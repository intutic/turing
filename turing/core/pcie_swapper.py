"""
Double-Buffered Asynchronous PCIe Virtual Page Swapper (128K+ Out-Of-Core Context).
"""

from typing import List, Optional, Tuple
import torch


class DoubleBufferedAsyncRingSwapper:
    """
    Manages asynchronous double-buffered ring paging between CPU Host Pinned Memory
    and GPU VRAM staging slots, hiding DMA transfer latency behind compute via lookahead (N+2).
    """
    def __init__(
        self,
        host_huge_pages: int = 64,
        ring_slots: int = 4,
        page_size: int = 512,
        num_heads: int = 64,
        rank_sub: int = 64,
        device: torch.device = torch.device("cpu")
    ):
        self.host_huge_pages = host_huge_pages
        self.ring_slots = ring_slots
        self.page_size = page_size
        self.num_heads = num_heads
        self.rank_sub = rank_sub
        self.device = device

        self.is_cuda = (device.type == "cuda")

        # 1. Allocate Pinned Host Pool
        if self.is_cuda:
            self.k_host = torch.zeros((host_huge_pages, num_heads, page_size, rank_sub), dtype=torch.int8, pin_memory=True)
            self.v_host = torch.zeros((host_huge_pages, num_heads, page_size, rank_sub), dtype=torch.int8, pin_memory=True)
        else:
            self.k_host = torch.zeros((host_huge_pages, num_heads, page_size, rank_sub), dtype=torch.int8)
            self.v_host = torch.zeros((host_huge_pages, num_heads, page_size, rank_sub), dtype=torch.int8)

        # 2. Allocate GPU VRAM Staging Ring Slots
        self.k_gpu_ring = torch.zeros((ring_slots, num_heads, page_size, rank_sub), dtype=torch.int8, device=device)
        self.v_gpu_ring = torch.zeros((ring_slots, num_heads, page_size, rank_sub), dtype=torch.int8, device=device)

        # 3. Dedicated CUDA Streams & Slot Events
        if self.is_cuda:
            self.pcie_stream = torch.cuda.Stream(device=device)
            self.slot_events = [torch.cuda.Event() for _ in range(ring_slots)]
        else:
            self.pcie_stream = None
            self.slot_events = None

    def prefetch_page_async(self, host_page_idx: int, ring_slot_idx: int):
        """
        Asynchronously streams host page into GPU ring slot over pcie_stream.
        """
        if self.is_cuda and self.pcie_stream is not None:
            with torch.cuda.stream(self.pcie_stream):
                self.k_gpu_ring[ring_slot_idx].copy_(self.k_host[host_page_idx], non_blocking=True)
                self.v_gpu_ring[ring_slot_idx].copy_(self.v_host[host_page_idx], non_blocking=True)
                self.slot_events[ring_slot_idx].record(self.pcie_stream)
        else:
            self.k_gpu_ring[ring_slot_idx].copy_(self.k_host[host_page_idx])
            self.v_gpu_ring[ring_slot_idx].copy_(self.v_host[host_page_idx])

    def wait_slot_ready(self, ring_slot_idx: int, compute_stream: Optional[torch.cuda.Stream] = None):
        """
        Ensures compute stream waits for DMA transfer to complete on the target ring slot.
        """
        if self.is_cuda and self.slot_events is not None and compute_stream is not None:
            compute_stream.wait_event(self.slot_events[ring_slot_idx])

    def get_ready_slot_tensor(self, ring_slot_idx: int) -> torch.Tensor:
        return self.k_gpu_ring[ring_slot_idx], self.v_gpu_ring[ring_slot_idx]


class NetworkKVPuller:
    """
    Asynchronous TCP/HTTP KV block puller for distributed and disaggregated serving.
    Pulls SVD-compressed KV payloads from remote prefiller pods and stages them into local memory.
    """

    def __init__(self, timeout_s: float = 10.0, device: torch.device = torch.device("cpu")):
        self.timeout_s = timeout_s
        self.device = device

    def pull_kv_block_sync(
        self,
        remote_host_port: str,
        request_id: str,
        block_id: int,
        u_proj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Synchronously pulls an SVD-compressed KV block from remote host and reconstructs it.
        """
        import urllib.request
        import urllib.error
        from ..serving.kv_transfer import deserialize_kv_block_svd

        url = f"http://{remote_host_port}/kv/blocks/{request_id}/{block_id}"
        req = urllib.request.Request(url, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = resp.read()
                return deserialize_kv_block_svd(data, u_proj=u_proj, device=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to pull KV block {block_id} for request {request_id} from {remote_host_port}: {e}")

