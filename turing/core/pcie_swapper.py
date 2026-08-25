"""
Double-Buffered Asynchronous PCIe Virtual Page Swapper (128K+ Out-Of-Core Context).
"""

from typing import List, Optional
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
