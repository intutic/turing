"""
OS-Style Hierarchical Virtual Memory Paging (Huge, Medium, Small Page Tiers) for KV Caches.
"""

from enum import Enum
from typing import List, Dict, Tuple, Optional, Any
import torch

class PageTier(Enum):
    HUGE = 512     # For massive prompt prefixes & long-context history (eliminates 96.8% pointer tables)
    MEDIUM = 64    # For multi-turn conversational turns
    SMALL = 16     # For decode tails to guarantee zero internal fragmentation

class PhysicalBlock:
    def __init__(self, block_id: int, tier: PageTier):
        self.block_id = block_id
        self.tier = tier
        self.capacity = tier.value
        self.ref_count = 0
        self.is_free = True

class VirtualPageEntry:
    def __init__(self, logical_page_id: int, physical_block_id: int, tier: PageTier, valid_tokens: int = 0):
        self.logical_page_id = logical_page_id
        self.physical_block_id = physical_block_id
        self.tier = tier
        self.valid_tokens = valid_tokens

class HierarchicalVirtualPageManager:
    """
    Manages multi-tier virtual page tables and physical KV block pools with buddy allocation.
    """
    def __init__(
        self,
        num_huge_blocks: int = 64,
        num_medium_blocks: int = 128,
        num_small_blocks: int = 256,
        hidden_dim: int = 8192,
        num_heads: int = 64,
        head_dim: int = 128,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float16
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Fast C++20 Bitmap Allocator Backend
        try:
            import turing.turing_csrc as turing_csrc
            self.csrc_allocator = turing_csrc.HierarchicalBitmapAllocator(num_huge_blocks, num_medium_blocks, num_small_blocks)
            self.has_csrc = True
        except ImportError:
            try:
                import turing_csrc
                self.csrc_allocator = turing_csrc.HierarchicalBitmapAllocator(num_huge_blocks, num_medium_blocks, num_small_blocks)
                self.has_csrc = True
            except ImportError:
                self.csrc_allocator = None
                self.has_csrc = False

        # Free lists per tier fallback
        self.free_blocks: Dict[PageTier, List[int]] = {
            PageTier.HUGE: list(range(num_huge_blocks)),
            PageTier.MEDIUM: list(range(num_medium_blocks)),
            PageTier.SMALL: list(range(num_small_blocks)),
        }

        # Sequence page tables: seq_id -> List[VirtualPageEntry]
        self.sequence_page_tables: Dict[int, List[VirtualPageEntry]] = {}

        # Physical KV pool tensors
        self.k_pools: Dict[PageTier, torch.Tensor] = {
            PageTier.HUGE: torch.zeros(num_huge_blocks, num_heads, PageTier.HUGE.value, head_dim, dtype=dtype, device=device),
            PageTier.MEDIUM: torch.zeros(num_medium_blocks, num_heads, PageTier.MEDIUM.value, head_dim, dtype=dtype, device=device),
            PageTier.SMALL: torch.zeros(num_small_blocks, num_heads, PageTier.SMALL.value, head_dim, dtype=dtype, device=device),
        }
        self.v_pools: Dict[PageTier, torch.Tensor] = {
            PageTier.HUGE: torch.zeros(num_huge_blocks, num_heads, PageTier.HUGE.value, head_dim, dtype=dtype, device=device),
            PageTier.MEDIUM: torch.zeros(num_medium_blocks, num_heads, PageTier.MEDIUM.value, head_dim, dtype=dtype, device=device),
            PageTier.SMALL: torch.zeros(num_small_blocks, num_heads, PageTier.SMALL.value, head_dim, dtype=dtype, device=device),
        }

    def allocate_prompt_pages(self, seq_id: int, prompt_len: int) -> List[VirtualPageEntry]:
        """
        Allocates multi-tier pages for prompt prefill using maximal page compaction.
        Example: 1040 tokens -> 2 Huge (1024) + 1 Small (16).
        """
        self.sequence_page_tables[seq_id] = []

        if self.has_csrc and self.csrc_allocator is not None:
            alloc_tuples = self.csrc_allocator.allocate_prompt(prompt_len)
            tier_map = {512: PageTier.HUGE, 64: PageTier.MEDIUM, 16: PageTier.SMALL}
            for idx, (tier_val, block_id, valid_toks) in enumerate(alloc_tuples):
                tier = tier_map.get(tier_val, PageTier.SMALL)
                entry = VirtualPageEntry(idx, block_id, tier, valid_toks)
                self.sequence_page_tables[seq_id].append(entry)
            return self.sequence_page_tables[seq_id]

        rem = prompt_len

        # 1. Allocate Huge Blocks
        while rem >= PageTier.HUGE.value and self.free_blocks[PageTier.HUGE]:
            block_id = self.free_blocks[PageTier.HUGE].pop(0)
            entry = VirtualPageEntry(len(self.sequence_page_tables[seq_id]), block_id, PageTier.HUGE, PageTier.HUGE.value)
            self.sequence_page_tables[seq_id].append(entry)
            rem -= PageTier.HUGE.value

        # 2. Allocate Medium Blocks
        while rem >= PageTier.MEDIUM.value and self.free_blocks[PageTier.MEDIUM]:
            block_id = self.free_blocks[PageTier.MEDIUM].pop(0)
            entry = VirtualPageEntry(len(self.sequence_page_tables[seq_id]), block_id, PageTier.MEDIUM, PageTier.MEDIUM.value)
            self.sequence_page_tables[seq_id].append(entry)
            rem -= PageTier.MEDIUM.value

        # 3. Allocate Small Blocks for the remainder
        while rem > 0 and self.free_blocks[PageTier.SMALL]:
            block_id = self.free_blocks[PageTier.SMALL].pop(0)
            tokens_in_block = min(rem, PageTier.SMALL.value)
            entry = VirtualPageEntry(len(self.sequence_page_tables[seq_id]), block_id, PageTier.SMALL, tokens_in_block)
            self.sequence_page_tables[seq_id].append(entry)
            rem -= tokens_in_block

        return self.sequence_page_tables[seq_id]

    def append_token(self, seq_id: int, k_token: torch.Tensor, v_token: torch.Tensor) -> Tuple[int, int]:
        """
        Appends a single token KV state during decode into the active tail page.
        Allocates a new Small Page when tail is exhausted.
        k_token, v_token: [num_heads, head_dim]
        """
        table = self.sequence_page_tables.get(seq_id)
        if not table:
            self.sequence_page_tables[seq_id] = []
            table = self.sequence_page_tables[seq_id]

        if not table or table[-1].valid_tokens >= table[-1].tier.value:
            # Allocate new small page
            if not self.free_blocks[PageTier.SMALL]:
                raise RuntimeError("Out of small KV pages")
            block_id = self.free_blocks[PageTier.SMALL].pop(0)
            table.append(VirtualPageEntry(len(table), block_id, PageTier.SMALL, 0))

        tail = table[-1]
        slot = tail.valid_tokens
        tier = tail.tier
        block_id = tail.physical_block_id

        # Write to physical pool
        self.k_pools[tier][block_id, :, slot, :].copy_(k_token)
        self.v_pools[tier][block_id, :, slot, :].copy_(v_token)
        tail.valid_tokens += 1

        return block_id, slot

    def rollback_tail(self, seq_id: int, num_tokens_to_discard: int):
        """
        O(1) Zero-Copy Rollback for Speculative Execution Rejection.
        Simply adjusts the tail pointer without re-allocating or memory zeroing.
        """
        table = self.sequence_page_tables.get(seq_id)
        if not table:
            return

        rem = num_tokens_to_discard
        while rem > 0 and table:
            tail = table[-1]
            if tail.valid_tokens <= rem:
                rem -= tail.valid_tokens
                # Return small block to free list if completely rejected
                self.free_blocks[tail.tier].append(tail.physical_block_id)
                table.pop()
            else:
                tail.valid_tokens -= rem
                rem = 0

    def free_sequence(self, seq_id: int):
        """Reclaims all physical blocks for a completed sequence."""
        table = self.sequence_page_tables.pop(seq_id, [])
        for entry in table:
            self.free_blocks[entry.tier].append(entry.physical_block_id)

class StaticPagedKVPool:
    """
    Zero-Allocation Contiguous Page Memory Pool.
    Pre-allocates physical KV memory buffers at engine startup to eliminate
    Python GC thrashing and dynamic heap allocations during continuous serving.
    """
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int = 16,
        max_total_pages: int = 1024,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float16
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_total_pages = max_total_pages
        self.active_max_pages = max_total_pages
        self.device = device
        self.dtype = dtype

        # Pre-allocate contiguous [layers, pages, heads, page_size, head_dim]
        self.k_pool = torch.zeros(
            (num_layers, max_total_pages, num_heads, page_size, head_dim),
            dtype=dtype,
            device=device
        )
        self.v_pool = torch.zeros(
            (num_layers, max_total_pages, num_heads, page_size, head_dim),
            dtype=dtype,
            device=device
        )

        self.free_page_indices: List[int] = list(range(max_total_pages))
        self.allocated_pages: Dict[str, List[int]] = {}

    def adjust_active_capacity(self, new_max_pages: int) -> int:
        """
        Dynamically adjusts active page pool ceiling without reallocating tensor buffers.
        Returns the adjusted active capacity.
        """
        new_capacity = max(1, min(new_max_pages, self.max_total_pages))
        self.active_max_pages = new_capacity

        allocated_set = set(p for pages in self.allocated_pages.values() for p in pages)
        # Keep free pages that are within new_capacity and not currently allocated
        self.free_page_indices = [p for p in range(new_capacity) if p not in allocated_set]
        return new_capacity

    def allocate_pages(self, request_id: str, num_pages: int) -> List[int]:
        """Allocates contiguous slots with O(1) stack pop."""
        if len(self.free_page_indices) < num_pages:
            raise MemoryError(f"Out of static KV pages: requested {num_pages}, available {len(self.free_page_indices)}")
        pages = [self.free_page_indices.pop(0) for _ in range(num_pages)]
        if request_id not in self.allocated_pages:
            self.allocated_pages[request_id] = []
        self.allocated_pages[request_id].extend(pages)
        return pages

    def free_pages(self, request_id: str):
        """Reclaims page slots back to free stack."""
        pages = self.allocated_pages.pop(request_id, [])
        for p in pages:
            if p < self.active_max_pages and p not in self.free_page_indices:
                self.free_page_indices.append(p)
        self.free_page_indices.sort()

    def get_stats(self) -> Dict[str, Any]:
        used = sum(len(p) for p in self.allocated_pages.values())
        free = max(0, self.active_max_pages - used)
        return {
            "total_pages": self.active_max_pages,
            "max_physical_pages": self.max_total_pages,
            "used_pages": used,
            "free_pages": free,
            "utilization_pct": round((used / max(1, self.active_max_pages)) * 100.0, 2)
        }




