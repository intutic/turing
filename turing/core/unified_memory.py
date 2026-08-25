"""
Unified Memory Management & Zero-Copy Staging Buffer Pool.
Provides aligned memory allocations for heterogeneous host-device data transfers.
"""

from typing import Optional, Tuple
import numpy as np

try:
    import turing.turing_csrc as turing_csrc
    HAS_CSRC = True
except ImportError:
    try:
        import turing_csrc
        HAS_CSRC = True
    except ImportError:
        HAS_CSRC = False

class UnifiedMemoryBufferPool:
    """
    Manages unified memory allocation slabs with prefetching and zero-copy semantics.
    """
    def __init__(self, capacity_bytes: int = 64 * 1024 * 1024):
        self.capacity_bytes = capacity_bytes
        if HAS_CSRC:
            self.csrc_pool = turing_csrc.UnifiedMemoryPool(capacity_bytes)
        else:
            self.csrc_pool = None
            self.used_bytes = 0

    def allocate(self, size_bytes: int) -> int:
        """
        Allocates a 64-byte aligned slab and returns its byte offset.
        """
        if HAS_CSRC and self.csrc_pool is not None:
            return self.csrc_pool.allocate_slab(size_bytes)
        aligned = (size_bytes + 63) & ~63
        if self.used_bytes + aligned > self.capacity_bytes:
            return -1
        offset = self.used_bytes
        self.used_bytes += aligned
        return offset

    def reset(self):
        if HAS_CSRC and self.csrc_pool is not None:
            self.csrc_pool.reset()
        else:
            self.used_bytes = 0

    @property
    def free_bytes(self) -> int:
        if HAS_CSRC and self.csrc_pool is not None:
            return self.csrc_pool.free
        return self.capacity_bytes - self.used_bytes

    @property
    def used_bytes_count(self) -> int:
        if HAS_CSRC and self.csrc_pool is not None:
            return self.csrc_pool.used
        return self.used_bytes
