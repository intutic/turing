"""
Persistent Thread-Local Reduction & Parallel CPU MoE Accumulation Engine.
Provides zero-allocation thread-local reductions across multi-core CPU workers.
"""

from typing import List, Optional
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

class PersistentThreadAccumulator:
    """
    Manages persistent thread-local buffers to eliminate OpenMP thread spawn and memory allocation overhead.
    """
    def __init__(self, num_threads: int = 4, dim: int = 256):
        self.num_threads = num_threads
        self.dim = dim
        if HAS_CSRC:
            self.csrc_reducer = turing_csrc.PersistentThreadReducer(num_threads, dim)
        else:
            self.csrc_reducer = None
            self.buffers = np.zeros((num_threads, dim), dtype=np.float32)

    def reduce_sum(self, thread_arrays: np.ndarray) -> np.ndarray:
        """
        thread_arrays: [NumThreads, Dim]
        """
        if HAS_CSRC:
            return turing_csrc.persistent_parallel_reduce(thread_arrays.astype(np.float32))
        return np.sum(thread_arrays, axis=0)
