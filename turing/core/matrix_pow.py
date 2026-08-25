"""
Logarithmic Recurrence State Evaluation & Matrix Power Jump-Ahead Engine.
Evaluates linear state transitions in O(log K) steps for parallel speculative decoding and SSMs.
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

class LogarithmicRecurrenceEngine:
    """
    Logarithmic Matrix Exponentiation Engine for Jump-Ahead Recurrences.
    Computes x_k = (A*x_{k-1} + B*x_{k-2} + C) mod M in O(log k) multiplications.
    """
    def __init__(self, x0: int, x1: int, A: int, B: int, C: int, M: int):
        self.x0 = x0
        self.x1 = x1
        self.A = A
        self.B = B
        self.C = C
        self.M = M

    def jump_ahead(self, k: int) -> int:
        """
        Returns the k-th state in O(log k) time.
        """
        if HAS_CSRC:
            return turing_csrc.matrix_power_transition(self.x0, self.x1, self.A, self.B, self.C, self.M, k)

        # Python fallback linear simulation for small k or binary power
        if k == 0:
            return self.x0 % self.M
        if k == 1:
            return self.x1 % self.M
        p2 = self.x0 % self.M
        p1 = self.x1 % self.M
        for _ in range(2, k + 1):
            nxt = (self.A * p1 + self.B * p2 + self.C) % self.M
            p2 = p1
            p1 = nxt
        return p1

    def generate_chunk(self, start_idx: int, length: int) -> np.ndarray:
        """
        Generates a slice [start_idx, start_idx + length) in parallel by initializing from jump-ahead state.
        """
        if length <= 0:
            return np.array([], dtype=np.int64)

        p1 = self.jump_ahead(start_idx)
        if length == 1:
            return np.array([p1], dtype=np.int64)

        if start_idx == 0:
            p2 = 0
        else:
            p2 = self.jump_ahead(start_idx - 1)

        out = np.zeros(length, dtype=np.int64)
        out[0] = p1

        for i in range(1, length):
            nxt = (self.A * p1 + self.B * p2 + self.C) % self.M
            out[i] = nxt
            p2 = p1
            p1 = nxt
        return out
