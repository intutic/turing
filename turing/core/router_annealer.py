"""
Router Temperature Annealing & Online Streaming Welford Normalization Engine.
Provides smooth parameter decay and real-time activation statistics.
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

class OnlineWelfordAccumulator:
    """
    Streaming Welford accumulator for computing running mean and variance.
    """
    def __init__(self):
        if HAS_CSRC:
            self.csrc_welford = turing_csrc.StreamingWelford()
        else:
            self.csrc_welford = None
            self.count = 0
            self.mean = 0.0
            self.m2 = 0.0

    def update(self, x: float):
        if HAS_CSRC and self.csrc_welford is not None:
            self.csrc_welford.update(float(x))
        else:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.m2 += delta * delta2

    @property
    def mean_value(self) -> float:
        if HAS_CSRC and self.csrc_welford is not None:
            return float(self.csrc_welford.mean)
        return self.mean

    @property
    def variance_value(self) -> float:
        if HAS_CSRC and self.csrc_welford is not None:
            return float(self.csrc_welford.variance)
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0

def compute_exponential_decay(init_val: float, min_val: float, current_step: int, max_steps: int) -> float:
    """
    Computes smooth exponential decay for temperature or sparsity.
    """
    if HAS_CSRC:
        return float(turing_csrc.exponential_decay_schedule(init_val, min_val, current_step, max_steps))
    if max_steps <= 0:
        return min_val
    if current_step <= 0:
        return init_val
    if current_step >= max_steps:
        return min_val
    return float(init_val * ((min_val / init_val) ** (current_step / max_steps)))
