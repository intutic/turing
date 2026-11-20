"""
Multi-Modal Objective Landscapes for Swarm Auto-Tuning Verification.
Evaluates Ackley, Rastrigin, Griewank, and Rosenbrock fitness functions.
"""

from typing import List, Union
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

def evaluate_objective(name: str, x: Union[List[float], np.ndarray]) -> float:
    """
    Evaluates benchmark objective function in C++ or Python fallback.
    Supported: 'ackley', 'rastrigin', 'griewank', 'rosenbrock'.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if HAS_CSRC:
        return float(turing_csrc.evaluate_pso_objective(name.lower(), x_arr))

    # Python fallback
    d = len(x_arr)
    if name.lower() == "ackley":
        sum_sq = np.sum(x_arr ** 2)
        sum_cos = np.sum(np.cos(2 * np.pi * x_arr))
        return float(-20.0 * np.exp(-0.2 * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + 20.0 + np.e)
    elif name.lower() == "rastrigin":
        return float(10.0 * d + np.sum(x_arr ** 2 - 10.0 * np.cos(2 * np.pi * x_arr)))
    elif name.lower() == "griewank":
        sum_sq = np.sum(x_arr ** 2) / 4000.0
        prod_cos = np.prod(np.cos(x_arr / np.sqrt(np.arange(1, d + 1))))
        return float(sum_sq - prod_cos + 1.0)
    elif name.lower() == "rosenbrock":
        return float(np.sum(100.0 * (x_arr[1:] - x_arr[:-1]**2)**2 + (1.0 - x_arr[:-1])**2))
    return 0.0
