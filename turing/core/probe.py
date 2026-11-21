"""
What this machine's accelerator delivers, measured, and the kernel settings that depend on it.

Batch-1 decoding reads every weight once per token, so its time per token cannot go below the weight bytes divided by
the memory read bandwidth (`read_bandwidth_gb_s`, `decode_floor_ms`). Prefill and large batches are bound by the
matmul rate instead (`matmul_tflops`). `turing bench-bw` prints both.

Some kernel choices are crossovers measured on one GPU (an NVIDIA L4): for example the row count up to which the
fused int4 GEMM beats dequantizing and calling cuBLAS. `turing bench-bw --tune` measures them on the GPU present and
keeps them in a JSON file per GPU model; `tuned(name, default)` returns the kept value, or the default when that GPU
has not been tuned.
"""

import functools
import json
import os
import statistics
import time
from typing import Callable, Dict

import torch


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def read_bandwidth_gb_s(dev: torch.device, gib: float = 2, iters: int = 7) -> float:
    """GB/s of one pass reading `gib` GiB of device memory (an int64 sum: a read-only, memory-bound kernel)."""
    x = torch.ones(int(gib * 2 ** 30) // 8, dtype=torch.int64, device=dev)
    ts = []
    for _ in range(iters):
        _sync(dev)
        t = time.perf_counter()
        x.sum()
        _sync(dev)
        ts.append(time.perf_counter() - t)
    del x
    return gib * 2 ** 30 / 1e9 / statistics.median(ts)


def matmul_tflops(dev: torch.device, dtype: torch.dtype, n: int = 4096, iters: int = 10) -> float:
    """TFLOP/s of an n x n matmul in dtype (2 n^3 floating-point operations)."""
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    a @ b
    ts = []
    for _ in range(iters):
        _sync(dev)
        t = time.perf_counter()
        a @ b
        _sync(dev)
        ts.append(time.perf_counter() - t)
    return 2 * n ** 3 / 1e12 / statistics.median(ts)


def decode_floor_ms(weight_bytes: int, bandwidth_gb_s: float) -> float:
    """The least time one batch-1 decode step can take: every weight byte read once at the measured bandwidth."""
    return weight_bytes / 1e9 / bandwidth_gb_s * 1e3


def graph_us(fn: Callable[[], object], iters: int = 15, reps: int = 20) -> float:
    """GPU microseconds per call of fn, from CUDA-graph replays of `reps` back-to-back calls (serving decode replays
    graphs; timing each launch from Python would measure the launcher once the kernel is faster than it)."""
    for _ in range(3):  # compilation and autotuning, outside the capture
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) / reps * 1e6)
    return statistics.median(ts)


# ---------------------------------------------------------------------------------------------- tuned settings

def _tuning_file() -> str:
    return os.path.join(os.environ.get("TURING_CACHE", os.path.expanduser("~/.cache/turing")), "tuning.json")


def _gpu_key() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


@functools.lru_cache(maxsize=None)
def _tuning() -> Dict[str, dict]:
    try:
        with open(_tuning_file(), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def tuned(name: str, default):
    """The value `turing bench-bw --tune` measured for this GPU model, or default when it was never tuned."""
    return _tuning().get(_gpu_key(), {}).get(name, default)


def save_tuning(values: dict) -> str:
    """Keep values for this GPU model (merged with what is there); returns the file."""
    data = dict(_tuning())
    data[_gpu_key()] = {**data.get(_gpu_key(), {}), **values}
    os.makedirs(os.path.dirname(_tuning_file()), exist_ok=True)
    with open(_tuning_file(), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    _tuning.cache_clear()
    return _tuning_file()


@torch.inference_mode()
def tune_int4_max_rows(dev: torch.device, shapes=((2048, 2048), (8192, 2048), (2048, 8192)),
                       rows=(16, 32, 64, 128, 256)) -> dict:
    """The largest row count at which the fused int4 GEMM is no slower than dequantizing into a buffer and calling
    cuBLAS, summed over layer shapes (out x in). Returns {"int4_max_rows": M, "int4_crossover_us": {M: [int4, dense]}}."""
    from ..kernels.triton_w4a16 import dequantize_cuda, w4a16_gemm
    from ..models.quant import GROUP, pack, rtn_quantize
    layers = []
    for n, k in shapes:
        c, s, z = rtn_quantize(torch.randn(n, k, device=dev) * 0.02)
        layers.append((pack(c), s, z, torch.empty(n, k, device=dev, dtype=torch.float16)))
    table, best = {}, rows[0]
    for m in rows:
        int4 = dense = 0.0
        for q, s, z, buf in layers:
            x = torch.randn(m, q.shape[1] * 2, device=dev, dtype=torch.float16)
            int4 += graph_us(lambda: w4a16_gemm(x, q, s, z, GROUP, None))
            dense += graph_us(lambda: torch.nn.functional.linear(x, dequantize_cuda(q, s, z, buf, GROUP)))
        table[m] = [round(int4, 1), round(dense, 1)]
        if int4 <= dense:
            best = m
    return {"int4_max_rows": best, "int4_crossover_us": table}
