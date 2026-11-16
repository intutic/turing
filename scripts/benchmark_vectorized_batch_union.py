#!/usr/bin/env python3
"""
Benchmark: MoE expert offload (HeterogeneousMoERunner) against the model's own SparseMoE, on the real weights of a
small Mixtral checkpoint. For each token count it times layer 0's MoE block three ways and checks the outputs agree:
1. SparseMoE (all experts resident on the device),
2. the runner with every expert in device cache slots (second call, all hits),
3. the runner with every expert run on the host CPU.
"""
import _bootstrap  # noqa: F401,E402  (this repo's root first on sys.path)

import json
import sys
import time
import torch

from turing.core.expert_cache import GPULRUExpertCache
from turing.core.heterogeneous_moe import BandwidthAdaptiveDecider, HeterogeneousMoERunner
from turing.models.hf_loader import RealHuggingFaceLoader

MODEL = "axolotl-ai-co/tiny-mixtral-30m"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _time_ms(fn, device, iters=20):
    fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    _sync(device)
    return (time.perf_counter() - t0) / iters * 1000.0, out


@torch.no_grad()
def benchmark_batch_union(device_str="cpu"):
    device = torch.device(device_str)
    model, _ = RealHuggingFaceLoader.load_hf_model_into_turing(hf_model_id=MODEL, device=device_str, dtype="fp32")
    cfg = model.config
    moe = model.layers[0].mlp
    print("\n" + "=" * 80)
    print("   [4/5] BENCHMARK: MoE EXPERT OFFLOAD vs IN-MODEL SparseMoE")
    print(f"   {MODEL} on {device_str.upper()} | hidden {cfg.hidden_dim} | experts {cfg.num_experts} "
          f"top-{cfg.num_experts_per_tok}")
    print("=" * 80)

    def runner(decider):
        cache = GPULRUExpertCache(num_slots=cfg.num_experts, hidden_dim=cfg.hidden_dim,
                                  active_subspace_dim=moe.down.shape[-1], device=device, eviction_policy="lru")
        return HeterogeneousMoERunner.from_model(model, cache, decider)

    on_device = runner(BandwidthAdaptiveDecider(device, pcie_bandwidth_gb_s=1e9, cpu_throughput_gflops=1e-9,
                                                gpu_throughput_gflops=1e9))
    on_host = runner(BandwidthAdaptiveDecider(device, pcie_bandwidth_gb_s=1e-9, cpu_throughput_gflops=1e9,
                                              gpu_throughput_gflops=1e-9))
    rows = []
    for n in (1, 16, 128, 512):
        x = torch.randn(1, n, cfg.hidden_dim, device=device)
        ref_ms, ref = _time_ms(lambda: moe(x), device)
        dev_ms, dev_out = _time_ms(lambda: on_device.route_and_execute(x, 0)[0], device)
        host_ms, host_out = _time_ms(lambda: on_host.route_and_execute(x, 0)[0], device)
        err = max((dev_out - ref).abs().max().item(), (host_out - ref).abs().max().item())
        rows.append({"tokens": n, "sparse_moe_ms": round(ref_ms, 3), "runner_cached_ms": round(dev_ms, 3),
                     "runner_host_ms": round(host_ms, 3), "max_abs_diff": err})
        print(f"    • N={n:4d}: SparseMoE {ref_ms:.3f} ms | runner, cached {dev_ms:.3f} ms | runner, host "
              f"{host_ms:.3f} ms | max |diff| {err:.2e}", flush=True)
    return {"model": MODEL, "device": device_str, "layer": 0, "rows": rows}


if __name__ == "__main__":
    res = benchmark_batch_union(sys.argv[1] if len(sys.argv) > 1 else "cpu")
    print(json.dumps(res, indent=2))
