#!/usr/bin/env python3
"""
MoE expert offload on real routing: allenai/OLMoE-1B-7B-0924 (16 MoE layers x 64 experts, top-8) run through
HeterogeneousMoERunner with every routed expert in host memory and a GPU expert cache (GPULRUExpertCache) of several
sizes, against the same model with all experts on the GPU.

The model is loaded once to host memory in bf16 (about 14 GB of host RAM; RealHuggingFaceLoader), the experts become
a HostExpertBank (the same tensors, no copy), every MoE layer's mlp becomes an OffloadedMoE and the remaining
weights (attention, embeddings, head: about 1 GB) move to the GPU. Each configuration gets a fresh cache and runner:
  slots    cache size in experts (16 x 64 = 1024 experts of 12 MiB each in bf16)
  policy   lru or lfru (GPULRUExpertCache eviction)
  mode     decider: the runner's calibrated BandwidthAdaptiveDecider picks copy-to-GPU or run-on-CPU per miss;
           device:  every miss is copied into a slot (no CPU execution)
Text: --windows WikiText-2 test windows; each is prefilled for --prompt-tokens, then --decode-tokens decode steps are
fed the window's real next tokens (teacher forcing, so every configuration routes the same text). The cache carries
over from window to window as it would in serving. Reported per configuration: expert-cache hit rate in prefill and in
decode (hits / experts dispatched, from the runner's counts), how misses were served, decode tok/s (mean and standard
error over windows, device synchronized), peak GPU memory, and how often the decode argmax equals the all-on-GPU
model's. The all-on-GPU run moves the original SparseMoE layers to the GPU at the end (needs about 14 GB of GPU).

python scripts/benchmark_moe_offload_olmoe.py [--slots 64,128,256,512] [--policies lru,lfru] [--modes decider,device]
"""
import _bootstrap  # noqa: F401,E402  (this repo's root first on sys.path)
import argparse
import json
import os
import platform
import resource
import statistics
import time

import torch

from turing.core.expert_cache import GPULRUExpertCache
from turing.core.heterogeneous_moe import BandwidthAdaptiveDecider, HeterogeneousMoERunner, HostExpertBank
from turing.models.causal_lm import activation
from turing.models.hf_loader import RealHuggingFaceLoader


def windows(tokenizer, length, n):
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n = min(n, len(ids) // length)
    return ids[: n * length].view(n, length)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def run_text(model, wins, prompt, totals, dev):
    """Per window: prefill, then teacher-forced decode. Returns (decode tok/s per window, decode argmaxes, prefill
    and decode dispatch counts)."""
    rates, argmax, pre, dec = [], [], {}, {}
    for w in wins:
        w = w.to(dev)[None]
        before = dict(totals)
        _, past = model(w[:, :prompt], logits_to_keep=1)
        sync(dev)
        mid = dict(totals)
        steps = []
        t0 = time.perf_counter()
        for t in range(prompt, w.shape[1]):
            logits, past = model(w[:, t:t + 1], past_key_values=past)
            steps.append(logits[0, -1].argmax())
        sync(dev)
        rates.append(len(steps) / (time.perf_counter() - t0))
        argmax.append(torch.stack(steps).tolist())
        for k in mid:
            pre[k] = pre.get(k, 0) + mid[k] - before.get(k, 0)
            dec[k] = dec.get(k, 0) + totals.get(k, 0) - mid[k]
        del past
    return rates, argmax, pre, dec


def summary(rates, pre, dec):
    def hit(c):
        return c.get("cache_hits", 0) / c["experts"] if c.get("experts") else None
    return dict(decode_tok_s=statistics.mean(rates),
                decode_tok_s_se=statistics.stdev(rates) / len(rates) ** 0.5 if len(rates) > 1 else None,
                n_windows=len(rates), decode_tok_s_per_window=rates,
                hit_rate_prefill=hit(pre), hit_rate_decode=hit(dec), prefill_counts=pre, decode_counts=dec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--slots", default="64,128,256,512")
    ap.add_argument("--policies", default="lru,lfru")
    ap.add_argument("--modes", default="decider,device")
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-gpu-baseline", action="store_true")
    ap.add_argument("--out", default="benchmarks/results/results_moe_offload_cuda.json")
    a = ap.parse_args(argv)
    dev = torch.device(a.device)

    t = time.perf_counter()
    model, tokenizer = RealHuggingFaceLoader.load_hf_model_into_turing(a.model, device="cpu", dtype="bf16")
    cfg = model.config
    cfg.kv_pool_gib = 1.0  # fixed, so the pool does not take the memory the expert caches need
    bank = HostExpertBank.from_model(model)
    act = activation(cfg.hidden_act)

    def make(cache, decider):
        return HeterogeneousMoERunner(bank, cache, decider, cfg.num_experts_per_tok, cfg.norm_topk_prob, act)
    first = make(GPULRUExpertCache(1, bank.hidden_dim, bank.ffn_dim, device=dev, dtype=torch.bfloat16),
                 BandwidthAdaptiveDecider(dev, 1.0, 1.0, 1.0))
    sparse_moe = first.attach(model)  # the original layers, kept on the host for the all-on-GPU run
    model.to(dev)
    del first
    load_s = time.perf_counter() - t
    print(f"loaded in {load_s:.0f}s: {len(bank.layers)} MoE layers x {bank.num_experts} experts, "
          f"{bank.bytes_per_expert / 2**20:.1f} MiB each", flush=True)

    wins = windows(tokenizer, a.prompt_tokens + a.decode_tokens, a.windows)
    calibrated = BandwidthAdaptiveDecider(dev)
    always_device = BandwidthAdaptiveDecider(dev, pcie_bandwidth_gb_s=1e9, cpu_throughput_gflops=1e-9,
                                             gpu_throughput_gflops=1e9)
    result = dict(script="scripts/benchmark_moe_offload_olmoe.py", args=vars(a),
                  env=dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                           cuda=torch.version.cuda, cpus=os.cpu_count(),
                           gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else None),
                  model=dict(moe_layers=len(bank.layers), experts=bank.num_experts, top_k=cfg.num_experts_per_tok,
                             bytes_per_expert=bank.bytes_per_expert, load_s=load_s),
                  decider=calibrated.get_telemetry(), windows=int(wins.shape[0]), rows=[])

    def save():
        result["host_max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with open(a.out, "w") as f:
            json.dump(result, f, indent=2)

    argmaxes = {}
    for slots in map(int, a.slots.split(",")):
        for policy in a.policies.split(","):
            for mode in a.modes.split(","):
                name = f"offload slots={slots} {policy} {mode}"
                cache = GPULRUExpertCache(slots, bank.hidden_dim, bank.ffn_dim, device=dev, dtype=torch.bfloat16,
                                          eviction_policy=policy)
                runner = make(cache, calibrated if mode == "decider" else always_device)
                runner.attach(model)
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                rates, argmax, pre, dec = run_text(model, wins, a.prompt_tokens, runner.totals, dev)
                row = dict(name=name, slots=slots, policy=policy, mode=mode,
                           slot_bytes=slots * bank.bytes_per_expert, **summary(rates, pre, dec),
                           peak_gpu_bytes=torch.cuda.max_memory_allocated() if dev.type == "cuda" else None)
                argmaxes[name] = argmax
                result["rows"].append(row)
                print(f"{name}: {row['decode_tok_s']:.2f} tok/s, hit rate decode {row['hit_rate_decode']:.3f} "
                      f"prefill {row['hit_rate_prefill']:.3f}", flush=True)
                del runner, cache
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                save()

    if not a.no_gpu_baseline:
        try:
            for i, m in sparse_moe.items():
                model.layers[i].mlp = m.to(dev)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            rates, argmax, _, _ = run_text(model, wins, a.prompt_tokens, {}, dev)
            row = dict(name="all experts on the GPU", **summary(rates, {}, {}),
                       peak_gpu_bytes=torch.cuda.max_memory_allocated() if dev.type == "cuda" else None)
            result["rows"].append(row)
            print(f"all on GPU: {row['decode_tok_s']:.2f} tok/s", flush=True)
            for r in result["rows"][:-1]:
                same = [x == y for wa, wb in zip(argmaxes[r["name"]], argmax) for x, y in zip(wa, wb)]
                r["argmax_agrees_with_gpu"] = sum(same) / len(same)
        except torch.cuda.OutOfMemoryError as e:
            result["gpu_baseline_error"] = f"out of memory: {e}"
    save()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
