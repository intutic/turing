#!/usr/bin/env python3
"""
Where one batch-1 decode step's time goes, per weight format: GPU kernel launches per step and device time by kernel
(torch.profiler over eager steps), the wall time of an eager step and of a CUDA-graph step (the serving path), prefill
time for 512 tokens and for a long prompt in chunks, and the bandwidth floor: weight bytes over the GPU's measured
read bandwidth. Prompts are random token ids (speed does not depend on the text).

python scripts/profile_decode_step.py --model Qwen/Qwen2.5-1.5B-Instruct --settings fp16,rtn4
"""
import _bootstrap  # noqa: F401,E402
import argparse
import copy
import json
import platform
import statistics
import time
from collections import defaultdict

import torch

from turing.models.hf_loader import RealHuggingFaceLoader
from turing.models.quant import quantize_model, weight_bytes


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def read_bandwidth_gb_s(dev, gib=2, iters=7):
    """GB/s of one pass reading `gib` GiB of device memory (an int64 sum: a read-only, memory-bound kernel)."""
    x = torch.ones(gib * 2 ** 30 // 8, dtype=torch.int64, device=dev)
    ts = []
    for _ in range(iters):
        sync(dev)
        t = time.perf_counter()
        x.sum()
        sync(dev)
        ts.append(time.perf_counter() - t)
    del x
    return gib * 2 ** 30 / 1e9 / statistics.median(ts)


@torch.inference_mode()
def steps_wall_ms(model, ids, n, dev, graphs):
    model.config.cuda_graphs = graphs
    _, past = model(ids, logits_to_keep=1)
    tok = ids[:, -1:]
    for _ in range(8):  # compilation, autotuning, graph capture
        logits, past = model(tok, past_key_values=past)
    sync(dev)
    t = time.perf_counter()
    for _ in range(n):
        logits, past = model(tok, past_key_values=past)
        tok = logits[:, -1:].argmax(-1)
    sync(dev)
    return (time.perf_counter() - t) / n * 1e3


@torch.inference_mode()
def kernel_profile(model, ids, n, dev):
    """Eager decode steps under torch.profiler: (launches per step, device ms per step, [name, launches, us] by time)."""
    from torch.profiler import ProfilerActivity, profile
    model.config.cuda_graphs = False
    _, past = model(ids, logits_to_keep=1)
    tok = ids[:, -1:]
    for _ in range(8):
        logits, past = model(tok, past_key_values=past)
    sync(dev)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            logits, past = model(tok, past_key_values=past)
        sync(dev)
    count, us = defaultdict(int), defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            count[e.name] += 1
            us[e.name] += e.time_range.elapsed_us()
    top = sorted(us, key=lambda k: -us[k])
    return (sum(count.values()) / n, sum(us.values()) / n / 1e3,
            [[k[:90], round(count[k] / n, 2), round(us[k] / n, 1)] for k in top[:30]])


@torch.inference_mode()
def prefill_s(model, ids, chunk, dev):
    """Seconds to prefill ids in chunks of `chunk` tokens through the cache (the engine's path), second of two runs."""
    out = None
    for _ in range(2):
        sync(dev)
        t = time.perf_counter()
        past = None
        for s in range(0, ids.shape[1], chunk):
            _, past = model(ids[:, s:s + chunk], past_key_values=past, logits_to_keep=1)
        sync(dev)
        out = time.perf_counter() - t
        del past
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--settings", default="fp16,rtn4",
                    help="fp16, rtn4 and/or rtn4h (the head too; speed is the same for GPTQ); a '+compile' suffix (fp16+compile) wraps "
                         "every layer's MLP and norms in torch.compile (speed plan P2.4, decision D11)")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--long-context", type=int, default=16384)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results_decode_profile_cuda.json")
    a = ap.parse_args()
    dev = torch.device(a.device)
    cuda = dev.type == "cuda"
    base, _ = RealHuggingFaceLoader.load_hf_model_into_turing(a.model, device=a.device, dtype="fp16" if cuda else "fp32")
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, base.config.vocab_size, (1, a.context), generator=g).to(dev)
    long_ids = torch.randint(0, base.config.vocab_size, (1, a.long_context), generator=g).to(dev)
    bw = read_bandwidth_gb_s(dev) if cuda else None
    rows = []
    for name in a.settings.split(","):
        model = copy.deepcopy(base)
        if name.split("+")[0] in ("rtn4", "rtn4h"):  # rtn4h: the LM head in 4 bits too (speed plan P3)
            quantize_model(model, "rtn", None, lm_head=name.split("+")[0] == "rtn4h")
        compile_s = None
        if name.endswith("+compile"):
            for layer in model.layers:
                for attr in ("mlp", "input_layernorm", "post_attention_layernorm"):
                    if hasattr(layer, attr):
                        setattr(layer, attr, torch.compile(getattr(layer, attr)))
            with torch.inference_mode():  # first call through every compiled module: the compile cost
                sync(dev)
                t = time.perf_counter()
                _, past = model(ids, logits_to_keep=1)
                model(ids[:, -1:], past_key_values=past)
                sync(dev)
                compile_s = time.perf_counter() - t
                del past
        wb = weight_bytes(model)
        row = dict(model=a.model, setting=name, weight_bytes=wb, context=a.context, compile_s=compile_s,
                   floor_ms=wb / 1e9 / bw * 1e3 if bw else None,
                   graph_step_ms=steps_wall_ms(model, ids, a.steps, dev, True) if cuda else None,
                   eager_step_ms=steps_wall_ms(model, ids, a.steps, dev, False))
        if cuda:
            launches, device_ms, top = kernel_profile(model, ids, 16, dev)
            row.update(kernel_launches_per_step=launches, device_ms_per_step=device_ms, top_kernels=top)
        row.update(prefill_s=prefill_s(model, ids, a.chunk, dev), long_context=a.long_context, chunk=a.chunk,
                   long_prefill_s=prefill_s(model, long_ids, a.chunk, dev))
        print(json.dumps({k: v for k, v in row.items() if k != "top_kernels"}), flush=True)
        for k in row.get("top_kernels", [])[:12]:
            print("   ", k, flush=True)
        rows.append(row)
        del model
        if cuda:
            torch.cuda.empty_cache()
    env = dict(platform=platform.platform(), torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if cuda else None,
               read_bandwidth_gb_s=bw)
    with open(a.out, "w") as f:
        json.dump(dict(env=env, rows=rows, note="top_kernels: [name, launches per step, device microseconds per step] "
                                                 "from eager steps; graph_step_ms is the serving path"), f, indent=2)
    print(f"[+] wrote {a.out}")


if __name__ == "__main__":
    main()
