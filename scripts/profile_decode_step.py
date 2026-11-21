#!/usr/bin/env python3
"""
Where one decode step's time goes, per weight format: GPU kernel launches per step and device time by kernel
(torch.profiler over eager steps), the wall time of an eager step and of a CUDA-graph step (the serving path), prefill
time for 512 tokens and for a long prompt in chunks, and the bandwidth floor: weight bytes over the GPU's measured
read bandwidth. Prompts are random token ids (a dense model's speed does not depend on the text) or, with --prompts
wikitext, WikiText-2 windows: an MoE model's floor counts only the experts the batch's tokens are routed to (averaged
over the profiled steps), and random tokens are routed to far fewer experts than text is.

python scripts/profile_decode_step.py --model Qwen/Qwen2.5-1.5B-Instruct --settings fp16,rtn4
python scripts/profile_decode_step.py --model allenai/OLMoE-1B-7B-0924-Instruct --dtype bf16 --settings bf16 \
    --batch 64 --prompts wikitext --long-context 0 --out benchmarks/results/results_moe_profile_cuda.json
"""
import _bootstrap  # noqa: F401,E402
import argparse
import copy
import json
import os
import platform
import statistics
import time
from collections import defaultdict

import torch

from turing.core.probe import read_bandwidth_gb_s
from turing.models.hf_loader import RealHuggingFaceLoader
from turing.models.quant import TGATE4_FILE, load_quantized, quantize_model, weight_bytes


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def prefilled(model, ids):
    """(caches, last logits): one row is prefilled into its own KVCache; several rows get a cache each and decode
    together, as the engine's requests do."""
    if ids.shape[0] == 1:
        logits, past = model(ids, logits_to_keep=1)
        return past, logits
    outs = [model(ids[i:i + 1], logits_to_keep=1) for i in range(ids.shape[0])]
    return [past for _, past in outs], torch.cat([logits for logits, _ in outs])


@torch.inference_mode()
def steps_wall_ms(model, ids, n, dev, graphs):
    model.config.cuda_graphs = graphs
    past, logits = prefilled(model, ids)
    tok = logits[:, -1:].argmax(-1)
    for _ in range(8):  # compilation, autotuning, graph capture
        logits, past = model(tok, past_key_values=past)
        tok = logits[:, -1:].argmax(-1)
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
    past, logits = prefilled(model, ids)
    tok = logits[:, -1:].argmax(-1)
    for _ in range(8):
        logits, past = model(tok, past_key_values=past)
        tok = logits[:, -1:].argmax(-1)
    sync(dev)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            logits, past = model(tok, past_key_values=past)
            tok = logits[:, -1:].argmax(-1)
        sync(dev)
    count, us = defaultdict(int), defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            count[e.name] += 1
            us[e.name] += e.time_range.elapsed_us()
    top = sorted(us, key=lambda k: -us[k])
    return (sum(count.values()) / n, sum(us.values()) / n / 1e3,
            [[k[:90], round(count[k] / n, 2), round(us[k] / n, 1)] for k in top[:30]])


def moe_read_bytes(model, run, rows):
    """Weight bytes one decode step of `rows` tokens reads: everything outside the experts plus, per MoE layer, the
    experts the tokens were routed to (mean over the decode steps run() makes). None for a dense model."""
    from turing.models.causal_lm import SparseMoE
    moes = [m for m in model.modules() if isinstance(m, SparseMoE)]
    if not moes:
        return None
    hit = defaultdict(list)

    def record(m):
        def hook(mod, inp, out):
            if out.shape[0] == rows:  # a decode step, not the prefill
                hit[m].append(out.float().topk(m.top_k, dim=-1).indices.unique().numel())
        return hook
    hooks = [m.router.register_forward_hook(record(m)) for m in moes]
    run()
    for h in hooks:
        h.remove()
    total = weight_bytes(model)
    for m in moes:
        names = ("gate_up", "down") if m.gate_up is not None else tuple(  # fp experts, or packed 4-bit ones
            w + part for w in ("gate_up", "down") for part in ("_q", "_scales", "_zeros"))
        per_expert = sum(getattr(m, n)[0].numel() * getattr(m, n).element_size() for n in names)
        total -= (m.num_experts - statistics.mean(hit[m])) * per_expert
    return int(total)


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
    ap.add_argument("--dtype", default=None, help="model dtype (default fp16 on CUDA, fp32 elsewhere)")
    ap.add_argument("--batch", type=int, default=1, help="rows per decode step")
    ap.add_argument("--prompts", default="random", choices=["random", "wikitext"])
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--long-context", type=int, default=16384, help="0 skips the long prefill")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="benchmarks/results/results_decode_profile_cuda.json")
    a = ap.parse_args()
    dev = torch.device(a.device)
    cuda = dev.type == "cuda"
    dtype = a.dtype or ("fp16" if cuda else "fp32")
    if os.path.isfile(os.path.join(a.model, TGATE4_FILE)):  # a directory written by `turing convert`: --settings as-is
        base, tok = load_quantized(a.model, device=a.device, dtype=dtype)
    else:
        base, tok = RealHuggingFaceLoader.load_hf_model_into_turing(a.model, device=a.device, dtype=dtype)[0], None
    g = torch.Generator().manual_seed(0)
    if a.prompts == "wikitext":
        from benchmark_e2e_generation import prompts
        from transformers import AutoTokenizer
        ids = torch.tensor(prompts(tok or AutoTokenizer.from_pretrained(a.model), a.batch, a.context)).to(dev)
    else:
        ids = torch.randint(0, base.config.vocab_size, (a.batch, a.context), generator=g).to(dev)
    long_ids = torch.randint(0, base.config.vocab_size, (1, a.long_context), generator=g).to(dev)
    bw = read_bandwidth_gb_s(dev) if cuda else None
    rows = []
    settings = a.settings.split(",")
    for name in settings:
        model = copy.deepcopy(base) if len(settings) > 1 else base  # one setting: no second copy of the weights
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
        read = moe_read_bytes(model, lambda m=model: steps_wall_ms(m, ids, 8, dev, False), a.batch)
        row = dict(model=a.model, setting=name, weight_bytes=wb, batch=a.batch, context=a.context,
                   prompts=a.prompts, compile_s=compile_s,
                   read_bytes_per_step=read or wb, floor_ms=(read or wb) / 1e9 / bw * 1e3 if bw else None,
                   graph_step_ms=steps_wall_ms(model, ids, a.steps, dev, True) if cuda else None,
                   eager_step_ms=steps_wall_ms(model, ids, a.steps, dev, False))
        if cuda:
            launches, device_ms, top = kernel_profile(model, ids, 16, dev)
            row.update(kernel_launches_per_step=launches, device_ms_per_step=device_ms, top_kernels=top)
        row.update(prefill_s=prefill_s(model, ids[:1], a.chunk, dev), long_context=a.long_context, chunk=a.chunk,
                   long_prefill_s=prefill_s(model, long_ids, a.chunk, dev) if a.long_context else None)
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
