#!/usr/bin/env python3
"""
End-to-end batch generation, Turing Engine against vLLM on the same GPU with identical requests: B prompts of C tokens
(WikiText-2 windows), each generating exactly N tokens (EOS ignored), all submitted at once; reported: time to finish
the batch, output tokens per second, and time to first token. Turing runs through ContinuousBatchEngine (paged pool,
CUDA graphs, prefix caching off so no request reuses another's pages); vLLM through vllm.LLM.generate (run it in its
own virtualenv: --engine vllm). Greedy decoding on both.

python scripts/benchmark_e2e_generation.py --engine turing --model HuggingFaceTB/SmolLM2-360M --batches 1,8,32,64
python scripts/benchmark_e2e_generation.py --engine vllm   --model HuggingFaceTB/SmolLM2-360M --batches 1,8,32,64

Long context at equal KV memory: run vLLM first (it records the KV budget it took at --gpu-memory-utilization), then
give Turing the same budget with --kv-pool-gib-from that results file, so capacity differs only by bytes per token:
python scripts/benchmark_e2e_generation.py --engine vllm --model Qwen/Qwen2.5-1.5B-Instruct --context 16384 \
    --batches 16,32,64,128 --kv-cache-dtype fp8 --warmup-context 512 --out results_longctx_cuda.json
python scripts/benchmark_e2e_generation.py --engine turing --model Qwen/Qwen2.5-1.5B-Instruct --context 16384 \
    --batches 16,32,64,128 --kv-cache-dtype int4 --kv-pool-gib-from results_longctx_cuda.json --out results_longctx_cuda.json
"""
import argparse
import asyncio
import json
import os
import platform
import sys
import time


def prompts(tokenizer, n, C, seed=0):
    from datasets import load_dataset
    import random
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tokenizer(text).input_ids
    rng = random.Random(seed)
    return [ids[s:s + C] for s in (rng.randrange(0, len(ids) - C) for _ in range(n))]


def run_turing(a, batches):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import torch
    from transformers import AutoTokenizer
    from turing.config import TuringConfig
    from turing.models.hf_loader import RealHuggingFaceLoader
    from turing.serving.engine import ContinuousBatchEngine
    from turing.models.quant import TGATE4_FILE, load_quantized
    load_dev = "cpu" if a.moe_expert_slots else "cuda"  # offload: weights to host memory first (see turing.cli)
    if os.path.isfile(os.path.join(a.model, TGATE4_FILE)):  # a directory written by `turing convert`
        model, tok = load_quantized(a.model, device=load_dev, dtype=a.dtype)
        a.quantization, a.model = model.config.quantization, os.path.basename(os.path.normpath(a.model))
        a.quantize_lm_head = type(model.lm_head).__name__ == "QuantLinear"  # as the file was converted
    else:
        tok = AutoTokenizer.from_pretrained(a.model)
        model, _ = RealHuggingFaceLoader.load_hf_model_into_turing(a.model, device=load_dev, dtype=a.dtype)
    # as turing serve --quantization: GPTQ calibrated on WikiText-2 train windows
    if a.quantization != "none" and model.config.quantization is None:
        from turing.models.quant import calibration_windows, quantize_model
        method = a.quantization.split("-")[-1]
        quantize_model(model, method, calibration_windows(tok, a.calib_windows, 2048) if method == "gptq" else None,
                       lm_head=a.quantize_lm_head)
    if a.moe_expert_slots:
        model.offload_experts(a.moe_expert_slots)
    if a.activation_sparsity:  # as --activation-sparsity: applies to one-row decode on CUDA (batch 1)
        from turing.models.activation_sparsity import calibrate_and_apply
        from turing.models.quant import calibration_windows
        calibrate_and_apply(model, calibration_windows(tok, 16, 512), a.activation_sparsity)
    model.config.kv_cache_dtype, model.config.kv_page_budget = a.kv_cache_dtype, a.kv_page_budget
    model.config.kv_pool_fraction = None  # as turing serve: all free memory but a reserve
    if a.kv_pool_gib:
        model.config.kv_pool_gib = a.kv_pool_gib
    engine = ContinuousBatchEngine(model.config, TuringConfig(device="cuda", max_batch_size=max(batches)), model=model,
                                   tokenizer=tok, prefix_caching=False, draft_tokens=0,
                                   prefill_chunk_size=a.prefill_chunk)
    engine.ride_prefill = not a.no_ride_prefill
    rows = []

    async def batch(ps):
        t0 = time.perf_counter()
        reqs = [await engine.add_request(prompt_tokens=p, max_new_tokens=a.new_tokens, temperature=0.0,
                                         stop_token_ids=set()) for p in ps]
        ttft = []

        async def drain(r):
            n = 0
            async for _ in engine.stream_request(r):
                if n == 0:
                    ttft.append(time.perf_counter() - t0)
                n += 1
            return n
        counts = await asyncio.gather(*[drain(r) for r in reqs])
        return time.perf_counter() - t0, sum(counts), ttft

    async def main():
        await engine.start()
        try:
            for B in batches:  # warm-up at this batch size: CUDA graph capture
                await batch(prompts(tok, B, a.warmup_context or a.context, seed=99))
            for B in batches:
                preempted = engine.preemptions
                secs, n, ttft = await batch(prompts(tok, B, a.context))
                pool = engine.kv_pool
                rows.append(dict(engine="turing", model=a.model, batch=B, context=a.context, new_tokens=a.new_tokens,
                                 seconds=round(secs, 3), output_tok_per_s=round(n / secs, 1),
                                 mean_ttft_s=round(sum(ttft) / len(ttft), 4), kv_cache_dtype=a.kv_cache_dtype,
                                 kv_page_budget=a.kv_page_budget, kv_capacity_tokens=pool.num_pages * pool.page_size,
                                 kv_pool_gib=round(pool.num_pages * pool.page_bytes() / 2 ** 30, 3),
                                 kv_native_layers=list(pool.native_layers), quantization=a.quantization,
                                 # peak device memory of this process: weights, the KV pool and everything else
                                 gpu_peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
                                 if pool.device.type == "cuda" else None,
                                 prefill_chunk=a.prefill_chunk, ride_prefill=engine.ride_prefill,
                                 moe_expert_slots=a.moe_expert_slots,
                                 moe_offload=model.expert_offload_stats(),
                                 preemptions=engine.preemptions - preempted,  # requests whose cache was recomputed
                                 # the engine's own moving average of a whole decode step, by row bucket
                                 engine_step_ms={str(k): round(v * 1e3, 3) for k, v in engine._step_s.items() if v},
                                 quantize_lm_head=a.quantize_lm_head, activation_sparsity=a.activation_sparsity))
                print(json.dumps(rows[-1]), flush=True)
        finally:
            await engine.stop()
    asyncio.run(main())
    return rows, torch.__version__


def vllm_ttft(out):
    """Seconds from a request's arrival to its first token, from vLLM's per-request stats (recorded only with
    disable_log_stats=False): RequestStateStats.first_token_latency in the v1 engine, first_token_time - arrival_time
    in older versions. None when the stats are missing."""
    m = getattr(out, "metrics", None)
    if m is None:
        return None
    if getattr(m, "first_token_latency", 0):
        return m.first_token_latency
    if getattr(m, "first_token_time", None) and getattr(m, "arrival_time", None):
        return m.first_token_time - m.arrival_time
    return None


def run_vllm(a, batches):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    kv = a.vllm_kv_cache_dtype or {"native": "auto", "int8": "fp8", "fp8": "fp8"}[a.kv_cache_dtype]
    skip = [x for x in a.kv_skip_layers.split(",") if x]
    llm = LLM(a.model, dtype=a.dtype or "float16", kv_cache_dtype=kv, enable_prefix_caching=False,
              disable_log_stats=False,  # request metrics (first-token time) are recorded only with stats on
              **({"kv_cache_dtype_skip_layers": skip} if skip else {}),
              max_model_len=a.context + a.new_tokens + 16, gpu_memory_utilization=a.gpu_memory_utilization)
    capacity, kv_gib = None, None
    try:  # KV capacity vLLM allocated: blocks x block size, and its bytes from the model shape
        from transformers import AutoConfig
        cc = llm.llm_engine.vllm_config.cache_config
        capacity = cc.num_gpu_blocks * cc.block_size
        hc = AutoConfig.from_pretrained(a.model)
        D = getattr(hc, "head_dim", None) or hc.hidden_size // hc.num_attention_heads
        if kv in ("auto", "fp8") and not skip:  # bytes known exactly only for these layouts
            per_token = 2 * hc.num_hidden_layers * hc.num_key_value_heads * D * (1 if kv == "fp8" else 2)
            kv_gib = round(capacity * per_token / 2 ** 30, 3)
    except Exception as e:  # recorded, not hidden
        print(f"[!] could not read vLLM's KV capacity: {e}", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=a.new_tokens, ignore_eos=True)
    for B in batches:  # warm-up at every batch size, as for Turing
        llm.generate([{"prompt_token_ids": p} for p in prompts(tok, B, a.warmup_context or a.context, seed=99)], sp,
                     use_tqdm=False)
    rows = []
    for B in batches:
        ps = prompts(tok, B, a.context)
        t0 = time.perf_counter()
        outs = llm.generate([{"prompt_token_ids": p} for p in ps], sp, use_tqdm=False)
        secs = time.perf_counter() - t0
        n = sum(len(o.outputs[0].token_ids) for o in outs)
        ttft = [vllm_ttft(o) for o in outs]
        ttft = [t for t in ttft if t is not None]
        rows.append(dict(engine="vllm", model=a.model, batch=B, context=a.context, new_tokens=a.new_tokens,
                         seconds=round(secs, 3), output_tok_per_s=round(n / secs, 1),
                         mean_ttft_s=round(sum(ttft) / len(ttft), 4) if ttft else None, kv_cache_dtype=kv,
                         kv_capacity_tokens=capacity, kv_pool_gib=kv_gib, kv_skip_layers=skip,
                         gpu_memory_utilization=a.gpu_memory_utilization))
        print(json.dumps(rows[-1]), flush=True)
    import vllm
    return rows, f"vllm {vllm.__version__}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["turing", "vllm"], required=True)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--batches", default="1,8,32,64")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--kv-cache-dtype", default="native", choices=["native", "int8", "int4", "fp8"],
                    help="turing: native/int8/int4; vllm: native (auto) or int8/fp8 (fp8)")
    ap.add_argument("--kv-page-budget", type=float, default=1.0, help="turing: Quest share of pages read per step")
    ap.add_argument("--kv-pool-gib", type=float, default=None, help="turing: KV pool size")
    ap.add_argument("--kv-pool-gib-from", default=None,
                    help="turing: take the KV budget vLLM recorded in this results file (same model and context)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="vllm")
    ap.add_argument("--vllm-kv-cache-dtype", default=None, help="vllm: any kv_cache_dtype (int8_per_token_head, ...)")
    ap.add_argument("--kv-skip-layers", default="", help="vllm: layers kept out of the quantized cache, e.g. 0")
    ap.add_argument("--warmup-context", type=int, default=None, help="prompt length of the warm-up batches")
    ap.add_argument("--moe-expert-slots", type=int, default=None,
                    help="turing: MoE experts kept on the GPU per layer; the rest in host memory")
    ap.add_argument("--prefill-chunk", type=int, default=2048,
                    help="turing: prompt tokens prefilled per scheduler step, over all waiting requests")
    ap.add_argument("--no-ride-prefill", action="store_true",
                    help="turing: decode steps of their own while prompts are prefilled (the engine before riders)")
    ap.add_argument("--quantization", default="none", choices=["none", "w4a16-gptq", "w4a16-rtn"], help="turing")
    ap.add_argument("--calib-windows", type=int, default=32, help="turing GPTQ calibration windows (speed only)")
    ap.add_argument("--quantize-lm-head", action="store_true", help="turing: with --quantization, the LM head too")
    ap.add_argument("--activation-sparsity", type=float, default=0.0, help="turing: TEAL sparsity")
    ap.add_argument("--out", default="benchmarks/results/results_e2e_generation_cuda.json")
    a = ap.parse_args()
    if a.engine == "vllm":
        a.dtype = {"fp16": "float16", "bf16": "bfloat16"}.get(a.dtype, a.dtype)
    batches = [int(b) for b in a.batches.split(",")]
    if a.engine == "turing" and a.kv_cache_dtype == "fp8" or a.engine == "vllm" and a.kv_cache_dtype == "int4":
        raise SystemExit(f"{a.engine} has no {a.kv_cache_dtype} KV cache here")
    if a.kv_pool_gib_from:
        got = [r["kv_pool_gib"] for r in json.load(open(a.kv_pool_gib_from))["rows"]
               if r["engine"] == "vllm" and r["model"] == a.model and r["context"] == a.context and r.get("kv_pool_gib")]
        if not got:
            raise SystemExit(f"no vLLM KV budget for {a.model} at context {a.context} in {a.kv_pool_gib_from}")
        a.kv_pool_gib = max(got)  # the budget vLLM gave its KV cache (fp16 and fp8 runs take the same bytes)
    rows, version = (run_vllm if a.engine == "vllm" else run_turing)(a, batches)
    old = json.load(open(a.out))["rows"] if os.path.exists(a.out) else []
    kvname = (a.vllm_kv_cache_dtype or {"native": "auto", "int8": "fp8"}.get(a.kv_cache_dtype, a.kv_cache_dtype)) \
        if a.engine == "vllm" else a.kv_cache_dtype
    skipped = [x for x in a.kv_skip_layers.split(",") if x] if a.engine == "vllm" else []
    keep = [r for r in old if not (r["engine"] == a.engine and r["model"] == a.model and r["context"] == a.context
                                   and r.get("kv_cache_dtype") == kvname and r.get("kv_skip_layers", []) == skipped
                                   and r.get("kv_page_budget", 1.0) == (a.kv_page_budget if a.engine == "turing" else 1.0)
                                   and r.get("quantization", "none") == (a.quantization if a.engine == "turing" else "none")
                                   and r.get("prefill_chunk", 2048) == (a.prefill_chunk if a.engine == "turing" else 2048)
                                   and r.get("moe_expert_slots") == (a.moe_expert_slots if a.engine == "turing" else None)
                                   and r.get("quantize_lm_head", False) == (a.quantize_lm_head and a.engine == "turing")
                                   and r.get("activation_sparsity", 0.0) == (a.activation_sparsity
                                                                              if a.engine == "turing" else 0.0))]
    env = dict(platform=platform.platform(), runtime=version)
    with open(a.out, "w") as f:
        json.dump(dict(env=env, rows=keep + rows), f, indent=2)
    print(f"[+] wrote {a.out}")


if __name__ == "__main__":
    main()
