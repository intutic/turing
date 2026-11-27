"""
Turing Engine Master Command-Line Interface (CLI).
"""

import argparse
import copy
import os
import sys
import json
import time
import uvicorn
import torch

from turing.config import TuringConfig
from turing.models.registry import get_model_config, MODEL_REGISTRY
from turing.models.causal_lm import SubspaceCausalLM
from turing.core.kv_cache import resolve_kv_cache_dtype
from turing.serving.engine import ContinuousBatchEngine
from turing.serving.spec_gate import SpeculationGatePolicy
from turing.serving.server import create_app

def _load_hf_or_exit(args, dev, dtype=None):
    """Real HuggingFace weights, or stop: random weights served for a mistyped name would only produce gibberish."""
    from turing.models.hf_loader import RealHuggingFaceLoader
    try:
        return RealHuggingFaceLoader.load_hf_model_into_turing(hf_model_id=args.model, sparsity_ratio=args.sparsity,
                                                              device=str(dev),
                                                              dtype=dtype or getattr(args, "dtype", None))
    except Exception as e:
        raise SystemExit(f"[!] Could not load weights for '{args.model}': {e}\n    Pass --mock to run on synthetic weights.")

def _load_weights(args, dev):
    """Real weights from a .tgate4 directory or file, a local .gguf file, or HuggingFace, with --quantization and the
    KV cache flags applied when the command has them."""
    from turing.models.hf_loader import resolve_dtype
    from turing.models.quant import TGATE4_FILE, load_quantized, quantize_model, calibration_windows
    # --moe-expert-slots: the experts must never all be on the GPU, so the weights are loaded in host memory, in the
    # dtype they will have on the device, and only the resident experts move there with the rest of the model
    slots = getattr(args, "moe_expert_slots", None)
    load_dev = torch.device("cpu") if slots else dev
    dtype = getattr(args, "dtype", None)
    if args.model.endswith(".tgate4") or os.path.isfile(os.path.join(args.model, TGATE4_FILE)):
        print(f"[*] Loading 4-bit .tgate4 weights from '{args.model}'...")
        model, tokenizer = load_quantized(args.model, device=str(load_dev),
                                          dtype=resolve_dtype(dtype, dev) if slots else dtype)
        if tokenizer is None:
            raise SystemExit(f"[!] '{args.model}' has no tokenizer files next to its model.tgate4")
    elif args.model.endswith(".gguf") and os.path.isfile(args.model):
        from turing.models.gguf_loader import GGUFModelLoader
        print(f"[*] Loading native GGUF weights from '{args.model}'...")
        model, tokenizer = GGUFModelLoader(args.model).load(sparsity_ratio=args.sparsity, device=str(load_dev))
    else:
        model, tokenizer = _load_hf_or_exit(args, load_dev, dtype or ("bf16" if slots and dev.type == "cuda" else None))
    sp = getattr(args, "activation_sparsity", 0.0)
    q = getattr(args, "quantization", "none")
    if q != "none" and model.config.quantization is None:
        method = q.split("-")[-1]
        print(f"[*] Quantizing weights to 4 bits ({method})...")
        calib = calibration_windows(tokenizer, args.calib_windows, 2048) if method == "gptq" else None
        quantize_model(model, method, calib, lm_head=getattr(args, "quantize_lm_head", False))
    if sp:
        from turing.models.activation_sparsity import calibrate_and_apply
        print(f"[*] Calibrating {sp:.0%} activation sparsity on WikiText-2 (16 windows of 512 tokens)...")
        calibrate_and_apply(model, calibration_windows(tokenizer, 16, 512), sp)
    if slots:
        layers = model.offload_experts(slots)
        print(f"[*] Expert offload: {slots:g} experts per layer on the device, the rest of {layers} MoE layers' "
              f"experts in host memory")
    cfg = model.config
    if hasattr(args, "kv_cache_dtype"):
        cfg.kv_cache_dtype, cfg.kv_page_budget = resolve_kv_cache_dtype(args.kv_cache_dtype, dev), args.kv_page_budget
        cfg.kv_smoothing = not getattr(args, "no_kv_smoothing", False)
    if hasattr(args, "no_cuda_graphs"):
        cfg.cuda_graphs, cfg.kv_pool_fraction = not args.no_cuda_graphs, args.kv_cache_memory_fraction
    return model, tokenizer

def _eval_engine(args):
    """Model and engine for eval-niah / eval-accuracy, with the serving settings under test."""
    dev = TuringConfig(device=args.device).resolve_device()
    model, tokenizer = _load_weights(args, dev)
    if args.kv_select_min_tokens is not None:
        model.config.kv_select_min_tokens = args.kv_select_min_tokens
    return ContinuousBatchEngine(model.config, TuringConfig(device=args.device), model=model, tokenizer=tokenizer,
                                 prefix_caching=False)


def _write_eval(args, res, model=None):
    # as the model runs: a converted (.tgate4) model carries its own quantization, whatever the flags say
    quant = getattr(getattr(model, "config", None), "quantization", None) or args.quantization
    res.update(model=args.model, kv_cache_dtype=args.kv_cache_dtype, kv_page_budget=args.kv_page_budget,
               quantization=quant,
               quantize_lm_head=type(getattr(model, "lm_head", None)).__name__ == "QuantLinear" if model is not None
               else args.quantization != "none" and getattr(args, "quantize_lm_head", False),
               moe_expert_slots=getattr(args, "moe_expert_slots", None),
               activation_sparsity=args.activation_sparsity, device=args.device,
               dtype=getattr(args, "dtype", None), kv_smoothing=not getattr(args, "no_kv_smoothing", False),
               batch_size=getattr(args, "batch_size", None))
    print(json.dumps(res, indent=2))
    if args.out:
        runs = json.load(open(args.out)) if os.path.exists(args.out) else []
        with open(args.out, "w") as f:
            json.dump(runs + [res], f, indent=2)


def main():
    parser = argparse.ArgumentParser(prog="turing", description="Turing Engine: Subspace-Compressed LLM Serving Runtime")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Serve
    serve_parser = subparsers.add_parser("serve", help="Launch OpenAI-compatible FastAPI serving server")
    serve_parser.add_argument("--model", type=str, default="smollm2", help="Model identifier (e.g. gpt2, smollm2, Qwen/Qwen2.5-0.5B, or HF repo ID)")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number")
    serve_parser.add_argument("--device", type=str, default="auto", help="Hardware device (auto, cuda, rocm, cpu, mps)")
    serve_parser.add_argument("--backend", type=str, default="turing", choices=["turing", "libllama"], help="turing: Turing's engine (default); libllama: the CPU backend on llama.cpp for a .gguf file, one request at a time (built with TURING_BUILD_LLAMA=1)")
    serve_parser.add_argument("--ctx", type=int, default=8192, help="--backend libllama: tokens of context, prompt plus output (default: 8192)")
    serve_parser.add_argument("--threads", type=int, default=0, help="--backend libllama: CPU threads (default 0: all)")
    serve_parser.add_argument("--expert-readahead", action="store_true", help="--backend libllama, MoE models larger than RAM: read the experts each layer chose from the file before their products, while >= 5%% of their bytes miss the page cache (gpt-oss-120b with 28 GB of cache: 3.8x faster decode); turns weight repacking off")
    serve_parser.add_argument("--served-model-name", type=str, default=None, help="The model id /v1/models, /health and /api/tags report (default: HF-<--model>; a request may name any model)")
    serve_parser.add_argument("--activation-sparsity", type=float, default=0.0, help="TEAL-style: zero this fraction of every projection's inputs by magnitude (0 = off; costs quality, speeds batch-1 decode on CUDA)")
    serve_parser.add_argument("--quantization", type=str, default="none", choices=["none", "w4a16-rtn", "w4a16-gptq"], help="4-bit weights at load (GPTQ calibrates on WikiText-2 and takes minutes; `turing convert` does it once)")
    serve_parser.add_argument("--quantize-lm-head", action=argparse.BooleanOptionalAction, default=True, help="with --quantization, the LM head in 4 bits too (the largest single read per token); --no-quantize-lm-head keeps it in the model dtype")
    serve_parser.add_argument("--moe-expert-slots", type=float, default=None, help="MoE models that do not fit the GPU: experts kept on it per layer (6.5 = 6 and 7 in turn); the others stay in host memory and are copied in, or for a few tokens of packed 4-bit experts computed on the CPU, when a token is routed to them")
    serve_parser.add_argument("--calib-windows", type=int, default=128, help="GPTQ calibration windows of 2048 tokens (default 128)")
    serve_parser.add_argument("--dtype", type=str, default=None, choices=["bf16", "fp16", "fp32"], help="Weight and activation dtype (default: the checkpoint's bf16/fp16 on CUDA, fp32 on CPU and MPS)")
    serve_parser.add_argument("--sparsity", type=float, default=0.0, help="Subspace channel sparsity ratio; 0 serves the exact model, pruning costs quality (default: 0.0)")
    serve_parser.add_argument("--kv-cache-dtype", type=str, default="auto", choices=["auto", "native", "int8", "int4", "fp8"], help="KV cache storage: native, int8 (per token), int4 (per 32 values), fp8 (E4M3, one scale per layer); auto = int8 on CUDA, native elsewhere (default: auto)")
    serve_parser.add_argument("--no-kv-smoothing", action="store_true", help="int4 KV: store keys without outlier-channel smoothing")
    serve_parser.add_argument("--kv-page-budget", type=float, default=1.0, help="Quest: share of 16-token KV pages each decode step reads; 1.0 reads all (default: 1.0)")
    serve_parser.add_argument("--no-cuda-graphs", action="store_true", help="Run decode steps eagerly instead of replaying CUDA graphs")
    serve_parser.add_argument("--kv-cache-memory-fraction", type=float, default=None, help="Share of free GPU memory (after the weights) for the paged KV pool (default: all of it but a reserve of a tenth of the GPU's memory, at least 2 GiB)")
    serve_parser.add_argument("--max-batch-size", type=int, default=None, help="Continuous batching size (default 64; --backend libllama: 1, one request at a time, or N sequences decoded together)")
    serve_parser.add_argument("--embedding-model", type=str, default=None, help="HF embedding model (e.g. BAAI/bge-small-en-v1.5) for /v1/embeddings; default: the served model's mean-pooled hidden states")
    serve_parser.add_argument("--rerank-model", type=str, default=None, help="HF cross-encoder (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2) for /v1/rerank; default: the served model's yes/no likelihood")
    serve_parser.add_argument("--draft-tokens", type=int, default=None, help="Speculative draft tokens per request and step (0 = off): from prompt lookup, or from --draft-head (default 8; --backend libllama: 0, prompt lookup for greedy requests)")
    serve_parser.add_argument("--draft-head", type=str, default=None, help="EAGLE-3 draft head for this model (a Hugging Face repository or a directory with config.json and its weights): it drafts instead of prompt lookup")
    serve_parser.add_argument("--api-key", type=str, default=os.getenv("TURING_API_KEY"), help="Require this key (Authorization: Bearer or x-api-key) on every endpoint but /health (default: $TURING_API_KEY; off when unset)")
    serve_parser.add_argument("--no-admission-control", action="store_true", help="Do not reject requests the KV pool cannot hold soon (429/503)")
    serve_parser.add_argument("--no-lanes", action="store_true", help="Disable the interactive/batch/background QoS lanes")
    serve_parser.add_argument("--prefill-worker", action="store_true", help="Prefill/decode disaggregation: run as a prefill worker that ships KV pages to decode servers over TCP (no HTTP server)")
    serve_parser.add_argument("--prefill-server", type=str, default=None, metavar="HOST:PORT", help="Prefill/decode disaggregation: prefill every prompt on this --prefill-worker and decode here")
    serve_parser.add_argument("--kv-port", type=int, default=5600, help="TCP port of --prefill-worker (default 5600)")
    serve_parser.add_argument("--no-prefix-caching", action="store_true", help="Disable automatic prefix caching")
    serve_parser.add_argument("--no-warmup", action="store_true", help="CUDA: listen before the kernels are compiled and tuned; the first requests of each shape then wait for that, and so does every request running beside them")
    serve_parser.add_argument("--lora", action="append", default=[], metavar="NAME=PATH", help="Load a PEFT LoRA adapter (repeatable); select it per request with model 'base:NAME' or X-Turing-Adapter")
    serve_parser.add_argument("--enable-kv-events", action="store_true", help="Enable ZeroMQ KV block event publisher for llm-d EPP prefix cache routing")
    serve_parser.add_argument("--kv-events-pub-port", type=int, default=5556, help="ZeroMQ PUB port for live KV cache block events (default: 5556)")
    serve_parser.add_argument("--kv-events-replay-port", type=int, default=5559, help="ZeroMQ ROUTER port for KV cache event replay (default: 5559)")
    serve_parser.add_argument("--mock", action="store_true", help="Run with dry-run synthetic architecture weights for isolated FLOP profiling")
    serve_parser.add_argument("--reasoning-effort", type=str, default=None, choices=["none", "minimal", "low", "medium", "high"], help="Default thinking-token budget for reasoning models when a request sets none (none disables thinking where the chat template allows)")
    serve_parser.add_argument("--tensor-parallel", "--tp", type=int, default=1, help="Tensor parallelism world size (default: 1)")
    serve_parser.add_argument("--pipeline-parallel", "--pp", type=int, default=1, help="Pipeline parallelism stage count (default: 1)")


    # 3. Generate
    gen_parser = subparsers.add_parser("generate", help="Run single-prompt autoregressive text generation")
    gen_parser.add_argument("--model", type=str, default="smollm2", help="Model identifier (e.g. gpt2, smollm2, or HF repo ID)")
    gen_parser.add_argument("--prompt", type=str, default="Artificial intelligence is", help="Input prompt")
    gen_parser.add_argument("--max-new-tokens", "--max-tokens", type=int, default=32, help="Tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    gen_parser.add_argument("--activation-sparsity", type=float, default=0.0, help="TEAL-style: zero this fraction of every projection's inputs by magnitude (0 = off; costs quality, speeds batch-1 decode on CUDA)")
    gen_parser.add_argument("--quantization", type=str, default="none", choices=["none", "w4a16-rtn", "w4a16-gptq"], help="4-bit weights at load (GPTQ calibrates on WikiText-2 and takes minutes; `turing convert` does it once)")
    gen_parser.add_argument("--quantize-lm-head", action=argparse.BooleanOptionalAction, default=True, help="with --quantization, the LM head in 4 bits too (the largest single read per token); --no-quantize-lm-head keeps it in the model dtype")
    gen_parser.add_argument("--moe-expert-slots", type=float, default=None, help="MoE models that do not fit the GPU: experts kept on it per layer (6.5 = 6 and 7 in turn); the others stay in host memory and are copied in, or for a few tokens of packed 4-bit experts computed on the CPU, when a token is routed to them")
    gen_parser.add_argument("--calib-windows", type=int, default=128, help="GPTQ calibration windows of 2048 tokens (default 128)")
    gen_parser.add_argument("--dtype", type=str, default=None, choices=["bf16", "fp16", "fp32"], help="Weight and activation dtype (default: the checkpoint's bf16/fp16 on CUDA, fp32 on CPU and MPS)")
    gen_parser.add_argument("--sparsity", type=float, default=0.0, help="Subspace channel sparsity ratio; 0 serves the exact model, pruning costs quality (default: 0.0)")
    gen_parser.add_argument("--kv-cache-dtype", type=str, default="auto", choices=["auto", "native", "int8", "int4", "fp8"], help="KV cache storage: native, int8 (per token), int4 (per 32 values), fp8 (E4M3, one scale per layer); auto = int8 on CUDA, native elsewhere (default: auto)")
    gen_parser.add_argument("--no-kv-smoothing", action="store_true", help="int4 KV: store keys without outlier-channel smoothing")
    gen_parser.add_argument("--kv-page-budget", type=float, default=1.0, help="Quest: share of 16-token KV pages each decode step reads; 1.0 reads all (default: 1.0)")
    gen_parser.add_argument("--device", type=str, default="auto", help="Hardware device")
    gen_parser.add_argument("--mock", action="store_true", help="Run with dry-run synthetic architecture weights")

    # 4. Convert
    conv_parser = subparsers.add_parser("convert", help="Quantize a HuggingFace checkpoint to 4-bit .tgate4 (GPTQ or RTN), layer by layer")
    conv_parser.add_argument("--model", type=str, required=True, help="HF repo id, alias, or local checkpoint directory")
    conv_parser.add_argument("--output", type=str, required=True, help="Output directory (model.tgate4 + tokenizer)")
    conv_parser.add_argument("--method", type=str, default="gptq", choices=["gptq", "rtn"])
    conv_parser.add_argument("--device", type=str, default="auto", help="Device for the per-layer work")
    conv_parser.add_argument("--calib-windows", type=int, default=128, help="GPTQ calibration windows (default 128)")
    conv_parser.add_argument("--calib-tokens", type=int, default=2048, help="Tokens per calibration window (default 2048)")
    conv_parser.add_argument("--quantize-lm-head", action=argparse.BooleanOptionalAction, default=True, help="the LM head in 4 bits too; --no-quantize-lm-head keeps it in the model dtype")
    conv_parser.add_argument("--symmetric", action="store_true", help="every group's zero point at 8 (GPTQ's sym=true, as Qwen's GPTQ-Int4 checkpoints): the Marlin kernels then read no zero points, at the cost of the levels an asymmetric grid would not waste on a one-sided group")

    # 5. Eval NIAH
    niah_parser = subparsers.add_parser("eval-niah", help="Pass-key needle-in-a-haystack retrieval through the engine on real weights")
    niah_parser.add_argument("--model", type=str, default="smollm2", help="HF repo id, alias, .gguf file or .tgate4 directory")
    niah_parser.add_argument("--context-len", type=int, nargs="+", default=[1024], help="Prompt lengths in tokens (default 1024)")
    niah_parser.add_argument("--depths", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0], help="Needle depths, 0 = start, 1 = end")
    niah_parser.add_argument("--sparsity", type=float, default=0.0, help="Subspace channel sparsity ratio; 0 serves the exact model (default: 0.0)")
    niah_parser.add_argument("--device", type=str, default="auto", help="Hardware device")
    niah_parser.add_argument("--kv-cache-dtype", type=str, default="native", choices=["native", "int8", "int4", "fp8"], help="KV cache storage (default native)")
    niah_parser.add_argument("--no-kv-smoothing", action="store_true", help="int4 KV: store keys without outlier-channel smoothing")
    niah_parser.add_argument("--kv-page-budget", type=float, default=1.0, help="Quest page budget (1.0 reads all pages)")
    niah_parser.add_argument("--kv-select-min-tokens", type=int, default=None, help="Quest only from this many tokens (default: the model config's 8192)")
    niah_parser.add_argument("--quantization", type=str, default="none", choices=["none", "w4a16-rtn", "w4a16-gptq"], help="4-bit weights at load")
    niah_parser.add_argument("--quantize-lm-head", action=argparse.BooleanOptionalAction, default=True, help="with --quantization, the LM head in 4 bits too (the largest single read per token); --no-quantize-lm-head keeps it in the model dtype")
    niah_parser.add_argument("--moe-expert-slots", type=float, default=None, help="MoE models that do not fit the GPU: experts kept on it per layer (6.5 = 6 and 7 in turn); the others stay in host memory and are copied in, or for a few tokens of packed 4-bit experts computed on the CPU, when a token is routed to them")
    niah_parser.add_argument("--calib-windows", type=int, default=128, help="GPTQ calibration windows")
    niah_parser.add_argument("--activation-sparsity", type=float, default=0.0, help="TEAL-style input sparsity (0 = off)")
    niah_parser.add_argument("--dtype", type=str, default=None, choices=["bf16", "fp16", "fp32"], help="Weight dtype")
    niah_parser.add_argument("--out", type=str, default=None, help="Append the result to this JSON file (a list of runs)")

    # 6. Eval Accuracy (GSM8K)
    acc_parser = subparsers.add_parser("eval-accuracy", help="GSM8K test exact match through the engine on real weights (k-shot, greedy)")
    acc_parser.add_argument("--model", type=str, default="smollm2", help="HF repo id, alias, .gguf file or .tgate4 directory")
    acc_parser.add_argument("--samples", type=int, default=None, help="First N GSM8K test problems (default: all 1319)")
    acc_parser.add_argument("--shots", type=int, default=8, help="Few-shot examples, the first k of the train split (default 8)")
    acc_parser.add_argument("--max-new-tokens", type=int, default=256, help="Tokens generated per problem (default 256)")
    acc_parser.add_argument("--batch-size", type=int, default=8, help="Requests in flight at a time (default 8)")
    acc_parser.add_argument("--sparsity", type=float, default=0.0, help="Subspace channel sparsity ratio; 0 serves the exact model, pruning costs quality (default: 0.0)")
    acc_parser.add_argument("--device", type=str, default="auto", help="Hardware device")
    acc_parser.add_argument("--kv-cache-dtype", type=str, default="native", choices=["native", "int8", "int4", "fp8"], help="KV cache storage (default native)")
    acc_parser.add_argument("--no-kv-smoothing", action="store_true", help="int4 KV: store keys without outlier-channel smoothing")
    acc_parser.add_argument("--kv-page-budget", type=float, default=1.0, help="Quest page budget (1.0 reads all pages)")
    acc_parser.add_argument("--kv-select-min-tokens", type=int, default=None, help="Quest only from this many tokens (default: the model config's 8192)")
    acc_parser.add_argument("--quantization", type=str, default="none", choices=["none", "w4a16-rtn", "w4a16-gptq"], help="4-bit weights at load")
    acc_parser.add_argument("--quantize-lm-head", action=argparse.BooleanOptionalAction, default=True, help="with --quantization, the LM head in 4 bits too (the largest single read per token); --no-quantize-lm-head keeps it in the model dtype")
    acc_parser.add_argument("--moe-expert-slots", type=float, default=None, help="MoE models that do not fit the GPU: experts kept on it per layer (6.5 = 6 and 7 in turn); the others stay in host memory and are copied in, or for a few tokens of packed 4-bit experts computed on the CPU, when a token is routed to them")
    acc_parser.add_argument("--calib-windows", type=int, default=128, help="GPTQ calibration windows")
    acc_parser.add_argument("--activation-sparsity", type=float, default=0.0, help="TEAL-style input sparsity (0 = off)")
    acc_parser.add_argument("--dtype", type=str, default=None, choices=["bf16", "fp16", "fp32"], help="Weight dtype")
    acc_parser.add_argument("--out", type=str, default=None, help="Append the result to this JSON file (a list of runs)")

    # 9. PCIe vs Host DRAM Bandwidth Calibration
    bw_parser = subparsers.add_parser("bench-bw", aliases=["doctor"], help="Measure this machine: copy and read bandwidth, matmul rates, and a model's batch-1 decode floor")
    bw_parser.add_argument("--device", type=str, default="auto", help="Hardware device")
    bw_parser.add_argument("--model", type=str, default=None, help="HF repo id: print the least time a batch-1 decode step can take for its weights as stored")
    bw_parser.add_argument("--tune", action="store_true", help="CUDA: measure the kernel crossovers on this GPU and keep them (~/.cache/turing/tuning.json) for later runs")

    # 12. Interactive Subspace Inference Demo
    demo_parser = subparsers.add_parser("demo", help="Run interactive Turing Engine subspace inference demo")
    demo_parser.add_argument("--model", type=str, default="smollm2", help="Model target (smollm2 or gpt2)")
    demo_parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, mps, cuda, cpu)")
    demo_parser.add_argument("--sparsity", type=float, default=0.0, help="Subspace channel sparsity ratio; 0 serves the exact model, pruning costs quality (default: 0.0)")
    demo_parser.add_argument("--prompt", type=str, default="Explain how high-performance subspace inference accelerates LLM token generation in 3 bullet points.", help="Prompt to evaluate")

    # 14. Terminal Interactive Chat
    chat_parser = subparsers.add_parser("chat", help="Start an instant interactive terminal chat session with real weights")
    chat_parser.add_argument("--model", type=str, default="smollm2", help="Model to chat with (e.g. smollm2, gpt2, qwen-2.5-7b, llama-3.1-8b)")
    chat_parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, mps, cpu)")
    chat_parser.add_argument("--activation-sparsity", type=float, default=0.0, help="TEAL-style: zero this fraction of every projection's inputs by magnitude (0 = off; costs quality, speeds batch-1 decode on CUDA)")
    chat_parser.add_argument("--quantization", type=str, default="none", choices=["none", "w4a16-rtn", "w4a16-gptq"], help="4-bit weights at load (GPTQ calibrates on WikiText-2 and takes minutes; `turing convert` does it once)")
    chat_parser.add_argument("--quantize-lm-head", action=argparse.BooleanOptionalAction, default=True, help="with --quantization, the LM head in 4 bits too (the largest single read per token); --no-quantize-lm-head keeps it in the model dtype")
    chat_parser.add_argument("--moe-expert-slots", type=float, default=None, help="MoE models that do not fit the GPU: experts kept on it per layer (6.5 = 6 and 7 in turn); the others stay in host memory and are copied in, or for a few tokens of packed 4-bit experts computed on the CPU, when a token is routed to them")
    chat_parser.add_argument("--calib-windows", type=int, default=128, help="GPTQ calibration windows of 2048 tokens (default 128)")
    chat_parser.add_argument("--dtype", type=str, default=None, choices=["bf16", "fp16", "fp32"], help="Weight and activation dtype (default: the checkpoint's bf16/fp16 on CUDA, fp32 on CPU and MPS)")
    chat_parser.add_argument("--sparsity", type=float, default=0.0, help="Subspace channel sparsity ratio; 0 serves the exact model, pruning costs quality (default: 0.0)")
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    chat_parser.add_argument("--max-new-tokens", "--max-tokens", type=int, default=128, help="Maximum new tokens per response")
    chat_parser.add_argument("--mock", action="store_true", help="Run with mock synthetic weights without downloading")

    # 15. Info
    info_parser = subparsers.add_parser("info", help="Display system, hardware, and Turing Engine runtime information")
    info_parser.add_argument("--device", type=str, default="auto", help="Hardware device")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "demo":
        from turing.experimental.demo.interactive_demo import run_demo
        run_demo(model_name=args.model, device=args.device, sparsity=args.sparsity, prompt=args.prompt)

    elif args.command == "eval-accuracy":
        from turing.serving.accuracy_eval import GSM8KEvaluator, load_gsm8k
        engine = _eval_engine(args)
        problems = load_gsm8k("test")[:args.samples]
        shots = load_gsm8k("train")[:args.shots]
        _write_eval(args, GSM8KEvaluator(engine).evaluate(problems, shots, args.max_new_tokens, args.batch_size),
                    engine.model)

    elif args.command == "chat":
        jcfg = TuringConfig(device=args.device)
        dev = jcfg.resolve_device()
        print("=" * 72)
        print(f"💬 Turing Engine Interactive Terminal Chat (Target: {dev})")
        print("=" * 72)

        if getattr(args, "mock", False) or args.model in ["test-tiny", "mock"]:
            cfg = get_model_config(args.model)
            model = SubspaceCausalLM(cfg).to(dev).eval()
            tokenizer = None
            print(f"[*] Loaded mock architecture: {cfg.name}")
        else:
            model, tokenizer = _load_weights(args, dev)
            print(f"[+] Loaded {model.config.name} (layers: {model.config.num_layers}, hidden: {model.config.hidden_dim})")

        print("\nType your message and press Enter. Type 'exit' or 'quit' to end.\n")
        while True:
            try:
                user_msg = input("User > ").strip()
                if not user_msg:
                    continue
                if user_msg.lower() in ["exit", "quit", "q"]:
                    print("\nGoodbye!")
                    break

                start_t = time.perf_counter()
                if tokenizer is not None:
                    try:
                        messages = [{"role": "user", "content": user_msg}]
                        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    except Exception:
                        formatted = f"User: {user_msg}\nAssistant:"
                    prompt_tokens = tokenizer.encode(formatted)
                else:
                    prompt_tokens = [ord(c) % model.config.vocab_size for c in user_msg]

                out_tokens = model.generate(prompt_tokens, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
                new_tokens = out_tokens[len(prompt_tokens):] if len(out_tokens) > len(prompt_tokens) else out_tokens
                elapsed = time.perf_counter() - start_t
                tok_s = len(new_tokens) / max(1e-4, elapsed)

                if tokenizer is not None:
                    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                else:
                    response_text = "".join([chr(t % 128) if (32 <= (t % 128) <= 126) else f"<{t}>" for t in new_tokens])

                print(f"\nAssistant > {response_text}\n(Generated {len(new_tokens)} tokens in {elapsed*1000:.1f}ms — {tok_s:.1f} tok/s)\n")
            except (KeyboardInterrupt, EOFError):
                print("\nSession ended.")
                break

    elif args.command == "serve" and args.backend == "libllama":
        # the CPU backend on libllama (decision CF1a): a GGUF file, one request at a time, the same HTTP APIs
        if not (args.model.endswith(".gguf") and os.path.isfile(args.model)):
            raise SystemExit("[!] --backend libllama serves a local .gguf file: pass its path to --model")
        torch.set_num_threads(1)  # torch only samples here; its OpenMP team beside ggml's slows each step (llama_engine.py)
        try:
            from turing.serving.llama_engine import LlamaCppEngine
            from turing.models.gguf_loader import GGUFReader
            meta = GGUFReader(args.model, willneed=False).metadata
            engine = LlamaCppEngine(args.model, None, args.served_model_name or os.path.basename(args.model),
                                    metadata=meta, n_ctx=args.ctx, n_threads=args.threads,
                                    draft_tokens=args.draft_tokens or 0, max_batch_size=args.max_batch_size or 1,
                                    **(dict(readahead=True, repack=False) if args.expert_readahead else {}))
        except ImportError as e:
            raise SystemExit(f"[!] The libllama backend is not built ({e}): git submodule update --init "
                             f"third_party/llama.cpp, then TURING_BUILD_LLAMA=1 pip install -e .")
        print(f"[+] libllama CPU backend: {engine.llm.desc}, {engine.llm.n_ctx} tokens of context")
        app = create_app(engine, embedding_model=args.embedding_model, rerank_model=args.rerank_model,
                         api_key=args.api_key, reasoning_effort=args.reasoning_effort, admission=False, lanes=False,
                         warmup=not args.no_warmup)
        print(f"[*] Starting Turing Engine OpenAI Server on http://{args.host}:{args.port} "
              f"(Model: {engine.model_config.name}, Device: cpu, backend: libllama)")
        uvicorn.run(app, host=args.host, port=args.port)

    elif args.command == "serve":
        jcfg = TuringConfig(device=args.device, max_batch_size=args.max_batch_size or 64)
        dev = jcfg.resolve_device()

        model_inst = None
        tokenizer_inst = None

        if getattr(args, "mock", False) or args.model in ["test-tiny", "mock"]:
            cfg = get_model_config(args.model)
        else:
            model_inst, tokenizer_inst = _load_weights(args, dev)
            cfg = model_inst.config
        if args.served_model_name:
            if model_inst is None:
                cfg = copy.copy(cfg)  # a preset is a shared object
            cfg.name = args.served_model_name

        # Multi-GPU / Multi-Node Distributed Driver setup
        # Under torchrun every rank runs this command with the same model; rank 0 serves HTTP, the others run steps.
        tp_size = getattr(args, "tensor_parallel", 1)
        pp_size = getattr(args, "pipeline_parallel", 1)
        dist_driver = None
        if tp_size > 1 or pp_size > 1 or int(os.getenv("WORLD_SIZE", "1")) > 1:
            from turing.serving.distributed import DistributedInferenceDriver, DistributedConfig
            if tp_size * pp_size != int(os.getenv("WORLD_SIZE", "1")):
                raise SystemExit(f"[!] --tp {tp_size} x --pp {pp_size} must equal WORLD_SIZE "
                                 f"{os.getenv('WORLD_SIZE', '1')}: launch with scripts/launch_distributed.sh (torchrun)")
            dist_cfg = DistributedConfig.from_env(tp_size=tp_size, pp_size=pp_size)
            if model_inst is None:
                torch.manual_seed(0)  # synthetic weights must be identical on every rank
                model_inst = SubspaceCausalLM(cfg).eval()
            dist_driver = DistributedInferenceDriver(model=model_inst, config=cfg, dist_config=dist_cfg)
            if dist_driver.rank != 0:
                dist_driver.worker_loop()
                import torch.distributed as torch_dist
                torch_dist.destroy_process_group()
                return
            print(f"[+] Initialized DistributedInferenceDriver (TP: {tp_size}, PP: {pp_size})")

        for spec in args.lora:
            from turing.models.adapters import load_lora_adapter
            name, _, path = spec.partition("=")
            if not path or model_inst is None:
                raise SystemExit(f"[!] --lora takes NAME=PATH and real weights, got '{spec}'")
            load_lora_adapter(model_inst, name, path)
            print(f"[+] Loaded LoRA adapter '{name}' from {path}")
        draft_head = None
        if args.draft_head:
            if model_inst is None:
                raise SystemExit("[!] --draft-head needs real weights")
            from turing.models.eagle3 import load_eagle3
            draft_head = load_eagle3(args.draft_head, cfg.hidden_dim, cfg.num_layers)
            print(f"[+] Loaded EAGLE-3 draft head from {args.draft_head} (reads layers {draft_head.aux_layers})")
        drafts = 8 if args.draft_tokens is None else args.draft_tokens
        engine = ContinuousBatchEngine(cfg, jcfg, model=model_inst, tokenizer=tokenizer_inst, dist_driver=dist_driver,
                                       draft_head=draft_head,
                                       prefix_caching=not args.no_prefix_caching and dist_driver is None
                                       and not args.prefill_worker, draft_tokens=drafts,
                                       spec_gate=SpeculationGatePolicy() if drafts else None,
                                       remote_prefill=(args.prefill_server.rsplit(":", 1)[0],
                                                       int(args.prefill_server.rsplit(":", 1)[1]))
                                       if args.prefill_server else None)
        if args.prefill_worker:  # prefill/decode disaggregation: serve prefills to decode servers over TCP
            from turing.serving.disagg_sidecar import PrefillWorker
            worker = PrefillWorker(engine, args.host, args.kv_port)
            print(f"[*] Prefill worker on {args.host}:{worker.port} (decode servers call add_remote_prefilled_request)")
            worker.serve_forever()
            return

        kv_pub = None
        if getattr(args, "enable_kv_events", False) or os.getenv("ENABLE_KV_EVENTS", "0").lower() in ("1", "true", "yes"):
            from turing.serving.kv_events import KVBlockEventPublisher
            pod_ip = os.getenv("POD_IP", args.host if args.host != "0.0.0.0" else "127.0.0.1")
            pod_port = args.port
            pub_port = getattr(args, "kv_events_pub_port", 5556)
            replay_port = getattr(args, "kv_events_replay_port", 5559)
            kv_pub = KVBlockEventPublisher(
                model_name=cfg.name,
                pod_ip=pod_ip,
                pod_port=pod_port,
                pub_endpoint=f"tcp://*:{pub_port}",
                replay_endpoint=f"tcp://*:{replay_port}",
                block_size=engine.kv_pool.page_size,  # events describe the pool's real 16-token pages
            )
            print(f"[+] Initialized llm-d KVBlockEventPublisher on {kv_pub.pub_endpoint} (topic: {kv_pub.topic})")

        # CUDA: the kernels are compiled, tuned and captured by a few requests before the server listens
        warm = (dev.type == "cuda" and model_inst is not None and dist_driver is None and not args.prefill_server
                and not args.no_warmup)
        app = create_app(engine, kv_publisher=kv_pub, embedding_model=args.embedding_model, rerank_model=args.rerank_model,
                         api_key=args.api_key, reasoning_effort=args.reasoning_effort,
                         admission=not args.no_admission_control, lanes=not args.no_lanes, warmup=warm)
        if warm:
            print("[*] Warming up: compiling and tuning kernels before listening (--no-warmup skips this)")
        print(f"[*] Starting Turing Engine OpenAI Server on http://{args.host}:{args.port} (Model: {cfg.name}, Device: {dev})")
        uvicorn.run(app, host=args.host, port=args.port)
        if dist_driver is not None:
            dist_driver.shutdown()


    elif args.command == "generate":
        jcfg = TuringConfig(device=args.device)
        dev = jcfg.resolve_device()

        if getattr(args, "mock", False) or args.model in ["test-tiny", "mock"]:
            cfg = get_model_config(args.model)
            model = SubspaceCausalLM(cfg).to(dev).eval()
            prompt_tokens = [ord(c) % cfg.vocab_size for c in args.prompt] or [1]
            print(f"[*] Generating from prompt: '{args.prompt}' on {cfg.name} (Mock Synthetic Weights)...")
            output_tokens = model.generate(prompt_tokens, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            out_text = "".join([chr(t % 128) if (32 <= (t % 128) <= 126) else f"<{t}>" for t in output_tokens])
        else:
            model, tokenizer = _load_weights(args, dev)
            cfg = model.config
            prompt_tokens = tokenizer.encode(args.prompt)
            print(f"[*] Generating from prompt: '{args.prompt}' on {cfg.name} ({len(prompt_tokens)} input tokens)...")
            output_tokens = model.generate(prompt_tokens, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            out_text = tokenizer.decode(output_tokens, skip_special_tokens=True)

        print(f"\n[+] Generated Output:\n{out_text}\n")

    elif args.command == "convert":
        from turing.models.converter import convert_checkpoint
        dev = TuringConfig(device=args.device).resolve_device()
        from turing.models.quant import symmetric
        with symmetric(args.symmetric):
            res = convert_checkpoint(args.model, args.output, args.method, str(dev), args.calib_windows,
                                     args.calib_tokens, lm_head=args.quantize_lm_head)
        print(json.dumps(res, indent=2))

    elif args.command == "eval-niah":
        from turing.serving.niah import LongContextNIAHEvaluator
        engine = _eval_engine(args)
        _write_eval(args, LongContextNIAHEvaluator(engine).evaluate_retrieval(args.context_len, args.depths),
                    engine.model)

    elif args.command in ("bench-bw", "doctor"):
        from turing.core.heterogeneous_moe import BandwidthAdaptiveDecider
        from turing.core import probe
        jcfg = TuringConfig(device=args.device)
        dev = jcfg.resolve_device()
        print(f"[*] Calibrating Live Hardware Interconnect & Compute on {dev}...")
        decider = BandwidthAdaptiveDecider(dev)
        print(f"\n[+] Hardware Calibration Results:")
        print(f"    Measured PCIe / Interconnect Bandwidth : {decider.pcie_bandwidth_gb_s:.2f} GB/s")
        print(f"    Measured Host CPU SIMD Throughput      : {decider.cpu_throughput_gflops:.2f} GFLOPs")
        print(f"    Measured GPU Matrix Throughput         : {decider.gpu_throughput_gflops:.2f} GFLOPs")
        accel = dev.type in ("cuda", "mps")
        bw = probe.read_bandwidth_gb_s(dev, 2 if accel else 0.5)
        print(f"    Measured Device Read Bandwidth         : {bw:.1f} GB/s (bounds batch-1 decode)")
        for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)) if dev.type == "cuda" else (("fp32", torch.float32),):
            print(f"    Measured Matmul Rate, {name}             : {probe.matmul_tflops(dev, dt, 4096 if accel else 1024):.2f} TFLOP/s (bounds prefill)")
        if args.model:
            from huggingface_hub import HfApi
            files = HfApi().model_info(args.model, files_metadata=True).siblings
            nbytes = sum(f.size or 0 for f in files if f.rfilename.endswith(".safetensors"))
            if not nbytes:
                raise SystemExit(f"[!] '{args.model}' lists no .safetensors weights")
            floor = probe.decode_floor_ms(nbytes, bw)
            print(f"    {args.model}: {nbytes / 1e9:.2f} GB of weights as stored -> at least {floor:.1f} ms per "
                  f"batch-1 decode step here (at most {1e3 / floor:.0f} tokens/s without speculation)")
        if args.tune:
            if dev.type != "cuda":
                raise SystemExit("[!] --tune measures CUDA kernels")
            values = probe.tune_int4_max_rows(dev)
            print(f"    int4 GEMM against dequantize + cuBLAS, microseconds by rows: {values['int4_crossover_us']}")
            values.update(probe.tune_decode_min_rows(dev))
            print(f"    dense decode GEMMs, milliseconds a pass by rows: {values['decode_rows_ms']}")
            print(f"[+] int4_max_rows = {values['int4_max_rows']} and decode_min_rows = {values['decode_min_rows']} "
                  f"kept in {probe.save_tuning(values)}")

    elif args.command == "info":
        jcfg = TuringConfig(device=args.device)
        dev = jcfg.resolve_device()
        print("================================================================================")
        print("   ⚡ TURING ENGINE RUNTIME ENVIRONMENT & HARDWARE STATUS")
        print("================================================================================")
        from turing import __version__
        print(f"[*] Version          : v{__version__} (Enterprise BSL 1.1)")
        print(f"[*] Hardware Device  : {str(dev).upper()} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else ('Apple Silicon Metal (MPS)' if dev.type == 'mps' else 'Host CPU')})")
        print(f"[*] PyTorch Version  : {torch.__version__}")
        print(f"[*] Registered Models: {len(MODEL_REGISTRY)} frontier model architectures")
        print("================================================================================")

if __name__ == "__main__":
    main()
