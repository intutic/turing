# Changelog

Releases before 1.6.0 are listed only as git tags (v0.1.0 to v1.5.0).

## Unreleased (2.0.0)

### Fixed (c547190)
- CLI: every `.gguf` path crashed with `UnboundLocalError` (missing module-level `import os`). `chat`, `serve` and
  `generate` share one weight loader, so GGUF models also get the KV cache flags. `--max-tokens` alias added;
  `--block-size` removed (KV events use the pool's 16-token pages).
- Speculative decoding: candidates, entropy and argmaxes cross to the host once per draft instead of once per tree
  node. The target's correction token no longer counts as an accepted draft token (a test had asserted 100%
  acceptance for 3 of 4 matches).
- C++ engine: `forward_token` throws past `max_pos` instead of reading beyond the KV cache; the embedded HTTP server
  answers 400 instead of crashing.
- Serving: rejected requests return 503 with `Retry-After` (KV pool full, admission shedding), 429 (admission queue
  full) or 400 (can never fit), with OpenAI, Anthropic or Ollama error bodies. Streaming endpoints admit a request
  before sending the 200.
- Telemetry: TTFT and inter-token latency are stamped after the forward pass and sampling; percentiles are null with
  fewer than two samples instead of reporting the mean; Prometheus prints NaN for them.
- `--device rocm` maps to the HIP build's `cuda` device, or explains why it cannot.

### Removed (c547190)
- The sparse prefill kernel, which was never launched.
- The Qwen 2.4T guide and its two scripts. They published a GCP project ID and a VM IP, and the model needs Blackwell
  FP4 on many GPUs. A test now rejects tracked files containing public IPs or GCP project IDs.

### Also since 1.7.0
- Expert-cache eviction fixes, a 1 MiB / queue-depth weight reader and the cold-ingest ladder (57bf57e).
- Benchmark claims audited; unmeasured numbers removed from the docs and scripts (53ddf5a, 0281443).

## 1.7.0 (2026-11-16, c80e7d2)

- Paged KV pool: one pool per model, sized once from free GPU memory, a device page table per request (at most one
  partial page wasted), copy-on-write forks. The engine reserves every page a request can need at admission and
  rejects what cannot fit (7f38fdd).
- Decode on CUDA: a fused quantize-and-append, split-K attention over each request's pages (optionally a Quest page
  list selected for the whole batch on the device) and a reduction; the decode step is captured as a CUDA graph per
  batch bucket and replayed.
- Pipeline stages and the distributed driver run on the new cache; the `(k, v)` tuple cache path is removed.
- HF parity for Qwen2/Qwen2.5 (q/k/v biases), Qwen3 (per-head q/k RMSNorm) and Llama-3.1 RoPE scaling; the loader
  refuses o_proj biases, sliding-window attention and unsupported RoPE scaling instead of serving wrong logits.
  `serve`, `generate` and `chat` exit when weights fail to load; random weights only with `--mock` (2cec292).
- Measured on an NVIDIA L4: SmolLM2-360M decodes 162 tok/s at batch 1 (HuggingFace transformers: 29) and 1,715 at
  batch 64 (827). int8 and int4 KV hold 1.87x and 3.17x the tokens of the native cache.

## 1.6.0 (2026-11-15, f936008)

- **Real-weight results before 1.6.0 came from a broken model.** From v0.1.0 (2026-08-25) until b9eafd7
  (2026-11-15) the HuggingFace loader never copied the per-layer RMSNorm weights (they stayed at 1.0), and under
  transformers 5 it read the wrong RoPE base (it fell back to 10000 instead of reading `rope_parameters`). b9eafd7
  fixes both, honours `head_dim` and `rms_norm_eps`, runs RMSNorm in fp32, and tests logits against HuggingFace
  through prefill, chunked prefill and decode (max difference 1.8e-7).
- KV cache with native, int8 and int4 storage and optional Quest page selection; SDPA with native GQA. On CUDA one
  fused append, split-K attention and reduce launch per layer decodes every request in the batch.
- Defaults: `--sparsity 0` and recirculation off (50% pruning measured at +6.8 to +15.8 nats NLL on SmolLM2);
  `--kv-cache-dtype auto` (int8 on CUDA).
- Quality and throughput harnesses: `scripts/eval_kv_quality.py` (paired NLL on WikiText-2) and
  `scripts/benchmark_kv_decode.py` (batched decode against HuggingFace transformers), with L4 and M4 Pro results
  (e3b18be).
- Docs quote measurements only. Removed: "75% less memory", SVD KV "-75% (2.5GB)", "57% pruning with zero accuracy
  loss", the unsourced GSM8K/HumanEval/MMLU table, NIAH 1M-token retrieval and the formula-generated "2.57x vLLM"
  serving table.
