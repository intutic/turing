# Turing Engine: Architecture

The full description is [docs/architecture.md](docs/architecture.md); measured results are in
[docs/benchmarks.md](docs/benchmarks.md) and the committed `results_*.json`. This page maps the repository.

## Request path

```
HTTP (server.py: OpenAI /v1, Anthropic /v1/messages, Ollama /api/*)
  -> generation.py        chat template, admission, tools, reasoning, constrained decoding, stop strings, usage
  -> engine.py            ContinuousBatchEngine: chunked prefill, batched decode, prefix cache, prompt-lookup
                          speculation, LoRA per row, lanes and admission, cancel
  -> causal_lm.py         SubspaceCausalLM: 16 HF model types, RoPE scaling, MoE, 4-bit QuantLinear, CUDA graphs
  -> kv_cache.py          KVPool / KVCache: 16-token pages, native / int8 / int4, copy-on-write, Quest bounds
  -> triton_kv_decode.py  fused append + split-K paged attention on CUDA (batched gather + SDPA elsewhere)
```

## Directories

| Path | Contents |
| :--- | :--- |
| `turing/models/` | `causal_lm.py` (the model), `hf_loader.py` (HF weights, `FAMILIES` allow-list), `gguf_loader.py` / `gguf_tokenizer.py` (llama.cpp files), `quant.py` (GPTQ/RTN, `QuantLinear`, `.tgate4`), `streaming_loader.py` (layer-by-layer `turing convert`), `activation_sparsity.py` (TEAL), `adapters.py` (PEFT LoRA), `tensor_parallel.py`, `architecture_registry.py`, `resolver.py` (repo ids and aliases), `sizing_profiles.py` (offline layer geometries for `--mock` synthetic models; never used for real weights) |
| `turing/core/` | `kv_cache.py` (serving KV cache), `rope.py`; library components outside the serving path: `cross_model_kv.py`, `lineage.py`, `kslot_pooling.py`, `hybrid_memory.py`, `hybrid_attention.py`, `speculation.py` (untrained draft heads), `heterogeneous_moe.py` / `expert_cache.py` / `expert_prefetch.py` (MoE offload), `elastic_memory.py`, `mhc.py`, `paging.py` |
| `turing/serving/` | `server.py`, `generation.py`, `engine.py`, `prefix_cache.py`, `traffic.py` (admission, lanes), `spec_gate.py`, `anthropic_api.py`, `ollama_api.py`, `tools.py`, `structured.py` (llguidance), `reasoning.py`, `embeddings.py`, `extract.py`, `pull.py`, `kv_events.py` (llm-d), `distributed.py`, `kv_transfer.py` / `disagg_sidecar.py` (P/D), `accuracy_eval.py` (GSM8K), `niah.py` (pass-key retrieval) |
| `turing/kernels/` | Triton kernels: `triton_kv_decode.py` (serving decode), `triton_w4a16.py` (int4 GEMM), `triton_sparse_gemv.py` (activation sparsity), and library kernels with CUDA parity tests |
| `turing/dsl/` | `@chain`, `gen`, `fork`, `join`, `select`, `constrain` on the engine |
| `turing/integrations/` | LangChain `ChatTuring` (a `BaseChatModel`) |
| `turing/demo/` | `MultiAgentCoordinator`, `DynamicEnvironmentModel`, `EpistemicUncertaintyGate`, `turing demo` |
| `csrc/` | PyBind11 bindings (`pybind_bindings.cpp`), SIMD kernels (`turing_simd.hpp`, AVX2/NEON), GGUF reader and dequantizer (`turing_gguf_cpp.hpp`), `ParallelReader` (`turing_parallel_reader.hpp`), cuFile loader (`turing_cufile.hpp`), and the standalone engine: `turing_model_cpp.hpp`, `turing_tokenizer_cpp.hpp`, `turing_http_server.hpp`, `turing_main.cpp` (`turing-cli`) |
| `integrations/` | LiteLLM, LlamaIndex (OpenAILike example), RunPod, Ray Serve, Modal, KServe |
| `deploy/` | `Dockerfile` (CPU and CUDA), Helm chart, llm-d manifests |
| `scripts/` | Benchmarks and evals (`eval_parity.py`, `eval_gguf_parity.py`, `eval_kv_quality.py`, `benchmark_kv_decode.py`, `benchmark_weight_formats.py`, `benchmark_e2e_generation.py`, `benchmark_cpp_engine.py`, …), `launch_distributed.sh`, `l4_session.sh` |

## Native code conventions

The C++ extension and `turing-cli` are C++20, built with `-fvisibility=hidden` and LTO (MSVC: `/GL` + `/LTCG`).
`setup.py` picks AVX2/FMA/F16C on x86_64 and NEON on arm64. The standalone engine allocates its weights, KV cache and
scratch buffers 64-byte aligned. A new SIMD intrinsic or Triton kernel needs a parity test against a NumPy or
PyTorch reference (`CONTRIBUTING.md`).
