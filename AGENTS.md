# AGENTS.md

Guidance for coding agents (and humans) changing this repository. Read [CONTRIBUTING.md](CONTRIBUTING.md) too.

## Build and test

```bash
pip install -e ".[dev]"                 # editable install with test dependencies
python setup.py build_ext --inplace     # rebuild the C++ extension after editing csrc/
python -W error -m pytest -q            # the whole suite; warnings are errors
mkdocs build --strict                   # the docs site; broken links fail the build
python scripts/make_doc_tables.py --check
```

CUDA-only tests skip on machines without an NVIDIA GPU, and tests that need a model download skip when it is not
cached.

## Layout

| Path | Contents |
| :--- | :--- |
| `turing/serving/` | HTTP server (OpenAI, Anthropic, Ollama APIs), generation core, continuous-batching engine, prefix cache, admission control |
| `turing/models/` | `SubspaceCausalLM` (the model, 16 HF model types), HF and GGUF loaders, 4-bit quantization, LoRA, tensor/pipeline parallelism |
| `turing/core/` | Paged KV cache (`kv_cache.py`), RoPE variants; experimental components outside the serving path |
| `turing/kernels/` | Triton kernels: paged decode/prefill, int4 GEMM, sparse GEMV, and their dispatch wrappers |
| `turing/dsl/` | `@chain`, `gen`, `fork`, `join`, `select`, `constrain` |
| `csrc/` | C++20 extension (SIMD, GGUF reader, file loaders) and the standalone `turing-cli` engine |
| `packages/` | Separately published packages: `kv-layer-guard`, `langchain-turing`, `llama-index-llms-turing` |
| `integrations/`, `deploy/` | LiteLLM, Ray, Modal, RunPod, KServe; Dockerfile, Helm chart, llm-d manifests |
| `scripts/` | Benchmarks and evaluations |
| `benchmarks/results/` | The results files the docs' numbers come from |

## Rules

- **Kernels need parity tests.** Every Triton kernel and SIMD routine is tested against a PyTorch reference
  (`atol <= 1e-3`).
- **No mocks for hardware or models.** Tests run real code; integration tests serve the real app in-process (the
  `turing_server` fixture in `tests/conftest.py`).
- **No warnings.** The suite runs with `-W error`; use current PyTorch APIs (`torch.amp.autocast("cuda")`, explicit
  `torch.meshgrid(..., indexing="ij")`).
- **No host syncs in GPU hot paths.** Avoid `.item()`, `.tolist()` and boolean-mask indexing in decode and drafting.
- **Every device.** Code paths must work on CUDA, MPS and CPU, dispatching to the best available implementation.
- **Numbers come from results files.** A benchmark writes JSON to `benchmarks/results/`; the tables in the docs are
  generated from it by `scripts/make_doc_tables.py`, and every number quoted in prose must match a file there.
- **Version** lives in `setup.py`, `pyproject.toml` and `turing/__init__.py`; change all three together.
- **Commits** carry a DCO sign-off (`git commit -s`).
