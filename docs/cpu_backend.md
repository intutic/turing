# The libllama backend (CPU and Apple GPU)

`turing serve --backend libllama` serves a GGUF file through llama.cpp's libllama instead of Turing's own engine. It
is for machines without an NVIDIA GPU: x86 and ARM CPUs, Apple silicon's CPU or GPU, and MoE models larger than RAM
read from an SSD. The server around it is the same: the OpenAI, Anthropic and Ollama endpoints, chat templates,
reasoning, tool calls and structured output ([Serving & API](serving.md)).

llama.cpp is a submodule pinned at one commit (`third_party/llama.cpp`, MIT; see `NOTICE`). The backend is built only
when asked for, so `pip install turing-engine` does not include it.

---

## 1. Build

```bash
git submodule update --init third_party/llama.cpp
TURING_BUILD_LLAMA=1 pip install -e .                                     # CPU
TURING_BUILD_LLAMA=1 TURING_BUILD_LLAMA_METAL=1 python setup.py build_ext --inplace   # Apple silicon: CPU and GPU
```

The build needs CMake. It applies `third_party/patches/` to the submodule when CMake configures, and on ARM it
fetches ARM's KleidiAI kernels. `python -c "from turing import turing_llama; print(turing_llama.gpu_devices())"`
lists the GPUs a build can use (empty without Metal).

## 2. Serve

```bash
turing serve --backend libllama --model ~/models/google_gemma-4-26B-A4B-it-Q4_K_M.gguf \
  --mmproj ~/models/mmproj-google_gemma-4-26B-A4B-it-f16.gguf --gpu-layers -1 --ctx 65536 \
  --host 127.0.0.1 --port 8000 --served-model-name gemma-4-26b-a4b
```

| Flag | Default | Effect |
| :--- | :--- | :--- |
| `--model` | required | A local `.gguf` file |
| `--ctx` | `8192` | Tokens of context, prompt plus output. A longer request fails with OpenAI's "maximum context length" message, which agents read as a cue to compact |
| `--threads` | all | CPU threads |
| `--gpu-layers` | `0` | Metal build: layers on the GPU, `-1` for all. On a build without a GPU it has no effect and the server says so |
| `--mmproj` | none | The model's vision projector (`mmproj-*.gguf`): chat requests may then carry images as base64 `data:` URLs, one request at a time |
| `--no-repack` | off | Compute from the mapped file instead of ggml's repacked copy of the weights, which needs memory for a second copy. Inkling 2 Mini needs it (llama.cpp aborts with repacking at the pinned commit) |
| `--expert-readahead` | off | MoE models larger than RAM: read the experts each layer's router chose, in parallel, before their products (§4) |
| `--max-batch-size` | `1` | Sequences decoded together |
| `--draft-tokens` | `0` | Prompt-lookup draft tokens for greedy requests, with a gate that stops drafting while drafts do not pay |

The startup line names the device: `Device: gpu MTL0` with layers on the Apple GPU, `Device: cpu` otherwise. A
following turn of a conversation reuses the KV cache of the prompt it shares with the previous one.

Not on this backend: LoRA adapters, a caller's KV cache, admission control and lanes. On recurrent and hybrid models
(Inkling 2, Qwen3-Next, Falcon-H1, Mamba) a prompt that differs from the cached one clears the sequence instead of
truncating it, and prompt-lookup drafts are off.

## 3. `turing-cpu`

The build also makes `turing-cpu` (in `build/temp.*/llama/bin/`), which runs the backend without a server:

```bash
turing-cpu bench --model m.gguf --prompt-tokens 512 --gen-tokens 128 --reps 3 [--seqs 8] [--gpu-layers -1]
turing-cpu generate --model m.gguf --prompt "..." --max-tokens 128 [--draft 4] [--readahead]
```

`bench` decodes random token ids after a random prompt, as `llama-bench` does. For an MoE model larger than RAM that
misses about half the expert bytes real text does and overstated real-text decode about 4x on an M4 Pro: measure such
models with real prompts. `--readahead-adaptive` reads experts only while 5% or more of the chosen experts' bytes miss
the cache (the default reads for every token).

## 4. Measurements

All in `benchmarks/results/results_cpu_*.json` and `bmad-output/cpu-first/plan.md`.

* **Against llama.cpp** on a Neoverse-V2 (GCP c4a-standard-16), Q4_K_M: Qwen3-8B decodes at 39.2 tok/s against
  llama-bench's 38.6, Qwen3-30B-A3B at 65.1 against 67.2. Over HTTP on a c4-standard-16 (Xeon Platinum 8581C),
  Qwen3-8B Q4_K_M, `--max-batch-size 8` against `llama-server -np 8` with equal prompts: 15.4 / 36.3 / 59.1-62.9
  tok/s at 1 / 4 / 8 requests against
  15.6-16.0 / 36.8-37.0 / 63.2-63.5; at 16 requests 90.4-92.5 against 98.2-98.7 (once each). vLLM's CPU backend was
  not measured.
* **Expert read-ahead**, gpt-oss-120b (63 GB) on real text with the page cache capped at 40 / 28 / 20 GB, same VM type,
  2.7 GB/s disk: 10.6 / 5.7 / 3.6 tok/s against plain mmap's 3.6 / 1.2 / 0.74, same text. On an M4 Pro's internal SSD
  (48 GB of RAM): 5.5 tok/s against 4.7.
* **Apple M4 Pro, CPU against GPU**, Q4_K_M, real-text decode: Qwen3-30B-A3B 69.8 / 73.5, Qwen3.6 35B-A3B
  47.4 / 60.4, Gemma 4 26B-A4B 46.8 / 58.9, Inkling 2 Mini 25.5 / 29.6 tok/s; on the GPU 512-token prompts run at
  760-840 tok/s.
* **Answer quality** (`scripts/model_quality`, 76 auto-scored tasks of code, bug fixes, word problems, JSON
  extraction and tool calls, through `turing serve` on the M4 Pro's GPU): reasoning off, Gemma 4 26B-A4B 75,
  Qwen3.6 35B-A3B 74, Qwen3-30B-A3B 70, Inkling 2 Mini 70; reasoning on (medium), 76, 74, 75 and 75.
