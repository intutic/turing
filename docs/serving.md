# Serving and API Reference

`turing serve` runs one model in a continuous-batching engine (`turing/serving/engine.py`) behind a FastAPI app
(`turing/serving/server.py`). The OpenAI, Anthropic and Ollama endpoints share one generation core
(`turing/serving/generation.py`) that handles the chat template, admission, constrained decoding, reasoning, tool
calls, stop strings and usage.

---

## 1. `turing serve`

```bash
turing serve --model Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --kv-cache-dtype int8
```

Each server serves one model. The request's `model` field does not select weights; only its `:NAME` suffix selects
a LoRA adapter (§8). The main flags by topic are below; `turing serve --help` lists all of them.

**Model and weights**

| Flag | Default | Effect |
| :--- | :--- | :--- |
| `--model` | `smollm2` | HuggingFace repo id, local directory, `.gguf` file, `.tgate4` directory or alias |
| `--device` | `auto` | `auto` picks CUDA, then XPU, then MPS, then Vulkan, then CPU; `rocm` needs PyTorch's HIP build |
| `--dtype` | checkpoint | `bf16`/`fp16`/`fp32`; default: the checkpoint's half type on CUDA, fp32 on CPU and MPS |
| `--quantization` | `none` | `w4a16-rtn` or `w4a16-gptq`: 4-bit weights at load (GPTQ calibrates on WikiText-2; `turing convert` does it once) |
| `--no-quantize-lm-head` | off | With `--quantization`, keep the LM head in the model dtype (it is quantized to 4 bits by default) |
| `--calib-windows` | `128` | GPTQ calibration windows of 2,048 tokens |
| `--activation-sparsity` | `0` | TEAL-style: in batch-1 decode on CUDA, zero this share of every projection's inputs by magnitude and skip their weight rows; prefill and batched decode stay dense. 3–7% faster than 4-bit weights alone at batch 1, at a cost of 5–9 GSM8K points ([benchmarks §3](benchmarks.md#3-4-bit-weights-and-activation-sparsity)) |
| `--sparsity` | `0.0` | FFN channel pruning. `0.0` is the exact model; 50% costs +6.8 to +15.8 nats NLL on SmolLM2 ([benchmarks §10](benchmarks.md#10-ffn-channel-pruning)) |
| `--lora NAME=PATH` | none | Load a PEFT LoRA adapter (repeatable) |
| `--mock` | off | Synthetic weights, for testing without a download |

**KV cache and decode**

| Flag | Default | Effect |
| :--- | :--- | :--- |
| `--kv-cache-dtype` | `auto` | `native`, `int8` (per token and KV head), `int4` (per 32 values) or `fp8` (float8 E4M3, the format of vLLM's fp8 cache); `auto` = `int8` on CUDA, `native` elsewhere. Layers the format damages are detected at startup and kept in the model dtype; int4 also scales down outlier key channels. Quality: [benchmarks §2](benchmarks.md#2-kv-cache-quality-and-the-layers-it-keeps) |
| `--no-kv-smoothing` | off | int4 KV: store keys without outlier-channel smoothing |
| `--kv-page-budget` | `1.0` | Quest: share of 16-token pages each decode step reads, for requests of at least 8,192 tokens. Runs inside CUDA graphs. Off by default: it is slower than reading every page at every length up to 8K ([benchmarks §4](benchmarks.md#4-decode-throughput-against-huggingface-transformers)) |
| `--kv-cache-memory-fraction` | all but a reserve | Share of free GPU memory, after the weights, for the paged KV pool. Unset, the pool takes all of it but a tenth of the GPU's memory (at least 2 GiB, at most half of what is free) |
| `--max-batch-size` | `64` | Requests running (prefilling or decoding) at once |
| `--no-cuda-graphs` | off | Run decode steps eagerly instead of replaying captured CUDA graphs |
| `--no-prefix-caching` | off | Disable automatic prefix caching |
| `--draft-tokens` | `8` | Speculative draft tokens per step, from prompt lookup or from `--draft-head` (`0` = off). A request stops drafting while its drafts do not pay for the rows they add to the decode step ([§5](#5-prefix-caching-and-speculative-decoding)) |
| `--draft-head` | none | An EAGLE-3 draft head for the served model (a Hugging Face repository or a directory with `config.json` and its weights); it drafts instead of prompt lookup |
| `--moe-expert-slots` | all | MoE models that do not fit the GPU: experts kept on it per layer (`6.5` keeps 6 and 7 in turn). The others stay in host memory; a token routed to one is served from there ([§5a](#5a-moe-expert-offload)) |

**API behaviour**

| Flag | Default | Effect |
| :--- | :--- | :--- |
| `--host` / `--port` | `0.0.0.0` / `8000` | Bind address |
| `--api-key` | `$TURING_API_KEY` | Require `Authorization: Bearer <key>` or `x-api-key: <key>` on every endpoint but `/health`; off when unset |
| `--reasoning-effort` | none | Default thinking budget for requests that set none (`none`, `minimal`, `low`, `medium`, `high`) |
| `--no-admission-control` | off | Do not reject requests the KV pool cannot hold soon |
| `--no-lanes` | off | Disable the interactive/batch/background lanes |
| `--embedding-model` | none | HF embedding model for `/v1/embeddings` (default: the served LLM, §6) |
| `--rerank-model` | none | HF cross-encoder for `/v1/rerank` (default: the served LLM, §6) |
| `--enable-kv-events` | off | Publish prefix-cache events for llm-d ([llm-d integration](llmd-integration.md)); also `--kv-events-pub-port` (5556), `--kv-events-replay-port` (5559) |

**Distributed**: `--tensor-parallel`/`--tp`, `--pipeline-parallel`/`--pp`, `--prefill-worker`, `--prefill-server`,
`--kv-port` (§11).

---

## 2. Endpoints

| Path | API |
| :--- | :--- |
| `POST /v1/chat/completions`, `POST /v1/completions` | OpenAI, streaming (SSE) or not |
| `POST /v1/chat/completions/render`, `POST /v1/completions/render` | The exact token ids the generation endpoint would feed the model |
| `POST /v1/embeddings`, `POST /v1/rerank`, `POST /v1/extract` | §6 |
| `GET /v1/models` | The served model |
| `POST /v1/messages` | Anthropic Messages, streaming or not |
| `POST /api/chat`, `/api/generate`, `/api/embed`, `/api/embeddings`, `/api/show`, `/api/pull`; `GET /api/tags`, `/api/ps`, `/api/version` | Ollama |
| `GET /health` | Liveness, running/waiting counts, KV utilisation (never needs the API key) |
| `GET /metrics` | Prometheus text; `?format=json` for the engine telemetry as JSON |

Admission rejections (429/503, §7) use the error shape of the path's API (OpenAI, Anthropic or Ollama). Other errors,
such as request validation (422) and bad inputs to embeddings, rerank and extract (400), use FastAPI's
`{"detail": ...}`.

---

## 3. Chat, completions and messages

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "messages": [{"role": "user", "content": "Write a binary search in Python."}],
  "temperature": 0.6, "top_p": 0.95, "max_tokens": 256, "stop": ["\n\n\n"],
  "stream": true, "stream_options": {"include_usage": true}
}'
```

* **Prompts** are rendered with the model's chat template (the `developer` role becomes `system`; assistant
  `tool_calls` and `tool` messages are passed to the template). Models without a template get a plain fallback.
* **Content parts**: text parts are joined; image and other non-text parts are a 400.
* **Sampling**: `temperature`, `top_p`, `top_k`, `max_tokens` (`max_completion_tokens` on chat). `n > 1` is a 400.
* **Stop**: requests end at the tokenizer's end-of-sequence ids (including chat end markers such as `<|im_end|>`,
  `<|eot_id|>`, `<end_of_turn>`), at `max_tokens`, or at a stop string (OpenAI `stop`, Anthropic
  `stop_sequences`, Ollama `options.stop`); `finish_reason` / `stop_reason` says which.
* **Usage** reports real token counts; `prompt_tokens_details.cached_tokens` is the number of prompt tokens served
  from the prefix cache.
* **Disconnects** cancel the request; its KV pages return to the pool at the next step.

### Reasoning

For models that think in `<think>…</think>` (DeepSeek-R1 distills, Qwen3), the thinking goes to
`message.reasoning_content` (streamed as `delta.reasoning_content`) on the OpenAI API and to `thinking` blocks on the
Anthropic API. A thinking budget is enforced by forcing `</think>` when it runs out:

| Request | Budget |
| :--- | :--- |
| `reasoning_effort: "low"` / `"medium"` / `"high"` | 1,024 / 4,096 / 16,384 thinking tokens |
| `reasoning_effort: "none"` or `"minimal"` | Thinking off where the chat template allows (Qwen3's `enable_thinking`) |
| Anthropic `thinking: {"type": "enabled", "budget_tokens": N}` | N |
| none of these | `turing serve --reasoning-effort`, else unlimited |

### Tools

OpenAI `tools` / `tool_choice` and Anthropic `tools` / `tool_choice` are passed to the chat template. Calls the
model writes in the Hermes/Qwen (`<tool_call>{…}</tool_call>`), Mistral (`[TOOL_CALLS][…]`) or Llama 3 (a bare
`{"name": …, "parameters": …}`) format come back as `tool_calls` (OpenAI, also in stream deltas) or `tool_use`
blocks (Anthropic). `tool_choice: "required"` or a named tool is enforced by a grammar during decoding (needs the
`[structured]` extra).

### Structured output

`response_format: {"type": "json_object"}` or `{"type": "json_schema", "json_schema": {"schema": {…}}}` (and Ollama's
`format`) constrain decoding token by token with [llguidance](https://github.com/guidance-ai/llguidance), then check
the result with `jsonschema`. Install it with `pip install turing-engine[structured]` (from a checkout:
`pip install -e ".[structured]"`). A schema llguidance cannot enforce, a model without a tokenizer, or a server
without llguidance answers 400; there is no prompt-only fallback.

---

## 4. Per-request controls and response headers

Set these per request as a body field or a header:

| Body field | Header | Effect |
| :--- | :--- | :--- |
| `draft_tokens` | `X-Turing-Draft-Tokens` | Speculative draft tokens for this request (`0` = off) |
| `lane` | `X-Turing-Lane` | `interactive`, `batch` or `background` (§7) |
| `model: "base:NAME"` | `X-Turing-Adapter` | LoRA adapter (§8) |

The body fields `use_svd_kv` and `sparsity_ratio` and the headers `X-Turing-SVD-KV`, `X-Turing-SVD-Rank` and
`X-Turing-Sparsity` are not supported; requests that set them get a 400 naming the load-time flag to use instead.

Generation responses report what the server did in these headers: `X-Turing-Model`, `X-Turing-Device`, `X-Turing-KV-Cache-Dtype`,
`X-Turing-Cached-Tokens`, `X-Turing-Draft-Tokens`, `X-Turing-KV-Utilization`, `X-Turing-Queue-Depth`, and
`X-Turing-Lane` / `X-Turing-Adapter` when set.

---

## 5. Prefix caching and speculative decoding

* **Prefix caching** (on by default): every full 16-token page of a finished prefill or request is indexed by a
  chained hash of its tokens and LoRA adapter. A new request reuses the longest cached prefix copy-on-write and
  prefills only the rest; unused cached pages are evicted least-recently-used when the pool runs short. It is off
  on the tensor/pipeline-parallel path and on a prefill worker.
* **Speculative decoding by prompt lookup** (no draft model): the engine drafts the tokens that followed the latest
  earlier occurrence of the request's last 3 (else 2, else 1) tokens. The request's last token and its drafts go
  into the batched decode step as consecutive rows on its cache, so one step verifies the drafts of every request
  next to the plain rows of the others. Greedy requests accept on argmax; sampled requests accept a draft token with
  its probability and otherwise sample from the rest, so the output distribution is that of plain decoding. A
  greedy output can still differ from an undrafted run where two tokens' logits tie within fp16 rounding, as it can
  between any two batch sizes: 7 of 60 chat requests and 0 of 60 rewrite requests in
  `results_spec_divergence_cuda.json`, each at a token whose two candidates were the model's top two and at most
  0.03 apart. It is skipped for requests with a logits processor and on the distributed path; `--draft-tokens 0`
  turns drafting off.
  With Quest page selection or LongRoPE, drafts are verified in a forward per request instead, and for those
  `turing serve` installs a `SpeculationGatePolicy` (`turing/serving/spec_gate.py`): it drafts while 2 or fewer
  requests decode and stops at 4 or more (hysteresis between).
* **Drafting only while it pays.** Draft rows make a step longer. The engine times decode steps per batch size (the
  decode graph's row bucket) and keeps a moving average, over about 16 steps, of the tokens each request gets per
  step with drafts; after 16 such steps, a request whose average is no more than the step's time over a plain
  step's time for the same requests decodes plainly for the next 64 tokens, then tries again. (Accepted drafts come
  in runs: a shorter average rested chat requests through a third of their steps although drafting doubled their
  speed.) A request for which no draft is found eight steps in a row rests the same way. Draft rows are capped so
  that a step fits the largest decode graph (128 rows).
* **EAGLE-3 draft head** (`--draft-head`, `turing/models/eagle3.py`): a published one-layer head that predicts the
  served model's next tokens from its hidden states after three layers. It drafts a chain of up to `--draft-tokens`
  tokens after every target forward, verified by the same rule, so outputs keep the target's distribution. The
  head's layers run once per draft position over the rows of every drafting request; on CUDA its keys and values
  live in a paged pool of their own and each head step is a replay of a CUDA graph. The steps of a chain are queued
  one behind the other, and when every request being served drafts (greedy requests, no prompt waiting), the step
  that verifies the drafts is queued behind the chain while they are still on the device: the host reads the
  drafts and the target's picks together when that step is done, and the GPU goes from the chain to the target and
  back without waiting for the scheduler in between. Each draft costs a forward of
  the head whether or not the target gets to it, so a request's chain is as long as its drafts are likely to be
  reached: with a share p of its recently checked drafts accepted, it drafts while p to the power of the draft's
  place in the chain is at least 0.15. Heads in the
  EAGLE authors' layout and in the speculators library's are loaded; variants with more than one layer or extra
  norms are refused. No draft tree, and no head is trained here.

## 5a. MoE expert offload

`--moe-expert-slots N` serves an MoE model whose experts do not all fit on the GPU. Every expert stays in host memory
(pinned) and `N` per layer are on the GPU; a fraction is an average over the layers. The weights are loaded into
host memory first, so the whole model never has to fit on the GPU. What happens when a token is routed to an expert
that is not on the GPU depends on the experts:

* **4-bit experts** (`turing convert`), decode steps of up to 8 rows: the missing experts are computed on the CPU
  from the packed weights in host memory (`csrc/turing_w4_expert_cpu.hpp`: AVX2, AVX-512 where the CPU has it, a
  thread pool that stays alive) while the GPU runs the resident ones, and nothing is copied. Which experts are
  resident follows use: every 64 such steps a layer brings in the expert it missed most in place of the resident it
  used least. With the Marlin extension built the resident experts are in Marlin's layout, and the decode step is
  captured as a chain of CUDA graphs with a stop for the host at each layer (`StepCuts`), so only the routing and
  the CPU's part run from Python.
* **Experts in the model dtype**, and 4-bit experts without the native extension: the missing expert is copied into
  the slot used least recently, and the grouped MoE kernels read the slots.
* **Larger inputs** (prefill chunks, more than 8 rows): the missing experts pass through the GPU one at a time
  without changing which ones are resident.

Limits: the CPU path costs one pass over an expert's weights in host memory per missing expert and step, so it suits
one or two concurrent requests, and a prefill streams every missing expert of every layer. The engine's telemetry
(`get_telemetry`) reports under `moe_offload` the expert uses served from the GPU (`hits`), computed on the CPU
(`cpu`, with `cpu_seconds`), copied in (`misses`) and streamed.

---

## 6. Embeddings, rerank and extraction

* **`/v1/embeddings`, `/api/embed`, `/api/embeddings`**: with `--embedding-model` (a BERT-style model, its own
  pooling), else the served LLM's mean-pooled final hidden states. Vectors are L2-normalised; `dimensions` truncates
  and renormalises; `encoding_format: "base64"` works; usage counts real tokens.
* **`/v1/rerank`** (Cohere/Jina result shape): with `--rerank-model`, the cross-encoder's sigmoid score; else the
  served LLM's probability of answering "yes" when asked whether the document is relevant. Results are sorted best
  first; `meta.scorer` says which scorer ran.
* **`/v1/extract`**: the served LLM extracts spans for the given `labels` as JSON (greedy); every returned span is
  located in the text with `start`/`end`, spans the text does not contain are listed under `unmatched`, and output
  that is not JSON is a 422 carrying the raw output.

```bash
curl http://localhost:8000/v1/rerank -H "Content-Type: application/json" -d '{
  "query": "What does the KV cache store?",
  "documents": ["Keys and values of past tokens.", "A list of HTTP routes."], "top_n": 1}'
```

---

## 7. Admission control and lanes

Both are on by default.

* **Admission** (`AdmissionController`, sized to the KV pool): the prompts of the admitted requests may fill 90% of
  the pool; a request beyond that gets 429 (`Retry-After: 2`), beyond 95% 503 (shed, no `Retry-After`), and 503 with
  `Retry-After: 1` when too many requests already wait for pages. A request whose prompt plus `max_tokens` can never
  fit gets 400. Streaming endpoints admit before sending the 200.
* **KV pages on demand.** A request holds the pages of its prompt when it is admitted and takes the pages of its
  output as it decodes, 32 tokens at a time, so a large `max_tokens` does not keep other requests out. If the pool
  runs out while requests decode, the newest request of the least urgent lane gives its pages back, waits at the
  head of the queue and has its cache recomputed from its prompt and the tokens it already produced (recompute
  preemption, as in vLLM): its stream pauses and then continues where it stopped. `/metrics` counts these as
  `turing_kv_preemptions_total`. A request that continues a caller's KV cache (`fork`, a remote prefill) reserves
  everything up front and is never preempted.
* **Lanes** (`LanePolicy`): `interactive`, `batch` and `background`, chosen by `lane` / `X-Turing-Lane` or by the
  request's shape. Waiting requests are ordered by lane, and each lane has its own prefill budget per scheduler step (2,048, 1,024
  and 512 prompt tokens, over all waiting requests). A prompt that fits a step's budget is not split over two
  steps while the step holds whole prompts only (such a step can be replayed from a CUDA graph): it waits for the
  next one if the rest of this step's budget is too small. A longer prompt takes what is left of each step. One
  forward takes at most 2,048 prompt tokens: an engine built with a larger budget spends it over several scheduler
  steps, and the requests already decoding take their next token with the last of them, as rows of that forward.
  How many rows a forward that cuts a prompt takes (2,040 or 2,048, those rows included or on top) is measured for
  the model on the GPU at startup: cuBLAS picks its kernels by the row count. Steps that prefill are launched
  ahead like decode steps: the next forward is queued before the host has read the tokens of the one in flight. `/metrics` reports waiting, running, admitted and finished requests per lane.

---

## 8. Multi-LoRA

```bash
turing serve --model meta-llama/Llama-3.2-1B-Instruct --lora sql=./adapters/sql --lora code=./adapters/code
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.2-1B-Instruct:sql", "messages": [{"role": "user", "content": "Count active users."}]}'
```

PEFT LoRA adapters (Llama-style layouts and Phi-3's fused projections, rsLoRA scaling) wrap the targeted attention
and MLP projections, dense or 4-bit. All loaded adapters stay resident, and one batched decode step serves base and
adapter requests together, each row with its own adapter (CUDA graphs are skipped for steps with adapters). The
prefix cache keys pages by adapter. Logits are tested against PEFT for Llama, Qwen2, Phi-3 and Gemma 2. An adapter
whose targets the loader cannot map is refused.

---

## 9. Metrics

`GET /metrics` returns Prometheus text (format 0.0.4); `?format=json` returns the telemetry as JSON. Exported
metrics: `turing_serving_throughput_tok_per_sec`, `turing_total_tokens_generated`, `turing_running_requests`,
`turing_waiting_queue_depth`, `turing_ttft_avg_ms`, `turing_ttft_p99_ms`, `turing_itl_avg_ms`,
`turing_num_requests_waiting`, `turing_num_requests_running`, `turing_kv_cache_usage_perc`,
`turing_cache_config_info` (labels `block_size`, `num_gpu_blocks`), with prefix caching on `turing_prefix_cache_hit_rate`,
`turing_prefix_cache_hit_tokens_total`, `turing_prefix_cache_queried_tokens_total`, `turing_prefix_cache_pages`,
`turing_spec_drafted_tokens_total`, `turing_spec_accepted_tokens_total`, `turing_spec_acceptance_rate`,
`turing_lane_requests_{waiting,running,admitted_total,finished_total}` (label `lane`), and with admission control
`turing_vram_utilization_ratio` and `turing_admission_shed_total`. Latency percentiles are `NaN` until there are two
samples.

---

## 10. Ollama API

`/api/chat` and `/api/generate` take Ollama's `options` (`temperature`, `top_p`, `top_k`, `num_predict`, `stop`)
and `format`. `/api/tags`, `/api/show` and `/api/ps` report the served model's real family, parameter count,
quantization (F32/F16/BF16 or Q4) and, for `/api/ps`, the memory held by the weights and KV pool. `/api/pull`
downloads a HuggingFace repository (`org/name` or `hf.co/org/name[:revision]`) into the local HuggingFace cache and
streams Ollama progress lines; it does not change the model being served.

---

## 11. Distributed serving

**Tensor and pipeline parallelism** run under torchrun; `--tp × --pp` must equal `WORLD_SIZE`:

```bash
./scripts/launch_distributed.sh Qwen/Qwen2.5-7B-Instruct 2 2   # MODEL TP PP
```

Tensor parallelism shards attention heads, MLP and expert widths and the vocabulary across ranks with all-reduces;
pipeline stages own contiguous layers and their KV pages and pass activations with send/recv, with forward-only 1F1B
micro-batching. Rank 0 runs the HTTP server and drives the steps; the other ranks run the worker loop.

**Limitation:** the test suite covers TP 2, TP 4, PP 2 and TP 2 × PP 2 as CPU (gloo) processes, comparing logits
with a single process (atol 1e-4). Multi-GPU (NCCL) runs have not been validated.

**Prefill/decode disaggregation** over TCP:

```bash
turing serve --model Qwen/Qwen2.5-0.5B-Instruct --prefill-worker --kv-port 5600            # prefill node, no HTTP
turing serve --model Qwen/Qwen2.5-0.5B-Instruct --prefill-server prefill-host:5600 --port 8000
```

The decode server sends each prompt to the prefill worker, which returns the first token and the request's KV pages
in the pool's storage format (native, int8 or int4 with scales, or fp8). In the tests, which run both sides as processes on
one CPU host, greedy output matches a single engine.

---

## 12. Standalone C++ server (`turing-cli serve`)

```bash
cmake -B build && cmake --build build
./build/turing-cli serve --model ./models/llama-3.2-1b-instruct-f16.gguf --host 127.0.0.1 --port 8000 --ctx 4096
```

A single-threaded server with `POST /v1/chat/completions` (messages rendered with the GGUF's chat template: ChatML,
Llama 3, Gemma or `[INST]`, else `role: content` lines), `POST /v1/completions` and `GET /health`. It runs
Llama-layout models on the CPU (AVX2 or NEON) with an optional int8 KV cache. See
[architecture §9](architecture.md#9-native-c-engine-turing-cli).
