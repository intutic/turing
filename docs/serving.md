# 🚀 Serving & API Reference

`turing serve` runs one model in a continuous-batching engine (`turing/serving/engine.py`) behind a FastAPI app
(`turing/serving/server.py`). Every generation endpoint, OpenAI, Anthropic and Ollama alike, goes through the same
core (`turing/serving/generation.py`): chat template, admission, constrained decoding, reasoning, tool calls, stop
strings and usage.

---

## 1. `turing serve`

```bash
turing serve --model Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --kv-cache-dtype int8
```

The server serves one model; the `model` field of a request is not used to pick weights (only its `:NAME` suffix
selects a LoRA adapter, §8). Run `turing serve --help` for the full list. The flags by topic:

**Model and weights**

| Flag | Default | Effect |
| :--- | :--- | :--- |
| `--model` | `smollm2` | HuggingFace repo id, local directory, `.gguf` file, `.tgate4` directory or alias |
| `--device` | `auto` | `auto` picks CUDA, then XPU, then MPS, then Vulkan, then CPU; `rocm` needs PyTorch's HIP build |
| `--dtype` | checkpoint | `bf16`/`fp16`/`fp32`; default: the checkpoint's half type on CUDA, fp32 on CPU and MPS |
| `--quantization` | `none` | `w4a16-rtn` or `w4a16-gptq`: 4-bit weights at load (GPTQ calibrates on WikiText-2; `turing convert` does it once) |
| `--calib-windows` | `128` | GPTQ calibration windows of 2,048 tokens |
| `--activation-sparsity` | `0` | TEAL-style: zero this share of every projection's inputs by magnitude. Lossy; on CUDA at batch 1 the projections use sparse GEMV kernels |
| `--sparsity` | `0.0` | FFN channel pruning. `0.0` is the exact model; 50% costs +6.8 to +15.8 nats NLL on SmolLM2 ([benchmarks](benchmarks.md#2-quality-on-real-weights)) |
| `--lora NAME=PATH` | none | Load a PEFT LoRA adapter (repeatable) |
| `--mock` | off | Synthetic weights, for testing without a download |

**KV cache and decode**

| Flag | Default | Effect |
| :--- | :--- | :--- |
| `--kv-cache-dtype` | `auto` | `native`, `int8` (per token and KV head) or `int4` (per 32 values); `auto` = `int8` on CUDA, `native` elsewhere. Measured quality: [benchmarks §2](benchmarks.md#2-quality-on-real-weights) |
| `--kv-page-budget` | `1.0` | Quest: share of 16-token pages each decode step reads, for requests of at least 8,192 tokens. Measured at v1.7.0 to pay only at large batches and long contexts ([benchmarks §3](benchmarks.md#3-kv-cache-memory-and-decode-throughput)); since v2.0 it also runs inside CUDA graphs (not re-measured) |
| `--kv-cache-memory-fraction` | `0.6` | Share of free GPU memory, after the weights, for the paged KV pool |
| `--max-batch-size` | `64` | Requests running (prefilling or decoding) at once |
| `--no-cuda-graphs` | off | Run decode steps eagerly instead of replaying captured CUDA graphs |
| `--no-prefix-caching` | off | Disable automatic prefix caching |
| `--draft-tokens` | `4` | Prompt-lookup speculative draft tokens per step (`0` = off) |

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
`--kv-port` (§10).

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

Errors come back in the shape of the API the path belongs to (OpenAI, Anthropic or Ollama).

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

Body fields or headers:

| Body field | Header | Effect |
| :--- | :--- | :--- |
| `draft_tokens` | `X-Turing-Draft-Tokens` | Speculative draft tokens for this request (`0` = off) |
| `lane` | `X-Turing-Lane` | `interactive`, `batch` or `background` (§7) |
| `model: "base:NAME"` | `X-Turing-Adapter` | LoRA adapter (§8) |

Removed parameters are refused with 400 and a message naming the load-time flag: body fields `use_svd_kv` and
`sparsity_ratio`, headers `X-Turing-SVD-KV`, `X-Turing-SVD-Rank` and `X-Turing-Sparsity`. Earlier versions echoed
them back without changing anything.

Generation responses carry what the server did: `X-Turing-Model`, `X-Turing-Device`, `X-Turing-KV-Cache-Dtype`,
`X-Turing-Cached-Tokens`, `X-Turing-Draft-Tokens`, `X-Turing-KV-Utilization`, `X-Turing-Queue-Depth`, and
`X-Turing-Lane` / `X-Turing-Adapter` when set.

---

## 5. Prefix caching and speculative decoding

* **Prefix caching** (on by default): every full 16-token page of a finished prefill or request is indexed by a
  chained hash of its tokens and LoRA adapter. A new request reuses the longest cached prefix copy-on-write and
  prefills only the rest; unused cached pages are evicted least-recently-used when the pool runs short. It is off
  on the tensor/pipeline-parallel path and on a prefill worker.
* **Speculative decoding by prompt lookup** (no draft model): the engine drafts the tokens that followed the latest
  earlier occurrence of the request's last 3 (else 2, else 1) tokens and verifies them in one forward. Greedy
  requests accept on argmax; sampled requests accept a draft token with its probability and otherwise sample from
  the rest, so the output distribution is that of plain decoding. It is skipped for requests with a logits processor
  and on the distributed path. The engine can take a `SpeculationGatePolicy` (`turing/serving/spec_gate.py`) that
  turns drafting off when many requests decode; `turing serve` does not install one, so `--draft-tokens 0` is the way
  to turn it off. Its speed has not been measured.

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

* **Admission** (`AdmissionController`, sized to the KV pool): a request that cannot be admitted gets 503 (pool
  full or shed) or 429 (queued), with `Retry-After`; a request that can never fit gets 400. Streaming endpoints
  admit before sending the 200.
* **Lanes** (`LanePolicy`): `interactive`, `batch` and `background`, chosen by `lane` / `X-Turing-Lane` or by the
  request's shape. Waiting requests are ordered by lane, and each lane has its own prefill chunk (512, 256 and 128
  tokens). `/metrics` reports waiting, running, admitted and finished requests per lane.

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
prefix cache keys pages by adapter. Logits match PEFT's in the tests (Llama, Qwen2, Phi-3, Gemma 2); an adapter
whose targets the loader cannot map is refused.

---

## 9. Metrics

`GET /metrics` returns Prometheus text (format 0.0.4); `?format=json` returns the telemetry as JSON. Metrics:
`turing_serving_throughput_tok_per_sec`, `turing_total_tokens_generated`, `turing_running_requests`,
`turing_waiting_queue_depth`, `turing_ttft_avg_ms`, `turing_ttft_p99_ms`, `turing_itl_avg_ms`,
`turing_num_requests_waiting`, `turing_num_requests_running`, `turing_kv_cache_usage_perc`,
`turing_cache_config_info` (labels `block_size`, `num_gpu_blocks`), `turing_prefix_cache_hit_rate`,
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
micro-batching. Rank 0 runs the HTTP server and drives the steps; the other ranks run the worker loop. The tests run
TP 2, TP 4, PP 2 and TP 2 × PP 2 as CPU (gloo) processes and compare logits with one process (atol 1e-4). No NCCL or
multi-GPU run has been recorded.

**Prefill/decode disaggregation** over TCP:

```bash
turing serve --model Qwen/Qwen2.5-0.5B-Instruct --prefill-worker --kv-port 5600            # prefill node, no HTTP
turing serve --model Qwen/Qwen2.5-0.5B-Instruct --prefill-server prefill-host:5600 --port 8000
```

The decode server sends each prompt to the prefill worker, which returns the first token and the request's KV pages
in the pool's storage format (native, int8 or int4 with scales). Greedy output equals a single engine in the tests,
which run both sides as processes on one CPU host.

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
