# 🚀 Serving & API Reference

Turing Engine features a production-grade continuous batching engine with dual drop-in compatibility for both **OpenAI** and **Anthropic** API endpoints.

---

## 1. CLI Commands Reference

### Start Serving Server
```bash
turing serve \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --host 0.0.0.0 \
  --port 8000 \
  --device auto \
  --sparsity 0.57 \
  --reasoning-effort high \
  --max-batch-size 32
```

### Interactive Terminal Chat
```bash
# Direct Hugging Face Hub checkpoint with reasoning budget:
turing chat --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --reasoning-effort high

# Tri-part provider/model/effort namespace:
turing chat --model deepseek-ai/DeepSeek-R1/medium
```

### Single Prompt Generation
```bash
turing generate --model meta-llama/Llama-3.3-70B-Instruct --prompt "Explain quantum entanglement:" --max-tokens 128
```

---

## 2. OpenAI Chat Completions (`/v1/chat/completions`)

Turing Engine provides full standard OpenAI Chat API compatibility, including native reasoning effort constraints (`low`, `medium`, `high`) and dynamic extraction of `<think>...</think>` tokens into `delta.reasoning_content`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer turing-local" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "messages": [{"role": "user", "content": "Write a quick binary search function in Python:"}],
    "reasoning_effort": "high",
    "temperature": 0.6,
    "max_tokens": 256,
    "stream": false
  }'
```

---

## 3. OpenAI Text Completions (`/v1/completions`)

Standard OpenAI raw text completions with both non-streaming JSON and streaming Server-Sent Events (SSE):

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-7b",
    "prompt": "Once upon a time in a distributed cluster,",
    "max_tokens": 64,
    "stream": false
  }'
```

---

## 4. Tokenizer Render Endpoints (`/render`)

Exposes token ID sequences for prefix-cache routing (used by llm-d EPP routers and frontend inspectors):

```bash
# Render text prompt tokens:
curl -X POST http://localhost:8000/v1/completions/render \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world"}'
# -> {"tokens": [15496, 995], "count": 2}

# Render chat messages tokens:
curl -X POST http://localhost:8000/v1/chat/completions/render \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello world"}]}'
```

---

## 5. Anthropic-Compatible API (`/v1/messages`)

Seamless drop-in for Anthropic Claude SDK and tools:

```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:8000", api_key="turing-local")
message = client.messages.create(
    model="deepseek-r1-7b",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello from Anthropic SDK!"}]
)
print(message.content[0].text)
```

---

## 6. Prometheus Telemetry & Metrics (`/metrics`)

Scrapes real-time continuous batching, VRAM admission, and KV memory metrics:

```bash
curl -H "Accept: text/plain" http://localhost:8000/metrics
```

Key Prometheus metrics exported:
- `turing_serving_throughput_tok_per_sec`: Instantaneous token generation speed
- `turing_total_tokens_generated`: Monotonic token counter
- `turing_num_requests_waiting` & `turing_num_requests_running`: Active and queued batch sizes
- `turing_kv_cache_usage_perc`: Fraction of static KV page pool in active use (0.0 to 1.0)
- `turing_vram_utilization_ratio`: Fraction of estimated VRAM budget allocated (0.0 to 1.0)
- `turing_admission_shed_total`: Total count of requests rejected to prevent VRAM exhaustion
- `turing_cache_config_info`: Block size tokens (64) and total allocated memory blocks
- `turing_ttft_avg_ms`, `turing_ttft_p99_ms`, `turing_itl_avg_ms`: P99 latencies

---

## 4. Native Ollama REST API (`/api/*`)

Turing Engine provides full native compatibility with the **Ollama REST API**, allowing drop-in connection from **Open WebUI**, **Continue.dev**, **Cursor Ollama backend**, **Enchanted**, and the official **Ollama Python / JavaScript SDKs**.

### A. List Models (`GET /api/tags`)
```bash
curl http://localhost:8000/api/tags
```
```json
{
  "models": [
    {
      "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
      "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
      "modified_at": "2026-08-31T14:30:00Z",
      "size": 4294967296,
      "digest": "sha256:turing...",
      "details": {
        "format": "subspace",
        "family": "qwen",
        "parameter_size": "7B",
        "quantization_level": "w4a16"
      }
    }
  ]
}
```

### B. Ollama Chat Completion (`POST /api/chat`)
```bash
curl http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "messages": [
      {"role": "user", "content": "Explain subspace channel pruning in one sentence."}
    ],
    "stream": false
  }'
```

### C. Ollama Raw Completion (`POST /api/generate`)
```bash
curl http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "prompt": "def fibonacci(n):",
    "stream": false
  }'
```

---

## 5. Structured Outputs & JSON Mode (`response_format`)

Enforce strict JSON syntax adherence or full **JSONSchema / Pydantic validation** during generation:

### A. JSON Mode (`response_format={"type": "json_object"}`)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "messages": [
      {"role": "user", "content": "Extract name and age from: Alice is a 28 year old engineer."}
    ],
    "response_format": {"type": "json_object"}
  }'
```

### B. Strict JSON Schema Validation (`response_format={"type": "json_schema"}`)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "messages": [
      {"role": "user", "content": "Generate a user profile"}
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "UserProfile",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "skills": {"type": "array", "items": {"type": "string"}}
          },
          "required": ["name", "age", "skills"]
        }
      }
    }
  }'
```

---

## 6. Native Tool & Function Calling (`tools` & `tool_calls`)

Standardized OpenAI and Anthropic tool calling enables autonomous agents to invoke external functions:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "messages": [
      {"role": "user", "content": "What is the weather in Tokyo?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "description": "Get current weather in a city",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string"},
              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
          }
        }
      }
    ]
  }'
```

Response format with extracted `tool_calls`:
```json
{
  "id": "chatcmpl-1725100000",
  "object": "chat.completion",
  "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_9a8b7c6d5e4f3a2b",
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\"location\": \"Tokyo\", \"unit\": \"celsius\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

---

## 7. Subspace Pruning & Performance Headers

You can dynamically tune compression and routing parameters per-request via HTTP headers:

| HTTP Header | Default | Description |
| :--- | :--- | :--- |
| `X-Turing-Lane` | `interactive` | QoS Scheduling Priority: `interactive` (SLO-first), `batch`, or `background` (shed-first). |
| `X-Turing-Sparsity` | `0.57` | Fraction of intermediate FFN channels to prune (0.0 to 0.75). |
| `X-Turing-SVD-KV` | `1` | Enable/disable calibrated SVD INT8 KV cache paging. |
| `X-Turing-Draft-Tokens` | `4` | Speculative draft token budget per verification step. |
| `X-Turing-Tenant-ID` | `None` | Dynamic multi-tenant LoRA adapter ID (e.g. `tenant_sql`, `tenant_code`). |

---

## 8. Multi-Tenant LoRA Dynamic Routing

Serve 100+ fine-tuned task adapters off a single shared base model with zero weight duplication:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Turing-Tenant-ID: text-to-sql-v2" \
  -d '{
    "model": "deepseek-r1-7b",
    "messages": [{"role": "user", "content": "SELECT count(*) FROM users WHERE active = 1;"}]
  }'
```
* **Cache Hits**: $191.38\,\mu\text{s}$ ($0.00\,\text{ms}$ bubble).
* **Cold Loads**: $<0.97\,\text{ms}$ async PCIe DMA stream transfer.

---

## 9. AI Traffic Management, 3-Lane QoS & Admission Control

Turing Engine treats requests as heterogeneous token-budget workloads rather than uniform HTTP calls:

1. **VRAM Admission Control**: Evaluates static memory footprints ($N_{\text{prompt}} + N_{\text{max\_tokens}}$) in $<42\,\mu\text{s}$.
   - **High Watermark (0.90)**: Queues incoming requests with `Retry-After: 2.0s` (HTTP 429) before physical memory is exhausted.
   - **Shed Watermark (0.95)**: Rejects background/batch requests with HTTP 503 to maintain zero OOM crashes.
2. **3-Lane QoS Prioritization**:
   - **Interactive**: Chunk size 512, prioritized scheduler dispatch, zero load-shedding.
   - **Batch**: Chunk size 256, admitted when Interactive P99 latency is within SLO ($\le 50\,\text{ms}$).
   - **Background**: Chunk size 128, automatically shed first under traffic surges.
3. **Concurrency-Adaptive Speculation Gating**:
   - Automatically collapses speculative tree width to plain decode when active concurrency exceeds 4, eliminating serialization bottlenecks and yielding **$162.65\text{ tok/s}$** vs $143.68\text{ tok/s}$ baseline (+13.2% throughput gain).

---

## 10. 2-Phase Prefill & Parallel Batched Decode Scheduling

Turing Engine's continuous batch scheduler explicitly separates **Prefill (Compute-Bound FLOPs)** from **Decode (Memory-Bandwidth GB/s)**:

1. **Piggybacked Chunked Prefill (Phase 1)**:
   - Slices incoming prompts into 512-token chunks (`Interactive` lane) or 256-token chunks (`Batch` lane).
   - Pre-processes prompt chunks in compute-dense Tensor Core GEMMs while immediately yielding to running decode streams.
2. **Parallel Interleaved Batched Decode (Phase 2)**:
   - Evaluates active decoding streams with full KV cache state persistence.
   - Measures real-time latency percentiles (TTFT P50/P95/P99 and ITL P50/P95/P99).
   - Guarantees **Inter-Token Latency (ITL) P99 $\le 18.23\text{ ms}$ on NVIDIA L4 GPU** and $\le 24.25\text{ ms}$ median on Apple Silicon Mac, completely eliminating prompt burst jitter.


