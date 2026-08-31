# 🚀 Serving & API Reference

Turing Engine features a production-grade continuous batching engine with dual drop-in compatibility for both **OpenAI** and **Anthropic** API endpoints.

---

## 1. CLI Commands Reference

### Start Serving Server
```bash
turing serve \
  --model deepseek-r1-7b \
  --host 0.0.0.0 \
  --port 8000 \
  --device auto \
  --sparsity 0.57 \
  --max-batch-size 32
```

### Interactive Terminal Chat
```bash
turing chat --model deepseek-r1-1.5b
```

### Single Prompt Generation
```bash
turing generate --model deepseek-r1-1.5b --prompt "Explain quantum entanglement:" --max-tokens 128
```

---

## 2. OpenAI Chat Completions (`/v1/chat/completions`)

Turing Engine provides full standard OpenAI Chat API compatibility:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer turing-local" \
  -d '{
    "model": "deepseek-r1-7b",
    "messages": [{"role": "user", "content": "Write a quick binary search function in Python:"}],
    "temperature": 0.7,
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

Scrapes real-time continuous batching and KV memory metrics:

```bash
curl -H "Accept: text/plain" http://localhost:8000/metrics
```

Key Prometheus metrics exported:
- `turing_serving_throughput_tok_per_sec`: Instantaneous token generation speed
- `turing_total_tokens_generated`: Monotonic token counter
- `turing_num_requests_waiting` & `turing_num_requests_running`: Active and queued batch sizes
- `turing_kv_cache_usage_perc`: Fraction of static KV page pool in active use (0.0 to 1.0)
- `turing_cache_config_info`: Block size tokens (64) and total allocated memory blocks
- `turing_ttft_avg_ms`, `turing_ttft_p99_ms`, `turing_itl_avg_ms`: P99 latencies

---

## 7. Subspace Pruning & Performance Headers

You can dynamically tune compression and routing parameters per-request via HTTP headers:

| HTTP Header | Default | Description |
| :--- | :--- | :--- |
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


