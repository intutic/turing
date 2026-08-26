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

## 2. OpenAI-Compatible API (`/v1/chat/completions`)

Turing Engine provides full standard OpenAI API compatibility:

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

### Python SDK Client
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="turing-local")
response = client.chat.completions.create(
    model="deepseek-r1-7b",
    messages=[{"role": "user", "content": "Explain SVD KV cache compression in 2 sentences:"}]
)
print(response.choices[0].message.content)
```

---

## 3. Anthropic-Compatible API (`/v1/messages`)

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

## 4. Subspace Pruning & Performance Headers

You can dynamically tune compression and routing parameters per-request via HTTP headers:

| HTTP Header | Default | Description |
| :--- | :---: | :--- |
| `X-Turing-Sparsity` | `0.57` | Fraction of intermediate FFN channels to prune (0.0 to 0.75). |
| `X-Turing-SVD-Rank` | `64` | Rank dimension for singular value KV cache compression (32, 64, 128). |
| `X-Turing-Draft-Tokens` | `4` | Speculative draft token budget per verification step. |
