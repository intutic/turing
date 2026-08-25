# OpenAI & Anthropic API Reference

Turing Engine includes a dual API gateway listening on port 8000.

---

## 1. OpenAI Chat Completions (`POST /v1/chat/completions`)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-70b",
    "messages": [
      {"role": "system", "content": "You are a fast AI assistant."},
      {"role": "user", "content": "Explain Turing Engine."}
    ],
    "temperature": 0.7,
    "max_tokens": 128,
    "stream": false
  }'
```

---

## 2. Anthropic Messages (`POST /v1/messages`)

```bash
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: turing-local" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Hello from Anthropic client!"}
    ]
  }'
```
