# Ray Serve Deployment Guide

[Ray Serve](https://docs.ray.io/en/latest/serve/index.html) enables multi-node, scalable distributed model serving.

---

## 1. Launching Turing Engine on Ray Serve

```bash
pip install "ray[serve]"
```

```bash
serve run integrations/ray/ray_serve_turing:app
```

## 2. Testing Ray Serve Endpoint

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-70b",
    "messages": [{"role": "user", "content": "Hello from Ray Serve!"}]
  }'
```
