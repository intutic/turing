# Open WebUI & LibreChat Integration Guide

[Open WebUI](https://github.com/open-webui/open-webui) and [LibreChat](https://github.com/danny-avila/LibreChat) provide rich ChatGPT-style web interfaces that connect directly to Turing Engine.

---

## 1. Connecting Open WebUI

### Option A: Running with Docker
When launching Open WebUI in Docker, set the `OPENAI_API_BASE_URL` to point to the host machine's Turing Engine server:

```bash
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL="http://host.docker.internal:8000/v1" \
  -e OPENAI_API_KEY="turing-local" \
  -v open-webui:/app/backend/data \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

### Option B: Open WebUI Settings UI
1. Navigate to **Admin Panel > Settings > Connections**.
2. Under **OpenAI API**, enter URL: `http://localhost:8000/v1` and Key: `turing-local`.
3. Click **Save**. Your models (`llama-3.1-70b`, `qwen-2.5-72b`, `deepseek-v4-flash-284b`) will appear in the model selector dropdown.
