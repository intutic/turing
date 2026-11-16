# 🔌 Ecosystem Integrations

Turing Engine integrates with agent frameworks, model routers, and deployment runtimes.

---

## 1. LangChain & LangGraph

Turing Engine provides native drop-in support via `langchain-openai` or built-in `ChatTuring`:

=== "Method A: Drop-In langchain-openai (Zero Extra Dependencies)"
    ```python
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="turing-local",
        model="deepseek-r1-7b"
    )
    response = llm.invoke("Explain quantum computing in simple terms:")
    print(response.content)
    ```

=== "Method B: ChatTuring (a LangChain BaseChatModel, with tool calling)"
    ```python
    from turing.integrations.langchain import ChatTuring  # needs langchain-core

    llm = ChatTuring(model="deepseek-r1-7b", base_url="http://localhost:8000/v1")
    print(llm.invoke("Write a quick C++ SIMD routine:").content)
    ```
    `ChatTuring` supports `invoke`, `stream`, `batch` and `bind_tools` (OpenAI-style tool calls returned as
    `AIMessage.tool_calls`). Pruning, KV cache format and quantization are chosen when the server starts, not per request.

---

## 2. LiteLLM Proxy & Router

Configure Turing Engine as a backend in your `litellm` `config.yaml`:

```yaml
model_list:
  - model_name: turing/deepseek-r1
    litellm_params:
      model: openai/deepseek-r1-7b
      api_base: http://localhost:8000/v1
      api_key: turing-local
      extra_headers:
        X-Turing-Draft-Tokens: "4"   # optional: speculative draft tokens per step
```

Requests carrying the removed `X-Turing-Sparsity` / `X-Turing-SVD-*` headers get HTTP 400.

---

## 3. LlamaIndex RAG & Agents

LlamaIndex talks to Turing Engine through its OpenAI-compatible API (`pip install llama-index-llms-openai-like`);
there is no separate Turing package. A tested example is in
[`integrations/llamaindex/openai_like_example.py`](https://github.com/intutic/turing/blob/master/integrations/llamaindex/openai_like_example.py).

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="deepseek-r1-7b",
    api_base="http://localhost:8000/v1",
    api_key="turing-local",
    is_chat_model=True
)
response = llm.complete("Summarize the main architectural trade-offs of microservices:")
print(response.text)
```

---

## 4. Open WebUI, Continue.dev & Ollama Frontends

Turing Engine serves the **Ollama** (`/api/*`) and **OpenAI** (`/v1/*`) protocols; the clients below are configured
the way their docs describe for those protocols (we test the protocols, not each client application):

=== "Method A: Native Ollama Protocol (Open WebUI / Continue.dev / Raycast / Enchanted)"
    1. In **Open WebUI**: Go to **Settings ➔ Connections ➔ Ollama API**.
    2. Set **Ollama Base URL**: `http://localhost:8000`
    3. Click **Verify Connection** — all loaded models will automatically sync via `/api/tags`.
    4. In **Continue.dev** (`~/.continue/config.json`):
    ```json
    {
      "models": [
        {
          "title": "Turing Engine DeepSeek-R1",
          "provider": "ollama",
          "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
          "apiBase": "http://localhost:8000"
        }
      ]
    }
    ```

=== "Method B: OpenAI Compatible Protocol (LibreChat / Chatbox / Cursor / Cline)"
    1. Open your chat UI Settings ➔ OpenAI API.
    2. Set **Base URL**: `http://localhost:8000/v1`
    3. Set **API Key**: anything, or the key given to `turing serve --api-key` (keys are not checked unless one is set)
    4. Models will automatically appear in your model selector dropdown.

---

## 5. Kubernetes Deployment (KServe, Helm & llm-d)

Turing Engine provides production manifests for Kubernetes orchestration:

* ☸️ **[Full llm-d & Kubernetes Distributed Serving Guide](llmd-integration.md)** — Prefix-cache aware routing, InferencePools, and P/D disaggregation.
* 📦 **[Turing Serving Helm Chart](https://github.com/intutic/turing/tree/master/deploy/helm/turing-serving)** — deployment with Prometheus scrape annotations (`/metrics` is Prometheus text), GPU limits and optional llm-d KV events.
* 🐳 **[Docker images](docker.md)** — CPU and CUDA images.
* 🚀 **KServe `ServingRuntime`**:

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ServingRuntime
metadata:
  name: turing-runtime
  labels:
    llm-d.ai/engine-type: "turing"
spec:
  supportedModelFormats:
    - name: turing-subspace
      version: "1"
      autoSelect: true
    - name: huggingface
      version: "1"
      autoSelect: false
  containers:
    - name: kserve-container
      image: ghcr.io/intutic/turing:v2.0.0-cuda



      command: ["python3", "-m", "turing.cli", "serve"]
      args: ["--model", "$(MODEL_NAME)", "--port", "8080", "--device", "cuda"]
      resources:
        limits:
          nvidia.com/gpu: "1"
          memory: 32Gi
          cpu: "8"
        requests:
          nvidia.com/gpu: "1"
          memory: 24Gi
          cpu: "4"
```

