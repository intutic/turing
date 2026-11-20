# 🔌 Ecosystem Integrations

Turing Engine integrates with agent frameworks, model routers, and deployment runtimes.

---

## 1. LangChain & LangGraph

Use `langchain-openai` against the OpenAI API, or `ChatTuring` from the `langchain-turing` package:

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

=== "Method B: ChatTuring (langchain-turing, with tool calling)"
    ```bash
    pip install langchain-turing
    ```
    ```python
    from langchain_turing import ChatTuring

    llm = ChatTuring(model="deepseek-r1-7b", base_url="http://localhost:8000/v1")
    print(llm.invoke("Write a quick C++ SIMD routine:").content)
    ```
    `ChatTuring` is a LangChain `BaseChatModel` in its own package
    ([`packages/langchain-turing`](https://github.com/intutic/turing/tree/master/packages/langchain-turing), depends
    only on `langchain-core` and `httpx`): `invoke`, `stream` (with token usage), `batch`, async, and `bind_tools`
    (OpenAI-style tool calls returned as `AIMessage.tool_calls`). It passed LangChain's standard chat-model test suite
    against `turing serve` with Qwen2.5-0.5B-Instruct on an Apple M4 Pro (8ecc6f8; the run is not committed). `turing.integrations.langchain.ChatTuring` re-exports it.
    Pruning, KV cache format and quantization are chosen when the server starts, not per request.

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

`llama-index-llms-turing` ([`packages/llama-index-llms-turing`](https://github.com/intutic/turing/tree/master/packages/llama-index-llms-turing))
is LlamaIndex's `OpenAILike` client with Turing's defaults: chat endpoint, tool calling on, `api_base` from
`$TURING_API_BASE` (else `http://localhost:8000/v1`).

```bash
pip install llama-index-llms-turing
```

```python
from llama_index.llms.turing import Turing

llm = Turing(model="deepseek-r1-7b", context_window=32768)
response = llm.complete("Summarize the main architectural trade-offs of microservices:")
print(response.text)
```

Set `context_window` to the served model's. `OpenAILike(api_base=..., is_chat_model=True)` from
`llama-index-llms-openai-like` works the same way without the package.

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
* 🚀 **KServe `ServingRuntime`**: [`integrations/kserve/serving_runtime.yaml`](https://github.com/intutic/turing/blob/master/integrations/kserve/serving_runtime.yaml)
  runs `turing serve --model /mnt/models` on port 8080, so an InferenceService's `storageUri` (`hf://`, `s3://`,
  `gs://`, `pvc://`) is what it serves ([`inferenceservice.yaml`](https://github.com/intutic/turing/blob/master/integrations/kserve/inferenceservice.yaml)).
  It uses the published `ghcr.io/intutic/turing:v2.0.0-cuda` image. The manifests were checked by hand against KServe's CRD schemas (not in CI); they have not been run on a cluster yet.

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: qwen25-1-5b-instruct
spec:
  predictor:
    model:
      modelFormat:
        name: huggingface
      runtime: turing-runtime
      storageUri: hf://Qwen/Qwen2.5-1.5B-Instruct
```
