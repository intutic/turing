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

=== "Method B: Native ChatTuring (Subspace Controls)"
    ```python
    from turing.integrations.langchain import ChatTuring

    llm = ChatTuring(
        model="deepseek-r1-7b",
        base_url="http://localhost:8000/v1",
        sparsity_ratio=0.57, # 57% FFN channel pruning
        svd_rank=64          # Rank-64 SVD INT8 KV cache
    )
    print(llm.invoke("Write a quick C++ SIMD routine:")["content"])
    ```

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
        X-Turing-Sparsity: "0.57"
        X-Turing-SVD-Rank: "64"
```

---

## 3. LlamaIndex RAG & Agents

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

## 4. Open WebUI & LibreChat

Connect your local web UI directly to Turing Engine:
1. Open **Open WebUI Settings ➔ Connections ➔ OpenAI API**.
2. Set **URL**: `http://localhost:8000/v1`
3. Set **API Key**: `turing-local`
4. Models like `deepseek-r1-7b` and `glm-5.3-flash` will automatically appear in your model dropdown.

---

## 5. Kubernetes Deployment (KServe / Helm)

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ServingRuntime
metadata:
  name: turing-engine
spec:
  supportedModelFormats:
    - name: turing-subspace
      version: "1"
  containers:
    - name: kserve-container
      image: ghcr.io/intutic/turing:v0.2.0
      command: ["turing", "serve", "--model", "deepseek-r1-7b", "--port", "8000"]
      resources:
        limits:
          nvidia.com/gpu: "1"
          memory: 32Gi
```
