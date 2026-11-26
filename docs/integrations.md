# Integrations

Turing Engine serves the OpenAI (`/v1/*`), Anthropic (`/v1/messages`) and Ollama (`/api/*`) protocols, so most
agent frameworks, model routers and chat frontends connect to it with their existing clients. The examples below
assume a server started with `turing serve` on `http://localhost:8000`.

## LangChain and LangGraph

Use `ChatOpenAI` from `langchain-openai` against the OpenAI API, or `ChatTuring` from the `langchain-turing` package.

=== "langchain-openai"
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

=== "langchain-turing (with tool calling)"
    ```bash
    pip install langchain-turing
    ```
    ```python
    from langchain_turing import ChatTuring

    llm = ChatTuring(model="deepseek-r1-7b", base_url="http://localhost:8000/v1")
    print(llm.invoke("Write a quick C++ SIMD routine:").content)
    ```
    `ChatTuring` is a LangChain `BaseChatModel`
    ([`packages/langchain-turing`](https://github.com/intutic/turing/tree/master/packages/langchain-turing)) that
    depends only on `langchain-core` and `httpx`. It supports `invoke`, `stream` (with token usage), `batch`, async,
    and `bind_tools`; tool calls come back OpenAI-style as `AIMessage.tool_calls`. It passes LangChain's standard
    chat-model test suite against `turing serve` (Qwen2.5-0.5B-Instruct, Apple M4 Pro).
    `turing.integrations.langchain.ChatTuring` re-exports it.

Pruning, KV cache format and quantization are set when the server starts, not per request.

## LiteLLM

LiteLLM reaches Turing Engine through its OpenAI-compatible provider: use the `openai/` model prefix and set
`api_base`. A native `turing` provider is proposed upstream
([BerriAI/litellm#38181](https://github.com/BerriAI/litellm/pull/38181)) and is not merged yet.

Two Turing servers behind one model name, with your own prices (`litellm --config litellm_config.yaml`):

```yaml
model_list:
  - model_name: qwen
    litellm_params:
      model: openai/Qwen/Qwen2.5-1.5B-Instruct   # the name after openai/ is free-form: a server serves one model
      api_base: http://gpu-a:8000/v1
      api_key: turing-local                     # any value unless the server runs with --api-key
      input_cost_per_token: 0.000001            # your own prices; LiteLLM's cost map has no Turing rows
      output_cost_per_token: 0.000002
      extra_headers:
        X-Turing-Draft-Tokens: "4"              # optional: speculative draft tokens per step
  - model_name: qwen
    litellm_params:
      model: openai/Qwen/Qwen2.5-1.5B-Instruct
      api_base: http://gpu-b:8000/v1
      api_key: turing-local
      input_cost_per_token: 0.000001
      output_cost_per_token: 0.000002

router_settings:
  num_retries: 2
  allowed_fails: 1      # a server that fails once leaves the rotation for cooldown_time seconds
  cooldown_time: 30

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY   # the proxy does not start without one
```

Checked on 2026-11-26 against two `turing serve` processes, with LiteLLM's `Router` and its proxy: requests were
spread over both servers (11 and 9 of 20); with one server stopped, all 40 requests were answered by the other.
Each response's cost (`response._hidden_params["response_cost"]`, or the proxy's `x-litellm-response-cost` header)
was its prompt and completion tokens times these prices; `litellm.completion_cost(completion_response=...)`
returned 0.0 with them. Without prices the cost is 0.0. With the native provider the model is
`turing/<name>`, with `TURING_API_BASE` and `TURING_API_KEY` (any value) from the environment.

The `X-Turing-Sparsity` and `X-Turing-SVD-*` headers are not supported; requests that send them get HTTP 400.

## DSPy

DSPy calls models through LiteLLM. Pass `model_type="chat"`: DSPy 3.5 otherwise calls `/v1/responses`, which
Turing does not serve (HTTP 404).

```python
import dspy

lm = dspy.LM("openai/Qwen/Qwen2.5-0.5B-Instruct", api_base="http://localhost:8000/v1", api_key="turing-local",
             model_type="chat", temperature=0.0)
dspy.configure(lm=lm)
print(dspy.ChainOfThought("question -> answer")(question="What is 12 + 7?").answer)
```

`dspy.BootstrapFewShot` compiles against it the same way: with Qwen2.5-0.5B-Instruct and 6 training questions,
exact match on 6 others went from 1 to 4 (DSPy 3.5.0, 2026-11-26).

## LlamaIndex

`llama-index-llms-turing`
([`packages/llama-index-llms-turing`](https://github.com/intutic/turing/tree/master/packages/llama-index-llms-turing))
is LlamaIndex's `OpenAILike` client with Turing's defaults: the chat endpoint, tool calling on, and `api_base` from
`$TURING_API_BASE` (default `http://localhost:8000/v1`).

```bash
pip install llama-index-llms-turing
```

```python
from llama_index.llms.turing import Turing

llm = Turing(model="deepseek-r1-7b", context_window=32768)
response = llm.complete("Summarize the main architectural trade-offs of microservices:")
print(response.text)
```

Set `context_window` to the served model's context length. Without the package, `OpenAILike(api_base=...,
is_chat_model=True)` from `llama-index-llms-openai-like` works the same way.

## Chat frontends and editors

Clients that speak the Ollama or OpenAI protocol connect with the settings below, taken from each client's own
documentation. The protocols are tested; the individual client applications are not.

=== "Ollama protocol (Open WebUI, Continue.dev, Raycast, Enchanted)"
    1. In **Open WebUI**, open **Settings ➔ Connections ➔ Ollama API**.
    2. Set **Ollama Base URL** to `http://localhost:8000`.
    3. Click **Verify Connection**. The loaded models appear through `/api/tags`.
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

=== "OpenAI protocol (LibreChat, Chatbox, Cursor, Cline)"
    1. Open the client's OpenAI API settings.
    2. Set **Base URL** to `http://localhost:8000/v1`.
    3. Set **API Key** to any value, or to the key passed to `turing serve --api-key` (keys are checked only when one
       is set).
    4. The served models appear in the model selector.

## Kubernetes

- **[Kubernetes with llm-d](llmd-integration.md):** prefix-cache aware routing across replicas with an
  `InferencePool`.
- **[Helm chart](https://github.com/intutic/turing/tree/master/deploy/helm/turing-serving)**
  (`deploy/helm/turing-serving`): a Deployment with GPU limits, Prometheus scrape annotations (`/metrics` is
  Prometheus text) and optional llm-d KV events. Defaults: model `Qwen/Qwen2.5-7B-Instruct`, image tag `v2.1.0-cuda`.
- **[Docker images](docker.md):** `ghcr.io/intutic/turing:v2.1.0` (CPU) and `ghcr.io/intutic/turing:v2.1.0-cuda`.

### KServe

[`integrations/kserve/serving_runtime.yaml`](https://github.com/intutic/turing/blob/master/integrations/kserve/serving_runtime.yaml)
defines a `ServingRuntime` that runs `turing serve --model /mnt/models` on port 8080 with the
`ghcr.io/intutic/turing:v2.1.0-cuda` image. KServe downloads the InferenceService's `storageUri` (`hf://`, `s3://`,
`gs://`, `pvc://`) to `/mnt/models`, so that is the model it serves
([`inferenceservice.yaml`](https://github.com/intutic/turing/blob/master/integrations/kserve/inferenceservice.yaml)):

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

`hf://` storage URIs need KServe 0.15 or later.

**Run on GKE on 2026-11-26**, with the published `v2.1.0-cuda` image and no override: the same cluster setup, with
the runtime as an opt-in `kserve-turingserver` ClusterServingRuntime (`--served-model-name={{.Name}}`, proposed
upstream in [kserve/kserve#6688](https://github.com/kserve/kserve/pull/6688)). The InferenceService was Ready 304 s
after `kubectl apply`, including the image's pull onto a new node (162 s); `/health`, `/v1/models` (listing the
InferenceService's name), a plain and a streamed chat completion and `/metrics` answered through the predictor
Service, and after `kubectl delete pod` it was Ready again in 93 s.

The same day in **Knative mode**, on a cluster set up with KServe v0.21.0's own Knative-mode install (Istio,
Knative Serving 1.21.1) and the InferenceService without a deployment-mode annotation: Ready 372 s after
`kubectl apply`, every endpoint answered through the Istio ingress gateway, and the revision was Ready again 123 s
after `kubectl delete pod`. KServe's e2e test for the runtime (`test/e2e/predictor/test_turing.py` in #6688: chat
and streamed chat on Qwen2.5-0.5B-Instruct on the CPU) passed against that cluster.

!!! note "Give the storage initializer 4Gi"
    KServe's storage initializer has a 1Gi memory limit by default, and downloading Qwen2.5-1.5B-Instruct's 3 GB
    checkpoint through `hf://` was killed for running out of memory at that limit. 4Gi worked: install the
    `kserve-resources` chart with `--set kserve.storage.resources.limits.memory=4Gi`, or set `memoryLimit` under
    `storageInitializer` in the `inferenceservice-config` ConfigMap and restart `kserve-controller-manager`.

**Run on GKE on 2026-11-23** (`benchmarks/results/results_kserve_gke.json`): GKE 1.36.4 with one Spot L4 node
(`g2-standard-8`), KServe v0.21.0 in Standard deployment mode, cert-manager v1.17.0.

- **The `v2.0.0-cuda` image does not start on a GPU.** With the manifests as they are the pod restarts in a loop.
  The image has no C compiler, which Triton needs at its first kernel (`Failed to find C compiler`), and it does
  not set the library path where GKE mounts the NVIDIA driver (`Found no NVIDIA driver`). `deploy/Dockerfile` now
  does both. An image built from it started on an L4 under `docker run --gpus all` and with the driver libraries
  mounted the way GKE mounts them (healthy 80 to 84 s after `docker run`, the model's download included); the
  published one failed both ways with the two errors above. The manifests now name `v2.1.0-cuda`, built from that
  Dockerfile; before its release the same build served chat requests on an L4 both ways.
- With those two things added to the same image at container start (a test-only override), the InferenceService was
  Ready 57 s after `kubectl apply` with the image already on the node (pulling its 4.1 GB took 4 min 8 s; the
  storage initializer's download 27 to 30 s). `/health`, `/v1/models`, a plain and a streamed chat completion and
  `/metrics` answered through the predictor Service, and after `kubectl delete pod` the service was Ready again in
  62 s. The server ran as the image's non-root user.
- In that run the first request waited 17.4 s for its first token and the second 2.1 s, and `/v1/models` listed
  the model as `HF-/mnt/models`. Three changes since (`benchmarks/results/results_serve_first_token_cuda.json`):
  `turing serve` warms up before it listens (24 s in the rebuilt image on an L4), so Triton's builds and tuning are
  over when the pod turns Ready; a chat request no longer builds the tokenizer's vocabulary twice, and a short
  prompt arriving alone is prefilled whole from a CUDA graph (0.26 s to 0.03 s to the first token of a 34-token
  prompt, 0.06 s for the first request after a start); and `--served-model-name` sets the listed name. A
  ServingRuntime can pass `--served-model-name={{.Name}}` (the InferenceService's name) with `v2.1.0-cuda`, which
  has the flag.
