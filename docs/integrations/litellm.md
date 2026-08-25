# LiteLLM Integration Guide for Turing Engine

[LiteLLM](https://github.com/BerriAI/litellm) provides an OpenAI-compatible proxy to route, load-balance, and manage AI model requests across backends.

Turing Engine provides native drop-in compatibility with LiteLLM for high-throughput serving of frontier models (LLaMA-3.1-70B, Qwen-2.5-72B, DeepSeek-V4-284B) on single-GPU hardware.

---

## 1. Quick Start with LiteLLM Proxy

### Step 1: Start Turing Engine Server
```bash
turing serve --model llama-3.1-70b --port 8000 --device auto
```

### Step 2: Configure `litellm_config.yaml`
Create a configuration file mapping model routes to your Turing Engine endpoint:

```yaml
model_list:
  - model_name: turing-llama-70b
    litellm_params:
      model: openai/llama-3.1-70b
      api_base: http://localhost:8000/v1
      api_key: turing-local
      rpm: 1000
      tpm: 100000

  - model_name: turing-qwen-72b
    litellm_params:
      model: openai/qwen-2.5-72b
      api_base: http://localhost:8000/v1
      api_key: turing-local

general_settings:
  master_key: sk-litellm-master-key
```

### Step 3: Launch LiteLLM Proxy
```bash
litellm --config litellm_config.yaml --port 4000
```

---

## 2. Python SDK Usage via LiteLLM

```python
import litellm

# Route request through LiteLLM to local Turing Engine
response = litellm.completion(
    model="openai/llama-3.1-70b",
    api_base="http://localhost:8000/v1",
    api_key="turing-local",
    messages=[
        {"role": "user", "content": "Explain subspace pruning in one sentence."}
    ],
    temperature=0.7,
    max_tokens=64
)

print(response.choices[0].message.content)
```

---

## 3. Streaming Example

```python
import litellm

response = litellm.completion(
    model="openai/llama-3.1-70b",
    api_base="http://localhost:8000/v1",
    api_key="turing-local",
    messages=[{"role": "user", "content": "Write a Python fibonacci generator."}],
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
```
