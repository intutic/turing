# ⚡ Quickstart

Get Turing Engine running locally in under 30 seconds.

---

## 1. Installation

=== "pip (Recommended)"
    ```bash
    pip install turing-engine
    ```

=== "Pre-Built Binary Wheel"
    ```bash
    # Direct wheel download for macOS Apple Silicon (arm64):
    pip install https://github.com/intutic/turing/releases/download/v0.6.0/turing_engine-0.6.0-cp311-cp311-macosx_15_0_arm64.whl
    ```





=== "Build from Source"
    ```bash
    git clone https://github.com/intutic/turing.git
    cd turing
    pip install -e ".[dev]"
    python setup.py build_ext --inplace
    ```

---

## 2. Interactive Terminal Chat

Stream reasoning weights dynamically straight from Hugging Face Hub:

```bash
# Chat with DeepSeek-R1 Distill with high reasoning effort
turing chat --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --reasoning-effort high

# Chat using provider/model/effort namespace:
turing chat --model deepseek-ai/DeepSeek-R1/medium

# Chat with Qwen 2.5 Coder
turing chat --model Qwen/Qwen2.5-Coder-7B-Instruct
```

---

## 3. Launch Triple Gateway Serving Server (OpenAI, Anthropic & Ollama API)

```bash
# Serve any model dynamically on port 8000:
turing serve --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8000
```

### A. Query with Python OpenAI SDK
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="turing-local")
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    messages=[{"role": "user", "content": "Write a quick binary search in Python:"}],
    response_format={"type": "json_object"}
)
print(response.choices[0].message.content)
```

### B. Query with Python Ollama SDK
```python
import ollama

client = ollama.Client(host="http://localhost:8000")
response = client.chat(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    messages=[{"role": "user", "content": "Explain Subspace KV paging in 2 sentences."}]
)
print(response["message"]["content"])
```

### C. Query with Python Anthropic SDK
```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:8000", api_key="turing-local")
message = client.messages.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    max_tokens=128,
    messages=[{"role": "user", "content": "Hello Turing Engine!"}]
)
print(message.content[0].text)
```

### D. Python SDK Programmatic Usage
```python
from turing.models.architecture_registry import AutoSubspaceModel

# Dynamically load any Hugging Face model into Subspace format
model, tokenizer = AutoSubspaceModel.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    sparsity_ratio=0.5,
    device="auto"
)
prompt_tokens = tokenizer.encode("def quicksort(arr):")
output = model.generate(prompt_tokens, max_new_tokens=64)
print(tokenizer.decode(output))
```

### E. Query with LangChain
```python
from turing.integrations.langchain import ChatTuring

llm = ChatTuring(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", base_url="http://localhost:8000/v1")
print(llm.invoke("Explain KV cache compression in 2 sentences:")["content"])
```
