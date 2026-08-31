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
    pip install https://github.com/intutic/turing/releases/download/v0.4.1/turing_engine-0.4.1-cp311-cp311-macosx_15_0_arm64.whl
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

## 3. Launch Serving Server (Dual OpenAI & Anthropic API)

```bash
turing serve --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8000
```

### Query with Python OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="turing-local")
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    messages=[{"role": "user", "content": "Write a quick Python script to calculate Fibonacci numbers:"}],
    extra_body={"reasoning_effort": "high"}
)
print(response.choices[0].message.content)
```

### Python SDK Programmatic Usage

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

### Query with LangChain

```python
from turing.integrations.langchain import ChatTuring

llm = ChatTuring(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", base_url="http://localhost:8000/v1")
print(llm.invoke("Explain KV cache compression in 2 sentences:")["content"])
```
