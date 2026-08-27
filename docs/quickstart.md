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
    pip install https://github.com/intutic/turing/releases/download/v0.2.1/turing_engine-0.2.1-cp311-cp311-macosx_15_0_arm64.whl
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

Stream reasoning weights straight from Hugging Face:

```bash
# Chat with DeepSeek-R1 Distill
turing chat --model deepseek-r1-1.5b

# Chat with Qwen Coder
turing chat --model qwen-2.5-coder-7b
```

---

## 3. Launch Serving Server (Dual OpenAI & Anthropic API)

```bash
turing serve --model deepseek-r1-7b --port 8000
```

### Query with Python OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="turing-local")
response = client.chat.completions.create(
    model="deepseek-r1-7b",
    messages=[{"role": "user", "content": "Write a quick Python script to calculate Fibonacci numbers:"}]
)
print(response.choices[0].message.content)
```

### Query with LangChain

```python
from turing.integrations.langchain import ChatTuring

llm = ChatTuring(model="deepseek-r1-7b", base_url="http://localhost:8000/v1")
print(llm.invoke("Explain KV cache compression in 2 sentences:")["content"])
```
