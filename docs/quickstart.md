# Quickstart

Install Turing Engine, chat with a model in the terminal, then serve it over the OpenAI, Anthropic and Ollama APIs.

---

## 1. Installation

=== "pip"
    ```bash
    pip install turing-engine                  # Python 3.10–3.12; wheels for Linux x86_64/aarch64 and macOS arm64
    pip install "turing-engine[structured]"    # optional: JSON schema / tool_choice enforcement (llguidance)
    ```

=== "From source"
    ```bash
    git clone https://github.com/intutic/turing.git
    cd turing
    pip install -e .                  # builds the C++ extension (AVX2 on x86_64, NEON on arm64)
    pip install -e ".[structured]"    # optional: JSON schema / tool_choice enforcement (llguidance)
    ```

=== "Development"
    ```bash
    pip install -e ".[dev]"
    python setup.py build_ext --inplace
    python -W error -m pytest -q
    ```

=== "Docker"
    ```bash
    docker run --rm -p 127.0.0.1:8000:8000 ghcr.io/intutic/turing:v2.1.0 --model HuggingFaceTB/SmolLM2-135M
    ```
    The CUDA image is `ghcr.io/intutic/turing:v2.1.0-cuda`. See [Docker](docker.md).

On other platforms (Windows, Intel Macs) `pip install turing-engine` builds the C++ extension from the source package,
which needs a C++20 compiler.

---

## 2. Terminal chat

```bash
turing chat --model smollm2                                     # HuggingFaceTB/SmolLM2-135M
turing chat --model deepseek-r1-1.5b                            # deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
turing chat --model Qwen/Qwen2.5-Coder-7B-Instruct --dtype bf16
turing chat --model ./models/qwen2.5-0.5b-instruct-q8_0.gguf    # llama.cpp GGUF file
```

`--model` takes a HuggingFace repo id, a local checkpoint directory, a `.gguf` file, a `.tgate4` directory written by
`turing convert`, or a short alias from `ModelResolver.CLI_ALIASES` (`turing/models/resolver.py`). If the weights fail
to load, the command exits with an error; `--mock` runs synthetic weights for testing without a download.

---

## 3. Serving

```bash
turing serve --model Qwen/Qwen2.5-7B-Instruct --port 8000
```

### A. OpenAI SDK
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Return a JSON object with the keys name and age for: Alice is 28."}],
    response_format={"type": "json_object"},   # needs the [structured] extra; 400 without it
)
print(response.choices[0].message.content)
```

### B. Anthropic SDK
```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:8000", api_key="unused")
message = client.messages.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    max_tokens=128,
    messages=[{"role": "user", "content": "Hello"}],
)
print(message.content[0].text)
```

### C. Ollama SDK
```python
import ollama

client = ollama.Client(host="http://localhost:8000")
response = client.chat(model="Qwen/Qwen2.5-7B-Instruct", messages=[{"role": "user", "content": "Hello"}])
print(response["message"]["content"])
```

The server ignores `api_key` unless it is started with `--api-key` (or `TURING_API_KEY`).

### D. LangChain
```python
from langchain_turing import ChatTuring   # pip install langchain-turing

llm = ChatTuring(model="Qwen/Qwen2.5-7B-Instruct", base_url="http://localhost:8000/v1")
print(llm.invoke("Explain KV caching in two sentences.").content)
```

---

## 4. Python API

```python
from turing.models.architecture_registry import AutoSubspaceModel

# A HuggingFace repo id, a local directory or a .gguf file; sparsity_ratio=0.0 (the default) is the exact model
model, tokenizer = AutoSubspaceModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device="auto")
ids = model.generate(tokenizer.encode("def quicksort(arr):"), max_new_tokens=64, temperature=0.0)
print(tokenizer.decode(ids))   # prompt and continuation
```

---

## 5. Turing DSL

```python
import turing

@turing.chain(model="smollm2", device="auto")
def three_answers(problem: str):
    turing.gen(f"Problem: {problem}\n", max_tokens=64)
    branches = turing.fork(3, temperature=0.8)   # prefix prefilled once; branches decode concurrently
    for i, b in enumerate(branches):
        b.gen(f"Candidate {i + 1}:", max_tokens=64)
    return turing.join(branches, strategy="best")  # or "vote", "concat", or a callable
```

`turing.select(options)` picks one of several strings and `turing.constrain(schema)` makes later `gen()` calls
produce JSON valid against the schema (needs the `[structured]` extra and a model with a tokenizer).

---

## 6. 4-bit weights

```bash
turing convert --model Qwen/Qwen2.5-7B-Instruct --output ./qwen7b-gptq --method gptq   # layer by layer
turing serve --model ./qwen7b-gptq
turing serve --model Qwen/Qwen2.5-7B-Instruct --quantization w4a16-rtn                 # quantize at load
```

---

## 7. Standalone C++ engine (`turing-cli`)

```bash
cmake -B build && cmake --build build
./build/turing-cli info
./build/turing-cli generate --model ./models/llama-3.2-1b-instruct-f16.gguf --prompt "Hello" --max-tokens 32
./build/turing-cli serve --model ./models/llama-3.2-1b-instruct-f16.gguf --port 8000 --kv-cache-dtype int8
```

The binary runs Llama-layout GGUF models on the CPU and serves `/v1/chat/completions`, `/v1/completions` and
`/health` on `127.0.0.1` unless `--host` says otherwise.

---

## 8. Tensor and pipeline parallelism

```bash
./scripts/launch_distributed.sh Qwen/Qwen2.5-7B-Instruct 2 1   # MODEL TP PP; one process per GPU via torchrun
```

Rank 0 runs the HTTP server and the engine; the other ranks run the worker loop. This path is tested with CPU
(gloo) processes; multi-GPU runs have not been validated. See [distributed serving](serving.md#11-distributed-serving).
