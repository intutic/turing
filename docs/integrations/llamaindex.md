# LlamaIndex Integration Guide

[LlamaIndex](https://github.com/run-llama/llama_index) connects data sources and retrieval-augmented generation (RAG) pipelines directly to Turing Engine.

---

## 1. Quick Start: `OpenAILike` Adapter

```bash
pip install llama-index llama-index-llms-openai-like
```

```python
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import VectorStoreIndex, Document

# Connect to Turing Engine
llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    api_key="turing-local",
    model="llama-3.1-70b",
    is_chat_model=True,
    max_tokens=256
)

documents = [
    Document(text="Turing Engine reduces 32K context KV cache from 10 GB to 2.5 GB using SVD INT8 quantization."),
    Document(text="Turing Engine bypasses 57.1% of inactive FFN channels for 2.32x CUDA speedup.")
]

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("What are the key memory savings in Turing Engine?")
print(str(response))
```

---

## 2. Streaming Query Responses

```python
response = query_engine.query("Explain how subspace pruning accelerates compute.")
for text in response.response_gen:
    print(text, end="", flush=True)
```
