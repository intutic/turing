# 🦜️🔗 langchain-turing

Official LangChain and LangGraph partner package for **Turing Engine** — high-performance LLM serving with Subspace channel activation pruning and SVD INT8 KV cache compression.

---

## ⚡ Installation

```bash
pip install langchain-turing
```

---

## 🚀 Quickstart

```python
from langchain_turing import ChatTuring

# Connect to local Turing Engine instance
llm = ChatTuring(
    model="deepseek-r1-1.5b",
    base_url="http://localhost:8000/v1",
    sparsity_ratio=0.57 # 57% Subspace Activation Pruning
)

response = llm.invoke("Explain why SVD INT8 KV cache preserves needle retrieval:")
print(response["content"])
```

### Streaming Responses
```python
for chunk in llm.stream("Write a fast PyTorch Triton kernel:"):
    print(chunk, end="", flush=True)
```
