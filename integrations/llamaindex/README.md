# LlamaIndex LLMs Integration: Turing Engine

`llama-index-llms-turing` connects LlamaIndex directly to **Turing Engine** — an open-source inference runtime that serves frontier 70B+ models on a single 24GB consumer GPU with 75% less KV cache memory.

## Installation

```bash
pip install llama-index-llms-turing
```

## Quickstart

```python
from llama_index.llms.turing import Turing

llm = Turing(
    model="deepseek-r1-7b",
    api_base="http://localhost:8000/v1",
    temperature=0.7,
    max_tokens=512,
    sparsity_ratio=0.57,  # 57% FFN subspace channel pruning
    svd_rank=64           # Calibrated SVD INT8 KV cache paging
)

response = llm.complete("Explain the vector indexing retrieval pipeline:")
print(response)
```

## RAG Query Engine Integration

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.turing import Turing

# Initialize documents and index
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Set Turing Engine as the LLM
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("Summarize the key architectural concepts:")
print(response)
```
