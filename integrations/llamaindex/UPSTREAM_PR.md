# Upstream PR Template: `run-llama/llama_index`

**PR Title**: `feat(llms): Add native Turing Engine serving adapter`

## Description
This PR introduces `TuringEngine` to `llama-index-llms-turing`, allowing developers to connect high-scale RAG retrieval pipelines directly to Turing Engine running on 1x 24GB GPUs.

## Example Usage
```python
from llama_index.llms.turing import TuringEngine

llm = TuringEngine(model="llama-3.1-70b", api_base="http://localhost:8000/v1")
response = llm.complete("Explain the RAG indexing pipeline.")
print(response.text)
```
