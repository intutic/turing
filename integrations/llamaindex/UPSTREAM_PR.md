# Upstream PR Template: `run-llama/llama_index`

**PR Title**: `feat(llms): Add native Turing Engine serving adapter`

## Summary & Description
This PR introduces the official `llama-index-llms-turing` integration package located under `llama-index-integrations/llms/llama-index-llms-turing/`.

Turing Engine is an inference and serving engine enabling developers to serve 70B+ frontier models (LLaMA-3.3-70B, DeepSeek-R1, Qwen-2.5-72B, GLM-5.3-Flash) on a single 24GB consumer GPU with 75% less KV cache memory.

## Package Structure
```text
llama-index-integrations/llms/llama-index-llms-turing/
├── README.md
├── pyproject.toml
├── llama_index/
│   └── llms/
│       └── turing/
│           ├── __init__.py
│           └── base.py
└── tests/
    └── test_llms_turing.py
```

## Implementation Highlights
* Subclasses `llama_index.core.llms.custom.CustomLLM` with standard `pydantic` fields.
* Implements `complete()`, `stream_complete()`, `chat()`, `stream_chat()`, and metadata reporting.
* Exposes Turing-specific optimization headers (`X-Turing-Sparsity` for 57% FFN channel pruning, `X-Turing-SVD-Rank` for SVD INT8 KV cache compression).

## Example Usage
```python
from llama_index.llms.turing import Turing

llm = Turing(
    model="deepseek-r1-7b",
    api_base="http://localhost:8000/v1",
    sparsity_ratio=0.57,
    svd_rank=64
)

response = llm.complete("Explain the vector indexing pipeline:")
print(response)
```
