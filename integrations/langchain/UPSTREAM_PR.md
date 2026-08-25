# Upstream PR Template: `langchain-ai/langchain` (langchain-community)

**PR Title**: `feat(community): Add ChatTuringEngine integration for Turing Engine`

## Description
This PR adds `ChatTuringEngine` to `langchain-community`, allowing LangChain and LangGraph users to run agentic loops and chains backed by Turing Engine's high-throughput sub-24GB serving runtime.

## Changes
- Created `libs/community/langchain_community/chat_models/turing.py`.
- Exposed hyperparameters: `sparsity_ratio`, `use_svd_kv`, and `speculative_draft_tokens`.
- Added unit and integration tests with mock server fixtures.

## Example
```python
from langchain_community.chat_models import ChatTuringEngine

llm = ChatTuringEngine(model="llama-3.1-70b", base_url="http://localhost:8000/v1")
response = llm.invoke("Summarize the Turing Subspace theorem.")
print(response.content)
```
