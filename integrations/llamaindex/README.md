# LlamaIndex

Turing Engine serves the OpenAI API, so LlamaIndex uses its own `OpenAILike` client; there is no
`llama-index-llms-turing` package.

```bash
pip install llama-index-llms-openai-like
turing serve --model smollm2 --port 8000
```

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(model="smollm2", api_base="http://localhost:8000/v1", api_key="turing-local", is_chat_model=True)
print(llm.complete("Explain KV cache paging in one sentence:").text)
```

`openai_like_example.py` is the same code as a script; `tests/test_llamaindex_example.py` runs it
against a live server.
