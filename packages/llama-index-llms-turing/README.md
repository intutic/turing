# llama-index-llms-turing

LlamaIndex LLM for [Turing Engine](https://github.com/intutic/turing). `Turing` is LlamaIndex's `OpenAILike`
client with Turing's defaults: chat endpoint, tool calling on, `api_base` from `$TURING_API_BASE` (else
`http://localhost:8000/v1`) and `api_key` from `$TURING_API_KEY` (checked only when the server runs with
`--api-key`). It does not depend on Turing Engine itself.

```bash
pip install llama-index-llms-turing
turing serve --model Qwen/Qwen2.5-1.5B-Instruct --port 8000   # on the machine with the GPU
```

```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.turing import Turing

llm = Turing(model="Qwen/Qwen2.5-1.5B-Instruct", context_window=32768, max_tokens=256)

print(llm.complete("Explain KV cache paging in one sentence:"))
print(llm.chat([ChatMessage(role="user", content="Write a haiku about GPUs")]))
for chunk in llm.stream_complete("Count to five:"):
    print(chunk.delta, end="", flush=True)
```

Set `context_window` to the served model's; LlamaIndex uses it to size prompts. Tool calling (`FunctionAgent`,
`llm.predict_and_call`) goes through the server's OpenAI `tools` API; whether the model calls a tool depends on the
model.

## Tests

```bash
pip install -e ".[test]"
TURING_API_BASE=http://localhost:8000/v1 TURING_MODEL=Qwen/Qwen2.5-1.5B-Instruct pytest tests
```

License: Apache-2.0 (Turing Engine itself is BSL 1.1).
