# langchain-turing

LangChain chat model for [Turing Engine](https://github.com/intutic/turing): `ChatTuring` talks to a
`turing serve` server over its OpenAI-compatible `/v1/chat/completions` endpoint. It depends only on
`langchain-core` and `httpx`, not on Turing Engine itself.

```bash
pip install langchain-turing
turing serve --model Qwen/Qwen2.5-1.5B-Instruct --port 8000   # on the machine with the GPU
```

```python
from langchain_turing import ChatTuring

llm = ChatTuring(model="Qwen/Qwen2.5-1.5B-Instruct", base_url="http://localhost:8000/v1", max_tokens=64)

print(llm.invoke("Explain KV cache paging in one sentence:").content)

for chunk in llm.stream("Write a haiku about GPUs:"):
    print(chunk.content, end="", flush=True)
```

`base_url` and `api_key` default to `$TURING_BASE_URL` and `$TURING_API_KEY` (the key is checked only when the
server runs with `--api-key`). `invoke`, `stream`, `batch`, their async forms and LangGraph work as with any
LangChain chat model.

## Tool calling

`bind_tools` sends the tools with each request; calls the model makes come back on `AIMessage.tool_calls`.

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Current weather for a city."""
    return "sunny"

msg = llm.bind_tools([get_weather]).invoke("What's the weather in Paris?")
print(msg.tool_calls)
```

Whether the model emits a tool call depends on the model; small models often answer in text. Pruning, KV cache
format and quantization are chosen when the server starts, not per request.

## Tests

```bash
pip install -e ".[test]"
pytest tests/unit_tests
TURING_BASE_URL=http://localhost:8000/v1 TURING_MODEL=Qwen/Qwen2.5-1.5B-Instruct pytest tests/integration_tests
```

The integration tests run LangChain's standard chat-model suite against a live server.

License: Apache-2.0 (Turing Engine itself is BSL 1.1).
