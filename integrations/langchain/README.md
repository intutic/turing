# LangChain

`ChatTuring` is a LangChain `BaseChatModel` for a Turing Engine server. It ships with the
`turing-engine` package as `turing.integrations.langchain` and needs `langchain-core`.

```bash
pip install turing-engine langchain-core
turing serve --model smollm2 --port 8000
```

```python
from turing.integrations.langchain import ChatTuring

llm = ChatTuring(model="smollm2", base_url="http://localhost:8000/v1", max_tokens=64)

print(llm.invoke("Explain KV cache paging in one sentence:").content)

for chunk in llm.stream("Write a haiku about GPUs:"):
    print(chunk.content, end="", flush=True)
```

## Tool calling

`bind_tools` sends the tools with each request through the server's OpenAI API; calls the model
makes come back on `AIMessage.tool_calls`.

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Current weather for a city."""
    return "sunny"

msg = llm.bind_tools([get_weather]).invoke("What's the weather in Paris?")
print(msg.tool_calls)
```

Whether the model emits a tool call depends on the model; small models often answer in text.
