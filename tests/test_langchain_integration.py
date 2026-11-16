"""
ChatTuring (turing.integrations.langchain) against a live in-process Turing server.
"""

import socket

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from turing.integrations.langchain import ChatTuring, _chat_result
from turing.serving.tools import ToolCallingHandler


@tool
def get_weather(city: str) -> str:
    """Current weather for a city."""
    return "sunny"


def _llm(url, **kw):
    return ChatTuring(model="test-tiny", base_url=url, max_tokens=8, **kw)


def test_is_a_langchain_chat_model():
    assert isinstance(ChatTuring(), BaseChatModel)


def test_invoke(turing_server):
    msg = _llm(turing_server).invoke("Explain KV cache paging:")
    assert isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content
    assert msg.usage_metadata["output_tokens"] == 8
    assert msg.response_metadata["finish_reason"] == "stop"


def test_stream(turing_server):
    chunks = list(_llm(turing_server).stream("Write a haiku:"))
    assert len(chunks) >= 8  # one chunk per generated token, then the finish chunk
    assert "".join(c.content for c in chunks)
    assert [c.response_metadata.get("finish_reason") for c in chunks if c.response_metadata] == ["stop"]


def test_batch(turing_server):
    out = _llm(turing_server).batch(["one", "two"])
    assert len(out) == 2 and all(isinstance(m, AIMessage) for m in out)


def test_tool_conversation_round_trip(turing_server):
    """A full tool-calling history (system, user, assistant tool call, tool result) is accepted by the server."""
    history = [
        SystemMessage("Be brief."),
        HumanMessage("Weather in Paris?"),
        AIMessage("", tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}, "id": "call_1"}]),
        ToolMessage("sunny", tool_call_id="call_1"),
    ]
    assert isinstance(_llm(turing_server).bind_tools([get_weather]).invoke(history), AIMessage)


def test_bind_tools_sends_openai_tools():
    bound = ChatTuring().bind_tools([get_weather], tool_choice="get_weather")
    tools = bound.kwargs["tools"]
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "get_weather" and "city" in tools[0]["function"]["parameters"]["properties"]
    assert bound.kwargs["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_tool_calls_from_server_response():
    """The server's own tool-call parser output, shaped as its /v1/chat/completions body, becomes AIMessage.tool_calls."""
    text, calls = ToolCallingHandler.extract_tool_calls(
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>')
    body = {"model": "test-tiny", "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14},
            "choices": [{"index": 0, "finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": text or None, "tool_calls": calls}}]}
    msg = _chat_result(body).generations[0].message
    assert [(c["name"], c["args"], c["id"]) for c in msg.tool_calls] == [("get_weather", {"city": "Paris"}, calls[0]["id"])]
    assert msg.usage_metadata["total_tokens"] == 14


def test_connection_error():
    with socket.socket() as s:  # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(ConnectionError, match="Failed to connect to Turing Engine"):
        _llm(f"http://127.0.0.1:{port}/v1").invoke("hello")
