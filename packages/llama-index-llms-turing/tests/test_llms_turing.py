"""Turing (llama-index-llms-turing): defaults without a server; complete, chat, stream and a tool call against a live
`turing serve` at $TURING_API_BASE serving $TURING_MODEL."""

import os

import pytest
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool

from llama_index.llms.turing import Turing

live = pytest.mark.skipif(not os.getenv("TURING_API_BASE"), reason="set TURING_API_BASE to a running turing serve")


def _llm(**kw):
    return Turing(model=os.getenv("TURING_MODEL", "smollm2"), max_tokens=kw.pop("max_tokens", 16), **kw)


def test_defaults(monkeypatch):
    monkeypatch.delenv("TURING_API_BASE", raising=False)
    monkeypatch.delenv("TURING_API_KEY", raising=False)
    llm = Turing(model="m")
    assert isinstance(llm, BaseLLM) and Turing.class_name() == "Turing"
    assert (llm.api_base, llm.api_key) == ("http://localhost:8000/v1", "turing-local")
    assert llm.metadata.is_chat_model and llm.metadata.is_function_calling_model
    monkeypatch.setenv("TURING_API_BASE", "http://gpu-host:8000/v1")
    monkeypatch.setenv("TURING_API_KEY", "k")
    assert (Turing(model="m").api_base, Turing(model="m").api_key) == ("http://gpu-host:8000/v1", "k")
    assert Turing(model="m", api_base="http://other/v1").api_base == "http://other/v1"


@live
def test_complete_and_chat():
    assert _llm().complete("Explain KV cache paging:").text
    resp = _llm().chat([ChatMessage(role="user", content="Say hello")])
    assert resp.message.role == "assistant" and resp.message.content


@live
def test_stream_complete():
    chunks = list(_llm(max_tokens=8).stream_complete("Write a haiku:"))
    assert len(chunks) >= 2 and chunks[-1].text


@live
async def test_async_chat():
    llm = _llm()
    resp = await llm.achat([ChatMessage(role="user", content="Say hello")])
    await llm._get_aclient().close()  # the reused AsyncOpenAI client keeps its connection open otherwise
    assert resp.message.content


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return "sunny"


@pytest.mark.skipif(not os.getenv("TURING_TOOL_MODEL"), reason="set TURING_TOOL_MODEL=1 when $TURING_MODEL calls tools")
def test_tool_call():
    resp = _llm(max_tokens=128, temperature=0).chat_with_tools(
        [FunctionTool.from_defaults(get_weather)], user_msg="What's the weather in Paris? Use the tool.")
    calls = _llm().get_tool_calls_from_response(resp, error_on_no_tool_call=True)
    assert calls[0].tool_name == "get_weather" and calls[0].tool_kwargs == {"city": "Paris"}
