"""
Turing (the llama-index-llms-turing package, packages/llama-index-llms-turing) against a live in-process Turing
server. Skips unless the package is installed.
"""

import pytest

pytest.importorskip("llama_index.llms.turing")  # pip install -e packages/llama-index-llms-turing

from llama_index.core.llms import ChatMessage
from llama_index.llms.turing import Turing


def _llm(url):
    return Turing(model="test-tiny", api_base=url, max_tokens=8)


def test_complete(turing_server):
    resp = _llm(turing_server).complete("Explain KV cache paging:")
    assert isinstance(resp.text, str) and resp.text


def test_chat(turing_server):
    resp = _llm(turing_server).chat([ChatMessage(role="user", content="hi")])
    assert resp.message.role == "assistant" and resp.message.content


def test_stream_complete(turing_server):
    chunks = list(_llm(turing_server).stream_complete("Write a haiku:"))
    assert len(chunks) >= 8 and chunks[-1].text
