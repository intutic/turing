"""
integrations/llamaindex/openai_like_example.py (llama-index's OpenAILike) against a live in-process Turing server.
Skips unless llama-index-llms-openai-like is installed.
"""

import pytest

pytest.importorskip("llama_index.llms.openai_like")

from integrations.llamaindex.openai_like_example import turing_llm


def test_complete(turing_server):
    resp = turing_llm(model="test-tiny", api_base=turing_server, max_tokens=8).complete("Explain KV cache paging:")
    assert isinstance(resp.text, str) and resp.text


def test_stream_complete(turing_server):
    chunks = list(turing_llm(model="test-tiny", api_base=turing_server, max_tokens=8).stream_complete("Write a haiku:"))
    assert len(chunks) >= 8 and chunks[-1].text
