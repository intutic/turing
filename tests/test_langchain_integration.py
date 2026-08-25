"""
Unit tests for Turing Engine LangChain and LangGraph integration adapter.
"""

import pytest
import unittest.mock as mock
import json
import io
import urllib.error

from turing.integrations.langchain import ChatTuring, TuringLLM

class DummyMessage:
    def __init__(self, content, msg_type):
        self.content = content
        self.type = msg_type

def test_chat_turing_message_formatting():
    chat = ChatTuring(model="deepseek-r1-1.5b")

    # 1. Plain string
    formatted = chat._format_messages("Hello world")
    assert formatted == [{"role": "user", "content": "Hello world"}]

    # 2. List of dicts
    dict_msgs = [{"role": "system", "content": "Be concise"}, {"role": "user", "content": "Hi"}]
    assert chat._format_messages(dict_msgs) == dict_msgs

    # 3. Object messages (HumanMessage, AIMessage, SystemMessage)
    obj_msgs = [
        DummyMessage("System instruction", "system"),
        DummyMessage("User question", "human"),
        DummyMessage("AI answer", "ai")
    ]
    res = chat._format_messages(obj_msgs)
    assert res == [
        {"role": "system", "content": "System instruction"},
        {"role": "user", "content": "User question"},
        {"role": "assistant", "content": "AI answer"}
    ]

def test_chat_turing_invoke_mocked():
    chat = ChatTuring(model="deepseek-r1-1.5b", sparsity_ratio=0.57, svd_rank=64)

    mock_response_data = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a verified test response from Turing Engine."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {"total_tokens": 12}
    }

    mock_resp = io.BytesIO(json.dumps(mock_response_data).encode("utf-8"))

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        res = chat.invoke("Explain KV cache compression")
        assert res["content"] == "This is a verified test response from Turing Engine."
        assert res["raw"]["id"] == "chatcmpl-123"

def test_turing_llm_callable_mocked():
    llm = TuringLLM(model="deepseek-r1-1.5b")

    mock_response_data = {
        "choices": [{"message": {"content": "Generated text output"}}]
    }
    mock_resp = io.BytesIO(json.dumps(mock_response_data).encode("utf-8"))

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        output = llm("Hello prompt")
        assert output == "Generated text output"

def test_chat_turing_connection_error_handling():
    chat = ChatTuring(model="deepseek-r1-1.5b", base_url="http://localhost:9999/v1")

    with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        with pytest.raises(ConnectionError) as exc_info:
            chat.invoke("Hello")
        assert "Failed to connect to Turing Engine" in str(exc_info.value)
