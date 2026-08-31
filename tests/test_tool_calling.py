"""
Unit tests for Tool Calling & Function Calling parser.
"""

import pytest
from fastapi.testclient import TestClient
from turing.config import TuringConfig
from turing.models.registry import get_model_config
from turing.serving.engine import ContinuousBatchEngine
from turing.serving.server import create_app
from turing.serving.tools import ToolCallingHandler, ToolCall, FunctionCall


def test_tool_calling_instruction_injection():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    prompt = "What is the weather in Tokyo?"
    injected = ToolCallingHandler.inject_tools_instruction(prompt, tools)
    assert "# TOOLS & FUNCTIONS" in injected
    assert "get_current_weather" in injected
    assert "<tool_call>" in injected


def test_tool_calling_extraction():
    # 1. Output containing <tool_call> tags
    raw_output = (
        "Let me check that for you.\n"
        "<tool_call>\n"
        '{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "unit": "celsius"}}\n'
        "</tool_call>\n"
        "I'll have the weather shortly."
    )
    clean_text, calls = ToolCallingHandler.extract_tool_calls(raw_output)
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "get_current_weather"
    assert "San Francisco, CA" in calls[0]["function"]["arguments"]
    assert "<tool_call>" not in clean_text

    # 2. Output containing direct JSON tool call
    json_output = '{"name": "calculate_tax", "arguments": {"income": 100000, "state": "CA"}}'
    clean_text, calls = ToolCallingHandler.extract_tool_calls(json_output)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "calculate_tax"


def test_server_chat_completions_with_tools():
    cfg = get_model_config("test-tiny")
    jcfg = TuringConfig(device="cpu", max_batch_size=4)
    engine = ContinuousBatchEngine(cfg, jcfg)
    app = create_app(engine)

    with TestClient(app) as client:
        payload = {
            "model": "test-tiny",
            "messages": [{"role": "user", "content": "What's the weather in Seattle?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Fetch weather",
                        "parameters": {"type": "object", "properties": {"loc": {"type": "string"}}}
                    }
                }
            ],
            "max_tokens": 16
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) == 1
