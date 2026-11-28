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


WEATHER_TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}}}},
    {"type": "function", "function": {"name": "convert_currency", "parameters": {"type": "object", "properties": {
        "amount": {"type": "number"}, "from_currency": {"type": "string"}, "to_currency": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "create_calendar_event", "parameters": {"type": "object", "properties": {
        "title": {"type": "string"}, "date": {"type": "string"}, "time": {"type": "string"},
        "duration_minutes": {"type": "integer"}, "remote": {"type": "boolean"}, "attendees": {"type": "array"}}}}},
]


def test_tool_call_formats_of_qwen36_gemma4_and_inkling2():
    """Calls as the models wrote them through turing serve (scripts/model_quality): Qwen3.5 / 3.6's XML
    (<function=...><parameter=...>), Gemma 4's call:name{key:<|"|>text<|"|>,n:1}, and Inkling 2's {"name", "args"}
    JSON (the channel split drops its <|message_model|> header). XML values take the type the tool's schema gives."""
    import json
    from turing.serving.tools import tool_format

    def one(text, tools=WEATHER_TOOLS):
        clean, calls = ToolCallingHandler.extract_tool_calls(text, tools)
        assert len(calls) == 1, (text, calls)
        return clean, calls[0]["function"]["name"], json.loads(calls[0]["function"]["arguments"])

    qwen = ("<tool_call>\n<function=convert_currency>\n<parameter=amount>\n250\n</parameter>\n<parameter=from_currency>\n"
            "USD\n</parameter>\n<parameter=to_currency>\nEUR\n</parameter>\n</function>\n</tool_call>")
    assert one(qwen) == ("", "convert_currency", {"amount": 250, "from_currency": "USD", "to_currency": "EUR"})
    said = ("I'll search for Python files related to tokenizers for you.\n\n<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nNew York\n</parameter>\n<parameter=unit>\nfahrenheit\n</parameter>\n</function>\n</tool_call>")
    assert one(said) == ("I'll search for Python files related to tokenizers for you.", "get_weather",
                         {"city": "New York", "unit": "fahrenheit"})
    typed = ("<tool_call>\n<function=create_calendar_event>\n<parameter=duration_minutes>\n45\n</parameter>\n"
             "<parameter=remote>\ntrue\n</parameter>\n<parameter=attendees>\n[\"a\", \"b\"]\n</parameter>\n"
             "<parameter=title>\n45\n</parameter>\n<parameter=notes>\n7\n</parameter>\n</function>\n</tool_call>")
    assert one(typed)[2] == {"duration_minutes": 45, "remote": True, "attendees": ["a", "b"], "title": "45", "notes": "7"}
    two = qwen + "\n" + qwen.replace("250", "99.5")
    assert [json.loads(c["function"]["arguments"])["amount"]
            for c in ToolCallingHandler.extract_tool_calls(two, WEATHER_TOOLS)[1]] == [250, 99.5]

    gemma = ('<|tool_call>call:create_calendar_event{date:<|"|>2026-12-03<|"|>,duration_minutes:45,time:<|"|>14:00<|"|>,'
             'title:<|"|>Design review: Q1, {draft}<|"|>,attendees:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>')
    assert one(gemma) == ("", "create_calendar_event", {"date": "2026-12-03", "duration_minutes": 45, "time": "14:00",
                                                       "title": "Design review: Q1, {draft}", "attendees": ["a", "b"]})
    assert one('<|tool_call>call:convert_currency{amount:1000,from_currency:<|"|>JPY<|"|>,to_currency:<|"|>GBP<|"|>}'
               '<tool_call|>')[2] == {"amount": 1000, "from_currency": "JPY", "to_currency": "GBP"}

    inkling = '{"name":"get_weather","args":{"city":"Tokyo","unit":"fahrenheit"}}'
    assert one(inkling) == ("", "get_weather", {"city": "Tokyo", "unit": "fahrenheit"})

    assert tool_format("... <tool_call>\n<function=example_function_name>\n<parameter=x> ...") == "qwen_xml"
    assert tool_format("{{- '<|tool_call>call:' + function['name'] + '{' -}} parameters tool") == "gemma"
    assert tool_format('<|message_model|>" ~ fn.name ~ "<|content_invoke_tool_json|> parameters tool') == "inkling"
    assert tool_format("{{ '<tool_call>' }}") == "hermes" and tool_format("parameters of the tool") == "llama"
    assert ToolCallingHandler.extract_tool_calls("<function=get_weather> unfinished", WEATHER_TOOLS)[1] == []
