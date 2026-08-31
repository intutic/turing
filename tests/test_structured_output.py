"""
Unit tests for Structured Outputs, JSON Mode, and JSONSchema validation.
"""

import pytest
from fastapi.testclient import TestClient
from turing.config import TuringConfig
from turing.models.registry import get_model_config
from turing.serving.engine import ContinuousBatchEngine
from turing.serving.server import create_app
from turing.serving.structured import StructuredOutputParser


def test_structured_output_json_injection():
    prompt = "Give me user info."
    injected = StructuredOutputParser.inject_json_instruction(prompt)
    assert "IMPORTANT: You must respond ONLY with a valid JSON object" in injected

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }
    injected_schema = StructuredOutputParser.inject_json_instruction(prompt, schema=schema, schema_name="UserProfile")
    assert "UserProfile" in injected_schema
    assert '"required": [' in injected_schema


def test_structured_output_extract_json():
    # 1. Plain valid JSON
    valid, obj, _ = StructuredOutputParser.extract_json('{"key": "value", "count": 42}')
    assert valid is True
    assert obj == {"key": "value", "count": 42}

    # 2. JSON in markdown block
    valid, obj, _ = StructuredOutputParser.extract_json('Here is the output:\n```json\n{"status": "ok"}\n```\nDone.')
    assert valid is True
    assert obj == {"status": "ok"}

    # 3. Invalid non-JSON
    valid, obj, _ = StructuredOutputParser.extract_json('Just some plain text.')
    assert valid is False


def test_structured_output_auto_repair_truncated():
    # Truncated string inside object
    truncated = '{"name": "Alice", "hobbies": ["reading", "swim'
    repaired = StructuredOutputParser.repair_truncated_json(truncated)
    valid, obj, _ = StructuredOutputParser.extract_json(repaired)
    assert valid is True
    assert obj["name"] == "Alice"
    assert "hobbies" in obj


def test_structured_output_schema_validation():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["name", "age"]
    }

    # Valid
    valid, err = StructuredOutputParser.validate_schema({"name": "Bob", "age": 30, "tags": ["admin", "dev"]}, schema)
    assert valid is True
    assert err is None

    # Missing required property
    valid, err = StructuredOutputParser.validate_schema({"name": "Bob"}, schema)
    assert valid is False
    assert "Missing required property 'age'" in err

    # Wrong type
    valid, err = StructuredOutputParser.validate_schema({"name": "Bob", "age": "thirty"}, schema)
    assert valid is False
    assert "Invalid property 'age'" in err


def test_server_chat_completions_with_response_format():
    cfg = get_model_config("test-tiny")
    jcfg = TuringConfig(device="cpu", max_batch_size=4)
    engine = ContinuousBatchEngine(cfg, jcfg)
    app = create_app(engine)

    with TestClient(app) as client:
        payload = {
            "model": "test-tiny",
            "messages": [{"role": "user", "content": "Return a JSON object"}],
            "response_format": {"type": "json_object"},
            "max_tokens": 16
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) == 1
