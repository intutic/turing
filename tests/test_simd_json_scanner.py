"""
Unit tests for Native C++20 AVX2 SIMD Fast JSON Syntax Scanner & Bracket Balancer.
"""

import pytest
import json
from turing.serving.structured import StructuredOutputParser


def test_simd_json_scanner_binding():
    """Verifies that scan_json_structure_fast detects boundaries and balances brackets correctly."""
    try:
        import turing.turing_csrc as turing_csrc
    except ImportError:
        pytest.skip("turing_csrc native extension not available")

    # Complete valid JSON
    valid_json = '{"name": "turing", "active": true, "items": [1, 2, 3]}'
    res = turing_csrc.scan_json_structure_fast(valid_json)
    assert res["is_valid"] is True
    assert res["first_brace_idx"] == 0
    assert res["last_brace_idx"] == len(valid_json) - 1
    assert res["repair_suffix"] == ""

    # Truncated inside string
    trunc1 = '{"name": "turing'
    res1 = turing_csrc.scan_json_structure_fast(trunc1)
    assert res1["is_valid"] is False
    assert res1["in_string"] is True
    assert res1["repair_suffix"] == '"}'

    # Truncated nested array inside object
    trunc2 = '{"tags": ["llm", "speed", "kernel'
    res2 = turing_csrc.scan_json_structure_fast(trunc2)
    assert res2["is_valid"] is False
    assert res2["repair_suffix"] == '"]}'


def test_structured_output_parser_repair():
    """Verifies that StructuredOutputParser auto-repairs truncated strings into valid parsable JSON."""
    truncated = '{"action": "lookup", "parameters": {"user_id": 42, "query": "hello world'
    repaired = StructuredOutputParser.repair_truncated_json(truncated)
    parsed = json.loads(repaired)
    assert parsed["action"] == "lookup"
    assert parsed["parameters"]["user_id"] == 42
    assert parsed["parameters"]["query"] == "hello world"
