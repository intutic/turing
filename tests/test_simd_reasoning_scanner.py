import pytest
from turing.serving.reasoning import ReasoningStreamFilter
import turing.turing_csrc as turing_csrc


def test_native_reasoning_filter_direct_csrc():
    """Verify C++ NativeReasoningFilter tag splitting behavior."""
    scanner = turing_csrc.NativeReasoningFilter()
    
    # 1. Normal before-think text
    r1, c1 = scanner.process_token("Hello! ")
    assert r1 is None
    assert c1 == "Hello! "
    assert not scanner.in_thinking_block

    # 2. Enter <think>
    r2, c2 = scanner.process_token("Let's consider: <think>")
    assert r2 is None
    assert c2 == "Let's consider: "
    assert scanner.in_thinking_block

    # 3. Stream inside think
    r3, c3 = scanner.process_token("The answer is 42 because 6 * 7 = 42.")
    assert r3 == "The answer is 42 because 6 * 7 = 42."
    assert c3 is None

    # 4. Exit </think>
    r4, c4 = scanner.process_token("</think> Therefore, 42.")
    assert r4 is None
    assert c4 == " Therefore, 42."
    assert not scanner.in_thinking_block
    assert "The answer is 42 because 6 * 7 = 42." in scanner.thinking_content


def test_reasoning_stream_filter_e2e():
    """Verify ReasoningStreamFilter correctly extracts reasoning and final content."""
    rf = ReasoningStreamFilter()
    tokens = ["Here is ", "the logic: ", "<think>", "Step 1: check inputs. ", "Step 2: compute.", "</think>", " Result is ready."]
    
    reasoning_collected = []
    content_collected = []

    for tok in tokens:
        r, c = rf.process_token(tok)
        if r:
            reasoning_collected.append(r)
        if c:
            content_collected.append(c)

    assert "Step 1: check inputs." in "".join(reasoning_collected)
    assert "Result is ready." in "".join(content_collected)
    assert "<think>" not in "".join(content_collected)
    assert "</think>" not in "".join(content_collected)
