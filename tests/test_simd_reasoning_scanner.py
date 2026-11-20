import pytest
from turing.serving.reasoning import ReasoningStreamFilter


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
