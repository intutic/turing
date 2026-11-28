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


def test_channel_split_routes_reasoning_marked_by_special_tokens():
    """Inkling 2 marks reasoning with special tokens that decoding drops (<|message_model|><|content_thinking|> ...
    <|end_message|><|message_model|><|content_text|> ...): Generation splits on the tokens themselves; the request's
    reasoning effort reaches templates that read one; the end-of-turn token stops the engine."""
    import asyncio
    from types import SimpleNamespace
    from turing.serving.engine import _eos_ids
    from turing.serving.generation import Generation, GenerationParams, thinking_kwargs

    words = {10: "The user asks", 11: " about tides.", 12: "Tides come", 13: " from the moon."}
    marks = {"<|message_model|>": 100, "<|content_thinking|>": 101, "<|content_text|>": 102, "<|end_message|>": 103,
             "<|content_model_end_sampling|>": 104}

    class Tok:
        chat_template = "{{ reasoning_effort }}"
        eos_token_id = 99

        def get_vocab(self):
            return {**marks, **{v: k for k, v in words.items()}}

        def convert_tokens_to_ids(self, token):
            return marks.get(token)

        def decode(self, ids, skip_special_tokens=True):
            return "".join(words.get(i, "") for i in ids)

    out = [100, 101, 10, 11, 103, 100, 102, 12, 13, 103]

    class Engine:
        tokenizer = Tok()
        default_draft_tokens = 0

        def decode_tokens(self, ids):
            return self.tokenizer.decode(ids)

        async def stream_request(self, req):
            for t in out:
                yield t

    req = SimpleNamespace(finish_reason="stop", cached_prompt_tokens=0, draft_tokens=0, logits_processor=None)
    gen = Generation(Engine(), req, GenerationParams(stop=[]), [1, 2], False, None, None)
    asyncio.run(gen.collect())
    assert gen.reasoning_text == "The user asks about tides."
    assert gen.content == "Tides come from the moon."
    assert gen.completion_tokens == len(out)
    plain = Generation(Engine(), req, GenerationParams(stop=[], reasoning=False), [1, 2], False, None, None)
    asyncio.run(plain.collect())  # raw completions keep everything as content
    assert plain.reasoning_text == "" and plain.content == "The user asks about tides.Tides come from the moon."
    assert thinking_kwargs(Engine(), False, "None") == {"reasoning_effort": "none"}
    Tok.chat_template = "no effort here"
    assert thinking_kwargs(Engine(), False, "low") == {}
    assert 104 in _eos_ids(Tok()) and 99 in _eos_ids(Tok())
