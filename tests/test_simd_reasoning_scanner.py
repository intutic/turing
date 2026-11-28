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
             "<|content_model_end_sampling|>": 104, "<|content_invoke_tool_json|>": 105}

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
    words.update({20: "get_weather", 21: '{"name":"get_weather","args":{}}'})
    out[:] = [100, 101, 10, 103, 100, 20, 105, 21, 103]  # reasoning, then a tool call: the header (its name) is not text
    call = Generation(Engine(), req, GenerationParams(stop=[]), [1, 2], False, None, None)
    asyncio.run(call.collect())
    assert call.reasoning_text == "The user asks" and call.content == '{"name":"get_weather","args":{}}'
    assert thinking_kwargs(Engine(), False, "None") == {"reasoning_effort": "none"}
    Tok.chat_template = "no effort here"
    assert thinking_kwargs(Engine(), False, "low") == {}
    assert 104 in _eos_ids(Tok()) and 99 in _eos_ids(Tok())


def test_gemma4_thinking_channel_is_split_and_switched_on():
    """Gemma 4 thinks only with enable_thinking (its template defaults it to false) and writes <|channel>thought ...
    <channel|> then the answer, as visible text: the text filter splits on those tags, held back across chunks, and a
    request's reasoning effort switches the template's enable_thinking either way."""
    from types import SimpleNamespace
    from turing.serving.generation import thinking_kwargs
    from turing.serving.reasoning import ReasoningStreamFilter, think_text_tags

    class Tok:
        chat_template = "{%- set enable_thinking = enable_thinking | default(false) -%}"

        def get_vocab(self):
            return {"<|channel>": 100, "<channel|>": 101, "<|think|>": 98}

    tags = think_text_tags(Tok())
    assert tags == ("<|channel>thought", "<channel|>")
    f = ReasoningStreamFilter(False, *tags)
    r, c = [], []
    for chunk in ["<|chan", "nel>tho", "ught\n17*23 = 391.", "<chann", "el|>", "Answer: 391"]:
        a, b = f.process_token(chunk)
        r.append(a or "")
        c.append(b or "")
    a, b = f.flush()
    assert "".join(r) + (a or "") == "\n17*23 = 391." and "".join(c) + (b or "") == "Answer: 391"
    engine = SimpleNamespace(tokenizer=Tok())
    assert thinking_kwargs(engine, True) == {"enable_thinking": True}
    assert thinking_kwargs(engine, False) == {"enable_thinking": False}
    assert thinking_kwargs(engine, None) == {}
    class Qwen:
        def get_vocab(self):
            return {"<think>": 1, "</think>": 2}

    assert think_text_tags(Qwen()) is None


def test_thinking_budget_covers_gemma4_and_inkling2():
    """A reasoning effort's thinking budget applies to every reasoning model, not only <think> ones: Gemma 4's block
    opens with <|channel> and closes with <channel|>, Inkling 2's opens with <|message_model|><|content_thinking|> and
    closes with <|end_message|>. At the budget the close sequence is forced, so the answer always gets room."""
    import torch
    from turing.serving.reasoning import ThinkingBudget, thinking_block_ids

    class Vocab:
        def __init__(self, v):
            self.v = v

        def get_vocab(self):
            return self.v

    gemma = thinking_block_ids(Vocab({"<|channel>": 100, "<channel|>": 101}))
    inkling = thinking_block_ids(Vocab({"<|message_model|>": 200001, "<|content_thinking|>": 200008,
                                        "<|end_message|>": 200010, "<|content_text|>": 200004}))
    qwen = thinking_block_ids(Vocab({"<think>": 151667, "</think>": 151668}))
    assert gemma == ([100], [101]) and inkling == ([200001, 200008], [200010]) and qwen == ([151667], [151668])
    assert thinking_block_ids(Vocab({"hello": 1})) is None
    for open_ids, close_ids in (gemma, inkling):
        budget = ThinkingBudget(open_ids, close_ids, budget=3)
        gen = list(open_ids) + [5, 6, 7]  # three thinking tokens: the budget is spent
        logits = torch.zeros(200100)
        budget(gen, logits)
        assert int(logits.argmax()) == close_ids[0] and torch.isinf(logits).sum() == logits.numel() - 1
