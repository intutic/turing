"""GSM8K evaluation: answer parsing, prompt format, and a run through the engine on real GPT-2 weights."""

import pytest

from turing.config import TuringConfig
from turing.models.hf_loader import RealHuggingFaceLoader
from turing.serving.accuracy_eval import (GSM8KEvaluator, build_prompt, extract_answer, first_answer, gold_answer)
from turing.serving.engine import ContinuousBatchEngine

SHOT = {"question": "Tom has 3 apples and buys 2 more. How many apples does he have?",
        "answer": "Tom has 3 + 2 = <<3+2=5>>5 apples.\n#### 5"}


def test_answer_parsing():
    assert gold_answer("48 / 2 = 24 clips in May.\n#### 1,072") == "1072"
    assert extract_answer("so 3 + 4 = 7.\n#### 7") == "7"
    assert extract_answer("#### 12 and later 99") == "12"          # the marked answer wins over later numbers
    assert extract_answer("She earns $10.00 in total.") == "10"     # no marker: last number, normalised
    assert extract_answer("no digits") is None
    assert first_answer(" 5 apples.\n#### 5\n\nQuestion: next one") == " 5 apples.\n#### 5"


def test_prompt_format():
    p = build_prompt("What is 2 + 2?", [SHOT])
    assert p == f"Question: {SHOT['question']}\nAnswer: {SHOT['answer']}\n\nQuestion: What is 2 + 2?\nAnswer:"


def test_gsm8k_through_engine_on_gpt2():
    model, tok = RealHuggingFaceLoader.load_hf_model_into_turing(hf_model_id="gpt2", device="cpu")
    engine = ContinuousBatchEngine(model.config, TuringConfig(device="cpu"), model=model, tokenizer=tok)
    problems = [{"question": "Ann has 4 pens and gets 3 more. How many pens does she have?", "answer": "#### 7"},
                {"question": "A box holds 6 eggs. How many eggs are in 2 boxes?", "answer": "#### 12"}]
    res = GSM8KEvaluator(engine).evaluate(problems, [SHOT], max_new_tokens=16, batch_size=2)
    assert res["n"] == 2 and res["shots"] == 1
    assert [r["gold"] for r in res["results"]] == ["7", "12"]
    for r in res["results"]:
        assert r["correct"] == (r["predicted"] == r["gold"])
        assert r["predicted"] == extract_answer(r["completion"])
    assert res["exact_match"] == sum(r["correct"] for r in res["results"]) / 2


def test_evaluator_needs_a_tokenizer():
    from turing.models.registry import get_model_config
    engine = ContinuousBatchEngine(get_model_config("test-tiny"), TuringConfig(device="cpu"))
    with pytest.raises(ValueError):
        GSM8KEvaluator(engine)
