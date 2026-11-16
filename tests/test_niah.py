"""Pass-key needle-in-a-haystack through the engine on real SmolLM2-135M weights."""

import pytest

from turing.config import TuringConfig
from turing.models.hf_loader import RealHuggingFaceLoader
from turing.serving.engine import ContinuousBatchEngine
from turing.serving.niah import FILLER, LongContextNIAHEvaluator, NEEDLE


@pytest.fixture(scope="module")
def evaluator():
    model, tok = RealHuggingFaceLoader.load_hf_model_into_turing(hf_model_id="HuggingFaceTB/SmolLM2-135M", device="cpu")
    return LongContextNIAHEvaluator(ContinuousBatchEngine(model.config, TuringConfig(device="cpu"), model=model,
                                                          tokenizer=tok))


def test_prompt_length_and_needle_depth(evaluator):
    tok = evaluator.engine.tokenizer
    filler_len = len(tok.encode(FILLER))
    for depth in (0.0, 0.5, 1.0):
        text = evaluator.build_prompt(512, depth, "12345")
        assert 512 - filler_len - 4 <= len(tok.encode(text)) <= 512 + 4
        before, after = text.split(NEEDLE.format(key="12345"))
        n_before, n_after = before.count(FILLER), after.count(FILLER)
        assert n_before == round(depth * (n_before + n_after))


def test_retrieval_runs_through_engine(evaluator):
    res = evaluator.evaluate_retrieval([256], [0.0, 1.0])
    assert res["n"] == 2
    assert res["accuracy"] == sum(r["found"] for r in res["results"]) / 2
    for r in res["results"]:
        assert r["found"] == (r["key"] in r["completion"])
        assert r["prompt_tokens"] <= 256 + 4
