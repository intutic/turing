"""
Unit & Integration Test Suite for Turing Programmatic DSL.
Verifies @chain decorator, sequential gen(), parallel fork(), multi-branch join(),
constrained select(), and structured output constrain().
"""

import json

import pytest
import torch
from turing.dsl import (
    chain,
    gen,
    fork,
    join,
    select,
    constrain,
    ChainContext,
    BranchContext,
    LocalExecutor
)
from turing.models.registry import get_model_config
from turing.models.causal_lm import SubspaceCausalLM


def test_gen_outside_context_raises():
    with pytest.raises(RuntimeError, match="No active Turing DSL context found"):
        gen("This should fail")


def test_basic_chain_sequential_gen():
    @chain(model="test-tiny", device="cpu")
    def simple_workflow(prompt: str) -> str:
        r1 = gen(prompt, max_tokens=4, temperature=0.0)
        r2 = gen(" Next step:", max_tokens=4, temperature=0.0)
        return r1 + r2

    out = simple_workflow("Artificial Intelligence")
    assert isinstance(out, str)
    assert len(out) > 0


def test_dsl_fork_and_join_best():
    @chain(model="test-tiny", device="cpu")
    def parallel_thought(question: str) -> str:
        gen(question, max_tokens=4, temperature=0.5)
        branches = fork(3, temperature=0.7)
        for b in branches:
            b.gen("Reasoning branch:", max_tokens=4)
        winner = join(branches, strategy="best")
        return winner

    winner_text = parallel_thought("Solve this problem:")
    assert isinstance(winner_text, str)
    assert len(winner_text) > 0


def test_dsl_fork_and_join_vote():
    @chain(model="test-tiny", device="cpu")
    def majority_vote_workflow():
        gen("Vote test:", max_tokens=2)
        branches = fork(3)
        branches[0].branch_text = " Option A "
        branches[1].branch_text = " Option B "
        branches[2].branch_text = " Option A "
        winner = join(branches, strategy="vote")
        return winner

    result = majority_vote_workflow()
    assert result == "Option A"


def test_dsl_fork_and_join_concat():
    @chain(model="test-tiny", device="cpu")
    def concat_workflow():
        branches = fork(2)
        branches[0].branch_text = "Part 1"
        branches[1].branch_text = "Part 2"
        combined = join(branches, strategy="concat")
        return combined

    result = concat_workflow()
    assert "Part 1" in result
    assert "Part 2" in result


def test_dsl_select_options():
    @chain(model="test-tiny", device="cpu")
    def classification_task(text: str) -> str:
        gen(f"Review: {text}\nSentiment: ")
        decision = select(["Positive", "Negative", "Neutral"])
        return decision

    res = classification_task("This is absolutely wonderful!")
    assert res in ["Positive", "Negative", "Neutral"]


def test_dsl_constrain_json():
    """constrain() is token-level constrained decoding: on a model with a tokenizer every gen() is schema-valid JSON;
    a model without one (test-tiny) refuses instead of falling back to prompting."""
    import jsonschema
    from tests.test_generation_api import tiny_chat_model, tokenizer
    from turing.serving.structured import StructuredOutputError
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "maxLength": 8},
            "score": {"type": "integer", "minimum": 0, "maximum": 100}
        },
        "required": ["name", "score"],
        "additionalProperties": False
    }

    @chain(model="test-tiny", device="cpu")
    def synthetic() -> dict:
        constrain(schema)
        from turing.dsl.context import get_active_context
        assert get_active_context().current_schema == schema
        return gen("Generate JSON:", max_tokens=4)

    with pytest.raises(StructuredOutputError, match="tokenizer"):
        synthetic()

    tok = tokenizer("Qwen/Qwen3-0.6B")
    executor = LocalExecutor(model=tiny_chat_model(tok), tokenizer=tok, device="cpu")

    @chain(executor=executor)
    def structured() -> list:
        constrain(schema)
        return [gen("A player:", max_tokens=48, temperature=1.0) for _ in range(3)]

    try:
        torch.manual_seed(0)
        for out in structured():
            jsonschema.validate(json.loads(out), schema)
    finally:
        executor.close()


def test_fork_prefills_the_prefix_once_and_branches_equal_sequential_greedy():
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    model = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"))
    model.config.kv_pool_gib = 0.01
    executor = LocalExecutor(model=model, device="cpu")
    prompts = [" alpha", " beta beta", " gamma!"]

    @chain(executor=executor)
    def workflow():
        from turing.dsl.context import get_active_context
        gen("A shared question that every branch continues:", max_tokens=6, temperature=0.0)
        prefix = list(get_active_context().history_tokens)
        branches = fork(len(prompts), temperature=0.0)
        futures = [b.gen(p, max_tokens=10) for b, p in zip(branches, prompts)]
        assert all(hasattr(f, "result") for f in futures)  # started, not finished: join() collects them
        join(branches, strategy="concat")
        return prefix, branches

    try:
        prefix, branches = workflow()
        assert executor.prefix_prefills == 1
        assert executor.engine.kv_pool.used_pages == 0  # the shared prefix was released at join()
        for b, p in zip(branches, prompts):
            suffix = [ord(c) % executor.vocab_size for c in p]
            text, tokens, logprobs = executor.generate(prefix + suffix, max_tokens=10, temperature=0.0)
            assert b.branch_tokens == suffix + tokens and b.branch_text == p + text
            assert len(b.token_logprobs) == len(tokens) and all(lp <= 0 for lp in b.token_logprobs)
    finally:
        executor.close()


def test_custom_executor_injection():
    cfg = get_model_config("test-tiny")
    model = SubspaceCausalLM(cfg).eval()
    executor = LocalExecutor(model=model)

    @chain(executor=executor)
    def custom_exec_chain():
        return gen("Custom executor test", max_tokens=4)

    out = custom_exec_chain()
    assert isinstance(out, str)
