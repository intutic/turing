"""
Unit & Integration Test Suite for Turing Programmatic DSL.
Verifies @chain decorator, sequential gen(), parallel fork(), multi-branch join(),
constrained select(), and structured output constrain().
"""

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
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "score": {"type": "integer"}
        },
        "required": ["name", "score"]
    }

    @chain(model="test-tiny", device="cpu")
    def structured_task() -> dict:
        constrain(schema)
        # Verify schema is stored in active context
        from turing.dsl.context import get_active_context
        ctx = get_active_context()
        assert ctx.current_schema == schema
        out = gen("Generate JSON:", max_tokens=4)
        return ctx.current_schema

    ret_schema = structured_task()
    assert ret_schema == schema


def test_custom_executor_injection():
    cfg = get_model_config("test-tiny")
    model = SubspaceCausalLM(cfg).eval()
    executor = LocalExecutor(model=model)

    @chain(executor=executor)
    def custom_exec_chain():
        return gen("Custom executor test", max_tokens=4)

    out = custom_exec_chain()
    assert isinstance(out, str)
