"""
Unit and integration tests for Prefill & Decode 2-phase scheduling,
chunk budget interleaving, and percentile latency telemetry in ContinuousBatchEngine.
"""

import asyncio
import time
import torch

from turing.config import ModelConfig, TuringConfig
from turing.models.registry import get_model_config
from turing.serving.engine import ContinuousBatchEngine, RequestState, AsyncSequenceRequest
from turing.serving.traffic import LanePolicy, Lane, AdmissionController


def _create_mini_engine(chunk_size: int = 16, lane_policy: LanePolicy = None) -> ContinuousBatchEngine:
    config = ModelConfig(
        name="test-mini",
        vocab_size=128,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ffn_dim=128,
        max_position_embeddings=1024,
        head_dim=16
    )
    turing_config = TuringConfig(device="cpu", max_batch_size=8)
    engine = ContinuousBatchEngine(
        model_config=config,
        turing_config=turing_config,
        prefill_chunk_size=chunk_size,
        lane_policy=lane_policy
    )
    return engine


def test_chunked_prefill_state_progression():
    """Verifies that a 35-token prompt advances in chunks of 16, transitions to DECODING, and completes."""
    async def _run():
        mini_engine = _create_mini_engine(chunk_size=16)
        prompt = list(range(35))
        await mini_engine.start()
        try:
            tokens = []
            async for token in mini_engine.stream_generate(prompt_tokens=prompt, max_new_tokens=5, temperature=0.0):
                tokens.append(token)

            assert len(tokens) == 5
            assert all(isinstance(t, int) for t in tokens)
        finally:
            await mini_engine.stop()

    asyncio.run(_run())


def test_a_prompt_that_fits_a_step_is_not_split_over_two():
    """With 16 prompt tokens a step, a 10-token prompt behind another waits for the next step instead of starting
    with the 6 tokens left; a 40-token prompt (longer than a step) takes what is left of each step."""
    async def _run():
        engine = _create_mini_engine(chunk_size=16)
        steps, forward, step = [], engine._forward, engine._step_batch

        def spy(ids, *a, **kw):
            if ids.shape[1] > 1:  # a prompt chunk
                steps[-1].append(ids.shape[1])
            return forward(ids, *a, **kw)

        def mark():
            steps.append([])
            step()
        engine._forward, engine._step_batch = spy, mark
        reqs = [await engine.add_request(prompt_tokens=list(range(1, n + 1)), max_new_tokens=2, temperature=0.0)
                for n in (10, 10, 40)]
        await engine.start()
        try:
            for r in reqs:
                async for _ in engine.stream_request(r):
                    pass
        finally:
            await engine.stop()
        assert [s for s in steps if s] == [[10], [10, 6], [16], [16], [2]]

    asyncio.run(_run())


def test_a_step_budget_above_one_forward_is_spent_over_several_steps():
    """Forwards of 16 prompt tokens: with 16 a step the request already decoding takes a token beside every forward
    of the long prompt; with 32 a step the prompt still moves 16 tokens a scheduler step and that request takes a
    token with every second forward (the last of each budget)."""
    async def _run(chunk):
        engine = _create_mini_engine(chunk_size=chunk)
        engine.FORWARD_TOKENS = 16
        log, forward = [], engine._forward

        def spy(ids, *a, **kw):
            log.append(ids.shape[1] if ids.shape[1] > 1 else "d")  # a prompt chunk's tokens, or a decode forward
            return forward(ids, *a, **kw)
        engine._forward = spy
        reqs = [await engine.add_request(prompt_tokens=list(range(1, n + 1)), max_new_tokens=m, temperature=0.0)
                for n, m in ((3, 8), (64, 2))]
        await engine.start()
        try:
            for r in reqs:
                async for _ in engine.stream_request(r):
                    pass
        finally:
            await engine.stop()
        return log

    assert asyncio.run(_run(16))[:9] == [3, 13, 16, "d", 16, "d", 16, "d", 3]
    assert asyncio.run(_run(32))[:9] == [3, 13, 16, "d", 16, 16, "d", 3, "d"]


def test_interleaved_prefill_decode_scheduling():
    """Verifies that an incoming 64-token prefill interleaves with an active decoding request without starving it."""
    async def _run():
        mini_engine = _create_mini_engine(chunk_size=16)
        await mini_engine.start()
        try:
            # Request 1: short prompt (starts decoding quickly)
            gen1 = mini_engine.stream_generate(prompt_tokens=[1, 2, 3], max_new_tokens=8, temperature=0.0)
            tok1_first = await gen1.__anext__()
            assert isinstance(tok1_first, int)

            # Request 2: long prompt (64 tokens, needs 4 chunks of 16)
            gen2 = mini_engine.stream_generate(prompt_tokens=list(range(64)), max_new_tokens=4, temperature=0.0)

            tokens1 = [tok1_first]
            tokens2 = []

            async def collect(gen, dest):
                async for t in gen:
                    dest.append(t)

            await asyncio.gather(
                collect(gen1, tokens1),
                collect(gen2, tokens2)
            )

            assert len(tokens1) == 8
            assert len(tokens2) == 4
        finally:
            await mini_engine.stop()

    asyncio.run(_run())


def test_percentile_telemetry_tracking():
    """Verifies that P50, P95, and P99 latency percentiles are tracked and populated in telemetry."""
    async def _run():
        mini_engine = _create_mini_engine(chunk_size=16)
        await mini_engine.start()
        try:
            # Run 4 requests to generate latency samples
            for _ in range(4):
                tokens = []
                async for t in mini_engine.stream_generate(prompt_tokens=[1, 2, 3, 4], max_new_tokens=6, temperature=0.0):
                    tokens.append(t)
                assert len(tokens) == 6

            telemetry = mini_engine.get_telemetry()
            assert telemetry["total_requests_completed"] == 4
            assert telemetry["total_tokens_generated"] == 24

            lat = telemetry["latency"]
            assert "avg_ttft_ms" in lat
            assert "p50_ttft_ms" in lat
            assert "p95_ttft_ms" in lat
            assert "p99_ttft_ms" in lat
            assert "avg_itl_ms" in lat
            assert "p50_itl_ms" in lat
            assert "p95_itl_ms" in lat
            assert "p99_itl_ms" in lat

            assert lat["p50_ttft_ms"] >= 0.0
            assert lat["p50_itl_ms"] >= 0.0
        finally:
            await mini_engine.stop()

    asyncio.run(_run())


def test_lane_budget_prefill_differentiation():
    """Verifies that LanePolicy custom prefill chunk budgets are respected."""
    async def _run():
        policy = LanePolicy(
            interactive_prefill_chunk=32,
            batch_prefill_chunk=16,
            background_prefill_chunk=8
        )
        engine = _create_mini_engine(chunk_size=16, lane_policy=policy)

        assert policy.prefill_chunk_budget(Lane.INTERACTIVE) == 32
        assert policy.prefill_chunk_budget(Lane.BATCH) == 16
        assert policy.prefill_chunk_budget(Lane.BACKGROUND) == 8

        await engine.start()
        try:
            tokens = []
            async for t in engine.stream_generate(
                prompt_tokens=list(range(40)),
                max_new_tokens=4,
                lane=Lane.INTERACTIVE,
                temperature=0.0
            ):
                tokens.append(t)
            assert len(tokens) == 4
        finally:
            await engine.stop()

    asyncio.run(_run())


def test_latency_percentiles_are_null_until_there_are_samples():
    """One request gives one TTFT: its mean is known, its percentiles are not (no average stands in for a P99)."""
    async def _run():
        engine = ContinuousBatchEngine(get_model_config("test-tiny"), TuringConfig(device="cpu", max_batch_size=4))
        await engine.start()
        try:
            async for _ in engine.stream_generate(prompt_tokens=[1, 2, 3], max_new_tokens=1, temperature=0.0):
                pass
            return engine.get_telemetry()["latency"]
        finally:
            await engine.stop()
    lat = asyncio.run(_run())
    assert lat["avg_ttft_ms"] > 0.0
    assert lat["p50_ttft_ms"] is None and lat["p99_ttft_ms"] is None and lat["avg_itl_ms"] is None


def test_forward_rows_plan_from_the_measured_times():
    """SubspaceCausalLM._rows_plan with the times per token an L4 measured (2,040, 2,048 and 2,056 rows)."""
    from turing.models.causal_lm import SubspaceCausalLM
    assert SubspaceCausalLM._rows_plan(58.5, 60.9, 62.2) == (2040, False)   # Qwen2.5-1.5B
    assert SubspaceCausalLM._rows_plan(59.1, 60.2, 62.7) == (2040, False)   # the same model, a noisier run
    assert SubspaceCausalLM._rows_plan(69.3, 68.9, 77.7) == (2048, True)    # SmolLM2-1.7B
    assert SubspaceCausalLM._rows_plan(20.0, 20.1, 20.3) == (2048, False)
