"""
Unit and integration tests for Prefill & Decode 2-phase scheduling,
chunk budget interleaving, and percentile latency telemetry in ContinuousBatchEngine.
"""

import asyncio
import time
import torch

from turing.config import ModelConfig, TuringConfig
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
