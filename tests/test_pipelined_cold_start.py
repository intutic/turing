"""
Tests for Pipelined Subspace Warmup Loader and Time-to-Ready Telemetry.
"""

import pytest
import torch
from turing.models.streaming_loader import PipelinedSubspaceWarmupLoader
from turing.models.registry import get_model_config

def test_pipelined_subspace_warmup_loader():
    cfg = get_model_config("test-tiny")
    loader = PipelinedSubspaceWarmupLoader(
        model_config=cfg,
        device="cpu",
        warmup_buckets=[1, 4, 16]
    )

    res = loader.pipelined_load_and_warmup(total_layers=cfg.num_layers)

    assert res["status"] == "ready"
    assert "time_to_ready_ms" in res
    assert res["time_to_ready_ms"] >= 0.0
    assert res["bootstrap_layers"] == min(3, cfg.num_layers)
    assert res["captured_buckets"] == [1, 4, 16]
