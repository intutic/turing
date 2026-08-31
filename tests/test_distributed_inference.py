"""
Unit & Integration Test Suite for Distributed Multi-GPU & Multi-Node Inference.
Verifies Tensor Parallelism (TP), Pipeline Parallelism (PP), Micro-Batch Scheduling,
Placement Policy, and DistributedInferenceDriver.
"""

import pytest
import torch
import torch.nn as nn

from turing.models.registry import get_model_config
from turing.models.causal_lm import SubspaceCausalLM
from turing.models.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
    ParallelLMHead,
    PipelineStage,
    partition_model_for_pipeline,
    MicroBatchScheduler,
)
from turing.serving.distributed import (
    DistributedConfig,
    PlacementPolicy,
    DistributedInferenceDriver,
)


def test_placement_policy_decomposition():
    assert PlacementPolicy.auto_decompose(1) == (1, 1)
    assert PlacementPolicy.auto_decompose(2) == (2, 1)
    assert PlacementPolicy.auto_decompose(4, model_params_billion=7.0) == (4, 1)
    assert PlacementPolicy.auto_decompose(4, model_params_billion=70.0) == (2, 2)
    assert PlacementPolicy.auto_decompose(8, model_params_billion=8.0) == (8, 1)
    assert PlacementPolicy.auto_decompose(8, model_params_billion=70.0) == (4, 2)
    assert PlacementPolicy.auto_decompose(16, model_params_billion=120.0) == (8, 2)


def test_column_and_row_parallel_linear():
    x = torch.randn(2, 4, 32)
    # Column parallel (splits out_features 64 -> 32)
    col = ColumnParallelLinear(in_features=32, out_features=64, tp_world_size=2, tp_rank=0)
    out_col = col(x)
    assert out_col.shape == (2, 4, 32)

    # Row parallel (splits in_features 32 -> 16)
    row = RowParallelLinear(in_features=32, out_features=64, tp_world_size=2, tp_rank=0)
    x_split = torch.randn(2, 4, 16)
    out_row = row(x_split)
    assert out_row.shape == (2, 4, 64)


def test_parallel_embedding_and_lm_head():
    input_ids = torch.tensor([[1, 5, 10]])
    embed = ParallelEmbedding(num_embeddings=128, embedding_dim=32, tp_world_size=2, tp_rank=0)
    out_emb = embed(input_ids)
    assert out_emb.shape == (1, 3, 32)

    head = ParallelLMHead(hidden_dim=32, vocab_size=128, tp_world_size=2, tp_rank=0)
    out_head = head(out_emb)
    assert out_head.shape == (1, 3, 64)


def test_pipeline_partitioning():
    cfg = get_model_config("test-tiny")
    model = SubspaceCausalLM(cfg).eval()
    
    stages = partition_model_for_pipeline(model, num_stages=2)
    assert len(stages) == 2
    assert stages[0].is_first_stage is True
    assert stages[0].is_last_stage is False
    assert stages[0].embed_tokens is not None
    assert stages[0].lm_head is None

    assert stages[1].is_first_stage is False
    assert stages[1].is_last_stage is True
    assert stages[1].embed_tokens is None
    assert stages[1].lm_head is not None


def test_micro_batch_scheduler():
    scheduler = MicroBatchScheduler(num_stages=2, num_micro_batches=4)
    # Bubble ratio = (2-1)/(4+2-1) = 1/5 = 0.20
    assert abs(scheduler.bubble_ratio - 0.20) < 1e-4

    cfg = get_model_config("test-tiny")
    model = SubspaceCausalLM(cfg).eval()
    stages = partition_model_for_pipeline(model, num_stages=2)

    micro_batches = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])]
    outs = scheduler.schedule_forward(micro_batches, stages)
    assert len(outs) == 2
    assert outs[0].shape == (1, 3, cfg.vocab_size)
    assert outs[1].shape == (1, 3, cfg.vocab_size)


def test_distributed_inference_driver_pp1():
    cfg = get_model_config("test-tiny")
    model = SubspaceCausalLM(cfg).eval()
    dist_cfg = DistributedConfig(tp_size=1, pp_size=1)
    driver = DistributedInferenceDriver(model=model, config=cfg, dist_config=dist_cfg)

    input_ids = torch.tensor([[1, 5, 10]])
    logits, kvs = driver.forward_distributed(input_ids)
    assert logits.shape == (1, 3, cfg.vocab_size)
    assert kvs is not None
    assert len(kvs) == cfg.num_layers


def test_distributed_inference_driver_pp2():
    cfg = get_model_config("test-tiny")
    model = SubspaceCausalLM(cfg).eval()
    dist_cfg = DistributedConfig(tp_size=1, pp_size=2)
    driver = DistributedInferenceDriver(model=model, config=cfg, dist_config=dist_cfg)

    input_ids = torch.tensor([[2, 4, 8]])
    logits, kvs = driver.forward_distributed(input_ids)
    assert logits.shape == (1, 3, cfg.vocab_size)
    assert kvs is not None
    assert len(kvs) == cfg.num_layers
