"""
Tests for Multi-Tenant LoRA GPULRUAdapterCache and Dynamic Routing in SubspaceCausalLM.
"""

import pytest
import torch
from turing.config import ModelConfig
from turing.models.adapters import GPULRUAdapterCache, MultiTenantAdapterManager, TenantLoRAAdapter
from turing.models.causal_lm import SubspaceCausalLM

def test_gpu_lru_adapter_cache_hit_miss_eviction():
    hidden_dim = 64
    rank = 4
    capacity = 3
    device = "cpu"

    cache = GPULRUAdapterCache(
        hidden_dim=hidden_dim,
        rank=rank,
        capacity=capacity,
        device=device
    )

    # 1. Register 4 tenant adapters into host store
    for i in range(1, 5):
        cache.register_host_adapter(f"tenant_{i}")

    assert len(cache.host_store) == 4
    assert len(cache.gpu_slots) == 0

    # 2. Access tenant_1, tenant_2, tenant_3 -> 3 misses, 0 evictions
    a1 = cache.get_adapter("tenant_1")
    a2 = cache.get_adapter("tenant_2")
    a3 = cache.get_adapter("tenant_3")

    assert cache.misses == 3
    assert cache.hits == 0
    assert cache.evictions == 0
    assert len(cache.gpu_slots) == 3

    # 3. Access tenant_1 again -> 1 hit
    a1_again = cache.get_adapter("tenant_1")
    assert cache.hits == 1
    assert a1 is a1_again

    # 4. Access tenant_4 -> triggers LRU eviction of tenant_2
    a4 = cache.get_adapter("tenant_4")
    assert cache.misses == 4
    assert cache.evictions == 1
    assert "tenant_2" not in cache.gpu_slots
    assert "tenant_1" in cache.gpu_slots
    assert "tenant_3" in cache.gpu_slots
    assert "tenant_4" in cache.gpu_slots


def test_subspace_causal_lm_dynamic_tenant_routing():
    config = ModelConfig(
        name="test-lora-model",
        vocab_size=100,
        hidden_dim=64,
        ffn_dim=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        tile_size=32,
        active_tiles=2,
        rank_sub=16,
        max_position_embeddings=128
    )


    adapter_manager = MultiTenantAdapterManager(
        hidden_dim=config.hidden_dim,
        rank=4,
        cache_capacity=4,
        device="cpu"
    )

    model = SubspaceCausalLM(config, adapter_manager=adapter_manager)

    input_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)

    # 1. Base forward (no adapter)
    logits_base, _ = model(input_ids)

    # 2. Tenant A forward
    logits_tenant_a, _ = model(input_ids, tenant_id="tenant_sql")

    # 3. Tenant B forward
    logits_tenant_b, _ = model(input_ids, tenant_id="tenant_code")

    # Tenant forward produces valid shapes
    assert logits_base.shape == logits_tenant_a.shape
    assert logits_tenant_a.shape == (1, 3, config.vocab_size)

    # Generation with tenant_id
    out_tokens = model.generate([10, 20], max_new_tokens=4, tenant_id="tenant_sql")
    assert len(out_tokens) == 6
