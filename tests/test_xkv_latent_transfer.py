"""
Unit tests for XKV Cross-Model Latent KV Cache Transfer and Auditable Semantic Inspector (arXiv:2608.20617).
"""

import pytest
import torch
from turing.config import ModelConfig
from turing.core.cross_model_kv import (
    XKVLayerAlignmentTransport,
    XKVHeadSummaryExtractor,
    XKVLatentAgentBridge
)
from turing.demo.epistemic_gate import AuditableSemanticInspector
from turing.demo.agent_system import MultiAgentCoordinator
from turing.demo.engine_wrapper import TuringAcceleratedGenerator

def test_xkv_layer_alignment_transport():
    src_layers, tgt_layers = 45, 32 # E.g., GLM-5.3-Flash (45 layers) -> LLaMA-3.3 (32 layers)
    transport = XKVLayerAlignmentTransport(src_layers=src_layers, tgt_layers=tgt_layers, sigma=0.12)
    matrix = transport()

    assert matrix.shape == (src_layers, tgt_layers)
    assert (matrix >= 0.0).all()
    # Check that each target layer receives a normalized sum of 1.0 from source layers
    col_sums = matrix.sum(dim=0)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-5)

def test_xkv_head_summary_extractor():
    batch, seq_len, num_heads, head_dim = 2, 64, 8, 128
    num_summary = 4
    extractor = XKVHeadSummaryExtractor(head_dim=head_dim, num_heads=num_heads, num_summary_tokens=num_summary)

    k_content = torch.randn(batch, seq_len, num_heads, head_dim)
    v_cache = torch.randn(batch, seq_len, num_heads, head_dim)

    k_sum, v_sum = extractor(k_content, v_cache)
    assert k_sum.shape == (batch, num_summary, num_heads, head_dim)
    assert v_sum.shape == (batch, num_summary, num_heads, head_dim)

def test_xkv_latent_agent_bridge_heterogeneous_transfer():
    # Source: 45 layers, 8 KV heads, 128 head dim
    src_cfg = ModelConfig(
        name="Source-Model",
        hidden_dim=4096,
        ffn_dim=11008,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        num_layers=45
    )
    # Target: 32 layers, 4 KV heads, 128 head dim
    tgt_cfg = ModelConfig(
        name="Target-Model",
        hidden_dim=2048,
        ffn_dim=5632,
        num_heads=16,
        num_kv_heads=4,
        head_dim=128,
        num_layers=32
    )

    bridge = XKVLatentAgentBridge(
        source_config=src_cfg,
        target_config=tgt_cfg,
        num_summary_tokens=4
    )

    batch, seq_len = 1, 32
    source_keys = [torch.randn(batch, seq_len, src_cfg.num_kv_heads, src_cfg.head_dim) for _ in range(src_cfg.num_layers)]
    source_values = [torch.randn(batch, seq_len, src_cfg.num_kv_heads, src_cfg.head_dim) for _ in range(src_cfg.num_layers)]

    tgt_keys, tgt_values, shared_latent = bridge.transfer_latent_kv(source_keys, source_values)

    assert len(tgt_keys) == tgt_cfg.num_layers
    assert len(tgt_values) == tgt_cfg.num_layers
    assert tgt_keys[0].shape == (batch, 4, tgt_cfg.num_kv_heads, tgt_cfg.head_dim)
    assert tgt_values[0].shape == (batch, 4, tgt_cfg.num_kv_heads, tgt_cfg.head_dim)
    assert shared_latent.shape == (batch, 4, src_cfg.num_kv_heads * src_cfg.head_dim)

def test_auditable_semantic_inspector():
    latent_dim = 512
    vocab_size = 1000
    inspector = AuditableSemanticInspector(latent_dim=latent_dim, vocab_size=vocab_size, top_k=3)

    batch, num_summary = 1, 4
    shared_latent = torch.randn(batch, num_summary, latent_dim)

    mock_vocab = {i: f"word_{i}" for i in range(vocab_size)}
    report = inspector.audit_latent_state(shared_latent, tokenizer_vocab=mock_vocab)

    assert report["audit_status"] == "PASSED"
    assert report["summary_positions"] == 4
    assert len(report["top_concepts"]) == 4 # 4 summary tokens
    assert len(report["top_concepts"][0]) == 3 # top 3 concepts
    assert "concept_id" in report["top_concepts"][0][0]
    assert "prob" in report["top_concepts"][0][0]

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(256, 64)
        self.head = torch.nn.Linear(64, 256)

    def forward(self, input_ids, **kwargs):
        class Output:
            def __init__(self, logits):
                self.logits = logits
        h = self.embed(input_ids)
        logits = self.head(h)
        return Output(logits)

    def generate(self, input_ids, max_new_tokens=10, **kwargs):
        new_ids = torch.tensor([[42] * 5], device=input_ids.device)
        return torch.cat([input_ids, new_ids], dim=-1)


class MockTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["content"] for m in messages)

    def __call__(self, text, **kwargs):
        class BatchEncoding(dict):
            def __init__(self):
                super().__init__({"input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long)})
            def to(self, dev):
                self["input_ids"] = self["input_ids"].to(dev)
                return self
        return BatchEncoding()

    def decode(self, token_ids, **kwargs):
        return "Configured resilient multi-cloud edge mesh."


def test_multi_agent_coordinator_latent_deliberation():
    engine = TuringAcceleratedGenerator(
        model_id_or_instance=MockModel(),
        tokenizer=MockTokenizer(),
        sparsity_ratio=0.57,
        device="cpu"
    )
    coordinator = MultiAgentCoordinator(engine=engine)

    result = coordinator.run_xkv_latent_deliberation(
        user_scenario="Configure resilient multi-cloud edge mesh."
    )

    assert result["mode"] == "XKV_ZERO_TOKEN_LATENT_DELIBERATION"
    assert result["total_latency_ms"] > 0
    assert "audit_report" in result
    assert result["audit_report"]["audit_status"] == "PASSED"
    assert "measured_speedup" in result
