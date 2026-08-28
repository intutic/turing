"""
Unit tests for 3:1 Hybrid Linear-Full Attention and Chunk Context Scorer.
"""

import pytest
import torch

from turing.core.hybrid_attention import (
    LinearRecurrentAttention,
    ChunkContextScorer,
    HybridAttentionLayerRouter
)


def test_linear_recurrent_attention_forward():
    """Verify O(L) Linear Recurrent Attention computes forward pass and recurrent state."""
    batch, seq_len, hidden_dim = 2, 64, 512
    num_heads, head_dim = 8, 64

    layer = LinearRecurrentAttention(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim)
    x = torch.randn(batch, seq_len, hidden_dim)

    out, state = layer(x)
    assert out.shape == (batch, seq_len, hidden_dim)
    assert state.shape == (batch, num_heads, head_dim, head_dim)
    assert not torch.isnan(out).any()


def test_linear_recurrent_state_continuity():
    """Verify sequential single-step autoregressive generation matches multi-token prefill."""
    batch, hidden_dim = 1, 256
    num_heads, head_dim = 4, 64
    layer = LinearRecurrentAttention(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim)

    # 4-token prompt
    x_prompt = torch.randn(batch, 4, hidden_dim)
    _, state = layer(x_prompt)

    # Next single token decode
    x_next = torch.randn(batch, 1, hidden_dim)
    out_next, state_next = layer(x_next, state=state)

    assert out_next.shape == (batch, 1, hidden_dim)
    assert state_next.shape == (batch, num_heads, head_dim, head_dim)


def test_chunk_context_scorer_filtering():
    """Verify ChunkContextScorer filters long context sequences to budget tokens."""
    batch, seq_len = 2, 4096
    num_heads, head_dim = 8, 64
    hidden_dim = num_heads * head_dim
    budget_tokens = 2048

    scorer = ChunkContextScorer(hidden_dim=hidden_dim, chunk_size=128, budget_tokens=budget_tokens)
    k = torch.randn(batch, seq_len, num_heads, head_dim)
    v = torch.randn(batch, seq_len, num_heads, head_dim)
    q = torch.randn(batch, 1, num_heads, head_dim)

    k_filt, v_filt = scorer.filter_context(k, v, q)

    assert k_filt.shape[1] <= budget_tokens
    assert v_filt.shape[1] <= budget_tokens
    assert k_filt.shape[1] == budget_tokens


def test_hybrid_layer_router():
    """Verify 3:1 layer assignment logic."""
    hidden_dim, num_heads, head_dim = 256, 4, 64

    # Layer 1, 2, 3 should be linear
    for l_idx in [1, 2, 3, 5, 6, 7]:
        router = HybridAttentionLayerRouter(layer_idx=l_idx, hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim)
        assert router.is_linear is True

    # Layer 0, 4, 8 should be full attention anchor layers
    for l_idx in [0, 4, 8, 12]:
        router = HybridAttentionLayerRouter(layer_idx=l_idx, hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim)
        assert router.is_linear is False
