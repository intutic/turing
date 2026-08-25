"""
Unit Tests for Compressed Convolutional Attention (CCA) & 1D Conv-Enhanced Speculation (ZAYA1-8B & Poolside Laguna XS.2).
"""

import pytest
import torch

from turing.core.cca import CompressedConvolutionalAttention, LayerwiseHeadBudgeter
from turing.core.speculation import EnhancedQuadtreeDraftHead, TreeNode

def test_compressed_convolutional_attention_forward():
    """
    Tests CCA forward pass with 1D depthwise sequence convolution.
    """
    batch = 2
    seq_len = 32
    hidden_dim = 128
    latent_dim = 64
    num_heads = 4

    cca = CompressedConvolutionalAttention(
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_heads=num_heads,
        kernel_size=3
    )

    x = torch.randn(batch, seq_len, hidden_dim)
    out = cca(x)

    assert out.shape == (batch, seq_len, hidden_dim)
    assert not torch.isnan(out).any()


def test_layerwise_head_budgeter():
    """
    Tests Laguna XS.2 layer-wise head budgeting.
    """
    budgeter = LayerwiseHeadBudgeter(
        num_layers=40,
        kv_heads=8,
        sliding_heads=8,
        global_heads=6,
        sliding_ratio=4
    )

    # Layer 0 (Global full attention)
    assert budgeter.get_query_heads(0) == 6
    # Layer 1, 2, 3 (Sliding window attention)
    assert budgeter.get_query_heads(1) == 8
    assert budgeter.get_query_heads(2) == 8
    assert budgeter.get_query_heads(3) == 8
    # Layer 4 (Global)
    assert budgeter.get_query_heads(4) == 6


def test_enhanced_quadtree_draft_head():
    """
    Tests 1D spatial conv-enhanced Quadtree MRP speculative candidate generation.
    """
    hidden_dim = 64
    vocab_size = 256
    seq_len = 16

    head = EnhancedQuadtreeDraftHead(hidden_dim=hidden_dim, vocab_size=vocab_size, kernel_size=3)

    hidden_states = torch.randn(1, seq_len, hidden_dim)

    nodes, dag_mask, token_ids = head(hidden_states)

    # Verify 21-node DAG tree structure
    assert len(nodes) == 21
    assert dag_mask.shape == (21, 21)
    assert len(token_ids) == 21
