"""
Unit Tests for Compressed Convolutional Attention (CCA) & 1D Conv-Enhanced Speculation (ZAYA1-8B & Poolside Laguna XS.2).
"""

import pytest
import torch

from turing.experimental.speculation import EnhancedQuadtreeDraftHead, TreeNode


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
