"""
Unit Tests for Hierarchical Sequence-Chunk Compression & Cross-Layer KV Sharing (DeepSeek V4 & Gemma 4).
"""

import pytest
import torch

from turing.core.hierarchical_compression import (
    HCAChunkCompressor,
    CSAChunkCompressor,
    CrossLayerKVSharingManager
)

def test_hca_chunk_compressor_128x_compression():
    """
    Tests HCA chunk pooling on 512-token Huge Page sequences.
    """
    batch = 2
    seq_len = 512
    num_heads = 4
    head_dim = 64
    hidden_dim = num_heads * head_dim

    hca = HCAChunkCompressor(hidden_dim=head_dim, chunk_size=128)

    k = torch.randn(batch, seq_len, num_heads, head_dim)
    v = torch.randn(batch, seq_len, num_heads, head_dim)

    k_summary, v_summary = hca.compress_chunk(k, v)

    # 512 / 128 = 4 summary tokens
    assert k_summary.shape == (batch, 4, num_heads, head_dim)
    assert v_summary.shape == (batch, 4, num_heads, head_dim)


def test_csa_chunk_compressor_and_topk_selection():
    """
    Tests CSA block-level compression and query selection.
    """
    batch = 2
    seq_len = 64
    num_heads = 4
    head_dim = 64
    hidden_dim = num_heads * head_dim

    csa = CSAChunkCompressor(hidden_dim=head_dim, chunk_size=4)

    k = torch.randn(batch, seq_len, num_heads, head_dim)
    v = torch.randn(batch, seq_len, num_heads, head_dim)

    k_blocks, v_blocks = csa.compress_blocks(k, v)

    # 64 / 4 = 16 blocks
    assert k_blocks.shape == (batch, 16, num_heads, head_dim)

    # Test top-k query selection
    q = torch.randn(batch, num_heads, head_dim)
    mask = csa.select_topk_blocks(q, k_blocks, top_k=4)

    assert mask.shape == (batch, 16)
    assert mask.sum(dim=-1).tolist() == [4, 4]


def test_cross_layer_kv_sharing_manager():
    """
    Tests Gemma 4 style cross-layer KV tensor sharing mapping.
    """
    num_layers = 32
    num_shared = 16
    sharing_mgr = CrossLayerKVSharingManager(num_layers=num_layers, num_shared_layers=num_shared, sliding_window_ratio=4)

    # First 16 layers compute their own KV
    for l in range(16):
        assert not sharing_mgr.is_shared(l)
        assert sharing_mgr.get_source_layer(l) == l

    # Layers 16..31 share KV with earlier layers
    for l in range(16, 32):
        assert sharing_mgr.is_shared(l)
        src = sharing_mgr.get_source_layer(l)
        assert src < 16
