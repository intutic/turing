"""
Unit tests for SVD-Compressed Network KV Wire Format (cross-pod transfer).
"""

import pytest
import torch
from turing.serving.kv_transfer import (
    SVDNetworkKVWireCodec,
    serialize_kv_block_svd,
    deserialize_kv_block_svd,
)
from turing.core.radix_svd import SpectralRadixSVDForest


def test_svd_wire_codec_encode_decode_roundtrip():
    seq_len = 64
    num_heads = 8
    head_dim = 128
    rank = 64

    # Create dummy KV tensors and orthonormal projection basis
    torch.manual_seed(42)
    k = torch.randn(seq_len, num_heads, head_dim)
    v = torch.randn(seq_len, num_heads, head_dim)
    u_proj, _ = torch.linalg.qr(torch.randn(head_dim, rank))

    toks = list(range(1000, 1000 + seq_len))

    # Encode to wire format
    wire_bytes = SVDNetworkKVWireCodec.encode(k, v, u_proj, token_ids=toks)
    assert isinstance(wire_bytes, bytes)
    assert len(wire_bytes) > 0

    # Compression ratio check: raw FP16 payload size vs wire payload size
    raw_fp16_bytes = (k.numel() + v.numel()) * 2  # 2 bytes per float16
    wire_size = len(wire_bytes)
    # Wire size should be substantially smaller than raw FP16
    assert wire_size < raw_fp16_bytes

    # Decode from wire format
    k_recon, v_recon, decoded_toks = SVDNetworkKVWireCodec.decode(wire_bytes, u_proj=u_proj)

    assert decoded_toks == toks
    assert k_recon.shape == k.shape
    assert v_recon.shape == v.shape

    # Project original into subspace to verify reconstruction fidelity
    k_sub_proj = torch.matmul(torch.matmul(k, u_proj), u_proj.t())
    v_sub_proj = torch.matmul(torch.matmul(v, u_proj), u_proj.t())

    # INT8 quantization error in subspace should be small
    k_diff = (k_recon - k_sub_proj).abs().mean().item()
    v_diff = (v_recon - v_sub_proj).abs().mean().item()
    assert k_diff < 0.05
    assert v_diff < 0.05


def test_svd_wire_codec_corrupted_magic():
    u_proj = torch.eye(128, 64)
    with pytest.raises(ValueError, match="Invalid SVDK magic"):
        SVDNetworkKVWireCodec.decode(b"CORRUPTED_BYTES_1234567890", u_proj=u_proj)


def test_radix_forest_list_hashes_and_compressed_insert():
    forest = SpectralRadixSVDForest(rank=64)
    u_proj = torch.eye(128, 64)

    k = torch.randn(32, 8, 128)
    v = torch.randn(32, 8, 128)
    toks = list(range(100, 132))

    forest.insert_prefix(toks, k, v, u_proj)

    # Test list_block_hashes
    hashes = forest.list_block_hashes()
    assert len(hashes) > 0
    assert hashes[0][1] == toks

    # Test insert_compressed_node directly
    k_sub_int8 = torch.randint(-127, 127, (16, 8, 64), dtype=torch.int8)
    k_scale = torch.ones(16, 8, 1) * 0.01
    v_sub_int8 = torch.randint(-127, 127, (16, 8, 64), dtype=torch.int8)
    v_scale = torch.ones(16, 8, 1) * 0.01

    forest.insert_compressed_node([201, 202, 203], k_sub_int8[:3], k_scale[:3], v_sub_int8[:3], v_scale[:3])
    matched, mk, mv = forest.match_prefix([201, 202, 203], u_proj)
    assert matched == 3
    assert mk is not None
    assert mv is not None
