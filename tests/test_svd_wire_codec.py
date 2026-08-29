"""
Unit tests for Native C++20 SVD Wire Codec.
Verifies project-and-quantize, dequantize-and-reconstruct, and round-trip fidelity.
"""

import pytest
import torch
import numpy as np

try:
    import turing.turing_csrc as turing_csrc
    HAS_CSRC = True
except ImportError:
    HAS_CSRC = False

from turing.serving.kv_transfer import SVDNetworkKVWireCodec


def test_svd_wire_cpp_quantize_and_reconstruct():
    if not HAS_CSRC:
        pytest.skip("turing_csrc not available")

    torch.manual_seed(42)
    N, Heads, HeadDim, Rank = 16, 4, 128, 64

    # Create orthogonal projection matrix U
    u_raw = torch.randn(HeadDim, Rank, dtype=torch.float32)
    q_u, _ = torch.linalg.qr(u_raw)
    u_proj = q_u[:, :Rank].contiguous()

    # Input tensor
    k_tensor = torch.randn(N, Heads, HeadDim, dtype=torch.float32)

    # PyTorch reference
    k_sub_ref = torch.matmul(k_tensor, u_proj)
    k_scale_ref = torch.amax(torch.abs(k_sub_ref), dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
    k_int8_ref = torch.clamp(torch.round(k_sub_ref / k_scale_ref), -128, 127).to(torch.int8)

    # C++ native
    k_np = k_tensor.numpy()
    u_np = u_proj.numpy()
    k_int8_cpp, k_scale_cpp = turing_csrc.svd_wire_project_quantize_cpu(k_np, u_np)

    np.testing.assert_allclose(k_scale_cpp, k_scale_ref.numpy(), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(k_int8_cpp, k_int8_ref.numpy(), atol=1.0) # Within integer rounding boundary

    # Reconstruction
    k_recon_cpp = turing_csrc.svd_wire_dequantize_reconstruct_cpu(k_int8_cpp, k_scale_cpp, u_np)
    k_recon_ref = torch.matmul(k_int8_ref.float() * k_scale_ref, u_proj.t()).numpy()

    np.testing.assert_allclose(k_recon_cpp, k_recon_ref, rtol=1e-3, atol=1e-3)


def test_svd_network_codec_full_roundtrip():
    torch.manual_seed(42)
    seq_len, num_heads, head_dim, rank = 32, 4, 128, 64

    u_raw = torch.randn(head_dim, rank, dtype=torch.float32)
    q_u, _ = torch.linalg.qr(u_raw)
    u_proj = q_u[:, :rank].contiguous()

    k_tensor = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float32)
    v_tensor = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float32)
    token_ids = list(range(100, 100 + seq_len))

    payload = SVDNetworkKVWireCodec.encode(k_tensor, v_tensor, u_proj, token_ids=token_ids)
    assert isinstance(payload, bytes)
    assert len(payload) > 0

    k_dec, v_dec, dec_tokens = SVDNetworkKVWireCodec.decode(payload, u_proj, device=torch.device("cpu"))

    assert dec_tokens == token_ids
    assert k_dec.shape == k_tensor.shape
    assert v_dec.shape == v_tensor.shape

    # Subspace reconstruction fidelity: reconstructed KV vs exact projected subspace (K @ U @ U^T)
    k_sub_ideal = torch.matmul(torch.matmul(k_tensor, u_proj), u_proj.t())
    v_sub_ideal = torch.matmul(torch.matmul(v_tensor, u_proj), u_proj.t())

    cos_sim_sub_k = torch.cosine_similarity(k_sub_ideal.view(-1, head_dim), k_dec.view(-1, head_dim), dim=-1).mean()
    cos_sim_sub_v = torch.cosine_similarity(v_sub_ideal.view(-1, head_dim), v_dec.view(-1, head_dim), dim=-1).mean()

    assert cos_sim_sub_k.item() > 0.99
    assert cos_sim_sub_v.item() > 0.99

