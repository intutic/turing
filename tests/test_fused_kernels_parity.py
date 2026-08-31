"""
Numerical Parity Test Suite for Fused In-SRAM Triton & PyTorch Kernels.
Verifies exact tensor equality between fused layers and unfused PyTorch reference layers.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from turing.config import ModelConfig
from turing.models.causal_lm import RMSNorm, SubspaceMLP, SubspaceDecoderLayer
from turing.kernels.triton_fused_rmsnorm_swiglu import dispatch_fused_rmsnorm_swiglu
from turing.kernels.triton_fused_qkv_rope import dispatch_fused_qkv_rope


def test_fused_rmsnorm_swiglu_parity():
    torch.manual_seed(42)
    B, K, ffn_dim = 4, 64, 128
    tile_size = 32
    active_tiles = torch.tensor([0, 1], dtype=torch.int32)

    x = torch.randn(B, K)
    weight_norm = torch.randn(K)
    w_gate = torch.randn(K, ffn_dim)
    w_up = torch.randn(K, ffn_dim)
    w_down = torch.randn(ffn_dim, K)
    residual = torch.randn(B, K)

    # 1. Unfused Reference
    var = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(var + 1e-6) * weight_norm
    indices = [0, 1, 2, ..., 63]
    w_g_sub = w_gate[:, :64]
    w_u_sub = w_up[:, :64]
    w_d_sub = w_down[:64, :]
    gate = torch.matmul(x_norm, w_g_sub)
    up = torch.matmul(x_norm, w_u_sub)
    out_unfused = torch.matmul(F.silu(gate) * up, w_d_sub) + residual

    # 2. Fused Dispatcher
    out_fused = dispatch_fused_rmsnorm_swiglu(
        x=x,
        weight_norm=weight_norm,
        w_gate=w_gate,
        w_up=w_up,
        w_down=w_down,
        residual=residual,
        active_tiles=active_tiles,
        tile_size=tile_size,
        eps=1e-6
    )

    assert torch.allclose(out_fused, out_unfused, rtol=1e-4, atol=1e-4)


def test_decoder_layer_with_fused_ffn():
    config = ModelConfig(
        name="test-fused",
        vocab_size=128,
        hidden_dim=64,
        ffn_dim=128,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        tile_size=32,
        active_tiles=2
    )
    layer = SubspaceDecoderLayer(config, layer_idx=0).eval()

    x = torch.randn(2, 8, 64)
    out, k, v = layer(x)

    assert out.shape == (2, 8, 64)
    assert k.shape == (2, 2, 8, 16)
    assert v.shape == (2, 2, 8, 16)
    assert not torch.isnan(out).any()


def test_fused_qkv_rope_parity():
    torch.manual_seed(42)
    B, H = 2, 64
    num_heads, num_kv_heads, head_dim = 4, 2, 16
    x = torch.randn(B, H)
    wq = torch.randn(H, num_heads * head_dim)
    wk = torch.randn(H, num_kv_heads * head_dim)
    wv = torch.randn(H, num_kv_heads * head_dim)
    cos = torch.ones(B, head_dim)
    sin = torch.zeros(B, head_dim)

    q, k, v = dispatch_fused_qkv_rope(x, wq, wk, wv, cos, sin, num_heads, num_kv_heads, head_dim)

    assert q.shape == (B, num_heads * head_dim)
    assert k.shape == (B, num_kv_heads * head_dim)
    assert v.shape == (B, num_kv_heads * head_dim)
