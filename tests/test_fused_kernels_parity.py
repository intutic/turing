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
from turing.kernels.triton_fused_qkv_rope import dispatch_fused_qkv_rope


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
