import pytest
import torch
import math
from turing.kernels.triton_paged_decode_gemv import batched_paged_decode_gemv_cuda


def test_paged_decode_gemv_cpu_shapes():
    """Verify batched paged decode GEMV output shapes and non-nan outputs."""
    torch.manual_seed(42)
    B, H, D = 4, 8, 64
    num_blocks = 16
    block_size = 16
    max_blocks = 4

    q = torch.randn(B, H, 1, D, dtype=torch.float32)
    k_pool = torch.randn(num_blocks, block_size, H, D, dtype=torch.float32)
    v_pool = torch.randn(num_blocks, block_size, H, D, dtype=torch.float32)

    block_tables = torch.tensor([
        [0, 1, -1, -1],
        [2, 3, 4, -1],
        [5, -1, -1, -1],
        [6, 7, 8, 9]
    ], dtype=torch.int32)
    seq_lens = torch.tensor([24, 40, 12, 60], dtype=torch.int32)

    out = batched_paged_decode_gemv_cuda(q, k_pool, v_pool, block_tables, seq_lens)
    assert out.shape == (B, H, 1, D)
    assert not torch.isnan(out).any()


def test_paged_decode_gemv_single_vs_multi_batch():
    """Verify single batch outputs match multi-batch outputs identically."""
    torch.manual_seed(123)
    H, D = 4, 32
    num_blocks = 8
    block_size = 8

    q1 = torch.randn(1, H, D, dtype=torch.float32)
    q2 = torch.randn(1, H, D, dtype=torch.float32)
    q_batched = torch.cat([q1, q2], dim=0)

    k_pool = torch.randn(num_blocks, block_size, H, D, dtype=torch.float32)
    v_pool = torch.randn(num_blocks, block_size, H, D, dtype=torch.float32)

    bt1 = torch.tensor([[0, 1]], dtype=torch.int32)
    bt2 = torch.tensor([[2, 3]], dtype=torch.int32)
    bt_batched = torch.cat([bt1, bt2], dim=0)

    sl1 = torch.tensor([12], dtype=torch.int32)
    sl2 = torch.tensor([15], dtype=torch.int32)
    sl_batched = torch.cat([sl1, sl2], dim=0)

    out1 = batched_paged_decode_gemv_cuda(q1, k_pool, v_pool, bt1, sl1)
    out2 = batched_paged_decode_gemv_cuda(q2, k_pool, v_pool, bt2, sl2)
    out_batched = batched_paged_decode_gemv_cuda(q_batched, k_pool, v_pool, bt_batched, sl_batched)

    assert torch.allclose(out_batched[0], out1[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_batched[1], out2[0], atol=1e-5, rtol=1e-5)
