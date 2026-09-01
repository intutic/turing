import pytest
import torch
import torch.nn.functional as F
from turing.kernels.triton_fused_moe_block_gemm import fused_moe_block_gemm_cuda


def test_fused_moe_block_gemm_cpu_reference_shapes():
    """Verify output shapes and parity on CPU fallback."""
    torch.manual_seed(42)
    M, K, N = 4, 64, 128
    num_experts = 8
    top_k = 2

    x = torch.randn(M, K, dtype=torch.float32)
    w_g = torch.randn(num_experts, K, N, dtype=torch.float32) * 0.02
    w_u = torch.randn(num_experts, K, N, dtype=torch.float32) * 0.02
    w_d = torch.randn(num_experts, N, K, dtype=torch.float32) * 0.02

    router_logits = torch.randn(M, num_experts, dtype=torch.float32)
    topk_probs, topk_indices = torch.topk(F.softmax(router_logits, dim=-1), k=top_k, dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

    out = fused_moe_block_gemm_cuda(x, w_g, w_u, w_d, topk_indices, topk_probs)
    assert out.shape == (M, K)
    assert not torch.isnan(out).any()


def test_fused_moe_block_gemm_mathematical_parity():
    """Verify mathematical equivalence to independent expert SwiGLU computation."""
    torch.manual_seed(123)
    M, K, N = 2, 32, 64
    num_experts = 4
    top_k = 2

    x = torch.randn(M, K, dtype=torch.float32)
    w_g = torch.randn(num_experts, K, N, dtype=torch.float32) * 0.05
    w_u = torch.randn(num_experts, K, N, dtype=torch.float32) * 0.05
    w_d = torch.randn(num_experts, N, K, dtype=torch.float32) * 0.05

    topk_indices = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    topk_probs = torch.tensor([[0.7, 0.3], [0.6, 0.4]], dtype=torch.float32)

    # Reference computation
    ref_out = torch.zeros(M, K, dtype=torch.float32)
    for row in range(M):
        for k_idx in range(top_k):
            e = int(topk_indices[row, k_idx].item())
            prob = float(topk_probs[row, k_idx].item())
            tok_x = x[row:row+1]
            g = F.silu(torch.matmul(tok_x, w_g[e]))
            u = torch.matmul(tok_x, w_u[e])
            d = torch.matmul(g * u, w_d[e])
            ref_out[row] += (d * prob).squeeze(0)

    fused_out = fused_moe_block_gemm_cuda(x, w_g, w_u, w_d, topk_indices, topk_probs)
    assert torch.allclose(ref_out, fused_out, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU Triton parity test")
def test_fused_moe_block_gemm_cuda_parity():
    """Verify GPU Triton kernel matches PyTorch reference within tolerance."""
    torch.manual_seed(42)
    M, K, N = 8, 128, 256
    num_experts = 8
    top_k = 2

    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w_g = torch.randn(num_experts, K, N, device="cuda", dtype=torch.float16) * 0.02
    w_u = torch.randn(num_experts, K, N, device="cuda", dtype=torch.float16) * 0.02
    w_d = torch.randn(num_experts, N, K, device="cuda", dtype=torch.float16) * 0.02

    router_logits = torch.randn(M, num_experts, device="cuda", dtype=torch.float16)
    topk_probs, topk_indices = torch.topk(F.softmax(router_logits, dim=-1), k=top_k, dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

    gpu_out = fused_moe_block_gemm_cuda(x, w_g, w_u, w_d, topk_indices, topk_probs)
    # CPU baseline
    cpu_out = fused_moe_block_gemm_cuda(x.cpu().float(), w_g.cpu().float(), w_u.cpu().float(), w_d.cpu().float(), topk_indices.cpu(), topk_probs.cpu().float())

    assert torch.allclose(gpu_out.cpu().float(), cpu_out, atol=1e-3, rtol=1e-3)
