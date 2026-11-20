"""
Triton GPU Kernel for Fused In-SRAM Adam Optimizer Step.
Directly updates parameters, first moments, and second moments in GPU SRAM.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _fused_adam_kernel(
    param_ptr,
    grad_ptr,
    m_ptr,
    v_ptr,
    lr,
    beta1,
    beta2,
    eps,
    bias_correction1,
    bias_correction2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    param = tl.load(param_ptr + offs, mask=mask)
    grad = tl.load(grad_ptr + offs, mask=mask)
    m = tl.load(m_ptr + offs, mask=mask)
    v = tl.load(v_ptr + offs, mask=mask)

    # 1. Update moments
    m_new = beta1 * m + (1.0 - beta1) * grad
    v_new = beta2 * v + (1.0 - beta2) * (grad * grad)

    # 2. Bias correction
    m_hat = m_new / bias_correction1
    v_hat = v_new / bias_correction2

    # 3. Update parameter
    param_new = param - lr * m_hat / (tl.sqrt(v_hat) + eps)

    # 4. Write back
    tl.store(m_ptr + offs, m_new, mask=mask)
    tl.store(v_ptr + offs, v_new, mask=mask)
    tl.store(param_ptr + offs, param_new, mask=mask)

def fused_adam_step_cuda(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg_m: torch.Tensor,
    exp_avg_v: torch.Tensor,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    timestep: int = 1
):
    """
    Executes fused Adam step in a single GPU kernel launch.
    """
    n_elements = param.numel()
    bias_correction1 = 1.0 - (beta1 ** timestep)
    bias_correction2 = 1.0 - (beta2 ** timestep)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _fused_adam_kernel[grid](
        param, grad, exp_avg_m, exp_avg_v,
        lr, beta1, beta2, eps,
        bias_correction1, bias_correction2,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

