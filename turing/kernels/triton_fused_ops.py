"""
Fused small operations of a decoder step, one launch each where PyTorch takes several (speed plan P2.1):

- rms_norm: RMSNorm in fp32 (Llama's offset 0, Gemma's offset 1), optionally after a residual add whose sum it also
  returns (x + r, then the norm of that sum: the residual stream and the next sublayer's input in one pass).
- silu_mul: SiLU(gate) * up (the gated MLP), reading gate and up through strides (views of one fused projection).
- rope: rotary embedding of q and k together, full or partial (the leading dims rotate, the rest pass through).

Each rounds where the PyTorch code it replaces rounds (an fp16/bf16 op computes in fp32 and rounds its result), so the
outputs match the eager path bit for bit or within one rounding; tests/test_fused_ops.py checks them against it.
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


if HAS_TRITON:
    @triton.jit
    def _rms_norm_kernel(X, R, W, Y, S, D, s_x, s_r, s_y, s_s, eps, offset,
                         HAS_RES: tl.constexpr, OFFSET: tl.constexpr, BLOCK: tl.constexpr):
        """One row: s = x (+ r, rounded to the dtype), y = rms_norm(s) * weight."""
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < D
        x = tl.load(X + row * s_x + cols, mask=mask, other=0.0)
        if HAS_RES:
            x = (x.to(tl.float32) + tl.load(R + row * s_r + cols, mask=mask, other=0.0).to(tl.float32)).to(x.dtype)
            tl.store(S + row * s_s + cols, x, mask=mask)
        xf = x.to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(xf * xf, axis=0) / D + eps)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        if OFFSET:  # Gemma: (normed * (offset + w)) in fp32, then cast
            y = xf * rstd * (offset + w)
        else:  # Llama: cast the normed value, then multiply by the weight in the dtype
            y = (xf * rstd).to(x.dtype).to(tl.float32) * w
        tl.store(Y + row * s_y + cols, y.to(Y.dtype.element_ty), mask=mask)

    @triton.jit
    def _silu_mul_kernel(G, U, Y, F, s_g, s_u, s_y, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < F
        g = tl.load(G + row * s_g + cols, mask=mask, other=0.0)
        u = tl.load(U + row * s_u + cols, mask=mask, other=0.0)
        gf = g.to(tl.float32)
        a = (gf / (1.0 + tl.exp(-gf))).to(g.dtype)  # F.silu rounds to the dtype, then the product rounds again
        tl.store(Y + row * s_y + cols, (a.to(tl.float32) * u.to(tl.float32)).to(Y.dtype.element_ty), mask=mask)

    @triton.jit(do_not_specialize=["L", "n_q"])  # no build per value of these (1, a multiple of 16)
    def _rope_kernel(X, XK, C, Sn, Y, YK, H, HK, L, n_q, s_xb, s_xh, s_xl, s_kb, s_kh, s_kl, s_cb, s_cl,
                     s_yb, s_yh, s_yl, s_zb, s_zh, s_zl, D: tl.constexpr, R: tl.constexpr, BLOCK_D: tl.constexpr,
                     BLOCK_L: tl.constexpr):
        """BLOCK_L consecutive positions of one (b, h) of q (the first n_q programs) or of k (the rest):
        y[:R] = x*cos + rotate_half(x)*sin (each product rounded, then the sum), y[R:] = x. One vector a program
        (BLOCK_L 1, a decode step's rows) is 131,072 programs for a 2,048-token step of SmolLM2-1.7B (64 q and k
        heads): 0.60 ms a layer on an L4, against 0.12 in tiles of 2,048 elements."""
        pid = tl.program_id(0).to(tl.int64)
        tiles = (L + BLOCK_L - 1) // BLOCK_L
        if pid < n_q:
            t = pid % tiles
            h = (pid // tiles) % H
            b = pid // (tiles * H)
            x_row = X + b * s_xb + h * s_xh
            y_row = Y + b * s_yb + h * s_yh
            s_x = pid * 0 + s_xl  # int64 in both branches
            s_y = pid * 0 + s_yl
        else:
            p = pid - n_q
            t = p % tiles
            h = (p // tiles) % HK
            b = p // (tiles * HK)
            x_row = XK + b * s_kb + h * s_kh
            y_row = YK + b * s_zb + h * s_zh
            s_x = pid * 0 + s_kl
            s_y = pid * 0 + s_zl
        l = (t * BLOCK_L + tl.arange(0, BLOCK_L))[:, None]
        d = tl.arange(0, BLOCK_D)[None, :]
        live = l < L
        x = tl.load(x_row + l * s_x + d, mask=live & (d < D), other=0.0)
        half = R // 2
        rot_mask = d < R
        partner = tl.where(d < half, d + half, d - half)
        xp = tl.load(x_row + l * s_x + partner, mask=live & rot_mask, other=0.0).to(tl.float32)
        xp = tl.where(d < half, -xp, xp).to(x.dtype)  # rotate_half: (-x2, x1)
        c = tl.load(C + b * s_cb + l * s_cl + d, mask=live & rot_mask, other=0.0).to(x.dtype)
        s = tl.load(Sn + b * s_cb + l * s_cl + d, mask=live & rot_mask, other=0.0).to(x.dtype)
        t1 = (x.to(tl.float32) * c.to(tl.float32)).to(x.dtype)
        t2 = (xp.to(tl.float32) * s.to(tl.float32)).to(x.dtype)
        y = tl.where(rot_mask, (t1.to(tl.float32) + t2.to(tl.float32)).to(x.dtype), x)
        tl.store(y_row + l * s_y + d, y, mask=live & (d < D))


def fused_ops_supported(x: torch.Tensor) -> bool:
    return HAS_TRITON and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, offset: float = 0.0, residual: torch.Tensor = None):
    """rms_norm(x) * weight (offset 0) or * (offset + weight) (offset != 0) over the last dim; with residual, the norm
    of s = x + residual, returning (norm, s)."""
    D = x.shape[-1]
    x2 = x.reshape(-1, D)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    y = torch.empty(x2.shape, device=x.device, dtype=x.dtype)
    r2 = s = None
    if residual is not None:
        r2 = residual.reshape(-1, D)
        if r2.stride(-1) != 1:
            r2 = r2.contiguous()
        s = torch.empty_like(y)
    BLOCK = triton.next_power_of_2(D)
    _rms_norm_kernel[(x2.shape[0],)](x2, r2 if r2 is not None else x2, weight, y, s if s is not None else y, D,
                                     x2.stride(0), r2.stride(0) if r2 is not None else 0, y.stride(0),
                                     s.stride(0) if s is not None else 0, eps, float(offset),
                                     HAS_RES=residual is not None, OFFSET=bool(offset), BLOCK=BLOCK,
                                     num_warps=4 if BLOCK <= 4096 else 8)
    y = y.view(x.shape)
    return y if residual is None else (y, s.view(x.shape))


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """F.silu(gate) * up for [..., F] tensors whose last dim is contiguous (they may be views of one wider tensor)."""
    F_ = gate.shape[-1]
    g2, u2 = gate.reshape(-1, F_), up.reshape(-1, F_)
    if g2.stride(-1) != 1 or u2.stride(-1) != 1:
        g2, u2 = g2.contiguous(), u2.contiguous()
    y = torch.empty(g2.shape, device=gate.device, dtype=gate.dtype)
    BLOCK = 1024
    _silu_mul_kernel[(g2.shape[0], triton.cdiv(F_, BLOCK))](g2, u2, y, F_, g2.stride(0), u2.stride(0), y.stride(0),
                                                            BLOCK=BLOCK)
    return y.view(gate.shape)


def _rope_both(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, tokens_first: bool = False):
    """q and k in one launch (k's programs follow q's). tokens_first: the rotated q's memory is [B, L, H, D] (the
    shape returned is [B, H, L, D] all the same), as a kernel that takes queries token by token reads them."""
    q = q if q.stride(-1) == 1 else q.contiguous()
    k = k if k.stride(-1) == 1 else k.contiguous()
    B, H, L, D = q.shape
    HK = k.shape[1]
    yq = torch.empty((B, L, H, D) if tokens_first else q.shape, device=q.device, dtype=q.dtype)
    yq = yq.transpose(1, 2) if tokens_first else yq
    yk = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    s_cb = cos.stride(0) if cos.dim() == 4 else 0
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_L = 1 if L == 1 else max(1, 2048 // BLOCK_D)  # two builds: a decode step's rows, and tiles of positions
    tiles = triton.cdiv(L, BLOCK_L)
    _rope_kernel[(B * (H + HK) * tiles,)](q, k, cos, sin, yq, yk, H, HK, L, B * H * tiles,
                                          q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                                          s_cb, cos.stride(-2), yq.stride(0), yq.stride(1), yq.stride(2),
                                          yk.stride(0), yk.stride(1), yk.stride(2), D=D, R=cos.shape[-1],
                                          BLOCK_D=BLOCK_D, BLOCK_L=BLOCK_L)
    return yq, yk


def rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, tokens_first: bool = False):
    """apply_rotary_pos_emb(q, k, cos.to(q.dtype), sin.to(q.dtype)) for q [B, H, L, D], k [B, Hkv, L, D] and cos, sin
    [L, R] or [B, 1, L, R] (R <= D, even; cast to q's dtype in the kernel)."""
    if cos.stride(-1) != 1:
        cos, sin = cos.contiguous(), sin.contiguous()
    if sin.stride() != cos.stride():
        cos, sin = cos.contiguous(), sin.contiguous()
    return _rope_both(q, k, cos, sin, tokens_first)
