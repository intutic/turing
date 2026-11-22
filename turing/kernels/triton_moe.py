"""
Grouped GEMM for top-k routed MoE layers on CUDA, with no host synchronisation, so a decode step that contains MoE
layers can be captured in a CUDA graph.

The token-expert pairs (token t's j-th choice is pair t * top_k + j) are sorted by expert on the device and each
expert's pairs padded to a multiple of BLOCK_M rows (`moe_align`). Two Triton kernels then run one program per
(block of rows, tile of columns), each block belonging to one expert, so every expert's weights are read once per
block of its tokens:

  gate_up: h[row] = act(x[token] @ gate[e].T) * (x[token] @ up[e].T)   (SiLU, as in SparseMoE)
  down:    y[pair] = routing_weight[pair] * (h[row] @ down[e].T)

and the output is the sum of each token's top_k rows of y. Shapes depend only on the number of pairs and experts.
"""

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # CPU and MPS builds: SparseMoE keeps its per-expert loop
    triton = tl = None
    HAS_TRITON = False


def moe_align(experts: torch.Tensor, num_experts: int, block_m: int
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pairs grouped by expert, on the device of `experts` ([T, top_k] expert ids).

    Returns sorted_pairs [P] (pair ids, each expert's group padded to a multiple of block_m with the sentinel
    T * top_k), block_expert [P // block_m] (the expert of each block; num_experts past the last group) and used
    [1] (int32: the rows of sorted_pairs in use, the layout of vLLM's moe_align_block_size). Rows of sorted_pairs
    past `used` are not written on CUDA. P = n + min(num_experts, n) * (block_m - 1) for n pairs, an upper bound that
    depends only on the shapes (n * block_m for a single token, which needs no sort)."""
    flat = experts.reshape(-1)
    n = flat.numel()
    if experts.shape[0] == 1:  # one token (batch-1 decode): its experts are distinct, so each pair is its own block
        pairs = _single_token_pairs(n, block_m, flat.device)
        return pairs[0], flat, pairs[1]
    total = n + min(num_experts, n) * (block_m - 1)
    total = -(-total // block_m) * block_m
    if HAS_TRITON and flat.is_cuda and block_m <= 64:
        return _moe_align_cuda(flat, num_experts, block_m, total)
    counts = torch.zeros(num_experts, dtype=torch.int64, device=flat.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    padded = (counts + block_m - 1) // block_m * block_m
    pad_end = torch.cumsum(padded, 0)
    pad_start = pad_end - padded
    order = torch.argsort(flat, stable=True)  # pair ids, grouped by expert
    sorted_e = flat[order]
    rank = torch.arange(n, device=flat.device) - (torch.cumsum(counts, 0) - counts)[sorted_e]
    sorted_pairs = torch.full((total,), n, dtype=torch.int64, device=flat.device)
    sorted_pairs.scatter_(0, pad_start[sorted_e] + rank, order)
    block_start = torch.arange(0, total, block_m, device=flat.device)
    return sorted_pairs, torch.searchsorted(pad_end, block_start, right=True), pad_end[-1:].to(torch.int32)


if HAS_TRITON:
    # moe_align on CUDA in three launches (PyTorch needs about fourteen: a scatter, a sort, cumulative sums, a
    # search; at 24 MoE layers they add up to 1-2 ms of a decode step).

    @triton.jit
    def _align_count_kernel(IDS, COUNTS, n, BLOCK: tl.constexpr):
        """Program e: how many pairs chose expert e."""
        e = tl.program_id(0)
        count = e * 0  # an int32 scalar carried through the loop
        for i0 in range(0, n, BLOCK):
            i = i0 + tl.arange(0, BLOCK)
            ids = tl.load(IDS + i, mask=i < n, other=-1)
            count += tl.sum((ids == e).to(tl.int32), 0)
        tl.store(COUNTS + e, count)

    @triton.jit
    def _align_blocks_kernel(COUNTS, STARTS, BLOCK_E, USED, num_experts, block_m, n_blocks,
                             BLOCK_B: tl.constexpr, BLOCK_X: tl.constexpr):
        """Each expert's first row (its pairs padded to whole blocks, experts in order), and for a tile of blocks the
        expert each belongs to (num_experts past the last)."""
        offs_e = tl.arange(0, BLOCK_X)
        counts = tl.load(COUNTS + offs_e, mask=offs_e < num_experts, other=0)
        padded = (counts + block_m - 1) // block_m * block_m
        end = tl.cumsum(padded, 0)
        if tl.program_id(0) == 0:
            tl.store(STARTS + offs_e, end - padded, mask=offs_e < num_experts)
            tl.store(USED, tl.sum(padded, 0))
        b = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
        done = (end[None, :] <= (b * block_m)[:, None]) & (offs_e < num_experts)[None, :]
        tl.store(BLOCK_E + b, tl.sum(done.to(tl.int32), 1), mask=b < n_blocks)

    @triton.jit
    def _align_fill_kernel(IDS, COUNTS, STARTS, PAIRS, n, block_m, BLOCK: tl.constexpr, PAD: tl.constexpr):
        """Program e: the ids of expert e's pairs, in order, from its first row on; then the sentinel n up to a whole
        block."""
        e = tl.program_id(0)
        base = tl.load(STARTS + e)
        seen = e * 0
        for i0 in range(0, n, BLOCK):
            i = i0 + tl.arange(0, BLOCK)
            mine = tl.load(IDS + i, mask=i < n, other=-1) == e
            rank = seen + tl.cumsum(mine.to(tl.int32), 0) - 1
            tl.store(PAIRS + base + rank, i, mask=mine)
            seen += tl.sum(mine.to(tl.int32), 0)
        count = tl.load(COUNTS + e)
        r = count + tl.arange(0, PAD)
        tl.store(PAIRS + base + r, tl.full((PAD,), n, dtype=tl.int32), mask=r < (count + block_m - 1) // block_m * block_m)


def _moe_align_cuda(flat: torch.Tensor, num_experts: int, block_m: int, total: int):
    n, dev = flat.numel(), flat.device
    counts = torch.empty(num_experts, dtype=torch.int32, device=dev)
    starts = torch.empty(num_experts, dtype=torch.int32, device=dev)
    pairs = torch.empty(total, dtype=torch.int32, device=dev)
    block_expert = torch.empty(total // block_m, dtype=torch.int32, device=dev)
    used = torch.empty(1, dtype=torch.int32, device=dev)
    _align_count_kernel[(num_experts,)](flat, counts, n, BLOCK=1024, num_warps=4)
    _align_blocks_kernel[(triton.cdiv(total // block_m, 64),)](
        counts, starts, block_expert, used, num_experts, block_m, total // block_m,
        BLOCK_B=64, BLOCK_X=triton.next_power_of_2(num_experts), num_warps=4)
    _align_fill_kernel[(num_experts,)](flat, counts, starts, pairs, n, block_m, BLOCK=1024, PAD=64, num_warps=4)
    return pairs, block_expert, used


_SINGLE = {}  # (pairs, block_m, device) -> the pair table of one token: the same every step


def _single_token_pairs(n: int, block_m: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """(pair i alone in block i, the rest of the block the sentinel n; the rows in use, as moe_align's `used`). Kept
    between calls: built on the spot it is three kernel launches per MoE layer and decode step, about as long as they
    take to run. Not kept when the first call is inside a CUDA graph capture (the tensors would live in that graph's
    memory)."""
    key = (n, block_m, device)
    pairs = _SINGLE.get(key)
    if pairs is None:
        table = torch.full((n, block_m), n, dtype=torch.int32, device=device)
        table[:, 0] = torch.arange(n, dtype=torch.int32, device=device)
        pairs = (table.reshape(-1), torch.full((1,), n * block_m, dtype=torch.int32, device=device))
        if not (device.type == "cuda" and torch.cuda.is_current_stream_capturing()):
            _SINGLE[key] = pairs
    return pairs


if HAS_TRITON:
    @triton.jit
    def _moe_route_kernel(L, W, E, n_experts, s_l,
                          TOP_K: tl.constexpr, NORM: tl.constexpr, BLOCK_E: tl.constexpr):
        """Program t: the top_k experts of row t of the router logits, best first, with their softmax probabilities
        (divided by their sum when NORM)."""
        t = tl.program_id(0)
        offs = tl.arange(0, BLOCK_E)
        ok = offs < n_experts
        x = tl.load(L + t * s_l + offs, mask=ok, other=float("-inf")).to(tl.float32)
        p = tl.where(ok, tl.exp(x - tl.max(x, 0)), 0.0)
        p = p / tl.sum(p, 0)
        picked = 0.0
        for j in tl.static_range(TOP_K):
            w = tl.max(p, 0)
            i = tl.argmax(p, 0)
            tl.store(E + t * TOP_K + j, i.to(tl.int64))
            tl.store(W + t * TOP_K + j, w)
            picked += w
            p = tl.where(offs == i, -1.0, p)
        if NORM:
            for j in tl.static_range(TOP_K):
                tl.store(W + t * TOP_K + j, tl.load(W + t * TOP_K + j) / picked)


def moe_route(logits: torch.Tensor, top_k: int, norm: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """torch.topk(softmax(logits), top_k) for router logits [T, experts] on CUDA, in one kernel: (fp32 probabilities,
    divided by their sum when norm; expert ids), each [T, top_k]. In PyTorch it is a cast, a softmax, a top-k (two
    kernels), a sum and a division per MoE layer, which at batch 1 take as long as a tenth of the layer's experts."""
    T, E = logits.shape
    logits = logits.contiguous()
    weights = torch.empty(T, top_k, device=logits.device, dtype=torch.float32)
    experts = torch.empty(T, top_k, device=logits.device, dtype=torch.int64)
    _moe_route_kernel[(T,)](logits, weights, experts, E, logits.stride(0), TOP_K=top_k, NORM=norm,
                            BLOCK_E=triton.next_power_of_2(E), num_warps=1)
    return weights, experts


# From this many rows per expert on average (a large prefill chunk, not a decode step), one cuBLAS GEMM pair per expert
# (gemm_moe) beats the grouped kernels, whose 32-row blocks each read their expert again. Below it the loop's fixed
# cost per expert dominates: on an L4 with OLMoE's 64 experts, 11.0 ms against 8.6 ms at 256 rows per expert and
# 30.3 ms against 39.5 ms at 1,024 (results_moe_kernel_prefill_cuda.json).
GEMM_MIN_ROWS = 1024


def gemm_moe(x: torch.Tensor, weights: torch.Tensor, experts: torch.Tensor, num_experts: int, expert_weights,
             act, only=None) -> torch.Tensor:
    """The same sum as fused_moe, as two GEMMs per expert on the rows routed to it: expert_weights(e) returns
    (gate_up [2F, H], down [H, F]) in x's dtype. cuBLAS reads an expert once however many rows it has, which is what
    prefill-sized inputs need. Runs on any device; one host sync (the experts' row counts). The top_k terms are
    summed in fp32. With `only` (expert ids), the terms of those experts alone."""
    top_k = experts.shape[1]
    flat = experts.reshape(-1)
    order = torch.argsort(flat, stable=True)  # pair ids, grouped by expert
    counts = torch.bincount(flat, minlength=num_experts).tolist()
    tok, w = order // top_k, weights.reshape(-1)[order].float()
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    start = 0
    for e, n in enumerate(counts):
        if n and (only is None or e in only):
            rows = tok[start:start + n]  # distinct: a token is routed to an expert once
            gate_up, down = expert_weights(e)
            gate, up = torch.nn.functional.linear(x[rows], gate_up).chunk(2, dim=-1)
            y = torch.nn.functional.linear(act(gate) * up, down)
            out.index_add_(0, rows, y.float() * w[start:start + n, None])
        start += n
    return out.to(x.dtype)


def fused_moe_supported(x: torch.Tensor, gate_up: torch.Tensor, act) -> bool:
    return (HAS_TRITON and x.is_cuda and gate_up.is_cuda and act is torch.nn.functional.silu
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32) and gate_up.dtype == x.dtype)


if HAS_TRITON:
    @triton.jit
    def _moe_gate_up_kernel(X, W, PAIRS, BLOCK_E, H, n_pairs, num_experts, K, F,
                            s_xt, s_we, s_wn, s_hr,
                            TOP_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                            BLOCK_K: tl.constexpr, IEEE: tl.constexpr):
        pid_m = tl.program_id(0)
        e = tl.load(BLOCK_E + pid_m)
        if e >= num_experts:
            return
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        pair = tl.load(PAIRS + rows)
        valid = pair < n_pairs
        tok = pair // TOP_K
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = offs_n < F
        offs_k = tl.arange(0, BLOCK_K)
        w = W + e.to(tl.int64) * s_we
        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            k_ok = k < K
            x = tl.load(X + tok[:, None].to(tl.int64) * s_xt + k[None, :], mask=valid[:, None] & k_ok[None, :],
                        other=0.0)
            wg = tl.load(w + offs_n[None, :] * s_wn + k[:, None], mask=k_ok[:, None] & n_ok[None, :], other=0.0)
            wu = tl.load(w + (offs_n[None, :] + F) * s_wn + k[:, None], mask=k_ok[:, None] & n_ok[None, :],
                         other=0.0)
            if IEEE:
                acc_g += tl.dot(x, wg, input_precision="ieee")
                acc_u += tl.dot(x, wu, input_precision="ieee")
            else:
                acc_g += tl.dot(x, wg)
                acc_u += tl.dot(x, wu)
        # SparseMoE rounds both projections to the model dtype before the activation; so does this
        g = acc_g.to(H.dtype.element_ty).to(tl.float32)
        u = acc_u.to(H.dtype.element_ty).to(tl.float32)
        h = (g * tl.sigmoid(g)).to(H.dtype.element_ty).to(tl.float32) * u
        tl.store(H + rows[:, None].to(tl.int64) * s_hr + offs_n[None, :], h.to(H.dtype.element_ty),
                 mask=valid[:, None] & n_ok[None, :])

    @triton.jit
    def _moe_down_kernel(Hs, W, PAIRS, BLOCK_E, RW, Y, n_pairs, num_experts, F, N,
                         s_hr, s_we, s_wn, s_yp,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, IEEE: tl.constexpr):
        pid_m = tl.program_id(0)
        e = tl.load(BLOCK_E + pid_m)
        if e >= num_experts:
            return
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        pair = tl.load(PAIRS + rows)
        valid = pair < n_pairs
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = offs_n < N
        offs_k = tl.arange(0, BLOCK_K)
        w = W + e.to(tl.int64) * s_we
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, F, BLOCK_K):
            k = k0 + offs_k
            k_ok = k < F
            h = tl.load(Hs + rows[:, None].to(tl.int64) * s_hr + k[None, :], mask=valid[:, None] & k_ok[None, :],
                        other=0.0)
            wd = tl.load(w + offs_n[None, :] * s_wn + k[:, None], mask=k_ok[:, None] & n_ok[None, :], other=0.0)
            if IEEE:
                acc += tl.dot(h, wd, input_precision="ieee")
            else:
                acc += tl.dot(h, wd)
        # as SparseMoE: the expert's output in the model dtype, then scaled by its fp32 routing weight
        rw = tl.load(RW + pair, mask=valid, other=0.0)
        y = acc.to(Hs.dtype.element_ty).to(tl.float32) * rw[:, None]
        tl.store(Y + pair[:, None].to(tl.int64) * s_yp + offs_n[None, :], y, mask=valid[:, None] & n_ok[None, :])


# Rows per block and tile sizes. On an L4 with OLMoE's shapes, 32 x 64 x 128 is within 1% of the fastest setting at
# 1 to 512 tokens and 7% faster than 16 x 64 x 64 (results_moe_kernel_sweep_cuda.json). A GPU whose shared memory
# cannot hold a 128-wide K tile gets 64 (set on the first OutOfResources).
_BLOCK_K = [128]


def fused_moe(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, weights: torch.Tensor,
              experts: torch.Tensor, block_m: int = 32, block_n: int = 64, block_k: Optional[int] = None,
              num_warps: int = 4) -> torch.Tensor:
    """sum_j weights[t, j] * down[e] @ (silu(gate[e] @ x[t]) * up[e] @ x[t]) with e = experts[t, j], for x [T, H],
    gate_up [E, 2F, H], down [E, H, F], weights [T, top_k] (fp32) and experts [T, top_k]. Returns [T, H] in x's
    dtype; the top_k terms are summed in fp32. block_m (rows per block), block_n and block_k (tile sizes) and
    num_warps are for tuning (scripts/benchmark_moe_kernel.py); the result does not depend on them."""
    T, H = x.shape
    E, two_f, _ = gate_up.shape
    F = two_f // 2
    top_k = experts.shape[1]
    n = T * top_k
    if block_k is None:
        try:
            return fused_moe(x, gate_up, down, weights, experts, block_m, block_n, _BLOCK_K[0], num_warps)
        except triton.runtime.errors.OutOfResources:
            if _BLOCK_K[0] == 64:
                raise
            _BLOCK_K[0] = 64
            return fused_moe(x, gate_up, down, weights, experts, block_m, block_n, 64, num_warps)
    sorted_pairs, block_expert, _ = moe_align(experts, E, block_m)
    x, gate_up, down = x.contiguous(), gate_up.contiguous(), down.contiguous()
    ieee = x.dtype == torch.float32
    h = torch.empty(sorted_pairs.numel(), F, device=x.device, dtype=x.dtype)
    nb = block_expert.numel()
    _moe_gate_up_kernel[(nb, triton.cdiv(F, block_n))](
        x, gate_up, sorted_pairs, block_expert, h, n, E, H, F,
        x.stride(0), gate_up.stride(0), gate_up.stride(1), h.stride(0),
        TOP_K=top_k, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, IEEE=ieee, num_warps=num_warps)
    y = torch.empty(n, H, device=x.device, dtype=torch.float32)
    _moe_down_kernel[(nb, triton.cdiv(H, block_n))](
        h, down, sorted_pairs, block_expert, weights.reshape(-1).float().contiguous(), y, n, E, F, H,
        h.stride(0), down.stride(0), down.stride(1), y.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, IEEE=ieee, num_warps=num_warps)
    return moe_sum(y, T, x.dtype)


# ------------------------------------------------------------------------------------------- 4-bit experts
#
# The same two kernels over experts stored as turing.models.quant packs them: unsigned 4-bit codes, two per byte
# (low nibble = even column), an fp16 scale and a uint8 zero point per `group` columns, w = (q - zero) * scale. Each
# packed byte is loaded once and split into the weights of an even and an odd input column, as in triton_w4a16.py.

if HAS_TRITON:
    @triton.jit
    def _moe_sum_kernel(Y, OUT, H, s_y, s_o, TOP_K: tl.constexpr, BLOCK: tl.constexpr):
        """OUT[t] = the sum of Y's TOP_K rows for token t (rows t * TOP_K ...), added in fp32."""
        t = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ok = offs < H
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for j in tl.static_range(TOP_K):
            acc += tl.load(Y + (t * TOP_K + j).to(tl.int64) * s_y + offs, mask=ok, other=0.0).to(tl.float32)
        tl.store(OUT + t.to(tl.int64) * s_o + offs, acc.to(OUT.dtype.element_ty), mask=ok)


def moe_sum(y: torch.Tensor, tokens: int, dtype: torch.dtype) -> torch.Tensor:
    """y [tokens * top_k, H] (a token's rows together) -> [tokens, H] in dtype: the terms of a token summed in fp32.
    One launch on CUDA where PyTorch takes a cast, a sum and a cast."""
    n, H = y.shape
    top_k = n // tokens
    if not (HAS_TRITON and y.is_cuda and y.stride(1) == 1):
        return y.view(tokens, top_k, H).float().sum(1).to(dtype)
    out = torch.empty(tokens, H, device=y.device, dtype=dtype)
    _moe_sum_kernel[(tokens, triton.cdiv(H, 1024))](y, out, H, y.stride(0), out.stride(0), TOP_K=top_k, BLOCK=1024)
    return out


if HAS_TRITON:
    @triton.jit
    def _gated_add_kernel(A, G, B, OUT, H, s_a, s_g, s_b, s_o, BLOCK: tl.constexpr):
        """OUT[t] = A[t] + sigmoid(G[t]) * B[t] for rows of H columns and one gate logit per row."""
        t = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ok = offs < H
        gate = tl.sigmoid(tl.load(G + t * s_g).to(tl.float32)).to(OUT.dtype.element_ty)
        a = tl.load(A + t * s_a + offs, mask=ok, other=0.0)
        b = tl.load(B + t * s_b + offs, mask=ok, other=0.0)
        tl.store(OUT + t * s_o + offs, a + gate * b, mask=ok)


def gated_add(a: torch.Tensor, gate_logit: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a + sigmoid(gate_logit) * b for a, b [T, H] and gate_logit [T, 1] (Qwen2-MoE's shared expert joining the
    routed experts): one launch on CUDA for half-precision rows, where PyTorch takes three. Each operation rounds to
    the rows' dtype, as PyTorch's do."""
    if not (HAS_TRITON and a.is_cuda and a.dtype in (torch.float16, torch.bfloat16) and a.dtype == b.dtype
            and gate_logit.dtype == a.dtype and a.dim() == 2 and a.stride(1) == 1 and b.stride(1) == 1):
        return a + torch.sigmoid(gate_logit) * b
    T, H = a.shape
    out = torch.empty_like(a)
    _gated_add_kernel[(T, triton.cdiv(H, 1024))](a, gate_logit, b, out, H, a.stride(0), gate_logit.stride(0),
                                                 b.stride(0), out.stride(0), BLOCK=1024)
    return out


def fused_moe_w4_supported(x: torch.Tensor, gate_up_q: torch.Tensor, act, group: int) -> bool:
    return (HAS_TRITON and x.is_cuda and gate_up_q.is_cuda and act is torch.nn.functional.silu
            and x.dtype in (torch.float16, torch.bfloat16) and group % 64 == 0)


if HAS_TRITON:
    @triton.jit
    def _w4_groups(q, sc, zp, offs_n, n_ok, k0, K, s_qn, s_sn, GROUP: tl.constexpr, BLOCK_K: tl.constexpr):
        """(q - zero) * scale for BLOCK_K columns from k0 on of the rows offs_n of one packed matrix, as two
        [rows, groups, GROUP / 2] tensors: the even and the odd columns. Whole quantization groups per step, so a
        row gives BLOCK_K / 2 contiguous bytes and one scale and zero per group. On an L4, rows read 128 bytes or
        more at a time stream from memory at 230-255 GB/s and rows read 32 bytes at a time at about 107."""
        g = k0 // GROUP + tl.arange(0, BLOCK_K // GROUP)
        kb = g[:, None] * (GROUP // 2) + tl.arange(0, GROUP // 2)[None, :]
        wb = tl.load(q + offs_n[:, None, None] * s_qn + kb[None, :, :],
                     mask=n_ok[:, None, None] & (kb < K // 2)[None, :, :], other=0)
        s_ok = n_ok[:, None] & (g < K // GROUP)[None, :]
        s = tl.load(sc + offs_n[:, None] * s_sn + g[None, :], mask=s_ok, other=0.0).to(tl.float32)[:, :, None]
        z = tl.load(zp + offs_n[:, None] * s_sn + g[None, :], mask=s_ok, other=0).to(tl.float32)[:, :, None]
        return ((wb & 0x0F).to(tl.float32) - z) * s, (((wb >> 4) & 0x0F).to(tl.float32) - z) * s

    @triton.jit
    def _moe_gate_up_w4_kernel(X, Q, S, Z, PAIRS, BLOCK_E, H, n_pairs, num_experts, K, F,
                               s_xt, s_qe, s_qn, s_se, s_sn, s_hr,
                               TOP_K: tl.constexpr, GROUP: tl.constexpr, BLOCK_M: tl.constexpr,
                               BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        e = tl.load(BLOCK_E + pid_m)
        if e >= num_experts:
            return
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        pair = tl.load(PAIRS + rows)
        valid = pair < n_pairs
        tok = pair // TOP_K
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = offs_n < F
        offs_b = tl.arange(0, BLOCK_K // 2)  # packed bytes of one K tile
        q = Q + e.to(tl.int64) * s_qe
        sc_e = S + e.to(tl.int64) * s_se
        zp_e = Z + e.to(tl.int64) * s_se
        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kb = k0 // 2 + offs_b
            b_ok = kb < K // 2
            g = k0 // GROUP
            x_even = tl.load(X + tok[:, None].to(tl.int64) * s_xt + (2 * kb)[None, :],
                             mask=valid[:, None] & b_ok[None, :], other=0.0)
            x_odd = tl.load(X + tok[:, None].to(tl.int64) * s_xt + (2 * kb + 1)[None, :],
                            mask=valid[:, None] & b_ok[None, :], other=0.0)
            w_ok = n_ok[:, None] & b_ok[None, :]
            if BLOCK_K > GROUP:  # several groups per step: gate rows
                lo, hi = _w4_groups(q, sc_e, zp_e, offs_n, n_ok, k0, K, s_qn, s_sn, GROUP, BLOCK_K)
                lo = tl.reshape(lo, (BLOCK_N, BLOCK_K // 2))
                hi = tl.reshape(hi, (BLOCK_N, BLOCK_K // 2))
            else:  # BLOCK_K divides GROUP: one scale and zero per weight row of this tile
                wb = tl.load(q + offs_n[:, None] * s_qn + kb[None, :], mask=w_ok, other=0)
                sc = tl.load(sc_e + offs_n * s_sn + g, mask=n_ok, other=0.0).to(tl.float32)
                zp = tl.load(zp_e + offs_n * s_sn + g, mask=n_ok, other=0).to(tl.float32)
                lo = ((wb & 0x0F).to(tl.float32) - zp[:, None]) * sc[:, None]
                hi = (((wb >> 4) & 0x0F).to(tl.float32) - zp[:, None]) * sc[:, None]
            acc_g += tl.dot(x_even, tl.trans(lo.to(x_even.dtype)))
            acc_g += tl.dot(x_odd, tl.trans(hi.to(x_odd.dtype)))
            if BLOCK_K > GROUP:  # up rows
                lo, hi = _w4_groups(q, sc_e, zp_e, offs_n + F, n_ok, k0, K, s_qn, s_sn, GROUP, BLOCK_K)
                lo = tl.reshape(lo, (BLOCK_N, BLOCK_K // 2))
                hi = tl.reshape(hi, (BLOCK_N, BLOCK_K // 2))
            else:
                wb = tl.load(q + (offs_n[:, None] + F) * s_qn + kb[None, :], mask=w_ok, other=0)
                sc = tl.load(sc_e + (offs_n + F) * s_sn + g, mask=n_ok, other=0.0).to(tl.float32)
                zp = tl.load(zp_e + (offs_n + F) * s_sn + g, mask=n_ok, other=0).to(tl.float32)
                lo = ((wb & 0x0F).to(tl.float32) - zp[:, None]) * sc[:, None]
                hi = (((wb >> 4) & 0x0F).to(tl.float32) - zp[:, None]) * sc[:, None]
            acc_u += tl.dot(x_even, tl.trans(lo.to(x_even.dtype)))
            acc_u += tl.dot(x_odd, tl.trans(hi.to(x_odd.dtype)))
        g_out = acc_g.to(H.dtype.element_ty).to(tl.float32)
        u_out = acc_u.to(H.dtype.element_ty).to(tl.float32)
        h = (g_out * tl.sigmoid(g_out)).to(H.dtype.element_ty).to(tl.float32) * u_out
        tl.store(H + rows[:, None].to(tl.int64) * s_hr + offs_n[None, :], h.to(H.dtype.element_ty),
                 mask=valid[:, None] & n_ok[None, :])

    @triton.jit
    def _moe_down_w4_kernel(Hs, Q, S, Z, PAIRS, BLOCK_E, RW, Y, n_pairs, num_experts, F, N,
                            s_hr, s_qe, s_qn, s_se, s_sn, s_yp,
                            GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                            BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        e = tl.load(BLOCK_E + pid_m)
        if e >= num_experts:
            return
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        pair = tl.load(PAIRS + rows)
        valid = pair < n_pairs
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = offs_n < N
        offs_b = tl.arange(0, BLOCK_K // 2)
        q = Q + e.to(tl.int64) * s_qe
        sc_e = S + e.to(tl.int64) * s_se
        zp_e = Z + e.to(tl.int64) * s_se
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, F, BLOCK_K):
            kb = k0 // 2 + offs_b
            b_ok = kb < F // 2
            g = k0 // GROUP
            h_even = tl.load(Hs + rows[:, None].to(tl.int64) * s_hr + (2 * kb)[None, :],
                             mask=valid[:, None] & b_ok[None, :], other=0.0)
            h_odd = tl.load(Hs + rows[:, None].to(tl.int64) * s_hr + (2 * kb + 1)[None, :],
                            mask=valid[:, None] & b_ok[None, :], other=0.0)
            if BLOCK_K > GROUP:
                lo, hi = _w4_groups(q, sc_e, zp_e, offs_n, n_ok, k0, F, s_qn, s_sn, GROUP, BLOCK_K)
                lo = tl.reshape(lo, (BLOCK_N, BLOCK_K // 2))
                hi = tl.reshape(hi, (BLOCK_N, BLOCK_K // 2))
            else:
                wb = tl.load(q + offs_n[:, None] * s_qn + kb[None, :], mask=n_ok[:, None] & b_ok[None, :], other=0)
                sc = tl.load(sc_e + offs_n * s_sn + g, mask=n_ok, other=0.0).to(tl.float32)
                zp = tl.load(zp_e + offs_n * s_sn + g, mask=n_ok, other=0).to(tl.float32)
                lo = ((wb & 0x0F).to(tl.float32) - zp[:, None]) * sc[:, None]
                hi = (((wb >> 4) & 0x0F).to(tl.float32) - zp[:, None]) * sc[:, None]
            acc += tl.dot(h_even, tl.trans(lo.to(h_even.dtype)))
            acc += tl.dot(h_odd, tl.trans(hi.to(h_odd.dtype)))
        rw = tl.load(RW + pair, mask=valid, other=0.0)
        y = acc.to(Hs.dtype.element_ty).to(tl.float32) * rw[:, None]
        tl.store(Y + pair[:, None].to(tl.int64) * s_yp + offs_n[None, :], y, mask=valid[:, None] & n_ok[None, :])

    # The same two kernels for blocks of a few rows (a decode step at small batch, where an expert has one or two
    # tokens). No tensor-core tile, of which most rows would be padding: the products of a step's columns are added
    # into registers, one per (row, weight row, column pair), and summed once after the last step.

    @triton.jit
    def _moe_gate_up_w4_rows_kernel(X, Q, S, Z, PAIRS, BLOCK_E, H, n_pairs, num_experts, K, F,
                                    s_xt, s_qe, s_qn, s_se, s_sn, s_hr,
                                    TOP_K: tl.constexpr, GROUP: tl.constexpr, BLOCK_M: tl.constexpr,
                                    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        e = tl.load(BLOCK_E + pid_m)
        if e >= num_experts:
            return
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        pair = tl.load(PAIRS + rows)
        valid = pair < n_pairs
        tok = pair // TOP_K
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = offs_n < F
        offs_g = tl.arange(0, BLOCK_K // GROUP)
        offs_b = tl.arange(0, GROUP // 2)
        q = Q + e.to(tl.int64) * s_qe
        sc_e = S + e.to(tl.int64) * s_se
        zp_e = Z + e.to(tl.int64) * s_se
        acc_g = tl.zeros((BLOCK_M, BLOCK_N, BLOCK_K // GROUP, GROUP // 2), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N, BLOCK_K // GROUP, GROUP // 2), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kb = (k0 // GROUP + offs_g)[:, None] * (GROUP // 2) + offs_b[None, :]  # [groups, bytes of a group]
            x_ok = valid[:, None, None] & (kb < K // 2)[None, :, :]
            xe = tl.load(X + tok[:, None, None].to(tl.int64) * s_xt + (2 * kb)[None, :, :], mask=x_ok,
                         other=0.0).to(tl.float32)[:, None, :, :]
            xo = tl.load(X + tok[:, None, None].to(tl.int64) * s_xt + (2 * kb + 1)[None, :, :], mask=x_ok,
                         other=0.0).to(tl.float32)[:, None, :, :]
            lo, hi = _w4_groups(q, sc_e, zp_e, offs_n, n_ok, k0, K, s_qn, s_sn, GROUP, BLOCK_K)  # gate rows
            acc_g += lo[None, :, :, :] * xe + hi[None, :, :, :] * xo
            lo, hi = _w4_groups(q, sc_e, zp_e, offs_n + F, n_ok, k0, K, s_qn, s_sn, GROUP, BLOCK_K)  # up rows
            acc_u += lo[None, :, :, :] * xe + hi[None, :, :, :] * xo
        g_out = tl.sum(tl.sum(acc_g, 3), 2).to(H.dtype.element_ty).to(tl.float32)
        u_out = tl.sum(tl.sum(acc_u, 3), 2).to(H.dtype.element_ty).to(tl.float32)
        h = (g_out * tl.sigmoid(g_out)).to(H.dtype.element_ty).to(tl.float32) * u_out
        tl.store(H + rows[:, None].to(tl.int64) * s_hr + offs_n[None, :], h.to(H.dtype.element_ty),
                 mask=valid[:, None] & n_ok[None, :])

    @triton.jit
    def _moe_down_w4_rows_kernel(Hs, Q, S, Z, PAIRS, BLOCK_E, RW, Y, n_pairs, num_experts, F, N,
                                 s_hr, s_qe, s_qn, s_se, s_sn, s_yp,
                                 GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                 BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        e = tl.load(BLOCK_E + pid_m)
        if e >= num_experts:
            return
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        pair = tl.load(PAIRS + rows)
        valid = pair < n_pairs
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = offs_n < N
        offs_g = tl.arange(0, BLOCK_K // GROUP)
        offs_b = tl.arange(0, GROUP // 2)
        q = Q + e.to(tl.int64) * s_qe
        sc_e = S + e.to(tl.int64) * s_se
        zp_e = Z + e.to(tl.int64) * s_se
        acc = tl.zeros((BLOCK_M, BLOCK_N, BLOCK_K // GROUP, GROUP // 2), dtype=tl.float32)
        for k0 in range(0, F, BLOCK_K):
            kb = (k0 // GROUP + offs_g)[:, None] * (GROUP // 2) + offs_b[None, :]
            h_ok = valid[:, None, None] & (kb < F // 2)[None, :, :]
            he = tl.load(Hs + rows[:, None, None].to(tl.int64) * s_hr + (2 * kb)[None, :, :], mask=h_ok,
                         other=0.0).to(tl.float32)[:, None, :, :]
            ho = tl.load(Hs + rows[:, None, None].to(tl.int64) * s_hr + (2 * kb + 1)[None, :, :], mask=h_ok,
                         other=0.0).to(tl.float32)[:, None, :, :]
            lo, hi = _w4_groups(q, sc_e, zp_e, offs_n, n_ok, k0, F, s_qn, s_sn, GROUP, BLOCK_K)
            acc += lo[None, :, :, :] * he + hi[None, :, :, :] * ho
        rw = tl.load(RW + pair, mask=valid, other=0.0)
        y = tl.sum(tl.sum(acc, 3), 2).to(Hs.dtype.element_ty).to(tl.float32) * rw[:, None]
        tl.store(Y + pair[:, None].to(tl.int64) * s_yp + offs_n[None, :], y, mask=valid[:, None] & n_ok[None, :])



def fused_moe_w4(x: torch.Tensor, gate_up, down, weights: torch.Tensor, experts: torch.Tensor, group: int,
                 block_m: Optional[int] = None, block_n: Optional[int] = None, block_k: Optional[int] = None,
                 num_warps: Optional[int] = None) -> torch.Tensor:
    """fused_moe for packed 4-bit experts: gate_up and down are (codes [E, rows, cols / 2] uint8, scales
    [E, rows, cols / group] fp16, zeros uint8) with rows x cols = 2F x H and H x F. Blocks of fewer than 16 rows
    (block_m) run the kernels without a tensor-core tile; block_k columns are read per step."""
    (gq, gs, gz), (dq, ds, dz) = gate_up, down
    T, H = x.shape
    E, two_f = gq.shape[:2]
    F = two_f // 2
    top_k = experts.shape[1]
    n = T * top_k
    if block_m is None:  # few pairs for the experts there are (most experts get one row): a block per pair, read
        # by the kernels without a tensor-core tile; else tiles of 16 rows, or 64 for prefill-sized inputs
        block_m = 1 if 2 * n <= E else 16 if n <= 1024 else 64
    few = block_m < 16
    # 256 columns per step: each weight row is read in pieces of 128 bytes, which stream from memory at about twice
    # the rate of the 32 bytes of a 64-column step. On an L4 with Qwen1.5-MoE's shapes and cold weights
    # (results_moe_kernel_w4_qwen_cuda.json), ms per layer: 0.159 -> 0.107 for one token, 1.01 -> 0.78 for 8,
    # 2.33 -> 1.90 for 64; OLMoE's (results_moe_kernel_w4_olmoe_cuda.json): 0.214 -> 0.157, 1.06 -> 0.83, 1.82 -> 1.51.
    # 64-row tiles (prefill-sized inputs) keep 64 columns: 5.5 ms against 11.2 with 256 at 512 tokens
    # (results_moe_kernel_w4_prefill_qwen_cuda.json).
    block_n, num_warps = block_n or (16 if few else 64), num_warps or (8 if few else 4)
    block_k = block_k or (64 if block_m >= 64 else 256)
    if block_k % group and (few or group % block_k):
        raise ValueError(f"block_k {block_k} must be a multiple of the quantization group {group}"
                         + ("" if few else " or divide it"))
    gate_up_kernel, down_kernel = ((_moe_gate_up_w4_rows_kernel, _moe_down_w4_rows_kernel) if few
                                   else (_moe_gate_up_w4_kernel, _moe_down_w4_kernel))
    sorted_pairs, block_expert, _ = moe_align(experts, E, block_m)
    x = x.contiguous()
    h = torch.empty(sorted_pairs.numel(), F, device=x.device, dtype=x.dtype)
    nb = block_expert.numel()
    gate_up_kernel[(nb, triton.cdiv(F, block_n))](
        x, gq, gs, gz, sorted_pairs, block_expert, h, n, E, H, F,
        x.stride(0), gq.stride(0), gq.stride(1), gs.stride(0), gs.stride(1), h.stride(0),
        TOP_K=top_k, GROUP=group, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=num_warps)
    y = torch.empty(n, H, device=x.device, dtype=torch.float32)
    down_kernel[(nb, triton.cdiv(H, block_n))](
        h, dq, ds, dz, sorted_pairs, block_expert, weights.reshape(-1).float().contiguous(), y, n, E, F, H,
        h.stride(0), dq.stride(0), dq.stride(1), ds.stride(0), ds.stride(1), y.stride(0),
        GROUP=group, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=num_warps)
    return moe_sum(y, T, x.dtype)
