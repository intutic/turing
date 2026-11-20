"""
Decoder-only causal LM (SubspaceCausalLM) covering the HuggingFace families the loader converts: Llama, Mistral, Qwen2/3,
Gemma 1/2/3, Phi-2/3, GPT-2, OPT, GPT-NeoX and the MoE families Mixtral, Qwen2/3-MoE and OLMoE. Each family's features
(norm type, MLP shape, biases, positions, partial rotary, sliding windows, soft-capping, MoE routing) come from
ModelConfig, and tests/test_hf_parity.py checks every family against HuggingFace's own logits.
"""

import math
from typing import List, Optional, Tuple, Union, Any
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..core.rope import RotaryEmbedding, apply_rotary_pos_emb
from ..core.kv_cache import INT4_GROUP, KVCache, KVPool, _dequantize, _quantize
from ..kernels.triton_kv_decode import pool_decode_supported, triton_pool_decode
from ..kernels import triton_fused_ops as fused_ops
from ..kernels.triton_kv_prefill import pool_prefill_preferred, pool_prefill_supported, triton_pool_prefill
from ..kernels.flashinfer_prefill import flashinfer_prefill, flashinfer_prefill_supported


class RMSNorm(nn.Module):
    """RMSNorm in fp32. offset 0 is Llama's (normalise, cast, scale by weight); offset 1 is Gemma's (scale by
    1 + weight in fp32, then cast)."""

    def __init__(self, dim: int, eps: float = 1e-6, offset: float = 0.0):
        super().__init__()
        self.eps, self.offset = eps, offset
        self.weight = nn.Parameter(torch.zeros(dim) if offset else torch.ones(dim))

    def fusable(self, x: torch.Tensor) -> bool:
        """The fused Triton norm applies (it reproduces this module's rounding; a weight in another dtype than x would
        promote the Llama-style output, which it does not)."""
        return fused_ops.fused_ops_supported(x) and (self.offset or self.weight.dtype == x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fusable(x):
            return fused_ops.rms_norm(x, self.weight, self.eps, self.offset)
        xf = x.float()  # fp16 squares overflow for large activations
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.offset:
            return (normed * (self.offset + self.weight.float())).to(x.dtype)
        return normed.to(x.dtype) * self.weight


def make_norm(config: ModelConfig, dim: int) -> nn.Module:
    if config.norm_type == "layer":
        return nn.LayerNorm(dim, eps=config.rms_norm_eps)
    return RMSNorm(dim, config.rms_norm_eps, config.norm_offset)


def activation(name: str):
    """HuggingFace's ACT2FN for the activations the supported families use."""
    if name in ("silu", "swish"):
        return F.silu
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name in ("gelu_new", "gelu_fast", "gelu_pytorch_tanh"):
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "quick_gelu":
        return lambda x: x * torch.sigmoid(1.702 * x)
    raise ValueError(f"activation '{name}' is not supported")


def _weight_of(m: nn.Module) -> torch.Tensor:
    return m.qweight if hasattr(m, "qweight") else m.weight


def fuse_linears(owner: nn.Module, key: str, parts: Tuple[nn.Module, ...]) -> bool:
    """One linear layer over parts that share an input (q/k/v, gate/up), so a step launches one GEMM instead of
    len(parts). The parts' weights become views of the fused weight: no extra memory, and the state dict, adapters and
    hooks still see the separate layers. Kept outside the module tree (owner.__dict__[key]); used by fused_outputs only
    while the parts are still the same objects on the same storage. False when the parts cannot be fused."""
    from .quant import QuantLinear
    kind = type(parts[0])
    if kind not in (nn.Linear, QuantLinear) or any(type(p) is not kind for p in parts):
        return False
    w0 = _weight_of(parts[0])
    if any(_weight_of(p).device != w0.device or _weight_of(p).dtype != w0.dtype or p.in_features != parts[0].in_features
           or (p.bias is None) != (parts[0].bias is None) for p in parts):
        return False
    if kind is QuantLinear and any(p.group != parts[0].group for p in parts):
        return False
    sizes = [p.out_features for p in parts]
    with torch.inference_mode(False), torch.no_grad():  # normal tensors: the model is used in and out of inference mode
        if kind is QuantLinear:
            fused = QuantLinear(parts[0].in_features, sum(sizes), parts[0].bias is not None, parts[0].group,
                                device="meta")
            names = ["qweight", "scales", "zeros"]
        else:
            fused = nn.Linear(parts[0].in_features, sum(sizes), bias=parts[0].bias is not None, device="meta")
            names = ["weight"]
        names += ["bias"] if parts[0].bias is not None else []
        for name in names:
            full = torch.cat([getattr(p, name).data for p in parts])
            is_param = isinstance(getattr(parts[0], name), nn.Parameter)
            grad = is_param and getattr(parts[0], name).requires_grad
            setattr(fused, name, nn.Parameter(full, requires_grad=grad) if is_param else full)
            lo = 0
            for p, n in zip(parts, sizes):
                view = full[lo:lo + n]
                setattr(p, name, nn.Parameter(view, requires_grad=grad) if is_param else view)
                lo += n
    owner.__dict__[key] = (fused, parts, sizes)
    return True


def fused_outputs(owner: nn.Module, key: str, parts: Tuple[nn.Module, ...], x: torch.Tensor):
    """The parts' outputs from their fused layer (fuse_linears), or None: not fused, a part replaced (LoRA, TEAL,
    tensor parallel), its weight moved (.to()), or a hook on a part (calibration) that must see the separate call."""
    f = owner.__dict__.get(key)
    if f is None:
        return None
    fused, fparts, sizes = f
    if any(a is not b or a._forward_hooks or a._forward_pre_hooks for a, b in zip(parts, fparts)) \
            or _weight_of(parts[0]).data_ptr() != _weight_of(fused).data_ptr():
        return None
    return fused(x).split(sizes, dim=-1)


class SubspaceMLP(nn.Module):
    """Feed-forward block: gated (down(act(gate(x)) * up(x)), Llama/Gemma/Phi-3) or plain (down(act(up(x))), GPT-2/OPT/
    NeoX/Phi-2). FFN channel pruning (sparsity_ratio > 0) keeps a subset of intermediate channels as smaller contiguous
    weights (set_active_tiles), so the pruned forward is three plain matmuls on every device."""

    def __init__(self, config: ModelConfig, ffn_dim: Optional[int] = None):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.ffn_dim = ffn_dim or config.ffn_dim
        self.tile_size = config.tile_size
        self.act = activation(config.hidden_act)
        bias = config.mlp_bias
        self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=bias) if config.mlp_gated else None
        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=bias)
        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=bias)

    def set_active_tiles(self, active_tile_indices: torch.Tensor):
        """Keep only these tiles (of tile_size intermediate channels): the weights become contiguous slices."""
        t = active_tile_indices.to(device=self.up_proj.weight.device, dtype=torch.long)
        ch = (t[:, None] * self.tile_size + torch.arange(self.tile_size, device=t.device)).flatten()
        ch = ch[ch < self.ffn_dim]

        def rows(lin):
            new = nn.Linear(lin.in_features, len(ch), bias=lin.bias is not None).to(lin.weight)
            new.weight.data.copy_(lin.weight.data[ch])
            if lin.bias is not None:
                new.bias.data.copy_(lin.bias.data[ch])
            return new
        if self.gate_proj is not None:
            self.gate_proj = rows(self.gate_proj)
        self.up_proj = rows(self.up_proj)
        down = nn.Linear(len(ch), self.hidden_dim, bias=self.down_proj.bias is not None).to(self.down_proj.weight)
        down.weight.data.copy_(self.down_proj.weight.data[:, ch])
        if self.down_proj.bias is not None:
            down.bias.data.copy_(self.down_proj.bias.data)
        self.down_proj = down
        self.ffn_dim = len(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_proj is not None:
            gu = fused_outputs(self, "_gate_up", (self.gate_proj, self.up_proj), x)
            g, u = gu if gu is not None else (self.gate_proj(x), self.up_proj(x))
            if self.act is F.silu and fused_ops.fused_ops_supported(g):
                return self.down_proj(fused_ops.silu_mul(g, u))
            return self.down_proj(self.act(g) * u)
        return self.down_proj(self.act(self.up_proj(x)))


class SparseMoE(nn.Module):
    """Top-k routed experts as in HuggingFace's Mixtral / Qwen2-MoE / Qwen3-MoE / OLMoE blocks: softmax over every
    expert in fp32, top-k, optional renormalisation, each chosen expert's gated MLP weighted by its probability, plus
    Qwen2-MoE's sigmoid-gated shared expert. Expert weights are stacked: gate_up [E, 2F, H], down [E, H, F]."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        E, H, Fd = config.num_experts, config.hidden_dim, config.moe_ffn_dim or config.ffn_dim
        self.num_experts, self.top_k, self.norm_topk_prob = E, config.num_experts_per_tok, config.norm_topk_prob
        self.router = nn.Linear(H, E, bias=False)
        self.gate_up = nn.Parameter(torch.empty(E, 2 * Fd, H))
        self.down = nn.Parameter(torch.empty(E, H, Fd))
        nn.init.normal_(self.gate_up, std=0.02)
        nn.init.normal_(self.down, std=0.02)
        self.act = activation(config.hidden_act)
        self.shared_expert = None
        if config.shared_expert_ffn_dim:
            self.shared_expert = SubspaceMLP(config, config.shared_expert_ffn_dim)
            self.shared_expert_gate = nn.Linear(H, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        probs = F.softmax(self.router(x).float(), dim=-1)
        weights, experts = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x)
        # shortcut: one host sync per layer for the experts hit (fine for prefill and eager decode); MoE models
        # therefore skip CUDA graphs. A grouped-GEMM kernel over all experts would remove it.
        hit = torch.unique(experts).tolist()
        for e in hit:
            token, slot = torch.where(experts == e)
            gate, up = F.linear(x[token], self.gate_up[e]).chunk(2, dim=-1)
            y = F.linear(self.act(gate) * up, self.down[e])
            out.index_add_(0, token, (y * weights[token, slot, None]).to(out.dtype))
        if self.shared_expert is not None:
            out = out + torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return out.view(shape)


class DecodeStep:
    """One batched decode step: the pool, each row's page-table slot and length before the step (device int32
    tensors), and, outside CUDA graphs, the rows' caches and host lengths. RoPE for the rows is computed once on the
    device and shared by every layer."""

    def __init__(self, pool: KVPool, slots: torch.Tensor, starts_t: torch.Tensor,
                 caches: Optional[List[KVCache]] = None, starts: Optional[List[int]] = None):
        self.pool, self.slots, self.starts_t, self.caches, self.starts = pool, slots, starts_t, caches, starts
        self._freqs = {}  # per RoPE configuration (Gemma 3's local and global layers differ)
        self.quest_plan = None
        self.quest_width: Optional[int] = None  # in a CUDA graph: the context bucket's page-table width for Quest

    @classmethod
    def of(cls, caches: List[KVCache], device) -> "DecodeStep":
        starts = [c.length for c in caches]
        rows = torch.tensor([[c.slot for c in caches], starts], dtype=torch.int32).to(device)
        return cls(caches[0].pool, rows[0], rows[1], caches, starts)

    def freqs(self, rope) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (rope.dim, rope.base_theta, rope.params)
        if key not in self._freqs:
            self._freqs[key] = rope.compute_freqs_device(self.starts_t)
        return self._freqs[key]


class PackedPrefill:
    """Prompt chunks of several requests as one row of tokens [1, T] (speed plan P4.3): the linear layers, norms and
    MLPs run once over every chunk; attention appends and attends per request through the paged prefill kernel. Built
    by SubspaceCausalLM.prefill_packed; positions [T] give RoPE (shared by every layer) and learned positions."""

    def __init__(self, caches: List[KVCache], lens: List[int], device):
        self.pool, self.caches, self.lens = caches[0].pool, caches, lens
        self.starts = [c.length for c in caches]
        self.offsets = [sum(lens[:i]) for i in range(len(lens))]
        self.positions = torch.cat([torch.arange(s, s + n) for s, n in zip(self.starts, lens)]).to(device)
        self.last = torch.tensor([o + n - 1 for o, n in zip(self.offsets, lens)], device=device)
        self.top = max(s + n for s, n in zip(self.starts, lens))
        self._freqs = {}

    def freqs(self, rope) -> Tuple[torch.Tensor, torch.Tensor]:
        """cos, sin [T, R] at each token's position (frequencies that do not depend on the sequence length)."""
        key = (rope.dim, rope.base_theta, rope.params)
        if key not in self._freqs:
            cos, sin = rope.compute_freqs(self.top, device=self.positions.device)
            self._freqs[key] = (cos[self.positions], sin[self.positions])
        return self._freqs[key]




class SubspaceAttention(nn.Module):
    """
    Multi-head / grouped-query attention with RoPE (full or partial) or none, optional q/k norms, sliding window,
    score soft-capping and a custom scale, over the paged KV cache.
    """
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.window = config.layer_window(layer_idx)
        self.scale = config.attn_scale if config.attn_scale is not None else self.head_dim ** -0.5
        self.softcap = config.attn_softcap

        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=config.o_proj_bias)
        if config.qk_norm or config.qk_norm_full:
            q_dim, k_dim = (self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim) if config.qk_norm_full \
                else (self.head_dim, self.head_dim)
            self.q_norm, self.k_norm = make_norm(config, q_dim), make_norm(config, k_dim)

        # Per-channel key smoothing for quantized KV caches (set by SubspaceCausalLM.calibrate_kv_smoothing): keys are
        # stored as k / s and queries scaled by s, so q.k is unchanged while outlier channels (Qwen2's key biases reach
        # 20x the median) no longer swamp the 4/8-bit grid of every other channel.
        self.register_buffer("k_smooth", None, persistent=False)   # [1, Hkv, 1, D]
        self.register_buffer("q_smooth", None, persistent=False)   # [1, H, 1, D]
        self._k_absmax = None  # running max |k| per (KV head, channel) during calibration
        self._kv_roundtrip = None  # (bits, group) during select_kv_native_layers: K and V take a low-bit round trip

        self.rope = None
        if not config.learned_positions and config.rotary_dim != 0:
            local = self.window is not None and config.rope_local_theta is not None
            self.rope = RotaryEmbedding(
                dim=config.rotary_dim or self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base_theta=config.rope_local_theta if local else config.rope_theta,
                params=config.rope_local_scaling if local else config.rope_scaling,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        start_pos: Union[int, List[int]] = 0,
        cache: Optional[KVCache] = None,
        step: Optional["DecodeStep"] = None,
        packed: Optional["PackedPrefill"] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Attention for new tokens at start_pos: through a KVCache (one request), a DecodeStep (one token per
        request), a PackedPrefill (several requests' chunks), or with none, causal attention over these tokens alone,
        returning their k and v."""
        batch, seq_len, _ = hidden_states.shape
        cfg = self.config

        qkv = fused_outputs(self, "_qkv", (self.q_proj, self.k_proj, self.v_proj), hidden_states)
        q, k, v = qkv if qkv is not None else \
            (self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states))
        if cfg.qk_norm_full:
            q, k = self.q_norm(q), self.k_norm(k)
        if cfg.clip_qkv is not None:  # OLMoE clamps after its norms
            q, k, v = (t.clamp(-cfg.clip_qkv, cfg.clip_qkv) for t in (q, k, v))
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if cfg.qk_norm and not cfg.qk_norm_full:
            q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            if step is not None:  # one position per row, on the device
                cos, sin = step.freqs(self.rope)
                r = cos.shape[-1]
                cos, sin = cos.view(batch, 1, 1, r), sin.view(batch, 1, 1, r)
            elif packed is not None:
                cos, sin = packed.freqs(self.rope)
            else:
                cos, sin = self.rope.compute_freqs(start_pos + seq_len, device=hidden_states.device, start=start_pos,
                                                   select_len=cache.rope_len if cache is not None else 0)
            if fused_ops.fused_ops_supported(q) and cos.shape[-1] % 2 == 0:
                q, k = fused_ops.rope(q, k, cos, sin)  # one pass per tensor, the cast to q's dtype inside
            else:
                q, k = apply_rotary_pos_emb(q, k, cos.to(q.dtype), sin.to(q.dtype))
        if self._k_absmax is not None:
            m = k.detach().abs().float().amax(dim=(0, 2))
            self._k_absmax = torch.maximum(self._k_absmax, m)
        if self.k_smooth is not None:
            k, q = k / self.k_smooth.to(k.dtype), q * self.q_smooth.to(q.dtype)
        if self._kv_roundtrip is not None:
            bits, group = self._kv_roundtrip
            k, v = (_dequantize(*_quantize(x, bits, group), group, x.dtype) for x in (k, v))

        if step is not None:  # one decode token for each request of the step
            return self.o_proj(self._decode_step(q, k, v, step)), None, None
        if packed is not None:
            return self.o_proj(self._attend_packed(q, k, v, packed)), None, None
        if cache is not None:
            return self.o_proj(self._attend_one(q, k, v, cache)), None, None

        mask = self._mask(0, seq_len, seq_len, q.device) if (self.window or self.softcap) else None
        out = self._attend(q, k, v, mask, causal=mask is None and seq_len > 1)
        return self.o_proj(out.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)), k, v

    def _mask(self, start: int, seq_len: int, total: int, device) -> torch.Tensor:
        """[seq_len, total] bool: query start + i sees key j when j <= start + i (and j > start + i - window)."""
        qpos = start + torch.arange(seq_len, device=device)[:, None]
        kpos = torch.arange(total, device=device)
        mask = kpos <= qpos
        if self.window:
            mask &= kpos > qpos - self.window
        return mask

    def _attend(self, q, k, v, mask, causal: bool) -> torch.Tensor:
        """softmax(q k^T * scale) v with an optional bool mask. Soft-capped scores, and every call on MPS, take the
        explicit path: PyTorch 2.9's MPS scaled_dot_product_attention caches one graph per tensor shape and reuses it
        with whatever `scale` the first call of that shape had, so a model whose scale differs from an earlier caller's
        (HF's OPT passes 1.0 after pre-scaling q) silently gets wrong attention."""
        if self.softcap is None and q.device.type != "mps":
            gqa = self.num_kv_groups > 1
            if gqa and mask is not None and q.is_cuda:
                # SDPA has no memory-efficient kernel for enable_gqa with a mask: it would fall back to the math path
                # and materialise [H, L, T] scores (GBs for a 2K chunk at 32K context). Repeating the KV heads costs
                # [H, T, D] instead.
                k, v = k.repeat_interleave(self.num_kv_groups, dim=1), v.repeat_interleave(self.num_kv_groups, dim=1)
                gqa = False
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, scale=self.scale,
                                                  enable_gqa=gqa)
        if self.num_kv_groups > 1:
            k, v = k.repeat_interleave(self.num_kv_groups, dim=1), v.repeat_interleave(self.num_kv_groups, dim=1)
        s = torch.matmul(q, k.transpose(2, 3)) * self.scale
        if self.softcap is not None:
            s = torch.tanh(s / self.softcap) * self.softcap
        if causal:
            mask = self._mask(0, q.shape[2], k.shape[2], q.device)
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
        p = F.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(p, v)

    def _attend_one(self, q, k, v, cache: KVCache) -> torch.Tensor:
        """One request: append k, v at cache.length, then attend q over its cache: [1, H, L, D] -> [1, L, H * D]."""
        batch, _, seq_len, _ = q.shape
        pool, layer, cfg = cache.pool, self.layer_idx, self.config
        start, total = cache.length, cache.length + seq_len
        pool.append(layer, cache, k, v)
        if seq_len > 1 and flashinfer_prefill_supported(pool, layer, q, self.window, self.softcap):  # D10 fallback
            return flashinfer_prefill(pool, layer, q, [cache.slot], [start], [seq_len], self.scale)
        if seq_len > 1 and pool_prefill_preferred(pool, layer, q, start):  # paged prefill kernel: no gather or copies
            return triton_pool_prefill(pool, layer, q, cache.slot, start, self.scale, self.window or 0,
                                       self.softcap or 0.0)
        mask, causal = None, False
        if (seq_len == 1 and self.window is None and cfg.kv_page_budget < 1.0 and pool.track_pages
                and total >= cfg.kv_select_min_tokens):
            sel = pool.select_pages(layer, [cache], [total], q, cfg.kv_page_budget, cfg.kv_select_min_tokens)[0]
            k_all, v_all, valid = pool.read_selected(layer, cache, sel, total)
            mask = valid.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_all, v_all = pool.read(layer, cache, total)
            if self.window is not None or (seq_len > 1 and start > 0):  # chunked prefill or a window
                mask = self._mask(start, seq_len, total, q.device)
            causal = mask is None and seq_len > 1  # start == 0
        out = self._attend(q, k_all, v_all, mask, causal)
        return out.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)

    def _attend_packed(self, q, k, v, packed: "PackedPrefill") -> torch.Tensor:
        """Each request's slice of the packed chunk: append its K/V, attend over its pages -> [1, T, H * D]."""
        pool, layer = packed.pool, self.layer_idx
        if flashinfer_prefill_supported(pool, layer, q, self.window, self.softcap):  # every request in one call
            for c, off, n in zip(packed.caches, packed.offsets, packed.lens):
                pool.append(layer, c, k[:, :, off:off + n], v[:, :, off:off + n])
            return flashinfer_prefill(pool, layer, q, [c.slot for c in packed.caches], packed.starts, packed.lens,
                                      self.scale)
        outs = []
        for c, off, n in zip(packed.caches, packed.offsets, packed.lens):  # packed.starts[i] is c.length
            sl = slice(off, off + n)
            outs.append(self._attend_one(q[:, :, sl], k[:, :, sl], v[:, :, sl], c))
        return torch.cat(outs, dim=1)

    def _decode_step(self, q, k, v, step: "DecodeStep") -> torch.Tensor:
        """One decode token per request: [B, H, 1, D] -> [B, 1, H * D]. On CUDA three fused launches serve every
        row (append, attention over each row's pages, split reduction); elsewhere each row takes _attend_one."""
        batch = q.shape[0]
        pool, layer, cfg = step.pool, self.layer_idx, self.config
        if not pool_decode_supported(pool, q) or cfg.decode_attention == "sdpa":
            quest = (self.window is None and cfg.kv_page_budget < 1.0 and pool.track_pages
                     and max(step.starts) + 1 >= cfg.kv_select_min_tokens)
            if quest:  # page selection per row
                return torch.cat([self._attend_one(q[b:b + 1], k[b:b + 1], v[b:b + 1], c)
                                  for b, c in enumerate(step.caches)])
            # one gather and one attention for every row (MPS and CPU)
            pool.append_batch(layer, step.slots, step.starts_t, k, v)
            lengths = [n + 1 for n in step.starts]
            k_all, v_all, valid = pool.read_batch(layer, step.slots, lengths)
            mask = valid[:, None, None, :]
            if self.window is not None:
                kpos = torch.arange(valid.shape[1], device=q.device)
                last = torch.tensor(step.starts, device=q.device)[:, None]
                mask = mask & (kpos[None, :] > last - self.window)[:, None, None, :]
            out = self._attend(q, k_all, v_all, mask, causal=False)
            return out.transpose(1, 2).reshape(batch, 1, self.num_heads * self.head_dim)
        sel = None
        if self.window is None and cfg.kv_page_budget < 1.0 and pool.track_pages:
            if step.quest_width is not None:  # CUDA graph: static shapes, all on the device
                if step.quest_plan is None:
                    step.quest_plan = pool.selection_plan_device(step.slots, step.starts_t, step.quest_width,
                                                                 cfg.kv_page_budget, cfg.kv_select_min_tokens)
                sel = pool.select_pages_planned(layer, q, step.quest_plan)
            elif step.starts is not None and max(step.starts) + 1 >= cfg.kv_select_min_tokens:
                if step.quest_plan is None:  # once per step: every layer shares the rows' page counts and pages
                    step.quest_plan = pool.selection_plan(step.caches, [n + 1 for n in step.starts],
                                                          cfg.kv_page_budget, cfg.kv_select_min_tokens)
                sel = pool.select_pages_planned(layer, q, step.quest_plan)
        out = triton_pool_decode(pool, layer, q, k, v, step.slots, step.starts_t, sel, scale=self.scale,
                                 window=self.window or 0, softcap=self.softcap or 0.0)
        return out.transpose(1, 2).reshape(batch, 1, self.num_heads * self.head_dim)


class SubspaceDecoderLayer(nn.Module):
    """
    Transformer decoder block: pre-norm (or post-norm) attention and MLP/MoE, sequential or parallel residual, with
    Gemma's extra output norms when sandwich_norm is set.
    """
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.input_layernorm = make_norm(config, config.hidden_dim)
        self.self_attn = SubspaceAttention(config, layer_idx)
        if not config.shared_layer_norm:
            self.post_attention_layernorm = make_norm(config, config.hidden_dim)  # the norm before the MLP
        if config.sandwich_norm:
            self.attn_out_norm = make_norm(config, config.hidden_dim)
            self.mlp_out_norm = make_norm(config, config.hidden_dim)
        self.mlp = SparseMoE(config) if config.layer_is_moe(layer_idx) else SubspaceMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        start_pos: Union[int, List[int]] = 0,
        cache: Optional[KVCache] = None,
        step: Optional["DecodeStep"] = None,
        packed: Optional["PackedPrefill"] = None,
        pending: Optional[torch.Tensor] = None,
        defer: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """pending: the previous layer's MLP output, not yet added to hidden_states; the add and this layer's input norm
        run as one fused pass. defer: return (x, MLP output, None) instead of (x + MLP output, k, v), for the next layer
        or the final norm to add (one launch fewer per layer; SubspaceCausalLM._body decides when both apply)."""
        cfg = self.config
        x = hidden_states
        if pending is not None:
            n = self.input_layernorm
            h, x = fused_ops.rms_norm(pending, n.weight, n.eps, n.offset, residual=x)
        else:
            h = self.input_layernorm(x) if cfg.pre_norm else x
        attn_out, k_out, v_out = self.self_attn(h, start_pos=start_pos, cache=cache, step=step, packed=packed)
        if cfg.parallel_residual:
            mlp_in = h if cfg.shared_layer_norm else self.post_attention_layernorm(x)
            return x + attn_out + self.mlp(mlp_in), k_out, v_out
        if cfg.sandwich_norm:
            attn_out = self.attn_out_norm(attn_out)
        norm = self.post_attention_layernorm if cfg.pre_norm else None
        if isinstance(norm, RMSNorm) and norm.fusable(attn_out) and x.dtype == attn_out.dtype:
            h, x = fused_ops.rms_norm(attn_out, norm.weight, norm.eps, norm.offset, residual=x)  # add + norm, one pass
        else:
            x = x + attn_out
            if not cfg.pre_norm:
                x = self.input_layernorm(x)
            h = self.post_attention_layernorm(x) if cfg.pre_norm else x
        m = self.mlp(h)
        if cfg.sandwich_norm:
            m = self.mlp_out_norm(m)
        if defer:
            return x, m, None
        x = x + m
        if not cfg.pre_norm:
            x = self.post_attention_layernorm(x)
        return x, k_out, v_out

class DecodeGraphs:
    """CUDA graphs of one decode step through SubspaceCausalLM._body, one per batch bucket. A step is copied into the
    graph's static buffers (token ids, page-table slots, lengths; padding rows point at the pool's scratch page) and
    replayed, so a step costs a few copies instead of every kernel launch. Rebuilt when the pool or its page table
    is replaced. With Quest (kv_page_budget < 1), steps whose longest row has reached kv_select_min_tokens use graphs
    captured per context bucket (page-table width), whose page selection runs on the device with static shapes. Not
    used for LoRA steps, MoE models, or off CUDA."""

    BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)
    CONTEXT_BUCKETS = tuple(2 ** i for i in range(9, 14))  # page-table widths for Quest graphs: 8K .. 128K tokens

    def __init__(self):
        # holds no reference to its model (the model passes itself in): no model <-> graphs cycle, so a dropped model
        # frees its KV pool at once, and a deep-copied model gets graphs of its own
        self.graphs = {}  # (batch bucket, Quest width or None) -> (key, graph, ids, rows, logits)

    def eligible(self, model: "SubspaceCausalLM", caches: List[KVCache], lora: bool) -> bool:
        cfg, pool = model.config, caches[0].pool
        return (cfg.cuda_graphs and pool.device.type == "cuda" and len(caches) <= self.BUCKETS[-1]
                and not lora and not torch.is_grad_enabled()
                and (not pool.track_pages or max(c.length for c in caches) + 1 <= self.CONTEXT_BUCKETS[-1] * 16)
                and cfg.num_experts == 0  # MoE routing picks experts on the host
                and cfg.decode_attention == "kernel"  # the SDPA path reads lengths on the host
                and pool_decode_supported(pool, model.embed_tokens.weight))

    # Capture and replay always run in inference mode: the graph's static buffers and the CUDA generator's graph-safe
    # state are created by the first capture, and an in-place update across modes (inference tensor touched outside
    # inference mode) fails mid-capture and leaves the generator unusable.
    @torch.inference_mode()
    def run(self, model: "SubspaceCausalLM", input_ids: torch.Tensor, caches: List[KVCache]) -> torch.Tensor:
        B, pool, cfg = len(caches), caches[0].pool, model.config
        bucket = next(b for b in self.BUCKETS if b >= B)
        width = None
        longest = max(c.length for c in caches) + 1
        if pool.track_pages and cfg.kv_page_budget < 1.0 and longest >= cfg.kv_select_min_tokens:
            width = next(w for w in self.CONTEXT_BUCKETS if w * 16 >= longest)
        key = (id(pool), pool.page_table.data_ptr())
        entry = self.graphs.get((bucket, width))
        if entry is None or entry[0] != key:
            entry = self._capture(model, bucket, pool, key, width)
        _, graph, ids, rows, logits = entry
        scratch = pool.scratch_slot()
        host = torch.tensor([[c.slot for c in caches] + [scratch] * (bucket - B),
                             [c.length for c in caches] + [0] * (bucket - B)], dtype=torch.int32)
        rows.copy_(host)
        ids[:B].copy_(input_ids)
        graph.replay()
        return logits[:B].clone()  # the next replay overwrites the static output

    @torch.inference_mode()
    def _capture(self, model: "SubspaceCausalLM", bucket: int, pool: KVPool, key, width: Optional[int] = None):
        device = pool.device
        torch.cuda.empty_cache()  # give capture the memory a long prefill left cached (rare: once per bucket and pool)
        ids = torch.zeros(bucket, 1, dtype=torch.long, device=device)
        rows = torch.zeros(2, bucket, dtype=torch.int32, device=device)
        rows[0] = pool.scratch_slot()
        step = DecodeStep(pool, rows[0], rows[1])
        step.quest_width = width
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):  # warm-up (Triton compilation, allocator) writes only the scratch page
            for _ in range(2):
                step._freqs, step.quest_plan = {}, None
                model._body(ids, rows[1], None, step)
        torch.cuda.current_stream(device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        step._freqs, step.quest_plan = {}, None
        with torch.cuda.graph(graph):
            logits = model._body(ids, rows[1], None, step)
        self.graphs[(bucket, width)] = (key, graph, ids, rows, logits)
        return self.graphs[(bucket, width)]


class SubspaceCausalLM(nn.Module):
    """
    End-to-end Autoregressive Subspace Causal Language Model with stateful generation.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_dim = config.hidden_dim
        self.adapter_names: Tuple[str, ...] = ()  # LoRA adapters loaded by turing.models.adapters.load_lora_adapter

        emb_dim = config.embed_proj_dim or config.hidden_dim
        self.embed_tokens = nn.Embedding(config.vocab_size, emb_dim)
        if config.learned_positions:
            self.embed_positions = nn.Embedding(config.max_position_embeddings + config.position_offset,
                                                config.hidden_dim)
        if config.embed_proj_dim:  # OPT-350m: narrower embeddings projected to the hidden size and back
            self.project_in = nn.Linear(emb_dim, config.hidden_dim, bias=False)
            self.project_out = nn.Linear(config.hidden_dim, emb_dim, bias=False)
        self.layers = nn.ModuleList([SubspaceDecoderLayer(config, i) for i in range(config.num_layers)])
        self.norm = make_norm(config, config.hidden_dim) if config.final_norm else None
        self.lm_head = nn.Linear(emb_dim, config.vocab_size, bias=config.lm_head_bias)
        self._graphs = DecodeGraphs()

        # Optional deep-to-shallow recirculation (off by default: it changes the model's output). Its SVD basis is
        # only built when enabled.
        self.router_layer_idx = config.router_layer_idx
        if config.enable_recirculation:
            from ..core.subspace import SubspaceRecirculation
            self.recirculation = SubspaceRecirculation(hidden_dim=config.hidden_dim, rank=config.rank_sub)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Union[KVCache, List[KVCache]]] = None,
        start_pos: int = 0,
        adapters: Optional[List[Optional[str]]] = None,
        logits_to_keep: int = 0
    ) -> Tuple[torch.Tensor, Union[KVCache, List[KVCache]]]:
        """past_key_values: None or a KVCache (filled in place and returned), or a list of KVCache, one per row of a
        [B, 1] decode step (batched decode across requests, returned as given). start_pos is taken from the cache.
        logits_to_keep > 0 computes logits for the last that many positions only (a prefill needs one).
        adapters: a LoRA adapter name (or None for the base model) per row of input_ids."""
        lora = adapters is not None and any(adapters)
        if lora:
            from .adapters import set_active_adapters
            set_active_adapters(self, list(adapters))
        try:
            return self._forward(input_ids, past_key_values, logits_to_keep, lora)
        finally:
            if lora:
                set_active_adapters(self, None)

    def can_pack_prefill(self, caches: List[KVCache]) -> bool:
        """Whether prefill_packed serves these caches: one CUDA pool the paged prefill kernel reads, RoPE whose
        frequencies do not depend on a sequence's length (not dynamic or LongRoPE), no recirculation."""
        from ..core.rope import RotaryEmbedding
        if not caches or self.config.enable_recirculation or any(c.pool is not caches[0].pool for c in caches):
            return False
        if not pool_prefill_supported(caches[0].pool, self.embed_tokens.weight):
            return False
        return all(m.kind not in ("dynamic", "longrope") for m in self.modules() if isinstance(m, RotaryEmbedding))

    @torch.inference_mode()
    def prefill_packed(self, chunks: List[List[int]], caches: List[KVCache]) -> torch.Tensor:
        """Prefill the next chunk of several requests in one forward (speed plan P4.3): chunks[i] continues caches[i]
        (filled in place). Returns the logits of each chunk's last token [len(chunks), V]. Check can_pack_prefill
        first."""
        dev = self.embed_tokens.weight.device
        if dev.type == "cuda" and not self.__dict__.get("_projections_fused"):
            self.fuse_projections()
        lens = [len(x) for x in chunks]
        for c, n in zip(caches, lens):
            c.reserve(c.length + n)
        packed = PackedPrefill(caches, lens, dev)
        ids = torch.tensor([[t for x in chunks for t in x]], dtype=torch.long, device=dev)
        logits = self._body(ids, 0, None, None, packed=packed)[0]
        for c, n in zip(caches, lens):
            c.length += n
        return logits

    def _prepare_marlin(self, dtype: torch.dtype) -> None:
        """Build the Marlin copy (turing.kernels.marlin) of every 4-bit layer a CUDA forward uses: the fused q/k/v and
        gate/up layers and the unfused rest. QuantLinear would build them on first use, after the KV pool has taken the
        free memory, where a 7B model's copies may not fit."""
        from .quant import QuantLinear
        from ..kernels.marlin import HAS_MARLIN
        if not HAS_MARLIN or not any(isinstance(m, QuantLinear) for m in self.modules()):
            return
        if not self.__dict__.get("_projections_fused"):
            self.fuse_projections()
        used, parts = [], set()
        for owner in self.modules():
            for key in ("_qkv", "_gate_up"):
                f = owner.__dict__.get(key)
                if f is not None:  # (fused layer, its parts, sizes): see fuse_linears
                    used.append(f[0])
                    parts.update(id(p) for p in f[1])
        used += [m for m in self.modules() if isinstance(m, QuantLinear) and id(m) not in parts]
        for m in used:
            if isinstance(m, QuantLinear):
                m._marlin_weights(dtype)

    def fuse_projections(self) -> int:
        """Fuse every layer's q/k/v and gate/up projections (fuse_linears); returns how many groups were fused."""
        n = 0
        for layer in self.layers:
            a = layer.self_attn
            n += fuse_linears(a, "_qkv", (a.q_proj, a.k_proj, a.v_proj))
            m = layer.mlp
            if isinstance(m, SubspaceMLP) and m.gate_proj is not None:
                n += fuse_linears(m, "_gate_up", (m.gate_proj, m.up_proj))
        self.__dict__["_projections_fused"] = True
        return n

    def _forward(self, input_ids, past_key_values, logits_to_keep, lora):
        if input_ids.is_cuda and not self.__dict__.get("_projections_fused") \
                and not torch.cuda.is_current_stream_capturing():
            self.fuse_projections()  # one GEMM per shared input on CUDA, where a launch costs as much as a small GEMM
        batch, seq_len = input_ids.shape
        cache, step = None, None
        if isinstance(past_key_values, list):
            if not all(isinstance(c, KVCache) for c in past_key_values):
                raise TypeError("past_key_values must be a KVCache or a list of KVCache; (k, v) tuples are not supported")
            if seq_len != 1 or batch != len(past_key_values):
                raise ValueError("a list of KVCache takes one decode token per request: input_ids [len(caches), 1]")
            step = past_key_values
        elif past_key_values is None and batch > 1:
            pass  # several sequences and no caches: a stateless forward, no cache kept (returns None)
        else:
            if batch != 1:
                raise ValueError("a KVCache holds one request: input_ids [1, L] (batch decode takes a list of caches)")
            cache = past_key_values if past_key_values is not None else KVCache(self.kv_pool())
            if seq_len == 1 and cache.length > 0 and cache.pool.device.type == "cuda":
                step = [cache]  # a single decode token on CUDA takes the fused path too
        returned = past_key_values if cache is None else cache  # the list of caches, or the cache
        if step is not None:
            for c in step:
                c.reserve(c.length + 1)
            if self._graphs.eligible(self, step, lora):
                logits = self._graphs.run(self, input_ids, step)
                for c in step:
                    c.length += 1
                return logits, returned
            step = DecodeStep.of(step, input_ids.device)
            start_pos = step.starts
        elif cache is not None:
            cache.reserve(cache.length + seq_len)
            start_pos = cache.length
        else:
            start_pos = 0
        logits = self._body(input_ids, start_pos, cache, step, logits_to_keep)
        if step is not None:
            for c in step.caches:
                c.length += 1
        elif cache is not None:
            cache.length += seq_len
        return logits, returned

    def _body(self, input_ids, start_pos, cache, step, logits_to_keep: int = 0, packed=None) -> torch.Tensor:
        """Embedding, layers, norm and head for one forward (what a CUDA graph captures for a decode step)."""
        cfg = self.config
        hidden_states = self.embed_tokens(input_ids)
        if cfg.embed_proj_dim:
            hidden_states = self.project_in(hidden_states)
        if cfg.embed_scale is not None:  # Gemma: in the activations' dtype, as HF rounds it
            hidden_states = hidden_states * torch.tensor(cfg.embed_scale, dtype=hidden_states.dtype)
        if cfg.learned_positions:
            if step is not None:  # [B] lengths before this token, on the device
                pos = step.starts_t.long()[:, None]
            elif packed is not None:
                pos = packed.positions[None]
            else:
                pos = torch.arange(start_pos, start_pos + input_ids.shape[1], device=input_ids.device)[None]
            hidden_states = hidden_states + self.embed_positions(pos + cfg.position_offset)

        shallow_state = None
        # each layer's residual add runs inside the next layer's input norm (or the final norm): pre-norm RMSNorm models
        defer = (not cfg.enable_recirculation and cfg.pre_norm and not cfg.parallel_residual and not cfg.sandwich_norm
                 and isinstance(self.norm, RMSNorm) and self.norm.fusable(hidden_states)
                 and all(isinstance(layer.input_layernorm, RMSNorm) for layer in self.layers))
        kw = dict(start_pos=start_pos, step=step) if step is not None else dict(packed=packed) if packed is not None \
            else dict(start_pos=start_pos, cache=cache)
        pending = None

        for i, layer in enumerate(self.layers):
            # Capture shallow state at router boundary
            if self.config.enable_recirculation and i == self.router_layer_idx:
                shallow_state = hidden_states.clone()

            hidden_states, out, _ = layer(hidden_states, pending=pending, defer=defer, **kw)
            pending = out if defer else None

            # Apply deep-to-shallow recurrence at the penultimate layer if state captured
            if self.config.enable_recirculation and i == len(self.layers) - 2 and shallow_state is not None:
                hidden_states = self.recirculation(hidden_states, shallow_state)

        if packed is not None:  # each chunk's last token
            hidden_states = hidden_states[:, packed.last]
            pending = pending[:, packed.last] if pending is not None else None
        elif logits_to_keep:
            hidden_states = hidden_states[:, -logits_to_keep:]
            pending = pending[:, -logits_to_keep:] if pending is not None else None
        if pending is not None:  # the last layer's residual add, fused with the final norm
            hidden_states, _ = fused_ops.rms_norm(pending, self.norm.weight, self.norm.eps, self.norm.offset,
                                                  residual=hidden_states)
        elif self.norm is not None:
            hidden_states = self.norm(hidden_states)
        if cfg.embed_proj_dim:
            hidden_states = self.project_out(hidden_states)
        logits = self.lm_head(hidden_states)
        if cfg.final_softcap is not None:
            logits = torch.tanh(logits / cfg.final_softcap) * cfg.final_softcap
        return logits

    @torch.no_grad()
    def calibrate_kv_smoothing(self, n: int = 4, tokens: int = 256, seed: int = 0) -> None:
        """Measure each layer's per-channel max |k| (after RoPE) on n random token sequences and set the key smoothing
        scale s = channel max / that head's median channel max, rounded to a power of two, where it is at least 16
        (1 elsewhere): only outlier channels are scaled, and q.k is unchanged bit for bit; it changes what a 4-bit
        KV cache can represent. Random tokens suffice:
        the outliers come from the projection biases and weights, not from the text."""
        dev = self.embed_tokens.weight.device
        attns = [layer.self_attn for layer in self.layers]
        for a in attns:
            a.k_smooth = a.q_smooth = None
            a._k_absmax = torch.zeros(a.num_kv_heads, a.head_dim, device=dev)
        ids = torch.randint(0, self.config.vocab_size, (n, tokens), generator=torch.Generator().manual_seed(seed))
        try:
            with torch.inference_mode(False):
                self._body(ids.to(dev), 0, None, None)  # stateless batched forward: no KV pool involved
        finally:
            for a in attns:
                m, a._k_absmax = a._k_absmax, None
                ref = m.median(dim=-1, keepdim=True).values.clamp_min(1e-6)
                # shrink outliers only (equalising all channels measured worse), by powers of two: q * s and k / s are
                # then exact in any float format, so q.k is bit-identical (in bf16, rounding q * s and k / s on the
                # outlier channels added logit noise: int8 KV fell from 57% to 40% on GSM8K, Qwen2.5-1.5B-Instruct)
                scale = torch.exp2(torch.round(torch.log2((m / ref).clamp_min(1.0))))
                # only real outliers (Qwen2's bias channels reach 64x; SmolLM2's largest is 8x): smoothing milder
                # channels raised SmolLM2's int4 cost from +0.025 to +0.05 nats
                scale = torch.where(scale >= 16, scale, torch.ones_like(scale))
                a.k_smooth = scale[None, :, None, :].to(self.embed_tokens.weight.dtype)
                a.q_smooth = scale.repeat_interleave(a.num_kv_groups, dim=0)[None, :, None, :].to(a.k_smooth.dtype)
        self._kv_smoothed = True

    @torch.no_grad()
    def select_kv_native_layers(self, threshold: float = 0.02, n: int = 4, tokens: int = 256, seed: int = 0) -> tuple:
        """Choose the layers a low-bit KV pool keeps in the model dtype: for each layer, the mean KL divergence of the
        next-token distribution when only that layer's K and V take the pool's int8/int4 round trip, on the random
        token sequences calibrate_kv_smoothing uses; layers above `threshold` nats are kept (at most a quarter of the
        layers, worst first). Measured: Qwen2.5-1.5B int4 layer 0 0.13 (median layer 0.0006), and on WikiText-2 that
        layer alone cost +1.77 nats; no SmolLM2-360M layer above 0.002. Sets config.kv_native_layers."""
        cfg = self.config
        dev = self.embed_tokens.weight.device
        attns = [layer.self_attn for layer in self.layers]
        ids = torch.randint(0, cfg.vocab_size, (n, tokens), generator=torch.Generator().manual_seed(seed)).to(dev)

        def logprobs():
            return [F.log_softmax(self._body(ids[i:i + 1], 0, None, None).float(), -1) for i in range(n)]
        with torch.inference_mode(False):
            ref = logprobs()
            kls = []
            for a in attns:
                a._kv_roundtrip = (8, a.head_dim) if cfg.kv_cache_dtype == "int8" else (4, INT4_GROUP)
                try:
                    lp = logprobs()
                finally:
                    a._kv_roundtrip = None
                kls.append(sum((r.exp() * (r - x)).sum(-1).mean().item() for r, x in zip(ref, lp)) / n)
        worst = sorted(range(len(attns)), key=lambda i: -kls[i])[:max(1, len(attns) // 4)]
        cfg.kv_native_layers = tuple(sorted(i for i in worst if kls[i] > threshold))
        self._kv_native_auto = cfg.kv_cache_dtype  # chosen for this storage; a different one chooses again
        self._kv_layer_kl = kls
        return cfg.kv_native_layers

    def _kv_calibration_file(self) -> Optional[str]:
        """Where this model's KV calibration is kept between processes (speed plan P7.2: it was most of a new process's
        time to first token): ~/.cache/turing/kv_calibration (TURING_CACHE_DIR overrides; "off" disables), one file
        per model config, weight fingerprint, dtype, device type and KV storage."""
        import hashlib
        import json
        import os
        root = os.environ.get("TURING_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "turing"))
        if root == "off":
            return None
        cfg, w = self.config, self.embed_tokens.weight
        h = hashlib.sha256(json.dumps({k: v for k, v in vars(cfg).items() if k not in ("kv_native_layers",)},
                                      sort_keys=True, default=str).encode())
        h.update(f"{w.dtype}|{w.device.type}".encode())
        a0, a1 = self.layers[0].self_attn, self.layers[-1].self_attn
        for t in (w[:8], getattr(a0.k_proj, "qweight", None), getattr(a0.k_proj, "weight", None),
                  getattr(a1.v_proj, "qweight", None), getattr(a1.v_proj, "weight", None)):
            if t is not None:  # a fingerprint of the weights the calibration depends on, not all of them
                h.update(t.detach().reshape(-1)[:1 << 16].float().cpu().numpy().tobytes())
        return os.path.join(root, "kv_calibration", h.hexdigest()[:32] + ".json")

    def _load_kv_calibration(self, smooth: bool, choose: bool) -> bool:
        import json
        path = self._kv_calibration_file()
        try:
            with open(path) as f:
                rec = json.load(f)
        except (TypeError, OSError, ValueError):
            return False
        if (smooth and rec.get("k_smooth") is None) or (choose and rec.get("native_layers") is None):
            return False
        dtype = self.embed_tokens.weight.dtype
        if smooth:
            for a, sc in zip((layer.self_attn for layer in self.layers), rec["k_smooth"]):
                scale = torch.tensor(sc, device=self.embed_tokens.weight.device)
                a.k_smooth = scale[None, :, None, :].to(dtype)
                a.q_smooth = scale.repeat_interleave(a.num_kv_groups, dim=0)[None, :, None, :].to(dtype)
            self._kv_smoothed = True
        if choose:
            self.config.kv_native_layers = tuple(rec["native_layers"])
            self._kv_native_auto = self.config.kv_cache_dtype
            self._kv_layer_kl = rec.get("layer_kl")
        return True

    def _save_kv_calibration(self, smooth: bool, choose: bool) -> None:
        import json
        import os
        path = self._kv_calibration_file()
        if path is None:
            return
        rec = {}
        try:
            with open(path) as f:
                rec = json.load(f)
        except (OSError, ValueError):
            pass
        attns = [layer.self_attn for layer in self.layers]
        if smooth and all(a.k_smooth is not None for a in attns):
            rec["k_smooth"] = [a.k_smooth[0, :, 0, :].float().cpu().tolist() for a in attns]
        if choose and self.config.kv_native_layers is not None:
            rec.update(native_layers=list(self.config.kv_native_layers), layer_kl=getattr(self, "_kv_layer_kl", None))
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(rec, f)
            os.replace(tmp, path)  # atomic: a concurrent process reads the old file or the new one
        except OSError:
            pass  # a read-only home: calibrate again next time

    def rope_switch_length(self) -> Optional[int]:
        """LongRoPE's original context, or None for other RoPE types. A sequence longer than it uses the long
        frequencies at every position; transformers recomputes the cache when generation crosses it."""
        from ..core.rope import RotaryEmbedding
        lens = {m._original for m in self.modules() if isinstance(m, RotaryEmbedding) and m.kind == "longrope"}
        return min(lens) if lens else None

    def kv_pool(self) -> KVPool:
        """This model's paged KV pool, created on first use (and again if the device, dtype or cache settings
        changed; caches of an older pool keep it alive)."""
        w, cfg = self.embed_tokens.weight, self.config
        pool = getattr(self, "_kv_pool", None)
        auto = getattr(self, "_kv_native_auto", None)
        if auto is not None and auto != cfg.kv_cache_dtype:
            cfg.kv_native_layers, self._kv_native_auto = None, None  # chosen for another storage: choose again
        if (pool is None or pool.device != w.device or pool.dtype != w.dtype
                or pool.kv_cache_dtype != cfg.kv_cache_dtype or pool.track_pages != (cfg.kv_page_budget < 1.0)
                or (cfg.kv_cache_dtype != "native" and (cfg.kv_native_layers is None
                                                        or pool.native_layers != tuple(cfg.kv_native_layers)))):
            self._kv_pool = None  # drop ours (and graphs captured on it) first, so its memory can back the new pool
            self._graphs.graphs.clear()
            gc.collect()
            if w.device.type == "cuda":
                torch.cuda.empty_cache()
            # int4 only: int8 needs no smoothing (+0.003 nats without it on Qwen2.5-1.5B-Instruct in bf16, +0.017 and
            # 16 GSM8K points lost with it)
            smooth = cfg.kv_cache_dtype == "int4" and cfg.kv_smoothing and not getattr(self, "_kv_smoothed", False)
            choose = cfg.kv_cache_dtype != "native" and cfg.kv_native_layers is None
            if (smooth or choose) and not self._load_kv_calibration(smooth, choose):
                if smooth:
                    self.calibrate_kv_smoothing()
                if choose:
                    self.select_kv_native_layers()
                self._save_kv_calibration(smooth, choose)
            if w.device.type == "cuda":
                self._prepare_marlin(w.dtype)  # before the pool takes the free memory
            pool = KVPool.for_model(cfg, w.dtype, w.device)
            self._kv_pool = pool
        return pool

    @torch.inference_mode()
    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_k: int = 50,
        adapter: Optional[str] = None
    ) -> List[int]:
        """
        Stateful autoregressive token generation loop with optional multi-tenant LoRA routing.
        """
        device = next(self.parameters()).device
        output_ids = list(prompt_token_ids)

        cur_input = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
        past_kv = None
        start_pos = 0

        for _ in range(max_new_tokens):
            logits, past_kv = self.forward(
                cur_input,
                past_key_values=past_kv,
                adapters=[adapter] if adapter else None
            )

            next_token_logits = logits[:, -1, :]
            if temperature > 0:
                scaled_logits = torch.nan_to_num(next_token_logits / max(temperature, 1e-4), nan=0.0, posinf=1e4, neginf=-1e4)
                probs = F.softmax(scaled_logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=1e-8).clamp(min=1e-8)
                if top_k > 0:
                    topk_probs, topk_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
                    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    idx = torch.multinomial(topk_probs, num_samples=1)
                    next_token = torch.gather(topk_indices, -1, idx).item()
                else:
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).item()

            output_ids.append(next_token)
            cur_input = torch.tensor([[next_token]], dtype=torch.long, device=device)

        return output_ids
