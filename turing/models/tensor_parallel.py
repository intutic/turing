"""
Tensor parallelism (Megatron style) and pipeline stages for a SubspaceCausalLM, applied to an existing model in place.

Tensor parallelism (`shard_tensor_parallel`): every rank of a TP group holds a slice of each layer.
- Attention: q_proj, k_proj, v_proj column-parallel by heads (each rank its num_heads / tp query heads and their KV
  heads; when num_kv_heads % tp != 0 each rank keeps a copy of the one KV head its query heads share), o_proj
  row-parallel followed by an all-reduce. Each rank's KV pool holds only its KV heads.
- MLP: gate_proj and up_proj column-parallel, down_proj row-parallel + all-reduce.
- MoE: every expert's intermediate dimension (and the shared expert's) is split across ranks; the router is
  replicated, so every rank picks the same experts, and the block's output is all-reduced once.
- Embedding replicated; lm_head vocab-parallel (all-gathered logits) when vocab % tp == 0, else replicated.

Pipeline stages (`keep_pipeline_stage`): a rank keeps a contiguous range of layers, renumbered so its KV pool holds
only those layers; the first stage keeps the embedding, the last the final norm and head. `stage_forward` runs one
stage over a request's cache (or a decode step's list of caches).
"""

import copy
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .causal_lm import DecodeStep, SparseMoE, SubspaceAttention, SubspaceCausalLM, SubspaceMLP
from ..core.kv_cache import KVCache

__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "shard_tensor_parallel",
    "keep_pipeline_stage",
    "stage_forward",
    "stage_layer_range",
]


class ColumnParallelLinear(nn.Module):
    """Output features [lo, hi) of a Linear (its rows of weight and bias). gather=True all-gathers the slices of every
    rank of `group` along the last dimension (a vocab-parallel lm_head)."""

    def __init__(self, linear: nn.Linear, lo: int, hi: int, group=None, gather: bool = False):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.data[lo:hi].clone(), requires_grad=False)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.data[lo:hi].clone(), requires_grad=False)
        self.group, self.gather = group, gather

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if not self.gather:
            return out
        parts = [torch.empty_like(out) for _ in range(dist.get_world_size(self.group))]
        dist.all_gather(parts, out.contiguous(), group=self.group)
        return torch.cat(parts, dim=-1)


class RowParallelLinear(nn.Module):
    """Input features [lo, hi) of a Linear (its columns of weight). reduce=True all-reduces the partial products over
    `group` and then adds the full bias; reduce=False leaves the sum to the caller (the bias is added on TP rank 0
    only, so the caller's sum counts it once)."""

    def __init__(self, linear: nn.Linear, lo: int, hi: int, group=None, reduce: bool = True, first: bool = True):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.data[:, lo:hi].clone(), requires_grad=False)
        keep_bias = linear.bias is not None and (reduce or first)
        self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False) if keep_bias else None
        self.group, self.reduce = group, reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight)
        if self.reduce:
            dist.all_reduce(out, group=self.group)
        return out if self.bias is None else out + self.bias


class _AllReduce(nn.Module):
    """Sums a module's output over a TP group (an MoE block whose experts each return a partial sum)."""

    def __init__(self, inner: nn.Module, group):
        super().__init__()
        self.inner, self.group = inner, group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x).contiguous()
        dist.all_reduce(out, group=self.group)
        return out


def _split(n: int, rank: int, world: int, what: str):
    if n % world:
        raise ValueError(f"tensor parallelism over {world} ranks needs {what} ({n}) divisible by {world}")
    size = n // world
    return rank * size, (rank + 1) * size


def _shard_mlp(mlp: SubspaceMLP, rank: int, world: int, group, reduce: bool) -> None:
    lo, hi = _split(mlp.up_proj.out_features, rank, world, "the MLP intermediate size")
    if mlp.gate_proj is not None:
        mlp.gate_proj = ColumnParallelLinear(mlp.gate_proj, lo, hi)
    mlp.up_proj = ColumnParallelLinear(mlp.up_proj, lo, hi)
    mlp.down_proj = RowParallelLinear(mlp.down_proj, lo, hi, group, reduce=reduce, first=rank == 0)
    mlp.ffn_dim = hi - lo


def _shard_moe(moe: SparseMoE, rank: int, world: int, group) -> None:
    if moe.gate_up is None or moe._host is not None:
        raise ValueError("tensor parallelism needs every fp expert on the device; this model's experts are packed in "
                         "4 bits or offloaded")
    ffn = moe.down.shape[2]
    lo, hi = _split(ffn, rank, world, "the expert intermediate size")
    gate_up = moe.gate_up.data  # [E, 2F, H]: gate rows then up rows
    moe.gate_up = nn.Parameter(torch.cat([gate_up[:, lo:hi], gate_up[:, ffn + lo:ffn + hi]], dim=1).clone(),
                               requires_grad=False)
    moe.down = nn.Parameter(moe.down.data[:, :, lo:hi].clone(), requires_grad=False)
    if moe.shared_expert is not None:  # sigmoid(gate(x)) * shared(x) is linear in shared(x): one reduce covers it
        _shard_mlp(moe.shared_expert, rank, world, group, reduce=False)


def _shard_attention(attn: SubspaceAttention, rank: int, world: int, group) -> None:
    H, Hkv, D = attn.num_heads, attn.num_kv_heads, attn.head_dim
    if getattr(attn.config, "qk_norm_full", False):
        raise NotImplementedError("tensor parallelism does not support a q/k norm over all heads (OLMoE)")
    q_lo, q_hi = _split(H, rank, world, "num_heads")
    groups = H // Hkv  # query heads per KV head
    hq = q_hi - q_lo
    if hq % groups and groups % hq:
        raise ValueError(f"{world} ranks split the {H} query heads unevenly over the {Hkv} KV heads")
    kv_lo, kv_hi = q_lo // groups, (q_hi - 1) // groups + 1  # a whole share of KV heads, or one copied head
    attn.q_proj = ColumnParallelLinear(attn.q_proj, q_lo * D, q_hi * D)
    attn.k_proj = ColumnParallelLinear(attn.k_proj, kv_lo * D, kv_hi * D)
    attn.v_proj = ColumnParallelLinear(attn.v_proj, kv_lo * D, kv_hi * D)
    attn.o_proj = RowParallelLinear(attn.o_proj, q_lo * D, q_hi * D, group)
    attn.num_heads, attn.num_kv_heads = hq, kv_hi - kv_lo
    attn.num_kv_groups = hq // attn.num_kv_heads


def _own_config(model: SubspaceCausalLM):
    """Give the model (and every submodule sharing its config) a private copy, so per-rank head and layer counts size
    this rank's KV pool without touching the caller's config."""
    old, cfg = model.config, copy.deepcopy(model.config)
    for m in model.modules():
        if getattr(m, "config", None) is old:
            m.config = cfg
    cfg.cuda_graphs = False  # decode-step graphs are not captured around collectives or stage boundaries
    model._kv_pool = None
    return cfg


def shard_tensor_parallel(model: SubspaceCausalLM, rank: int, world: int, group=None) -> SubspaceCausalLM:
    """Keep this rank's slice of every layer (see the module docstring). Every rank must call it on the same full
    model; collectives run over `group` (None = the default group)."""
    if world == 1:
        return model
    cfg = _own_config(model)
    for layer in model.layers:
        _shard_attention(layer.self_attn, rank, world, group)
        if isinstance(layer.mlp, SparseMoE):
            _shard_moe(layer.mlp, rank, world, group)
            layer.mlp = _AllReduce(layer.mlp, group)
        else:
            _shard_mlp(layer.mlp, rank, world, group, reduce=True)
    attn = model.layers[0].self_attn
    cfg.num_heads, cfg.num_kv_heads = attn.num_heads, attn.num_kv_heads
    if model.lm_head.out_features % world == 0:
        lo, hi = _split(model.lm_head.out_features, rank, world, "vocab_size")
        model.lm_head = ColumnParallelLinear(model.lm_head, lo, hi, group, gather=True)
    return model


def stage_layer_range(num_layers: int, stage: int, num_stages: int):
    """Contiguous layers [lo, hi) of a stage; the first num_layers % num_stages stages take one extra."""
    if num_layers < num_stages:
        raise ValueError(f"{num_stages} pipeline stages need at least {num_stages} layers, the model has {num_layers}")
    base, extra = divmod(num_layers, num_stages)
    lo = stage * base + min(stage, extra)
    return lo, lo + base + (stage < extra)


def keep_pipeline_stage(model: SubspaceCausalLM, stage: int, num_stages: int) -> SubspaceCausalLM:
    """Drop every layer outside this stage's range and renumber the rest from 0, so the stage's KV pool holds only its
    layers. Non-first stages keep a one-row embedding (only so model.kv_pool() knows the dtype and device);
    non-last stages drop the final norm and head."""
    if num_stages == 1:
        return model
    if model.config.enable_recirculation:
        raise NotImplementedError("pipeline parallelism does not support recirculation (it spans stages)")
    cfg = _own_config(model)
    lo, hi = stage_layer_range(len(model.layers), stage, num_stages)
    model.layers = nn.ModuleList(model.layers[lo:hi])
    for i, layer in enumerate(model.layers):
        layer.layer_idx = layer.self_attn.layer_idx = i  # the pool layer it reads and writes
    cfg.num_layers = hi - lo
    w = model.embed_tokens.weight
    if stage > 0:
        model.embed_tokens = nn.Embedding(1, w.shape[1], device=w.device, dtype=w.dtype)
        if cfg.learned_positions:
            del model.embed_positions
    if stage < num_stages - 1:
        model.norm = model.lm_head = None
    return model


def _embed(model: SubspaceCausalLM, ids: torch.Tensor, start_pos, step: Optional[DecodeStep]) -> torch.Tensor:
    """The embedding part of SubspaceCausalLM._body."""
    cfg = model.config
    h = model.embed_tokens(ids)
    if cfg.embed_proj_dim:
        h = model.project_in(h)
    if cfg.embed_scale is not None:
        h = h * torch.tensor(cfg.embed_scale, dtype=h.dtype)
    if cfg.learned_positions:
        if step is not None:
            pos = step.starts_t.long()[:, None]
        else:
            pos = torch.arange(start_pos, start_pos + ids.shape[1], device=ids.device)[None]
        h = h + model.embed_positions(pos + cfg.position_offset)
    return h


def _head(model: SubspaceCausalLM, h: torch.Tensor, logits_to_keep: int) -> torch.Tensor:
    """The final norm and head part of SubspaceCausalLM._body."""
    cfg = model.config
    if logits_to_keep:
        h = h[:, -logits_to_keep:]
    if model.norm is not None:
        h = model.norm(h)
    if cfg.embed_proj_dim:
        h = model.project_out(h)
    logits = model.lm_head(h)
    if cfg.final_softcap is not None:
        logits = torch.tanh(logits / cfg.final_softcap) * cfg.final_softcap
    return logits


def stage_forward(model: SubspaceCausalLM, x: torch.Tensor, caches: Union[KVCache, List[KVCache]], first: bool,
                  last: bool, logits_to_keep: int = 0) -> torch.Tensor:
    """One pipeline stage for one request (caches a KVCache, x [1, L]) or one decode step (caches a list, x [B, 1]):
    x is token ids on the first stage and the previous stage's hidden states [B, L, hidden] otherwise. Writes this
    stage's layers into the caches and advances their lengths. Returns hidden states, or logits on the last stage."""
    rows = caches if isinstance(caches, list) else [caches]
    n = x.shape[1]
    for c in rows:
        c.reserve(c.length + n)
    step = DecodeStep.of(rows, x.device) if isinstance(caches, list) else None
    start = step.starts if step is not None else rows[0].length
    h = _embed(model, x, start, step) if first else x
    for layer in model.layers:
        h = layer(h, start_pos=start, cache=None if step is not None else rows[0], step=step)[0]
    for c in rows:
        c.length += n
    return _head(model, h, logits_to_keep) if last else h
