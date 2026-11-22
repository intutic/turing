"""
Decoder-only causal LM (SubspaceCausalLM) covering the HuggingFace families the loader converts: Llama, Mistral, Qwen2/3,
Gemma 1/2/3, Phi-2/3, GPT-2, OPT, GPT-NeoX and the MoE families Mixtral, Qwen2/3-MoE and OLMoE. Each family's features
(norm type, MLP shape, biases, positions, partial rotary, sliding windows, soft-capping, MoE routing) come from
ModelConfig, and tests/test_hf_parity.py checks every family against HuggingFace's own logits.
"""

import math
import os
import time
from typing import List, Optional, Tuple, Union, Any
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..core.rope import RotaryEmbedding, apply_rotary_pos_emb
from ..core.kv_cache import PAGE, KVCache, KVPool, KVPoolExhausted, kv_roundtrip
try:  # the native extension: one packed 4-bit expert on the CPU (csrc/turing_w4_expert_cpu.hpp)
    from ..turing_csrc import w4_expert_cpu
except ImportError:  # a build without it: offload copies missing experts in, as before
    w4_expert_cpu = None
from ..kernels.triton_kv_decode import (pool_append_supported, pool_decode_supported, triton_pool_append,
                                        triton_pool_decode)
from ..kernels import triton_fused_ops as fused_ops
from ..kernels.triton_kv_prefill import pool_prefill_preferred, pool_prefill_supported, triton_pool_prefill
from ..kernels.flashinfer_prefill import flashinfer_prefill, flashinfer_prefill_supported
from ..kernels.triton_moe import HAS_TRITON as HAS_MOE_KERNELS
from ..kernels.triton_moe import (GEMM_MIN_ROWS, fused_moe, fused_moe_supported, fused_moe_w4,
                                  fused_moe_w4_supported, gated_add, gemm_moe, moe_route)


_CPU_THREADS = min(os.cpu_count() or 1, 16)  # for the CPU expert kernel


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
    Qwen2-MoE's sigmoid-gated shared expert. Expert weights are stacked: gate_up [E, 2F, H], down [E, H, F]. After
    pack_experts they are packed 4-bit buffers instead (turing.models.quant) and gate_up and down are None. After
    offload only some experts are on the device (see offload)."""

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
        self._host = None  # offload: every expert's arrays in host memory, by name
        self._cuts = None  # StepCuts while DecodeGraphs captures a step through this (offloaded) layer
        self._marlin = None  # use_marlin: (activation dtype, the experts in Marlin's layout)

    def route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(fp32 weights, expert ids), each [T, top_k], for x [T, H]."""
        logits = self.router(x)
        if logits.is_cuda and HAS_MOE_KERNELS and logits.dtype in (torch.float16, torch.bfloat16):
            return moe_route(logits, self.top_k, self.norm_topk_prob)  # one kernel for the six below
        probs = F.softmax(logits.float(), dim=-1)
        weights, experts = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights, experts

    def pack_experts(self, gate_up, down, group: int) -> None:
        """Keep the experts as packed 4-bit weights: gate_up and down are (packed codes, scales, zeros) with a
        leading expert dimension, as turing.models.quant produces them. The fp weights are dropped."""
        self.gate_up = self.down = None
        for name, (q, scales, zeros) in (("gate_up", gate_up), ("down", down)):
            for suffix, t in (("_q", q), ("_scales", scales), ("_zeros", zeros)):
                self.register_buffer(name + suffix, t)
        self.group = group

    def use_marlin(self, dtype: torch.dtype) -> bool:
        """Serve packed experts with the vendored Marlin MoE kernel (turing.kernels.marlin_moe) where it is built and
        the shapes fit: the experts are converted to Marlin's layout for activations of `dtype`, and the packed
        buffers move to host memory, so the device holds each expert once (they still save and convert from there).
        An offloaded layer converts its slots: the resident experts are then Marlin's, the packed slot arrays are
        dropped (host memory has every expert already), and an expert that takes a slot is converted on the way in."""
        from ..kernels.marlin_moe import marlin_moe_pack, marlin_moe_supported
        if self.gate_up is not None or self._marlin is not None:
            return self._marlin is not None
        q = self.gate_up_q
        if not marlin_moe_supported(q.shape[2] * 2, self.down_q.shape[2] * 2, self.group, dtype, q.device):
            return False
        self._marlin = (dtype, marlin_moe_pack((q, self.gate_up_scales, self.gate_up_zeros),
                                               (self.down_q, self.down_scales, self.down_zeros), dtype))
        for name in self._arrays():
            self._buffers[name] = None if self._host is not None else self._buffers[name].cpu()
        return True

    def _arrays(self) -> Tuple[str, ...]:
        """Names of the arrays that hold the experts (leading dimension: expert, or slot after offload)."""
        if self.gate_up is not None:
            return ("gate_up", "down")
        return tuple(w + part for w in ("gate_up", "down") for part in ("_q", "_scales", "_zeros"))

    def offload(self, slots: int) -> None:
        """Keep `slots` of this layer's experts on its device; all of them stay in host memory (pinned where CUDA is
        available) and one is copied into a slot when a decode step routes to it, replacing the expert used least
        recently. The device arrays keep their names with `slots` as their leading dimension; slot_of maps an expert
        to its slot (-1: on the host only). A decode step then syncs with the host once per layer, so it is not
        captured in a CUDA graph. Prefill-sized inputs, and a step that routes to more experts than there are slots,
        stream the missing experts through the device without changing which ones are resident. Packed 4-bit experts
        take another road for a few tokens (_cpu_experts): the missing ones run on the CPU, nothing is copied, and
        the residents become the experts used most (_rebalance)."""
        if self._host is not None:
            raise ValueError("this layer's experts are already offloaded")
        if self._marlin is not None:
            raise ValueError("offload before the experts are converted for the Marlin kernel (use_marlin)")
        if not 0 < slots < self.num_experts:
            raise ValueError(f"expert slots must be between 1 and {self.num_experts - 1}, got {slots}")
        pin, host = torch.cuda.is_available(), {}
        for name in self._arrays():
            t = getattr(self, name).detach()
            host[name] = t.cpu().pin_memory() if pin else t.cpu()
            first = host[name][:slots].to(t.device).clone()
            if name in self._parameters:
                self._parameters[name] = nn.Parameter(first, requires_grad=False)
            else:
                self._buffers[name] = first
        dev = self.router.weight.device
        self.register_buffer("slot_of", torch.cat([torch.arange(slots), torch.full((self.num_experts - slots,), -1)])
                             .to(dev), persistent=False)
        self._host, self.num_slots = host, slots
        self._expert_slot = list(range(slots)) + [-1] * (self.num_experts - slots)  # host-side copy of slot_of
        self._slot_expert, self._slot_used, self._tick = list(range(slots)), [0] * slots, 0
        self.offload_hits = self.offload_misses = self.offload_streamed = self.offload_cpu = 0
        self.offload_cpu_s = 0.0  # seconds in the CPU kernel
        self._uses = [0] * self.num_experts  # small steps that needed each expert (halved at every rebalance)
        self._small_steps = 0

    def _ensure(self, needed: List[int]) -> None:
        """Bring the experts in `needed` (at most num_slots of them) onto the device, each replacing the resident
        expert used least recently among those this step does not need."""
        self._tick += 1
        missing = []
        for e in needed:
            slot = self._expert_slot[e]
            if slot < 0:
                missing.append(e)
            else:
                self._slot_used[slot] = self._tick
        self.offload_hits += len(needed) - len(missing)
        self.offload_misses += len(missing)
        for e in missing:
            self._load(e, min((i for i in range(self.num_slots) if self._slot_used[i] != self._tick),
                              key=self._slot_used.__getitem__))

    def _load(self, e: int, slot: int) -> None:
        """Copy expert e from host memory into `slot`, replacing the expert there."""
        if self._marlin is not None:  # the slots are Marlin's: converted on the device, then stored
            from ..kernels.marlin_moe import marlin_moe_store
            p = {name: host[e].to(self.slot_of.device, non_blocking=True) for name, host in self._host.items()}
            marlin_moe_store(self._marlin[1], slot, (p["gate_up_q"], p["gate_up_scales"], p["gate_up_zeros"]),
                             (p["down_q"], p["down_scales"], p["down_zeros"]), self._marlin[0])
        else:
            for name, host in self._host.items():
                getattr(self, name)[slot].copy_(host[e], non_blocking=True)
        old = self._slot_expert[slot]
        self._expert_slot[old], self._expert_slot[e] = -1, slot
        self._slot_expert[slot], self._slot_used[slot] = e, self._tick
        self.slot_of[old], self.slot_of[e] = -1, slot

    # A missing packed expert costs a copy of all its weights to the device (92 MB for Mixtral-8x7B) or one pass over
    # them on the CPU (csrc/turing_w4_expert_cpu.hpp); for a few tokens the pass is cheaper and leaves the slots alone.
    CPU_ROWS = 8
    REBALANCE = 64  # steps of at most CPU_ROWS tokens between looks at which experts are used most

    def _cpu_ok(self) -> bool:
        return (self.gate_up is None and self.act is F.silu and w4_expert_cpu is not None
                and self.group <= 128 and self.group % 16 == 0)  # the CPU kernel's groups

    def _cpu_serves(self, x: torch.Tensor) -> bool:
        return x.shape[0] <= self.CPU_ROWS and x.is_cuda and self._cpu_ok()

    def cut_safe(self, x: torch.Tensor, rows: int) -> bool:
        """A decode step of `rows` rows through this offloaded layer can be captured as CUDA graphs with a stop for
        the host (StepCuts): its slots are Marlin's, and the CPU takes the experts that are not resident."""
        return self._host is not None and self._marlin_for(x) is not None and rows <= self.CPU_ROWS and self._cpu_ok()

    def between_graphs(self, site: list, rows: int, launch) -> None:
        """The host's part of a captured decode step at this layer, between the graph that made the routing and the
        one that adds this layer's output: which routed experts are not resident, `launch` (the graph with the
        resident ones), and the others on the CPU meanwhile, into the site's static tensor. site: [layer, x, weights,
        experts, extra, extra is not zero]; the first `rows` rows are requests, the rest padding."""
        _, x, weights, experts, extra, dirty = site
        en = experts[:rows].cpu().numpy()  # the wait for the graph before
        needed = np.unique(en).tolist()
        self._count(needed)
        missing = [e for e in needed if self._expert_slot[e] < 0]
        host = (x[:rows].float().cpu(), weights[:rows].cpu()) if missing else None
        launch()
        if missing:
            extra[:rows].copy_(self._cpu_experts(host[0], host[1], en, needed, missing))
            site[5] = True
        else:
            self.offload_hits += len(needed)
            if dirty:
                extra.zero_()
                site[5] = False

    def _count(self, needed: List[int]) -> None:
        """A small step's experts, hits and misses alike, towards which stay resident; every REBALANCE such steps
        the residency is looked at, before the step reads its slots."""
        for e in needed:
            self._uses[e] += 1
        self._small_steps += 1
        if self._small_steps % self.REBALANCE == 0:
            self._rebalance()

    def _cpu_experts(self, xs: torch.Tensor, w: torch.Tensor, ex: torch.Tensor, needed: List[int],
                     missing: List[int]) -> torch.Tensor:
        """The weighted outputs of the experts in `missing` for the tokens routed to them, computed on the CPU from
        the packed host arrays: [T, H] in fp32 on the host (zero rows for the other tokens). xs [T, H] fp32, w and ex
        [T, top_k] (fp32 routing weights, expert ids): host copies (tensors or arrays) of the step's input and
        routing."""
        self._tick += 1
        for e in needed:
            if self._expert_slot[e] >= 0:
                self._slot_used[self._expert_slot[e]] = self._tick
        self.offload_hits += len(needed) - len(missing)
        self.offload_cpu += len(missing)
        h = self.__dict__.get("_host_np")
        if h is None:  # the host arrays as the kernel takes them (fp16 scales as their bits), viewed once
            h = self._host_np = {k: v.numpy().view("uint16") if v.dtype == torch.float16 else v.numpy()
                                 for k, v in self._host.items()}
        # numpy from here on: torch's CPU nonzero opens an OpenMP region for any size, and its threads then spin for
        # milliseconds beside the kernel's (an expert took 5.4 ms instead of 2.6 on a 16-thread L4 host)
        xn, wn, en = (np.asarray(t) for t in (xs, w, ex))
        out, t = np.zeros_like(xn), time.perf_counter()
        for e in missing:
            rows, cols = np.nonzero(en == e)
            y = w4_expert_cpu(xn[rows], h["gate_up_q"][e], h["gate_up_scales"][e], h["gate_up_zeros"][e],
                              h["down_q"][e], h["down_scales"][e], h["down_zeros"][e], self.group, _CPU_THREADS)
            out[rows] += y * wn[rows, cols, None]  # a token is routed to an expert once: the rows are distinct
        self.offload_cpu_s += time.perf_counter() - t
        return torch.from_numpy(out)

    def _rebalance(self) -> None:
        """Make the most-used expert that is not resident resident, in place of the least-used resident one, when it
        was needed by half as many steps again (one copy); then halve the counts so they follow the traffic."""
        out = max((e for e in range(self.num_experts) if self._expert_slot[e] < 0), key=self._uses.__getitem__)
        slot = min(range(self.num_slots), key=lambda i: self._uses[self._slot_expert[i]])
        if self._uses[out] > 1.5 * self._uses[self._slot_expert[slot]] + 4:
            self._load(out, slot)
            self.offload_misses += 1
        self._uses = [u // 2 for u in self._uses]

    def dequant_numel(self) -> int:
        """Elements of one packed expert's two matrices once dequantized (the scratch buffer expert_weights fills)."""
        arrays = self._host if self._host is not None else self._buffers  # offloaded: the device may hold Marlin's only
        return 2 * (arrays["gate_up_q"][0].numel() + arrays["down_q"][0].numel())

    def expert_weights(self, e: int, dtype: torch.dtype, scratch: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expert e's (gate_up [2F, H], down [H, F]) in dtype, dequantized when the experts are packed. An offloaded
        expert that is not resident is copied to the device for this call only. With scratch, packed experts on CUDA
        are dequantized by one launch each into a buffer shared by every layer, valid until the next such call: the
        reference makes temporaries of three times a matrix (0.7 GiB for a Mixtral gate/up), which did not fit
        beside six of eight experts and the KV pool on a 24 GB card."""
        if self._host is None:
            parts = {name: getattr(self, name)[e] for name in self._arrays()}
        elif self._expert_slot[e] >= 0 and self._marlin is None:
            parts = {name: getattr(self, name)[self._expert_slot[e]] for name in self._host}
        else:  # not resident (or resident in Marlin's layout only): from host memory
            self.offload_streamed += 1
            dev = self.router.weight.device
            parts = {name: host[e].to(dev, non_blocking=True) for name, host in self._host.items()}
        if "gate_up" in parts:
            return parts["gate_up"].to(dtype), parts["down"].to(dtype)
        from .quant import _scratch, dequantize
        dev = self.router.weight.device  # after use_marlin the packed buffers are in host memory
        gq, dq = parts["gate_up_q"], parts["down_q"]
        if scratch and dev.type == "cuda" and gq.device == dev and dtype in (torch.float16, torch.bfloat16):
            from ..kernels.triton_w4a16 import HAS_TRITON, dequantize_cuda
            if HAS_TRITON:
                n = 2 * gq.numel()
                buf = _scratch(self.dequant_numel(), dev, dtype)
                return (dequantize_cuda(gq, parts["gate_up_scales"], parts["gate_up_zeros"],
                                        buf[:n].view(gq.shape[0], -1), self.group),
                        dequantize_cuda(dq, parts["down_scales"], parts["down_zeros"],
                                        buf[n:].view(dq.shape[0], -1), self.group))
        return (dequantize(gq, parts["gate_up_scales"], parts["gate_up_zeros"], dtype).to(dev),
                dequantize(dq, parts["down_scales"], parts["down_zeros"], dtype).to(dev))

    def _grouped(self, x: torch.Tensor) -> bool:
        """The grouped Triton kernels serve inputs like x on this device."""
        if self.gate_up is None:
            return fused_moe_w4_supported(x, self.gate_up_q, self.act, self.group)
        return fused_moe_supported(x, self.gate_up, self.act)

    def _marlin_for(self, x: torch.Tensor):
        """The Marlin copy of the experts if it serves inputs like x (use_marlin), else None."""
        return self._marlin[1] if self._marlin is not None and x.is_cuda and x.dtype == self._marlin[0] else None

    def graph_safe(self, x: torch.Tensor, tokens: int = 1) -> bool:
        """The forward for `tokens` inputs like x makes no host sync (it can go in a CUDA graph): not offloaded, and
        served by Marlin or by the grouped kernels (which hand over to a loop over experts at GEMM_MIN_ROWS rows
        per expert)."""
        return self._host is None and (self._marlin_for(x) is not None or (
            self._grouped(x) and tokens * self.top_k < GEMM_MIN_ROWS * self.num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        weights, experts = self.route(x)
        marlin = self._marlin_for(x)
        if marlin is not None:  # every input size: decode steps and prefill chunks
            from ..kernels.marlin_moe import marlin_moe
            cpu = stream = None
            if self._host is not None and self._cuts is not None:
                # a decode step being captured as graphs (StepCuts): nothing here waits for the host. The graph
                # before this point made x and the routing, the next holds the resident experts (the slots are
                # looked up there: the host may change them in between), and what the CPU computed is added after.
                cuts = self._cuts
                cuts.cut()
                slots = self.slot_of[experts]
                out = marlin_moe(x, marlin, weights.masked_fill(slots < 0, 0), slots.clamp(min=0), self.act)
                extra = cuts.cut(lambda: torch.zeros(x.shape, dtype=torch.float32, device=x.device))
                if cuts.capturing:
                    cuts.sites.append([self, x, weights, experts, extra, False])
                out = (out.float() + extra).to(out.dtype)
                if self.shared_expert is not None:
                    out = gated_add(out, self.shared_expert_gate(x), self.shared_expert(x))
                return out.view(shape)
            if self._host is not None:  # offloaded: the slots hold the resident experts; one host sync for the rest
                en = experts.cpu().numpy()  # the layer's wait for the device; a unique on the device costs more launches
                needed = np.unique(en).tolist()
                small = self._cpu_serves(x)
                if small:
                    self._count(needed)
                missing = [e for e in needed if self._expert_slot[e] < 0]
                if missing and small:  # host copies before the launch below (see the Triton path)
                    cpu = (x.float().cpu(), weights.cpu(), en, needed, missing)
                elif missing:  # a prefill chunk: the missing experts pass through the device, one at a time
                    stream = (weights, experts, missing)
                elif small:
                    self.offload_hits += len(needed)
                slots = self.slot_of[experts]
                weights, experts = weights.masked_fill(slots < 0, 0), slots.clamp(min=0)
            out = marlin_moe(x, marlin, weights, experts, self.act)
            if cpu is not None:
                out = (out.float() + self._cpu_experts(*cpu).to(out.device)).to(out.dtype)
            if stream is not None:
                rest = gemm_moe(x, stream[0], stream[1], self.num_experts,
                                lambda e: self.expert_weights(e, x.dtype, scratch=True), self.act, only=stream[2])
                out = (out.float() + rest.float()).to(out.dtype)
            if self.shared_expert is not None:
                out = gated_add(out, self.shared_expert_gate(x), self.shared_expert(x))
            return out.view(shape)
        if self._marlin is not None:
            raise RuntimeError(f"this layer's experts were converted for {self._marlin[0]} activations on CUDA "
                               f"(SparseMoE.use_marlin); got {x.dtype} on {x.device}")
        # the grouped kernels, up to GEMM_MIN_ROWS rows per expert (a large prefill chunk): then cuBLAS per expert
        ids, grouped = experts, experts.numel() < GEMM_MIN_ROWS * self.num_experts and self._grouped(x)
        extra = None
        if grouped and self._host is not None:  # offloaded: one host sync to see which experts this step needs
            en = experts.cpu().numpy()
            needed = np.unique(en).tolist()
            cpu = self._cpu_serves(x)
            if cpu:
                self._count(needed)
            missing = [e for e in needed if self._expert_slot[e] < 0]
            if missing and cpu:  # a few tokens: the missing experts on the CPU, the rest here
                # host copies now: after the launch below, a copy would wait for the device to finish it, and the
                # CPU is to work while the device runs the resident experts
                extra = (x.float().cpu(), weights.cpu(), en, needed, missing)
                slots = self.slot_of[experts]
                weights, ids = weights.masked_fill(slots < 0, 0), slots.clamp(min=0)
            else:
                grouped = len(needed) <= self.num_slots
                if grouped:
                    self._ensure(needed)
                    ids = self.slot_of[experts]
        if not grouped:
            out = gemm_moe(x, weights, experts, self.num_experts,
                           lambda e: self.expert_weights(e, x.dtype, scratch=True), self.act)
        elif self.gate_up is None:
            out = fused_moe_w4(x, (self.gate_up_q, self.gate_up_scales, self.gate_up_zeros),
                               (self.down_q, self.down_scales, self.down_zeros), weights, ids, self.group)
        else:  # no host sync unless offloaded, so a decode step with this layer can go in a CUDA graph
            out = fused_moe(x, self.gate_up, self.down, weights, ids)
        if extra is not None:
            out = (out.float() + self._cpu_experts(*extra).to(out.device)).to(out.dtype)
        if self.shared_expert is not None:
            out = gated_add(out, self.shared_expert_gate(x), self.shared_expert(x))
        return out.view(shape)


def append_chunk(pool: KVPool, layer: int, cache: KVCache, k: torch.Tensor, v: torch.Tensor) -> None:
    """KVPool.append; a chunk for a low-bit layer on CUDA is quantized by the append kernel."""
    if k.shape[2] > 1 and pool_append_supported(pool, layer):
        triton_pool_append(pool, layer, cache, k, v)
    else:
        pool.append(layer, cache, k, v)


def row_starts(caches: List[KVCache]) -> List[int]:
    """The position each row of a decode step writes: a cache's length, and for a cache listed several times the
    positions after it, in order. Listing a request's cache k + 1 times with its last token and k drafted tokens
    verifies the drafts inside an ordinary decode step: every row's key and value are appended before any row
    reads, and each row reads only up to its own position."""
    seen, starts = {}, []
    for c in caches:
        n = seen.get(id(c), 0)
        starts.append(c.length + n)
        seen[id(c)] = n + 1
    return starts


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
    def of(cls, caches: List[KVCache], device, starts: Optional[List[int]] = None) -> "DecodeStep":
        starts = row_starts(caches) if starts is None else starts
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

    def __init__(self, caches: List[KVCache], lens: List[int], device, riders: int = 0):
        self.pool, self.caches, self.lens = caches[0].pool, caches, lens
        self.starts = row_starts(caches)  # a rider's cache listed again: its drafts, at the positions after it
        self.offsets = [sum(lens[:i]) for i in range(len(lens))]
        self.positions = torch.cat([torch.arange(s, s + n) for s, n in zip(self.starts, lens)]).to(device)
        self.last = torch.tensor([o + n - 1 for o, n in zip(self.offsets, lens)], device=device)
        self._freqs = {}
        self.solo = None
        # the last `riders` chunks are decode rows (one token each): the decode kernel appends and attends for them
        self.riders = riders
        if riders:
            rows = torch.tensor([[c.slot for c in caches[-riders:]], self.starts[-riders:]], dtype=torch.int32)
            self.rider_slots, self.rider_starts = rows.to(device)
            # splits as a decode step of the next graph size up: no kernel build of its own for each row count
            size = next((b for b in DecodeGraphs.BUCKETS if b >= riders), riders)
            self.rider_splits = max(1, min(32, 256 // (size * self.pool.num_kv_heads)))

    @classmethod
    def fresh(cls, pool: KVPool, n: int, b: int, slots: torch.Tensor, positions: torch.Tensor, last: torch.Tensor,
              rows: Optional[torch.Tensor] = None) -> "PackedPrefill":
        """n chunks of b tokens each at positions 0 .. b - 1 with no cached prefix, then decode rows, described by
        device tensors only: slots [n * b] int32 (each token's page-table row), positions [n * b + rows] (the
        chunks' part is fixed, the decode rows' part is rows[1]), last (the indices whose logits are wanted) and
        rows [2, R] int32 (the decode rows' page-table rows and positions). Nothing in the forward then depends on
        a host value, which is what a CUDA graph needs (PrefillGraphs)."""
        self = cls.__new__(cls)
        self.pool, self._freqs, self.last, self.positions = pool, {}, last, positions
        self.solo = (slots, positions[:n * b].to(torch.int32), n, b)
        self.riders = 0 if rows is None else rows.shape[1]
        if self.riders:
            self.rider_slots, self.rider_starts = rows[0], rows[1]
            self.rider_splits = max(1, min(32, 256 // (self.riders * pool.num_kv_heads)))
        return self

    def freqs(self, rope) -> Tuple[torch.Tensor, torch.Tensor]:
        """cos, sin [T, R] at each token's position (frequencies that do not depend on the sequence length), computed
        from the positions as a decode step does: a table up to the longest sequence would be built, and cached,
        again for every length a riding request reaches (32 MiB each at 16K tokens)."""
        key = (rope.dim, rope.base_theta, rope.params)
        if key not in self._freqs:
            self._freqs[key] = rope.compute_freqs_device(self.positions)
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
        self._kv_roundtrip = None  # kv_roundtrip's arguments during select_kv_native_layers: K and V as a low-bit
                                   # pool would give them back

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
            k, v = kv_roundtrip(k, v, *self._kv_roundtrip)

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
        append_chunk(pool, layer, cache, k, v)
        if (start == 0 and 1 < seq_len <= 2048 and self.window is None and self.softcap is None
                and pool.layer_fields(layer, "k")[3] == "native"):
            # no cached prefix, and the layer stores what the chunk just computed: plain causal attention over its
            # own keys and values. One call instead of a paged kernel's plan and wrapper, which cost the host 3 ms
            # of a 512-token prompt's first token on an L4 for the same device time (measured up to 2,048 tokens,
            # the engine's chunk; longer ones keep the paged kernels, whose memory does not grow with the square).
            out = self._attend(q, k, v, None, causal=True)
            return out.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)
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
        """Each request's slice of the packed chunk: append its K/V, attend over its pages -> [1, T, H * D]. The
        decode rows at the end (packed.riders) take the decode kernel, as in a step of their own."""
        pool, layer, R = packed.pool, self.layer_idx, packed.riders
        if packed.solo is not None:
            # chunks of one padded length with no cached prefix (PrefillGraphs): append, then plain causal attention
            # over each chunk by itself, as a batch; the decode rows after them take the decode kernel
            slots, starts, n, b = packed.solo
            T = n * b
            if R:
                qr, kr, vr = (x[0, :, T:].transpose(0, 1)[:, :, None] for x in (q, k, v))  # [R, heads, 1, D]
                rode = triton_pool_decode(pool, layer, qr, kr, vr, packed.rider_slots, packed.rider_starts,
                                          scale=self.scale, window=self.window or 0, softcap=self.softcap or 0.0,
                                          n_splits=packed.rider_splits)
                q, k, v = q[:, :, :T], k[:, :, :T], v[:, :, :T]
            triton_pool_append(pool, layer, None, k, v, slots, starts)
            storage = pool.layer_fields(layer, "k")[3]
            if storage != "native":  # attend over what the layer's pages give back, as the eager forward reads them
                k, v = kv_roundtrip(k, v, storage, pool.fp8_scales[layer] if storage == "fp8" else (1.0, 1.0))
            qc, kc, vc = (x[0].unflatten(1, (n, b)).transpose(0, 1) for x in (q, k, v))  # [n, heads, b, D]
            out = self._attend(qc, kc, vc, None, causal=True).transpose(1, 2).reshape(1, T, self.num_heads * self.head_dim)
            if R:
                out = torch.cat([out, rode.transpose(1, 2).reshape(1, R, self.num_heads * self.head_dim)], dim=1)
            return out
        n = len(packed.caches) - R
        caches, offsets, lens, starts = packed.caches[:n], packed.offsets[:n], packed.lens[:n], packed.starts[:n]
        outs = []
        if R:
            T = q.shape[2] - R
            if pool.layer_fields(layer, "k")[3] in ("native", "fp8") \
                    and flashinfer_prefill_supported(pool, layer, q, self.window, self.softcap, min_chunk=1):
                # pages FlashInfer reads in place: the decode rows as requests of one query each, in a call of
                # their own (in the chunks' call they would set its tile size: 1.66 ms a row at 16K tokens on
                # Qwen2.5-1.5B, against 1.05 from the decode kernel)
                triton_pool_append(pool, layer, None, k[:, :, T:], v[:, :, T:], packed.rider_slots, packed.rider_starts)
                rode = flashinfer_prefill(pool, layer, q[:, :, T:], [c.slot for c in packed.caches[n:]],
                                          packed.starts[n:], [1] * R, self.scale, tag="rows")
            else:
                qr, kr, vr = (x[0, :, T:].transpose(0, 1)[:, :, None] for x in (q, k, v))  # [R, heads, 1, D]
                rode = triton_pool_decode(pool, layer, qr, kr, vr, packed.rider_slots, packed.rider_starts,
                                          scale=self.scale, window=self.window or 0, softcap=self.softcap or 0.0,
                                          n_splits=packed.rider_splits).transpose(1, 2).reshape(
                                              1, R, self.num_heads * self.head_dim)
            q, k, v = q[:, :, :T], k[:, :, :T], v[:, :, :T]
        if n and flashinfer_prefill_supported(pool, layer, q, self.window, self.softcap):  # every request in one call
            for c, off, m in zip(caches, offsets, lens):
                append_chunk(pool, layer, c, k[:, :, off:off + m], v[:, :, off:off + m])
            outs.append(flashinfer_prefill(pool, layer, q, [c.slot for c in caches], starts, lens, self.scale))
        else:
            for c, off, m in zip(caches, offsets, lens):  # starts[i] is c.length
                sl = slice(off, off + m)
                outs.append(self._attend_one(q[:, :, sl], k[:, :, sl], v[:, :, sl], c))
        if R:
            outs.append(rode)
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)

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

class StepCuts:
    """A decode step captured as consecutive CUDA graphs sharing one memory pool, cut where a layer needs the host:
    an offloaded MoE layer reads its routing there, launches the graph with its resident experts and runs the others
    on the CPU meanwhile (SparseMoE.between_graphs). Replayed in capture order by DecodeGraphs.run. Run eagerly, such
    a step is bound by the host: 32 layers of Mixtral-8x7B took 54 ms a token on an L4 host besides the CPU's
    experts, for about 30 ms of device time."""

    def __init__(self):
        self.graphs, self.sites, self.pool, self._live = [], [], torch.cuda.graph_pool_handle(), None

    @property
    def capturing(self) -> bool:
        return self._live is not None

    def begin(self) -> None:
        graph = torch.cuda.CUDAGraph()
        live = torch.cuda.graph(graph, pool=self.pool)
        live.__enter__()
        self._live = live  # only a capture that began is ended
        self.graphs.append(graph)

    def end(self) -> None:
        live, self._live = self._live, None
        live.__exit__(None, None, None)

    def cut(self, between=None):
        """End the graph being captured, call `between` outside any capture (static tensors are made there) and
        begin the next graph; returns what `between` returned. In the warm-up passes only `between` runs."""
        live = self.capturing
        if live:
            self.end()
        made = between() if between is not None else None
        if live:
            self.begin()
        return made


class DecodeGraphs:
    """CUDA graphs of one decode step through SubspaceCausalLM._body, one per batch bucket. A step is copied into the
    graph's static buffers (token ids, page-table slots, lengths; padding rows point at the pool's scratch page) and
    replayed, so a step costs a few copies instead of every kernel launch. Rebuilt when the pool or its page table
    is replaced. With Quest (kv_page_budget < 1), steps whose longest row has reached kv_select_min_tokens use graphs
    captured per context bucket (page-table width), whose page selection runs on the device with static shapes. Not
    used for LoRA steps, MoE layers that would sync with the host (SparseMoE.graph_safe), or off CUDA."""

    # a step is padded to its bucket, and a padding row costs what a row costs (all but its KV read): steps of 1.5x
    # between the powers of two keep that under a third of the rows
    BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
    CONTEXT_BUCKETS = tuple(2 ** i for i in range(9, 14))  # page-table widths for Quest graphs: 8K .. 128K tokens

    def __init__(self):
        # holds no reference to its model (the model passes itself in): no model <-> graphs cycle, so a dropped model
        # frees its KV pool at once, and a deep-copied model gets graphs of its own
        self.graphs = {}  # (batch bucket, Quest width) -> (key, graph, ids, rows, logits, host rows, turn, aux)

    def eligible(self, model: "SubspaceCausalLM", caches: List[KVCache], lora: bool,
                 starts: Optional[List[int]] = None) -> bool:
        cfg, pool, w = model.config, caches[0].pool, model.embed_tokens.weight
        longest = max(starts if starts is not None else [c.length for c in caches]) + 1
        bucket = next((b for b in self.BUCKETS if b >= len(caches)), 0)
        return (cfg.cuda_graphs and pool.device.type == "cuda" and len(caches) <= self.BUCKETS[-1]
                and not lora and not torch.is_grad_enabled()
                and (not pool.track_pages or longest <= self.CONTEXT_BUCKETS[-1] * 16)
                and all(layer.mlp.graph_safe(w) or layer.mlp.cut_safe(w, bucket)  # else routing syncs: eager
                        for layer in model.layers if isinstance(layer.mlp, SparseMoE))
                and cfg.decode_attention == "kernel"  # the SDPA path reads lengths on the host
                and pool_decode_supported(pool, model.embed_tokens.weight))

    # Capture and replay always run in inference mode: the graph's static buffers and the CUDA generator's graph-safe
    # state are created by the first capture, and an in-place update across modes (inference tensor touched outside
    # inference mode) fails mid-capture and leaves the generator unusable.
    @torch.inference_mode()
    def run(self, model: "SubspaceCausalLM", input_ids: torch.Tensor, caches: List[KVCache],
            starts: Optional[List[int]] = None) -> torch.Tensor:
        B, pool, cfg = len(caches), caches[0].pool, model.config
        bucket = next(b for b in self.BUCKETS if b >= B)
        width = None
        starts = row_starts(caches) if starts is None else starts
        longest = max(starts) + 1
        if pool.track_pages and cfg.kv_page_budget < 1.0 and longest >= cfg.kv_select_min_tokens:
            width = next(w for w in self.CONTEXT_BUCKETS if w * 16 >= longest)
        key = (id(pool), pool.page_table.data_ptr(), model.aux_layers)
        entry = self.graphs.get((bucket, width))
        if entry is None or entry[0] != key:
            entry = self._capture(model, bucket, pool, key, width)
        _, graph, ids, rows, logits, pinned, turn, aux = entry
        scratch = pool.scratch_slot()
        # The step's rows go through pinned memory without waiting, so a caller that keeps its tokens on the device
        # can queue this step while the previous one runs (the engine's overlapped decode). Two buffers in turn; a
        # buffer is written again only once the copy that last read it has run, which is already the case unless the
        # caller queued the two steps since without reading any result.
        slot = pinned[turn[0]]
        turn[0] ^= 1
        host = slot[0]
        if slot[1] is None:
            slot[1] = torch.cuda.Event()
        else:
            slot[1].synchronize()
        host.copy_(torch.tensor([[c.slot for c in caches] + [scratch] * (bucket - B),
                                 starts + [0] * (bucket - B)], dtype=torch.int32))
        rows.copy_(host, non_blocking=True)
        slot[1].record()
        ids[:B].copy_(input_ids)
        if isinstance(graph, StepCuts):  # graphs 2i and 2i + 1 end at offloaded layer i's two cuts; then the rest
            for i, site in enumerate(graph.sites):
                graph.graphs[2 * i].replay()
                site[0].between_graphs(site, B, graph.graphs[2 * i + 1].replay)
            graph.graphs[-1].replay()
        else:
            graph.replay()
        if aux is not None:
            model.last_aux = aux[:B].clone()
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
        # offloaded MoE layers (eligible only when cut_safe): the step becomes a chain of graphs, and their forward
        # takes the path without host syncs from the warm-up on
        offloaded = [layer.mlp for layer in model.layers
                     if isinstance(layer.mlp, SparseMoE) and layer.mlp._host is not None]
        cuts = StepCuts() if offloaded else None
        for moe in offloaded:
            moe._cuts = cuts
        try:
            side = torch.cuda.Stream(device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):  # warm-up (Triton compilation, allocator) writes only the scratch page
                for _ in range(2):
                    step._freqs, step.quest_plan = {}, None
                    model._body(ids, rows[1], None, step)
            torch.cuda.current_stream(device).wait_stream(side)
            step._freqs, step.quest_plan = {}, None
            if cuts is None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    logits = model._body(ids, rows[1], None, step)
            else:
                graph = cuts
                cuts.begin()
                logits = model._body(ids, rows[1], None, step)
        finally:
            for moe in offloaded:  # first: a layer left in capture mode would add its resident experts only
                moe._cuts = None
            if cuts is not None and cuts.capturing:
                cuts.end()
        # [buffer, the CUDA event after the last copy out of it]
        pinned = [[torch.empty(2, bucket, dtype=torch.int32, pin_memory=True), None] for _ in range(2)]
        aux = model.last_aux if model.aux_layers else None  # the captured step's tapped states: a static output too
        self.graphs[(bucket, width)] = (key, graph, ids, rows, logits, pinned, [0], aux)
        return self.graphs[(bucket, width)]


class PrefillGraphs:
    """CUDA graphs of prefill steps whose prompts all start with no cached prefix: one prompt arriving alone, or a
    step of a burst (several prompts' first chunks, plus the decode rows of the requests already generating). The
    eager forward issues about 20 launches a layer from Python, and a packed step more for each prompt in it, which
    takes longer than the device needs for short prompts (one 512-token prompt on an L4: 37 ms on the host for 27 ms
    of device time on Qwen2.5-1.5B, 40 for 9 on SmolLM2-360M); a replay costs the device time.

    One graph per step shape (chunks, their padded length, decode rows padded to a decode bucket). Every chunk is
    padded to the bucket of the longest: a padding token costs a token's compute, comes after every real token
    (causal attention never looks at it) and its keys and values land past the cache's length, where the next tokens
    overwrite them. A shape is captured at its second use, so one-off shapes do not pay a capture (about one eager
    forward more), and all graphs share one memory pool (only one runs at a time). The forward is
    SubspaceCausalLM._body on a PackedPrefill.fresh: appends through the decode step's append kernel, plain causal
    attention over the chunks as a batch (over what an int8 or fp8 layer gives back for them), and the decode kernel
    for the decode rows. Not with an int4 or Quest pool, a sliding window or soft-capped scores, length-dependent
    RoPE, LoRA, a draft head (it reads every position's states), or MoE layers that would sync."""

    # Longer chunks are bound by the device: on an L4 a replay took the first token of a 64 / 512 / 1,024 / 2,048-token
    # prompt from 36 to 5 / 11 / 21 / 45 ms on SmolLM2-360M and from 34 / 35 / 63 / 130 to 16 / 30 / 62 / 130 ms on
    # Qwen2.5-1.5B, so graphs above 1,024 tokens would cost their captures and memory for under 1%.
    MAX_TOKENS = 1024
    # A step of more prompt tokens is bound by the device whatever its shape (64 prompts of 1,024 tokens on
    # SmolLM2-360M, two a step: 49.6 ms eager, 47 from a graph), and its graph would hold a second copy of the
    # step's activations: at 8,192 tokens a step that ran SmolLM2-1.7B out of memory beside a full KV pool.
    MAX_STEP_TOKENS = 2048
    MAX_GRAPHS = 48  # each holds its step's activations in the shared pool's address space and a few MB of launches
    FILL = 0.75  # chunks of unlike lengths stay eager when they fill less than this of n times the longest
    KEEP = 0.9

    def __init__(self):
        self.graphs = {}  # (chunks, bucket, decode rows) -> [pool stamp, graph, static inputs, logits, PackedPrefill]
        self.seen = {}  # step shape -> eligible steps so far
        self.pool = None  # the graphs' shared memory pool
        # A graph has to earn its place: a shape's eager forward is timed at its first use and its first two replays
        # after it; if the faster replay took more than KEEP of the eager time, the device bounds that step on this
        # model and GPU. For one chunk alone (an 8B model on an L4 at any length) that length and every longer one
        # then stay eager, in steps of any shape; another shape is dropped by itself.
        self.trial = {}  # step shape -> [eager ms, replay ms, ...] until judged
        self.limit = self.MAX_TOKENS
        self.dead = set()

    @staticmethod
    def bucket(tokens: int) -> int:
        """Padded length: steps of 32 to 512 tokens and 64 above (at most 6% more tokens than asked for from 512)."""
        step = 32 if tokens <= 512 else 64
        return -(-tokens // step) * step

    def _ok(self, model: "SubspaceCausalLM", cache: KVCache, tokens: int) -> bool:
        """What a graph over `tokens` positions needs of the model and the pool."""
        cfg, pool = model.config, cache.pool
        if not (cfg.cuda_graphs and pool.device.type == "cuda" and not torch.is_grad_enabled() and not model.aux_layers
                and pool.kv_cache_dtype != "int4" and not pool.track_pages):  # int4 keys are smoothed per layer
            return False
        w = model.embed_tokens.weight
        ok = model.__dict__.get("_prefill_graph_ok")
        if ok is None:  # what does not change with the request
            ok = model.__dict__["_prefill_graph_ok"] = (
                model.can_pack_prefill([cache]) and pool_decode_supported(pool, w)
                and not any(m.window is not None or m.softcap is not None
                            for m in model.modules() if isinstance(m, SubspaceAttention)))
        return ok and all(layer.mlp.graph_safe(w, tokens)
                          for layer in model.layers if isinstance(layer.mlp, SparseMoE))

    def eligible(self, model: "SubspaceCausalLM", cache: KVCache, tokens: int, logits_to_keep: int, lora: bool) -> bool:
        """One chunk alone (its step shape is (1, bucket(tokens), 0))."""
        return (cache.length == 0 and 1 < tokens and self.bucket(tokens) <= self.limit and logits_to_keep == 1
                and not lora and self._ok(model, cache, self.bucket(tokens)))

    def step_key(self, model: "SubspaceCausalLM", caches: List[KVCache], lens: List[int], riders: int) -> Optional[tuple]:
        """The shape of a packed step (prefill_packed's arguments) if a graph can serve it, else None: every chunk
        starts an empty cache and the chunks are of like lengths."""
        n = len(caches) - riders
        if n < 1 or min(lens[:n]) < 2 or any(c.length for c in caches[:n]):
            return None
        b = self.bucket(max(lens[:n]))
        r = next((x for x in DecodeGraphs.BUCKETS if x >= riders), None) if riders else 0
        key = (n, b, r)
        if r is None or b > self.limit or n * b > self.MAX_STEP_TOKENS or key in self.dead \
                or sum(lens[:n]) < self.FILL * n * max(lens[:n]) \
                or not self._ok(model, caches[0], n * b + r):
            return None
        return key

    @torch.inference_mode()  # as DecodeGraphs: every capture and replay in one mode
    def run(self, model: "SubspaceCausalLM", key: tuple, tokens, caches: List[KVCache],
            lens: List[int]) -> Optional[torch.Tensor]:
        """Logits [chunks + decode rows, V] of the step of shape `key`: caches' first key[0] entries are empty and
        take chunks of lens tokens, the rest are decode rows (a cache listed again: row_starts). tokens is one
        chunk as a device tensor [1, n], or every chunk's and row's list of ids. The keys and values are then in the
        caches (the caller advances their lengths). None when the step takes the eager path: its shape's first use,
        no room for another graph, or no pages for the padding."""
        n, b, r = key
        pool, riding = caches[0].pool, len(caches) - n
        self.seen[key] = self.seen.get(key, 0) + 1
        if self.seen[key] < 2 or (key not in self.graphs and len(self.graphs) >= self.MAX_GRAPHS):
            return None
        starts = row_starts(caches)
        # the padding's pages are held for this forward only: each prompt keeps what it had, or what its tokens need
        keep = [max(len(c.pages), -(-m // PAGE)) for c, m in zip(caches[:n], lens)]
        try:
            scratch = pool.scratch_slot() if r else 0  # before the stamp: its first use writes the page table
            for c in caches[:n]:
                c.reserve(b)
            for c, s in zip(caches[n:], starts[n:]):
                c.reserve(s + 1)
        except KVPoolExhausted:
            for c, pages in zip(caches[:n], keep):
                c.trim(pages)
            return None
        stamp = (id(pool), pool.page_table.data_ptr())
        entry = self.graphs.get(key)
        if entry is None or entry[0] != stamp:
            entry = self.graphs[key] = [stamp, None, self._inputs(key, pool.device), None, None]
        ids, slots, positions, last, rows = entry[2]
        dev = pool.device
        if torch.is_tensor(tokens):
            ids[0, :lens[0]].copy_(tokens[0])
        else:
            flat = [t for x in tokens[:n] for t in x + [0] * (b - len(x))] + [x[0] for x in tokens[n:]]
            ids[0, :len(flat)].copy_(torch.tensor(flat, device=dev))
        slots.view(n, b).copy_(torch.tensor([[c.slot] for c in caches[:n]], dtype=torch.int32, device=dev))
        last[:n].copy_(torch.tensor([i * b + m - 1 for i, m in enumerate(lens[:n])], device=dev))
        if r:  # padding rows write and read the pool's scratch page
            pad = r - riding
            rows.copy_(torch.tensor([[c.slot for c in caches[n:]] + [scratch] * pad, starts[n:] + [0] * pad],
                                    dtype=torch.int32, device=dev))
            positions[n * b:].copy_(rows[1])
        if entry[1] is None:
            try:
                self._capture(model, key, pool, entry)
            except BaseException as e:
                del self.graphs[key]
                for c, pages in zip(caches[:n], keep):
                    c.trim(pages)
                if not isinstance(e, torch.OutOfMemoryError):
                    raise
                self._drop(key)  # no room for the graph's copy of the step's activations: the eager forward
                return None
        graph, logits = entry[1], entry[3]
        trial = self.trial.get(key)
        if trial is None:
            graph.replay()
            for c, pages in zip(caches[:n], keep):  # whoever takes these pages writes them after this replay does
                c.trim(pages)
            return logits[0, :n + riding].clone()  # the next replay overwrites the static output
        torch.cuda.synchronize()  # the caller reads the logits next, so the waits cost nothing
        t = time.perf_counter()
        graph.replay()
        for c, pages in zip(caches[:n], keep):
            c.trim(pages)
        out = logits[0, :n + riding].clone()
        torch.cuda.synchronize()
        trial.append((time.perf_counter() - t) * 1e3)
        if len(trial) == 3:
            del self.trial[key]
            if min(trial[1:]) > self.KEEP * trial[0]:
                self._drop(key)
        return out

    def _drop(self, key: tuple) -> None:
        n, b, r = key
        if (n, r) == (1, 0):
            self.limit = min(self.limit, b - 1)
        else:
            self.dead.add(key)
        for gone in [k for k in self.graphs if k == key or (n, r) == (1, 0) and k[1] >= b]:
            del self.graphs[gone]

    def time_eager(self, key: tuple) -> Optional[float]:
        """Called before the eager forward of a step run() declined: the start time if this is its shape's first
        use (the forward is then timed, see eager_took), else None."""
        if self.seen.get(key) != 1:
            return None
        torch.cuda.synchronize()
        return time.perf_counter()

    def eager_took(self, key: tuple, since: float, scale: float = 1.0) -> None:
        """scale: 1 for the step's own eager forward at its first use; for a forward of fewer prompt tokens timed
        in its place (prefill_packed), the step's tokens over that forward's."""
        torch.cuda.synchronize()
        self.trial[key] = [(time.perf_counter() - since) * 1e3 * scale]

    @staticmethod
    def _inputs(key: tuple, device):
        """A graph's static inputs: ids [1, T], each chunk token's page-table row [n * b], positions [T], the indices
        whose logits are read [n + r] and the decode rows' page-table rows and positions [2, r] (None without)."""
        n, b, r = key
        positions = torch.cat([torch.arange(b).repeat(n), torch.zeros(r, dtype=torch.long)]).to(device)
        last = torch.cat([torch.zeros(n, dtype=torch.long), n * b + torch.arange(r)]).to(device)
        return (torch.zeros(1, n * b + r, dtype=torch.long, device=device),
                torch.zeros(n * b, dtype=torch.int32, device=device), positions, last,
                torch.zeros(2, r, dtype=torch.int32, device=device) if r else None)

    @torch.inference_mode()
    def _capture(self, model: "SubspaceCausalLM", key: tuple, pool: KVPool, entry: list) -> None:
        """Capture the step into entry; its inputs already hold the step that asked for it."""
        if self.pool is None or not any(e[1] is not None for e in self.graphs.values()):
            # no graph holds the pool any more (all dropped, or the only one replaced): a capture into it would
            # fail PyTorch's allocator check (use_count > 0), so this one starts a new pool
            self.pool = torch.cuda.graph_pool_handle()
        ids, slots, positions, last, rows = entry[2]
        chunk = PackedPrefill.fresh(pool, key[0], key[1], slots, positions, last, rows)
        # warm-up (Triton compilation, allocator): it writes these requests' reserved pages, as the replay will
        model._body(ids, 0, None, None, packed=chunk)
        chunk._freqs = {}
        torch.cuda.empty_cache()  # the capture allocates the step's activations again, in the graphs' pool
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            logits = model._body(ids, 0, None, None, packed=chunk)
        # the chunk too: the graph reads its position tensors (RoPE, the append kernel) on every replay
        entry[1], entry[3], entry[4] = graph, logits, chunk


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
        self._prefill_graphs = PrefillGraphs()
        # EAGLE-3 drafting (turing.models.eagle3): with aux_layers set, a forward leaves the residual stream after
        # those layers in last_aux [B, T, len(aux_layers) * H] (T = 1 for a decode step)
        self.aux_layers: Optional[Tuple[int, ...]] = None
        self.last_aux: Optional[torch.Tensor] = None

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
        logits_to_keep: int = 0,
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

    def offload_experts(self, slots: float) -> int:
        """Keep `slots` experts per MoE layer on the device and the rest in host memory (SparseMoE.offload), for a
        model whose experts do not all fit on the GPU: call it before moving the model there. A fraction is an
        average: 6.5 keeps 6 in one layer and 7 in the next (an expert of Mixtral-8x7B is 92 MB, and half an expert a
        layer is 1.4 GiB that a whole one more would not leave). Returns the number of layers offloaded."""
        moes = [layer.mlp for layer in self.layers if isinstance(layer.mlp, SparseMoE)]
        if not moes:
            raise ValueError("expert offload needs a model with MoE layers")
        counts = [int((i + 1) * slots) - int(i * slots) for i in range(len(moes))]
        for moe, n in zip(moes, counts):  # before any layer changes
            if not 0 < n < moe.num_experts:
                raise ValueError(f"expert slots must be between 1 and {moe.num_experts - 1} in every layer; "
                                 f"{slots:g} gives a layer {n}")
        for i, (moe, n) in enumerate(zip(moes, counts)):
            moe.offload(n)
            # the layers look at their residents on different steps: a change copies an expert in (and converts it
            # when the slots are Marlin's), and all of them on one step would stall that token
            moe._small_steps = i * moe.REBALANCE // len(moes)
        self._graphs.graphs.clear()
        self._prefill_graphs.graphs.clear()
        return len(moes)

    def decode_syncs(self) -> bool:
        """A decode step waits for the host inside the forward (offloaded experts): launching the next step before
        reading this one's tokens gains nothing then (ContinuousBatchEngine.overlap_decode)."""
        return any(isinstance(layer.mlp, SparseMoE) and layer.mlp._host is not None for layer in self.layers)

    def expert_offload_stats(self) -> Optional[dict]:
        """Decode-step expert uses served from the device (hits), copied in first (misses) or run on the CPU (cpu),
        and experts streamed for prefill-sized inputs, summed over the offloaded layers; None when nothing is
        offloaded. hit_rate is the share served from the device."""
        moes = [layer.mlp for layer in self.layers if isinstance(layer.mlp, SparseMoE) and layer.mlp._host is not None]
        if not moes:
            return None
        hits, misses = sum(m.offload_hits for m in moes), sum(m.offload_misses for m in moes)
        cpu = sum(m.offload_cpu for m in moes)
        return {"slots_per_layer": round(sum(m.num_slots for m in moes) / len(moes), 2),
                "experts_per_layer": moes[0].num_experts, "hits": hits,
                "misses": misses, "cpu": cpu, "cpu_seconds": round(sum(m.offload_cpu_s for m in moes), 3),
                "hit_rate": round(hits / (hits + misses + cpu), 4) if hits + misses + cpu else None,
                "streamed": sum(m.offload_streamed for m in moes)}

    def can_pack_prefill(self, caches: List[KVCache]) -> bool:
        """Whether prefill_packed serves these caches: one CUDA pool the paged prefill kernel reads, RoPE whose
        frequencies do not depend on a sequence's length (not dynamic or LongRoPE), no recirculation."""
        from ..core.rope import RotaryEmbedding
        if not caches or self.config.enable_recirculation or any(c.pool is not caches[0].pool for c in caches):
            return False
        if not pool_prefill_supported(caches[0].pool, self.embed_tokens.weight):
            return False
        return all(m.kind not in ("dynamic", "longrope") for m in self.modules() if isinstance(m, RotaryEmbedding))

    def can_ride_prefill(self, caches: List[KVCache]) -> bool:
        """Whether decoding requests can take their next token in prefill_packed (its `riders`): the paged decode
        kernel serves their attention there as in a decode step. Not with Quest page selection, which a decode step
        plans per row."""
        return (not caches[0].pool.track_pages and self.config.decode_attention != "sdpa"
                and self.can_pack_prefill(caches))

    def packed_step_graph(self, caches: List[KVCache], lens: List[int], riders: int = 0) -> bool:
        """Whether prefill_packed would replay this step from a CUDA graph, or is still timing its shape for one
        (PrefillGraphs.step_key): caches as prefill_packed takes them, lens the prompt chunks' lengths."""
        return self._prefill_graphs.step_key(self, caches, lens + [1] * riders, riders) is not None

    @torch.inference_mode()
    def prefill_packed(self, chunks: List[List[int]], caches: List[KVCache], riders: int = 0) -> torch.Tensor:
        """Prefill the next chunk of several requests in one forward (speed plan P4.3): chunks[i] continues caches[i]
        (filled in place). Returns the logits of each chunk's last token [len(chunks), V]. Check can_pack_prefill
        first. The last `riders` chunks are decode rows, one token each (check can_ride_prefill): the same pass over
        the weights gives requests that are already decoding their next token. As in a decode step, a cache listed
        k + 1 times there verifies k drafts (row_starts); the caller then sets its length to what it accepted."""
        dev = self.embed_tokens.weight.device
        if dev.type == "cuda" and not self.__dict__.get("_projections_fused"):
            self.fuse_projections()
        lens = [len(x) for x in chunks]
        graphs = self._prefill_graphs
        shape = graphs.step_key(self, caches, lens, riders)  # a step of prompts with no cached prefix: a CUDA graph
        logits = graphs.run(self, shape, chunks, caches, lens) if shape is not None else None
        if logits is None:
            timed = graphs.time_eager(shape) if shape is not None else None
            for c, s, n in zip(caches, row_starts(caches), lens):
                c.reserve(s + n)
            packed = PackedPrefill(caches, lens, dev, riders)
            ids = torch.tensor([[t for x in chunks for t in x]], dtype=torch.long, device=dev)
            logits = self._body(ids, 0, None, None, packed=packed)[0]
            n, over = len(lens) - riders, sum(lens) - graphs.MAX_STEP_TOKENS
            if timed is not None and riders and 0 < over < lens[n - 1]:
                # A step past MAX_STEP_TOKENS rows: its graph is judged against the forward the engine runs
                # without one, with the decode rows inside that many rows and the last chunk `over` tokens
                # shorter (ContinuousBatchEngine._step_batch), by the prompt token. Timed now, after the pass
                # above, whose keys and values it writes again.
                cut = lens[:n - 1] + [lens[n - 1] - over] + lens[n:]
                flat = [t for x in chunks[:n - 1] for t in x] + chunks[n - 1][:-over] + [x[0] for x in chunks[n:]]
                short = PackedPrefill(caches, cut, dev, riders)
                timed = graphs.time_eager(shape)
                self._body(torch.tensor([flat], dtype=torch.long, device=dev), 0, None, None, packed=short)
                graphs.eager_took(shape, timed, (sum(lens) - riders) / (sum(cut) - riders))
            elif timed is not None:
                graphs.eager_took(shape, timed)
        for c, n in zip(caches, lens):
            c.length += n
        return logits

    def _prepare_marlin(self, dtype: torch.dtype) -> None:
        """Build the Marlin copy (turing.kernels.marlin) of every 4-bit layer a CUDA forward uses: the fused q/k/v and
        gate/up layers and the unfused rest. QuantLinear would build them on first use, after the KV pool has taken the
        free memory, where a 7B model's copies may not fit."""
        from .quant import QuantLinear
        from ..kernels.marlin import HAS_MARLIN
        from ..kernels.marlin_moe import HAS_MARLIN_MOE
        if HAS_MARLIN_MOE:  # packed MoE experts: Marlin's layout on the device, the packed buffers to host memory
            for m in self.modules():
                if isinstance(m, SparseMoE):
                    m.use_marlin(dtype)
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
        timed = None  # set where a prefill's eager forward is timed for PrefillGraphs
        if step is not None:
            starts = row_starts(step)  # a cache listed more than once: consecutive positions (verify_in_step)
            if step[0].pool.track_pages and len({id(c) for c in step}) < len(step):
                # rows on one page would race on its Quest bounds
                raise ValueError("several rows of one request in a decode step are not supported with Quest page tracking")
            for c, s in zip(step, starts):
                c.reserve(s + 1)
            if self._graphs.eligible(self, step, lora, starts):
                logits = self._graphs.run(self, input_ids, step, starts)
                for c in step:
                    c.length += 1
                return logits, returned
            step = DecodeStep.of(step, input_ids.device, starts)
            start_pos = step.starts
        elif cache is not None:
            if self._prefill_graphs.eligible(self, cache, seq_len, logits_to_keep, lora):
                shape = (1, PrefillGraphs.bucket(seq_len), 0)
                logits = self._prefill_graphs.run(self, shape, input_ids, [cache], [seq_len])
                if logits is not None:
                    cache.length += seq_len
                    return logits[None], returned
                timed = self._prefill_graphs.time_eager(shape)
            cache.reserve(cache.length + seq_len)
            start_pos = cache.length
        else:
            start_pos = 0
        logits = self._body(input_ids, start_pos, cache, step, logits_to_keep)
        if timed is not None:
            self._prefill_graphs.eager_took(shape, timed)
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
        aux = []

        for i, layer in enumerate(self.layers):
            # Capture shallow state at router boundary
            if self.config.enable_recirculation and i == self.router_layer_idx:
                shallow_state = hidden_states.clone()

            hidden_states, out, _ = layer(hidden_states, pending=pending, defer=defer, **kw)
            pending = out if defer else None
            if self.aux_layers and i + 1 in self.aux_layers:
                aux.append(hidden_states + pending if pending is not None else hidden_states)

            # Apply deep-to-shallow recurrence at the penultimate layer if state captured
            if self.config.enable_recirculation and i == len(self.layers) - 2 and shallow_state is not None:
                hidden_states = self.recirculation(hidden_states, shallow_state)

        if aux:
            self.last_aux = torch.cat(aux, dim=-1)
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
        next-token distribution when only that layer's K and V take the pool's int8/int4/fp8 round trip, on the random
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
                a._kv_roundtrip = (cfg.kv_cache_dtype,)
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
            # drop ours (and graphs captured on it) first, so its memory can back the new pool; the local name too,
            # or the old pool outlives the calibration below and the new pool is sized from what it leaves
            self._kv_pool = pool = None
            self._graphs.graphs.clear()
            self._prefill_graphs.graphs.clear()
            gc.collect()
            if w.device.type == "cuda":
                torch.cuda.empty_cache()
            # int4 only: int8 needs no smoothing (+0.003 nats without it on Qwen2.5-1.5B-Instruct in bf16, +0.017 and
            # 16 GSM8K points lost with it)
            smooth = cfg.kv_cache_dtype == "int4" and cfg.kv_smoothing and not getattr(self, "_kv_smoothed", False)
            if cfg.kv_cache_dtype != "int4" and getattr(self, "_kv_smoothed", False):
                # a model that served int4 before: the other storages are calibrated, and stored, without smoothing
                # (their calibration files do not record it)
                for layer in self.layers:
                    layer.self_attn.k_smooth = layer.self_attn.q_smooth = None
                self._kv_smoothed = False
            choose = cfg.kv_cache_dtype != "native" and cfg.kv_native_layers is None
            if (smooth or choose) and not self._load_kv_calibration(smooth, choose):
                if smooth:
                    self.calibrate_kv_smoothing()
                if choose:
                    self.select_kv_native_layers()
                self._save_kv_calibration(smooth, choose)
            if w.device.type == "cuda":
                self._prepare_marlin(w.dtype)  # before the pool takes the free memory
                from .quant import _scratch
                for m in self.modules():  # and the buffer an offloaded 4-bit layer dequantizes an expert into
                    if isinstance(m, SparseMoE) and m._host is not None and m.gate_up is None:
                        _scratch(m.dequant_numel(), w.device, w.dtype)
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
