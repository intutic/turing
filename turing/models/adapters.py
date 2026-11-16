"""
Multi-LoRA serving: PEFT LoRA adapters on a converted model's attention and MLP projections, several loaded at once and
chosen per request (per batch row), as vLLM's multi-LoRA does.

load_lora_adapter(model, name, path) reads a PEFT adapter (adapter_config.json + adapter_model.safetensors or .bin)
and wraps each targeted projection in a LoRALinear that keeps the base layer (dense or 4-bit QuantLinear) and every
adapter's (A, B, scale). SubspaceCausalLM.forward(..., adapters=[name or None per row]) selects them; rows without an
adapter get the base model. The output equals PEFT's for the same adapter (tests/test_lora_adapters.py).
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# HF module path suffix -> Turing module path suffix, for the llama-style layouts (llama, mistral, qwen2/3, gemma 1-3,
# mixtral/qwen-moe/olmoe attention). Phi-3's fused projections are split below.
_LLAMA_TARGETS = {
    "self_attn.q_proj": "self_attn.q_proj", "self_attn.k_proj": "self_attn.k_proj",
    "self_attn.v_proj": "self_attn.v_proj", "self_attn.o_proj": "self_attn.o_proj",
    "mlp.gate_proj": "mlp.gate_proj", "mlp.up_proj": "mlp.up_proj", "mlp.down_proj": "mlp.down_proj",
}


class LoRALinear(nn.Module):
    """A projection with any number of LoRA adapters: y = base(x) + scale * (x A^T) B^T for the row's adapter."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.in_features = getattr(base, "in_features", None)
        self.out_features = getattr(base, "out_features", None)
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.scales: Dict[str, float] = {}
        self.active: Optional[List[Optional[str]]] = None  # set by SubspaceCausalLM.forward for one call

    @property
    def weight(self):  # code that reads .weight (pruning, quantization) sees the base layer
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def add(self, name: str, A: torch.Tensor, B: torch.Tensor, scale: float) -> None:
        dev = next(self.base.buffers(), None)
        dev = dev.device if dev is not None else self.base.weight.device
        dt = self.base.weight.dtype if hasattr(self.base, "weight") and self.base.weight.is_floating_point() \
            else torch.float16
        self.lora_A[name] = nn.Parameter(A.to(device=dev, dtype=dt), requires_grad=False)
        self.lora_B[name] = nn.Parameter(B.to(device=dev, dtype=dt), requires_grad=False)
        self.scales[name] = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.active:
            return y
        names = self.active
        if len(set(names)) == 1:  # one adapter (or none) for the whole batch
            n = names[0]
            if n is None or n not in self.scales:
                return y
            return y + (x @ self.lora_A[n].T.to(x.dtype)) @ self.lora_B[n].T.to(x.dtype) * self.scales[n]
        y = y.clone()
        for n in set(names) - {None}:
            if n not in self.scales:
                continue
            rows = torch.tensor([i for i, m in enumerate(names) if m == n], device=x.device)
            xs = x.index_select(0, rows)
            y.index_add_(0, rows, (xs @ self.lora_A[n].T.to(x.dtype)) @ self.lora_B[n].T.to(x.dtype) * self.scales[n])
        return y


def _read_adapter(path: str) -> Tuple[dict, Dict[str, torch.Tensor]]:
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(path, allow_patterns=["adapter_config.json", "adapter_model.*"])
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    st = os.path.join(path, "adapter_model.safetensors")
    if os.path.exists(st):
        from safetensors.torch import load_file
        weights = load_file(st)
    else:
        weights = torch.load(os.path.join(path, "adapter_model.bin"), map_location="cpu", weights_only=True)
    return cfg, weights


def load_lora_adapter(model: nn.Module, name: str, path: str) -> List[str]:
    """Load the PEFT LoRA adapter at path (a directory or an HF repo id) into model under `name`. Returns the Turing
    module names it touches. Unsupported adapters (DoRA, targets other than the attention/MLP projections, layouts
    other than the llama style and Phi-3) raise."""
    cfg, weights = _read_adapter(path)
    if cfg.get("peft_type", "LORA") != "LORA" or cfg.get("use_dora"):
        raise ValueError(f"adapter {path}: only plain LoRA is supported (peft_type {cfg.get('peft_type')}, "
                         f"use_dora {cfg.get('use_dora')})")
    arch = model.config.arch
    if arch in ("gpt2", "opt", "gpt_neox", "phi"):
        raise ValueError(f"LoRA adapters for the '{arch}' layout are not supported")
    r, alpha = cfg["r"], cfg.get("lora_alpha", cfg["r"])
    scale = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r
    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, t in weights.items():
        if ".lora_A." not in key and ".lora_B." not in key:
            raise ValueError(f"adapter {path}: tensor {key} is not a LoRA A/B matrix (modules_to_save is not supported)")
        mod, ab = key.split(".lora_")[0], "A" if ".lora_A." in key else "B"
        pairs.setdefault(mod, {})[ab] = t
    touched = []
    for mod, ab in pairs.items():
        i = mod.index("layers.")
        layer, rest = mod[i:].split(".", 2)[1], mod[i:].split(".", 2)[2]
        A, B = ab["A"], ab["B"]
        if arch == "phi3" and rest in ("self_attn.qkv_proj", "mlp.gate_up_proj"):
            cfgm = model.config
            sizes = [cfgm.num_heads * cfgm.head_dim, cfgm.num_kv_heads * cfgm.head_dim, cfgm.num_kv_heads * cfgm.head_dim] \
                if rest == "self_attn.qkv_proj" else [B.shape[0] // 2] * 2
            names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] if rest == "self_attn.qkv_proj" \
                else ["mlp.gate_proj", "mlp.up_proj"]
            targets = list(zip(names, B.split(sizes)))
        elif rest in _LLAMA_TARGETS:
            targets = [(_LLAMA_TARGETS[rest], B)]
        else:
            raise ValueError(f"adapter {path}: target module '{rest}' is not supported")
        for tname, Bpart in targets:
            full = f"layers.{layer}.{tname}"
            parent, _, attr = full.rpartition(".")
            owner = model.get_submodule(parent)
            lin = getattr(owner, attr)
            if not isinstance(lin, LoRALinear):
                lin = LoRALinear(lin)
                setattr(owner, attr, lin)
            lin.add(name, A, Bpart, scale)
            touched.append(full)
    if not touched:
        raise ValueError(f"adapter {path} has no LoRA weights")
    names = set(getattr(model, "adapter_names", ()))
    model.adapter_names = tuple(sorted(names | {name}))
    return touched


def set_active_adapters(model: nn.Module, adapters: Optional[List[Optional[str]]]) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.active = adapters
