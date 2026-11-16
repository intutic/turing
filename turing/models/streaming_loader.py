"""
Layer-by-layer conversion of a HuggingFace checkpoint into a 4-bit .tgate4 file with bounded memory: the safetensors
shards are memory-mapped and read one tensor at a time, each decoder layer is mapped onto SubspaceCausalLM,
quantized (RTN, or GPTQ on the outputs of the already-quantized layers before it) and moved to the host as packed
codes before the next layer is read. Peak memory is one dense layer plus the calibration activations, not the model.
The checkpoint is downloaded (or read from the HF cache) first; nothing is streamed from the network mid-conversion.
"""

import json
import os
import time
from collections.abc import Mapping
from typing import Any, Dict, Iterator, Optional

import torch

from ..config import ModelConfig
from .causal_lm import SubspaceCausalLM
from .hf_loader import FAMILIES, _Writer, _config_kwargs, _copy_weights, resolve_dtype
from .quant import calibration_windows, embed_inputs, quantize_layer, save_quantized


class ShardedStateDict(Mapping):
    """name -> tensor over a checkpoint's safetensors shards, loading each tensor on access. Names follow the
    transformers model's state dict; checkpoints that store MoE experts one by one (Mixtral, Qwen-MoE, OLMoE) or
    GPT-2's unprefixed names are mapped on the fly."""

    def __init__(self, folder: str):
        from safetensors import safe_open
        idx = os.path.join(folder, "model.safetensors.index.json")
        if os.path.exists(idx):
            weight_map = json.load(open(idx))["weight_map"]
        else:
            single = os.path.join(folder, "model.safetensors")
            if not os.path.exists(single):
                raise ValueError(f"no safetensors weights in {folder}")
            with safe_open(single, "pt") as f:
                weight_map = {k: "model.safetensors" for k in f.keys()}
        self.folder, self.map = folder, weight_map
        self._open = {}

    def _raw(self, name: str) -> torch.Tensor:
        from safetensors import safe_open
        shard = self.map[name]
        if shard not in self._open:
            self._open[shard] = safe_open(os.path.join(self.folder, shard), "pt")
        return self._open[shard].get_tensor(name)

    def _resolve(self, name: str) -> Optional[torch.Tensor]:
        if name in self.map:
            return self._raw(name)
        if name.startswith("transformer.") and name[len("transformer."):] in self.map:  # GPT-2's original names
            return self._raw(name[len("transformer."):])
        for suffix, parts in ((".mlp.experts.gate_up_proj", ("gate", "up")), (".mlp.experts.down_proj", ("down",))):
            if name.endswith(suffix):
                return self._stacked_experts(name[: -len(suffix)], parts)
        if name.endswith(".mlp.gate.weight"):  # Mixtral's original router name
            alt = name.replace(".mlp.gate.weight", ".block_sparse_moe.gate.weight")
            if alt in self.map:
                return self._raw(alt)
        return None

    def _stacked_experts(self, prefix: str, parts) -> Optional[torch.Tensor]:
        names = {"gate": ("mlp.experts.{e}.gate_proj.weight", "block_sparse_moe.experts.{e}.w1.weight"),
                 "up": ("mlp.experts.{e}.up_proj.weight", "block_sparse_moe.experts.{e}.w3.weight"),
                 "down": ("mlp.experts.{e}.down_proj.weight", "block_sparse_moe.experts.{e}.w2.weight")}
        experts = []
        for e in range(4096):
            row = []
            for part in parts:
                found = next((f"{prefix}.{pat.format(e=e)}" for pat in names[part]
                              if f"{prefix}.{pat.format(e=e)}" in self.map), None)
                if found is None:
                    break
                row.append(self._raw(found))
            if len(row) != len(parts):
                break
            experts.append(torch.cat(row, dim=0))
        return torch.stack(experts) if experts else None

    def __getitem__(self, name: str) -> torch.Tensor:
        t = self._resolve(name)
        if t is None:
            raise KeyError(name)
        return t

    def get(self, name, default=None):
        t = self._resolve(name)
        return default if t is None else t

    def __contains__(self, name) -> bool:
        return name in self.map or self._resolve(name) is not None

    def __iter__(self) -> Iterator[str]:
        return iter(self.map)

    def __len__(self) -> int:
        return len(self.map)


class StreamingHFSubspaceLoader:
    """Converts one HuggingFace checkpoint into out_dir/model.tgate4 (see the module docstring)."""

    def __init__(self, repo_id: str, token: Optional[str] = None, target_device: str = "cpu",
                 method: str = "gptq", calib_windows: int = 128, calib_tokens: int = 2048):
        self.repo_id, self.token = repo_id, token or os.environ.get("HF_TOKEN")
        self.device = torch.device(target_device)
        self.method, self.calib_windows, self.calib_tokens = method, calib_windows, calib_tokens

    def _folder(self) -> str:
        if os.path.isdir(self.repo_id):
            return self.repo_id
        from huggingface_hub import snapshot_download
        return snapshot_download(self.repo_id, token=self.token,
                                 allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"])

    @torch.no_grad()
    def convert(self, out_dir: str) -> Dict[str, Any]:
        from transformers import AutoConfig, AutoTokenizer
        t0 = time.perf_counter()
        folder = self._folder()
        hf_cfg = AutoConfig.from_pretrained(folder)
        if hf_cfg.model_type not in FAMILIES:
            raise ValueError(f"model type '{hf_cfg.model_type}' is not supported; supported: {', '.join(sorted(FAMILIES))}")
        kw = _config_kwargs(hf_cfg)
        ffn = kw["ffn_dim"]
        cfg = ModelConfig(name=f"HF-{self.repo_id}", tile_size=64 if ffn <= 4096 else 256,
                          active_tiles=-(-ffn // (64 if ffn <= 4096 else 256)), **kw)
        dtype = resolve_dtype(None, self.device, torch.bfloat16 if self.device.type == "cuda" else None)
        with torch.device("meta"):
            model = SubspaceCausalLM(cfg)
        sd = ShardedStateDict(folder)
        put = _Writer(model, self.device, dtype)
        layout = FAMILIES[hf_cfg.model_type]
        _copy_weights(cfg, sd, put, layout, layers=[], glob=True)
        tokenizer = AutoTokenizer.from_pretrained(folder)
        hs = None
        if self.method == "gptq":
            calib = calibration_windows(tokenizer, self.calib_windows, self.calib_tokens)
            hs = embed_inputs(model, calib)
        peak = 0
        for li in range(cfg.num_layers):
            _copy_weights(cfg, sd, put, layout, layers=[li], glob=False)
            hs = quantize_layer(model, li, self.method, hs)
            model.layers[li].to("cpu")  # packed codes on the host; the next layer gets the device
            if self.device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated())
        put.check()
        model.to("cpu")
        cfg.quantization = f"w4a16-{self.method}"
        path = save_quantized(model, out_dir, tokenizer)
        return {"model_id": self.repo_id, "layers_converted": cfg.num_layers, "method": self.method,
                "output": path, "bytes": os.path.getsize(path), "elapsed_seconds": round(time.perf_counter() - t0, 1),
                "peak_device_bytes": peak or None}
