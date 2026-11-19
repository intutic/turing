"""
4-bit weight quantization (W4A16): round-to-nearest and GPTQ, a packed format, and QuantLinear.

Format (".tgate4" v2): per output row, groups of GROUP input columns share an fp16 scale and a uint8 zero point;
codes are unsigned 4-bit, two per byte (low nibble = even column): w = (q - zero) * scale. Bytes per weight:
0.5 + 3 / GROUP (0.547 at GROUP 64).

GPTQ (Frantar et al. 2022), as cift E11 measured it (gptq4 beat rtn4 on SmolLM2-135M, WikiText-2 ppl 26.55 vs 27.06):
columns left to right, each column's rounding error spread over the columns not yet quantized through the inverse
Hessian of the layer's calibration inputs, a group's grid set from the partly updated weights when the group is
reached, act-order off, damping 1% of the mean diagonal. Layers are quantized in order, each calibrated on the
outputs of the already-quantized layers before it (the reference algorithm's sequential mode).

QuantLinear picks its kernel by the number of rows: on CUDA up to INT4_MAX_ROWS rows (decode) the fused Triton
W4A16 GEMM reads the packed weights; above it (prefill) the layer is dequantized into a shared scratch buffer and
multiplied by cuBLAS. Off CUDA it dequantizes and multiplies. Where the optional Marlin extension is built
(turing.kernels.marlin), MARLIN_MIN_ROWS to MARLIN_MAX_ROWS rows go to Marlin first, from a repacked copy of the layer.
"""

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

BITS, GROUP = 4, 64
QMAX = 2 ** BITS - 1
# Rows at or below which the fused int4 kernel beats dequantize + cuBLAS on CUDA. Measured on an L4 with CUDA-graph
# replays (results_w4a16_crossover_cuda.json, summed over six layer shapes from 2048x2048 to 14336x4096): 1847 against
# 3000 us at 64 rows, about equal at 128, 6330 against 3812 at 256.
INT4_MAX_ROWS = 64
# Rows routed to the Marlin kernel when the optional extension is built (turing.kernels.marlin); M = 1 stays on the
# Triton GEMV. Starting values, to be set from scripts/benchmark_w4a16.py's marlin_us column.
MARLIN_MIN_ROWS, MARLIN_MAX_ROWS = 2, 1024


def _grid(w: torch.Tensor, qmax: int = QMAX) -> Tuple[torch.Tensor, torch.Tensor]:
    """Asymmetric grid of groups w [..., GROUP]: fp16-rounded scale and integer zero point, as cift E10/E11."""
    lo, hi = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
    scale = ((hi - lo) / qmax).clamp_min(1e-8).half().to(w.dtype)
    return scale, (-lo / scale).round().clamp(0, qmax)


def rtn_quantize(W: torch.Tensor, group: int = GROUP) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Round to nearest. W [N, K] -> codes uint8 [N, K], scales fp16 [N, K / group], zeros uint8 [N, K / group]."""
    N, K = W.shape
    g = W.double().view(N, K // group, group)
    scale, zero = _grid(g)
    codes = (g / scale + zero).round().clamp(0, QMAX)
    return codes.view(N, K).to(torch.uint8), scale[..., 0].half(), zero[..., 0].to(torch.uint8)


def gptq_quantize(W: torch.Tensor, H: torch.Tensor, group: int = GROUP, damp: float = 0.01, block: int = 128
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPTQ with act-order off. W [N, K], H = sum of x x^T over calibration inputs [K, K]. Same outputs as rtn_quantize."""
    N, K = W.shape
    assert K % group == 0 and block % group == 0
    W, H = W.double().clone(), H.double().clone()
    dead = H.diagonal() == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * H.diagonal().mean() * torch.eye(K, dtype=H.dtype, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    codes = torch.zeros(N, K, dtype=torch.uint8, device=W.device)
    scales = torch.zeros(N, K // group, dtype=torch.float16, device=W.device)
    zeros = torch.zeros(N, K // group, dtype=torch.uint8, device=W.device)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1, Hinv1 = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        Err1 = torch.zeros(N, i2 - i1, dtype=W.dtype, device=W.device)
        for i in range(i2 - i1):
            col = i1 + i
            if col % group == 0:
                scale, zero = _grid(W1[:, i:i + group])
                scale, zero = scale[:, 0], zero[:, 0]
                scales[:, col // group], zeros[:, col // group] = scale.half(), zero.to(torch.uint8)
            w = W1[:, i]
            q = (w / scale + zero).round().clamp(0, QMAX)
            codes[:, col] = q.to(torch.uint8)
            err = (w - (q - zero) * scale) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return codes, scales, zeros


def pack(codes: torch.Tensor) -> torch.Tensor:
    """uint8 codes [N, K] (values 0..15) -> [N, K / 2] bytes, low nibble = even column."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def unpack(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack((packed & 0x0F, packed >> 4), dim=-1).view(packed.shape[0], -1)


def dequantize(packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, dtype: torch.dtype,
               out: Optional[torch.Tensor] = None) -> torch.Tensor:
    N = packed.shape[0]
    G = scales.shape[1]
    w = unpack(packed).view(N, G, -1).to(dtype)
    w = (w - zeros.to(dtype)[..., None]) * scales.to(dtype)[..., None]
    if out is not None:
        out.copy_(w.view(N, -1))
        return out
    return w.view(N, -1)


_SCRATCH: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _scratch(numel: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """One reusable buffer per device and dtype, as large as the largest layer dequantized so far."""
    buf = _SCRATCH.get((device, dtype))
    if buf is None or buf.numel() < numel:
        with torch.inference_mode(False):  # a normal tensor: written both inside and outside inference mode
            buf = torch.empty(numel, device=device, dtype=dtype)
        _SCRATCH[(device, dtype)] = buf
    return buf[:numel]


class QuantLinear(nn.Module):
    """A linear layer stored as packed 4-bit weights (see the module docstring)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, group: int = GROUP,
                 device=None, dtype=torch.float16):
        super().__init__()
        if in_features % group:
            raise ValueError(f"in_features {in_features} is not a multiple of the quantization group {group}")
        self.in_features, self.out_features, self.group = in_features, out_features, group
        self.register_buffer("qweight", torch.zeros(out_features, in_features // 2, dtype=torch.uint8, device=device))
        self.register_buffer("scales", torch.zeros(out_features, in_features // group, dtype=torch.float16, device=device))
        self.register_buffer("zeros", torch.zeros(out_features, in_features // group, dtype=torch.uint8, device=device))
        self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype), requires_grad=False) if bias else None
        self._marlin = {}  # activation dtype -> MarlinWeights or None (unsupported), built on first use

    @classmethod
    def from_codes(cls, codes, scales, zeros, bias: Optional[torch.Tensor], group: int = GROUP) -> "QuantLinear":
        N, K = codes.shape
        m = cls(K, N, bias is not None, group, device=codes.device,
                dtype=bias.dtype if bias is not None else torch.float16)
        m.qweight.copy_(pack(codes))
        m.scales.copy_(scales)
        m.zeros.copy_(zeros)
        if bias is not None:
            m.bias.data.copy_(bias)
        return m

    def dequantized(self, dtype: torch.dtype) -> torch.Tensor:
        return dequantize(self.qweight, self.scales, self.zeros, dtype)

    def _marlin_weights(self, dtype: torch.dtype):
        """Marlin's copy of this layer for `dtype` activations (turing.kernels.marlin), or None where it does not run.
        Built once; assumes the buffers are not modified afterwards."""
        if dtype not in self._marlin:
            if torch.cuda.is_current_stream_capturing():  # the repack cannot run inside a CUDA-graph capture
                return None
            from ..kernels.marlin import marlin_pack, marlin_supported
            ok = marlin_supported(self.out_features, self.in_features, self.group, dtype, self.qweight.device)
            self._marlin[dtype] = marlin_pack(self.qweight, self.scales, self.zeros, dtype, self.bias) if ok else None
        return self._marlin[dtype]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, self.in_features)
        if x.is_cuda and MARLIN_MIN_ROWS <= x2.shape[0] <= MARLIN_MAX_ROWS and x.dtype in (torch.float16, torch.bfloat16):
            from ..kernels.marlin import HAS_MARLIN, marlin_gemm
            w = self._marlin_weights(x.dtype) if HAS_MARLIN else None
            if w is not None:
                return marlin_gemm(x2, w).view(*shape[:-1], self.out_features)
        if x.is_cuda and x2.shape[0] <= INT4_MAX_ROWS and x.dtype in (torch.float16, torch.bfloat16):
            from ..kernels.triton_w4a16 import HAS_TRITON, w4a16_gemm
            if HAS_TRITON:
                y = w4a16_gemm(x2, self.qweight, self.scales, self.zeros, self.group, self.bias)
                return y.view(*shape[:-1], self.out_features)
        n = self.out_features * self.in_features
        if x.is_cuda:
            from ..kernels.triton_w4a16 import HAS_TRITON, dequantize_cuda
            buf = _scratch(n, x.device, x.dtype).view(self.out_features, self.in_features)
            w = dequantize_cuda(self.qweight, self.scales, self.zeros, buf, self.group) if HAS_TRITON \
                else dequantize(self.qweight, self.scales, self.zeros, x.dtype, out=buf)
        else:
            w = self.dequantized(x.dtype)
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bits=4, group={self.group}, bias={self.bias is not None}"


def _quantizable(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    """Every attention and dense-MLP linear layer (embeddings, the LM head, routers and MoE experts stay as they are)."""
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.startswith("layers.") and ".router" not in name \
                and "shared_expert_gate" not in name and mod.in_features % GROUP == 0:
            out.append((name, mod))
    return out


def _replace(model: nn.Module, name: str, new: nn.Module):
    parent, _, attr = name.rpartition(".")
    setattr(model.get_submodule(parent), attr, new)


def embed_inputs(model: nn.Module, calib: torch.Tensor, batch: int = 4) -> List[torch.Tensor]:
    """Hidden states entering layer 0 for calibration ids [n, T], as SubspaceCausalLM._body computes them."""
    dev, cfg = model.embed_tokens.weight.device, model.config
    hs = []
    for i in range(0, len(calib), batch):
        ids = calib[i:i + batch].to(dev)
        h = model.embed_tokens(ids)
        if cfg.embed_proj_dim:
            h = model.project_in(h)
        if cfg.embed_scale is not None:
            h = h * torch.tensor(cfg.embed_scale, dtype=h.dtype)
        if cfg.learned_positions:
            h = h + model.embed_positions(torch.arange(ids.shape[1], device=dev)[None] + cfg.position_offset)
        hs.append(h)
    return hs


@torch.no_grad()
def quantize_layer(model: nn.Module, li: int, method: str, hs: Optional[List[torch.Tensor]] = None,
                   group: int = GROUP) -> Optional[List[torch.Tensor]]:
    """Replace layer li's linear layers by QuantLinear. GPTQ calibrates on hs (the layer's inputs) and returns the
    next layer's inputs through the quantized layer."""
    layer = model.layers[li]
    lins = [(n, m) for n, m in _quantizable(model) if n.startswith(f"layers.{li}.")]
    if method == "rtn":
        for n, m in lins:
            work = m.weight.device if m.weight.device.type == "cuda" else torch.device("cpu")  # MPS has no float64
            c, s, z = rtn_quantize(m.weight.float().to(work), group)
            bias = m.bias.to(work) if m.bias is not None else None
            _replace(model, n, QuantLinear.from_codes(c, s, z, bias, group).to(m.weight.device))
        return hs
    if method != "gptq":
        raise ValueError(f"quantization method '{method}' is not 'rtn' or 'gptq'")
    dev = next(layer.parameters()).device
    work = dev if dev.type == "cuda" else torch.device("cpu")  # the fp64 solve; MPS has no float64
    grams = {n: torch.zeros(m.in_features, m.in_features, dtype=torch.float32, device=dev) for n, m in lins}
    hooks = []
    for n, m in lins:
        def hook(mod, inp, out, n=n):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            grams[n] += x.T @ x
        hooks.append(m.register_forward_hook(hook))
    for h in hs:
        layer(h)
    for hk in hooks:
        hk.remove()
    for n, m in lins:
        c, s, z = gptq_quantize(m.weight.float().to(work), grams[n].to(work), group)
        bias = m.bias.to(work) if m.bias is not None else None
        _replace(model, n, QuantLinear.from_codes(c, s, z, bias, group).to(dev))
    del grams
    return [layer(h)[0] for h in hs]


@torch.no_grad()
def quantize_model(model: nn.Module, method: str = "rtn", calib: Optional[torch.Tensor] = None,
                   group: int = GROUP, batch: int = 4, lm_head: bool = False) -> nn.Module:
    """Replace the model's linear layers by QuantLinear in place. method "rtn" needs no data; "gptq" needs calib
    [n, T] token ids (WikiText-2 train windows, see calibration_windows) and runs the layers in order. lm_head also
    quantizes the LM head (quantize_head)."""
    if method == "gptq" and calib is None:
        raise ValueError("GPTQ needs calibration token ids")
    hs = embed_inputs(model, calib, batch) if method == "gptq" else None
    for li in range(len(model.layers)):
        hs = quantize_layer(model, li, method, hs, group)
    if lm_head:
        quantize_head(model, method, hs, group)
    model.config.quantization = f"w4a16-{method}"
    return model


@torch.no_grad()
def quantize_head(model: nn.Module, method: str, hs: Optional[List[torch.Tensor]] = None, group: int = GROUP) -> bool:
    """The LM head as QuantLinear (speed plan P3: at batch 1 it is the largest single read of a 4-bit model, 466 MB of
    fp16 on Qwen2.5-1.5B). GPTQ calibrates on hs, the last layer's outputs, through the final norm. A tied head gets
    its own packed copy; the embedding keeps its weight for the lookup. False when the head's width is not a multiple
    of the group (it stays as it is)."""
    head = model.lm_head
    if not isinstance(head, nn.Linear) or head.in_features % group:
        return False
    dev = head.weight.device
    work = dev if dev.type == "cuda" else torch.device("cpu")
    W = head.weight.float().to(work)
    if method == "gptq":
        H = torch.zeros(head.in_features, head.in_features, dtype=torch.float32, device=dev)
        for h in hs:
            if model.norm is not None:
                h = model.norm(h)
            if model.config.embed_proj_dim:
                h = model.project_out(h)
            x = h.reshape(-1, h.shape[-1]).float()
            H += x.T @ x
        c, s, z = gptq_quantize(W, H.to(work), group)
    elif method == "rtn":
        c, s, z = rtn_quantize(W, group)
    else:
        raise ValueError(f"quantization method '{method}' is not 'rtn' or 'gptq'")
    bias = head.bias.to(work) if head.bias is not None else None
    model.lm_head = QuantLinear.from_codes(c, s, z, bias, group).to(dev)
    return True


def calibration_windows(tokenizer, n: int = 128, T: int = 2048, seed: int = 0) -> torch.Tensor:
    """n random windows of T tokens from WikiText-2 train (GPTQ's usual calibration set)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("GPTQ and activation-sparsity calibration read WikiText-2: pip install datasets") from e
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(ids) - T - 1, (n,), generator=g)
    return torch.stack([ids[s:s + T] for s in starts])


def weight_bytes(model: nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))


TGATE4_FILE = "model.tgate4"


def save_quantized(model: nn.Module, out_dir: str, tokenizer=None) -> str:
    """Write out_dir/model.tgate4: a safetensors file (packed codes, scales, zeros, and the unquantized tensors) whose
    metadata holds the format version, group size and the ModelConfig; plus the tokenizer files if given."""
    import dataclasses
    import json
    import os
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    tensors = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    for k, v in tensors.items():  # fused projections (fuse_projections) are views of one tensor; safetensors refuses
        if v.untyped_storage().nbytes() != v.numel() * v.element_size():
            tensors[k] = v.clone()
    for k in [k for k in tensors if k.endswith("lm_head.weight")]:  # tied heads share storage, safetensors refuses
        if tensors[k].data_ptr() == tensors.get("embed_tokens.weight", torch.empty(0)).data_ptr():
            tensors[k] = tensors[k].clone()
    meta = {"format": "tgate4", "version": "2", "bits": str(BITS), "group": str(GROUP),
            "config": json.dumps(dataclasses.asdict(model.config))}
    path = os.path.join(out_dir, TGATE4_FILE)
    save_file(tensors, path, metadata=meta)
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
    return path


def load_quantized(path: str, device: str = "cpu", dtype: Optional[torch.dtype] = None):
    """(model, tokenizer or None) from a directory holding model.tgate4 (or the file itself). The packed weights are
    loaded as they are: no dequantized copy is ever made."""
    import json
    import os
    from safetensors import safe_open
    from safetensors.torch import load_file
    from ..config import ModelConfig
    from .causal_lm import SubspaceCausalLM
    from .hf_loader import resolve_dtype
    file = path if path.endswith(".tgate4") else os.path.join(path, TGATE4_FILE)
    with safe_open(file, "pt") as f:
        meta = f.metadata() or {}
    if meta.get("format") != "tgate4" or meta.get("version") != "2":
        raise ValueError(f"{file} is not a .tgate4 v2 file (metadata {meta})")
    cfg = ModelConfig(**json.loads(meta["config"]))
    dev = torch.device(device)
    dtype = resolve_dtype(dtype, dev)
    group = int(meta["group"])
    with torch.device("meta"):
        model = SubspaceCausalLM(cfg)
    sd = load_file(file, device=str(dev))
    for name in {k.rsplit(".", 1)[0] for k in sd if k.endswith(".qweight")}:
        out_f, in_half = sd[name + ".qweight"].shape
        _replace(model, name, QuantLinear(in_half * 2, out_f, name + ".bias" in sd, group, device="meta"))
    for name, t in sd.items():
        owner_name, _, attr = name.rpartition(".")
        owner = model.get_submodule(owner_name)
        if attr in owner._buffers:
            owner._buffers[attr] = t
        else:
            owner._parameters[attr] = nn.Parameter(t.to(dtype) if t.is_floating_point() else t, requires_grad=False)
    left = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.is_meta]
    if left:
        raise ValueError(f"{file} has no tensors for {left[:5]}")
    tokenizer = None
    folder = os.path.dirname(file)
    if os.path.exists(os.path.join(folder, "tokenizer_config.json")):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(folder)
    return model.eval(), tokenizer
