"""
4-bit weight quantization (W4A16): round-to-nearest and GPTQ, a packed format, QuantLinear, and the same packing for
the stacked experts of an MoE layer (quantize_experts; SparseMoE reads them through turing.kernels.triton_moe).

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

import functools
import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

BITS, GROUP = 4, 64
QMAX = 2 ** BITS - 1
# Rows at or below which the fused int4 kernel beats dequantize + cuBLAS on CUDA. Measured on an L4 with CUDA-graph
# replays (results_w4a16_crossover_cuda.json, summed over seven layer shapes of Qwen2.5-1.5B and 7B): 3776 against
# 7805 us at 64 rows, 12823 against 9767 at 256.
INT4_MAX_ROWS = 64  # on another GPU, `turing bench-bw --tune` measures it (turing.core.probe)
# Rows routed to the Marlin kernel when the optional extension is built (turing.kernels.marlin). On the L4 it beat the
# Triton kernels at every row count from 1 to 256 on Qwen2.5-1.5B's shapes (results_w4a16_crossover_ckpt7_cuda.json:
# 22.6 against 98.8 us at one row on the 1536x17920 gate/up layer), and dequantize + cuBLAS at prefill sizes too
# (results_w4a16_prefill_rows_cuda.json, summed over that model's four shapes: 3,397 against 4,405 us at 2,048 rows,
# 6,846 against 7,682 at 4,096, the most rows measured).
MARLIN_MIN_ROWS, MARLIN_MAX_ROWS = 1, 4096


def _grid(w: torch.Tensor, qmax: int = QMAX) -> Tuple[torch.Tensor, torch.Tensor]:
    """Asymmetric grid of groups w [..., GROUP]: fp16-rounded scale and integer zero point, as cift E10/E11."""
    lo, hi = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
    scale = ((hi - lo) / qmax).clamp_min(1e-8).half().to(w.dtype)
    return scale, (-lo / scale).round().clamp(0, qmax)


def rtn_quantize(W: torch.Tensor, group: int = GROUP) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Round to nearest. W [..., N, K] -> codes uint8 [..., N, K], scales fp16 and zeros uint8 [..., N, K / group]."""
    g = W.double().unflatten(-1, (-1, group))
    scale, zero = _grid(g)
    codes = (g / scale + zero).round().clamp(0, QMAX)
    return codes.flatten(-2).to(torch.uint8), scale[..., 0].half(), zero[..., 0].to(torch.uint8)


def gptq_quantize(W: torch.Tensor, H: torch.Tensor, group: int = GROUP, damp: float = 0.01, block: int = 128
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPTQ with act-order off. W [..., N, K], H = sum of x x^T over calibration inputs [..., K, K]; leading
    dimensions are independent problems solved together (stacked MoE experts). Same outputs as rtn_quantize."""
    *lead, N, K = W.shape
    assert K % group == 0 and block % group == 0
    W, H = W.double().clone(), H.double().clone()
    diag = H.diagonal(dim1=-2, dim2=-1)  # a view: writes go to H
    dead = diag == 0
    diag[dead] = 1
    W.masked_fill_(dead[..., None, :], 0)
    H += damp * diag.mean(-1)[..., None, None] * torch.eye(K, dtype=H.dtype, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    codes = torch.zeros(*lead, N, K, dtype=torch.uint8, device=W.device)
    scales = torch.zeros(*lead, N, K // group, dtype=torch.float16, device=W.device)
    zeros = torch.zeros(*lead, N, K // group, dtype=torch.uint8, device=W.device)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1, Hinv1 = W[..., i1:i2].clone(), Hinv[..., i1:i2, i1:i2]
        Err1 = torch.zeros(*lead, N, i2 - i1, dtype=W.dtype, device=W.device)
        for i in range(i2 - i1):
            col = i1 + i
            if col % group == 0:
                scale, zero = _grid(W1[..., i:i + group])
                scale, zero = scale[..., 0], zero[..., 0]
                scales[..., col // group], zeros[..., col // group] = scale.half(), zero.to(torch.uint8)
            w = W1[..., i]
            q = (w / scale + zero).round().clamp(0, QMAX)
            codes[..., col] = q.to(torch.uint8)
            err = (w - (q - zero) * scale) / Hinv1[..., i, i][..., None]
            W1[..., i:] -= err[..., None] * Hinv1[..., i, i:][..., None, :]
            Err1[..., i] = err
        W[..., i2:] -= Err1 @ Hinv[..., i1:i2, i2:]
    return codes, scales, zeros


def pack(codes: torch.Tensor) -> torch.Tensor:
    """uint8 codes [..., N, K] (values 0..15) -> [..., N, K / 2] bytes, low nibble = even column."""
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def unpack(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)


def dequantize(packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, dtype: torch.dtype,
               out: Optional[torch.Tensor] = None) -> torch.Tensor:
    w = unpack(packed).unflatten(-1, (scales.shape[-1], -1)).to(dtype)
    w = ((w - zeros.to(dtype)[..., None]) * scales.to(dtype)[..., None]).flatten(-2)
    if out is not None:
        out.copy_(w)
        return out
    return w


@functools.lru_cache(maxsize=None)
def _int4_max_rows() -> int:
    from ..core.probe import tuned
    return int(tuned("int4_max_rows", INT4_MAX_ROWS))


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
        if x.is_cuda and x2.shape[0] <= _int4_max_rows() and x.dtype in (torch.float16, torch.bfloat16):
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
    """Every attention and dense-MLP linear layer (embeddings, the LM head and routers stay as they are; MoE experts
    are stacked tensors, packed by quantize_experts)."""
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


def _moe_to_pack(layer: nn.Module, group: int):
    """The layer's SparseMoE when its experts are still fp and both of their widths are multiples of the group."""
    moe = getattr(layer, "mlp", None)
    if getattr(moe, "_host", None) is not None:
        raise ValueError("quantize the model before offloading its experts")
    if hasattr(moe, "pack_experts") and moe.gate_up is not None \
            and moe.gate_up.shape[-1] % group == 0 and moe.down.shape[-1] % group == 0:
        return moe
    return None


def _expert_grams(moe: nn.Module, x: torch.Tensor, grams: Dict[str, torch.Tensor]) -> None:
    """Add the tokens x [T, H] to the Gram matrices of the experts they are routed to: gate_up's inputs (x itself)
    and down's (the gated hidden states), and count the tokens each expert saw."""
    _, experts = moe.route(x)
    for e in torch.unique(experts).tolist():
        xe = x[(experts == e).any(-1)]
        gate, up = F.linear(xe, moe.gate_up[e]).chunk(2, dim=-1)
        hid = (moe.act(gate) * up).float()
        xe = xe.float()
        grams["gate_up"][e] += xe.T @ xe
        grams["down"][e] += hid.T @ hid
        grams["tokens"][e] += len(xe)


@torch.no_grad()
def quantize_experts(moe: nn.Module, method: str, grams: Optional[Dict[str, torch.Tensor]] = None,
                     group: int = GROUP) -> None:
    """Pack a SparseMoE's stacked experts (gate_up [E, 2F, H], down [E, H, F]) in 4 bits, in place. GPTQ uses each
    expert's own Gram matrices (grams, from _expert_grams). An expert that saw fewer calibration tokens than it has
    input columns is rounded to nearest instead: its Gram matrix is singular, and GPTQ zeroes the weights of inputs
    it never saw. Experts are solved a few at a time (about 2 GiB of fp64 per array: the weights [rows, columns] and
    GPTQ's Gram matrix [columns, columns]; all eight of Mixtral's 28,672 x 4,096 gate/up matrices at once needed
    more than an L4 holds)."""
    dev = moe.gate_up.device
    work = dev if dev.type == "cuda" else torch.device("cpu")  # the fp64 solve; MPS has no float64
    packed = {}
    for name in ("gate_up", "down"):
        W = getattr(moe, name)
        chunk = max(1, min(8, 2 ** 31 // (8 * W.shape[-1] * max(W.shape[-2:]))))
        parts = []
        for i in range(0, W.shape[0], chunk):
            w = W[i:i + chunk].float().to(work)
            c, sc, z = rtn_quantize(w, group)
            if method == "gptq":
                seen = (grams["tokens"][i:i + chunk] >= w.shape[-1]).to(work)
                if seen.any():
                    c[seen], sc[seen], z[seen] = gptq_quantize(w[seen], grams[name][i:i + chunk].to(work)[seen], group)
            elif method != "rtn":
                raise ValueError(f"quantization method '{method}' is not 'rtn' or 'gptq'")
            parts.append((pack(c).to(dev), sc.to(dev), z.to(dev)))
        packed[name] = tuple(torch.cat(t) for t in zip(*parts))
    moe.pack_experts(packed["gate_up"], packed["down"], group)


@torch.no_grad()
def quantize_layer(model: nn.Module, li: int, method: str, hs: Optional[List[torch.Tensor]] = None,
                   group: int = GROUP) -> Optional[List[torch.Tensor]]:
    """Replace layer li's linear layers by QuantLinear and pack its MoE experts. GPTQ calibrates on hs (the layer's
    inputs) and returns the next layer's inputs through the quantized layer."""
    layer = model.layers[li]
    lins = [(n, m) for n, m in _quantizable(model) if n.startswith(f"layers.{li}.")]
    moe = _moe_to_pack(layer, group)
    if method == "rtn":
        for n, m in lins:
            work = m.weight.device if m.weight.device.type == "cuda" else torch.device("cpu")  # MPS has no float64
            c, s, z = rtn_quantize(m.weight.float().to(work), group)
            bias = m.bias.to(work) if m.bias is not None else None
            _replace(model, n, QuantLinear.from_codes(c, s, z, bias, group).to(m.weight.device))
        if moe is not None:
            quantize_experts(moe, "rtn", None, group)
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
    expert_grams = None
    if moe is not None:
        E, two_f, H = moe.gate_up.shape
        expert_grams = {"gate_up": torch.zeros(E, H, H, dtype=torch.float32, device=dev),
                        "down": torch.zeros(E, two_f // 2, two_f // 2, dtype=torch.float32, device=dev),
                        "tokens": torch.zeros(E, dtype=torch.int64, device=dev)}
        hooks.append(moe.register_forward_pre_hook(
            lambda mod, args, g=expert_grams: _expert_grams(mod, args[0].reshape(-1, args[0].shape[-1]), g)))
    for h in hs:
        layer(h)
    for hk in hooks:
        hk.remove()
    for n, m in lins:
        c, s, z = gptq_quantize(m.weight.float().to(work), grams[n].to(work), group)
        bias = m.bias.to(work) if m.bias is not None else None
        _replace(model, n, QuantLinear.from_codes(c, s, z, bias, group).to(dev))
    if moe is not None:
        quantize_experts(moe, "gptq", expert_grams, group)
    del grams, expert_grams
    return [layer(h)[0] for h in hs]


@torch.no_grad()
def quantize_model(model: nn.Module, method: str = "rtn", calib: Optional[torch.Tensor] = None,
                   group: int = GROUP, batch: int = 4, lm_head: bool = False) -> nn.Module:
    """Replace the model's linear layers by QuantLinear and pack its MoE experts, in place. method "rtn" needs no
    data; "gptq" needs calib [n, T] token ids (WikiText-2 train windows, see calibration_windows) and runs the layers
    in order. lm_head also quantizes the LM head (quantize_head)."""
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
    if any(getattr(m, "_host", None) is not None for m in model.modules()):
        raise ValueError("a model with offloaded experts holds only some of them on its device: save it before offloading")
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
    for name in {k.rsplit(".", 1)[0] for k in sd if k.endswith(".gate_up_q")}:  # packed MoE experts
        model.get_submodule(name).pack_experts(*(tuple(sd[f"{name}.{w}{part}"] for part in ("_q", "_scales", "_zeros"))
                                                 for w in ("gate_up", "down")), group)
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
