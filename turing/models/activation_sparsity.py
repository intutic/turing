"""
TEAL-style training-free activation sparsity (opt-in): the input of every attention and dense-MLP projection is
thresholded by magnitude so that a chosen fraction of its entries is zero, with one threshold per projection input
calibrated on WikiText-2 activations. Quality cost is measured with scripts/eval_kv_quality.py (arms teal25/40/50);
decode speed at batch 1 on CUDA comes from triton_sparse_gemv (zero inputs skip their weight rows). Larger batches
gain nothing (a weight row is needed when any row's input is nonzero), so the sparse kernel runs only for one row.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseInputLinear(nn.Module):
    """base(x * (|x| >= threshold)); one-row CUDA inputs use the input-sparse GEMV over a transposed weight copy."""

    def __init__(self, base: nn.Module, threshold: float):
        super().__init__()
        self.base, self.threshold = base, float(threshold)
        self.in_features, self.out_features = base.in_features, base.out_features
        self._wt = None  # [K, N] for the sparse kernel, built on first CUDA use (dense weights only)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * (x.abs() >= self.threshold)
        rows = x.numel() // self.in_features
        quant = type(self.base).__name__ == "QuantLinear"
        if x.is_cuda and rows == 1 and (isinstance(self.base, nn.Linear) or quant) and self.out_features % 64 == 0:
            from ..kernels.triton_sparse_gemv import HAS_TRITON, quantize_transposed_int4, sparse_gemv, sparse_gemv_int4
            if HAS_TRITON:
                if self._wt is None:  # int4: a transposed 4-bit copy (groups of 64 outputs per input row)
                    self._wt = quantize_transposed_int4(self.base.dequantized(torch.float32)) if quant \
                        else self.base.weight.t().contiguous()
                y = sparse_gemv_int4(x.reshape(-1), *self._wt) if quant else sparse_gemv(x.reshape(-1), self._wt)
                if self.base.bias is not None:
                    y = y + self.base.bias
                return y.view(*x.shape[:-1], self.out_features)
        return self.base(x)


def _targets(model: nn.Module) -> List[str]:
    out = []
    for name, mod in model.named_modules():
        if name.startswith("layers.") and ".router" not in name and "shared_expert_gate" not in name and \
                (isinstance(mod, nn.Linear) or type(mod).__name__ in ("QuantLinear", "LoRALinear")):
            out.append(name)
    return out


@torch.no_grad()
def calibrate_and_apply(model: nn.Module, calib: torch.Tensor, sparsity: float, max_samples: int = 2 ** 20) -> Dict[str, float]:
    """Wrap every attention/MLP projection so `sparsity` of its inputs are zeroed. calib [n, T] token ids. Thresholds
    are the per-projection |x| quantiles of the dense model's inputs (TEAL's uniform-sparsity setting)."""
    if not 0 < sparsity < 1:
        raise ValueError("activation sparsity must be in (0, 1)")
    names = _targets(model)
    samples: Dict[str, List[torch.Tensor]] = {n: [] for n in names}
    hooks = []
    for n in names:
        def hook(mod, inp, out, n=n):
            x = inp[0].reshape(-1).abs().float()
            if x.numel() > max_samples // 8:
                x = x[torch.randint(0, x.numel(), (max_samples // 8,), device=x.device)]
            samples[n].append(x.cpu())
        hooks.append(model.get_submodule(n).register_forward_hook(hook))
    dev = model.embed_tokens.weight.device
    for ids in calib:
        model(ids[None].to(dev))
    for h in hooks:
        h.remove()
    thresholds = {}
    for n in names:
        x = torch.cat(samples[n])
        x = x[torch.randint(0, x.numel(), (min(x.numel(), max_samples),))]
        thresholds[n] = float(torch.quantile(x, sparsity))
        parent, _, attr = n.rpartition(".")
        setattr(model.get_submodule(parent), attr, SparseInputLinear(model.get_submodule(n), thresholds[n]))
    model.config.activation_sparsity = sparsity
    return thresholds
