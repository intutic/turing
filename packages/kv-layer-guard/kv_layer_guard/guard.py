# Copyright 2026 Ishan Gupta
# SPDX-License-Identifier: Apache-2.0
"""
Find the attention layers a low-bit KV cache breaks, before serving with it.

For each layer, only that layer's keys and values take the cache format's round trip (quantize, dequantize) during a
forward pass over random tokens, and the mean KL divergence of the next-token distribution from the exact model's is
measured. Layers above a threshold are the ones to keep in full precision (vLLM: `kv_cache_dtype_skip_layers`).

Why random tokens: the damage comes from the weights (for example Qwen2's key-projection biases, which put a few key
channels at 30-70x the others in layer 0), not from the text, and random tokens need no dataset. On
Qwen2.5-1.5B-Instruct this finds layer 0 for fp8, int8 and int4; with layer 0 skipped, vLLM's fp8 cache goes from
2.4% to 56.8% on GSM8K (bf16: 57.4%).
"""
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import DynamicCache

FORMATS = ("fp8", "int8", "int4")
FP8_E4M3_MAX = 448.0


def roundtrip(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """What a KV cache in `fmt` gives back for x [..., head_dim], in x's dtype.

    fp8:  float8 e4m3 with scale 1.0, saturating at +-448 (vLLM's default fp8 cache, no calibrated scales).
    int8, int4: asymmetric round to nearest, one scale and zero per token and head (per-token-head caches).
    """
    if fmt == "fp8":
        y = x.float().clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        dev = y.device
        if dev.type == "mps":  # MPS has no float8 cast
            y = y.cpu()
        return y.to(torch.float8_e4m3fn).to(torch.float32).to(dev).to(x.dtype)
    if fmt not in ("int8", "int4"):
        raise ValueError(f"format must be one of {FORMATS}, not {fmt!r}")
    levels = 255 if fmt == "int8" else 15
    g = x.float()
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    scale = ((hi - lo) / levels).clamp(min=1e-8)
    return (torch.round((g - lo) / scale).clamp(0, levels) * scale + lo).to(x.dtype)


class RoundTripCache(DynamicCache):
    """A DynamicCache whose chosen layers store their keys and values through `roundtrip` (keys arrive after RoPE,
    as a KV cache stores them), so attention reads what the low-bit cache would give back."""

    def __init__(self, config, layers: Iterable[int], fmt: str):
        super().__init__(config=config)
        self._layers, self._fmt = set(layers), fmt

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if layer_idx in self._layers:
            key_states, value_states = roundtrip(key_states, self._fmt), roundtrip(value_states, self._fmt)
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)


def _num_layers(config) -> int:
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    return text.num_hidden_layers


@torch.no_grad()
def layer_kl(model, fmt: str, n: int = 4, tokens: int = 256, seed: int = 0) -> List[float]:
    """Per layer: mean KL(exact || that layer's K and V round-tripped) of the next-token distribution over n random
    sequences of `tokens` tokens."""
    config = model.config
    vocab = (config.get_text_config() if hasattr(config, "get_text_config") else config).vocab_size
    dev = next(model.parameters()).device
    ids = torch.randint(0, vocab, (n, tokens), generator=torch.Generator().manual_seed(seed)).to(dev)

    def logprobs(layers: Sequence[int]):
        return [F.log_softmax(model(ids[i:i + 1], past_key_values=RoundTripCache(config, layers, fmt),
                                    use_cache=True).logits.float(), -1) for i in range(n)]
    ref = logprobs(())
    kls = []
    for layer in range(_num_layers(config)):
        lp = logprobs((layer,))
        kls.append(sum((r.exp() * (r - x)).sum(-1).mean().item() for r, x in zip(ref, lp)) / n)
    return kls


def choose_layers(model, fmt: str, threshold: float = 0.02, max_fraction: float = 0.25,
                  **kwargs) -> Tuple[List[int], List[float]]:
    """(layers to keep in full precision, per-layer KL): those above `threshold` nats, worst first, at most
    max(1, max_fraction of the layers)."""
    kls = layer_kl(model, fmt, **kwargs)
    worst = sorted(range(len(kls)), key=lambda i: -kls[i])[:max(1, int(len(kls) * max_fraction))]
    return sorted(i for i in worst if kls[i] > threshold), kls


VLLM_FORMATS = {"fp8": "fp8", "fp8_e4m3": "fp8", "int8_per_token_head": "int8", "int4_per_token_head": "int4"}


def vllm_skip_layers(model_id: str, kv_cache_dtype: str = "fp8", device: str = None, dtype=torch.bfloat16,
                     **kwargs) -> List[str]:
    """The layer indices to pass as vLLM's `kv_cache_dtype_skip_layers` for `kv_cache_dtype`, as strings."""
    from transformers import AutoModelForCausalLM
    if kv_cache_dtype not in VLLM_FORMATS:
        raise ValueError(f"kv_cache_dtype must be one of {sorted(VLLM_FORMATS)}, not {kv_cache_dtype!r}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    layers, _ = choose_layers(model, VLLM_FORMATS[kv_cache_dtype], **kwargs)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return [str(i) for i in layers]
