"""
Preallocated per-request KV cache for the serving decode path, with optional low-bit storage and Quest page selection.

Layout per layer: [B, Hkv, capacity, D], written in place at the current length; capacity doubles when full, so a
decode step never copies the whole cache. Storage modes (`kv_cache_dtype`):
- "native": the model's dtype.
- "int8":   asymmetric round-to-nearest per token and KV head, fp16 scale and zero (D + 4 bytes per token-head).
- "int4":   asymmetric round-to-nearest per group of 32 values along D, fp16 scale and zero per group, two codes per
            byte (D/2 + D/8 bytes per token-head).
K is stored after RoPE. Quest metadata: per 16-token page and KV head, the elementwise min and max of K (from the
values before quantization), so a decode query can bound q.k over a page without reading it.
"""

from typing import List, NamedTuple, Optional, Tuple
import math
import torch

PAGE = 16
INT4_GROUP = 32
KV_CACHE_DTYPES = ("native", "int8", "int4")


def resolve_kv_cache_dtype(choice: str, device) -> str:
    """'auto' = int8 on CUDA (measured free in quality, fused kernels read it directly), native elsewhere (the
    reference path dequantizes the whole cache each step, so int8 there saves memory but costs speed)."""
    return ("int8" if torch.device(device).type == "cuda" else "native") if choice == "auto" else choice


def _quantize(x: torch.Tensor, bits: int, group: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [..., D] -> codes uint8 [..., D], scale and zero fp16 [..., D // group]."""
    g = x.float().unflatten(-1, (-1, group))
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    # tensor / tensor is IEEE division on every device (CUDA computes tensor / scalar via a reciprocal), which the
    # fused Triton append kernel reproduces exactly
    scale = ((hi - lo) / torch.full_like(hi, 2 ** bits - 1)).clamp(min=1e-8)
    codes = torch.round((g - lo) / scale).clamp(0, 2 ** bits - 1).to(torch.uint8).flatten(-2)
    return codes, scale.squeeze(-1).half(), lo.squeeze(-1).half()


def _dequantize(codes: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, group: int, dtype) -> torch.Tensor:
    g = codes.float().unflatten(-1, (-1, group))
    return (g * scale.float().unsqueeze(-1) + zero.float().unsqueeze(-1)).flatten(-2).to(dtype)


def _pack4(codes: torch.Tensor) -> torch.Tensor:
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def _unpack4(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)


class LayerKVCache:
    """K and V of one layer for a batch of B sequences that share a length."""

    def __init__(self, kv_cache_dtype: str = "native", track_pages: bool = False):
        if kv_cache_dtype not in KV_CACHE_DTYPES:
            raise ValueError(f"kv_cache_dtype must be one of {KV_CACHE_DTYPES}, not {kv_cache_dtype!r}")
        self.kv_cache_dtype = kv_cache_dtype
        self.track_pages = track_pages  # keep Quest page min/max (a dozen small ops per append); off unless selecting
        self.length = 0
        self.capacity = 0
        self.store = {}  # name -> tensor [B, Hkv, capacity, ...]
        self.dtype = None

    # ------------------------------------------------------------------ storage

    def _fields(self, B: int, H: int, D: int, dtype) -> dict:
        """name -> (trailing shape, dtype, initial value) of every stored tensor."""
        if self.kv_cache_dtype == "native":
            f = {"k": ((D,), dtype, 0), "v": ((D,), dtype, 0)}
        elif self.kv_cache_dtype == "int8":
            f = {n: ((D,), torch.uint8, 0) for n in ("k", "v")}
            f |= {n + s: ((1,), torch.float16, 0) for n in ("k", "v") for s in ("_scale", "_zero")}
        else:
            if D % INT4_GROUP:
                raise ValueError(f"int4 KV cache needs head_dim divisible by {INT4_GROUP}, got {D}")
            f = {n: ((D // 2,), torch.uint8, 0) for n in ("k", "v")}
            f |= {n + s: ((D // INT4_GROUP,), torch.float16, 0) for n in ("k", "v") for s in ("_scale", "_zero")}
        return f

    def _reserve(self, B: int, H: int, D: int, need: int, dtype, device) -> None:
        if need <= self.capacity:
            return
        cap = max(need, self.capacity + self.capacity // 4, 256)  # grow by a quarter: doubling wasted up to half
        cap = -(-cap // 256) * 256
        new = {}
        for name, (tail, dt, init) in self._fields(B, H, D, dtype).items():
            t = torch.full((B, H, cap) + tail, init, dtype=dt, device=device)
            if name in self.store:
                t[:, :, :self.length] = self.store[name][:, :, :self.length]
            new[name] = t
        for name, init in ((("kmin", float("inf")), ("kmax", float("-inf"))) if self.track_pages else ()):
            t = torch.full((B, H, cap // PAGE, D), init, dtype=torch.float16, device=device)
            if name in self.store:
                t[:, :, :self.capacity // PAGE] = self.store[name]
            new[name] = t
        self.store, self.capacity, self.dtype = new, cap, dtype

    def _write(self, name: str, x: torch.Tensor, start: int, end: int) -> None:
        s = self.store
        if self.kv_cache_dtype == "native":
            s[name][:, :, start:end] = x
        else:
            bits, group = (8, x.shape[-1]) if self.kv_cache_dtype == "int8" else (4, INT4_GROUP)
            codes, scale, zero = _quantize(x, bits, group)
            s[name][:, :, start:end] = codes if bits == 8 else _pack4(codes)
            s[name + "_scale"][:, :, start:end] = scale
            s[name + "_zero"][:, :, start:end] = zero

    def _update_page_bounds(self, k: torch.Tensor, start: int, end: int) -> None:
        p0, p1 = start // PAGE, -(-end // PAGE)
        pad_lo, pad_hi = start - p0 * PAGE, p1 * PAGE - end
        kf = k.float()
        lo = torch.nn.functional.pad(kf, (0, 0, pad_lo, pad_hi), value=float("inf"))
        hi = torch.nn.functional.pad(kf, (0, 0, pad_lo, pad_hi), value=float("-inf"))
        B, H, _, D = k.shape
        pmin = lo.view(B, H, p1 - p0, PAGE, D).amin(3).half()
        pmax = hi.view(B, H, p1 - p0, PAGE, D).amax(3).half()
        s = self.store
        s["kmin"][:, :, p0:p1] = torch.minimum(s["kmin"][:, :, p0:p1], pmin)
        s["kmax"][:, :, p0:p1] = torch.maximum(s["kmax"][:, :, p0:p1], pmax)

    def ensure(self, need: int) -> None:
        """Reserve capacity for `need` tokens in an allocated cache (before a fused kernel writes past its end)."""
        k = self.store["k"]
        D = k.shape[-1] * (2 if self.kv_cache_dtype == "int4" else 1)
        self._reserve(k.shape[0], k.shape[1], D, need, self.dtype, k.device)

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """k, v [B, Hkv, L, D] (K after RoPE) written at positions length .. length + L - 1."""
        B, H, L, D = k.shape
        start, end = self.length, self.length + L
        self._reserve(B, H, D, end, k.dtype, k.device)
        self._write("k", k, start, end)
        self._write("v", v, start, end)
        if self.track_pages:
            self._update_page_bounds(k, start, end)
        self.length = end

    # ------------------------------------------------------------------ reads

    def _read(self, name: str, index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """name 'k' or 'v' -> [B, Hkv, T, D] in the model dtype; index [B, Hkv, N] picks token positions."""
        s = self.store

        def take(t):
            t = t[:, :, :self.length]
            if index is None:
                return t
            return torch.gather(t, 2, index.unsqueeze(-1).expand(-1, -1, -1, t.shape[-1]))

        if self.kv_cache_dtype == "native":
            return take(s[name])
        if self.kv_cache_dtype == "int8":
            codes, group = take(s[name]), s[name].shape[-1]
        else:
            codes, group = _unpack4(take(s[name])), INT4_GROUP
        return _dequantize(codes, take(s[name + "_scale"]), take(s[name + "_zero"]), group, self.dtype)

    def read(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._read("k"), self._read("v")

    def select_pages(self, q: torch.Tensor, budget: float) -> torch.Tensor:
        """Quest: q [B, H, 1, D] -> token positions [B, Hkv, N] of the kept pages (first, last, then the highest
        bound sum_d max(q_d kmax_d, q_d kmin_d), max over the query heads sharing a KV head). N = pages kept * 16;
        positions past the end of the cache are clamped to the last token (masked by the caller)."""
        B, H, _, D = q.shape
        Hkv = self.store["kmin"].shape[1]
        n_pages = -(-self.length // PAGE)
        keep = min(n_pages, max(2, math.ceil(budget * n_pages)))
        qg = q.view(B, Hkv, H // Hkv, D).float()
        kmin = self.store["kmin"][:, :, :n_pages].float()
        kmax = self.store["kmax"][:, :, :n_pages].float()
        bound = qg.clamp(min=0) @ kmax.transpose(-1, -2) + qg.clamp(max=0) @ kmin.transpose(-1, -2)
        bound = bound.amax(2)  # [B, Hkv, P]
        bound[:, :, 0] = float("inf")
        bound[:, :, n_pages - 1] = float("inf")
        pages = bound.topk(keep, dim=-1).indices.sort(-1).values
        pos = (pages.unsqueeze(-1) * PAGE + torch.arange(PAGE, device=q.device)).flatten(-2)
        return pos

    def read_positions(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gathered K, V [B, Hkv, N, D] at pos, and a bool mask [B, Hkv, 1, N] of the positions inside the cache."""
        valid = pos < self.length
        idx = pos.clamp(max=self.length - 1)
        return self._read("k", idx), self._read("v", idx), valid.unsqueeze(2)

    def clone(self) -> "LayerKVCache":
        c = LayerKVCache(self.kv_cache_dtype, self.track_pages)
        c.length, c.capacity, c.dtype = self.length, self.capacity, self.dtype
        c.store = {n: t.clone() for n, t in self.store.items()}
        return c

    def nbytes(self) -> int:
        """Bytes held for the current length (K, V, scales; not the Quest page metadata)."""
        return sum(t[:, :, :self.length].numel() * t.element_size()
                   for n, t in self.store.items() if n not in ("kmin", "kmax"))


class KVCache:
    """All layers of one request's cache. Passed as `past_key_values` and returned by SubspaceCausalLM.forward."""

    def __init__(self, num_layers: int, kv_cache_dtype: str = "native", track_pages: bool = False):
        self.layers: List[LayerKVCache] = [LayerKVCache(kv_cache_dtype, track_pages) for _ in range(num_layers)]

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, i: int) -> LayerKVCache:
        return self.layers[i]

    @property
    def length(self) -> int:
        return self.layers[0].length if self.layers else 0

    def clone(self) -> "KVCache":
        c = KVCache(0)
        c.layers = [layer.clone() for layer in self.layers]
        return c

    def nbytes(self) -> int:
        return sum(layer.nbytes() for layer in self.layers)


def kv_bytes_per_token(num_layers: int, num_kv_heads: int, head_dim: int, kv_cache_dtype: str = "native",
                       native_bytes: int = 2) -> int:
    """K and V bytes one token adds to the cache across all layers (scales included, Quest metadata excluded)."""
    per_head = {"native": head_dim * native_bytes,
                "int8": head_dim + 4,
                "int4": head_dim // 2 + 4 * (head_dim // INT4_GROUP)}[kv_cache_dtype]
    return 2 * num_layers * num_kv_heads * per_head


class DecodeBatchLayer(NamedTuple):
    """One layer of a batched decode step on CUDA: each row's cache, and their device table
    [B, 8] = (k, v, k_scale, k_zero, v_scale, v_zero data pointers, length before the step, capacity)."""
    rows: List[LayerKVCache]
    table: torch.Tensor
