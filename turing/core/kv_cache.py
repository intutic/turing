"""
Paged KV cache for the serving path: one fixed pool per model, a page table per request.

Pool layout per layer (`KVPool`): `k`, `v` [num_pages, Hkv, 16, D'], plus for low-bit storage `k_scale`, `k_zero`,
`v_scale`, `v_zero` [num_pages, Hkv, 16, G], and for Quest page selection `kmin`, `kmax` [num_pages, Hkv, D] (the
elementwise min and max of K over each 16-token page, from the values before quantization). A page id addresses the
same slot in every layer. Storage modes (`kv_cache_dtype`):
- "native": the model's dtype (D' = D).
- "int8":   asymmetric round-to-nearest per token and KV head, fp16 scale and zero (D' = D, G = 1).
- native_layers: layers whose K and V a low-bit pool stores in the model dtype anyway (`k_hi`, `v_hi`
  [len(native_layers), num_pages, Hkv, 16, D]); set when calibration finds a layer the low-bit format damages.
- "int4":   asymmetric round-to-nearest per group of 32 values along D, fp16 scale and zero per group, two codes per
            byte (D' = D / 2, G = D / 32).
K is stored after RoPE. A request (`KVCache`) owns a row of the pool's device page table and the pages it lists; a
clone shares its full pages copy-on-write (refcounted) and copies only the partial last page, so a fork costs one page
per layer. At most one partial page per request is unused, instead of a per-request capacity margin.

The methods here are the reference path (any device). On CUDA, `turing/kernels/triton_kv_decode.py` reads and
writes the same layout with fused kernels.
"""

from typing import Optional, Sequence, List, Tuple
import math
import torch

PAGE = 16
INT4_GROUP = 32
KV_CACHE_DTYPES = ("native", "int8", "int4")


def resolve_kv_cache_dtype(choice: str, device) -> str:
    """'auto' = int8 on CUDA (measured free in quality, fused kernels read it directly), native elsewhere (the
    reference path dequantizes what it reads each step, so int8 there saves memory but costs speed)."""
    return ("int8" if torch.device(device).type == "cuda" else "native") if choice == "auto" else choice


def kv_bytes_per_token(num_layers: int, num_kv_heads: int, head_dim: int, kv_cache_dtype: str = "native",
                       native_bytes: int = 2) -> int:
    """K and V bytes one token adds to the cache across all layers (scales included, Quest metadata excluded)."""
    per_head = {"native": head_dim * native_bytes,
                "int8": head_dim + 4,
                "int4": head_dim // 2 + 4 * (head_dim // INT4_GROUP)}[kv_cache_dtype]
    return 2 * num_layers * num_kv_heads * per_head


class KVPoolExhausted(RuntimeError):
    """The pool has no free page. Raise ModelConfig.kv_pool_gib / kv_pool_fraction, or admit fewer requests."""


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


class KVPool:
    """Every layer's K and V pages for one model, plus the page tables of the requests using it."""

    # The pool lives for the whole process and is written both inside inference mode (forward passes) and outside it
    # (admission, page tables): its tensors are always created as normal tensors.
    @torch.inference_mode(False)
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int, num_pages: int, dtype, device,
                 kv_cache_dtype: str = "native", track_pages: bool = False, max_tokens_per_request: int = 131072,
                 native_layers: Sequence[int] = ()):
        if kv_cache_dtype not in KV_CACHE_DTYPES:
            raise ValueError(f"kv_cache_dtype must be one of {KV_CACHE_DTYPES}, not {kv_cache_dtype!r}")
        if kv_cache_dtype == "int4" and head_dim % INT4_GROUP:
            raise ValueError(f"int4 KV cache needs head_dim divisible by {INT4_GROUP}, got {head_dim}")
        self.num_layers, self.num_kv_heads, self.head_dim = num_layers, num_kv_heads, head_dim
        self.num_pages, self.dtype, self.device = num_pages, dtype, torch.device(device)
        self.kv_cache_dtype, self.track_pages = kv_cache_dtype, track_pages
        L, P, H, D = num_layers, num_pages, num_kv_heads, head_dim
        shape = lambda tail: (L, P, H, PAGE) + tail
        # uninitialized: a slot is read only after it is written (reads are bounded by each request's length or
        # masked), and untouched host pages are never backed by memory
        if kv_cache_dtype == "native":
            self.k, self.v = (torch.empty(shape((D,)), dtype=dtype, device=device) for _ in range(2))
            self.k_scale = self.k_zero = self.v_scale = self.v_zero = None
        else:
            width, groups = (D, 1) if kv_cache_dtype == "int8" else (D // 2, D // INT4_GROUP)
            self.k, self.v = (torch.empty(shape((width,)), dtype=torch.uint8, device=device) for _ in range(2))
            self.k_scale, self.k_zero, self.v_scale, self.v_zero = (
                torch.empty(shape((groups,)), dtype=torch.float16, device=device) for _ in range(4))
        # layers a low-bit pool keeps in the model dtype (their slots in the low-bit tensors stay unused)
        self.native_layers = tuple(sorted(set(native_layers))) if kv_cache_dtype != "native" else ()
        self._hi = {layer: i for i, layer in enumerate(self.native_layers)}
        if self._hi:
            self.k_hi, self.v_hi = (torch.empty((len(self._hi), P, H, PAGE, D), dtype=dtype, device=device)
                                    for _ in range(2))
        else:
            self.k_hi = self.v_hi = None
        if track_pages:
            self.kmin = torch.full((L, P, H, D), float("inf"), dtype=torch.float16, device=device)
            self.kmax = torch.full((L, P, H, D), float("-inf"), dtype=torch.float16, device=device)
        else:
            self.kmin = self.kmax = None
        self.table_width = -(-max_tokens_per_request // PAGE)
        # 256 request rows from the start (4 bytes per page of a request each): growing the table moves it, and every
        # decode graph captured before that is captured again (0.1 s each, results in a stall after the first burst)
        self.page_table = torch.zeros((256, self.table_width), dtype=torch.int32, device=device)
        self._free_pages = list(range(P - 1, -1, -1))  # a stack: pop() hands out low page ids first
        self._refs = [0] * P
        self.page_limit = P  # pages that may be in use at once (lowered by ElasticMemoryBudgetManager)
        self._free_slots = list(range(255, -1, -1))

    @staticmethod
    def cuda_budget(free: float, total: float, fraction: Optional[float] = None) -> float:
        """Bytes of `free` CUDA memory for the pool: `fraction` of it when given (ModelConfig's default is 0.6, which
        leaves room for another model in the process), else all but a reserve for what steps allocate besides the
        pool (prefill activations, attention workspaces, CUDA graphs, adapters loaded later): a tenth of the device's
        memory, at least 1.5 GiB, and never more than half of what is free. That is `turing serve`'s default, where
        one model owns the GPU: on an L4 the memory in use outside the weights and the pool peaked at 0.4-1.2 GiB
        across the benchmark sessions, and 60% of the free memory left 4-7 GiB of its 22.5 unused."""
        if fraction is not None:
            return fraction * free
        return free - min(max(0.1 * total, 1.5 * 2 ** 30), free / 2)

    @classmethod
    def for_model(cls, config, dtype, device) -> "KVPool":
        """Sized from config.kv_pool_gib if set, else from the free CUDA memory (cuda_budget; counting memory PyTorch
        has cached but not used), else off CUDA 4 GiB or what 64 full-length requests need, whichever is less
        (Windows commits the whole allocation up front, and MPS shares the machine's memory)."""
        device = torch.device(device)
        D, H, L = config.head_dim, config.num_kv_heads, config.num_layers
        native = tuple(getattr(config, "kv_native_layers", None) or ()) if config.kv_cache_dtype != "native" else ()
        page_bytes = PAGE * kv_bytes_per_token(L, H, D, config.kv_cache_dtype, torch.finfo(dtype).bits // 8)
        page_bytes += PAGE * kv_bytes_per_token(len(native), H, D, "native", torch.finfo(dtype).bits // 8)
        track = config.kv_page_budget < 1.0
        if track:
            page_bytes += 2 * L * H * D * 2
        if config.kv_pool_gib:
            budget = config.kv_pool_gib * 2 ** 30
        elif device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
            budget = cls.cuda_budget(free, total, config.kv_pool_fraction)
        else:
            budget = min(4 * 2 ** 30, 64 * -(-config.max_position_embeddings // PAGE) * page_bytes)
        return cls(L, H, D, max(1, int(budget // page_bytes)), dtype, device, config.kv_cache_dtype, track,
                   native_layers=native)

    # ------------------------------------------------------------------ pages and slots

    @property
    def num_free_pages(self) -> int:
        return len(self._free_pages)

    def fields(self) -> List[torch.Tensor]:
        return [t for t in (self.k, self.v, self.k_scale, self.k_zero, self.v_scale, self.v_zero, self.k_hi, self.v_hi)
                if t is not None]

    def layer_fields(self, layer: int, which: str):
        """(codes, scale, zero, storage) of one layer's K or V ("k" / "v"): tensors indexed [page, head, slot, ...],
        scale and zero None for native storage; storage is "native", "int8" or "int4"."""
        if layer in self._hi:
            return (self.k_hi if which == "k" else self.v_hi)[self._hi[layer]], None, None, "native"
        codes, scale, zero = ((self.k, self.k_scale, self.k_zero) if which == "k"
                              else (self.v, self.v_scale, self.v_zero))
        if self.kv_cache_dtype == "native":
            return codes[layer], None, None, "native"
        return codes[layer], scale[layer], zero[layer], self.kv_cache_dtype

    def _store(self, layer: int, which: str, phys: torch.Tensor, slot: torch.Tensor, xt: torch.Tensor) -> None:
        """xt [N, Hkv, D] into (page phys[n], slot slot[n]) of one layer's K or V."""
        codes_t, scale_t, zero_t, storage = self.layer_fields(layer, which)
        if storage == "native":
            codes_t[phys, :, slot] = xt.to(codes_t.dtype)
            return
        bits, group = (8, self.head_dim) if storage == "int8" else (4, INT4_GROUP)
        codes, scale, zero = _quantize(xt, bits, group)
        codes_t[phys, :, slot] = codes if bits == 8 else _pack4(codes)
        scale_t[phys, :, slot] = scale
        zero_t[phys, :, slot] = zero

    def page_bytes(self) -> int:
        """K, V and scale bytes of one page across all layers (Quest metadata excluded)."""
        return sum(t[:, :1].numel() * t.element_size() for t in self.fields())

    @property
    def page_size(self) -> int:
        return PAGE

    @property
    def max_total_pages(self) -> int:
        return self.num_pages

    @property
    def used_pages(self) -> int:
        return self.num_pages - len(self._free_pages)

    @property
    def free_pages(self) -> int:
        """Pages alloc_pages can still hand out."""
        return max(0, min(len(self._free_pages), self.page_limit - self.used_pages))

    def adjust_active_capacity(self, new_max_pages: int) -> int:
        """Cap the pages in use at once (no reallocation; pages already in use stay valid)."""
        self.page_limit = max(1, min(new_max_pages, self.num_pages))
        return self.page_limit

    def get_stats(self) -> dict:
        used = self.used_pages
        return {"total_pages": self.page_limit, "max_physical_pages": self.num_pages, "used_pages": used,
                "free_pages": max(0, self.page_limit - used), "page_size": PAGE,
                "kv_cache_dtype": self.kv_cache_dtype, "native_layers": list(self.native_layers),
                "bytes_per_page": self.page_bytes(),
                "utilization_pct": round(used / self.page_limit * 100.0, 2)}

    def alloc_pages(self, n: int) -> List[int]:
        if n > self.free_pages:
            raise KVPoolExhausted(f"KV pool has {self.free_pages} free pages of "
                                  f"{self.page_limit}, {n} needed")
        pages = [self._free_pages.pop() for _ in range(n)]
        for p in pages:
            self._refs[p] = 1
        if self.track_pages and pages:
            idx = torch.tensor(pages, device=self.device)
            self.kmin[:, idx] = float("inf")
            self.kmax[:, idx] = float("-inf")
        return pages

    def incref(self, pages: List[int]) -> None:
        for p in pages:
            self._refs[p] += 1

    def decref(self, pages: List[int]) -> None:
        for p in pages:
            self._refs[p] -= 1
            if self._refs[p] == 0:
                self._free_pages.append(p)

    @torch.inference_mode(False)
    def new_slot(self) -> int:
        if not self._free_slots:  # grow the page table by doubling its rows (a few KB; rare)
            rows = self.page_table.shape[0]
            self.page_table = torch.cat([self.page_table, torch.zeros_like(self.page_table)])
            self._free_slots = list(range(2 * rows - 1, rows - 1, -1))
        return self._free_slots.pop()

    def release_slot(self, slot: int) -> None:
        self._free_slots.append(slot)

    def scratch_slot(self) -> int:
        """A page-table row with one page of its own, for the padding rows of a CUDA-graph decode step (they write and
        read only there). Reserved on first use."""
        if getattr(self, "_scratch", None) is None:
            slot, page = self.new_slot(), self.alloc_pages(1)[0]
            self.page_table[slot, 0] = page
            self._scratch = slot
        return self._scratch

    def copy_page(self, src: int, dst: int) -> None:
        for t in self.fields() + ([self.kmin, self.kmax] if self.track_pages else []):
            t[:, dst] = t[:, src]

    # ------------------------------------------------------------------ reference reads and writes

    def append(self, layer: int, cache: "KVCache", k: torch.Tensor, v: torch.Tensor) -> None:
        """k, v [1, Hkv, L, D] (K after RoPE) at positions cache.length .. + L - 1 of one request; the caller has
        reserved the pages and advances cache.length after the last layer."""
        L = k.shape[2]
        idx = cache._append_idx
        if idx is None or idx[0] != (cache.length, L):
            # each token's page and slot: the same in every layer of a forward (a request's pages never move), and
            # computing them took 9% of the host's time in a 512-token prefill
            pos = torch.arange(cache.length, cache.length + L, device=self.device)
            idx = cache._append_idx = ((cache.length, L), self.page_table[cache.slot, pos // PAGE].long(), pos % PAGE)
        _, phys, slot = idx
        for which, x in (("k", k), ("v", v)):
            self._store(layer, which, phys, slot, x[0].transpose(0, 1))  # [L, Hkv, D]
        if self.track_pages:
            kt = k[0].transpose(0, 1).half()
            idx = phys[:, None, None].expand_as(kt)
            self.kmin[layer].scatter_reduce_(0, idx, kt, reduce="amin")
            self.kmax[layer].scatter_reduce_(0, idx, kt, reduce="amax")

    def append_batch(self, layer: int, slots: torch.Tensor, starts: torch.Tensor, k: torch.Tensor,
                     v: torch.Tensor) -> None:
        """One token for each of B requests: k, v [B, Hkv, 1, D] at position starts[b] of page-table row slots[b]
        (device tensors; pages already reserved). The batched form of append, for decode off CUDA."""
        pos = starts.long()
        phys = self.page_table[slots.long(), pos // PAGE].long()
        slot = pos % PAGE
        for which, x in (("k", k), ("v", v)):
            self._store(layer, which, phys, slot, x[:, :, 0])  # [B, Hkv, D]
        if self.track_pages:
            kt = k[:, :, 0].half()
            idx = phys[:, None, None].expand_as(kt)
            self.kmin[layer].scatter_reduce_(0, idx, kt, reduce="amin")
            self.kmax[layer].scatter_reduce_(0, idx, kt, reduce="amax")

    def read_batch(self, layer: int, slots: torch.Tensor, lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor,
                                                                                         torch.Tensor]:
        """K, V [B, Hkv, T, D] for B requests, padded to the longest (T = its pages * 16), and valid [B, T]: one
        gather per field for the whole batch instead of one read per request."""
        width = -(-max(lengths) // PAGE)
        phys = self.page_table[slots.long(), :width].long()  # [B, W]; entries past a row's pages are masked below
        B = phys.shape[0]
        out = []
        for which in ("k", "v"):
            codes_t, scale_t, zero_t, storage = self.layer_fields(layer, which)
            g = lambda t: t[phys].permute(0, 2, 1, 3, 4).flatten(2, 3)  # [B, W, Hkv, 16, w] -> [B, Hkv, W*16, w]
            codes = g(codes_t)
            if storage != "native":
                group = self.head_dim if storage == "int8" else INT4_GROUP
                if storage == "int4":
                    codes = _unpack4(codes)
                codes = _dequantize(codes, g(scale_t), g(zero_t), group, self.dtype)
            out.append(codes)
        pos = torch.arange(width * PAGE, device=self.device)
        valid = pos[None, :] < torch.tensor(lengths, device=self.device)[:, None]
        k, v = (x.masked_fill(~valid[:, None, :, None], 0) for x in out)  # unwritten slots may hold NaN
        return k, v, valid

    def _gather(self, t: torch.Tensor, phys: torch.Tensor) -> torch.Tensor:
        """One layer's field t [pages, Hkv, 16, w]: phys [N] -> [Hkv, N * 16, w]; phys [Hkv, N] (one page list per KV
        head) -> the same."""
        if phys.dim() == 1:
            return t[phys].transpose(0, 1).flatten(1, 2)          # [N, Hkv, 16, w] -> [Hkv, N * 16, w]
        heads = torch.arange(phys.shape[0], device=self.device)[:, None]
        return t[phys, heads].flatten(1, 2)                        # [Hkv, N, 16, w] -> [Hkv, N * 16, w]

    def _decode_field(self, layer: int, phys: torch.Tensor, which: str) -> torch.Tensor:
        codes_t, scale_t, zero_t, storage = self.layer_fields(layer, which)
        codes = self._gather(codes_t, phys)
        if storage == "native":
            return codes
        group = self.head_dim if storage == "int8" else INT4_GROUP
        if storage == "int4":
            codes = _unpack4(codes)
        return _dequantize(codes, self._gather(scale_t, phys), self._gather(zero_t, phys), group, self.dtype)

    def read(self, layer: int, cache: "KVCache", length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """K, V [1, Hkv, length, D] of one request, in the model dtype."""
        phys = self.page_table[cache.slot, :-(-length // PAGE)].long()
        k, v = (self._decode_field(layer, phys, w)[:, :length].unsqueeze(0) for w in ("k", "v"))
        return k, v

    def selection_plan(self, caches: List["KVCache"], lengths: List[int], budget: float, min_tokens: int) -> tuple:
        """What Quest needs per step, shared by every layer: table width W, pages kept K, and on the device each row's
        page count and keep count [B] and its physical pages [B, W]. lengths are after this step's append."""
        width = -(-max(lengths) // PAGE)
        n_pages = [-(-n // PAGE) for n in lengths]
        keep = [p if n < min_tokens else min(p, max(2, math.ceil(budget * p))) for n, p in zip(lengths, n_pages)]
        meta = torch.tensor([n_pages, keep, [c.slot for c in caches]], device=self.device)
        phys = self.page_table[meta[2], :width].long()
        return width, max(keep), meta[0], meta[1], phys

    def selection_plan_device(self, slots: torch.Tensor, starts: torch.Tensor, width: int, budget: float,
                              min_tokens: int) -> tuple:
        """selection_plan with static shapes and no host values, for a CUDA graph: the table width W is fixed (a
        context bucket) and K = the most pages any row can keep at that width. starts [B] are lengths before this
        step's append."""
        lengths = starts.long() + 1
        n_pages = (lengths + PAGE - 1) // PAGE
        quota = torch.clamp(torch.ceil(budget * n_pages.float()).long(), min=2)
        keep = torch.where(lengths < min_tokens, n_pages, torch.minimum(n_pages, quota))
        k_max = min(width, max(math.ceil(budget * width), 2, -(-min_tokens // PAGE)))
        phys = self.page_table[slots.long(), :width].long()
        return width, k_max, n_pages, keep, phys

    def select_pages_planned(self, layer: int, q: torch.Tensor, plan: tuple) -> torch.Tensor:
        """Quest for a batch: q [B, H, 1, D] -> logical page indices [B, Hkv, K], -1 = none. Per row and KV head: the
        first and the last page, then the pages with the highest bound sum_d max(q_d kmax_d, q_d kmin_d) (max over
        the query heads sharing the KV head), up to its keep count. All on the device."""
        width, k_max, n_pages, keep, phys = plan
        B, H, _, D = q.shape
        Hkv = self.num_kv_heads
        kmin, kmax = self.kmin[layer][phys].float(), self.kmax[layer][phys].float()     # [B, W, Hkv, D]
        qg = q.view(B, Hkv, H // Hkv, D).float()
        bound = (torch.einsum("bhgd,bwhd->bhgw", qg.clamp(min=0), kmax)
                 + torch.einsum("bhgd,bwhd->bhgw", qg.clamp(max=0), kmin)).amax(2)     # [B, Hkv, W]
        lp = torch.arange(width, device=self.device)
        inside = (lp < n_pages[:, None])[:, None]                                      # [B, 1, W]
        forced = ((lp == 0) | (lp == n_pages[:, None] - 1))[:, None]
        bound = bound.masked_fill(~inside, float("-inf")).masked_fill(forced & inside, float("inf"))
        top = bound.topk(k_max, dim=-1)
        ok = (torch.arange(k_max, device=self.device) < keep[:, None, None]) & (top.values > float("-inf"))
        return torch.where(ok, top.indices, torch.full_like(top.indices, -1))

    def select_pages(self, layer: int, caches: List["KVCache"], lengths: List[int], q: torch.Tensor,
                     budget: float, min_tokens: int) -> torch.Tensor:
        """Quest for a batch in one call (selection_plan + select_pages_planned); a row shorter than min_tokens keeps
        every page."""
        return self.select_pages_planned(layer, q, self.selection_plan(caches, lengths, budget, min_tokens))

    def read_selected(self, layer: int, cache: "KVCache", sel: torch.Tensor, length: int):
        """Reference read of selected logical pages sel [Hkv, K] of one request -> K, V [1, Hkv, K * 16, D] and a
        bool mask [1, Hkv, 1, K * 16] of the positions inside the cache."""
        phys = self.page_table[cache.slot, sel.clamp(min=0)].long()                     # [Hkv, K]
        pos = (sel[..., None] * PAGE + torch.arange(PAGE, device=self.device)).flatten(-2)
        valid = (sel[..., None] >= 0).expand(-1, -1, PAGE).flatten(-2) & (pos < length)
        k, v = (self._decode_field(layer, phys, w).masked_fill(~valid[..., None], 0).unsqueeze(0) for w in ("k", "v"))
        return k, v, valid[None, :, None]  # unwritten slots zeroed: a masked 0 * NaN would still poison attention


class KVCache:
    """One request's cache in a pool: a row of the pool's page table and a length. Passed as `past_key_values` and
    returned by SubspaceCausalLM.forward. Frees its pages when freed or garbage collected."""

    def __init__(self, pool: KVPool):
        self.pool = pool
        self.slot = pool.new_slot()
        self.pages: List[int] = []
        self.length = 0
        self.rope_len = 0  # the length that picks length-dependent RoPE frequencies in prefill (its prompt's length)
        self._append_idx = None  # KVPool.append's page and slot of each token of the chunk being appended

    def __len__(self) -> int:
        return self.pool.num_layers

    def reserve(self, tokens: int) -> None:
        """Make the cache hold `tokens` positions (allocates pages, writes their ids into the device page table)."""
        need = -(-tokens // PAGE) - len(self.pages)
        if need <= 0:
            return
        if len(self.pages) + need > self.pool.table_width:
            raise KVPoolExhausted(f"a request may hold at most {self.pool.table_width * PAGE} tokens")
        new = self.pool.alloc_pages(need)
        ids = torch.tensor(new, dtype=torch.int32)
        row = self.pool.page_table[self.slot, len(self.pages):len(self.pages) + need]
        # on CUDA through pinned memory without waiting: a plain copy waits for every kernel queued before it, which
        # is the decode step the engine launched ahead
        row.copy_(ids.pin_memory(), non_blocking=True) if row.is_cuda else row.copy_(ids)
        self.pages += new

    def clone(self, share: bool = True) -> "KVCache":
        """share=True: full pages are shared copy-on-write and only the partial last page is copied (a fork);
        share=False: every page is copied (an independent request with the same content)."""
        c = KVCache(self.pool)
        full = self.length // PAGE if share else 0
        c.pages = self.pages[:full]
        self.pool.incref(c.pages)
        rest = self.pool.alloc_pages(len(self.pages) - full)
        for src, dst in zip(self.pages[full:], rest):
            self.pool.copy_page(src, dst)
        c.pages += rest
        if c.pages:
            self.pool.page_table[c.slot, :len(c.pages)] = torch.tensor(c.pages, dtype=torch.int32)
        c.length, c.rope_len = self.length, self.rope_len
        return c

    def free(self) -> None:
        if self.pages is not None:
            self.pool.decref(self.pages)
            self.pool.release_slot(self.slot)
            self.pages = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    def nbytes(self) -> int:
        """Bytes of the pages this request holds (K, V and scales across all layers; shared pages counted in full)."""
        return len(self.pages) * self.pool.page_bytes()
