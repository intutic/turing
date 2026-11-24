"""
EAGLE-3 draft head (Li et al. 2025, arXiv:2503.01840), inference only: a one-layer decoder that predicts the target
model's next tokens from the target's own hidden states, for speculative decoding. The engine verifies its drafts
in one target forward, so the output is exactly the target's (turing.serving.engine).

The head reads the target's residual stream after three layers (by default 2, L // 2 and L - 3 of L), fused by a
bias-free linear layer `fc` (3H -> H). At a verified position p its input is the pair (embedding of the token at
p + 1, fused feature at p); its output there stands in for the target's feature at p + 1, and `lm_head(norm(.))`
of it scores the token at p + 2 over a reduced draft vocabulary (`d2t` maps a draft id to its target id as an
offset). Further drafts feed the head its own last output with the embedding of its last draft. The layer is
Llama's, except that attention reads the concatenation [input_layernorm(embedding), hidden_norm(feature)] (so q, k
and v take 2H inputs) and the attention residual is the feature (normed first when `norm_before_residual`).

Loads the published checkpoints in both layouts: the EAGLE authors' (`midlayer.*`, flat Llama config) and the
speculators library's (`layers.0.*`, `transformer_layer_config`). Variants with more than one layer, extra norms
around `fc` or attention biases are refused. Drafting is a greedy chain (no draft tree).
"""

import json
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.kv_cache import KVCache, KVPool, KVPoolExhausted
from ..core.rope import RotaryEmbedding, apply_rotary_pos_emb
from .causal_lm import RMSNorm, host_tensor

# keys and values [1, kv_heads, positions, head_dim]; or, once use_pool() was called, a KVCache of the head's own pool
HeadCache = Optional[Tuple[torch.Tensor, torch.Tensor]]


def default_aux_layers(num_layers: int) -> Tuple[int, int, int]:
    """The target layers whose outputs the head reads: id k is the residual stream after k layers."""
    return (2, num_layers // 2, num_layers - 3)


class _StepGraph:
    """One step of the head's pool path for up to `rows` rows, captured in a CUDA graph: a step is some 45 small
    kernels, which launched one by one take about as long again as the head's weights take to read. first: the
    verified positions (tokens and the target's tapped states in, layer outputs out); else a chain step (the previous
    outputs in; the next draft token and the layer outputs for it out). Padding rows write to the pool's scratch page."""

    def __init__(self, head: "Eagle3Head", embed: nn.Module, rows: int, first: bool):
        w, pool = head.q_proj.weight, head.pool
        self.rows, self.first, self.scratch = rows, first, pool.scratch_slot()
        self.key = (id(embed), pool.page_table.data_ptr())
        self.meta = torch.zeros(3, rows, dtype=torch.long, device=w.device)  # positions, page-table rows, starts
        self.meta[1] = self.scratch
        self.tokens = torch.zeros(rows, dtype=torch.long, device=w.device)
        self.feat = torch.zeros(rows, head.fc.in_features if first else head.fc.out_features, dtype=w.dtype,
                                device=w.device)

        def step():
            slots, starts = self.meta[1].to(torch.int32), self.meta[2].to(torch.int32)
            if first:
                return None, head._step(embed(self.tokens), head.fc(self.feat), self.meta[0], slots, starts)
            tok = head.lm_head(head.norm(self.feat)).argmax(-1)
            tok = tok + head.d2t[tok]
            return tok, head._step(embed(tok), self.feat, self.meta[0], slots, starts)
        for _ in range(2):  # compilation and library set-up outside the capture; every row is on the scratch page
            step()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.tok, self.out = step()

    def run(self, feat: torch.Tensor, positions: List[int], slots: List[int], starts: List[int],
            tokens: Optional[torch.Tensor] = None, shift: Optional[torch.Tensor] = None
            ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        R, pad = len(positions), self.rows - len(positions)
        # through pinned memory: a plain copy waits for the head step still running, and a chain of k drafts is k
        # steps the host can queue one behind the other
        self.meta.copy_(host_tensor([positions + [0] * pad, slots + [self.scratch] * pad, starts + [0] * pad],
                                    self.meta.dtype, self.meta.device))
        if shift is not None:  # long [R] on the device: added to the rows' positions and cache positions
            self.meta[0, :R] += shift
            self.meta[2, :R] += shift
        self.feat[:R].copy_(feat)
        if tokens is not None:
            self.tokens[:R].copy_(tokens)
        self.graph.replay()
        return (self.tok[:R] if self.tok is not None else None), self.out[:R]


class Eagle3Head(nn.Module):
    def __init__(self, hidden: int, intermediate: int, num_heads: int, num_kv_heads: int, head_dim: int,
                 draft_vocab: int, target_hidden: Optional[int] = None, rms_norm_eps: float = 1e-6,
                 rope_theta: float = 10000.0, max_positions: int = 4096, norm_before_residual: bool = False,
                 aux_layers: Optional[Tuple[int, ...]] = None):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.norm_before_residual, self.aux_layers = norm_before_residual, aux_layers
        self.fc = nn.Linear(3 * (target_hidden or hidden), hidden, bias=False)
        self.input_layernorm = RMSNorm(hidden, rms_norm_eps)
        self.hidden_norm = RMSNorm(hidden, rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden, rms_norm_eps)
        self.norm = RMSNorm(hidden, rms_norm_eps)
        self.q_proj = nn.Linear(2 * hidden, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(2 * hidden, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(2 * hidden, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.lm_head = nn.Linear(hidden, draft_vocab, bias=False)
        self.register_buffer("d2t", torch.zeros(draft_vocab, dtype=torch.long))
        self.rope = RotaryEmbedding(head_dim, max_positions, rope_theta)
        self.pool: Optional[KVPool] = None  # see use_pool
        self.graphs: Optional[dict] = None  # (first step?, rows) -> _StepGraph, when the pool's steps are captured

    def _rows(self, emb: torch.Tensor, feat: torch.Tensor, positions: torch.Tensor):
        """Rows of (token embedding, feature) [R, H] at positions [R], of any number of requests: the attention
        residual [R, H] and each row's query [R, heads, D], key and value [R, kv_heads, D], rotated to its position."""
        if self.norm_before_residual:
            residual = hidden = self.hidden_norm(feat)
        else:
            residual, hidden = feat, self.hidden_norm(feat)
        x = torch.cat([self.input_layernorm(emb), hidden], dim=-1)
        R = x.shape[0]
        q = self.q_proj(x).view(R, self.num_heads, 1, self.head_dim)
        k = self.k_proj(x).view(R, self.num_kv_heads, 1, self.head_dim)
        v = self.v_proj(x).view(R, self.num_kv_heads, self.head_dim)
        cos, sin = self.rope.compute_freqs_device(positions)
        q, k = apply_rotary_pos_emb(q, k, cos[:, None, None, :].to(x.dtype), sin[:, None, None, :].to(x.dtype))
        return residual, q[:, :, 0], k[:, :, 0], v

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """One request: queries [T, heads, D] of its last T positions over its keys and values [1, kv_heads, S, D]
        (those T positions last) -> [T, heads * D]. Each key head serves heads / kv_heads query heads, taken as extra
        query rows so that keys and values are not copied per head."""
        T, S, rep = q.shape[0], k.shape[2], self.num_heads // self.num_kv_heads
        q = q.view(T, self.num_kv_heads, rep, self.head_dim).permute(1, 2, 0, 3).reshape(1, self.num_kv_heads, rep * T, -1)
        mask = None if T == 1 else torch.ones(T, S, dtype=torch.bool, device=q.device).tril(S - T).repeat(rep, 1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # [1, kv_heads, rep * T, D]
        return out.view(self.num_kv_heads, rep, T, self.head_dim).permute(2, 0, 1, 3).reshape(T, -1)

    def _out(self, attn: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """The layer's outputs before the final norm for rows of attention output [R, heads * D]."""
        residual = self.o_proj(attn) + residual
        x = self.post_attention_layernorm(residual)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)) + residual

    def _layer(self, emb: torch.Tensor, feat: torch.Tensor, start: int, cache: HeadCache
               ) -> Tuple[torch.Tensor, torch.Tensor, HeadCache]:
        """T consecutive positions from `start`: embeddings and features [T, H] -> (draft logits of the last position
        [draft_vocab], the layer's outputs before the final norm [T, H], the cache extended by the T positions)."""
        T = emb.shape[0]
        residual, q, k, v = self._rows(emb, feat, start + torch.arange(T, device=emb.device))
        k, v = k.transpose(0, 1)[None], v.transpose(0, 1)[None]
        if cache is not None:
            k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
        out = self._out(self._attend(q, k, v), residual)
        return self.lm_head(self.norm(out[-1])), out, (k, v)

    @torch.inference_mode()
    def extend(self, embed: nn.Module, aux: torch.Tensor, next_tokens: torch.Tensor, start: int, cache: HeadCache
               ) -> HeadCache:
        """Take in T verified positions without drafting (a prompt chunk that is not the last): aux [T, 3H] are the
        target's tapped states at positions start .. start + T - 1 and next_tokens [T] the tokens that follow them."""
        return self._layer(embed(next_tokens), self.fc(aux), start, cache)[2]

    def use_pool(self, tokens: int) -> bool:
        """Keep the head's keys and values in a paged pool of its own (one layer, room for `tokens` positions) and
        attend with the engine's decode kernel: a head step then has the same shapes whatever the requests' lengths,
        which the SDPA path over per-request tensors does not. On CUDA where that kernel runs; False elsewhere."""
        from ..kernels.triton_kv_decode import pool_decode_supported
        w = self.q_proj.weight
        pool = KVPool(1, self.num_kv_heads, self.head_dim, -(-tokens // 16) + 1, w.dtype, w.device)
        if pool_decode_supported(pool, w):
            self.pool, self.graphs = pool, {}
        return self.pool is not None

    BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)  # rows of a captured step, as the target's decode graphs

    def _run(self, embed: nn.Module, first: bool, feat: torch.Tensor, positions: List[int], slots: List[int],
             starts: List[int], tokens: Optional[torch.Tensor] = None, shift: Optional[torch.Tensor] = None):
        """One step of the pool path, (draft tokens of a chain step or None, layer outputs [R, H]): a replay of the
        step's CUDA graph when graphs are on and the rows fit one, else the kernels launched one by one. shift: long
        [R] on the device, added to positions and starts (rows whose place only the device knows yet)."""
        R = len(positions)
        if self.graphs is not None and R <= self.BUCKETS[-1] and not torch.is_grad_enabled():
            rows = next(b for b in self.BUCKETS if b >= R)
            graph = self.graphs.get((first, rows))
            if graph is None or graph.key != (id(embed), self.pool.page_table.data_ptr()):
                graph = self.graphs[(first, rows)] = _StepGraph(self, embed, rows, first)
            return graph.run(feat, positions, slots, starts, tokens, shift)
        dev = feat.device
        tok = None
        if first:
            emb, feat = embed(tokens), self.fc(feat)
        else:
            tok = self.lm_head(self.norm(feat)).argmax(-1)
            tok = tok + self.d2t[tok]
            emb = embed(tok)
        positions, starts = host_tensor(positions, torch.long, dev), host_tensor(starts, torch.long, dev)
        if shift is not None:
            positions, starts = positions + shift, starts + shift
        return tok, self._step(emb, feat, positions, host_tensor(slots, torch.int32, dev), starts.to(torch.int32))

    def _step(self, emb: torch.Tensor, feat: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor,
              starts: torch.Tensor) -> torch.Tensor:
        """Pool path: one row per (request, position), position starts[r] of page-table row slots[r] (several rows
        of one request take consecutive positions, each reading up to its own). Appends the rows' keys and values
        and returns the layer's outputs before the final norm [R, H]."""
        from ..kernels.triton_kv_decode import triton_pool_decode
        residual, q, k, v = self._rows(emb, feat, positions)
        attn = triton_pool_decode(self.pool, 0, q[:, :, None, :], k[:, :, None, :], v[:, :, None, :], slots, starts)
        return self._out(attn.reshape(emb.shape[0], -1), residual)

    @torch.inference_mode()
    def _draft_many_pool(self, embed: nn.Module, work: list, host: bool = True,
                         accepted: Optional[torch.Tensor] = None) -> Optional[list]:
        """accepted (long [len(work)] on the device): the rows of each request are every position of a verify step
        that is still in flight, of which the first accepted[r] + 1 turn out verified. All of them are taken in
        (a row reads up to its own position, and those past the verified ones are written over by what follows);
        the chain starts from row accepted[r], one position after it. The caches' lengths are left for the caller
        to set when it has read the step. None, with nothing launched, if a request's cache has no room."""
        dev, lens, ks, caches = work[0][0].device, [w[0].shape[0] for w in work], [w[3] for w in work], []
        for (_, _, _, n, cache), T in zip(work, lens):
            cache = cache if isinstance(cache, KVCache) else KVCache(self.pool)
            try:
                cache.reserve(cache.length + T + max(n - 1, 0))  # the verified positions and the drafts fed back
            except KVPoolExhausted:  # no room: this request goes without the head (its cache starts again later)
                cache.free()
                cache = None
            caches.append(cache)
        live = [r for r, c in enumerate(caches) if c is not None]
        results = [([], None)] * len(work)
        if accepted is not None and len(live) < len(work):
            return None
        if not live:
            return results
        _, out = self._run(embed, True, torch.cat([work[r][0] for r in live]),
                           [work[r][2] + j for r in live for j in range(lens[r])],
                           [caches[r].slot for r in live for _ in range(lens[r])],
                           [caches[r].length + j for r in live for j in range(lens[r])],
                           torch.cat([work[r][1] for r in live]))
        last, base, at = {}, {}, 0
        for r in live:
            if accepted is None:
                caches[r].length += lens[r]  # the verified positions stay; drafted ones are written past the length
            at += lens[r]
            # the row the chain starts from and the positions after it, in the request and in its cache: the last
            # row, or the first one with accepted[r] added on the device
            last[r] = at - 1 if accepted is None else at - lens[r]
            base[r] = (work[r][2] + lens[r], caches[r].length) if accepted is None else \
                (work[r][2] + 1, caches[r].length + 1)
        rows = [r for r in live if ks[r] > 0]  # the requests still drafting; `out` holds their last outputs
        drafts = torch.zeros(len(work), max(ks), dtype=torch.long, device=dev)
        # no index made from a host list below (it is copied to the device with a wait for the head step in flight):
        # the chain's steps are queued one behind the other, and the drafts are read once at the end
        rowsel = lambda x: host_tensor(x, torch.long, dev)  # noqa: E731
        shift = None
        if rows:
            first = rowsel([last[r] for r in rows])
            if accepted is not None:
                shift = accepted.index_select(0, rowsel(rows))
                first = first + shift
            out = out.index_select(0, first)
        for i in range(max(ks) if rows else 0):
            keep = [j for j, r in enumerate(rows) if ks[r] > i + 1]  # they feed this draft back for another
            if len(keep) < len(rows):  # the others take their last draft from `out` and leave
                tok = self.lm_head(self.norm(out)).argmax(-1)
                drafts[:, i].index_copy_(0, rowsel(rows), tok + self.d2t[tok])
                if not keep:
                    break
                out = out.index_select(0, rowsel(keep))
                shift = shift if shift is None else shift.index_select(0, rowsel(keep))
                rows = [rows[j] for j in keep]
            tok, out = self._run(embed, False, out, [base[r][0] + i for r in rows],
                                 [caches[r].slot for r in rows], [base[r][1] + i for r in rows], shift=shift)
            drafts[:, i].index_copy_(0, rowsel(rows), tok)  # the same draft the leavers' pick gave for these rows
        if host:
            drafts = drafts.tolist()
        for r in live:
            results[r] = (drafts[r][:ks[r]], caches[r])
        return results

    def draft(self, embed: nn.Module, aux: torch.Tensor, next_tokens: torch.Tensor, start: int, k: int,
              cache: HeadCache) -> Tuple[List[int], HeadCache]:
        """Take in T newly verified positions (as extend; the last of next_tokens is the token just sampled) and draft
        the k tokens after it, greedily. Returns (draft token ids, the cache over the verified positions only)."""
        return self.draft_many(embed, [(aux, next_tokens, start, k, cache)])[0]

    @torch.inference_mode()
    def draft_many(self, embed: nn.Module, work: List[Tuple[torch.Tensor, torch.Tensor, int, int, HeadCache]],
                   host: bool = True, accepted: Optional[torch.Tensor] = None) -> List[Tuple[List[int], HeadCache]]:
        """draft for several requests at once: work holds (aux, next_tokens, start, k, cache) per request (k = 0:
        take the positions in without drafting). Every linear layer runs once per draft position over the rows of
        all requests (a head step costs about one read of the head's weights, however many requests draft);
        attention runs per request over its own cache, or for all rows in the decode kernel (use_pool).
        host=False (with a pool): the drafts stay on the device, one tensor per request, and nothing here waits for
        them, so the caller can queue the step that verifies them behind the chain."""
        if self.pool is not None:
            return self._draft_many_pool(embed, work, host, accepted)
        dev, kv = work[0][0].device, self.num_kv_heads
        lens = [w[0].shape[0] for w in work]
        positions = torch.cat([w[2] + torch.arange(T, device=dev) for w, T in zip(work, lens)])
        residual, q, k, v = self._rows(embed(torch.cat([w[1] for w in work])), self.fc(torch.cat([w[0] for w in work])),
                                       positions)
        attn, caches, chains, at = [], [], [], 0
        for (_, _, _, n, cache), T in zip(work, lens):
            ki, vi = k[at:at + T].transpose(0, 1)[None], v[at:at + T].transpose(0, 1)[None]
            if cache is not None:
                ki, vi = torch.cat([cache[0], ki], dim=2), torch.cat([cache[1], vi], dim=2)
            attn.append(self._attend(q[at:at + T], ki, vi))
            caches.append((ki, vi))
            # room after the verified positions for this request's own drafts (every draft but the last is fed back)
            pad = ki.new_empty(1, kv, max(n - 1, 0), self.head_dim)
            chains.append([torch.cat([ki, pad], dim=2), torch.cat([vi, pad], dim=2), ki.shape[2]])
            at += T
        rowsel = lambda x: host_tensor(x, torch.long, dev)  # noqa: E731
        last = rowsel([sum(lens[:i + 1]) - 1 for i in range(len(work))])
        out = self._out(torch.cat(attn), residual)[last]  # each request's last verified position
        ks = [w[3] for w in work]
        drafts = torch.zeros(len(work), max(ks), dtype=torch.long, device=dev)
        rows = list(range(len(work)))  # the requests still drafting; `out` holds their last outputs
        for i in range(max(ks)):
            tok = self.lm_head(self.norm(out)).argmax(-1)
            tok = tok + self.d2t[tok]
            drafts[:, i].index_copy_(0, rowsel(rows), tok)
            keep = [j for j, r in enumerate(rows) if ks[r] > i + 1]
            if not keep:
                break
            if len(keep) < len(rows):
                sel = rowsel(keep)
                tok, out = tok[sel], out[sel]
            rows = [rows[j] for j in keep]
            positions = rowsel([work[r][2] + lens[r] + i for r in rows])
            residual, q, k, v = self._rows(embed(tok), out, positions)
            attn = []
            for j, r in enumerate(rows):
                ck, cv, n = chains[r]
                ck[0, :, n], cv[0, :, n] = k[j], v[j]
                chains[r][2] = n + 1
                attn.append(self._attend(q[j:j + 1], ck[:, :, :n + 1], cv[:, :, :n + 1]))
            out = self._out(torch.cat(attn), residual)
        drafts = drafts.tolist()
        return [(drafts[r][:ks[r]], caches[r]) for r in range(len(work))]


def load_eagle3(path: str, target_hidden: int, target_layers: int, device="cpu", dtype=torch.float32,
                token: Optional[str] = None) -> Eagle3Head:
    """An Eagle3Head from a local directory or a Hugging Face repository holding config.json and model.safetensors
    (or pytorch_model.bin, loaded with weights_only)."""
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(path, token=token, allow_patterns=["config.json", "model.safetensors", "pytorch_model.bin"])
    cfg = json.load(open(os.path.join(path, "config.json"), encoding="utf-8"))
    layer = cfg.get("transformer_layer_config") or cfg  # speculators nests the layer's config; the original is flat
    eagle = cfg.get("eagle_config") or {}
    unsupported = [name for name, on in (
        ("more than one layer", layer.get("num_hidden_layers", 1) != 1),
        ("attention_bias", layer.get("attention_bias", False)), ("mlp_bias", layer.get("mlp_bias", False)),
        ("fc_norm", cfg.get("fc_norm", False)), ("norm_output", cfg.get("norm_output", False)),
        ("norm_before_fc", cfg.get("norm_before_fc", False) or eagle.get("norm_before_fc", False)),
        ("use_aux_hidden_state=false", eagle.get("use_aux_hidden_state", True) is False),
        ("rope_scaling", layer.get("rope_scaling"))) if on]
    if unsupported:
        raise ValueError(f"EAGLE-3 head at {path} uses {', '.join(unsupported)}, which this loader does not support")
    hidden = layer["hidden_size"]
    if (cfg.get("target_hidden_size") or hidden) != target_hidden:
        raise ValueError(f"EAGLE-3 head at {path} is for a target of hidden size "
                         f"{cfg.get('target_hidden_size') or hidden}, not {target_hidden}")
    file = os.path.join(path, "model.safetensors")
    if os.path.exists(file):
        from safetensors.torch import load_file
        sd = load_file(file)
    else:
        sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    sd = {k.replace("midlayer.", "").replace("layers.0.", "").replace("self_attn.", "").replace("mlp.", ""): v
          for k, v in sd.items() if k not in ("t2d", "embed_tokens.weight")}  # the target's embeddings are used
    heads = layer["num_attention_heads"]
    aux = eagle.get("eagle_aux_hidden_state_layer_ids") or default_aux_layers(target_layers)
    head = Eagle3Head(hidden, layer["intermediate_size"], heads, layer.get("num_key_value_heads", heads),
                      layer.get("head_dim") or hidden // heads, sd["lm_head.weight"].shape[0], target_hidden,
                      layer.get("rms_norm_eps", 1e-6), layer.get("rope_theta", 10000.0),
                      layer.get("max_position_embeddings", 4096), bool(cfg.get("norm_before_residual", False)),
                      tuple(aux))
    if "d2t" not in sd:  # a head over the full vocabulary
        sd["d2t"] = torch.zeros(sd["lm_head.weight"].shape[0], dtype=torch.long)
    head.load_state_dict(sd)
    return head.to(device=device, dtype=dtype).eval()
