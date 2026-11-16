"""
Full Subspace Causal Language Model (SubspaceCausalLM) Architecture with GQA, RoPE & Subspace SwiGLU.
"""

from typing import List, Optional, Tuple, Union, Any
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..core.subspace import SubspaceRecirculation
from ..core.router import SubspaceStructuredRouter
from ..core.rope import NTKDynamicRoPEScaling, apply_rotary_pos_emb
from ..core.hybrid_attention import ChunkContextScorer
from ..core.kv_cache import KVCache, KVPool
from ..kernels.triton_kv_decode import pool_decode_supported, triton_pool_decode
from ..kernels.dispatch import dispatch_swiglu
from ..kernels.triton_fused_rmsnorm_swiglu import dispatch_fused_rmsnorm_swiglu


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()  # in fp32, as HF Llama does: fp16 squares overflow for large activations
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight

class SubspaceMLP(nn.Module):
    """
    Subspace SwiGLU Feed-Forward Network executing over pruned active channel tiles.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.ffn_dim = config.ffn_dim
        self.tile_size = config.tile_size
        self.active_tiles_count = config.active_tiles

        # Full weight representations (can be sliced / quantized)
        self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)

        # Default contiguous active tile indices [0 .. active_tiles - 1]
        default_tiles = torch.arange(self.active_tiles_count, dtype=torch.int32)
        self.register_buffer("active_tiles", default_tiles)

    def set_active_tiles(self, active_tile_indices: torch.Tensor):
        self.active_tiles = active_tile_indices.to(dtype=torch.int32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.view(-1, self.hidden_dim)

        out = dispatch_swiglu(
            x=x_2d,
            w_gate=self.gate_proj.weight.t(),
            w_up=self.up_proj.weight.t(),
            w_down=self.down_proj.weight.t(),
            active_tiles=self.active_tiles,
            tile_size=self.tile_size
        )

        return out.view(orig_shape)

class DecodeStep:
    """One batched decode step: the pool, each row's page-table slot and length before the step (device int32
    tensors), and, outside CUDA graphs, the rows' caches and host lengths. RoPE for the rows is computed once on the
    device and shared by every layer."""

    def __init__(self, pool: KVPool, slots: torch.Tensor, starts_t: torch.Tensor,
                 caches: Optional[List[KVCache]] = None, starts: Optional[List[int]] = None):
        self.pool, self.slots, self.starts_t, self.caches, self.starts = pool, slots, starts_t, caches, starts
        self._freqs = None
        self.quest_plan = None

    @classmethod
    def of(cls, caches: List[KVCache], device) -> "DecodeStep":
        starts = [c.length for c in caches]
        rows = torch.tensor([[c.slot for c in caches], starts], dtype=torch.int32).to(device)
        return cls(caches[0].pool, rows[0], rows[1], caches, starts)

    def freqs(self, rope) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._freqs is None:
            self._freqs = rope.compute_freqs_device(self.starts_t)
        return self._freqs


class SubspaceAttention(nn.Module):
    """
    Multi-Head & Grouped-Query Attention with Rotary Position Embeddings and Stateful KV Cache.
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

        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=False)
        if config.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

        self.rope = NTKDynamicRoPEScaling(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base_theta=config.rope_theta,
            llama3_scaling=config.rope_scaling
        )
        self.chunk_scorer = ChunkContextScorer(hidden_dim=self.num_kv_heads * self.head_dim, budget_tokens=2048)

    def forward(
        self,
        hidden_states: torch.Tensor,
        start_pos: Union[int, List[int]] = 0,
        cache: Optional[KVCache] = None,
        step: Optional["DecodeStep"] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Attention for new tokens at start_pos: through a KVCache (one request), a DecodeStep (one token per
        request), or with neither, causal attention over these tokens alone, returning their k and v."""
        batch, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.config.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        if step is not None:  # one position per row, on the device
            cos, sin = step.freqs(self.rope)
            cos, sin = cos.view(batch, 1, 1, self.head_dim), sin.view(batch, 1, 1, self.head_dim)
        else:
            cos, sin = self.rope.compute_freqs(start_pos + seq_len, device=hidden_states.device, start=start_pos)
        q, k = apply_rotary_pos_emb(q, k, cos.to(q.dtype), sin.to(q.dtype))

        if step is not None:  # one decode token for each request of the step
            return self.o_proj(self._decode_step(q, k, v, step)), None, None
        if cache is not None:
            return self.o_proj(self._attend_one(q, k, v, cache)), None, None

        out = F.scaled_dot_product_attention(q, k, v, is_causal=seq_len > 1, enable_gqa=self.num_kv_groups > 1)
        return self.o_proj(out.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)), k, v

    def _attend_one(self, q, k, v, cache: KVCache) -> torch.Tensor:
        """One request: append k, v at cache.length, then attend q over its cache: [1, H, L, D] -> [1, L, H * D]."""
        batch, _, seq_len, _ = q.shape
        pool, layer, cfg = cache.pool, self.layer_idx, self.config
        start, total = cache.length, cache.length + seq_len
        pool.append(layer, cache, k, v)
        gqa, mask, causal = self.num_kv_groups > 1, None, False
        if seq_len == 1 and cfg.kv_page_budget < 1.0 and pool.track_pages and total >= cfg.kv_select_min_tokens:
            sel = pool.select_pages(layer, [cache], [total], q, cfg.kv_page_budget, cfg.kv_select_min_tokens)[0]
            k_all, v_all, valid = pool.read_selected(layer, cache, sel, total)
            mask = valid.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_all, v_all = pool.read(layer, cache, total)
            if seq_len > 1 and start > 0:  # chunked prefill: query i sees keys 0 .. start + i
                mask = torch.arange(total, device=q.device) <= (start + torch.arange(seq_len, device=q.device))[:, None]
            causal = seq_len > 1 and start == 0
        out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask, is_causal=causal, enable_gqa=gqa)
        return out.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)

    def _decode_step(self, q, k, v, step: "DecodeStep") -> torch.Tensor:
        """One decode token per request: [B, H, 1, D] -> [B, 1, H * D]. On CUDA three fused launches serve every
        row (append, attention over each row's pages, split reduction); elsewhere each row takes _attend_one."""
        batch = q.shape[0]
        pool, layer, cfg = step.pool, self.layer_idx, self.config
        if not pool_decode_supported(pool, q):
            return torch.cat([self._attend_one(q[b:b + 1], k[b:b + 1], v[b:b + 1], c) for b, c in enumerate(step.caches)])
        sel = None
        if cfg.kv_page_budget < 1.0 and pool.track_pages and max(step.starts) + 1 >= cfg.kv_select_min_tokens:
            if step.quest_plan is None:  # once per step: every layer shares the rows' page counts and pages
                step.quest_plan = pool.selection_plan(step.caches, [n + 1 for n in step.starts],
                                                      cfg.kv_page_budget, cfg.kv_select_min_tokens)
            sel = pool.select_pages_planned(layer, q, step.quest_plan)
        out = triton_pool_decode(pool, layer, q, k, v, step.slots, step.starts_t, sel)
        return out.transpose(1, 2).reshape(batch, 1, self.num_heads * self.head_dim)


class SubspaceDecoderLayer(nn.Module):
    """
    Transformer Decoder block with Subspace Attention and Subspace SwiGLU MLP.
    """
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.self_attn = SubspaceAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.mlp = SubspaceMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        start_pos: Union[int, List[int]] = 0,
        cache: Optional[KVCache] = None,
        step: Optional["DecodeStep"] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, k_out, v_out = self.self_attn(hidden_states, start_pos=start_pos, cache=cache, step=step)
        hidden_states = residual + attn_out

        # Fused RMSNorm + Subspace SwiGLU + Residual
        orig_shape = hidden_states.shape
        x_2d = hidden_states.view(-1, self.mlp.hidden_dim)
        res_2d = hidden_states.view(-1, self.mlp.hidden_dim)

        fused_out = dispatch_fused_rmsnorm_swiglu(
            x=x_2d,
            weight_norm=self.post_attention_layernorm.weight,
            w_gate=self.mlp.gate_proj.weight.t(),
            w_up=self.mlp.up_proj.weight.t(),
            w_down=self.mlp.down_proj.weight.t(),
            residual=res_2d,
            active_tiles=self.mlp.active_tiles,
            tile_size=self.mlp.tile_size,
            eps=self.post_attention_layernorm.eps
        )
        hidden_states = fused_out.view(orig_shape)

        return hidden_states, k_out, v_out

class DecodeGraphs:
    """CUDA graphs of one decode step through SubspaceCausalLM._body, one per batch bucket. A step is copied into the
    graph's static buffers (token ids, page-table slots, lengths; padding rows point at the pool's scratch page) and
    replayed, so a step costs a few copies instead of every kernel launch. Rebuilt when the pool or its page table
    is replaced. Not used for Quest selection, LoRA tenants, or off CUDA."""

    BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)

    def __init__(self):
        # holds no reference to its model (the model passes itself in): no model <-> graphs cycle, so a dropped model
        # frees its KV pool at once, and a deep-copied model gets graphs of its own
        self.graphs = {}  # bucket -> (key, graph, ids, rows, logits)

    def eligible(self, model: "SubspaceCausalLM", caches: List[KVCache], tenant_id) -> bool:
        cfg, pool = model.config, caches[0].pool
        return (cfg.cuda_graphs and pool.device.type == "cuda" and len(caches) <= self.BUCKETS[-1]
                and tenant_id is None and not pool.track_pages and not torch.is_grad_enabled()
                and pool_decode_supported(pool, model.embed_tokens.weight))

    def run(self, model: "SubspaceCausalLM", input_ids: torch.Tensor, caches: List[KVCache]) -> torch.Tensor:
        B, pool = len(caches), caches[0].pool
        bucket = next(b for b in self.BUCKETS if b >= B)
        key = (id(pool), pool.page_table.data_ptr())
        entry = self.graphs.get(bucket)
        if entry is None or entry[0] != key:
            entry = self._capture(model, bucket, pool, key)
        _, graph, ids, rows, logits = entry
        scratch = pool.scratch_slot()
        host = torch.tensor([[c.slot for c in caches] + [scratch] * (bucket - B),
                             [c.length for c in caches] + [0] * (bucket - B)], dtype=torch.int32)
        rows.copy_(host)
        ids[:B].copy_(input_ids)
        graph.replay()
        return logits[:B].clone()  # the next replay overwrites the static output

    def _capture(self, model: "SubspaceCausalLM", bucket: int, pool: KVPool, key):
        device = pool.device
        torch.cuda.empty_cache()  # give capture the memory a long prefill left cached (rare: once per bucket and pool)
        ids = torch.zeros(bucket, 1, dtype=torch.long, device=device)
        rows = torch.zeros(2, bucket, dtype=torch.int32, device=device)
        rows[0] = pool.scratch_slot()
        step = DecodeStep(pool, rows[0], rows[1])
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):  # warm-up (Triton compilation, allocator) writes only the scratch page
            for _ in range(2):
                step._freqs = None
                model._body(ids, rows[1], None, step)
        torch.cuda.current_stream(device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        step._freqs = None
        with torch.cuda.graph(graph):
            logits = model._body(ids, rows[1], None, step)
        self.graphs[bucket] = (key, graph, ids, rows, logits)
        return self.graphs[bucket]


class SubspaceCausalLM(nn.Module):
    """
    End-to-end Autoregressive Subspace Causal Language Model with stateful generation.
    """
    def __init__(self, config: ModelConfig, adapter_manager: Optional[Any] = None):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_dim = config.hidden_dim
        self.adapter_manager = adapter_manager

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([SubspaceDecoderLayer(config, i) for i in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self._graphs = DecodeGraphs()

        # Recirculation layer at layer L/3
        self.router_layer_idx = config.router_layer_idx
        self.recirculation = SubspaceRecirculation(hidden_dim=config.hidden_dim, rank=config.rank_sub)
        self.router = SubspaceStructuredRouter(
            hidden_dim=config.hidden_dim,
            total_tiles=config.total_tiles,
            min_tiles=max(1, config.active_tiles // 2),
            max_tiles=config.active_tiles
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Union[KVCache, List[KVCache]]] = None,
        start_pos: int = 0,
        tenant_id: Optional[str] = None,
        logits_to_keep: int = 0
    ) -> Tuple[torch.Tensor, Union[KVCache, List[KVCache]]]:
        """past_key_values: None or a KVCache (filled in place and returned), or a list of KVCache, one per row of a
        [B, 1] decode step (batched decode across requests, returned as given). start_pos is taken from the cache.
        logits_to_keep > 0 computes logits for the last that many positions only (a prefill needs one)."""
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
        if step is not None:
            for c in step:
                c.reserve(c.length + 1)
            if self._graphs.eligible(self, step, tenant_id):
                logits = self._graphs.run(self, input_ids, step)
                for c in step:
                    c.length += 1
                return logits, returned
            step = DecodeStep.of(step, input_ids.device)
            start_pos = step.starts
        elif cache is not None:
            cache.reserve(cache.length + seq_len)
            start_pos = cache.length
        else:
            start_pos = 0
        logits = self._body(input_ids, start_pos, cache, step, tenant_id, logits_to_keep)
        if step is not None:
            for c in step.caches:
                c.length += 1
        elif cache is not None:
            cache.length += seq_len
        return logits, returned

    def _body(self, input_ids, start_pos, cache, step, tenant_id=None, logits_to_keep: int = 0) -> torch.Tensor:
        """Embedding, layers, norm and head for one forward (what a CUDA graph captures for a decode step)."""
        hidden_states = self.embed_tokens(input_ids)

        shallow_state = None

        for i, layer in enumerate(self.layers):
            # Capture shallow state at router boundary
            if self.config.enable_recirculation and i == self.router_layer_idx:
                shallow_state = hidden_states.clone()

            if step is not None:
                hidden_states, _, _ = layer(hidden_states, start_pos=start_pos, step=step)
            else:
                hidden_states, _, _ = layer(hidden_states, start_pos=start_pos, cache=cache)

            # Apply deep-to-shallow recurrence at the penultimate layer if state captured
            if self.config.enable_recirculation and i == len(self.layers) - 2 and shallow_state is not None:
                hidden_states = self.recirculation(hidden_states, shallow_state)

        # Apply multi-tenant dynamic LoRA adapter if active
        if tenant_id is not None and self.adapter_manager is not None:
            hidden_states = self.adapter_manager.forward_tenant(hidden_states, tenant_id)

        if logits_to_keep:
            hidden_states = hidden_states[:, -logits_to_keep:]
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    def kv_pool(self) -> KVPool:
        """This model's paged KV pool, created on first use (and again if the device, dtype or cache settings
        changed; caches of an older pool keep it alive)."""
        w, cfg = self.embed_tokens.weight, self.config
        pool = getattr(self, "_kv_pool", None)
        if (pool is None or pool.device != w.device or pool.dtype != w.dtype
                or pool.kv_cache_dtype != cfg.kv_cache_dtype or pool.track_pages != (cfg.kv_page_budget < 1.0)):
            self._kv_pool = None  # drop ours (and graphs captured on it) first, so its memory can back the new pool
            self._graphs.graphs.clear()
            gc.collect()
            if w.device.type == "cuda":
                torch.cuda.empty_cache()
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
        tenant_id: Optional[str] = None
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
                start_pos=start_pos,
                tenant_id=tenant_id
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
