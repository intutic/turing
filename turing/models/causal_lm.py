"""
Full Subspace Causal Language Model (SubspaceCausalLM) Architecture with GQA, RoPE & Subspace SwiGLU.
"""

from typing import List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..core.subspace import SubspaceRecirculation
from ..core.router import SubspaceStructuredRouter
from ..core.rope import NTKDynamicRoPEScaling, apply_rotary_pos_emb
from ..core.hybrid_attention import ChunkContextScorer
from ..core.kv_cache import DecodeBatchLayer, KVCache, LayerKVCache
from ..kernels.triton_kv_decode import (batched_decode_supported, build_decode_tables, kv_decode_supported,
                                        triton_kv_decode, triton_kv_decode_batch)
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

class SubspaceAttention(nn.Module):
    """
    Multi-Head & Grouped-Query Attention with Rotary Position Embeddings and Stateful KV Cache.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
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
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
        start_pos: Union[int, List[int]] = 0,
        cache: Union[LayerKVCache, List[LayerKVCache], None] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.config.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        if isinstance(start_pos, list):  # batched decode: one position per row
            cos, sin = self.rope.compute_freqs_at(tuple(start_pos), device=hidden_states.device)
            cos, sin = cos.view(batch, 1, 1, self.head_dim), sin.view(batch, 1, 1, self.head_dim)
        else:
            cos, sin = self.rope.compute_freqs(start_pos + seq_len, device=hidden_states.device, start=start_pos)
        q, k = apply_rotary_pos_emb(q, k, cos.to(q.dtype), sin.to(q.dtype))

        if isinstance(cache, DecodeBatchLayer):  # CUDA decode step: one fused append + attention for every row
            out = triton_kv_decode_batch(q, k, v, cache.rows, cache.table)
            return self.o_proj(out.transpose(1, 2).reshape(batch, 1, self.num_heads * self.head_dim)), None, None
        if isinstance(cache, list):  # batched decode: each row attends over its own request's cache
            out = torch.cat([self._attend_cache(q[b:b + 1], k[b:b + 1], v[b:b + 1], cache[b], start_pos[b])
                             for b in range(batch)])
            return self.o_proj(out), None, None
        if cache is not None:
            return self.o_proj(self._attend_cache(q, k, v, cache, start_pos)), None, None

        if past_k is not None and past_v is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        cur_k = k
        cur_v = v

        # Grouped Query Attention (repeat KV heads if needed)
        if self.num_kv_groups > 1:
            k_expanded = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_expanded = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_expanded = k
            v_expanded = v

        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q, k_expanded.transpose(-1, -2)) * scale

        # Causal mask if prefilling
        if seq_len > 1:
            causal_mask = torch.triu(torch.full((seq_len, k_expanded.shape[2]), float("-inf"), device=scores.device, dtype=scores.dtype), diagonal=1 + start_pos)
            scores = scores + causal_mask

        attn_weights = F.softmax(scores.float(), dim=-1).to(v_expanded.dtype)
        attn_out = torch.matmul(attn_weights, v_expanded) # [Batch, Heads, SeqLen, HeadDim]

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.head_dim)
        output = self.o_proj(attn_out)

        return output, cur_k, cur_v

    def _attend_cache(self, q, k, v, cache: LayerKVCache, start_pos: int) -> torch.Tensor:
        """Append k, v to the cache, then attend q over it: [B, H, L, D] -> [B, L, H * D]."""
        batch, _, seq_len, _ = q.shape
        cache.append(k, v)
        cfg, gqa, mask, causal = self.config, self.num_kv_groups > 1, None, False
        select = (seq_len == 1 and cfg.kv_page_budget < 1.0 and cache.track_pages
                  and cache.length >= cfg.kv_select_min_tokens)
        if seq_len == 1 and kv_decode_supported(q, cache):  # CUDA: fused kernel reads the stored format directly
            out = triton_kv_decode(q, cache, cache.select_pages(q, cfg.kv_page_budget) if select else None)
            return out.transpose(1, 2).reshape(batch, 1, self.num_heads * self.head_dim)
        if select:
            k_all, v_all, valid = cache.read_positions(cache.select_pages(q, cfg.kv_page_budget))
            mask = valid.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_all, v_all = cache.read()
            if seq_len > 1 and start_pos > 0:  # chunked prefill: query i sees keys 0 .. start_pos + i
                pos = torch.arange(cache.length, device=q.device)
                mask = pos <= (start_pos + torch.arange(seq_len, device=q.device))[:, None]
            causal = seq_len > 1 and start_pos == 0
        out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask, is_causal=causal, enable_gqa=gqa)
        return out.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)


class SubspaceDecoderLayer(nn.Module):
    """
    Transformer Decoder block with Subspace Attention and Subspace SwiGLU MLP.
    """
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.self_attn = SubspaceAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.mlp = SubspaceMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
        start_pos: Union[int, List[int]] = 0,
        cache: Union[LayerKVCache, List[LayerKVCache], None] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, k_out, v_out = self.self_attn(hidden_states, past_k=past_k, past_v=past_v, start_pos=start_pos, cache=cache)
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
        past_key_values: Optional[Union[KVCache, List[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        start_pos: int = 0,
        tenant_id: Optional[str] = None
    ) -> Tuple[torch.Tensor, Union[KVCache, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """past_key_values: None or a KVCache (filled in place and returned); a list of KVCache, one per row of a
        [B, 1] decode step (batched decode across requests); or a legacy list of (k, v) per layer."""
        batch, seq_len = input_ids.shape
        cache = None
        if isinstance(past_key_values, list) and past_key_values and isinstance(past_key_values[0], KVCache):
            if seq_len != 1 or batch != len(past_key_values):
                raise ValueError("a list of KVCache takes one decode token per request: input_ids [len(caches), 1]")
            cache, start_pos = past_key_values, [c.length for c in past_key_values]
        elif past_key_values is None or isinstance(past_key_values, KVCache):
            cache = past_key_values if past_key_values is not None else KVCache(
                len(self.layers), self.config.kv_cache_dtype, track_pages=self.config.kv_page_budget < 1.0)
            start_pos = cache.length
        elif start_pos == 0 and len(past_key_values) > 0 and past_key_values[0][0] is not None:
            start_pos = past_key_values[0][0].shape[2]

        # CUDA decode step: one pointer table per step, then fused kernels per layer for every request at once
        rows, tables = None, None
        if seq_len == 1 and cache is not None:
            rows = cache if isinstance(cache, list) else ([cache] if cache.length > 0 else None)
        if rows and batched_decode_supported(rows, input_ids.device, self.config.head_dim):
            for c in rows:
                if c.length + 1 > c.layers[0].capacity:
                    for layer_cache in c.layers:
                        layer_cache.ensure(c.length + 1)
            tables = build_decode_tables(rows, input_ids.device)
            start_pos = [c.length for c in rows]
        hidden_states = self.embed_tokens(input_ids)

        new_key_values = []
        shallow_state = None

        for i, layer in enumerate(self.layers):
            # Capture shallow state at router boundary
            if self.config.enable_recirculation and i == self.router_layer_idx:
                shallow_state = hidden_states.clone()

            if tables is not None:
                batch_layer = DecodeBatchLayer([c[i] for c in rows], tables[i])
                hidden_states, _, _ = layer(hidden_states, start_pos=start_pos, cache=batch_layer)
            elif isinstance(cache, list):
                hidden_states, _, _ = layer(hidden_states, start_pos=start_pos, cache=[c[i] for c in cache])
            elif cache is not None:
                hidden_states, _, _ = layer(hidden_states, start_pos=start_pos, cache=cache[i])
            else:
                p_k, p_v = past_key_values[i]
                hidden_states, k_out, v_out = layer(hidden_states, past_k=p_k, past_v=p_v, start_pos=start_pos)
                new_key_values.append((k_out, v_out))

            # Apply deep-to-shallow recurrence at the penultimate layer if state captured
            if self.config.enable_recirculation and i == len(self.layers) - 2 and shallow_state is not None:
                hidden_states = self.recirculation(hidden_states, shallow_state)

        # Apply multi-tenant dynamic LoRA adapter if active
        if tenant_id is not None and self.adapter_manager is not None:
            hidden_states = self.adapter_manager.forward_tenant(hidden_states, tenant_id)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, (cache if cache is not None else new_key_values)

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
