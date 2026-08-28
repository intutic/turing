"""
Cross-Model Closed-Form KV Cache Transfer (arXiv:2608.03893).
Implements per-head closed-form Ridge regression mapping with RoPE content-space decoupling,
top-k source layer selection, and attention null-space error projection.
"""

import math
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..core.rope import NTKDynamicRoPEScaling


class RoPEContentDecoupler:
    """
    Decouples position-dependent RoPE rotation from semantic content vectors:
    k_content = R_{theta}^{-1}(t) * k_RoPE
    Allows linear Ridge mapping weights to be position-free and reusable across arbitrary context lengths.
    """
    @staticmethod
    def strip_rope(k_rope: torch.Tensor, base: float = 500000.0) -> torch.Tensor:
        """
        k_rope: [Batch, SeqLen, NumHeads, HeadDim] or [SeqLen, HeadDim]
        Applies inverse 2D rotation R_{-theta}(t) = R_{theta}(t)^T.
        """
        try:
            import turing.turing_csrc as turing_csrc
            HAS_CSRC = True
        except ImportError:
            try:
                import turing_csrc
                HAS_CSRC = True
            except ImportError:
                HAS_CSRC = False

        if k_rope.is_cuda:
            try:
                from ..kernels.triton_cross_kv import fused_inv_rope_cuda
                return fused_inv_rope_cuda(k_rope, base=base)
            except Exception:
                pass

        if HAS_CSRC and not k_rope.is_cuda:
            k_cpu = k_rope.detach().to(torch.float32).cpu().contiguous().numpy()
            out_np = turing_csrc.fused_rope_transform(k_cpu, base, 0, True)
            return torch.from_numpy(out_np).to(device=k_rope.device, dtype=k_rope.dtype)


        orig_shape = k_rope.shape
        if k_rope.ndim == 2:
            seq_len, head_dim = orig_shape
            batch, num_heads = 1, 1
            k = k_rope.unsqueeze(0).unsqueeze(2)
        elif k_rope.ndim == 4:
            batch, seq_len, num_heads, head_dim = orig_shape
            k = k_rope
        else:
            raise ValueError(f"Unsupported tensor shape {orig_shape}")

        dim_half = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=k.device) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=k.device)
        freqs = torch.outer(t, inv_freq) # [SeqLen, dim_half]
        cos = freqs.cos().unsqueeze(0).unsqueeze(2) # [1, SeqLen, 1, dim_half]
        sin = freqs.sin().unsqueeze(0).unsqueeze(2)

        # Inverse rotation: [cos, sin; -sin, cos]
        k1 = k[..., :dim_half]
        k2 = k[..., dim_half:]
        k1_unrot = k1 * cos + k2 * sin
        k2_unrot = -k1 * sin + k2 * cos

        k_unrot = torch.cat([k1_unrot, k2_unrot], dim=-1)
        return k_unrot.view(orig_shape)

    @staticmethod
    def apply_rope(k_content: torch.Tensor, base: float = 500000.0) -> torch.Tensor:
        """
        Re-encodes content keys with target model's positional rotation.
        """
        try:
            import turing.turing_csrc as turing_csrc
            HAS_CSRC = True
        except ImportError:
            try:
                import turing_csrc
                HAS_CSRC = True
            except ImportError:
                HAS_CSRC = False

        if HAS_CSRC and not k_content.is_cuda:
            k_cpu = k_content.detach().to(torch.float32).cpu().contiguous().numpy()
            out_np = turing_csrc.fused_rope_transform(k_cpu, base, 0, False)
            return torch.from_numpy(out_np).to(device=k_content.device, dtype=k_content.dtype)

        orig_shape = k_content.shape
        if k_content.ndim == 2:
            seq_len, head_dim = orig_shape
            batch, num_heads = 1, 1
            k = k_content.unsqueeze(0).unsqueeze(2)
        elif k_content.ndim == 4:
            batch, seq_len, num_heads, head_dim = orig_shape
            k = k_content
        else:
            raise ValueError(f"Unsupported tensor shape {orig_shape}")

        dim_half = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=k.device) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=k.device)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().unsqueeze(0).unsqueeze(2)
        sin = freqs.sin().unsqueeze(0).unsqueeze(2)

        k1 = k[..., :dim_half]
        k2 = k[..., dim_half:]
        k1_rot = k1 * cos - k2 * sin
        k2_rot = k1 * sin + k2 * cos

        k_rot = torch.cat([k1_rot, k2_rot], dim=-1)
        return k_rot.view(orig_shape)


class ClosedFormRidgeMapper(nn.Module):
    """
    Closed-Form Per-Head Ridge Regression Mapper:
    W* = (X^T * X + lambda * I)^(-1) * X^T * Y
    Maps source KV cache representations to target model's expected KV format.
    """
    def __init__(
        self,
        source_heads: int,
        target_heads: int,
        head_dim: int,
        top_k_source_layers: int = 8,
        ridge_lambda: float = 0.01
    ):
        super().__init__()
        self.source_heads = source_heads
        self.target_heads = target_heads
        self.head_dim = head_dim
        self.top_k = top_k_source_layers
        self.ridge_lambda = ridge_lambda

        # In_dim = top_k * source_heads * head_dim
        self.in_dim = self.top_k * self.source_heads * self.head_dim
        self.out_dim = self.head_dim

        # Per-head weights and biases for K and V
        # Shape: [target_heads, in_dim, out_dim]
        self.register_buffer("w_k", torch.zeros(target_heads, self.in_dim, self.out_dim))
        self.register_buffer("b_k", torch.zeros(target_heads, self.out_dim))
        self.register_buffer("w_v", torch.zeros(target_heads, self.in_dim, self.out_dim))
        self.register_buffer("b_v", torch.zeros(target_heads, self.out_dim))
        self.is_fit = False

    def fit(self, x_source_kv: torch.Tensor, y_target_kv: torch.Tensor, is_key: bool = True):
        """
        Closed-form Tikhonov Regularized Least Squares fit.
        x_source_kv: [N_tokens, in_dim]
        y_target_kv: [N_tokens, target_heads, out_dim]
        Uses chunked covariance accumulation for memory and multi-core efficiency.
        """
        n_tokens = x_source_kv.shape[0]
        device = x_source_kv.device
        dtype = x_source_kv.dtype

        x_mean = x_source_kv.mean(dim=0, keepdim=True)
        x_centered = x_source_kv - x_mean

        # Chunked Covariance Matrix computation: (X^T * X + lambda * I)
        chunk_size = 4096
        if n_tokens > chunk_size:
            xtx = torch.zeros(self.in_dim, self.in_dim, device=device, dtype=dtype)
            for chunk_start in range(0, n_tokens, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_tokens)
                x_chunk = x_centered[chunk_start:chunk_end]
                xtx.addmm_(x_chunk.t(), x_chunk)
        else:
            xtx = torch.matmul(x_centered.t(), x_centered)

        reg_eye = self.ridge_lambda * torch.eye(self.in_dim, device=device, dtype=dtype)
        xtx_reg = xtx + reg_eye

        # Proposal Agentve system for each target head
        for h in range(self.target_heads):
            y_h = y_target_kv[:, h, :] # [N_tokens, out_dim]
            y_mean = y_h.mean(dim=0, keepdim=True)
            y_centered = y_h - y_mean

            if n_tokens > chunk_size:
                xty = torch.zeros(self.in_dim, self.out_dim, device=device, dtype=dtype)
                for chunk_start in range(0, n_tokens, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n_tokens)
                    x_c = x_centered[chunk_start:chunk_end]
                    y_c = y_centered[chunk_start:chunk_end]
                    xty.addmm_(x_c.t(), y_c)
            else:
                xty = torch.matmul(x_centered.t(), y_centered) # [in_dim, out_dim]

            # Proposal Agentve xtx_reg * W = xty
            w_h = torch.linalg.solve(xtx_reg, xty) # [in_dim, out_dim]
            b_h = y_mean - torch.matmul(x_mean, w_h)

            if is_key:
                self.w_k[h].copy_(w_h)
                self.b_k[h].copy_(b_h.squeeze(0))
            else:
                self.w_v[h].copy_(w_h)
                self.b_v[h].copy_(b_h.squeeze(0))

        self.is_fit = True


    def forward(self, x_source: torch.Tensor, is_key: bool = True) -> torch.Tensor:
        """
        x_source: [Batch, SeqLen, in_dim] or [SeqLen, in_dim]
        Returns: [Batch, SeqLen, target_heads, out_dim]
        """
        orig_ndim = x_source.ndim
        if orig_ndim == 2:
            x = x_source.unsqueeze(0)
        else:
            x = x_source

        batch_size, seq_len, _ = x.shape
        w = (self.w_k if is_key else self.w_v).to(device=x.device, dtype=x.dtype)
        b = (self.b_k if is_key else self.b_v).to(device=x.device, dtype=x.dtype)

        try:
            import turing.turing_csrc as turing_csrc
            HAS_CSRC = True
        except ImportError:
            HAS_CSRC = False

        if HAS_CSRC and not x.is_cuda:
            x_flat = x.reshape(-1, self.in_dim).detach().to(torch.float32).cpu().contiguous().numpy()
            w_flat = w.permute(1, 0, 2).reshape(self.in_dim, self.target_heads * self.out_dim).detach().to(torch.float32).cpu().contiguous().numpy()
            b_flat = b.reshape(self.target_heads * self.out_dim).detach().to(torch.float32).cpu().contiguous().numpy()
            out_np = turing_csrc.fused_ridge_forward(x_flat, w_flat, b_flat)
            out_flat = torch.from_numpy(out_np).to(device=x.device, dtype=x.dtype)
            out = out_flat.view(batch_size, seq_len, self.target_heads, self.out_dim)
            if orig_ndim == 2:
                return out.squeeze(0)
            return out

        # Reshape to standard GEMM for hardware compatibility across CPU, MPS, and CUDA
        w_flat = w.permute(1, 0, 2).reshape(self.in_dim, self.target_heads * self.out_dim)
        out_flat = torch.matmul(x.reshape(-1, self.in_dim), w_flat)
        out = out_flat.view(batch_size, seq_len, self.target_heads, self.out_dim) + b.unsqueeze(0).unsqueeze(1)

        if orig_ndim == 2:
            return out.squeeze(0)
        return out



class SVDNullSpaceProjector:
    """
    Projects Ridge transfer residual errors onto the null-space of target query SVD singular vectors,
    preventing transfer noise from corrupting high-attention-weight subspaces.
    """
    def __init__(self, query_svd_basis: torch.Tensor, top_r: int = 16):
        """
        query_svd_basis: [HeadDim, Rank] orthonormal right singular vectors of target queries.
        """
        self.top_r = top_r
        # U_sensitive: [HeadDim, top_r]
        self.u_sens = query_svd_basis[:, :top_r]
        # Null-space projector P_null = I - U_sens * U_sens^T
        eye = torch.eye(query_svd_basis.shape[0], device=query_svd_basis.device, dtype=query_svd_basis.dtype)
        self.p_null = eye - torch.matmul(self.u_sens, self.u_sens.t())

    def filter_residual(self, error_vector: torch.Tensor) -> torch.Tensor:
        """
        error_vector: [..., HeadDim]
        Returns error components lying purely in attention-irrelevant directions.
        """
        return torch.matmul(error_vector, self.p_null)


class CrossModelKVPipeline:
    """
    End-to-End Cross-Model KV Cache Transfer Cascading Pipeline.
    1. Runs fast prefill on small source model (e.g. LLaMA-3-8B).
    2. Decouples source RoPE rotation from keys.
    3. Maps selected source layers via closed-form Ridge regressions into target (70B) KV space.
    4. Applies target model RoPE rotation.
    5. Injects populated KV cache into target model, skipping 70B prefill entirely.
    """
    def __init__(
        self,
        source_config: ModelConfig,
        target_config: ModelConfig,
        top_k_layers: int = 8,
        ridge_lambda: float = 0.01
    ):
        self.source_config = source_config
        self.target_config = target_config
        self.top_k = top_k_layers

        # Target has target_config.num_layers layers
        self.layer_mappers_k: Dict[int, ClosedFormRidgeMapper] = {}
        self.layer_mappers_v: Dict[int, ClosedFormRidgeMapper] = {}

        for l_idx in range(target_config.num_layers):
            self.layer_mappers_k[l_idx] = ClosedFormRidgeMapper(
                source_heads=source_config.num_kv_heads,
                target_heads=target_config.num_kv_heads,
                head_dim=target_config.head_dim,
                top_k_source_layers=self.top_k,
                ridge_lambda=ridge_lambda
            )
            self.layer_mappers_v[l_idx] = ClosedFormRidgeMapper(
                source_heads=source_config.num_kv_heads,
                target_heads=target_config.num_kv_heads,
                head_dim=target_config.head_dim,
                top_k_source_layers=self.top_k,
                ridge_lambda=ridge_lambda
            )

    def select_source_layer_indices(self, target_layer_idx: int) -> List[int]:
        """
        Heuristic top-k layer selection mapping target layer to nearest source layers.
        """
        src_layers = self.source_config.num_layers
        tgt_layers = self.target_config.num_layers

        # Center around proportional source layer
        center_src = int(round((target_layer_idx / tgt_layers) * src_layers))
        half_k = self.top_k // 2
        start = max(0, min(src_layers - self.top_k, center_src - half_k))
        return list(range(start, min(src_layers, start + self.top_k)))

    def transfer_cache(
        self,
        source_keys_by_layer: List[torch.Tensor],
        source_values_by_layer: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        source_keys_by_layer: List of [Batch, SeqLen, SrcKVHeads, HeadDim]
        source_values_by_layer: List of [Batch, SeqLen, SrcKVHeads, HeadDim]
        Returns: target_keys_by_layer, target_values_by_layer
        """
        tgt_keys = []
        tgt_values = []

        # Ensure format is [Batch, SeqLen, Heads, HeadDim]
        src_keys_formatted = []
        src_vals_formatted = []
        for k, v in zip(source_keys_by_layer, source_values_by_layer):
            if k.ndim == 4 and k.shape[1] == self.source_config.num_kv_heads and k.shape[2] != self.source_config.num_kv_heads:
                src_keys_formatted.append(k.transpose(1, 2).contiguous())
                src_vals_formatted.append(v.transpose(1, 2).contiguous())
            else:
                src_keys_formatted.append(k)
                src_vals_formatted.append(v)

        batch, seq_len, _, head_dim = src_keys_formatted[0].shape

        # Step 1: Strip RoPE from all source key layers
        stripped_source_keys = [
            RoPEContentDecoupler.strip_rope(k, base=self.source_config.rope_theta)
            for k in src_keys_formatted
        ]

        # Step 2: Map each target layer
        for t_idx in range(self.target_config.num_layers):
            selected_src_indices = self.select_source_layer_indices(t_idx)

            # Concatenate selected source layer features: [Batch, SeqLen, top_k * SrcKVHeads * HeadDim]
            k_feats = torch.cat([
                stripped_source_keys[s_idx].reshape(batch, seq_len, -1)
                for s_idx in selected_src_indices
            ], dim=-1)

            v_feats = torch.cat([
                src_vals_formatted[s_idx].reshape(batch, seq_len, -1)
                for s_idx in selected_src_indices
            ], dim=-1)

            # Apply per-head linear ridge map
            k_mapper = self.layer_mappers_k[t_idx]
            v_mapper = self.layer_mappers_v[t_idx]

            # Projected target keys in content space
            mapped_k_content = k_mapper(k_feats, is_key=True) # [Batch, SeqLen, TgtKVHeads, HeadDim]
            mapped_v = v_mapper(v_feats, is_key=False)

            # Step 3: Re-encode with target model RoPE
            mapped_k_rope = RoPEContentDecoupler.apply_rope(mapped_k_content, base=self.target_config.rope_theta)

            tgt_keys.append(mapped_k_rope)
            tgt_values.append(mapped_v)

        return tgt_keys, tgt_values


class XKVLayerAlignmentTransport(nn.Module):
    """
    Continuous Gaussian Cross-Layer Alignment Transport (arXiv:2608.20617).
    Constructs a smooth, differentiable transport matrix between heterogeneous transformer depths:
    A_{i, j} = exp( - (i / L_src - j / L_tgt)^2 / (2 * sigma^2) ) / sum_k (...)
    """
    def __init__(self, src_layers: int, tgt_layers: int, sigma: float = 0.12):
        super().__init__()
        self.src_layers = src_layers
        self.tgt_layers = tgt_layers
        self.sigma = sigma

        # Build normalized transport matrix: [src_layers, tgt_layers]
        src_pos = torch.linspace(0.0, 1.0, src_layers).unsqueeze(1) # [L_src, 1]
        tgt_pos = torch.linspace(0.0, 1.0, tgt_layers).unsqueeze(0) # [1, L_tgt]
        
        diff = (src_pos - tgt_pos) ** 2
        raw_transport = torch.exp(-diff / (2.0 * (sigma ** 2)))
        transport_matrix = raw_transport / raw_transport.sum(dim=0, keepdim=True)
        
        self.register_buffer("transport_matrix", transport_matrix)

    def forward(self) -> torch.Tensor:
        return self.transport_matrix


class XKVHeadSummaryExtractor(nn.Module):
    """
    Per-Head Latent Summary Extractor (XKV - arXiv:2608.20617).
    Extracts compact, position-decoupled semantic summaries from an agent's KV cache.
    """
    def __init__(self, head_dim: int, num_heads: int, num_summary_tokens: int = 4):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_summary_tokens = num_summary_tokens
        
        # Learned query matrix for extracting multi-token temporal summaries
        self.summary_queries = nn.Parameter(
            torch.randn(num_summary_tokens, num_heads, head_dim) * 0.02
        )
        self.norm_k = nn.LayerNorm(head_dim)
        self.norm_v = nn.LayerNorm(head_dim)

    def forward(
        self,
        k_content: torch.Tensor,
        v_cache: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        k_content: [Batch, SeqLen, NumHeads, HeadDim] (RoPE stripped)
        v_cache: [Batch, SeqLen, NumHeads, HeadDim]
        Returns:
          k_summary: [Batch, NumSummaryTokens, NumHeads, HeadDim]
          v_summary: [Batch, NumSummaryTokens, NumHeads, HeadDim]
        """
        batch, seq_len, num_heads, head_dim = k_content.shape
        
        k_norm = self.norm_k(k_content)
        v_norm = self.norm_v(v_cache)
        
        # Attention scores between summary queries and cached keys:
        # queries: [num_summary_tokens, num_heads, head_dim]
        # keys: [batch, seq_len, num_heads, head_dim]
        # einsum: q=summary_tokens, h=num_heads, d=head_dim, b=batch, s=seq_len
        attn_logits = torch.einsum('qhd,bshd->bqhs', self.summary_queries, k_norm) / math.sqrt(head_dim)
        attn_weights = F.softmax(attn_logits, dim=-1) # [batch, q, h, seq_len]

        # Weighted pooling over sequence dimension:
        k_summary = torch.einsum('bqhs,bshd->bqhd', attn_weights, k_content)
        v_summary = torch.einsum('bqhs,bshd->bqhd', attn_weights, v_norm)

        return k_summary, v_summary


class XKVLatentAgentBridge(nn.Module):
    """
    XKV Cross-Model Latent Agent Bridge (arXiv:2608.20617).
    Enables zero-token, sub-symbolic KV cache transfer between heterogeneous agents.
    Provides 6.8x-8.2x lower latency than natural language message passing.
    """
    def __init__(
        self,
        source_config: ModelConfig,
        target_config: ModelConfig,
        num_summary_tokens: int = 4
    ):
        super().__init__()
        self.source_config = source_config
        self.target_config = target_config
        self.num_summary_tokens = num_summary_tokens

        # Cross-layer depth alignment transport
        self.alignment = XKVLayerAlignmentTransport(
            src_layers=source_config.num_layers,
            tgt_layers=target_config.num_layers
        )

        # Per-layer summary extractors for source model
        self.extractors = nn.ModuleList([
            XKVHeadSummaryExtractor(
                head_dim=source_config.head_dim,
                num_heads=source_config.num_kv_heads,
                num_summary_tokens=num_summary_tokens
            )
            for _ in range(source_config.num_layers)
        ])

        # Per-layer cross-model linear adapters
        src_feat_dim = source_config.num_kv_heads * source_config.head_dim
        tgt_feat_dim = target_config.num_kv_heads * target_config.head_dim

        self.k_proj = nn.ModuleList([
            nn.Linear(src_feat_dim, tgt_feat_dim, bias=False)
            for _ in range(target_config.num_layers)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(src_feat_dim, tgt_feat_dim, bias=False)
            for _ in range(target_config.num_layers)
        ])

        # Initialize projections with orthonormal identity-like scaling
        for proj in self.k_proj + self.v_proj:
            nn.init.orthogonal_(proj.weight)

    def transfer_latent_kv(
        self,
        source_keys: List[torch.Tensor],
        source_values: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Transfers source agent KV state directly to target agent KV cache.
        Returns:
          tgt_keys: List of [Batch, NumSummaryTokens, TgtKVHeads, HeadDim]
          tgt_values: List of [Batch, NumSummaryTokens, TgtKVHeads, HeadDim]
          shared_latent_state: Combined audit tensor [Batch, NumSummaryTokens, SharedDim]
        """
        assert len(source_keys) == self.source_config.num_layers
        
        batch = source_keys[0].shape[0]
        
        # Step 1: Strip RoPE and extract position-free per-layer summaries
        src_k_summaries = []
        src_v_summaries = []

        for l_idx in range(self.source_config.num_layers):
            k = source_keys[l_idx]
            v = source_values[l_idx]

            # Format to [Batch, SeqLen, NumHeads, HeadDim] if needed
            if k.ndim == 4 and k.shape[1] == self.source_config.num_kv_heads:
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()

            # Decouple RoPE
            k_content = RoPEContentDecoupler.strip_rope(k, base=self.source_config.rope_theta)
            
            # Extract head summaries: [Batch, NumSummaryTokens, NumHeads, HeadDim]
            k_sum, v_sum = self.extractors[l_idx](k_content, v)
            src_k_summaries.append(k_sum.reshape(batch, self.num_summary_tokens, -1))
            src_v_summaries.append(v_sum.reshape(batch, self.num_summary_tokens, -1))

        # Stack summaries across source layers: [L_src, Batch, NumSummaryTokens, SrcFeatDim]
        stacked_k = torch.stack(src_k_summaries, dim=0)
        stacked_v = torch.stack(src_v_summaries, dim=0)

        # Step 2: Continuous Gaussian Layer Alignment Transport
        # transport_matrix: [L_src, L_tgt]
        transport_a = self.alignment()

        # Step 3: Project into target layers
        tgt_keys = []
        tgt_values = []
        
        for t_idx in range(self.target_config.num_layers):
            # Mix source layer summaries according to alignment column: [Batch, NumSummaryTokens, SrcFeatDim]
            weights = transport_a[:, t_idx].view(-1, 1, 1, 1)
            mixed_k = (stacked_k * weights).sum(dim=0)
            mixed_v = (stacked_v * weights).sum(dim=0)

            # Linear projection to target head dimensions: [Batch, NumSummaryTokens, TgtFeatDim]
            proj_k = self.k_proj[t_idx](mixed_k).view(
                batch, self.num_summary_tokens, self.target_config.num_kv_heads, self.target_config.head_dim
            )
            proj_v = self.v_proj[t_idx](mixed_v).view(
                batch, self.num_summary_tokens, self.target_config.num_kv_heads, self.target_config.head_dim
            )

            # Re-apply target model RoPE
            proj_k_rope = RoPEContentDecoupler.apply_rope(proj_k, base=self.target_config.rope_theta)

            tgt_keys.append(proj_k_rope)
            tgt_values.append(proj_v)

        # Global shared latent memory for auditing
        shared_latent_state = stacked_k.mean(dim=0) # [Batch, NumSummaryTokens, SrcFeatDim]

        return tgt_keys, tgt_values, shared_latent_state

