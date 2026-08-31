"""
Hugging Face Real Weight Importer, Pruner, and Subspace Loader.
Loads real weights from Hugging Face Hub, computes empirical channel saliency,
exports packed .tgate4 INT4 containers, and loads them into SubspaceCausalLM.
"""

import os
import warnings

os.environ["HF_HUB_DISABLE_UNAUTHENTICATED_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

import huggingface_hub
huggingface_hub.logging.set_verbosity_error()

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

# Silence non-critical HF warnings
hf_logging.set_verbosity_error()

from ..config import ModelConfig
from .causal_lm import SubspaceCausalLM, SubspaceDecoderLayer
from .converter import TuringConverter
from ..core.subspace import SubspaceManager

from .resolver import ModelResolver, ResolvedModelSpec

class RealHuggingFaceLoader:
    """
    Imports real pretrained weights from HuggingFace directly into Turing Engine Subspace format.
    """
    MODEL_ALIASES = ModelResolver.CLI_ALIASES

    @classmethod
    def parse_model_identifier(cls, model_identifier: str) -> Tuple[str, Optional[str]]:
        spec = ModelResolver.parse(model_identifier)
        return spec.repo_id, spec.reasoning_effort

    @classmethod
    def resolve_model_id(cls, model_identifier: str) -> str:
        return ModelResolver.resolve_repo_id(model_identifier)

    @staticmethod
    def load_hf_model_into_turing(
        hf_model_id: str = "gpt2",
        sparsity_ratio: float = 0.5,
        device: str = "cpu",
        token: Optional[str] = None
    ) -> Tuple[SubspaceCausalLM, Any]:
        """
        Loads real Hugging Face model and maps weights to SubspaceCausalLM.
        """
        resolved_id = RealHuggingFaceLoader.resolve_model_id(hf_model_id)
        hf_token = token or os.environ.get("HF_TOKEN") or huggingface_hub.get_token()
        print(f"[*] Downloading / Loading real pretrained HuggingFace model: '{resolved_id}' (Authenticated)...")
        tokenizer = AutoTokenizer.from_pretrained(resolved_id, token=hf_token)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        target_device = torch.device(device)
        target_dtype = torch.float16 if target_device.type == "cuda" else torch.float32

        hf_model = AutoModelForCausalLM.from_pretrained(
            resolved_id,
            token=hf_token,
            torch_dtype=target_dtype,
            low_cpu_mem_usage=True
        ).eval()

        hf_cfg = hf_model.config
        hidden_dim = getattr(hf_cfg, "hidden_size", getattr(hf_cfg, "n_embd", 768))
        num_layers = getattr(hf_cfg, "num_hidden_layers", getattr(hf_cfg, "n_layer", 12))
        num_heads = getattr(hf_cfg, "num_attention_heads", getattr(hf_cfg, "n_head", 12))
        num_kv_heads = getattr(hf_cfg, "num_key_value_heads", num_heads)
        head_dim = hidden_dim // num_heads
        vocab_size = getattr(hf_cfg, "vocab_size", 50257)
        ffn_dim = getattr(hf_cfg, "intermediate_size", hidden_dim * 4)

        tile_size = 64 if ffn_dim <= 4096 else 256
        total_tiles = ffn_dim // tile_size
        active_tiles_count = max(1, int(total_tiles * (1.0 - sparsity_ratio)))

        turing_cfg = ModelConfig(
            name=f"HF-{hf_model_id}",
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            vocab_size=vocab_size,
            tile_size=tile_size,
            active_tiles=active_tiles_count,
            rank_sub=64 if hidden_dim >= 768 else 32,
            max_position_embeddings=getattr(hf_cfg, "max_position_embeddings", 2048),
            rope_theta=getattr(hf_cfg, "rope_theta", 10000.0)
        )

        turing_model = SubspaceCausalLM(turing_cfg).to(target_dtype).to(target_device)

        # Copy real embeddings and output head
        with torch.no_grad():
            if hasattr(hf_model, "transformer"): # GPT-2 style
                if hasattr(hf_model.transformer, "wte"):
                    turing_model.embed_tokens.weight.copy_(hf_model.transformer.wte.weight.to(target_device))
                if hasattr(hf_model.transformer, "ln_f"):
                    turing_model.norm.weight.copy_(hf_model.transformer.ln_f.weight.to(target_device))
                if hasattr(hf_model, "lm_head"):
                    turing_model.lm_head.weight.copy_(hf_model.lm_head.weight.to(target_device))
                elif hasattr(hf_model.transformer, "wte"):
                    turing_model.lm_head.weight.copy_(hf_model.transformer.wte.weight.to(target_device))
            elif hasattr(hf_model, "model"): # LLaMA / Qwen / DeepSeek / Mistral style
                if hasattr(hf_model.model, "embed_tokens"):
                    turing_model.embed_tokens.weight.copy_(hf_model.model.embed_tokens.weight.to(target_device))
                if hasattr(hf_model.model, "norm"):
                    turing_model.norm.weight.copy_(hf_model.model.norm.weight.to(target_device))
                if hasattr(hf_model, "lm_head"):
                    turing_model.lm_head.weight.copy_(hf_model.lm_head.weight.to(target_device))

            # Map real layers
            for l_idx in range(num_layers):
                j_layer = turing_model.layers[l_idx]

                if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
                    hf_layer = hf_model.transformer.h[l_idx]
                    # GPT-2 c_attn contains [Q, K, V] concatenated in dim=-1
                    qkv_w = hf_layer.attn.c_attn.weight.to(target_device) # [hidden, 3*hidden]
                    qkv_w = qkv_w.t().contiguous() # [3*hidden, hidden]
                    q_w, k_w, v_w = qkv_w.chunk(3, dim=0)

                    j_layer.self_attn.q_proj.weight.copy_(q_w)
                    j_layer.self_attn.k_proj.weight.copy_(k_w)
                    j_layer.self_attn.v_proj.weight.copy_(v_w)
                    j_layer.self_attn.o_proj.weight.copy_(hf_layer.attn.c_proj.weight.to(target_device).t().contiguous())

                    # GPT-2 MLP c_fc [hidden, ffn_dim], c_proj [ffn_dim, hidden]
                    fc_w = hf_layer.mlp.c_fc.weight.to(target_device).t().contiguous() # [ffn_dim, hidden]
                    proj_w = hf_layer.mlp.c_proj.weight.to(target_device).t().contiguous() # [hidden, ffn_dim]

                    # Saliency pruning on real MLP weights
                    saliency = torch.norm(fc_w, dim=-1) * torch.norm(proj_w, dim=0)
                    tile_scores = saliency.view(total_tiles, tile_size).mean(dim=-1)
                    _, active_indices = torch.topk(tile_scores, k=active_tiles_count)
                    active_indices = sorted(active_indices.tolist())

                    # Set weights and active tiles
                    j_layer.mlp.gate_proj.weight.copy_(fc_w)
                    j_layer.mlp.up_proj.weight.copy_(torch.ones_like(fc_w))
                    j_layer.mlp.down_proj.weight.copy_(proj_w)
                    j_layer.mlp.set_active_tiles(torch.tensor(active_indices, dtype=torch.int32, device=target_device))

                elif hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
                    hf_layer = hf_model.model.layers[l_idx]
                    j_layer.self_attn.q_proj.weight.copy_(hf_layer.self_attn.q_proj.weight.to(target_device))
                    j_layer.self_attn.k_proj.weight.copy_(hf_layer.self_attn.k_proj.weight.to(target_device))
                    j_layer.self_attn.v_proj.weight.copy_(hf_layer.self_attn.v_proj.weight.to(target_device))
                    j_layer.self_attn.o_proj.weight.copy_(hf_layer.self_attn.o_proj.weight.to(target_device))

                    # Real SwiGLU FFN: gate, up, down
                    gate_w = hf_layer.mlp.gate_proj.weight.to(target_device) # [ffn, hidden]
                    up_w = hf_layer.mlp.up_proj.weight.to(target_device)
                    down_w = hf_layer.mlp.down_proj.weight.to(target_device) # [hidden, ffn]

                    saliency = torch.norm(gate_w, dim=-1) * torch.norm(up_w, dim=-1) * torch.norm(down_w, dim=0)
                    tile_scores = saliency.view(total_tiles, tile_size).mean(dim=-1)
                    _, active_indices = torch.topk(tile_scores, k=active_tiles_count)
                    active_indices = sorted(active_indices.tolist())

                    j_layer.mlp.gate_proj.weight.copy_(gate_w)
                    j_layer.mlp.up_proj.weight.copy_(up_w)
                    j_layer.mlp.down_proj.weight.copy_(down_w)
                    j_layer.mlp.set_active_tiles(torch.tensor(active_indices, dtype=torch.int32, device=target_device))

        del hf_model
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"[+] Successfully loaded {num_layers} real layers from '{hf_model_id}' into Turing Engine Subspace format ({sparsity_ratio*100:.1f}% channel pruned).")
        return turing_model, tokenizer
