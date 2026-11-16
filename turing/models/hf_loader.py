"""
HuggingFace weight importer: maps a HuggingFace causal LM onto SubspaceCausalLM for the families listed in FAMILIES,
reproducing HuggingFace's logits (tests/test_hf_parity.py). Any other architecture is refused, and every Turing weight
must be written by the mapping, so nothing is ever served at random initialisation.
"""

import os

os.environ["HF_HUB_DISABLE_UNAUTHENTICATED_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import huggingface_hub
huggingface_hub.logging.set_verbosity_error()

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Callable
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

# Silence non-critical HF warnings
hf_logging.set_verbosity_error()

from ..config import ModelConfig, hf_rope_scaling, hf_rope_theta
from .causal_lm import SubspaceCausalLM
from .resolver import ModelResolver, ResolvedModelSpec

# HF model_type -> weight layout handled by _copy_weights. Everything else is refused.
FAMILIES = {
    "llama": "llama", "mistral": "llama", "qwen2": "llama", "qwen3": "llama",
    "gemma": "llama", "gemma2": "gemma2", "gemma3_text": "gemma2", "phi3": "phi3",
    "mixtral": "llama", "qwen2_moe": "llama", "qwen3_moe": "llama", "olmoe": "llama",
    "phi": "phi", "gpt2": "gpt2", "opt": "opt", "gpt_neox": "gpt_neox",
}

_DTYPES = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
           "float32": torch.float32, "fp32": torch.float32}


def resolve_dtype(dtype, device: torch.device, checkpoint_dtype=None) -> torch.dtype:
    """None picks the default: on CUDA the checkpoint's half-precision type (bf16 checkpoints such as Gemma overflow in
    fp16) or fp16; fp32 on CPU and MPS."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is not None:
        if dtype not in _DTYPES:
            raise ValueError(f"dtype '{dtype}' is not one of {sorted(_DTYPES)}")
        return _DTYPES[dtype]
    if device.type == "cuda":
        return torch.bfloat16 if checkpoint_dtype == torch.bfloat16 else torch.float16
    return torch.float32


def _get(c, *names, default=None):
    for n in names:
        v = getattr(c, n, None)
        if v is not None:
            return v
    return default


def _config_kwargs(hf_cfg) -> Dict[str, Any]:
    """ModelConfig fields for one HF config of a supported family."""
    mt = hf_cfg.model_type
    hidden = _get(hf_cfg, "hidden_size", "n_embd")
    heads = _get(hf_cfg, "num_attention_heads", "n_head")
    head_dim = _get(hf_cfg, "head_dim") or hidden // heads
    rp = getattr(hf_cfg, "rope_parameters", None) or {}
    partial = _get(hf_cfg, "partial_rotary_factor", "rotary_pct",
                   default=rp.get("partial_rotary_factor", 1.0) if isinstance(rp.get("partial_rotary_factor", 1.0), float) else 1.0)
    kw = dict(
        arch=mt, hidden_dim=hidden, num_heads=heads, head_dim=head_dim,
        num_kv_heads=_get(hf_cfg, "num_key_value_heads", default=heads),
        num_layers=_get(hf_cfg, "num_hidden_layers", "n_layer"),
        vocab_size=hf_cfg.vocab_size,
        ffn_dim=_get(hf_cfg, "intermediate_size", "n_inner", "ffn_dim") or 4 * hidden,
        max_position_embeddings=_get(hf_cfg, "max_position_embeddings", "n_positions", default=2048),
        rms_norm_eps=_get(hf_cfg, "rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon", default=1e-5),
        hidden_act=_get(hf_cfg, "hidden_activation", "hidden_act", "activation_function", default="silu"),
        attention_bias=bool(getattr(hf_cfg, "attention_bias", False)),
        rotary_dim=int(head_dim * partial) if partial != 1.0 else None,
    )
    layer_types = getattr(hf_cfg, "layer_types", None)
    window = getattr(hf_cfg, "sliding_window", None)
    if window and (layer_types is None and getattr(hf_cfg, "use_sliding_window", True) is not False):
        kw.update(sliding_window=window)
    elif window and layer_types and "sliding_attention" in layer_types:
        kw.update(sliding_window=window, layer_types=list(layer_types))
    per_type = isinstance(rp.get("full_attention"), dict)
    kw.update(rope_theta=hf_rope_theta(hf_cfg, "full_attention" if per_type else None),
              rope_scaling=hf_rope_scaling(hf_cfg, "full_attention" if per_type else None))
    if per_type and isinstance(rp.get("sliding_attention"), dict):
        kw.update(rope_local_theta=hf_rope_theta(hf_cfg, "sliding_attention"),
                  rope_local_scaling=hf_rope_scaling(hf_cfg, "sliding_attention"))

    if mt in ("qwen2", "qwen2_moe"):
        kw["attention_bias"] = True
    if mt in ("qwen3", "qwen3_moe", "gemma3_text"):
        kw["qk_norm"] = True
    if mt in ("gemma", "gemma2", "gemma3_text"):
        kw.update(norm_offset=1.0, embed_scale=hidden ** 0.5)
    if mt in ("gemma2", "gemma3_text"):
        kw.update(sandwich_norm=True, attn_scale=hf_cfg.query_pre_attn_scalar ** -0.5,
                  attn_softcap=getattr(hf_cfg, "attn_logit_softcapping", None),
                  final_softcap=getattr(hf_cfg, "final_logit_softcapping", None))
    if mt in ("mixtral", "qwen2_moe", "qwen3_moe", "olmoe"):
        E = _get(hf_cfg, "num_local_experts", "num_experts")
        kw.update(num_experts=E, num_experts_per_tok=hf_cfg.num_experts_per_tok,
                  moe_ffn_dim=getattr(hf_cfg, "moe_intermediate_size", None),
                  norm_topk_prob=True if mt == "mixtral" else bool(hf_cfg.norm_topk_prob))
        if mt in ("qwen2_moe", "qwen3_moe"):
            step, dense = hf_cfg.decoder_sparse_step or 1, set(hf_cfg.mlp_only_layers or [])
            kw["moe_layers"] = [i for i in range(kw["num_layers"]) if i not in dense and (i + 1) % step == 0]
        if mt == "qwen2_moe":
            kw["shared_expert_ffn_dim"] = hf_cfg.shared_expert_intermediate_size
    if mt == "olmoe":
        kw.update(qk_norm_full=True, clip_qkv=getattr(hf_cfg, "clip_qkv", None))
    if mt in ("gpt2", "opt", "gpt_neox", "phi"):
        kw.update(norm_type="layer", mlp_gated=False, mlp_bias=True)
    if mt == "gpt2":
        if getattr(hf_cfg, "scale_attn_by_inverse_layer_idx", False) or getattr(hf_cfg, "reorder_and_upcast_attn", False):
            raise ValueError("GPT-2 with scale_attn_by_inverse_layer_idx or reorder_and_upcast_attn is not supported")
        if not getattr(hf_cfg, "scale_attn_weights", True):
            kw["attn_scale"] = 1.0
        kw.update(attention_bias=True, o_proj_bias=True, learned_positions=True)
    if mt == "opt":
        if not getattr(hf_cfg, "layer_norm_elementwise_affine", True):
            raise ValueError("OPT without layer-norm weights (layer_norm_elementwise_affine=False) is not supported")
        kw.update(attention_bias=hf_cfg.enable_bias, o_proj_bias=hf_cfg.enable_bias, mlp_bias=hf_cfg.enable_bias,
                  learned_positions=True, position_offset=2, rms_norm_eps=1e-5, pre_norm=hf_cfg.do_layer_norm_before,
                  final_norm=hf_cfg.do_layer_norm_before and not getattr(hf_cfg, "_remove_final_layer_norm", False))
        if hf_cfg.word_embed_proj_dim != hidden:
            kw["embed_proj_dim"] = hf_cfg.word_embed_proj_dim
    if mt == "gpt_neox":
        kw.update(o_proj_bias=kw["attention_bias"], parallel_residual=hf_cfg.use_parallel_residual)
    if mt == "phi":
        kw.update(attention_bias=True, o_proj_bias=True, lm_head_bias=True, parallel_residual=True,
                  shared_layer_norm=True, qk_norm=bool(getattr(hf_cfg, "qk_layernorm", False)))
    return kw


class _Writer:
    """Puts HF tensors into a Turing model built on the meta device, by parameter name: a tensor already in the target
    dtype and device is adopted without a copy (so a conversion holds one copy of the weights, not two). Remembers
    which parameters were written; check() fails if any was not."""

    def __init__(self, model: nn.Module, device: torch.device, dtype: torch.dtype):
        self.model, self.device, self.dtype = model, device, dtype
        self.shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
        self.written = set()

    def __call__(self, name: str, tensor: torch.Tensor):
        if self.shapes[name] != tuple(tensor.shape):
            raise ValueError(f"{name}: HF tensor {tuple(tensor.shape)} does not fit {self.shapes[name]}")
        owner_name, _, attr = name.rpartition(".")
        owner = self.model.get_submodule(owner_name)
        t = tensor.detach().to(device=self.device, dtype=self.dtype).contiguous()
        owner._parameters[attr] = nn.Parameter(t, requires_grad=False)
        self.written.add(name)

    def check(self):
        missing = sorted(set(self.shapes) - self.written)
        if missing:
            raise ValueError(f"no HF weights for {len(missing)} Turing parameters, e.g. {missing[:5]}")


def _copy_weights(cfg: ModelConfig, sd: Dict[str, torch.Tensor], put: _Writer, layout: str) -> None:
    L = cfg.num_layers
    if layout in ("llama", "gemma2", "phi3"):
        put("embed_tokens.weight", sd["model.embed_tokens.weight"])
        put("norm.weight", sd["model.norm.weight"])
        put("lm_head.weight", sd.get("lm_head.weight", sd["model.embed_tokens.weight"]))
        for i in range(L):
            s, t = f"model.layers.{i}.", f"layers.{i}."
            if layout == "phi3":
                qd, kd = cfg.num_heads * cfg.head_dim, cfg.num_kv_heads * cfg.head_dim
                q, k, v = sd[s + "self_attn.qkv_proj.weight"].split([qd, kd, kd])
                put(t + "self_attn.q_proj.weight", q)
                put(t + "self_attn.k_proj.weight", k)
                put(t + "self_attn.v_proj.weight", v)
                gate, up = sd[s + "mlp.gate_up_proj.weight"].chunk(2, dim=0)
                put(t + "mlp.gate_proj.weight", gate)
                put(t + "mlp.up_proj.weight", up)
                put(t + "mlp.down_proj.weight", sd[s + "mlp.down_proj.weight"])
            else:
                for p in ("q_proj", "k_proj", "v_proj"):
                    put(t + f"self_attn.{p}.weight", sd[s + f"self_attn.{p}.weight"])
                    if cfg.attention_bias:
                        put(t + f"self_attn.{p}.bias", sd[s + f"self_attn.{p}.bias"])
            put(t + "self_attn.o_proj.weight", sd[s + "self_attn.o_proj.weight"])
            if cfg.qk_norm or cfg.qk_norm_full:
                put(t + "self_attn.q_norm.weight", sd[s + "self_attn.q_norm.weight"])
                put(t + "self_attn.k_norm.weight", sd[s + "self_attn.k_norm.weight"])
            put(t + "input_layernorm.weight", sd[s + "input_layernorm.weight"])
            if layout == "gemma2":  # HF's post_attention_layernorm is the attention-output norm here
                put(t + "attn_out_norm.weight", sd[s + "post_attention_layernorm.weight"])
                put(t + "post_attention_layernorm.weight", sd[s + "pre_feedforward_layernorm.weight"])
                put(t + "mlp_out_norm.weight", sd[s + "post_feedforward_layernorm.weight"])
            else:
                put(t + "post_attention_layernorm.weight", sd[s + "post_attention_layernorm.weight"])
            if cfg.layer_is_moe(i):
                put(t + "mlp.router.weight", sd[s + "mlp.gate.weight"])
                put(t + "mlp.gate_up", sd[s + "mlp.experts.gate_up_proj"])
                put(t + "mlp.down", sd[s + "mlp.experts.down_proj"])
                if cfg.shared_expert_ffn_dim:
                    for p in ("gate_proj", "up_proj", "down_proj"):
                        put(t + f"mlp.shared_expert.{p}.weight", sd[s + f"mlp.shared_expert.{p}.weight"])
                    put(t + "mlp.shared_expert_gate.weight", sd[s + "mlp.shared_expert_gate.weight"])
            elif layout != "phi3":
                for p in ("gate_proj", "up_proj", "down_proj"):
                    put(t + f"mlp.{p}.weight", sd[s + f"mlp.{p}.weight"])
        return

    def ln(dst: str, src: str):
        put(dst + ".weight", sd[src + ".weight"])
        put(dst + ".bias", sd[src + ".bias"])

    def lin(dst: str, src: str, bias: bool = True, w: Optional[torch.Tensor] = None, b: Optional[torch.Tensor] = None):
        put(dst + ".weight", sd[src + ".weight"] if w is None else w)
        if bias:
            put(dst + ".bias", sd[src + ".bias"] if b is None else b)

    if layout == "gpt2":
        put("embed_tokens.weight", sd["transformer.wte.weight"])
        put("embed_positions.weight", sd["transformer.wpe.weight"])
        ln("norm", "transformer.ln_f")
        put("lm_head.weight", sd.get("lm_head.weight", sd["transformer.wte.weight"]))
        for i in range(L):
            s, t = f"transformer.h.{i}.", f"layers.{i}."
            ln(t + "input_layernorm", s + "ln_1")
            ln(t + "post_attention_layernorm", s + "ln_2")
            qkv_w, qkv_b = sd[s + "attn.c_attn.weight"].t().chunk(3, dim=0), sd[s + "attn.c_attn.bias"].chunk(3)
            for p, w, b in zip(("q_proj", "k_proj", "v_proj"), qkv_w, qkv_b):
                lin(t + f"self_attn.{p}", "", w=w, b=b)
            lin(t + "self_attn.o_proj", s + "attn.c_proj", w=sd[s + "attn.c_proj.weight"].t())
            lin(t + "mlp.up_proj", s + "mlp.c_fc", w=sd[s + "mlp.c_fc.weight"].t())
            lin(t + "mlp.down_proj", s + "mlp.c_proj", w=sd[s + "mlp.c_proj.weight"].t())
    elif layout == "opt":
        d = "model.decoder."
        put("embed_tokens.weight", sd[d + "embed_tokens.weight"])
        put("embed_positions.weight", sd[d + "embed_positions.weight"])
        if cfg.embed_proj_dim:
            put("project_in.weight", sd[d + "project_in.weight"])
            put("project_out.weight", sd[d + "project_out.weight"])
        if cfg.final_norm:
            ln("norm", d + "final_layer_norm")
        put("lm_head.weight", sd.get("lm_head.weight", sd[d + "embed_tokens.weight"]))
        bias = cfg.attention_bias
        for i in range(L):
            s, t = f"{d}layers.{i}.", f"layers.{i}."
            for p in ("q_proj", "k_proj", "v_proj"):
                lin(t + f"self_attn.{p}", s + f"self_attn.{p}", bias)
            lin(t + "self_attn.o_proj", s + "self_attn.out_proj", bias)
            ln(t + "input_layernorm", s + "self_attn_layer_norm")
            ln(t + "post_attention_layernorm", s + "final_layer_norm")
            lin(t + "mlp.up_proj", s + "fc1", cfg.mlp_bias)
            lin(t + "mlp.down_proj", s + "fc2", cfg.mlp_bias)
    elif layout == "gpt_neox":
        put("embed_tokens.weight", sd["gpt_neox.embed_in.weight"])
        ln("norm", "gpt_neox.final_layer_norm")
        put("lm_head.weight", sd.get("embed_out.weight", sd.get("lm_head.weight")))
        H, D = cfg.num_heads, cfg.head_dim
        for i in range(L):
            s, t = f"gpt_neox.layers.{i}.", f"layers.{i}."
            ln(t + "input_layernorm", s + "input_layernorm")
            ln(t + "post_attention_layernorm", s + "post_attention_layernorm")
            w = sd[s + "attention.query_key_value.weight"].view(H, 3, D, -1)  # per head: q, k, v
            b = sd[s + "attention.query_key_value.bias"].view(H, 3, D) if cfg.attention_bias else None
            for j, p in enumerate(("q_proj", "k_proj", "v_proj")):
                lin(t + f"self_attn.{p}", "", cfg.attention_bias, w=w[:, j].reshape(H * D, -1),
                    b=b[:, j].reshape(H * D) if b is not None else None)
            lin(t + "self_attn.o_proj", s + "attention.dense", cfg.o_proj_bias)
            lin(t + "mlp.up_proj", s + "mlp.dense_h_to_4h")
            lin(t + "mlp.down_proj", s + "mlp.dense_4h_to_h")
    elif layout == "phi":
        put("embed_tokens.weight", sd["model.embed_tokens.weight"])
        ln("norm", "model.final_layernorm")
        lin("lm_head", "lm_head")
        for i in range(L):
            s, t = f"model.layers.{i}.", f"layers.{i}."
            ln(t + "input_layernorm", s + "input_layernorm")
            for p in ("q_proj", "k_proj", "v_proj"):
                lin(t + f"self_attn.{p}", s + f"self_attn.{p}")
            lin(t + "self_attn.o_proj", s + "self_attn.dense")
            if cfg.qk_norm:
                ln(t + "self_attn.q_norm", s + "self_attn.q_layernorm")
                ln(t + "self_attn.k_norm", s + "self_attn.k_layernorm")
            lin(t + "mlp.up_proj", s + "mlp.fc1")
            lin(t + "mlp.down_proj", s + "mlp.fc2")
    else:
        raise ValueError(f"unknown weight layout '{layout}'")


def _prune(model: SubspaceCausalLM, sparsity_ratio: float) -> None:
    """Keep the most salient (1 - sparsity_ratio) of each dense MLP's tiles of intermediate channels."""
    cfg = model.config
    if cfg.num_experts:
        raise ValueError("FFN channel pruning is not supported for MoE models")
    for layer in model.layers:
        mlp = layer.mlp
        score = torch.norm(mlp.up_proj.weight.float(), dim=-1) * torch.norm(mlp.down_proj.weight.float(), dim=0)
        if mlp.gate_proj is not None:
            score = score * torch.norm(mlp.gate_proj.weight.float(), dim=-1)
        tiles = -(-mlp.ffn_dim // mlp.tile_size)
        score = torch.nn.functional.pad(score, (0, tiles * mlp.tile_size - mlp.ffn_dim)).view(tiles, mlp.tile_size)
        keep = max(1, int(tiles * (1.0 - sparsity_ratio)))
        mlp.set_active_tiles(torch.topk(score.mean(-1), k=keep).indices.sort().values)


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
        sparsity_ratio: float = 0.0,
        device: str = "cpu",
        token: Optional[str] = None,
        enable_recirculation: bool = False,
        dtype=None,
    ) -> Tuple[SubspaceCausalLM, Any]:
        """
        Loads a HuggingFace checkpoint of a supported family and maps its weights to SubspaceCausalLM.
        dtype: "bf16", "fp16", "fp32" or None (see resolve_dtype).
        """
        from transformers import AutoConfig
        resolved_id = RealHuggingFaceLoader.resolve_model_id(hf_model_id)
        hf_token = token or os.environ.get("HF_TOKEN") or huggingface_hub.get_token()
        hf_cfg = AutoConfig.from_pretrained(resolved_id, token=hf_token)
        if hf_cfg.model_type not in FAMILIES:  # refuse before downloading the weights
            raise ValueError(f"model type '{hf_cfg.model_type}' is not supported; supported: {', '.join(sorted(FAMILIES))}")
        print(f"[*] Downloading / Loading real pretrained HuggingFace model: '{resolved_id}'...")
        tokenizer = AutoTokenizer.from_pretrained(resolved_id, token=hf_token)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        target_device = torch.device(device)
        ckpt_dtype = getattr(hf_cfg, "dtype", None) or getattr(hf_cfg, "torch_dtype", None)
        if isinstance(ckpt_dtype, str):
            ckpt_dtype = getattr(torch, ckpt_dtype, None)
        target_dtype = resolve_dtype(dtype, target_device, ckpt_dtype)

        hf_model = AutoModelForCausalLM.from_pretrained(resolved_id, token=hf_token, dtype=target_dtype,
                                                        low_cpu_mem_usage=True).eval()
        turing_model = RealHuggingFaceLoader.convert_hf_model(
            hf_model, sparsity_ratio=sparsity_ratio, device=device, enable_recirculation=enable_recirculation,
            name=f"HF-{hf_model_id}", dtype=target_dtype)
        del hf_model
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"[+] Loaded {turing_model.config.num_layers} layers of '{hf_model_id}' ({turing_model.config.arch}, "
              f"{str(target_dtype)[6:]}, {sparsity_ratio * 100:.1f}% FFN channels pruned).")
        return turing_model, tokenizer

    @staticmethod
    def convert_hf_model(
        hf_model: nn.Module,
        sparsity_ratio: float = 0.0,
        device: str = "cpu",
        enable_recirculation: bool = False,
        name: str = "HF-model",
        dtype=None,
    ) -> SubspaceCausalLM:
        """
        Maps the weights of an already-loaded HuggingFace causal LM into SubspaceCausalLM.
        sparsity_ratio=0 with enable_recirculation=False is exact mode: the HF model's own logits.
        """
        target_device = torch.device(device)
        target_dtype = resolve_dtype(dtype, target_device, next(hf_model.parameters()).dtype)
        hf_cfg = hf_model.config
        mt = hf_cfg.model_type
        if mt not in FAMILIES:
            raise ValueError(f"model type '{mt}' is not supported; supported: {', '.join(sorted(FAMILIES))}")
        kw = _config_kwargs(hf_cfg)
        ffn = kw["ffn_dim"]
        tile_size = 64 if ffn <= 4096 else 256
        cfg = ModelConfig(name=name, tile_size=tile_size, active_tiles=-(-ffn // tile_size),
                          rank_sub=64 if kw["hidden_dim"] >= 768 else 32,
                          enable_recirculation=enable_recirculation, **kw)

        with torch.device("meta"):  # no random init: every parameter is put from the HF weights below
            model = SubspaceCausalLM(cfg)
        sd = {k: v.detach() for k, v in hf_model.state_dict().items()}
        if "lm_head.weight" not in sd and getattr(hf_model, "get_output_embeddings", None) and hf_model.get_output_embeddings() is not None:
            sd["lm_head.weight"] = hf_model.get_output_embeddings().weight.detach()
        put = _Writer(model, target_device, target_dtype)
        with torch.no_grad():
            _copy_weights(cfg, sd, put, FAMILIES[mt])
            put.check()
            if enable_recirculation:  # its SVD basis is not an HF weight: build it for real now
                from ..core.subspace import SubspaceRecirculation
                model.recirculation = SubspaceRecirculation(hidden_dim=cfg.hidden_dim, rank=cfg.rank_sub).to(
                    device=target_device, dtype=target_dtype)
            if sparsity_ratio > 0:
                _prune(model, sparsity_ratio)
                cfg.active_tiles = max(1, int(cfg.total_tiles * (1.0 - sparsity_ratio)))
        return model
