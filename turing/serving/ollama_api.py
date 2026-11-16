"""
Native Ollama REST API (/api/*) Adapter for Turing Engine.
Provides full drop-in compatibility for Open WebUI, Continue.dev, Cursor Ollama backend,
Ollama Python SDK, Chatbox, and Ollama CLI.
"""

import json
import time
import datetime
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

from ..config import ModelConfig
from ..models.resolver import ModelResolver


class OllamaMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: Optional[bool] = True
    raw: Optional[bool] = False
    format: Optional[Union[str, Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = None
    images: Optional[List[str]] = None


class OllamaChatRequest(BaseModel):
    model: str
    messages: List[OllamaMessage]
    stream: Optional[bool] = True
    format: Optional[Union[str, Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None


class OllamaShowRequest(BaseModel):
    model: str
    verbose: Optional[bool] = False


class OllamaModelDetails(BaseModel):
    parent_model: str = ""
    format: str = "subspace"
    family: str = "llama"
    families: List[str] = []
    parameter_size: str = ""
    quantization_level: str = ""


class OllamaModelInfo(BaseModel):
    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: OllamaModelDetails


class OllamaTagsResponse(BaseModel):
    models: List[OllamaModelInfo]


class OllamaProcessInfo(BaseModel):
    name: str
    model: str
    size: int
    digest: str
    details: OllamaModelDetails
    expires_at: str
    size_vram: int


class OllamaPsResponse(BaseModel):
    models: List[OllamaProcessInfo]


def _model_facts(model_config: ModelConfig, model=None) -> Dict[str, Any]:
    """Family, parameter count, weight bytes and quantization label of the served model, from the model itself."""
    params, nbytes, dtype = 0, 0, None
    if model is not None:
        for t in list(model.parameters()) + list(model.buffers()):
            nbytes += t.numel() * t.element_size()
            if t.is_floating_point():
                params += t.numel()
                dtype = dtype or t.dtype
        for m in model.modules():  # packed 4-bit layers hold two weights per byte of qweight
            if hasattr(m, "qweight"):
                params += m.qweight.numel() * 2
    q = getattr(model_config, "quantization", None)
    label = {"w4a16-gptq": "Q4 (GPTQ)", "w4a16-rtn": "Q4 (RTN)"}.get(q) or \
        {"torch.float16": "F16", "torch.bfloat16": "BF16", "torch.float32": "F32"}.get(str(dtype), "unknown")
    size = f"{params / 1e9:.1f}B" if params >= 1e9 else f"{params / 1e6:.0f}M"
    return {"family": getattr(model_config, "arch", "llama"), "params": params, "bytes": nbytes, "label": label,
            "parameter_size": size}


class OllamaAPIHandler:
    """
    Transforms Ollama /api/* requests and responses into Turing Engine representations.
    """

    @staticmethod
    def extract_prompt_from_generate_request(req: OllamaGenerateRequest) -> str:
        parts = []
        if req.system:
            parts.append(f"System: {req.system}")
        parts.append(f"User: {req.prompt}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    @staticmethod
    def extract_prompt_from_chat_request(req: OllamaChatRequest) -> str:
        parts = []
        for m in req.messages:
            role = "System" if m.role == "system" else ("User" if m.role == "user" else "Assistant")
            parts.append(f"{role}: {m.content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    @staticmethod
    def format_non_streaming_generate(
        req: OllamaGenerateRequest,
        response_text: str,
        context_tokens: List[int],
        prompt_tokens: int,
        completion_tokens: int,
        total_duration_ns: int = 0,
        eval_duration_ns: int = 0
    ) -> Dict[str, Any]:
        iso_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return {
            "model": req.model,
            "created_at": iso_now,
            "response": response_text,
            "done": True,
            "done_reason": "stop",
            "context": context_tokens,
            "total_duration": total_duration_ns or 50_000_000,
            "load_duration": 1_000_000,
            "prompt_eval_count": prompt_tokens,
            "prompt_eval_duration": 10_000_000,
            "eval_count": completion_tokens,
            "eval_duration": eval_duration_ns or 40_000_000,
        }

    @staticmethod
    def format_streaming_generate_chunk(
        req: OllamaGenerateRequest,
        token_text: str,
        done: bool = False,
        context_tokens: Optional[List[int]] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0
    ) -> str:
        iso_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        payload = {
            "model": req.model,
            "created_at": iso_now,
            "response": token_text,
            "done": done
        }
        if done:
            payload.update({
                "done_reason": "stop",
                "context": context_tokens or [],
                "total_duration": 50_000_000,
                "load_duration": 1_000_000,
                "prompt_eval_count": prompt_tokens,
                "prompt_eval_duration": 10_000_000,
                "eval_count": completion_tokens,
                "eval_duration": 40_000_000,
            })
        return json.dumps(payload) + "\n"

    @staticmethod
    def format_non_streaming_chat(
        req: OllamaChatRequest,
        message_content: str,
        prompt_tokens: int,
        completion_tokens: int,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        total_duration_ns: int = 0
    ) -> Dict[str, Any]:
        iso_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": message_content
        }
        if tool_calls:
            msg["tool_calls"] = tool_calls
            
        return {
            "model": req.model,
            "created_at": iso_now,
            "message": msg,
            "done": True,
            "done_reason": "stop",
            "total_duration": total_duration_ns or 50_000_000,
            "load_duration": 1_000_000,
            "prompt_eval_count": prompt_tokens,
            "prompt_eval_duration": 10_000_000,
            "eval_count": completion_tokens,
            "eval_duration": 40_000_000,
        }

    @staticmethod
    def format_streaming_chat_chunk(
        req: OllamaChatRequest,
        delta_text: str,
        done: bool = False,
        prompt_tokens: int = 0,
        completion_tokens: int = 0
    ) -> str:
        iso_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        payload = {
            "model": req.model,
            "created_at": iso_now,
            "message": {
                "role": "assistant",
                "content": delta_text
            },
            "done": done
        }
        if done:
            payload.update({
                "done_reason": "stop",
                "total_duration": 50_000_000,
                "load_duration": 1_000_000,
                "prompt_eval_count": prompt_tokens,
                "prompt_eval_duration": 10_000_000,
                "eval_count": completion_tokens,
                "eval_duration": 40_000_000,
            })
        return json.dumps(payload) + "\n"

    @staticmethod
    def format_tags_response(active_model_name: str, model_config: ModelConfig, model=None) -> Dict[str, Any]:
        iso_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        spec = ModelResolver.parse(active_model_name)
        f = _model_facts(model_config, model)
        details = OllamaModelDetails(
            parent_model="",
            format="subspace",
            family=f["family"],
            families=[f["family"]],
            parameter_size=f["parameter_size"],
            quantization_level=f["label"]
        )
        
        info = OllamaModelInfo(
            name=active_model_name,
            model=spec.repo_id,
            modified_at=iso_now,
            size=f["bytes"],
            digest=f"sha256:turing{abs(hash(active_model_name)):016x}",
            details=details
        )
        return {"models": [info.model_dump()]}

    @staticmethod
    def format_show_response(model_name: str, model_config: ModelConfig, model=None, chat_template=None) -> Dict[str, Any]:
        spec = ModelResolver.parse(model_name)
        iso_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        f = _model_facts(model_config, model)
        
        return {
            "license": "Business Source License 1.1 (BSL 1.1) / Open Apache 2.0 conversion 2030",
            "modelfile": f"# Turing Engine Native Modelfile\nFROM {spec.repo_id}\nPARAMETER temperature 0.7\nPARAMETER top_p 0.9\nPARAMETER stop <|im_end|>\n",
            "parameters": f"stop                           \"<|im_end|>\"\ntemperature                    0.7\ntop_p                          0.9\n",
            "template": chat_template or "",  # the tokenizer's own (Jinja) chat template, which the server applies
            "details": {
                "parent_model": "",
                "format": "subspace",
                "family": f["family"],
                "families": [f["family"]],
                "parameter_size": f["parameter_size"],
                "quantization_level": f["label"]
            },
            "model_info": {
                "general.architecture": f["family"],
                "general.parameter_count": f["params"],
                "turing.hidden_size": model_config.hidden_dim,
                "turing.intermediate_size": model_config.ffn_dim,
                "turing.num_hidden_layers": model_config.num_layers,
                "turing.num_attention_heads": model_config.num_heads,
                "turing.num_key_value_heads": model_config.num_kv_heads,
                "turing.active_tiles": model_config.active_tiles,
                "turing.sparsity_ratio": model_config.sparsity_ratio,
            },
            "modified_at": iso_now
        }

    @staticmethod
    def format_ps_response(active_model_name: str, model_config: ModelConfig, vram_bytes: int = 0, model=None) -> Dict[str, Any]:
        iso_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        expires_at = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=30)).isoformat()
        spec = ModelResolver.parse(active_model_name)
        
        f = _model_facts(model_config, model)
        proc = OllamaProcessInfo(
            name=active_model_name,
            model=spec.repo_id,
            size=vram_bytes,
            digest=f"sha256:turing{abs(hash(active_model_name)):016x}",
            details=OllamaModelDetails(
                parent_model="",
                format="subspace",
                family=f["family"],
                families=[f["family"]],
                parameter_size=f["parameter_size"],
                quantization_level=f["label"]
            ),
            expires_at=expires_at,
            size_vram=vram_bytes
        )
        return {"models": [proc.model_dump()]}
