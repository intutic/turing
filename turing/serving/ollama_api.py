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
    parameter_size: str = "7B"
    quantization_level: str = "w4a16"


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
    def format_tags_response(active_model_name: str, model_config: ModelConfig) -> Dict[str, Any]:
        iso_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        spec = ModelResolver.parse(active_model_name)
        param_est = f"{model_config.num_layers * model_config.hidden_dim * 12 // 1_000_000_000}B" if model_config.num_layers else "7B"
        
        details = OllamaModelDetails(
            parent_model="",
            format="subspace",
            family="llama" if "llama" in spec.model_name.lower() else ("qwen" if "qwen" in spec.model_name.lower() else "deepseek"),
            families=[spec.model_name.lower()],
            parameter_size=param_est,
            quantization_level="w4a16"
        )
        
        info = OllamaModelInfo(
            name=active_model_name,
            model=spec.repo_id,
            modified_at=iso_now,
            size=int(model_config.hidden_dim * model_config.ffn_dim * model_config.num_layers * 0.5) if model_config else 4_000_000_000,
            digest=f"sha256:turing{abs(hash(active_model_name)):016x}",
            details=details
        )
        return {"models": [info.model_dump()]}

    @staticmethod
    def format_show_response(model_name: str, model_config: ModelConfig) -> Dict[str, Any]:
        spec = ModelResolver.parse(model_name)
        iso_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        param_est = f"{model_config.num_layers * model_config.hidden_dim * 12 // 1_000_000_000}B" if model_config.num_layers else "7B"
        
        return {
            "license": "Business Source License 1.1 (BSL 1.1) / Open Apache 2.0 conversion 2030",
            "modelfile": f"# Turing Engine Native Modelfile\nFROM {spec.repo_id}\nPARAMETER temperature 0.7\nPARAMETER top_p 0.9\nPARAMETER stop <|im_end|>\n",
            "parameters": f"stop                           \"<|im_end|>\"\ntemperature                    0.7\ntop_p                          0.9\n",
            "template": "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}<|im_start|>assistant\n{{ .Response }}<|im_end|>",
            "details": {
                "parent_model": "",
                "format": "subspace",
                "family": "llama" if "llama" in spec.model_name.lower() else ("qwen" if "qwen" in spec.model_name.lower() else "deepseek"),
                "families": [spec.model_name.lower()],
                "parameter_size": param_est,
                "quantization_level": "w4a16"
            },
            "model_info": {
                "general.architecture": "subspace_causal_lm",
                "general.file_type": 1,
                "general.parameter_count": model_config.num_layers * model_config.hidden_dim * 12,
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
    def format_ps_response(active_model_name: str, model_config: ModelConfig, vram_bytes: int = 0) -> Dict[str, Any]:
        iso_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        expires_at = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=30)).isoformat()
        spec = ModelResolver.parse(active_model_name)
        
        proc = OllamaProcessInfo(
            name=active_model_name,
            model=spec.repo_id,
            size=vram_bytes or 4_294_967_296,
            digest=f"sha256:turing{abs(hash(active_model_name)):016x}",
            details=OllamaModelDetails(
                parent_model="",
                format="subspace",
                family="llama",
                families=[spec.model_name.lower()],
                parameter_size="7B",
                quantization_level="w4a16"
            ),
            expires_at=expires_at,
            size_vram=vram_bytes or 4_294_967_296
        )
        return {"models": [proc.model_dump()]}
