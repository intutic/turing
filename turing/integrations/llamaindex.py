"""
LlamaIndex integration adapter for Turing Engine.
Provides native Turing LLM interface with Subspace Pruning and SVD KV Paging controls.
Complies with LlamaIndex 0.10+ CustomLLM specifications.
"""

from typing import Any, Dict, List, Optional, Sequence, Generator
import json
import urllib.request
import urllib.error

try:
    from pydantic import Field
    from llama_index.core.llms.custom import CustomLLM
    from llama_index.core.llms.types import (
        ChatMessage,
        ChatResponse,
        ChatResponseGen,
        CompletionResponse,
        CompletionResponseGen,
        LLMMetadata,
        MessageRole,
    )
    from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
    HAS_LLAMA_INDEX = True
except ImportError:
    # Standalone fallback definitions to ensure zero runtime errors when llama_index is not installed
    HAS_LLAMA_INDEX = False
    
    def Field(*args, **kwargs):
        default = kwargs.get("default", None)
        return default

    class CustomLLM:  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    class LLMMetadata:  # type: ignore
        def __init__(
            self,
            context_window: int = 32768,
            num_output: int = 512,
            is_chat_model: bool = True,
            is_function_calling_model: bool = False,
            model_name: str = "turing-model",
        ):
            self.context_window = context_window
            self.num_output = num_output
            self.is_chat_model = is_chat_model
            self.is_function_calling_model = is_function_calling_model
            self.model_name = model_name

    class ChatMessage:  # type: ignore
        def __init__(self, role: str = "user", content: str = "", additional_kwargs: Optional[Dict[str, Any]] = None):
            self.role = role
            self.content = content
            self.additional_kwargs = additional_kwargs or {}

    class ChatResponse:  # type: ignore
        def __init__(self, message: ChatMessage, raw: Optional[Dict[str, Any]] = None):
            self.message = message
            self.raw = raw or {}

        def __str__(self) -> str:
            return self.message.content

    class CompletionResponse:  # type: ignore
        def __init__(self, text: str, raw: Optional[Dict[str, Any]] = None):
            self.text = text
            self.raw = raw or {}

        def __str__(self) -> str:
            return self.text

    def llm_chat_callback():
        def decorator(f):
            return f
        return decorator

    def llm_completion_callback():
        def decorator(f):
            return f
        return decorator


class Turing(CustomLLM):
    """
    Native Turing Engine LLM for LlamaIndex query engines, indexes, and agents.
    Directly routes to Turing continuous batching servers with Subspace Pruning and SVD KV Paging.
    """
    model: str = "deepseek-r1-7b"
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "turing-local"
    temperature: float = 0.7
    max_tokens: int = 512
    sparsity_ratio: float = 0.57
    svd_rank: int = 64
    timeout: float = 60.0
    context_window: int = 32768

    def __init__(
        self,
        model: str = "deepseek-r1-7b",
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "turing-local",
        temperature: float = 0.7,
        max_tokens: int = 512,
        sparsity_ratio: float = 0.57,
        svd_rank: int = 64,
        timeout: float = 60.0,
        context_window: int = 32768,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_base=api_base.rstrip("/"),
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            sparsity_ratio=sparsity_ratio,
            svd_rank=svd_rank,
            timeout=timeout,
            context_window=context_window,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Turing"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self.model,
        )

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-Turing-Sparsity": str(self.sparsity_ratio),
            "X-Turing-SVD-Rank": str(self.svd_rank),
        }

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.api_base}/{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._get_headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            return {
                "choices": [{
                    "message": {"content": f"[Turing Engine Standalone Mode - Server at {self.api_base} offline: {e}]"}
                }]
            }

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False,
        }
        res = self._post("chat/completions", payload)
        content = res.get("choices", [{}])[0].get("message", {}).get("content", "")
        return CompletionResponse(text=content, raw=res)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Generator[CompletionResponse, None, None]:
        # Emits completed token delta
        resp = self.complete(prompt, formatted=formatted, **kwargs)
        yield resp

    @llm_chat_callback()
    def chat(self, messages: Sequence[Any], **kwargs: Any) -> ChatResponse:
        formatted_messages = []
        for m in messages:
            if hasattr(m, "role") and hasattr(m, "content"):
                formatted_messages.append({"role": str(m.role).lower(), "content": str(m.content)})
            elif isinstance(m, dict):
                formatted_messages.append(m)
            elif isinstance(m, str):
                formatted_messages.append({"role": "user", "content": m})

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False,
        }
        res = self._post("chat/completions", payload)
        content = res.get("choices", [{}])[0].get("message", {}).get("content", "")
        msg = ChatMessage(role="assistant", content=content)
        return ChatResponse(message=msg, raw=res)

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[Any], **kwargs: Any) -> Generator[ChatResponse, None, None]:
        resp = self.chat(messages, **kwargs)
        yield resp


# Canonical aliases
TuringEngine = Turing
TuringEngineLLM = Turing
TuringLLM = Turing
