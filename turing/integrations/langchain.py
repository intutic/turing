"""
LangChain chat model for a Turing Engine server.

`ChatTuring` is a `langchain_core` `BaseChatModel` that talks to `turing serve` over its
OpenAI-compatible `/v1/chat/completions` endpoint: `invoke`, `stream`, `batch`, async (via the
base class) and tool calling with `bind_tools`. Needs `pip install langchain-core`.
"""

import json
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

import httpx

try:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        BaseMessage,
        ChatMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langchain_core.tools import BaseTool
    from langchain_core.utils.function_calling import convert_to_openai_tool
except ImportError as e:  # optional dependency
    raise ImportError("turing.integrations.langchain needs langchain-core: pip install langchain-core") from e


def _text(content: Union[str, List[Any]]) -> str:
    """The server takes string content only; keep the text parts of a multi-part message."""
    if isinstance(content, str):
        return content
    return "".join(p if isinstance(p, str) else p.get("text", "") for p in content if isinstance(p, (str, dict)))


def _to_openai(m: BaseMessage) -> Dict[str, Any]:
    if isinstance(m, SystemMessage):
        return {"role": "system", "content": _text(m.content)}
    if isinstance(m, HumanMessage):
        return {"role": "user", "content": _text(m.content)}
    if isinstance(m, AIMessage):
        out: Dict[str, Any] = {"role": "assistant", "content": _text(m.content)}
        if m.tool_calls:
            out["tool_calls"] = [
                {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                for tc in m.tool_calls
            ]
        return out
    if isinstance(m, ToolMessage):
        return {"role": "tool", "content": _text(m.content), "tool_call_id": m.tool_call_id}
    if isinstance(m, ChatMessage):
        return {"role": m.role, "content": _text(m.content)}
    raise TypeError(f"Unsupported message type: {type(m).__name__}")


def _chat_result(data: Dict[str, Any]) -> ChatResult:
    """An OpenAI chat completion (the server's response body) as a LangChain ChatResult."""
    choice = data["choices"][0]
    msg = choice["message"]
    usage = data.get("usage") or {}
    tool_calls = [
        {"name": tc["function"]["name"], "args": json.loads(tc["function"]["arguments"] or "{}"), "id": tc.get("id")}
        for tc in msg.get("tool_calls") or []
    ]
    message = AIMessage(
        content=msg.get("content") or "",
        tool_calls=tool_calls,
        response_metadata={"finish_reason": choice.get("finish_reason"), "model_name": data.get("model")},
        usage_metadata={"input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)} if usage else None,
    )
    return ChatResult(generations=[ChatGeneration(message=message)],
                      llm_output={"token_usage": usage, "model_name": data.get("model")})


class ChatTuring(BaseChatModel):
    """Chat model backed by a Turing Engine server (`turing serve`)."""

    model: str = "smollm2"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "turing-local"
    temperature: float = 0.7
    max_tokens: int = 512
    sparsity_ratio: Optional[float] = None  # None: the server's default (0.0, the exact model)
    timeout: float = 60.0

    @property
    def _llm_type(self) -> str:
        return "turing"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model, "base_url": self.base_url, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "sparsity_ratio": self.sparsity_ratio}

    def _request(self, messages: List[BaseMessage], stop: Optional[List[str]], stream: bool, **kwargs: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [_to_openai(m) for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if self.sparsity_ratio is not None:
            payload["sparsity_ratio"] = self.sparsity_ratio
        if stop:
            payload["stop"] = stop
        payload.update(kwargs)  # per-call overrides and bind_tools' tools / tool_choice
        return {"url": f"{self.base_url.rstrip('/')}/chat/completions", "json": payload,
                "headers": {"Authorization": f"Bearer {self.api_key}"}, "timeout": self.timeout}

    def _connection_error(self, e: Exception) -> ConnectionError:
        return ConnectionError(f"Failed to connect to Turing Engine at {self.base_url}. "
                               f"Is `turing serve --model {self.model}` running? ({e})")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            resp = httpx.post(**self._request(messages, stop, False, **kwargs))
        except httpx.TransportError as e:
            raise self._connection_error(e) from e
        resp.raise_for_status()
        return _chat_result(resp.json())

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        try:
            with httpx.stream("POST", **self._request(messages, stop, True, **kwargs)) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    choice = json.loads(line[len("data: "):])["choices"][0]
                    text = choice["delta"].get("content") or ""
                    finish = choice.get("finish_reason")
                    meta = {"finish_reason": finish} if finish else {}
                    chunk = ChatGenerationChunk(message=AIMessageChunk(content=text, response_metadata=meta),
                                                generation_info=meta or None)
                    if run_manager:
                        run_manager.on_llm_new_token(text, chunk=chunk)
                    yield chunk
        except httpx.TransportError as e:
            raise self._connection_error(e) from e

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ):
        """Send `tools` (and `tool_choice`) with every request; tool calls come back on `AIMessage.tool_calls`."""
        kwargs["tools"] = [convert_to_openai_tool(t) for t in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = (tool_choice if tool_choice in ("auto", "none", "required")
                                     else {"type": "function", "function": {"name": tool_choice}})
        return self.bind(**kwargs)


ChatTuringEngine = ChatTuring

__all__ = ["ChatTuring", "ChatTuringEngine"]
