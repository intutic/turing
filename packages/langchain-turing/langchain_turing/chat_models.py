"""
LangChain chat model for a Turing Engine server.

`ChatTuring` is a `langchain_core` `BaseChatModel` that talks to `turing serve` over its
OpenAI-compatible `/v1/chat/completions` endpoint: `invoke`, `stream`, `batch`, async (via the
base class) and tool calling with `bind_tools`.
"""

import json
import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

import httpx

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LangSmithParams
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
from pydantic import Field


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


def _raise_for_status(resp: httpx.Response) -> None:
    """raise_for_status with the server's error message (it says which field or value it rejected)."""
    if resp.is_error:
        resp.read()
        raise httpx.HTTPStatusError(f"Turing Engine returned {resp.status_code}: {resp.text}",
                                    request=resp.request, response=resp)


def _usage(u: Dict[str, Any]) -> Dict[str, int]:
    return {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0),
            "total_tokens": u.get("total_tokens", 0)}


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
        usage_metadata=_usage(usage) if usage else None,
    )
    return ChatResult(generations=[ChatGeneration(message=message)],
                      llm_output={"token_usage": usage, "model_name": data.get("model")})


class ChatTuring(BaseChatModel):
    """Chat model backed by a Turing Engine server (`turing serve`)."""

    model: str = "smollm2"
    base_url: str = Field(default_factory=lambda: os.getenv("TURING_BASE_URL", "http://localhost:8000/v1"))
    api_key: str = Field(default_factory=lambda: os.getenv("TURING_API_KEY", "turing-local"))  # checked only with `turing serve --api-key`
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: float = 60.0

    @property
    def _llm_type(self) -> str:
        return "turing"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model, "base_url": self.base_url, "temperature": self.temperature,
                "max_tokens": self.max_tokens}

    def _get_ls_params(self, stop: Optional[List[str]] = None, **kwargs: Any) -> LangSmithParams:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params.update(ls_provider="turing", ls_model_name=kwargs.get("model", self.model),
                      ls_temperature=kwargs.get("temperature", self.temperature),
                      ls_max_tokens=kwargs.get("max_tokens", self.max_tokens))
        return params

    def _request(self, messages: List[BaseMessage], stop: Optional[List[str]], stream: bool, **kwargs: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [_to_openai(m) for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
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
        _raise_for_status(resp)
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
                _raise_for_status(resp)
                for line in resp.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    data = json.loads(line[len("data: "):])
                    if not data["choices"]:  # the stream_options.include_usage chunk
                        yield ChatGenerationChunk(message=AIMessageChunk(
                            content="", usage_metadata=_usage(data["usage"]),
                            response_metadata={"model_name": data.get("model"), "model_provider": "turing"}))
                        continue
                    choice = data["choices"][0]
                    text = choice["delta"].get("content") or ""
                    tool_chunks = [{"name": tc["function"]["name"], "args": tc["function"]["arguments"],
                                    "id": tc.get("id"), "index": tc.get("index", i)}
                                   for i, tc in enumerate(choice["delta"].get("tool_calls") or [])]
                    finish = choice.get("finish_reason")
                    meta = {"finish_reason": finish} if finish else {}
                    chunk = ChatGenerationChunk(message=AIMessageChunk(content=text, tool_call_chunks=tool_chunks,
                                                                       response_metadata=meta),
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
        tool_choice: Optional[Union[str, bool, Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        """Send `tools` (and `tool_choice`) with every request; tool calls come back on `AIMessage.tool_calls`.
        tool_choice: "auto", "none", "required" (or LangChain's "any" / True), a tool name, or an OpenAI dict."""
        kwargs["tools"] = [convert_to_openai_tool(t) for t in tools]
        if tool_choice is True or tool_choice == "any":
            tool_choice = "required"
        if tool_choice is not None and tool_choice is not False:
            kwargs["tool_choice"] = (tool_choice if isinstance(tool_choice, dict) or tool_choice in ("auto", "none", "required")
                                     else {"type": "function", "function": {"name": tool_choice}})
        return self.bind(**kwargs)


__all__ = ["ChatTuring"]
