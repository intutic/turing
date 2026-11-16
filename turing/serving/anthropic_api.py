"""
Anthropic Messages API (/v1/messages) Server Adapter (Turing Engine Integration).
Provides compatibility with Claude Code, Cursor, Cline, and the Anthropic SDK: system prompts, text / tool_use /
tool_result content blocks, tools and tool_choice, stop_sequences, extended thinking, and the SSE event stream.
"""

import json
import uuid
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel

from .generation import bad_request, plain_prompt


class AnthropicContentBlock(BaseModel):
    type: str = "text"
    text: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    thinking: Optional[str] = None
    signature: Optional[str] = None


class AnthropicMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class AnthropicMessageRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 50
    stream: Optional[bool] = False
    stop_sequences: Optional[List[str]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    draft_tokens: Optional[int] = None
    lane: Optional[str] = None


class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0


class AnthropicMessageResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[AnthropicContentBlock]
    model: str
    stop_reason: Optional[str] = "end_turn"
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage


def _block_text(block: Dict[str, Any]) -> str:
    kind = block.get("type")
    if kind == "text":
        return block.get("text") or ""
    raise bad_request(f"content block type '{kind}' is not supported here (text only)")


def _result_text(content: Any) -> str:
    if isinstance(content, str) or content is None:
        return content or ""
    return "\n".join(_block_text(b) for b in content)


def _stop_reason(finish_reason: str, stop_sequence: Optional[str]) -> str:
    if stop_sequence is not None:
        return "stop_sequence"
    return {"tool_calls": "tool_use", "length": "max_tokens"}.get(finish_reason, "end_turn")


def _tool_use_id(call_id: str) -> str:
    return "toolu_" + call_id.split("_", 1)[-1]


def tool_input(arguments: str) -> Dict[str, Any]:
    """A call's JSON-string arguments as the tool_use input object."""
    try:
        value = json.loads(arguments or "{}")
    except ValueError:
        return {"_raw": arguments}
    return value if isinstance(value, dict) else {"_value": value}


class AnthropicAPIHandler:
    """
    Converts Anthropic Messages requests into chat-template messages and generation results into Anthropic
    responses and SSE events.
    """

    @staticmethod
    def to_chat_messages(req: AnthropicMessageRequest) -> List[Dict[str, Any]]:
        """system + messages as chat-template messages: tool_use blocks become assistant tool_calls, tool_result
        blocks become tool messages; image (or any other non-text) blocks are refused."""
        out: List[Dict[str, Any]] = []
        if req.system:
            sys_text = req.system if isinstance(req.system, str) else "\n".join(_block_text(b) for b in req.system)
            out.append({"role": "system", "content": sys_text})
        for m in req.messages:
            if m.role not in ("user", "assistant"):
                raise bad_request(f"message role must be user or assistant, not '{m.role}'")
            if isinstance(m.content, str):
                out.append({"role": m.role, "content": m.content})
                continue
            text, calls, results = [], [], []
            for b in m.content:
                kind = b.get("type")
                if kind == "tool_use" and m.role == "assistant":
                    calls.append({"id": b.get("id"), "type": "function",
                                  "function": {"name": b.get("name"), "arguments": b.get("input") or {}}})
                elif kind == "tool_result" and m.role == "user":
                    results.append({"role": "tool", "tool_call_id": b.get("tool_use_id"),
                                    "content": _result_text(b.get("content"))})
                elif kind in ("thinking", "redacted_thinking") and m.role == "assistant":
                    continue  # earlier turns' thinking is not fed back (as the chat templates do)
                else:
                    text.append(_block_text(b))
            out.extend(results)
            if text or calls or not results:
                msg: Dict[str, Any] = {"role": m.role, "content": "\n".join(text)}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
        return out

    @staticmethod
    def extract_prompt_from_request(req: AnthropicMessageRequest) -> str:
        """The plain prompt (models without a chat template)."""
        return plain_prompt(AnthropicAPIHandler.to_chat_messages(req))

    @staticmethod
    def thinking(req: AnthropicMessageRequest):
        """(thinking enabled, budget_tokens) from the request's thinking field."""
        t = req.thinking
        if not t:
            return None, None
        if t.get("type") == "disabled":
            return False, 0
        if t.get("type") == "enabled":
            budget = t.get("budget_tokens")
            if not isinstance(budget, int) or budget < 0:
                raise bad_request("thinking.budget_tokens must be a non-negative integer")
            return True, budget
        raise bad_request("thinking.type must be enabled or disabled")

    @staticmethod
    def format_non_streaming_response(
        req: AnthropicMessageRequest,
        generated_text: str,
        input_tokens_count: int,
        output_tokens_count: int,
        reasoning: str = "",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        finish_reason: str = "stop",
        stop_sequence: Optional[str] = None,
        cached_tokens: int = 0,
    ) -> AnthropicMessageResponse:
        content = []
        if reasoning:
            content.append(AnthropicContentBlock(type="thinking", thinking=reasoning, signature=""))
        if generated_text or not (reasoning or tool_calls):
            content.append(AnthropicContentBlock(type="text", text=generated_text))
        for tc in tool_calls or []:
            content.append(AnthropicContentBlock(type="tool_use", id=_tool_use_id(tc["id"]),
                                                 name=tc["function"]["name"],
                                                 input=tool_input(tc["function"]["arguments"])))
        return AnthropicMessageResponse(
            id=f"msg_{uuid.uuid4().hex[:24]}",
            content=content,
            model=req.model,
            stop_reason=_stop_reason(finish_reason, stop_sequence),
            stop_sequence=stop_sequence,
            usage=AnthropicUsage(input_tokens=input_tokens_count, output_tokens=output_tokens_count,
                                 cache_read_input_tokens=cached_tokens),
        )

    @staticmethod
    async def stream_generator(req: AnthropicMessageRequest, gen):
        """Anthropic SSE events for a Generation: thinking and text blocks as they stream, tool_use blocks at the
        end, then message_delta with the stop reason and usage."""
        def event(name: str, payload: Dict[str, Any]) -> str:
            return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

        yield event("message_start", {"type": "message_start", "message": {
            "id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant", "content": [],
            "model": req.model, "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": gen.prompt_tokens, "output_tokens": 0,
                      "cache_read_input_tokens": gen.cached_tokens}}})
        index, open_kind = -1, None

        def close():
            return event("content_block_stop", {"type": "content_block_stop", "index": index})

        async for kind, text in gen.events():
            block = "thinking" if kind == "reasoning" else "text"
            if block != open_kind:
                if open_kind is not None:
                    yield close()
                index, open_kind = index + 1, block
                start = {"type": "thinking", "thinking": "", "signature": ""} if block == "thinking" else \
                    {"type": "text", "text": ""}
                yield event("content_block_start", {"type": "content_block_start", "index": index,
                                                    "content_block": start})
            delta = {"type": "thinking_delta", "thinking": text} if block == "thinking" else \
                {"type": "text_delta", "text": text}
            yield event("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})
        if open_kind is not None:
            yield close()
        for tc in gen.tool_calls:
            index += 1
            yield event("content_block_start", {"type": "content_block_start", "index": index, "content_block": {
                "type": "tool_use", "id": _tool_use_id(tc["id"]), "name": tc["function"]["name"], "input": {}}})
            yield event("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {
                "type": "input_json_delta", "partial_json": tc["function"]["arguments"]}})
            yield close()
        delta = {"stop_reason": _stop_reason(gen.finish_reason, gen.stop_sequence),
                 "stop_sequence": gen.stop_sequence}
        yield event("message_delta", {"type": "message_delta", "delta": delta,
                                      "usage": {"output_tokens": gen.completion_tokens}})
        yield event("message_stop", {"type": "message_stop"})
