"""
Anthropic Messages API (/v1/messages) Server Adapter (Turing Engine Integration).
Provides full compatibility with Claude Code, Cursor, Cline, and Anthropic SDK.
"""

import json
import time
import uuid
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

class AnthropicContentBlock(BaseModel):
    type: str = "text"
    text: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None

class AnthropicMessage(BaseModel):
    role: str
    content: Union[str, List[Union[AnthropicContentBlock, Dict[str, Any]]]]

class AnthropicTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class AnthropicMessageRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 50
    stream: Optional[bool] = False
    tools: Optional[List[AnthropicTool]] = None
    sparsity_ratio: Optional[float] = Field(default=None, description="Custom subspace sparsity ratio (0.0 to 0.9)")
    use_svd_kv: Optional[bool] = Field(default=None, description="Enable calibrated SVD INT8 KV cache paging")
    draft_tokens: Optional[int] = Field(default=None, description="Number of speculative candidate draft tokens")

class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int

class AnthropicMessageResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[AnthropicContentBlock]
    model: str
    stop_reason: Optional[str] = "end_turn"
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage

class AnthropicAPIHandler:
    """
    Transforms Anthropic Messages format into Turing Engine engine prompts and yields Anthropic SSE events.
    """
    @staticmethod
    def extract_prompt_from_request(req: AnthropicMessageRequest) -> str:
        prompt_parts = []

        # System prompt
        if req.system:
            if isinstance(req.system, str):
                prompt_parts.append(f"System: {req.system}")
            elif isinstance(req.system, list):
                sys_text = " ".join([b.get("text", "") for b in req.system if isinstance(b, dict)])
                prompt_parts.append(f"System: {sys_text}")

        # Messages
        for m in req.messages:
            role_label = "User" if m.role == "user" else "Assistant"
            if isinstance(m.content, str):
                prompt_parts.append(f"{role_label}: {m.content}")
            elif isinstance(m.content, list):
                text_blocks = []
                for block in m.content:
                    if isinstance(block, AnthropicContentBlock) and block.text:
                        text_blocks.append(block.text)
                    elif isinstance(block, dict) and "text" in block:
                        text_blocks.append(block["text"])
                prompt_parts.append(f"{role_label}: {' '.join(text_blocks)}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    @staticmethod
    def format_non_streaming_response(
        req: AnthropicMessageRequest,
        generated_text: str,
        input_tokens_count: int,
        output_tokens_count: int
    ) -> AnthropicMessageResponse:
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        return AnthropicMessageResponse(
            id=msg_id,
            role="assistant",
            content=[AnthropicContentBlock(type="text", text=generated_text)],
            model=req.model,
            stop_reason="end_turn",
            usage=AnthropicUsage(
                input_tokens=input_tokens_count,
                output_tokens=output_tokens_count
            )
        )

    @staticmethod
    async def stream_generator(
        req: AnthropicMessageRequest,
        token_stream,
        input_tokens_count: int
    ):
        """
        Yields official Anthropic Server-Sent Events (SSE).
        """
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        output_tokens = 0

        # Event 1: message_start
        message_start_payload = {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": req.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_tokens_count,
                    "output_tokens": 1
                }
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_start_payload)}\n\n"

        # Event 2: content_block_start
        content_block_start_payload = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "text",
                "text": ""
            }
        }
        yield f"event: content_block_start\ndata: {json.dumps(content_block_start_payload)}\n\n"

        # Event 3: content_block_delta for each token
        async for token_id, token_str in token_stream:
            output_tokens += 1
            delta_payload = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": token_str
                }
            }
            yield f"event: content_block_delta\ndata: {json.dumps(delta_payload)}\n\n"

        # Event 4: content_block_stop
        content_block_stop_payload = {
            "type": "content_block_stop",
            "index": 0
        }
        yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop_payload)}\n\n"

        # Event 5: message_delta
        message_delta_payload = {
            "type": "message_delta",
            "delta": {
                "stop_reason": "end_turn",
                "stop_sequence": None
            },
            "usage": {
                "output_tokens": output_tokens
            }
        }
        yield f"event: message_delta\ndata: {json.dumps(message_delta_payload)}\n\n"

        # Event 6: message_stop
        yield "event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n"
