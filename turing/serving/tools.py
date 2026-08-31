"""
Native Tool & Function Calling Parser for Turing Engine.
Standardizes tool schemas across OpenAI tools/tool_calls and Anthropic tool_use interfaces.
"""

import json
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:20]}")
    type: str = "function"
    function: FunctionCall


class ToolCallingHandler:
    """
    Handles tool schema injection into system prompts and extracts tool_calls from output.
    """

    TOOL_CALL_REGEX = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>")
    JSON_TOOL_REGEX = re.compile(r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[\s\S]*?\})\s*\}')

    @classmethod
    def inject_tools_instruction(cls, prompt: str, tools: List[Dict[str, Any]]) -> str:
        """
        Injects tool descriptions and invocation syntax into the prompt.
        """
        tool_descriptions = []
        for t in tools:
            # Handle OpenAI tool format: {"type": "function", "function": {...}}
            func = t.get("function", t) if isinstance(t, dict) else t
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", func.get("input_schema", {}))
            
            tool_descriptions.append({
                "name": name,
                "description": desc,
                "parameters": params
            })

        tools_json = json.dumps(tool_descriptions, indent=2)
        instruction = (
            f"\n\n# TOOLS & FUNCTIONS\nYou have access to the following tools:\n```json\n{tools_json}\n```\n"
            f"If you need to call a tool, format your tool call using the following tag:\n"
            f"<tool_call>\n{{\"name\": \"<function_name>\", \"arguments\": {{...}}}}\n</tool_call>\n"
            f"If no tool is needed, respond directly with text."
        )
        return prompt + instruction

    @classmethod
    def extract_tool_calls(cls, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extracts tool calls from generated text.
        Returns (clean_text_without_tool_tags, list_of_openai_tool_calls).
        """
        tool_calls: List[Dict[str, Any]] = []
        clean_text = text

        # 1. Search for <tool_call> tags
        for match in cls.TOOL_CALL_REGEX.finditer(text):
            raw_json = match.group(1)
            try:
                data = json.loads(raw_json)
                func_name = data.get("name")
                args = data.get("arguments", {})
                args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                
                if func_name:
                    tc = ToolCall(function=FunctionCall(name=func_name, arguments=args_str))
                    tool_calls.append(tc.model_dump())
            except Exception:
                pass
            
            clean_text = clean_text.replace(match.group(0), "")

        # 2. Fallback: Check if the entire response is a JSON tool call
        if not tool_calls:
            trimmed = text.strip()
            if trimmed.startswith("{") and '"name"' in trimmed and '"arguments"' in trimmed:
                try:
                    data = json.loads(trimmed)
                    func_name = data.get("name")
                    args = data.get("arguments", {})
                    args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                    if func_name and isinstance(args, (dict, list)):
                        tc = ToolCall(function=FunctionCall(name=func_name, arguments=args_str))
                        tool_calls.append(tc.model_dump())
                        clean_text = ""
                except Exception:
                    pass

        return clean_text.strip(), tool_calls
