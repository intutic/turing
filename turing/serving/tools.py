"""
Tool (function) calling: the OpenAI / Anthropic tool lists, the call formats chat models emit, and the grammar that
forces a call when tool_choice requires one.

Formats parsed from model output:
  Hermes / Qwen   <tool_call>{"name": ..., "arguments": {...}}</tool_call>   (one tag per call)
  Mistral         [TOOL_CALLS][{"name": ..., "arguments": {...}}, ...]
  Llama 3         {"name": ..., "parameters": {...}}                          (the whole answer is the call)
"""

import json
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple

from pydantic import BaseModel, Field

HERMES_OPEN, HERMES_CLOSE, MISTRAL_MARK = "<tool_call>", "</tool_call>", "[TOOL_CALLS]"


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")
    type: str = "function"
    function: FunctionCall


def tool_format(chat_template: Optional[str]) -> str:
    """The call format a chat template teaches its model: 'hermes', 'mistral' or 'llama' (bare JSON)."""
    t = chat_template or ""
    if HERMES_OPEN in t:
        return "hermes"
    if MISTRAL_MARK in t:
        return "mistral"
    if "parameters" in t and "tool" in t:
        return "llama"
    return "hermes"  # what the plain-prompt tool instruction asks for


def _calls_from(objs: List[Any]) -> List[Dict[str, Any]]:
    calls = []
    for data in objs:
        if not isinstance(data, dict) or not isinstance(data.get("name"), str):
            continue
        args = data.get("arguments", data.get("parameters", {}))
        args_str = args if isinstance(args, str) else json.dumps(args)
        calls.append(ToolCall(function=FunctionCall(name=data["name"], arguments=args_str)).model_dump())
    return calls


def _json_at(text: str, i: int) -> Tuple[Any, int]:
    """The JSON value starting at text[i] (after whitespace) and the index after it."""
    while i < len(text) and text[i].isspace():
        i += 1
    return json.JSONDecoder().raw_decode(text, i)


class ToolCallingHandler:
    """
    Plain-prompt tool instructions (models without a chat template) and tool-call extraction from output.
    """

    TOOL_CALL_REGEX = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*(?:</tool_call>|$)")

    @classmethod
    def inject_tools_instruction(cls, prompt: str, tools: List[Dict[str, Any]]) -> str:
        """
        Tool descriptions and the call syntax, for models without a chat template (templates render tools themselves).
        """
        tool_descriptions = []
        for t in tools:
            # Handle OpenAI tool format: {"type": "function", "function": {...}}
            func = t.get("function", t) if isinstance(t, dict) else t
            tool_descriptions.append({
                "name": func.get("name", "unknown"),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", func.get("input_schema", {})),
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
        Extracts tool calls from generated text in any of the formats above.
        Returns (text outside the calls, list of OpenAI tool_calls with JSON-string arguments).
        """
        # 1. Hermes / Qwen tags
        if HERMES_OPEN in text:
            objs, clean = [], text
            for match in cls.TOOL_CALL_REGEX.finditer(text):
                try:
                    objs.append(json.loads(match.group(1)))
                except ValueError:
                    continue
                clean = clean.replace(match.group(0), "")
            calls = _calls_from(objs)
            if calls:
                return clean.strip(), calls

        # 2. Mistral: [TOOL_CALLS] followed by a JSON list of calls
        i = text.find(MISTRAL_MARK)
        if i >= 0:
            try:
                value, end = _json_at(text, i + len(MISTRAL_MARK))
                calls = _calls_from(value if isinstance(value, list) else [value])
                if calls:
                    return (text[:i] + text[end:]).strip(), calls
            except ValueError:
                pass

        # 3. Llama 3 / bare JSON: the whole answer is one call (or a list of them)
        trimmed = text.strip()
        if trimmed[:1] in ("{", "["):
            try:
                value, end = _json_at(trimmed, 0)
                objs = value if isinstance(value, list) else [value]
                if all(isinstance(o, dict) and ("arguments" in o or "parameters" in o) for o in objs):
                    calls = _calls_from(objs)
                    if calls:
                        return trimmed[end:].strip(), calls
            except ValueError:
                pass

        return text.strip(), []


def normalize_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """OpenAI ({type: function, function: {...}}) or Anthropic ({name, description, input_schema}) tools as the
    OpenAI list chat templates take."""
    if not tools:
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict):
            raise ValueError("each tool must be an object")
        if "function" in t:
            f = t["function"]
        elif "name" in t:
            f = {"name": t["name"], "description": t.get("description", ""),
                 "parameters": t.get("input_schema", t.get("parameters", {"type": "object"}))}
        else:
            raise ValueError("a tool needs a 'function' (OpenAI) or a 'name' and 'input_schema' (Anthropic)")
        if not isinstance(f.get("name"), str) or not f["name"]:
            raise ValueError("every tool needs a name")
        out.append({"type": "function", "function": {"name": f["name"], "description": f.get("description", ""),
                                                      "parameters": f.get("parameters") or {"type": "object"}}})
    return out


def call_schema(tools: List[Dict[str, Any]], fmt: str, name: Optional[str] = None) -> Dict[str, Any]:
    """JSON schema of one call to any of tools (or the named one), in the format's key for the arguments."""
    key = "parameters" if fmt == "llama" else "arguments"
    options = [{"type": "object", "properties": {"name": {"const": t["function"]["name"]},
                                                 key: t["function"]["parameters"]},
                "required": ["name", key], "additionalProperties": False}
               for t in tools if name is None or t["function"]["name"] == name]
    if not options:
        raise ValueError(f"tool_choice names '{name}', which is not in tools")
    return options[0] if len(options) == 1 else {"anyOf": options}


def call_grammar(tools: List[Dict[str, Any]], fmt: str, tokenizer, name: Optional[str] = None) -> str:
    """llguidance Lark grammar for exactly one call in the model's format (tool_choice required or named)."""
    schema = json.dumps({**call_schema(tools, fmt, name), "x-guidance": {"whitespace_flexible": False}})

    def lit(s: str) -> str:
        """A tag the tokenizer holds as one token is matched as that token (llguidance treats added tokens as
        special), else as text."""
        ids = tokenizer.encode(s, add_special_tokens=False) if tokenizer is not None else []
        return f"<[{ids[0]}]>" if len(ids) == 1 else json.dumps(s)

    if fmt == "hermes":
        return f'start: {lit(HERMES_OPEN)} "\\n" call "\\n" {lit(HERMES_CLOSE)}\ncall: %json {schema}'
    if fmt == "mistral":
        return f'start: {lit(MISTRAL_MARK)} "[" call "]"\ncall: %json {schema}'
    return f"start: call\ncall: %json {schema}"
