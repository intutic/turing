"""
Shared generation core for every HTTP endpoint (OpenAI, Anthropic, Ollama) and the DSL.

render_chat() turns chat messages into the exact token ids the model sees (the tokenizer's chat template, tools
included); start() builds the logits processors a request needs (JSON-schema / tool-call grammar, thinking budget),
admits it to the engine (so a rejection happens before any streamed 200), and returns a Generation whose events()
stream text with incremental detokenization, stop strings, reasoning split from content and tool calls held back for
parsing. After the stream: finish_reason, tool_calls, usage and the schema check.
"""

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

import torch

from ..core.kv_cache import PAGE, KVCache
from .engine import ContinuousBatchEngine, RequestRejected
from .reasoning import ReasoningBudgetManager, ReasoningStreamFilter, ThinkingBudget, think_token_ids
from .structured import GrammarConstraint, StructuredOutputError, validate_json
from .tools import (HERMES_OPEN, MISTRAL_MARK, ToolCallingHandler, call_grammar, normalize_tools, tool_format)
from .traffic import Lane

# Request fields and headers that once claimed per-request effects the engine does not have
REMOVED_FIELDS = {
    "use_svd_kv": "KV cache storage is chosen at load time with --kv-cache-dtype (native, int8, int4)",
    "sparsity_ratio": "sparsity is chosen at load time with --sparsity",
}
REMOVED_HEADERS = {
    "x-turing-svd-kv": REMOVED_FIELDS["use_svd_kv"],
    "x-turing-svd-rank": REMOVED_FIELDS["use_svd_kv"],
    "x-turing-sparsity": REMOVED_FIELDS["sparsity_ratio"],
}


def bad_request(message: str) -> RequestRejected:
    return RequestRejected(message, status_code=400)


# --------------------------------------------------------------------------------------------- prompts

def _text(content: Any) -> str:
    """Text of a message's content: a string, None, or a list of content parts (text parts are joined)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise bad_request("message content must be a string or a list of content parts")
    parts = []
    for p in content:
        kind = p.get("type") if isinstance(p, dict) else None
        if kind in ("text", "input_text"):
            parts.append(p.get("text") or "")
        else:
            raise bad_request(f"content part type '{kind}' is not supported by this server (text only)")
    return "\n".join(parts)


def _arguments(args: Any) -> Any:
    """Tool-call arguments as the dict chat templates render (OpenAI sends a JSON string)."""
    if isinstance(args, str):
        try:
            return json.loads(args) if args.strip() else {}
        except ValueError:
            return args
    return args if args is not None else {}


def openai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """OpenAI chat messages as template messages: content parts joined, developer as system, tool calls with dict
    arguments, tool results with their tool_call_id."""
    out = []
    for m in messages:
        if not isinstance(m, dict):
            raise bad_request("each message must be an object")
        role = {"developer": "system", "function": "tool"}.get(m.get("role"), m.get("role"))
        if role not in ("system", "user", "assistant", "tool"):
            raise bad_request(f"unknown message role '{m.get('role')}'")
        msg: Dict[str, Any] = {"role": role, "content": _text(m.get("content"))}
        if role == "assistant" and m.get("tool_calls"):
            msg["tool_calls"] = [{"id": tc.get("id"), "type": "function",
                                  "function": {"name": tc.get("function", {}).get("name"),
                                               "arguments": _arguments(tc.get("function", {}).get("arguments"))}}
                                 for tc in m["tool_calls"]]
        if role == "tool":
            msg["tool_call_id"] = m.get("tool_call_id")
            if m.get("name"):
                msg["name"] = m["name"]
        out.append(msg)
    return out


def plain_prompt(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
    """The prompt for models without a chat template: 'Role: text' turns, then 'Assistant:'."""
    parts = []
    for m in messages:
        role = m["role"]
        text = m.get("content") or ""
        for tc in m.get("tool_calls") or []:
            f = tc["function"]
            text += f'\n<tool_call>{json.dumps({"name": f["name"], "arguments": f["arguments"]})}</tool_call>'
        label = {"system": "System", "user": "User", "assistant": "Assistant"}.get(role, "Tool")
        parts.append(f"{label}: {text.strip()}")
    if tools:
        parts.insert(0, ToolCallingHandler.inject_tools_instruction("System:", tools).strip())
    parts.append("Assistant:")
    return "\n\n".join(parts)


def render_chat(engine: ContinuousBatchEngine, messages: List[Dict[str, Any]],
                tools: Optional[List[Dict[str, Any]]] = None, **template_kwargs) -> Tuple[List[int], str]:
    """Token ids (and text) the model is fed for messages: the tokenizer's chat template with tools and
    add_generation_prompt, or the plain prompt for models without one."""
    tok = engine.tokenizer
    if tok is None or not getattr(tok, "chat_template", None):
        text = plain_prompt(messages, tools)
        return engine.encode_prompt(text), text
    try:
        text = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False,
                                       **template_kwargs)
    except Exception as e:  # the template rejects the conversation (role order, missing fields)
        raise bad_request(f"chat template error: {e}") from e
    return tok.encode(text, add_special_tokens=False), text  # the template already holds BOS


def render_text(engine: ContinuousBatchEngine, prompt: Any) -> List[int]:
    """Token ids of a completion prompt: a string (tokenized as the model's tokenizer does) or token ids."""
    if isinstance(prompt, list):
        if not all(isinstance(t, int) and 0 <= t < engine.model_config.vocab_size for t in prompt):
            raise bad_request(f"prompt token ids must be integers in [0, {engine.model_config.vocab_size})")
        return list(prompt)
    if not isinstance(prompt, str):
        raise bad_request("prompt must be a string or a list of token ids")
    return engine.encode_prompt(prompt)


def thinking_kwargs(engine: ContinuousBatchEngine, enabled: Optional[bool]) -> Dict[str, Any]:
    """Chat-template kwargs that switch thinking off (Qwen3's enable_thinking) when a request disables it."""
    if enabled is False and think_token_ids(engine.tokenizer) is not None:
        return {"enable_thinking": False}
    return {}


def effort_budget(effort: Optional[str]) -> Tuple[Optional[bool], Optional[int]]:
    """OpenAI reasoning_effort -> (thinking enabled, thinking-token budget)."""
    if effort is None:
        return None, None
    e = effort.lower()
    if e in ("none", "minimal"):
        return False, 0
    if e not in ("low", "medium", "high"):
        raise bad_request(f"reasoning_effort must be none, minimal, low, medium or high, not '{effort}'")
    return True, ReasoningBudgetManager.EFFORT_BUDGET_MAP[e]


# --------------------------------------------------------------------------------------------- streaming pieces

class Detokenizer:
    """Incremental detokenization: decode a window of the growing token list and emit what it adds, holding back
    text that ends in an incomplete multi-byte character (U+FFFD) until the next token completes it."""

    def __init__(self, engine: ContinuousBatchEngine):
        self.decode = engine.decode_tokens
        self.ids: List[int] = []
        self.prefix = self.read = 0

    def push(self, token: int) -> str:
        self.ids.append(token)
        before = self.decode(self.ids[self.prefix:self.read])
        now = self.decode(self.ids[self.prefix:])
        if len(now) > len(before) and not now.endswith("�"):
            self.prefix, self.read = self.read, len(self.ids)
            return now[len(before):]
        return ""

    def flush(self) -> str:
        before = self.decode(self.ids[self.prefix:self.read])
        now = self.decode(self.ids[self.prefix:])
        self.prefix = self.read = len(self.ids)
        return now[len(before):]


def _held(buf: str, words: List[str]) -> int:
    """Length of the longest suffix of buf that could still grow into one of words."""
    best = 0
    for w in words:
        for j in range(min(len(w) - 1, len(buf)), best, -1):
            if buf.endswith(w[:j]):
                best = j
                break
    return best


class StopStrings:
    """Finds stop strings in streamed text. push() returns (text safe to emit, the stop string hit or None)."""

    def __init__(self, stops: List[str]):
        self.stops = [s for s in stops if s]
        self.buf = ""

    def push(self, text: str) -> Tuple[str, Optional[str]]:
        self.buf += text
        hits = [(self.buf.find(s), s) for s in self.stops if s in self.buf]
        if hits:
            i, s = min(hits)
            out, self.buf = self.buf[:i], ""
            return out, s
        k = _held(self.buf, self.stops)
        out, self.buf = self.buf[:len(self.buf) - k], self.buf[len(self.buf) - k:]
        return out, None

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return out


class ToolGate:
    """Holds content back from the stream once it may be a tool call (a call tag, or a JSON object opening the
    answer for the bare-JSON format), so calls are emitted parsed rather than as text."""

    def __init__(self, fmt: str):
        self.marks = [HERMES_OPEN, MISTRAL_MARK]
        self.bare_json = fmt == "llama"
        self.released = ""
        self.buf = ""
        self.holding = False

    def push(self, text: str) -> str:
        if self.holding:
            self.buf += text
            return ""
        self.buf += text
        if self.bare_json and not self.released.strip() and self.buf.lstrip()[:1] in ("{", "["):
            self.holding = True
            return ""
        idx = [self.buf.find(m) for m in self.marks if m in self.buf]
        if idx:
            out, self.buf = self.buf[:min(idx)], self.buf[min(idx):]
            self.holding = True
        else:
            k = _held(self.buf, self.marks)
            if self.bare_json and not (self.released + self.buf[:len(self.buf) - k]).strip():
                k = len(self.buf)  # only whitespace so far: the answer may still open with a JSON call
            out, self.buf = self.buf[:len(self.buf) - k], self.buf[len(self.buf) - k:]
        self.released += out
        return out


# --------------------------------------------------------------------------------------------- the request

@dataclass
class GenerationParams:
    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 50
    stop: List[str] = field(default_factory=list)
    draft_tokens: Optional[int] = None
    lane: Optional[Lane] = None
    adapter: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None  # normalized OpenAI tools: parse calls from the output
    tool_choice: Any = "auto"  # "auto" | "none" | "required" | {"name": ...}
    schema: Optional[Dict[str, Any]] = None  # constrain the output to this JSON schema, then validate it
    thinking_budget: Optional[int] = None  # thinking tokens before </think> is forced (reasoning models)
    reasoning: bool = True  # split <think> blocks into reasoning (chat endpoints); off for raw completions
    logits_processor: Optional[Callable] = None  # applied first (the DSL records log-probabilities with it)


def normalize_tool_choice(choice: Any) -> Any:
    """OpenAI ("auto" | "none" | "required" | {type: function, function: {name}}) or Anthropic ({type: auto | any |
    none | tool, name}) tool_choice as "auto" | "none" | "required" | {"name": ...}."""
    if choice is None:
        return "auto"
    if isinstance(choice, str):
        if choice not in ("auto", "none", "required"):
            raise bad_request(f"tool_choice must be auto, none, required or a named tool, not '{choice}'")
        return choice
    if isinstance(choice, dict):
        kind = choice.get("type")
        if kind is None and isinstance(choice.get("name"), str):  # already normalized
            return choice
        if kind == "function" and isinstance(choice.get("function"), dict):
            return {"name": choice["function"].get("name")}
        if kind == "tool":
            return {"name": choice.get("name")}
        if kind in ("auto", "none"):
            return kind
        if kind == "any":
            return "required"
    raise bad_request(f"unsupported tool_choice {choice!r}")


class Generation:
    """One admitted request. Iterate events() for ("reasoning" | "content", text) pieces; afterwards finish_reason
    ("stop" | "length" | "tool_calls"), stop_sequence, tool_calls, completion_tokens and schema_error are set."""

    def __init__(self, engine, req, params: GenerationParams, prompt_ids: List[int], in_thinking: bool,
                 constraint: Optional[GrammarConstraint], tool_fmt: Optional[str]):
        self.engine, self.req, self.params = engine, req, params
        self.prompt_ids = prompt_ids
        self.constraint = constraint
        self.reasoning = ReasoningStreamFilter(in_thinking) if params.reasoning and \
            think_token_ids(engine.tokenizer) is not None else None
        self.tool_fmt = tool_fmt
        self.stats: Optional[dict] = None  # the server's per-lane counters ("finished" counts this request's end)
        self.finish_reason: Optional[str] = None
        self.stop_sequence: Optional[str] = None
        self.tool_calls: List[Dict[str, Any]] = []
        self.output_ids: List[int] = []
        self.content = ""
        self.reasoning_text = ""
        self.schema_error: Optional[str] = None

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_ids)

    @property
    def completion_tokens(self) -> int:
        return len(self.output_ids)

    @property
    def cached_tokens(self) -> int:
        return self.req.cached_prompt_tokens

    @property
    def draft_tokens(self) -> int:
        """Draft tokens per step this request may use (requests with a logits processor never speculate)."""
        if self.req.logits_processor is not None:
            return 0
        d = self.req.draft_tokens
        return self.engine.default_draft_tokens if d is None else d

    def _route(self, text: str, gate: Optional[ToolGate]):
        pieces = [(None, text)] if self.reasoning is None else [self.reasoning.process_token(text)]
        return self._emit(pieces, gate)

    def _emit(self, pieces, gate):
        for r, c in pieces:
            if r:
                self.reasoning_text += r
                yield "reasoning", r
            if c:
                self.content += c
                c = gate.push(c) if gate is not None else c
                if c:
                    yield "content", c

    async def events(self) -> AsyncIterator[Tuple[str, str]]:
        detok = Detokenizer(self.engine)
        stops = StopStrings(self.params.stop)
        gate = ToolGate(self.tool_fmt) if self.tool_fmt else None
        stream = self.engine.stream_request(self.req)
        try:
            async for token in stream:
                self.output_ids.append(token)
                out, hit = stops.push(detok.push(token))
                for ev in self._route(out, gate):
                    yield ev
                if hit is not None:
                    self.stop_sequence = hit
                    break
            if self.stop_sequence is None:
                out, _ = stops.push(detok.flush())
                for ev in self._route(out + stops.flush(), gate):
                    yield ev
            if self.reasoning is not None:
                for ev in self._emit([self.reasoning.flush()], gate):
                    yield ev
        finally:
            await stream.aclose()  # cancels the engine request when the client went away or a stop string hit
            if self.stats is not None:
                self.stats["finished"] += 1
        self._finish(gate)
        if gate is not None and not self.tool_calls and gate.buf:
            yield "content", gate.buf  # held back but no call parsed: it was text after all

    def _finish(self, gate: Optional[ToolGate]) -> None:
        if gate is not None:
            clean, calls = ToolCallingHandler.extract_tool_calls(self.content)
            if calls:
                self.tool_calls, self.content = calls, clean
        if self.tool_calls:
            self.finish_reason = "tool_calls"
        elif self.stop_sequence is None and self.req.finish_reason == "length":
            self.finish_reason = "length"
        else:
            self.finish_reason = "stop"
        if self.params.schema is not None:
            self.schema_error = (self.constraint.error if self.constraint is not None else None) or \
                validate_json(self.content, self.params.schema)

    async def collect(self) -> "Generation":
        async for _ in self.events():
            pass
        return self


async def start(engine: ContinuousBatchEngine, prompt_ids: List[int], params: GenerationParams,
                prompt_text: str = "", prefix_kv: Optional[KVCache] = None) -> Generation:
    """Build the request's logits processors and admit it to the engine. Raises RequestRejected (400 for a request
    the server cannot serve as asked, 429/503 for load) before anything is streamed."""
    if params.max_tokens is None or params.max_tokens < 1:
        raise bad_request("max_tokens must be at least 1")
    if not prompt_ids:
        raise bad_request("empty prompt")
    tok = engine.tokenizer
    eos = engine.eos_token_ids
    vocab = engine.model_config.vocab_size
    stop = params.stop
    if isinstance(stop, str):
        params.stop = [stop]
    elif stop is None:
        params.stop = []

    tags = think_token_ids(tok) if params.reasoning else None
    in_thinking = tags is not None and prompt_text.rstrip().endswith("<think>")
    think = ThinkingBudget(*tags, budget=params.thinking_budget, in_thinking=in_thinking) if tags else None

    choice = normalize_tool_choice(params.tool_choice)
    tools = params.tools if choice != "none" else None
    fmt = tool_format(getattr(tok, "chat_template", None)) if tools else None
    try:
        constraint = None
        if tools and (choice == "required" or isinstance(choice, dict)):
            name = choice["name"] if isinstance(choice, dict) else None
            constraint = GrammarConstraint(tok, vocab, eos, lark=call_grammar(tools, fmt, tok, name))
        elif params.schema is not None:
            constraint = GrammarConstraint(tok, vocab, eos, schema=params.schema)
    except StructuredOutputError as e:
        raise bad_request(str(e)) from e
    except ValueError as e:  # a named tool that is not in tools
        raise bad_request(str(e)) from e

    extra = params.logits_processor
    processor = None
    if extra is not None or constraint is not None or (think is not None and think.budget is not None):
        def processor(gen: List[int], logits: torch.Tensor):
            if extra is not None:
                out = extra(gen, logits)
                logits = logits if out is None else out
            if think is not None:
                think(gen, logits)
            if constraint is not None:
                s = think.content_start(gen) if think is not None else 0
                if s is not None:  # the grammar governs the answer, not the thinking before it
                    constraint(gen[s:], logits)
            return logits

    # a request larger than the KV pool is the client's error even when admission control would shed it first
    pool = engine.kv_pool
    need = (prefix_kv.length if prefix_kv is not None else 0) + len(prompt_ids) + params.max_tokens
    cap = min(pool.table_width, pool.page_limit) * PAGE
    if need > cap:
        raise bad_request(f"prompt plus max_tokens is {need} tokens, but this server holds at most {cap} per request")

    req = await engine.add_request(
        prompt_tokens=prompt_ids, max_new_tokens=params.max_tokens,
        temperature=0.7 if params.temperature is None else params.temperature,
        top_k=params.top_k if params.top_k is not None and params.top_k > 0 else 0,
        top_p=params.top_p if params.top_p is not None else 1.0, draft_tokens=params.draft_tokens,
        lane=params.lane, prefix_kv=prefix_kv, logits_processor=processor, adapter=params.adapter)
    return Generation(engine, req, params, prompt_ids, in_thinking, constraint, fmt)


def prefill(engine: ContinuousBatchEngine, tokens: List[int]) -> KVCache:
    """Prefill tokens into a new KV cache of the engine's pool (a shared prefix that requests continue with
    prefix_kv=). Call it on the engine's event loop, between scheduler steps."""
    cache = KVCache(engine.kv_pool)
    try:
        engine._reserve(cache, len(tokens))
        ids = torch.tensor([tokens], dtype=torch.long, device=engine.device)
        with torch.inference_mode():
            engine.model(ids, past_key_values=cache, logits_to_keep=1)
    except Exception:
        cache.free()
        raise
    return cache
