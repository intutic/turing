"""
Structured output for Turing Engine.

GrammarConstraint is token-level constrained decoding with llguidance (MIT; `pip install turing-engine[structured]`):
a logits processor that masks every token the JSON schema (or Lark grammar) does not allow next, so the output is
valid by construction, and a final jsonschema check reports any output cut short by max_tokens.
StructuredOutputParser holds client-side helpers for extracting and repairing JSON in free text.
"""

import json
import re
from typing import Dict, Any, Optional, Tuple, Union, List

import torch

INSTALL_HINT = "structured output needs llguidance: pip install turing-engine[structured]"


class StructuredOutputError(ValueError):
    """A schema or grammar the server cannot enforce (the message says why); the API returns 400."""


_LL_TOKENIZERS: Dict[Tuple[int, int, Tuple[int, ...]], Any] = {}


def _ll_tokenizer(tokenizer, vocab_size: int, eos_ids):
    """llguidance's view of an HF tokenizer (building one takes about a second, so it is kept per tokenizer)."""
    try:
        import llguidance.hf
    except ImportError as e:
        raise StructuredOutputError(INSTALL_HINT) from e
    if tokenizer is None:
        raise StructuredOutputError("structured output needs the model's tokenizer; this model has none")
    eos = tuple(sorted(eos_ids)) or ((tokenizer.eos_token_id,) if tokenizer.eos_token_id is not None else ())
    key = (id(tokenizer), vocab_size, eos)
    if key not in _LL_TOKENIZERS:
        try:
            _LL_TOKENIZERS[key] = llguidance.hf.from_tokenizer(tokenizer, n_vocab=vocab_size, eos_token=list(eos))
        except ValueError as e:
            raise StructuredOutputError(f"structured output: {e}") from e
    return _LL_TOKENIZERS[key]


class GrammarConstraint:
    """
    Logits processor for the engine: before each sampled token, masks the logits to the tokens the grammar allows
    after the tokens generated so far (once the grammar is complete, only end-of-sequence tokens). Built from a
    JSON schema or a Lark grammar; an unsupported schema raises StructuredOutputError with llguidance's message.
    """

    def __init__(self, tokenizer, vocab_size: int, eos_ids, schema: Optional[Dict[str, Any]] = None,
                 lark: Optional[str] = None):
        from llguidance import LLMatcher
        ll = _ll_tokenizer(tokenizer, vocab_size, eos_ids)
        if lark is None:
            if not isinstance(schema, dict):
                raise StructuredOutputError("the JSON schema must be an object")
            try:  # compact JSON: free whitespace would let a model pad forever
                lark = LLMatcher.grammar_from_json_schema(schema, overrides={"whitespace_flexible": False})
            except ValueError as e:
                raise StructuredOutputError(f"invalid JSON schema: {e}") from e
        else:
            lark = LLMatcher.grammar_from_lark(lark)
        self.matcher = LLMatcher(ll, lark, log_level=0)
        if self.matcher.is_error():
            raise StructuredOutputError(f"unsupported schema: {self.matcher.get_error()}")
        self.vocab_size = vocab_size
        self._mask = torch.full(((vocab_size + 31) // 32,), -1, dtype=torch.int32)
        self._shifts = torch.arange(32, dtype=torch.int32)
        self._consumed = 0

    @property
    def error(self) -> Optional[str]:
        return self.matcher.get_error() if self.matcher.is_error() else None

    def done(self) -> bool:
        """The grammar is complete (or broken): only end-of-sequence can follow."""
        return self.matcher.is_stopped()

    def __call__(self, generated: List[int], logits: torch.Tensor) -> None:
        for t in generated[self._consumed:]:
            self.matcher.consume_token(t)
        self._consumed = len(generated)
        if self.matcher.is_error():  # never mask everything: the request ends normally and reports the error
            return
        self.matcher.unsafe_compute_mask_ptr(self._mask.data_ptr(), self._mask.numel() * 4)
        allowed = ((self._mask.unsqueeze(1) >> self._shifts) & 1).reshape(-1)[:logits.shape[-1]]
        logits[:allowed.numel()].masked_fill_(allowed.to(logits.device) == 0, float("-inf"))
        if logits.shape[-1] > allowed.numel():
            logits[allowed.numel():] = float("-inf")


def validate_json(text: str, schema: Optional[Dict[str, Any]]) -> Optional[str]:
    """None when text is JSON valid against schema, else what is wrong (jsonschema's message)."""
    try:
        obj = json.loads(text)
    except ValueError as e:
        return f"output is not valid JSON: {e}"
    if schema:
        try:
            import jsonschema
        except ImportError as e:
            raise StructuredOutputError(INSTALL_HINT) from e
        try:
            jsonschema.validate(obj, schema)
        except jsonschema.ValidationError as e:
            return f"output does not match the schema: {e.message}"
        except jsonschema.SchemaError as e:
            return f"invalid JSON schema: {e.message}"
    return None


class StructuredOutputParser:
    """
    Client-side helpers: JSON extraction from free text and truncation repair.
    """


    @staticmethod
    def extract_json(text: str) -> Tuple[bool, Optional[Any], str]:
        """
        Extracts JSON from raw text or markdown codeblocks.
        Returns (is_valid, parsed_obj, cleaned_text).
        """
        raw = text.strip()

        # Check for ```json ... ``` or ``` ... ``` code blocks
        json_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        candidate = json_block_match.group(1).strip() if json_block_match else raw

        # Direct JSON parse attempt
        try:
            parsed = json.loads(candidate)
            return True, parsed, candidate
        except Exception:
            pass

        # Try to find outermost { ... } or [ ... ]
        first_brace = candidate.find("{")
        first_bracket = candidate.find("[")
        
        start_idx = -1
        if first_brace != -1 and first_bracket != -1:
            start_idx = min(first_brace, first_bracket)
        elif first_brace != -1:
            start_idx = first_brace
        elif first_bracket != -1:
            start_idx = first_bracket

        if start_idx != -1:
            sub = candidate[start_idx:]
            try:
                parsed = json.loads(sub)
                return True, parsed, sub
            except Exception:
                # Try auto-repairing truncated JSON
                repaired = StructuredOutputParser.repair_truncated_json(sub)
                try:
                    parsed = json.loads(repaired)
                    return True, parsed, repaired
                except Exception:
                    pass

        return False, None, candidate

    @staticmethod
    def repair_truncated_json(text: str) -> str:
        """
        Repairs truncated JSON strings caused by max_tokens limits by closing open quotes,
        brackets, and braces.
        """
        s = text.strip()
        if not s:
            return "{}"

        try:
            import turing.turing_csrc as turing_csrc
            res = turing_csrc.scan_json_structure_fast(s)
            suffix = res.get("repair_suffix", "")
            if suffix:
                s_repaired = re.sub(r",\s*$", "", s)
                s_repaired = re.sub(r":\s*$", ': null', s_repaired)
                return s_repaired + suffix
        except Exception:
            pass

        # State tracking
        in_string = False
        escape = False
        stack: List[str] = []

        for char in s:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue

            if char in ("{", "["):
                stack.append(char)
            elif char == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif char == "]":
                if stack and stack[-1] == "[":
                    stack.pop()

        # If inside an open string, close it
        if in_string:
            s += '"'

        # Remove trailing dangling commas or colons before closing
        s = re.sub(r",\s*$", "", s)
        s = re.sub(r":\s*$", ': null', s)

        # Close all remaining unclosed containers in reverse order
        while stack:
            opener = stack.pop()
            if opener == "{":
                s += "}"
            elif opener == "[":
                s += "]"

        return s
