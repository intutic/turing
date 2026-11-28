"""
Reasoning (thinking) support for models that emit <think>...</think> (Qwen3, DeepSeek-R1 distills):
splitting the text stream into reasoning and content, and a thinking-token budget enforced while decoding.
"""

import functools
from typing import Optional, Tuple, List

import torch

OPEN, CLOSE = "<think>", "</think>"


class ReasoningBudgetManager:
    """Thinking-token budgets per OpenAI reasoning_effort (Anthropic passes budget_tokens directly)."""
    EFFORT_BUDGET_MAP = {
        "low": 1024,
        "medium": 4096,
        "high": 16384,
        "default": 2048,
    }

    EFFORT_TEMPERATURE_MAP = {
        "low": 0.6,
        "medium": 0.6,
        "high": 0.6,
        "default": 0.7,
    }

    @classmethod
    def get_max_tokens(cls, effort: Optional[str], user_max_tokens: Optional[int] = None) -> int:
        """
        Determines the effective maximum output token budget given an effort constraint.
        """
        if not effort or effort.lower() not in cls.EFFORT_BUDGET_MAP:
            return user_max_tokens or 2048

        effort_budget = cls.EFFORT_BUDGET_MAP[effort.lower()]
        if user_max_tokens is not None:
            return max(user_max_tokens, effort_budget)
        return effort_budget

    @classmethod
    def get_temperature(cls, effort: Optional[str], user_temp: Optional[float] = None) -> float:
        """
        Calibrates recommended sampling temperature for reasoning chains (e.g. DeepSeek-R1 standard 0.6).
        """
        if not effort or effort.lower() not in cls.EFFORT_TEMPERATURE_MAP:
            return user_temp if user_temp is not None else 0.7
        return cls.EFFORT_TEMPERATURE_MAP[effort.lower()]


def _partial(buf: str, tag: str) -> int:
    """Length of the longest suffix of buf that is a proper prefix of tag (a tag possibly split across chunks)."""
    for j in range(min(len(tag) - 1, len(buf)), 0, -1):
        if buf.endswith(tag[:j]):
            return j
    return 0


class ReasoningStreamFilter:
    """
    Splits streamed text into (reasoning, content) at <think> / </think>. A tag split across chunks is held back
    until it is complete; in_thinking=True starts inside a thinking block (templates that end the prompt with
    <think>, as DeepSeek-R1's does). Newlines right after </think> are dropped from the content.
    """

    def __init__(self, in_thinking: bool = False):
        self.in_thinking_block = in_thinking
        self.buffered_text = ""
        self.thinking_content = ""
        self.final_content = ""
        self._after_close = False

    def process_token(self, token_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Returns the (reasoning_chunk, content_chunk) this text completes; None for an empty chunk."""
        buf, r, c = self.buffered_text + token_text, "", ""
        while True:
            tag = CLOSE if self.in_thinking_block else OPEN
            i = buf.find(tag)
            cut = i if i >= 0 else len(buf) - _partial(buf, tag)
            if self.in_thinking_block:
                r += buf[:cut]
            else:
                c += self._content(buf[:cut])
            if i < 0:
                self.buffered_text = buf[cut:]
                break
            buf = buf[i + len(tag):]
            self.in_thinking_block = not self.in_thinking_block
            self._after_close = not self.in_thinking_block
        self.thinking_content += r
        self.final_content += c
        return r or None, c or None

    def flush(self) -> Tuple[Optional[str], Optional[str]]:
        """The held-back text at the end of the stream."""
        rest, self.buffered_text = self.buffered_text, ""
        if self.in_thinking_block:
            self.thinking_content += rest
            return rest or None, None
        rest = self._content(rest)
        self.final_content += rest
        return None, rest or None

    def _content(self, text: str) -> str:
        if self._after_close and text:
            text = text.lstrip("\n")
            self._after_close = not text
        return text


@functools.lru_cache(maxsize=8)
def _think_ids(tokenizer) -> Optional[Tuple[int, int]]:
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        return None
    if OPEN not in vocab or CLOSE not in vocab:
        return None
    return vocab[OPEN], vocab[CLOSE]


CHANNEL_THINKING, CHANNEL_TEXT = "<|content_thinking|>", "<|content_text|>"
CHANNEL_MARKS = ("<|message_model|>", "<|end_message|>", CHANNEL_THINKING, CHANNEL_TEXT)


@functools.lru_cache(maxsize=8)
def _channel_ids(tokenizer):
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        return None
    if CHANNEL_THINKING not in vocab or CHANNEL_TEXT not in vocab:
        return None
    return vocab[CHANNEL_THINKING], vocab[CHANNEL_TEXT], frozenset(vocab[t] for t in CHANNEL_MARKS if t in vocab)


def channel_token_ids(tokenizer) -> Optional[Tuple[int, int, frozenset]]:
    """(thinking id, text id, every marker id) for models that mark reasoning with special tokens instead of <think>
    text (Inkling 2), else None. Read once per tokenizer, as think_token_ids."""
    return _channel_ids(tokenizer) if tokenizer is not None else None


class ChannelSplit:
    """Reasoning marked with special tokens (Inkling 2): the model writes <|message_model|><|content_thinking|> reasoning
    <|end_message|> <|message_model|><|content_text|> answer <|end_message|>. push(token) is the part a generated token
    belongs to ("reasoning" | "content"), or None for a marker, which is not text. Decoding skips special tokens, so the
    text filter (ReasoningStreamFilter) would never see these markers."""

    def __init__(self, ids: Tuple[int, int, frozenset]):
        self.thinking, self.text, self.marks = ids
        self.part = "content"

    def push(self, token: int) -> Optional[str]:
        if token == self.thinking:
            self.part = "reasoning"
        elif token == self.text:
            self.part = "content"
        return None if token in self.marks else self.part


def think_token_ids(tokenizer) -> Optional[Tuple[List[int], List[int]]]:
    """(<think> ids, </think> ids) when the tokenizer has both tags as tokens (a reasoning model), else None. The
    vocabulary is read once per tokenizer: get_vocab() builds a new dict of every token (0.1 s for Qwen2.5's
    151,936), and a chat request asks twice."""
    ids = _think_ids(tokenizer) if tokenizer is not None else None
    return ([ids[0]], [ids[1]]) if ids else None


class ThinkingBudget:
    """
    Logits processor that ends a thinking block after `budget` thinking tokens by forcing the </think> token
    sequence (in order when it is several tokens). Also tells where the content starts in the generated ids:
    content_start() is None while thinking, else the index after </think> (0 when the model never thinks).
    """

    def __init__(self, open_ids: List[int], close_ids: List[int], budget: Optional[int], in_thinking: bool = False):
        self.open_ids, self.close_ids, self.budget = list(open_ids), list(close_ids), budget
        self.start: Optional[int] = 0 if in_thinking else None
        self.end: Optional[int] = None
        self._seen = 0
        self._forced = 0
        self._decided = in_thinking  # whether the answer opened with <think> is known

    def _scan(self, gen: List[int]) -> None:
        if not self._decided and len(gen) >= len(self.open_ids):  # a thinking block opens the answer or never
            self._decided = True
            if gen[:len(self.open_ids)] == self.open_ids:
                self.start = len(self.open_ids)
        if self.start is not None and self.end is None:
            k = len(self.close_ids)
            for n in range(max(self._seen + 1, self.start + k), len(gen) + 1):
                if gen[n - k:n] == self.close_ids:
                    self.end = n
                    break
        self._seen = len(gen)

    def content_start(self, gen: List[int]) -> Optional[int]:
        self._scan(gen)
        if self.start is None:
            return 0
        return self.end

    def __call__(self, gen: List[int], logits: torch.Tensor) -> None:
        self._scan(gen)
        if self.budget is None or self.start is None or self.end is not None:
            return
        if len(gen) - self.start < self.budget:
            return
        tok = self.close_ids[min(self._forced, len(self.close_ids) - 1)]
        self._forced += 1
        logits.fill_(float("-inf"))
        logits[tok] = 0.0
