"""
Reasoning Effort & Thinking Token Budgeting Engine for Turing Engine.
Handles OpenAI reasoning_effort ('low', 'medium', 'high'), Anthropic thinking budgets,
and streaming token extraction for <think>...</think> sequences.
"""

from typing import Optional, Tuple, Dict, Any, List
import re

class ReasoningBudgetManager:
    """
    Manages max generation budgets and sampling dynamics based on reasoning effort constraints.
    """
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


class ReasoningStreamFilter:
    """
    Processes token streams and extracts <think>...</think> reasoning chains for SSE streaming.
    Supports both OpenAI delta.reasoning_content and Anthropic thinking blocks.
    """
    def __init__(self):
        self.in_thinking_block = False
        self.buffered_text = ""
        self.thinking_content = ""
        self.final_content = ""

    def process_token(self, token_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Processes a chunk of newly generated text.
        Returns a tuple of (reasoning_chunk, content_chunk).
        """
        self.buffered_text += token_text

        # Check entry into <think>
        if "<think>" in self.buffered_text and not self.in_thinking_block:
            self.in_thinking_block = True
            before_think, self.buffered_text = self.buffered_text.split("<think>", 1)
            return None, before_think if before_think else None

        # Check exit from </think>
        if "</think>" in self.buffered_text and self.in_thinking_block:
            self.in_thinking_block = False
            think_part, content_part = self.buffered_text.split("</think>", 1)
            self.thinking_content += think_part
            self.final_content += content_part
            self.buffered_text = ""
            return think_part, (content_part if content_part else None)

        if self.in_thinking_block:
            chunk = self.buffered_text
            self.thinking_content += chunk
            self.buffered_text = ""
            return chunk, None
        else:
            chunk = self.buffered_text
            self.final_content += chunk
            self.buffered_text = ""
            return None, chunk
