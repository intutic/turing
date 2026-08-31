"""
Chain & Branch Execution Context for Turing Programmatic DSL.
Manages prompt state, active token histories, and zero-copy prefix branches.
"""

import contextvars
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
import torch

__all__ = ["ChainContext", "BranchContext", "get_active_context", "set_active_context"]

_ACTIVE_CONTEXT: contextvars.ContextVar[Optional["ChainContext"]] = contextvars.ContextVar(
    "_ACTIVE_CONTEXT", default=None
)


def get_active_context() -> "ChainContext":
    ctx = _ACTIVE_CONTEXT.get()
    if ctx is None:
        raise RuntimeError(
            "No active Turing DSL context found. Ensure your code is wrapped inside a '@turing.chain' decorator or 'with ChainContext(...):' block."
        )
    return ctx


def set_active_context(ctx: Optional["ChainContext"]) -> contextvars.Token:
    return _ACTIVE_CONTEXT.set(ctx)


@dataclass
class BranchContext:
    """
    Child execution branch spawned by fork().
    Shares parent's prefix tokens and prefix KV cache, maintaining its own generation suffix.
    """
    branch_id: int
    parent: "ChainContext"
    branch_text: str = ""
    branch_tokens: List[int] = field(default_factory=list)
    token_logprobs: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return self.parent.history_text + self.branch_text

    @property
    def full_tokens(self) -> List[int]:
        return self.parent.history_tokens + self.branch_tokens

    @property
    def mean_logprob(self) -> float:
        if not self.token_logprobs:
            return 0.0
        return float(sum(self.token_logprobs) / len(self.token_logprobs))

    def gen(
        self,
        prompt: str = "",
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generates text from this branch context."""
        if prompt:
            self.branch_text += prompt
            if self.parent.tokenizer is not None:
                p_tokens = self.parent.tokenizer.encode(prompt)
                self.branch_tokens.extend(p_tokens)
            else:
                self.branch_tokens.extend([ord(c) % self.parent.vocab_size for c in prompt])

        # Execute generation via parent's executor
        new_text, new_tokens, logprobs = self.parent.executor.generate_branch(
            prefix_tokens=self.parent.history_tokens,
            branch_tokens=self.branch_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            stop=stop,
            schema=schema or self.parent.current_schema
        )

        self.branch_text += new_text
        self.branch_tokens.extend(new_tokens)
        self.token_logprobs.extend(logprobs)
        return new_text

    def select(self, options: List[str], prompt: str = "") -> str:
        """Constrains generation to select one option from the provided list."""
        if prompt:
            self.branch_text += prompt
            if self.parent.tokenizer is not None:
                p_tokens = self.parent.tokenizer.encode(prompt)
                self.branch_tokens.extend(p_tokens)
            else:
                self.branch_tokens.extend([ord(c) % self.parent.vocab_size for c in prompt])

        selected = self.parent.executor.select_option(
            tokens=self.full_tokens,
            options=options
        )
        self.branch_text += selected
        if self.parent.tokenizer is not None:
            self.branch_tokens.extend(self.parent.tokenizer.encode(selected))
        else:
            self.branch_tokens.extend([ord(c) % self.parent.vocab_size for c in selected])
        return selected


class ChainContext:
    """
    Main stateful context for a @turing.chain workflow.
    """

    def __init__(
        self,
        model: Optional[str] = "test-tiny",
        device: str = "auto",
        sparsity: float = 0.5,
        executor: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        self.model_name = model
        self.device = device
        self.sparsity = sparsity
        self.tokenizer = tokenizer
        self.executor = executor
        
        self.history_text: str = ""
        self.history_tokens: List[int] = []
        self.token_logprobs: List[float] = []
        self.current_schema: Optional[Dict[str, Any]] = None
        self.vocab_size: int = 32000
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> "ChainContext":
        if self.executor is None:
            from .executor import LocalExecutor
            self.executor = LocalExecutor(
                model_name_or_id=self.model_name,
                device=self.device,
                sparsity=self.sparsity
            )
            self.tokenizer = self.executor.tokenizer
            self.vocab_size = self.executor.vocab_size

        self._token = set_active_context(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token is not None:
            _ACTIVE_CONTEXT.reset(self._token)
            self._token = None

    def append_prompt(self, prompt: str):
        self.history_text += prompt
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(prompt)
            self.history_tokens.extend(tokens)
        else:
            self.history_tokens.extend([ord(c) % self.vocab_size for c in prompt])

    def append_output(self, text: str, tokens: List[int], logprobs: List[float]):
        self.history_text += text
        self.history_tokens.extend(tokens)
        self.token_logprobs.extend(logprobs)

    def fork(self, n: int, temperature: float = 0.7) -> List[BranchContext]:
        """
        Creates n child BranchContext instances that share the current context prefix.
        """
        branches = []
        for i in range(n):
            b = BranchContext(
                branch_id=i,
                parent=self,
                metadata={"temperature": temperature}
            )
            branches.append(b)
        return branches
