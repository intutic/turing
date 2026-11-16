"""
Chain & Branch Execution Context for Turing Programmatic DSL.
Manages prompt state, active token histories, and branches that share one prefilled prefix KV cache.
"""

import contextvars
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

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
    Child execution branch spawned by fork(). Its requests continue the parent's prefix, prefilled once at fork()
    (`prefix`: {"handle": the executor's KV handle or None once released}), with the branch's own suffix.
    gen() starts the branch and returns a future of its text; join() (or wait()) collects it.
    """
    branch_id: int
    parent: "ChainContext"
    branch_text: str = ""
    branch_tokens: List[int] = field(default_factory=list)
    token_logprobs: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    prefix: Dict[str, Any] = field(default_factory=dict)
    prefix_tokens: List[int] = field(default_factory=list)
    _pending: Any = None

    @property
    def full_text(self) -> str:
        return self.parent.history_text + self.branch_text

    @property
    def full_tokens(self) -> List[int]:
        return self.prefix_tokens + self.branch_tokens

    @property
    def mean_logprob(self) -> float:
        if not self.token_logprobs:
            return 0.0
        return float(sum(self.token_logprobs) / len(self.token_logprobs))

    def wait(self) -> str:
        """Finish the branch's running gen() (if any) and return the text it generated."""
        if self._pending is None:
            return ""
        text, tokens, logprobs = self._pending.result()
        self._pending = None
        self.branch_text += text
        self.branch_tokens.extend(tokens)
        self.token_logprobs.extend(logprobs)
        return text

    def _append(self, text: str):
        self.branch_text += text
        self.branch_tokens.extend(self.parent.encode(text))

    def gen(
        self,
        prompt: str = "",
        max_tokens: int = 64,
        temperature: Optional[float] = None,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
        schema: Optional[Dict[str, Any]] = None
    ):
        """Starts generating on this branch (temperature defaults to fork()'s); branches started one after another
        decode concurrently. Returns a concurrent.futures.Future of (text, tokens, logprobs); join() collects it."""
        self.wait()
        if prompt:
            self._append(prompt)
        self._pending = self.parent.executor.submit_branch(
            self.prefix_tokens, self.prefix.get("handle"), self.branch_tokens, max_tokens=max_tokens,
            temperature=self.metadata.get("temperature", 0.7) if temperature is None else temperature,
            top_k=top_k, stop=stop, schema=schema or self.parent.current_schema)
        return self._pending

    def select(self, options: List[str], prompt: str = "") -> str:
        """Constrains generation to select one option from the provided list."""
        self.wait()
        if prompt:
            self._append(prompt)
        selected = self.parent.executor.select_option(tokens=self.full_tokens, options=options)
        self._append(selected)
        return selected


class ChainContext:
    """
    Main stateful context for a @turing.chain workflow.
    """

    def __init__(
        self,
        model: Optional[str] = "test-tiny",
        device: str = "auto",
        sparsity: float = 0.0,
        executor: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        self.model_name = model
        self.device = device
        self.sparsity = sparsity
        self.tokenizer = tokenizer
        self.executor = executor
        self._owns_executor = False

        self.history_text: str = ""
        self.history_tokens: List[int] = []
        self.token_logprobs: List[float] = []
        self.current_schema: Optional[Dict[str, Any]] = None
        self.vocab_size: int = 32000
        self._prefixes: List[Dict[str, Any]] = []
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> "ChainContext":
        if self.executor is None:
            from .executor import LocalExecutor
            self.executor = LocalExecutor(
                model_name_or_id=self.model_name,
                device=self.device,
                sparsity=self.sparsity
            )
            self._owns_executor = True
        if self.tokenizer is None:
            self.tokenizer = getattr(self.executor, "tokenizer", None)
        self.vocab_size = getattr(self.executor, "vocab_size", self.vocab_size)

        self._token = set_active_context(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in self._prefixes:
            self.release(p)
        self._prefixes = []
        if self._owns_executor and hasattr(self.executor, "close"):
            self.executor.close()
        if self._token is not None:
            _ACTIVE_CONTEXT.reset(self._token)
            self._token = None

    def encode(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return [ord(c) % self.vocab_size for c in text]

    def append_prompt(self, prompt: str):
        self.history_text += prompt
        self.history_tokens.extend(self.encode(prompt))

    def append_output(self, text: str, tokens: List[int], logprobs: List[float]):
        self.history_text += text
        self.history_tokens.extend(tokens)
        self.token_logprobs.extend(logprobs)

    def fork(self, n: int, temperature: float = 0.7) -> List[BranchContext]:
        """
        n branches that continue the current context: its tokens are prefilled once, here, and every branch's
        requests share that KV cache copy-on-write.
        """
        prefix = {"handle": self.executor.prefill_prefix(self.history_tokens)}
        self._prefixes.append(prefix)
        return [BranchContext(branch_id=i, parent=self, metadata={"temperature": temperature}, prefix=prefix,
                              prefix_tokens=list(self.history_tokens)) for i in range(n)]

    def release(self, prefix: Dict[str, Any]) -> None:
        """Free a fork's shared prefix cache (its branches' later requests re-send the prefix)."""
        handle, prefix["handle"] = prefix.get("handle"), None
        if handle is not None:
            self.executor.release_prefix(handle)
