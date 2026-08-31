"""
Core Primitives for Turing Programmatic DSL.
Exposes @chain, gen(), fork(), join(), select(), and constrain().
"""

import functools
from collections import Counter
from typing import List, Dict, Any, Optional, Callable, Union
from .context import ChainContext, BranchContext, get_active_context, set_active_context

__all__ = ["chain", "gen", "fork", "join", "select", "constrain"]


def chain(
    model: Optional[str] = "test-tiny",
    device: str = "auto",
    sparsity: float = 0.5,
    executor: Optional[Any] = None
):
    """
    Decorator that transforms a Python function into an autonomous Turing DSL workflow chain.
    """
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            ctx = ChainContext(
                model=model,
                device=device,
                sparsity=sparsity,
                executor=executor
            )
            with ctx:
                return fn(*args, **kwargs)
        return wrapper
    return decorator


def gen(
    prompt: str = "",
    max_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 50,
    stop: Optional[List[str]] = None,
    schema: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generates text sequentially in the active @turing.chain context.
    """
    ctx = get_active_context()
    if prompt:
        ctx.append_prompt(prompt)

    text, tokens, logprobs = ctx.executor.generate(
        tokens=ctx.history_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        stop=stop,
        schema=schema or ctx.current_schema
    )

    ctx.append_output(text, tokens, logprobs)
    return text


def fork(n: int = 2, temperature: float = 0.7) -> List[BranchContext]:
    """
    Forks the active context into n parallel child branches that share the parent's prefix KV cache.
    """
    ctx = get_active_context()
    return ctx.fork(n=n, temperature=temperature)


def join(
    branches: List[BranchContext],
    strategy: Union[str, Callable[[List[BranchContext]], Any]] = "best"
) -> Any:
    """
    Merges outputs across forked branches according to the chosen strategy:
    - 'best': Selects branch with highest mean log-probability
    - 'vote': Majority vote on trimmed branch text
    - 'concat': Concatenates all branch texts
    - Callable: Custom aggregator function
    """
    if not branches:
        return ""

    if callable(strategy):
        return strategy(branches)

    if strategy == "best":
        # Sort by mean log probability
        best_b = max(branches, key=lambda b: b.mean_logprob)
        # Update active context with best branch
        ctx = get_active_context()
        ctx.append_output(best_b.branch_text, best_b.branch_tokens, best_b.token_logprobs)
        return best_b.branch_text

    elif strategy == "vote":
        # Extract text responses and vote
        responses = [b.branch_text.strip() for b in branches if b.branch_text.strip()]
        if not responses:
            return ""
        counts = Counter(responses)
        winner_text, _ = counts.most_common(1)[0]
        # Find matching branch to update context
        for b in branches:
            if b.branch_text.strip() == winner_text:
                ctx = get_active_context()
                ctx.append_output(b.branch_text, b.branch_tokens, b.token_logprobs)
                break
        return winner_text

    elif strategy == "concat":
        combined = "\n".join([b.branch_text for b in branches])
        ctx = get_active_context()
        ctx.append_output(combined, [], [])
        return combined

    else:
        raise ValueError(f"Unknown join strategy: '{strategy}'. Choose 'best', 'vote', 'concat', or pass a custom callable.")


def select(options: List[str], prompt: str = "") -> str:
    """
    Forces the model to choose one option from a discrete set of strings.
    """
    ctx = get_active_context()
    if prompt:
        ctx.append_prompt(prompt)

    selected = ctx.executor.select_option(
        tokens=ctx.history_tokens,
        options=options
    )
    if ctx.tokenizer is not None:
        sel_tokens = ctx.tokenizer.encode(selected)
    else:
        sel_tokens = [ord(c) % ctx.vocab_size for c in selected]

    ctx.append_output(selected, sel_tokens, [0.0] * len(sel_tokens))
    return selected


def constrain(schema: Dict[str, Any]):
    """
    Applies JSON Schema validation constraints to subsequent gen() calls in this context.
    """
    ctx = get_active_context()
    ctx.current_schema = schema
