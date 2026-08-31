"""
Turing Programmatic DSL Package.
High-performance prompt chaining, parallel tree-of-thought exploration, and structured decoding.
"""

from .primitives import chain, gen, fork, join, select, constrain
from .context import ChainContext, BranchContext, get_active_context, set_active_context
from .executor import BaseExecutor, LocalExecutor, RemoteExecutor

__all__ = [
    "chain",
    "gen",
    "fork",
    "join",
    "select",
    "constrain",
    "ChainContext",
    "BranchContext",
    "get_active_context",
    "set_active_context",
    "BaseExecutor",
    "LocalExecutor",
    "RemoteExecutor",
]
