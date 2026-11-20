"""`ChatTuring` moved to the `langchain-turing` package (pip install langchain-turing); this name re-exports it."""

try:
    from langchain_turing import ChatTuring
except ImportError as e:  # optional dependency
    raise ImportError("ChatTuring is in the langchain-turing package: pip install langchain-turing") from e

ChatTuringEngine = ChatTuring

__all__ = ["ChatTuring", "ChatTuringEngine"]
