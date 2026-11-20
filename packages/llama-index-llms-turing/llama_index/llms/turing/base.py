"""LlamaIndex LLM for a Turing Engine server (`turing serve`), on LlamaIndex's OpenAI-compatible client."""

import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike

DEFAULT_API_BASE = "http://localhost:8000/v1"


class Turing(OpenAILike):
    """
    Turing Engine LLM: chat completions, streaming, async and tool calling against `turing serve`.

    Examples:
        `pip install llama-index-llms-turing`

        ```python
        from llama_index.llms.turing import Turing

        llm = Turing(model="Qwen/Qwen2.5-1.5B-Instruct", api_base="http://localhost:8000/v1", context_window=32768)
        print(llm.complete("Explain KV cache paging in one sentence:"))
        ```

    `api_base` and `api_key` default to `$TURING_API_BASE` (else http://localhost:8000/v1) and `$TURING_API_KEY`
    (checked only when the server runs with `--api-key`). Set `context_window` to the served model's.
    """

    def __init__(self, model: str, api_base: Optional[str] = None, api_key: Optional[str] = None,
                 is_chat_model: bool = True, is_function_calling_model: bool = True, **kwargs: Any) -> None:
        super().__init__(
            model=model,
            api_base=api_base or os.getenv("TURING_API_BASE", DEFAULT_API_BASE),
            api_key=api_key or os.getenv("TURING_API_KEY", "turing-local"),
            is_chat_model=is_chat_model,
            is_function_calling_model=is_function_calling_model,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Turing"
