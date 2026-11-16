"""
Turing Engine Provider Adapter for LiteLLM.
Points LiteLLM at a Turing Engine server's OpenAI-compatible API; the only Turing-specific request header is the
number of speculative draft tokens (KV cache format, pruning and quantization are chosen when the server starts).
"""

import os
from typing import Dict, Optional, List


class TuringEngineConfig:
    """
    Configuration helper for Turing Engine in LiteLLM.
    Drop-in compatibility with OpenAI-compatible gateway structures.
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        speculative_draft_tokens: int = 4,
    ):
        self.api_base = api_base or os.getenv(
            "TURING_API_BASE", "http://localhost:8000/v1"
        )
        self.api_key = api_key or os.getenv("TURING_API_KEY", "turing-local")
        self.speculative_draft_tokens = speculative_draft_tokens

    def get_supported_openai_params(self) -> List[str]:
        return [
            "temperature",
            "top_p",
            "max_tokens",
            "stream",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "user",
            "model",
        ]

    def map_request_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Turing-Draft-Tokens": str(self.speculative_draft_tokens),
        }

    def transform_model_name(self, model: str) -> str:
        if model.startswith("turing/"):
            return model[len("turing/") :]
        return model
