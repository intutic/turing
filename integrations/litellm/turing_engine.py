"""
Turing Engine Provider Adapter for LiteLLM.
Translates LiteLLM generic completion requests into Turing Engine endpoints
supporting subspace channel pruning, SVD INT8 KV compression, and speculative decoding.
"""

import os
from typing import Dict, Any, Optional, List, Union

class TuringEngineConfig:
    """
    Configuration helper for Turing Engine in LiteLLM.
    Drop-in compatibility with OpenAI-compatible gateway structures.
    """
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        sparsity_ratio: float = 0.57,
        use_svd_kv_cache: bool = True,
        speculative_draft_tokens: int = 4
    ):
        self.api_base = api_base or os.getenv("TURING_API_BASE", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("TURING_API_KEY", "turing-local")
        self.sparsity_ratio = sparsity_ratio
        self.use_svd_kv_cache = use_svd_kv_cache
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
            "model"
        ]

    def map_request_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Turing-Sparsity": str(self.sparsity_ratio),
            "X-Turing-SVD-KV": "1" if self.use_svd_kv_cache else "0",
            "X-Turing-Draft-Tokens": str(self.speculative_draft_tokens)
        }

    def transform_model_name(self, model: str) -> str:
        if model.startswith("turing/"):
            return model[len("turing/"):]
        return model
