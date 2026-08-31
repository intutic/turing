"""
Universal Model Namespace & Gateway Resolver for Turing Engine.
Parses canonical Hugging Face Hub IDs, provider namespaces (LiteLLM/OpenAI format),
reasoning effort controls, local filesystem paths, and developer CLI aliases.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

@dataclass(frozen=True)
class ResolvedModelSpec:
    """
    Structured specification of a resolved model target.
    """
    raw_identifier: str
    repo_id: str
    provider: str
    model_name: str
    reasoning_effort: Optional[str] = None
    is_local_path: bool = False
    is_alias: bool = False

    @property
    def canonical_name(self) -> str:
        if self.reasoning_effort:
            return f"{self.repo_id}:{self.reasoning_effort}"
        return self.repo_id


class ModelResolver:
    """
    Universal model identifier parser and gateway router.
    """
    # Ergonomic CLI Shortcuts for local testing & interactive demos
    CLI_ALIASES: Dict[str, str] = {
        "gpt2": "gpt2",
        "gpt-2": "gpt2",
        "smollm2": "HuggingFaceTB/SmolLM2-135M",
        "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
        "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
        "qwen-2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
        "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
        "qwen-2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
        "qwen-32b": "Qwen/Qwen2.5-32B-Instruct",
        "qwen-2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
        "qwen-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "qwen-2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "qwen-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "qwen-2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
        "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
        "gemma-2-2b": "google/gemma-2-2b-it",
        "gemma-2-9b": "google/gemma-2-9b-it",
        "gemma-2-27b": "google/gemma-2-27b-it",
        "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
        "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
        "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama-3.1-70b": "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
        "llama-3.3": "meta-llama/Llama-3.3-70B-Instruct",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
        "deepseek-r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-r1-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-r1-distill-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-r1-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-r1-distill-qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-r1-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-r1-distill-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-r1-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-r1-distill-llama-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
        "mistral-small": "mistralai/Mistral-Small-24B-Instruct-2501",
        "mistral-small-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
        "phi-4": "microsoft/phi-4",
        "phi-4-mini": "microsoft/Phi-4-mini-instruct",
        "glm-4-9b": "THUDM/glm-4-9b-chat",
        "internlm3-8b": "internlm/internlm3-8b-instruct",
        "minicpm3-4b": "openbmb/MiniCPM3-4B",
        "yi-1.5-9b": "01-ai/Yi-1.5-9B-Chat",
        "yi-1.5-34b": "01-ai/Yi-1.5-34B-Chat",
        "test-tiny": "test-tiny",
        "mock": "test-tiny",
    }

    KNOWN_EFFORT_LEVELS = {"high", "medium", "low", "default", "none"}

    @classmethod
    def parse(cls, raw_identifier: str) -> ResolvedModelSpec:
        """
        Parses any model string into a ResolvedModelSpec.
        
        Handles:
        1. Local path: `/models/llama-3-8b` or `./weights/`
        2. Colon reasoning suffix: `meta-llama/Llama-3.3-70B:high`
        3. Tri-part provider/model/effort: `deepseek-ai/DeepSeek-R1/high`
        4. Provider prefix (LiteLLM / Gateway): `huggingface/meta-llama/Llama-3-8B`
        5. CLI convenience aliases: `deepseek-r1-1.5b` -> `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
        """
        raw = raw_identifier.strip()
        effort: Optional[str] = None
        is_local = os.path.exists(raw) or raw.startswith(("./", "/", "~"))
        is_alias = False
        provider = "huggingface"

        # 1. Local path check
        if is_local:
            model_name = os.path.basename(os.path.normpath(raw))
            return ResolvedModelSpec(
                raw_identifier=raw_identifier,
                repo_id=raw,
                provider="local",
                model_name=model_name,
                reasoning_effort=None,
                is_local_path=True,
                is_alias=False
            )

        # 2. Extract colon reasoning suffix (e.g. repo:high)
        if ":" in raw:
            parts = raw.split(":", 1)
            candidate_effort = parts[1].strip().lower()
            if candidate_effort in cls.KNOWN_EFFORT_LEVELS:
                effort = candidate_effort if candidate_effort != "none" else None
                raw = parts[0].strip()

        # 3. Check for Provider Prefixes (e.g. huggingface/org/model or openai/o3-mini)
        slash_parts = [p.strip() for p in raw.split("/") if p.strip()]
        
        if len(slash_parts) == 3:
            # Case A: provider/org/model (e.g. huggingface/meta-llama/Llama-3-8B)
            if slash_parts[0].lower() in ("huggingface", "hf", "local", "vllm"):
                provider = slash_parts[0].lower()
                raw = f"{slash_parts[1]}/{slash_parts[2]}"
            # Case B: org/model/effort (e.g. deepseek-ai/DeepSeek-R1/high)
            elif slash_parts[2].lower() in cls.KNOWN_EFFORT_LEVELS:
                effort = slash_parts[2].lower()
                raw = f"{slash_parts[0]}/{slash_parts[1]}"
            else:
                provider = slash_parts[0]
                raw = f"{slash_parts[1]}/{slash_parts[2]}"

        elif len(slash_parts) == 4:
            # provider/org/model/effort (e.g. hf/deepseek-ai/DeepSeek-R1/high)
            provider = slash_parts[0].lower()
            if slash_parts[3].lower() in cls.KNOWN_EFFORT_LEVELS:
                effort = slash_parts[3].lower()
            raw = f"{slash_parts[1]}/{slash_parts[2]}"

        # 4. Resolve CLI Aliases
        key = raw.lower().replace("_", "-")
        if key in cls.CLI_ALIASES:
            repo_id = cls.CLI_ALIASES[key]
            is_alias = (repo_id != raw)
        else:
            repo_id = raw

        model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

        return ResolvedModelSpec(
            raw_identifier=raw_identifier,
            repo_id=repo_id,
            provider=provider,
            model_name=model_name,
            reasoning_effort=effort,
            is_local_path=False,
            is_alias=is_alias
        )

    @classmethod
    def resolve_repo_id(cls, model_identifier: str) -> str:
        return cls.parse(model_identifier).repo_id

    @classmethod
    def resolve_reasoning_effort(cls, model_identifier: str) -> Optional[str]:
        return cls.parse(model_identifier).reasoning_effort
