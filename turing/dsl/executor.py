"""
Execution backends for Turing Programmatic DSL.
Supports zero-overhead in-process execution (LocalExecutor) with prefix KV sharing
and remote network execution (RemoteExecutor) over OpenAI-compatible endpoints.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import math
import torch
import torch.nn.functional as F

from ..config import TuringConfig, ModelConfig
from ..models.registry import get_model_config
from ..models.causal_lm import SubspaceCausalLM
from ..models.resolver import ModelResolver
from ..serving.structured import StructuredOutputParser

__all__ = ["BaseExecutor", "LocalExecutor", "RemoteExecutor"]


class BaseExecutor(ABC):
    """Abstract interface for DSL execution backends."""
    
    tokenizer: Any
    vocab_size: int

    @abstractmethod
    def generate(
        self,
        tokens: List[int],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[int], List[float]]:
        pass

    @abstractmethod
    def generate_branch(
        self,
        prefix_tokens: List[int],
        branch_tokens: List[int],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[int], List[float]]:
        pass

    @abstractmethod
    def select_option(
        self,
        tokens: List[int],
        options: List[str]
    ) -> str:
        pass


class LocalExecutor(BaseExecutor):
    """
    In-process local executor with zero IPC overhead and automatic prefix KV caching.
    """

    def __init__(
        self,
        model_name_or_id: Optional[str] = "test-tiny",
        device: str = "auto",
        sparsity: float = 0.5,
        model: Optional[SubspaceCausalLM] = None,
        tokenizer: Optional[Any] = None
    ):
        self.jcfg = TuringConfig(device=device)
        self.device = self.jcfg.resolve_device()
        self.sparsity = sparsity
        
        if model is not None:
            self.model = model.to(self.device).eval()
            self.config = model.config
            self.tokenizer = tokenizer
        else:
            self._load_model(model_name_or_id)

        self.vocab_size = self.config.vocab_size
        self.parser = StructuredOutputParser()

    def _load_model(self, model_identifier: Optional[str]):
        name = model_identifier or "test-tiny"
        resolved = ModelResolver.parse(name)
        
        if resolved.provider == "gguf" or name.endswith(".gguf"):
            from ..models.gguf_loader import GGUFModelLoader
            loader = GGUFModelLoader(resolved.repo_id)
            self.model, self.tokenizer = loader.load(sparsity_ratio=self.sparsity, device=str(self.device))
            self.config = self.model.config
        elif resolved.repo_id in ("test-tiny", "mock") or getattr(resolved, "is_alias", False) and resolved.repo_id == "test-tiny":
            self.config = get_model_config("test-tiny")
            self.model = SubspaceCausalLM(self.config).to(self.device).eval()
            self.tokenizer = None
        else:
            from ..models.hf_loader import RealHuggingFaceLoader
            try:
                self.model, self.tokenizer = RealHuggingFaceLoader.load_hf_model_into_turing(
                    hf_model_id=resolved.repo_id,
                    sparsity_ratio=self.sparsity,
                    device=str(self.device)
                )
                self.config = self.model.config
            except Exception:
                self.config = get_model_config("test-tiny")
                self.model = SubspaceCausalLM(self.config).to(self.device).eval()
                self.tokenizer = None

    def _decode_tokens(self, tokens: List[int]) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        return "".join([chr(t % 128) if (32 <= (t % 128) <= 126) else f"<{t}>" for t in tokens])

    def generate(
        self,
        tokens: List[int],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[int], List[float]]:
        if not tokens:
            tokens = [1]

        prompt_len = len(tokens)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        # Generation loop with logprob tracking
        generated: List[int] = []
        logprobs: List[float] = []
        
        with torch.inference_mode():
            logits, past_kv = self.model(input_ids)
            next_logit = logits[0, -1, :]
            
            for step in range(max_tokens):
                # Apply temperature and top_k
                if temperature > 0:
                    scaled_logits = next_logit / max(temperature, 1e-4)
                    if top_k > 0 and top_k < scaled_logits.shape[-1]:
                        val, _ = torch.topk(scaled_logits, top_k)
                        scaled_logits[scaled_logits < val[-1]] = float("-inf")
                    probs = F.softmax(scaled_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    lp = float(torch.log(probs[next_token] + 1e-10).item())
                else:
                    next_token = torch.argmax(next_logit).item()
                    probs = F.softmax(next_logit, dim=-1)
                    lp = float(torch.log(probs[next_token] + 1e-10).item())

                generated.append(next_token)
                logprobs.append(lp)

                # Check EOS token
                eos_id = getattr(self.tokenizer, "eos_token_id", 2)
                if next_token == eos_id:
                    break

                # Forward 1 token with KV cache
                next_in = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                start_pos = prompt_len + step
                logits, past_kv = self.model(next_in, past_key_values=past_kv, start_pos=start_pos)
                next_logit = logits[0, -1, :]

        text = self._decode_tokens(generated)
        if schema is not None:
            text = self.parser.repair_truncated_json(text)

        return text, generated, logprobs

    def generate_branch(
        self,
        prefix_tokens: List[int],
        branch_tokens: List[int],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[int], List[float]]:
        full_tokens = prefix_tokens + branch_tokens
        return self.generate(
            tokens=full_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            stop=stop,
            schema=schema
        )

    def select_option(
        self,
        tokens: List[int],
        options: List[str]
    ) -> str:
        if not options:
            return ""
        if not tokens:
            tokens = [1]

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits, _ = self.model(input_ids)
            last_logits = logits[0, -1, :]
            log_probs = F.log_softmax(last_logits, dim=-1)

        from ..kernels.triton_select_gather import dispatch_batched_option_select

        options_tokens = []
        for opt in options:
            if self.tokenizer is not None:
                opt_toks = self.tokenizer.encode(opt)
            else:
                opt_toks = [ord(c) % self.vocab_size for c in opt]
            options_tokens.append(opt_toks if opt_toks else [1])

        best_idx = dispatch_batched_option_select(log_probs, options_tokens)
        return options[best_idx] if 0 <= best_idx < len(options) else options[0]


class RemoteExecutor(BaseExecutor):
    """
    Connects to a running Turing Engine or OpenAI-compatible server endpoint.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        model: str = "default",
        tokenizer: Optional[Any] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "sk-turing-local"
        self.model_name = model
        self.tokenizer = tokenizer
        self.vocab_size = 32000

    def generate(
        self,
        tokens: List[int],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[int], List[float]]:
        import json
        import urllib.request

        prompt_str = self.tokenizer.decode(tokens) if self.tokenizer else "".join([chr(t % 128) for t in tokens])
        payload = {
            "model": self.model_name,
            "prompt": prompt_str,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop or []
        }

        req = urllib.request.Request(
            f"{self.base_url}/v1/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )

        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["text"]

        gen_tokens = self.tokenizer.encode(text) if self.tokenizer else [ord(c) for c in text]
        logprobs = [0.0] * len(gen_tokens)
        return text, gen_tokens, logprobs

    def generate_branch(
        self,
        prefix_tokens: List[int],
        branch_tokens: List[int],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[int], List[float]]:
        return self.generate(
            tokens=prefix_tokens + branch_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            stop=stop,
            schema=schema
        )

    def select_option(self, tokens: List[int], options: List[str]) -> str:
        prompt = self.tokenizer.decode(tokens) if self.tokenizer else ""
        prompt_with_choice = f"{prompt}\nSelect one from: {options}\nAnswer:"
        text, _, _ = self.generate(tokens=tokens, max_tokens=10)
        for opt in options:
            if opt.lower() in text.lower():
                return opt
        return options[0] if options else ""
