"""
Execution backends for Turing Programmatic DSL.
LocalExecutor runs in process on the serving engine (ContinuousBatchEngine): gen() is one engine request, fork()
prefills the shared prefix once into a KV cache that every branch continues copy-on-write, and the branches decode
concurrently in one batch. RemoteExecutor talks to a running server over its OpenAI-compatible endpoints.
"""

import asyncio
import concurrent.futures
import threading
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F

from ..config import TuringConfig
from ..models.registry import get_model_config
from ..models.causal_lm import SubspaceCausalLM
from ..models.resolver import ModelResolver
from ..serving import generation
from ..serving.engine import ContinuousBatchEngine
from ..serving.structured import StructuredOutputError

__all__ = ["BaseExecutor", "LocalExecutor", "RemoteExecutor"]

Result = Tuple[str, List[int], List[float]]


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
    ) -> Result:
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
    ) -> Result:
        pass

    @abstractmethod
    def select_option(
        self,
        tokens: List[int],
        options: List[str]
    ) -> str:
        pass

    def prefill_prefix(self, tokens: List[int]) -> Any:
        """A handle to tokens prefilled once for fork() branches (None: this backend re-sends the prefix)."""
        return None

    def release_prefix(self, handle: Any) -> None:
        pass

    def submit_branch(self, prefix_tokens: List[int], handle: Any, branch_tokens: List[int],
                      **kw) -> "concurrent.futures.Future[Result]":
        """Start a branch; the future holds (text, tokens, logprobs). This default runs it to completion now."""
        fut: "concurrent.futures.Future[Result]" = concurrent.futures.Future()
        fut.set_result(self.generate_branch(prefix_tokens, branch_tokens, **kw))
        return fut


class _LogProbs:
    """Logits processor that records the model's log-probability of each sampled token (resolved one step later,
    when the sampled token is known; the last one after the request ends)."""

    def __init__(self):
        self.values: List[float] = []
        self._row: Optional[torch.Tensor] = None

    def __call__(self, gen: List[int], logits: torch.Tensor) -> None:
        self._resolve(gen)
        self._row = F.log_softmax(logits.float(), -1)

    def _resolve(self, gen: List[int]) -> None:
        if self._row is not None and len(gen) > len(self.values):
            self.values.append(float(self._row[gen[len(self.values)]]))
        self._row = None

    def finish(self, gen: List[int]) -> List[float]:
        self._resolve(gen)
        return self.values[:len(gen)]


class LocalExecutor(BaseExecutor):
    """
    In-process executor on a ContinuousBatchEngine driven by its own event-loop thread.
    """

    def __init__(
        self,
        model_name_or_id: Optional[str] = "test-tiny",
        device: str = "auto",
        sparsity: float = 0.0,
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
        # prefix caching off: branch requests continue a shared prefix KV (prefix_kv=), which the engine's prefix
        # index would record under the branch tokens alone
        self.engine = ContinuousBatchEngine(self.config, self.jcfg, model=self.model, tokenizer=self.tokenizer,
                                            prefix_caching=False)
        self.prefix_prefills = 0  # shared prefixes prefilled by fork()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

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
            except Exception as e:
                # Random weights served for a mistyped name would only produce gibberish
                raise RuntimeError(
                    f"Could not load weights for '{name}': {e}\n"
                    "Pass model_name_or_id='test-tiny' (or 'mock') to run on synthetic weights."
                ) from e

    # ------------------------------------------------------------------ the engine's loop

    def _submit(self, coro) -> "concurrent.futures.Future":
        """Run coro on the engine's event loop (started on first use)."""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, name="turing-dsl-engine", daemon=True)
            self._thread.start()
            asyncio.run_coroutine_threadsafe(self.engine.start(), self._loop).result()
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def close(self) -> None:
        """Stop the engine and its loop thread (the next call starts them again)."""
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.engine.stop(), self._loop).result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()
        self._loop = self._thread = None

    async def _run(self, prompt: List[int], max_tokens: int, temperature: float, top_k: int,
                   stop: Optional[List[str]], schema: Optional[Dict[str, Any]], prefix_kv=None) -> Result:
        lp = _LogProbs()
        params = generation.GenerationParams(max_tokens=max_tokens, temperature=temperature, top_k=top_k,
                                             stop=list(stop or []), schema=schema, reasoning=False,
                                             logits_processor=lp)
        try:
            gen = await generation.start(self.engine, prompt, params, prefix_kv=prefix_kv)
        except generation.RequestRejected as e:
            raise StructuredOutputError(str(e)) if schema is not None and e.status_code == 400 else e
        await gen.collect()
        if gen.schema_error:
            raise StructuredOutputError(f"constrained output does not satisfy the schema ({gen.schema_error}): "
                                        f"{gen.content!r}")
        return gen.content, gen.output_ids, lp.finish(gen.output_ids)

    # ------------------------------------------------------------------ BaseExecutor

    def generate(
        self,
        tokens: List[int],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> Result:
        if not tokens:
            tokens = [self.tokenizer.bos_token_id if getattr(self.tokenizer, "bos_token_id", None) is not None else 1]
        return self._submit(self._run(tokens, max_tokens, temperature, top_k, stop, schema)).result()

    def generate_branch(
        self,
        prefix_tokens: List[int],
        branch_tokens: List[int],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> Result:
        return self.generate(prefix_tokens + branch_tokens, max_tokens, temperature, top_k, stop, schema)

    def prefill_prefix(self, tokens: List[int]) -> Any:
        """All but the last prefix token prefilled once into a KV cache of the engine's pool (each branch request
        starts with the last one, so it has at least one token to prefill)."""
        if len(tokens) < 2:
            return None

        async def run():
            return generation.prefill(self.engine, tokens[:-1])

        handle = self._submit(run()).result()
        self.prefix_prefills += 1
        return handle

    def release_prefix(self, handle: Any) -> None:
        if handle is not None:
            async def run():
                handle.free()

            self._submit(run()).result()

    def submit_branch(self, prefix_tokens: List[int], handle: Any, branch_tokens: List[int],
                      max_tokens: int = 64, temperature: float = 0.7, top_k: int = 50,
                      stop: Optional[List[str]] = None, schema: Optional[Dict[str, Any]] = None):
        """Start a branch continuing the shared prefix (copy-on-write); branches submitted together decode in one
        batch. The future holds (text, tokens, logprobs)."""
        if handle is None:
            prompt, kv = (prefix_tokens + branch_tokens) or [1], None
        else:
            prompt, kv = prefix_tokens[-1:] + branch_tokens, handle
        return self._submit(self._run(prompt, max_tokens, temperature, top_k, stop, schema, prefix_kv=kv))

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
                opt_toks = self.tokenizer.encode(opt, add_special_tokens=False)
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
