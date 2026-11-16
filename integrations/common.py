"""
Shared pieces of the Ray Serve, RunPod and Modal recipes: load real weights into the serving engine,
and call Turing's OpenAI-compatible app in-process.
"""

from typing import Any, Dict

import httpx

from turing.config import TuringConfig
from turing.models.hf_loader import RealHuggingFaceLoader
from turing.serving.engine import ContinuousBatchEngine
from turing.serving.server import create_app


def load_engine(model_id: str, device: str = "auto", sparsity: float = 0.0) -> ContinuousBatchEngine:
    """Real HuggingFace weights (repo id or a `turing serve` alias such as smollm2) in a ContinuousBatchEngine."""
    cfg = TuringConfig(device=device)
    model, tokenizer = RealHuggingFaceLoader.load_hf_model_into_turing(
        hf_model_id=model_id, sparsity_ratio=sparsity, device=str(cfg.resolve_device()))
    return ContinuousBatchEngine(model.config, cfg, model=model, tokenizer=tokenizer)


class InProcessTuring:
    """The app `turing serve` runs (create_app), called through httpx without a socket."""

    def __init__(self, engine: ContinuousBatchEngine):
        self.engine = engine
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(engine)),
                                        base_url="http://turing", timeout=None)

    async def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        # ASGITransport does not run the app's lifespan, so start the scheduler here (a no-op once running).
        # The response body is buffered: a "stream": true request gets its whole SSE body at once.
        await self.engine.start()
        return await self.client.request(method, path, **kwargs)

    async def chat(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """One /v1/chat/completions request (OpenAI request body in, OpenAI response body out)."""
        resp = await self.request("POST", "/v1/chat/completions",
                                  json={"model": self.engine.model_config.name, **body, "stream": False})
        resp.raise_for_status()
        return resp.json()
