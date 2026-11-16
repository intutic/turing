"""
Ray Serve deployment for Turing Engine: one replica per GPU, each serving Turing's full HTTP API
(OpenAI, Anthropic and Ollama routes). Run from the repository root:

    serve run integrations.ray.ray_serve_turing:app

Set TURING_MODEL (default smollm2) and TURING_DEVICE (default auto). Responses are buffered: a
"stream": true request gets its whole SSE body at once.
"""

import os

from starlette.requests import Request
from starlette.responses import Response

from integrations.common import InProcessTuring, load_engine

try:
    from ray import serve
except ImportError:
    serve = None


async def forward(turing: InProcessTuring, request: Request) -> Response:
    """Send a request Ray Serve received to the in-process Turing app and return its response."""
    resp = await turing.request(request.method, request.url.path, params=request.query_params,
                                content=await request.body(),
                                headers={"content-type": request.headers.get("content-type", "application/json")})
    return Response(resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


if serve is not None:
    @serve.deployment(ray_actor_options={"num_gpus": 1})
    class TuringRayServeDeployment:
        def __init__(self, model_id: str = "", device: str = ""):
            self.turing = InProcessTuring(load_engine(model_id or os.environ.get("TURING_MODEL", "smollm2"),
                                                      device or os.environ.get("TURING_DEVICE", "auto")))

        async def __call__(self, request: Request) -> Response:
            return await forward(self.turing, request)

    app = TuringRayServeDeployment.bind()
