"""
RunPod serverless worker for Turing Engine.

The job input is an OpenAI chat request body ({"messages": [...], "max_tokens": ...}) or {"prompt": "..."};
the output is the OpenAI chat completion. Run from the repository root (the image needs this repo and
`pip install .`):

    TURING_MODEL=smollm2 python -m integrations.runpod.runpod_handler
"""

import os
from typing import Any, Dict, Optional

from turing.serving.engine import ContinuousBatchEngine

from integrations.common import InProcessTuring, load_engine


class RunPodTuringWorker:
    def __init__(self, model_id: str = "smollm2", device: str = "auto", engine: Optional[ContinuousBatchEngine] = None):
        self.turing = InProcessTuring(engine or load_engine(model_id, device))

    async def handler(self, job: Dict[str, Any]) -> Dict[str, Any]:
        body = dict(job.get("input") or {})
        if "messages" not in body:
            body["messages"] = [{"role": "user", "content": body.pop("prompt", "")}]
        return await self.turing.chat(body)


def start_worker():
    import runpod

    worker = RunPodTuringWorker(os.environ.get("TURING_MODEL", "smollm2"), os.environ.get("TURING_DEVICE", "auto"))
    runpod.serverless.start({"handler": worker.handler})


if __name__ == "__main__":
    start_worker()
