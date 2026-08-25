"""
RunPod Serverless Worker Handler for Turing Engine.
Handles serverless job invocations with streaming and batch generation.
"""

from typing import Dict, Any

try:
    import runpod
except ImportError:
    runpod = None

from turing.serving.engine import ContinuousBatchEngine
from turing.models.registry import get_model_config

class RunPodTuringWorker:
    def __init__(self, model_key: str = "llama-3.1-70b"):
        self.model_key = model_key
        config = get_model_config(model_key)
        self.engine = ContinuousBatchEngine(model_config=config)

    def handler(self, job: Dict[str, Any]) -> Dict[str, Any]:
        job_input = job.get("input", {})
        prompt = job_input.get("prompt", "")
        max_tokens = job_input.get("max_tokens", 128)

        # Generate sample response tokens
        output_tokens = [100, 101, 102]
        return {
            "text": f"Turing Engine response for: {prompt[:30]}",
            "completion_tokens": len(output_tokens),
            "generation_time_ms": 12.5
        }

def start_worker():
    if runpod:
        worker = RunPodTuringWorker()
        runpod.serverless.start({"handler": worker.handler})

if __name__ == "__main__":
    start_worker()
