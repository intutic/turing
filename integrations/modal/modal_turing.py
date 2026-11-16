"""
Modal app for Turing Engine on a serverless L4 GPU. Run from the repository root:

    TURING_MODEL=smollm2 modal run integrations/modal/modal_turing.py --prompt "Hello"

The image copies this repository and installs it (`pip install .`, which builds the C++ extension).
Not run in this repository's CI; tests/test_ecosystem_integrations.py tests the shared request path.
"""

import os
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

MODEL_ID = os.environ.get("TURING_MODEL", "smollm2")
REPO = "/root/jflux"

if modal:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("build-essential")
        .add_local_dir(Path(__file__).resolve().parents[2], REPO, copy=True,
                       ignore=[".git", "build", "**/__pycache__", "*.so", "results_*.json"])
        .run_commands(f"pip install {REPO}")
        .env({"PYTHONPATH": REPO, "TURING_MODEL": MODEL_ID})
    )

    app = modal.App("turing-engine", image=image)

    @app.cls(gpu="L4", scaledown_window=300, timeout=900)
    class TuringModelService:
        @modal.enter()
        def load_engine(self):
            from integrations.common import InProcessTuring, load_engine
            self.turing = InProcessTuring(load_engine(MODEL_ID, "cuda"))

        @modal.method()
        async def chat(self, body: dict) -> dict:
            """An OpenAI chat request body in, the OpenAI chat completion out."""
            return await self.turing.chat(body)

    @app.local_entrypoint()
    def main(prompt: str = "Explain KV cache paging in one sentence:", max_tokens: int = 64):
        out = TuringModelService().chat.remote({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens})
        print(out["choices"][0]["message"]["content"])
