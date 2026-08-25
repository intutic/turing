"""
Modal Labs Serverless App for Turing Engine.
Deploys Turing Engine on serverless NVIDIA L4 / A10G GPUs with instant cold-start caching.
"""

try:
    import modal
except ImportError:
    modal = None

if modal:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch", "transformers", "accelerate", "fastapi", "uvicorn")
        .run_commands("pip install git+https://github.com/intutic/turing.git")
    )

    app = modal.App("turing-engine-serverless", image=image)

    @app.cls(gpu="L4", container_idle_timeout=300)
    class TuringModelService:
        @modal.enter()
        def load_engine(self):
            from turing.serving.engine import ContinuousBatchEngine
            from turing.models.registry import get_model_config
            self.engine = ContinuousBatchEngine(config=get_model_config("llama-3.1-70b"))

        @modal.method()
        def generate(self, prompt: str, max_tokens: int = 128) -> str:
            result = self.engine.generate(prompt=prompt, max_tokens=max_tokens)
            return result.text

    @app.local_entrypoint()
    def main(prompt: str = "Explain the Birkhoff doubly stochastic hyper-connections:"):
        service = TuringModelService()
        output = service.generate.remote(prompt)
        print("--- Output from Modal ---\n", output)
