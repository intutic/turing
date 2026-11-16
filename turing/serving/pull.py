"""
Ollama /api/pull for HuggingFace repos: the repo's files are downloaded into the local HuggingFace cache (the same
cache `turing serve --model <repo>` loads from) with huggingface_hub, reported as Ollama progress lines.
"""

import asyncio
import json
from typing import AsyncIterator, List, Optional, Tuple

# Other frameworks' copies of the weights: not needed to serve with PyTorch
_SKIP = ("onnx/", "openvino/", "coreml/", "flax_model", "tf_model", "rust_model", ".h5", ".msgpack", ".ot", ".gguf")


def repo_and_revision(model: str) -> Tuple[str, Optional[str]]:
    """'hf.co/org/name:rev' (Ollama's form), 'org/name:rev' or 'org/name' -> (repo id, revision or None)."""
    for prefix in ("https://huggingface.co/", "huggingface.co/", "hf.co/"):
        if model.startswith(prefix):
            model = model[len(prefix):]
    repo, _, rev = model.partition(":")
    return repo, (rev if rev and rev != "latest" else None)


def pull_plan(repo: str, revision: Optional[str]) -> List[Tuple[str, int, str]]:
    """The files to download: (name, size in bytes, digest). Raises huggingface_hub errors
    (RepositoryNotFoundError, RevisionNotFoundError, ...) for a repo that cannot be pulled."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    files = [s for s in info.siblings if not any(k in s.rfilename for k in _SKIP)]
    if any(s.rfilename.endswith(".safetensors") for s in files):  # the same weights as pickles: skip them
        files = [s for s in files if not s.rfilename.endswith((".bin", ".pt", ".pth"))
                 or not s.rfilename.split("/")[-1].startswith(("pytorch_model", "model"))]
    return [(s.rfilename, s.size or 0, f"sha256:{s.lfs.sha256}" if s.lfs else s.blob_id) for s in files]


async def pull_progress(repo: str, revision: Optional[str], files: List[Tuple[str, int, str]]) -> AsyncIterator[dict]:
    """Ollama progress objects while the files download: one before and one after each file, then success.
    A failure ends the stream with {"error": ...}. Downloading by the requested revision (not its commit hash)
    records the branch in the cache, so the repo then loads offline too."""
    from huggingface_hub import hf_hub_download
    yield {"status": "pulling manifest"}
    try:
        for name, size, digest in files:
            yield {"status": f"pulling {name}", "digest": digest, "total": size, "completed": 0}
            await asyncio.to_thread(hf_hub_download, repo, name, revision=revision)
            yield {"status": f"pulling {name}", "digest": digest, "total": size, "completed": size}
    except Exception as e:  # network, disk or hub errors: Ollama reports them as an error line
        yield {"error": f"pull {repo}: {e}"}
        return
    yield {"status": "success"}


async def ndjson(progress: AsyncIterator[dict]) -> AsyncIterator[str]:
    async for p in progress:
        yield json.dumps(p) + "\n"
