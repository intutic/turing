"""
The model-backed side endpoints: /v1/embeddings, /api/embed, /api/embeddings, /v1/rerank, /v1/extract, /api/pull.
Real models only: the served LLM (test-tiny's synthetic weights, Qwen2.5-0.5B-Instruct), BAAI/bge-small-en-v1.5 and
cross-encoder/ms-marco-MiniLM-L-6-v2 when they are in the local HuggingFace cache, and a real pull from the Hub.
"""

import base64
import json
import os

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from huggingface_hub import try_to_load_from_cache

from turing.config import TuringConfig
from turing.models.causal_lm import SubspaceCausalLM
from turing.models.registry import get_model_config
from turing.serving.embeddings import _padded, llm_hidden_states, llm_logits
from turing.serving.engine import ContinuousBatchEngine
from turing.serving.server import create_app

EMBED, RERANK, LLM = "BAAI/bge-small-en-v1.5", "cross-encoder/ms-marco-MiniLM-L-6-v2", "Qwen/Qwen2.5-0.5B-Instruct"


def _cached(repo: str) -> bool:
    return isinstance(try_to_load_from_cache(repo, "config.json"), str) and \
        isinstance(try_to_load_from_cache(repo, "model.safetensors"), str)


def _need(repo: str):
    if not _cached(repo):
        pytest.skip(f"{repo} is not in the local HuggingFace cache")


def _tiny_app(**kw):
    engine = ContinuousBatchEngine(get_model_config("test-tiny"), TuringConfig(device="cpu"))
    return engine, create_app(engine, **kw)


@pytest.fixture(scope="module")
def qwen_client():
    _need(LLM)
    from turing.models.hf_loader import RealHuggingFaceLoader
    model, tok = RealHuggingFaceLoader.load_hf_model_into_turing(LLM, device="cpu")
    engine = ContinuousBatchEngine(model.config, TuringConfig(device="cpu"), model=model, tokenizer=tok)
    with TestClient(create_app(engine)) as client:
        yield client


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ------------------------------------------------------------------ the served LLM's hidden states

def test_llm_hidden_states_match_the_model_forward():
    """llm_hidden_states + llm_logits on a right-padded batch reproduce SubspaceCausalLM's own logits per row."""
    torch.manual_seed(0)
    model = SubspaceCausalLM(get_model_config("test-tiny")).eval()
    seqs = [[5, 9, 11, 3, 7, 2, 40], [4, 8, 15]]
    ids, _ = _padded(seqs, "cpu")
    h = llm_hidden_states(model, ids)
    for i, s in enumerate(seqs):
        with torch.inference_mode():
            ref = model(torch.tensor([s]))[0][0]
        torch.testing.assert_close(llm_logits(model, h[i, :len(s)]), ref, atol=1e-4, rtol=1e-4)


def test_llm_embeddings_endpoint():
    engine, app = _tiny_app()
    texts = ["the quick brown fox", "jumps over the lazy dog"]
    with TestClient(app) as client:
        r = client.post("/v1/embeddings", json={"input": texts})
        assert r.status_code == 200
        d = r.json()
        vecs = np.array([e["embedding"] for e in d["data"]])
        assert vecs.shape == (2, engine.model_config.hidden_dim)
        np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
        assert d["model"] == engine.model_config.name
        assert d["usage"]["prompt_tokens"] == sum(len(engine.encode_prompt(t)) for t in texts)

        again = client.post("/v1/embeddings", json={"input": texts[0]}).json()["data"][0]["embedding"]
        np.testing.assert_allclose(again, vecs[0], atol=1e-5)  # deterministic, and batching does not change it

        short = client.post("/v1/embeddings", json={"input": texts, "dimensions": 64}).json()["data"]
        s0 = np.array(short[0]["embedding"])
        assert s0.shape == (64,)
        np.testing.assert_allclose(s0, vecs[0][:64] / np.linalg.norm(vecs[0][:64]), atol=1e-5)

        b64 = client.post("/v1/embeddings", json={"input": texts, "encoding_format": "base64"}).json()["data"]
        np.testing.assert_allclose(np.frombuffer(base64.b64decode(b64[1]["embedding"]), dtype="<f4"), vecs[1],
                                   atol=1e-6)

        assert client.post("/v1/embeddings", json={"input": texts, "dimensions": 10_000}).status_code == 400
        assert client.post("/v1/embeddings", json={"input": ""}).status_code == 400


def test_ollama_embed_endpoints():
    engine, app = _tiny_app()
    with TestClient(app) as client:
        r = client.post("/api/embed", json={"model": "x", "input": ["first text", "second text"]})
        assert r.status_code == 200
        d = r.json()
        assert len(d["embeddings"]) == 2 and len(d["embeddings"][0]) == engine.model_config.hidden_dim
        assert d["prompt_eval_count"] == len("first text") + len("second text")  # test-tiny: one token per char
        legacy = client.post("/api/embeddings", json={"model": "x", "prompt": "first text"}).json()["embedding"]
        np.testing.assert_allclose(legacy, d["embeddings"][0], atol=1e-5)


def test_embedding_model_paraphrases_are_closer():
    """--embedding-model BAAI/bge-small-en-v1.5 (CLS pooling): paraphrases score above unrelated sentences."""
    _need(EMBED)
    _, app = _tiny_app(embedding_model=EMBED)
    texts = ["A man is playing a guitar.", "Someone is strumming a guitar.",
             "The stock market fell sharply today.", "How do I bake sourdough bread?"]
    with TestClient(app) as client:
        d = client.post("/v1/embeddings", json={"input": texts}).json()
        v = [np.array(e["embedding"]) for e in d["data"]]
        assert d["model"] == EMBED and len(v[0]) == 384
        np.testing.assert_allclose([np.linalg.norm(x) for x in v], 1.0, atol=1e-5)
        assert _cos(v[0], v[1]) > _cos(v[0], v[2]) + 0.1
        assert _cos(v[0], v[1]) > _cos(v[0], v[3]) + 0.1
        again = client.post("/v1/embeddings", json={"input": texts[1]}).json()["data"][0]["embedding"]
        np.testing.assert_allclose(again, v[1], atol=1e-5)
        assert d["usage"]["prompt_tokens"] > len(texts)  # word pieces plus [CLS] and [SEP]


# ------------------------------------------------------------------ rerank

QUERY = "What is the capital of France?"
DOCS = ["The mitochondria is the powerhouse of the cell.", "Paris is the capital and largest city of France.",
        "Stock markets fell sharply today."]


def test_rerank_with_cross_encoder():
    _need(RERANK)
    _, app = _tiny_app(rerank_model=RERANK)
    with TestClient(app) as client:
        d = client.post("/v1/rerank", json={"query": QUERY, "documents": DOCS}).json()
        res = d["results"]
        assert d["model"] == RERANK and "cross-encoder" in d["meta"]["scorer"]
        assert [r["index"] for r in res][0] == 1
        assert res[0]["document"]["text"] == DOCS[1]
        scores = [r["relevance_score"] for r in res]
        assert scores == sorted(scores, reverse=True) and all(0.0 < s < 1.0 for s in scores)
        assert scores[0] > 0.9 and scores[-1] < 0.1
        top = client.post("/v1/rerank", json={"query": QUERY, "documents": DOCS, "top_n": 1,
                                              "return_documents": False}).json()["results"]
        assert top == [{"index": 1, "relevance_score": pytest.approx(scores[0], abs=1e-6)}]


def test_rerank_with_served_llm(qwen_client):
    d = qwen_client.post("/v1/rerank", json={"query": QUERY, "documents": DOCS}).json()
    assert d["meta"]["scorer"].startswith("llm-yes-probability")
    assert d["results"][0]["index"] == 1
    assert d["results"][0]["relevance_score"] > 0.5 > d["results"][-1]["relevance_score"]


# ------------------------------------------------------------------ extract

def test_extract_with_instruct_model(qwen_client):
    text = "Dr. Alice Smith at Oxford University announced the finding in London on Tuesday."
    r = qwen_client.post("/v1/extract", json={"text": text, "labels": ["person", "organization", "location"]})
    assert r.status_code == 200
    d = r.json()
    assert d["count"] == len(d["data"]) > 0
    for span in d["data"]:  # every span is a real substring at its offsets
        assert text[span["start"]:span["end"]] == span["text"]
        assert span["label"] in ("person", "organization", "location")
        assert 0.0 < span["score"] <= 1.0
    found = {(s["label"], s["text"]) for s in d["data"]}
    assert ("organization", "Oxford University") in found
    assert d["usage"]["completion_tokens"] > 0 and d["raw_output"]


def test_extract_refuses_non_json_output():
    """test-tiny's synthetic weights cannot write JSON: the endpoint says so instead of inventing spans."""
    _, app = _tiny_app()
    with TestClient(app) as client:
        r = client.post("/v1/extract", json={"text": "Alice works at Acme.", "labels": ["person"], "max_tokens": 16})
        assert r.status_code == 422
        err = r.json()["error"]
        assert err["type"] == "invalid_model_output" and isinstance(err["raw_output"], str)


# ------------------------------------------------------------------ pull

TINY_REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _hub_reachable() -> bool:
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
        return True
    except OSError:
        return False


def test_ollama_pull_downloads_into_the_hf_cache():
    if os.environ.get("HF_HUB_OFFLINE") == "1" or not _hub_reachable():
        pytest.skip("the HuggingFace Hub is not reachable")
    _, app = _tiny_app()
    with TestClient(app) as client:
        with client.stream("POST", "/api/pull", json={"model": f"hf.co/{TINY_REPO}"}) as r:
            assert r.status_code == 200
            lines = [json.loads(line) for line in r.iter_lines() if line]
        assert lines[0] == {"status": "pulling manifest"} and lines[-1] == {"status": "success"}
        done = {ln["status"][len("pulling "):]: ln for ln in lines[1:-1] if ln["completed"] == ln["total"]}
        assert "model.safetensors" in done and "config.json" in done
        assert not any(name.startswith("onnx/") for name in done)  # other frameworks' weights are skipped
        assert done["model.safetensors"]["digest"].startswith("sha256:")
        for name, ln in done.items():
            path = try_to_load_from_cache(TINY_REPO, name)
            assert isinstance(path, str) and os.path.getsize(path) == ln["total"]

        final = client.post("/api/pull", json={"model": TINY_REPO, "stream": False})
        assert final.status_code == 200 and final.json() == {"status": "success"}
        missing = client.post("/api/pull", json={"model": "hf-internal-testing/no-such-repo-turing-test"})
        assert missing.status_code == 404 and "error" in missing.json()
