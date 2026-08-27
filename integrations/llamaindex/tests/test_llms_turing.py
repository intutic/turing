import pytest
from turing.integrations.llamaindex import Turing, TuringEngine, TuringEngineLLM

def test_turing_llm_initialization():
    llm = Turing(
        model="deepseek-r1-7b",
        api_base="http://localhost:8000/v1",
        api_key="sk-test",
        temperature=0.5,
        max_tokens=256,
        sparsity_ratio=0.57,
        svd_rank=64,
        context_window=65536
    )
    assert llm.model == "deepseek-r1-7b"
    assert llm.api_base == "http://localhost:8000/v1"
    assert llm.temperature == 0.5
    assert llm.max_tokens == 256
    assert llm.sparsity_ratio == 0.57
    assert llm.svd_rank == 64
    assert llm.context_window == 65536
    assert llm.class_name() == "Turing"

def test_turing_llm_metadata():
    llm = Turing(model="qwen-2.5-72b", max_tokens=1024, context_window=131072)
    meta = llm.metadata
    assert meta.model_name == "qwen-2.5-72b"
    assert meta.num_output == 1024
    assert meta.context_window == 131072
    assert meta.is_chat_model is True

def test_turing_llm_headers():
    llm = Turing(api_key="test-auth-key", sparsity_ratio=0.60, svd_rank=32)
    headers = llm._get_headers()
    assert headers["Authorization"] == "Bearer test-auth-key"
    assert headers["X-Turing-Sparsity"] == "0.6"
    assert headers["X-Turing-SVD-Rank"] == "32"

def test_turing_llm_standalone_offline():
    llm = Turing(model="deepseek-r1-1.5b", api_base="http://127.0.0.1:9999/v1")
    # Should gracefully return offline notice without unhandled exception
    res = llm.complete("Hello world")
    assert "Turing Engine Standalone Mode" in res.text
