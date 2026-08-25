import os
import sys
import yaml
import pytest
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.litellm.turing_engine import TuringEngineConfig
from integrations.langchain.chat_turing import ChatTuringEngine
from integrations.llamaindex.turing_llm import TuringEngineLLM
from integrations.runpod.runpod_handler import RunPodTuringWorker
from scripts.create_ecosystem_tickets import TICKETS, create_tickets

def test_litellm_adapter_config():
    cfg = TuringEngineConfig(
        api_base="http://localhost:8000/v1",
        api_key="sk-test-key",
        sparsity_ratio=0.57,
        use_svd_kv_cache=True,
        speculative_draft_tokens=4
    )
    headers = cfg.map_request_headers()
    assert headers["Authorization"] == "Bearer sk-test-key"
    assert headers["X-Turing-Sparsity"] == "0.57"
    assert headers["X-Turing-SVD-KV"] == "1"
    assert headers["X-Turing-Draft-Tokens"] == "4"
    assert cfg.transform_model_name("turing/llama-3.1-70b") == "llama-3.1-70b"
    assert cfg.transform_model_name("gpt-4o") == "gpt-4o"

def test_langchain_adapter_message_formatting():
    llm = ChatTuringEngine(model="llama-3.1-70b", base_url="http://localhost:8000/v1")
    raw_msgs = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Hello!"}
    ]
    formatted = llm._format_messages(raw_msgs)
    assert len(formatted) == 2
    assert formatted[0]["role"] == "system"
    assert formatted[1]["content"] == "Hello!"

def test_llamaindex_adapter_init():
    llm = TuringEngineLLM(model="qwen-2.5-72b", api_base="http://localhost:8000/v1")
    assert llm.model == "qwen-2.5-72b"
    assert llm.api_base == "http://localhost:8000/v1"

def test_kserve_manifest_validity():
    manifest_path = os.path.join(os.path.dirname(__file__), "..", "integrations", "kserve", "serving_runtime.yaml")
    with open(manifest_path, "r") as f:
        data = yaml.safe_load(f)
    assert data["apiVersion"] == "serving.kserve.io/v1alpha1"
    assert data["kind"] == "ServingRuntime"
    assert data["metadata"]["name"] == "turing-runtime"
    assert len(data["spec"]["containers"]) >= 1

def test_runpod_worker_handler():
    worker = RunPodTuringWorker(model_key="test-tiny")
    res = worker.handler({"input": {"prompt": "Test prompt", "max_tokens": 16}})
    assert "text" in res
    assert "completion_tokens" in res
    assert res["completion_tokens"] > 0

def test_ticket_generator_dry_run(capsys):
    create_tickets(dry_run=True)
    captured = capsys.readouterr()
    assert "TURING ENGINE ECOSYSTEM TICKET GENERATOR" in captured.out
    assert "DRY RUN" in captured.out
    assert len(TICKETS) == 8
