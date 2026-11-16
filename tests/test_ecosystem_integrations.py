import os
import sys
import yaml
import pytest
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.litellm.turing_engine import TuringEngineConfig
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

def test_kserve_manifest_validity():
    manifest_path = os.path.join(os.path.dirname(__file__), "..", "integrations", "kserve", "serving_runtime.yaml")
    with open(manifest_path, "r") as f:
        data = yaml.safe_load(f)
    assert data["apiVersion"] == "serving.kserve.io/v1alpha1"
    assert data["kind"] == "ServingRuntime"
    assert data["metadata"]["name"] == "turing-runtime"
    assert len(data["spec"]["containers"]) >= 1

def test_ticket_generator_dry_run(capsys):
    create_tickets(dry_run=True)
    captured = capsys.readouterr()
    assert "TURING ENGINE ECOSYSTEM TICKET GENERATOR" in captured.out
    assert "DRY RUN" in captured.out
    assert len(TICKETS) == 8
