import os
import sys
import yaml
import pytest
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.litellm.turing_engine import TuringEngineConfig

def test_litellm_adapter_config():
    cfg = TuringEngineConfig(
        api_base="http://localhost:8000/v1",
        api_key="sk-test-key",
        speculative_draft_tokens=4
    )
    headers = cfg.map_request_headers()
    assert headers["Authorization"] == "Bearer sk-test-key"
    assert not {"X-Turing-Sparsity", "X-Turing-SVD-KV"} & set(headers)  # the server rejects these with 400
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
    c = data["spec"]["containers"][0]
    assert c["args"][:2] == ["--model", "/mnt/models"]  # where KServe's storage initializer puts storageUri
    assert c["readinessProbe"]["httpGet"]["path"] == "/health"
    assert not c["image"].startswith("ghcr.io/intutic/")  # no image is published there yet
