"""
Unit and Integration Tests for Universal Model Resolver, Architecture Registry, and Reasoning Engine.
"""

import pytest
import torch
from turing.models.resolver import ModelResolver, ResolvedModelSpec
from turing.models.architecture_registry import ArchitectureRegistry, AutoSubspaceModel
from turing.models.registry import get_model_config, MODEL_REGISTRY, SIZING_PROFILES
from turing.config import ModelConfig
from turing.serving.reasoning import ReasoningBudgetManager, ReasoningStreamFilter

def test_model_resolver_canonical_hub_id():
    spec = ModelResolver.parse("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    assert spec.repo_id == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    assert spec.provider == "huggingface"
    assert spec.model_name == "DeepSeek-R1-Distill-Qwen-1.5B"
    assert spec.reasoning_effort is None
    assert not spec.is_local_path
    assert not spec.is_alias

def test_model_resolver_provider_model_effort():
    spec = ModelResolver.parse("deepseek-ai/DeepSeek-R1/high")
    assert spec.repo_id == "deepseek-ai/DeepSeek-R1"
    assert spec.reasoning_effort == "high"
    assert spec.canonical_name == "deepseek-ai/DeepSeek-R1:high"

def test_model_resolver_colon_effort():
    spec = ModelResolver.parse("meta-llama/Llama-3.3-70B-Instruct:low")
    assert spec.repo_id == "meta-llama/Llama-3.3-70B-Instruct"
    assert spec.reasoning_effort == "low"

def test_model_resolver_litellm_prefix():
    spec = ModelResolver.parse("huggingface/meta-llama/Llama-3.1-8B-Instruct")
    assert spec.provider == "huggingface"
    assert spec.repo_id == "meta-llama/Llama-3.1-8B-Instruct"

def test_model_resolver_cli_alias():
    spec = ModelResolver.parse("deepseek-r1-1.5b")
    assert spec.repo_id == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    assert spec.is_alias

def test_model_resolver_local_path():
    spec = ModelResolver.parse("/tmp/my_local_model")
    assert spec.provider == "local"
    assert spec.is_local_path
    assert spec.repo_id == "/tmp/my_local_model"

def test_architecture_registry_dispatch():
    cls_llama = ArchitectureRegistry.get_model_class("LlamaForCausalLM")
    assert cls_llama is not None
    cls_qwen = ArchitectureRegistry.get_model_class("qwen2")
    assert cls_qwen is not None
    cls_deepseek = ArchitectureRegistry.get_model_class("deepseek_v3")
    assert cls_deepseek is not None

def test_get_model_config_offline_sizing():
    cfg = get_model_config("llama-3.1-70b")
    assert cfg.hidden_dim == 8192
    assert cfg.num_heads == 64
    assert cfg.num_layers == 80

def test_get_model_config_dynamic_hf():
    cfg = get_model_config("gpt2")
    assert cfg.vocab_size == 50257
    assert cfg.num_layers == 12

def test_reasoning_budget_manager():
    assert ReasoningBudgetManager.get_max_tokens("low") == 1024
    assert ReasoningBudgetManager.get_max_tokens("medium") == 4096
    assert ReasoningBudgetManager.get_max_tokens("high") == 16384
    assert ReasoningBudgetManager.get_temperature("high") == 0.6

def test_reasoning_stream_filter():
    rf = ReasoningStreamFilter()
    r1, c1 = rf.process_token("Hello! <think>Let me calculate")
    assert r1 is None and c1 == "Hello! "
    
    r2, c2 = rf.process_token(" 2+2=4</think> The answer is 4.")
    assert "2+2=4" in r2
    assert "The answer is 4." in c2
