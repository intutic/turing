"""
Unit tests for expanded model aliases and interactive chat CLI.
"""

import pytest
from turing.models.hf_loader import RealHuggingFaceLoader

def test_expanded_model_aliases():
    # Test Qwen aliases
    assert RealHuggingFaceLoader.resolve_model_id("qwen-2.5-7b") == "Qwen/Qwen2.5-7B-Instruct"
    assert RealHuggingFaceLoader.resolve_model_id("qwen-7b") == "Qwen/Qwen2.5-7B-Instruct"
    assert RealHuggingFaceLoader.resolve_model_id("qwen-2.5-14b") == "Qwen/Qwen2.5-14B-Instruct"
    assert RealHuggingFaceLoader.resolve_model_id("qwen-32b") == "Qwen/Qwen2.5-32B-Instruct"
    assert RealHuggingFaceLoader.resolve_model_id("qwen-coder-32b") == "Qwen/Qwen2.5-Coder-32B-Instruct"
    assert RealHuggingFaceLoader.resolve_model_id("qwen-72b") == "Qwen/Qwen2.5-72B-Instruct"

    # Test Gemma & LLaMA 3.3 aliases
    assert RealHuggingFaceLoader.resolve_model_id("gemma-2-27b") == "google/gemma-2-27b-it"
    assert RealHuggingFaceLoader.resolve_model_id("llama-3.3-70b") == "meta-llama/Llama-3.3-70B-Instruct"
    assert RealHuggingFaceLoader.resolve_model_id("phi-4") == "microsoft/phi-4"

    # Test DeepSeek-R1 Distill aliases
    assert RealHuggingFaceLoader.resolve_model_id("deepseek-r1-1.5b") == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    assert RealHuggingFaceLoader.resolve_model_id("deepseek-r1-7b") == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    assert RealHuggingFaceLoader.resolve_model_id("deepseek-r1-14b") == "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    assert RealHuggingFaceLoader.resolve_model_id("deepseek-r1-32b") == "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    assert RealHuggingFaceLoader.resolve_model_id("deepseek-r1-70b") == "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

    # Test fallback on raw repo identifier
    assert RealHuggingFaceLoader.resolve_model_id("custom/my-fine-tuned-model") == "custom/my-fine-tuned-model"
