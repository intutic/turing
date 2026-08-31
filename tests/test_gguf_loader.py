"""
Comprehensive Unit & Integration Test Suite for Turing Engine Native GGUF Loader.
Verifies binary parsing, metadata extraction, Q4_0 / Q8_0 / FP16 dequantization,
tokenizer extraction, and end-to-end SubspaceCausalLM forward execution.
"""

import os
import tempfile
import pytest
import numpy as np
import torch

from turing.models.gguf_loader import (
    GGUFReader,
    GGUFHeader,
    GGUFTensorInfo,
    GGUFDequantizer,
    GGUFModelLoader,
    GGMLType,
    GGUFValueType,
    GGUF_MAGIC,
    create_test_gguf_file
)
from turing.models.gguf_tokenizer import GGUFTokenizer
from turing.models.resolver import ModelResolver
from turing.config import ModelConfig
from turing.models.causal_lm import SubspaceCausalLM


@pytest.fixture(scope="module")
def temp_gguf_f16():
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        path = f.name
    create_test_gguf_file(
        path,
        architecture="llama",
        hidden_dim=64,
        ffn_dim=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        vocab_size=128,
        quant_type=GGMLType.F16
    )
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture(scope="module")
def temp_gguf_q8():
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        path = f.name
    create_test_gguf_file(
        path,
        architecture="llama",
        hidden_dim=64,
        ffn_dim=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        vocab_size=128,
        quant_type=GGMLType.Q8_0
    )
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_gguf_header_and_metadata(temp_gguf_f16):
    with GGUFReader(temp_gguf_f16) as reader:
        assert reader.header is not None
        assert reader.header.magic == GGUF_MAGIC
        assert reader.header.version == 3
        assert reader.header.tensor_count > 0
        assert reader.header.kv_count > 0
        
        # Verify metadata keys
        assert reader.metadata["general.architecture"] == "llama"
        assert reader.metadata["llama.embedding_length"] == 64
        assert reader.metadata["llama.block_count"] == 2
        assert reader.metadata["llama.attention.head_count"] == 4
        assert "tokenizer.ggml.tokens" in reader.metadata
        assert len(reader.metadata["tokenizer.ggml.tokens"]) == 128


def test_invalid_magic_raises(tmp_path):
    corrupt_path = str(tmp_path / "corrupt.gguf")
    with open(corrupt_path, "wb") as f:
        f.write(b"BAD_MAGIC_BYTES_1234")
    
    with pytest.raises(ValueError, match="Invalid GGUF magic bytes"):
        GGUFReader(corrupt_path)


def test_dequantizer_fp32_fp16():
    raw_f32 = np.array([1.0, 2.5, -3.0, 0.0], dtype=np.float32).tobytes()
    t_f32 = GGUFDequantizer.dequantize(raw_f32, GGMLType.F32, [4], target_dtype=torch.float32)
    assert t_f32.shape == (4,)
    assert torch.allclose(t_f32, torch.tensor([1.0, 2.5, -3.0, 0.0]))

    raw_f16 = np.array([1.5, -2.0], dtype=np.float16).tobytes()
    t_f16 = GGUFDequantizer.dequantize(raw_f16, GGMLType.F16, [2], target_dtype=torch.float16)
    assert t_f16.shape == (2,)
    assert torch.allclose(t_f16, torch.tensor([1.5, -2.0], dtype=torch.float16))


def test_dequantizer_q8_0():
    # 32 elements: delta = 0.5 (fp16), quants = [0, 1, 2, ..., 31]
    delta = np.float16(0.5).tobytes()
    quants = np.arange(32, dtype=np.int8).tobytes()
    block = delta + quants
    
    t_q8 = GGUFDequantizer.dequantize(block, GGMLType.Q8_0, [32], target_dtype=torch.float32)
    assert t_q8.shape == (32,)
    expected = torch.arange(32, dtype=torch.float32) * 0.5
    assert torch.allclose(t_q8, expected, atol=1e-3)


def test_dequantizer_q4_0():
    # 32 elements: delta = 1.0 (fp16), 16 bytes containing nibbles
    delta = np.float16(1.0).tobytes()
    # Nibbles with value 8 (which dequantizes to (8-8)*1.0 = 0.0)
    nibbles = np.full(16, 0x88, dtype=np.uint8).tobytes()
    block = delta + nibbles
    
    t_q4 = GGUFDequantizer.dequantize(block, GGMLType.Q4_0, [32], target_dtype=torch.float32)
    assert t_q4.shape == (32,)
    assert torch.allclose(t_q4, torch.zeros(32, dtype=torch.float32))


def test_gguf_tokenizer_encode_decode(temp_gguf_f16):
    with GGUFReader(temp_gguf_f16) as reader:
        tokenizer = GGUFTokenizer(reader.metadata)
        assert tokenizer.vocab_size == 128
        
        # Test encode with direct match
        tok_id = tokenizer.encode("<tok_5>")
        assert 5 in tok_id
        
        # Test decode
        text = tokenizer.decode([5, 6, 7])
        assert "<tok_5>" in text
        assert "<tok_6>" in text
        
        # Test apply_chat_template
        chat = [{"role": "user", "content": "hello"}]
        tmpl = tokenizer.apply_chat_template(chat, tokenize=False)
        assert "<|im_start|>user" in tmpl
        assert "hello" in tmpl
        assert "<|im_start|>assistant" in tmpl


def test_gguf_model_loader_e2e(temp_gguf_f16):
    loader = GGUFModelLoader(temp_gguf_f16)
    model, tokenizer = loader.load(device="cpu", dtype=torch.float32)
    
    assert isinstance(model, SubspaceCausalLM)
    assert model.config.hidden_dim == 64
    assert model.config.num_layers == 2
    assert model.config.num_heads == 4
    
    # Test forward pass
    input_ids = torch.tensor([[1, 5, 10, 20]], dtype=torch.long)
    logits, kv = model(input_ids)
    assert logits.shape == (1, 4, 128)
    assert kv is not None
    assert len(kv) == 2  # 2 layers
    
    # Test autoregressive generate
    prompt_tokens = [1, 5, 10]
    out_tokens = model.generate(prompt_tokens, max_new_tokens=4, temperature=0.7)
    assert len(out_tokens) == 7
    assert out_tokens[:3] == prompt_tokens


def test_gguf_model_loader_q8_e2e(temp_gguf_q8):
    loader = GGUFModelLoader(temp_gguf_q8)
    model, tokenizer = loader.load(device="cpu", dtype=torch.float32)
    
    assert isinstance(model, SubspaceCausalLM)
    input_ids = torch.tensor([[2, 4, 8]], dtype=torch.long)
    logits, _ = model(input_ids)
    assert logits.shape == (1, 3, 128)
    assert not torch.isnan(logits).any()


def test_model_resolver_gguf_path(temp_gguf_f16):
    spec = ModelResolver.parse(temp_gguf_f16)
    assert spec.provider == "gguf"
    assert spec.is_local_path is True
    assert spec.repo_id == temp_gguf_f16


def test_model_config_from_pretrained_gguf(temp_gguf_f16):
    cfg = ModelConfig.from_pretrained(temp_gguf_f16)
    assert cfg.hidden_dim == 64
    assert cfg.ffn_dim == 128
    assert cfg.num_layers == 2
    assert cfg.num_heads == 4
