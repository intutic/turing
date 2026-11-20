"""
TensorSerializer round trips (fp16 and int8).
"""

import pytest
import torch
from turing.experimental.hybrid_mesh import TensorSerializer

def test_tensor_serializer_fp16_and_int8():
    dev = torch.device("cpu")
    original_tensor = torch.randn(1, 16, 2048, device=dev)

    # 1. Test FP16 serialization
    payload_fp16 = TensorSerializer.serialize(original_tensor, compress_int8=False)
    deserialized_fp16 = TensorSerializer.deserialize(payload_fp16[4:], device=dev)

    assert deserialized_fp16.shape == original_tensor.shape
    fp16_err = torch.norm(original_tensor - deserialized_fp16).item() / torch.norm(original_tensor).item()
    assert fp16_err < 1e-3

    # 2. Test Dynamic INT8 Quantized serialization
    payload_int8 = TensorSerializer.serialize(original_tensor, compress_int8=True)
    deserialized_int8 = TensorSerializer.deserialize(payload_int8[4:], device=dev)

    assert deserialized_int8.shape == original_tensor.shape
    int8_err = torch.norm(original_tensor - deserialized_int8).item() / torch.norm(original_tensor).item()
    assert int8_err < 0.05 # < 5% quantization noise
    assert len(payload_int8) < len(payload_fp16)
