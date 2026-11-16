"""
Binary tensor serializer (fp16, or symmetric per-tensor int8) for sending tensors between processes or machines.
"""

import struct
import numpy as np
import torch

try:
    import turing.turing_csrc as turing_csrc
    HAS_CSRC = True
except ImportError:
    try:
        import turing_csrc
        HAS_CSRC = True
    except ImportError:
        HAS_CSRC = False

class TensorSerializer:
    """
    serialize() returns a length-prefixed payload; deserialize() takes it with or without the 4-byte prefix.
    """
    @staticmethod
    def serialize(tensor: torch.Tensor, compress_int8: bool = False) -> bytes:
        tensor_cpu = tensor.detach().cpu().contiguous()
        shape = list(tensor_cpu.shape)
        ndim = len(shape)

        if compress_int8:
            if HAS_CSRC:
                return turing_csrc.serialize_tensor_int8(tensor_cpu.to(torch.float32).numpy())
            # Dynamic symmetric INT8 quantization: x_int8 = round(x / scale)
            scale = (tensor_cpu.abs().max() / 127.0).item()
            scale = max(1e-8, scale)
            quantized = torch.clamp(torch.round(tensor_cpu / scale), -128, 127).to(torch.int8)
            data_bytes = quantized.numpy().tobytes()
            dtype_code = 1 # INT8
            header = struct.pack(f"<BfI{ndim}I", dtype_code, scale, ndim, *shape)
        else:
            fp16_tensor = tensor_cpu.to(torch.float16)
            data_bytes = fp16_tensor.numpy().tobytes()
            dtype_code = 2 # FP16
            scale = 1.0
            header = struct.pack(f"<BfI{ndim}I", dtype_code, scale, ndim, *shape)

        payload_len = len(header) + len(data_bytes)
        return struct.pack("<I", payload_len) + header + data_bytes

    @staticmethod
    def deserialize(data: bytes, device: torch.device) -> torch.Tensor:
        if len(data) >= 4:
            potential_len = struct.unpack_from("<I", data, 0)[0]
            if potential_len == len(data) - 4:
                offset = 4
            else:
                offset = 0
        else:
            offset = 0

        dtype_code, scale, ndim = struct.unpack_from("<BfI", data, offset)
        if dtype_code == 1 and HAS_CSRC:
            arr_np = turing_csrc.deserialize_tensor_int8(data[offset:])
            return torch.from_numpy(arr_np).to(device=device, dtype=torch.float32)

        shape = struct.unpack_from(f"<{ndim}I", data, offset + 9)
        header_len = offset + 9 + 4 * ndim
        raw_bytes = data[header_len:]

        if dtype_code == 1: # INT8
            arr = np.frombuffer(raw_bytes, dtype=np.int8).reshape(shape)
            tensor = torch.from_numpy(arr.copy()).to(torch.float32) * scale
        else: # FP16
            arr = np.frombuffer(raw_bytes, dtype=np.float16).reshape(shape)
            tensor = torch.from_numpy(arr.copy()).to(torch.float32)

        return tensor.to(device=device)
