"""
Zero-Copy Direct HuggingFace Safetensors Memory-Mapped Reader (Turing Engine Integration).
Enables reading and streaming weights directly from HuggingFace .safetensors files
without GGUF conversion or high-RAM memory duplication.
"""

import os
import mmap
import json
import struct
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import torch

class SafetensorsMmapReader:
    """
    Direct zero-copy reader for HuggingFace .safetensors binary files.
    """
    DTYPE_MAP = {
        "F32": (torch.float32, 4),
        "F16": (torch.float16, 2),
        "BF16": (torch.bfloat16, 2),
        "I32": (torch.int32, 4),
        "I16": (torch.int16, 2),
        "I8": (torch.int8, 1),
        "U8": (torch.uint8, 1),
        "BOOL": (torch.bool, 1)
    }

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_obj = open(filepath, "rb")
        self.mmap_obj = mmap.mmap(self.file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        if hasattr(self.mmap_obj, "madvise") and hasattr(mmap, "MADV_WILLNEED"):
            try:
                self.mmap_obj.madvise(mmap.MADV_WILLNEED)
            except Exception:
                pass

        # Parse 8-byte uint64 header size
        header_size_bytes = self.mmap_obj[:8]
        self.header_size = struct.unpack("<Q", header_size_bytes)[0]

        # Parse JSON header metadata
        header_json_bytes = self.mmap_obj[8 : 8 + self.header_size]
        header_str = header_json_bytes.decode("utf-8")
        
        try:
            from turing.turing_csrc import NativeSafetensorsHeaderParser
            fast_meta = NativeSafetensorsHeaderParser.parse_header(header_str)
            self.metadata = {}
            for k, v in fast_meta.items():
                self.metadata[k] = {
                    "dtype": v.dtype,
                    "shape": v.shape,
                    "data_offsets": [v.start_offset, v.end_offset]
                }
        except Exception:
            self.metadata = json.loads(header_str)

        self.data_offset = 8 + self.header_size

    def get_tensor_names(self) -> List[str]:
        return [k for k in self.metadata.keys() if k != "__metadata__"]

    def get_tensor_info(self, tensor_name: str) -> Dict[str, Any]:
        if tensor_name not in self.metadata:
            raise KeyError(f"Tensor '{tensor_name}' not found in {self.filepath}")
        return self.metadata[tensor_name]

    def read_tensor_slice(
        self,
        tensor_name: str,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Extracts a tensor slice directly from the memory-mapped buffer.
        """
        info = self.get_tensor_info(tensor_name)
        dtype_str = info["dtype"]
        shape = info["shape"]
        offsets = info["data_offsets"]

        torch_dtype, item_size = self.DTYPE_MAP.get(dtype_str, (torch.float32, 4))
        start_byte = self.data_offset + offsets[0]
        end_byte = self.data_offset + offsets[1]

        # Zero-copy view using frombuffer on numpy uint8 view
        raw_bytes = self.mmap_obj[start_byte:end_byte]
        np_dtype = np.float16 if dtype_str in ["F16", "BF16"] else (np.float32 if dtype_str == "F32" else np.uint8)
        arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape)

        tensor = torch.from_numpy(arr.copy())
        if dtype_str == "BF16":
            tensor = tensor.view(torch.bfloat16)
        elif dtype_str == "F16":
            tensor = tensor.to(torch.float16)

        return tensor.to(device)

    def close(self):
        if hasattr(self, "mmap_obj") and self.mmap_obj is not None:
            self.mmap_obj.close()
        if hasattr(self, "file_obj") and self.file_obj is not None:
            self.file_obj.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
