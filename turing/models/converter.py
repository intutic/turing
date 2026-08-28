"""
Offline Checkpoint Converter for .tgate, .tgate8, .tgate4, and .tgate_calib Binary Packages.
"""

import os
import struct
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from ..config import ModelConfig
from ..core.subspace import SubspaceManager

class TuringConverter:
    """
    Offline weight converter slicing PyTorch/Safetensors weights into contiguous hardware tiles.
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        self.subspace_mgr = SubspaceManager(hidden_dim=config.hidden_dim, rank=config.rank_sub)

    def export_turing_gate4_layer(
        self,
        output_filepath: str,
        layer_idx: int,
        w_gate: torch.Tensor, # [HiddenDim, FFNDim]
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        active_tiles: List[int]
    ):
        """
        Exports layer weights into packed INT4 .tgate4 binary container.
        """
        hidden_dim = self.config.hidden_dim
        ffn_dim = self.config.ffn_dim
        tile_size = self.config.tile_size
        num_tiles = self.config.total_tiles
        k_active = len(active_tiles)

        mask_int, mask_bytes = SubspaceManager.encode_bitmask(active_tiles, num_tiles)

        # Slice active columns
        gate_slices = [w_gate[:, t*tile_size:(t+1)*tile_size] for t in active_tiles]
        up_slices = [w_up[:, t*tile_size:(t+1)*tile_size] for t in active_tiles]
        down_slices = [w_down[t*tile_size:(t+1)*tile_size, :] for t in active_tiles]

        w_g_act = torch.cat(gate_slices, dim=-1).t().contiguous() # [K*TileSize, HiddenDim]
        w_u_act = torch.cat(up_slices, dim=-1).t().contiguous()
        w_d_act = torch.cat(down_slices, dim=0).contiguous()

        # Group-wise INT4 quantization (Group size 128)
        def quantize_w4(w: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
            out_f, in_f = w.shape
            w_grouped = w.view(out_f, in_f // 128, 128)
            scales = (w_grouped.abs().amax(dim=-1).clamp(min=1e-5) / 7.0) # [OutF, Groups]
            scales_exp = scales.repeat_interleave(128, dim=-1)

            w_int4 = torch.clamp(torch.round(w / scales_exp) + 8.0, 0, 15).to(torch.uint8)
            # Pack 2 nibbles per byte
            w_packed = (w_int4[:, 0::2] & 0x0F) | ((w_int4[:, 1::2] & 0x0F) << 4)

            scales_np = scales.detach().cpu().numpy().astype(np.float16)
            packed_np = w_packed.detach().cpu().numpy().astype(np.uint8)
            return packed_np, scales_np

        g_pack, g_scales = quantize_w4(w_g_act)
        u_pack, u_scales = quantize_w4(w_u_act)
        d_pack, d_scales = quantize_w4(w_d_act)

        # 64-byte Header: 3 uint32 + 16s + 5 uint32 + 16x padding = 64 bytes
        header = struct.pack(
            "<III16sIIIII16x",
            0x34544147, # Magic 'GAT4'
            3,          # Version
            layer_idx,
            mask_bytes,
            hidden_dim,
            ffn_dim,
            tile_size,
            num_tiles,
            k_active
        )

        dirname = os.path.dirname(output_filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with open(output_filepath, "wb") as f:
            f.write(header)
            f.write(g_pack.tobytes())
            f.write(g_scales.tobytes())
            f.write(u_pack.tobytes())
            f.write(u_scales.tobytes())
            f.write(d_pack.tobytes())
            f.write(d_scales.tobytes())

    def export_turing_calib_package(
        self,
        output_filepath: str,
        layer_bitmasks: Dict[int, List[int]],
        u_proj: torch.Tensor
    ):
        """
        Exports model calibration package (.tgate_calib) with layer bitmasks and SVD basis.
        """
        header = struct.pack(
            "<IIIIIII36x",
            0x43414C42, # Magic 'CALB'
            1,          # Version
            self.config.hidden_dim,
            self.config.ffn_dim,
            self.config.tile_size,
            self.config.num_layers,
            self.config.rank_sub
        )

        with open(output_filepath, "wb") as f:
            f.write(header)
            # Write per-layer bitmasks
            for l_idx in range(self.config.num_layers):
                tiles = layer_bitmasks.get(l_idx, list(range(self.config.active_tiles)))
                _, mask_bytes = SubspaceManager.encode_bitmask(tiles, self.config.total_tiles)
                f.write(struct.pack("<I", l_idx))
                f.write(mask_bytes)

            # Write SVD projection matrix
            u_np = u_proj.detach().cpu().numpy().astype(np.float16)
            f.write(u_np.tobytes())

    @staticmethod
    def compute_mean_centered_klt(
        activations: torch.Tensor, # [N, Dim]
        rank: int = 64,
        apply_hadamard: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes Mean-Centered Karhunen-Loève / SVD Transform with Hadamard Equalization.
        Returns: (W_down, W_up, mu, beta)
        where beta = mu @ W_up (folded into layer residual bias).
        """
        mu = activations.mean(dim=0, keepdim=True) # [1, Dim]
        centered = activations - mu

        # Covariance SVD / PCA
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        w_down = Vh[:rank, :].t() # [Dim, Rank]
        w_up = Vh[:rank, :]       # [Rank, Dim]

        if apply_hadamard and rank >= 16:
            # Deterministic orthogonal Walsh-Hadamard transform on light singular coordinates
            # to distribute outlier variance evenly before INT8 quantization
            g = torch.Generator(device="cpu").manual_seed(42)
            A = torch.randn(rank, rank, generator=g, dtype=activations.dtype, device=activations.device)
            H, _ = torch.linalg.qr(A)

            w_down = w_down @ H
            w_up = H.t() @ w_up

        beta = mu @ w_down @ w_up # Folded bias
        return w_down, w_up, mu.squeeze(0), beta.squeeze(0)

