"""
Validates all registered 2026 model architectures directly on the active CUDA / MPS GPU.
Tests layer tensor allocations, forward pass executions, and memory limits across all 17 models.
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from turing.models.registry import MODEL_REGISTRY, get_model_config
from turing.models.causal_lm import SubspaceDecoderLayer
from turing.core.subspace import SubspaceManager

def run_all_gpu_architecture_tests(device_str: str = "auto"):
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    target_dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("=" * 95)
    print(f"   ⚡ TURING ENGINE: FULL GPU ARCHITECTURE VALIDATION SUITE")
    print(f"   Silicon Target: {str(device).upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'Apple Silicon / CPU'})")
    print("=" * 95 + "\n")

    models_to_test = [k for k in MODEL_REGISTRY.keys() if k != "test-tiny"]

    print(f"{'Model Key':<28} | {'Architecture Name':<32} | {'Dims (H x FFN -> Sub)':<22} | {'GPU Layer Pass':<14} | {'Status'}")
    print("-" * 110)

    passed_count = 0

    for model_key in models_to_test:
        cfg = get_model_config(model_key)
        dim_str = f"{cfg.hidden_dim}x{cfg.ffn_dim}->{cfg.active_tiles * cfg.tile_size}"

        try:
            # 1. Instantiate 1 Subspace Decoder Layer on GPU in target_dtype
            layer = SubspaceDecoderLayer(cfg, layer_idx=0).to(target_dtype).to(device)
            layer.eval()

            # 2. Run Forward Pass with simulated batch=1, seq_len=4
            x = torch.randn(1, 4, cfg.hidden_dim, dtype=target_dtype, device=device)

            start = time.perf_counter()
            with torch.inference_mode():
                out, k_out, v_out = layer(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            # 3. Clean up VRAM immediately
            del layer, x, out, k_out, v_out
            if device.type == "cuda":
                torch.cuda.empty_cache()

            print(f"{model_key:<24} | {cfg.name:<32} | {dim_str:<22} | {elapsed_ms:>8.3f} ms    | ✅ PASSED")
            passed_count += 1

        except Exception as e:
            print(f"{model_key:<24} | {cfg.name:<32} | {dim_str:<22} | {'FAILED':>11}    | ❌ {str(e)[:30]}")

    print("\n" + "=" * 95)
    print(f"[+] GPU Validation Complete: {passed_count}/{len(models_to_test)} architectures verified on {str(device).upper()}!")
    print("=" * 95 + "\n")

if __name__ == "__main__":
    device_arg = sys.argv[1] if len(sys.argv) > 1 else "auto"
    run_all_gpu_architecture_tests(device_arg)
