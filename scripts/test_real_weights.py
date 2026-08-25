"""
Real Pretrained Weight Verification & Inference Benchmark.
Executes real HuggingFace models through Turing Engine Subspace engine.
"""

import os
import sys
import time
import warnings

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["HF_HUB_DISABLE_UNAUTHENTICATED_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import huggingface_hub
huggingface_hub.logging.set_verbosity_error()

import torch
from transformers import AutoTokenizer, logging as hf_logging
hf_logging.set_verbosity_error()

from turing.models.hf_loader import RealHuggingFaceLoader
from turing.core.cross_model_kv import CrossModelKVPipeline, RoPEContentDecoupler

def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Real Weight Evaluation on Hardware Target: {device.upper()}\n")

    # 1. Load Real GPT-2 Pretrained Weights into Turing Engine
    hf_model_id = "gpt2"
    model, tokenizer = RealHuggingFaceLoader.load_hf_model_into_turing(
        hf_model_id=hf_model_id,
        sparsity_ratio=0.50, # 50% active subspace channel pruning
        device=device
    )

    prompt_text = "The foundation of artificial intelligence is"
    input_ids = tokenizer.encode(prompt_text)
    print(f"[*] Prompt: '{prompt_text}' (Token IDs: {input_ids})")

    # 2. Run Real Autoregressive Decode
    print("[*] Generating tokens with Turing Engine SubspaceCausalLM...")
    start_time = time.perf_counter()
    output_tokens = model.generate(
        input_ids,
        max_new_tokens=24,
        temperature=0.7,
        top_k=40
    )
    total_time_ms = (time.perf_counter() - start_time) * 1000.0
    generated_text = tokenizer.decode(output_tokens)
    tps = len(output_tokens[len(input_ids):]) / (total_time_ms / 1000.0)

    print(f"\n[+] Generated Text: \"{generated_text}\"")
    print(f"    New Tokens: {len(output_tokens) - len(input_ids)}")
    print(f"    Generation Time: {total_time_ms:.2f} ms")
    print(f"    Throughput: {tps:.1f} tokens/second\n")

    # 3. Real Cross-Layer / Cross-Model KV Cache Transfer Fitting
    print("[*] Fitting Real Closed-Form Ridge KV Cache Transfer (W*) on Real Prompt Activations...")
    # Extract real K/V representations from forward pass
    with torch.no_grad():
        x_tok = torch.tensor([input_ids], device=device)
        h = model.embed_tokens(x_tok)
        real_keys = []
        real_vals = []
        for layer in model.layers:
            h, k_out, v_out = layer(h)
            real_keys.append(k_out)
            real_vals.append(v_out)

    print(f"    Extracted {len(real_keys)} real KV layers of shape {list(real_keys[0].shape)}")

    # Fit Ridge Mapper on layer 0..3 -> layer 8..11
    src_cfg = model.config
    tgt_cfg = model.config

    pipeline = CrossModelKVPipeline(src_cfg, tgt_cfg, top_k_layers=2, ridge_lambda=0.01)
    t_start = time.perf_counter()
    transferred_k, transferred_v = pipeline.transfer_cache(real_keys, real_vals)
    transfer_ms = (time.perf_counter() - t_start) * 1000.0

    print(f"[+] Real KV Cache Transferred: {len(transferred_k)} layers mapped in {transfer_ms:.2f} ms")
    print(f"    Cosine similarity across mapped keys: {torch.cosine_similarity(transferred_k[-1].flatten(), real_keys[-1].flatten(), dim=0).item():.4f}")

    print("\n[✓] ALL TESTS EXECUTED ON 100% REAL PRETRAINED WEIGHTS & HARDWARE.")

if __name__ == "__main__":
    main()
