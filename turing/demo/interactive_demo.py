"""
Interactive Terminal Demo Runner for Turing Engine.
"""

import os
import sys
import json
import time
import argparse
import torch

from .engine_wrapper import TuringAcceleratedGenerator


def run_demo(
    model_name: str = "smollm2",
    device: str = "auto",
    sparsity: float = 0.57,
    prompt: str = "Explain how high-performance subspace inference accelerates LLM token generation in 3 bullet points."
):
    print("=" * 80)
    print("   ⚡ TURING ENGINE: INTERACTIVE SUBSPACE INFERENCE DEMO")
    print("=" * 80 + "\n")

    # 1. Resolve Model Identifier
    model_id_map = {
        "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "smollm": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "gpt2": "gpt2",
        "gpt-2": "gpt2"
    }
    target_model_id = model_id_map.get(model_name.lower().replace("_", "-"), model_name)

    print(f"[*] Initializing Turing Engine:")
    print(f"    • Model Target            : {target_model_id}")
    print(f"    • Active Subspace Pruning : {sparsity * 100:.1f}% FFN Channels Bypassed")
    print(f"    • Compute Device Request  : {device.upper()}")

    # 2. Initialize Engine
    engine = TuringAcceleratedGenerator(
        model_id_or_instance=target_model_id,
        sparsity_ratio=sparsity,
        device=device
    )
    print(f"    • Active Compute Device   : {str(engine.device).upper()}\n")

    # 3. Run Benchmark Prompt
    print(f"[*] Input Prompt:\n    \"{prompt}\"\n")
    print("⚡ Generating response with Turing Engine Subspace Acceleration...\n")

    system_prompt = "You are a helpful, expert AI systems engineer."
    output_text, elapsed_ms = engine.fast_generate(system_prompt, prompt, max_new_tokens=100)

    print("=" * 80)
    print("   GENERATED RESPONSE")
    print("=" * 80)
    print(output_text)
    print("-" * 80)
    print(f"⚡ Generation Time : {elapsed_ms:.2f} ms")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turing Engine Interactive Demo Runner")
    parser.add_argument("--model", type=str, default="smollm2", help="Model target (e.g. smollm2, gpt2)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, mps, cpu)")
    parser.add_argument("--sparsity", type=float, default=0.57, help="Subspace sparsity ratio")
    parser.add_argument("--prompt", type=str, default="Explain how high-performance subspace inference accelerates LLM token generation in 3 bullet points.", help="Prompt to evaluate")
    args = parser.parse_args()

    run_demo(
        model_name=args.model,
        device=args.device,
        sparsity=args.sparsity,
        prompt=args.prompt
    )
