#!/usr/bin/env python3
"""
Exhaustive Master Benchmark Suite: Runs EVERY Documented Benchmark on Real Silicon (Zero Mocks).
Executes across all 15 benchmark suites in scripts/ with live measurements.
"""

import sys
import os
import time
import subprocess
import json
import torch

BENCHMARK_SUITES = [
    # 1. Real Weight GPU Ingestion & Storage Ladder
    ("6-Tier Storage Ingestion Speed Ladder", "scripts/benchmark_ingest_ladder_all_tiers.py", []),
    
    # 2. Fused GPU Kernels & C++20 SIMD Subsystems
    ("Fused GPU Kernels & C++20 SIMD Subsystems", "scripts/benchmark_fused_kernels_and_simd.py", []),
    
    # 3. Real Weight Model Prefill/Decode & 7-Architecture Matrix
    ("Real Model Comprehensive Matrix (7 Architectures)", "scripts/benchmark_comprehensive_real_gpu.py", []),
    
    # 4. Triple Serving Gateway & Microsecond Structured JSON
    ("Triple Serving Gateway & Structured JSON Parser", "scripts/benchmark_triple_gateway_and_structured.py", []),
    
    # 5. Multi-Turn Lineage Deliberation & k-Slot Pooling
    ("Multi-Turn Clean-Base Lineage & Cache Pooling", "scripts/benchmark_lineage_strategies.py", []),
    
    # 6. AI Traffic Management, 3-Lane QoS & Speculation Gating
    ("AI Traffic Management & Adaptive Speculation Gating", "scripts/benchmark_traffic_and_spec.py", []),
    
    # 7. Prefill-Decode Duality & Interleaved Scheduling Jitter
    ("Prefill-Decode Duality & Interleaved Throughput", "scripts/benchmark_prefill_decode_duality.py", ["--device", "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")]),
    
    # 8. Latent Flash-Decode (SPECTRA Subspace Attention)
    ("Latent Flash-Decode & Hybrid Prefill Scaling", "scripts/benchmark_latent_decode.py", ["--device", "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")]),
    
    # 9. Matryoshka Parameter-Sliced Speculation & Semantic Anchors
    ("Matryoshka Sliced Speculation & Semantic Anchors", "scripts/benchmark_freetoken_matryoshka.py", ["--device", "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")]),
    
    # 10. Native C++20 SIMD & Fused GPU Kernel Micro-benchmarks
    ("Native C++20 SIMD & Fused Kernel Micro-benchmarks", "scripts/benchmark_native_fusions_v2.py", ["--device", "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")]),
    
    # 11. GGUF Q8_0 SIMD Dequantization & Fused RMSNorm+SwiGLU
    ("C++20 SIMD Dequantizer & In-SRAM Fused FFN Layers", "scripts/benchmark_fused_layers.py", []),
    
    # 12. Multi-Tenant LoRA Hot-Swapping & Speculative Drafting
    ("Multi-Tenant LoRA Hot-Swapping & Cold Starts", "scripts/benchmark_lora_and_speculation.py", []),
    
    # 13. Long-Context NIAH Breaking Point Analysis
    ("Long-Context NIAH Depth & Rank Breaking-Point", "scripts/stress_test_niah_breaking_point.py", ["cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")]),
    
    # 14. Universal Architecture & Hardware Verification
    ("Universal Architecture & Hardware Verification", "scripts/test_all_gpu_architectures.py", []),
    
    # 15. End-to-End Serving SLA Replay
    ("End-to-End Serving SLA Concurrency Load Test", "scripts/benchmark_serving_e2e.py", [])
]

def main():
    device_type = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ("Apple Silicon Metal (MPS)" if torch.backends.mps.is_available() else "CPU")
    
    print("=" * 90)
    print("   ⚡ TURING ENGINE: EXHAUSTIVE UNMOCKED PHYSICAL BENCHMARK SUITE")
    print("=" * 90)
    print(f"[*] Execution Silicon  : {device_name} ({device_type.upper()})")
    print(f"[*] Total Suites to Run: {len(BENCHMARK_SUITES)}")
    print(f"[*] Timestamp          : {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print("=" * 90)

    summary_results = []
    total_start = time.perf_counter()

    for idx, (suite_name, script_path, extra_args) in enumerate(BENCHMARK_SUITES, 1):
        if not os.path.exists(script_path):
            print(f"[-] Skipping {script_path} (File not found)")
            continue

        cmd = [sys.executable, script_path] + extra_args
        cmd_str = " ".join(cmd)

        print(f"\n[{idx}/{len(BENCHMARK_SUITES)}] 🚀 RUNNING: {suite_name}")
        print(f"    Command: {cmd_str}")
        print("-" * 90)

        t0 = time.perf_counter()
        res = subprocess.run(cmd, capture_output=True, text=True)
        t1 = time.perf_counter()
        dur = t1 - t0

        print(res.stdout)
        if res.stderr and res.returncode != 0:
            print("[!] Stderr:", res.stderr)

        passed = (res.returncode == 0)
        status_icon = "✅ PASSED" if passed else "❌ FAILED"
        print(f"[*] Status: {status_icon} in {dur:.2f}s")

        summary_results.append({
            "index": idx,
            "suite_name": suite_name,
            "script_path": script_path,
            "passed": passed,
            "duration_seconds": dur,
            "stdout": res.stdout,
            "stderr": res.stderr
        })

    total_duration = time.perf_counter() - total_start

    print("\n" + "=" * 90)
    print(f"   📊 EXHAUSTIVE BENCHMARK SUITE EXECUTION SUMMARY ({device_name})")
    print("=" * 90)
    all_passed = True
    for r in summary_results:
        icon = "✅" if r["passed"] else "❌"
        print(f"  {icon} [{r['index']:02d}/{len(BENCHMARK_SUITES):02d}] {r['suite_name']:<60} : {r['duration_seconds']:.2f}s")
        if not r["passed"]:
            all_passed = False

    print("=" * 90)
    print(f"[*] Total Execution Time: {total_duration:.2f}s (~{total_duration/60.0:.1f} minutes)")
    if all_passed:
        print(f"🎉 ALL {len(summary_results)} BENCHMARK SUITES COMPLETED WITH 100% SUCCESS!")
    else:
        print("⚠️ SOME BENCHMARK SUITES REPORTED FAILURES.")
    print("=" * 90)

    # Save summary report JSON
    out_file = f"exhaustive_benchmark_results_{device_type}.json"
    with open(out_file, "w") as f:
        json.dump({
            "device": device_name,
            "device_type": device_type,
            "total_duration_seconds": total_duration,
            "suites": summary_results
        }, f, indent=2)
    print(f"[*] Results exported to: {out_file}")

if __name__ == "__main__":
    main()
