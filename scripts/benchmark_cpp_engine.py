"""
Decode throughput of the standalone C++ engine (turing-cli generate, greedy) on a synthetic F16 GGUF.

  python scripts/benchmark_cpp_engine.py --binary base=/tmp/turing-cli-base --binary current=build/turing-cli \
      --out results_cpp_engine_mps.json

Each binary runs `generate` --runs times per KV cache dtype (int8 only where the binary has --kv-cache-dtype);
tok/s = new tokens / wall time of generate (prefill included), as turing-cli itself prints it.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import tempfile

import numpy as np

from turing.models.gguf_loader import GGMLType, create_test_gguf_file

SHAPE = dict(hidden_dim=1024, ffn_dim=2816, num_layers=8, num_heads=16, num_kv_heads=16, vocab_size=8192,
             max_position_embeddings=2048)


def run(binary, gguf, prompt, new_tokens, kv):
    cmd = [binary, "generate", "--model", gguf, "--prompt", prompt, "--max-tokens", str(new_tokens), "--temp", "0"]
    if kv != "native":
        cmd += ["--kv-cache-dtype", kv]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    line = res.stderr.strip().splitlines()[-1]  # "(Generated N tokens in X ms — Y tok/s)"
    n = int(line.split("Generated ")[1].split(" tokens")[0])
    ms = float(line.split(" tokens in ")[1].split("ms")[0])
    return n, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", action="append", required=True, help="label=path (repeatable)")
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default="results_cpp_engine_mps.json")
    a = ap.parse_args()

    np.random.seed(0)
    gguf = os.path.join(tempfile.mkdtemp(), "bench.gguf")
    create_test_gguf_file(gguf, quant_type=GGMLType.F16, **SHAPE)
    # Token 1 first: the base binary always prepended BOS id 1, the current one only when the GGUF names one.
    prompt = " ".join(f"<tok_{i}>" for i in [1] + list(range(10, 26)))

    load_start = os.getloadavg()[0]
    rows = []
    for label, path in (b.split("=", 1) for b in a.binary):
        has_int8 = "--kv-cache-dtype" in subprocess.run([path], capture_output=True, text=True).stdout
        for kv in ("native", "int8") if has_int8 else ("native",):
            samples = [run(path, gguf, prompt, a.new_tokens, kv) for _ in range(a.runs)]
            tok_s = [n / (ms / 1000.0) for n, ms in samples]
            row = {"binary": label, "kv_cache_dtype": kv, "prompt_tokens": 17, "new_tokens": samples[0][0],
                   "n": len(tok_s), "tok_s_median": statistics.median(tok_s), "tok_s_mean": statistics.mean(tok_s),
                   "tok_s_se": statistics.stdev(tok_s) / len(tok_s) ** 0.5 if len(tok_s) > 1 else 0.0}
            rows.append(row)
            print(json.dumps(row))

    out = {
        "script": "scripts/benchmark_cpp_engine.py",
        "model": {**SHAPE, "dtype": "F16 GGUF (synthetic, seed 0), fp32 weights in the engine"},
        "env": {"machine": platform.machine(), "platform": platform.platform(), "processor": platform.processor(),
                "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                      text=True).stdout.strip() if platform.system() == "Darwin" else platform.processor(),
                "compiler_flags": "-std=c++20 -O3 -flto -fvisibility=hidden (+ -mavx2 -mfma -mf16c on x86_64)",
                "threads": 1, "load_average_start": load_start, "load_average_end": os.getloadavg()[0]},
        "rows": rows,
        "note": "tok/s = new tokens / wall time of generate (prefill of 17 tokens included), greedy; se over n runs.",
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
