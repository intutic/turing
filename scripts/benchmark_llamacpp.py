#!/usr/bin/env python3
"""
llama.cpp (CUDA build) for the 4-bit comparison with Turing and vLLM on the same GPU.

bench: llama-batched-bench with B parallel sequences of a --context-token prompt and --new-tokens generated tokens;
       rows match scripts/benchmark_e2e_generation.py (output_tok_per_s = generated tokens / total time, prefill
       included) plus llama.cpp's own decode-only rate (s_tg).
gsm8k: llama-server through turing/serving/accuracy_eval.py's GSM8K evaluator (the prompts and scoring Turing and
       vLLM get: first --samples test problems, --shots train shots, greedy, 256 new tokens); prompts go as token ids.

python scripts/benchmark_llamacpp.py bench --gguf qwen2.5-1.5b-instruct-q4_k_m.gguf --bin ~/llama.cpp/build/bin
python scripts/benchmark_llamacpp.py gsm8k --gguf qwen2.5-1.5b-instruct-q4_k_m.gguf --bin ~/llama.cpp/build/bin
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_vllm_quality import Engine, turing_evals  # noqa: E402  (loads the evaluators without the serving engine)


def parse_batched_bench(text):
    """Rows of llama-batched-bench's table: | PP | TG | B | N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s | T s | S t/s |"""
    rows = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 10 and re.fullmatch(r"\d+", cells[0] or ""):
            pp, tg, b, n_kv = map(int, cells[:4])
            t_pp, s_pp, t_tg, s_tg, t, s = map(float, cells[4:])
            rows.append(dict(pp=pp, tg=tg, batch=b, t_pp=t_pp, s_pp=s_pp, t_tg=t_tg, s_tg=s_tg, t=t, s=s))
    return rows


def git_rev(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def save(out, rows, keep):
    old = json.load(open(out))["rows"] if os.path.exists(out) else []
    rows = [r for r in old if not keep(r)] + rows
    with open(out, "w") as f:
        json.dump(dict(env=dict(platform=platform.platform()), rows=rows), f, indent=2)
    print(f"[+] wrote {out}")


def bench(a):
    cmd = [os.path.join(a.bin, "llama-batched-bench"), "-m", a.gguf, "-ngl", "99",
           "-c", str(max(a.batches) * (a.context + a.new_tokens) + 64), "-b", "2048", "-ub", "512",
           "-npp", str(a.context), "-ntg", str(a.new_tokens), "-npl", ",".join(map(str, a.batches))]
    p = subprocess.run(cmd, capture_output=True, text=True)
    print(p.stdout[-3000:], p.stderr[-2000:], sep="\n")
    got = parse_batched_bench(p.stdout)
    if p.returncode or not got:
        raise SystemExit(f"llama-batched-bench failed (exit {p.returncode})")
    rev = git_rev(os.path.dirname(os.path.dirname(os.path.abspath(a.bin))))
    rows = [dict(engine="llama.cpp", model=os.path.basename(a.gguf), batch=r["batch"], context=r["pp"],
                 new_tokens=r["tg"], seconds=r["t"], output_tok_per_s=round(r["batch"] * r["tg"] / r["t"], 1),
                 decode_tok_per_s=r["s_tg"], runtime=f"llama.cpp {rev}") for r in got]
    for r in rows:
        print(json.dumps(r), flush=True)
    name = os.path.basename(a.gguf)
    save(a.out, rows, lambda r: r["engine"] == "llama.cpp" and r["model"] == name and r["context"] == a.context)


def gsm8k(a):
    import requests
    from transformers import AutoTokenizer
    acc_eval, _ = turing_evals()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    srv = subprocess.Popen([os.path.join(a.bin, "llama-server"), "-m", a.gguf, "-ngl", "99", "-c",
                            str(a.parallel * 4096), "-np", str(a.parallel), "--port", str(a.port)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{a.port}"
    try:
        for _ in range(300):
            try:
                if requests.get(url + "/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise SystemExit("llama-server did not become ready")

        def one(ids, n):
            r = requests.post(url + "/completion", json={"prompt": ids, "n_predict": n, "temperature": 0.0,
                                                         "top_k": 1, "cache_prompt": False}, timeout=600)
            r.raise_for_status()
            return tok.encode(r.json()["content"], add_special_tokens=False)

        def generate(engine, prompts, max_new_tokens, batch_size=8):
            with ThreadPoolExecutor(a.parallel) as ex:
                return list(ex.map(lambda p: one(p, max_new_tokens), prompts))
        acc_eval.engine_generate = generate  # the evaluator's only engine call
        g = acc_eval.GSM8KEvaluator(Engine(tok)).evaluate(acc_eval.load_gsm8k("test")[:a.samples],
                                                          acc_eval.load_gsm8k("train")[:a.shots], 256)
    finally:
        srv.terminate()
        srv.wait(30)
    row = dict(engine="llama.cpp", model=os.path.basename(a.gguf), tokenizer=a.tokenizer,
               gsm8k={k: g[k] for k in ("n", "shots", "exact_match", "stderr")})
    print(json.dumps(row), flush=True)
    save(a.out, [row], lambda r: r["engine"] == "llama.cpp" and r["model"] == row["model"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["bench", "gsm8k"])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--bin", default=os.path.expanduser("~/llama.cpp/build/bin"))
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.batches = [int(b) for b in a.batches.split(",")]
    a.out = a.out or ("results_4bit_speed_cuda.json" if a.mode == "bench" else "results_4bit_quality_llamacpp_cuda.json")
    (bench if a.mode == "bench" else gsm8k)(a)


if __name__ == "__main__":
    main()
