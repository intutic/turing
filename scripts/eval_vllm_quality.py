#!/usr/bin/env python3
"""
vLLM's KV-cache quality on the checks Turing's low-bit KV gets, with Turing's own prompt builders and scorers:
GSM8K exact match (turing/serving/accuracy_eval.py: first --samples test problems, --shots train shots, greedy, 256
new tokens), pass-key retrieval (turing/serving/niah.py: same contexts, depths and key seed), and WikiText-2 NLL of the
last --decode tokens of --windows windows of --context tokens (the windows scripts/eval_kv_quality.py scores), read
from vLLM's prompt log-probabilities. Only generation is vLLM's: the evaluators' engine_generate is replaced.

Run in the vLLM virtualenv, once per KV cache dtype; rows are appended to --out:
~/vllm-venv/bin/python scripts/eval_vllm_quality.py --kv-cache-dtype auto
~/vllm-venv/bin/python scripts/eval_vllm_quality.py --kv-cache-dtype fp8
"""
import argparse
import importlib
import json
import math
import os
import platform
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def turing_evals():
    """turing/serving/accuracy_eval.py and niah.py without turing/serving/__init__.py (which loads the whole serving
    engine, not installed in vLLM's virtualenv): a bare package over that folder."""
    pkg = types.ModuleType("turing_evals")
    pkg.__path__ = [os.path.join(ROOT, "turing", "serving")]
    sys.modules["turing_evals"] = pkg
    return importlib.import_module("turing_evals.accuracy_eval"), importlib.import_module("turing_evals.niah")


def wikitext_windows(tok, T, n):
    """The windows scripts/eval_kv_quality.py scores: consecutive T-token windows of WikiText-2 test."""
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text).input_ids
    n = min(n, len(ids) // T)
    return [ids[i * T:(i + 1) * T] for i in range(n)]


def window_nll(ids, prompt_logprobs, start):
    """Mean NLL of tokens start+1 .. T-1 given their prefixes (what eval_kv_quality's score() averages)."""
    return -sum(prompt_logprobs[i][ids[i]].logprob for i in range(start + 1, len(ids))) / (len(ids) - start - 1)


def mean_se(v):
    m = sum(v) / len(v)
    return m, (math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1) / len(v)) if len(v) > 1 else None)


class Engine:
    """What the Turing evaluators need from an engine: its tokenizer (generation goes through vllm_generate)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--kv-cache-dtype", default="auto", help="any vLLM kv_cache_dtype (auto, fp8, int8_per_token_head, ...)")
    ap.add_argument("--kv-skip-layers", default="", help="layers vLLM keeps out of the quantized cache, e.g. 0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--context-len", type=int, nargs="+", default=[2048, 8192, 16384, 32768])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--decode", type=int, default=256)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--checks", default="gsm8k,niah,nll", help="any of gsm8k, niah, nll")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="vLLM's max_model_len (default: the longest pass-key context + 32, or 4128); for a LongRoPE "
                         "model (Phi-3) it may decide which frequencies vLLM uses for every request")
    ap.add_argument("--out", default="results_quality_vllm_cuda.json")
    a = ap.parse_args()

    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")  # a 32K needle prompt plus 8 new tokens, as for Turing
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import vllm
    acc_eval, niah = turing_evals()
    tok = AutoTokenizer.from_pretrained(a.model)
    skip = [x for x in a.kv_skip_layers.split(",") if x]
    max_model_len = a.max_model_len or (max(a.context_len) if "niah" in a.checks else 4096) + 32
    llm = LLM(a.model, dtype=a.dtype, kv_cache_dtype=a.kv_cache_dtype, enable_prefix_caching=False,
              **({"kv_cache_dtype_skip_layers": skip} if skip else {}),
              max_model_len=max_model_len,
              gpu_memory_utilization=0.85)

    def vllm_generate(engine, prompts, max_new_tokens, batch_size=8):
        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outs]
    acc_eval.engine_generate = niah.engine_generate = vllm_generate  # the evaluators' only engine call
    engine = Engine(tok)
    row = dict(engine="vllm", runtime=f"vllm {vllm.__version__}", model=a.model, dtype=a.dtype,
               kv_cache_dtype=a.kv_cache_dtype, kv_skip_layers=skip, max_model_len=max_model_len)

    checks = set(a.checks.split(","))
    t0 = time.perf_counter()
    if "gsm8k" in checks:
        g = acc_eval.GSM8KEvaluator(engine).evaluate(acc_eval.load_gsm8k("test")[:a.samples],
                                                     acc_eval.load_gsm8k("train")[:a.shots], 256)
        row["gsm8k"] = {k: g[k] for k in ("n", "shots", "exact_match", "stderr")}
    if "niah" in checks:
        nh = niah.LongContextNIAHEvaluator(engine).evaluate_retrieval(a.context_len, a.depths)
        row["niah"] = {"n": nh["n"], "accuracy": nh["accuracy"],
                       "by_context": {c: sum(r["found"] for r in nh["results"] if r["context_length"] == c)
                                      / sum(1 for r in nh["results"] if r["context_length"] == c)
                                      for c in a.context_len}}
    if "nll" in checks:
        wins = wikitext_windows(tok, a.context, a.windows)
        outs = llm.generate([{"prompt_token_ids": w} for w in wins], SamplingParams(max_tokens=1, prompt_logprobs=0),
                            use_tqdm=False)
        per_window = [window_nll(w, o.prompt_logprobs, a.context - a.decode) for w, o in zip(wins, outs)]
        m, se = mean_se(per_window)
        row["wikitext_nll"] = {"n": len(per_window), "mean": m, "se": se, "per_window": per_window,
                               "context": a.context, "scored": a.decode - 1}
        print(f"wikitext nll {m:.4f} ± {se:.4f}", flush=True)
    row["seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps({k: v for k, v in row.items() if k != "wikitext_nll"}), flush=True)

    old = json.load(open(a.out))["rows"] if os.path.exists(a.out) else []
    same = lambda r: (r["model"] == a.model and r["dtype"] == a.dtype and r["kv_cache_dtype"] == a.kv_cache_dtype
                      and r.get("kv_skip_layers", []) == skip)
    rows = [r for r in old if not same(r)] + [row]
    # paired NLL change against the fp16 (auto) run, per window, for every quantized cache of this model and dtype
    base = next((r for r in rows if r["model"] == a.model and r["dtype"] == a.dtype and r["kv_cache_dtype"] == "auto"
                 and "wikitext_nll" in r), None)
    for r in rows:
        if base and r is not base and r["model"] == a.model and r["dtype"] == a.dtype and "wikitext_nll" in r:
            d = [x - y for x, y in zip(r["wikitext_nll"]["per_window"], base["wikitext_nll"]["per_window"])]
            dm, dse = mean_se(d)
            r["wikitext_nll"]["delta_vs_auto"] = {"mean": dm, "se": dse, "n": len(d)}
            print(f"{r['kv_cache_dtype']} skip={r.get('kv_skip_layers', [])} - fp16 nll {dm:+.4f} ± {dse:.4f}",
                  flush=True)
    env = dict(platform=platform.platform(), python=platform.python_version())
    with open(a.out, "w") as f:
        json.dump(dict(env=env, text="GSM8K and pass-key prompts and scoring from turing/serving; WikiText-2 windows "
                                     "as scripts/eval_kv_quality.py", rows=rows), f, indent=2)
    print(f"[+] wrote {a.out}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # results are written; vLLM 0.31's helper processes kept the interpreter alive for hours at exit
