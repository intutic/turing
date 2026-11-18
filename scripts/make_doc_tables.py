#!/usr/bin/env python3
"""
Every numeric table in the docs, generated from the committed results_*.json files. A table lives between
`<!-- table:NAME -->` and `<!-- /table -->` in a docs page or the README; the prose around it is written by hand.

python scripts/make_doc_tables.py          # rewrite the tables in place
python scripts/make_doc_tables.py --check  # exit 1 if any table differs from what the results files give
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["README.md", "packages/kv-layer-guard/README.md",
         *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs").glob("*.md"))]


def load(name):
    d = json.loads((ROOT / name).read_text())
    return d["rows"] if isinstance(d, dict) and "rows" in d else d


def short(model):
    return model.split("/")[-1]


def pm(v, se, digits=3, sign=True):
    f = f"{{:{'+' if sign else ''}.{digits}f}}"
    return f.format(v) + (f" ± {se:.{digits}f}" if se is not None else "")


def pct(v):
    return f"{100 * v:.1f}%"


def table(header, rows, align=None):
    align = align or [":---"] + [":---:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- comparisons

def t_vs_short():
    rows = load("results_e2e_generation_cuda.json")
    batches = sorted({r["batch"] for r in rows})
    out = []
    for model in sorted({r["model"] for r in rows}):
        for eng in ("vllm", "turing"):
            got = {r["batch"]: r for r in rows if r["model"] == model and r["engine"] == eng and r["context"] == 1024}
            if got:
                ttft = got[1].get("mean_ttft_s") if 1 in got else None
                out.append([short(model), "vLLM" if eng == "vllm" else "Turing",
                            *[f"{got[b]['output_tok_per_s']:,.0f}" if b in got else "—" for b in batches],
                            f"{ttft:.3f} s" if ttft else "not recorded"])
    return table(["Model", "Engine", *[f"B={b}" for b in batches], "TTFT, B=1"], out)


def t_vs_4bit():
    speed = load("results_4bit_speed_cuda.json")
    q_t = {(r["quantization"], r["activation_sparsity"]): r for r in load("results_4bit_quality_turing_cuda.json")}
    q_v = {r["model"]: r for r in load("results_4bit_quality_vllm_cuda.json")}
    q_l = {r["model"]: r for r in load("results_4bit_quality_llamacpp_cuda.json")}

    def tok(match, b):
        r = [x for x in speed if match(x) and x["batch"] == b]
        return f"{r[0]['output_tok_per_s']:,.0f}" if r else "—"

    def gsm(r):
        return f"{pct(r['exact_match'])} ± {pct(r['stderr'])}" if r else "—"
    arms = [
        ("llama.cpp Q4_K_M", lambda x: x["engine"] == "llama.cpp", q_l.get("qwen2.5-1.5b-instruct-q4_k_m.gguf", {}).get("gsm8k")),
        ("vLLM GPTQ-Int4 (Qwen's checkpoint)", lambda x: x["engine"] == "vllm" and x["model"].endswith("GPTQ-Int4"),
         q_v.get("Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4", {}).get("gsm8k")),
        ("Turing GPTQ", lambda x: x["engine"] == "turing" and x.get("quantization") == "w4a16-gptq"
         and not x.get("activation_sparsity"), q_t.get(("w4a16-gptq", 0.0))),
        ("Turing GPTQ + TEAL 25%", lambda x: x["engine"] == "turing" and x.get("quantization") == "w4a16-gptq"
         and x.get("activation_sparsity") == 0.25, q_t.get(("w4a16-gptq", 0.25))),
        ("Turing GPTQ + TEAL 40%", lambda x: x["engine"] == "turing" and x.get("quantization") == "w4a16-gptq"
         and x.get("activation_sparsity") == 0.4, q_t.get(("w4a16-gptq", 0.4))),
        ("vLLM fp16 (reference)", lambda x: x["engine"] == "vllm" and x["model"] == "Qwen/Qwen2.5-1.5B-Instruct", None),
        ("Turing fp16 (reference)", lambda x: x["engine"] == "turing" and x.get("quantization") == "none"
         and not x.get("activation_sparsity"), None),
    ]
    return table(["Setting", "Batch 1, tok/s", "Batch 8, tok/s", "GSM8K (n=500)"],
                 [[n, tok(m, 1), tok(m, 8), gsm(q)] for n, m, q in arms])


def _turing_kv_quality():
    g = {r["kv_cache_dtype"]: r for r in load("results_gsm8k_final_cuda.json")}
    n = {r["kv_cache_dtype"]: r for r in load("results_niah_final_cuda.json")}
    d = {r["arm"]: r for r in load("results_quality_instruct_bf16_cuda.json") if r["metric"] != "nll"}
    return g, n, d


def t_vs_kv():
    g, n, d = _turing_kv_quality()
    lc = load("results_longctx_cuda.json")

    def sreq(match):
        r = [x for x in lc if match(x) and x["batch"] == 128]
        return f"{r[0]['seconds'] / 128:.2f}" if r else "—"
    out = []
    for kv, label in (("native", "Turing bf16 (reference)"), ("int8", "Turing int8"), ("int4", "Turing int4")):
        delta = "—" if kv == "native" else pm(d[kv]["value"], d[kv]["se"])
        out.append([label, "automatic" if kv == "int4" else "—", pct(g[kv]["exact_match"]), pct(n[kv]["accuracy"]),
                    delta, sreq(lambda x, kv=kv: x["engine"] == "turing" and x["kv_cache_dtype"] == kv
                                and x.get("kv_page_budget", 1.0) == 1.0)])
    for r in load("results_quality_vllm_cuda.json"):
        if r["model"] != "Qwen/Qwen2.5-1.5B-Instruct":
            continue
        kv, skip = r["kv_cache_dtype"], r.get("kv_skip_layers", [])
        dl = (r.get("wikitext_nll") or {}).get("delta_vs_auto")
        out.append([f"vLLM {'fp16 (reference)' if kv == 'auto' else kv}", f"layer {','.join(skip)} (manual)" if skip else "—",
                    pct(r["gsm8k"]["exact_match"]), pct(r["niah"]["accuracy"]),
                    pm(dl["mean"], dl["se"]) if dl else "—",
                    sreq(lambda x, kv=kv, skip=skip: x["engine"] == "vllm" and x["kv_cache_dtype"] == kv
                         and x.get("kv_skip_layers", []) == skip)])
    return table(["KV cache", "Layers kept in bf16", "GSM8K", "Pass-key 2K–32K", "ΔNLL vs bf16", "s/request (16K, B=128)"],
                 out)


def t_validation():
    """kv-layer-guard and Turing's KV pools on six instruct models (L4 validation run, 2026-11-18): GSM8K (8-shot,
    500 problems unless noted) and pass-key retrieval. Phi-3.5's Turing pass-key rows come from the rerun after the
    LongRoPE fix (results_niah_phi35_longrope_fix_cuda.json); the first run's are superseded."""
    vllm = load("results_quality_vllm_validation_cuda.json")
    gsm = {(r["model"], r["kv_cache_dtype"]): r for r in load("results_gsm8k_validation_cuda.json")}
    niah = {(r["model"], r["kv_cache_dtype"]): r for r in load("results_niah_validation_cuda.json")
            if "Phi-3.5" not in r["model"]}
    niah.update({(r["model"], r["kv_cache_dtype"]): r for r in load("results_niah_phi35_longrope_fix_cuda.json")})
    guard = {}
    for f in sorted(ROOT.glob("results_guard_*_cuda.json")):
        g = json.loads(f.read_text())
        guard[g["model"]] = g["kv_cache_dtype_skip_layers"]

    def em(r):
        n = r["gsm8k"]["n"] if "gsm8k" in r else r.get("samples", r.get("n"))
        v = r["gsm8k"]["exact_match"] if "gsm8k" in r else r["exact_match"]
        return pct(v) + ("" if n == 500 else f" (n={n})")
    out = []
    for model in dict.fromkeys(r["model"] for r in vllm):
        rows = {(r["kv_cache_dtype"], tuple(r.get("kv_skip_layers", []))): r for r in vllm if r["model"] == model}
        skip = tuple(guard.get(model, ()))
        bf16, fp8 = rows[("auto", ())], rows[("fp8", ())]
        guarded = rows.get(("fp8", skip)) if skip else fp8
        tur = [gsm.get((model, kv)) for kv in ("int8", "int4")]
        pk = [pct(x["niah"]["accuracy"]) for x in (fp8, guarded)] + \
             [pct(niah[(model, "int4")]["accuracy"]) if (model, "int4") in niah else "—"]
        out.append([short(model), em(bf16), em(fp8), ", ".join(skip) or "none", em(guarded) if skip else "= fp8",
                    em(tur[0]) if tur[0] else "—", em(tur[1]) if tur[1] else "—", " / ".join(pk)])
    return table(["Model", "vLLM bf16", "vLLM fp8 (default)", "Guard keeps in bf16", "vLLM fp8 + guard",
                  "Turing int8 KV", "Turing int4 KV", "Pass-key: fp8 / fp8 + guard / Turing int4"], out)


def t_longctx():
    lc = load("results_longctx_cuda.json")
    batches = sorted({r["batch"] for r in lc})
    keys = []
    for r in lc:
        k = (r["engine"], r["kv_cache_dtype"], r.get("kv_page_budget", 1.0), tuple(r.get("kv_skip_layers", [])))
        if k not in keys:
            keys.append(k)
    out = []
    for eng, kv, q, skip in keys:
        got = {r["batch"]: r for r in lc if (r["engine"], r["kv_cache_dtype"], r.get("kv_page_budget", 1.0),
                                             tuple(r.get("kv_skip_layers", []))) == (eng, kv, q, skip)}
        name = f"{'vLLM' if eng == 'vllm' else 'Turing'} {kv}" + (f", skip layer {','.join(skip)}" if skip else "") \
            + (f", Quest {q:.0%}" if q < 1 else "")
        cap = next(iter(got.values())).get("kv_capacity_tokens")
        out.append([name, f"{cap:,}" if cap else "—", *[f"{got[b]['seconds'] / b:.2f}" if b in got else "—"
                                                          for b in batches]])
    return table(["Engine, KV cache", "KV capacity (tokens)", *[f"B={b}" for b in batches]], out)


# ---------------------------------------------------------------------------------------------- Turing quality

def t_kv_quality():
    rows = [r for r in load("results_quality_cuda.json") if r["metric"] != "nll"]
    models = list(dict.fromkeys(r["model"] for r in rows))
    labels = {"int8": "int8 KV", "int4": "int4 KV (smoothed, protected layers)", "int4_nosmooth": "int4 KV, no smoothing",
              "quest25": "Quest, 25% of pages", "rtn4": "4-bit weights, RTN", "gptq4": "4-bit weights, GPTQ",
              "teal25": "TEAL 25% (decode)", "teal40": "TEAL 40% (decode)", "teal50": "TEAL 50% (decode)",
              "exact_decode": "token by token vs one pass (sanity)"}
    out = []
    for arm, label in labels.items():
        got = {r["model"]: r for r in rows if r["arm"] == arm}
        if got:
            out.append([label, *[pm(got[m]["value"], got[m]["se"], 4) if m in got else "—" for m in models]])
    return table(["Setting (ΔNLL vs exact)", *[short(m) for m in models]], out)


def t_gsm8k():
    out = []
    for r in load("results_gsm8k_final_cuda.json"):
        out.append([f"{r['kv_cache_dtype']} KV", r["dtype"], r["batch_size"], pct(r["exact_match"]) + f" ± {pct(r['stderr'])}"])
    for r in load("results_gsm8k_teal_cuda.json"):
        out.append([f"TEAL {r['activation_sparsity']:.0%}" if r["activation_sparsity"] else "native (TEAL baseline)",
                    r["dtype"], r["batch_size"], pct(r["exact_match"]) + f" ± {pct(r['stderr'])}" + f" (n={r['n']})"])
    for r in load("results_gsm8k_diag_cuda.json"):
        out.append([f"{r['kv_cache_dtype']} KV{'' if r['kv_smoothing'] else ', smoothing off'}", r["dtype"],
                    r["batch_size"], pct(r["exact_match"]) + f" ± {pct(r['stderr'])}"])
    return table(["Setting", "dtype", "Batch", "GSM8K exact match"], out)


def t_niah():
    rows = load("results_niah_final_cuda.json")
    ctxs = sorted({x["context_length"] for x in rows[0]["results"]})
    out = []
    for r in rows:
        by = {c: [x["found"] for x in r["results"] if x["context_length"] == c] for c in ctxs}
        out.append([f"{r['kv_cache_dtype']} KV", *[f"{sum(v)}/{len(v)}" for v in by.values()], pct(r["accuracy"])])
    return table(["KV cache", *[f"{c // 1024}K" for c in ctxs], "All"], out)


def t_formats():
    out = []
    for f in ("results_weight_formats_cuda.json", "results_weight_formats_qwen_cuda.json"):
        for r in load(f):
            out.append([short(r["model"]), r["setting"], f"{r['tokens_per_second']:.1f}",
                        f"{r['eager_tokens_per_second']:.1f}" if r.get("eager_tokens_per_second") else "—",
                        f"{r['weight_bytes'] / 2 ** 30:.2f}"])
    return table(["Model", "Weights", "Decode tok/s (CUDA graphs)", "Eager", "Weight GiB"], out)


def t_crossover():
    rows = load("results_w4a16_crossover_cuda.json")
    ms = sorted({r["M"] for r in rows})
    s = {m: [sum(r[k] for r in rows if r["M"] == m) for k in ("int4_kernel_us", "int4_kernel_no_split_us",
                                                             "dequant_cublas_us", "fp16_cublas_us")] for m in ms}
    return table(["Rows (M)", "int4 kernel", "int4, no K split", "dequantize + cuBLAS", "fp16 cuBLAS"],
                 [[m, *[f"{v:,.0f}" for v in s[m]]] for m in ms])


# ---------------------------------------------------------------------------------------------- decode throughput

def _decode(name):
    d = json.loads((ROOT / name).read_text())
    rows = d["rows"]
    ctxs = sorted({r["context"] for r in rows})
    bats = sorted({r["batch"] for r in rows})
    labels = {"hf_transformers": "HuggingFace transformers", "native_eager": "Turing, native, no CUDA graph",
              "native": "Turing, native cache", "int8": "Turing, int8", "int4": "Turing, int4",
              "int8_quest25": "Turing, int8 + Quest 25%"}
    out = []
    for setting, label in labels.items():
        got = {(r["context"], r["batch"]): r for r in rows if r["setting"] == setting}
        if not got:
            continue

        def cell(c, b):
            r = got.get((c, b))
            if r is None:
                return "—"
            if r.get("fit"):
                return f"{r['tokens_per_s']:,.0f}"
            return "OOM" if "out of memory" in r.get("limit", "") else "pool full"
        out.append([label, *[cell(c, b) for c in ctxs for b in bats]])
    return table(["Setting", *[f"{c // 1024}K × B={b}" for c in ctxs for b in bats]], out)


def t_decode_360m():
    return _decode("results_kv_decode_cuda_smollm2_360m.json")


def t_decode_1p7b():
    return _decode("results_kv_decode_cuda_smollm2_1p7b.json")


def t_quest():
    rows = load("results_quest_threshold_cuda_smollm2_360m.json")
    ctxs = sorted({r["context"] for r in rows})
    bats = sorted({r["batch"] for r in rows})

    def ratio(c, b):
        a = [r for r in rows if r["context"] == c and r["batch"] == b and r["setting"] == "int8"]
        q = [r for r in rows if r["context"] == c and r["batch"] == b and r["setting"] == "int8_quest25"]
        if not a or not q:
            return "—"
        if not (a[0].get("fit") and q[0].get("fit")):
            return "OOM"
        return f"{q[0]['tokens_per_s'] / a[0]['tokens_per_s']:.2f}×"
    return table(["Batch", *[f"{c // 1024}K" for c in ctxs]], [[b, *[ratio(c, b) for c in ctxs]] for b in bats])


# ---------------------------------------------------------------------------------------------- systems

def t_cold_start():
    out = []
    for r in load("results_cold_start_cuda.json"):
        if r.get("n"):
            out.append([short(r["model"]), r["format"], r["cache"], f"{r['bytes'] / 1e9:.2f}", r["n"],
                        pm(r["ttft_s"]["mean"], r["ttft_s"]["se"], 2, False), pm(r["load_s"]["mean"], r["load_s"]["se"], 2, False),
                        f"{r['import_s']['mean']:.2f}", f"{r['first_token_s']['mean']:.2f}"])
    return table(["Model", "Format", "Page cache", "GB", "n", "Process start → first token, s", "Load, s", "Imports, s",
                  "Load → first token, s"], out)


def t_moe():
    d = json.loads((ROOT / "results_moe_offload_cuda.json").read_text())
    out = []
    for r in d["rows"]:
        hit = r.get("hit_rate_decode")
        out.append([r["name"], f"{r['decode_tok_s']:.2f}", pct(hit) if hit is not None else "—",
                    f"{(r.get('peak_gpu_bytes') or 0) / 2 ** 30:.1f}",
                    pct(r["argmax_agrees_with_gpu"]) if r.get("argmax_agrees_with_gpu") is not None else "—"])
    return table(["Configuration", "Decode tok/s", "Expert-cache hit rate (decode)", "Peak GPU GiB",
                  "Same greedy token as all-on-GPU"], out)


def t_cpp():
    out = []
    for f, where in (("results_cpp_engine_mps.json", "Apple M4 Pro (NEON)"), ("results_cpp_engine_x86.json", "L4 VM CPU (AVX2)")):
        for r in load(f):
            out.append([where, r["binary"], r["kv_cache_dtype"], r["n"], pm(r["tok_s_mean"], r["tok_s_se"], 1, False)])
    return table(["Machine", "Build", "KV cache", "n", "tok/s"], out)


def _ingest(name):
    out = []
    for r in load(name):
        if r.get("measured"):
            out.append([r["tier"], r["cache"], pm(r["gb_per_s"], r["gb_per_s_se"], 2, False), f"{r['latency_ms']:,.0f}", r["n"]])
        else:
            out.append([r["tier"], "—", "not measured", "—", "—"])
    return table(["Row", "Cache", "GB/s", "ms", "n"], out, [":---", ":---:", "---:", "---:", ":---:"])


def t_ingest_mac():
    return _ingest("results_ingest_ladder.json")


def t_ingest_linux():
    return _ingest("results_ingest_ladder_linux_l4.json")


def t_ring_ab():
    return table(["Build", "GB/s (mean ± se)", "GB/s (median)", "ms", "n"],
                 [[r["arm"], pm(r["gb_per_s"], r["gb_per_s_se"], 2, False), f"{r['gb_per_s_median']:.2f}",
                   f"{r['latency_ms']:,.0f}", r["n"]] for r in load("results_ring_reader_ab_mps.json")],
                 [":---", "---:", "---:", "---:", ":---:"])


# ---------------------------------------------------------------------------------------------- models

def _parity(name):
    out = []
    for r in load(name):
        if "error" in r:
            out.append([r["model"], r.get("dtype", ""), "error: " + r["error"][:60], "", "", "", ""])
            continue
        floor = f"{r['hf_self_max_abs_diff']:.3f}" if r.get("hf_self_max_abs_diff") is not None else "—"
        out.append([r["model"], r["arch"], r["dtype"], f"{r['max_abs_diff_incremental']:.3g}", floor,
                    pct(r["top1_agreement"]), f"{r['nll_turing'] - r['nll_hf']:+.4f}"])
    return table(["Checkpoint", "Type", "dtype", "max \\|Δlogit\\| vs HF", "HF cached vs full (noise floor)",
                  "Top-1 agreement", "ΔNLL vs HF"], out)


def t_parity_mps():
    return _parity("results_parity_mps.json")


def t_parity_cuda():
    return _parity("results_parity_cuda.json")


def t_parity_cpu():
    return _parity("results_parity_cpu.json")


def t_gguf():
    out = []
    for r in load("results_gguf_parity_mps.json"):
        if r.get("status") == "not run":
            continue
        out.append([r["name"], "`" + r["gguf"].split("/")[-1] + "`", ", ".join(t for t in r["tensor_types"] if t != "F32"),
                    f"{r['vs_llamacpp']['max_abs_dlogit']:.3f}", pct(r["vs_llamacpp"]["top1_agreement"]),
                    f"{r['llamacpp_metal_vs_cpu']['max_abs_dlogit']:.3f}", f"{r['nll_turing']:.3f}",
                    f"{r['nll_llamacpp']:.3f}", f"{r['nll_hf']:.3f}"])
    return table(["Family", "GGUF file", "Quantized types", "max \\|Δlogit\\| vs llama.cpp", "Top-1 agreement",
                  "llama.cpp Metal vs CPU max \\|Δlogit\\|", "NLL Turing", "NLL llama.cpp", "NLL HF fp32"], out,
                 [":---", ":---", ":---", "---:", "---:", "---:", "---:", "---:", "---:"])


def t_serving_defaults():
    rows = load("results_serving_defaults_mps.json")
    models = list(dict.fromkeys(r["model"] for r in rows))
    out = []
    for m in models:
        got = {r["arm"]: r for r in rows if r["model"] == m and r["metric"] != "nll"}
        ex = [r for r in rows if r["model"] == m and r["arm"] == "exact_forward" and r["metric"] == "nll"][0]
        cell = lambda a: pm(got[a]["value"], got[a]["se"], 3) if a in got else "—"  # noqa: E731
        out.append([short(m), f"{ex['value']:.3f}", cell("sparsity_50"), cell("recirculation")])
    return table(["Model", "Exact NLL (= HuggingFace)", "FFN pruning 50% (old default)", "Recirculation (old default)"], out)


# ---------------------------------------------------------------------------------------------- research findings

def t_xkv():
    out = []
    for f, lam in (("results_xkv_transfer_cuda.json", "0.01"), ("results_xkv_transfer_lambda100_cuda.json", "100"),
                   ("results_xkv_transfer_selfcontrol_cuda.json", "0.01 (self)")):
        for p in json.loads((ROOT / f).read_text())["pairs"]:
            r = {(x["arm"], x["metric"]): x for x in p["rows"]}
            ret = r[("transfer", "context_benefit_retained")]
            out.append([f"{short(p['sender'])} → {short(p['receiver'])}", lam,
                        pm(ret["value"], ret["se"], 3, False), f"{r[('transfer', 'rel_err_K')]['value']:.3f}"])
    return table(["Sender → receiver", "Ridge λ", "Context benefit retained", "Key error (relative)"], out)


def t_lineage():
    rows = load("results_lineage_cuda.json")
    turns = [1, 2, 4, 6]
    out = []
    for arm in ("clean_base_alpha1", "naive_alpha1", "clean_base_alpha0.5", "naive_alpha0.5"):
        got = {r["turn"]: r for r in rows if r["arm"] == arm}
        out.append([arm.replace("_alpha", ", α="), *[f"{got[t]['wiki_nll']:.2f} / {pct(got[t]['recall_exact'])}"
                                                    for t in turns]])
    base = {r["arm"]: r for r in rows if r["turn"] == 0}
    out.append(["receiver alone, exact context", *[f"{base['exact']['wiki_nll']:.2f} / {pct(base['exact']['recall_exact'])}"] * len(turns)])
    out.append(["receiver, no context", *[f"{base['none']['wiki_nll']:.2f} / {pct(base['none']['recall_exact'])}"] * len(turns)])
    return table(["Arm (WikiText NLL / pass-code recall)", *[f"Turn {t}" for t in turns]], out)


def t_kslot():
    d = json.loads((ROOT / "results_kslot_cuda.json").read_text())
    out = []
    for m in d["models"]:
        for r in m["rows"]:
            if r["arm"] == "exact":
                continue
            out.append([short(m["model"]), r["arm"], f"{r['bytes_vs_fp16']:.3f}", pm(r["delta_vs_exact"], r["delta_se"], 3)])
    return table(["Model", "Context kept as", "Bytes vs fp16", "ΔNLL vs exact context"], out)


def t_debate():
    return table(["Arm", "GSM8K accuracy", "Paired Δ vs single", "Tokens generated"],
                 [[r["arm"], pm(r["accuracy"], r["se"], 2, False), pm(r["paired_delta_vs_single"], r["paired_delta_se"], 2),
                   f"{r['tokens_generated']:,}"] for r in load("results_debate_cuda.json")])


TABLES = {k[2:]: v for k, v in globals().items() if k.startswith("t_") and callable(v)}
BLOCK = re.compile(r"(<!-- table:([a-z0-9_]+) -->\n)(.*?)(<!-- /table -->)", re.S)


def render(text):
    def one(m):
        if m.group(2) not in TABLES:
            sys.exit(f"unknown table {m.group(2)!r}; known: {sorted(TABLES)}")
        return m.group(1) + TABLES[m.group(2)]() + "\n" + m.group(4)
    return BLOCK.sub(one, text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    stale, used = [], set()
    for page in PAGES:
        path = ROOT / page
        text = path.read_text()
        used |= {m.group(2) for m in BLOCK.finditer(text)}
        new = render(text)
        if new != text:
            stale.append(page)
            if not a.check:
                path.write_text(new)
    if a.check and stale:
        sys.exit(f"tables out of date in {stale}: run python scripts/make_doc_tables.py")
    print(("checked" if a.check else "wrote") + f" {len(used)} tables" + (f"; updated {stale}" if stale and not a.check else ""))


if __name__ == "__main__":
    main()
