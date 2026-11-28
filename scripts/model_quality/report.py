#!/usr/bin/env python3
"""report.py RESULTS.jsonl: per model and category, passed / n with the standard error of the pass rate
(sqrt(p(1-p)/n)); tool calls a model got right that the server left unparsed; seconds and completion tokens."""
import json
import math
import sys
from collections import OrderedDict

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import score  # noqa: E402
from tasks import hard_tasks, tasks  # noqa: E402

rows = list({(r["model"], r["id"]): r for r in map(json.loads, open(sys.argv[1]))}.values())  # a rerun replaces
by_id = {t["id"]: t for t in tasks() + hard_tasks()}
for r in rows:  # score again from the saved answers, so a scorer fix applies to the whole run
    r["ok"], r["note"], r["unparsed_tool_call_ok"] = score(by_id[r["id"]], r)
cats = list(OrderedDict.fromkeys(r["cat"] for r in rows))
models = list(OrderedDict.fromkeys(r["model"] for r in rows))
summary = {}
for m in models:
    mine = [r for r in rows if r["model"] == m]
    s = {}
    for c in cats + ["all"]:
        rs = [r for r in mine if c == "all" or r["cat"] == c]
        n, k = len(rs), sum(r["ok"] for r in rs)
        p = k / n if n else 0.0
        s[c] = {"passed": k, "n": n, "rate": round(p, 3), "se": round(math.sqrt(p * (1 - p) / n), 3) if n else None}
    s["unparsed_tool_calls_correct"] = sum(r.get("unparsed_tool_call_ok", False) for r in mine)
    s["errors"] = sum(1 for r in mine if r.get("error"))
    s["cut_off"] = sum(1 for r in mine if r.get("finish") == "length")  # answer ended by max_tokens
    s["seconds"] = round(sum(r.get("seconds", 0) for r in mine), 1)
    s["completion_tokens"] = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in mine)
    summary[m] = s
w = max(len(m) for m in models)
print(f"{'model':{w}s} " + " ".join(f"{c:>15s}" for c in cats + ["all"]) + "   seconds  tokens  unparsed-ok  cut-off")
for m, s in summary.items():
    cells = [f"{s[c]['passed']:>2d}/{s[c]['n']:<2d} ±{s[c]['se'] * 100:4.1f}%" if s[c]["n"] else "-" for c in cats + ["all"]]
    print(f"{m:{w}s} " + " ".join(f"{x:>15s}" for x in cells) +
          f"   {s['seconds']:7.0f} {s['completion_tokens']:7d} {s['unparsed_tool_calls_correct']:>6d} {s['cut_off']:>12d}")
json.dump(summary, open(sys.argv[1].replace(".jsonl", "_summary.json"), "w"), indent=2)
