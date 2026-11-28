#!/usr/bin/env python3
"""run_eval.py selfcheck | run | run-hard OUT.jsonl MODEL.gguf [MODEL.gguf ...]: the task set (tasks.py) against models served by
turing serve --backend libllama on the Mac's GPU, one model at a time; or the scorers against the reference answers.
Every request: greedy, reasoning_effort none, the task's max_tokens. Generated code runs in a separate Python process
(-I, 10 s timeout, a temporary directory) after a check for process, network and file-deleting calls."""
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import hard_tasks, tasks  # noqa: E402

PORT = 8095
DENY = re.compile(r"\b(subprocess|socket|urllib|requests|shutil|ctypes|os\.(system|remove|unlink|rmdir|popen|kill|exec)|"
                  r"__import__|eval\(|exec\(|open\()")


def code_of(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    return blocks[-1] if blocks else (text or "")


def run_tests(code, tests):
    if DENY.search(code):
        return False, "refused: code calls processes, the network or file operations"
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "t.py")
        open(f, "w").write(code + "\n\n" + "\n".join(f"assert {t}, {t!r}" for t in tests) + "\n")
        try:
            r = subprocess.run([sys.executable, "-I", f], cwd=d, capture_output=True, text=True, timeout=10,
                               env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1][:200]


def number_of(text):
    m = re.findall(r"Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text or "")
    if not m:
        m = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    try:
        return float(m[-1].replace(",", "")) if m else None
    except ValueError:
        return None


def same(a, b):
    if isinstance(b, bool) or b is None:
        return a == b
    if isinstance(b, (int, float)):
        return isinstance(a, (int, float)) and not isinstance(a, bool) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(b, str):
        return isinstance(a, str) and a.strip() == b
    if isinstance(b, list):
        return isinstance(a, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(b, dict):
        return isinstance(a, dict) and all(k in a and same(a[k], v) for k, v in b.items())
    return a == b


def json_of(text):
    t = text or ""
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return None


def score(task, resp):
    """(passed, note, unparsed_tool_call): resp has content and tool_calls (as the OpenAI API returns them)."""
    content = resp.get("content") or ""
    cat = task["cat"]
    if cat in ("write code", "fix a bug"):
        ok, note = run_tests(code_of(content), task["tests"])
        return ok, note, False
    if cat == "word problem":
        n = number_of(content)
        return n is not None and math.isclose(n, task["answer"], abs_tol=1e-6), f"got {n}", False
    if cat == "extract JSON":
        obj = json_of(content)
        return obj is not None and same(obj, task["expected"]), "no JSON" if obj is None else "fields differ", False
    calls = resp.get("tool_calls") or []
    if calls:
        f = calls[0]["function"]
        args = f.get("arguments")
        args = json.loads(args) if isinstance(args, str) else (args or {})
        ok = f.get("name") == task["tool"] and bool(task["check"](args))
        return ok, f"{f.get('name')} {json.dumps(args)[:120]}", False
    # a call written as text the server did not parse: JSON, or the XML form of Qwen3.5 / 3.6 (Qwen3-Coder style)
    xml = re.search(r"<function=([\w.-]+)>(.*?)</function>", content, re.S)
    if xml:
        args = {k: v.strip() for k, v in re.findall(r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>", xml.group(2), re.S)}
        if xml.group(1) == task["tool"]:
            return False, "tool call left in the text (XML), not parsed", bool(_safe(task["check"], args))
    gemma = re.search(r"call:([\w.-]+)(\{.*?\})<tool_call\|>", content, re.S)  # Gemma 4: call:name{key:<|"|>v<|"|>,n:1}
    if gemma and gemma.group(1) == task["tool"]:
        body = re.sub(r'([{,])\s*([A-Za-z_]\w*)\s*:', r'\1"\2":', gemma.group(2).replace('<|"|>', '"'))
        try:
            args = json.loads(body)
        except json.JSONDecodeError:
            args = {}
        return False, "tool call left in the text (Gemma form), not parsed", bool(_safe(task["check"], args))
    obj = json_of(content)
    if obj and obj.get("name") == task["tool"]:
        args = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
        args = json.loads(args) if isinstance(args, str) else args
        return False, "tool call left in the text, not parsed", bool(_safe(task["check"], args or {}))
    return False, "no tool call", False


def _safe(check, args):
    try:
        return check(args)
    except (TypeError, ValueError):
        return False


def selfcheck():
    bad = 0
    for t in tasks() + hard_tasks():
        if t["cat"] == "tool call":
            continue
        ok, note, _ = score(t, {"content": t["reference"]})
        if not ok:
            bad += 1
            print("REFERENCE FAILS", t["id"], note)
    print("selfcheck:", "ok" if bad == 0 else f"{bad} failing")
    return bad == 0


def post(body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def run(out, models, which="base"):
    ts = hard_tasks() if which == "hard" else tasks()
    ts = [t for t in ts if t["id"].startswith(os.environ.get("ONLY", ""))]  # ONLY=tool/ reruns a subset
    with open(out, "a") as f:
        for path in models:
            name = os.path.basename(path)
            log = open(out + f".{name}.serve.log", "w")
            srv = subprocess.Popen(["turing", "serve", "--backend", "libllama", "--model", path, "--gpu-layers", "-1",
                                    "--no-repack", "--ctx", "8192", "--threads", "10", "--host", "127.0.0.1",
                                    "--port", str(PORT)], stdout=log, stderr=subprocess.STDOUT)
            try:
                for _ in range(180):
                    try:
                        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)
                        break
                    except Exception:
                        if srv.poll() is not None:
                            raise RuntimeError(f"turing serve exited for {name}")
                        time.sleep(1)
                for t in ts:
                    body = {"model": name, "messages": t["messages"], "max_tokens": t["max_tokens"], "temperature": 0,
                            "reasoning_effort": "none"}
                    if t.get("tools"):
                        body["tools"] = t["tools"]
                    t0 = time.perf_counter()
                    try:
                        r = post(body)
                        m = r["choices"][0]["message"]
                        resp = {"content": m.get("content"), "tool_calls": m.get("tool_calls"),
                                "reasoning": m.get("reasoning_content"), "usage": r.get("usage"),
                                "finish": r["choices"][0].get("finish_reason")}
                    except urllib.error.HTTPError as e:
                        resp = {"content": None, "error": f"{e.code} {e.read().decode()[:300]}"}
                    resp["seconds"] = round(time.perf_counter() - t0, 2)
                    ok, note, unparsed = score(t, resp)
                    row = {"model": name, "id": t["id"], "cat": t["cat"], "ok": ok, "note": note,
                           "unparsed_tool_call_ok": unparsed, **resp}
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(f"{name[:28]:28s} {t['id']:28s} {'PASS' if ok else 'fail'} {resp['seconds']:6.1f}s {note[:60]}",
                          flush=True)
            finally:
                srv.terminate()
                try:
                    srv.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    srv.kill()
                log.close()


if __name__ == "__main__":
    if sys.argv[1] == "selfcheck":
        sys.exit(0 if selfcheck() else 1)
    if sys.argv[1] == "run-hard":
        run(sys.argv[2], sys.argv[3:], "hard")
    else:
        run(sys.argv[2], sys.argv[3:])
