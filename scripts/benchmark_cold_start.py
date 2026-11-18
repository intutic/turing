#!/usr/bin/env python3
"""
Cold start: time to first token from process start, and the load time inside it, for a 16-bit safetensors checkpoint
(the HuggingFace snapshot) and its GPTQ .tgate4 directory (`turing convert`, made here if missing).

Each run is a fresh `python -c` process that does what `turing generate --max-new-tokens 1 --temperature 0` does
(turing.cli._load_weights, then model.generate) and stamps the wall clock at four points:
    t0 (parent, just before spawning) -> imported (torch + turing.cli) -> loaded (weights on the device, synced)
    -> first token (one greedy token from a 4-token prompt).
ttft_s = first token - t0 (W9.5); load_s = loaded - imported (W9.3, the format comparison).

cold: before every run each model file (every file of the snapshot or the .tgate4 directory) is fsynced and dropped
from the page cache with posix_fadvise(POSIX_FADV_DONTNEED) (Linux only); mincore reports the resident fraction
before the run (resident_fraction from benchmark_ingest_ladder_all_tiers.py). Python, torch and the CUDA libraries
stay cached: this is a cold model, not a cold machine. warm: the same runs with the files in the page cache.
The children run with HF_HUB_OFFLINE=1 so no run waits on the network.

python scripts/benchmark_cold_start.py --models HuggingFaceTB/SmolLM2-1.7B,Qwen/Qwen2.5-1.5B --runs 5
"""
import _bootstrap  # noqa: F401,E402  (this repo's root first on sys.path)
import argparse
import json
import os
import platform
import statistics
import signal
import subprocess
import tempfile
import sys
import time

from benchmark_ingest_ladder_all_tiers import resident_fraction

ROOT = _bootstrap.ROOT
CHILD = r"""
import time; t_start = time.time()
import argparse, json, sys
import torch
from turing.cli import _load_weights
from turing.config import TuringConfig
t_imported = time.time()
a = argparse.Namespace(model=sys.argv[1], device=sys.argv[2], dtype=sys.argv[3] or None, sparsity=0.0,
                       activation_sparsity=0.0, quantization="none", kv_cache_dtype="auto", kv_page_budget=1.0)
dev = TuringConfig(device=a.device).resolve_device()
model, tok = _load_weights(a, dev)
if dev.type == "cuda":
    torch.cuda.synchronize()
t_loaded = time.time()
ids = tok.encode("Artificial intelligence is")
out = model.generate(ids, max_new_tokens=1, temperature=0.0)  # returns host ints: the token is on the host
t_first = time.time()
print("COLDSTART " + json.dumps(dict(t_start=t_start, t_imported=t_imported, t_loaded=t_loaded, t_first=t_first,
      token=out[-1], dtype=str(next(model.parameters()).dtype), device=str(dev),
      peak_device_bytes=torch.cuda.max_memory_allocated() if dev.type == "cuda" else None)))
"""


# vLLM, for the same measurement (run with --engine vllm --python ~/vllm-venv/bin/python --formats safetensors): its
# engine start includes torch.compile and CUDA-graph capture; the compile cache on disk is left as vLLM leaves it.
CHILD_VLLM = r"""
import time; t_start = time.time()
import json, os, sys
from vllm import LLM, SamplingParams
t_imported = time.time()
llm = LLM(sys.argv[1], dtype="float16", gpu_memory_utilization=0.85)
t_loaded = time.time()
out = llm.generate(["Artificial intelligence is"], SamplingParams(temperature=0.0, max_tokens=1), use_tqdm=False)
t_first = time.time()
print("COLDSTART " + json.dumps(dict(t_start=t_start, t_imported=t_imported, t_loaded=t_loaded, t_first=t_first,
      token=out[0].outputs[0].token_ids[0], dtype="float16", device="cuda", peak_device_bytes=None)), flush=True)
os._exit(0)  # vLLM's helper processes keep the interpreter alive at exit
"""


def model_files(folder):
    """Every regular file under folder, symlinks (the HF cache's blobs) resolved, each once."""
    out = set()
    for d, _, names in os.walk(folder):
        for n in names:
            p = os.path.realpath(os.path.join(d, n))
            if os.path.isfile(p):
                out.add(p)
    return sorted(out)


def evict(paths):
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.fsync(fd)  # dirty pages (a fresh .tgate4) cannot be dropped
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def resident(paths):
    """Byte-weighted fraction of the files in the page cache."""
    sizes = [os.path.getsize(p) for p in paths]
    return sum(resident_fraction(p) * s for p, s in zip(paths, sizes) if s) / max(1, sum(sizes))


def run_once(target, device, dtype, engine="turing", python=sys.executable):
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env["PATH"] = os.path.dirname(python) + os.pathsep + env["PATH"]  # a venv's own tools (vLLM's ninja)
    # Output goes to files and the child leads its own process group: vLLM's engine-core helper inherits the child's
    # pipes and outlives it, so waiting for a pipe's EOF (capture_output) blocked forever; the group is killed instead.
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        t0 = time.time()
        p = subprocess.Popen([python, "-c", CHILD_VLLM if engine == "vllm" else CHILD, target, device, dtype or ""],
                             cwd=ROOT, env=env, stdout=out, stderr=err, text=True, start_new_session=True)
        try:
            p.wait(timeout=900)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()
        out.seek(0); err.seek(0)
        stdout, stderr = out.read(), err.read()
    line = next((ln for ln in stdout.splitlines() if ln.startswith("COLDSTART ")), None)
    if p.returncode or line is None:
        return dict(error=f"exit {p.returncode}: {(stderr or stdout)[-2000:]}")
    r = json.loads(line[len("COLDSTART "):])
    return dict(ttft_s=r["t_first"] - t0, interpreter_s=r["t_start"] - t0, import_s=r["t_imported"] - r["t_start"],
                load_s=r["t_loaded"] - r["t_imported"], first_token_s=r["t_first"] - r["t_loaded"],
                token=r["token"], dtype=r["dtype"], device=r["device"], peak_device_bytes=r["peak_device_bytes"])


def stats(values):
    return dict(mean=statistics.mean(values),
                se=statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else None)


def ensure_tgate4(model, out_dir, a):
    """The .tgate4 directory of model, made with the real `turing convert` if it is missing."""
    if os.path.isfile(os.path.join(out_dir, "model.tgate4")):
        return dict(created=False)
    t = time.perf_counter()
    subprocess.run([sys.executable, "-m", "turing.cli", "convert", "--model", model, "--output", out_dir,
                    "--method", a.method, "--device", a.device, "--calib-windows", str(a.calib_windows),
                    "--calib-tokens", str(a.calib_tokens)], cwd=ROOT, check=True)
    return dict(created=True, method=a.method, calib_windows=a.calib_windows, calib_tokens=a.calib_tokens,
                elapsed_s=round(time.perf_counter() - t, 1))


def env_info():
    import torch
    import turing
    info = dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                cuda=torch.version.cuda, turing=turing.__version__, cpus=os.cpu_count(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    if os.path.exists("/proc/meminfo"):
        info["host_mem_kib"] = int(open("/proc/meminfo").readline().split()[1])
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="HuggingFaceTB/SmolLM2-1.7B,Qwen/Qwen2.5-1.5B")
    ap.add_argument("--formats", default="safetensors,tgate4")
    ap.add_argument("--caches", default="cold,warm")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"],
                    help="weight dtype (default: the CLI's, the checkpoint's bf16/fp16 on CUDA)")
    ap.add_argument("--tgate4-root", default=os.path.join(ROOT, "build", "tgate4"))
    ap.add_argument("--method", default="gptq", choices=["gptq", "rtn"], help="for missing .tgate4 directories")
    ap.add_argument("--calib-windows", type=int, default=128)
    ap.add_argument("--calib-tokens", type=int, default=2048)
    ap.add_argument("--engine", default="turing", choices=["turing", "vllm"])
    ap.add_argument("--python", default=sys.executable, help="interpreter of the child process (vLLM's virtualenv)")
    ap.add_argument("--out", default="results_cold_start_cuda.json")
    a = ap.parse_args(argv)
    caches = a.caches.split(",")
    if "cold" in caches and not hasattr(os, "posix_fadvise"):
        raise SystemExit("cold runs need posix_fadvise (Linux); pass --caches warm here")

    from huggingface_hub import snapshot_download
    rows, conversions = [], {}
    for model in a.models.split(","):
        snapshot = snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt",
                                                            "*.tiktoken"])
        targets = {}
        for fmt in a.formats.split(","):
            if fmt == "safetensors":
                targets[fmt] = (model, snapshot)
            elif fmt == "tgate4":
                d = os.path.join(a.tgate4_root, model.replace("/", "--"))
                conversions[model] = ensure_tgate4(model, d, a)
                targets[fmt] = (d, d)
            else:
                raise SystemExit(f"unknown format {fmt}")
        for fmt, (target, folder) in targets.items():
            files = model_files(folder)
            for cache in caches:
                if cache == "warm":  # read every file once, so the first warm run finds them cached too
                    for p in files:
                        with open(p, "rb") as f:
                            while f.read(64 << 20):
                                pass
                runs = []
                for r in range(a.runs):
                    if cache == "cold":
                        evict(files)
                    before = resident(files)
                    res = run_once(target, a.device, a.dtype, a.engine, os.path.expanduser(a.python))
                    res.update(resident_before=before)
                    runs.append(res)
                    print(model, fmt, cache, r, json.dumps(res)[:300], flush=True)
                ok = [x for x in runs if "error" not in x]
                row = dict(engine=a.engine, model=model, format=fmt, cache=cache, n=len(ok), errors=len(runs) - len(ok),
                           files=len(files), bytes=sum(os.path.getsize(p) for p in files), runs=runs)
                if ok:
                    row.update({k: stats([x[k] for x in ok]) for k in
                                ("ttft_s", "load_s", "import_s", "first_token_s", "interpreter_s")})
                    row.update(resident_before_max=max(x["resident_before"] for x in ok),
                               resident_before_min=min(x["resident_before"] for x in ok),
                               dtype=ok[0]["dtype"], tokens_agree=len({x["token"] for x in ok}) == 1)
                    print(f"== {model} {fmt} {cache}: ttft {row['ttft_s']['mean']:.2f}s load "
                          f"{row['load_s']['mean']:.2f}s (n={row['n']})", flush=True)
                rows.append(row)
                with open(a.out, "w") as f:  # rewritten after every row: a killed run keeps what it measured
                    json.dump(dict(script="scripts/benchmark_cold_start.py", env=env_info(), args=vars(a),
                                   conversions=conversions, rows=rows,
                                   note="seconds of wall clock; se = standard error over n runs; cold = model "
                                        "files dropped from the page cache, Python/torch/CUDA libraries cached"),
                              f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
