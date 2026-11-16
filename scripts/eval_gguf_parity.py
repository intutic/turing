#!/usr/bin/env python3
"""
Real GGUF files through Turing's GGUF loader, against (a) llama.cpp itself on the same file and (b) the original
HuggingFace checkpoint in fp32. This is what proves the loader reads each family's llama.cpp conventions (tensor names,
q/k permutation, Gemma's stored 1 + w norms, attention scales, sliding windows, RoPE scaling, fused tensors, experts).

For each file, one 64-token WikiText-2 window (raw test split, tokenized by the HF tokenizer; the same ids go to all
three) is run as one prefill:
- llama.cpp via llama-cpp-python (Metal on a Mac, logits_all=True): max |Δlogit| and top-1 agreement with Turing,
  next to llama.cpp's own Metal-vs-CPU difference on the file (neither llama.cpp backend computes in exact fp32: Metal
  multiplies in half precision, the CPU quantizes activations to 8 bits), which sets the scale of "matches";
- HF fp32 on CPU: top-1 agreement with Turing and the NLL difference (the GGUF is quantized, HF is not).
Each row is written as it finishes; a failure is recorded as a row with its error, never skipped.

pip install llama-cpp-python   # builds llama.cpp (with Metal on Apple Silicon)
python scripts/eval_gguf_parity.py --device mps [--only llama,qwen2] [--out results_gguf_parity_mps.json]
"""
import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout's turing, not an installed one

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

# name: (GGUF repo, GGUF file, original HF checkpoint, Turing device override). Small official or widely used files;
# Q8_0 where the repo has it under ~4 GB.
FILES = {
    "llama": ("unsloth/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf", "unsloth/Llama-3.2-1B-Instruct", None),
    "llama-mixtral": ("DroidEverything/tiny-mixtral-30m-Q8_0-GGUF", "tiny-mixtral-30m-q8_0.gguf",
                      "axolotl-ai-co/tiny-mixtral-30m", None),
    "qwen2": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf", "Qwen/Qwen2.5-0.5B-Instruct", None),
    "qwen3": ("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf", "Qwen/Qwen3-0.6B", None),
    "gemma": ("ggml-org/gemma-1.1-2b-it-Q8_0-GGUF", "gemma-1.1-2b-it.Q8_0.gguf", "unsloth/gemma-1.1-2b-it", None),
    "gemma2": ("bartowski/gemma-2-2b-it-GGUF", "gemma-2-2b-it-Q8_0.gguf", "unsloth/gemma-2-2b-it", None),
    "gemma3": ("unsloth/gemma-3-270m-it-GGUF", "gemma-3-270m-it-Q8_0.gguf", "unsloth/gemma-3-270m-it", None),
    "phi2": ("TheBloke/phi-2-GGUF", "phi-2.Q8_0.gguf", "microsoft/phi-2", None),
    "phi3": ("microsoft/Phi-3-mini-4k-instruct-gguf", "Phi-3-mini-4k-instruct-q4.gguf", "microsoft/Phi-3-mini-4k-instruct", None),
    "phi3-longrope": ("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q8_0.gguf",
                      "microsoft/Phi-3.5-mini-instruct", None),
    "gpt2": ("QuantFactory/gpt2-GGUF", "gpt2.Q8_0.gguf", "gpt2", None),
    "gptneox": ("mradermacher/pythia-160m-GGUF", "pythia-160m.Q8_0.gguf", "EleutherAI/pythia-160m", None),
    "olmoe": ("allenai/OLMoE-1B-7B-0924-GGUF", "olmoe-1b-7b-0924-q8_0.gguf", "allenai/OLMoE-1B-7B-0924", "cpu"),
}
# families GGUFModelLoader reads that have no real file small enough to check here (fp32 must fit in 48 GB of RAM)
NOT_RUN = {
    "qwen2moe": "smallest real model is Qwen1.5-MoE-A2.7B (14.3B parameters: 57 GB in fp32, more than this machine's RAM)",
    "qwen3moe": "smallest real model is Qwen3-30B-A3B (30.5B parameters: 122 GB in fp32)",
}


def window(tok, T: int) -> list:
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"][:400])
    return tok(text).input_ids[:T]


def nll(logits: torch.Tensor, ids: list) -> float:
    return F.cross_entropy(logits[:-1].float(), torch.tensor(ids[1:])).item()


def free():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def hf_logits(repo: str, ids: list) -> torch.Tensor:
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32, attn_implementation="eager").eval()
    with torch.inference_mode():
        out = hf(torch.tensor([ids])).logits[0].float()
    del hf
    free()
    return out


def _llamacpp_worker(path: str, ids: list, gpu: bool, out_q):
    from llama_cpp import Llama
    llm = Llama(model_path=path, n_ctx=128, n_batch=128, n_ubatch=128, n_gpu_layers=-1 if gpu else 0, logits_all=True,
                verbose=False)
    llm.eval(ids)
    out_q.put(np.array(llm.scores[:len(ids)], dtype=np.float32))
    llm.close()


def llamacpp_logits(path: str, ids: list, gpu: bool = True, timeout: float = 600.0) -> torch.Tensor:
    """llama.cpp's logits in a child process, so a hung backend (seen: a Metal command buffer that never completes)
    is killed after `timeout` seconds and recorded as an error instead of stalling the run."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_llamacpp_worker, args=(path, ids, gpu, q))
    proc.start()
    try:
        out = q.get(timeout=timeout)
    except Exception:
        proc.kill()
        raise TimeoutError(f"llama.cpp ({'Metal' if gpu else 'CPU'}) gave no logits in {timeout:.0f} s")
    proc.join(30)
    return torch.from_numpy(out)


def turing_logits(path: str, ids: list, device: str):
    from turing.models.gguf_loader import GGUFModelLoader
    model, _ = GGUFModelLoader(path).load(device=device)
    cfg = model.config
    with torch.inference_mode():
        out = model(torch.tensor([ids], device=device))[0][0].float().cpu()
    del model
    free()
    return out, cfg


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    v = min(a.shape[-1], b.shape[-1])  # HF pads some vocabularies past the GGUF's token count
    a, b = a[:, :v], b[:, :v]
    d = (a - b).abs()
    return dict(max_abs_dlogit=d.max().item(), mean_abs_dlogit=d.mean().item(),
                top1_agreement=(a.argmax(-1) == b.argmax(-1)).float().mean().item())


def run(name: str, device: str, T: int) -> dict:
    from transformers import AutoTokenizer
    from turing.models.gguf_loader import GGUFReader, GGMLType
    repo, fname, hf_repo, dev_override = FILES[name]
    device = dev_override or device
    row = dict(name=name, gguf=f"{repo}/{fname}", hf=hf_repo, turing_device=device, tokens=T)
    t0 = time.perf_counter()
    path = hf_hub_download(repo, fname)
    with GGUFReader(path) as r:
        types = sorted({GGMLType(i.ggml_type).name for i in r.tensor_infos.values()})
        row.update(arch=r.metadata.get("general.architecture"), tensor_types=types)
    ids = window(AutoTokenizer.from_pretrained(hf_repo), T)
    row["first_ids"] = ids[:4]
    logits = {}
    for stage, fn in (("hf", lambda: hf_logits(hf_repo, ids)), ("llamacpp", lambda: llamacpp_logits(path, ids)),
                      ("llamacpp_cpu", lambda: llamacpp_logits(path, ids, gpu=False)),
                      ("turing", lambda: turing_logits(path, ids, device))):
        try:
            out = fn()
            if stage == "turing":
                out, cfg = out
                rope = cfg.rope_scaling or {}
                row["turing_config"] = dict(arch=cfg.arch, rope_type=rope.get("rope_type"), rope_factor=rope.get("factor"),
                                            sliding_window=cfg.sliding_window, attn_scale=cfg.attn_scale,
                                            num_experts=cfg.num_experts)
            logits[stage] = out
            row[f"nll_{stage}"] = nll(out, ids)
        except Exception as e:  # recorded, not hidden
            row[f"error_{stage}"] = f"{type(e).__name__}: {e}"[:500]
    if "turing" in logits and "llamacpp" in logits:
        row["vs_llamacpp"] = compare(logits["turing"], logits["llamacpp"])
    if "turing" in logits and "hf" in logits:
        row["vs_hf"] = compare(logits["turing"], logits["hf"])
        row["nll_diff_turing_minus_hf"] = row["nll_turing"] - row["nll_hf"]
    if "llamacpp" in logits and "llamacpp_cpu" in logits:  # llama.cpp's own backend noise, for scale
        row["llamacpp_metal_vs_cpu"] = compare(logits["llamacpp"], logits["llamacpp_cpu"])
    if "turing" in logits and "llamacpp_cpu" in logits:
        row["vs_llamacpp_cpu"] = compare(logits["turing"], logits["llamacpp_cpu"])
    if "llamacpp" in logits and "hf" in logits:
        row["llamacpp_vs_hf"] = compare(logits["llamacpp"], logits["hf"])
    row["status"] = "ok" if not any(k.startswith("error_") for k in row) else "error"
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def env(device: str) -> dict:
    import transformers
    import gguf
    import llama_cpp
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() \
        if sys.platform == "darwin" else platform.processor()
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    return dict(device=device, chip=chip, platform=platform.platform(), python=platform.python_version(),
                torch=torch.__version__, transformers=transformers.__version__, llama_cpp_python=llama_cpp.__version__,
                gguf=getattr(gguf, "__version__", None) or "0.19+", turing_commit=sha,
                llamacpp_backend="Metal, all layers offloaded" if sys.platform == "darwin" else "default",
                text="WikiText-2 raw test split, first 400 lines joined, first 64 tokens (HF tokenizer, its special tokens)",
                date=time.strftime("%Y-%m-%d"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--only", default="", help="comma-separated names from FILES (default: all)")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--out", default=None, help="results JSON (default results_gguf_parity_<device>.json; rows merged)")
    a = ap.parse_args()
    out = Path(a.out or f"results_gguf_parity_{a.device}.json")
    rows = {r["name"]: r for r in (json.loads(out.read_text())["rows"] if out.exists() else [])}
    for name, why in NOT_RUN.items():
        rows[name] = dict(name=name, arch=name, status="not run", reason=why)
    for name in [n for n in a.only.split(",") if n] or list(FILES):
        try:
            r = run(name, a.device, a.tokens)
        except Exception as e:  # keep going; the failure is recorded, not hidden
            r = dict(name=name, gguf="/".join(FILES[name][:2]), status="error", error=f"{type(e).__name__}: {e}"[:500])
        print(json.dumps({k: v for k, v in r.items() if k != "turing_config"}), flush=True)
        rows[name] = r
        out.write_text(json.dumps(dict(env=env(a.device), rows=list(rows.values())), indent=2))
    print(f"[+] wrote {out}")


if __name__ == "__main__":
    main()
