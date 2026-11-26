"""What an int4 KV cache does to the Colab notebook's one-line answer, which on a T4 can come back as
`reversed(s) = s[::-1]` (code that does not run).

On a GPU, Qwen2.5-1.5B-Instruct in fp16, for native / int8 / int4 KV (and int4 with 2 and 5 more layers kept): the
first answer token's top 5 logits, the log-probability of the right and the wrong answer (teacher-forced through the
cache), and the greedy answer decoded token by token through the cache. Then, with native KV, the greedy continuation
after the wrong answer's opening, to see whether the error follows from it.

python scripts/probe_int4_answer.py probe_int4_answer.json
"""
import _bootstrap  # noqa: F401,E402  (this repo's root first on sys.path)
import json
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import turing
from turing.models.hf_loader import RealHuggingFaceLoader

M = "Qwen/Qwen2.5-1.5B-Instruct"
Q = "Write a Python one-liner that reverses a string s."
GOOD = ("You can reverse a string in Python using slicing. Here is a one-liner to reverse a string `s`:\n\n"
        "```python\nreversed_string = s[::-1]\n```")
BAD = "Certainly! Here's a one-liner in Python that reverses a string `s`:\n\n```python\nreversed(s) = s[::-1]\n```"
OPENING = "Certainly! Here's a one-liner in Python that reverses a string `s`:\n\n```python\n"

tok = AutoTokenizer.from_pretrained(M)
model, _ = RealHuggingFaceLoader.load_hf_model_into_turing(M, device="cuda", dtype="fp16")
prompt = tok.apply_chat_template([{"role": "user", "content": Q}], tokenize=False, add_generation_prompt=True)
p = tok(prompt, return_tensors="pt").input_ids.cuda()
out = {"turing": turing.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "rows": []}
print(out)


@torch.inference_mode()
def run(ids, cont=None, steps=48):
    """Prefill all but the last token, then the rest through the cache (each query reads the stored keys)."""
    _, past = model(ids[:, :-1], logits_to_keep=1)
    if cont is not None:
        x = torch.cat([ids[:, -1:], cont[:, :-1]], 1)
        logits, _ = model(x, past_key_values=past)
        lp = F.log_softmax(logits.float(), -1)[0].gather(-1, cont[0, :, None]).sum().item()
        return logits[0, 0].float(), lp
    cur, gen = ids[:, -1:], []
    for _ in range(steps):
        logits, past = model(cur, past_key_values=past)
        cur = logits[:, -1].argmax(-1, keepdim=True)
        if cur.item() == tok.eos_token_id:
            break
        gen.append(cur.item())
    return tok.decode(gen)


def ids_of(text):
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()


good, bad = ids_of(GOOD), ids_of(BAD)
kl = None
for kv, extra in [("native", 0), ("int8", 0), ("int4", 0), ("int4", 2), ("int4", 5)]:
    model.config.kv_cache_dtype, model.config.kv_native_layers = kv, None
    pool = model.kv_pool()
    if kv == "int4":
        kl = list(getattr(model, "_kv_layer_kl", None) or kl)
        if extra:
            worst = sorted(range(len(kl)), key=lambda i: -kl[i])
            kept = tuple(pool.native_layers or ())
            model.config.kv_native_layers = tuple(sorted(set(kept) | set(worst[:len(kept) + extra])))
            pool = model.kv_pool()
    first, lp_good = run(p, good)
    _, lp_bad = run(p, bad)
    top = torch.topk(first, 5)
    row = dict(kv=kv, native_layers=list(pool.native_layers or ()), lp_good=lp_good, lp_bad=lp_bad,
               top5=[(tok.decode([i]), round(v, 3)) for v, i in zip(top.values.tolist(), top.indices.tolist())],
               greedy=run(p))
    out["rows"].append(row)
    print(json.dumps(row, ensure_ascii=False))
if kl:
    out["int4_layer_kl"] = [round(x, 4) for x in kl]
    print("int4 per-layer KL, largest first:", sorted(((round(x, 4), i) for i, x in enumerate(kl)), reverse=True)[:10])
model.config.kv_cache_dtype, model.config.kv_native_layers = "native", None
model.kv_pool()
out["native_after_bad_opening"] = OPENING + run(torch.cat([p, ids_of(OPENING)], 1), steps=24)
print("native KV after the wrong answer's opening:", json.dumps(out["native_after_bad_opening"]))
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "probe_int4_answer.json", "w"), indent=1, ensure_ascii=False)
