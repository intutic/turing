#!/usr/bin/env python3
"""
W6.4: multi-agent deliberation on GSM8K with a real instruct model on the Turing engine (exact HF weights, paged KV
cache, batched greedy decode).

Arms, on the same problems (a fixed-seed sample of the GSM8K main test split):
  single   one agent, greedy, prompt P0
  vote_r1  three agents (prompts P0, P1, P2: same task, different solving instructions, so greedy answers differ),
           majority vote of their independent answers (no exchange)
  debate   the same three agents, round 2 (Du et al. 2023): each sees its own round-1 solution and the other two
           agents' solutions as text and gives an updated answer; majority vote of the round-2 answers
A KV-exchange arm (agents reading each other's KV through a W* transfer) is not run: MultiAgentCoordinator
(turing/demo/agent_system.py) does not pass KV between agents (its XKV demo feeds torch.randn KV through an untrained
bridge and drops the result before generating), so there is nothing end to end to measure; see the results JSON.
Reported: exact-match accuracy (final number) with a binomial SE, the paired difference vs single with its SE,
per-agent accuracy per round, tokens generated, wall time (shared machine: context only).

python scripts/eval_debate.py --model Qwen/Qwen2.5-1.5B-Instruct --problems 100
"""
import _bootstrap  # noqa: F401,E402  (this repo's root first on sys.path)
import argparse
import json
import math
import random
import re
import time
from collections import Counter

import torch
from datasets import load_dataset

from eval_xkv_transfer import env_info, load_model, pick_device, sync
from turing.core.kv_cache import KVCache

END = " End your response with 'The answer is N.' where N is the final number."
PROMPTS = [
    "{q}\nSolve the problem step by step." + END,
    "{q}\nFirst list the quantities the problem gives, then compute the result one step at a time." + END,
    "{q}\nSolve the problem, then check your arithmetic again before you state the result." + END,
]
DEBATE = ("These are the solutions to the problem from other agents:\n\nOne agent solution: ```{a}```\n\n"
          "One agent solution: ```{b}```\n\nUsing the reasoning from other agents as additional advice, can you give "
          "an updated answer? Examine your solution and that of the other agents step by step." + END)
NUM = r"-?\d[\d,]*(?:\.\d+)?"


def extract(text):
    """The number after the last 'answer is', else in the last \\boxed{}, else the last number in the text."""
    m = (re.findall(r"answer is\s*:?\s*\$?\s*(?:\\boxed\{)?\s*\$?(" + NUM + ")", text, re.I)
         or re.findall(r"\\boxed\{\s*\$?(" + NUM + ")", text) or re.findall(NUM, text))
    if not m:
        return None
    try:
        return float(m[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def correct(pred, gold):
    return pred is not None and abs(pred - gold) < 1e-6


def vote(answers):
    """Majority over the non-None answers; a tie goes to the earliest agent's answer among the tied."""
    c = Counter(a for a in answers if a is not None)
    if not c:
        return None
    best = max(c.values())
    return next(a for a in answers if a is not None and c[a] == best)


@torch.inference_mode()
def generate(model, prompts, max_new, eos, device):
    """Greedy continuations of several token lists: each prefilled into its own paged cache, then decoded together
    one token per step (a list of KVCache). Returns the generated token lists (EOS excluded)."""
    caches, nxt, outs = [], [], [[] for _ in prompts]
    for ids in prompts:
        c = KVCache(model.kv_pool())
        logits, _ = model(torch.tensor([ids], device=device), past_key_values=c, logits_to_keep=1)
        caches.append(c)
        nxt.append(int(logits[0, -1].argmax()))
    active = list(range(len(prompts)))
    while active:
        for i in active:
            if nxt[i] not in eos:
                outs[i].append(nxt[i])
        active = [i for i in active if nxt[i] not in eos and len(outs[i]) < max_new]
        if not active:
            break
        x = torch.tensor([[nxt[i]] for i in active], device=device)
        logits, _ = model(x, past_key_values=[caches[i] for i in active])
        for i, t in zip(active, logits[:, -1].argmax(-1).tolist()):
            nxt[i] = t
    for c in caches:
        c.free()
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--problems", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4, help="problems decoded together")
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"], help="weights and KV cache")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    # 3 rows per problem, prompts + answers up to ~2,000 tokens: about 0.25 GiB per problem of the batch in bf16
    model, tok = load_model(args.model, device, dtype=dtype, kv_pool_gib=max(1.0, 0.25 * args.batch * (2 if args.dtype == "fp32" else 1)))
    eos = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")} | {tok.eos_token_id}
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = sorted(random.Random(args.seed).sample(range(len(ds)), args.problems))
    probs = [(ds[i]["question"], float(ds[i]["answer"].split("####")[-1].strip().replace(",", ""))) for i in idx]

    chat = lambda msgs: tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)  # noqa: E731
    enc = lambda text: tok(text, add_special_tokens=False).input_ids  # noqa: E731
    dec = lambda ids: tok.decode(ids, skip_special_tokens=True)  # noqa: E731
    recs, wall, toks = [], {"single": 0.0, "round1": 0.0, "round2": 0.0}, Counter()

    def timed(arm, prompts):
        sync(device)
        t0 = time.perf_counter()
        out = generate(model, prompts, args.max_new, eos, device)
        sync(device)
        wall[arm] += time.perf_counter() - t0
        toks[arm] += sum(len(o) for o in out)
        return out

    for s in range(0, len(probs), args.batch):
        batch = probs[s:s + args.batch]
        first = [[{"role": "user", "content": p.format(q=q)}] for q, _ in batch for p in PROMPTS]
        single = [dec(o) for o in timed("single", [enc(chat([{"role": "user", "content": PROMPTS[0].format(q=q)}]))
                                                   for q, _ in batch])]
        r1 = [dec(o) for o in timed("round1", [enc(chat(m)) for m in first])]
        second = []
        for j in range(len(batch)):
            sols = r1[3 * j:3 * j + 3]
            for a in range(3):
                others = [sols[b] for b in range(3) if b != a]
                second.append(first[3 * j + a] + [{"role": "assistant", "content": sols[a]},
                                                  {"role": "user", "content": DEBATE.format(a=others[0], b=others[1])}])
        r2 = [dec(o) for o in timed("round2", [enc(chat(m)) for m in second])]
        for j, (q, gold) in enumerate(batch):
            a1 = [extract(t) for t in r1[3 * j:3 * j + 3]]
            a2 = [extract(t) for t in r2[3 * j:3 * j + 3]]
            recs.append(dict(gold=gold, single=extract(single[j]), round1=a1, round2=a2, vote_r1=vote(a1),
                             debate=vote(a2), single_text=single[j], round1_text=r1[3 * j:3 * j + 3],
                             round2_text=r2[3 * j:3 * j + 3]))
        done = len(recs)
        acc = sum(correct(r["single"], r["gold"]) for r in recs) / done
        accd = sum(correct(r["debate"], r["gold"]) for r in recs) / done
        print(f"{done}/{len(probs)} problems: single {acc:.3f}, debate {accd:.3f}, wall {sum(wall.values()):.0f}s",
              flush=True)

    n = len(recs)
    hits = {arm: [float(correct(r[arm], r["gold"])) for r in recs] for arm in ("single", "vote_r1", "debate")}
    rows = []
    for arm, h in hits.items():
        p = sum(h) / n
        d = [x - y for x, y in zip(h, hits["single"])]
        dm = sum(d) / n
        dse = math.sqrt(sum((x - dm) ** 2 for x in d) / (n - 1) / n)
        gen = toks["single"] if arm == "single" else toks["round1"] + (toks["round2"] if arm == "debate" else 0)
        secs = wall["single"] if arm == "single" else wall["round1"] + (wall["round2"] if arm == "debate" else 0)
        rows.append(dict(arm=arm, accuracy=p, se=math.sqrt(p * (1 - p) / n), n=n, paired_delta_vs_single=dm,
                         paired_delta_se=dse, tokens_generated=gen, wall_seconds=secs))
    per_agent = {f"round{r}_agent{a}": sum(correct(x[f"round{r}"][a], x["gold"]) for x in recs) / n
                 for r in (1, 2) for a in range(3)}
    for r in rows:
        print(f"  {r['arm']:<8} acc {r['accuracy']:.3f} ± {r['se']:.3f}  vs single {r['paired_delta_vs_single']:+.3f} "
              f"± {r['paired_delta_se']:.3f}  tokens {r['tokens_generated']}  wall {r['wall_seconds']:.0f}s")
    print("  per agent:", {k: round(v, 3) for k, v in per_agent.items()})
    out = args.out or f"results_debate_{device.type}.json"
    with open(out, "w") as f:
        json.dump(dict(env=env_info(device), args=vars(args), dtype=str(dtype), problems=idx, rows=rows,
                       per_agent_accuracy=per_agent,
                       kv_exchange=dict(status="not run", reason=(
                           "MultiAgentCoordinator has no KV exchange between agents: run_xkv_latent_deliberation "
                           "builds torch.randn source KV, maps it with a randomly initialised XKVLatentAgentBridge, "
                           "and calls fast_generate without the transferred KV; run_multi_turn_deliberation is the "
                           "same with lineage bookkeeping; run_multi_agent_workflow exchanges text, and the "
                           "reuse_previous_kv flag it passes is ignored by TuringAcceleratedGenerator.fast_generate "
                           "(its anchors record token ids only). No path feeds one agent's KV into another's "
                           "generation.")),
                       records=recs), f, indent=2)
    print(f"\n[+] wrote {out}")


if __name__ == "__main__":
    main()
