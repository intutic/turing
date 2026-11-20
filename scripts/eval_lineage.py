#!/usr/bin/env python3
"""
W6.2: multi-turn KV lineage on real models (turing/experimental/lineage.py): naive chaining vs clean_base.

Two agents that share a tokenizer, A (sender) and B (receiver, whose cache is tracked). Closed-form ridge maps
(CrossModelKVPipeline, eval_xkv_transfer.py) are fitted both ways on WikiText-2 train caches. B's turn-0 cache C_0 is
its own prefill of a context that opens with a fact ("... the secret passcode is 482913.") followed by WikiText-2 test
text. One deliberation turn sends B's cache to A and A's translation back:
    read  = strategy.cache_for_turn(t, original=C_0, previous=C_{t-1})   (lineage.py: clean_base reads C_0, naive
                                                                          reads the previous turn's cache)
    delta = RT(read) - read,  RT = map_{A->B}(map_{B->A}(.))             (the translated residual)
    C_t   = read + alpha * delta
so naive with alpha = 1 chains translations on the mutated cache, and clean_base recomputes against the frozen C_0.
Per turn and strategy: the drift ||C_t - C_0|| and the turn's residual ||delta|| (Frobenius over every layer's K and V
as stored, K after RoPE; absolute and relative to ||C_0||), recorded through CacheLineage; recall of the fact (NLL of
the passcode tokens after a question, and the share of windows where every passcode token is the argmax); NLL of the
next D WikiText tokens. Baselines: B's exact cache (turn 0), A's cache mapped one way (one_way), no context (none).

python scripts/eval_lineage.py --sender Qwen/Qwen2.5-0.5B --receiver Qwen/Qwen2.5-1.5B
"""
import _bootstrap  # noqa: F401,E402  (this repo's root first on sys.path)
import argparse
import json
import math
import random

import torch
import torch.nn.functional as F

from eval_xkv_transfer import (cache_from_kv, continuation_nll, env_info, load_model, mean_se, pick_device,
                               prefill_kv, token_windows)
from turing.experimental.cross_model_kv import CrossModelKVPipeline
from turing.experimental.lineage import CacheLineage, CleanBaseStrategy, NaiveStrategy

QUESTION = "\nQuestion: What is the secret passcode mentioned above? Answer: The secret passcode is"


@torch.inference_mode()
def fit_both_ways(a, b, tok, args, device):
    cal = token_windows(tok, "train", args.context, args.calib_windows)
    kv = {m: ([[] for _ in range(m.config.num_layers)], [[] for _ in range(m.config.num_layers)]) for m in (a, b)}
    for w in cal:
        for m in (a, b):
            cache, k, v = prefill_kv(m, w[None].to(device))
            for layer in range(m.config.num_layers):
                kv[m][0][layer].append(k[layer].cpu())
                kv[m][1][layer].append(v[layer].cpu())
            cache.free()
    cat = {m: ([torch.cat(x) for x in kv[m][0]], [torch.cat(x) for x in kv[m][1]]) for m in (a, b)}
    pipes = {}
    for s, t, name in ((a, b, "a_to_b"), (b, a, "b_to_a")):
        p = CrossModelKVPipeline(s.config, t.config, top_k_layers=args.top_k, ridge_lambda=args.ridge_lambda)
        p.fit(*cat[s], *cat[t])
        for mp in list(p.layer_mappers_k.values()) + list(p.layer_mappers_v.values()):
            mp.to(device)
        pipes[name] = p
    return pipes, cal.numel()


def norm2(caches):
    return sum((x.float() ** 2).sum().item() for x in caches)


@torch.inference_mode()
def recall(model, kv, query, target):
    """NLL of the passcode tokens after the question, and whether each is the argmax, on top of cache kv (or None)."""
    cache = cache_from_kv(model, *kv) if kv is not None else None
    ids = torch.cat([query, target], 1)
    logits, _ = model(ids, past_key_values=cache)
    pred = logits[0, query.shape[1] - 1:-1].float()
    nll = F.cross_entropy(pred, target[0]).item()
    hit = bool((pred.argmax(-1) == target[0]).all())
    if cache is not None:
        cache.free()
    return nll, hit


@torch.inference_mode()
def wiki(model, kv, cont):
    cache = cache_from_kv(model, *kv) if kv is not None else None
    out = continuation_nll(model, cache, cont)
    if cache is not None:
        cache.free()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sender", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--receiver", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--cont", type=int, default=64)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--alphas", default="1.0,0.5", help="merge strengths of the translated residual")
    ap.add_argument("--calib-windows", type=int, default=64)
    ap.add_argument("--eval-windows", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--ridge-lambda", type=float, default=0.01)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    torch.manual_seed(0)
    rng = random.Random(0)
    a, tok = load_model(args.sender, device)
    b, _ = load_model(args.receiver, device)
    pipes, calib_tokens = fit_both_ways(a, b, tok, args, device)
    rt = lambda kv: pipes["a_to_b"].transfer_cache(*pipes["b_to_a"].transfer_cache(*kv))  # noqa: E731
    alphas = [float(x) for x in args.alphas.split(",")]

    query = torch.tensor([tok(QUESTION).input_ids], device=device)
    wins = token_windows(tok, "test", args.context + args.cont, args.eval_windows)
    rec = {}  # (arm, turn) -> dict of per-window lists

    def put(arm, turn, **vals):
        d = rec.setdefault((arm, turn), {})
        for k, v in vals.items():
            d.setdefault(k, []).append(v)

    for wi, w in enumerate(wins):
        code = f"{rng.randrange(10 ** 5, 10 ** 6)}"
        fact = tok(f"Note: the secret passcode is {code}. Remember it.\n").input_ids
        # the fact replaces the window's first tokens, so the context still ends right where the continuation starts
        ctx = torch.tensor([fact + w[len(fact):args.context].tolist()], device=device)
        cont = w[None, args.context:].to(device)
        target = torch.tensor([tok(" " + code).input_ids], device=device)

        cache0, k0, v0 = prefill_kv(b, ctx)
        cache0.free()
        c0 = (k0, v0)
        base2 = norm2(k0 + v0)
        cache_a, ka, va = prefill_kv(a, ctx)
        cache_a.free()
        for arm, kv in (("exact", c0), ("one_way", pipes["a_to_b"].transfer_cache(ka, va)), ("none", None)):
            n, h = recall(b, kv, query, target)
            put(arm, 0, recall_nll=n, recall_hit=h, wiki_nll=wiki(b, kv, cont))

        for alpha in alphas:
            for strategy in (CleanBaseStrategy(), NaiveStrategy()):
                arm = f"{strategy.name}_alpha{alpha:g}"
                lineage = CacheLineage()
                lineage.record(0, strategy.name, k0 + v0, k0 + v0, residual_norm=0.0)
                prev = c0
                for t in range(1, args.turns + 1):
                    read = strategy.cache_for_turn(t, list(c0), list(prev))
                    trans = rt(read)
                    delta = [x - y for x, y in zip(trans[0] + trans[1], read[0] + read[1])]
                    new = [y + alpha * d for y, d in zip(read[0] + read[1], delta)]
                    L = len(k0)
                    prev = (new[:L], new[L:])
                    drift2 = sum(((x - y).float() ** 2).sum().item() for x, y in zip(new, k0 + v0))
                    res = math.sqrt(norm2(delta))
                    lineage.record(t, strategy.name, read[0] + read[1], new, residual_norm=res)
                    n, h = recall(b, prev, query, target)
                    put(arm, t, recall_nll=n, recall_hit=h, wiki_nll=wiki(b, prev, cont),
                        residual_norm=res, rel_residual=res / math.sqrt(base2),
                        drift_norm=math.sqrt(drift2), rel_drift=math.sqrt(drift2 / base2), base_norm=math.sqrt(base2))
                    if not strategy.translates_on_turn(t):
                        raise AssertionError("both strategies translate every turn")
        print(f"window {wi + 1}/{len(wins)} done", flush=True)

    rows = []
    for (arm, turn), d in rec.items():
        row = dict(arm=arm, turn=turn, n=len(d["recall_nll"]))
        for k, v in d.items():
            if k == "recall_hit":
                p = sum(v) / len(v)
                row["recall_exact"], row["recall_exact_se"] = p, math.sqrt(p * (1 - p) / len(v))
            else:
                row[k], row[k + "_se"] = mean_se([float(x) for x in v])
        rows.append(row)
    print(f"\n{args.sender} <-> {args.receiver}: {len(wins)} windows, context {args.context}, calib {calib_tokens} tokens")
    print(f"{'arm':<22}{'turn':>5} {'rel_drift':>10} {'|residual|':>11} {'recall_nll':>16} {'recall_exact':>13} "
          f"{'wiki_nll':>16}")
    for r in rows:
        print(f"{r['arm']:<22}{r['turn']:>5} {r.get('rel_drift', 0):>10.4f} {r.get('residual_norm', 0):>11.1f} "
              f"{r['recall_nll']:>9.3f}±{r['recall_nll_se']:.3f} {r['recall_exact']:>8.2f}±{r['recall_exact_se']:.2f} "
              f"{r['wiki_nll']:>9.3f}±{r['wiki_nll_se']:.3f}")
    out = args.out or f"results_lineage_{device.type}.json"
    with open(out, "w") as f:
        json.dump(dict(env=env_info(device), args=vars(args), dtype="float32", calib_tokens=calib_tokens,
                       text="WikiText-2 raw: calibration on train, contexts from test; a 6-digit passcode fact opens "
                            "each context", rows=rows), f, indent=2)
    print(f"\n[+] wrote {out}")


if __name__ == "__main__":
    main()
