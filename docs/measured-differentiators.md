# Measured differentiators (draft)

Draft for integration; not linked from the mkdocs nav. Every number below comes from a committed script and its
results JSON (environment, `n`, standard errors). Machine: Apple M4 Pro, 48 GB, PyTorch 2.9.1, MPS, fp32 unless
stated. The machine was shared and heavily loaded (other jobs, swap close to full) while these ran, so wall times are
context only; the quality numbers are the point.

Models run through the Turing engine (`RealHuggingFaceLoader.convert_hf_model`, exact mode: HuggingFace's logits) and
its paged KV cache (`KVPool` / `KVCache`). NLL is in nats per token. "± x" is one standard error over windows
(problems for GSM8K); paired deltas are computed window by window.

**Status.** W6.1, W6.2 and W6.3 are complete on MPS (results JSONs committed). W6.4 (GSM8K debate) and the
ridge-λ sensitivity runs of W6.1 and W6.2 were stopped part-way on the Mac (too slow on the shared machine) and wrote
no results; they are to be run on the GCP L4 (`--device cuda`, results_*_cuda.json). No number from those partial runs
is quoted here.

## Library fixes made for these runs

- `RoPEContentDecoupler.strip_rope` / `apply_rope` (`turing/core/cross_model_kv.py`): with the native extension built,
  any input that was not on CUDA went to `turing_csrc.fused_rope_transform`, which only understands 2-D and 3-D arrays
  and silently returns a 4-D `[batch, seq, heads, dim]` array unchanged. `CrossModelKVPipeline.transfer_cache` passes
  4-D tensors, so on Mac and CPU the "RoPE content decoupling" was a no-op. 4-D input now goes through the native op
  one batch row at a time (test: `test_rope_4d_matches_per_slice`).
- `ClosedFormRidgeMapper.forward` used `turing_csrc.fused_ridge_forward` off CUDA: 7.0 s for a 512 × 2560 × 2048 map
  that `torch.matmul` does in 4 ms on the same CPU, with identical output. Removed; the mapper uses torch's GEMM.
- `ClosedFormRidgeMapper` sized its input from the target's head size, so a sender and receiver with different head
  sizes (Qwen2.5-0.5B: 64, Qwen2.5-1.5B: 128) could not be mapped. It takes `source_head_dim` now, and
  `CrossModelKVPipeline` passes it.
- `CrossModelKVPipeline` had no way to fit its maps. `CrossModelKVPipeline.fit(source K, V, target K, V)` strips RoPE
  from both caches and solves every target layer's K and V maps in closed form with the existing mapper
  (test: `test_pipeline_fit_recovers_linear_map_across_head_dims`).
- `MultiAgentCoordinator.run_xkv_latent_deliberation` (`turing/demo/agent_system.py`) returned
  `"measured_speedup": "7.2x faster than natural language message passing"` and a "baseline" latency that was its own
  latency times 7.2. Both removed; the docstring now says what the demo does (below, W6.4).

## W6.1 Cross-model KV transfer (closed-form ridge W*)

`python scripts/eval_xkv_transfer.py --calib-windows 64 --eval-windows 64` → `results_xkv_transfer_mps.json`;
control: `--pairs HuggingFaceTB/SmolLM2-360M:HuggingFaceTB/SmolLM2-360M,Qwen/Qwen2.5-0.5B:Qwen/Qwen2.5-0.5B
--out results_xkv_transfer_selfcontrol_mps.json`.

The sender prefills the context; `CrossModelKVPipeline` (library defaults: 8 source layers per target layer, ridge
λ = 0.01) maps its cache into the receiver's KV space; the result is written into the receiver's paged cache and the
receiver scores a 64-token continuation (63 predictions) on top of it. Maps fitted on 64 × 512 = 32,768 WikiText-2
train tokens; 64 test windows of 512 context + 64 continuation tokens.

| Sender → receiver | Receiver NLL: own prefill | transferred KV | no context | Sender's own NLL | Context benefit kept |
|---|---|---|---|---|---|
| SmolLM2-360M → SmolLM2-1.7B | 2.107 ± 0.065 | **7.174 ± 0.112** | 3.439 ± 0.056 | 2.441 ± 0.067 | **−2.80 ± 0.14** |
| Qwen2.5-0.5B → Qwen2.5-1.5B | 2.202 ± 0.073 | **2.598 ± 0.073** | 3.394 ± 0.066 | 2.547 ± 0.075 | **0.668 ± 0.019** |

Paired NLL deltas vs the receiver's own prefill: transfer +5.068 ± 0.088 (SmolLM2), +0.396 ± 0.030 (Qwen); no context
+1.333 ± 0.044 (SmolLM2), +1.192 ± 0.050 (Qwen). Relative error of the transferred cache against the receiver's own:
K 0.471, V 0.666 (SmolLM2); K 0.041, V 0.693 (Qwen).

- **SmolLM2 360M → 1.7B: negative.** The transferred cache is far worse than giving the receiver no context at all
  (NLL 7.17 vs 3.44). The 1.7B has 32 KV heads per layer (no GQA); the linear map from the 360M's 5 KV heads × 8 layers
  leaves a relative error of 47 % on K and 67 % on V, and the receiver does worse with that cache than with none.
- **Qwen2.5 0.5B → 1.5B: works partly.** The receiver keeps two thirds of the benefit of its own prefill, and ends up
  slightly worse than the sender reading its own cache (2.598 vs 2.547): the transfer hands the receiver about the
  sender's understanding of the context, not its own. K maps almost exactly (4 % error); V does not (69 %).
- **Control: the pipeline is exact.** Mapping a model onto itself (same calibration, same code) reproduces its cache
  (K error ≤ 0.0002) and its NLL to four decimals, with 100 % of the context benefit kept, on both families. The
  SmolLM2 result is the method, not a bug in the transfer code.
- **Time (context only, loaded machine):** per 512-token window, receiver prefill 1.82 s vs sender prefill 0.56 s +
  map-and-write 0.46 s (SmolLM2); 0.53 s vs 0.23 s + 0.19 s (Qwen). The transfer is cheaper than the receiver's own
  prefill here, but the maps cost a calibration pass and a fit (97 s for SmolLM2, 6 s for Qwen) per model pair.
- **Caveat: λ = 0.01 is effectively no regularisation** at these feature scales (W6.2 below: the same Qwen maps, given
  a context that opens with a sentence unlike WikiText, make the receiver worse than no context). A sensitivity run with
  λ = 100 (`--ridge-lambda 100 --out results_xkv_transfer_lambda100_cuda.json`) is pending; until it lands, the
  SmolLM2 result is a result for the library default, not for ridge transfer in general.

## W6.2 Multi-turn lineage: naive chaining vs `clean_base`

`python scripts/eval_lineage.py` → `results_lineage_mps.json`. Agents Qwen2.5-0.5B (A) and Qwen2.5-1.5B (B, whose
cache is tracked); ridge maps fitted both ways (library defaults, 32,768 WikiText-2 train tokens). Each of 32 contexts
is 512 tokens of WikiText-2 test whose first tokens are replaced by a fact, "Note: the secret passcode is NNNNNN.
Remember it." One turn sends B's cache to A and back: `read` is chosen by the library's
`CleanBaseStrategy` / `NaiveStrategy.cache_for_turn` (the frozen C_0, or the previous turn's cache), `delta =
RT(read) − read` with `RT = map_A→B ∘ map_B→A`, and `C_t = read + α·delta`. Measured per turn: the turn's residual
‖ΔC_R‖ (Frobenius over all layers' K and V as stored), the drift ‖C_t − C_0‖ relative to ‖C_0‖ (= 25,610), recall of the
passcode after a question (NLL of its tokens; share of contexts where every passcode token is the argmax) and NLL of
the next 64 WikiText tokens.

| Arm (α = 1) | Turn | ‖ΔC_R‖ | ‖C_t − C_0‖ / ‖C_0‖ | Passcode NLL | Passcode recalled | WikiText NLL |
|---|---|---|---|---|---|---|
| B's own cache | 0 | — | 0 | 0.005 ± 0.000 | 100 % | 2.062 ± 0.110 |
| no context | — | — | — | 2.422 ± 0.016 | 0 % | 3.259 ± 0.085 |
| clean_base | 1 … 8 (identical) | 1,595 | 0.062 | 1.844 ± 0.034 | 0 % | 2.428 ± 0.108 |
| naive | 1 | 1,595 | 0.062 | 1.844 ± 0.034 | 0 % | 2.428 ± 0.108 |
| naive | 2 | 12,198 | 0.46 | 9.27 ± 0.22 | 0 % | 9.14 ± 0.12 |
| naive | 3 | 3.19 × 10^5 | 12.0 | 9.45 ± 0.22 | 0 % | 8.98 ± 0.10 |
| naive | 6 | 5.72 × 10^9 | 2.15 × 10^5 | 11.25 ± 0.25 | 0 % | 9.05 ± 0.10 |
| naive | 8 | 3.92 × 10^12 | 1.47 × 10^8 | 11.25 ± 0.24 | 0 % | 9.69 ± 0.10 |

With α = 0.5: clean_base holds the passcode (NLL 0.081 ± 0.015, recalled in 94 ± 4 % of contexts, WikiText 2.135 ±
0.110) on every turn; naive matches it at turn 1, then loses it at turn 2 (NLL 9.23, 0 %, WikiText 9.11) while its drift
is still only 0.115 × ‖C_0‖, and its residual grows about 12.6× per turn (5,927 at turn 2, 1.47 × 10^8 at turn 6,
2.33 × 10^10 at turn 8).

- **The `clean_base` invariant holds up on real models.** Chaining translations on the mutated cache grows the residual
  geometrically (≈ 26× per turn at α = 1) and destroys both the fact and the language modelling by the second turn;
  recomputing against the frozen cache keeps every turn identical to the first.
- **CLAUDE.md's ">20,000 by turn 6" is an understatement in this setting**: the residual passes 12,000 at turn 2 and is
  5.7 × 10^9 at turn 6 (α = 1). The absolute norm scales with context length and model, so the relative drift
  (2 × 10^5 × ‖C_0‖ at turn 6) is the more portable number.
- **But `clean_base` does not make the translation good**: one round trip at full strength (α = 1) already loses the
  passcode (NLL 1.84 vs 2.42 without context, never recalled); only the half-strength merge keeps it.
- **One-way transfer fails on these contexts.** B reading A's cache mapped one way gives passcode NLL 2.250 ± 0.122
  (never recalled) and WikiText NLL 3.592 ± 0.163, worse than no context (3.259 ± 0.085), although the same maps
  help on plain WikiText windows (W6.1, 67 % of the benefit kept). The contexts differ only in their opening sentence;
  with λ = 0.01 the closed-form maps extrapolate badly off the calibration distribution. Whether a larger λ fixes this,
  and whether naive chaining still explodes when the round-trip map is shrunk by stronger regularisation, is what the
  pending `--ridge-lambda 100` run (`results_lineage_lambda100_cuda.json`) will show.

## W6.3 k-slot pooling vs low-bit KV vs dropping old tokens, at matched bytes

`python scripts/eval_kslot.py` → `results_kslot_mps.json`. 64 WikiText-2 test windows, 512 context + 64 continuation
tokens. Cache bytes are counted against an fp16 cache (the CUDA serving dtype); int8 is 0.531× and int4 0.312× of it
for head size 64, i.e. the bytes of 272 and 160 fp16 tokens out of 512. `kslot_k` pools the exact cache into k slots
per layer and KV head with `KSlotCachePooler` (RoPE stripped before pooling, re-applied at positions 0 .. k−1);
`last_m` keeps only the newest m context tokens. NLL deltas are paired against the exact cache.

| Arm | Bytes vs fp16 | SmolLM2-360M Δ NLL | SmolLM2-1.7B Δ NLL |
|---|---|---|---|
| int8 KV | 0.531 | +0.0000 ± 0.0004 | −0.0000 ± 0.0004 |
| k-slot, 272 slots | 0.531 | +3.037 ± 0.316 | +4.016 ± 0.184 |
| newest 272 tokens | 0.531 | +0.109 ± 0.019 | +0.099 ± 0.017 |
| int4 KV | 0.312 | +0.024 ± 0.004 | +0.018 ± 0.003 |
| k-slot, 160 slots | 0.312 | +3.303 ± 0.355 | +4.091 ± 0.151 |
| newest 160 tokens | 0.312 | +0.247 ± 0.028 | +0.218 ± 0.025 |
| k-slot, 64 slots | 0.125 | +3.346 ± 0.326 | +4.817 ± 0.168 |
| newest 64 tokens | 0.125 | +0.441 ± 0.035 | +0.414 ± 0.031 |
| k-slot, 4 slots | 0.008 | +6.663 ± 0.262 | +5.526 ± 0.238 |
| newest 4 tokens | 0.008 | +1.142 ± 0.046 | +1.144 ± 0.044 |
| no context | 0 | +1.327 ± 0.047 | +1.333 ± 0.044 |

Exact-cache NLL: 2.441 ± 0.067 (360M), 2.107 ± 0.065 (1.7B).

- **Negative for k-slot pooling as shipped.** At every budget the pooled cache is worse than no context at all, by
  1.7 to 5.3 nats. The pooler's slot queries are a random initialisation and the library has no procedure to train
  them, so what is measured is an untrained pooler; a trained one might do better, but there is nothing in the
  repository to train it with.
- **int8 KV is free and int4 nearly so** (+0.02 nats) at 0.53× and 0.31× of the fp16 bytes. At the same bytes,
  dropping the oldest tokens costs +0.10 to +0.11 nats (int8's bytes) and +0.22 to +0.25 nats (int4's bytes, 10 to 12
  times int4's cost). Quantizing beats truncating here.

## W6.4 Multi-agent deliberation on GSM8K (pending)

`python scripts/eval_debate.py --model Qwen/Qwen2.5-1.5B-Instruct --problems 100` → `results_debate_<device>.json`.
100 GSM8K test problems (`random.Random(0).sample`), greedy, bf16, on the Turing engine (batched decode over the paged
cache). Arms: one agent (`single`); three agents with differently worded instructions, majority vote of their
independent answers (`vote_r1`, no exchange); the same three agents after one round of reading each other's solutions
as text (`debate`, Du et al. 2023), majority vote. Exact match on the final number, binomial SE, paired difference vs
single, tokens generated, wall time. Not yet run to completion (see Status).

**KV exchange between agents cannot be measured: it does not exist.** `MultiAgentCoordinator` has no path that feeds
one agent's KV into another's generation:
- `run_xkv_latent_deliberation` and `run_multi_turn_deliberation` built the "source agent" KV with `torch.randn` and
  discarded the mapped KV before generating (one multiplied its own latency by 7.2 to report "7.2x faster"). Both
  were removed.
- `run_multi_agent_workflow` exchanges text only. Each agent ends its answer with a `NODES=…; ZONES=…` line, which
  the world model scores; an answer without that line is left unscored (`revision_successful` is `None`).

A KV-exchange debate arm would need new code: each agent's prompt prefix as a cache, the other agents' answers
appended as KV rather than text (with positions), and for heterogeneous agents a fitted `CrossModelKVPipeline`. Given
W6.1/W6.2 (transfer quality depends on λ and on the text being in-distribution), that is a research task, not a switch.
