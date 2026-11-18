# 🔬 Research Findings

The components outside the serving path, measured on real models on an NVIDIA L4 (fp32 for the cross-model maps).
Several did not work; those results are here too. Tables are generated from the committed results by
`scripts/make_doc_tables.py`.

---

## Cross-model KV transfer (closed-form ridge W\*)

A sender model prefills a context; `CrossModelKVPipeline` (8 source layers per target layer, RoPE stripped before the
map and re-applied after) maps its cache into a receiver model's KV space, the receiver scores a 64-token continuation
on top of it, and the result is compared with the receiver reading the context itself (benefit 1) and with no context
(benefit 0). Maps fitted on 64 × 512 WikiText-2 train tokens; 64 test windows (`scripts/eval_xkv_transfer.py`,
`results_xkv_transfer_*_cuda.json`):

<!-- table:xkv -->
| Sender → receiver | Ridge λ | Context benefit retained | Key error (relative) |
| :--- | :---: | :---: | :---: |
| SmolLM2-360M → SmolLM2-1.7B | 0.01 | -0.018 ± 0.103 | 0.244 |
| Qwen2.5-0.5B → Qwen2.5-1.5B | 0.01 | 0.668 ± 0.019 | 0.041 |
| SmolLM2-360M → SmolLM2-1.7B | 100 | -0.129 ± 0.111 | 0.240 |
| Qwen2.5-0.5B → Qwen2.5-1.5B | 100 | 0.669 ± 0.019 | 0.041 |
| SmolLM2-360M → SmolLM2-360M | 0.01 (self) | 1.000 ± 0.000 | 0.000 |
| Qwen2.5-0.5B → Qwen2.5-0.5B | 0.01 (self) | 1.000 ± 0.000 | 0.000 |
<!-- /table -->

* **Qwen2.5 0.5B → 1.5B keeps two thirds of the context's benefit**, with keys mapped almost exactly and values not.
* **SmolLM2 360M → 1.7B keeps none of it.** The 1.7B has 32 KV heads per layer against the 360M's 5; the linear map
  leaves a quarter of the key and most of the value unexplained, and the receiver does no better than without context.
* **The mapping is exact on itself** (self-transfer keeps 100%), so the SmolLM2 result belongs to the method.
* **Regularisation does not change either result** (ridge λ = 100 against 0.01).
* The SmolLM2 map is ill-conditioned: the same fit on the Apple M4 Pro left a key error of 0.47 and a retained benefit
  of −2.80 (worse than no context); here it is 0.24 and −0.02. Qwen's matches on both machines (0.668).
* An earlier CUDA run collapsed because a Triton RoPE kernel rotated every head after the first at the wrong position;
  it was removed, and these numbers are from after the fix.

## Multi-turn lineage: naive chaining vs `clean_base`

Two agents (Qwen2.5-0.5B → Qwen2.5-1.5B) exchange translated KV over six turns; a 6-digit pass code opens each context.
"naive" adds each turn's translated residual to the receiver's already-modified cache; `clean_base` always recomputes
it against the receiver's original cache. α scales the injected residual (`scripts/eval_lineage.py`,
`results_lineage_cuda.json`; WikiText NLL / pass-code recall over 32 contexts):

<!-- table:lineage -->
| Arm (WikiText NLL / pass-code recall) | Turn 1 | Turn 2 | Turn 4 | Turn 6 |
| :--- | :---: | :---: | :---: | :---: |
| clean_base, α=1 | 2.43 / 0.0% | 2.43 / 0.0% | 2.43 / 0.0% | 2.43 / 0.0% |
| naive, α=1 | 2.43 / 0.0% | 2.88 / 0.0% | 4.56 / 0.0% | 5.47 / 0.0% |
| clean_base, α=0.5 | 2.14 / 90.6% | 2.14 / 90.6% | 2.14 / 90.6% | 2.14 / 90.6% |
| naive, α=0.5 | 2.14 / 90.6% | 2.32 / 9.4% | 2.75 / 0.0% | 3.69 / 0.0% |
| receiver alone, exact context | 2.06 / 100.0% | 2.06 / 100.0% | 2.06 / 100.0% | 2.06 / 100.0% |
| receiver, no context | 3.26 / 0.0% | 3.26 / 0.0% | 3.26 / 0.0% | 3.26 / 0.0% |
<!-- /table -->

* **`clean_base` is stable and naive chaining drifts**, as the multi-turn invariant in `turing/core/lineage.py`
  states: naive α = 1 goes from 2.43 to 5.47 nats by turn 6, while `clean_base` holds its first-turn value.
* **Neither carries the pass code at α = 1**; at α = 0.5 `clean_base` keeps it in 91% of contexts on every turn, and
  naive chaining loses it from turn 2.

## k-slot pooling against low-bit KV and against dropping old tokens

At matched cache bytes on SmolLM2 (512 context + 64 continuation tokens, 64 windows; `scripts/eval_kslot.py`,
`results_kslot_cuda.json`). `kslot_k` pools the cache into k slots per layer and KV head with `KSlotCachePooler`;
`last_m` keeps only the newest m tokens:

<!-- table:kslot -->
| Model | Context kept as | Bytes vs fp16 | ΔNLL vs exact context |
| :--- | :---: | :---: | :---: |
| SmolLM2-360M | none | 0.000 | +1.326 ± 0.047 |
| SmolLM2-360M | int8 | 0.531 | -0.000 ± 0.000 |
| SmolLM2-360M | int4 | 0.312 | +0.032 ± 0.004 |
| SmolLM2-360M | kslot_272 | 0.531 | +3.037 ± 0.316 |
| SmolLM2-360M | last_272 | 0.531 | +0.109 ± 0.018 |
| SmolLM2-360M | kslot_160 | 0.312 | +3.303 ± 0.355 |
| SmolLM2-360M | last_160 | 0.312 | +0.247 ± 0.028 |
| SmolLM2-360M | kslot_64 | 0.125 | +3.346 ± 0.326 |
| SmolLM2-360M | last_64 | 0.125 | +0.441 ± 0.035 |
| SmolLM2-360M | kslot_4 | 0.008 | +6.663 ± 0.262 |
| SmolLM2-360M | last_4 | 0.008 | +1.142 ± 0.046 |
| SmolLM2-1.7B | none | 0.000 | +1.333 ± 0.044 |
| SmolLM2-1.7B | int8 | 0.531 | -0.000 ± 0.000 |
| SmolLM2-1.7B | int4 | 0.312 | +0.024 ± 0.004 |
| SmolLM2-1.7B | kslot_272 | 0.531 | +4.012 ± 0.182 |
| SmolLM2-1.7B | last_272 | 0.531 | +0.098 ± 0.017 |
| SmolLM2-1.7B | kslot_160 | 0.312 | +4.023 ± 0.148 |
| SmolLM2-1.7B | last_160 | 0.312 | +0.218 ± 0.025 |
| SmolLM2-1.7B | kslot_64 | 0.125 | +4.600 ± 0.186 |
| SmolLM2-1.7B | last_64 | 0.125 | +0.414 ± 0.031 |
| SmolLM2-1.7B | kslot_4 | 0.008 | +5.447 ± 0.244 |
| SmolLM2-1.7B | last_4 | 0.008 | +1.144 ± 0.043 |
<!-- /table -->

* **k-slot pooling is worse than no context at every budget.** Its slot queries are a random initialisation and the
  library has no procedure to train them; what is measured is an untrained pooler.
* **Quantizing beats truncating**: int8 is free and int4 costs +0.02 to +0.03 nats, while keeping only the newest
  tokens at the same bytes costs +0.1 to +0.25.

## Multi-agent debate on GSM8K

Qwen2.5-1.5B-Instruct, 100 GSM8K test problems: one answer, a majority vote of three, and a three-agent debate (each
agent revises after reading the others' answers as text; `scripts/eval_debate.py`, `results_debate_cuda.json`):

<!-- table:debate -->
| Arm | GSM8K accuracy | Paired Δ vs single | Tokens generated |
| :--- | :---: | :---: | :---: |
| single | 0.72 ± 0.04 | +0.00 ± 0.00 | 23,537 |
| vote_r1 | 0.77 ± 0.04 | +0.05 ± 0.03 | 67,947 |
| debate | 0.79 ± 0.04 | +0.07 ± 0.03 | 123,775 |
<!-- /table -->

* **Debate helps by 7 points (paired, ± 3.3)** at 5.3× the tokens; most of it is the vote (+5).
* **The agents exchange text.** `MultiAgentCoordinator` has no path that hands one agent's KV cache to another, so a
  KV-exchange debate was not measured; building one would need the cross-model maps above, which transfer two thirds
  of a context at best.
