# Decision log: CPU-first inference

## CF1 Kernel base: ggml's CPU kernels inside Turing — 2026-11-27
- Decision: Turing's C++ engine calls llama.cpp's ggml CPU kernels (MIT, notices kept) for its matrix products; Turing's
  own work goes into expert streaming, CPU speculative decoding, sparsity, ternary integration and serving.
- Rationale: ggml already has ARM repacking, KleidiAI SME2, AMX and ternary types; matching them from scratch would take
  months before any new idea could be measured. User chose it over writing every kernel or patching llama.cpp.

## CF1a Model code too: libllama, not bare ggml — 2026-11-27
- Decision: Turing's CPU backend links libllama (llama.cpp's model graphs on ggml), pinned to commit 171527522 under
  `third_party/`, instead of Turing-written graphs on bare ggml.
- Rationale: libllama already runs Qwen3, Qwen3-MoE and gpt-oss with ARM repacking, KleidiAI and AMX, and exposes the
  eval callback Wave 2's expert read-ahead hooks into. User confirmed after the Wave 1-2 design.

## CF2 First target: one user (batch 1) — 2026-11-27
- Decision: single-user decode first; multi-user CPU serving is the last wave.
- Rationale: batch 1 is bandwidth bound, where a CPU can approach a GPU; many users are compute bound, where the research
  puts a two-socket server at 15-40% of one H100.

## CF3 Where measurements run: cloud only — 2026-11-27
- Decision: no benchmark runs on the user's Mac. ARM numbers on a rented ARM VM (GCP C4A, Axion / Neoverse V2: NEON,
  SVE2, i8mm, bf16), x86 numbers on a rented Xeon 6 VM (GCP C4, AMX); each VM created only with the user's OK, and it
  deletes itself when done. Unit tests and small smoke checks on the Mac stay allowed.
- Rationale: the user's choice (heavy runs slow their Mac, as in the 2026-11-16 note). Consequence: the SPEC's M4 Pro
  targets are measured on the cloud ARM VM instead, and the M4 numbers stay unmeasured.

## CF3a Mac and SanDisk measurements allowed — 2026-11-27
- Decision: the user's Mac (M4 Pro, 48 GB) and the SanDisk Extreme Pro on it are measured too, beside the cloud VMs;
  each Mac run with the user's go-ahead, while the Mac is otherwise idle.
- Rationale: the user's choice ("test on my mac as well apart from cloud vm", "Yes, measure first"). A model bigger than
  RAM needs no memory cap there: gpt-oss-120b (63 GB) is already larger than 48 GB.

## CF4 Lossy modes: opt-in, quality printed — 2026-11-27
- Decision: ternary, activation sparsity and fewer-experts modes are built as switches that print their measured quality
  cost (delta NLL with standard error); exact output stays the default.
- Rationale: user's choice; matches the SPEC's "exact by default" constraint.

## CF5 Wave 5 kernel changes as a patch file — 2026-11-27
- Decision: activation sparsity's changes to llama.cpp's matrix kernels live as a patch under `third_party/patches/`,
  applied to the pinned submodule at build time, behind an opt-in flag that prints its quality cost.
- Rationale: user's choice over an intutic fork of llama.cpp or skipping the wave; the patch stays visible and is rebased
  when the pin moves.
