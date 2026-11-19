// Turing Engine: the Marlin kernels this build instantiates, replacing vLLM v0.31.0's generate_kernels.py (jinja2).
// Only its AWQ-INT4 entry (b_type kU4, integer zero points) for a_type = c_type = s_type in {kFloat16, kBFloat16},
// group_blocks in {4, 8} (groups of 64 and 128), 4 pipeline stages (sm_80+). Per (a_type, group_blocks) these are
// exactly the kernels generate_kernels.py emits: THREAD_CONFIGS x THREAD_M_BLOCKS {0.5, 1, 2, 3, 4} after its
// 256-thread filter ((128, 128, 256) only for m_blocks <= 1, (64, 256, 256) only for m_blocks > 1).
// X(AT, GB, threads, thread_m_blocks, thread_n_blocks, thread_k_blocks, m_block_size_8)
#pragma once

#define TURING_MARLIN_SHAPES(X, AT, GB)    \
  X(AT, GB, 256, 1, 8, 8, true)            \
  X(AT, GB, 128, 1, 8, 4, true)            \
  X(AT, GB, 128, 1, 4, 8, true)            \
  X(AT, GB, 256, 1, 8, 8, false)           \
  X(AT, GB, 128, 1, 8, 4, false)           \
  X(AT, GB, 128, 1, 4, 8, false)           \
  X(AT, GB, 256, 2, 16, 4, false)          \
  X(AT, GB, 128, 2, 8, 4, false)           \
  X(AT, GB, 128, 2, 4, 8, false)           \
  X(AT, GB, 256, 3, 16, 4, false)          \
  X(AT, GB, 128, 3, 8, 4, false)           \
  X(AT, GB, 128, 3, 4, 8, false)           \
  X(AT, GB, 256, 4, 16, 4, false)          \
  X(AT, GB, 128, 4, 8, 4, false)           \
  X(AT, GB, 128, 4, 4, 8, false)

#define TURING_MARLIN_ALL(X)              \
  TURING_MARLIN_SHAPES(X, kFloat16, 4)    \
  TURING_MARLIN_SHAPES(X, kFloat16, 8)    \
  TURING_MARLIN_SHAPES(X, kBFloat16, 4)   \
  TURING_MARLIN_SHAPES(X, kBFloat16, 8)

#define TURING_MARLIN_KERNEL(AT, GB, TH, MB, NB, KB, M8)                                              \
  Marlin<vllm::AT.id(), vllm::kU4.id(), vllm::AT.id(), vllm::AT.id(), TH, MB, NB, KB, M8, 4, GB, false>

#define TURING_MARLIN_INSTANTIATE(AT, GB, TH, MB, NB, KB, M8) \
  template __global__ void TURING_MARLIN_KERNEL(AT, GB, TH, MB, NB, KB, M8)(MARLIN_KERNEL_PARAMS);
