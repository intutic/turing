/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

// Vendored from vLLM v0.31.0, csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu (Apache-2.0, see csrc/marlin/LICENSE
// and NOTICE). Modified for Turing Engine: the classic torch extension API (at::Tensor, TORCH_CHECK, pybind11) instead
// of torch's stable ABI; only the W4A16 AWQ-INT4 path (fp16/bf16 activations, kU4 weights with integer zero points,
// sm_80+); the kernel selector comes from csrc/marlin/kernel_configs.h instead of the generated kernel_selector.h.
// The thread configs, shared-memory sizing, exec-config choice and launch are vLLM's.

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

#include "kernel.h"
#include "kernel_configs.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

namespace MARLIN_NAMESPACE_NAME {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int num_bits, int group_size,
                          int stages) {
  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  int tb_scales = tb_groups * tb_n * 2;
  return tb_scales * stages;
}

int get_kernel_cache_size(thread_config_t const& th_config, bool m_block_size_8,
                          int thread_m_blocks, int prob_m, int prob_n,
                          int prob_k, int num_bits, int group_size, int has_zp,
                          int is_zp_float, bool is_a_8bit, int stages) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;

  // shm size for block_sorted_ids/rd_block_sorted_ids/block_topk_weights
  // both of them requires tb_m * 4 bytes (tb_m * int32 or tb_m * float32)
  int sh_block_meta_size = tb_m * 16;
  int sh_a_size = stages * (tb_m * tb_k) * (is_a_8bit ? 1 : 2);
  int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size =
      (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);

  int sh_s_size = get_scales_cache_size(th_config, prob_m, prob_n, prob_k,
                                        num_bits, group_size, stages);
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float)
      sh_zp_size = sh_s_size;
    else if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }

  int total_size =
      tmp_size + sh_a_size + sh_s_size + sh_zp_size + sh_block_meta_size;

  return total_size;
}

bool is_valid_config(thread_config_t const& th_config, bool m_block_size_8,
                     int thread_m_blocks, int prob_m, int prob_n, int prob_k,
                     int num_bits, int group_size, int has_zp, int is_zp_float,
                     bool is_a_8bit, int stages, int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  // Check that pipeline fits into cache
  int cache_size = get_kernel_cache_size(
      th_config, m_block_size_8, thread_m_blocks, prob_m, prob_n, prob_k,
      num_bits, group_size, has_zp, is_zp_float, is_a_8bit, stages);
  return cache_size <= max_shared_mem;
}

#define TURING_MARLIN_SELECT(AT, GB, TH, MB, NB, KB, M8)                     \
  if (a_type == vllm::AT && group_blocks == GB && threads == TH &&           \
      thread_m_blocks == MB && thread_n_blocks == NB &&                      \
      thread_k_blocks == KB && m_block_size_8 == M8)                         \
    return b_type == vllm::kU4B8                                             \
               ? TURING_MARLIN_KERNEL_B8(AT, GB, TH, MB, NB, KB, M8)         \
               : TURING_MARLIN_KERNEL(AT, GB, TH, MB, NB, KB, M8);

MarlinFuncPtr get_marlin_kernel(const vllm::ScalarType a_type,
                                const vllm::ScalarType b_type,
                                const vllm::ScalarType c_type,
                                const vllm::ScalarType s_type,
                                int thread_m_blocks, int thread_n_blocks,
                                int thread_k_blocks, bool m_block_size_8,
                                bool has_zp, int group_blocks, int threads,
                                bool is_zp_float, int stages) {
  // Only the instantiations in csrc/marlin/kernel_configs.h exist; anything else is MarlinDefault (skipped or
  // rejected).
  if (!(((b_type == vllm::kU4 && has_zp) || (b_type == vllm::kU4B8 && !has_zp)) &&
        !is_zp_float && stages == 4 && c_type == a_type && s_type == a_type))
    return MarlinDefault;
  TURING_MARLIN_ALL(TURING_MARLIN_SELECT)
  return MarlinDefault;
}

exec_config_t determine_exec_config(
    const vllm::ScalarType& a_type, const vllm::ScalarType& b_type,
    const vllm::ScalarType& c_type, const vllm::ScalarType& s_type, int prob_m,
    int prob_n, int prob_k, int num_experts, int top_k, int thread_m_blocks,
    bool m_block_size_8, int num_bits, int group_size, bool has_zp,
    bool is_zp_float, bool is_a_8bit, int stages, int max_shared_mem, int sms) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1
                                        ? large_batch_thread_configs
                                        : small_batch_thread_configs;
  int thread_configs_size =
      thread_m_blocks > 1
          ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
          : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  int count = 0;
  constexpr int device_max_reg_size = 255 * 1024;
  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(th_config, m_block_size_8, thread_m_blocks, prob_m,
                         prob_n, prob_k, num_bits, group_size, has_zp,
                         is_zp_float, is_a_8bit, stages,
                         max_shared_mem - 512)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(
        th_config, m_block_size_8, thread_m_blocks, prob_m, prob_n, prob_k,
        num_bits, group_size, has_zp, is_zp_float, is_a_8bit, stages);

    int group_blocks = group_size == -1 ? -1 : (group_size / 16);

    auto kernel = get_marlin_kernel(
        a_type, b_type, c_type, s_type, thread_m_blocks,
        th_config.thread_n / 16, th_config.thread_k / 16, m_block_size_8,
        has_zp, group_blocks, th_config.num_threads, is_zp_float, stages);

    if (kernel == MarlinDefault) continue;

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel);
    int reg_size = max(attr.numRegs, 1) * th_config.num_threads * 4;
    int allow_count = min(device_max_reg_size / reg_size,
                          max_shared_mem / (cache_size + 1536));
    if (thread_m_blocks == 1)
      allow_count = max(min(allow_count, 4), 1);
    else
      allow_count = max(min(allow_count, 2), 1);

    if (prob_n / th_config.thread_n * prob_m * top_k * 4 < sms * allow_count) {
      allow_count =
          max(prob_n / th_config.thread_n * prob_m * top_k * 4 / sms, 1);
    }

    if (allow_count > count) {
      count = allow_count;
      exec_cfg = {count, th_config};
    };
  }

  return exec_cfg;
}

void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, void* b_s, void* g_s, void* zp,
               void* sorted_token_ids, void* expert_ids,
               void* num_tokens_past_padded, void* topk_weights,
               int moe_block_size, int num_experts, int top_k,
               bool mul_topk_weights, int prob_m, int prob_n, int prob_k,
               void* workspace, vllm::ScalarType const& a_type,
               vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
               vllm::ScalarType const& s_type, bool has_bias, bool has_zp,
               int group_size, int dev, cudaStream_t stream, int thread_k,
               int thread_n, int sms, int blocks_per_sm, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float) {
  int thread_m_blocks = div_ceil(moe_block_size, 16);
  bool m_block_size_8 = moe_block_size == 8;
  bool is_a_8bit = a_type.size_bits() == 8;

  TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [",
                  prob_m, ", ", prob_n, ", ", prob_k, "]");

  int group_blocks;
  if (group_size == -1) {
    group_blocks = -1;
  } else {
    group_blocks = group_size / 16;
    TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                    " is not divisible by group_blocks = ", group_blocks);
  }

  int num_bits = b_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;
  const int4* bias_ptr = (const int4*)b_bias;
  const float* a_s_ptr = (const float*)a_s;
  const int4* b_s_ptr = (const int4*)b_s;
  const float* g_s_ptr = (const float*)g_s;
  const int4* zp_ptr = (const int4*)zp;
  const int32_t* sorted_token_ids_ptr = (const int32_t*)sorted_token_ids;
  const int32_t* expert_ids_ptr = (const int32_t*)expert_ids;
  const int32_t* num_tokens_past_padded_ptr =
      (const int32_t*)num_tokens_past_padded;
  const float* topk_weights_ptr = (const float*)topk_weights;
  int* locks = (int*)workspace;

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(max_shared_mem > 0);

  int major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         dev);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         dev);
  TORCH_CHECK(major_capability * 10 + minor_capability >= 75,
                  "marlin kernel only support Turing or newer GPUs.");
  int stages = 4;
  if (major_capability == 7 && minor_capability == 5) {
    stages = 2;
    TORCH_CHECK(a_type == vllm::kFloat16 || a_type == vllm::kS8,
                    "Turing only support FP16 or INT8 activation.");
  }
  if (a_type == vllm::kFE4M3fn) {
    TORCH_CHECK(major_capability * 10 + minor_capability >= 89,
                    "FP8 only support Ada Lovelace or newer GPUs.");
    TORCH_CHECK(
        major_capability * 10 + minor_capability == 89 ||
            major_capability == 12,
        "Marlin W4A8-FP8 only support SM89 or SM12x device (It is slower than "
        "Marlin W4A16 on other devices).");
  }

  // Set thread config
  exec_config_t exec_cfg;
  thread_config_t thread_tfg;
  if (thread_k != -1 && thread_n != -1) {
    thread_tfg = thread_config_t{thread_k, thread_n, thread_k * thread_n / 64};
    if (blocks_per_sm == -1) blocks_per_sm = 1;
    exec_cfg = exec_config_t{blocks_per_sm, thread_tfg};
    TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
                    " is not divisible by thread_n = ", thread_n);
    TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
                    " is not divisible by thread_k = ", thread_k);
  } else {
    // Auto config
    exec_cfg = determine_exec_config(
        a_type, b_type, c_type, s_type, prob_m, prob_n, prob_k, num_experts,
        top_k, thread_m_blocks, m_block_size_8, num_bits, group_size, has_zp,
        is_zp_float, is_a_8bit, stages, max_shared_mem, sms);
    thread_tfg = exec_cfg.tb_cfg;
  }

  int num_threads = thread_tfg.num_threads;
  thread_k = thread_tfg.thread_k;
  thread_n = thread_tfg.thread_n;
  int blocks = sms * exec_cfg.blocks_per_sm;
  if (exec_cfg.blocks_per_sm > 1)
    max_shared_mem = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;

  TORCH_CHECK(
      is_valid_config(thread_tfg, m_block_size_8, thread_m_blocks, prob_m,
                      prob_n, prob_k, num_bits, group_size, has_zp, is_zp_float,
                      is_a_8bit, stages, max_shared_mem),
      "Invalid thread config: thread_m_blocks = ", thread_m_blocks,
      ", thread_k = ", thread_tfg.thread_k,
      ", thread_n = ", thread_tfg.thread_n,
      ", num_threads = ", thread_tfg.num_threads, " for MKN = [", prob_m, ", ",
      prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
      ", group_size = ", group_size, ", has_zp = ", has_zp,
      ", is_zp_float = ", is_zp_float, ", max_shared_mem = ", max_shared_mem);

  int sh_cache_size = get_kernel_cache_size(
      thread_tfg, m_block_size_8, thread_m_blocks, prob_m, prob_n, prob_k,
      num_bits, group_size, has_zp, is_zp_float, is_a_8bit, stages);

  auto kernel =
      get_marlin_kernel(a_type, b_type, c_type, s_type, thread_m_blocks,
                        thread_n_blocks, thread_k_blocks, m_block_size_8,
                        has_zp, group_blocks, num_threads, is_zp_float, stages);

  if (kernel == MarlinDefault) {
    TORCH_CHECK(false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n,
                    ", ", prob_k, "]", ", group_size = ", group_size,
                    ", thread_m_blocks = ", thread_m_blocks,
                    ", thread_n_blocks = ", thread_n_blocks,
                    ", thread_k_blocks = ", thread_k_blocks,
                    ", num_bits = ", num_bits);
  }

  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       max_shared_mem);
  // avoid ">>>" being formatted to "> > >"
  // clang-format off
  kernel<<<blocks, num_threads, max_shared_mem, stream>>>(
      A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr,
      sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
      topk_weights_ptr, top_k, mul_topk_weights, prob_m, prob_n, prob_k, locks,
      has_bias, use_atomic_add, use_fp32_reduce);
  // clang-format on
}

}  // namespace MARLIN_NAMESPACE_NAME

// y[pair] = dequant(B[expert of pair]) a[token of pair] for the token-expert pairs listed block by block in
// sorted_token_ids (vLLM's moe_align_block_size layout): a [size_m, size_k], b_q_weight [E, size_k / 16, size_n * 2]
// int32 Marlin tiles, b_scales [E, groups, size_n] in a's dtype, b_zeros [E, groups, size_n / 8] int32 (packed kU4
// zero points), sorted_token_ids / expert_ids / num_tokens_past_padded int32, topk_weights fp32. Returns
// c [size_m * top_k, size_n] in a's dtype; a row of c is a pair id (token * top_k + choice).
at::Tensor moe_wna16_marlin_gemm(
    const at::Tensor& a, const at::Tensor& b_q_weight,
    const at::Tensor& b_scales, const at::Tensor& b_zeros,
    const at::Tensor& workspace, const at::Tensor& sorted_token_ids,
    const at::Tensor& expert_ids, const at::Tensor& num_tokens_past_padded,
    const at::Tensor& topk_weights, int64_t moe_block_size, int64_t top_k,
    bool mul_topk_weights, int64_t size_m, int64_t size_n, int64_t size_k) {
  TORCH_CHECK(a.scalar_type() == at::ScalarType::Half ||
                  a.scalar_type() == at::ScalarType::BFloat16,
              "moe_wna16_marlin_gemm: a must be float16 or bfloat16");
  vllm::ScalarType a_type = a.scalar_type() == at::ScalarType::Half
                                ? vllm::kFloat16
                                : vllm::kBFloat16;
  // no zero points given: every one is 8, the kernel without them (kU4B8, as for a symmetric GPTQ checkpoint)
  bool has_zp = b_zeros.numel() > 0;
  vllm::ScalarType b_type = has_zp ? vllm::kU4 : vllm::kU4B8;
  vllm::ScalarType c_type = a_type;
  vllm::ScalarType s_type = a_type;
  bool const is_zp_float = false;
  bool const use_atomic_add = false;  // vLLM's fused_marlin_moe passes these
  bool const use_fp32_reduce = true;

  int pack_factor = 32 / b_type.size_bits();
  int num_experts = b_q_weight.size(0);

  if (moe_block_size != 8) {
    TORCH_CHECK(moe_block_size % 16 == 0,
                "unsupported moe_block_size=", moe_block_size);
    TORCH_CHECK(moe_block_size >= 16 && moe_block_size <= 64,
                "unsupported moe_block_size=", moe_block_size);
  }

  // Verify A
  TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
              ", size_k = ", size_k);

  // Verify B
  TORCH_CHECK(size_k % MARLIN_NAMESPACE_NAME::tile_size == 0, "size_k = ",
              size_k, " is not divisible by tile_size = ",
              MARLIN_NAMESPACE_NAME::tile_size);
  TORCH_CHECK((size_k / MARLIN_NAMESPACE_NAME::tile_size) == b_q_weight.size(1),
              "Shape mismatch: b_q_weight.size(1) = ", b_q_weight.size(1),
              ", size_k = ", size_k,
              ", tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  TORCH_CHECK(b_q_weight.size(2) % MARLIN_NAMESPACE_NAME::tile_size == 0,
              "b_q_weight.size(2) = ", b_q_weight.size(2),
              " is not divisible by tile_size = ",
              MARLIN_NAMESPACE_NAME::tile_size);
  int actual_size_n =
      (b_q_weight.size(2) / MARLIN_NAMESPACE_NAME::tile_size) * pack_factor;
  TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
              ", actual_size_n = ", actual_size_n);
  TORCH_CHECK(b_q_weight.scalar_type() == at::ScalarType::Int,
              "b_q_weight must be int32");

  // Verify device and strides
  TORCH_CHECK(a.is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.is_contiguous(), "A is not contiguous");
  // rows of A are loaded as int4 (16 bytes)
  TORCH_CHECK(((uint64_t)a.data_ptr()) % 16 == 0, "A must aligned to 16 bytes");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");
  TORCH_CHECK(b_scales.is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");
  TORCH_CHECK(b_scales.scalar_type() == a.scalar_type(),
              "b_scales must have a's dtype");
  TORCH_CHECK(sorted_token_ids.is_cuda() && sorted_token_ids.is_contiguous() &&
                  sorted_token_ids.scalar_type() == at::ScalarType::Int,
              "sorted_token_ids must be a contiguous int32 CUDA tensor");
  TORCH_CHECK(expert_ids.is_cuda() && expert_ids.is_contiguous() &&
                  expert_ids.scalar_type() == at::ScalarType::Int,
              "expert_ids must be a contiguous int32 CUDA tensor");
  TORCH_CHECK(num_tokens_past_padded.is_cuda() &&
                  num_tokens_past_padded.scalar_type() == at::ScalarType::Int &&
                  num_tokens_past_padded.numel() >= 1,
              "num_tokens_past_padded must be an int32 CUDA tensor");
  TORCH_CHECK(topk_weights.is_cuda() && topk_weights.is_contiguous() &&
                  topk_weights.scalar_type() == at::ScalarType::Float,
              "topk_weights must be a contiguous float32 CUDA tensor");
  TORCH_CHECK(sorted_token_ids.numel() % moe_block_size == 0 &&
                  expert_ids.numel() >= sorted_token_ids.numel() / moe_block_size,
              "sorted_token_ids must hold whole blocks, with an expert id each");

  // sms: number of SMs to use for the kernel
  int sms = -1;
  const int dev = a.get_device();
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

  // Alloc buffers
  const c10::cuda::OptionalCUDAGuard device_guard(a.device());
  at::Tensor c = at::empty({size_m * top_k, size_n}, a.options());
  auto f32 = a.options().dtype(at::ScalarType::Float);

  // Alloc C tmp buffer that is going to be used for the global reduce
  // max num of threadblocks is sms * 4
  long max_c_tmp_size = min(
      (long)size_n * sorted_token_ids.size(0),
      (long)sms * 4 * moe_block_size * MARLIN_NAMESPACE_NAME::max_thread_n);
  if (moe_block_size == 8) max_c_tmp_size *= 2;
  at::Tensor c_tmp = at::empty({max_c_tmp_size}, f32);

  // Detect group size.
  TORCH_CHECK(b_scales.dim() == 3, "b_scales rank = ", b_scales.dim(),
              " is not 3");
  TORCH_CHECK(b_scales.size(2) == size_n, "b_scales dim 2 = ",
              b_scales.size(2), " is not size_n = ", size_n);
  int num_groups = b_scales.size(1);
  int group_size = -1;
  if (num_groups > 1) {
    TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                ", is not divisible by b_scales.size(1) = ", b_scales.size(1));
    group_size = size_k / num_groups;
  }

  // Verify b_zeros (packed kU4 zero points, (E, num_groups, size_n / 8) int32)
  if (has_zp) {
    TORCH_CHECK(b_zeros.is_cuda(), "b_zeros is not on GPU");
    TORCH_CHECK(b_zeros.is_contiguous(), "b_zeros is not contiguous");
    TORCH_CHECK(b_zeros.scalar_type() == at::ScalarType::Int,
                "b_zeros must be int32");
    TORCH_CHECK(b_zeros.dim() == 3, "b_zeros rank = ", b_zeros.dim(),
                " is not 3");
    TORCH_CHECK(b_zeros.size(1) == num_groups, "b_zeros dim 1 = ",
                b_zeros.size(1), " is not num_groups = ", num_groups);
    TORCH_CHECK(b_zeros.size(2) == size_n / pack_factor, "b_zeros dim 2 = ",
                b_zeros.size(2), " is not size_n / pack_factor = ",
                size_n / pack_factor);
  }
  bool has_bias = false;
  at::Tensor b_bias = at::empty({0}, a.options());
  at::Tensor a_scales = at::empty({0}, f32);
  at::Tensor global_scale = at::empty({0}, f32);

  // Verify workspace size
  TORCH_CHECK(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0, "size_n = ",
              size_n, ", is not divisible by min_thread_n = ",
              MARLIN_NAMESPACE_NAME::min_thread_n);
  int max_n_tiles = size_n / MARLIN_NAMESPACE_NAME::min_thread_n;
  int min_workspace_size = min(
      max_n_tiles * (int)(sorted_token_ids.size(0) / moe_block_size), sms * 4);
  TORCH_CHECK(workspace.is_cuda() &&
                  workspace.scalar_type() == at::ScalarType::Int,
              "workspace must be an int32 CUDA tensor");
  TORCH_CHECK(workspace.numel() >= min_workspace_size,
              "workspace.numel = ", workspace.numel(),
              " is below min_workspace_size = ", min_workspace_size);

  MARLIN_NAMESPACE_NAME::marlin_mm(
      a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(), c_tmp.data_ptr(),
      b_bias.data_ptr(), a_scales.data_ptr(), b_scales.data_ptr(),
      global_scale.data_ptr(), b_zeros.data_ptr(), sorted_token_ids.data_ptr(),
      expert_ids.data_ptr(), num_tokens_past_padded.data_ptr(),
      topk_weights.data_ptr(), moe_block_size, num_experts, top_k,
      mul_topk_weights, size_m, size_n, size_k, workspace.data_ptr(), a_type,
      b_type, c_type, s_type, has_bias, has_zp, group_size, dev,
      at::cuda::getCurrentCUDAStream(dev), -1, -1, sms, -1, use_atomic_add,
      use_fp32_reduce, is_zp_float);

  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Marlin W4A16 MoE GEMM (vendored from vLLM v0.31.0, Apache-2.0)";
  m.def("moe_wna16_marlin_gemm", &moe_wna16_marlin_gemm,
        "c[pair] = dequant(b[expert]) a[token]: (a, b_q_weight, b_scales, b_zeros, workspace, sorted_token_ids, "
        "expert_ids, num_tokens_past_padded, topk_weights, moe_block_size, top_k, mul_topk_weights, size_m, size_n, "
        "size_k)");
}
