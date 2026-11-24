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

// Vendored from vLLM v0.31.0, csrc/libtorch_stable/quantization/marlin/marlin.cu (Apache-2.0, see LICENSE and
// NOTICE). Modified for Turing Engine: the classic torch extension API (at::Tensor, TORCH_CHECK, pybind11) instead
// of torch's stable ABI; only the W4A16 AWQ-INT4 path (fp16/bf16 activations, kU4 weights with integer zero points,
// sm_80+); the kernel selector comes from kernel_configs.h instead of the generated kernel_selector.h. The thread
// configs, shared-memory sizing, M splitting and launch are vLLM's.

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

#include "kernel.h"
#include "kernel_configs.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace marlin {

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

int get_kernel_cache_size(thread_config_t const& th_config, int thread_m_blocks,
                          int prob_m, int prob_n, int prob_k, int num_bits,
                          int group_size, int has_zp, bool is_zp_float,
                          bool is_a_8bit, int stages) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;
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

  int total_size = tmp_size + sh_a_size + sh_s_size + sh_zp_size;

  return total_size;
}

bool is_valid_config(thread_config_t const& th_config, int thread_m_blocks,
                     int prob_m, int prob_n, int prob_k, int num_bits,
                     int group_size, int has_zp, bool is_zp_float,
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
      th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits, group_size,
      has_zp, is_zp_float, is_a_8bit, stages);
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
  // Only the instantiations in kernel_configs.h exist; anything else is MarlinDefault (skipped or rejected).
  if (!(((b_type == vllm::kU4 && has_zp) || (b_type == vllm::kU4B8 && !has_zp)) &&
        !is_zp_float && stages == 4 && c_type == a_type && s_type == a_type))
    return MarlinDefault;
  TURING_MARLIN_ALL(TURING_MARLIN_SELECT)
  return MarlinDefault;
}

exec_config_t determine_exec_config(
    const vllm::ScalarType& a_type, const vllm::ScalarType& b_type,
    const vllm::ScalarType& c_type, const vllm::ScalarType& s_type, int prob_m,
    int prob_n, int prob_k, int thread_m_blocks, bool m_block_size_8,
    int num_bits, int group_size, bool has_zp, bool is_zp_float, int is_a_8bit,
    int stages, int max_shared_mem, int sms) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1
                                        ? large_batch_thread_configs
                                        : small_batch_thread_configs;
  int thread_configs_size =
      thread_m_blocks > 1
          ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
          : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(th_config, thread_m_blocks, prob_m, prob_n, prob_k,
                         num_bits, group_size, has_zp, is_zp_float, is_a_8bit,
                         stages, max_shared_mem - 512)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(
        th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits,
        group_size, has_zp, is_zp_float, is_a_8bit, stages);

    int group_blocks = group_size == -1 ? -1 : group_size / 16;

    auto kernel = get_marlin_kernel(
        a_type, b_type, c_type, s_type, thread_m_blocks,
        th_config.thread_n / 16, th_config.thread_k / 16, m_block_size_8,
        has_zp, group_blocks, th_config.num_threads, is_zp_float, stages);

    if (kernel == MarlinDefault) continue;

    return {1, th_config};
  }

  return exec_cfg;
}

void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, void* b_s, void* g_s, void* zp, int prob_m,
               int prob_n, int prob_k, int lda, void* workspace,
               vllm::ScalarType const& a_type, vllm::ScalarType const& b_type,
               vllm::ScalarType const& c_type, vllm::ScalarType const& s_type,
               bool has_bias, bool has_zp, int num_groups, int group_size,
               int dev, cudaStream_t stream, int thread_k_init,
               int thread_n_init, int sms, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float) {
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
  // This build has only the 4-stage (sm_80+) kernels; vLLM's sm_75 2-stage ones are not vendored.
  TORCH_CHECK(major_capability * 10 + minor_capability >= 80,
              "this Marlin build supports Ampere (sm_80) or newer GPUs only.");
  int stages = 4;

  int max_par = 16;
  if (prob_n <= 4096) max_par = 16 * 8;
  int max_shared_mem_new = max_shared_mem;
  int rest_m = prob_m;
  int max_thread_m_blocks = 4;
  while (rest_m) {
    int par_count = rest_m / (max_thread_m_blocks * 16);
    if (par_count > max_par) par_count = max_par;
    int prob_m_split =
        par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

    int thread_k = thread_k_init;
    int thread_n = thread_n_init;

    int thread_m_blocks = min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
    int m_block_size_8 = prob_m_split <= 8 && a_type.size_bits() == 16;

    // Set thread config
    exec_config_t exec_cfg;
    thread_config_t thread_tfg;
    if (thread_k != -1 && thread_n != -1) {
      thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
      exec_cfg = exec_config_t{1, thread_tfg};
      TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
                  " is not divisible by thread_n = ", thread_n);
      TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
                  " is not divisible by thread_k = ", thread_k);
    } else {
      // Auto config
      exec_cfg = determine_exec_config(
          a_type, b_type, c_type, s_type, prob_m_split, prob_n, prob_k,
          thread_m_blocks, m_block_size_8, num_bits, group_size, has_zp,
          is_zp_float, is_a_8bit, stages, max_shared_mem, sms);
      thread_tfg = exec_cfg.tb_cfg;
      if (thread_tfg.thread_n != -1) {
        if (prob_n / thread_tfg.thread_n *
                div_ceil(prob_m_split, thread_m_blocks * 16) * 4 <=
            sms) {
          if (is_valid_config({128, 64, 128}, thread_m_blocks, prob_m_split,
                              prob_n, prob_k, num_bits, group_size, has_zp,
                              is_zp_float, is_a_8bit, stages,
                              max_shared_mem_new)) {
            thread_tfg = {128, 64, 128};
            exec_cfg = {1, thread_tfg};
          }
        }
      }

      if (thread_tfg.thread_k == -1 && max_thread_m_blocks > 1) {
        max_thread_m_blocks--;
        continue;
      }
    }

    int num_threads = thread_tfg.num_threads;
    thread_k = thread_tfg.thread_k;
    thread_n = thread_tfg.thread_n;
    int blocks = sms * exec_cfg.blocks_per_sm;
    if (exec_cfg.blocks_per_sm > 1)
      max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

    TORCH_CHECK(
        is_valid_config(thread_tfg, thread_m_blocks, prob_m_split, prob_n,
                        prob_k, num_bits, group_size, has_zp, is_zp_float,
                        is_a_8bit, stages, max_shared_mem_new),
        "Invalid thread config: thread_m_blocks = ", thread_m_blocks,
        ", thread_k = ", thread_tfg.thread_k,
        ", thread_n = ", thread_tfg.thread_n,
        ", num_threads = ", thread_tfg.num_threads, " for MKN = [", prob_m,
        ", ", prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
        ", prob_m_split = ", prob_m_split, ", group_size = ", group_size,
        ", has_zp = ", has_zp, ", is_zp_float = ", is_zp_float,
        ", stages = ", stages, ", max_shared_mem_new = ", max_shared_mem_new);

    auto kernel = get_marlin_kernel(
        a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
        thread_k_blocks, m_block_size_8, has_zp, group_blocks, num_threads,
        is_zp_float, stages);

    if (kernel == MarlinDefault) {
      TORCH_CHECK(
          false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n, ", ",
          prob_k, "]", ", num_groups = ", num_groups,
          ", group_size = ", group_size, ", prob_m_split = ", prob_m_split,
          ", thread_m_blocks = ", thread_m_blocks,
          ", thread_n_blocks = ", thread_n_blocks,
          ", thread_k_blocks = ", thread_k_blocks,
          ", num_threads = ", num_threads, ", num_bits = ", num_bits);
    }

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_new);

    bool part_use_atomic_add =
        use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;

    // avoid ">>>" being formatted to "> > >"
    // clang-format off
    kernel<<<blocks, num_threads, max_shared_mem_new, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr,
        prob_m_split, prob_n, prob_k, lda, locks, has_bias, part_use_atomic_add,
        use_fp32_reduce, max_shared_mem_new);
    // clang-format on
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    A_ptr += prob_m_split * (lda / (is_a_8bit ? 16 : 8));
    a_s_ptr += prob_m_split;
    C_ptr += prob_m_split * (prob_n / 8);
    rest_m -= prob_m_split;
  }
}

}  // namespace marlin

// vLLM's marlin_gemm with its W4A16 AWQ-INT4 arguments only: b_type is kU4 with integer zero points (b_zeros
// required, is_zp_float false), no activation scales, no global scale, c allocated here.
at::Tensor marlin_gemm(const at::Tensor& a, const at::Tensor& b_q_weight,
                       std::optional<at::Tensor> const& b_bias_or_none,
                       const at::Tensor& b_scales, const at::Tensor& b_zeros,
                       const at::Tensor& workspace, int64_t size_m, int64_t size_n,
                       int64_t size_k, bool use_atomic_add,
                       bool use_fp32_reduce) {
  TORCH_CHECK(a.scalar_type() == at::ScalarType::Half ||
                  a.scalar_type() == at::ScalarType::BFloat16,
              "marlin_gemm: a must be float16 or bfloat16");
  vllm::ScalarType a_type = a.scalar_type() == at::ScalarType::Half
                                ? vllm::kFloat16
                                : vllm::kBFloat16;
  // no zero points given: every one is 8, the kernel without them (kU4B8, as for a symmetric GPTQ checkpoint)
  bool has_zp = b_zeros.numel() > 0;
  vllm::ScalarType b_type = has_zp ? vllm::kU4 : vllm::kU4B8;
  vllm::ScalarType c_type = a_type;
  vllm::ScalarType s_type = a_type;
  bool const is_zp_float = false;

  int pack_factor = 32 / b_type.size_bits();

  // Verify A
  TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
              ", size_k = ", size_k);

  // Verify B
  TORCH_CHECK(size_k % MARLIN_NAMESPACE_NAME::tile_size == 0, "size_k = ",
              size_k, " is not divisible by tile_size = ",
              MARLIN_NAMESPACE_NAME::tile_size);
  TORCH_CHECK((size_k / MARLIN_NAMESPACE_NAME::tile_size) == b_q_weight.size(0),
              "Shape mismatch: b_q_weight.size(0) = ", b_q_weight.size(0),
              ", size_k = ", size_k,
              ", tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  TORCH_CHECK(b_q_weight.size(1) % MARLIN_NAMESPACE_NAME::tile_size == 0,
              "b_q_weight.size(1) = ", b_q_weight.size(1),
              " is not divisible by tile_size = ",
              MARLIN_NAMESPACE_NAME::tile_size);
  int actual_size_n =
      (b_q_weight.size(1) / MARLIN_NAMESPACE_NAME::tile_size) * pack_factor;
  TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
              ", actual_size_n = ", actual_size_n);
  TORCH_CHECK(b_q_weight.scalar_type() == at::ScalarType::Int,
              "b_q_weight must be int32");

  // Verify device and strides
  TORCH_CHECK(a.is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.stride(1) == 1, "A.stride(1) is not 1");
  // We use int4 (16 bytes) to load A, so A must aligned to 16 bytes
  TORCH_CHECK(a.stride(0) % 8 == 0, "A.stride(0) must divisible by 8");
  TORCH_CHECK(((uint64_t)a.data_ptr()) % 16 == 0, "A must aligned to 16 bytes");

  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  TORCH_CHECK(b_scales.is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");
  TORCH_CHECK(b_scales.scalar_type() == a.scalar_type(),
              "b_scales must have a's dtype");

  // thread_k / thread_n: left to the auto config (-1), as vLLM does
  int thread_k = -1;
  int thread_n = -1;
  int sms = -1;
  const int device_index = a.get_device();
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_index);

  // Alloc buffers
  const c10::cuda::OptionalCUDAGuard device_guard(a.device());
  at::Tensor c = at::empty({size_m, size_n}, a.options());
  if (size_m == 0) return c;

  // Alloc C tmp buffer that is going to be used for the global reduce
  at::Tensor c_tmp;
  auto f32 = a.options().dtype(at::ScalarType::Float);
  if (use_fp32_reduce) {
    int max_m_block_size = (size_m + 16 - 1) / 16 * 16;
    max_m_block_size = min(max_m_block_size, 64);
    int max_c_tmp_size =
        sms * max_m_block_size * MARLIN_NAMESPACE_NAME::max_thread_n;
    c_tmp = at::empty({max_c_tmp_size}, f32);
  } else {
    c_tmp = at::empty({0}, f32);
  }

  // Detect group size.
  int num_groups = -1;
  int group_size = -1;

  int rank = b_scales.dim();
  TORCH_CHECK(rank == 2, "b_scales rank = ", rank, " is not 2");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales dim 1 = ",
              b_scales.size(1), " is not size_n = ", size_n);
  num_groups = b_scales.size(0);

  if (num_groups > 1) {
    TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                ", is not divisible by b_scales.size(0) = ", b_scales.size(0));
    group_size = size_k / num_groups;
  } else {
    group_size = -1;
  }

  bool has_bias = b_bias_or_none.has_value();
  at::Tensor b_bias;
  if (has_bias) {
    b_bias = b_bias_or_none.value();
    TORCH_CHECK(b_bias.is_cuda(), "b_bias is not on GPU");
    TORCH_CHECK(b_bias.is_contiguous(), "b_bias is not contiguous");
    TORCH_CHECK(b_bias.size(0) == size_n, "b_bias.size(0) != size_n");
    TORCH_CHECK(b_bias.stride(0) == 1, "b_bias.stride(0) != 1");
    TORCH_CHECK(b_bias.scalar_type() == a.scalar_type(),
                "b_bias must have a's dtype");
  } else {
    b_bias = at::empty({0}, a.options());
  }

  // Verify b_zeros (packed kU4 zero points, (num_groups, size_n / 8) int32)
  if (has_zp) {
    TORCH_CHECK(b_zeros.is_cuda(), "b_zeros is not on GPU");
    TORCH_CHECK(b_zeros.is_contiguous(), "b_zeros is not contiguous");
    TORCH_CHECK(b_zeros.scalar_type() == at::ScalarType::Int,
                "b_zeros must be int32");
    TORCH_CHECK(b_zeros.dim() == 2, "b_zeros rank = ", b_zeros.dim(),
                " is not 2");
    TORCH_CHECK(b_zeros.size(0) == num_groups, "b_zeros dim 0 = ",
                b_zeros.size(0), " is not num_groups = ", num_groups);
    TORCH_CHECK(b_zeros.size(1) == size_n / pack_factor, "b_zeros dim 1 = ",
                b_zeros.size(1), " is not size_n / pack_factor = ",
                size_n / pack_factor);
  }

  // Verify workspace size
  TORCH_CHECK(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0, "size_n = ",
              size_n, ", is not divisible by min_thread_n = ",
              MARLIN_NAMESPACE_NAME::min_thread_n);

  int min_workspace_size = sms;
  TORCH_CHECK(workspace.is_cuda() && workspace.scalar_type() == at::ScalarType::Int,
              "workspace must be an int32 CUDA tensor");
  TORCH_CHECK(workspace.numel() >= min_workspace_size,
              "workspace.numel = ", workspace.numel(),
              " is below min_workspace_size = ", min_workspace_size);

  marlin::marlin_mm(
      a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(), c_tmp.data_ptr(),
      b_bias.data_ptr(), nullptr, b_scales.data_ptr(), nullptr,
      b_zeros.data_ptr(), size_m, size_n, size_k, a.stride(0),
      workspace.data_ptr(), a_type, b_type, c_type, s_type, has_bias, has_zp,
      num_groups, group_size, device_index,
      at::cuda::getCurrentCUDAStream(device_index), thread_k, thread_n, sms,
      use_atomic_add, use_fp32_reduce, is_zp_float);

  return c;
}

at::Tensor awq_marlin_repack(const at::Tensor& b_q_weight, int64_t size_k,
                             int64_t size_n);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Marlin W4A16 GEMM (vendored from vLLM v0.31.0, Apache-2.0)";
  m.def("marlin_gemm", &marlin_gemm,
        "y = a @ dequant(b)^T: (a, b_q_weight, b_bias, b_scales, b_zeros, workspace, size_m, size_n, size_k, "
        "use_atomic_add, use_fp32_reduce)");
  m.def("awq_marlin_repack", &awq_marlin_repack,
        "AWQ int32 [size_k, size_n / 8] -> Marlin int32 [size_k / 16, size_n * 2]");
}
