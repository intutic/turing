#include "turing_simd.hpp"
#include "turing_mmap.hpp"
#include "turing_paged_attention.hpp"
#include "turing_convolution_shared.hpp"
#include "turing_nbody_attention.hpp"
#include "turing_hex_bmu.hpp"
#include "turing_pso.hpp"
#include "turing_threadpool.hpp"
#include "turing_adam_kernel.hpp"
#include "turing_cpu_moe_kernel.hpp"
#include "turing_birkhoff.hpp"
#include "turing_dag_tree_mask.hpp"
#include "turing_sinkhorn_ot.hpp"
#include "turing_hierarchical.hpp"
#include "turing_serializer.hpp"
#include "turing_radix_trie.hpp"
#include "turing_apc_hash.hpp"
#include "turing_asynch_scheduler.hpp"
#include "turing_halo_exchange.hpp"
#include "turing_conv2d_shared.hpp"
#include "turing_nbody_recirculator.hpp"
#include "turing_pso_tuner.hpp"
#include "turing_hex_quantizer.hpp"
#include "turing_rope.hpp"
#include "turing_lru_cache.hpp"
#include "turing_paged_memory.hpp"
#include "turing_shannon_entropy.hpp"
#include "turing_matrix_pow.hpp"
#include "turing_persistent_reducer.hpp"
#include "turing_unified_memory.hpp"
#include "turing_pso_objectives.hpp"
#include "turing_laplacian_2d.hpp"
#include "turing_welford_anneal.hpp"
#include "turing_matryoshka_quadtree.hpp"
#include "turing_svd_quant.hpp"
#include "turing_ridge_solver.hpp"
#include "turing_latent_decode.hpp"


#include <pybind11/pybind11.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// PyBind wrapper for AVX2 / Portable dual-subspace quantization
void subspace_quantize_cpp(
    py::array_t<float, py::array::c_style | py::array::forcecast> input_arr,
    py::array_t<int8_t, py::array::c_style> content_out_arr,
    py::array_t<uint8_t, py::array::c_style> fluency_out_arr,
    float scale_content,
    float scale_fluency,
    int content_dim,
    int fluency_dim
) {
    py::buffer_info in_buf = input_arr.request();
    py::buffer_info c_buf = content_out_arr.request();
    py::buffer_info f_buf = fluency_out_arr.request();

    const float* in_ptr = static_cast<const float*>(in_buf.ptr);
    int8_t* c_ptr = static_cast<int8_t*>(c_buf.ptr);
    uint8_t* f_ptr = static_cast<uint8_t*>(f_buf.ptr);

    float inv_sc = 1.0f / (scale_content > 1e-8f ? scale_content : 1.0f);
    float inv_sf = 1.0f / (scale_fluency > 1e-8f ? scale_fluency : 1.0f);

    // 1. Content Subspace (INT8)
    for (int i = 0; i < content_dim; ++i) {
        float val = in_ptr[i] * inv_sc;
        float clamped = std::clamp(std::round(val), -128.0f, 127.0f);
        c_ptr[i] = static_cast<int8_t>(clamped);
    }

    // 2. Fluency Subspace (Packed INT4 nibbles into uint8)
    const float* f_in_ptr = in_ptr + content_dim;
    for (int i = 0; i < fluency_dim; i += 2) {
        float val0 = (f_in_ptr[i] * inv_sf) + 8.0f;
        float val1 = (f_in_ptr[i + 1] * inv_sf) + 8.0f;

        uint8_t v0 = static_cast<uint8_t>(std::clamp(std::round(val0), 0.0f, 15.0f));
        uint8_t v1 = static_cast<uint8_t>(std::clamp(std::round(val1), 0.0f, 15.0f));

        f_ptr[i / 2] = (v0 & 0x0F) | ((v1 & 0x0F) << 4);
    }
}

// PyBind wrapper for FP32 sparse GEMV
py::array_t<float> gemv_fp32_sparse_py(
    py::array_t<float, py::array::c_style> input_arr,
    py::array_t<float, py::array::c_style> weight_arr,
    uint32_t active_mask,
    py::array_t<float, py::array::c_style> bias_arr,
    int tile_size
) {
    py::buffer_info in_buf = input_arr.request();
    py::buffer_info w_buf = weight_arr.request();
    py::buffer_info b_buf = bias_arr.request();

    int in_features = static_cast<int>(in_buf.shape[0]);
    int out_features = static_cast<int>(w_buf.shape[0]);

    auto result = py::array_t<float>(out_features);
    py::buffer_info res_buf = result.request();

    const float* in_ptr = static_cast<const float*>(in_buf.ptr);
    const float* w_ptr = static_cast<const float*>(w_buf.ptr);
    const float* b_ptr = (b_buf.size > 0) ? static_cast<const float*>(b_buf.ptr) : nullptr;
    float* out_ptr = static_cast<float*>(res_buf.ptr);

    // Initialize output with bias or zero
    if (b_ptr != nullptr) {
        std::copy(b_ptr, b_ptr + out_features, out_ptr);
    } else {
        std::fill(out_ptr, out_ptr + out_features, 0.0f);
    }

    // Iterate over active tiles according to 32-bit mask
    int num_tiles = in_features / tile_size;
    for (int t = 0; t < num_tiles; ++t) {
        if (!((active_mask >> t) & 1)) continue;

        const float* i_tile = in_ptr + (t * tile_size);
        const float* w_tile = w_ptr + (t * tile_size * out_features);
        turing::gemv_fp32_tile(i_tile, w_tile, out_ptr, out_features, tile_size);
    }

    return result;
}

// PyBind wrapper for Cooperative Shared Memory 1D/2D Convolution
py::array_t<float> cooperative_shared_conv1d_py(
    py::array_t<float, py::array::c_style> input_arr,
    py::array_t<float, py::array::c_style> weight_arr,
    py::array_t<float, py::array::c_style> bias_arr,
    int stride,
    int padding
) {
    py::buffer_info in_buf = input_arr.request();
    py::buffer_info w_buf = weight_arr.request();
    py::buffer_info b_buf = bias_arr.request();

    int batch = static_cast<int>(in_buf.shape[0]);
    int in_channels = static_cast<int>(in_buf.shape[1]);
    int in_len = static_cast<int>(in_buf.shape[2]);

    int out_channels = static_cast<int>(w_buf.shape[0]);
    int kernel_size = static_cast<int>(w_buf.shape[2]);

    int out_len = (in_len + 2 * padding - kernel_size) / stride + 1;
    auto result = py::array_t<float>({batch, out_channels, out_len});
    py::buffer_info res_buf = result.request();

    const float* in_ptr = static_cast<const float*>(in_buf.ptr);
    const float* w_ptr = static_cast<const float*>(w_buf.ptr);
    const float* b_ptr = (b_buf.size > 0) ? static_cast<const float*>(b_buf.ptr) : nullptr;
    float* out_ptr = static_cast<float*>(res_buf.ptr);

    turing::cooperative_shared_conv1d(
        in_ptr, w_ptr, b_ptr, out_ptr,
        batch, in_channels, out_channels, in_len,
        kernel_size, stride, padding
    );

    return result;
}

// PyBind wrapper for Softened N-Body Attention
py::array_t<float> softened_nbody_attention_py(
    py::array_t<float, py::array::c_style> query_arr,
    py::array_t<float, py::array::c_style> key_arr,
    py::array_t<float, py::array::c_style> value_arr,
    float softening_sq,
    float scale
) {
    py::buffer_info q_buf = query_arr.request();
    py::buffer_info k_buf = key_arr.request();
    py::buffer_info v_buf = value_arr.request();

    int batch = static_cast<int>(q_buf.shape[0]);
    int heads = static_cast<int>(q_buf.shape[1]);
    int seq_q = static_cast<int>(q_buf.shape[2]);
    int head_dim = static_cast<int>(q_buf.shape[3]);
    int seq_k = static_cast<int>(k_buf.shape[2]);

    auto result = py::array_t<float>({batch, heads, seq_q, head_dim});
    py::buffer_info res_buf = result.request();

    turing::softened_nbody_attention_forward(
        static_cast<const float*>(q_buf.ptr),
        static_cast<const float*>(k_buf.ptr),
        static_cast<const float*>(v_buf.ptr),
        static_cast<float*>(res_buf.ptr),
        batch, heads, seq_q, seq_k, head_dim,
        softening_sq, scale
    );

    return result;
}

// PyBind wrapper for Hexagonal BMU Search
py::tuple hexagonal_bmu_search_py(
    py::array_t<float, py::array::c_style> activations_arr,
    py::array_t<float, py::array::c_style> codebook_arr
) {
    py::buffer_info act_buf = activations_arr.request();
    py::buffer_info cb_buf = codebook_arr.request();

    int batch = static_cast<int>(act_buf.shape[0]);
    int codebook_dim = static_cast<int>(act_buf.shape[1]);
    int total_cells = static_cast<int>(cb_buf.shape[0]);

    auto out_indices = py::array_t<int64_t>(batch);
    auto out_dists = py::array_t<float>(batch);

    py::buffer_info idx_buf = out_indices.request();
    py::buffer_info dist_buf = out_dists.request();

    turing::hexagonal_bmu_search(
        static_cast<const float*>(act_buf.ptr),
        static_cast<const float*>(cb_buf.ptr),
        static_cast<int64_t*>(idx_buf.ptr),
        static_cast<float*>(dist_buf.ptr),
        batch, codebook_dim, total_cells
    );

    return py::make_tuple(out_indices, out_dists);
}

// PyBind wrapper for Hexagonal Metric Distance
float hexagonal_distance_py(float u1, float v1, float u2, float v2) {
    return turing::hexagonal_distance(u1, v1, u2, v2);
}

// PyBind wrapper for Native Fused Adam Step (Fused High-Performance Kernel)
void fused_adam_step_py(
    py::array_t<float, py::array::c_style> param_arr,
    py::array_t<float, py::array::c_style> grad_arr,
    py::array_t<float, py::array::c_style> m_arr,
    py::array_t<float, py::array::c_style> v_arr,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    int timestep
) {
    py::buffer_info p_buf = param_arr.request();
    py::buffer_info g_buf = grad_arr.request();
    py::buffer_info m_buf = m_arr.request();
    py::buffer_info v_buf = v_arr.request();

    int dim = static_cast<int>(p_buf.size);

    turing::fused_adam_step(
        static_cast<float*>(p_buf.ptr),
        static_cast<const float*>(g_buf.ptr),
        static_cast<float*>(m_buf.ptr),
        static_cast<float*>(v_buf.ptr),
        dim, lr, beta1, beta2, epsilon, timestep
    );
}

// PyBind wrapper for Multi-Threaded CPU MoE GEMV (Fused High-Performance Kernel)
py::array_t<float> parallel_cpu_moe_gemv_py(
    py::array_t<float, py::array::c_style> input_arr,
    py::array_t<float, py::array::c_style> expert_weights_arr,
    py::array_t<int32_t, py::array::c_style> expert_indices_arr,
    py::array_t<float, py::array::c_style> routing_weights_arr
) {
    py::buffer_info in_buf = input_arr.request();
    py::buffer_info w_buf = expert_weights_arr.request();
    py::buffer_info idx_buf = expert_indices_arr.request();
    py::buffer_info rw_buf = routing_weights_arr.request();

    int batch = static_cast<int>(in_buf.shape[0]);
    int in_features = static_cast<int>(in_buf.shape[1]);
    int out_features = static_cast<int>(w_buf.shape[1]);
    int top_k = static_cast<int>(idx_buf.shape[1]);

    auto result = py::array_t<float>({batch, out_features});
    py::buffer_info res_buf = result.request();

    turing::parallel_cpu_moe_gemv(
        static_cast<const float*>(in_buf.ptr),
        static_cast<const float*>(w_buf.ptr),
        static_cast<const int32_t*>(idx_buf.ptr),
        static_cast<const float*>(rw_buf.ptr),
        static_cast<float*>(res_buf.ptr),
        batch, in_features, out_features, top_k
    );

    return result;
}

// Pillar 1: Birkhoff Manifold Projection
py::array_t<float> birkhoff_project_py(
    py::array_t<float, py::array::c_style> matrix_arr,
    int num_iterations,
    float eps
) {
    py::buffer_info buf = matrix_arr.request();
    int ndim = static_cast<int>(buf.ndim);
    int n = static_cast<int>(buf.shape[ndim - 1]);
    
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; ++i) {
        batch_size *= static_cast<int>(buf.shape[i]);
    }

    auto result = py::array_t<float>(buf.shape);
    py::buffer_info res_buf = result.request();

    turing::birkhoff_manifold_project(
        static_cast<const float*>(buf.ptr),
        static_cast<float*>(res_buf.ptr),
        batch_size,
        n,
        num_iterations,
        eps
    );

    return result;
}

// Pillar 2: DAG Tree Attention Mask Engine
py::array_t<float> build_dag_tree_mask_py(
    py::array_t<int32_t, py::array::c_style> parent_indices_arr
) {
    py::buffer_info buf = parent_indices_arr.request();
    int num_nodes = static_cast<int>(buf.size);

    auto result = py::array_t<float>({num_nodes, num_nodes});
    py::buffer_info res_buf = result.request();

    turing::build_dag_tree_mask_cpp(
        static_cast<const int32_t*>(buf.ptr),
        static_cast<float*>(res_buf.ptr),
        num_nodes
    );

    return result;
}

// Pillar 3: In-SRAM Entropic Optimal Transport (OT) KV Eviction
std::pair<py::array_t<int32_t>, py::array_t<float>> sinkhorn_ot_eviction_py(
    py::array_t<float, py::array::c_style> query_arr,
    py::array_t<float, py::array::c_style> key_arr,
    int budget,
    float epsilon,
    int num_iters
) {
    py::buffer_info q_buf = query_arr.request();
    py::buffer_info k_buf = key_arr.request();

    int m_queries = static_cast<int>(q_buf.shape[0]);
    int head_dim = static_cast<int>(q_buf.shape[1]);
    int n_keys = static_cast<int>(k_buf.shape[0]);

    int actual_budget = std::min(budget, n_keys);
    auto retained = py::array_t<int32_t>(actual_budget);
    auto mass = py::array_t<float>(n_keys);

    py::buffer_info ret_buf = retained.request();
    py::buffer_info mass_buf = mass.request();

    turing::sinkhorn_ot_eviction_cpp(
        static_cast<const float*>(q_buf.ptr),
        static_cast<const float*>(k_buf.ptr),
        static_cast<int32_t*>(ret_buf.ptr),
        static_cast<float*>(mass_buf.ptr),
        m_queries,
        n_keys,
        head_dim,
        actual_budget,
        epsilon,
        num_iters
    );

    return {retained, mass};
}

// Pillar 4: Hierarchical Sequence-Chunk Pooling (HCA & CSA)
py::array_t<float> hca_chunk_pool_py(
    py::array_t<float, py::array::c_style> input_arr,
    int chunk_size
) {
    py::buffer_info buf = input_arr.request();
    int seq_len = static_cast<int>(buf.shape[0]);
    int num_heads = static_cast<int>(buf.shape[1]);
    int head_dim = static_cast<int>(buf.shape[2]);

    int num_chunks = (seq_len + chunk_size - 1) / chunk_size;
    auto result = py::array_t<float>({num_chunks, num_heads, head_dim});
    py::buffer_info res_buf = result.request();

    turing::hca_chunk_pool_cpp(
        static_cast<const float*>(buf.ptr),
        static_cast<float*>(res_buf.ptr),
        seq_len,
        num_heads,
        head_dim,
        chunk_size
    );

    return result;
}

// Pillar 5: Zero-Overhead Binary Tensor Serializer
py::bytes serialize_tensor_int8_py(py::array_t<float, py::array::c_style> input_arr) {
    py::buffer_info buf = input_arr.request();
    std::vector<uint32_t> shape;
    for (auto d : buf.shape) shape.push_back(static_cast<uint32_t>(d));

    auto bytes_vec = turing::serialize_tensor_int8_cpp(
        static_cast<const float*>(buf.ptr),
        shape
    );

    return py::bytes(reinterpret_cast<const char*>(bytes_vec.data()), bytes_vec.size());
}

py::array_t<float> deserialize_tensor_int8_py(py::bytes data_bytes) {
    std::string str = data_bytes;
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(str.data());

    float scale = 1.0f;
    std::vector<uint32_t> shape;

    const uint8_t* temp_ptr = ptr + 1; // skip dtype (buffer starts directly at dtype)
    std::memcpy(&scale, temp_ptr, 4); temp_ptr += 4;
    uint32_t ndim = 0;
    std::memcpy(&ndim, temp_ptr, 4); temp_ptr += 4;
    std::vector<py::ssize_t> py_shape(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        uint32_t dim_val = 0;
        std::memcpy(&dim_val, temp_ptr, 4); temp_ptr += 4;
        py_shape[i] = static_cast<py::ssize_t>(dim_val);
    }

    auto result = py::array_t<float>(py_shape);
    py::buffer_info res_buf = result.request();

    turing::deserialize_tensor_int8_cpp(
        ptr,
        static_cast<float*>(res_buf.ptr),
        scale,
        shape
    );

    return result;
}

// Pillar 7: SIMD 64-Bit APC Hash Table
uint64_t apc_hash_mask_py(py::array_t<uint8_t, py::array::c_style> mask_arr) {
    py::buffer_info buf = mask_arr.request();
    return turing::apc_murmurhash64A(buf.ptr, buf.size);
}

// Component 1: Asynchronous Dynamic Master-Worker Task Scheduler
py::array_t<float> asynch_schedule_tasks_py(
    py::array_t<float, py::array::c_style> input_arr,
    float scale,
    int num_workers
) {
    py::buffer_info buf = input_arr.request();
    int num_tokens = static_cast<int>(buf.shape[0]);
    int dim = static_cast<int>(buf.shape[1]);

    auto result = py::array_t<float>({num_tokens, dim});
    py::buffer_info res_buf = result.request();

    turing::asynch_schedule_token_slices_cpp(
        static_cast<const float*>(buf.ptr),
        static_cast<float*>(res_buf.ptr),
        num_tokens,
        dim,
        scale,
        num_workers
    );

    return result;
}

// Component 2: 2D Spatial Mesh Halo Exchange Engine
std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>> halo_exchange_step_py(
    py::array_t<float, py::array::c_style> local_grid_arr,
    py::array_t<float, py::array::c_style> top_halo_in_arr,
    py::array_t<float, py::array::c_style> bottom_halo_in_arr,
    float diffusion_alpha
) {
    py::buffer_info g_buf = local_grid_arr.request();
    py::buffer_info t_buf = top_halo_in_arr.request();
    py::buffer_info b_buf = bottom_halo_in_arr.request();

    int height = static_cast<int>(g_buf.shape[0]);
    int width = static_cast<int>(g_buf.shape[1]);

    auto next_grid = py::array_t<float>({height, width});
    auto top_halo_out = py::array_t<float>(width);
    auto bottom_halo_out = py::array_t<float>(width);

    py::buffer_info n_buf = next_grid.request();
    py::buffer_info to_buf = top_halo_out.request();
    py::buffer_info bo_buf = bottom_halo_out.request();

    turing::halo_exchange_step_cpp(
        static_cast<const float*>(g_buf.ptr),
        static_cast<float*>(to_buf.ptr),
        static_cast<float*>(bo_buf.ptr),
        (t_buf.size > 0) ? static_cast<const float*>(t_buf.ptr) : nullptr,
        (b_buf.size > 0) ? static_cast<const float*>(b_buf.ptr) : nullptr,
        static_cast<float*>(n_buf.ptr),
        height,
        width,
        diffusion_alpha
    );

    return std::make_tuple(next_grid, top_halo_out, bottom_halo_out);
}

// Component 3: Cooperative Shared Memory 2D Convolution
py::array_t<float> cooperative_conv2d_shared_py(
    py::array_t<float, py::array::c_style> input_arr,
    py::array_t<float, py::array::c_style> weights_arr,
    py::array_t<float, py::array::c_style> bias_arr,
    int stride,
    int padding
) {
    py::buffer_info in_buf = input_arr.request();
    py::buffer_info w_buf = weights_arr.request();
    py::buffer_info b_buf = bias_arr.request();

    int in_channels = static_cast<int>(in_buf.shape[0]);
    int in_h = static_cast<int>(in_buf.shape[1]);
    int in_w = static_cast<int>(in_buf.shape[2]);

    int out_channels = static_cast<int>(w_buf.shape[0]);
    int kernel_h = static_cast<int>(w_buf.shape[2]);
    int kernel_w = static_cast<int>(w_buf.shape[3]);

    int out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_w) / stride + 1;

    auto result = py::array_t<float>({out_channels, out_h, out_w});
    py::buffer_info res_buf = result.request();

    turing::cooperative_conv2d_shared_cpp(
        static_cast<const float*>(in_buf.ptr),
        static_cast<const float*>(w_buf.ptr),
        (b_buf.size > 0) ? static_cast<const float*>(b_buf.ptr) : nullptr,
        static_cast<float*>(res_buf.ptr),
        in_channels,
        out_channels,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride,
        padding
    );

    return result;
}

// Component 4: Softened N-Body Multi-Agent Belief Recirculator
py::array_t<float> nbody_belief_recirculate_py(
    py::array_t<float, py::array::c_style> belief_states_arr,
    float softening_sq,
    float step_size
) {
    py::buffer_info buf = belief_states_arr.request();
    int num_agents = static_cast<int>(buf.shape[0]);
    int state_dim = static_cast<int>(buf.shape[1]);

    auto result = py::array_t<float>({num_agents, state_dim});
    py::buffer_info res_buf = result.request();

    turing::nbody_belief_recirculate_cpp(
        static_cast<const float*>(buf.ptr),
        static_cast<float*>(res_buf.ptr),
        num_agents,
        state_dim,
        softening_sq,
        step_size
    );

    return result;
}

// Component 5: Asynchronous Swarm Hyper-Tuner Engine (PSO)
std::vector<float> pso_optimize_hyperparams_py(
    int num_particles,
    int num_dims,
    int num_iterations,
    std::vector<float> lower_bounds,
    std::vector<float> upper_bounds,
    float w,
    float c1,
    float c2
) {
    return turing::pso_optimize_hyperparams_cpp(
        num_particles,
        num_dims,
        num_iterations,
        lower_bounds,
        upper_bounds,
        w,
        c1,
        c2
    );
}

// Component 6: Hexagonal Spatial Codebook Quantizer
std::pair<py::array_t<int32_t>, py::array_t<float>> hex_quantize_activations_py(
    py::array_t<float, py::array::c_style> input_arr,
    py::array_t<float, py::array::c_style> codebook_arr
) {
    py::buffer_info in_buf = input_arr.request();
    py::buffer_info cd_buf = codebook_arr.request();

    int num_vectors = static_cast<int>(in_buf.shape[0]);
    int dim = static_cast<int>(in_buf.shape[1]);
    int num_cells = static_cast<int>(cd_buf.shape[0]);

    auto bmu_indices = py::array_t<int32_t>(num_vectors);
    auto quantized = py::array_t<float>({num_vectors, dim});

    py::buffer_info bmu_buf = bmu_indices.request();
    py::buffer_info q_buf = quantized.request();

    turing::hex_quantize_activations_cpp(
        static_cast<const float*>(in_buf.ptr),
        static_cast<const float*>(cd_buf.ptr),
        static_cast<int32_t*>(bmu_buf.ptr),
        static_cast<float*>(q_buf.ptr),
        num_vectors,
        num_cells,
        dim
    );

    return {bmu_indices, quantized};
}

// Component 7: Fused In-Place RoPE Decoupler
py::array_t<float> fused_rope_transform_py(
    py::array_t<float, py::array::c_style> data_arr,
    float base,
    int pos_offset,
    bool is_inverse
) {
    py::buffer_info buf = data_arr.request();
    int seq_len = 1;
    int num_heads = 1;
    int head_dim = 1;

    if (buf.ndim == 3) {
        seq_len = static_cast<int>(buf.shape[0]);
        num_heads = static_cast<int>(buf.shape[1]);
        head_dim = static_cast<int>(buf.shape[2]);
    } else if (buf.ndim == 2) {
        seq_len = static_cast<int>(buf.shape[0]);
        head_dim = static_cast<int>(buf.shape[1]);
    }

    auto result = py::array_t<float>(buf.shape);
    py::buffer_info res_buf = result.request();
    std::memcpy(res_buf.ptr, buf.ptr, buf.size * sizeof(float));

    turing::fused_rope_transform_cpp(
        static_cast<float*>(res_buf.ptr),
        seq_len,
        num_heads,
        head_dim,
        base,
        pos_offset,
        is_inverse
    );

    return result;
}

// Component 8: Fused In-SRAM Shannon Entropy Kernel
py::array_t<float> compute_shannon_entropy_py(py::array_t<float, py::array::c_style> logits_arr) {
    py::buffer_info buf = logits_arr.request();
    if (buf.ndim == 1) {
        int vocab_size = static_cast<int>(buf.shape[0]);
        float ent = turing::compute_shannon_entropy_single(static_cast<const float*>(buf.ptr), vocab_size);
        auto res = py::array_t<float>(1);
        *static_cast<float*>(res.request().ptr) = ent;
        return res;
    } else {
        int batch_size = static_cast<int>(buf.shape[0]);
        int vocab_size = static_cast<int>(buf.shape[1]);
        auto ents = turing::compute_shannon_entropy_batch_cpp(static_cast<const float*>(buf.ptr), batch_size, vocab_size);
        auto res = py::array_t<float>(batch_size);
        std::memcpy(res.request().ptr, ents.data(), batch_size * sizeof(float));
        return res;
    }
}

// Component 9: Logarithmic Matrix Exponentiation Recurrence
int64_t matrix_power_transition_py(int64_t x0, int64_t x1, int64_t A, int64_t B, int64_t C, int64_t M, int64_t target_idx) {
    return turing::evaluate_recurrence_jump(x0, x1, A, B, C, M, target_idx);
}

// Component 10: Persistent Thread-Local OpenMP Reducer
py::array_t<float> persistent_parallel_reduce_py(py::array_t<float, py::array::c_style> thread_data_arr) {
    py::buffer_info buf = thread_data_arr.request();
    int num_threads = static_cast<int>(buf.shape[0]);
    int dim = static_cast<int>(buf.shape[1]);
    auto res_vec = turing::parallel_reduce_sum_cpp(static_cast<const float*>(buf.ptr), num_threads, dim);
    auto res = py::array_t<float>(dim);
    std::memcpy(res.request().ptr, res_vec.data(), dim * sizeof(float));
    return res;
}

// Component 11: Multi-Modal PSO Objectives
double evaluate_pso_objective_py(const std::string& name, py::array_t<double, py::array::c_style> x_arr) {
    py::buffer_info buf = x_arr.request();
    int d = static_cast<int>(buf.size);
    return turing::evaluate_pso_objective_cpp(name, static_cast<const double*>(buf.ptr), d);
}

// Component 12: 9-Point 2D Laplacian Stencil Step
py::array_t<float> laplacian_2d_step_py(py::array_t<float, py::array::c_style> in_grid_arr, float alpha) {
    py::buffer_info buf = in_grid_arr.request();
    int height = static_cast<int>(buf.shape[0]);
    int width = static_cast<int>(buf.shape[1]);
    auto out = py::array_t<float>({height, width});
    turing::laplacian_2d_step_cpp(
        static_cast<const float*>(buf.ptr),
        static_cast<float*>(out.request().ptr),
        height, width, alpha
    );
    return out;
}

// Component 13: Exponential Parameter Annealing
double exponential_decay_schedule_py(double init_val, double min_val, int64_t current_step, int64_t max_steps) {
    return turing::evaluate_exponential_decay(init_val, min_val, current_step, max_steps);
}

PYBIND11_MODULE(turing_csrc, m) {
    m.doc() = "Turing Engine C++20 SIMD and Memory-Mapped Acceleration Extension";

    m.def("subspace_quantize", &subspace_quantize_cpp,
          "Quantize input activations into INT8 content and packed INT4 fluency subspaces",
          py::arg("input_arr"), py::arg("content_out_arr"), py::arg("fluency_out_arr"),
          py::arg("scale_content"), py::arg("scale_fluency"),
          py::arg("content_dim"), py::arg("fluency_dim"));

    m.def("gemv_fp32_sparse", &gemv_fp32_sparse_py,
          "Execute FP32 sparse pointer-skipping GEMV",
          py::arg("input_arr"), py::arg("weight_arr"), py::arg("active_mask"),
          py::arg("bias_arr"), py::arg("tile_size") = 256);

    m.def("cooperative_shared_conv1d", &cooperative_shared_conv1d_py,
          "Execute Cooperative Shared Memory 1D Convolution (Spatial HPC Stencil Engine)",
          py::arg("input_arr"), py::arg("weight_arr"), py::arg("bias_arr"),
          py::arg("stride") = 1, py::arg("padding") = 0);

    m.def("softened_nbody_attention", &softened_nbody_attention_py,
          "Execute Softened N-Body All-to-All Attention (Spatial HPC Stencil Engine)",
          py::arg("query_arr"), py::arg("key_arr"), py::arg("value_arr"),
          py::arg("softening_sq") = 1e-4f, py::arg("scale") = 1.0f);

    m.def("hexagonal_bmu_search", &hexagonal_bmu_search_py,
          "Execute Parallel Hexagonal BMU Search (Spatial HPC Stencil Engine)",
          py::arg("activations_arr"), py::arg("codebook_arr"));

    m.def("hexagonal_distance", &hexagonal_distance_py,
          "Calculate Hexagonal Coordinate Metric Distance (Spatial HPC Stencil Engine)",
          py::arg("u1"), py::arg("v1"), py::arg("u2"), py::arg("v2"));

    m.def("fused_adam_step", &fused_adam_step_py,
          "Execute Native Fused In-SRAM Adam Step (Fused High-Performance Kernel)",
          py::arg("param_arr"), py::arg("grad_arr"),
          py::arg("m_arr"), py::arg("v_arr"),
          py::arg("lr"), py::arg("beta1"), py::arg("beta2"),
          py::arg("epsilon"), py::arg("timestep"));

    m.def("parallel_cpu_moe_gemv", &parallel_cpu_moe_gemv_py,
          "Execute Multi-Threaded CPU MoE GEMV via Persistent ThreadPool (Fused High-Performance Kernel)",
          py::arg("input_arr"), py::arg("expert_weights_arr"),
          py::arg("expert_indices_arr"), py::arg("routing_weights_arr"));

    // Pillar 1-7 Native Bindings
    m.def("birkhoff_project", &birkhoff_project_py,
          "Execute Native C++20 Birkhoff Manifold Projector (mHC)",
          py::arg("matrix_arr"), py::arg("num_iterations") = 20, py::arg("eps") = 1e-6f);

    m.def("build_dag_tree_mask", &build_dag_tree_mask_py,
          "Construct Additive DAG Tree Attention Mask in C++20",
          py::arg("parent_indices_arr"));

    m.def("sinkhorn_ot_eviction", &sinkhorn_ot_eviction_py,
          "Execute In-SRAM Entropic Optimal Transport (OT) KV Eviction",
          py::arg("query_arr"), py::arg("key_arr"), py::arg("budget"),
          py::arg("epsilon") = 0.05f, py::arg("num_iters") = 15);

    m.def("hca_chunk_pool", &hca_chunk_pool_py,
          "Execute Native Hierarchical Sequence-Chunk Pooling (HCA/CSA)",
          py::arg("input_arr"), py::arg("chunk_size") = 128);

    m.def("serialize_tensor_int8", &serialize_tensor_int8_py,
          "Serialize Tensor to Zero-Overhead INT8 Binary Buffer",
          py::arg("input_arr"));

    m.def("deserialize_tensor_int8", &deserialize_tensor_int8_py,
          "Deserialize Zero-Overhead INT8 Binary Buffer to Float Tensor",
          py::arg("data_bytes"));

    m.def("apc_hash_mask", &apc_hash_mask_py,
          "Compute 64-Bit MurmurHash3 for Attention Pattern Cache (APC)",
          py::arg("mask_arr"));

    // High-Performance Systems & Math Advanced HPC Bindings
    m.def("asynch_schedule_tasks", &asynch_schedule_tasks_py,
          "Execute Asynchronous Dynamic Master-Worker Task Scheduling (Asynchronous Master-Worker Engine)",
          py::arg("input_arr"), py::arg("scale") = 1.0f, py::arg("num_workers") = 4);

    m.def("halo_exchange_step", &halo_exchange_step_py,
          "Execute 2D Spatial Mesh Halo Exchange Step (Spatial HPC Stencil Engine)",
          py::arg("local_grid_arr"), py::arg("top_halo_in_arr"),
          py::arg("bottom_halo_in_arr"), py::arg("diffusion_alpha") = 0.25f);

    m.def("cooperative_conv2d_shared", &cooperative_conv2d_shared_py,
          "Execute Cooperative 2D Shared Memory Convolution (Spatial HPC Stencil Engine)",
          py::arg("input_arr"), py::arg("weights_arr"), py::arg("bias_arr"),
          py::arg("stride") = 1, py::arg("padding") = 0);

    m.def("nbody_belief_recirculate", &nbody_belief_recirculate_py,
          "Execute Softened N-Body Multi-Agent Belief Recirculation (Spatial HPC Stencil Engine)",
          py::arg("belief_states_arr"), py::arg("softening_sq") = 1e-4f, py::arg("step_size") = 0.05f);

    m.def("pso_optimize_hyperparams", &pso_optimize_hyperparams_py,
          "Execute Native C++20 PSO Hyper-Parameter Optimizer (Spatial HPC Stencil Engine)",
          py::arg("num_particles"), py::arg("num_dims"), py::arg("num_iterations"),
          py::arg("lower_bounds"), py::arg("upper_bounds"),
          py::arg("w") = 0.729f, py::arg("c1") = 1.494f, py::arg("c2") = 1.494f);

    m.def("hex_quantize_activations", &hex_quantize_activations_py,
          "Execute Hexagonal Spatial Codebook Quantization (Spatial HPC Stencil Engine)",
          py::arg("input_arr"), py::arg("codebook_arr"));

    // 4-Pillar Hot-Path Native Bindings
    m.def("fused_rope_transform", &fused_rope_transform_py,
          "Execute Native C++20 Fused In-Place RoPE Transform and Positional Decoupling",
          py::arg("data_arr"), py::arg("base") = 500000.0f,
          py::arg("pos_offset") = 0, py::arg("is_inverse") = false);

    m.def("compute_shannon_entropy", &compute_shannon_entropy_py,
          "Execute Native C++20 Fused Single-Pass Shannon Entropy from Unnormalized Logits",
          py::arg("logits_arr"));

    // New 6-Pillar High-Performance Systems & Math Advanced HPC Functions
    m.def("matrix_power_transition", &matrix_power_transition_py,
          "Execute Logarithmic Matrix Exponentiation Jump-Ahead (Fused High-Performance Kernel)",
          py::arg("x0"), py::arg("x1"), py::arg("A"), py::arg("B"), py::arg("C"), py::arg("M"), py::arg("target_idx"));

    m.def("persistent_parallel_reduce", &persistent_parallel_reduce_py,
          "Execute Thread-Local Persistent Parallel Reduction (Fused High-Performance Kernel)",
          py::arg("thread_data_arr"));

    m.def("evaluate_pso_objective", &evaluate_pso_objective_py,
          "Evaluate Non-Convex Benchmark Objective Surface (Spatial HPC Stencil Engine)",
          py::arg("name"), py::arg("x_arr"));

    m.def("laplacian_2d_stencil_step", &laplacian_2d_step_py,
          "Execute 9-Point 2D Laplacian Spatial Diffusion Stencil (Spatial HPC Stencil Engine)",
          py::arg("in_grid_arr"), py::arg("alpha") = 0.1f);

    m.def("exponential_decay_schedule", &exponential_decay_schedule_py,
          "Evaluate Exponential Parameter Decay Schedule (Spatial HPC Stencil Engine)",
          py::arg("init_val"), py::arg("min_val"), py::arg("current_step"), py::arg("max_steps"));

    // Fast C++20 LRU Expert Slot Cache Class
    py::class_<turing::LRUExpertCacheFast>(m, "LRUExpertCacheFast")
        .def(py::init<int>(), py::arg("num_slots") = 32)
        .def("contains", &turing::LRUExpertCacheFast::contains, py::arg("layer_idx"), py::arg("expert_idx"))
        .def("get_slot", &turing::LRUExpertCacheFast::get_slot, py::arg("layer_idx"), py::arg("expert_idx"))
        .def("allocate_or_evict_slot", [](turing::LRUExpertCacheFast& self, int layer_idx, int expert_idx) {
            int evicted_layer = -1;
            int evicted_expert = -1;
            int slot = self.allocate_or_evict_slot(layer_idx, expert_idx, evicted_layer, evicted_expert);
            return std::make_tuple(slot, evicted_layer, evicted_expert);
        }, py::arg("layer_idx"), py::arg("expert_idx"))
        .def_property_readonly("hit_rate", &turing::LRUExpertCacheFast::get_hit_rate)
        .def_property_readonly("hits", &turing::LRUExpertCacheFast::get_hits)
        .def_property_readonly("misses", &turing::LRUExpertCacheFast::get_misses)
        .def_property_readonly("used_slots", &turing::LRUExpertCacheFast::get_used_slots);

    // Hierarchical Virtual Memory Page Bitmap Allocator Class
    py::class_<turing::HierarchicalBitmapAllocator>(m, "HierarchicalBitmapAllocator")
        .def(py::init<int, int, int>(), py::arg("num_huge") = 64, py::arg("num_medium") = 128, py::arg("num_small") = 256)
        .def("allocate_prompt", &turing::HierarchicalBitmapAllocator::allocate_prompt, py::arg("prompt_len"))
        .def("free_block", &turing::HierarchicalBitmapAllocator::free_block, py::arg("tier"), py::arg("block_id"))
        .def("get_num_free", &turing::HierarchicalBitmapAllocator::get_num_free, py::arg("tier"));

    // Persistent Thread Reducer Class
    py::class_<turing::PersistentThreadReducer>(m, "PersistentThreadReducer")
        .def(py::init<int, int>(), py::arg("num_threads") = 4, py::arg("dim") = 256)
        .def("clear_all", &turing::PersistentThreadReducer::clear_all)
        .def_property_readonly("num_threads", &turing::PersistentThreadReducer::get_num_threads)
        .def_property_readonly("dim", &turing::PersistentThreadReducer::get_dim);

    // Unified Memory Pool Class
    py::class_<turing::UnifiedMemoryPool>(m, "UnifiedMemoryPool")
        .def(py::init<size_t>(), py::arg("capacity_bytes") = 64 * 1024 * 1024)
        .def("allocate_slab", &turing::UnifiedMemoryPool::allocate_slab, py::arg("bytes"))
        .def("reset", &turing::UnifiedMemoryPool::reset)
        .def_property_readonly("capacity", &turing::UnifiedMemoryPool::get_capacity)
        .def_property_readonly("used", &turing::UnifiedMemoryPool::get_used)
        .def_property_readonly("free", &turing::UnifiedMemoryPool::get_free);

    // Online Streaming Welford Statistical Accumulator Class
    py::class_<turing::StreamingWelford>(m, "StreamingWelford")
        .def(py::init<>())
        .def("update", &turing::StreamingWelford::update, py::arg("x"))
        .def("reset", &turing::StreamingWelford::reset)
        .def_property_readonly("count", &turing::StreamingWelford::get_count)
        .def_property_readonly("mean", &turing::StreamingWelford::get_mean)
        .def_property_readonly("variance", &turing::StreamingWelford::get_variance)
        .def_property_readonly("stdev", &turing::StreamingWelford::get_stdev);

    // Pillar 6: Radix Trie Index
    py::class_<turing::RadixTrieIndex>(m, "RadixTrieIndex")
        .def(py::init<>())
        .def("insert", &turing::RadixTrieIndex::insert, py::arg("tokens"))
        .def("match_longest_prefix", [](const turing::RadixTrieIndex& self, const std::vector<int32_t>& tokens) {
            int32_t matched_len = 0;
            int32_t node_id = self.match_longest_prefix(tokens, matched_len);
            return std::make_pair(node_id, matched_len);
        }, py::arg("tokens"));

    py::class_<turing::TuringPagedAttentionEngine>(m, "TuringPagedAttentionEngine")
        .def(py::init<int, int, int, int>(),
             py::arg("num_heads"), py::arg("head_dim"), py::arg("block_size"), py::arg("num_blocks"))
        .def("forward_selective_attention", [](
            turing::TuringPagedAttentionEngine& self,
            py::array_t<float, py::array::c_style> query,
            py::array_t<int, py::array::c_style> block_table,
            uint32_t active_page_mask
        ) {
            py::buffer_info q_buf = query.request();
            py::buffer_info bt_buf = block_table.request();

            int num_heads = static_cast<int>(q_buf.shape[0]);
            int head_dim = static_cast<int>(q_buf.shape[1]);
            int num_logical_pages = static_cast<int>(bt_buf.shape[0]);

            auto out = py::array_t<float>({num_heads, head_dim});
            py::buffer_info out_buf = out.request();

            self.forward_selective_attention(
                static_cast<const float*>(q_buf.ptr),
                static_cast<const int*>(bt_buf.ptr),
                num_logical_pages,
                active_page_mask,
                static_cast<float*>(out_buf.ptr)
            );

            return out;
        }, py::arg("query"), py::arg("block_table"), py::arg("active_page_mask"));

    // Matryoshka Sliced GEMV & Quadtree Candidate Generator
    m.def("generate_matryoshka_quadtree", [](
        py::array_t<float, py::array::c_style> hidden_state_arr,
        py::array_t<float, py::array::c_style> draft_weight_arr,
        py::array_t<float, py::array::c_style> spatial_proj_arr,
        int slice_width
    ) {
        py::buffer_info h_buf = hidden_state_arr.request();
        py::buffer_info w_buf = draft_weight_arr.request();
        py::buffer_info s_buf = spatial_proj_arr.request();

        int hidden_dim = static_cast<int>(h_buf.shape[0]);
        int vocab_size = static_cast<int>(w_buf.shape[0]);

        auto res = turing::generate_matryoshka_quadtree_cpp(
            static_cast<const float*>(h_buf.ptr),
            static_cast<const float*>(w_buf.ptr),
            static_cast<const float*>(s_buf.ptr),
            hidden_dim,
            vocab_size,
            slice_width
        );

        int num_nodes = static_cast<int>(res.token_ids.size());
        auto tok_arr = py::array_t<int32_t>(num_nodes);
        auto parent_arr = py::array_t<int32_t>(num_nodes);
        auto mask_arr = py::array_t<float>({num_nodes, num_nodes});

        std::memcpy(tok_arr.request().ptr, res.token_ids.data(), num_nodes * sizeof(int32_t));
        std::memcpy(parent_arr.request().ptr, res.parent_indices.data(), num_nodes * sizeof(int32_t));
        std::memcpy(mask_arr.request().ptr, res.dag_mask.data(), num_nodes * num_nodes * sizeof(float));

        return py::make_tuple(tok_arr, parent_arr, mask_arr);
    }, "Native C++20 Matryoshka Sliced GEMV & Quadtree Generator");

    // Fused SVD INT8 Quantizer
    m.def("fused_svd_int8_quant", [](
        py::array_t<float, py::array::c_style> k_input_arr,
        py::array_t<float, py::array::c_style> u_proj_arr
    ) {
        py::buffer_info k_buf = k_input_arr.request();
        py::buffer_info u_buf = u_proj_arr.request();

        int seq_len = static_cast<int>(k_buf.shape[0]);
        int head_dim = static_cast<int>(k_buf.shape[1]);
        int rank = static_cast<int>(u_buf.shape[1]);

        auto q_out = py::array_t<int8_t>({seq_len, rank});
        auto scale_out = py::array_t<float>({seq_len, 1});

        turing::fused_svd_int8_quant_cpp(
            static_cast<const float*>(k_buf.ptr),
            static_cast<const float*>(u_buf.ptr),
            static_cast<int8_t*>(q_out.request().ptr),
            static_cast<float*>(scale_out.request().ptr),
            seq_len,
            head_dim,
            rank
        );

        return py::make_tuple(q_out, scale_out);
    }, "Native C++20 Fused SVD INT8 Quantizer");

    // Fused INT8 Dequantize & SVD Reconstruct GEMM
    m.def("fused_int8_dequant_svd_recon", [](
        py::array_t<int8_t, py::array::c_style> q_int8_arr,
        py::array_t<float, py::array::c_style> scale_arr,
        py::array_t<float, py::array::c_style> u_proj_arr
    ) {
        py::buffer_info q_buf = q_int8_arr.request();
        py::buffer_info s_buf = scale_arr.request();
        py::buffer_info u_buf = u_proj_arr.request();

        int seq_len = static_cast<int>(q_buf.shape[0]);
        int rank = static_cast<int>(q_buf.shape[1]);
        int head_dim = static_cast<int>(u_buf.shape[0]);

        auto out_recon = py::array_t<float>({seq_len, head_dim});

        turing::fused_int8_dequant_svd_recon_cpp(
            static_cast<const int8_t*>(q_buf.ptr),
            static_cast<const float*>(s_buf.ptr),
            static_cast<const float*>(u_buf.ptr),
            static_cast<float*>(out_recon.request().ptr),
            seq_len,
            head_dim,
            rank
        );

        return out_recon;
    }, "Native C++20 Fused INT8 Dequantize & SVD Reconstruct GEMM");

    // Fused Ridge Forward
    m.def("fused_ridge_forward", [](
        py::array_t<float, py::array::c_style> x_source_arr,
        py::array_t<float, py::array::c_style> w_flat_arr,
        py::array_t<float, py::array::c_style> b_flat_arr
    ) {
        py::buffer_info x_buf = x_source_arr.request();
        py::buffer_info w_buf = w_flat_arr.request();
        py::buffer_info b_buf = b_flat_arr.request();

        int n_tokens = static_cast<int>(x_buf.shape[0]);
        int in_dim = static_cast<int>(x_buf.shape[1]);
        int out_features = static_cast<int>(w_buf.shape[1]);

        auto out_arr = py::array_t<float>({n_tokens, out_features});

        turing::fused_ridge_forward_cpp(
            static_cast<const float*>(x_buf.ptr),
            static_cast<const float*>(w_buf.ptr),
            b_buf.size > 0 ? static_cast<const float*>(b_buf.ptr) : nullptr,
            static_cast<float*>(out_arr.request().ptr),
            n_tokens,
            in_dim,
            out_features
        );

        return out_arr;
    }, "Native C++20 Fused Ridge Representation Projection");

    // Native C++20 Latent Flash-Decode (Mode-B)
    m.def("latent_decode_cpu", [](
        py::array_t<float, py::array::c_style> qp_arr,
        py::array_t<int8_t, py::array::c_style> ck_arr,
        py::array_t<float, py::array::c_style> sk_arr,
        py::array_t<int8_t, py::array::c_style> cv_arr,
        py::array_t<float, py::array::c_style> sv_arr,
        float scale
    ) {
        py::buffer_info q_buf = qp_arr.request();
        py::buffer_info ck_buf = ck_arr.request();
        py::buffer_info sk_buf = sk_arr.request();
        py::buffer_info cv_buf = cv_arr.request();
        py::buffer_info sv_buf = sv_arr.request();

        int B = static_cast<int>(ck_buf.shape[0]);
        int N = static_cast<int>(ck_buf.shape[1]);
        int R = static_cast<int>(ck_buf.shape[2]);
        int total_q = static_cast<int>(q_buf.shape[0]); // B * NKV * GRP
        int GRP = (total_q >= B) ? (total_q / B) : 1;
        int NKV = 1;

        auto out_arr = py::array_t<float>({total_q, R});

        turing::latent_decode_avx2(
            static_cast<const float*>(q_buf.ptr),
            static_cast<const int8_t*>(ck_buf.ptr),
            static_cast<const float*>(sk_buf.ptr),
            static_cast<const int8_t*>(cv_buf.ptr),
            static_cast<const float*>(sv_buf.ptr),
            static_cast<float*>(out_arr.request().ptr),
            B, NKV, GRP, R, N, scale
        );

        return out_arr;
    }, "Native C++20 AVX2 Latent Flash-Decode Attention (Mode-B)");
}



