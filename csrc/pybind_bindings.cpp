#include "turing_simd.hpp"
#include "turing_dag_tree_mask.hpp"
#include "turing_hierarchical.hpp"
#include "turing_halo_exchange.hpp"
#include "turing_nbody_recirculator.hpp"
#include "turing_rope.hpp"
#include "turing_paged_memory.hpp"
#include "turing_shannon_entropy.hpp"
#include "turing_matryoshka_quadtree.hpp"
#include "turing_ridge_solver.hpp"
#include "turing_linear_recurrence.hpp"
#include "turing_fast_hash.hpp"
#include "turing_dilated_conv1d.hpp"
#include "turing_lora_gemv.hpp"
#include "turing_elastic_budget.hpp"
#include "turing_prefix_router.hpp"
#include "turing_spec_verifier.hpp"
#include "turing_fast_hash_tensor.hpp"
#include "turing_sampler.hpp"
#include "turing_json_fast_scan.hpp"
#include "turing_gguf_simd.hpp"
#include "turing_model_cpp.hpp"
#include "turing_parallel_reader.hpp"
#include "turing_cufile.hpp"
#include "turing_safetensors_fast_header.hpp"
#include "turing_traffic_manager.hpp"
#include "turing_prefix_match.hpp"
#include "turing_routing_history.hpp"
#include "turing_gated_zero_identity.hpp"

#include <pybind11/pybind11.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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

PYBIND11_MODULE(turing_csrc, m) {
    m.doc() = "Turing Engine C++20 SIMD and Memory-Mapped Acceleration Extension";

    m.def("build_dag_tree_mask", &build_dag_tree_mask_py,
          "Construct Additive DAG Tree Attention Mask in C++20",
          py::arg("parent_indices_arr"));

    m.def("hca_chunk_pool", &hca_chunk_pool_py,
          "Execute Native Hierarchical Sequence-Chunk Pooling (HCA/CSA)",
          py::arg("input_arr"), py::arg("chunk_size") = 128);

    m.def("halo_exchange_step", &halo_exchange_step_py,
          "Execute 2D Spatial Mesh Halo Exchange Step (Spatial HPC Stencil Engine)",
          py::arg("local_grid_arr"), py::arg("top_halo_in_arr"),
          py::arg("bottom_halo_in_arr"), py::arg("diffusion_alpha") = 0.25f);

    m.def("nbody_belief_recirculate", &nbody_belief_recirculate_py,
          "Execute Softened N-Body Multi-Agent Belief Recirculation (Spatial HPC Stencil Engine)",
          py::arg("belief_states_arr"), py::arg("softening_sq") = 1e-4f, py::arg("step_size") = 0.05f);

    // 4-Pillar Hot-Path Native Bindings
    m.def("fused_rope_transform", &fused_rope_transform_py,
          "Execute Native C++20 Fused In-Place RoPE Transform and Positional Decoupling",
          py::arg("data_arr"), py::arg("base") = 500000.0f,
          py::arg("pos_offset") = 0, py::arg("is_inverse") = false);

    m.def("compute_shannon_entropy", &compute_shannon_entropy_py,
          "Execute Native C++20 Fused Single-Pass Shannon Entropy from Unnormalized Logits",
          py::arg("logits_arr"));

    // Hierarchical Virtual Memory Page Bitmap Allocator Class
    py::class_<turing::HierarchicalBitmapAllocator>(m, "HierarchicalBitmapAllocator")
        .def(py::init<int, int, int>(), py::arg("num_huge") = 64, py::arg("num_medium") = 128, py::arg("num_small") = 256)
        .def("allocate_prompt", &turing::HierarchicalBitmapAllocator::allocate_prompt, py::arg("prompt_len"))
        .def("free_block", &turing::HierarchicalBitmapAllocator::free_block, py::arg("tier"), py::arg("block_id"))
        .def("get_num_free", &turing::HierarchicalBitmapAllocator::get_num_free, py::arg("tier"));

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

        size_t num_nodes = res.token_ids.size();
        auto tok_arr = py::array_t<int32_t>(static_cast<py::ssize_t>(num_nodes));
        auto parent_arr = py::array_t<int32_t>(static_cast<py::ssize_t>(num_nodes));
        auto mask_arr = py::array_t<float>({static_cast<py::ssize_t>(num_nodes), static_cast<py::ssize_t>(num_nodes)});

        std::memcpy(tok_arr.request().ptr, res.token_ids.data(), num_nodes * sizeof(int32_t));
        std::memcpy(parent_arr.request().ptr, res.parent_indices.data(), num_nodes * sizeof(int32_t));
        std::memcpy(mask_arr.request().ptr, res.dag_mask.data(), res.dag_mask.size() * sizeof(float));

        return py::make_tuple(tok_arr, parent_arr, mask_arr);
    }, "Native C++20 Matryoshka Sliced GEMV & Quadtree Generator");

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

    // Native C++20 Linear Recurrent Attention Step (Decode L=1)
    m.def("linear_recurrence_step_cpu", [](
        py::array_t<float, py::array::c_style> q_arr,
        py::array_t<float, py::array::c_style> k_arr,
        py::array_t<float, py::array::c_style> v_arr,
        py::array_t<float, py::array::c_style> state_arr,
        float decay
    ) {
        py::buffer_info q_buf = q_arr.request();
        py::buffer_info k_buf = k_arr.request();
        py::buffer_info v_buf = v_arr.request();
        py::buffer_info s_buf = state_arr.request();

        size_t B = static_cast<size_t>(q_buf.shape[0]);
        size_t H = static_cast<size_t>(q_buf.shape[1]);
        size_t D = static_cast<size_t>(q_buf.shape[2]);

        auto out_arr = py::array_t<float>({static_cast<py::ssize_t>(B), static_cast<py::ssize_t>(H), static_cast<py::ssize_t>(D)});
        auto next_state_arr = py::array_t<float>({static_cast<py::ssize_t>(B), static_cast<py::ssize_t>(H), static_cast<py::ssize_t>(D), static_cast<py::ssize_t>(D)});

        // Copy current state into next_state_arr
        size_t total_state_elements = B * H * D * D;
        std::memcpy(next_state_arr.request().ptr, s_buf.ptr, total_state_elements * sizeof(float));

        turing::linear_recurrence_step_avx2(
            static_cast<const float*>(q_buf.ptr),
            static_cast<const float*>(k_buf.ptr),
            static_cast<const float*>(v_buf.ptr),
            static_cast<float*>(next_state_arr.request().ptr),
            static_cast<float*>(out_arr.request().ptr),
            static_cast<int>(B), static_cast<int>(H), static_cast<int>(D), decay
        );

        return py::make_tuple(out_arr, next_state_arr);
    }, "Native C++20 AVX2 Linear Recurrent Attention Step");

    // Native C++20 High-Throughput Deterministic Token Hasher
    m.def("deterministic_token_hash_cpu", [](
        std::vector<uint32_t> token_ids,
        uint64_t seed
    ) -> uint64_t {
        return turing::deterministic_token_hash_cpp(token_ids.data(), token_ids.size(), seed);
    }, "Native C++20 Deterministic 64-bit Token Block Hasher");

    // Native C++20 AVX2 1D Dilated Depthwise Causal Convolution
    m.def("dilated_causal_conv1d_cpu", [](
        py::array_t<float, py::array::c_style> input_arr,
        py::array_t<float, py::array::c_style> weight_arr,
        int dilation
    ) {
        py::buffer_info in_buf = input_arr.request();
        py::buffer_info w_buf = weight_arr.request();

        int B = static_cast<int>(in_buf.shape[0]);
        int S = static_cast<int>(in_buf.shape[1]);
        int C = static_cast<int>(in_buf.shape[2]);
        int K = static_cast<int>(w_buf.shape[1]);

        auto out_arr = py::array_t<float>({B, S, C});

        turing::dilated_causal_conv1d_simd(
            static_cast<const float*>(in_buf.ptr),
            static_cast<const float*>(w_buf.ptr),
            static_cast<float*>(out_arr.request().ptr),
            B, S, C, K, dilation
        );

        return out_arr;
    }, "Native C++20 AVX2 1D Dilated Depthwise Causal Convolution");

    // Native C++20 AVX2 Fused Base GEMV + LoRA Rank-8 Additive Contraction
    m.def("fused_lora_gemv_cpu", [](
        py::array_t<float, py::array::c_style> x_arr,
        py::array_t<float, py::array::c_style> w_base_arr,
        py::array_t<float, py::array::c_style> w_a_arr,
        py::array_t<float, py::array::c_style> w_b_arr,
        float alpha
    ) {
        py::buffer_info x_buf = x_arr.request();
        py::buffer_info wb_buf = w_base_arr.request();
        py::buffer_info wa_buf = w_a_arr.request();
        py::buffer_info wb2_buf = w_b_arr.request();

        int B = static_cast<int>(x_buf.shape[0]);
        int InDim = static_cast<int>(x_buf.shape[1]);
        int OutDim = static_cast<int>(wb_buf.shape[1]);
        int Rank = static_cast<int>(wa_buf.shape[1]);

        auto out_arr = py::array_t<float>({B, OutDim});

        turing::fused_lora_gemv_simd(
            static_cast<const float*>(x_buf.ptr),
            static_cast<const float*>(wb_buf.ptr),
            static_cast<const float*>(wa_buf.ptr),
            static_cast<const float*>(wb2_buf.ptr),
            alpha,
            static_cast<float*>(out_arr.request().ptr),
            B, InDim, OutDim, Rank
        );

        return out_arr;
    }, "Native C++20 AVX2 Fused Base GEMV + LoRA Rank-8 Contraction");

    // Native C++20 Lock-Free Elastic Memory Budget Controller
    py::class_<turing::ElasticBudgetController>(m, "NativeElasticBudgetController")
        .def(py::init<int, int, int, int, int, int, int, int, int, float>(),
             py::arg("initial_expert_slots"),
             py::arg("min_expert_slots"),
             py::arg("max_expert_slots"),
             py::arg("initial_kv_pages"),
             py::arg("min_kv_pages"),
             py::arg("max_kv_pages"),
             py::arg("bytes_per_slot"),
             py::arg("bytes_per_page"),
             py::arg("page_size_tokens"),
             py::arg("target_headroom") = 0.25f)
        .def("evaluate_and_rebalance", [](turing::ElasticBudgetController& self, int active_tokens, bool force) {
            auto res = self.evaluate_and_rebalance(active_tokens, force);
            py::dict d;
            d["rebalanced"] = res.rebalanced;
            d["new_expert_slots"] = res.new_expert_slots;
            d["new_kv_pages"] = res.new_kv_pages;
            d["rebalance_count"] = res.rebalance_count;
            d["action"] = res.action;
            return d;
        }, py::arg("active_tokens"), py::arg("force") = false)
        .def("get_expert_slots", &turing::ElasticBudgetController::get_expert_slots)
        .def("get_kv_pages", &turing::ElasticBudgetController::get_kv_pages)
        .def("get_rebalance_count", &turing::ElasticBudgetController::get_rebalance_count);

    // Native C++20 AVX2 Prefix Token Hasher
    m.def("compute_prefix_hash_fast", [](
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> tokens_arr,
        int window
    ) -> uint64_t {
        py::buffer_info buf = tokens_arr.request();
        const int32_t* ptr = static_cast<const int32_t*>(buf.ptr);
        size_t len = std::min(static_cast<size_t>(buf.size), static_cast<size_t>(window));
        return turing::compute_prefix_hash_fnv1a_cpp(ptr, len);
    }, py::arg("tokens"), py::arg("window") = 128, "Fast AVX2 SIMD FNV-1a Prefix Hasher");

    // Native C++20 AVX2 Speculative Parity Verifier
    m.def("verify_greedy_parity_fast", [](
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> spec_arr,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> plain_arr
    ) -> py::tuple {
        py::buffer_info s_buf = spec_arr.request();
        py::buffer_info p_buf = plain_arr.request();

        size_t count = std::min(s_buf.size, p_buf.size);
        const int32_t* s_ptr = static_cast<const int32_t*>(s_buf.ptr);
        const int32_t* p_ptr = static_cast<const int32_t*>(p_buf.ptr);

        auto res = turing::verify_greedy_parity_simd_cpp(s_ptr, p_ptr, count);
        if (res.passed && s_buf.size != p_buf.size) {
            return py::make_tuple(false, static_cast<int>(count), static_cast<int>(count));
        }
        return py::make_tuple(res.passed, static_cast<int>(res.num_compared), res.divergence_index);
    }, py::arg("spec_tokens"), py::arg("plain_tokens"), "Fast AVX2 SIMD Speculative Parity Verifier");

    // Native C++20 AVX2 Fast Tensor Pointer Checksum
    m.def("hash_tensor_buffer_fast", [](uintptr_t data_ptr, size_t num_bytes) -> std::string {
        if (data_ptr == 0 || num_bytes == 0) {
            return "0000000000000000";
        }
        return turing::hash_tensor_buffer_hex(reinterpret_cast<const uint8_t*>(data_ptr), num_bytes);
    }, py::arg("data_ptr"), py::arg("num_bytes"), "Fast AVX2 SIMD Tensor Memory Checksum");

    // Native C++20 AVX2 Batched Logits Sampler
    m.def("sample_batched_logits_simd", [](
        py::array_t<float, py::array::c_style | py::array::forcecast> logits_arr,
        py::array_t<float, py::array::c_style | py::array::forcecast> temps_arr,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> topks_arr,
        py::array_t<float, py::array::c_style | py::array::forcecast> unifs_arr
    ) -> std::vector<int32_t> {
        py::buffer_info l_buf = logits_arr.request();
        py::buffer_info t_buf = temps_arr.request();
        py::buffer_info k_buf = topks_arr.request();
        py::buffer_info u_buf = unifs_arr.request();

        if (l_buf.ndim != 2) {
            throw std::runtime_error("Logits array must be 2D [Batch, Vocab]");
        }

        size_t batch_size = static_cast<size_t>(l_buf.shape[0]);
        size_t vocab_size = static_cast<size_t>(l_buf.shape[1]);

        const float* l_ptr = static_cast<const float*>(l_buf.ptr);
        const float* t_ptr = static_cast<const float*>(t_buf.ptr);
        const int32_t* k_ptr = static_cast<const int32_t*>(k_buf.ptr);
        const float* u_ptr = static_cast<const float*>(u_buf.ptr);

        return turing::sample_batched_logits_simd_cpp(l_ptr, batch_size, vocab_size, t_ptr, k_ptr, u_ptr);
    }, py::arg("logits"), py::arg("temperatures"), py::arg("top_ks"), py::arg("uniform_samples"),
       "Native AVX2 SIMD Batched Logits Sampler");

    // Native C++20 AVX2 Fast JSON Syntax Scanner & Bracket Balancer
    m.def("scan_json_structure_fast", [](const std::string& text) -> py::dict {
        auto res = turing::scan_json_structure_simd_cpp(text.data(), text.size());
        py::dict d;
        d["is_valid"] = res.is_valid_json_boundary;
        d["first_brace_idx"] = res.first_brace_idx;
        d["last_brace_idx"] = res.last_brace_idx;
        d["depth_curly"] = res.depth_curly;
        d["depth_square"] = res.depth_square;
        d["in_string"] = res.in_string;
        d["repair_suffix"] = res.suggested_repair_suffix;
        return d;
    }, py::arg("text"), "Native AVX2 SIMD Fast JSON Syntax Scanner & Bracket Balancer");

    // Native C++20 AVX2 SIMD GGUF Block Dequantizer
    m.def("dequantize_gguf_simd", [](py::bytes data_bytes, int ggml_type, const std::vector<size_t>& shape) -> py::array_t<float> {
        std::string_view src = data_bytes;
        size_t total_elements = 1;
        for (auto dim : shape) total_elements *= dim;
        const size_t nbytes = turing::gguf_tensor_nbytes(static_cast<uint32_t>(ggml_type), total_elements);
        if (src.size() < nbytes) {
            throw std::runtime_error("GGUF tensor data is truncated: " + std::to_string(src.size()) + " bytes, " +
                                     std::to_string(nbytes) + " needed");
        }

        py::array_t<float> result(shape);
        float* dst_ptr = static_cast<float*>(result.request().ptr);
        {
            py::gil_scoped_release release;
            turing::dequantize_gguf(static_cast<uint32_t>(ggml_type), src.data(), dst_ptr, total_elements);
        }
        return result;
    }, py::arg("data_bytes"), py::arg("ggml_type"), py::arg("shape"), "Native AVX2 / NEON GGUF block dequantizer");

    // The standalone C++ engine (turing-cli) on a GGUF file: logits [len(token_ids), vocab] of one sequence
    m.def("gguf_cpp_engine_logits", [](const std::string& path, const std::vector<int32_t>& token_ids) {
        turing::GGUFReaderCpp reader(path);
        auto model = turing::TransformerModelCpp::load_from_gguf(reader);
        const size_t vocab = model->config.vocab_size;
        py::array_t<float> logits({token_ids.size(), vocab});
        float* out = static_cast<float*>(logits.request().ptr);
        for (size_t pos = 0; pos < token_ids.size(); ++pos) model->forward_token(token_ids[pos], pos, out + pos * vocab);
        return logits;
    }, py::arg("path"), py::arg("token_ids"), "Logits of the standalone C++ GGUF engine for one sequence");

    // Tier 2: parallel positional reader (io_uring when built with liburing, else pread / ReadFile threads)
    py::class_<turing::ParallelReader>(m, "ParallelReader")
        .def(py::init<int, size_t>(), py::arg("num_workers") = 8, py::arg("queue_depth") = 64)
        .def("read_exact", [](turing::ParallelReader& self, const std::string& filepath, uint64_t file_offset, uint64_t num_bytes) -> py::bytes {
            std::vector<uint8_t> buffer(num_bytes);
            self.read_exact(filepath, file_offset, num_bytes, buffer.data());
            return py::bytes(reinterpret_cast<const char*>(buffer.data()), num_bytes);
        }, py::arg("filepath"), py::arg("file_offset"), py::arg("num_bytes"))
        .def("read_segments_parallel", [](turing::ParallelReader& self, const std::string& filepath, 
                                          const std::vector<uint64_t>& file_offsets, 
                                          const std::vector<uint64_t>& byte_lengths) -> py::bytes {
            uint64_t total_bytes = 0;
            std::vector<uint64_t> dest_offsets(file_offsets.size());
            for (size_t i = 0; i < file_offsets.size(); ++i) {
                dest_offsets[i] = total_bytes;
                total_bytes += byte_lengths[i];
            }
            // Read straight into a fresh bytes object (no zero-filled staging vector, no second copy).
            PyObject* obj = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(total_bytes));
            if (!obj) throw py::error_already_set();
            py::bytes result = py::reinterpret_steal<py::bytes>(obj);
            uint8_t* dest = reinterpret_cast<uint8_t*>(PyBytes_AS_STRING(obj));
            {
                py::gil_scoped_release release;
                self.read_segments_parallel(filepath, file_offsets, byte_lengths, dest_offsets, dest);
            }
            return result;
        }, py::arg("filepath"), py::arg("file_offsets"), py::arg("byte_lengths"))
        .def("read_segments_into", [](turing::ParallelReader& self, const std::string& filepath,
                                      const std::vector<uint64_t>& file_offsets,
                                      const std::vector<uint64_t>& byte_lengths, py::buffer out) {
            // Segments land back to back in `out` (any writable C-contiguous buffer), with no extra copy.
            py::buffer_info info = out.request(true);
            if (info.ndim > 0 && info.strides.back() != info.itemsize) {
                throw std::invalid_argument("out must be C-contiguous");
            }
            uint64_t capacity = static_cast<uint64_t>(info.size) * static_cast<uint64_t>(info.itemsize);
            std::vector<uint64_t> dest_offsets(file_offsets.size());
            uint64_t total_bytes = 0;
            for (size_t i = 0; i < file_offsets.size() && i < byte_lengths.size(); ++i) {
                dest_offsets[i] = total_bytes;
                total_bytes += byte_lengths[i];
            }
            if (total_bytes > capacity) {
                throw std::invalid_argument("out is smaller than the sum of byte_lengths");
            }
            py::gil_scoped_release release;
            self.read_segments_parallel(filepath, file_offsets, byte_lengths, dest_offsets, static_cast<uint8_t*>(info.ptr));
            return total_bytes;
        }, py::arg("filepath"), py::arg("file_offsets"), py::arg("byte_lengths"), py::arg("out"))
        .def_property_readonly("num_workers", &turing::ParallelReader::get_num_workers)
        .def_property_readonly("queue_depth", &turing::ParallelReader::get_queue_depth)
        .def_property_readonly("backend", [](const turing::ParallelReader& self) { return std::string(self.backend()); });
    m.attr("NativeAsyncRingReader") = m.attr("ParallelReader");  // pre-1.8 name

    // Tier 4: Native GDS Loader (GPUDirect Storage cuFile dynamic driver)
    py::class_<turing::NativeGDSLoader>(m, "NativeGDSLoader")
        .def(py::init<>())
        .def("is_available", &turing::NativeGDSLoader::is_available)
        .def("has_cuda", &turing::NativeGDSLoader::has_cuda)
        .def("read_to_device", [](turing::NativeGDSLoader& self, const std::string& filepath, uint64_t file_offset,
                                  uint64_t num_bytes, uintptr_t dev_ptr, bool allow_cufile) {
            turing::NativeGDSLoader::ReadResult r;
            {
                py::gil_scoped_release release;
                r = self.read_to_device(filepath, file_offset, num_bytes, reinterpret_cast<void*>(dev_ptr), allow_cufile);
            }
            return py::make_tuple(r.bytes, r.path);
        }, py::arg("filepath"), py::arg("file_offset"), py::arg("num_bytes"), py::arg("dev_ptr"), py::arg("allow_cufile") = true,
           "Read file bytes into device memory at dev_ptr; returns (bytes_read, path) with path cufile, pinned_host_copy or no_cuda")
        .def("get_status_info", &turing::NativeGDSLoader::get_status_info);

    // Fast Safetensors Header Parser
    py::class_<turing::FastTensorInfo>(m, "FastTensorInfo")
        .def_readonly("dtype", &turing::FastTensorInfo::dtype)
        .def_readonly("shape", &turing::FastTensorInfo::shape)
        .def_readonly("start_offset", &turing::FastTensorInfo::start_offset)
        .def_readonly("end_offset", &turing::FastTensorInfo::end_offset);

    py::class_<turing::NativeSafetensorsHeaderParser>(m, "NativeSafetensorsHeaderParser")
        .def_static("parse_header", &turing::NativeSafetensorsHeaderParser::parse_header, py::arg("json_str"));

    // Native C++20 AI Traffic Manager
    py::class_<turing::NativeAdmissionResult>(m, "NativeAdmissionResult")
        .def_readonly("admitted", &turing::NativeAdmissionResult::admitted)
        .def_readonly("decision", &turing::NativeAdmissionResult::decision)
        .def_readonly("retry_after", &turing::NativeAdmissionResult::retry_after)
        .def_readonly("reason", &turing::NativeAdmissionResult::reason);

    py::class_<turing::NativeTrafficManager>(m, "NativeTrafficManager")
        .def(py::init<uint64_t, double, double>(), py::arg("vram_budget_bytes"), py::arg("high_watermark") = 0.90, py::arg("shed_watermark") = 0.95)
        .def_static("estimate_kv_bytes", &turing::NativeTrafficManager::estimate_kv_bytes, 
                    py::arg("num_prompt_tokens"), py::arg("max_new_tokens"), py::arg("num_layers"), py::arg("num_kv_heads"), py::arg("head_dim"), py::arg("dtype_bytes") = 2)
        .def_static("compute_prefix_hash", &turing::NativeTrafficManager::compute_prefix_hash, py::arg("token_ids"), py::arg("window") = 128)
        .def("admit", &turing::NativeTrafficManager::admit, py::arg("request_id"), py::arg("estimated_bytes"))
        .def("release", &turing::NativeTrafficManager::release, py::arg("request_id"))
        .def_property_readonly("utilization", &turing::NativeTrafficManager::get_utilization)
        .def_property_readonly("allocated_bytes", &turing::NativeTrafficManager::get_allocated_bytes)
        .def_property_readonly("shed_count", &turing::NativeTrafficManager::get_shed_count)
        .def_property_readonly("queue_count", &turing::NativeTrafficManager::get_queue_count);

    // Fast SIMD Common Token Prefix Matcher
    m.def("find_common_prefix_fast", [](py::array_t<int32_t, py::array::c_style | py::array::forcecast> tokens_a,
                                        py::array_t<int32_t, py::array::c_style | py::array::forcecast> tokens_b) -> int {
        py::buffer_info buf_a = tokens_a.request();
        py::buffer_info buf_b = tokens_b.request();
        const int32_t* ptr_a = static_cast<const int32_t*>(buf_a.ptr);
        const int32_t* ptr_b = static_cast<const int32_t*>(buf_b.ptr);
        return turing::find_common_token_prefix(ptr_a, static_cast<int>(buf_a.size), ptr_b, static_cast<int>(buf_b.size));
    }, py::arg("tokens_a"), py::arg("tokens_b"), "Fast SIMD common prefix matcher");

    // Fast Zero-Copy Variable-Length Token Packing using Direct C-API
    m.def("pack_varlen_tokens_fast", [](py::list token_lists) {
        size_t num_seqs = token_lists.size();
        std::vector<int32_t> cu_seqlens(num_seqs + 1, 0);
        size_t total_len = 0;
        
        PyObject* lists_ptr = token_lists.ptr();
        for (size_t i = 0; i < num_seqs; ++i) {
            PyObject* item = PyList_GET_ITEM(lists_ptr, i);
            size_t seq_len = PyList_Check(item) ? PyList_GET_SIZE(item) : 0;
            total_len += seq_len;
            cu_seqlens[i + 1] = static_cast<int32_t>(total_len);
        }

        std::vector<int32_t> flat_tokens(total_len);
        int32_t* dest_ptr = flat_tokens.data();

        for (size_t i = 0; i < num_seqs; ++i) {
            PyObject* item = PyList_GET_ITEM(lists_ptr, i);
            if (PyList_Check(item)) {
                size_t seq_len = PyList_GET_SIZE(item);
                for (size_t j = 0; j < seq_len; ++j) {
                    *dest_ptr++ = static_cast<int32_t>(PyLong_AsLong(PyList_GET_ITEM(item, j)));
                }
            }
        }

        return py::make_tuple(flat_tokens, cu_seqlens);
    }, py::arg("token_lists"), "Fast SIMD variable-length token packing and cu_seqlens generation");

    // High-Concurrency Native Routing History Tracker
    py::class_<turing::NativeRoutingHistoryTracker>(m, "NativeRoutingHistoryTracker")
        .def(py::init<size_t, size_t, float>(), py::arg("num_layers") = 32, py::arg("num_experts") = 64, py::arg("decay_factor") = 0.99f)
        .def("reset", &turing::NativeRoutingHistoryTracker::reset)
        .def("record_route", &turing::NativeRoutingHistoryTracker::record_route, py::arg("layer_idx"), py::arg("expert_indices"))
        .def("record_single", &turing::NativeRoutingHistoryTracker::record_single, py::arg("layer_idx"), py::arg("expert_idx"))
        .def("decay", &turing::NativeRoutingHistoryTracker::decay, py::arg("factor") = -1.0f)
        .def("get_top_k_experts", &turing::NativeRoutingHistoryTracker::get_top_k_experts, py::arg("k"), py::arg("layer_idx") = -1)
        .def("get_count", &turing::NativeRoutingHistoryTracker::get_count, py::arg("layer_idx"), py::arg("expert_idx"))
        .def("get_clock", &turing::NativeRoutingHistoryTracker::get_clock);

    // Fast SIMD Gated Zero-Identity Head Forward Pass
    m.def("gated_zero_identity_forward_fast", [](
        py::array_t<float, py::array::c_style | py::array::forcecast> gate_arr,
        py::array_t<float, py::array::c_style | py::array::forcecast> raw_k_arr,
        py::array_t<float, py::array::c_style | py::array::forcecast> raw_v_arr
    ) {
        py::buffer_info g_buf = gate_arr.request();
        py::buffer_info k_buf = raw_k_arr.request();
        py::buffer_info v_buf = raw_v_arr.request();

        int num_heads = static_cast<int>(k_buf.shape[0]);
        int head_dim = static_cast<int>(k_buf.shape[1]);

        py::array_t<float> out_k({num_heads, head_dim});
        py::array_t<float> out_v({num_heads, head_dim});

        py::buffer_info ok_buf = out_k.request();
        py::buffer_info ov_buf = out_v.request();

        turing::gated_zero_identity_simd(
            static_cast<const float*>(g_buf.ptr),
            static_cast<const float*>(k_buf.ptr),
            static_cast<const float*>(v_buf.ptr),
            static_cast<float*>(ok_buf.ptr),
            static_cast<float*>(ov_buf.ptr),
            static_cast<size_t>(num_heads),
            static_cast<size_t>(head_dim)
        );

        return py::make_tuple(out_k, out_v);
    }, py::arg("gate"), py::arg("raw_k"), py::arg("raw_v"), "Fast SIMD gated zero-identity head forward pass");

    // Engine SIMD kernels (turing_simd.hpp), exposed for parity tests against NumPy.
    using f32arr = py::array_t<float, py::array::c_style | py::array::forcecast>;
    using u8arr = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
    m.def("simd_backend", []() { return std::string(turing::simd_backend()); },
          "SIMD path the extension was built with: avx2, neon or scalar");
    m.def("simd_dot_f32", [](f32arr a, f32arr b) {
        if (a.size() != b.size()) throw std::invalid_argument("length mismatch");
        return turing::dot_f32(a.data(), b.data(), static_cast<size_t>(a.size()));
    });
    m.def("simd_gemv_f32", [](f32arr w, f32arr x) {
        if (w.ndim() != 2 || w.shape(1) != x.size()) throw std::invalid_argument("w must be [M, K] with K == len(x)");
        f32arr y(w.shape(0));
        turing::gemv_f32(w.data(), x.data(), y.mutable_data(), static_cast<size_t>(w.shape(0)), static_cast<size_t>(w.shape(1)));
        return y;
    });
    m.def("simd_softmax_f32", [](f32arr x) {
        f32arr y(x.size());
        std::memcpy(y.mutable_data(), x.data(), x.size() * sizeof(float));
        turing::softmax_f32(y.mutable_data(), static_cast<size_t>(y.size()));
        return y;
    });
    m.def("simd_rmsnorm_f32", [](f32arr x, f32arr w, float eps) {
        if (x.size() != w.size()) throw std::invalid_argument("length mismatch");
        f32arr y(x.size());
        turing::rmsnorm_f32(x.data(), w.data(), y.mutable_data(), static_cast<size_t>(x.size()), eps);
        return y;
    });
    m.def("simd_silu_mul_f32", [](f32arr gate, f32arr up) {
        if (gate.size() != up.size()) throw std::invalid_argument("length mismatch");
        f32arr y(gate.size());
        std::memcpy(y.mutable_data(), gate.data(), gate.size() * sizeof(float));
        turing::silu_mul_f32(y.mutable_data(), up.data(), static_cast<size_t>(y.size()));
        return y;
    });
    m.def("simd_dot_f32_u8", [](f32arr a, u8arr codes) {
        if (a.size() != codes.size()) throw std::invalid_argument("length mismatch");
        return turing::dot_f32_u8(a.data(), codes.data(), static_cast<size_t>(a.size()));
    });
    m.def("simd_axpby_u8_f32", [](float a, float b, u8arr codes, f32arr y) {
        if (y.size() != codes.size()) throw std::invalid_argument("length mismatch");
        f32arr out(y.size());
        std::memcpy(out.mutable_data(), y.data(), y.size() * sizeof(float));
        turing::axpby_u8_f32(a, b, codes.data(), out.mutable_data(), static_cast<size_t>(out.size()));
        return out;
    });
    m.def("simd_gemv_fp32_tile", [](f32arr x, f32arr w) {
        if (w.ndim() != 2 || w.shape(1) != x.size()) throw std::invalid_argument("w must be [M, K] with K == len(x)");
        f32arr y(w.shape(0));
        std::fill(y.mutable_data(), y.mutable_data() + y.size(), 0.0f);
        turing::gemv_fp32_tile(x.data(), w.data(), y.mutable_data(), static_cast<int>(w.shape(0)), static_cast<int>(w.shape(1)));
        return y;
    });
    m.def("simd_gemv_int8_tile", [](u8arr x, py::array_t<int8_t, py::array::c_style | py::array::forcecast> w) {
        if (w.ndim() != 2 || w.shape(1) != x.size()) throw std::invalid_argument("w must be [M, K] with K == len(x)");
        py::array_t<int32_t> y(w.shape(0));
        std::fill(y.mutable_data(), y.mutable_data() + y.size(), 0);
        turing::gemv_int8_tile(x.data(), w.data(), y.mutable_data(), static_cast<int>(w.shape(0)), static_cast<int>(w.shape(1)));
        return y;
    });
}

