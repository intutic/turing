// turing.turing_llama: the libllama CPU backend (csrc/cpu/turing_llama.hpp) for Python. Built only with
// TURING_BUILD_LLAMA=1 (setup.py) or -DTURING_LLAMA=ON (CMake), from the pinned third_party/llama.cpp submodule.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <cstring>

#include "cpu/turing_llama.hpp"

namespace py = pybind11;
using turing::LlamaOptions;
using turing::TuringLlama;

PYBIND11_MODULE(turing_llama, m) {
    m.doc() = "Turing Engine's CPU backend on libllama (llama.cpp, pinned in third_party/llama.cpp)";
    m.def(
        "gpu_devices",
        [] {
            llama_log_set([](ggml_log_level l, const char* t, void*) { if (l == GGML_LOG_LEVEL_ERROR) std::fputs(t, stderr); },
                          nullptr);  // the Metal device's start-up lines otherwise
            llama_backend_init();
            std::vector<std::string> out;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) out.emplace_back(ggml_backend_dev_name(d));
            }
            return out;
        },
        "GPU devices this build can put layers on (n_gpu_layers): empty unless built with TURING_BUILD_LLAMA_METAL=1.");
    py::class_<TuringLlama>(m, "Llama")
        .def(py::init([](const std::string& path, int n_ctx, int n_batch, int n_threads, bool use_mmap, bool repack,
                         bool exact, bool verbose, bool readahead, int readahead_threads, size_t readahead_chunk,
                         bool readahead_check_resident, bool readahead_wait, bool readahead_adaptive,
                         int readahead_window, int n_seq, int n_gpu_layers) {
                 LlamaOptions o;
                 o.n_ctx = n_ctx;
                 o.n_batch = n_batch;
                 o.n_threads = n_threads;
                 o.use_mmap = use_mmap;
                 o.repack = repack;
                 o.exact = exact;
                 o.verbose = verbose;
                 o.readahead = readahead;
                 o.readahead_threads = readahead_threads;
                 o.readahead_chunk = readahead_chunk;
                 o.readahead_check_resident = readahead_check_resident;
                 o.readahead_wait = readahead_wait;
                 o.readahead_adaptive = readahead_adaptive;
                 o.readahead_window = readahead_window;
                 o.n_seq = n_seq;
                 o.n_gpu_layers = n_gpu_layers;
                 py::gil_scoped_release release;
                 return std::make_unique<TuringLlama>(path, o);
             }),
             py::arg("path"), py::kw_only(), py::arg("n_ctx") = 4096, py::arg("n_batch") = 2048, py::arg("n_threads") = 0,
             py::arg("use_mmap") = true, py::arg("repack") = true, py::arg("exact") = false, py::arg("verbose") = false,
             py::arg("readahead") = false, py::arg("readahead_threads") = 16, py::arg("readahead_chunk") = size_t(4) << 20,
             py::arg("readahead_check_resident") = true, py::arg("readahead_wait") = true,
             py::arg("readahead_adaptive") = false, py::arg("readahead_window") = 64, py::arg("n_seq") = 1,
             py::arg("n_gpu_layers") = 0)
        .def(
            "readahead_stats",
            [](const TuringLlama& self) {
                const turing::ReadAheadStats s = self.readahead_stats();
                py::dict d;
                d["calls"] = s.calls;
                d["experts"] = s.experts;
                d["bytes_wanted"] = s.bytes_wanted;
                d["bytes_read"] = s.bytes_read;
                d["bytes_missing"] = s.bytes_missing;
                d["active_calls"] = s.active_calls;
                d["seconds"] = s.seconds;
                d["moe_layers"] = self.readahead_layers();
                return d;
            },
            "Expert read-ahead counters: router outputs seen, experts chosen, their bytes, bytes read, seconds.")
        .def_property_readonly("n_vocab", &TuringLlama::n_vocab)
        .def_property_readonly("n_ctx", &TuringLlama::n_ctx)
        .def_property_readonly("n_seq", &TuringLlama::n_seq)
        .def("seq_clear", &TuringLlama::seq_clear, py::arg("seq"))
        .def(
            "step",
            [](TuringLlama& self, const std::vector<int>& seqs, const std::vector<std::vector<int>>& tokens,
               const std::vector<int>& pos) {
                std::vector<float> v;
                {
                    py::gil_scoped_release release;
                    v = self.step(seqs, tokens, pos);
                }
                const py::ssize_t V = self.n_vocab();
                py::array_t<float> out({static_cast<py::ssize_t>(seqs.size()), V});
                std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(float));
                return out;
            },
            py::arg("seqs"), py::arg("tokens"), py::arg("pos"),
            "Several sequences in one batch: tokens[i] appended to seqs[i] from position pos[i]; last-token logits [n, V].")
        .def(
            "load_mmproj",
            [](TuringLlama& self, const std::string& path) {
                py::gil_scoped_release release;
                self.load_mmproj(path);
            },
            py::arg("path"), "Load the model's vision projector (an mmproj GGUF) for prefill_media.")
        .def_property_readonly("has_vision", &TuringLlama::has_vision)
        .def_property_readonly_static("media_marker", [](py::object) { return TuringLlama::media_marker(); })
        .def(
            "prefill_media",
            [](TuringLlama& self, const std::string& text, const std::vector<py::bytes>& images) {
                std::vector<std::string> raw(images.begin(), images.end());
                std::vector<float> v;
                {
                    py::gil_scoped_release release;
                    v = self.prefill_media(text, raw);
                }
                py::array_t<float> out(static_cast<py::ssize_t>(v.size()));
                std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(float));
                return out;
            },
            py::arg("text"), py::arg("images"),
            "Evaluate a prompt with one media_marker per image (encoded bytes, in order) from an empty cache; the last "
            "token's logits [n_vocab].")
        .def_property_readonly("n_past", &TuringLlama::n_past)
        .def_property_readonly("bos", &TuringLlama::bos)
        .def_property_readonly("eos", &TuringLlama::eos)
        .def_property_readonly("n_params", &TuringLlama::n_params)
        .def_property_readonly("size_bytes", &TuringLlama::size_bytes)
        .def_property_readonly("desc", &TuringLlama::desc)
        .def("is_eog", &TuringLlama::is_eog, py::arg("token"))
        .def("tokenize", &TuringLlama::tokenize, py::arg("text"), py::arg("add_special") = true,
             py::arg("parse_special") = true)
        .def(
            "piece",
            [](const TuringLlama& self, int token, bool special) { return py::bytes(self.piece(token, special)); },
            py::arg("token"), py::arg("special") = true,
            "The token's bytes (a byte-level token can hold part of a UTF-8 character); special=False: empty for control "
            "tokens.")
        .def("truncate", &TuringLlama::truncate, py::arg("n"))
        .def(
            "eval",
            [](TuringLlama& self, const std::vector<int>& tokens, bool all_logits) {
                std::vector<float> v;
                {
                    py::gil_scoped_release release;
                    v = self.eval(tokens, all_logits);
                }
                const py::ssize_t V = self.n_vocab();
                py::array_t<float> out = all_logits ? py::array_t<float>({static_cast<py::ssize_t>(v.size()) / V, V})
                                                    : py::array_t<float>(V);
                std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(float));
                return out;
            },
            py::arg("tokens"), py::arg("all_logits") = false,
            "Append tokens after the cached ones; logits of the last token [n_vocab], or of all of them [n, n_vocab].")
        .def(
            "prefill",
            [](TuringLlama& self, const std::vector<int>& prompt) {
                std::vector<float> v;
                {
                    py::gil_scoped_release release;
                    v = self.prefill(prompt);
                }
                py::array_t<float> out(static_cast<py::ssize_t>(v.size()));
                std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(float));
                return out;
            },
            py::arg("prompt"),
            "Make the cache hold exactly prompt (reusing the prefix it shares with the cached tokens); last logits.")
        .def_property_readonly("cached_prompt_tokens", &TuringLlama::cached_prompt_tokens)
        .def(
            "generate",
            [](TuringLlama& self, const std::vector<int>& prompt, int max_new, float temperature, float top_p, int top_k,
               uint32_t seed, py::object on_token, int n_draft) {
                std::function<bool(int)> cb;
                if (!on_token.is_none()) {
                    cb = [&on_token](int t) {
                        py::gil_scoped_acquire acquire;
                        py::object r = on_token(t);
                        return r.is_none() || r.cast<bool>();
                    };
                }
                py::gil_scoped_release release;
                if (n_draft > 0 && temperature <= 0.0f) return self.generate_lookup(prompt, max_new, n_draft, cb);
                return self.generate(prompt, max_new, temperature, top_p, top_k, seed, cb);
            },
            py::arg("prompt"), py::arg("max_new"), py::arg("temperature") = 0.0f, py::arg("top_p") = 1.0f,
            py::arg("top_k") = 0, py::arg("seed") = 0, py::arg("on_token") = py::none(), py::arg("n_draft") = 0,
            "Generate after prompt (token ids), reusing the cached prefix; on_token(id) may return False to stop. "
            "n_draft > 0 with temperature 0: prompt-lookup speculation; greedy in exact arithmetic (libllama's batched "
            "kernels round differently from its one-token kernels, so a near-tie can flip; exact=True avoids it).")
        .def(
            "spec_stats",
            [](const TuringLlama& self) {
                const auto s = self.spec_stats();
                py::dict d;
                d["steps"] = s.steps;
                d["drafted"] = s.drafted;
                d["accepted"] = s.accepted;
                return d;
            },
            "Prompt-lookup speculation: verification steps, drafts verified, drafts accepted.");
}
