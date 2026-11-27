// turing.turing_llama: the libllama CPU backend (csrc/cpu/turing_llama.hpp) for Python. Built only with
// TURING_BUILD_LLAMA=1 (setup.py) or -DTURING_LLAMA=ON (CMake), from the pinned third_party/llama.cpp submodule.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>

#include "cpu/turing_llama.hpp"

namespace py = pybind11;
using turing::LlamaOptions;
using turing::TuringLlama;

PYBIND11_MODULE(turing_llama, m) {
    m.doc() = "Turing Engine's CPU backend on libllama (llama.cpp, pinned in third_party/llama.cpp)";
    py::class_<TuringLlama>(m, "Llama")
        .def(py::init([](const std::string& path, int n_ctx, int n_batch, int n_threads, bool use_mmap, bool repack,
                         bool exact, bool verbose) {
                 LlamaOptions o;
                 o.n_ctx = n_ctx;
                 o.n_batch = n_batch;
                 o.n_threads = n_threads;
                 o.use_mmap = use_mmap;
                 o.repack = repack;
                 o.exact = exact;
                 o.verbose = verbose;
                 py::gil_scoped_release release;
                 return std::make_unique<TuringLlama>(path, o);
             }),
             py::arg("path"), py::kw_only(), py::arg("n_ctx") = 4096, py::arg("n_batch") = 2048, py::arg("n_threads") = 0,
             py::arg("use_mmap") = true, py::arg("repack") = true, py::arg("exact") = false, py::arg("verbose") = false)
        .def_property_readonly("n_vocab", &TuringLlama::n_vocab)
        .def_property_readonly("n_ctx", &TuringLlama::n_ctx)
        .def_property_readonly("n_past", &TuringLlama::n_past)
        .def_property_readonly("bos", &TuringLlama::bos)
        .def_property_readonly("eos", &TuringLlama::eos)
        .def_property_readonly("n_params", &TuringLlama::n_params)
        .def_property_readonly("size_bytes", &TuringLlama::size_bytes)
        .def_property_readonly("desc", &TuringLlama::desc)
        .def("is_eog", &TuringLlama::is_eog, py::arg("token"))
        .def("tokenize", &TuringLlama::tokenize, py::arg("text"), py::arg("add_special") = true,
             py::arg("parse_special") = true)
        .def("piece", &TuringLlama::piece, py::arg("token"))
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
                                                    : py::array_t<float>({V});
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
                py::array_t<float> out({static_cast<py::ssize_t>(v.size())});
                std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(float));
                return out;
            },
            py::arg("prompt"),
            "Make the cache hold exactly prompt (reusing the prefix it shares with the cached tokens); last logits.")
        .def_property_readonly("cached_prompt_tokens", &TuringLlama::cached_prompt_tokens)
        .def(
            "generate",
            [](TuringLlama& self, const std::vector<int>& prompt, int max_new, float temperature, float top_p, int top_k,
               uint32_t seed, py::object on_token) {
                std::function<bool(int)> cb;
                if (!on_token.is_none()) {
                    cb = [&on_token](int t) {
                        py::gil_scoped_acquire acquire;
                        py::object r = on_token(t);
                        return r.is_none() || r.cast<bool>();
                    };
                }
                py::gil_scoped_release release;
                return self.generate(prompt, max_new, temperature, top_p, top_k, seed, cb);
            },
            py::arg("prompt"), py::arg("max_new"), py::arg("temperature") = 0.0f, py::arg("top_p") = 1.0f,
            py::arg("top_k") = 0, py::arg("seed") = 0, py::arg("on_token") = py::none(),
            "Generate after prompt (token ids), reusing the cached prefix; on_token(id) may return False to stop.");
}
