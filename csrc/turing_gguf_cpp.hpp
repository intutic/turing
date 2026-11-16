#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cmath>
#include <fcntl.h>
#include <sys/stat.h>
#include "turing_gguf_simd.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

namespace turing {

constexpr uint32_t GGUF_MAGIC_CPP = 0x46554747; // 'GGUF' (Little Endian)
constexpr uint32_t GGUF_DEFAULT_ALIGNMENT = 32;

enum class GGMLTypeCpp : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    BF16 = 30
};

enum class GGUFValueTypeCpp : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12
};

struct GGUFTensorInfoCpp {
    std::string name;
    uint32_t n_dims;
    std::vector<uint64_t> shape; // PyTorch order [d0, d1, ...]
    GGMLTypeCpp ggml_type;
    uint64_t offset;
};

class GGUFReaderCpp {
public:
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    uint32_t alignment = GGUF_DEFAULT_ALIGNMENT;
    uint64_t tensor_data_offset = 0;

    std::unordered_map<std::string, std::string> metadata_strings;
    std::unordered_map<std::string, int64_t> metadata_ints;
    std::unordered_map<std::string, float> metadata_floats;
    std::vector<std::string> tokenizer_tokens;
    std::unordered_map<std::string, GGUFTensorInfoCpp> tensor_infos;

private:
    uint8_t* mmap_data = nullptr;
    size_t file_size = 0;
    int fd = -1;

public:
    GGUFReaderCpp(const std::string& filepath) {
        open_and_mmap(filepath);
        parse_header_and_metadata();
    }

    ~GGUFReaderCpp() {
        cleanup();
    }

    bool has_tensor(const std::string& name) const {
        return tensor_infos.find(name) != tensor_infos.end();
    }

    const GGUFTensorInfoCpp& get_tensor_info(const std::string& name) const {
        auto it = tensor_infos.find(name);
        if (it == tensor_infos.end()) {
            throw std::runtime_error("Tensor not found in GGUF: " + name);
        }
        return it->second;
    }

    // Dequantizes a tensor into a preallocated FP32 buffer, in the file's layout: llama q/k rows stay in the
    // order llama.cpp's converter permuted them to, which is what the engine's interleaved (GPT-J) RoPE expects.
    void read_tensor_fp32(const std::string& name, float* dst, size_t dst_capacity) const {
        const auto& info = get_tensor_info(name);
        size_t num_elements = 1;
        for (uint64_t d : info.shape) {
            num_elements *= d;
        }
        if (num_elements > dst_capacity) {
            throw std::runtime_error("Destination buffer too small for tensor " + name);
        }
        const size_t nbytes = gguf_tensor_nbytes(static_cast<uint32_t>(info.ggml_type), num_elements);
        if (tensor_data_offset + info.offset + nbytes > file_size) {
            throw std::runtime_error("GGUF file is truncated: tensor " + name + " ends past the end of the file");
        }
        dequantize_gguf(static_cast<uint32_t>(info.ggml_type), mmap_data + tensor_data_offset + info.offset, dst,
                        num_elements);
    }

private:
    // Byte size of a scalar GGUF metadata value (0 for STRING and ARRAY, which carry their own lengths)
    static size_t scalar_size(GGUFValueTypeCpp t) {
        switch (t) {
            case GGUFValueTypeCpp::UINT8: case GGUFValueTypeCpp::INT8: case GGUFValueTypeCpp::BOOL: return 1;
            case GGUFValueTypeCpp::UINT16: case GGUFValueTypeCpp::INT16: return 2;
            case GGUFValueTypeCpp::UINT32: case GGUFValueTypeCpp::INT32: case GGUFValueTypeCpp::FLOAT32: return 4;
            case GGUFValueTypeCpp::UINT64: case GGUFValueTypeCpp::INT64: case GGUFValueTypeCpp::FLOAT64: return 8;
            case GGUFValueTypeCpp::STRING: case GGUFValueTypeCpp::ARRAY: return 0;
        }
        throw std::runtime_error("Unknown GGUF metadata value type " + std::to_string(static_cast<uint32_t>(t)));
    }

    void open_and_mmap(const std::string& filepath) {
#if defined(_WIN32)
        HANDLE hFile = CreateFileA(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) throw std::runtime_error("Could not open GGUF file: " + filepath);
        LARGE_INTEGER size;
        GetFileSizeEx(hFile, &size);
        file_size = static_cast<size_t>(size.QuadPart);
        HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        mmap_data = static_cast<uint8_t*>(MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0));
        CloseHandle(hMap);
        CloseHandle(hFile);
#else
        fd = open(filepath.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Could not open GGUF file: " + filepath);
        struct stat sb;
        if (fstat(fd, &sb) < 0) throw std::runtime_error("Could not stat GGUF file");
        file_size = sb.st_size;
        mmap_data = static_cast<uint8_t*>(mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0));
        if (mmap_data == MAP_FAILED) throw std::runtime_error("mmap failed for GGUF file");
#if defined(MADV_WILLNEED)
        madvise(mmap_data, file_size, MADV_WILLNEED);
#endif
#endif
    }

    void cleanup() {
#if defined(_WIN32)
        if (mmap_data) UnmapViewOfFile(mmap_data);
#else
        if (mmap_data && mmap_data != MAP_FAILED) munmap(mmap_data, file_size);
        if (fd >= 0) close(fd);
#endif
        mmap_data = nullptr;
    }

    std::string read_str(size_t& pos) {
        if (pos + 8 > file_size) throw std::runtime_error("Unexpected EOF reading string length");
        uint64_t len = *reinterpret_cast<const uint64_t*>(mmap_data + pos);
        pos += 8;
        if (pos + len > file_size) throw std::runtime_error("Unexpected EOF reading string body");
        std::string s(reinterpret_cast<const char*>(mmap_data + pos), len);
        pos += len;
        return s;
    }

    void parse_header_and_metadata() {
        if (file_size < 24) throw std::runtime_error("File size is smaller than GGUF header");
        size_t pos = 0;

        uint32_t magic = *reinterpret_cast<const uint32_t*>(mmap_data + pos); pos += 4;
        if (magic != GGUF_MAGIC_CPP) {
            throw std::runtime_error("Invalid GGUF magic bytes");
        }
        version = *reinterpret_cast<const uint32_t*>(mmap_data + pos); pos += 4;
        tensor_count = *reinterpret_cast<const uint64_t*>(mmap_data + pos); pos += 8;
        kv_count = *reinterpret_cast<const uint64_t*>(mmap_data + pos); pos += 8;

        // Parse KV metadata
        for (uint64_t i = 0; i < kv_count; ++i) {
            std::string key = read_str(pos);
            uint32_t vtype_raw = *reinterpret_cast<const uint32_t*>(mmap_data + pos); pos += 4;
            auto vtype = static_cast<GGUFValueTypeCpp>(vtype_raw);

            if (vtype == GGUFValueTypeCpp::STRING) {
                metadata_strings[key] = read_str(pos);
            }
            else if (vtype == GGUFValueTypeCpp::UINT32) {
                metadata_ints[key] = *reinterpret_cast<const uint32_t*>(mmap_data + pos); pos += 4;
            }
            else if (vtype == GGUFValueTypeCpp::INT32) {
                metadata_ints[key] = *reinterpret_cast<const int32_t*>(mmap_data + pos); pos += 4;
            }
            else if (vtype == GGUFValueTypeCpp::UINT64) {
                metadata_ints[key] = *reinterpret_cast<const uint64_t*>(mmap_data + pos); pos += 8;
            }
            else if (vtype == GGUFValueTypeCpp::INT64) {
                metadata_ints[key] = *reinterpret_cast<const int64_t*>(mmap_data + pos); pos += 8;
            }
            else if (vtype == GGUFValueTypeCpp::FLOAT32) {
                metadata_floats[key] = *reinterpret_cast<const float*>(mmap_data + pos); pos += 4;
            }
            else if (vtype == GGUFValueTypeCpp::BOOL) {
                metadata_ints[key] = *reinterpret_cast<const uint8_t*>(mmap_data + pos); pos += 1;
            }
            else if (vtype == GGUFValueTypeCpp::ARRAY) {
                uint32_t elem_type_raw = *reinterpret_cast<const uint32_t*>(mmap_data + pos); pos += 4;
                uint64_t arr_len = *reinterpret_cast<const uint64_t*>(mmap_data + pos); pos += 8;
                auto elem_type = static_cast<GGUFValueTypeCpp>(elem_type_raw);

                for (uint64_t a = 0; a < arr_len; ++a) {
                    if (elem_type == GGUFValueTypeCpp::STRING) {
                        std::string elem_str = read_str(pos);
                        if (key == "tokenizer.ggml.tokens") {
                            tokenizer_tokens.push_back(elem_str);
                        }
                    } else if (elem_type == GGUFValueTypeCpp::ARRAY) {
                        throw std::runtime_error("Nested GGUF metadata arrays are not supported: " + key);
                    } else {
                        pos += scalar_size(elem_type);
                    }
                }
            } else {
                pos += scalar_size(vtype);  // other scalar types are not used
            }
        }

        if (metadata_ints.find("general.alignment") != metadata_ints.end()) {
            alignment = static_cast<uint32_t>(metadata_ints["general.alignment"]);
        }

        // Parse Tensor Infos
        for (uint64_t i = 0; i < tensor_count; ++i) {
            std::string t_name = read_str(pos);
            uint32_t n_dims = *reinterpret_cast<const uint32_t*>(mmap_data + pos); pos += 4;
            std::vector<uint64_t> dims(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d) {
                dims[d] = *reinterpret_cast<const uint64_t*>(mmap_data + pos); pos += 8;
            }
            // Reverse GGML column-major dimensions into row-major
            std::vector<uint64_t> torch_shape(dims.rbegin(), dims.rend());

            uint32_t type_raw = *reinterpret_cast<const uint32_t*>(mmap_data + pos); pos += 4;
            uint64_t offset = *reinterpret_cast<const uint64_t*>(mmap_data + pos); pos += 8;

            tensor_infos[t_name] = GGUFTensorInfoCpp{
                t_name,
                n_dims,
                torch_shape,
                static_cast<GGMLTypeCpp>(type_raw),
                offset
            };
        }

        // Align tensor data section
        size_t rem = pos % alignment;
        tensor_data_offset = (rem == 0) ? pos : (pos + (alignment - rem));
    }
};

} // namespace turing
