#pragma once

#include "turing_simd.hpp"
#include <string>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdint>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#endif

namespace turing {

#pragma pack(push, 1)
struct TGateFileHeader {
    uint32_t magic;         // 0x4A474154 ("JGAT")
    uint32_t version;       // 1
    uint32_t in_features;   // D_in
    uint32_t out_features;  // D_out
    uint32_t tile_size;     // 256
    uint32_t num_tiles;     // in_features / tile_size
    uint32_t precision;     // 0 = FP32, 1 = INT8
    uint8_t  padding[36];   // Alignment padding to 64 bytes
};

struct TGate4Header {
    uint32_t magic;         // 0x34544147 ('GAT4')
    uint32_t version;       // 3
    uint32_t layer_idx;     // Layer index
    uint8_t  mask_bytes[16];// 128-bit active tile bitmask
    uint32_t hidden_dim;    // Hidden dimension (e.g. 8192)
    uint32_t ffn_dim;       // FFN intermediate dimension (e.g. 28672)
    uint32_t tile_size;     // Tile size (256)
    uint32_t num_tiles;     // Total tiles
    uint32_t k_active;      // Active tiles count
    uint8_t  reserved[16];  // 16-byte padding
};
#pragma pack(pop)

class TGateMmapEngine {
private:
#if defined(_WIN32)
    HANDLE hFile;
    HANDLE hMap;
#else
    int fd;
#endif
    size_t file_size;
    uint8_t* mapped_data;
    TGateFileHeader header;
    std::vector<uint64_t> tile_offsets;
    const float* bias_ptr;

public:
    TGateMmapEngine(const std::string& filepath) 
#if defined(_WIN32)
        : hFile(INVALID_HANDLE_VALUE), hMap(NULL), file_size(0), mapped_data(nullptr), bias_ptr(nullptr)
#else
        : fd(-1), file_size(0), mapped_data(nullptr), bias_ptr(nullptr)
#endif
    {
#if defined(_WIN32)
        hFile = CreateFileA(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Failed to open file on Windows: " + filepath);
        }
        LARGE_INTEGER size;
        if (!GetFileSizeEx(hFile, &size)) {
            CloseHandle(hFile);
            hFile = INVALID_HANDLE_VALUE;
            throw std::runtime_error("Failed to get file size: " + filepath);
        }
        file_size = static_cast<size_t>(size.QuadPart);
        hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMap) {
            CloseHandle(hFile);
            hFile = INVALID_HANDLE_VALUE;
            throw std::runtime_error("Failed to create file mapping on Windows: " + filepath);
        }
        void* addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, file_size);
        if (!addr) {
            CloseHandle(hMap);
            CloseHandle(hFile);
            hMap = NULL;
            hFile = INVALID_HANDLE_VALUE;
            throw std::runtime_error("Failed to map view of file on Windows: " + filepath);
        }
        mapped_data = static_cast<uint8_t*>(addr);
#else
        fd = open(filepath.c_str(), O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Failed to open file: " + filepath);
        }

        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            close(fd);
            throw std::runtime_error("Failed to stat file: " + filepath);
        }
        file_size = static_cast<size_t>(sb.st_size);

        void* addr = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to mmap file: " + filepath);
        }
        mapped_data = static_cast<uint8_t*>(addr);

#if defined(MADV_WILLNEED)
        madvise(mapped_data, file_size, MADV_WILLNEED);
#endif
#endif

        // Parse header
        if (file_size < sizeof(TGateFileHeader)) {
            cleanup();
            throw std::runtime_error("File size smaller than TGate header");
        }
        std::memcpy(&header, mapped_data, sizeof(TGateFileHeader));

        if (header.magic != 0x4A474154) { // "JGAT"
            cleanup();
            throw std::runtime_error("Invalid TGate magic bytes");
        }

        tile_offsets.resize(header.num_tiles);
        size_t offset_pos = sizeof(TGateFileHeader);
        for (uint32_t t = 0; t < header.num_tiles; ++t) {
            std::memcpy(&tile_offsets[t], mapped_data + offset_pos, sizeof(uint64_t));
            offset_pos += sizeof(uint64_t);
        }

        bias_ptr = reinterpret_cast<const float*>(mapped_data + offset_pos);
    }

    ~TGateMmapEngine() {
        cleanup();
    }

    void cleanup() {
#if defined(_WIN32)
        if (mapped_data) {
            UnmapViewOfFile(mapped_data);
            mapped_data = nullptr;
        }
        if (hMap) {
            CloseHandle(hMap);
            hMap = NULL;
        }
        if (hFile != INVALID_HANDLE_VALUE) {
            CloseHandle(hFile);
            hFile = INVALID_HANDLE_VALUE;
        }
#else
        if (mapped_data && mapped_data != MAP_FAILED) {
            munmap(mapped_data, file_size);
            mapped_data = nullptr;
        }
        if (fd != -1) {
            close(fd);
            fd = -1;
        }
#endif
    }

    const TGateFileHeader& get_header() const { return header; }

    void forward_sparse(const float* input_vec, uint32_t active_mask, float* output_vec) {
        // Initialize with bias
        std::memcpy(output_vec, bias_ptr, header.out_features * sizeof(float));

        // Iterate through active tiles
        for (uint32_t t = 0; t < header.num_tiles; ++t) {
            if (!(active_mask & (1U << t))) {
                continue; // Hardware pointer bypass
            }
            const float* w_tile = reinterpret_cast<const float*>(mapped_data + tile_offsets[t]);
            const float* i_tile = input_vec + (t * header.tile_size);
            gemv_fp32_tile(i_tile, w_tile, output_vec, header.out_features, header.tile_size);
        }
    }
};

} // namespace turing
