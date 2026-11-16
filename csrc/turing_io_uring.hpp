#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <thread>
#include <future>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <exception>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif
#endif

namespace turing {

/**
 * Parallel pread reader (Tier 2 storage ingestion).
 * read_segments_parallel splits segments into requests of at most 1 MiB and keeps up to queue_depth of them
 * in flight (one thread each). Large requests with many in flight are what an SSD needs to reach its rated
 * bandwidth; see docs/benchmarks.md for the measured numbers. num_workers is kept for API compatibility.
 */
class NativeAsyncRingReader {
public:
    NativeAsyncRingReader(int num_workers = 8, size_t queue_depth = 64)
        : num_workers_(num_workers > 0 ? num_workers : 8), queue_depth_(queue_depth) {
    }

    ~NativeAsyncRingReader() = default;

    /**
     * Reads a contiguous slice of bytes from a binary file into a pre-allocated 64-byte aligned buffer.
     */
    void read_exact(const std::string& filepath, uint64_t file_offset, uint64_t num_bytes, uint8_t* dest_buffer) {
        if (!dest_buffer) {
            throw std::invalid_argument("Destination buffer pointer cannot be null");
        }

#if defined(_WIN32)
        HANDLE hFile = CreateFileA(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Failed to open file: " + filepath);
        }

        OVERLAPPED ol;
        std::memset(&ol, 0, sizeof(ol));
        ol.Offset = static_cast<DWORD>(file_offset & 0xFFFFFFFF);
        ol.OffsetHigh = static_cast<DWORD>((file_offset >> 32) & 0xFFFFFFFF);

        DWORD bytesRead = 0;
        BOOL ok = ReadFile(hFile, dest_buffer, static_cast<DWORD>(num_bytes), &bytesRead, &ol);
        CloseHandle(hFile);
        if (!ok || bytesRead != num_bytes) {
            throw std::runtime_error("ReadFile failed or incomplete read");
        }
#else
        int fd = open(filepath.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("Failed to open file: " + filepath);
        }

        uint64_t total_read = 0;
        while (total_read < num_bytes) {
            ssize_t n = pread(fd, dest_buffer + total_read, num_bytes - total_read, file_offset + total_read);
            if (n <= 0) {
                close(fd);
                throw std::runtime_error("pread failed or premature EOF for: " + filepath);
            }
            total_read += static_cast<uint64_t>(n);
        }
        close(fd);
#endif
    }

    /**
     * Reads multiple discontinuous tensor segments in parallel across worker threads.
     * Segments are (file_offset, num_bytes, dest_offset_in_buffer).
     */
    void read_segments_parallel(
        const std::string& filepath,
        const std::vector<uint64_t>& file_offsets,
        const std::vector<uint64_t>& byte_lengths,
        const std::vector<uint64_t>& dest_offsets,
        uint8_t* dest_buffer
    ) {
        if (file_offsets.size() != byte_lengths.size() || file_offsets.size() != dest_offsets.size()) {
            throw std::invalid_argument("Segment offset vectors must have matching lengths");
        }

        size_t num_segments = file_offsets.size();
        if (num_segments == 0) return;

#if defined(_WIN32)
        for (size_t i = 0; i < num_segments; ++i) {
            read_exact(filepath, file_offsets[i], byte_lengths[i], dest_buffer + dest_offsets[i]);
        }
#else
        // Split every segment into requests of at most kMaxRequestBytes and keep up to queue_depth of them in
        // flight. pread is synchronous, so each request in flight needs its own thread.
        struct Request { uint64_t file_offset; uint64_t len; uint8_t* dest; };
        std::vector<Request> requests;
        for (size_t i = 0; i < num_segments; ++i) {
            for (uint64_t done = 0; done < byte_lengths[i]; done += kMaxRequestBytes) {
                requests.push_back({file_offsets[i] + done, std::min(kMaxRequestBytes, byte_lengths[i] - done),
                                    dest_buffer + dest_offsets[i] + done});
            }
        }
        if (requests.empty()) return;

        int fd = open(filepath.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("Failed to open file: " + filepath);
        }

        size_t in_flight = std::min(std::max<size_t>(queue_depth_, 1), requests.size());
        std::atomic<size_t> next{0};
        std::vector<std::future<void>> futures;
        futures.reserve(in_flight);
        for (size_t w = 0; w < in_flight; ++w) {
            futures.push_back(std::async(std::launch::async, [fd, &requests, &next]() {
                for (size_t i = next++; i < requests.size(); i = next++) {
                    const Request& r = requests[i];
                    uint64_t bytes_done = 0;
                    while (bytes_done < r.len) {
                        ssize_t n = pread(fd, r.dest + bytes_done, r.len - bytes_done, r.file_offset + bytes_done);
                        if (n <= 0) {
                            throw std::runtime_error("Parallel segment pread failed or premature EOF");
                        }
                        bytes_done += static_cast<uint64_t>(n);
                    }
                }
            }));
        }

        std::exception_ptr error;
        for (auto& f : futures) {
            try { f.get(); } catch (...) { if (!error) error = std::current_exception(); }
        }
        close(fd);
        if (error) std::rethrow_exception(error);
#endif
    }

    static constexpr uint64_t kMaxRequestBytes = 1ull << 20;

    int get_num_workers() const { return num_workers_; }
    size_t get_queue_depth() const { return queue_depth_; }

private:
    int num_workers_;
    size_t queue_depth_;
};

} // namespace turing
