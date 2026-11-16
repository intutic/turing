#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <cerrno>
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
#endif

// setup.py defines TURING_HAVE_LIBURING (and links -luring) when liburing is installed at build time on Linux.
#if defined(__linux__) && defined(TURING_HAVE_LIBURING)
#include <liburing.h>
#define TURING_IO_URING 1
#endif

namespace turing {

/**
 * Parallel positional file reader (Tier 2 storage ingestion).
 * read_segments_parallel splits segments into requests of at most 1 MiB and keeps up to queue_depth of them in
 * flight. Backends, chosen once at construction:
 *   "io_uring" - Linux built with liburing and a ring that can be created at run time: one ring, queue_depth reads
 *                submitted together, short reads resubmitted;
 *   "pread"    - POSIX: one thread per in-flight request, each doing pread;
 *   "win32"    - Windows: one thread per in-flight request, each doing a positional ReadFile.
 * Large requests with many in flight are what an SSD needs to reach its rated bandwidth.
 * num_workers is kept for API compatibility.
 */
class ParallelReader {
public:
    ParallelReader(int num_workers = 8, size_t queue_depth = 64)
        : num_workers_(num_workers > 0 ? num_workers : 8), queue_depth_(queue_depth) {
#if defined(TURING_IO_URING)
        io_uring probe;
        if (io_uring_queue_init(1, &probe, 0) == 0) {  // seccomp, old kernels and io_uring_disabled refuse here
            io_uring_queue_exit(&probe);
            use_uring_ = true;
        }
#endif
    }

    ~ParallelReader() = default;

    const char* backend() const {
#if defined(_WIN32)
        return "win32";
#else
        return use_uring_ ? "io_uring" : "pread";
#endif
    }

    /**
     * Reads a contiguous slice of bytes from a binary file into a caller-provided buffer.
     */
    void read_exact(const std::string& filepath, uint64_t file_offset, uint64_t num_bytes, uint8_t* dest_buffer) {
        if (!dest_buffer) {
            throw std::invalid_argument("Destination buffer pointer cannot be null");
        }
        File f(filepath);
        read_at(f, dest_buffer, num_bytes, file_offset);
    }

    /**
     * Reads multiple discontinuous tensor segments in parallel.
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
        std::vector<Request> requests;
        for (size_t i = 0; i < file_offsets.size(); ++i) {
            for (uint64_t done = 0; done < byte_lengths[i]; done += kMaxRequestBytes) {
                requests.push_back({file_offsets[i] + done, std::min(kMaxRequestBytes, byte_lengths[i] - done),
                                    dest_buffer + dest_offsets[i] + done});
            }
        }
        if (requests.empty()) return;

        File f(filepath);
#if defined(TURING_IO_URING)
        if (use_uring_) {
            read_uring(f.fd, requests);
            return;
        }
#endif
        // Positional reads are synchronous, so each request in flight needs its own thread.
        size_t in_flight = std::min(std::max<size_t>(queue_depth_, 1), requests.size());
        std::atomic<size_t> next{0};
        std::vector<std::future<void>> futures;
        futures.reserve(in_flight);
        for (size_t w = 0; w < in_flight; ++w) {
            futures.push_back(std::async(std::launch::async, [&f, &requests, &next]() {
                for (size_t i = next++; i < requests.size(); i = next++) {
                    read_at(f, requests[i].dest, requests[i].len, requests[i].file_offset);
                }
            }));
        }
        std::exception_ptr error;
        for (auto& fut : futures) {
            try { fut.get(); } catch (...) { if (!error) error = std::current_exception(); }
        }
        if (error) std::rethrow_exception(error);
    }

    static constexpr uint64_t kMaxRequestBytes = 1ull << 20;

    int get_num_workers() const { return num_workers_; }
    size_t get_queue_depth() const { return queue_depth_; }

private:
    struct Request { uint64_t file_offset; uint64_t len; uint8_t* dest; };

    // Read-only file handle, closed on scope exit.
    struct File {
#if defined(_WIN32)
        HANDLE h;
        explicit File(const std::string& path) {
            h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
            if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("Failed to open file: " + path);
        }
        ~File() { CloseHandle(h); }
#else
        int fd;
        explicit File(const std::string& path) {
            fd = open(path.c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("Failed to open file: " + path);
        }
        ~File() { close(fd); }
#endif
        File(const File&) = delete;
        File& operator=(const File&) = delete;
    };

    // Reads exactly len bytes at offset (looping over short reads); throws on error or premature EOF.
    static void read_at(const File& f, uint8_t* dest, uint64_t len, uint64_t offset) {
        uint64_t done = 0;
        while (done < len) {
#if defined(_WIN32)
            OVERLAPPED ol;
            std::memset(&ol, 0, sizeof(ol));
            uint64_t off = offset + done;
            ol.Offset = static_cast<DWORD>(off & 0xFFFFFFFF);
            ol.OffsetHigh = static_cast<DWORD>(off >> 32);
            DWORD chunk = static_cast<DWORD>(std::min<uint64_t>(len - done, 1u << 30));
            DWORD n = 0;
            if (!ReadFile(f.h, dest + done, chunk, &n, &ol) || n == 0) {
                throw std::runtime_error("ReadFile failed or premature EOF");
            }
#else
            ssize_t n = pread(f.fd, dest + done, len - done, static_cast<off_t>(offset + done));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) throw std::runtime_error("pread failed or premature EOF");
#endif
            done += static_cast<uint64_t>(n);
        }
    }

#if defined(TURING_IO_URING)
    // One ring of queue_depth entries; each completion either finishes its request or resubmits the remainder.
    void read_uring(int fd, const std::vector<Request>& requests) const {
        const unsigned depth = static_cast<unsigned>(std::min<size_t>(std::max<size_t>(queue_depth_, 1),
                                                                      std::min<size_t>(requests.size(), 4096)));
        io_uring ring;
        int rc = io_uring_queue_init(depth, &ring, 0);
        if (rc < 0) throw std::runtime_error(std::string("io_uring_queue_init failed: ") + std::strerror(-rc));

        std::vector<uint64_t> done(requests.size(), 0);
        size_t next = 0, in_flight = 0;
        std::string error;
        auto queue = [&](size_t i) {
            io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            if (!sqe) {  // only if the SQ is full of unsubmitted entries; flush and retry
                io_uring_submit(&ring);
                sqe = io_uring_get_sqe(&ring);
            }
            const Request& r = requests[i];
            io_uring_prep_read(sqe, fd, r.dest + done[i], static_cast<unsigned>(r.len - done[i]), r.file_offset + done[i]);
            io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(static_cast<uintptr_t>(i)));
        };

        while (in_flight > 0 || (error.empty() && next < requests.size())) {
            while (error.empty() && in_flight < depth && next < requests.size()) {
                queue(next++);
                ++in_flight;
            }
            rc = io_uring_submit_and_wait(&ring, 1);
            const bool fatal = rc < 0 && rc != -EINTR && rc != -EAGAIN && rc != -EBUSY;
            if (fatal && error.empty()) error = std::string("io_uring_submit_and_wait failed: ") + std::strerror(-rc);
            io_uring_cqe* cqe = nullptr;
            unsigned head, seen = 0;
            io_uring_for_each_cqe(&ring, head, cqe) {
                ++seen;
                size_t i = static_cast<size_t>(reinterpret_cast<uintptr_t>(io_uring_cqe_get_data(cqe)));
                int res = cqe->res;
                if (res > 0) done[i] += static_cast<uint64_t>(res);
                bool retry = (res == -EINTR || res == -EAGAIN || res > 0) && done[i] < requests[i].len;
                if (res == 0 || (res < 0 && !retry)) {
                    if (error.empty()) error = res == 0 ? "io_uring read hit premature EOF"
                                                        : std::string("io_uring read failed: ") + std::strerror(-res);
                }
                if (retry && error.empty()) {
                    queue(i);  // short read: the remainder goes back on the ring, in_flight unchanged
                } else {
                    --in_flight;
                }
            }
            io_uring_cq_advance(&ring, seen);
            // shortcut: a ring that can neither submit nor complete is abandoned; never seen, revisit if it is.
            if (fatal && seen == 0) break;
        }
        io_uring_queue_exit(&ring);
        if (!error.empty()) throw std::runtime_error(error);
    }
#endif

    int num_workers_;
    size_t queue_depth_;
    bool use_uring_ = false;
};

// The pre-1.8 name; the reader never was a ring on non-Linux builds.
using NativeAsyncRingReader = ParallelReader;

} // namespace turing
