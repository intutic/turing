#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <cerrno>
#include <mutex>
#include <algorithm>

#if defined(__linux__)
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#endif

namespace turing {

/**
 * File -> GPU memory reads (Tier 4).
 * Two paths, both loaded at run time with dlopen so the extension never links CUDA:
 *   "cufile"           - NVIDIA GPUDirect Storage: libcufile's cuFileRead DMAs from the file into device memory;
 *   "pinned_host_copy" - libcudart only: pread into pinned host memory, cudaMemcpy to the device, chunk by chunk.
 * Without a CUDA runtime (macOS, Windows, a Linux host without CUDA) read_to_device reads nothing and says
 * "no_cuda". A path is reported only after all num_bytes have landed on the device.
 */
class NativeGDSLoader {
public:
    struct ReadResult {
        uint64_t bytes = 0;
        std::string path = "no_cuda";
    };

    NativeGDSLoader() {
#if defined(__linux__)
        for (const char* name : {"libcudart.so", "libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0"}) {
            if ((cudart_ = dlopen(name, RTLD_NOW | RTLD_LOCAL))) break;
        }
        if (cudart_) {
            cuda_malloc_host_ = reinterpret_cast<int (*)(void**, size_t)>(dlsym(cudart_, "cudaMallocHost"));
            cuda_free_host_ = reinterpret_cast<int (*)(void*)>(dlsym(cudart_, "cudaFreeHost"));
            cuda_memcpy_ = reinterpret_cast<int (*)(void*, const void*, size_t, int)>(dlsym(cudart_, "cudaMemcpy"));
            cuda_error_string_ = reinterpret_cast<const char* (*)(int)>(dlsym(cudart_, "cudaGetErrorString"));
            has_cuda_ = cuda_malloc_host_ && cuda_free_host_ && cuda_memcpy_;
        }
        for (const char* name : {"libcufile.so", "libcufile.so.0"}) {
            if ((cufile_ = dlopen(name, RTLD_NOW | RTLD_LOCAL))) break;
        }
        if (cufile_ && has_cuda_) {
            driver_open_ = reinterpret_cast<CUfileError (*)()>(dlsym(cufile_, "cuFileDriverOpen"));
            handle_register_ = reinterpret_cast<CUfileError (*)(void**, CUfileDescr*)>(dlsym(cufile_, "cuFileHandleRegister"));
            handle_deregister_ = reinterpret_cast<void (*)(void*)>(dlsym(cufile_, "cuFileHandleDeregister"));
            file_read_ = reinterpret_cast<ssize_t (*)(void*, void*, size_t, off_t, off_t)>(dlsym(cufile_, "cuFileRead"));
            cufile_ok_ = driver_open_ && handle_register_ && handle_deregister_ && file_read_;
        }
#endif
    }

    ~NativeGDSLoader() {
#if defined(__linux__)
        if (cufile_) dlclose(cufile_);
        if (cudart_) dlclose(cudart_);
#endif
    }

    NativeGDSLoader(const NativeGDSLoader&) = delete;
    NativeGDSLoader& operator=(const NativeGDSLoader&) = delete;

    // libcufile and its entry points were found (the GPUDirect path can be tried).
    bool is_available() const { return cufile_ok_; }

    // A CUDA runtime was found (the pinned-host fallback works).
    bool has_cuda() const { return has_cuda_; }

    /**
     * Reads num_bytes at file_offset into device memory at dev_ptr. Tries cuFile first (unless allow_cufile is
     * false), falls back to pinned host memory + cudaMemcpy, and returns which path ran. Throws on I/O or CUDA
     * errors; without a CUDA runtime returns {0, "no_cuda"} having read nothing.
     */
    ReadResult read_to_device(const std::string& filepath, uint64_t file_offset, uint64_t num_bytes, void* dev_ptr,
                              bool allow_cufile = true) {
        ReadResult result;
        if (!has_cuda_) return result;
        if (!dev_ptr) throw std::invalid_argument("dev_ptr is null");
#if defined(__linux__)
        if (allow_cufile && cufile_ok_ && read_cufile(filepath, file_offset, num_bytes, dev_ptr)) {
            return {num_bytes, "cufile"};
        }
        read_pinned(filepath, file_offset, num_bytes, dev_ptr);
        return {num_bytes, "pinned_host_copy"};
#else
        (void)filepath; (void)file_offset; (void)num_bytes; (void)allow_cufile;
        return result;
#endif
    }

    std::string get_status_info() const {
        if (cufile_ok_) return "GPUDirect Storage (libcufile) found; reads try cuFileRead, then pinned host copies";
        if (has_cuda_) return "GPUDirect Storage unavailable; reads use pinned host memory + cudaMemcpy";
        return "No CUDA runtime loaded (read_to_device reads nothing and reports no_cuda)";
    }

private:
    // cufile.h ABI: CUfileError_t {CUfileOpError err; CUresult cu_err;}, CUfileDescr_t {type; union {int fd; void*
    // handle;}; const CUfileFSOps_t* fs_ops;}, CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1.
    struct CUfileError { int err; int cu_err; };
    struct CUfileDescr { int type; union { int fd; void* handle; } handle; const void* fs_ops; };

    bool has_cuda_ = false;
    bool cufile_ok_ = false;
#if defined(__linux__)
    void* cudart_ = nullptr;
    void* cufile_ = nullptr;
    int (*cuda_malloc_host_)(void**, size_t) = nullptr;
    int (*cuda_free_host_)(void*) = nullptr;
    int (*cuda_memcpy_)(void*, const void*, size_t, int) = nullptr;
    const char* (*cuda_error_string_)(int) = nullptr;
    CUfileError (*driver_open_)() = nullptr;
    CUfileError (*handle_register_)(void**, CUfileDescr*) = nullptr;
    void (*handle_deregister_)(void*) = nullptr;
    ssize_t (*file_read_)(void*, void*, size_t, off_t, off_t) = nullptr;
    std::once_flag driver_once_;
    bool driver_open_ok_ = false;

    std::string cuda_error(int code) const {
        return cuda_error_string_ ? cuda_error_string_(code) : ("CUDA error " + std::to_string(code));
    }

    // false when GPUDirect cannot serve this read (driver, registration or read failure): the caller falls back.
    bool read_cufile(const std::string& filepath, uint64_t file_offset, uint64_t num_bytes, void* dev_ptr) {
        std::call_once(driver_once_, [this] { driver_open_ok_ = driver_open_().err == 0; });
        if (!driver_open_ok_) return false;
        int fd = open(filepath.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) fd = open(filepath.c_str(), O_RDONLY);  // filesystems without O_DIRECT: cuFile's compat mode
        if (fd < 0) throw std::runtime_error("Failed to open file: " + filepath);
        CUfileDescr descr{};
        descr.type = 1;
        descr.handle.fd = fd;
        void* handle = nullptr;
        bool ok = handle_register_(&handle, &descr).err == 0;
        uint64_t done = 0;
        while (ok && done < num_bytes) {
            ssize_t n = file_read_(handle, dev_ptr, num_bytes - done, static_cast<off_t>(file_offset + done),
                                   static_cast<off_t>(done));
            if (n <= 0) ok = false;
            else done += static_cast<uint64_t>(n);
        }
        if (handle) handle_deregister_(handle);
        close(fd);
        return ok;
    }

    void read_pinned(const std::string& filepath, uint64_t file_offset, uint64_t num_bytes, void* dev_ptr) {
        constexpr uint64_t kChunk = 8ull << 20;
        int fd = open(filepath.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Failed to open file: " + filepath);
        void* host = nullptr;
        int rc = cuda_malloc_host_(&host, static_cast<size_t>(std::min(kChunk, std::max<uint64_t>(num_bytes, 1))));
        if (rc != 0) {
            close(fd);
            throw std::runtime_error("cudaMallocHost failed: " + cuda_error(rc));
        }
        std::string error;
        for (uint64_t done = 0; done < num_bytes && error.empty();) {
            uint64_t want = std::min(kChunk, num_bytes - done), got = 0;
            while (got < want) {
                ssize_t n = pread(fd, static_cast<uint8_t*>(host) + got, want - got, static_cast<off_t>(file_offset + done + got));
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) { error = "pread failed or premature EOF for: " + filepath; break; }
                got += static_cast<uint64_t>(n);
            }
            if (!error.empty()) break;
            rc = cuda_memcpy_(static_cast<uint8_t*>(dev_ptr) + done, host, want, 1 /* cudaMemcpyHostToDevice */);
            if (rc != 0) error = "cudaMemcpy failed: " + cuda_error(rc);
            done += want;
        }
        cuda_free_host_(host);
        close(fd);
        if (!error.empty()) throw std::runtime_error(error);
    }
#endif
};

} // namespace turing
