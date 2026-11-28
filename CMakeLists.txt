cmake_minimum_required(VERSION 3.20)
project(turing_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional CPU backend on libllama (third_party/llama.cpp, a pinned submodule): -DTURING_LLAMA=ON builds the Python module
# turing_llama and turing-cpu. Ahead of the options below, so llama.cpp builds with its own flags (no LTO, its visibility).
option(TURING_LLAMA "Build the CPU backend on libllama (third_party/llama.cpp)" OFF)
if (TURING_LLAMA)
    if (NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
        message(FATAL_ERROR "TURING_LLAMA=ON needs the submodule: git submodule update --init third_party/llama.cpp")
    endif()
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # its static libraries go into a Python module
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
    set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")  # ggml refuses KleidiAI on other targets
        set(GGML_CPU_KLEIDIAI ON CACHE BOOL "ARM: KleidiAI kernels")
    endif()
    # Metal (Apple silicon GPU) is opt-in: a CPU backend by default. With it, LlamaOptions::n_gpu_layers says how many
    # layers run on the GPU; 0 keeps everything on the CPU (op offload is then off too, so prompts stay on the CPU).
    option(TURING_LLAMA_METAL "libllama backend: also build ggml's Metal GPU backend" OFF)
    set(GGML_METAL ${TURING_LLAMA_METAL} CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ${TURING_LLAMA_METAL} CACHE BOOL "" FORCE)  # the shaders inside the module, not beside it
    set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)  # libmtmd alone: image input through a model's vision projector (mmproj)
    # Turing's patches to the pinned llama.cpp (third_party/patches, decision CF5), applied once at configure time: a
    # patch that already applies in reverse is in. They also cover its Python converter, so configure before converting.
    file(GLOB TURING_LLAMA_PATCHES "${CMAKE_SOURCE_DIR}/third_party/patches/*.patch")
    list(SORT TURING_LLAMA_PATCHES)
    find_package(Git REQUIRED)
    foreach(patch ${TURING_LLAMA_PATCHES})
        execute_process(COMMAND ${GIT_EXECUTABLE} apply --reverse --check ${patch}
                        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/llama.cpp RESULT_VARIABLE in OUTPUT_QUIET ERROR_QUIET)
        if (NOT in EQUAL 0)
            execute_process(COMMAND ${GIT_EXECUTABLE} apply ${patch}
                            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/llama.cpp RESULT_VARIABLE rc)
            if (NOT rc EQUAL 0)
                message(FATAL_ERROR "cannot apply ${patch} to third_party/llama.cpp (is the submodule at the pinned commit?)")
            endif()
            message(STATUS "applied ${patch}")
        endif()
    endforeach()
    add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(turing_llama csrc/cpu/llama_bindings.cpp)
    target_include_directories(turing_llama PRIVATE csrc)
    target_link_libraries(turing_llama PRIVATE llama mtmd)
    add_executable(turing-cpu csrc/cpu/turing_cpu_main.cpp)
    target_include_directories(turing-cpu PRIVATE csrc)
    target_link_libraries(turing-cpu PRIVATE llama mtmd)
    set_target_properties(turing-cpu PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_executable(turing-teal-bench csrc/cpu/teal_bench.cpp)  # Wave 5 go / no-go microbenchmark
    target_link_libraries(turing-teal-bench PRIVATE ggml)
    if (NOT MSVC)
        target_compile_options(turing-teal-bench PRIVATE -march=native)  # as ggml's own kernels (GGML_NATIVE)
    endif()
    set_target_properties(turing-teal-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

# Compile options: AVX2/FMA/F16C on x86_64 only; 64-bit ARM gets NEON by default.
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    add_compile_options(-O3 -flto -fvisibility=hidden)
    add_link_options(-flto)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        add_compile_options(-mavx2 -mfma -mf16c)
    endif()
elseif (MSVC)
    add_compile_options(/O2 /EHsc /utf-8 /GL)
    add_link_options(/LTCG)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        add_compile_options(/arch:AVX2)
    endif()
endif()

# Include directories
include_directories(csrc)

# Standalone executable
add_executable(turing-cli csrc/turing_main.cpp)
if (WIN32)
    target_link_libraries(turing-cli PRIVATE ws2_32)
endif()

# Output directory
set_target_properties(turing-cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
