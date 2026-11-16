cmake_minimum_required(VERSION 3.20)
project(turing_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
