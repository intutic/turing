cmake_minimum_required(VERSION 3.20)
project(turing_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile options
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    add_compile_options(-O3 -fvisibility=hidden)
    if (NOT APPLE)
        add_compile_options(-mavx2 -mfma)
    endif()
elseif (MSVC)
    add_compile_options(/O2 /arch:AVX2 /EHsc /utf-8)
endif()

# Include directories
include_directories(csrc)

# Standalone executable
add_executable(turing-cli csrc/turing_main.cpp)

# Output directory
set_target_properties(turing-cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
