cmake_minimum_required(VERSION 3.18)
project(tiny-llama VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Build options ─────────────────────────────────────────────────────────────
option(TINYLLAMA_BUILD_TESTS "Build unit tests" ON)

# ── CUDA architecture ─────────────────────────────────────────────────────────
# 未指定时自动检测；可通过 -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" 覆盖
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# ── Compiler flags ────────────────────────────────────────────────────────────
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3> $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O0> $<$<COMPILE_LANGUAGE:CXX>:-g>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O0> $<$<COMPILE_LANGUAGE:CUDA>:-g> $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# ── cuBLAS ────────────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── third-party: safetensors-cpp (header-only, implementation compiled here) ──
set(SAFETENSORS_DIR ${CMAKE_SOURCE_DIR}/third_party/safetensors-cpp)

# ── Source files ──────────────────────────────────────────────────────────────
set(TENSOR_SRCS tensor/tensor.cu)
set(OPS_SRCS
    ops/ops_basic.cu
    ops/ops_rope.cu
    ops/ops_attention.cu
    ops/ops_activation.cu
)
set(LOADER_SRCS loader/loader.cc)
set(MODEL_SRCS model/qwen3_model.cc)

# ── Core library ──────────────────────────────────────────────────────────────
add_library(tinyllama STATIC
    ${TENSOR_SRCS}
    ${OPS_SRCS}
    ${LOADER_SRCS}
    ${MODEL_SRCS}
)
target_include_directories(tinyllama PUBLIC include)
target_include_directories(tinyllama PRIVATE ${SAFETENSORS_DIR})
target_include_directories(tinyllama PRIVATE ${CMAKE_SOURCE_DIR}/third_party/nlohmann)
target_link_libraries(tinyllama PUBLIC CUDA::cublas CUDA::cudart)

# ── Tests ─────────────────────────────────────────────────────────────────────
if(TINYLLAMA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Examples ──────────────────────────────────────────────────────────────────
if(EXISTS ${CMAKE_SOURCE_DIR}/examples/CMakeLists.txt)
    add_subdirectory(examples)
endif()
