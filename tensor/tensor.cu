// tensor.cu — CUDA device memory management helpers
// 这里是唯一直接调用 CUDA runtime API 的地方。
// tensor.h 中声明的函数在此实现，所有 .cu 算子文件共享。

#include "tinyllama/tensor.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace tinyllama {

std::shared_ptr<void> cuda_alloc(size_t bytes) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return std::shared_ptr<void>(ptr, [](void* p){ cudaFree(p); });
}

void cuda_memset_zero(void* ptr, size_t bytes) {
    CUDA_CHECK(cudaMemset(ptr, 0, bytes));
}

void cuda_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

void cuda_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

void cuda_memcpy_d2d(void* dst, const void* src, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
}

}  // namespace tinyllama
