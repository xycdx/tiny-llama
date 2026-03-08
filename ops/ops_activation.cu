// ops_activation.cu — SiLU, GELU, element-wise mul, embedding lookup (CUDA)

#include "tinyllama/ops.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cassert>
#include <stdexcept>

namespace tinyllama {
namespace ops {

// ─────────────────────────────────────────────────────────────────────────────
//  SiLU kernel:  out[i] = x[i] / (1 + exp(-x[i]))
// ─────────────────────────────────────────────────────────────────────────────
__global__ void silu_kernel(const float* __restrict__ x,
                             float*       __restrict__ out,
                             int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        out[i]  = v / (1.0f + expf(-v));
    }
}

Tensor silu(const Tensor& x) {
    assert(x.device() == Device::CUDA);
    Tensor out(x.shape(), DType::Float32, Device::CUDA);
    int64_t n = x.numel();
    int threads = 256;
    silu_kernel<<<(int)((n + threads - 1) / threads), threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Element-wise multiply kernel
// ─────────────────────────────────────────────────────────────────────────────
__global__ void mul_kernel(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float*       __restrict__ out,
                            int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

Tensor mul(const Tensor& a, const Tensor& b) {
    assert(a.device() == Device::CUDA && b.device() == Device::CUDA);
    assert(a.numel() == b.numel());
    Tensor out(a.shape(), DType::Float32, Device::CUDA);
    int64_t n = a.numel();
    int threads = 256;
    mul_kernel<<<(int)((n + threads - 1) / threads), threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Element-wise add kernel (for residual connections)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void add_kernel(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float*       __restrict__ out,
                            int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

Tensor add(const Tensor& a, const Tensor& b) {
    assert(a.device() == Device::CUDA && b.device() == Device::CUDA);
    assert(a.numel() == b.numel());
    Tensor out(a.shape(), DType::Float32, Device::CUDA);
    int64_t n = a.numel();
    int threads = 256;
    add_kernel<<<(int)((n + threads - 1) / threads), threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  GELU kernel (tanh approximation)
//  f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
// ─────────────────────────────────────────────────────────────────────────────
__global__ void gelu_kernel(const float* __restrict__ x,
                             float*       __restrict__ out,
                             int64_t n) {
    constexpr float kS = 0.7978845608028654f;  // sqrt(2/pi)
    constexpr float kC = 0.044715f;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        out[i]  = 0.5f * v * (1.0f + tanhf(kS * (v + kC * v * v * v)));
    }
}

Tensor gelu(const Tensor& x) {
    assert(x.device() == Device::CUDA);
    Tensor out(x.shape(), DType::Float32, Device::CUDA);
    int64_t n = x.numel();
    int threads = 256;
    gelu_kernel<<<(int)((n + threads - 1) / threads), threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Embedding lookup kernel
//  每个 thread 负责输出 tensor 中的一个元素 out[i, j]。
//  grid: (n_ids,)  block: min(d_model, 1024)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void embedding_kernel(const float*   __restrict__ table,
                                  const int32_t* __restrict__ ids,
                                  float*         __restrict__ out,
                                  int64_t d_model, int64_t vocab_size) {
    int64_t token_pos = blockIdx.x;
    int32_t token_id  = ids[token_pos];
    // bounds check (device-side assert)
    assert(token_id >= 0 && (int64_t)token_id < vocab_size);

    const float* row = table + (int64_t)token_id * d_model;
    float*       dst = out   + token_pos          * d_model;
    for (int d = threadIdx.x; d < d_model; d += blockDim.x)
        dst[d] = row[d];
}

Tensor embedding(const Tensor& table, const Tensor& ids) {
    assert(table.ndim() == 2 && table.device() == Device::CUDA);
    assert(ids.device() == Device::CUDA);
    if (ids.dtype() != DType::Int32)
        throw std::runtime_error("embedding: ids must have dtype Int32");

    int64_t vocab_size = table.dim(0);
    int64_t d_model    = table.dim(1);
    int64_t n_ids      = ids.numel();

    auto out_shape = ids.shape();
    out_shape.push_back(d_model);
    Tensor out(out_shape, DType::Float32, Device::CUDA);

    int threads = (int)min(d_model, (int64_t)1024);
    embedding_kernel<<<(int)n_ids, threads>>>(
        table.data_ptr<float>(),
        ids.data_ptr<int32_t>(),
        out.data_ptr<float>(),
        d_model, vocab_size);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

}  // namespace ops
}  // namespace tinyllama
