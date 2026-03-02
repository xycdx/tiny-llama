// ops_basic.cu — softmax, rms_norm, layer_norm, matmul (cuBLAS), bmm, linear
//
// CUDA kernel 设计原则：
//   - softmax / norm：每个 block 处理一行（outer 维），warp reduce 做规约
//   - matmul / bmm：调用 cuBLAS SGEMM / SGEMM_BATCHED，达到接近峰值吞吐
//   - linear：转化为 matmul 后广播加 bias

#include "tinyllama/ops.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>
#include <stdexcept>
#include <cmath>

namespace tinyllama {
namespace ops {

// ─────────────────────────────────────────────────────────────────────────────
//  cuBLAS handle（进程内单例）
// ─────────────────────────────────────────────────────────────────────────────
static cublasHandle_t g_cublas_handle = nullptr;

static cublasHandle_t cublas_handle() {
    if (!g_cublas_handle) {
        cublasStatus_t st = cublasCreate(&g_cublas_handle);
        if (st != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasCreate failed");
    }
    return g_cublas_handle;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Warp-level reduce helpers
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Softmax kernel
//  Grid : (outer,)   Block : min(last_dim, 1024) threads
//  每个 block 处理 x[i, :]，先规约求 max，再求 sum，最后归一化。
// ─────────────────────────────────────────────────────────────────────────────
__global__ void softmax_kernel(float* x, int64_t last_dim) {
    extern __shared__ float smem[];   // size = blockDim.x

    int row   = blockIdx.x;
    float*  row_ptr = x + row * last_dim;
    int tid = threadIdx.x;

    // Step 1: parallel max
    float maxval = -1e38f;
    for (int i = tid; i < last_dim; i += blockDim.x)
        maxval = fmaxf(maxval, row_ptr[i]);
    maxval = warp_reduce_max(maxval);
    if (tid % 32 == 0) smem[tid / 32] = maxval;
    __syncthreads();
    if (tid < (blockDim.x + 31) / 32)
        maxval = smem[tid];
    maxval = warp_reduce_max(maxval);

    // Step 2: exp & parallel sum
    float sumval = 0.0f;
    for (int i = tid; i < last_dim; i += blockDim.x) {
        float e = expf(row_ptr[i] - maxval);
        row_ptr[i] = e;
        sumval += e;
    }
    sumval = warp_reduce_sum(sumval);
    if (tid % 32 == 0) smem[tid / 32] = sumval;
    __syncthreads();
    if (tid < (blockDim.x + 31) / 32)
        sumval = smem[tid];
    sumval = warp_reduce_sum(sumval);

    // Step 3: normalize
    float inv = 1.0f / sumval;
    for (int i = tid; i < last_dim; i += blockDim.x)
        row_ptr[i] *= inv;
}

void softmax_(Tensor& x) {
    assert(x.device() == Device::CUDA);
    int64_t last_dim = x.dim(-1);
    int64_t outer    = x.numel() / last_dim;
    int threads = (int)min((int64_t)1024, last_dim);
    int smem    = (threads / 32 + 1) * sizeof(float);
    softmax_kernel<<<(int)outer, threads, smem>>>(x.data_ptr<float>(), last_dim);
    CUDA_CHECK(cudaGetLastError());
}

Tensor softmax(const Tensor& x) {
    Tensor out = x.contiguous();
    softmax_(out);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  RMS Norm kernel
//  Grid: (outer,)  Block: min(hidden, 1024)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void rms_norm_kernel(const float* __restrict__ x,
                                 const float* __restrict__ w,
                                 float* __restrict__ out,
                                 int64_t hidden, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_in  = x   + row * hidden;
    float*       row_out = out + row * hidden;

    // sum of squares
    float ss = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x)
        ss += row_in[i] * row_in[i];
    ss = warp_reduce_sum(ss);
    if (tid % 32 == 0) smem[tid / 32] = ss;
    __syncthreads();
    if (tid < (blockDim.x + 31) / 32) ss = smem[tid];
    ss = warp_reduce_sum(ss);

    float scale = rsqrtf(ss / (float)hidden + eps);
    for (int i = tid; i < hidden; i += blockDim.x)
        row_out[i] = row_in[i] * scale * w[i];
}

Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) {
    assert(x.device() == Device::CUDA && weight.device() == Device::CUDA);
    int64_t hidden = x.dim(-1);
    assert(weight.numel() == hidden);
    Tensor out(x.shape(), DType::Float32, Device::CUDA);
    int64_t outer   = x.numel() / hidden;
    int     threads = (int)min((int64_t)1024, hidden);
    int     smem    = (threads / 32 + 1) * sizeof(float);
    rms_norm_kernel<<<(int)outer, threads, smem>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(),
        out.data_ptr<float>(), hidden, eps);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Layer Norm kernel
//  Grid: (outer,)  Block: min(hidden, 1024)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void layer_norm_kernel(const float* __restrict__ x,
                                   const float* __restrict__ w,
                                   const float* __restrict__ b,
                                   float* __restrict__ out,
                                   int64_t hidden, float eps) {
    extern __shared__ float smem[];   // [0..W-1]=sum, [W..2W-1]=sum_sq (W=warp_count)
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_in  = x   + row * hidden;
    float*       row_out = out + row * hidden;
    int warp_count = (blockDim.x + 31) / 32;

    // mean
    float sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) sum += row_in[i];
    sum = warp_reduce_sum(sum);
    if (tid % 32 == 0) smem[tid / 32] = sum;
    __syncthreads();
    if (tid < warp_count) sum = smem[tid];
    sum = warp_reduce_sum(sum);
    float mean = sum / (float)hidden;

    // variance
    float var = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float d = row_in[i] - mean;
        var += d * d;
    }
    var = warp_reduce_sum(var);
    if (tid % 32 == 0) smem[tid / 32] = var;
    __syncthreads();
    if (tid < warp_count) var = smem[tid];
    var = warp_reduce_sum(var);
    var /= (float)hidden;

    float inv_std = rsqrtf(var + eps);
    for (int i = tid; i < hidden; i += blockDim.x)
        row_out[i] = (row_in[i] - mean) * inv_std * w[i] + b[i];
}

Tensor layer_norm(const Tensor& x, const Tensor& weight,
                  const Tensor& bias, float eps) {
    assert(x.device() == Device::CUDA);
    int64_t hidden = x.dim(-1);
    assert(weight.numel() == hidden && bias.numel() == hidden);
    Tensor out(x.shape(), DType::Float32, Device::CUDA);
    int64_t outer   = x.numel() / hidden;
    int     threads = (int)min((int64_t)1024, hidden);
    int     smem    = (threads / 32 + 1) * sizeof(float);
    layer_norm_kernel<<<(int)outer, threads, smem>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(),
        bias.data_ptr<float>(), out.data_ptr<float>(), hidden, eps);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Bias add kernel（linear 用）
//  Grid: (outer,)  Block: min(out_feat, 1024)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void bias_add_kernel(float* out, const float* bias,
                                 int64_t out_feat) {
    int row = blockIdx.x;
    float* row_ptr = out + row * out_feat;
    for (int i = threadIdx.x; i < out_feat; i += blockDim.x)
        row_ptr[i] += bias[i];
}

// ─────────────────────────────────────────────────────────────────────────────
//  MatMul via cuBLAS SGEMM
//  C = alpha * op(A) @ op(B) + beta * C
//  cuBLAS 是 column-major；row-major 的 A@B 等价于 col-major 的 B^T @ A^T。
// ─────────────────────────────────────────────────────────────────────────────
Tensor matmul(const Tensor& A_in, const Tensor& B_in,
              float alpha, float beta, bool trans_A, bool trans_B) {
    Tensor A = A_in.contiguous();
    Tensor B = B_in.contiguous();

    // Row-major M×K  @  K×N  →  M×N
    int64_t M  = trans_A ? A.dim(1) : A.dim(0);
    int64_t Ka = trans_A ? A.dim(0) : A.dim(1);
    int64_t Kb = trans_B ? B.dim(1) : B.dim(0);
    int64_t N  = trans_B ? B.dim(0) : B.dim(1);
    if (Ka != Kb) throw std::runtime_error("matmul: inner dim mismatch");
    int64_t K = Ka;

    Tensor C({M, N}, DType::Float32, Device::CUDA);

    // cuBLAS col-major trick: C^T = B^T @ A^T
    // op(A_row) @ op(B_row) → in col-major: op_col(B) @ op_col(A)
    cublasOperation_t opA = trans_A ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t opB = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;
    // cuBLAS sees: C(N×M) = opA(B,N×K) @ opB(A,K×M)
    int lda = (int)(trans_B ? K : N);   // leading dim of B in col-major
    int ldb = (int)(trans_A ? M : K);   // leading dim of A in col-major
    int ldc = (int)N;

    cublasStatus_t st = cublasSgemm(
        cublas_handle(),
        opA, opB,
        (int)N, (int)M, (int)K,
        &alpha,
        B.data_ptr<float>(), lda,
        A.data_ptr<float>(), ldb,
        &beta,
        C.data_ptr<float>(), ldc);
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasSgemm failed");
    return C;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Batched MatMul via cuBLAS SGEMM_STRIDED_BATCHED
// ─────────────────────────────────────────────────────────────────────────────
Tensor bmm(const Tensor& A_in, const Tensor& B_in,
           bool trans_A, bool trans_B) {
    Tensor A = A_in.contiguous();
    Tensor B = B_in.contiguous();
    assert(A.ndim() == 3 && B.ndim() == 3);
    int64_t batch = A.dim(0);
    assert(B.dim(0) == batch);

    int64_t M  = trans_A ? A.dim(2) : A.dim(1);
    int64_t Ka = trans_A ? A.dim(1) : A.dim(2);
    int64_t Kb = trans_B ? B.dim(2) : B.dim(1);
    int64_t N  = trans_B ? B.dim(1) : B.dim(2);
    assert(Ka == Kb);
    int64_t K = Ka;

    Tensor C({batch, M, N}, DType::Float32, Device::CUDA);

    cublasOperation_t opA = trans_A ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t opB = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = (int)(trans_B ? K : N);
    int ldb = (int)(trans_A ? M : K);
    int ldc = (int)N;
    long long strideA = (long long)(B.dim(1) * B.dim(2));
    long long strideB = (long long)(A.dim(1) * A.dim(2));
    long long strideC = (long long)(M * N);
    float alpha = 1.0f, beta = 0.0f;

    cublasStatus_t st = cublasSgemmStridedBatched(
        cublas_handle(),
        opA, opB,
        (int)N, (int)M, (int)K,
        &alpha,
        B.data_ptr<float>(), lda, strideA,
        A.data_ptr<float>(), ldb, strideB,
        &beta,
        C.data_ptr<float>(), ldc, strideC,
        (int)batch);
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasSgemmStridedBatched failed");
    return C;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Linear  y = x @ W^T + b
// ─────────────────────────────────────────────────────────────────────────────
Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    int64_t in_feat  = weight.dim(1);
    int64_t out_feat = weight.dim(0);
    int64_t outer    = x.numel() / in_feat;

    Tensor x2d    = x.view({outer, in_feat});
    Tensor out2d  = matmul(x2d, weight, 1.0f, 0.0f, false, true);

    if (bias.numel() > 0) {
        int threads = (int)min((int64_t)1024, out_feat);
        bias_add_kernel<<<(int)outer, threads>>>(
            out2d.data_ptr<float>(), bias.data_ptr<float>(), out_feat);
        CUDA_CHECK(cudaGetLastError());
    }

    auto new_shape = x.shape();
    new_shape.back() = out_feat;
    return out2d.view(new_shape);
}

}  // namespace ops
}  // namespace tinyllama
