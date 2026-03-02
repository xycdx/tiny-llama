// ops_rope.cu — Rotary Position Embedding (RoPE) CUDA kernel
//
// 对 Q/K 张量 [batch, seq, n_heads, head_dim] 做原地旋转编码：
//   [ x0' ]   [ cos(θ)  -sin(θ) ] [ x0 ]
//   [ x1' ] = [ sin(θ)   cos(θ) ] [ x1 ]
//
// θ_i = pos / theta^(2i / head_dim)，半程旋转：前半 head_dim/2 与后半配对。

#include "tinyllama/ops.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <cassert>

namespace tinyllama {
namespace ops {

// ─────────────────────────────────────────────────────────────────────────────
//  RoPE kernel
//  每个 thread 负责一个 (batch, seq, head, freq) 四元组中的一对 (x0, x1)。
//
//  Grid : (batch * seq * n_heads, head_dim/2)  — 1D展开
//  Block: 1
//  （对于大 head_dim 可以合并，但这样最清晰）
//
//  实际发布时用 1 block/row 更高效；此处用扁平化 grid 兼顾可读性。
// ─────────────────────────────────────────────────────────────────────────────
__global__ void rope_kernel(float* __restrict__ t,
                             int64_t seq, int64_t n_heads,
                             int64_t head_dim, int64_t half,
                             int64_t pos_offset, float theta) {
    // 全局线程 id → (b_s_h, i)
    int64_t bsh = blockIdx.x;          // batch * seq * n_heads 中的第几个
    int64_t i   = threadIdx.x + (int64_t)blockIdx.y * blockDim.x;
    if (i >= half) return;

    // 还原 (b, s, h)
    int64_t h = bsh % n_heads;
    int64_t s = (bsh / n_heads) % seq;
    int64_t b = bsh / (n_heads * seq);

    // 指向该 head 向量起始
    float* vec = t + ((b * seq + s) * n_heads + h) * head_dim;

    int64_t pos   = pos_offset + s;
    float   freq  = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
    float   angle = (float)pos * freq;
    float   cos_a, sin_a;
    sincosf(angle, &sin_a, &cos_a);

    float x0 = vec[i];
    float x1 = vec[i + half];
    vec[i]        = x0 * cos_a - x1 * sin_a;
    vec[i + half] = x0 * sin_a + x1 * cos_a;
}

static void apply_rope_cuda(Tensor& t, int64_t pos_offset, float theta) {
    assert(t.device() == Device::CUDA);
    assert(t.is_contiguous());

    int64_t batch, seq, n_heads, head_dim;
    if (t.ndim() == 3) {
        batch = 1; seq = t.dim(0); n_heads = t.dim(1); head_dim = t.dim(2);
    } else if (t.ndim() == 4) {
        batch = t.dim(0); seq = t.dim(1); n_heads = t.dim(2); head_dim = t.dim(3);
    } else {
        throw std::runtime_error("RoPE: expected 3D or 4D tensor");
    }
    assert(head_dim % 2 == 0);
    int64_t half = head_dim / 2;

    // Grid: (batch*seq*n_heads, ceil(half/32))
    int threads_x = 32;
    dim3 grid((int)(batch * seq * n_heads), (int)((half + threads_x - 1) / threads_x));
    rope_kernel<<<grid, threads_x>>>(
        t.data_ptr<float>(),
        seq, n_heads, head_dim, half,
        pos_offset, theta);
    CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
//  rope_precompute_freqs（CPU 端生成，供调试/可视化用）
// ─────────────────────────────────────────────────────────────────────────────
Tensor rope_precompute_freqs(int64_t max_seq_len, int64_t head_dim, float theta) {
    assert(head_dim % 2 == 0);
    int64_t half = head_dim / 2;
    // 先在 CPU 生成，再上传 GPU
    Tensor cpu_t({max_seq_len, half, 2}, DType::Float32, Device::CPU);
    float* pf = cpu_t.data_ptr<float>();
    for (int64_t pos = 0; pos < max_seq_len; ++pos) {
        for (int64_t i = 0; i < half; ++i) {
            float freq  = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;
            int64_t idx = (pos * half + i) * 2;
            pf[idx]     = cosf(angle);
            pf[idx + 1] = sinf(angle);
        }
    }
    return cpu_t.cuda();
}

// ─────────────────────────────────────────────────────────────────────────────
//  rope_ — 对 Q 和 K 原地施加 RoPE
// ─────────────────────────────────────────────────────────────────────────────
void rope_(Tensor& q, Tensor& k, int64_t pos_offset, float theta) {
    apply_rope_cuda(q, pos_offset, theta);
    apply_rope_cuda(k, pos_offset, theta);
}

}  // namespace ops
}  // namespace tinyllama
