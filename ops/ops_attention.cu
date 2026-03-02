// ops_attention.cu — Scaled Dot-Product Attention + Multi-Head Attention (CUDA)
//
// SDPA:
//   scores = Q @ K^T / sqrt(d_k)   [batch, seq_q, seq_k]
//   scores += mask (可选)
//   out    = softmax(scores) @ V    [batch, seq_q, head_dim]
//
// MHA:
//   Q,K,V = linear(x)
//   reshape → [batch, n_heads, seq, head_dim]
//   rope_(Q, K)
//   KV-cache 追加
//   per-head SDPA (via cuBLAS batched SGEMM)
//   reshape → [batch, seq, d_model]
//   output projection

#include "tinyllama/ops.h"
#include <cuda_runtime.h>
#include <cassert>
#include <stdexcept>
#include <cstring>
#include <cmath>

namespace tinyllama {
namespace ops {

// Forward declarations (defined in ops_basic.cu)
Tensor matmul(const Tensor&, const Tensor&, float, float, bool, bool);
Tensor bmm   (const Tensor&, const Tensor&, bool, bool);
Tensor linear(const Tensor&, const Tensor&, const Tensor&);
void   softmax_(Tensor&);

// ─────────────────────────────────────────────────────────────────────────────
//  Causal mask kernel
//  每个 thread 填充 mask[i, j]：j > i 则 -1e9f，否则 0。
// ─────────────────────────────────────────────────────────────────────────────
__global__ void causal_mask_kernel(float* mask, int64_t seq_len) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < seq_len && j < seq_len)
        mask[i * seq_len + j] = (j <= i) ? 0.0f : -1e9f;
}

static Tensor make_causal_mask_cuda(int64_t seq_len) {
    Tensor mask({seq_len, seq_len}, DType::Float32, Device::CUDA);
    dim3 block(16, 16);
    dim3 grid((int)((seq_len + 15) / 16), (int)((seq_len + 15) / 16));
    causal_mask_kernel<<<grid, block>>>(mask.data_ptr<float>(), seq_len);
    CUDA_CHECK(cudaGetLastError());
    return mask;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Mask-add kernel
//  scores[b, q, k] += mask[q, k]  (broadcast over batch)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void add_mask_broadcast_kernel(float* scores, const float* mask,
                                           int64_t batch, int64_t seq_q,
                                           int64_t seq_k) {
    int64_t b = blockIdx.z;
    int64_t q = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch && q < seq_q && k < seq_k)
        scores[(b * seq_q + q) * seq_k + k] += mask[q * seq_k + k];
}

// ─────────────────────────────────────────────────────────────────────────────
//  Scale kernel: x *= scale  (element-wise)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void scale_kernel(float* x, int64_t n, float scale) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Transpose [batch, seq, n_heads, head_dim] ↔ [batch, n_heads, seq, head_dim]
// ─────────────────────────────────────────────────────────────────────────────
__global__ void transpose_bsnh_bnsh(const float* __restrict__ src,
                                     float*       __restrict__ dst,
                                     int64_t batch, int64_t s,
                                     int64_t n_heads, int64_t head_dim) {
    // one thread copies one element of head vector
    int64_t b = blockIdx.z;
    int64_t h = blockIdx.y;
    int64_t p = blockIdx.x;
    int64_t d = threadIdx.x;
    if (d < head_dim) {
        // src layout: [b, p, h, d] → dst layout: [b, h, p, d]
        dst[((b * n_heads + h) * s + p) * head_dim + d] =
            src[((b * s + p) * n_heads + h) * head_dim + d];
    }
}

static Tensor bsnh_to_bnsh(const Tensor& t, int64_t s) {
    int64_t batch    = t.dim(0);
    int64_t n_heads  = t.dim(2);
    int64_t head_dim = t.dim(3);
    Tensor out({batch, n_heads, s, head_dim}, DType::Float32, Device::CUDA);
    dim3 grid((int)s, (int)n_heads, (int)batch);
    int threads = (int)min(head_dim, (int64_t)1024);
    transpose_bsnh_bnsh<<<grid, threads>>>(
        t.data_ptr<float>(), out.data_ptr<float>(),
        batch, s, n_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

__global__ void transpose_bnsh_bsnh(const float* __restrict__ src,
                                     float*       __restrict__ dst,
                                     int64_t batch, int64_t s,
                                     int64_t n_heads, int64_t head_dim) {
    int64_t b = blockIdx.z;
    int64_t h = blockIdx.y;
    int64_t p = blockIdx.x;
    int64_t d = threadIdx.x;
    if (d < head_dim) {
        // src [b, h, p, d] → dst [b, p, h, d]
        dst[((b * s + p) * n_heads + h) * head_dim + d] =
            src[((b * n_heads + h) * s + p) * head_dim + d];
    }
}

static Tensor bnsh_to_bsnh(const Tensor& t, int64_t s) {
    int64_t batch    = t.dim(0);
    int64_t n_heads  = t.dim(1);
    int64_t head_dim = t.dim(3);
    Tensor out({batch, s, n_heads, head_dim}, DType::Float32, Device::CUDA);
    dim3 grid((int)s, (int)n_heads, (int)batch);
    int threads = (int)min(head_dim, (int64_t)1024);
    transpose_bnsh_bsnh<<<grid, threads>>>(
        t.data_ptr<float>(), out.data_ptr<float>(),
        batch, s, n_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  KV-cache 写入 kernel
//  dst[b, pos_offset : pos_offset+seq, :, :] = src[b, :, :, :]
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kvcache_write_kernel(const float* __restrict__ src,
                                      float*       __restrict__ dst,
                                      int64_t batch, int64_t seq,
                                      int64_t n_heads, int64_t head_dim,
                                      int64_t max_seq, int64_t pos_offset) {
    int64_t b = blockIdx.z;
    int64_t s = blockIdx.y;
    int64_t h = blockIdx.x;
    int64_t d = threadIdx.x;
    if (d < head_dim) {
        int64_t src_idx = ((b * seq + s) * n_heads + h) * head_dim + d;
        int64_t dst_idx = ((b * max_seq + (pos_offset + s)) * n_heads + h) * head_dim + d;
        dst[dst_idx] = src[src_idx];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  scaled_dot_product_attention
// ─────────────────────────────────────────────────────────────────────────────
Tensor scaled_dot_product_attention(const Tensor& Q, const Tensor& K,
                                     const Tensor& V,
                                     const Tensor* mask) {
    assert(Q.ndim() == 3 && K.ndim() == 3 && V.ndim() == 3);
    assert(Q.device() == Device::CUDA);
    int64_t batch    = Q.dim(0);
    int64_t seq_q    = Q.dim(1);
    int64_t head_dim = Q.dim(2);
    int64_t seq_k    = K.dim(1);

    float scale = 1.0f / sqrtf((float)head_dim);

    // scores = Q @ K^T  [batch, seq_q, seq_k]
    Tensor scores = bmm(Q, K, false, true);

    // scale
    int64_t n = scores.numel();
    int threads = 256;
    scale_kernel<<<(int)((n + threads - 1) / threads), threads>>>(
        scores.data_ptr<float>(), n, scale);
    CUDA_CHECK(cudaGetLastError());

    // optional additive mask (2D broadcast)
    if (mask && mask->numel() > 0) {
        assert(mask->ndim() == 2);
        dim3 blk(16, 16);
        dim3 grd((int)((seq_k + 15) / 16),
                 (int)((seq_q + 15) / 16),
                 (int)batch);
        add_mask_broadcast_kernel<<<grd, blk>>>(
            scores.data_ptr<float>(), mask->data_ptr<float>(),
            batch, seq_q, seq_k);
        CUDA_CHECK(cudaGetLastError());
    }

    softmax_(scores);

    // out = scores @ V  [batch, seq_q, head_dim]
    return bmm(scores, V);
}

// ─────────────────────────────────────────────────────────────────────────────
//  mha — Multi-Head Attention
// ─────────────────────────────────────────────────────────────────────────────
Tensor mha(const Tensor& x,
           const Tensor& w_q, const Tensor& b_q,
           const Tensor& w_k, const Tensor& b_k,
           const Tensor& w_v, const Tensor& b_v,
           const Tensor& w_o, const Tensor& b_o,
           int64_t n_heads,
           Tensor* kv_cache_k,
           Tensor* kv_cache_v,
           int64_t pos_offset) {
    assert(x.ndim() == 3 && x.device() == Device::CUDA);
    int64_t batch    = x.dim(0);
    int64_t seq_len  = x.dim(1);
    int64_t d_model  = x.dim(2);
    assert(d_model % n_heads == 0);
    int64_t head_dim = d_model / n_heads;

    // ── 1. 线性投影 ───────────────────────────────────────────────────────────
    Tensor Q = linear(x, w_q, b_q);   // [batch, seq, d_model]
    Tensor K = linear(x, w_k, b_k);
    Tensor V = linear(x, w_v, b_v);

    // ── 2. reshape → [batch, seq, n_heads, head_dim] ─────────────────────────
    Q = Q.view({batch, seq_len, n_heads, head_dim});
    K = K.view({batch, seq_len, n_heads, head_dim});
    V = V.view({batch, seq_len, n_heads, head_dim});

    // ── 3. RoPE ───────────────────────────────────────────────────────────────
    rope_(Q, K, pos_offset);

    // ── 4. KV-Cache 写入 ──────────────────────────────────────────────────────
    int64_t seq_k;
    Tensor K_full, V_full;
    if (kv_cache_k && kv_cache_v) {
        int64_t max_seq = kv_cache_k->dim(1);
        dim3 grd((int)n_heads, (int)seq_len, (int)batch);
        int  thr = (int)min(head_dim, (int64_t)1024);
        kvcache_write_kernel<<<grd, thr>>>(
            K.data_ptr<float>(), kv_cache_k->data_ptr<float>(),
            batch, seq_len, n_heads, head_dim, max_seq, pos_offset);
        CUDA_CHECK(cudaGetLastError());
        kvcache_write_kernel<<<grd, thr>>>(
            V.data_ptr<float>(), kv_cache_v->data_ptr<float>(),
            batch, seq_len, n_heads, head_dim, max_seq, pos_offset);
        CUDA_CHECK(cudaGetLastError());

        seq_k  = pos_offset + seq_len;
        K_full = kv_cache_k->view({batch, seq_k, n_heads, head_dim});
        V_full = kv_cache_v->view({batch, seq_k, n_heads, head_dim});
    } else {
        seq_k  = seq_len;
        K_full = K;
        V_full = V;
    }

    // ── 5. 转置 [batch,seq,n_heads,head_dim] → [batch,n_heads,seq,head_dim] ──
    Tensor Qh = bsnh_to_bnsh(Q,      seq_len)
                    .view({batch * n_heads, seq_len, head_dim});
    Tensor Kh = bsnh_to_bnsh(K_full, seq_k)
                    .view({batch * n_heads, seq_k,   head_dim});
    Tensor Vh = bsnh_to_bnsh(V_full, seq_k)
                    .view({batch * n_heads, seq_k,   head_dim});

    // ── 6. Causal mask（prefill 阶段）────────────────────────────────────────
    Tensor causal;
    const Tensor* pmask = nullptr;
    if (seq_len > 1 && pos_offset == 0) {
        causal = make_causal_mask_cuda(seq_len);
        pmask  = &causal;
    }

    // ── 7. SDPA ───────────────────────────────────────────────────────────────
    Tensor attn_out = scaled_dot_product_attention(Qh, Kh, Vh, pmask);
    // [batch*n_heads, seq_len, head_dim]

    // ── 8. 拼合 heads ─────────────────────────────────────────────────────────
    attn_out = attn_out.view({batch, n_heads, seq_len, head_dim});
    Tensor concat = bnsh_to_bsnh(attn_out, seq_len);   // [batch, seq, n_heads, head_dim]
    Tensor out_flat = concat.view({batch, seq_len, d_model});

    // ── 9. 输出投影 ───────────────────────────────────────────────────────────
    return linear(out_flat, w_o, b_o);
}

}  // namespace ops
}  // namespace tinyllama
