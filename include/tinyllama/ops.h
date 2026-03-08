#pragma once

#include "tensor.h"

// ─────────────────────────────────────────────────────────────────────────────
//  ops.h — public interface for all LLM inference operators
//
//  Naming convention:
//    All ops are free functions in the `tinyllama::ops` namespace.
//    Input tensors are passed by const reference, outputs are returned.
//    In-place variants are suffixed with `_`.
// ─────────────────────────────────────────────────────────────────────────────

namespace tinyllama {
namespace ops {

// ── Element-wise / reduction ─────────────────────────────────────────────────

// Softmax over the last dimension: out[..., :] = softmax(x[..., :])
Tensor softmax(const Tensor& x);

// In-place softmax over the last dimension
void softmax_(Tensor& x);

// ── Normalization ────────────────────────────────────────────────────────────

// RMS LayerNorm:  out = x / rms(x) * weight
// x     : [*, hidden]
// weight: [hidden]
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps = 1e-5f);

// Standard LayerNorm: out = (x - mean) / std * weight + bias
// x     : [*, hidden]
// weight: [hidden]
// bias  : [hidden]
Tensor layer_norm(const Tensor& x, const Tensor& weight,
                  const Tensor& bias, float eps = 1e-5f);

// ── Positional Encoding ──────────────────────────────────────────────────────

// Rotary Position Embedding (RoPE) — applied in-place.
// q, k  : [batch, seq_len, n_heads, head_dim]  (or [seq_len, n_heads, head_dim])
// freqs : precomputed [seq_len, head_dim/2] cos/sin table
// pos_offset: starting position (used for KV-cache decoding)
void rope_(Tensor& q, Tensor& k, int64_t pos_offset = 0,
           float theta = 10000.0f);

// Precompute RoPE frequency table → [max_seq_len, head_dim/2, 2]  (cos, sin)
Tensor rope_precompute_freqs(int64_t max_seq_len, int64_t head_dim,
                             float theta = 10000.0f);

// ── Linear / MatMul ──────────────────────────────────────────────────────────

// General matrix multiplication: out = alpha * A @ B + beta * C
// A: [M, K], B: [K, N], C (optional): [M, N]
Tensor matmul(const Tensor& A, const Tensor& B,
              float alpha = 1.0f, float beta = 0.0f,
              bool trans_A = false, bool trans_B = false);

// Batched matmul: A [batch, M, K] @ B [batch, K, N] → [batch, M, N]
Tensor bmm(const Tensor& A, const Tensor& B,
           bool trans_A = false, bool trans_B = false);

// Linear projection: out = x @ weight^T + bias
// x     : [*, in_features]
// weight: [out_features, in_features]
// bias  : [out_features]  (optional, pass empty tensor to skip)
Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias);

// ── Attention ────────────────────────────────────────────────────────────────

// Scaled dot-product attention (single head).
//   attn_out = softmax(Q @ K^T / sqrt(d_k)) @ V
// Q: [batch, seq_q, head_dim]
// K: [batch, seq_k, head_dim]
// V: [batch, seq_k, head_dim]
// mask: optional additive mask [seq_q, seq_k] or [batch, 1, seq_q, seq_k]
//       (typically a causal mask filled with -inf / 0)
Tensor scaled_dot_product_attention(const Tensor& Q, const Tensor& K,
                                     const Tensor& V,
                                     const Tensor* mask = nullptr);

// Multi-Head Attention (MHA) with GQA support.
//   Projects Q/K/V, runs n_heads independent attention heads, projects out.
// x           : [batch, seq_len, d_model]
// w_q/k/v/o   : projection weights
//               Q: [n_heads * head_dim, d_model]
//               K/V: [n_kv_heads * head_dim, d_model]  (GQA)
//               O: [d_model, n_heads * head_dim]
// b_q/k/v/o   : [d_model]            projection biases (pass empty to skip)
// n_heads     : number of query heads
// n_kv_heads  : number of key/value heads (for GQA, typically n_heads / 2 or n_heads / 4)
//               if n_kv_heads == 0, defaults to n_heads (standard MHA)
// kv_cache_k/v: optional KV-cache tensors, updated in-place
// pos_offset  : current decoding position (0 during prefill)
Tensor mha(const Tensor& x,
           const Tensor& w_q, const Tensor& b_q,
           const Tensor& w_k, const Tensor& b_k,
           const Tensor& w_v, const Tensor& b_v,
           const Tensor& w_o, const Tensor& b_o,
           int64_t n_heads,
           int64_t n_kv_heads = 0,
           Tensor* kv_cache_k = nullptr,
           Tensor* kv_cache_v = nullptr,
           int64_t pos_offset = 0);

// ── Activation functions ─────────────────────────────────────────────────────

// SiLU: x * sigmoid(x)  (used in LLaMA FFN gate)
Tensor silu(const Tensor& x);

// Element-wise multiply (used in SwiGLU: silu(gate) * up)
Tensor mul(const Tensor& a, const Tensor& b);

// Element-wise add (used for residual connections)
Tensor add(const Tensor& a, const Tensor& b);

// GELU approximation
Tensor gelu(const Tensor& x);

// ── Embedding ────────────────────────────────────────────────────────────────

// Token embedding lookup: ids [batch, seq] → [batch, seq, d_model]
// table: [vocab_size, d_model]
Tensor embedding(const Tensor& table, const Tensor& ids);

}  // namespace ops
}  // namespace tinyllama
