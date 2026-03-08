# Qwen3 推理 Ops 完整分析报告

## 一、已实现的 Ops 清单

### 1. 基础运算 (ops_basic.cu)
- ✅ `softmax()` / `softmax_()` - Softmax 归一化
- ✅ `rms_norm()` - RMS LayerNorm（Qwen3 使用）
- ✅ `layer_norm()` - 标准 LayerNorm（Qwen3 不使用）
- ✅ `matmul()` - 矩阵乘法
- ✅ `bmm()` - 批量矩阵乘法
- ✅ `linear()` - 线性投影

### 2. 位置编码 (ops_rope.cu)
- ✅ `rope_()` - Rotary Position Embedding（in-place）
- ✅ `rope_precompute_freqs()` - 预计算 RoPE 频率表

### 3. 注意力机制 (ops_attention.cu)
- ✅ `scaled_dot_product_attention()` - 单头注意力
- ✅ `mha()` - 多头注意力（支持 KV-cache）
  - 包含 RoPE
  - 包含 causal mask
  - 支持 KV-cache

### 4. 激活函数 (ops_activation.cu)
- ✅ `silu()` - SiLU 激活（Qwen3 使用）
- ✅ `mul()` - 逐元素乘法
- ✅ `add()` - 逐元素加法（刚实现）
- ✅ `gelu()` - GELU 激活（Qwen3 不使用）
- ✅ `embedding()` - Token embedding 查找

## 二、Qwen3 forward 实际使用的 Ops

### forward() 主函数：
1. ✅ `ops::embedding()` - Token embedding
2. ✅ `ops::rms_norm()` - Final LayerNorm
3. ✅ `ops::linear()` - LM head projection

### forward_layer() 每层：
1. ✅ `ops::rms_norm()` - Input LayerNorm (2次)
2. ✅ `ops::mha()` - Multi-Head Attention
3. ✅ `ops::linear()` - Q/K/V/O 投影 + FFN 投影 (7次)
4. ✅ `ops::silu()` - SwiGLU 激活
5. ✅ `ops::mul()` - SwiGLU 的 gate * up
6. ✅ `ops::add()` - 残差连接 (2次)

**总计每层使用：**
- rms_norm: 2次
- linear: 7次 (Q/K/V/O + gate/up/down)
- mha: 1次
- silu: 1次
- mul: 1次
- add: 2次

## 三、已实现但未使用的 Ops

1. ❌ `layer_norm()` - Qwen3 使用 RMS norm
2. ❌ `gelu()` - Qwen3 使用 SiLU
3. ❌ `rope_precompute_freqs()` - MHA 内部直接计算
4. ❌ `scaled_dot_product_attention()` - 被 MHA 封装
5. ❌ `softmax()` - MHA 内部使用 softmax_()
6. ❌ `matmul()` / `bmm()` - 被 linear/MHA 内部使用

**结论：这些 ops 都有用，只是被更高层的 ops 封装了。**

## 四、关键问题分析

### 问题 1: GQA (Grouped Query Attention) 支持 ⚠️

**Qwen3 配置：**
- num_attention_heads = 16 (Q heads)
- num_key_value_heads = 8 (KV heads)
- 需要 2:1 的 GQA

**当前 MHA 实现：**
```cpp
// ops_attention.cu line 217-218
assert(d_model % n_heads == 0);
int64_t head_dim = d_model / n_heads;
```

**问题：**
- MHA 假设 Q/K/V 的 head 数量相同
- 传入的 `n_heads` 参数只有一个
- K/V 权重维度：[1024, 1024] = [n_kv_heads * head_dim, hidden_size]
- Q 权重维度：[2048, 1024] = [n_heads * head_dim, hidden_size]

**当前调用：**
```cpp
// qwen3_model.cc line 220
Tensor attn_output = ops::mha(
    normed,
    wq, empty_bias,  // [2048, 1024]
    wk, empty_bias,  // [1024, 1024] ← 维度不匹配！
    wv, empty_bias,  // [1024, 1024]
    wo, empty_bias,
    cfg.num_attention_heads,  // 16
    ...
);
```

**解决方案：**
1. 修改 MHA 签名，添加 `n_kv_heads` 参数
2. 在 MHA 内部实现 K/V repeat
3. 或者在 forward_layer 中手动实现 GQA

### 问题 2: Q/K Normalization ⚠️

**Qwen3 特有：**
- 在 attention 之前对 Q 和 K 做 RMS norm
- 权重：`self_attn.q_norm.weight` [128] (head_dim)
- 权重：`self_attn.k_norm.weight` [128]

**当前实现：**
- 完全跳过了 Q/K norm
- 可能影响模型精度

**解决方案：**
1. 在 forward_layer 中手动实现（需要 reshape）
2. 或修改 MHA 添加 Q/K norm 支持

### 问题 3: Causal Mask ✅

**检查结果：**
```cpp
// ops_attention.cu line 37-51
__global__ void causal_mask_kernel(...)
static Tensor make_causal_mask_cuda(int64_t seq_len)
```

**结论：MHA 已正确实现 causal mask，无问题。**

## 五、推理所需但缺失的功能

### 1. GQA 支持（必须）
**状态：** ❌ 缺失
**优先级：** 🔴 最高
**影响：** 无法正确运行 Qwen3

### 2. Q/K Normalization（重要）
**状态：** ❌ 缺失
**优先级：** 🟡 中等
**影响：** 可能影响精度，但不影响运行

### 3. Tensor 操作（可选）
如果要手动实现 GQA，需要：
- `Tensor::reshape()` - 已有 view()，基本够用
- `Tensor::permute()` - 需要实现
- `Tensor::repeat_interleave()` - 需要实现

## 六、实现建议

### 方案 A：修改 MHA 支持 GQA（推荐）

**优点：**
- 一次修改，所有模型受益
- 代码更清晰

**缺点：**
- 需要修改核心 op

**步骤：**
1. 修改 MHA 签名：`mha(..., int64_t n_heads, int64_t n_kv_heads, ...)`
2. 在 MHA 内部实现 K/V repeat
3. 更新 forward_layer 调用

### 方案 B：手动实现 GQA Attention

**优点：**
- 不修改现有 ops
- 更灵活

**缺点：**
- 代码重复
- 需要实现更多 Tensor 操作

### 方案 C：简化版（快速验证）

**临时方案：**
1. 修改 Qwen3 权重，将 K/V 权重复制扩展到 16 heads
2. 跳过 Q/K norm
3. 先验证整体流程

## 七、总结

### ✅ 好消息：
1. **所有基础 ops 都已实现**
2. **Causal mask 正确实现**
3. **RoPE 已集成到 MHA**
4. **KV-cache 已支持**

### ⚠️ 需要解决：
1. **GQA 支持**（阻塞问题）
2. **Q/K normalization**（精度问题）

### 📊 实现完整度：
- 基础 ops: 100% ✅
- Attention 机制: 80% ⚠️ (缺 GQA)
- 特殊功能: 0% ❌ (缺 Q/K norm)

**建议：先实现 GQA 支持，再添加 Q/K norm。**
