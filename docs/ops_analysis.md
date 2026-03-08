# Qwen3 推理 Ops 分析

## 当前实现中使用的 Ops

### forward() 函数使用的 ops：
1. ✅ `ops::embedding()` - Token embedding 查找
2. ✅ `ops::rms_norm()` - RMS LayerNorm
3. ✅ `ops::linear()` - 线性投影
4. ✅ `ops::add()` - 残差连接（刚实现）

### forward_layer() 函数使用的 ops：
1. ✅ `ops::rms_norm()` - Input LayerNorm 和 Post-Attention LayerNorm
2. ✅ `ops::mha()` - Multi-Head Attention（简化版，未处理 GQA）
3. ✅ `ops::linear()` - Q/K/V/O 投影和 FFN 投影
4. ✅ `ops::silu()` - SwiGLU 激活函数
5. ✅ `ops::mul()` - SwiGLU 的 element-wise multiply
6. ✅ `ops::add()` - 残差连接

## 已实现但当前未使用的 Ops

1. ❌ `ops::softmax()` / `ops::softmax_()` - 未直接使用（MHA 内部使用）
2. ❌ `ops::layer_norm()` - Qwen3 使用 RMS norm，不需要标准 LayerNorm
3. ❌ `ops::rope_()` - 未使用（MHA 内部应该使用，但可能有问题）
4. ❌ `ops::rope_precompute_freqs()` - 未使用
5. ❌ `ops::matmul()` - 未直接使用（linear 内部使用）
6. ❌ `ops::bmm()` - 未直接使用（MHA 内部使用）
7. ❌ `ops::scaled_dot_product_attention()` - 未直接使用（MHA 内部使用）
8. ❌ `ops::gelu()` - Qwen3 使用 SiLU，不需要 GELU

## 推理所需但未实现的 Ops

### 关键缺失功能：

1. **GQA (Grouped Query Attention) 支持**
   - 当前 MHA 不支持 GQA（num_key_value_heads != num_attention_heads）
   - Qwen3-0.6B: 16 个 Q heads, 8 个 KV heads
   - 需要：K/V repeat 或专门的 GQA 实现

2. **Q/K Normalization**
   - Qwen3 特有：在 attention 之前对 Q 和 K 做 RMS norm
   - 权重：`self_attn.q_norm.weight` 和 `self_attn.k_norm.weight`
   - 需要：在 head_dim 维度上做 RMS norm

3. **Causal Mask**
   - 当前 MHA 可能没有正确应用 causal mask
   - Prefill 阶段需要 causal mask 防止看到未来 token

4. **Tensor 操作**（用于手动实现 GQA）：
   - `Tensor::reshape()` - 已有 view()，但可能需要 reshape
   - `Tensor::permute()` - 转置多个维度
   - `Tensor::repeat_interleave()` - 用于 GQA 的 K/V 扩展

## 建议的实现优先级

### 高优先级（必须）：
1. ✅ `ops::add()` - 已实现
2. 🔧 修复 MHA 以支持 GQA
3. 🔧 实现 Q/K norm（或在 forward_layer 中手动处理）
4. 🔧 确保 causal mask 正确应用

### 中优先级（优化）：
1. 实现 Tensor::reshape/permute/repeat_interleave
2. 手动实现 GQA attention（不依赖 MHA）
3. 添加 KV-cache 支持

### 低优先级（可选）：
1. 移除不需要的 ops（layer_norm, gelu）
2. 性能优化

## 当前实现的问题

1. **MHA 不支持 GQA**：
   - Qwen3 使用 16 个 Q heads 和 8 个 KV heads
   - 当前 MHA 假设 num_heads 相同
   - 解决方案：修改 MHA 或手动实现 attention

2. **缺少 Q/K norm**：
   - Qwen3 在 attention 前对 Q/K 做 norm
   - 当前实现跳过了这一步
   - 可能影响模型精度

3. **Causal mask 不确定**：
   - 需要验证 MHA 是否正确应用 causal mask
   - Prefill 阶段必须有 causal mask

## 总结

**推理所需的核心 ops 都已实现**，但有以下关键问题需要解决：
- GQA 支持（最重要）
- Q/K normalization
- Causal mask 验证

建议先实现一个简化版本（忽略 Q/K norm），然后逐步完善。
