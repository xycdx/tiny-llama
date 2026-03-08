#pragma once

#include "tensor.h"
#include <string>
#include <unordered_map>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  qwen3_model.h — Qwen3 模型权重存储
//
//  Usage:
//    Qwen3Model model("models/Qwen3-0.6B/model.safetensors");
//    Tensor emb = model.get_embedding();
//    Tensor q_weight = model.get_layer_weight(0, "self_attn.q_proj.weight");
//
//  模型结构基于 Qwen3ForCausalLM:
//    - model.embed_tokens.weight
//    - model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
//    - model.layers.{i}.self_attn.{q,k}_norm.weight
//    - model.layers.{i}.mlp.{gate,up,down}_proj.weight
//    - model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight
//    - model.norm.weight
//    - lm_head.weight
// ─────────────────────────────────────────────────────────────────────────────

namespace tinyllama {

// Qwen3 模型配置
struct Qwen3Config {
    int64_t vocab_size;              // 词表大小
    int64_t hidden_size;             // 隐藏层维度
    int64_t intermediate_size;       // FFN 中间层维度
    int64_t num_hidden_layers;       // Transformer 层数
    int64_t num_attention_heads;     // 注意力头数
    int64_t num_key_value_heads;     // KV 头数（GQA）
    int64_t head_dim;                // 每个注意力头的维度
    int64_t max_position_embeddings; // 最大位置编码
    double rms_norm_eps;             // RMS Norm epsilon
    double rope_theta;               // RoPE theta
    std::string hidden_act;          // 激活函数（silu/gelu）

    // 从 config.json 加载配置
    static Qwen3Config from_json(const std::string& config_path);
};

// Qwen3 模型权重
class Qwen3Model {
public:
    // 从 safetensors 文件加载模型
    explicit Qwen3Model(const std::string& model_path);

    // 从 safetensors 文件和配置文件加载模型
    Qwen3Model(const std::string& model_path, const std::string& config_path);

    // 获取模型配置
    const Qwen3Config& config() const { return config_; }

    // ── Embedding ──
    Tensor& get_embedding() { return get_weight("model.embed_tokens.weight"); }
    const Tensor& get_embedding() const { return get_weight("model.embed_tokens.weight"); }

    // ── Layer weights ──
    // 获取指定层的权重，例如：
    //   get_layer_weight(0, "self_attn.q_proj.weight")
    //   get_layer_weight(5, "mlp.gate_proj.weight")
    Tensor& get_layer_weight(int layer_idx, const std::string& name);
    const Tensor& get_layer_weight(int layer_idx, const std::string& name) const;

    // ── Output ──
    Tensor& get_norm() { return get_weight("model.norm.weight"); }
    const Tensor& get_norm() const { return get_weight("model.norm.weight"); }

    Tensor& get_lm_head() { return get_weight("lm_head.weight"); }
    const Tensor& get_lm_head() const { return get_weight("lm_head.weight"); }

    // ── 直接访问权重字典 ──
    Tensor& get_weight(const std::string& name);
    const Tensor& get_weight(const std::string& name) const;

    bool has_weight(const std::string& name) const {
        return weights_.count(name) > 0;
    }

    // 获取所有权重名称
    std::vector<std::string> weight_names() const;

    // 统计信息
    size_t num_parameters() const;
    size_t memory_bytes() const;

    // 将所有权重移动到 CUDA
    void to_cuda();

    // 将所有权重移动到 CPU
    void to_cpu();

    // ── 推理功能 ──

    // 单次前向传播（不带 KV-cache）
    // input_ids: [batch, seq_len] 输入 token IDs (Int32)
    // 返回: [batch, seq_len, vocab_size] logits (Float32)
    Tensor forward(const Tensor& input_ids);

private:
    Qwen3Config config_;
    std::unordered_map<std::string, Tensor> weights_;

    // 从 config.json 推断配置
    void infer_config_from_weights();

    // Transformer 单层前向传播
    Tensor forward_layer(const Tensor& x, int layer_idx);
};

}  // namespace tinyllama
