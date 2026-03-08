// model/qwen3_model.cc — Qwen3 模型权重存储实现

#include "tinyllama/qwen3_model.h"
#include "tinyllama/loader.h"
#include "tinyllama/ops.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

// 使用 nlohmann/json 解析配置文件
#include "json.hpp"

namespace tinyllama {

// ── Qwen3Config ──────────────────────────────────────────────────────────────

Qwen3Config Qwen3Config::from_json(const std::string& config_path) {
    std::ifstream f(config_path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open config file: " + config_path);
    }

    nlohmann::json j;
    f >> j;

    Qwen3Config cfg;
    cfg.vocab_size = j.value("vocab_size", 151936);
    cfg.hidden_size = j.value("hidden_size", 1024);
    cfg.intermediate_size = j.value("intermediate_size", 3072);
    cfg.num_hidden_layers = j.value("num_hidden_layers", 28);
    cfg.num_attention_heads = j.value("num_attention_heads", 16);
    cfg.num_key_value_heads = j.value("num_key_value_heads", 8);
    cfg.head_dim = j.value("head_dim", 128);
    cfg.max_position_embeddings = j.value("max_position_embeddings", 40960);
    cfg.rms_norm_eps = j.value("rms_norm_eps", 1e-6);
    cfg.rope_theta = j.value("rope_theta", 1000000.0);
    cfg.hidden_act = j.value("hidden_act", "silu");

    return cfg;
}

// ── Qwen3Model ───────────────────────────────────────────────────────────────

Qwen3Model::Qwen3Model(const std::string& model_path) {
    // 加载权重
    weights_ = load_safetensors(model_path);

    // 从权重推断配置
    infer_config_from_weights();
}

Qwen3Model::Qwen3Model(const std::string& model_path,
                       const std::string& config_path) {
    // 加载配置
    config_ = Qwen3Config::from_json(config_path);

    // 加载权重
    weights_ = load_safetensors(model_path);
}

void Qwen3Model::infer_config_from_weights() {
    // 从 embedding 推断 vocab_size 和 hidden_size
    if (has_weight("model.embed_tokens.weight")) {
        const auto& emb = get_embedding();
        if (emb.ndim() == 2) {
            config_.vocab_size = emb.dim(0);
            config_.hidden_size = emb.dim(1);
        }
    }

    // 从第一层的权重推断其他配置
    if (has_weight("model.layers.0.mlp.gate_proj.weight")) {
        const auto& gate = get_layer_weight(0, "mlp.gate_proj.weight");
        if (gate.ndim() == 2) {
            config_.intermediate_size = gate.dim(0);
        }
    }

    if (has_weight("model.layers.0.self_attn.q_proj.weight")) {
        const auto& q = get_layer_weight(0, "self_attn.q_proj.weight");
        if (q.ndim() == 2) {
            // q_proj: [num_heads * head_dim, hidden_size]
            int64_t q_dim = q.dim(0);
            config_.num_attention_heads = q_dim / config_.hidden_size;
        }
    }

    if (has_weight("model.layers.0.self_attn.k_proj.weight")) {
        const auto& k = get_layer_weight(0, "self_attn.k_proj.weight");
        if (k.ndim() == 2) {
            // k_proj: [num_kv_heads * head_dim, hidden_size]
            int64_t k_dim = k.dim(0);
            config_.num_key_value_heads = k_dim / config_.hidden_size;
        }
    }

    // 推断 head_dim
    if (config_.num_attention_heads > 0) {
        if (has_weight("model.layers.0.self_attn.q_proj.weight")) {
            const auto& q = get_layer_weight(0, "self_attn.q_proj.weight");
            config_.head_dim = q.dim(0) / config_.num_attention_heads;
        }
    }

    // 统计层数
    config_.num_hidden_layers = 0;
    for (int i = 0; i < 100; ++i) {  // 最多检查 100 层
        std::string key = "model.layers." + std::to_string(i) +
                         ".self_attn.q_proj.weight";
        if (has_weight(key)) {
            config_.num_hidden_layers = i + 1;
        } else {
            break;
        }
    }

    // 设置默认值
    config_.max_position_embeddings = 40960;
    config_.rms_norm_eps = 1e-6;
    config_.rope_theta = 1000000.0;
    config_.hidden_act = "silu";
}

Tensor& Qwen3Model::get_layer_weight(int layer_idx, const std::string& name) {
    std::string full_name = "model.layers." + std::to_string(layer_idx) +
                           "." + name;
    return get_weight(full_name);
}

const Tensor& Qwen3Model::get_layer_weight(int layer_idx,
                                           const std::string& name) const {
    std::string full_name = "model.layers." + std::to_string(layer_idx) +
                           "." + name;
    return get_weight(full_name);
}

Tensor& Qwen3Model::get_weight(const std::string& name) {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return it->second;
}

const Tensor& Qwen3Model::get_weight(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return it->second;
}

std::vector<std::string> Qwen3Model::weight_names() const {
    std::vector<std::string> names;
    names.reserve(weights_.size());
    for (const auto& [name, _] : weights_) {
        names.push_back(name);
    }
    return names;
}

size_t Qwen3Model::num_parameters() const {
    size_t total = 0;
    for (const auto& [_, tensor] : weights_) {
        total += static_cast<size_t>(tensor.numel());
    }
    return total;
}

size_t Qwen3Model::memory_bytes() const {
    size_t total = 0;
    for (const auto& [_, tensor] : weights_) {
        total += static_cast<size_t>(tensor.numel()) *
                 dtype_size(tensor.dtype());
    }
    return total;
}

void Qwen3Model::to_cuda() {
    for (auto& [_, tensor] : weights_) {
        if (tensor.device() == Device::CPU) {
            tensor = tensor.cuda();
        }
    }
}

void Qwen3Model::to_cpu() {
    for (auto& [_, tensor] : weights_) {
        if (tensor.device() == Device::CUDA) {
            tensor = tensor.cpu();
        }
    }
}

// ── 推理功能 ─────────────────────────────────────────────────────────────────

Tensor Qwen3Model::forward_layer(const Tensor& x, int layer_idx) {
    // Qwen3 Transformer Layer (简化版本)
    // 1. Input LayerNorm + Self-Attention + Residual
    // 2. Post-Attention LayerNorm + FFN + Residual

    const auto& cfg = config_;

    // ── Self-Attention ──
    // Input LayerNorm
    const auto& ln1_weight = get_layer_weight(layer_idx, "input_layernorm.weight");
    Tensor normed = ops::rms_norm(x, ln1_weight, cfg.rms_norm_eps);

    // Q, K, V projections
    const auto& wq = get_layer_weight(layer_idx, "self_attn.q_proj.weight");
    const auto& wk = get_layer_weight(layer_idx, "self_attn.k_proj.weight");
    const auto& wv = get_layer_weight(layer_idx, "self_attn.v_proj.weight");
    const auto& wo = get_layer_weight(layer_idx, "self_attn.o_proj.weight");

    Tensor empty_bias;  // Qwen3 没有 bias

    // 使用 MHA op（支持 GQA）
    Tensor attn_output = ops::mha(
        normed,
        wq, empty_bias,
        wk, empty_bias,
        wv, empty_bias,
        wo, empty_bias,
        cfg.num_attention_heads,  // n_heads = 16
        cfg.num_key_value_heads   // n_kv_heads = 8 (GQA)
        // kv_cache, pos_offset 使用默认值
    );

    // Residual connection
    Tensor hidden = ops::add(x, attn_output);

    // ── FFN (SwiGLU) ──
    // Post-attention LayerNorm
    const auto& ln2_weight = get_layer_weight(layer_idx, "post_attention_layernorm.weight");
    normed = ops::rms_norm(hidden, ln2_weight, cfg.rms_norm_eps);

    // SwiGLU: gate_proj, up_proj, down_proj
    const auto& w_gate = get_layer_weight(layer_idx, "mlp.gate_proj.weight");
    const auto& w_up = get_layer_weight(layer_idx, "mlp.up_proj.weight");
    const auto& w_down = get_layer_weight(layer_idx, "mlp.down_proj.weight");

    Tensor gate = ops::linear(normed, w_gate, empty_bias);
    Tensor up = ops::linear(normed, w_up, empty_bias);

    // SwiGLU: silu(gate) * up
    Tensor activated = ops::mul(ops::silu(gate), up);

    // Down projection
    Tensor ffn_output = ops::linear(activated, w_down, empty_bias);

    // Residual connection
    hidden = ops::add(hidden, ffn_output);

    return hidden;
}

Tensor Qwen3Model::forward(const Tensor& input_ids) {
    // input_ids: [batch, seq_len] Int32

    // 1. Token embedding
    const auto& emb_table = get_embedding();
    Tensor hidden = ops::embedding(emb_table, input_ids);  // [batch, seq, hidden_size]

    // 2. Transformer layers
    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        hidden = forward_layer(hidden, i);
    }

    // 3. Final LayerNorm
    const auto& norm_weight = get_norm();
    hidden = ops::rms_norm(hidden, norm_weight, config_.rms_norm_eps);

    // 4. LM head (output projection)
    const auto& lm_head_weight = get_lm_head();
    Tensor empty_bias;
    Tensor logits = ops::linear(hidden, lm_head_weight, empty_bias);  // [batch, seq, vocab_size]

    return logits;
}

}  // namespace tinyllama
