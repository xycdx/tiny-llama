// examples/qwen3_model_example.cc
// 演示如何使用 Qwen3Model 类

#include "tinyllama/qwen3_model.h"
#include <iostream>

using namespace tinyllama;

int main() {
    // 加载 Qwen3 模型
    std::cout << "Loading Qwen3-0.6B model...\n";
    Qwen3Model model("../models/Qwen3-0.6B/model.safetensors",
                     "../models/Qwen3-0.6B/config.json");

    std::cout << "✓ Model loaded\n\n";

    // 打印模型配置
    const auto& cfg = model.config();
    std::cout << "Model config:\n";
    std::cout << "  Vocab size: " << cfg.vocab_size << "\n";
    std::cout << "  Hidden size: " << cfg.hidden_size << "\n";
    std::cout << "  Layers: " << cfg.num_hidden_layers << "\n";
    std::cout << "  Attention heads: " << cfg.num_attention_heads << "\n";
    std::cout << "  KV heads: " << cfg.num_key_value_heads << "\n\n";

    // 访问权重
    std::cout << "Accessing weights:\n";

    auto& embedding = model.get_embedding();
    std::cout << "  Embedding: [" << embedding.dim(0) << ", "
              << embedding.dim(1) << "]\n";

    auto& q_weight = model.get_layer_weight(0, "self_attn.q_proj.weight");
    std::cout << "  Layer 0 Q: [" << q_weight.dim(0) << ", "
              << q_weight.dim(1) << "]\n";

    auto& lm_head = model.get_lm_head();
    std::cout << "  LM head: [" << lm_head.dim(0) << ", "
              << lm_head.dim(1) << "]\n\n";

    // 统计信息
    std::cout << "Model statistics:\n";
    std::cout << "  Parameters: " << (model.num_parameters() / 1e6) << "M\n";
    std::cout << "  Memory: " << (model.memory_bytes() / 1024.0 / 1024.0)
              << " MB\n";

    return 0;
}
