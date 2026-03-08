// tests/test_qwen3_model.cc — 测试 Qwen3Model 类

#include "tinyllama/qwen3_model.h"
#include <iostream>
#include <iomanip>

using namespace tinyllama;

int main() {
    std::cout << "=== Qwen3Model Test ===\n\n";

    try {
        // 加载模型（带配置文件）
        std::string model_path = "../../models/Qwen3-0.6B/model.safetensors";
        std::string config_path = "../../models/Qwen3-0.6B/config.json";

        std::cout << "Loading model...\n";
        Qwen3Model model(model_path, config_path);
        std::cout << "✓ Model loaded successfully\n\n";

        // 打印配置
        const auto& cfg = model.config();
        std::cout << "Model Configuration:\n";
        std::cout << "  vocab_size: " << cfg.vocab_size << "\n";
        std::cout << "  hidden_size: " << cfg.hidden_size << "\n";
        std::cout << "  intermediate_size: " << cfg.intermediate_size << "\n";
        std::cout << "  num_hidden_layers: " << cfg.num_hidden_layers << "\n";
        std::cout << "  num_attention_heads: " << cfg.num_attention_heads << "\n";
        std::cout << "  num_key_value_heads: " << cfg.num_key_value_heads << "\n";
        std::cout << "  head_dim: " << cfg.head_dim << "\n";
        std::cout << "  max_position_embeddings: " << cfg.max_position_embeddings << "\n";
        std::cout << "  rms_norm_eps: " << cfg.rms_norm_eps << "\n";
        std::cout << "  rope_theta: " << cfg.rope_theta << "\n";
        std::cout << "  hidden_act: " << cfg.hidden_act << "\n\n";

        // 统计信息
        std::cout << "Model Statistics:\n";
        std::cout << "  Total parameters: " << model.num_parameters()
                  << " (" << (model.num_parameters() / 1e6) << "M)\n";
        std::cout << "  Memory usage: "
                  << std::fixed << std::setprecision(2)
                  << (model.memory_bytes() / 1024.0 / 1024.0) << " MB\n\n";

        // 测试权重访问
        std::cout << "Testing weight access:\n";

        // Embedding
        const auto& emb = model.get_embedding();
        std::cout << "  ✓ Embedding: [" << emb.dim(0) << ", " << emb.dim(1) << "]\n";

        // Layer 0 weights
        const auto& q0 = model.get_layer_weight(0, "self_attn.q_proj.weight");
        std::cout << "  ✓ Layer 0 Q proj: [" << q0.dim(0) << ", " << q0.dim(1) << "]\n";

        const auto& k0 = model.get_layer_weight(0, "self_attn.k_proj.weight");
        std::cout << "  ✓ Layer 0 K proj: [" << k0.dim(0) << ", " << k0.dim(1) << "]\n";

        const auto& v0 = model.get_layer_weight(0, "self_attn.v_proj.weight");
        std::cout << "  ✓ Layer 0 V proj: [" << v0.dim(0) << ", " << v0.dim(1) << "]\n";

        const auto& o0 = model.get_layer_weight(0, "self_attn.o_proj.weight");
        std::cout << "  ✓ Layer 0 O proj: [" << o0.dim(0) << ", " << o0.dim(1) << "]\n";

        const auto& gate0 = model.get_layer_weight(0, "mlp.gate_proj.weight");
        std::cout << "  ✓ Layer 0 Gate proj: [" << gate0.dim(0) << ", " << gate0.dim(1) << "]\n";

        const auto& up0 = model.get_layer_weight(0, "mlp.up_proj.weight");
        std::cout << "  ✓ Layer 0 Up proj: [" << up0.dim(0) << ", " << up0.dim(1) << "]\n";

        const auto& down0 = model.get_layer_weight(0, "mlp.down_proj.weight");
        std::cout << "  ✓ Layer 0 Down proj: [" << down0.dim(0) << ", " << down0.dim(1) << "]\n";

        // Norm weights
        const auto& norm = model.get_norm();
        std::cout << "  ✓ Final norm: [" << norm.dim(0) << "]\n";

        const auto& lm_head = model.get_lm_head();
        std::cout << "  ✓ LM head: [" << lm_head.dim(0) << ", " << lm_head.dim(1) << "]\n\n";

        // 测试最后一层
        int last_layer = cfg.num_hidden_layers - 1;
        const auto& q_last = model.get_layer_weight(last_layer, "self_attn.q_proj.weight");
        std::cout << "  ✓ Layer " << last_layer << " Q proj: ["
                  << q_last.dim(0) << ", " << q_last.dim(1) << "]\n\n";

        // 验证权重数量
        auto names = model.weight_names();
        std::cout << "Total weight tensors: " << names.size() << "\n";

        // 检查关键权重是否存在
        std::cout << "\nKey weights check:\n";
        std::vector<std::string> key_weights = {
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight"
        };

        for (const auto& name : key_weights) {
            if (model.has_weight(name)) {
                std::cout << "  ✓ " << name << "\n";
            } else {
                std::cout << "  ✗ " << name << " (missing)\n";
            }
        }

        std::cout << "\n[PASS] Qwen3Model test completed successfully!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Error: " << e.what() << "\n";
        return 1;
    }
}
