// tests/test_qwen3_simple.cc — 简单测试 Qwen3 单层

#include "tinyllama/qwen3_model.h"
#include "tinyllama/ops.h"
#include <iostream>

using namespace tinyllama;

int main() {
    std::cout << "=== Qwen3 Simple Test ===\n\n";

    try {
        // 加载模型
        std::string model_path = "../../models/Qwen3-0.6B/model.safetensors";
        std::string config_path = "../../models/Qwen3-0.6B/config.json";

        std::cout << "Loading model...\n";
        Qwen3Model model(model_path, config_path);
        model.to_cuda();
        std::cout << "✓ Model loaded on CUDA\n\n";

        const auto& cfg = model.config();
        std::cout << "Config: " << cfg.num_attention_heads << " Q heads, "
                  << cfg.num_key_value_heads << " KV heads\n\n";

        // 测试 embedding
        std::cout << "Testing embedding...\n";
        Tensor input_ids({1, 4}, DType::Int32, Device::CPU);
        int32_t* ids = input_ids.data_ptr<int32_t>();
        ids[0] = 1; ids[1] = 2; ids[2] = 3; ids[3] = 4;

        Tensor input_cuda = input_ids.cuda();
        const auto& emb_table = model.get_embedding();
        Tensor hidden = ops::embedding(emb_table, input_cuda);

        std::cout << "  Embedding output shape: [" << hidden.dim(0) << ", "
                  << hidden.dim(1) << ", " << hidden.dim(2) << "]\n";

        // 检查 embedding 输出
        Tensor hidden_cpu = hidden.cpu();
        const float* h_ptr = hidden_cpu.data_ptr<float>();
        bool emb_ok = true;
        for (int i = 0; i < 10; ++i) {
            if (h_ptr[i] != h_ptr[i]) {  // NaN check
                emb_ok = false;
                break;
            }
        }
        std::cout << "  Embedding values: " << (emb_ok ? "OK" : "NaN") << "\n\n";

        if (!emb_ok) {
            std::cerr << "[FAIL] Embedding produced NaN\n";
            return 1;
        }

        // 测试 RMS norm
        std::cout << "Testing RMS norm...\n";
        const auto& ln_weight = model.get_layer_weight(0, "input_layernorm.weight");
        Tensor normed = ops::rms_norm(hidden, ln_weight, cfg.rms_norm_eps);

        Tensor normed_cpu = normed.cpu();
        const float* n_ptr = normed_cpu.data_ptr<float>();
        bool norm_ok = true;
        for (int i = 0; i < 10; ++i) {
            if (n_ptr[i] != n_ptr[i]) {
                norm_ok = false;
                break;
            }
        }
        std::cout << "  RMS norm values: " << (norm_ok ? "OK" : "NaN") << "\n\n";

        if (!norm_ok) {
            std::cerr << "[FAIL] RMS norm produced NaN\n";
            return 1;
        }

        // 测试 linear projection
        std::cout << "Testing Q projection...\n";
        const auto& wq = model.get_layer_weight(0, "self_attn.q_proj.weight");
        std::cout << "  Q weight shape: [" << wq.dim(0) << ", " << wq.dim(1) << "]\n";

        Tensor empty_bias;
        Tensor q = ops::linear(normed, wq, empty_bias);
        std::cout << "  Q output shape: [" << q.dim(0) << ", " << q.dim(1) << ", " << q.dim(2) << "]\n";

        Tensor q_cpu = q.cpu();
        const float* q_ptr = q_cpu.data_ptr<float>();
        bool q_ok = true;
        for (int i = 0; i < 10; ++i) {
            if (q_ptr[i] != q_ptr[i]) {
                q_ok = false;
                break;
            }
        }
        std::cout << "  Q values: " << (q_ok ? "OK" : "NaN") << "\n\n";

        if (!q_ok) {
            std::cerr << "[FAIL] Q projection produced NaN\n";
            return 1;
        }

        std::cout << "[PASS] All basic ops working!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Error: " << e.what() << "\n";
        return 1;
    }
}
