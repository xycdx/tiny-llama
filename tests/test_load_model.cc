// tests/test_load_model.cc — 测试从 models/ 目录加载真实模型
//
// 编译并运行：
//   cd build && make test_load_model && ./tests/test_load_model

#include "tinyllama/loader.h"
#include "tinyllama/tensor.h"

#include <iostream>
#include <string>
#include <cstdio>

using namespace tinyllama;

int main() {
    std::string model_path = "../../models/Qwen3-0.6B/model.safetensors";

    std::cout << "=== Safetensors Model Loading Test ===\n\n";
    std::cout << "Loading model from: " << model_path << "\n";

    try {
        // 加载模型权重
        auto weights = load_safetensors(model_path);

        std::cout << "✓ Successfully loaded model\n";
        std::cout << "Total tensors: " << weights.size() << "\n\n";

        // 统计信息
        size_t total_params = 0;
        size_t total_bytes = 0;

        std::cout << "Tensor details:\n";
        std::cout << "----------------------------------------\n";

        int count = 0;
        for (const auto& [name, tensor] : weights) {
            size_t params = 1;
            for (int i = 0; i < tensor.ndim(); i++) {
                params *= tensor.dim(i);
            }
            total_params += params;

            size_t bytes = params * sizeof(float);  // 所有 tensor 都转换为 F32
            total_bytes += bytes;

            // 只打印前 10 个 tensor 的详细信息
            if (count < 10) {
                std::cout << count + 1 << ". " << name << "\n";
                std::cout << "   Shape: [";
                for (int i = 0; i < tensor.ndim(); i++) {
                    if (i > 0) std::cout << ", ";
                    std::cout << tensor.dim(i);
                }
                std::cout << "]\n";
                std::cout << "   Params: " << params << "\n";
                std::cout << "   Device: " << (tensor.device() == Device::CPU ? "CPU" : "CUDA") << "\n";
                std::cout << "   DType: " << (tensor.dtype() == DType::Float32 ? "Float32" : "Int32") << "\n\n";
            }
            count++;
        }

        if (weights.size() > 10) {
            std::cout << "... (" << (weights.size() - 10) << " more tensors)\n\n";
        }

        std::cout << "----------------------------------------\n";
        std::cout << "Summary:\n";
        std::cout << "  Total parameters: " << total_params << " ("
                  << (total_params / 1e6) << "M)\n";
        std::cout << "  Total memory (F32): " << (total_bytes / 1024.0 / 1024.0)
                  << " MB\n";

        // 验证一些关键层是否存在
        std::cout << "\nKey layers check:\n";
        const char* key_layers[] = {
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "lm_head.weight"
        };

        for (const char* layer : key_layers) {
            if (weights.count(layer)) {
                std::cout << "  ✓ " << layer << "\n";
            } else {
                std::cout << "  ✗ " << layer << " (missing)\n";
            }
        }

        std::cout << "\n[PASS] Model loading test completed successfully!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Error loading model: " << e.what() << "\n";
        return 1;
    }
}
