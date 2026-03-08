// tests/test_qwen3_cuda.cc — 测试 Qwen3Model 的 CUDA 功能

#include "tinyllama/qwen3_model.h"
#include <iostream>
#include <iomanip>
#include <unistd.h>

using namespace tinyllama;

int main() {
    std::cout << "=== Qwen3Model CUDA Transfer Test ===\n\n";

    try {
        // 加载模型（默认在 CPU）
        std::string model_path = "../../models/Qwen3-0.6B/model.safetensors";
        std::string config_path = "../../models/Qwen3-0.6B/config.json";

        std::cout << "Loading model to CPU...\n";
        Qwen3Model model(model_path, config_path);
        std::cout << "✓ Model loaded on CPU\n\n";

        // 验证初始设备
        std::cout << "Checking initial device (should be CPU):\n";
        const auto& emb_cpu = model.get_embedding();
        const auto& q0_cpu = model.get_layer_weight(0, "self_attn.q_proj.weight");
        const auto& lm_head_cpu = model.get_lm_head();

        std::cout << "  Embedding device: "
                  << (emb_cpu.device() == Device::CPU ? "CPU" : "CUDA") << "\n";
        std::cout << "  Layer 0 Q device: "
                  << (q0_cpu.device() == Device::CPU ? "CPU" : "CUDA") << "\n";
        std::cout << "  LM head device: "
                  << (lm_head_cpu.device() == Device::CPU ? "CPU" : "CUDA") << "\n\n";

        // 测试 to_cuda()
        std::cout << "Transferring model to CUDA...\n";
        model.to_cuda();
        std::cout << "✓ Transfer complete\n\n";
        sleep(20);

        // 验证转移后的设备
        std::cout << "Checking device after to_cuda() (should be CUDA):\n";
        const auto& emb_cuda = model.get_embedding();
        const auto& q0_cuda = model.get_layer_weight(0, "self_attn.q_proj.weight");
        const auto& lm_head_cuda = model.get_lm_head();

        std::cout << "  Embedding device: "
                  << (emb_cuda.device() == Device::CPU ? "CPU" : "CUDA") << "\n";
        std::cout << "  Layer 0 Q device: "
                  << (q0_cuda.device() == Device::CPU ? "CPU" : "CUDA") << "\n";
        std::cout << "  LM head device: "
                  << (lm_head_cuda.device() == Device::CPU ? "CPU" : "CUDA") << "\n\n";

        // 验证所有权重都在 CUDA 上
        std::cout << "Verifying all weights are on CUDA:\n";
        auto names = model.weight_names();
        int cuda_count = 0;
        int cpu_count = 0;

        for (const auto& name : names) {
            const auto& tensor = model.get_weight(name);
            if (tensor.device() == Device::CUDA) {
                cuda_count++;
            } else {
                cpu_count++;
            }
        }

        std::cout << "  Total weights: " << names.size() << "\n";
        std::cout << "  On CUDA: " << cuda_count << "\n";
        std::cout << "  On CPU: " << cpu_count << "\n\n";

        if (cpu_count > 0) {
            std::cerr << "[FAIL] Some weights are still on CPU!\n";
            return 1;
        }

        // 测试 to_cpu()
        std::cout << "Transferring model back to CPU...\n";
        model.to_cpu();
        std::cout << "✓ Transfer complete\n\n";

        // 验证转移回 CPU
        std::cout << "Checking device after to_cpu() (should be CPU):\n";
        const auto& emb_cpu2 = model.get_embedding();
        const auto& q0_cpu2 = model.get_layer_weight(0, "self_attn.q_proj.weight");
        const auto& lm_head_cpu2 = model.get_lm_head();

        std::cout << "  Embedding device: "
                  << (emb_cpu2.device() == Device::CPU ? "CPU" : "CUDA") << "\n";
        std::cout << "  Layer 0 Q device: "
                  << (q0_cpu2.device() == Device::CPU ? "CPU" : "CUDA") << "\n";
        std::cout << "  LM head device: "
                  << (lm_head_cpu2.device() == Device::CPU ? "CPU" : "CUDA") << "\n\n";

        // 验证所有权重都回到 CPU
        cuda_count = 0;
        cpu_count = 0;
        for (const auto& name : names) {
            const auto& tensor = model.get_weight(name);
            if (tensor.device() == Device::CUDA) {
                cuda_count++;
            } else {
                cpu_count++;
            }
        }

        std::cout << "Final verification:\n";
        std::cout << "  On CUDA: " << cuda_count << "\n";
        std::cout << "  On CPU: " << cpu_count << "\n\n";

        if (cuda_count > 0) {
            std::cerr << "[FAIL] Some weights are still on CUDA!\n";
            return 1;
        }

        // 测试数据完整性（读取一些值）
        std::cout << "Testing data integrity:\n";
        const auto& emb_final = model.get_embedding();
        std::cout << "  Embedding shape: [" << emb_final.dim(0) << ", "
                  << emb_final.dim(1) << "]\n";
        std::cout << "  Embedding numel: " << emb_final.numel() << "\n";
        std::cout << "  ✓ Data accessible\n\n";

        std::cout << "[PASS] CUDA transfer test completed successfully!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Error: " << e.what() << "\n";
        return 1;
    }
}
