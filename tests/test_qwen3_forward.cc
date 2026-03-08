// tests/test_qwen3_forward.cc — 测试 Qwen3Model 的前向传播

#include "tinyllama/qwen3_model.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace tinyllama;

int main() {
    std::cout << "=== Qwen3Model Forward Pass Test ===\n\n";

    try {
        // 加载模型
        std::string model_path = "../../models/Qwen3-0.6B/model.safetensors";
        std::string config_path = "../../models/Qwen3-0.6B/config.json";

        std::cout << "Loading model...\n";
        Qwen3Model model(model_path, config_path);
        std::cout << "✓ Model loaded\n\n";

        // 打印配置
        const auto& cfg = model.config();
        std::cout << "Model Configuration:\n";
        std::cout << "  vocab_size: " << cfg.vocab_size << "\n";
        std::cout << "  hidden_size: " << cfg.hidden_size << "\n";
        std::cout << "  num_hidden_layers: " << cfg.num_hidden_layers << "\n";
        std::cout << "  num_attention_heads: " << cfg.num_attention_heads << "\n";
        std::cout << "  num_key_value_heads: " << cfg.num_key_value_heads << " (GQA)\n";
        std::cout << "  head_dim: " << cfg.head_dim << "\n\n";

        // 将模型移到 CUDA
        std::cout << "Moving model to CUDA...\n";
        model.to_cuda();
        std::cout << "✓ Model on CUDA\n\n";

        // 创建测试输入：[batch=1, seq_len=4]
        std::cout << "Creating test input...\n";
        int batch = 1;
        int seq_len = 4;

        Tensor input_ids({batch, seq_len}, DType::Int32, Device::CPU);
        int32_t* ids_ptr = input_ids.data_ptr<int32_t>();
        // 简单的测试 token IDs
        ids_ptr[0] = 1;
        ids_ptr[1] = 2;
        ids_ptr[2] = 3;
        ids_ptr[3] = 4;

        // 移到 CUDA
        Tensor input_ids_cuda = input_ids.cuda();
        std::cout << "  Input shape: [" << batch << ", " << seq_len << "]\n";
        std::cout << "  Input IDs: [1, 2, 3, 4]\n\n";

        // 前向传播
        std::cout << "Running forward pass...\n";
        Tensor logits = model.forward(input_ids_cuda);
        std::cout << "✓ Forward pass completed\n\n";

        // 检查输出形状
        std::cout << "Output shape: [";
        for (int i = 0; i < logits.ndim(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << logits.dim(i);
        }
        std::cout << "]\n";

        // 验证输出形状
        bool shape_ok = (logits.ndim() == 3) &&
                       (logits.dim(0) == batch) &&
                       (logits.dim(1) == seq_len) &&
                       (logits.dim(2) == cfg.vocab_size);

        if (!shape_ok) {
            std::cerr << "[FAIL] Output shape mismatch!\n";
            std::cerr << "  Expected: [" << batch << ", " << seq_len << ", "
                      << cfg.vocab_size << "]\n";
            return 1;
        }

        std::cout << "✓ Output shape correct: [batch, seq_len, vocab_size]\n\n";

        // 将 logits 拷贝回 CPU 查看一些值
        Tensor logits_cpu = logits.cpu();
        const float* logits_ptr = logits_cpu.data_ptr<float>();

        std::cout << "Sample logits (first token, first 10 vocab entries):\n";
        for (int i = 0; i < 10; ++i) {
            std::cout << "  vocab[" << i << "]: "
                      << std::fixed << std::setprecision(4)
                      << logits_ptr[i] << "\n";
        }
        std::cout << "\n";

        // 检查是否有 NaN 或 Inf
        bool has_nan = false;
        bool has_inf = false;
        size_t total = logits_cpu.numel();
        for (size_t i = 0; i < total; ++i) {
            float v = logits_ptr[i];
            if (v != v) has_nan = true;  // NaN check
            if (v == INFINITY || v == -INFINITY) has_inf = true;
        }

        if (has_nan) {
            std::cerr << "[FAIL] Output contains NaN values!\n";
            return 1;
        }
        if (has_inf) {
            std::cerr << "[FAIL] Output contains Inf values!\n";
            return 1;
        }

        std::cout << "✓ No NaN or Inf in output\n";
        std::cout << "\n[PASS] Forward pass test completed successfully!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Error: " << e.what() << "\n";
        return 1;
    }
}
