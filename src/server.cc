// server.cc — OpenAI-compatible /v1/chat/completions server
//
// 仅支持 POST /v1/chat/completions，单次同步推理（greedy decode）。
// 协议参考 https://platform.openai.com/docs/api-reference/chat/create
//
// 启动方式:
//   ./server --model   models/model.safetensors \
//            --tokenizer models/tokenizer.json  \
//            [--config  models/config.json]     \
//            [--port    8080]                   \
//            [--host    0.0.0.0]                \
//            [--max-tokens 512]
//
// Mock 模式（不需要 GPU / 模型文件，仅测试 HTTP 层）:
//   ./server --mock [--tokenizer models/tokenizer.json] [--port 8080]
//   --mock 时 tokenizer 可选；推理结果为固定 echo 文本。

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "tinyllama/qwen3_model.h"
#include "tinyllama/tensor.h"
#include "tinyllama/tokenizer.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace tinyllama;

// ─────────────────────────────────────────────────────────────────────────────
//  CLI 参数
// ─────────────────────────────────────────────────────────────────────────────
struct ServerArgs {
    std::string model_path;
    std::string tokenizer_path;   // tokenizer.json
    std::string config_path;      // config.json（可选）
    std::string host         = "0.0.0.0";
    int         port         = 8080;
    int         max_new_tokens = 512;
    bool        mock         = false;  // --mock：跳过模型加载，用 echo 函数代替推理
};

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --model <path> --tokenizer <path>"
        " [--config <path>] [--host <h>] [--port <p>] [--max-tokens <n>]\n"
        "       %s --mock [--tokenizer <path>] [--port <p>]\n",
        prog, prog);
}

static ServerArgs parse_args(int argc, char** argv) {
    ServerArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string key(argv[i]);
        if ((key == "--model" || key == "-m") && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if ((key == "--tokenizer" || key == "-t") && i + 1 < argc) {
            args.tokenizer_path = argv[++i];
        } else if ((key == "--config" || key == "-c") && i + 1 < argc) {
            args.config_path = argv[++i];
        } else if (key == "--host" && i + 1 < argc) {
            args.host = argv[++i];
        } else if (key == "--port" && i + 1 < argc) {
            args.port = std::stoi(argv[++i]);
        } else if (key == "--max-tokens" && i + 1 < argc) {
            args.max_new_tokens = std::stoi(argv[++i]);
        } else if (key == "--mock") {
            args.mock = true;
        } else if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
    }
    if (!args.mock && (args.model_path.empty() || args.tokenizer_path.empty())) {
        print_usage(argv[0]);
        std::exit(1);
    }
    return args;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Qwen3 chat 模板
//
//  格式（ChatML）：
//    <|im_start|>system
//    {content}<|im_end|>
//    <|im_start|>user
//    {content}<|im_end|>
//    <|im_start|>assistant
//    （等待模型生成）
// ─────────────────────────────────────────────────────────────────────────────
static std::string build_prompt(const json& messages) {
    std::string prompt;
    for (const auto& msg : messages) {
        std::string role    = msg.value("role",    "user");
        std::string content = msg.value("content", "");
        prompt += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
    }
    prompt += "<|im_start|>assistant\n";
    return prompt;
}

// ─────────────────────────────────────────────────────────────────────────────
//  greedy sampling
// ─────────────────────────────────────────────────────────────────────────────
static int32_t greedy_sample(const float* logits, int64_t vocab_size) {
    int32_t best_id  = 0;
    float   best_val = logits[0];
    for (int64_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > best_val) { best_val = logits[i]; best_id = static_cast<int32_t>(i); }
    }
    return best_id;
}

// ─────────────────────────────────────────────────────────────────────────────
//  生成请求 ID
// ─────────────────────────────────────────────────────────────────────────────
static std::string make_id() {
    static std::atomic<uint64_t> counter{0};
    return "chatcmpl-" + std::to_string(counter.fetch_add(1));
}

// ─────────────────────────────────────────────────────────────────────────────
//  构造 OpenAI chat/completions 响应
// ─────────────────────────────────────────────────────────────────────────────
static json make_response(const std::string& id,
                           const std::string& model_name,
                           const std::string& content,
                           const std::string& finish_reason,
                           int prompt_tokens,
                           int completion_tokens) {
    int64_t ts = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    return json{
        {"id",      id},
        {"object",  "chat.completion"},
        {"created", ts},
        {"model",   model_name},
        {"choices", json::array({json{
            {"index", 0},
            {"message", json{{"role", "assistant"}, {"content", content}}},
            {"finish_reason", finish_reason}
        }})},
        {"usage", json{
            {"prompt_tokens",     prompt_tokens},
            {"completion_tokens", completion_tokens},
            {"total_tokens",      prompt_tokens + completion_tokens}
        }}
    };
}

// ─────────────────────────────────────────────────────────────────────────────
//  推理：自回归 greedy decode
//    - 每步以完整上下文调用 forward()，取最后位置 logits
//    - 遇到 <|im_end|>（151645）或达到 max_new_tokens 停止
//    - 所有生成 token 最终一次性 decode 成 UTF-8 文本
// ─────────────────────────────────────────────────────────────────────────────
static std::string run_inference(Qwen3Model&           model,
                                  const Qwen3Tokenizer& tokenizer,
                                  const std::vector<int32_t>& prompt_ids,
                                  int max_new_tokens,
                                  std::string& finish_reason) {
    std::vector<int32_t> ids = prompt_ids;
    std::vector<int32_t> generated_ids;
    const int64_t vocab_size = model.config().vocab_size;

    for (int step = 0; step < max_new_tokens; ++step) {
        int64_t seq_len = static_cast<int64_t>(ids.size());

        // 构造 input_ids: [1, seq_len] Int32 CUDA tensor
        Tensor input_ids({1, seq_len}, DType::Int32, Device::CUDA);
        input_ids.copy_from_cpu(ids.data(), static_cast<size_t>(seq_len) * sizeof(int32_t));

        // forward → [1, seq_len, vocab_size]
        Tensor logits     = model.forward(input_ids);
        Tensor logits_cpu = logits.cpu();

        const float* last_logits =
            logits_cpu.data_ptr<float>() + (seq_len - 1) * vocab_size;

        int32_t next_id = greedy_sample(last_logits, vocab_size);

        // EOS：<|im_end|> = 151645
        if (next_id == Qwen3Tokenizer::kImEndId) {
            finish_reason = "stop";
            break;
        }

        generated_ids.push_back(next_id);
        ids.push_back(next_id);

        if (step == max_new_tokens - 1) {
            finish_reason = "length";
        }
    }

    return tokenizer.decode(generated_ids);
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

// 推理函数签名：(prompt_ids, max_new_tokens, finish_reason) → output_text
using InferFn = std::function<std::string(const std::vector<int32_t>&, int, std::string&)>;

int main(int argc, char** argv) {
    ServerArgs args = parse_args(argc, argv);

    // ── Tokenizer（mock 模式下可选）────────────────────────────────────────
    std::unique_ptr<Qwen3Tokenizer> tokenizer;
    if (!args.tokenizer_path.empty()) {
        printf("[server] Loading tokenizer: %s\n", args.tokenizer_path.c_str());
        tokenizer = std::make_unique<Qwen3Tokenizer>(args.tokenizer_path);
        printf("[server] Tokenizer loaded. vocab_size=%d\n", tokenizer->vocab_size());
    }

    // ── 构造推理函数 ──────────────────────────────────────────────────────
    InferFn infer_fn;

    if (args.mock) {
        // Mock 模式：直接返回固定文本，完全不需要模型 / GPU
        printf("[server] Running in MOCK mode (no model loaded)\n");
        infer_fn = [](const std::vector<int32_t>& /*ids*/, int /*max_tokens*/,
                      std::string& finish_reason) -> std::string {
            finish_reason = "stop";
            return "[mock response] Hello from mock server!";
        };
    } else {
        // 真实模式：加载模型
        printf("[server] Loading model: %s\n", args.model_path.c_str());
        auto model = std::make_shared<Qwen3Model>(
            args.config_path.empty()
                ? Qwen3Model(args.model_path)
                : Qwen3Model(args.model_path, args.config_path));
        model->to_cuda();
        printf("[server] Model loaded. vocab_size=%ld hidden=%ld layers=%ld\n",
               model->config().vocab_size,
               model->config().hidden_size,
               model->config().num_hidden_layers);

        infer_fn = [model, &tokenizer](const std::vector<int32_t>& ids, int max_tokens,
                                       std::string& finish_reason) -> std::string {
            return run_inference(*model, *tokenizer, ids, max_tokens, finish_reason);
        };
    }

    // ── HTTP server ───────────────────────────────────────────────────────
    httplib::Server svr;

    // POST /v1/chat/completions
    svr.Post("/v1/chat/completions",
        [&](const httplib::Request& req, httplib::Response& res) {
            json body;
            try {
                body = json::parse(req.body);
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json{{"error", json{
                    {"message", std::string("Invalid JSON: ") + e.what()},
                    {"type", "invalid_request_error"}}}}.dump(), "application/json");
                return;
            }

            if (!body.contains("messages") || !body["messages"].is_array()) {
                res.status = 400;
                res.set_content(json{{"error", json{
                    {"message", "Field 'messages' is required and must be an array"},
                    {"type", "invalid_request_error"}}}}.dump(), "application/json");
                return;
            }

            std::string model_name = body.value("model", "qwen3");
            int max_tokens = body.value("max_tokens", args.max_new_tokens);
            if (max_tokens <= 0 || max_tokens > 4096) max_tokens = args.max_new_tokens;

            // 构造 ChatML prompt
            std::string prompt       = build_prompt(body["messages"]);
            std::vector<int32_t> ids;
            int prompt_tokens = 0;
            if (tokenizer) {
                ids           = tokenizer->encode(prompt);
                prompt_tokens = static_cast<int>(ids.size());
            } else {
                // mock 无 tokenizer：用字节数近似 token 数
                prompt_tokens = static_cast<int>(prompt.size());
            }

            printf("[server] request: model=%s max_tokens=%d prompt_tokens=%d\n",
                   model_name.c_str(), max_tokens, prompt_tokens);

            std::string output;
            std::string finish_reason = "stop";
            try {
                output = infer_fn(ids, max_tokens, finish_reason);
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content(json{{"error", json{
                    {"message", std::string("Inference error: ") + e.what()},
                    {"type", "server_error"}}}}.dump(), "application/json");
                return;
            }

            int completion_tokens = tokenizer
                ? static_cast<int>(tokenizer->encode(output).size())
                : static_cast<int>(output.size());

            res.status = 200;
            res.set_content(
                make_response(make_id(), model_name, output,
                              finish_reason, prompt_tokens, completion_tokens).dump(),
                "application/json");

            printf("[server] response: finish_reason=%s completion_tokens=%d\n",
                   finish_reason.c_str(), completion_tokens);
        });

    // GET /health
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(json{{"status", "ok"}}.dump(), "application/json");
    });

    // GET /v1/models
    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        int64_t ts = static_cast<int64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        res.set_content(json{
            {"object", "list"},
            {"data", json::array({json{
                {"id", args.mock ? "mock" : "qwen3"}, {"object", "model"},
                {"created", ts}, {"owned_by", "local"}
            }})}
        }.dump(), "application/json");
    });

    printf("[server] Listening on %s:%d\n", args.host.c_str(), args.port);
    svr.listen(args.host.c_str(), args.port);
    return 0;
}
