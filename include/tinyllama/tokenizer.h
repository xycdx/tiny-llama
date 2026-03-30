#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  tokenizer.h — Qwen3 BPE tokenizer
//
//  实现 HuggingFace tokenizer.json 格式的 Qwen3 tokenizer：
//    - GPT-2 ByteLevel 字节编码（256 字节 → unicode 字符）
//    - 简化的 Qwen3 pre-tokenizer（regex split 近似实现）
//    - BPE 合并（从 tokenizer.json model.merges 加载）
//    - 特殊 token 直接映射（<|im_start|>, <|im_end|> 等）
//
//  Usage:
//    Qwen3Tokenizer tok("models/tokenizer.json");
//    auto ids = tok.encode("<|im_start|>user\nHello<|im_end|>\n");
//    std::string text = tok.decode(ids);
// ─────────────────────────────────────────────────────────────────────────────

namespace tinyllama {

class Qwen3Tokenizer {
public:
    // 特殊 token ID（Qwen3 固定值）
    static constexpr int32_t kEotId     = 151643;  // <|endoftext|>
    static constexpr int32_t kImStartId = 151644;  // <|im_start|>
    static constexpr int32_t kImEndId   = 151645;  // <|im_end|>  (EOS)

    // 从 HuggingFace tokenizer.json 加载
    explicit Qwen3Tokenizer(const std::string& tokenizer_json_path);

    // 编码：text → token id 列表（自动识别特殊 token，其余走 BPE）
    std::vector<int32_t> encode(const std::string& text) const;

    // 解码：token id 列表 → text
    std::string decode(const std::vector<int32_t>& ids) const;

    // 获取词表大小
    int32_t vocab_size() const { return static_cast<int32_t>(id_to_token_.size()); }

private:
    // ── 词表 ──────────────────────────────────────────────
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<std::string>                  id_to_token_;

    // ── BPE 合并规则：pair → rank（rank 越小优先级越高）──
    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
            size_t h1 = std::hash<std::string>{}(p.first);
            size_t h2 = std::hash<std::string>{}(p.second);
            return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL);
        }
    };
    std::unordered_map<std::pair<std::string, std::string>, int32_t, PairHash> merge_ranks_;

    // ── 特殊 token（按长度降序排列，用于文本中的贪心匹配）──
    std::vector<std::pair<std::string, int32_t>> special_tokens_sorted_;  // (token_str, id)

    // ── GPT-2 字节编码映射 ─────────────────────────────────
    std::array<std::string, 256> byte_to_char_;              // byte → UTF-8 string (unicode char)
    std::unordered_map<std::string, uint8_t> char_to_byte_;  // UTF-8 string → byte

    // ── 内部方法 ──────────────────────────────────────────

    // 初始化 GPT-2 字节编码表
    void init_byte_encoder();

    // Pre-tokenizer：将文本拆分为 word 列表（近似 Qwen3 regex）
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    // 对单个 pre-token 应用 BPE，返回 token 字符串列表
    std::vector<std::string> apply_bpe(const std::string& word) const;

    // 编码不含特殊 token 的文本片段
    std::vector<int32_t> encode_segment(const std::string& text) const;
};

}  // namespace tinyllama
