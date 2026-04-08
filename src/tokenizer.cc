// tokenizer.cc — Qwen3 BPE tokenizer 实现
//
// 编码流水线：
//   1. 贪心匹配特殊 token（<|im_start|> 等） → 直接输出 ID，跳过 BPE
//   2. pre_tokenize：按词边界拆分（近似 Qwen3 regex）
//   3. 每个 word：字节 → GPT-2 byte_to_char_ 映射 → BPE 合并
//   4. 查 token_to_id_ 得到 ID
//
// 解码流水线：
//   id → token 字符串 → GPT-2 char_to_byte_ 逆映射 → UTF-8 bytes

#include "tinyllama/tokenizer.h"

#include <algorithm>
#include <climits>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "nlohmann/json.hpp"

namespace tinyllama {

using json = nlohmann::json;

// ─────────────────────────────────────────────────────────────────────────────
//  UTF-8 工具
// ─────────────────────────────────────────────────────────────────────────────

// 将 unicode codepoint 编码为 UTF-8 字符串
static std::string codepoint_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xF0 | (cp >> 18));
        s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

// UTF-8 字节序列解码为 codepoint，返回消耗的字节数
static int utf8_decode(const char* s, size_t n, uint32_t& cp) {
    unsigned char c0 = static_cast<unsigned char>(s[0]);
    if (c0 < 0x80) {
        cp = c0; return 1;
    } else if (c0 < 0xE0 && n >= 2) {
        cp = ((c0 & 0x1F) << 6) | (static_cast<unsigned char>(s[1]) & 0x3F);
        return 2;
    } else if (c0 < 0xF0 && n >= 3) {
        cp = ((c0 & 0x0F) << 12) | ((static_cast<unsigned char>(s[1]) & 0x3F) << 6)
           | (static_cast<unsigned char>(s[2]) & 0x3F);
        return 3;
    } else if (n >= 4) {
        cp = ((c0 & 0x07) << 18) | ((static_cast<unsigned char>(s[1]) & 0x3F) << 12)
           | ((static_cast<unsigned char>(s[2]) & 0x3F) << 6)
           | (static_cast<unsigned char>(s[3]) & 0x3F);
        return 4;
    }
    cp = 0xFFFD; return 1;
}

// UTF-8 字节序列中下一个字符占用的字节数（根据首字节）
static inline int utf8_char_len(unsigned char c) {
    if (c < 0x80) return 1;
    if (c < 0xE0) return 2;
    if (c < 0xF0) return 3;
    return 4;
}

// ─────────────────────────────────────────────────────────────────────────────
//  GPT-2 字节编码（bytes_to_unicode）
//
//  将 256 个字节映射为可打印的 unicode 字符，规则：
//    - bytes 33–126 (ASCII 可打印) → 自身
//    - bytes 161–172, 174–255 (Latin supplement) → 自身
//    - 其余 68 个字节（0–32, 127, 128–160, 173） → U+0100 起顺序分配
// ─────────────────────────────────────────────────────────────────────────────
void Qwen3Tokenizer::init_byte_encoder() {
    // 1. 先把"直接映射"的字节收集起来
    std::vector<int> bs, cs;
    for (int c = 33; c <= 126; ++c) { bs.push_back(c); cs.push_back(c); }
    for (int c = 161; c <= 172; ++c) { bs.push_back(c); cs.push_back(c); }
    for (int c = 174; c <= 255; ++c) { bs.push_back(c); cs.push_back(c); }

    // 2. 剩余字节从 U+0100 起依次分配
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n++);
        }
    }

    // 3. 建立双向映射表
    for (int i = 0; i < 256; ++i) {
        std::string utf8 = codepoint_to_utf8(static_cast<uint32_t>(cs[i]));
        byte_to_char_[bs[i]] = utf8;
        char_to_byte_[utf8]  = static_cast<uint8_t>(bs[i]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  构造函数：加载 tokenizer.json
// ─────────────────────────────────────────────────────────────────────────────
Qwen3Tokenizer::Qwen3Tokenizer(const std::string& tokenizer_json_path) {
    init_byte_encoder();

    // 读取 tokenizer.json
    std::ifstream f(tokenizer_json_path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open tokenizer: " + tokenizer_json_path);

    json t;
    try {
        f >> t;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse tokenizer.json: " + std::string(e.what()));
    }

    // ── 加载词表（model.vocab） ────────────────────────────
    const auto& vocab = t["model"]["vocab"];
    size_t vocab_size = vocab.size();
    id_to_token_.resize(vocab_size);
    for (auto& [token_str, id_val] : vocab.items()) {
        int32_t id = id_val.get<int32_t>();
        token_to_id_[token_str] = id;
        if (id >= 0 && static_cast<size_t>(id) < vocab_size)
            id_to_token_[id] = token_str;
    }

    // ── 加载 added_tokens（特殊 token，会超出 vocab_size 范围） ──
    if (t.contains("added_tokens")) {
        for (const auto& entry : t["added_tokens"]) {
            std::string content = entry["content"].get<std::string>();
            int32_t     id      = entry["id"].get<int32_t>();
            token_to_id_[content] = id;
            // 扩展 id_to_token_ 以容纳特殊 token
            if (id >= static_cast<int32_t>(id_to_token_.size()))
                id_to_token_.resize(id + 1);
            id_to_token_[id] = content;
            special_tokens_sorted_.emplace_back(content, id);
        }
    }

    // 按字符串长度降序排列，保证贪心匹配时先匹配较长的特殊 token
    std::sort(special_tokens_sorted_.begin(), special_tokens_sorted_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    // ── 加载 BPE 合并规则（model.merges） ────────────────────
    const auto& merges = t["model"]["merges"];
    merge_ranks_.reserve(merges.size());
    for (size_t i = 0; i < merges.size(); ++i) {
        std::string left  = merges[i][0].get<std::string>();
        std::string right = merges[i][1].get<std::string>();
        merge_ranks_[{left, right}] = static_cast<int32_t>(i);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  pre_tokenize — 近似 Qwen3 regex pre-tokenizer
//
//  原始 regex（Unicode 属性版本，C++ std::regex 不直接支持）：
//    (?i:'s|'t|'re|'ve|'m|'ll|'d)
//    |[^\r\n\p{L}\p{N}]?\p{L}+
//    |\p{N}
//    | ?[^\s\p{L}\p{N}]+[\r\n]*
//    |\s*[\r\n]+
//    |\s+(?!\S)
//    |\s+
//
//  实现策略：
//    - ASCII 字母(a-z,A-Z) 和多字节 UTF-8 起始字节(>=0xC2) 视为 letter
//    - ASCII 数字每次取一个（\p{N} 匹配单个数字字符）
//    - 空格 + letter → 一个 pre-token（匹配 [^\r\n\p{L}\p{N}]?\p{L}+）
//    - 空格 + 标点 → 一个 pre-token（匹配 ?[^\s\p{L}\p{N}]+[\r\n]*）
//    - 连续换行 → 一个 pre-token
//    - 缩略词 ('s, 't 等) → 一个 pre-token
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::string> Qwen3Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> result;
    const size_t n = text.size();
    size_t i = 0;

    auto is_letter_start = [](unsigned char c) -> bool {
        // ASCII 字母 或 多字节 UTF-8 起始字节
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0xC2;
    };
    auto is_digit = [](unsigned char c) -> bool {
        return c >= '0' && c <= '9';
    };
    auto is_newline = [](unsigned char c) -> bool {
        return c == '\n' || c == '\r';
    };
    auto is_space = [](unsigned char c) -> bool {
        return c == ' ' || c == '\t';
    };
    auto is_ws = [&](unsigned char c) -> bool {
        return is_space(c) || is_newline(c);
    };
    // 是否为"非空白、非字母、非数字"（= ASCII 标点/符号 或 0x80-0xBF 续接字节）
    auto is_punct = [&](unsigned char c) -> bool {
        return !is_ws(c) && !is_letter_start(c) && !is_digit(c) && c != '\'';
    };

    while (i < n) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        // ── 1. 英文缩略词 ('s, 't, 're, 've, 'm, 'll, 'd)，大小写不敏感 ──
        if (c == '\'') {
            static const std::pair<const char*, int> contracs[] = {
                {"'ll", 3}, {"'re", 3}, {"'ve", 3},
                {"'s",  2}, {"'t",  2}, {"'m",  2}, {"'d", 2}
            };
            bool matched = false;
            for (auto& [pat, len] : contracs) {
                if (i + static_cast<size_t>(len) <= n) {
                    bool ok = true;
                    for (int k = 0; k < len; ++k) {
                        if (tolower(static_cast<unsigned char>(text[i + k])) != pat[k]) {
                            ok = false; break;
                        }
                    }
                    if (ok) {
                        result.push_back(text.substr(i, len));
                        i += len; matched = true; break;
                    }
                }
            }
            if (!matched) {
                // 单独的撇号当标点处理
                result.push_back(text.substr(i, 1));
                ++i;
            }
            continue;
        }

        // ── 2. 单个数字（\p{N}） ──
        if (is_digit(c)) {
            result.push_back(text.substr(i, 1));
            ++i;
            continue;
        }

        // ── 3. 换行（含前导空白 \s*[\r\n]+） ──
        if (is_newline(c)) {
            size_t j = i;
            while (j < n && is_newline(static_cast<unsigned char>(text[j]))) ++j;
            result.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        // ── 4. 空格 ──
        if (c == ' ') {
            if (i + 1 < n) {
                unsigned char nxt = static_cast<unsigned char>(text[i + 1]);

                // 4a. 空格 + letter：[^\r\n\p{L}\p{N}]?\p{L}+
                if (is_letter_start(nxt)) {
                    size_t j = i + 1;
                    while (j < n) {
                        unsigned char nc = static_cast<unsigned char>(text[j]);
                        if (!is_letter_start(nc) && !(nc >= 0x80 && nc < 0xC0)) break;
                        j += utf8_char_len(nc);
                    }
                    result.push_back(text.substr(i, j - i));
                    i = j;
                    continue;
                }

                // 4b. 空格 + 标点：?[^\s\p{L}\p{N}]+[\r\n]*
                if (is_punct(nxt)) {
                    size_t j = i + 1;
                    while (j < n && is_punct(static_cast<unsigned char>(text[j]))) ++j;
                    // 消耗尾随换行
                    while (j < n && is_newline(static_cast<unsigned char>(text[j]))) ++j;
                    result.push_back(text.substr(i, j - i));
                    i = j;
                    continue;
                }
            }
            // 其余空格（连续空格、行尾空格）
            size_t j = i;
            while (j < n && text[j] == ' ') ++j;
            result.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        // ── 5. Tab / 其他空白 ──
        if (is_space(c)) {
            size_t j = i;
            while (j < n && is_space(static_cast<unsigned char>(text[j]))) ++j;
            result.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        // ── 6. Letter 序列（无前导空格） [^\r\n\p{L}\p{N}]?\p{L}+ 中无前缀的情形 ──
        if (is_letter_start(c)) {
            size_t j = i;
            while (j < n) {
                unsigned char nc = static_cast<unsigned char>(text[j]);
                // 字母起始字节 或 UTF-8 续接字节
                if (!is_letter_start(nc) && !(nc >= 0x80 && nc < 0xC0)) break;
                j += utf8_char_len(nc);
            }
            result.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        // ── 7. 标点 / 符号序列（可能有空格前缀，已在 4b 处理）──
        {
            size_t j = i;
            while (j < n && is_punct(static_cast<unsigned char>(text[j]))) ++j;
            if (j > i) {
                result.push_back(text.substr(i, j - i));
                i = j;
            } else {
                // 兜底：单字符推进，避免死循环
                result.push_back(text.substr(i, 1));
                ++i;
            }
        }
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  apply_bpe — 对单个 pre-token 应用 BPE
//
//  算法：
//    1. 将 word 的每个字节通过 byte_to_char_ 转为 unicode 字符（GPT-2 编码）
//    2. 循环找优先级最高（rank 最小）的相邻 pair
//    3. 合并该 pair 的所有出现，直到无可合并的 pair
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::string> Qwen3Tokenizer::apply_bpe(const std::string& word) const {
    // 字节 → GPT-2 unicode 字符 token 列表
    std::vector<std::string> tokens;
    tokens.reserve(word.size());
    for (unsigned char b : word) {
        tokens.push_back(byte_to_char_[b]);
    }
    if (tokens.size() <= 1) return tokens;

    while (tokens.size() >= 2) {
        // 找 rank 最小的相邻 pair
        int best_rank = INT_MAX;
        int best_pos  = -1;
        for (size_t k = 0; k + 1 < tokens.size(); ++k) {
            auto it = merge_ranks_.find({tokens[k], tokens[k + 1]});
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos  = static_cast<int>(k);
            }
        }
        if (best_pos == -1) break;

        // 合并所有该 pair 的出现
        const std::string& left  = tokens[best_pos];
        const std::string& right = tokens[best_pos + 1];
        std::string merged = left + right;

        std::vector<std::string> new_tokens;
        new_tokens.reserve(tokens.size());
        for (size_t k = 0; k < tokens.size(); ) {
            if (k + 1 < tokens.size() && tokens[k] == left && tokens[k + 1] == right) {
                new_tokens.push_back(merged);
                k += 2;
            } else {
                new_tokens.push_back(tokens[k]);
                ++k;
            }
        }
        tokens = std::move(new_tokens);
    }

    return tokens;
}

// ─────────────────────────────────────────────────────────────────────────────
//  encode_segment — 对不含特殊 token 的文本片段编码
// ─────────────────────────────────────────────────────────────────────────────
std::vector<int32_t> Qwen3Tokenizer::encode_segment(const std::string& text) const {
    if (text.empty()) return {};

    std::vector<int32_t> ids;
    for (const std::string& word : pre_tokenize(text)) {
        for (const std::string& tok_str : apply_bpe(word)) {
            auto it = token_to_id_.find(tok_str);
            if (it != token_to_id_.end()) {
                ids.push_back(it->second);
            }
            // 未找到的 token 直接跳过（不应出现，词表完备）
        }
    }
    return ids;
}

// ─────────────────────────────────────────────────────────────────────────────
//  encode — 完整编码（先提取特殊 token，再对普通文本走 BPE）
// ─────────────────────────────────────────────────────────────────────────────
std::vector<int32_t> Qwen3Tokenizer::encode(const std::string& text) const {
    // 贪心匹配特殊 token，将 text 切分为 [segment, special, segment, ...] 序列
    std::vector<int32_t> ids;
    size_t i = 0;
    const size_t n = text.size();

    while (i < n) {
        // 尝试匹配特殊 token（按长度降序，保证最长匹配优先）
        bool found_special = false;
        for (const auto& [tok_str, tok_id] : special_tokens_sorted_) {
            if (n - i >= tok_str.size() &&
                text.compare(i, tok_str.size(), tok_str) == 0) {
                ids.push_back(tok_id);
                i += tok_str.size();
                found_special = true;
                break;
            }
        }
        if (found_special) continue;

        // 找下一个特殊 token 的起始位置，中间的普通文本走 BPE
        size_t next_special = n;
        for (const auto& [tok_str, tok_id] : special_tokens_sorted_) {
            size_t pos = text.find(tok_str, i);
            if (pos != std::string::npos && pos < next_special)
                next_special = pos;
        }

        // 对 [i, next_special) 的普通文本编码
        if (next_special > i) {
            auto seg_ids = encode_segment(text.substr(i, next_special - i));
            ids.insert(ids.end(), seg_ids.begin(), seg_ids.end());
        }
        i = next_special;
    }

    return ids;
}

// ─────────────────────────────────────────────────────────────────────────────
//  decode — token id 列表 → UTF-8 文本
//
//  流程：
//    - 特殊 token ID（>= kEotId）：直接输出其字符串内容
//    - 普通 token：token 字符串中每个 unicode char → char_to_byte_ → 字节 → UTF-8
// ─────────────────────────────────────────────────────────────────────────────
std::string Qwen3Tokenizer::decode(const std::vector<int32_t>& ids) const {
    // 先把所有 token 的字节还原出来
    std::vector<uint8_t> bytes;
    bytes.reserve(ids.size() * 3);

    for (int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) continue;
        const std::string& tok = id_to_token_[id];
        if (tok.empty()) continue;

        // 特殊 token 直接当 UTF-8 字符串输出（先刷已有字节）
        if (id >= kEotId) {
            // 输出之前积累的字节
            for (uint8_t b : bytes) {}  // 已经 reserve，下面统一处理
            // 特殊 token 直接追加原始字符串字节
            for (unsigned char c : tok)
                bytes.push_back(c);
            continue;
        }

        // 普通 token：遍历 UTF-8 字符，每个字符通过 char_to_byte_ 还原字节
        const char* p = tok.data();
        const char* end = p + tok.size();
        while (p < end) {
            uint32_t cp;
            int consumed = utf8_decode(p, end - p, cp);
            // 重新编码为 UTF-8 字符串以查表
            std::string ch = codepoint_to_utf8(cp);
            auto it = char_to_byte_.find(ch);
            if (it != char_to_byte_.end()) {
                bytes.push_back(it->second);
            }
            p += consumed;
        }
    }

    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

}  // namespace tinyllama
