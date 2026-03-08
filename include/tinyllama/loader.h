#pragma once

#include "tensor.h"
#include <string>
#include <unordered_map>

// ─────────────────────────────────────────────────────────────────────────────
//  loader.h — load model weights from .safetensors files
//
//  Usage:
//    auto weights = tinyllama::load_safetensors("model.safetensors");
//    Tensor wq = weights.at("model.layers.0.self_attn.q_proj.weight");
//
//  All tensors are returned on Device::CPU by default.
//  Call .cuda() on individual tensors to move them to GPU.
//
//  Supported dtypes: F32, F16, BF16 (F16/BF16 are converted to F32),
//                    I32, I64 (I64 truncated to I32).
// ─────────────────────────────────────────────────────────────────────────────

namespace tinyllama {

// Load all tensors from a safetensors file.
// Throws std::runtime_error on failure.
std::unordered_map<std::string, Tensor>
load_safetensors(const std::string& path);

}  // namespace tinyllama
