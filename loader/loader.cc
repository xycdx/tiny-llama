// loader/loader.cc — load .safetensors weights into tinyllama::Tensor
//
// Uses third_party/safetensors-cpp (safetensors.hh, single-header).
// Strategy: mmap the file (zero-copy), then copy each tensor's raw bytes
// into a CPU Tensor, converting F16/BF16 → F32 on the fly.

#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"

#include "tinyllama/loader.h"
#include "tinyllama/tensor.h"

#include <cstring>
#include <stdexcept>
#include <cstdint>

namespace tinyllama {

// ── dtype helpers ─────────────────────────────────────────────────────────────

static DType st_dtype_to_tinyllama(safetensors::dtype d) {
    switch (d) {
        
        case safetensors::dtype::kFLOAT32: return DType::Float32;
        case safetensors::dtype::kFLOAT16: return DType::Float32;  // converted
        case safetensors::dtype::kBFLOAT16: return DType::Float32; // converted
        case safetensors::dtype::kINT32:   return DType::Int32;
        case safetensors::dtype::kINT64:   return DType::Int32;    // truncated
        default:
            throw std::runtime_error(
                "load_safetensors: unsupported dtype " +
                safetensors::get_dtype_str(d));
    }
}

// Copy raw bytes → F32 Tensor, handling dtype conversion.
static void copy_to_f32(float* dst,
                        const uint8_t* src,
                        size_t n,
                        safetensors::dtype d) {
    switch (d) {
        case safetensors::dtype::kFLOAT32:
            std::memcpy(dst, src, n * sizeof(float));
            break;
        case safetensors::dtype::kFLOAT16: {
            const uint16_t* s16 = reinterpret_cast<const uint16_t*>(src);
            for (size_t i = 0; i < n; ++i)
                dst[i] = safetensors::fp16_to_float(s16[i]);
            break;
        }
        case safetensors::dtype::kBFLOAT16: {
            const uint16_t* s16 = reinterpret_cast<const uint16_t*>(src);
            for (size_t i = 0; i < n; ++i)
                dst[i] = safetensors::bfloat16_to_float(s16[i]);
            break;
        }
        default:
            break;
    }
}

static void copy_to_i32(int32_t* dst,
                        const uint8_t* src,
                        size_t n,
                        safetensors::dtype d) {
    switch (d) {
        case safetensors::dtype::kINT32:
            std::memcpy(dst, src, n * sizeof(int32_t));
            break;
        case safetensors::dtype::kINT64: {
            const int64_t* s64 = reinterpret_cast<const int64_t*>(src);
            for (size_t i = 0; i < n; ++i)
                dst[i] = static_cast<int32_t>(s64[i]);
            break;
        }
        default:
            break;
    }
}

// ── public API ────────────────────────────────────────────────────────────────

std::unordered_map<std::string, Tensor>
load_safetensors(const std::string& path) {
    safetensors::safetensors_t st;
    std::string warn, err;

    bool ok = safetensors::mmap_from_file(path, &st, &warn, &err);
    if (!ok)
        throw std::runtime_error("load_safetensors: " + err);

    if (!safetensors::validate_data_offsets(st, err))
        throw std::runtime_error("load_safetensors: invalid offsets: " + err);

    const uint8_t* databuf = st.mmaped ? st.databuffer_addr
                                       : st.storage.data();

    std::unordered_map<std::string, Tensor> result;
    result.reserve(st.tensors.size());

    for (size_t i = 0; i < st.tensors.size(); ++i) {
        std::string name;
        safetensors::tensor_t ti;
        // ordered_dict::keys()[i] gives the key; use at() to get value
        name = st.tensors.keys()[i];
        st.tensors.at(i, &ti);

        // Build shape
        std::vector<int64_t> shape;
        shape.reserve(ti.shape.size());
        for (auto s : ti.shape) shape.push_back(static_cast<int64_t>(s));

        size_t n = safetensors::get_shape_size(ti);
        const uint8_t* raw = databuf + ti.data_offsets[0];

        DType out_dtype = st_dtype_to_tinyllama(ti.dtype);
        Tensor t(shape, out_dtype, Device::CPU);

        if (out_dtype == DType::Float32) {
            copy_to_f32(t.data_ptr<float>(), raw, n, ti.dtype);
        } else {  // Int32
            copy_to_i32(t.data_ptr<int32_t>(), raw, n, ti.dtype);
        }

        result.emplace(std::move(name), std::move(t));
    }
    return result;
}

}  // namespace tinyllama
