#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <memory>
#include <string>

// cuda_runtime.h 只在 nvcc 编译时可见；
// clangd / 普通 C++ 编译器走 CPU-only 路径。
#ifdef __CUDACC__
#  include <cuda_runtime.h>
#endif

namespace tinyllama {

// ─────────────────────────────────────────────────────────
//  CUDA error-check helper（仅在 nvcc 编译时展开）
// ─────────────────────────────────────────────────────────
#ifdef __CUDACC__
#  define CUDA_CHECK(expr)                                                \
     do {                                                                  \
         cudaError_t _e = (expr);                                          \
         if (_e != cudaSuccess) {                                           \
             throw std::runtime_error(std::string("[CUDA] ") +             \
                 cudaGetErrorString(_e) + " at " __FILE__ ":" +            \
                 std::to_string(__LINE__));                                 \
         }                                                                 \
     } while (0)
#else
#  define CUDA_CHECK(expr) (void)(expr)
#endif

// ─────────────────────────────────────────────────────────
//  DType / Device
// ─────────────────────────────────────────────────────────
enum class DType  { Float32, Float16, Int8, Int32 };
enum class Device { CPU, CUDA };

inline size_t dtype_size(DType d) {
    switch (d) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::Int8:    return 1;
        case DType::Int32:   return 4;
    }
    throw std::runtime_error("Unknown dtype");
}

// ─────────────────────────────────────────────────────────
//  Helpers implemented in tensor.cu
//  （声明在此，定义由 nvcc 编译的 tensor.cu 提供）
// ─────────────────────────────────────────────────────────
std::shared_ptr<void> cuda_alloc(size_t bytes);
void cuda_memset_zero(void* ptr, size_t bytes);
void cuda_memcpy_h2d(void* dst, const void* src, size_t bytes);
void cuda_memcpy_d2h(void* dst, const void* src, size_t bytes);
void cuda_memcpy_d2d(void* dst, const void* src, size_t bytes);

// ─────────────────────────────────────────────────────────
//  Tensor
//  CPU 和 CUDA 均可存储；LLM 算子在 CUDA Tensor 上运行。
// ─────────────────────────────────────────────────────────
class Tensor {
public:
    Tensor() = default;

    explicit Tensor(std::vector<int64_t> shape,
                    DType  dtype  = DType::Float32,
                    Device device = Device::CUDA)
        : shape_(std::move(shape)), dtype_(dtype), device_(device)
    {
        compute_strides();
        size_t bytes = static_cast<size_t>(numel()) * dtype_size(dtype_);
        if (bytes == 0) return;

        if (device_ == Device::CUDA) {
            cuda_storage_ = cuda_alloc(bytes);        // 调 tensor.cu
            cuda_memset_zero(cuda_storage_.get(), bytes);
            data_ = static_cast<uint8_t*>(cuda_storage_.get());
        } else {
            cpu_storage_.assign(bytes, 0);
            data_ = cpu_storage_.data();
        }
    }

    // 非 owning view（指向外部 device 内存）
    Tensor(void* data, std::vector<int64_t> shape,
           std::vector<int64_t> strides, DType dtype, Device device)
        : shape_(std::move(shape)), strides_(std::move(strides)),
          dtype_(dtype), device_(device),
          data_(static_cast<uint8_t*>(data)), owns_data_(false)
    {}

    // ── Metadata ──────────────────────────────────────────

    int64_t ndim()   const { return static_cast<int64_t>(shape_.size()); }
    const std::vector<int64_t>& shape()   const { return shape_;   }
    const std::vector<int64_t>& strides() const { return strides_; }
    DType   dtype()  const { return dtype_;  }
    Device  device() const { return device_; }

    int64_t dim(int d) const {
        if (d < 0) d += ndim();
        return shape_[d];
    }

    int64_t numel() const {
        if (shape_.empty()) return 0;
        int64_t n = 1;
        for (auto s : shape_) n *= s;
        return n;
    }

    // ── Data pointer（device 指针在 GPU 上，不可在 CPU 直接解引用）──

    template<typename T = float>
    T* data_ptr() { return reinterpret_cast<T*>(data_); }

    template<typename T = float>
    const T* data_ptr() const { return reinterpret_cast<const T*>(data_); }

    // ── View / reshape（零拷贝）────────────────────────────

    Tensor view(std::vector<int64_t> new_shape) const {
        assert(is_contiguous());
        int64_t total = numel(), neg_idx = -1, inferred = 1;
        for (int i = 0; i < (int)new_shape.size(); ++i) {
            if (new_shape[i] == -1) neg_idx = i;
            else                    inferred *= new_shape[i];
        }
        if (neg_idx >= 0) new_shape[neg_idx] = total / inferred;

        Tensor out;
        out.shape_        = new_shape;
        out.dtype_        = dtype_;
        out.device_       = device_;
        out.data_         = data_;
        out.owns_data_    = false;
        out.cuda_storage_ = cuda_storage_;
        out.cpu_storage_  = cpu_storage_;
        out.compute_strides();
        return out;
    }

    Tensor transpose(int dim0, int dim1) const {
        if (dim0 < 0) dim0 += ndim();
        if (dim1 < 0) dim1 += ndim();
        Tensor out;
        out.shape_        = shape_;
        out.strides_      = strides_;
        out.dtype_        = dtype_;
        out.device_       = device_;
        out.data_         = data_;
        out.owns_data_    = false;
        out.cuda_storage_ = cuda_storage_;
        out.cpu_storage_  = cpu_storage_;
        std::swap(out.shape_[dim0],   out.shape_[dim1]);
        std::swap(out.strides_[dim0], out.strides_[dim1]);
        return out;
    }

    bool is_contiguous() const {
        int64_t expected = 1;
        for (int i = ndim() - 1; i >= 0; --i) {
            if (strides_[i] != expected) return false;
            expected *= shape_[i];
        }
        return true;
    }

    Tensor contiguous() const {
        if (is_contiguous()) return *this;
        Tensor out(shape_, dtype_, device_);
        size_t bytes = static_cast<size_t>(numel()) * dtype_size(dtype_);
        if (device_ == Device::CUDA)
            cuda_memcpy_d2d(out.data_, data_, bytes);
        else
            std::memcpy(out.data_, data_, bytes);
        return out;
    }

    // ── Host ↔ Device 拷贝 ────────────────────────────────

    // 把 host buffer 拷进此 CUDA tensor
    void copy_from_cpu(const void* host_ptr, size_t bytes) {
        assert(device_ == Device::CUDA);
        cuda_memcpy_h2d(data_, host_ptr, bytes);
    }

    // 把此 CUDA tensor 拷出到 host buffer
    void copy_to_cpu(void* host_ptr, size_t bytes) const {
        assert(device_ == Device::CUDA);
        cuda_memcpy_d2h(host_ptr, data_, bytes);
    }

    // 返回 CPU 副本
    Tensor cpu() const {
        Tensor out(shape_, dtype_, Device::CPU);
        size_t bytes = static_cast<size_t>(numel()) * dtype_size(dtype_);
        if (device_ == Device::CUDA) cuda_memcpy_d2h(out.data_, data_, bytes);
        else                         std::memcpy(out.data_, data_, bytes);
        return out;
    }

    // 返回 CUDA 副本
    Tensor cuda() const {
        Tensor out(shape_, dtype_, Device::CUDA);
        size_t bytes = static_cast<size_t>(numel()) * dtype_size(dtype_);
        if (device_ == Device::CPU)   cuda_memcpy_h2d(out.data_, data_, bytes);
        else                          cuda_memcpy_d2d(out.data_, data_, bytes);
        return out;
    }

    // ── Fill ─────────────────────────────────────────────

    void fill_zero() {
        size_t bytes = static_cast<size_t>(numel()) * dtype_size(dtype_);
        if (device_ == Device::CUDA) cuda_memset_zero(data_, bytes);
        else                         std::memset(data_, 0, bytes);
    }

    template<typename T = float>
    void fill_cpu(T val) {
        assert(device_ == Device::CPU);
        T* ptr = data_ptr<T>();
        for (int64_t i = 0; i < numel(); ++i) ptr[i] = val;
    }

    // ── Debug ─────────────────────────────────────────────

    void print_shape() const {
        printf("Tensor(");
        for (size_t i = 0; i < shape_.size(); ++i) {
            printf("%ld", (long)shape_[i]);
            if (i + 1 < shape_.size()) printf(", ");
        }
        printf(") dtype=%d device=%s\n",
               (int)dtype_,
               device_ == Device::CUDA ? "cuda" : "cpu");
    }

private:
    std::vector<int64_t>  shape_;
    std::vector<int64_t>  strides_;
    DType                 dtype_     = DType::Float32;
    Device                device_    = Device::CUDA;
    uint8_t*              data_      = nullptr;
    bool                  owns_data_ = true;

    std::shared_ptr<void> cuda_storage_;   // device 内存（shared_ptr 管理生命期）
    std::vector<uint8_t>  cpu_storage_;    // host 内存

    void compute_strides() {
        strides_.resize(shape_.size());
        int64_t s = 1;
        for (int i = (int)shape_.size() - 1; i >= 0; --i) {
            strides_[i] = s;
            s *= shape_[i];
        }
    }
};

}  // namespace tinyllama
