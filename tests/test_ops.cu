// tests/test_ops.cu — 核心算子正确性验证
//
// 编译：通过 CMake（tests/CMakeLists.txt）
// 每个测试独立运行，通过 PASS/FAIL 打印结果。

#include "tinyllama/ops.h"
#include "tinyllama/tensor.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>

using namespace tinyllama;
using namespace tinyllama::ops;

// ── 辅助 ──────────────────────────────────────────────────────────────────────

static bool approx_eq(float a, float b, float tol = 1e-4f) {
    return fabsf(a - b) <= tol + 1e-6f * fabsf(b);
}

// 从 CPU vector 创建 CUDA Tensor
static Tensor make_cuda(const std::vector<float>& data,
                         std::vector<int64_t> shape) {
    Tensor cpu(shape, DType::Float32, Device::CPU);
    float* p = cpu.data_ptr<float>();
    for (size_t i = 0; i < data.size(); ++i) p[i] = data[i];
    return cpu.cuda();
}

// 把 CUDA Tensor 读回 CPU vector
static std::vector<float> to_vec(const Tensor& t) {
    Tensor cpu = t.cpu();
    const float* p = cpu.data_ptr<float>();
    return std::vector<float>(p, p + cpu.numel());
}

// ── Tests ─────────────────────────────────────────────────────────────────────

static bool test_softmax() {
    // 输入 [1, 4]，期望结果为标准 softmax
    Tensor x = make_cuda({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4});
    Tensor y = softmax(x);
    auto v = to_vec(y);
    // sum should be 1
    float s = v[0] + v[1] + v[2] + v[3];
    // 值应单调递增
    bool ok = approx_eq(s, 1.0f) && v[0] < v[1] && v[1] < v[2] && v[2] < v[3];
    printf("[%s] softmax  (sum=%.6f)\n", ok ? "PASS" : "FAIL", s);
    return ok;
}

static bool test_rms_norm() {
    // x = [1,2,3,4], weight = [1,1,1,1]
    Tensor x = make_cuda({1.f, 2.f, 3.f, 4.f}, {1, 4});
    Tensor w = make_cuda({1.f, 1.f, 1.f, 1.f}, {4});
    Tensor y = rms_norm(x, w, 1e-5f);
    auto v = to_vec(y);
    // rms = sqrt((1+4+9+16)/4) = sqrt(7.5)
    float rms = sqrtf(7.5f);
    bool ok = true;
    float ref[4] = {1/rms, 2/rms, 3/rms, 4/rms};
    for (int i = 0; i < 4; ++i) ok &= approx_eq(v[i], ref[i]);
    printf("[%s] rms_norm\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_matmul() {
    // 2x3 @ 3x2 = 2x2
    Tensor A = make_cuda({1,2,3, 4,5,6}, {2,3});
    Tensor B = make_cuda({7,8, 9,10, 11,12}, {3,2});
    Tensor C = matmul(A, B);
    auto v = to_vec(C);
    // [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
    // = [58, 64, 139, 154]
    bool ok = approx_eq(v[0],58) && approx_eq(v[1],64)
           && approx_eq(v[2],139) && approx_eq(v[3],154);
    printf("[%s] matmul  [58,64,139,154] got [%.1f,%.1f,%.1f,%.1f]\n",
           ok?"PASS":"FAIL", v[0],v[1],v[2],v[3]);
    return ok;
}

static bool test_silu() {
    Tensor x = make_cuda({0.f, 1.f, -1.f, 2.f}, {4});
    Tensor y = silu(x);
    auto v = to_vec(y);
    // silu(0)=0, silu(1)=1/(1+e^-1)*1≈0.7311, silu(-1)≈-0.2689, silu(2)≈1.7616
    float ref[4] = {0.f, 0.7310586f, -0.2689414f, 1.7615942f};
    bool ok = true;
    for (int i = 0; i < 4; ++i) ok &= approx_eq(v[i], ref[i]);
    printf("[%s] silu\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_rope() {
    // 构造 q, k [1, 1, 2, 4]（batch=1, seq=1, heads=2, head_dim=4）
    // 应用 RoPE 后检验 x0'^2 + x1'^2 == x0^2 + x1^2（旋转保模）
    std::vector<float> data(16);
    for (int i = 0; i < 16; ++i) data[i] = (float)(i + 1);
    Tensor q = make_cuda(data, {1, 1, 2, 4});
    Tensor k = make_cuda(data, {1, 1, 2, 4});
    std::vector<float> q_orig = data;
    rope_(q, k, 0);
    auto qv = to_vec(q);
    // 检验旋转保模：对每个 pair (i, i+2) 验证 norm 不变
    bool ok = true;
    for (int h = 0; h < 2; ++h) {
        for (int i = 0; i < 2; ++i) {
            int base = h * 4 + i;
            float orig_norm = q_orig[base]*q_orig[base] + q_orig[base+2]*q_orig[base+2];
            float new_norm  = qv[base]*qv[base]         + qv[base+2]*qv[base+2];
            ok &= approx_eq(orig_norm, new_norm, 1e-3f);
        }
    }
    printf("[%s] rope (rotation preserves norm)\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_embedding() {
    // vocab=4, d_model=3
    Tensor table = make_cuda({
        1,2,3,   4,5,6,   7,8,9,   10,11,12
    }, {4, 3});
    // CPU ids
    Tensor ids_cpu({2}, DType::Int32, Device::CPU);
    ids_cpu.data_ptr<int32_t>()[0] = 2;
    ids_cpu.data_ptr<int32_t>()[1] = 0;
    Tensor ids = ids_cpu.cuda();
    Tensor out = embedding(table, ids);
    auto v = to_vec(out);
    // ids[0]=2 → row[2]=[7,8,9]; ids[1]=0 → row[0]=[1,2,3]
    bool ok = approx_eq(v[0],7) && approx_eq(v[1],8) && approx_eq(v[2],9)
           && approx_eq(v[3],1) && approx_eq(v[4],2) && approx_eq(v[5],3);
    printf("[%s] embedding\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_mha_shape() {
    // 仅验证 MHA 输出 shape 正确，不验证数值
    int64_t batch = 1, seq = 4, d_model = 8, n_heads = 2;
    Tensor x = make_cuda(std::vector<float>(batch*seq*d_model, 0.1f),
                          {batch, seq, d_model});
    auto rand_w = [&](){ return make_cuda(std::vector<float>(d_model*d_model,0.01f),
                                          {d_model, d_model}); };
    auto rand_b = [&](){ return make_cuda(std::vector<float>(d_model,0.f), {d_model}); };
    Tensor wq=rand_w(), bq=rand_b();
    Tensor wk=rand_w(), bk=rand_b();
    Tensor wv=rand_w(), bv=rand_b();
    Tensor wo=rand_w(), bo=rand_b();

    Tensor out = mha(x, wq,bq, wk,bk, wv,bv, wo,bo, n_heads);
    bool ok = (out.dim(0)==batch && out.dim(1)==seq && out.dim(2)==d_model);
    printf("[%s] mha output shape [%ld,%ld,%ld]\n",
           ok?"PASS":"FAIL", (long)out.dim(0),(long)out.dim(1),(long)out.dim(2));
    return ok;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    printf("=== tiny-llama CUDA ops tests ===\n\n");
    int pass = 0, total = 0;
#define RUN(t) do { ++total; if(t()) ++pass; } while(0)
    RUN(test_softmax);
    RUN(test_rms_norm);
    RUN(test_matmul);
    RUN(test_silu);
    RUN(test_rope);
    RUN(test_embedding);
    RUN(test_mha_shape);
#undef RUN
    printf("\n%d / %d tests passed.\n", pass, total);
    return (pass == total) ? 0 : 1;
}
