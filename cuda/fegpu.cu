// flashembed GPU engine: native CUDA forward passes for EmbeddingGemma 2 (text backbone + Gemma 4 vision tower) and
// NeoMME-260M-Retriever (text + document images). Called from Rust (src/gpu.rs) through the C API at the bottom.
//
// Design (docs/GPU.md):
//  * varlen packing: a batch is a flat list of tokens with cu_seqlens; no padding anywhere, every GEMM and kernel runs
//    on exactly the real tokens;
//  * cuBLAS GEMMs (true FP32, or BF16 with FP32 accumulation) on fused weights (QKV(+gate), gate|up);
//  * everything between GEMMs is one fused kernel per step (norm + residual + next norm, QK-norm + RoPE in place,
//    gated activations, PLE gating) with FP32 math and an FP32 residual stream;
//  * attention is a fused flash-style kernel (online softmax, GQA, symmetric sliding window with key-tile skipping,
//    NeoMME's exclusive-attention + sigmoid gate as epilogue); BF16 runs on tensor cores (mma.sync m16n8k16);
//  * all activation buffers come from one arena sized at build time (no allocation while running).
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <type_traits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using bf16 = __nv_bfloat16;

namespace {

std::string g_err;
struct FeErr : std::runtime_error {
  using std::runtime_error::runtime_error;
};
#define CK(x)                                                                                   \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) throw FeErr(std::string(#x) + ": " + cudaGetErrorString(e_));        \
  } while (0)
#define CB(x)                                                                                   \
  do {                                                                                          \
    cublasStatus_t s_ = (x);                                                                    \
    if (s_ != CUBLAS_STATUS_SUCCESS) throw FeErr(std::string(#x) + ": cublas " + std::to_string((int)s_)); \
  } while (0)
#define REQ(c, msg)                         \
  do {                                      \
    if (!(c)) throw FeErr(std::string(msg)); \
  } while (0)

enum { DT_F32 = 0, DT_BF16 = 1 };
enum { ARCH_GEMMA = 0, ARCH_NEOMME = 1 };

// ------------------------------------------------------------------------------------------------- device helpers
__device__ __forceinline__ float tof(float x) { return x; }
__device__ __forceinline__ float tof(bf16 x) { return __bfloat162float(x); }
template <typename T>
__device__ __forceinline__ T fromf(float x);
template <>
__device__ __forceinline__ float fromf<float>(float x) { return x; }
template <>
__device__ __forceinline__ bf16 fromf<bf16>(float x) { return __float2bfloat16(x); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
// sum over the block (blockDim.x multiple of 32); safe to call repeatedly
__device__ float block_sum(float v) {
  __shared__ float sh[33];
  v = warp_sum(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  if (w == 0) {
    float x = l < nw ? sh[l] : 0.f;
    x = warp_sum(x);
    if (l == 0) sh[32] = x;
  }
  __syncthreads();
  return sh[32];
}
__device__ __forceinline__ float gelu_tanh(float x) {
  const float k = 0.7978845608028654f;  // sqrt(2/pi)
  return 0.5f * x * (1.f + tanhf(k * (x + 0.044715f * x * x * x)));
}
__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.f + erff(x * 0.7071067811865476f)); }

// ------------------------------------------------------------------------------------------------- small kernels
__global__ void k_bf16_to_f32(const bf16* a, float* b, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    b[i] = __bfloat162float(a[i]);
}

// Residual / norm step for one token row (block per row, n = hidden size):
//   x_mode 1: h += rms(x) * w_post      x_mode 2: h += xmul * x      x_mode 3: h = x
//   h *= hscale
//   z = init ? la * h + lb * init : h
//   y_mode 1: y = rms(z) * (w_pre or 1)  y_mode 2: y = z      (y written as T, and/or y32 as float)
template <typename T>
__global__ void k_resid_norm(int n, float eps, const T* __restrict__ x, int x_mode, const float* __restrict__ w_post,
                             float xmul, float* __restrict__ h, float hscale, const float* __restrict__ init, float la,
                             float lb, int y_mode, const float* __restrict__ w_pre, T* __restrict__ y,
                             float* __restrict__ y32) {
  constexpr int MAXV = 8;  // n <= 8 * blockDim
  const size_t row = blockIdx.x;
  float v[MAXV];
  int cnt = 0;
  for (int c = threadIdx.x; c < n; c += blockDim.x) v[cnt++] = h ? h[row * n + c] : 0.f;
  if (x_mode) {
    float xv[MAXV];
    int k = 0;
    float ss = 0.f;
    for (int c = threadIdx.x; c < n; c += blockDim.x, ++k) {
      xv[k] = tof(x[row * n + c]);
      ss += xv[k] * xv[k];
    }
    float r = 1.f;
    if (x_mode == 1) r = 1.f / sqrtf(block_sum(ss) / n + eps);
    k = 0;
    for (int c = threadIdx.x; c < n; c += blockDim.x, ++k) {
      if (x_mode == 1) v[k] += xv[k] * r * w_post[c];
      else if (x_mode == 2) v[k] += xmul * xv[k];
      else v[k] = xv[k];
    }
  }
  int k = 0;
  for (int c = threadIdx.x; c < n; c += blockDim.x, ++k) {
    v[k] *= hscale;
    if (h) h[row * n + c] = v[k];
  }
  if (!y_mode) return;
  float ss = 0.f;
  k = 0;
  for (int c = threadIdx.x; c < n; c += blockDim.x, ++k) {
    if (init) v[k] = la * v[k] + lb * init[row * n + c];
    ss += v[k] * v[k];
  }
  float r = 1.f;
  if (y_mode == 1) r = 1.f / sqrtf(block_sum(ss) / n + eps);
  k = 0;
  for (int c = threadIdx.x; c < n; c += blockDim.x, ++k) {
    float o = v[k] * r;
    if (y_mode == 1 && w_pre) o *= w_pre[c];
    if (y) y[row * n + c] = fromf<T>(o);
    if (y32) y32[row * n + c] = o;
  }
}

// ---- vectorised row helpers: 4 consecutive elements per access
__device__ __forceinline__ void ld4(const float* p, float* o) {
  const float4 v = *(const float4*)p;
  o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w;
}
__device__ __forceinline__ void ld4(const bf16* p, float* o) {
  const uint2 u = *(const uint2*)p;
  const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&u.x), b = *reinterpret_cast<const __nv_bfloat162*>(&u.y);
  o[0] = __low2float(a); o[1] = __high2float(a); o[2] = __low2float(b); o[3] = __high2float(b);
}
__device__ __forceinline__ void st4(float* p, const float* o) { *(float4*)p = make_float4(o[0], o[1], o[2], o[3]); }
__device__ __forceinline__ void st4(bf16* p, const float* o) {
  __nv_bfloat162 a = __floats2bfloat162_rn(o[0], o[1]), b = __floats2bfloat162_rn(o[2], o[3]);
  uint2 u;
  u.x = *reinterpret_cast<uint32_t*>(&a);
  u.y = *reinterpret_cast<uint32_t*>(&b);
  *(uint2*)p = u;
}

// Warp-per-row version of k_resid_norm (8 rows per 256-thread block, NV x 4 elements per lane, n = 128 * NV).
template <typename T, int NV>
__global__ void __launch_bounds__(256) k_resid_norm_w(int ntok, int n, float eps, const T* __restrict__ x, int x_mode,
                                                      const float* __restrict__ w_post, float xmul, float* __restrict__ h,
                                                      float hscale, const float* __restrict__ init, float la, float lb,
                                                      int y_mode, const float* __restrict__ w_pre, T* __restrict__ y,
                                                      float* __restrict__ y32) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const size_t row = (size_t)blockIdx.x * 8 + warp;
  if (row >= (size_t)ntok) return;
  float v[NV][4];
#pragma unroll
  for (int i = 0; i < NV; ++i) {
    const int c = (lane + 32 * i) * 4;
    if (h) ld4(h + row * n + c, v[i]);
    else v[i][0] = v[i][1] = v[i][2] = v[i][3] = 0.f;
  }
  if (x_mode) {
    float xv[NV][4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < NV; ++i) {
      ld4(x + row * n + (lane + 32 * i) * 4, xv[i]);
#pragma unroll
      for (int j = 0; j < 4; ++j) ss += xv[i][j] * xv[i][j];
    }
    const float r = x_mode == 1 ? 1.f / sqrtf(warp_sum(ss) / n + eps) : 1.f;
#pragma unroll
    for (int i = 0; i < NV; ++i) {
      const int c = (lane + 32 * i) * 4;
      float w[4] = {1.f, 1.f, 1.f, 1.f};
      if (x_mode == 1) ld4(w_post + c, w);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (x_mode == 1) v[i][j] += xv[i][j] * r * w[j];
        else if (x_mode == 2) v[i][j] += xmul * xv[i][j];
        else v[i][j] = xv[i][j];
      }
    }
  }
#pragma unroll
  for (int i = 0; i < NV; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) v[i][j] *= hscale;
    if (h) st4(h + row * n + (lane + 32 * i) * 4, v[i]);
  }
  if (!y_mode) return;
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < NV; ++i) {
    if (init) {
      float b[4];
      ld4(init + row * n + (lane + 32 * i) * 4, b);
#pragma unroll
      for (int j = 0; j < 4; ++j) v[i][j] = la * v[i][j] + lb * b[j];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) ss += v[i][j] * v[i][j];
  }
  const float r = y_mode == 1 ? 1.f / sqrtf(warp_sum(ss) / n + eps) : 1.f;
#pragma unroll
  for (int i = 0; i < NV; ++i) {
    const int c = (lane + 32 * i) * 4;
    float w[4] = {1.f, 1.f, 1.f, 1.f};
    if (y_mode == 1 && w_pre) ld4(w_pre + c, w);
    float o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = v[i][j] * r * w[j];
    if (y) st4(y + row * n + c, o);
    if (y32) st4(y32 + row * n + c, o);
  }
}

// out[t, c] = gelu_tanh(gu[t, c]) * gu[t, I + c], 4 elements per thread
template <typename T>
__global__ void k_gelu_gate4(const T* __restrict__ gu, T* __restrict__ out, int I, int ntok) {
  const int q = I / 4;
  const long long nch = (long long)ntok * q;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < nch; i += (long long)gridDim.x * blockDim.x) {
    const int t = (int)(i / q), c = (int)(i - (long long)t * q) * 4;
    float g[4], u[4], o[4];
    ld4(gu + (size_t)t * 2 * I + c, g);
    ld4(gu + (size_t)t * 2 * I + I + c, u);
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = gelu_tanh(g[j]) * u[j];
    st4(out + (size_t)t * I + c, o);
  }
}
template <typename T>
__global__ void k_relu2_4(T* __restrict__ x, long long nch) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < nch; i += (long long)gridDim.x * blockDim.x) {
    float v[4];
    ld4(x + i * 4, v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[j] = fmaxf(v[j], 0.f);
      v[j] *= v[j];
    }
    st4(x + i * 4, v);
  }
}
// PLE gate, warp per row (n = 128 * NV)
template <typename T, int NV>
__global__ void __launch_bounds__(256) k_ple_w(const T* __restrict__ g, const T* __restrict__ pp, float s,
                                               const float* __restrict__ w, float eps, T* __restrict__ z, int n, int ntok) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const size_t row = (size_t)blockIdx.x * 8 + warp;
  if (row >= (size_t)ntok) return;
  float v[NV][4], ss = 0.f;
#pragma unroll
  for (int i = 0; i < NV; ++i) {
    ld4(pp + row * n + (lane + 32 * i) * 4, v[i]);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[i][j] *= s;
      ss += v[i][j] * v[i][j];
    }
  }
  const float r = 1.f / sqrtf(warp_sum(ss) / n + eps);
#pragma unroll
  for (int i = 0; i < NV; ++i) {
    const int c = (lane + 32 * i) * 4;
    float gv[4], wv[4], o[4];
    ld4(g + row * n + c, gv);
    ld4(w + c, wv);
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = gelu_tanh(gv[j]) * v[i][j] * r * wv[j];
    st4(z + row * n + c, o);
  }
}

// RoPE description: rotated channels c < 2*half*parts are rotated in pairs (c, c +- half) inside their part
// (parts = 1, or 2 for the Gemma 4 vision axial RoPE); angle = pos[axis] * inv[k].
struct Rope {
  int type;  // 0 none, 1 standard 1-D (Gemma text), 2 NeoMME 2-axis interleaved (M-RoPE), 3 Gemma 4 vision axial
  int rot;   // rotated channels (type 1/2); per-part width for type 3 = D/2
  float inv[256];
};

__device__ __forceinline__ void rope_chan(const Rope& r, int c, int D, int& k, int& axis, int& partner) {
  k = -1;
  if (r.type == 1 || r.type == 2) {
    if (c >= r.rot) return;
    const int half = r.rot / 2;
    k = c % half;
    axis = r.type == 2 ? (k & 1) : 0;
    partner = c < half ? c + half : c - half;
  } else if (r.type == 3) {
    const int pw = r.rot, half = pw / 2, part = c / pw, cc = c % pw;
    k = cc % half;
    axis = part;
    partner = part * pw + (cc < half ? cc + half : cc - half);
  }
}

// In-place per-head QK(V)-norm + RoPE (+ NeoMME value embeddings) on the fused QKV GEMM output [ntok, ld].
// One warp per (token, head vector); head vectors 0..Hq-1 = q, then Hkv k, then Hkv v.
template <typename T, int D>
__global__ void k_qkv_prep(T* __restrict__ qkv, int ld, int ntok, int Hq, int Hkv, const float* __restrict__ qw,
                           const float* __restrict__ kw, int qk_norm, int v_norm, float eps, Rope rope,
                           const int* __restrict__ pos, int naxes, const bf16* __restrict__ vemb,
                           const int* __restrict__ ids) {
  constexpr int CPL = D / 32;
  __shared__ float sm[8][D];
  __shared__ float sinv[256];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // a per-lane index into the by-value Rope parameter would serialise on the constant cache: stage it in smem
  for (int i = threadIdx.x; i < 256; i += blockDim.x) sinv[i] = rope.inv[i];
  __syncthreads();
  const int t = blockIdx.x;
  const int hv = blockIdx.y * 8 + warp;
  if (t >= ntok || hv >= Hq + 2 * Hkv) return;
  const bool isq = hv < Hq, isk = !isq && hv < Hq + Hkv, isv = hv >= Hq + Hkv;
  T* p = qkv + (size_t)t * ld + (size_t)hv * D;
  float x[CPL];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < CPL; ++i) {
    x[i] = tof(p[lane + 32 * i]);
    ss += x[i] * x[i];
  }
  const bool norm = isv ? v_norm : qk_norm;
  if (norm) {
    const float r = 1.f / sqrtf(warp_sum(ss) / D + eps);
    const float* w = isq ? qw : (isk ? kw : nullptr);
#pragma unroll
    for (int i = 0; i < CPL; ++i) x[i] = x[i] * r * ((w && !isv) ? w[lane + 32 * i] : 1.f);
  }
  if (isv) {
    if (vemb) {
      const int kvh = hv - Hq - Hkv;
      const bf16* e = vemb + (size_t)ids[t] * Hkv * D + (size_t)kvh * D;
#pragma unroll
      for (int i = 0; i < CPL; ++i) x[i] += __bfloat162float(e[lane + 32 * i]);
    }
  } else if (rope.type) {
#pragma unroll
    for (int i = 0; i < CPL; ++i) sm[warp][lane + 32 * i] = x[i];
    __syncwarp();
#pragma unroll
    for (int i = 0; i < CPL; ++i) {
      const int c = lane + 32 * i;
      int k, axis = 0, partner = 0;
      rope_chan(rope, c, D, k, axis, partner);
      if (k < 0) continue;
      const float ang = (float)pos[(size_t)t * naxes + (axis < naxes ? axis : 0)] * sinv[k];
      float s, co;
      sincosf(ang, &s, &co);
      const float other = sm[warp][partner];
      x[i] = partner > c ? x[i] * co - other * s : x[i] * co + other * s;
    }
  }
#pragma unroll
  for (int i = 0; i < CPL; ++i) p[lane + 32 * i] = fromf<T>(x[i]);
}

// out[t, c] = gelu_tanh(gu[t, c]) * gu[t, I + c]
template <typename T>
__global__ void k_gelu_gate(const T* __restrict__ gu, T* __restrict__ out, int I, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const size_t t = i / I, c = i % I;
    out[i] = fromf<T>(gelu_tanh(tof(gu[t * 2 * I + c])) * tof(gu[t * 2 * I + I + c]));
  }
}
template <typename T>
__global__ void k_relu2(T* __restrict__ x, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const float v = fmaxf(tof(x[i]), 0.f);
    x[i] = fromf<T>(v * v);
  }
}
// EmbeddingGemma 2 PLE gate: z = gelu(g) * rms(pp * s) * w   (row per block, n = 512)
template <typename T>
__global__ void k_ple(const T* __restrict__ g, const T* __restrict__ pp, float s, const float* __restrict__ w, float eps,
                      T* __restrict__ z, int n) {
  const size_t row = blockIdx.x;
  float v[8];
  int k = 0;
  float ss = 0.f;
  for (int c = threadIdx.x; c < n; c += blockDim.x, ++k) {
    v[k] = tof(pp[row * n + c]) * s;
    ss += v[k] * v[k];
  }
  const float r = 1.f / sqrtf(block_sum(ss) / n + eps);
  k = 0;
  for (int c = threadIdx.x; c < n; c += blockDim.x, ++k)
    z[row * n + c] = fromf<T>(gelu_tanh(tof(g[row * n + c])) * v[k] * r * w[c]);
}

// Gemma token embeddings: h = table[id] * scale (fp32); multimodal rows take soft tokens; e = T(h) for the PLE GEMM.
template <typename T>
__global__ void k_gemma_embed(const int* __restrict__ ids, const bf16* __restrict__ table, float scale,
                              const int* __restrict__ mm_idx, const T* __restrict__ soft, float* __restrict__ h,
                              T* __restrict__ e, int n) {
  const size_t t = blockIdx.x;
  const int mi = mm_idx ? mm_idx[t] : -1;
  for (int c = threadIdx.x; c < n; c += blockDim.x) {
    const float v = mi >= 0 ? tof(soft[(size_t)mi * n + c]) : __bfloat162float(table[(size_t)ids[t] * n + c]) * scale;
    h[t * n + c] = v;
    e[t * n + c] = fromf<T>(v);
  }
}
// gather bf16 table rows into T
template <typename T>
__global__ void k_gather(const int* __restrict__ ids, const bf16* __restrict__ table, T* __restrict__ out, int n) {
  const size_t t = blockIdx.x;
  for (int c = threadIdx.x; c < n; c += blockDim.x) out[t * n + c] = fromf<T>(__bfloat162float(table[(size_t)ids[t] * n + c]));
}
// fp32 host patches -> T: (a + add) * mul (Gemma 4 vision: 2 * (p - 0.5), HF's order of operations)
template <typename T>
__global__ void k_affine(const float* __restrict__ a, T* __restrict__ b, float mul, float add, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    b[i] = fromf<T>(__fmul_rn(__fadd_rn(a[i], add), mul));
}
// vision: h = x + pos_table[0][px] + pos_table[1][py]
template <typename T>
__global__ void k_vis_pos(const T* __restrict__ x, const int* __restrict__ pos, const bf16* __restrict__ tab, int tabn,
                          float* __restrict__ h, int n) {
  const size_t t = blockIdx.x;
  const int px = pos[2 * t], py = pos[2 * t + 1];
  for (int c = threadIdx.x; c < n; c += blockDim.x)
    h[t * n + c] = tof(x[t * n + c]) + (__bfloat162float(tab[(size_t)px * n + c]) + __bfloat162float(tab[((size_t)tabn + py) * n + c]));
}
// vision pooler + embed_vision pre-norm: out[s] = rms(sum_{p in slot s} h[p] / k2 * root) (no weight)
template <typename T>
__global__ void k_vis_pool(const float* __restrict__ h, const int* __restrict__ slot_ptr, const int* __restrict__ slot_patch,
                           float inv_k2, float root, float eps, T* __restrict__ out, int n) {
  const int s = blockIdx.x;
  float v[8];
  int k = 0;
  float ss = 0.f;
  for (int c = threadIdx.x; c < n; c += blockDim.x, ++k) {
    float a = 0.f;
    for (int j = slot_ptr[s]; j < slot_ptr[s + 1]; ++j) a += h[(size_t)slot_patch[j] * n + c];
    v[k] = a * inv_k2 * root;
    ss += v[k] * v[k];
  }
  const float r = 1.f / sqrtf(block_sum(ss) / n + eps);
  k = 0;
  for (int c = threadIdx.x; c < n; c += blockDim.x, ++k) out[(size_t)s * n + c] = fromf<T>(v[k] * r);
}
// NeoMME patch stem: processor normalisation of uint8 patches ((p/255 - 0.5) / 0.5; padding 0 -> -1), then
// LayerNorm(3072, w, b) -> T. 12 values per thread (n = 3072, 256 threads).
template <typename T>
__global__ void k_layernorm_u8(const uint8_t* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                               float eps, T* __restrict__ y, int n) {
  constexpr int MAXV = 12;
  const size_t row = blockIdx.x;
  float v[MAXV];
  int k = 0;
  float s = 0.f;
  for (int c = threadIdx.x; c < n; c += blockDim.x, ++k) {
    v[k] = __fdiv_rn(__fsub_rn(__fmul_rn((float)x[row * n + c], 1.0f / 255.0f), 0.5f), 0.5f);
    s += v[k];
  }
  const float mean = block_sum(s) / n;
  float ss = 0.f;
  k = 0;
  for (int c = threadIdx.x; c < n; c += blockDim.x, ++k) {
    const float d = v[k] - mean;
    ss += d * d;
  }
  const float r = 1.f / sqrtf(block_sum(ss) / n + eps);
  k = 0;
  for (int c = threadIdx.x; c < n; c += blockDim.x, ++k) y[row * n + c] = fromf<T>((v[k] - mean) * r * w[c] + b[c]);
}

// ---- Gemma 4 image preprocessing on the GPU: Pillow/ATen uint8 antialiased bicubic resize (same integer algorithm and
// coefficients as src/pil_resize.rs, so bit-identical), then 16 px patches as (p/255 - 0.5) * 2 in T.
// Per image descriptor (ints): see ImgDesc in the host code.
enum { D_SRC, D_W, D_H, D_TW, D_TH, D_TMP, D_ROWS, D_YB0, D_CH, D_KSH, D_PH, D_CV, D_KSV, D_PV, D_PATCH, D_NEEDH, D_NEEDV,
       D_N = 20 };
__device__ __forceinline__ uint8_t clip8(int v, int prec) { v >>= prec; return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }
__global__ void k_resize_h(const int* __restrict__ desc, const uint8_t* __restrict__ src, const int* __restrict__ coef,
                           uint8_t* __restrict__ tmp) {
  const int* d = desc + blockIdx.y * D_N;
  if (!d[D_NEEDH]) return;
  const int tw = d[D_TW], rows = d[D_ROWS], w = d[D_W], ks = d[D_KSH], prec = d[D_PH];
  const int* bounds = coef + d[D_CH];
  const int* kk = bounds + 2 * tw;
  const uint8_t* s0 = src + d[D_SRC] + (size_t)d[D_YB0] * w * 3;
  uint8_t* t0 = tmp + d[D_TMP];
  const int half = 1 << (prec - 1);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * tw; i += gridDim.x * blockDim.x) {
    const int y = i / tw, x = i % tw, xmin = bounds[2 * x], n = bounds[2 * x + 1];
    const int* k = kk + x * ks;
    int a0 = half, a1 = half, a2 = half;
    const uint8_t* p = s0 + ((size_t)y * w + xmin) * 3;
    for (int j = 0; j < n; ++j) {
      a0 += p[3 * j] * k[j];
      a1 += p[3 * j + 1] * k[j];
      a2 += p[3 * j + 2] * k[j];
    }
    uint8_t* o = t0 + ((size_t)y * tw + x) * 3;
    o[0] = clip8(a0, prec);
    o[1] = clip8(a1, prec);
    o[2] = clip8(a2, prec);
  }
}
template <typename T>
__global__ void k_resize_v_patch(const int* __restrict__ desc, const uint8_t* __restrict__ src,
                                 const int* __restrict__ coef, const uint8_t* __restrict__ tmp, T* __restrict__ pix) {
  const int* d = desc + blockIdx.y * D_N;
  const int tw = d[D_TW], th = d[D_TH], ks = d[D_KSV], prec = d[D_PV], pw = tw / 16;
  const bool nh = d[D_NEEDH], nv = d[D_NEEDV];
  const uint8_t* img = nh ? tmp + d[D_TMP] : src + d[D_SRC];  // tw columns either way (no horizontal pass: tw == w)
  const int* bounds = coef + d[D_CV];
  const int* kk = bounds + 2 * th;
  const int yoff = nh ? d[D_YB0] : 0;  // tmp rows start at the first source row the vertical pass needs
  const int half = 1 << (prec - 1);
  T* out = pix + (size_t)d[D_PATCH] * 768;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < th * tw; i += gridDim.x * blockDim.x) {
    const int y = i / tw, x = i % tw;
    int v[3];
    if (nv) {
      const int ymin = bounds[2 * y] - yoff, n = bounds[2 * y + 1];
      const int* k = kk + y * ks;
      int a0 = half, a1 = half, a2 = half;
      for (int j = 0; j < n; ++j) {
        const uint8_t* p = img + ((size_t)(ymin + j) * tw + x) * 3;
        a0 += p[0] * k[j];
        a1 += p[1] * k[j];
        a2 += p[2] * k[j];
      }
      v[0] = clip8(a0, prec);
      v[1] = clip8(a1, prec);
      v[2] = clip8(a2, prec);
    } else {
      const uint8_t* p = img + ((size_t)y * tw + x) * 3;
      v[0] = p[0];
      v[1] = p[1];
      v[2] = p[2];
    }
    T* o = out + (size_t)((y / 16) * pw + x / 16) * 768 + ((y % 16) * 16 + x % 16) * 3;
#pragma unroll
    for (int c = 0; c < 3; ++c)  // explicit roundings: no FMA contraction, same bits as the CPU path
      o[c] = fromf<T>(__fmul_rn(__fsub_rn(__fmul_rn((float)v[c], 1.0f / 255.0f), 0.5f), 2.f));
  }
}
template <typename T>
__global__ void k_gelu_erf(T* __restrict__ x, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    x[i] = fromf<T>(gelu_erf(tof(x[i])));
}
// scatter patch-stem outputs (+ bias) into the token rows
template <typename T>
__global__ void k_scatter_rows(const T* __restrict__ src, const float* __restrict__ bias, const int* __restrict__ rows,
                               T* __restrict__ dst, int n) {
  const size_t i = blockIdx.x;
  const size_t r = rows[i];
  for (int c = threadIdx.x; c < n; c += blockDim.x) dst[r * n + c] = fromf<T>(tof(src[i * n + c]) + (bias ? bias[c] : 0.f));
}
// mean over each sequence's rows: out[s, c] (fp32)
__global__ void k_mean_pool(const float* __restrict__ x, const int* __restrict__ cu, float* __restrict__ out, int n) {
  const int s = blockIdx.x;
  const int a = cu[s], b = cu[s + 1];
  for (int c = threadIdx.x; c < n; c += blockDim.x) {
    float acc = 0.f;
    for (int t = a; t < b; ++t) acc += x[(size_t)t * n + c];
    out[(size_t)s * n + c] = acc / fmaxf((float)(b - a), 1e-9f);
  }
}
// in-place row L2 normalisation (F.normalize, eps 1e-12), T in, fp32 out
template <typename T>
__global__ void k_l2norm(const T* __restrict__ x, float* __restrict__ y, int n) {
  const size_t row = blockIdx.x;
  float ss = 0.f;
  for (int c = threadIdx.x; c < n; c += blockDim.x) {
    const float v = tof(x[row * n + c]);
    ss += v * v;
  }
  const float r = 1.f / fmaxf(sqrtf(block_sum(ss)), 1e-12f);
  for (int c = threadIdx.x; c < n; c += blockDim.x) y[row * n + c] = tof(x[row * n + c]) * r;
}

// ------------------------------------------------------------------------------------------------- attention
struct AttnArgs {
  const void* qkv;
  int ld, q_off, k_off, v_off, Hq, Hkv, window;
  float scale;
  const int* cu;
  const int2* tiles;  // (sequence, first query) per query tile
  void* out;
  int ld_out;
  const float* alpha_t;  // NeoMME exclusive attention: tanh(alpha) per head (null = off)
  int g_off;             // NeoMME sigmoid gate columns in qkv
};

// SIMT flash attention (FP32 math; T = storage type). Block = 4 warps, BR query rows x BC key tiles.
template <typename T, int D, int BR, int BC>
__global__ void __launch_bounds__(128) k_attn_simt(AttnArgs a) {
  constexpr int NW = 4, RW = BR / NW, G = BC < 32 ? BC : 32, KPL = BC / G, RPL = RW * G / 32, CPL = D / 32;
  extern __shared__ float sm[];
  float* Qs = sm;                  // [BR][D]
  float* Ks = Qs + BR * D;         // [BC][D+1]
  float* Vs = Ks + BC * (D + 1);   // [BC][D]
  float* Ps = Vs + BC * D;         // [BR][BC]
  const int2 tile = a.tiles[blockIdx.x];
  const int s = tile.x, q0 = tile.y, h = blockIdx.y, hk = h / (a.Hq / a.Hkv);
  const int start = a.cu[s], len = a.cu[s + 1] - start;
  const T* base = (const T*)a.qkv;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < BR * D; i += 128) {
    const int r = i / D, c = i % D, q = q0 + r;
    Qs[i] = q < len ? tof(base[(size_t)(start + q) * a.ld + a.q_off + h * D + c]) * a.scale : 0.f;
  }
  float acc[RW][CPL], m[RW], l[RW];
#pragma unroll
  for (int r = 0; r < RW; ++r) {
    m[r] = -INFINITY;
    l[r] = 0.f;
#pragma unroll
    for (int c = 0; c < CPL; ++c) acc[r][c] = 0.f;
  }
  int lo = 0, hi = len;
  if (a.window >= 0) {
    lo = max(0, q0 - a.window);
    hi = min(len, q0 + BR - 1 + a.window + 1);
  }
  for (int k0 = lo; k0 < hi; k0 += BC) {
    __syncthreads();
    for (int i = tid; i < BC * D; i += 128) {
      const int r = i / D, c = i % D, kk = k0 + r;
      float kv = 0.f, vv = 0.f;
      if (kk < hi) {
        const size_t row = (size_t)(start + kk) * a.ld;
        kv = tof(base[row + a.k_off + hk * D + c]);
        vv = tof(base[row + a.v_off + hk * D + c]);
      }
      Ks[r * (D + 1) + c] = kv;
      Vs[r * D + c] = vv;
    }
    __syncthreads();
    float sc[RPL][KPL];
#pragma unroll
    for (int k = 0; k < RPL; ++k)
#pragma unroll
      for (int j = 0; j < KPL; ++j) sc[k][j] = 0.f;
#pragma unroll 4
    for (int d = 0; d < D; ++d) {
      float kd[KPL];
#pragma unroll
      for (int j = 0; j < KPL; ++j) kd[j] = Ks[((lane % G) + 32 * j) * (D + 1) + d];
#pragma unroll
      for (int k = 0; k < RPL; ++k) {
        const float qd = Qs[(warp * RW + lane / G + (32 / G) * k) * D + d];
#pragma unroll
        for (int j = 0; j < KPL; ++j) sc[k][j] += qd * kd[j];
      }
    }
#pragma unroll
    for (int k = 0; k < RPL; ++k) {
      const int r = lane / G + (32 / G) * k, qi = q0 + warp * RW + r;
#pragma unroll
      for (int j = 0; j < KPL; ++j) {
        const int jj = (lane % G) + 32 * j, kj = k0 + jj;
        const bool ok = kj < hi && (a.window < 0 || abs(qi - kj) <= a.window);
        Ps[(warp * RW + r) * BC + jj] = ok ? sc[k][j] : -INFINITY;
      }
    }
    __syncwarp();
#pragma unroll
    for (int r = 0; r < RW; ++r) {
      float* row = Ps + (warp * RW + r) * BC;
      float mx = -INFINITY;
      for (int j = lane; j < BC; j += 32) mx = fmaxf(mx, row[j]);
      mx = warp_max(mx);
      const float mn = fmaxf(m[r], mx);
      const float corr = mn == -INFINITY ? 1.f : expf(m[r] - mn);
      float sum = 0.f;
      for (int j = lane; j < BC; j += 32) {
        const float p = row[j] == -INFINITY ? 0.f : expf(row[j] - mn);
        row[j] = p;
        sum += p;
      }
      sum = warp_sum(sum);
      l[r] = l[r] * corr + sum;
      m[r] = mn;
#pragma unroll
      for (int c = 0; c < CPL; ++c) acc[r][c] *= corr;
    }
    __syncwarp();
    for (int j = 0; j < BC; ++j) {
      float vv[CPL];
#pragma unroll
      for (int c = 0; c < CPL; ++c) vv[c] = Vs[j * D + lane + 32 * c];
#pragma unroll
      for (int r = 0; r < RW; ++r) {
        const float p = Ps[(warp * RW + r) * BC + j];
#pragma unroll
        for (int c = 0; c < CPL; ++c) acc[r][c] += p * vv[c];
      }
    }
  }
  T* out = (T*)a.out;
#pragma unroll
  for (int r = 0; r < RW; ++r) {
    const int qi = q0 + warp * RW + r;
    if (qi >= len) continue;
    const size_t row = (size_t)(start + qi);
    const float inv = 1.f / l[r];
    float o[CPL];
#pragma unroll
    for (int c = 0; c < CPL; ++c) o[c] = acc[r][c] * inv;
    if (a.alpha_t) {
      float v[CPL], vn = 0.f;
#pragma unroll
      for (int c = 0; c < CPL; ++c) {
        v[c] = tof(base[row * a.ld + a.v_off + hk * D + lane + 32 * c]);
        vn += v[c] * v[c];
      }
      const float nrm = fmaxf(sqrtf(warp_sum(vn)), 1e-12f);
      float dot = 0.f;
#pragma unroll
      for (int c = 0; c < CPL; ++c) {
        v[c] /= nrm;
        dot += o[c] * v[c];
      }
      dot = warp_sum(dot);
      const float ta = a.alpha_t[h];
#pragma unroll
      for (int c = 0; c < CPL; ++c) {
        const float g = tof(base[row * a.ld + a.g_off + h * D + lane + 32 * c]);
        o[c] = (o[c] - ta * dot * v[c]) * (1.f / (1.f + expf(-g)));
      }
    }
#pragma unroll
    for (int c = 0; c < CPL; ++c) out[row * a.ld_out + h * D + lane + 32 * c] = fromf<T>(o[c]);
  }
}

// ---- BF16 tensor-core flash attention (mma.sync m16n8k16, FP32 accumulate), FA2-style: 4 warps x 16 query rows,
// K/V tiles of BC keys in shared memory, S and O in registers, P re-packed from the S accumulators as the A operand.
// DQK = head dim of Q.K; DV = slice of the value/output dims this block computes (grid.z = D / DV, so D = 512 runs
// as two 256-wide halves that each recompute S, keeping the O accumulator at <= 128 registers).
__device__ __forceinline__ void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}
__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
  const uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
  const int n = pred ? 16 : 0;  // 0 bytes read -> zero fill
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <int DQK, int DV, int BC, int STAGES>
__global__ void __launch_bounds__(128) k_attn_mma(AttnArgs a) {
  constexpr int BR = 64, SQ = DQK + 8, SV = DV + 8, NT = BC / 8, NO = DV / 8;
  static_assert(NT % 2 == 0 && NO % 2 == 0, "tile shape");
  extern __shared__ __align__(16) unsigned char smraw[];
  bf16* Qs = (bf16*)smraw;                 // [BR][SQ]
  bf16* Ks0 = Qs + BR * SQ;                // STAGES x [BC][SQ]
  bf16* Vs0 = Ks0 + STAGES * BC * SQ;      // STAGES x [BC][SV]
  const int2 tile = a.tiles[blockIdx.x];
  const int s = tile.x, q0 = tile.y, h = blockIdx.y, dv0 = blockIdx.z * DV;
  const int hk = h / (a.Hq / a.Hkv);
  const int start = a.cu[s], len = a.cu[s + 1] - start;
  const bf16* base = (const bf16*)a.qkv;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  int lo = 0, hi = len;
  if (a.window >= 0) {
    lo = max(0, q0 - a.window);
    hi = min(len, q0 + BR - 1 + a.window + 1);
  }
  auto load_kv = [&](int stage, int k0) {
    bf16* Ks = Ks0 + stage * BC * SQ;
    bf16* Vs = Vs0 + stage * BC * SV;
    for (int i = tid; i < BC * DQK / 8; i += 128) {
      const int r = i / (DQK / 8), c = (i % (DQK / 8)) * 8, kk = k0 + r;
      const bool ok = kk < hi;
      cp_async16(Ks + r * SQ + c, base + (size_t)(start + (ok ? kk : 0)) * a.ld + a.k_off + hk * DQK + c, ok);
    }
    for (int i = tid; i < BC * DV / 8; i += 128) {
      const int r = i / (DV / 8), c = (i % (DV / 8)) * 8, kk = k0 + r;
      const bool ok = kk < hi;
      cp_async16(Vs + r * SV + c, base + (size_t)(start + (ok ? kk : 0)) * a.ld + a.v_off + hk * DQK + dv0 + c, ok);
    }
  };
  for (int i = tid; i < BR * DQK / 8; i += 128) {
    const int r = i / (DQK / 8), c = (i % (DQK / 8)) * 8, q = q0 + r;
    const bool ok = q < len;
    cp_async16(Qs + r * SQ + c, base + (size_t)(start + (ok ? q : 0)) * a.ld + a.q_off + h * DQK + c, ok);
  }
  load_kv(0, lo);
  cp_async_commit();
  float o[NO][4];
#pragma unroll
  for (int n = 0; n < NO; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float mrow0 = -INFINITY, mrow1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const float sl2 = a.scale * 1.4426950408889634f;
  const int qr0 = q0 + warp * 16 + g, qr1 = qr0 + 8;
  int it = 0;
  for (int k0 = lo; k0 < hi; k0 += BC, ++it) {
    const int st = STAGES == 2 ? (it & 1) : 0;
    if (STAGES == 2) {
      if (k0 + BC < hi) load_kv(st ^ 1, k0 + BC);  // prefetch the next tile while this one is used
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    const bf16* Ks = Ks0 + st * BC * SQ;
    const bf16* Vs = Vs0 + st * BC * SV;
    float sc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < DQK; kk += 16) {
      uint32_t a0, a1, a2, a3;
      ldsm4(a0, a1, a2, a3, Qs + (warp * 16 + (lane & 15)) * SQ + kk + (lane >> 4) * 8);
#pragma unroll
      for (int j = 0; j < NT; j += 2) {
        uint32_t b0, b1, b2, b3;
        ldsm4(b0, b1, b2, b3, Ks + (j * 8 + (lane >> 4) * 8 + (lane & 7)) * SQ + kk + ((lane >> 3) & 1) * 8);
        mma16816(sc[j], a0, a1, a2, a3, b0, b1);
        mma16816(sc[j + 1], a0, a1, a2, a3, b2, b3);
      }
    }
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int key = k0 + j * 8 + 2 * t4 + e;
        const bool ok0 = key < hi && (a.window < 0 || abs(qr0 - key) <= a.window);
        const bool ok1 = key < hi && (a.window < 0 || abs(qr1 - key) <= a.window);
        sc[j][e] = ok0 ? sc[j][e] * sl2 : -INFINITY;
        sc[j][2 + e] = ok1 ? sc[j][2 + e] * sl2 : -INFINITY;
        mx0 = fmaxf(mx0, sc[j][e]);
        mx1 = fmaxf(mx1, sc[j][2 + e]);
      }
    mx0 = quad_max(mx0);
    mx1 = quad_max(mx1);
    const float mn0 = fmaxf(mrow0, mx0), mn1 = fmaxf(mrow1, mx1);
    const float r0 = mn0 == -INFINITY ? 0.f : mn0, r1 = mn1 == -INFINITY ? 0.f : mn1;
    const float c0 = exp2f(mrow0 - r0), c1 = exp2f(mrow1 - r1);
    mrow0 = mn0;
    mrow1 = mn1;
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      sc[j][0] = exp2f(sc[j][0] - r0);
      sc[j][1] = exp2f(sc[j][1] - r0);
      sc[j][2] = exp2f(sc[j][2] - r1);
      sc[j][3] = exp2f(sc[j][3] - r1);
      s0 += sc[j][0] + sc[j][1];
      s1 += sc[j][2] + sc[j][3];
    }
    l0 = l0 * c0 + s0;
    l1 = l1 * c1 + s1;
#pragma unroll
    for (int n = 0; n < NO; ++n) {
      o[n][0] *= c0;
      o[n][1] *= c0;
      o[n][2] *= c1;
      o[n][3] *= c1;
    }
#pragma unroll
    for (int ks = 0; ks < BC / 16; ++ks) {
      const uint32_t p0 = pack_bf16(sc[2 * ks][0], sc[2 * ks][1]), p1 = pack_bf16(sc[2 * ks][2], sc[2 * ks][3]);
      const uint32_t p2 = pack_bf16(sc[2 * ks + 1][0], sc[2 * ks + 1][1]), p3 = pack_bf16(sc[2 * ks + 1][2], sc[2 * ks + 1][3]);
#pragma unroll
      for (int n = 0; n < NO; n += 2) {
        uint32_t b0, b1, b2, b3;
        ldsm4t(b0, b1, b2, b3, Vs + (ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * SV + n * 8 + (lane >> 4) * 8);
        mma16816(o[n], p0, p1, p2, p3, b0, b1);
        mma16816(o[n + 1], p0, p1, p2, p3, b2, b3);
      }
    }
    __syncthreads();  // everyone is done with this stage before it is refilled
    if (STAGES == 1 && k0 + BC < hi) {
      load_kv(0, k0 + BC);
      cp_async_commit();
    }
  }
  cp_async_wait<0>();
  l0 = quad_sum(l0);
  l1 = quad_sum(l1);
  const float i0 = 1.f / l0, i1 = 1.f / l1;
#pragma unroll
  for (int n = 0; n < NO; ++n) {
    o[n][0] *= i0;
    o[n][1] *= i0;
    o[n][2] *= i1;
    o[n][3] *= i1;
  }
  bf16* out = (bf16*)a.out;
  if (a.alpha_t) {  // NeoMME: exclusive self-attention against the token's own value, then the sigmoid gate (DV == DQK)
    const float ta = a.alpha_t[h];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int qr = half ? qr1 : qr0;
      const bool valid = qr < len;
      const size_t row = (size_t)(start + (valid ? qr : 0));
      float vn = 0.f, dot = 0.f;
      float vv[NO][2];
#pragma unroll
      for (int n = 0; n < NO; ++n) {
        const int col = n * 8 + 2 * t4;
        const __nv_bfloat162 v2 = *(const __nv_bfloat162*)(base + row * a.ld + a.v_off + hk * DQK + col);
        vv[n][0] = __bfloat162float(v2.x);
        vv[n][1] = __bfloat162float(v2.y);
        vn += vv[n][0] * vv[n][0] + vv[n][1] * vv[n][1];
        dot += o[n][2 * half] * vv[n][0] + o[n][2 * half + 1] * vv[n][1];
      }
      vn = quad_sum(vn);
      dot = quad_sum(dot);
      const float nrm = fmaxf(sqrtf(vn), 1e-12f), proj = dot / (nrm * nrm);
#pragma unroll
      for (int n = 0; n < NO; ++n) {
        const int col = n * 8 + 2 * t4;
        const __nv_bfloat162 g2 = *(const __nv_bfloat162*)(base + row * a.ld + a.g_off + h * DQK + col);
        const float x0 = (o[n][2 * half] - ta * proj * vv[n][0]) / (1.f + expf(-__bfloat162float(g2.x)));
        const float x1 = (o[n][2 * half + 1] - ta * proj * vv[n][1]) / (1.f + expf(-__bfloat162float(g2.y)));
        if (valid) *(uint32_t*)(out + row * a.ld_out + h * DQK + col) = pack_bf16(x0, x1);
      }
    }
    return;
  }
#pragma unroll
  for (int n = 0; n < NO; ++n) {
    const int col = h * DQK + dv0 + n * 8 + 2 * t4;
    if (qr0 < len) *(uint32_t*)(out + (size_t)(start + qr0) * a.ld_out + col) = pack_bf16(o[n][0], o[n][1]);
    if (qr1 < len) *(uint32_t*)(out + (size_t)(start + qr1) * a.ld_out + col) = pack_bf16(o[n][2], o[n][3]);
  }
}

// Head dim 512 (EmbeddingGemma 2 global layers) without recomputing S: 8 warps, warp w handles query rows
// 16 * (w % 4) .. +15 and dims half = w / 4. Each warp of a pair computes the partial Q.K over its 256 dims, the pair
// swaps partials through shared memory (identical full S in both, a + b == b + a), runs the same online softmax and
// multiplies P by its own 256-dim half of V. Needs ~216 KB of shared memory (B200-class GPUs).
template <int BC, int STAGES>
__global__ void __launch_bounds__(256) k_attn_mma512p(AttnArgs a) {
  constexpr int D = 512, DH = 256, BR = 64, SQ = D + 8, NT = BC / 8, NO = DH / 8;
  extern __shared__ __align__(16) unsigned char smraw[];
  bf16* Qs = (bf16*)smraw;                        // [BR][SQ]
  bf16* Ks0 = Qs + BR * SQ;                       // STAGES x [BC][SQ]
  bf16* Vs0 = Ks0 + STAGES * BC * SQ;             // STAGES x [BC][SQ]
  float* Sx = (float*)(Vs0 + STAGES * BC * SQ);   // [2 halves][4 row groups][32 lanes][NT*4]
  const int2 tile = a.tiles[blockIdx.x];
  const int s = tile.x, q0 = tile.y, h = blockIdx.y;
  const int hk = h / (a.Hq / a.Hkv);
  const int start = a.cu[s], len = a.cu[s + 1] - start;
  const bf16* base = (const bf16*)a.qkv;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int rg = warp & 3, half = warp >> 2, d0 = half * DH;
  int lo = 0, hi = len;
  if (a.window >= 0) {
    lo = max(0, q0 - a.window);
    hi = min(len, q0 + BR - 1 + a.window + 1);
  }
  auto load_kv = [&](int stage, int k0) {
    bf16* Ks = Ks0 + stage * BC * SQ;
    bf16* Vs = Vs0 + stage * BC * SQ;
    for (int i = tid; i < BC * D / 8; i += 256) {
      const int r = i / (D / 8), c = (i % (D / 8)) * 8, kk = k0 + r;
      const bool ok = kk < hi;
      const size_t row = (size_t)(start + (ok ? kk : 0)) * a.ld;
      cp_async16(Ks + r * SQ + c, base + row + a.k_off + hk * D + c, ok);
      cp_async16(Vs + r * SQ + c, base + row + a.v_off + hk * D + c, ok);
    }
  };
  for (int i = tid; i < BR * D / 8; i += 256) {
    const int r = i / (D / 8), c = (i % (D / 8)) * 8, q = q0 + r;
    const bool ok = q < len;
    cp_async16(Qs + r * SQ + c, base + (size_t)(start + (ok ? q : 0)) * a.ld + a.q_off + h * D + c, ok);
  }
  load_kv(0, lo);
  cp_async_commit();
  float o[NO][4];
#pragma unroll
  for (int n = 0; n < NO; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float mrow0 = -INFINITY, mrow1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const float sl2 = a.scale * 1.4426950408889634f;
  const int qr0 = q0 + rg * 16 + g, qr1 = qr0 + 8;
  float* mine = Sx + ((half * 4 + rg) * 32 + lane) * (NT * 4);
  const float* other = Sx + (((half ^ 1) * 4 + rg) * 32 + lane) * (NT * 4);
  int it = 0;
  for (int k0 = lo; k0 < hi; k0 += BC, ++it) {
    const int st = STAGES == 2 ? (it & 1) : 0;
    if (STAGES == 2) {
      if (k0 + BC < hi) load_kv(st ^ 1, k0 + BC);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    const bf16* Ks = Ks0 + st * BC * SQ;
    const bf16* Vs = Vs0 + st * BC * SQ;
    float sc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
#pragma unroll
    for (int kk = d0; kk < d0 + DH; kk += 16) {
      uint32_t a0, a1, a2, a3;
      ldsm4(a0, a1, a2, a3, Qs + (rg * 16 + (lane & 15)) * SQ + kk + (lane >> 4) * 8);
#pragma unroll
      for (int j = 0; j < NT; j += 2) {
        uint32_t b0, b1, b2, b3;
        ldsm4(b0, b1, b2, b3, Ks + (j * 8 + (lane >> 4) * 8 + (lane & 7)) * SQ + kk + ((lane >> 3) & 1) * 8);
        mma16816(sc[j], a0, a1, a2, a3, b0, b1);
        mma16816(sc[j + 1], a0, a1, a2, a3, b2, b3);
      }
    }
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) mine[j * 4 + e] = sc[j][e];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) sc[j][e] = half ? other[j * 4 + e] + sc[j][e] : sc[j][e] + other[j * 4 + e];
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int key = k0 + j * 8 + 2 * t4 + e;
        const bool ok0 = key < hi && (a.window < 0 || abs(qr0 - key) <= a.window);
        const bool ok1 = key < hi && (a.window < 0 || abs(qr1 - key) <= a.window);
        sc[j][e] = ok0 ? sc[j][e] * sl2 : -INFINITY;
        sc[j][2 + e] = ok1 ? sc[j][2 + e] * sl2 : -INFINITY;
        mx0 = fmaxf(mx0, sc[j][e]);
        mx1 = fmaxf(mx1, sc[j][2 + e]);
      }
    mx0 = quad_max(mx0);
    mx1 = quad_max(mx1);
    const float mn0 = fmaxf(mrow0, mx0), mn1 = fmaxf(mrow1, mx1);
    const float r0 = mn0 == -INFINITY ? 0.f : mn0, r1 = mn1 == -INFINITY ? 0.f : mn1;
    const float c0 = exp2f(mrow0 - r0), c1 = exp2f(mrow1 - r1);
    mrow0 = mn0;
    mrow1 = mn1;
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      sc[j][0] = exp2f(sc[j][0] - r0);
      sc[j][1] = exp2f(sc[j][1] - r0);
      sc[j][2] = exp2f(sc[j][2] - r1);
      sc[j][3] = exp2f(sc[j][3] - r1);
      s0 += sc[j][0] + sc[j][1];
      s1 += sc[j][2] + sc[j][3];
    }
    l0 = l0 * c0 + s0;
    l1 = l1 * c1 + s1;
#pragma unroll
    for (int n = 0; n < NO; ++n) {
      o[n][0] *= c0;
      o[n][1] *= c0;
      o[n][2] *= c1;
      o[n][3] *= c1;
    }
#pragma unroll
    for (int ks = 0; ks < BC / 16; ++ks) {
      const uint32_t p0 = pack_bf16(sc[2 * ks][0], sc[2 * ks][1]), p1 = pack_bf16(sc[2 * ks][2], sc[2 * ks][3]);
      const uint32_t p2 = pack_bf16(sc[2 * ks + 1][0], sc[2 * ks + 1][1]), p3 = pack_bf16(sc[2 * ks + 1][2], sc[2 * ks + 1][3]);
#pragma unroll
      for (int n = 0; n < NO; n += 2) {
        uint32_t b0, b1, b2, b3;
        ldsm4t(b0, b1, b2, b3, Vs + (ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * SQ + d0 + n * 8 + (lane >> 4) * 8);
        mma16816(o[n], p0, p1, p2, p3, b0, b1);
        mma16816(o[n + 1], p0, p1, p2, p3, b2, b3);
      }
    }
    __syncthreads();
    if (STAGES == 1 && k0 + BC < hi) {
      load_kv(0, k0 + BC);
      cp_async_commit();
    }
  }
  cp_async_wait<0>();
  l0 = quad_sum(l0);
  l1 = quad_sum(l1);
  const float i0 = 1.f / l0, i1 = 1.f / l1;
  bf16* out = (bf16*)a.out;
#pragma unroll
  for (int n = 0; n < NO; ++n) {
    const int col = h * D + d0 + n * 8 + 2 * t4;
    if (qr0 < len) *(uint32_t*)(out + (size_t)(start + qr0) * a.ld_out + col) = pack_bf16(o[n][0] * i0, o[n][1] * i0);
    if (qr1 < len) *(uint32_t*)(out + (size_t)(start + qr1) * a.ld_out + col) = pack_bf16(o[n][2] * i1, o[n][3] * i1);
  }
}

template <int D>
struct SimtCfg;
template <>
struct SimtCfg<64> {
  static constexpr int BR = 32, BC = 64;
};
template <>
struct SimtCfg<256> {
  static constexpr int BR = 16, BC = 32;
};
template <>
struct SimtCfg<512> {
  static constexpr int BR = 16, BC = 16;
};

// ------------------------------------------------------------------------------------------------- host side
struct DBuf {
  void* p = nullptr;
  size_t bytes = 0;
};

struct Lin {  // weight [n, k] row-major in the compute dtype
  void* w = nullptr;
  int n = 0, k = 0;
};

struct GLayer {  // EmbeddingGemma 2 text layer / Gemma 4 vision layer
  Lin qkv, o, gu, down, pg, pp;
  float *w_in = nullptr, *w_pa = nullptr, *w_pff = nullptr, *w_pf = nullptr, *w_qn = nullptr, *w_kn = nullptr,
        *w_ppn = nullptr;
  float ls = 1.f;
  int hd = 0, nkv = 0, window = -1;
  Rope rope{};
};

struct NLayer {  // NeoMME layer
  Lin qkvg, o, up, down;
  float la = 1.f, lb = 0.f;
  float* alpha_t = nullptr;
  int window = -1, vemb = 0;
  Rope rope{};
};

struct Engine {
  int arch = 0, dt = DT_BF16, dev = 0;
  cudaStream_t st = nullptr;
  cublasHandle_t cb = nullptr;
  void* ws = nullptr;
  std::unordered_map<std::string, DBuf> raw;           // uploaded bf16 tensors by name
  std::unordered_map<std::string, std::vector<int64_t>> shapes;
  std::unordered_map<std::string, double> cfg;
  std::vector<void*> owned;                             // everything to free
  size_t weight_bytes = 0, arena_bytes = 0;
  // model
  int H = 0, L = 0, Hq = 0, I = 0, ple = 0, out_dim = 0, vocab = 0;
  float eps = 1e-6f, embed_scale = 1.f, ple_scale = 1.f, resid_mult = 1.f;
  bf16* embed = nullptr;  // token table (bf16)
  bf16* vemb = nullptr;   // NeoMME value embeddings
  Lin ple_w;              // per-layer model projection [L*ple, H]
  float *w_ple_norm = nullptr, *w_final = nullptr;
  Lin proj_f32;           // final projection in fp32 (Gemma embedding_projection) / NeoMME late head (compute dtype)
  Lin late;
  Lin nemb_proj, nup, ndown;  // NeoMME embedding projection, patch stem
  float *n_ln_w = nullptr, *n_ln_b = nullptr, *n_down_b = nullptr;
  std::vector<GLayer> gl;
  std::vector<NLayer> nl;
  // vision
  bool has_vision = false;
  int VH = 0, VL = 0, VHq = 0, VI = 0, vpos_n = 0;
  float veps = 1e-6f;
  Lin vin, vev;
  bf16* vpos = nullptr;
  std::vector<GLayer> vl;
  // arena
  int max_tok = 0, max_patch = 0;
  char* arena = nullptr;
  size_t arena_off = 0;
  // pinned host staging for batch metadata / inputs
  char* pin = nullptr;
  size_t pin_bytes = 0;
  // per-buffer pointers (set by plan)
  void *b_y, *b_qkv, *b_o, *b_t1, *b_t2, *b_gu, *b_m, *b_e, *b_late;
  float *b_h, *b_init, *b_y32, *b_pool, *b_out;
  int* b_meta;
  size_t meta_ints = 0;
  // vision buffers
  void *v_pix, *v_y, *v_qkv, *v_o, *v_t1, *v_gu, *v_m, *v_ev, *v_soft;
  float *v_h, *v_patch_f32;
  // NeoMME patch buffers
  uint8_t* n_patch_u8 = nullptr;
  // Gemma GPU image preprocessing: decoded source images, horizontal-pass output, coefficients
  uint8_t *v_src = nullptr, *v_tmp = nullptr;
  int* v_coef = nullptr;
  size_t img_bytes = 0;
  void *n_ln = nullptr, *n_up = nullptr, *n_dn = nullptr;

  size_t esz() const { return dt == DT_F32 ? 4 : 2; }
};

void* dmalloc(Engine& e, size_t bytes) {
  void* p = nullptr;
  CK(cudaMalloc(&p, std::max<size_t>(bytes, 256)));
  e.owned.push_back(p);
  return p;
}

const DBuf& rawt(Engine& e, const std::string& n) {
  auto it = e.raw.find(n);
  if (it == e.raw.end()) throw FeErr("missing tensor " + n);
  return it->second;
}
bool has(Engine& e, const std::string& n) { return e.raw.count(n) > 0; }
double cfgv(Engine& e, const std::string& k, double def = NAN) {
  auto it = e.cfg.find(k);
  if (it == e.cfg.end()) {
    if (std::isnan(def)) throw FeErr("missing config " + k);
    return def;
  }
  return it->second;
}

float* to_f32(Engine& e, const std::string& n) {  // small vectors (norm weights, ...) -> fp32
  const DBuf& b = rawt(e, n);
  const size_t cnt = b.bytes / 2;
  float* p = (float*)dmalloc(e, cnt * 4);
  k_bf16_to_f32<<<(unsigned)std::min<size_t>((cnt + 255) / 256, 4096), 256, 0, e.st>>>((const bf16*)b.p, p, cnt);
  CK(cudaGetLastError());
  e.weight_bytes += cnt * 4;
  return p;
}
float host_scalar(Engine& e, const std::string& n, int i = 0) {
  const DBuf& b = rawt(e, n);
  bf16 v;
  CK(cudaMemcpy(&v, (const char*)b.p + 2 * i, 2, cudaMemcpyDeviceToHost));
  return __bfloat162float(v);
}
// concatenate [n_i, k] bf16 weights along n into one weight in the compute dtype
Lin make_lin(Engine& e, const std::vector<std::string>& names, int force_dt = -1) {
  const int dt = force_dt < 0 ? e.dt : force_dt;
  Lin L;
  for (auto& n : names) {
    auto& sh = e.shapes.at(n);
    REQ(sh.size() == 2, "weight not 2-D: " + n);
    if (L.k == 0) L.k = (int)sh[1];
    REQ(L.k == sh[1], "concat k mismatch: " + n);
    L.n += (int)sh[0];
  }
  const size_t es = dt == DT_F32 ? 4 : 2;
  L.w = dmalloc(e, (size_t)L.n * L.k * es);
  e.weight_bytes += (size_t)L.n * L.k * es;
  size_t off = 0;
  for (auto& n : names) {
    const DBuf& b = rawt(e, n);
    const size_t cnt = b.bytes / 2;
    if (dt == DT_F32)
      k_bf16_to_f32<<<(unsigned)std::min<size_t>((cnt + 255) / 256, 4096), 256, 0, e.st>>>((const bf16*)b.p,
                                                                                         (float*)L.w + off, cnt);
    else
      CK(cudaMemcpyAsync((bf16*)L.w + off, b.p, b.bytes, cudaMemcpyDeviceToDevice, e.st));
    CK(cudaGetLastError());
    off += cnt;
  }
  return L;
}
bf16* keep_bf16(Engine& e, const std::string& n) {  // take ownership of an uploaded bf16 tensor as is
  auto it = e.raw.find(n);
  if (it == e.raw.end()) throw FeErr("missing tensor " + n);
  bf16* p = (bf16*)it->second.p;
  e.weight_bytes += it->second.bytes;
  e.raw.erase(it);
  return p;
}

Rope make_rope(int type, int rot, double theta, int dim_for_freq) {
  Rope r{};
  r.type = type;
  r.rot = rot;
  const int nf = type == 3 ? rot / 2 : rot / 2;
  REQ(nf <= 256, "rope too wide");
  for (int k = 0; k < nf; ++k) {
    const float ex = (float)(2 * k) / (float)dim_for_freq;
    r.inv[k] = 1.0f / powf((float)theta, ex);
  }
  return r;
}

// FLASHEMBED_FP32_GEMM=emu: FP32 GEMMs through cuBLAS's BF16x9 FP32 emulation on Blackwell tensor cores
// (CUBLAS_COMPUTE_32F_EMULATED_16BFX9); default: plain SGEMM (no TF32)
int g_fp32_emu = -1;  // FLASHEMBED_FP32_GEMM=sgemm turns the emulation off; cleared if cuBLAS rejects it
bool fp32_emu() {
  if (g_fp32_emu < 0) {
    const char* x = getenv("FLASHEMBED_FP32_GEMM");
    g_fp32_emu = !(x && std::string(x) == "sgemm");
  }
  return g_fp32_emu;
}

// Y[M, n] = X[M, k] * W[n, k]^T  (row-major everything)
void gemm(Engine& e, const void* X, const Lin& W, void* Y, int M, int dt = -1) {
  if (M == 0) return;
  if (dt < 0) dt = e.dt;
  const float al = 1.f, be = 0.f;
  const cudaDataType_t t = dt == DT_F32 ? CUDA_R_32F : CUDA_R_16BF;
  const cublasComputeType_t ct = (dt == DT_F32 && fp32_emu()) ? CUBLAS_COMPUTE_32F_EMULATED_16BFX9 : CUBLAS_COMPUTE_32F;
  cublasStatus_t st = cublasGemmEx(e.cb, CUBLAS_OP_T, CUBLAS_OP_N, W.n, M, W.k, &al, W.w, t, W.k, X, t, W.k, &be, Y, t,
                                   W.n, ct, CUBLAS_GEMM_DEFAULT);
  if (st != CUBLAS_STATUS_SUCCESS && ct == CUBLAS_COMPUTE_32F_EMULATED_16BFX9) {  // GPU without BF16x9 emulation
    g_fp32_emu = 0;
    st = cublasGemmEx(e.cb, CUBLAS_OP_T, CUBLAS_OP_N, W.n, M, W.k, &al, W.w, t, W.k, X, t, W.k, &be, Y, t, W.n,
                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  }
  CB(st);
}

inline unsigned nblk(size_t n, int per = 256) { return (unsigned)std::min<size_t>((n + per - 1) / per, 65535 * 8); }

template <typename T>
void resid_norm(Engine& e, int ntok, int n, const void* x, int x_mode, const float* w_post, float xmul, float* h,
                float hscale, const float* init, float la, float lb, int y_mode, const float* w_pre, void* y, float* y32) {
  if (!ntok) return;
  const unsigned g = (ntok + 7) / 8;
#define RN(NV) k_resid_norm_w<T, NV><<<g, 256, 0, e.st>>>(ntok, n, e.eps, (const T*)x, x_mode, w_post, xmul, h, hscale, \
                                                          init, la, lb, y_mode, w_pre, (T*)y, y32)
  switch (n) {
    case 512: RN(4); break;
    case 768: RN(6); break;
    case 1024: RN(8); break;
    default: {
      const int th = n <= 512 ? 128 : 256;
      REQ(n <= 8 * th, "resid_norm: n too large");
      k_resid_norm<T><<<ntok, th, 0, e.st>>>(n, e.eps, (const T*)x, x_mode, w_post, xmul, h, hscale, init, la, lb, y_mode,
                                             w_pre, (T*)y, y32);
    }
  }
#undef RN
  CK(cudaGetLastError());
}

template <typename T, int D>
void qkv_prep_t(Engine& e, void* qkv, int ld, int ntok, int Hq, int Hkv, const float* qw, const float* kw, int qk_norm,
                int v_norm, float eps, const Rope& rope, const int* pos, int naxes, const bf16* vemb, const int* ids) {
  dim3 grid(ntok, (Hq + 2 * Hkv + 7) / 8);
  k_qkv_prep<T, D><<<grid, 256, 0, e.st>>>((T*)qkv, ld, ntok, Hq, Hkv, qw, kw, qk_norm, v_norm, eps, rope, pos, naxes,
                                           vemb, ids);
  CK(cudaGetLastError());
}
template <typename T>
void qkv_prep(Engine& e, int D, void* qkv, int ld, int ntok, int Hq, int Hkv, const float* qw, const float* kw,
              int qk_norm, int v_norm, float eps, const Rope& rope, const int* pos, int naxes, const bf16* vemb,
              const int* ids) {
  if (!ntok) return;
  switch (D) {
    case 64: return qkv_prep_t<T, 64>(e, qkv, ld, ntok, Hq, Hkv, qw, kw, qk_norm, v_norm, eps, rope, pos, naxes, vemb, ids);
    case 256: return qkv_prep_t<T, 256>(e, qkv, ld, ntok, Hq, Hkv, qw, kw, qk_norm, v_norm, eps, rope, pos, naxes, vemb, ids);
    case 512: return qkv_prep_t<T, 512>(e, qkv, ld, ntok, Hq, Hkv, qw, kw, qk_norm, v_norm, eps, rope, pos, naxes, vemb, ids);
  }
  throw FeErr("qkv_prep: unsupported head dim " + std::to_string(D));
}

// tile lists per query-tile height, built on the host per batch
struct Tiles {
  std::vector<int> host;  // pairs (seq, q0)
  int n = 0;
};
void build_tiles(const std::vector<int>& cu, int BR, Tiles& t) {
  t.host.clear();
  for (size_t s = 0; s + 1 < cu.size(); ++s)
    for (int q = 0; q < cu[s + 1] - cu[s]; q += BR) {
      t.host.push_back((int)s);
      t.host.push_back(q);
    }
  t.n = (int)t.host.size() / 2;
}

template <typename T, int D>
void attn_simt_t(Engine& e, const AttnArgs& a, int ntiles) {
  using C = SimtCfg<D>;
  const size_t smem = (size_t)(C::BR * D + C::BC * (D + 1) + C::BC * D + C::BR * C::BC) * 4;
  static bool set = false;
  if (!set) {
    CK(cudaFuncSetAttribute(k_attn_simt<T, D, C::BR, C::BC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    set = true;
  }
  dim3 grid(ntiles, a.Hq);
  k_attn_simt<T, D, C::BR, C::BC><<<grid, 128, smem, e.st>>>(a);
  CK(cudaGetLastError());
}
int simt_br(int D) { return D == 64 ? SimtCfg<64>::BR : D == 256 ? SimtCfg<256>::BR : SimtCfg<512>::BR; }

// Batch metadata on the device: ids, cu, pos, tiles for each BR, extra index arrays
struct Meta {
  int* ids = nullptr;
  int* cu = nullptr;
  int* pos = nullptr;
  int* mm = nullptr;
  int2* tiles16 = nullptr;
  int2* tiles32 = nullptr;
  int2* tiles64 = nullptr;
  int nt16 = 0, nt32 = 0, nt64 = 0;
  int* x1 = nullptr;  // extra arrays (vision slots / NeoMME patch rows)
  int* x2 = nullptr;
  int* x3 = nullptr;
};

// pack host int arrays into the pinned buffer, copy once, return device pointers
struct Packer {
  Engine& e;
  std::vector<std::pair<const int*, size_t>> parts;
  explicit Packer(Engine& e) : e(e) {}
  size_t add(const int* p, size_t n) {
    parts.push_back({p, n});
    return parts.size() - 1;
  }
  std::vector<int*> go() {
    CK(cudaStreamSynchronize(e.st));  // the pinned staging buffer may still feed an earlier async copy
    size_t tot = 0;
    for (auto& p : parts) tot += (p.second + 63) / 64 * 64;
    REQ(tot <= e.meta_ints, "batch metadata exceeds the planned size (raise max tokens)");
    int* hp = (int*)e.pin;
    std::vector<int*> out;
    size_t off = 0;
    for (auto& p : parts) {
      if (p.second) std::memcpy(hp + off, p.first, p.second * 4);
      out.push_back(e.b_meta + off);
      off += (p.second + 63) / 64 * 64;
    }
    CK(cudaMemcpyAsync(e.b_meta, hp, off * 4, cudaMemcpyHostToDevice, e.st));
    return out;
  }
};

template <int DQK, int DV, int BC, int STAGES>
void attn_mma_t(Engine& e, const AttnArgs& a, int ntiles) {
  const size_t smem = (size_t)(64 * (DQK + 8) + STAGES * BC * ((DQK + 8) + (DV + 8))) * 2;
  static bool set = false;
  if (!set) {
    CK(cudaFuncSetAttribute(k_attn_mma<DQK, DV, BC, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    set = true;
  }
  dim3 grid(ntiles, a.Hq, DQK / DV);
  k_attn_mma<DQK, DV, BC, STAGES><<<grid, 128, smem, e.st>>>(a);
  CK(cudaGetLastError());
}
template <int BC, int STAGES>
void attn_512p_t(Engine& e, const AttnArgs& a, int ntiles) {
  const size_t smem = (size_t)(64 * 520 + 2 * STAGES * BC * 520) * 2 + (size_t)2 * 4 * 32 * (BC / 8) * 4 * 4;
  static bool set = false;
  if (!set) {
    CK(cudaFuncSetAttribute(k_attn_mma512p<BC, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    set = true;
  }
  dim3 grid(ntiles, a.Hq);
  k_attn_mma512p<BC, STAGES><<<grid, 256, smem, e.st>>>(a);
  CK(cudaGetLastError());
}

// shared memory per block available on this GPU (opt-in): B200 227 KB, RTX PRO 6000 Blackwell 99 KB
int smem_optin() {
  static int v = -1;
  if (v < 0) {
    int d = 0;
    cudaGetDevice(&d);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, d);
  }
  return v;
}

bool force_simt() {
  static int v = -1;
  if (v < 0) {
    const char* x = getenv("FLASHEMBED_ATTN");
    v = x && std::string(x) == "simt";
  }
  return v;
}

template <typename T>
void attention(Engine& e, int D, const AttnArgs& a0, const Meta& m) {
  AttnArgs a = a0;
  if (std::is_same<T, bf16>::value && !force_simt()) {
    a.tiles = m.tiles64;
    if (!m.nt64) return;
    const bool big = smem_optin() >= 170 * 1024;  // B200-class shared memory: bigger / double-buffered tiles
    switch (D) {
      case 64: return attn_mma_t<64, 64, 64, 2>(e, a, m.nt64);
      case 256:  // 99 KB GPUs (RTX PRO 6000): 16-key tiles so two K/V stages still fit
        return big ? attn_mma_t<256, 256, 32, 2>(e, a, m.nt64)
                   : (getenv("FLASHEMBED_ATTN_1STAGE") ? attn_mma_t<256, 256, 32, 1>(e, a, m.nt64)
                                                       : attn_mma_t<256, 256, 16, 2>(e, a, m.nt64));
      case 512:
        if (smem_optin() >= 220 * 1024 && !getenv("FLASHEMBED_ATTN_SPLIT512")) return attn_512p_t<32, 2>(e, a, m.nt64);
        return big ? attn_mma_t<512, 256, 32, 2>(e, a, m.nt64) : attn_mma_t<512, 256, 16, 1>(e, a, m.nt64);
    }
    throw FeErr("attention: unsupported head dim");
  }
  const int br = simt_br(D);
  a.tiles = br == 16 ? m.tiles16 : m.tiles32;
  const int nt = br == 16 ? m.nt16 : m.nt32;
  if (!nt) return;
  switch (D) {
    case 64: return attn_simt_t<T, 64>(e, a, nt);
    case 256: return attn_simt_t<T, 256>(e, a, nt);
    case 512: return attn_simt_t<T, 512>(e, a, nt);
  }
  throw FeErr("attention: unsupported head dim");
}

// ------------------------------------------------------------------------------------------------- arena planning
void* carve(Engine& e, size_t bytes) {
  bytes = (bytes + 255) / 256 * 256;
  REQ(e.arena_off + bytes <= e.arena_bytes, "arena overflow");
  void* p = e.arena + e.arena_off;
  e.arena_off += bytes;
  return p;
}

void plan(Engine& e) {
  const size_t T = e.max_tok, es = e.esz();
  size_t qkvw = 0, ow = 0, guw = 0;
  if (e.arch == ARCH_GEMMA) {
    for (auto& l : e.gl) {
      qkvw = std::max<size_t>(qkvw, l.qkv.n);
      ow = std::max<size_t>(ow, (size_t)e.Hq * l.hd);
    }
    guw = 2 * (size_t)e.I;
  } else {
    for (auto& l : e.nl) qkvw = std::max<size_t>(qkvw, l.qkvg.n);
    ow = e.H;
    guw = e.I;
  }
  const size_t H = e.H;
  // metadata ints: ids, cu, pos(2), mm, 2 tile lists (2 ints each), extras
  e.meta_ints = (T * 12 + 64 * 1024 + (size_t)e.max_patch * 4) + 4096;
  size_t need = 0;
  auto add = [&](size_t b) { need += (b + 255) / 256 * 256; };
  add(T * H * es);            // y
  add(T * qkvw * es);         // qkv
  add(T * ow * es);           // o
  add(T * H * es);            // t1
  add(T * H * es);            // t2
  add(T * guw * es);          // gu
  add(T * e.I * es);          // m
  add(T * H * es);            // e (embeddings in compute dtype)
  add(T * 128 * es);          // late
  add(T * H * 4);             // h
  add(T * H * 4);             // init (NeoMME)
  add(T * H * 4);             // y32
  add(4096 * H * 4);          // pool
  add(4096 * 1024 * 4);       // out
  add(e.meta_ints * 4);       // meta
  if (e.arch == ARCH_NEOMME) {
    const size_t P = e.max_patch;
    add(P * 3072);
    add(P * 3072 * es);
    add(P * 2048 * es);
    add(P * H * es);
  }
  if (e.has_vision) {
    const size_t P = e.max_patch, VH = e.VH;
    size_t vq = 0;
    for (auto& l : e.vl) vq = std::max<size_t>(vq, l.qkv.n);
    add(P * 768 * 4);           // patches fp32 (host-preprocessed path)
    add(e.img_bytes);           // decoded source images (GPU preprocessing)
    add(e.img_bytes);           // horizontal-pass output
    add(4 << 20);               // resize coefficients
    add(P * 768 * es);          // pix
    add(P * VH * es);           // y
    add(P * vq * es);           // qkv
    add(P * VH * es);           // o
    add(P * VH * es);           // t1
    add(P * 2 * e.VI * es);     // gu
    add(P * e.VI * es);         // m
    add(P * VH * 4);            // h
    add(P / 4 * VH * es);       // ev (pooled, pre-projection)
    add(P / 4 * H * es);        // soft tokens
  }
  e.arena_bytes = need;
  CK(cudaMalloc((void**)&e.arena, need));
  e.owned.push_back(e.arena);
  e.arena_off = 0;
  e.b_y = carve(e, T * H * es);
  e.b_qkv = carve(e, T * qkvw * es);
  e.b_o = carve(e, T * ow * es);
  e.b_t1 = carve(e, T * H * es);
  e.b_t2 = carve(e, T * H * es);
  e.b_gu = carve(e, T * guw * es);
  e.b_m = carve(e, T * e.I * es);
  e.b_e = carve(e, T * H * es);
  e.b_late = carve(e, T * 128 * es);
  e.b_h = (float*)carve(e, T * H * 4);
  e.b_init = (float*)carve(e, T * H * 4);
  e.b_y32 = (float*)carve(e, T * H * 4);
  e.b_pool = (float*)carve(e, 4096 * H * 4);
  e.b_out = (float*)carve(e, 4096 * 1024 * 4);
  e.b_meta = (int*)carve(e, e.meta_ints * 4);
  if (e.arch == ARCH_NEOMME) {
    const size_t P = e.max_patch;
    e.n_patch_u8 = (uint8_t*)carve(e, P * 3072);
    e.n_ln = carve(e, P * 3072 * es);
    e.n_up = carve(e, P * 2048 * es);
    e.n_dn = carve(e, P * H * es);
  }
  if (e.has_vision) {
    const size_t P = e.max_patch, VH = e.VH;
    size_t vq = 0;
    for (auto& l : e.vl) vq = std::max<size_t>(vq, l.qkv.n);
    e.v_patch_f32 = (float*)carve(e, P * 768 * 4);
    e.v_src = (uint8_t*)carve(e, e.img_bytes);
    e.v_tmp = (uint8_t*)carve(e, e.img_bytes);
    e.v_coef = (int*)carve(e, 4 << 20);
    e.v_pix = carve(e, P * 768 * es);
    e.v_y = carve(e, P * VH * es);
    e.v_qkv = carve(e, P * vq * es);
    e.v_o = carve(e, P * VH * es);
    e.v_t1 = carve(e, P * VH * es);
    e.v_gu = carve(e, P * 2 * e.VI * es);
    e.v_m = carve(e, P * e.VI * es);
    e.v_h = (float*)carve(e, P * VH * 4);
    e.v_ev = carve(e, P / 4 * VH * es);
    e.v_soft = carve(e, P / 4 * H * es);
  }
  // pinned staging: metadata + largest float input (patches)
  e.pin_bytes = e.meta_ints * 4 + 64 +
                (e.arch == ARCH_NEOMME ? (size_t)e.max_patch * 3072
                                       : (e.has_vision ? std::max<size_t>((size_t)e.max_patch * 768 * 4, e.img_bytes + (4 << 20)) : 0));
  CK(cudaMallocHost((void**)&e.pin, e.pin_bytes));
}

// ------------------------------------------------------------------------------------------------- build
void build_gemma(Engine& e) {
  const std::string P = "language_model.";
  e.H = (int)cfgv(e, "hidden");
  e.L = (int)cfgv(e, "layers");
  e.Hq = (int)cfgv(e, "heads");
  e.I = (int)cfgv(e, "inter");
  e.ple = (int)cfgv(e, "ple");
  e.eps = (float)cfgv(e, "eps");
  e.embed_scale = sqrtf((float)e.H);
  e.ple_scale = 1.f / sqrtf((float)e.H);
  const int sw = (int)cfgv(e, "sliding_window");
  e.embed = keep_bf16(e, P + "embed_tokens.weight");
  e.vocab = (int)e.shapes[P + "embed_tokens.weight"][0];
  e.ple_w = make_lin(e, {P + "ple.per_layer_model_projection.weight"});
  e.w_ple_norm = to_f32(e, P + "ple.per_layer_projection_norm.weight");
  e.w_final = to_f32(e, P + "norm.weight");
  e.proj_f32 = make_lin(e, {P + "embedding_projection.weight"}, DT_F32);
  e.out_dim = e.proj_f32.n;
  for (int i = 0; i < e.L; ++i) {
    const std::string p = P + "layers." + std::to_string(i) + ".";
    GLayer l;
    const bool full = cfgv(e, "full." + std::to_string(i), 0) != 0;
    l.hd = (int)e.shapes[p + "self_attn.q_norm.weight"][0];
    l.nkv = (int)e.shapes[p + "self_attn.k_proj.weight"][0] / l.hd;
    l.window = full ? -1 : sw;
    l.qkv = make_lin(e, {p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"});
    l.o = make_lin(e, {p + "self_attn.o_proj.weight"});
    l.gu = make_lin(e, {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"});
    l.down = make_lin(e, {p + "mlp.down_proj.weight"});
    l.pg = make_lin(e, {p + "ple_block.per_layer_input_gate.weight"});
    l.pp = make_lin(e, {p + "ple_block.per_layer_projection.weight"});
    l.w_in = to_f32(e, p + "input_layernorm.weight");
    l.w_pa = to_f32(e, p + "post_attention_layernorm.weight");
    l.w_pff = to_f32(e, p + "pre_feedforward_layernorm.weight");
    l.w_pf = to_f32(e, p + "post_feedforward_layernorm.weight");
    l.w_qn = to_f32(e, p + "self_attn.q_norm.weight");
    l.w_kn = to_f32(e, p + "self_attn.k_norm.weight");
    l.w_ppn = to_f32(e, p + "ple_block.post_per_layer_input_norm.weight");
    l.ls = host_scalar(e, p + "layer_scalar");
    const double th = cfgv(e, full ? "rope_theta_full" : "rope_theta_sliding");
    l.rope = make_rope(1, l.hd, th, l.hd);
    e.gl.push_back(l);
  }
  if (has(e, "vision_tower.patch_embedder.input_proj.weight") && cfgv(e, "vision", 0) != 0) {
    e.has_vision = true;
    const std::string V = "vision_tower.";
    e.VH = (int)cfgv(e, "v_hidden");
    e.VL = (int)cfgv(e, "v_layers");
    e.VHq = (int)cfgv(e, "v_heads");
    e.VI = (int)cfgv(e, "v_inter");
    e.veps = (float)cfgv(e, "v_eps");
    e.vin = make_lin(e, {V + "patch_embedder.input_proj.weight"});
    e.vpos = keep_bf16(e, V + "patch_embedder.position_embedding_table");
    e.vpos_n = (int)e.shapes[V + "patch_embedder.position_embedding_table"][1];
    e.vev = make_lin(e, {"embed_vision.embedding_projection.weight"});
    for (int i = 0; i < e.VL; ++i) {
      const std::string p = V + "encoder.layers." + std::to_string(i) + ".";
      GLayer l;
      l.hd = (int)e.shapes[p + "self_attn.q_norm.weight"][0];
      l.nkv = (int)e.shapes[p + "self_attn.k_proj.linear.weight"][0] / l.hd;
      l.window = -1;
      l.qkv = make_lin(e, {p + "self_attn.q_proj.linear.weight", p + "self_attn.k_proj.linear.weight",
                           p + "self_attn.v_proj.linear.weight"});
      l.o = make_lin(e, {p + "self_attn.o_proj.linear.weight"});
      l.gu = make_lin(e, {p + "mlp.gate_proj.linear.weight", p + "mlp.up_proj.linear.weight"});
      l.down = make_lin(e, {p + "mlp.down_proj.linear.weight"});
      l.w_in = to_f32(e, p + "input_layernorm.weight");
      l.w_pa = to_f32(e, p + "post_attention_layernorm.weight");
      l.w_pff = to_f32(e, p + "pre_feedforward_layernorm.weight");
      l.w_pf = to_f32(e, p + "post_feedforward_layernorm.weight");
      l.w_qn = to_f32(e, p + "self_attn.q_norm.weight");
      l.w_kn = to_f32(e, p + "self_attn.k_norm.weight");
      l.rope = make_rope(3, l.hd / 2, cfgv(e, "v_rope_theta"), l.hd / 2);
      e.vl.push_back(l);
    }
  }
}

void build_neomme(Engine& e) {
  e.H = (int)cfgv(e, "hidden");
  e.L = (int)cfgv(e, "layers");
  e.Hq = (int)cfgv(e, "heads");
  e.I = (int)cfgv(e, "inter");
  e.eps = (float)cfgv(e, "eps");
  e.resid_mult = (float)cfgv(e, "resid_mult");
  const int hd = (int)cfgv(e, "head_dim");
  e.embed = keep_bf16(e, "model.embed_tokens.word_embeddings.weight");
  e.vemb = keep_bf16(e, "model.value_embeddings.weight");
  e.nemb_proj = make_lin(e, {"model.embed_tokens.embedding_projection.weight"});
  e.late = make_lin(e, {"multi_vector_head.proj.weight"});
  e.nup = make_lin(e, {"model.patch_embeddings.up_proj.weight"});
  e.ndown = make_lin(e, {"model.patch_embeddings.down_proj.weight"});
  e.n_ln_w = to_f32(e, "model.patch_embeddings.norm.weight");
  e.n_ln_b = to_f32(e, "model.patch_embeddings.norm.bias");
  e.n_down_b = to_f32(e, "model.patch_embeddings.down_proj.bias");
  for (int i = 0; i < e.L; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    NLayer l;
    l.qkvg = make_lin(e, {p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight",
                          p + "self_attn.output_projection.gate_proj.weight"});
    l.o = make_lin(e, {p + "self_attn.output_projection.o_proj.weight"});
    l.up = make_lin(e, {p + "mlp.up_proj.weight"});
    l.down = make_lin(e, {p + "mlp.down_proj.weight"});
    l.la = host_scalar(e, p + "lambdas", 0);
    l.lb = host_scalar(e, p + "lambdas", 1);
    std::vector<float> at(e.Hq);
    for (int h = 0; h < e.Hq; ++h) at[h] = tanhf(host_scalar(e, p + "self_attn.exclusive_self_attention.alpha", h));
    l.alpha_t = (float*)dmalloc(e, e.Hq * 4);
    CK(cudaMemcpy(l.alpha_t, at.data(), e.Hq * 4, cudaMemcpyHostToDevice));
    l.window = (int)cfgv(e, "window." + std::to_string(i));
    l.vemb = (int)cfgv(e, "vemb." + std::to_string(i), 0);
    const bool full = l.window < 0;
    const double frac = cfgv(e, full ? "rope_frac_full" : "rope_frac_sliding");
    const int rot = (int)(hd * frac);
    l.rope = make_rope(2, rot, cfgv(e, full ? "rope_theta_full" : "rope_theta_sliding"), rot);
    e.nl.push_back(l);
  }
}

// ------------------------------------------------------------------------------------------------- forwards
template <typename T>
void gemma_layers(Engine& e, int ntok, const Meta& m, std::vector<GLayer>& layers, int H, int Hq, int I, void* y,
                  void* qkv, void* o, void* t1, void* gu, void* mbuf, float* h, bool ple, void* t2, void* emb,
                  int naxes, float* y32_final, const float* w_final) {
  for (size_t li = 0; li < layers.size(); ++li) {
    GLayer& l = layers[li];
    const int D = l.hd;
    gemm(e, y, l.qkv, qkv, ntok);
    qkv_prep<T>(e, D, qkv, l.qkv.n, ntok, Hq, l.nkv, l.w_qn, l.w_kn, 1, 1, e.eps, l.rope, m.pos, naxes, nullptr, nullptr);
    AttnArgs a{};
    a.qkv = qkv;
    a.ld = l.qkv.n;
    a.q_off = 0;
    a.k_off = Hq * D;
    a.v_off = Hq * D + l.nkv * D;
    a.Hq = Hq;
    a.Hkv = l.nkv;
    a.window = l.window;
    a.scale = 1.f;
    a.cu = m.cu;
    a.out = o;
    a.ld_out = Hq * D;
    attention<T>(e, D, a, m);
    gemm(e, o, l.o, t1, ntok);
    resid_norm<T>(e, ntok, H, t1, 1, l.w_pa, 0.f, h, 1.f, nullptr, 0, 0, 1, l.w_pff, y, nullptr);
    gemm(e, y, l.gu, gu, ntok);
    k_gelu_gate4<T><<<nblk((size_t)ntok * I / 4), 256, 0, e.st>>>((const T*)gu, (T*)mbuf, I, ntok);
    CK(cudaGetLastError());
    gemm(e, mbuf, l.down, t1, ntok);
    const bool last = li + 1 == layers.size();
    if (ple) {
      resid_norm<T>(e, ntok, H, t1, 1, l.w_pf, 0.f, h, 1.f, nullptr, 0, 0, 2, nullptr, y, nullptr);
      gemm(e, y, l.pg, t1, ntok);  // gate
      Lin sl{(char*)e.ple_w.w + (size_t)li * e.ple * e.ple_w.k * e.esz(), e.ple, e.ple_w.k};
      gemm(e, emb, sl, t2, ntok);  // this layer's slice of the per-layer projection
      REQ(e.ple == 512, "PLE width 512 expected");
      k_ple_w<T, 4><<<(ntok + 7) / 8, 256, 0, e.st>>>((const T*)t1, (const T*)t2, e.ple_scale, e.w_ple_norm, e.eps, (T*)y,
                                                      e.ple, ntok);
      CK(cudaGetLastError());
      gemm(e, y, l.pp, t1, ntok);
      if (last)
        resid_norm<T>(e, ntok, H, t1, 1, l.w_ppn, 0.f, h, l.ls, nullptr, 0, 0, 1, w_final, nullptr, y32_final);
      else
        resid_norm<T>(e, ntok, H, t1, 1, l.w_ppn, 0.f, h, l.ls, nullptr, 0, 0, 1, layers[li + 1].w_in, y, nullptr);
    } else {
      if (last)
        resid_norm<T>(e, ntok, H, t1, 1, l.w_pf, 0.f, h, 1.f, nullptr, 0, 0, 0, nullptr, nullptr, nullptr);
      else
        resid_norm<T>(e, ntok, H, t1, 1, l.w_pf, 0.f, h, 1.f, nullptr, 0, 0, 1, layers[li + 1].w_in, y, nullptr);
    }
  }
}

struct ImgIn {  // GPU image preprocessing inputs (null src = host-preprocessed float patches)
  const uint8_t* src = nullptr;
  size_t src_bytes = 0;
  const int* desc = nullptr;
  const int* coef = nullptr;
  size_t ncoef = 0;
  int max_pix = 0;  // largest tw * th or rows * tw in the batch
};

template <typename T>
void gemma_vision(Engine& e, int npatch, int nimg, const std::vector<int>& vcu, const int* pos_host,
                  const float* patches_host, const ImgIn& im, int nslot, const std::vector<int>& slot_ptr,
                  const std::vector<int>& slot_patch) {
  // metadata
  Tiles t16, t32, t64;
  build_tiles(vcu, 16, t16);
  build_tiles(vcu, 32, t32);
  build_tiles(vcu, 64, t64);
  Packer pk(e);
  pk.add(vcu.data(), vcu.size());
  pk.add(pos_host, (size_t)npatch * 2);
  pk.add(t16.host.data(), t16.host.size());
  pk.add(t32.host.data(), t32.host.size());
  pk.add(slot_ptr.data(), slot_ptr.size());
  pk.add(slot_patch.data(), slot_patch.size());
  pk.add(t64.host.data(), t64.host.size());
  if (im.src) pk.add(im.desc, (size_t)nimg * D_N);
  auto d = pk.go();
  Meta m;
  m.cu = d[0];
  m.pos = d[1];
  m.tiles16 = (int2*)d[2];
  m.nt16 = t16.n;
  m.tiles32 = (int2*)d[3];
  m.nt32 = t32.n;
  m.tiles64 = (int2*)d[6];
  m.nt64 = t64.n;
  const size_t pin_off = e.meta_ints * 4;
  if (im.src) {
    REQ(im.src_bytes <= e.img_bytes, "image batch exceeds the source-image budget");
    REQ(im.ncoef * 4 <= (4u << 20), "resize coefficient table too large");
    const size_t coff = (pin_off + im.src_bytes + 15) / 16 * 16;
    std::memcpy(e.pin + pin_off, im.src, im.src_bytes);
    std::memcpy(e.pin + coff, im.coef, im.ncoef * 4);
    CK(cudaMemcpyAsync(e.v_src, e.pin + pin_off, im.src_bytes, cudaMemcpyHostToDevice, e.st));
    CK(cudaMemcpyAsync(e.v_coef, e.pin + coff, im.ncoef * 4, cudaMemcpyHostToDevice, e.st));
    const int* desc = d[7];
    dim3 grid((im.max_pix + 255) / 256, nimg);
    k_resize_h<<<grid, 256, 0, e.st>>>(desc, e.v_src, e.v_coef, e.v_tmp);
    k_resize_v_patch<T><<<grid, 256, 0, e.st>>>(desc, e.v_src, e.v_coef, e.v_tmp, (T*)e.v_pix);
    CK(cudaGetLastError());
  } else {
    std::memcpy(e.pin + pin_off, patches_host, (size_t)npatch * 768 * 4);
    CK(cudaMemcpyAsync(e.v_patch_f32, e.pin + pin_off, (size_t)npatch * 768 * 4, cudaMemcpyHostToDevice, e.st));
    k_affine<T><<<nblk((size_t)npatch * 768), 256, 0, e.st>>>(e.v_patch_f32, (T*)e.v_pix, 2.f, -0.5f,
                                                              (size_t)npatch * 768);
  }
  gemm(e, e.v_pix, e.vin, e.v_t1, npatch);
  k_vis_pos<T><<<npatch, 256, 0, e.st>>>((const T*)e.v_t1, m.pos, e.vpos, e.vpos_n, e.v_h, e.VH);
  CK(cudaGetLastError());
  resid_norm<T>(e, npatch, e.VH, nullptr, 0, nullptr, 0.f, e.v_h, 1.f, nullptr, 0, 0, 1, e.vl[0].w_in, e.v_y, nullptr);
  const float saved_eps = e.eps;
  e.eps = e.veps;
  gemma_layers<T>(e, npatch, m, e.vl, e.VH, e.VHq, e.VI, e.v_y, e.v_qkv, e.v_o, e.v_t1, e.v_gu, e.v_m, e.v_h, false,
                  nullptr, nullptr, 2, nullptr, nullptr);
  e.eps = saved_eps;
  k_vis_pool<T><<<nslot, 256, 0, e.st>>>(e.v_h, d[4], d[5], 1.f / 9.f, sqrtf((float)e.VH), e.veps, (T*)e.v_ev, e.VH);
  CK(cudaGetLastError());
  gemm(e, e.v_ev, e.vev, e.v_soft, nslot);
}

template <typename T>
void gemma_text(Engine& e, int ntok, int nseq, const int* ids, const int* cu_h, const int* mm_idx, float* out_host) {
  REQ(ntok <= e.max_tok, "batch exceeds max tokens");
  std::vector<int> cu(cu_h, cu_h + nseq + 1), pos(ntok);
  for (int s = 0; s < nseq; ++s)
    for (int t = cu[s]; t < cu[s + 1]; ++t) pos[t] = t - cu[s];
  Tiles t16, t32, t64;
  build_tiles(cu, 16, t16);
  build_tiles(cu, 32, t32);
  build_tiles(cu, 64, t64);
  Packer pk(e);
  pk.add(ids, ntok);
  pk.add(cu.data(), cu.size());
  pk.add(pos.data(), ntok);
  pk.add(t16.host.data(), t16.host.size());
  pk.add(t32.host.data(), t32.host.size());
  pk.add(t64.host.data(), t64.host.size());
  if (mm_idx) pk.add(mm_idx, ntok);
  auto d = pk.go();
  Meta m;
  m.ids = d[0];
  m.cu = d[1];
  m.pos = d[2];
  m.tiles16 = (int2*)d[3];
  m.nt16 = t16.n;
  m.tiles32 = (int2*)d[4];
  m.nt32 = t32.n;
  m.tiles64 = (int2*)d[5];
  m.nt64 = t64.n;
  const int H = e.H;
  k_gemma_embed<T><<<ntok, 128, 0, e.st>>>(m.ids, e.embed, e.embed_scale, mm_idx ? d[6] : nullptr, (const T*)e.v_soft,
                                           e.b_h, (T*)e.b_e, H);
  CK(cudaGetLastError());
  resid_norm<T>(e, ntok, H, nullptr, 0, nullptr, 0.f, e.b_h, 1.f, nullptr, 0, 0, 1, e.gl[0].w_in, e.b_y, nullptr);
  gemma_layers<T>(e, ntok, m, e.gl, H, e.Hq, e.I, e.b_y, e.b_qkv, e.b_o, e.b_t1, e.b_gu, e.b_m, e.b_h, true, e.b_t2,
                  e.b_e, 1, e.b_y32, e.w_final);
  k_mean_pool<<<nseq, 256, 0, e.st>>>(e.b_y32, m.cu, e.b_pool, H);
  gemm(e, e.b_pool, e.proj_f32, e.b_out, nseq, DT_F32);
  k_l2norm<float><<<nseq, 256, 0, e.st>>>(e.b_out, e.b_out, e.out_dim);
  CK(cudaGetLastError());
  CK(cudaMemcpyAsync(out_host, e.b_out, (size_t)nseq * e.out_dim * 4, cudaMemcpyDeviceToHost, e.st));
  CK(cudaStreamSynchronize(e.st));
}

template <typename T>
void neomme(Engine& e, int ntok, int nseq, const int* ids, const int* cu_h, const int* pos2, int npatch,
            const uint8_t* patches, const int* patch_rows, float* dense_host, float* late_host) {
  REQ(ntok <= e.max_tok, "batch exceeds max tokens");
  REQ(npatch <= e.max_patch, "batch exceeds max patches");
  std::vector<int> cu(cu_h, cu_h + nseq + 1);
  std::vector<int> pos;
  if (!pos2) {
    pos.resize((size_t)ntok * 2);
    for (int s = 0; s < nseq; ++s)
      for (int t = cu[s]; t < cu[s + 1]; ++t) pos[2 * t] = pos[2 * t + 1] = t - cu[s];
    pos2 = pos.data();
  }
  Tiles t16, t32, t64;
  build_tiles(cu, 16, t16);
  build_tiles(cu, 32, t32);
  build_tiles(cu, 64, t64);
  Packer pk(e);
  pk.add(ids, ntok);
  pk.add(cu.data(), cu.size());
  pk.add(pos2, (size_t)ntok * 2);
  pk.add(t16.host.data(), t16.host.size());
  pk.add(t32.host.data(), t32.host.size());
  pk.add(patch_rows, npatch);
  pk.add(t64.host.data(), t64.host.size());
  auto d = pk.go();
  Meta m;
  m.ids = d[0];
  m.cu = d[1];
  m.pos = d[2];
  m.tiles16 = (int2*)d[3];
  m.nt16 = t16.n;
  m.tiles32 = (int2*)d[4];
  m.nt32 = t32.n;
  m.tiles64 = (int2*)d[6];
  m.nt64 = t64.n;
  const int H = e.H;
  // embeddings: factorised token embeddings for every row, patch stem rows overwrite theirs
  k_gather<T><<<ntok, 256, 0, e.st>>>(m.ids, e.embed, (T*)e.b_t1, e.nemb_proj.k);
  CK(cudaGetLastError());
  gemm(e, e.b_t1, e.nemb_proj, e.b_e, ntok);
  if (npatch) {
    const size_t pin_off = e.meta_ints * 4;
    std::memcpy(e.pin + pin_off, patches, (size_t)npatch * 3072);
    CK(cudaMemcpyAsync(e.n_patch_u8, e.pin + pin_off, (size_t)npatch * 3072, cudaMemcpyHostToDevice, e.st));
    k_layernorm_u8<T><<<npatch, 256, 0, e.st>>>(e.n_patch_u8, e.n_ln_w, e.n_ln_b, 1e-5f, (T*)e.n_ln, 3072);
    gemm(e, e.n_ln, e.nup, e.n_up, npatch);
    k_gelu_erf<T><<<nblk((size_t)npatch * 2048), 256, 0, e.st>>>((T*)e.n_up, (size_t)npatch * 2048);
    gemm(e, e.n_up, e.ndown, e.n_dn, npatch);
    k_scatter_rows<T><<<npatch, 256, 0, e.st>>>((const T*)e.n_dn, e.n_down_b, d[5], (T*)e.b_e, H);
    CK(cudaGetLastError());
  }
  // embedding_norm -> h and init; y = rms(la*h + lb*init) for layer 0
  resid_norm<T>(e, ntok, H, e.b_e, 3, nullptr, 0.f, e.b_h, 1.f, nullptr, 0, 0, 1, nullptr, nullptr, e.b_init);
  CK(cudaMemcpyAsync(e.b_h, e.b_init, (size_t)ntok * H * 4, cudaMemcpyDeviceToDevice, e.st));
  resid_norm<T>(e, ntok, H, nullptr, 0, nullptr, 0.f, e.b_h, 1.f, e.b_init, e.nl[0].la, e.nl[0].lb, 1, nullptr, e.b_y,
                nullptr);
  const float rm = e.resid_mult;
  for (int li = 0; li < e.L; ++li) {
    NLayer& l = e.nl[li];
    const int D = 64, Hkv = (l.qkvg.n - 2 * e.Hq * D) / (2 * D);
    gemm(e, e.b_y, l.qkvg, e.b_qkv, ntok);
    qkv_prep<T>(e, D, e.b_qkv, l.qkvg.n, ntok, e.Hq, Hkv, nullptr, nullptr, 1, 0, e.eps, l.rope, m.pos, 2,
                l.vemb ? e.vemb : nullptr, m.ids);
    AttnArgs a{};
    a.qkv = e.b_qkv;
    a.ld = l.qkvg.n;
    a.q_off = 0;
    a.k_off = e.Hq * D;
    a.v_off = e.Hq * D + Hkv * D;
    a.g_off = e.Hq * D + 2 * Hkv * D;
    a.Hq = e.Hq;
    a.Hkv = Hkv;
    a.window = l.window;
    a.scale = 1.f / sqrtf((float)D);
    a.cu = m.cu;
    a.out = e.b_o;
    a.ld_out = e.Hq * D;
    a.alpha_t = l.alpha_t;
    attention<T>(e, D, a, m);
    gemm(e, e.b_o, l.o, e.b_t1, ntok);
    resid_norm<T>(e, ntok, H, e.b_t1, 2, nullptr, rm, e.b_h, 1.f, nullptr, 0, 0, 1, nullptr, e.b_y, nullptr);
    gemm(e, e.b_y, l.up, e.b_gu, ntok);
    k_relu2_4<T><<<nblk((size_t)ntok * e.I / 4), 256, 0, e.st>>>((T*)e.b_gu, (long long)ntok * e.I / 4);
    CK(cudaGetLastError());
    gemm(e, e.b_gu, l.down, e.b_t1, ntok);
    if (li + 1 < e.L)
      resid_norm<T>(e, ntok, H, e.b_t1, 2, nullptr, rm, e.b_h, 1.f, e.b_init, e.nl[li + 1].la, e.nl[li + 1].lb, 1,
                    nullptr, e.b_y, nullptr);
    else
      resid_norm<T>(e, ntok, H, e.b_t1, 2, nullptr, rm, e.b_h, 1.f, nullptr, 0, 0, 1, nullptr, e.b_y, e.b_y32);
  }
  k_mean_pool<<<nseq, 256, 0, e.st>>>(e.b_y32, m.cu, e.b_pool, H);
  k_l2norm<float><<<nseq, 256, 0, e.st>>>(e.b_pool, e.b_out, H);
  CK(cudaGetLastError());
  CK(cudaMemcpyAsync(dense_host, e.b_out, (size_t)nseq * H * 4, cudaMemcpyDeviceToHost, e.st));
  if (late_host) {
    gemm(e, e.b_y, e.late, e.b_late, ntok);
    k_l2norm<T><<<ntok, 128, 0, e.st>>>((const T*)e.b_late, e.b_y32, e.late.n);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(late_host, e.b_y32, (size_t)ntok * e.late.n * 4, cudaMemcpyDeviceToHost, e.st));
  }
  CK(cudaStreamSynchronize(e.st));
}

template <typename F>
int guard(F&& f) {
  try {
    f();
    return 0;
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return -1;
  }
}

}  // namespace

// ================================================================================================== C API
extern "C" {

const char* fe_error() { return g_err.c_str(); }

void* fe_new(int arch, int dtype, int device) {
  Engine* e = new Engine();
  e->arch = arch;
  e->dt = dtype;
  e->dev = device;
  if (guard([&] {
        CK(cudaSetDevice(device));
        CK(cudaStreamCreateWithFlags(&e->st, cudaStreamNonBlocking));
        CB(cublasCreate(&e->cb));
        CB(cublasSetStream(e->cb, e->st));
        CB(cublasSetMathMode(e->cb, CUBLAS_DEFAULT_MATH));  // FP32 GEMMs stay true FP32 (no TF32)
        if (fp32_emu() && cublasSetEmulationStrategy(e->cb, CUBLAS_EMULATION_STRATEGY_EAGER) != CUBLAS_STATUS_SUCCESS)
          g_fp32_emu = 0;
        const size_t wsb = 64ull << 20;
        CK(cudaMalloc(&e->ws, wsb));
        e->owned.push_back(e->ws);
        CB(cublasSetWorkspace(e->cb, e->ws, wsb));
      }) != 0) {
    delete e;
    return nullptr;
  }
  return e;
}

int fe_cfg(void* h, const char* key, double v) {
  return guard([&] { ((Engine*)h)->cfg[key] = v; });
}

// src_dtype: 1 = bf16 (the checkpoints' type)
int fe_add(void* h, const char* name, int src_dtype, const int64_t* shape, int ndim, const void* data) {
  return guard([&] {
    Engine& e = *(Engine*)h;
    REQ(src_dtype == 1, "only bf16 checkpoints are supported");
    size_t n = 1;
    std::vector<int64_t> sh(shape, shape + ndim);
    for (auto s : sh) n *= (size_t)s;
    DBuf b;
    b.bytes = n * 2;
    CK(cudaMalloc(&b.p, std::max<size_t>(b.bytes, 2)));
    CK(cudaMemcpy(b.p, data, b.bytes, cudaMemcpyHostToDevice));
    e.raw[name] = b;
    e.shapes[name] = sh;
  });
}

int fe_build(void* h, int max_tokens, int max_patches) {
  return guard([&] {
    Engine& e = *(Engine*)h;
    e.max_tok = max_tokens;
    e.max_patch = max_patches;
    e.img_bytes = (size_t)cfgv(e, "img_bytes", 0);
    if (e.arch == ARCH_GEMMA) build_gemma(e);
    else build_neomme(e);
    CK(cudaStreamSynchronize(e.st));
    for (auto& kv : e.raw) cudaFree(kv.second.p);
    e.raw.clear();
    plan(e);
  });
}

// EmbeddingGemma 2 text (and multimodal rows when n_soft > 0 after fe_gemma_vision): out [nseq, 768]
int fe_gemma_text(void* h, int ntok, int nseq, const int32_t* ids, const int32_t* cu, const int32_t* mm_idx, float* out) {
  return guard([&] {
    Engine& e = *(Engine*)h;
    if (e.dt == DT_F32) gemma_text<float>(e, ntok, nseq, ids, cu, mm_idx, out);
    else gemma_text<bf16>(e, ntok, nseq, ids, cu, mm_idx, out);
  });
}

// Gemma 4 vision tower: patches [npatch, 768] (processor values in [0,1]), pos [npatch, 2] (x, y), images packed by
// vcu [nimg+1]; slots: soft token s averages patches slot_patch[slot_ptr[s]..slot_ptr[s+1]]. Leaves nslot soft tokens
// on the device for the next fe_gemma_text call (mm_idx refers to them).
int fe_gemma_vision(void* h, int npatch, int nimg, const int32_t* vcu, const int32_t* pos, const float* patches,
                    int nslot, const int32_t* slot_ptr, const int32_t* slot_patch) {
  return guard([&] {
    Engine& e = *(Engine*)h;
    REQ(e.has_vision, "engine built without the vision tower");
    REQ(npatch <= e.max_patch, "batch exceeds max patches");
    REQ(nslot <= e.max_patch / 4, "too many soft tokens");
    std::vector<int> c(vcu, vcu + nimg + 1), sp(slot_ptr, slot_ptr + nslot + 1), spp(slot_patch, slot_patch + sp[nslot]);
    ImgIn im;
    if (e.dt == DT_F32) gemma_vision<float>(e, npatch, nimg, c, pos, patches, im, nslot, sp, spp);
    else gemma_vision<bf16>(e, npatch, nimg, c, pos, patches, im, nslot, sp, spp);
  });
}

// Same, with the image preprocessing on the GPU: src = decoded RGB8 images back to back, desc [nimg, 20] per-image
// resize/patch descriptor, coef = resize bounds + int16-range coefficients (src/gpu.rs builds both).
int fe_gemma_vision_u8(void* h, int npatch, int nimg, const int32_t* vcu, const int32_t* pos, const uint8_t* src,
                       uint64_t src_bytes, const int32_t* desc, const int32_t* coef, uint64_t ncoef, int max_pix,
                       int nslot, const int32_t* slot_ptr, const int32_t* slot_patch) {
  return guard([&] {
    Engine& e = *(Engine*)h;
    REQ(e.has_vision && e.img_bytes, "engine built without GPU image preprocessing");
    REQ(npatch <= e.max_patch, "batch exceeds max patches");
    REQ(nslot <= e.max_patch / 4, "too many soft tokens");
    std::vector<int> c(vcu, vcu + nimg + 1), sp(slot_ptr, slot_ptr + nslot + 1), spp(slot_patch, slot_patch + sp[nslot]);
    ImgIn im;
    im.src = src;
    im.src_bytes = src_bytes;
    im.desc = desc;
    im.coef = coef;
    im.ncoef = ncoef;
    im.max_pix = max_pix;
    if (e.dt == DT_F32) gemma_vision<float>(e, npatch, nimg, c, pos, nullptr, im, nslot, sp, spp);
    else gemma_vision<bf16>(e, npatch, nimg, c, pos, nullptr, im, nslot, sp, spp);
  });
}

// NeoMME: text (pos2 = null) or document images (pos2 [ntok, 2] (row, col); patches [npatch, 3072] for rows
// patch_rows). dense [nseq, 1024]; late [ntok, 128] (nullable)
int fe_neomme(void* h, int ntok, int nseq, const int32_t* ids, const int32_t* cu, const int32_t* pos2, int npatch,
              const uint8_t* patches, const int32_t* patch_rows, float* dense, float* late) {
  return guard([&] {
    Engine& e = *(Engine*)h;
    if (e.dt == DT_F32) neomme<float>(e, ntok, nseq, ids, cu, pos2, npatch, patches, patch_rows, dense, late);
    else neomme<bf16>(e, ntok, nseq, ids, cu, pos2, npatch, patches, patch_rows, dense, late);
  });
}

// bytes: [0] weights, [1] arena, [2] device used (total - free), [3] total
int fe_mem(void* h, uint64_t* out) {
  return guard([&] {
    Engine& e = *(Engine*)h;
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    out[0] = e.weight_bytes;
    out[1] = e.arena_bytes;
    out[2] = tot - fr;
    out[3] = tot;
  });
}

int fe_device_name(char* buf, int n) {
  return guard([&] {
    int d = 0;
    CK(cudaGetDevice(&d));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, d));
    std::snprintf(buf, n, "%s", p.name);
  });
}

void fe_free(void* h) {
  Engine* e = (Engine*)h;
  if (!e) return;
  cudaStreamSynchronize(e->st);
  for (auto& kv : e->raw) cudaFree(kv.second.p);
  for (void* p : e->owned) cudaFree(p);
  if (e->pin) cudaFreeHost(e->pin);
  if (e->cb) cublasDestroy(e->cb);
  if (e->st) cudaStreamDestroy(e->st);
  delete e;
}

}  // extern "C"
