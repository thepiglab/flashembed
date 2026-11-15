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
#include <cublasLt.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
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

// optional cuDNN fused attention (cuda/fe_cudnn.cu; a stub unless built with FLASHEMBED_CUDNN_INCLUDE)
extern "C" {
const char* fecd_error();
void* fecd_new(void* stream);
int fecd_attn(void* ctx, int nb, int smax, int Hq, int Hkv, int D, int window, float scale, const void* q, const void* k,
              const void* v, void* o, int ld, int ld_out, const int* rag, const int* rago, const int* sl);
int fecd_graphs(void* ctx);
size_t fecd_bytes(void* ctx);
void fecd_free(void* ctx);
}
__global__ void k_cd_meta(const int* cu, int n, int nb, int ld, int ld_out, int* rag, int* rago, int* sl) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b > nb) return;
  const int c = cu[b < n ? b : n];
  rag[b] = c * ld;
  rago[b] = c * ld_out;
  if (b < nb) sl[b] = b < n ? cu[b + 1] - cu[b] : 0;
}

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
//   x_mode 4: h += xmul * rms(x) * w_post (Gemma 4 audio feed-forward, residual weight 0.5)
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
    if (x_mode == 1 || x_mode == 4) r = 1.f / sqrtf(block_sum(ss) / n + eps);
    k = 0;
    for (int c = threadIdx.x; c < n; c += blockDim.x, ++k) {
      if (x_mode == 1) v[k] += xv[k] * r * w_post[c];
      else if (x_mode == 4) v[k] += xmul * (xv[k] * r * w_post[c]);
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

// ---- block-scaled quantization for cuBLASLt narrow-precision GEMMs (FMT 1: MXFP8 = E4M3 values + UE8M0 scale per
// 32 K-elements; FMT 2: NVFP4 = E2M1 values (two per byte, lower K index in the low nibble) + UE4M3 scale per 16).
// Scales use cuBLASLt's tiled layout: 128 (outer) x 4 (scale columns) tiles of 512 bytes, row-major over tiles,
// inside a tile offset = (outer % 32) * 16 + (outer % 128 / 32) * 4 + inner % 4 (cuBLAS docs, "1D Block Scaling
// Factors Layout"). Scale rounding follows the docs' "Computing scaling and conversion factors" steps.
__device__ __forceinline__ size_t sf_offset(int outer, int kb, int sfk_pad) {
  return ((size_t)(outer / 128) * (sfk_pad / 4) + kb / 4) * 512 + (outer % 32) * 16 + ((outer % 128) / 32) * 4 + (kb % 4);
}
template <int FMT>
__global__ void k_quant_blocks(const bf16* __restrict__ x, int rows, int K, float gscale, uint8_t* __restrict__ q,
                               uint8_t* __restrict__ sf, int sfk_pad) {
  constexpr int BLK = FMT == 1 ? 32 : 16;
  const int nb = K / BLK;
  const long long total = (long long)rows * nb;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
    const int r = (int)(i / nb), b = (int)(i % nb);
    const bf16* p = x + (size_t)r * K + (size_t)b * BLK;
    float v[BLK];
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < BLK; j += 8) {
      const uint4 u = *(const uint4*)(p + j);
      const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        v[j + 2 * t] = __low2float(h2[t]) * gscale;
        v[j + 2 * t + 1] = __high2float(h2[t]) * gscale;
      }
    }
#pragma unroll
    for (int j = 0; j < BLK; ++j) amax = fmaxf(amax, fabsf(v[j]));
    uint8_t sbyte;
    float R;
    if (FMT == 1) {
      int e = 0;
      const float m = frexpf(amax / 448.f, &e);  // S = m * 2^e, m in [0.5, 1)
      int E = amax > 0.f ? (m == 0.5f ? e - 1 : e) : 0;  // round the exponent up unless S is a power of two
      E = max(-127, min(127, E));
      sbyte = (uint8_t)(E + 127);
      R = exp2f((float)-E);
      uint8_t* o = q + (size_t)r * K + (size_t)b * BLK;
#pragma unroll
      for (int j = 0; j < BLK; j += 16) {
        uint8_t tmp[16];
#pragma unroll
        for (int t = 0; t < 16; ++t) tmp[t] = __nv_cvt_float_to_fp8(v[j + t] * R, __NV_SATFINITE, __NV_E4M3);
        *(uint4*)(o + j) = *(const uint4*)tmp;
      }
    } else {
      sbyte = __nv_cvt_float_to_fp8(amax / 6.f, __NV_SATFINITE, __NV_E4M3);
      const float sq = __half2float(__half(__nv_cvt_fp8_to_halfraw(sbyte, __NV_E4M3)));
      R = sq > 0.f ? 1.f / sq : 0.f;
      uint8_t tmp[8];
#pragma unroll
      for (int t = 0; t < 8; ++t)
        tmp[t] = __nv_cvt_float2_to_fp4x2(make_float2(v[2 * t] * R, v[2 * t + 1] * R), __NV_E2M1, cudaRoundNearest);
      *(uint2*)(q + ((size_t)r * K + (size_t)b * BLK) / 2) = *(const uint2*)tmp;
    }
    sf[sf_offset(r, b, sfk_pad)] = sbyte;
  }
}
__global__ void k_amax_bf16(const bf16* __restrict__ x, size_t n, unsigned* __restrict__ out) {
  float a = 0.f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    a = fmaxf(a, fabsf(__bfloat162float(x[i])));
  a = warp_max(a);
  if ((threadIdx.x & 31) == 0) atomicMax(out, __float_as_uint(a));  // non-negative floats order like their bits
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
    const float r = (x_mode == 1 || x_mode == 4) ? 1.f / sqrtf(warp_sum(ss) / n + eps) : 1.f;
#pragma unroll
    for (int i = 0; i < NV; ++i) {
      const int c = (lane + 32 * i) * 4;
      float w[4] = {1.f, 1.f, 1.f, 1.f};
      if (x_mode == 1 || x_mode == 4) ld4(w_post + c, w);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (x_mode == 1) v[i][j] += xv[i][j] * r * w[j];
        else if (x_mode == 4) v[i][j] += xmul * (xv[i][j] * r * w[j]);
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

// ---- Gemma 4 audio tower (EmbeddingGemma 2 audio), on packed clips. Row-map arrays give, for every packed row of a
// stage, the packed start of its clip in the previous stage, that clip's length there and the row's local index.
// Subsampling conv 0: features [T0, 128] (time x mel) -> [T1, 64, 128] (channels last), 3x3 stride 2 pad 1, input
// frames multiplied by the frame mask, then LayerNorm over the 128 channels (no bias) + ReLU (CPU op ChannelLayerNorm)
__global__ void k_aconv0(const float* __restrict__ feats, const int* __restrict__ m0, const int* __restrict__ rows1,
                         const float* __restrict__ w, const float* __restrict__ nw, float eps, float* __restrict__ out) {
  const int r = blockIdx.x, f = blockIdx.y, c = threadIdx.x;  // 128 threads = channels
  const int base0 = rows1[3 * r], len0 = rows1[3 * r + 1], lt = rows1[3 * r + 2];
  float acc = 0.f;
#pragma unroll
  for (int dt = 0; dt < 3; ++dt) {
    const int t = 2 * lt - 1 + dt;
    if (t < 0 || t >= len0 || !m0[base0 + t]) continue;
#pragma unroll
    for (int df = 0; df < 3; ++df) {
      const int ff = 2 * f - 1 + df;
      if (ff < 0 || ff >= 128) continue;
      acc += w[c * 9 + dt * 3 + df] * feats[(size_t)(base0 + t) * 128 + ff];
    }
  }
  const float mean = block_sum(acc) / 128.f;
  const float d = acc - mean;
  const float var = block_sum(d * d) / 128.f;
  out[((size_t)r * 64 + f) * 128 + c] = fmaxf(d / sqrtf(var + eps) * nw[c], 0.f);
}
// Subsampling conv 1: [T1, 64, 128] -> [T2, 32 (freq), 32 (ch)] flattened to 1024 (= HF permute(0, 2, 3, 1)), in T
template <typename T>
__global__ void k_aconv1(const float* __restrict__ in, const int* __restrict__ m1, const int* __restrict__ rows2,
                         const float* __restrict__ w, const float* __restrict__ nw, float eps, T* __restrict__ out) {
  __shared__ float x[9][128];
  const int r = blockIdx.x, f = blockIdx.y, c = threadIdx.x;  // 32 threads = output channels
  const int base1 = rows2[5 * r], len1 = rows2[5 * r + 1], lt = rows2[5 * r + 2];
  for (int i = c; i < 9 * 128; i += 32) {
    const int tap = i / 128, ci = i % 128, t = 2 * lt - 1 + tap / 3, ff = 2 * f - 1 + tap % 3;
    const bool ok = t >= 0 && t < len1 && ff >= 0 && ff < 64 && m1[base1 + t];
    x[tap][ci] = ok ? in[((size_t)(base1 + t) * 64 + ff) * 128 + ci] : 0.f;
  }
  __syncthreads();
  float acc = 0.f;
  for (int ci = 0; ci < 128; ++ci)
#pragma unroll
    for (int tap = 0; tap < 9; ++tap) acc += w[(c * 128 + ci) * 9 + tap] * x[tap][ci];
  const float mean = warp_sum(acc) / 32.f;
  const float d = acc - mean;
  const float var = warp_sum(d * d) / 32.f;
  out[(size_t)r * 1024 + f * 32 + c] = fromf<T>(fmaxf(d / sqrtf(var + eps) * nw[c], 0.f));
}
template <typename T>
__global__ void k_clamp(const T* __restrict__ a, T* __restrict__ b, size_t n, float lo, float hi) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    b[i] = fromf<T>(fminf(fmaxf(tof(a[i]), lo), hi));
}
template <typename T>
__global__ void k_silu(T* __restrict__ x, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const float v = tof(x[i]);
    x[i] = fromf<T>(v / (1.f + expf(-v)));
  }
}
// light conv: GLU over [a | b] (2 x 1024) -> causal depthwise conv (kernel 5, left pad 4, within the clip) ->
// RMSNorm(conv_norm) -> SiLU, one row per block (256 threads x 4 channels)
template <typename T>
__global__ void k_lconv_mid(const T* __restrict__ ab, const int* __restrict__ rows2, const float* __restrict__ dw,
                            const float* __restrict__ nw, float eps, T* __restrict__ out) {
  const int r = blockIdx.x, lt = rows2[5 * r + 2];
  float v[4];
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int c = threadIdx.x + 256 * k;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 5; ++j) {
      const int back = 4 - j;
      if (lt - back < 0) continue;
      const size_t row = (size_t)(r - back) * 2048;
      const float a = tof(ab[row + c]), b = tof(ab[row + 1024 + c]);
      acc += dw[c * 5 + j] * (a / (1.f + expf(-b)));
    }
    acc = fminf(fmaxf(acc, -1e10f), 1e10f);
    v[k] = acc;
    ss += acc * acc;
  }
  const float rr = 1.f / sqrtf(block_sum(ss) / 1024.f + eps);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int c = threadIdx.x + 256 * k;
    const float y = v[k] * rr * nw[c];
    out[(size_t)r * 1024 + c] = fromf<T>(y / (1.f + expf(-y)));
  }
}
// chunked local attention (port of the CPU op AudioChunkAttention): block = one query row, warp = head (8 x 128).
// Query i of chunk b sees context slots o = 0..23 (key j = 12 b + o - 12, clip-local); logit = q.k_j + q.relk[o - i]
// (when 0 <= o - i <= 12), soft-capped at 50; slots out of the clip, out of the 12-frame window or masked get the
// `invalid` logit; out-of-clip slots contribute no value.
template <typename T>
__global__ void k_audio_attn(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
                             const float* __restrict__ qsc, float ksc, const float* __restrict__ relk,
                             const int* __restrict__ m2, const int* __restrict__ rows2, float softcap, float invalid,
                             T* __restrict__ out) {
  const int r = blockIdx.x, h = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int lt = rows2[5 * r + 2], base = rows2[5 * r + 3], len = rows2[5 * r + 4];
  const int blk = lt / 12, i = lt % 12, hd = h * 128 + lane * 4;
  float qv[4];
#pragma unroll
  for (int d = 0; d < 4; ++d) qv[d] = tof(q[(size_t)r * 1024 + hd + d]) * qsc[lane * 4 + d];
  float lg[24];
#pragma unroll
  for (int o = 0; o < 24; ++o) {
    const int j = blk * 12 + o - 12;
    const bool inr = j >= 0 && j < len;
    float part = 0.f;
    if (inr) {
#pragma unroll
      for (int d = 0; d < 4; ++d) part += qv[d] * (tof(k[(size_t)(base + j) * 1024 + hd + d]) * ksc);
    }
    const int p = o - i;
    if (p >= 0 && p <= 12) {
#pragma unroll
      for (int d = 0; d < 4; ++d) part += qv[d] * relk[p * 1024 + hd + d];
    }
    const float acc = warp_sum(part);
    const float capped = tanhf(acc / softcap) * softcap;
    const int dd = lt - j;
    const bool ok = inr && dd >= 0 && dd < 12 && m2[base + j];
    lg[o] = ok ? capped : invalid;
  }
  float mx = -INFINITY;
#pragma unroll
  for (int o = 0; o < 24; ++o) mx = fmaxf(mx, lg[o]);
  float z = 0.f;
#pragma unroll
  for (int o = 0; o < 24; ++o) {
    lg[o] = expf(lg[o] - mx);
    z += lg[o];
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int o = 0; o < 24; ++o) {
    const int j = blk * 12 + o - 12;
    if (j < 0 || j >= len) continue;
    const float wgt = lg[o] / z;
#pragma unroll
    for (int d = 0; d < 4; ++d) acc[d] += wgt * tof(v[(size_t)(base + j) * 1024 + hd + d]);
  }
#pragma unroll
  for (int d = 0; d < 4; ++d) out[(size_t)r * 1024 + hd + d] = fromf<T>(acc[d]);
}
// output_proj bias + embed_audio pre-norm (RMS, no weight) for the soft-token rows, compacted
template <typename T>
__global__ void k_bias_rms_gather(const T* __restrict__ x, const float* __restrict__ bias, const int* __restrict__ rows,
                                  float eps, T* __restrict__ out, int n) {
  const int s = blockIdx.x;
  const size_t src = rows[s];
  float v[8];
  int k = 0;
  float ss = 0.f;
  for (int c = threadIdx.x; c < n; c += blockDim.x, ++k) {
    v[k] = tof(x[src * n + c]) + bias[c];
    ss += v[k] * v[k];
  }
  const float r = 1.f / sqrtf(block_sum(ss) / n + eps);
  k = 0;
  for (int c = threadIdx.x; c < n; c += blockDim.x, ++k) out[(size_t)s * n + c] = fromf<T>(v[k] * r);
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
  __shared__ float scos[256], ssin[256];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int t = blockIdx.x;
  // cos/sin depend only on (token, frequency): one sincosf per table entry for the block's token, shared by every
  // head and both channels of each rotated pair (HF computes the same pos * inv_freq products in FP32)
  const int nk = rope.type == 3 ? rope.rot : rope.rot / 2;
  if (rope.type && t < ntok)
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
      int k = i, axis = rope.type == 2 ? (i & 1) : 0;
      if (rope.type == 3) {
        axis = i / (rope.rot / 2);
        k = i % (rope.rot / 2);
      }
      const float ang = (float)pos[(size_t)t * naxes + (axis < naxes ? axis : 0)] * rope.inv[k];
      sincosf(ang, &ssin[i], &scos[i]);
    }
  __syncthreads();
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
      const int ti = rope.type == 3 ? axis * (rope.rot / 2) + k : k;
      const float s = ssin[ti], co = scos[ti];
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
// late chunking: mean of rows lo[s] .. hi[s]-1 (packed token indices; spans may overlap)
__global__ void k_span_pool(const float* __restrict__ x, const int* __restrict__ lo, const int* __restrict__ hi,
                            float* __restrict__ out, int n) {
  const int s = blockIdx.x;
  const int a = lo[s], b = hi[s];
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

// ---- FP32 attention on tensor cores, 3xTF32 (mma.sync m16n8k8 tf32): every FP32 operand x is split into
// hi = tf32(x) and lo = tf32(x - hi); a*b ~= hi_a*hi_b + hi_a*lo_b + lo_a*hi_b with FP32 accumulation, i.e. FP32-level
// accuracy (the dropped lo*lo term is ~2^-22 relative) at tensor-core speed. Same tiling as k_attn_mma (4 warps x 16
// query rows); Q fragments are read straight from global/L1, K/V tiles go through shared memory (cp.async).
__device__ __forceinline__ uint32_t tf32_hi(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ void split_tf32(float x, uint32_t& hi, uint32_t& lo) {
  hi = tf32_hi(x);
  lo = tf32_hi(x - __uint_as_float(hi));
}
__device__ __forceinline__ void mma1688(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// c += a * b in 3xTF32 (a: 4 floats A fragment, b: 2 floats B fragment)
__device__ __forceinline__ void mma3x(float* c, const float* a, float b0, float b1) {
  uint32_t ah[4], al[4], bh0, bl0, bh1, bl1;
#pragma unroll
  for (int i = 0; i < 4; ++i) split_tf32(a[i], ah[i], al[i]);
  split_tf32(b0, bh0, bl0);
  split_tf32(b1, bh1, bl1);
  mma1688(c, al, bh0, bh1);
  mma1688(c, ah, bl0, bl1);
  mma1688(c, ah, bh0, bh1);
}

template <int D, int DV, int BC, int STAGES>
__global__ void __launch_bounds__(128) k_attn_tf32(AttnArgs a) {
  constexpr int BR = 64, SK = D + 4, SV = DV + 8, NT = BC / 8, NO = DV / 8;
  extern __shared__ __align__(16) unsigned char smraw[];
  float* Ks0 = (float*)smraw;              // STAGES x [BC][SK]
  float* Vs0 = Ks0 + STAGES * BC * SK;     // STAGES x [BC][SV]
  const int2 tile = a.tiles[blockIdx.x];
  const int s = tile.x, q0 = tile.y, h = blockIdx.y, dv0 = blockIdx.z * DV;
  const int hk = h / (a.Hq / a.Hkv);
  const int start = a.cu[s], len = a.cu[s + 1] - start;
  const float* base = (const float*)a.qkv;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  int lo = 0, hi = len;
  if (a.window >= 0) {
    lo = max(0, q0 - a.window);
    hi = min(len, q0 + BR - 1 + a.window + 1);
  }
  auto load_kv = [&](int stage, int k0) {
    float* Ks = Ks0 + stage * BC * SK;
    float* Vs = Vs0 + stage * BC * SV;
    for (int i = tid; i < BC * D / 4; i += 128) {
      const int r = i / (D / 4), c = (i % (D / 4)) * 4, kk = k0 + r;
      const bool ok = kk < hi;
      cp_async16(Ks + r * SK + c, base + (size_t)(start + (ok ? kk : 0)) * a.ld + a.k_off + hk * D + c, ok);
    }
    for (int i = tid; i < BC * DV / 4; i += 128) {
      const int r = i / (DV / 4), c = (i % (DV / 4)) * 4, kk = k0 + r;
      const bool ok = kk < hi;
      cp_async16(Vs + r * SV + c, base + (size_t)(start + (ok ? kk : 0)) * a.ld + a.v_off + hk * D + dv0 + c, ok);
    }
  };
  load_kv(0, lo);
  cp_async_commit();
  const int qr0 = q0 + warp * 16 + g, qr1 = qr0 + 8;
  const float* q0p = base + (size_t)(start + min(qr0, len - 1)) * a.ld + a.q_off + h * D;
  const float* q1p = base + (size_t)(start + min(qr1, len - 1)) * a.ld + a.q_off + h * D;
  float o[NO][4];
#pragma unroll
  for (int n = 0; n < NO; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float mrow0 = -INFINITY, mrow1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const float sl2 = a.scale * 1.4426950408889634f;
  const int src1 = (lane & ~3) | (t4 >> 1), src2 = src1 + 2;
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
    const float* Ks = Ks0 + st * BC * SK;
    const float* Vs = Vs0 + st * BC * SV;
    float sc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
#pragma unroll 4
    for (int kk = 0; kk < D; kk += 8) {
      const float af[4] = {__ldg(q0p + kk + t4), __ldg(q1p + kk + t4), __ldg(q0p + kk + t4 + 4), __ldg(q1p + kk + t4 + 4)};
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const float* kr = Ks + (j * 8 + g) * SK + kk + t4;
        mma3x(sc[j], af, kr[0], kr[4]);
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
    for (int j = 0; j < NT; ++j) {
      // S accumulator layout (cols 2t, 2t+1) -> tf32 A fragment layout (cols t, t + 4) via quad shuffles
      const float v0 = __shfl_sync(0xffffffffu, sc[j][0], src1), v1 = __shfl_sync(0xffffffffu, sc[j][1], src1);
      const float v2 = __shfl_sync(0xffffffffu, sc[j][2], src1), v3 = __shfl_sync(0xffffffffu, sc[j][3], src1);
      const float w0 = __shfl_sync(0xffffffffu, sc[j][0], src2), w1 = __shfl_sync(0xffffffffu, sc[j][1], src2);
      const float w2 = __shfl_sync(0xffffffffu, sc[j][2], src2), w3 = __shfl_sync(0xffffffffu, sc[j][3], src2);
      const bool odd = t4 & 1;
      const float pa[4] = {odd ? v1 : v0, odd ? v3 : v2, odd ? w1 : w0, odd ? w3 : w2};
#pragma unroll
      for (int n = 0; n < NO; ++n) {
        const float* vr = Vs + (j * 8 + t4) * SV + n * 8 + g;
        mma3x(o[n], pa, vr[0], vr[4 * SV]);
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
#pragma unroll
  for (int n = 0; n < NO; ++n) {
    o[n][0] *= i0;
    o[n][1] *= i0;
    o[n][2] *= i1;
    o[n][3] *= i1;
  }
  float* out = (float*)a.out;
  if (a.alpha_t) {  // NeoMME epilogue (DV == D)
    const float ta = a.alpha_t[h];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int qr = half ? qr1 : qr0;
      const bool valid = qr < len;
      const size_t row = (size_t)(start + (valid ? qr : 0));
      float vn = 0.f, dot = 0.f, vv[NO][2];
#pragma unroll
      for (int n = 0; n < NO; ++n) {
        const int col = n * 8 + 2 * t4;
        const float2 v2 = *(const float2*)(base + row * a.ld + a.v_off + hk * D + col);
        vv[n][0] = v2.x;
        vv[n][1] = v2.y;
        vn += v2.x * v2.x + v2.y * v2.y;
        dot += o[n][2 * half] * v2.x + o[n][2 * half + 1] * v2.y;
      }
      vn = quad_sum(vn);
      dot = quad_sum(dot);
      const float nrm = fmaxf(sqrtf(vn), 1e-12f), proj = dot / (nrm * nrm);
#pragma unroll
      for (int n = 0; n < NO; ++n) {
        const int col = n * 8 + 2 * t4;
        const float2 g2 = *(const float2*)(base + row * a.ld + a.g_off + h * D + col);
        const float x0 = (o[n][2 * half] - ta * proj * vv[n][0]) / (1.f + expf(-g2.x));
        const float x1 = (o[n][2 * half + 1] - ta * proj * vv[n][1]) / (1.f + expf(-g2.y));
        if (valid) *(float2*)(out + row * a.ld_out + h * D + col) = make_float2(x0, x1);
      }
    }
    return;
  }
#pragma unroll
  for (int n = 0; n < NO; ++n) {
    const int col = h * D + dv0 + n * 8 + 2 * t4;
    if (qr0 < len) *(float2*)(out + (size_t)(start + qr0) * a.ld_out + col) = make_float2(o[n][0], o[n][1]);
    if (qr1 < len) *(float2*)(out + (size_t)(start + qr1) * a.ld_out + col) = make_float2(o[n][2], o[n][3]);
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
  // block-scaled narrow-precision copy (--quant fp8 / nvfp4): data [n, k] (fp8 bytes or packed fp4), scales in
  // cuBLASLt's tiled layout, alpha = 1 / per-tensor pre-scale
  void* wq = nullptr;
  uint8_t* wsf = nullptr;
  float alpha = 1.f;
  int qfmt = 0;
  float clip[4] = {-INFINITY, INFINITY, -INFINITY, INFINITY};  // Gemma 4 clippable linear: input / output bounds
};

struct ALayer {  // Gemma 4 audio conformer layer
  Lin f1a, f1b, f2a, f2b, ls, le, q, k, v, post;
  float *f1_pre, *f1_post, *f2_pre, *f2_post, *lc_pre, *lc_conv, *pre_attn, *post_attn, *out;
  float *dw, *qsc, *relk;  // depthwise conv [1024 x 5], q scale per dim, relative keys [13 x 1024] (fp32)
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
  int quant = 0;  // 0 off, 1 MXFP8 (E4M3 + UE8M0 per 32), 2 NVFP4 (E2M1 + UE4M3 per 16): GEMM inputs only
  cublasLtHandle_t lt = nullptr;
  uint8_t *qx = nullptr, *qsf = nullptr;  // quantized activations + their scales
  size_t qx_bytes = 0, qsf_bytes = 0;
  std::unordered_map<std::string, cublasLtMatmulAlgo_t> algos;
  cudaStream_t st = nullptr;
  cublasHandle_t cb = nullptr;
  void* cd = nullptr;  // cuDNN attention context (bf16), cd_state: 0 untried, 1 on, -1 off
  int cd_state = 0;
  int* span_buf = nullptr;  // late chunking: span lo | hi, span_cap each
  int span_cap = 0;
  int* cd_meta = nullptr;  // ragged offsets / lengths for cuDNN, 3 x cd_cap ints
  int cd_cap = 0;
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
  // audio tower
  bool has_audio = false;
  int max_frames = 0;
  float a_ksc = 1.f, a_softcap = 50.f, a_invalid = -1e9f, a_resw = 0.5f;
  float *a_c0w = nullptr, *a_c0n = nullptr, *a_c1w = nullptr, *a_c1n = nullptr, *a_out_b = nullptr;
  Lin a_in, a_outp, a_emb;
  std::vector<ALayer> al;
  float *a_feat = nullptr, *a_l0 = nullptr, *a_h = nullptr;
  void *a_x = nullptr, *a_y = nullptr, *a_big = nullptr, *a_tmp = nullptr, *a_q = nullptr, *a_k = nullptr, *a_v = nullptr,
       *a_o = nullptr, *a_t1 = nullptr, *a_proj = nullptr, *a_e = nullptr, *a_soft = nullptr;
  const void* soft_src = nullptr;  // soft tokens for the next text forward (vision or audio)
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

inline unsigned nblk(size_t n, int per = 256) { return (unsigned)std::min<size_t>((n + per - 1) / per, 65535 * 8); }

inline int qblk(int fmt) { return fmt == 1 ? 32 : 16; }
inline int sfk_pad(int K, int fmt) { return (K / qblk(fmt) + 3) / 4 * 4; }
inline size_t sf_bytes(int rows, int K, int fmt) { return (size_t)((rows + 127) / 128) * 128 * sfk_pad(K, fmt); }

void quantize(Engine& e, const bf16* x, int rows, int K, int fmt, float gscale, void* q, uint8_t* sf) {
  const int nb = K / qblk(fmt);
  const size_t total = (size_t)rows * nb;
  // tiles past the last row must read as zero scales
  const size_t full = (size_t)(rows / 128) * (sfk_pad(K, fmt) / 4) * 512, all = sf_bytes(rows, K, fmt);
  if (all > full) CK(cudaMemsetAsync(sf + full, 0, all - full, e.st));
  if (fmt == 1)
    k_quant_blocks<1><<<nblk(total), 256, 0, e.st>>>(x, rows, K, gscale, (uint8_t*)q, sf, sfk_pad(K, fmt));
  else
    k_quant_blocks<2><<<nblk(total), 256, 0, e.st>>>(x, rows, K, gscale, (uint8_t*)q, sf, sfk_pad(K, fmt));
  CK(cudaGetLastError());
}

// Y[M, n] (bf16) = (block-scaled X) * (block-scaled W)^T on cuBLASLt (TN, FP32 compute)
void gemm_q(Engine& e, const void* X, const Lin& W, void* Y, int M) {
  const int K = W.k, fmt = W.qfmt;
  REQ((size_t)M * K / (fmt == 1 ? 1 : 2) <= e.qx_bytes && sf_bytes(M, K, fmt) <= e.qsf_bytes,
      "quantized activation buffer too small");
  quantize(e, (const bf16*)X, M, K, fmt, 1.f, e.qx, e.qsf);
  const cudaDataType_t qt = fmt == 1 ? CUDA_R_8F_E4M3 : CUDA_R_4F_E2M1;
  cublasLtMatmulDesc_t d;
  CB(cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
  const int32_t mode = fmt == 1 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 : CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  const void *sa = W.wsf, *sb = e.qsf;
  CB(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
  CB(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
  CB(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
  CB(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));
  CB(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof(sa)));
  CB(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof(sb)));
  cublasLtMatrixLayout_t la, lb, lc;
  CB(cublasLtMatrixLayoutCreate(&la, qt, K, W.n, K));
  CB(cublasLtMatrixLayoutCreate(&lb, qt, K, M, K));
  CB(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, W.n, M, W.n));
  const std::string key = std::to_string(W.n) + "x" + std::to_string(K) + "x" + std::to_string(M) + "f" + std::to_string(fmt);
  auto it = e.algos.find(key);
  if (it == e.algos.end()) {
    cublasLtMatmulPreference_t pref;
    CB(cublasLtMatmulPreferenceCreate(&pref));
    const size_t wsb = 64ull << 20;
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof(wsb)));
    cublasLtMatmulHeuristicResult_t res;
    int got = 0;
    const cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(e.lt, d, la, lb, lc, lc, pref, 1, &res, &got);
    cublasLtMatmulPreferenceDestroy(pref);
    REQ(hs == CUBLAS_STATUS_SUCCESS && got > 0,
        "cuBLASLt has no " + std::string(fmt == 1 ? "MXFP8" : "NVFP4") + " kernel for " + key + " on this GPU");
    it = e.algos.emplace(key, res.algo).first;
  }
  const float al = W.alpha, be = 0.f;
  CB(cublasLtMatmul(e.lt, d, &al, W.wq, la, e.qx, lb, &be, Y, lc, Y, lc, &it->second, e.ws, 64ull << 20, e.st));
  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatmulDescDestroy(d);
}

// Y[M, n] = X[M, k] * W[n, k]^T  (row-major everything)
void gemm(Engine& e, const void* X, const Lin& W, void* Y, int M, int dt = -1) {
  if (M == 0) return;
  if (dt < 0) dt = e.dt;
  if (W.qfmt && dt == DT_BF16) return gemm_q(e, X, W, Y, M);
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



void dfree(Engine& e, void* p) {
  auto it = std::find(e.owned.begin(), e.owned.end(), p);
  if (it != e.owned.end()) {
    e.owned.erase(it);
    cudaFree(p);
  }
}

// Narrow-precision copy of a bf16 weight (the bf16 copy is released). NVFP4 weights get a per-tensor pre-scale
// g = 448 * 6 / amax so that block scales use the UE4M3 range; alpha = 1 / g undoes it in the GEMM.
void quantize_lin(Engine& e, Lin& L, int fmt) {
  if (!L.w || !L.n) return;
  REQ(e.dt == DT_BF16, "--quant needs --dtype bf16");
  REQ(L.k % 32 == 0, "quantized GEMM needs K % 32 == 0");
  float g = 1.f;
  if (fmt == 2) {
    unsigned* d = (unsigned*)dmalloc(e, 4);
    CK(cudaMemsetAsync(d, 0, 4, e.st));
    k_amax_bf16<<<1024, 256, 0, e.st>>>((const bf16*)L.w, (size_t)L.n * L.k, d);
    unsigned h = 0;
    CK(cudaMemcpyAsync(&h, d, 4, cudaMemcpyDeviceToHost, e.st));
    CK(cudaStreamSynchronize(e.st));
    dfree(e, d);
    float amax;
    std::memcpy(&amax, &h, 4);
    g = amax > 0.f ? 448.f * 6.f / amax : 1.f;
  }
  const size_t qb = fmt == 1 ? (size_t)L.n * L.k : (size_t)L.n * L.k / 2, sb = sf_bytes(L.n, L.k, fmt);
  L.wq = dmalloc(e, qb);
  L.wsf = (uint8_t*)dmalloc(e, sb);
  CK(cudaMemsetAsync(L.wsf, 0, sb, e.st));
  quantize(e, (const bf16*)L.w, L.n, L.k, fmt, g, L.wq, L.wsf);
  CK(cudaStreamSynchronize(e.st));
  e.weight_bytes += qb + sb;
  e.weight_bytes -= (size_t)L.n * L.k * 2;
  dfree(e, L.w);
  L.w = nullptr;
  L.alpha = 1.f / g;
  L.qfmt = fmt;
}

// rows [r0, r0 + n) of a weight (EmbeddingGemma 2's per-layer slice of the PLE projection)
Lin slice_rows(const Engine& e, const Lin& L, int r0, int n) {
  Lin s = L;
  s.n = n;
  if (L.qfmt) {
    REQ(r0 % 128 == 0, "quantized row slice must start on a 128-row scale tile");
    s.wq = (char*)L.wq + (size_t)r0 * L.k / (L.qfmt == 1 ? 1 : 2);
    s.wsf = L.wsf + (size_t)(r0 / 128) * (sfk_pad(L.k, L.qfmt) / 4) * 512;
  } else {
    s.w = (char*)L.w + (size_t)r0 * L.k * e.esz();
  }
  return s;
}

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
  int nseq = 0, smax = 0, ntok = 0;  // host copies (cuDNN attention)
};
void set_seqs(Meta& m, const std::vector<int>& cu) {
  m.nseq = (int)cu.size() - 1;
  m.ntok = cu.empty() ? 0 : cu.back();
  m.smax = 0;
  for (int s = 0; s < m.nseq; ++s) m.smax = std::max(m.smax, cu[s + 1] - cu[s]);
}

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

template <int D, int DV, int BC, int STAGES>
void attn_tf32_t(Engine& e, const AttnArgs& a, int ntiles) {
  const size_t smem = (size_t)STAGES * BC * ((D + 4) + (DV + 8)) * 4;
  static bool set = false;
  if (!set) {
    CK(cudaFuncSetAttribute(k_attn_tf32<D, DV, BC, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    set = true;
  }
  dim3 grid(ntiles, a.Hq, D / DV);
  k_attn_tf32<D, DV, BC, STAGES><<<grid, 128, smem, e.st>>>(a);
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

// NeoMME after cuDNN attention: exclusive self-attention against the token's own value vector, then the sigmoid
// gate (the same math as the k_attn_mma epilogue, from the bf16 attention output). One warp per (token, head).
template <int D>
__global__ void k_excl_gate(AttnArgs a, int ntok) {
  constexpr int CPL = D / 32;
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (w >= ntok * a.Hq) return;
  const int row = w / a.Hq, h = w % a.Hq, hk = h / (a.Hq / a.Hkv);
  const bf16* base = (const bf16*)a.qkv + (size_t)row * a.ld;
  bf16* out = (bf16*)a.out + (size_t)row * a.ld_out + h * D;
  float o[CPL], v[CPL], vn = 0.f, dot = 0.f;
#pragma unroll
  for (int c = 0; c < CPL; ++c) {
    o[c] = __bfloat162float(out[lane + 32 * c]);
    v[c] = __bfloat162float(base[a.v_off + hk * D + lane + 32 * c]);
    vn += v[c] * v[c];
    dot += o[c] * v[c];
  }
  vn = warp_sum(vn);
  dot = warp_sum(dot);
  const float nrm = fmaxf(sqrtf(vn), 1e-12f), proj = dot / (nrm * nrm), ta = a.alpha_t[h];
#pragma unroll
  for (int c = 0; c < CPL; ++c) {
    const float g = __bfloat162float(base[a.g_off + h * D + lane + 32 * c]);
    out[lane + 32 * c] = __float2bfloat16((o[c] - ta * proj * v[c]) / (1.f + expf(-g)));
  }
}

// cuDNN attention with FLASHEMBED_ATTN=cudnn (opt-in while it is being evaluated); off for the run after a failure
bool use_cudnn(Engine& e) {
  if (e.cd_state == 0) {
    const char* x = getenv("FLASHEMBED_ATTN");
    const std::string v = x ? x : "";
    if (v != "cudnn") {
      e.cd_state = -1;
    } else {
      e.cd = fecd_new(e.st);
      e.cd_state = e.cd ? 1 : -1;
      if (!e.cd && v == "cudnn") throw FeErr(std::string("FLASHEMBED_ATTN=cudnn: ") + fecd_error());
    }
  }
  return e.cd_state == 1;
}

template <typename T>
void attention(Engine& e, int D, const AttnArgs& a0, const Meta& m) {
  AttnArgs a = a0;
  if (std::is_same<T, bf16>::value && m.nseq > 0 && use_cudnn(e)) {
    const bf16* b = (const bf16*)a.qkv;
    int nb = 1;
    while (nb < m.nseq) nb <<= 1;  // power-of-two sequence slots: few distinct cuDNN graphs
    if (nb + 1 > e.cd_cap) {
      if (e.cd_meta) CK(cudaFree(e.cd_meta));
      e.cd_cap = std::max(nb + 1, 1024);
      CK(cudaMalloc(&e.cd_meta, (size_t)3 * e.cd_cap * sizeof(int)));
    }
    int *rag = e.cd_meta, *rago = rag + e.cd_cap, *sl = rago + e.cd_cap;
    k_cd_meta<<<(nb + 1 + 255) / 256, 256, 0, e.st>>>(a.cu, m.nseq, nb, a.ld, a.ld_out, rag, rago, sl);
    CK(cudaGetLastError());
    if (fecd_attn(e.cd, nb, m.smax, a.Hq, a.Hkv, D, a.window, a.scale, b + a.q_off, b + a.k_off, b + a.v_off, a.out,
                  a.ld, a.ld_out, rag, rago, sl) == 0) {
      if (a.alpha_t) {
        REQ(D == 64 || D == 128 || D == 256, "exclusive attention head dim");
        const size_t thr = (size_t)m.ntok * a.Hq * 32;
        const int nb = (int)((thr + 255) / 256);
        if (D == 64) k_excl_gate<64><<<nb, 256, 0, e.st>>>(a, m.ntok);
        else if (D == 128) k_excl_gate<128><<<nb, 256, 0, e.st>>>(a, m.ntok);
        else k_excl_gate<256><<<nb, 256, 0, e.st>>>(a, m.ntok);
        CK(cudaGetLastError());
      }
      return;
    }
    if (getenv("FLASHEMBED_ATTN") && std::string(getenv("FLASHEMBED_ATTN")) == "cudnn")
      throw FeErr(std::string("cuDNN attention: ") + fecd_error());
    fprintf(stderr, "flashembed-gpu: cuDNN attention unavailable for this shape (%s); using the built-in kernels\n",
            fecd_error());
    e.cd_state = -1;
  }
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
  if (std::is_same<T, float>::value && !force_simt()) {  // FP32: 3xTF32 tensor-core attention
    a.tiles = m.tiles64;
    if (!m.nt64) return;
    const bool big = smem_optin() >= 170 * 1024;
    switch (D) {
      case 64: return attn_tf32_t<64, 64, 64, 2>(e, a, m.nt64);
      case 256: return big ? attn_tf32_t<256, 256, 32, 2>(e, a, m.nt64) : attn_tf32_t<256, 256, 16, 2>(e, a, m.nt64);
      case 512: return big ? attn_tf32_t<512, 256, 32, 2>(e, a, m.nt64) : attn_tf32_t<512, 256, 16, 2>(e, a, m.nt64);
    }
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
  e.meta_ints = (T * 12 + 64 * 1024 + (size_t)e.max_patch * 4) + 4096 + (e.has_audio ? (size_t)e.max_frames * 7 + 8192 : 0);
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
  if (e.has_audio) {
    const size_t F0 = e.max_frames, F1 = F0 / 2 + 1024, F2 = F0 / 4 + 1024;
    add(F0 * 128 * 4);                 // features
    add(F1 * 64 * 128 * 4);            // conv 0 output
    add(F2 * 1024 * 4);                // h
    add(F2 * 1024 * es * 8);           // x, y, q, k, v, o, t1, tmp(1024)
    add(F2 * 4096 * es * 2);           // big, tmp
    add(F2 * 1536 * es * 2);           // proj, e
    add(F2 * 512 * es);                // soft
  }
  if (e.quant) {
    int kmax = 0;
    for (auto& l : e.gl) kmax = std::max({kmax, l.qkv.k, l.o.k, l.gu.k, l.down.k});
    for (auto& l : e.vl) kmax = std::max({kmax, l.qkv.k, l.o.k, l.gu.k, l.down.k});
    for (auto& l : e.nl) kmax = std::max({kmax, l.qkvg.k, l.o.k, l.up.k, l.down.k});
    const int rows = std::max(e.max_tok, e.has_vision ? e.max_patch : 0);
    e.qx_bytes = (size_t)rows * kmax / (e.quant == 1 ? 1 : 2);
    e.qsf_bytes = sf_bytes(rows, kmax, e.quant);
    add(e.qx_bytes);
    add(e.qsf_bytes);
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
  if (e.quant) {
    e.qx = (uint8_t*)carve(e, e.qx_bytes);
    e.qsf = (uint8_t*)carve(e, e.qsf_bytes);
  }
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
  if (e.has_audio) {
    const size_t F0 = e.max_frames, F1 = F0 / 2 + 1024, F2 = F0 / 4 + 1024;
    e.a_feat = (float*)carve(e, F0 * 128 * 4);
    e.a_l0 = (float*)carve(e, F1 * 64 * 128 * 4);
    e.a_h = (float*)carve(e, F2 * 1024 * 4);
    void** bufs[] = {&e.a_x, &e.a_y, &e.a_q, &e.a_k, &e.a_v, &e.a_o, &e.a_t1};
    for (void** b : bufs) *b = carve(e, F2 * 1024 * es);
    carve(e, F2 * 1024 * es);
    e.a_big = carve(e, F2 * 4096 * es);
    e.a_tmp = carve(e, F2 * 4096 * es);
    e.a_proj = carve(e, F2 * 1536 * es);
    e.a_e = carve(e, F2 * 1536 * es);
    e.a_soft = carve(e, F2 * 512 * es);
  }
  // pinned staging: metadata + largest float input (patches)
  e.pin_bytes = e.meta_ints * 4 + 64 + (e.has_audio ? (size_t)e.max_frames * 128 * 4 : 0) +
                (e.arch == ARCH_NEOMME ? (size_t)e.max_patch * 3072
                                       : (e.has_vision ? std::max<size_t>((size_t)e.max_patch * 768 * 4, e.img_bytes + (4 << 20)) : 0));
  CK(cudaMallocHost((void**)&e.pin, e.pin_bytes));
}

// ------------------------------------------------------------------------------------------------- build

// ---- Gemma 4 audio tower weights
void set_clip(Engine& e, Lin& L, const std::string& p) {  // Gemma4ClippableLinear bounds (inf = no clamp)
  if (!has(e, p + "input_min")) return;
  L.clip[0] = host_scalar(e, p + "input_min");
  L.clip[1] = host_scalar(e, p + "input_max");
  L.clip[2] = host_scalar(e, p + "output_min");
  L.clip[3] = host_scalar(e, p + "output_max");
}
Lin clip_lin_w(Engine& e, const std::string& p) {
  Lin L = make_lin(e, {p + "linear.weight"});
  set_clip(e, L, p);
  return L;
}
void build_audio(Engine& e) {
  const std::string A = "audio_tower.";
  REQ((int)cfgv(e, "a_hidden") == 1024 && (int)cfgv(e, "a_heads") == 8 && (int)cfgv(e, "a_chunk") == 12 &&
          (int)cfgv(e, "a_left") == 13 && (int)cfgv(e, "a_right") == 0 && (int)cfgv(e, "a_kernel") == 5,
      "unsupported Gemma 4 audio config");
  e.has_audio = true;
  const int L = (int)cfgv(e, "a_layers");
  e.a_softcap = (float)cfgv(e, "a_softcap");
  e.a_invalid = (float)cfgv(e, "a_invalid");
  e.a_resw = (float)cfgv(e, "a_resw");
  const float eps = (float)cfgv(e, "a_eps");
  REQ(fabsf(eps - e.eps) < 1e-12f, "audio eps differs from the text eps");
  e.a_c0w = to_f32(e, A + "subsample_conv_projection.layer0.conv.weight");
  e.a_c0n = to_f32(e, A + "subsample_conv_projection.layer0.norm.weight");
  e.a_c1w = to_f32(e, A + "subsample_conv_projection.layer1.conv.weight");
  e.a_c1n = to_f32(e, A + "subsample_conv_projection.layer1.norm.weight");
  e.a_in = make_lin(e, {A + "subsample_conv_projection.input_proj_linear.weight"});
  e.a_outp = make_lin(e, {A + "output_proj.weight"});
  e.a_out_b = to_f32(e, A + "output_proj.bias");
  e.a_emb = make_lin(e, {"embed_audio.embedding_projection.weight"});
  // relative position table: positions 12..0, [sin | cos] over 512 timescales (Gemma4AudioRelPositionalEncoding)
  const int nts = 512;
  const double inc = std::log(10000.0) / (nts - 1);
  std::vector<float> pe(13 * 1024);
  for (int r = 0; r < 13; ++r)
    for (int i = 0; i < nts; ++i) {
      const float inv = (float)std::exp((float)(i * -inc));  // torch: exp of a float32 tensor
      const float st = (float)(12 - r) * inv;
      pe[r * 1024 + i] = std::sin(st);
      pe[r * 1024 + nts + i] = std::cos(st);
    }
  float* pe_d = (float*)dmalloc(e, pe.size() * 4);
  CK(cudaMemcpy(pe_d, pe.data(), pe.size() * 4, cudaMemcpyHostToDevice));
  void* pe_t = pe_d;
  if (e.dt == DT_BF16) {  // the reference casts the table to the hidden dtype before relative_k_proj
    pe_t = dmalloc(e, pe.size() * 2);
    k_affine<bf16><<<64, 256, 0, e.st>>>(pe_d, (bf16*)pe_t, 1.f, 0.f, pe.size());
  }
  void* rk_t = dmalloc(e, 13 * 1024 * 4);
  const double q_scale = std::pow(128.0, -0.5) / std::log(2.0);
  e.a_ksc = (float)(std::log1p(std::exp(1.0)) / std::log(2.0));
  for (int i = 0; i < L; ++i) {
    const std::string p = A + "layers." + std::to_string(i) + ".";
    ALayer l;
    l.f1a = clip_lin_w(e, p + "feed_forward1.ffw_layer_1.");
    l.f1b = clip_lin_w(e, p + "feed_forward1.ffw_layer_2.");
    l.f2a = clip_lin_w(e, p + "feed_forward2.ffw_layer_1.");
    l.f2b = clip_lin_w(e, p + "feed_forward2.ffw_layer_2.");
    l.ls = clip_lin_w(e, p + "lconv1d.linear_start.");
    l.le = clip_lin_w(e, p + "lconv1d.linear_end.");
    l.q = clip_lin_w(e, p + "self_attn.q_proj.");
    l.k = clip_lin_w(e, p + "self_attn.k_proj.");
    l.v = clip_lin_w(e, p + "self_attn.v_proj.");
    l.post = clip_lin_w(e, p + "self_attn.post.");
    l.f1_pre = to_f32(e, p + "feed_forward1.pre_layer_norm.weight");
    l.f1_post = to_f32(e, p + "feed_forward1.post_layer_norm.weight");
    l.f2_pre = to_f32(e, p + "feed_forward2.pre_layer_norm.weight");
    l.f2_post = to_f32(e, p + "feed_forward2.post_layer_norm.weight");
    l.lc_pre = to_f32(e, p + "lconv1d.pre_layer_norm.weight");
    l.lc_conv = to_f32(e, p + "lconv1d.conv_norm.weight");
    l.pre_attn = to_f32(e, p + "norm_pre_attn.weight");
    l.post_attn = to_f32(e, p + "norm_post_attn.weight");
    l.out = to_f32(e, p + "norm_out.weight");
    l.dw = to_f32(e, p + "lconv1d.depthwise_conv1d.weight");
    std::vector<float> qs(128);
    for (int d = 0; d < 128; ++d) {
      const double x = host_scalar(e, p + "self_attn.per_dim_scale", d);
      qs[d] = (float)(q_scale * (x > 20 ? x : std::log1p(std::exp(x))));
    }
    l.qsc = (float*)dmalloc(e, 128 * 4);
    CK(cudaMemcpy(l.qsc, qs.data(), 128 * 4, cudaMemcpyHostToDevice));
    Lin rk = make_lin(e, {p + "self_attn.relative_k_proj.weight"});
    gemm(e, pe_t, rk, rk_t, 13);
    l.relk = (float*)dmalloc(e, 13 * 1024 * 4);
    if (e.dt == DT_BF16) k_bf16_to_f32<<<64, 256, 0, e.st>>>((const bf16*)rk_t, l.relk, 13 * 1024);
    else CK(cudaMemcpyAsync(l.relk, rk_t, 13 * 1024 * 4, cudaMemcpyDeviceToDevice, e.st));
    CK(cudaStreamSynchronize(e.st));
    dfree(e, rk.w);
    e.weight_bytes -= (size_t)rk.n * rk.k * e.esz();
    e.al.push_back(l);
  }
  CK(cudaStreamSynchronize(e.st));
}

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
      const Lin sl = slice_rows(e, e.ple_w, (int)li * e.ple, e.ple);
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
  set_seqs(m, vcu);
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
  e.soft_src = e.v_soft;
}


// Gemma 4 audio tower on packed clips -> soft tokens (e.a_soft) for the next text forward
template <typename T>
void clip_gemm(Engine& e, const void* X, const Lin& L, void* Y, int M) {
  const void* in = X;
  if (std::isfinite(L.clip[0]) || std::isfinite(L.clip[1])) {
    k_clamp<T><<<nblk((size_t)M * L.k), 256, 0, e.st>>>((const T*)X, (T*)e.a_tmp, (size_t)M * L.k, L.clip[0], L.clip[1]);
    in = e.a_tmp;
  }
  gemm(e, in, L, Y, M);
  if (std::isfinite(L.clip[2]) || std::isfinite(L.clip[3]))
    k_clamp<T><<<nblk((size_t)M * L.n), 256, 0, e.st>>>((const T*)Y, (T*)Y, (size_t)M * L.n, L.clip[2], L.clip[3]);
  CK(cudaGetLastError());
}

template <typename T>
void gemma_audio(Engine& e, int nclip, const int* cu0, const float* feats, const int* m0, int n1, const int* rows1,
                 const int* m1, int n2, const int* rows2, const int* m2, int nsoft, const int* soft_rows) {
  const int n0 = cu0[nclip];
  REQ(n0 <= e.max_frames, "audio batch exceeds the frame budget");
  Packer pk(e);
  pk.add(m0, n0);
  pk.add(rows1, (size_t)n1 * 3);
  pk.add(m1, n1);
  pk.add(rows2, (size_t)n2 * 5);
  pk.add(m2, n2);
  pk.add(soft_rows, nsoft);
  auto d = pk.go();
  const size_t pin_off = e.meta_ints * 4;
  std::memcpy(e.pin + pin_off, feats, (size_t)n0 * 128 * 4);
  CK(cudaMemcpyAsync(e.a_feat, e.pin + pin_off, (size_t)n0 * 128 * 4, cudaMemcpyHostToDevice, e.st));
  k_aconv0<<<dim3(n1, 64), 128, 0, e.st>>>(e.a_feat, d[0], d[1], e.a_c0w, e.a_c0n, e.eps, e.a_l0);
  k_aconv1<T><<<dim3(n2, 32), 32, 0, e.st>>>(e.a_l0, d[2], d[3], e.a_c1w, e.a_c1n, e.eps, (T*)e.a_x);
  CK(cudaGetLastError());
  gemm(e, e.a_x, e.a_in, e.a_t1, n2);
  const int H = 1024;
  // h = input projection; y = ff1 pre-norm
  resid_norm<T>(e, n2, H, e.a_t1, 3, nullptr, 0.f, e.a_h, 1.f, nullptr, 0, 0, 1, e.al[0].f1_pre, e.a_y, nullptr);
  for (size_t li = 0; li < e.al.size(); ++li) {
    ALayer& l = e.al[li];
    // feed-forward 1: h += 0.5 * rms(ffw2(silu(ffw1(y)))); y = attention pre-norm
    clip_gemm<T>(e, e.a_y, l.f1a, e.a_big, n2);
    k_silu<T><<<nblk((size_t)n2 * 4096), 256, 0, e.st>>>((T*)e.a_big, (size_t)n2 * 4096);
    clip_gemm<T>(e, e.a_big, l.f1b, e.a_t1, n2);
    resid_norm<T>(e, n2, H, e.a_t1, 4, l.f1_post, e.a_resw, e.a_h, 1.f, nullptr, 0, 0, 1, l.pre_attn, e.a_y, nullptr);
    // chunked attention
    clip_gemm<T>(e, e.a_y, l.q, e.a_q, n2);
    clip_gemm<T>(e, e.a_y, l.k, e.a_k, n2);
    clip_gemm<T>(e, e.a_y, l.v, e.a_v, n2);
    k_audio_attn<T><<<n2, 256, 0, e.st>>>((const T*)e.a_q, (const T*)e.a_k, (const T*)e.a_v, l.qsc, e.a_ksc, l.relk, d[4],
                                         d[3], e.a_softcap, e.a_invalid, (T*)e.a_o);
    CK(cudaGetLastError());
    clip_gemm<T>(e, e.a_o, l.post, e.a_t1, n2);
    resid_norm<T>(e, n2, H, e.a_t1, 1, l.post_attn, 0.f, e.a_h, 1.f, nullptr, 0, 0, 1, l.lc_pre, e.a_y, nullptr);
    // light conv: h += linear_end(silu(rms(dwconv(glu(linear_start(y))))))
    clip_gemm<T>(e, e.a_y, l.ls, e.a_big, n2);
    k_lconv_mid<T><<<n2, 256, 0, e.st>>>((const T*)e.a_big, d[3], l.dw, l.lc_conv, e.eps, (T*)e.a_o);
    CK(cudaGetLastError());
    clip_gemm<T>(e, e.a_o, l.le, e.a_t1, n2);
    resid_norm<T>(e, n2, H, e.a_t1, 2, nullptr, 1.f, e.a_h, 1.f, nullptr, 0, 0, 1, l.f2_pre, e.a_y, nullptr);
    // feed-forward 2, then h = norm_out(h)
    clip_gemm<T>(e, e.a_y, l.f2a, e.a_big, n2);
    k_silu<T><<<nblk((size_t)n2 * 4096), 256, 0, e.st>>>((T*)e.a_big, (size_t)n2 * 4096);
    clip_gemm<T>(e, e.a_big, l.f2b, e.a_t1, n2);
    resid_norm<T>(e, n2, H, e.a_t1, 4, l.f2_post, e.a_resw, e.a_h, 1.f, nullptr, 0, 0, 1, l.out, nullptr, e.a_h);
    // next layer's ff1 pre-norm (or the input of output_proj)
    if (li + 1 < e.al.size())
      resid_norm<T>(e, n2, H, nullptr, 0, nullptr, 0.f, e.a_h, 1.f, nullptr, 0, 0, 1, e.al[li + 1].f1_pre, e.a_y, nullptr);
    else
      resid_norm<T>(e, n2, H, nullptr, 0, nullptr, 0.f, e.a_h, 1.f, nullptr, 0, 0, 2, nullptr, e.a_y, nullptr);
  }
  gemm(e, e.a_y, e.a_outp, e.a_proj, n2);
  if (nsoft) {
    k_bias_rms_gather<T><<<nsoft, 256, 0, e.st>>>((const T*)e.a_proj, e.a_out_b, d[5], e.eps, (T*)e.a_e, e.a_outp.n);
    CK(cudaGetLastError());
    gemm(e, e.a_e, e.a_emb, e.a_soft, nsoft);
  }
  e.soft_src = e.a_soft;
}

// nspan > 0 (late chunking): instead of one mean over each sequence, one vector per token span [span_lo, span_hi)
// (packed indices): mean of the final hidden states, then the same projection and L2 normalisation
template <typename T>
void gemma_text(Engine& e, int ntok, int nseq, const int* ids, const int* cu_h, const int* mm_idx, float* out_host,
                int nspan = 0, const int* span_lo = nullptr, const int* span_hi = nullptr) {
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
  set_seqs(m, cu);
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
  k_gemma_embed<T><<<ntok, 128, 0, e.st>>>(m.ids, e.embed, e.embed_scale, mm_idx ? d[6] : nullptr, (const T*)e.soft_src,
                                           e.b_h, (T*)e.b_e, H);
  CK(cudaGetLastError());
  resid_norm<T>(e, ntok, H, nullptr, 0, nullptr, 0.f, e.b_h, 1.f, nullptr, 0, 0, 1, e.gl[0].w_in, e.b_y, nullptr);
  gemma_layers<T>(e, ntok, m, e.gl, H, e.Hq, e.I, e.b_y, e.b_qkv, e.b_o, e.b_t1, e.b_gu, e.b_m, e.b_h, true, e.b_t2,
                  e.b_e, 1, e.b_y32, e.w_final);
  if (nspan > 0) {
    if (nspan > e.span_cap) {
      if (e.span_buf) CK(cudaFree(e.span_buf));
      e.span_cap = std::max(nspan, 4096);
      CK(cudaMalloc(&e.span_buf, (size_t)2 * e.span_cap * sizeof(int)));
    }
    for (int s = 0; s < nspan; ++s)
      REQ(span_lo[s] >= 0 && span_hi[s] > span_lo[s] && span_hi[s] <= ntok, "span out of range or empty");
    int* dlo = e.span_buf;
    int* dhi = e.span_buf + e.span_cap;
    CK(cudaMemcpyAsync(dlo, span_lo, (size_t)nspan * 4, cudaMemcpyHostToDevice, e.st));
    CK(cudaMemcpyAsync(dhi, span_hi, (size_t)nspan * 4, cudaMemcpyHostToDevice, e.st));
    for (int s0 = 0; s0 < nspan; s0 += 4096) {  // b_pool / b_out hold 4096 rows
      const int n = std::min(4096, nspan - s0);
      k_span_pool<<<n, 256, 0, e.st>>>(e.b_y32, dlo + s0, dhi + s0, e.b_pool, H);
      gemm(e, e.b_pool, e.proj_f32, e.b_out, n, DT_F32);
      k_l2norm<float><<<n, 256, 0, e.st>>>(e.b_out, e.b_out, e.out_dim);
      CK(cudaGetLastError());
      CK(cudaMemcpyAsync(out_host + (size_t)s0 * e.out_dim, e.b_out, (size_t)n * e.out_dim * 4, cudaMemcpyDeviceToHost, e.st));
    }
    CK(cudaStreamSynchronize(e.st));  // also keeps span_lo / span_hi alive until the copies are done
    return;
  }
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
  set_seqs(m, cu);
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
        CB(cublasLtCreate(&e->lt));
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
    if (e.arch == ARCH_GEMMA && cfgv(e, "audio", 0) != 0) {
      e.max_frames = (int)cfgv(e, "max_frames", 12000);
      build_audio(e);
    }
    e.quant = (int)cfgv(e, "quant", 0);
    if (e.quant) {  // the layer GEMMs (embedding / patch / output heads stay bf16)
      for (auto& l : e.gl)
        for (Lin* L : {&l.qkv, &l.o, &l.gu, &l.down, &l.pg, &l.pp}) quantize_lin(e, *L, e.quant);
      for (auto& l : e.vl)
        for (Lin* L : {&l.qkv, &l.o, &l.gu, &l.down}) quantize_lin(e, *L, e.quant);
      for (auto& l : e.nl)
        for (Lin* L : {&l.qkvg, &l.o, &l.up, &l.down}) quantize_lin(e, *L, e.quant);
      if (e.ple_w.w) quantize_lin(e, e.ple_w, e.quant);
    }
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

// late chunking: like fe_gemma_text, but out = [nspan, out_dim], one vector per packed token span [lo, hi)
int fe_gemma_text_spans(void* h, int ntok, int nseq, const int32_t* ids, const int32_t* cu, int nspan, const int32_t* lo,
                        const int32_t* hi, float* out) {
  return guard([&] {
    Engine& e = *(Engine*)h;
    REQ(e.arch == ARCH_GEMMA, "late chunking is EmbeddingGemma 2 only");
    if (e.dt == DT_F32) gemma_text<float>(e, ntok, nseq, ids, cu, nullptr, out, nspan, lo, hi);
    else gemma_text<bf16>(e, ntok, nseq, ids, cu, nullptr, out, nspan, lo, hi);
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

// Gemma 4 audio tower on packed clips (log-mel features [frames, 128] per clip back to back). Row maps per stage
// (see k_aconv0 / k_aconv1 / k_audio_attn): rows1 [n1 x 3] = (clip start in frames, clip frames, local index),
// rows2 [n2 x 5] = (clip start in stage-1 rows, clip stage-1 rows, local index, clip start in stage-2 rows, clip
// stage-2 rows); m0 / m1 / m2 = validity masks per stage; soft_rows = stage-2 rows that become soft tokens, in order.
int fe_gemma_audio(void* h, int nclip, const int32_t* cu0, const float* feats, const int32_t* m0, int n1,
                   const int32_t* rows1, const int32_t* m1, int n2, const int32_t* rows2, const int32_t* m2, int nsoft,
                   const int32_t* soft_rows) {
  return guard([&] {
    Engine& e = *(Engine*)h;
    REQ(e.has_audio, "engine built without the audio tower");
    REQ(n2 <= e.max_frames / 4 + 1024 && n1 <= e.max_frames / 2 + 1024, "audio batch too large");
    if (e.dt == DT_F32) gemma_audio<float>(e, nclip, cu0, feats, m0, n1, rows1, m1, n2, rows2, m2, nsoft, soft_rows);
    else gemma_audio<bf16>(e, nclip, cu0, feats, m0, n1, rows1, m1, n2, rows2, m2, nsoft, soft_rows);
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
    out[1] = e.arena_bytes + fecd_bytes(e.cd) + (size_t)3 * e.cd_cap * sizeof(int);
    out[2] = tot - fr;
    out[3] = tot;
  });
}

// attention micro-benchmark: nseq packed sequences of S tokens, random bf16 / fp32 q, k, v; mean ms per call
int fe_attn_bench(int dtype, int D, int Hq, int Hkv, int S, int nseq, int window, int iters, float* ms) {
  return guard([&] {
    Engine e;
    e.dt = dtype;
    CK(cudaStreamCreateWithFlags(&e.st, cudaStreamNonBlocking));
    const int ntok = S * nseq, ld = (Hq + 2 * Hkv) * D;
    const size_t es = dtype == DT_BF16 ? 2 : 4;
    void *qkv = nullptr, *out = nullptr, *dm = nullptr;
    CK(cudaMalloc(&qkv, (size_t)ntok * ld * es));
    CK(cudaMalloc(&out, (size_t)ntok * Hq * D * es));
    std::vector<int> cu(nseq + 1);
    for (int i = 0; i <= nseq; ++i) cu[i] = i * S;
    Tiles t;
    build_tiles(cu, 64, t);
    const size_t cpad = (cu.size() + 63) / 64 * 64;  // keep the int2 tile array 8-byte (here 256-byte) aligned
    CK(cudaMalloc(&dm, (cpad + t.host.size()) * 4));
    CK(cudaMemcpy(dm, cu.data(), cu.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy((int*)dm + cpad, t.host.data(), t.host.size() * 4, cudaMemcpyHostToDevice));
    {  // deterministic pseudo-random fill in [-1, 1)
      std::vector<uint16_t> hb;
      std::vector<float> hf;
      const size_t n = (size_t)ntok * ld;
      uint32_t x = 12345;
      if (dtype == DT_BF16) {
        hb.resize(n);
        for (auto& v : hb) { x = x * 1664525u + 1013904223u; float f = (x >> 8) * (2.f / 16777216.f) - 1.f; uint32_t u; std::memcpy(&u, &f, 4); v = (uint16_t)(u >> 16); }
        CK(cudaMemcpy(qkv, hb.data(), n * 2, cudaMemcpyHostToDevice));
      } else {
        hf.resize(n);
        for (auto& v : hf) { x = x * 1664525u + 1013904223u; v = (x >> 8) * (2.f / 16777216.f) - 1.f; }
        CK(cudaMemcpy(qkv, hf.data(), n * 4, cudaMemcpyHostToDevice));
      }
    }
    Meta m;
    set_seqs(m, cu);
    m.cu = (int*)dm;
    m.tiles64 = (int2*)((int*)dm + cpad);
    m.nt64 = t.n;
    Tiles t16, t32;
    build_tiles(cu, 16, t16);
    build_tiles(cu, 32, t32);
    void* dm2 = nullptr;
    const size_t t16pad = (t16.host.size() + 63) / 64 * 64;
    CK(cudaMalloc(&dm2, (t16pad + t32.host.size()) * 4));
    CK(cudaMemcpy(dm2, t16.host.data(), t16.host.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy((int*)dm2 + t16pad, t32.host.data(), t32.host.size() * 4, cudaMemcpyHostToDevice));
    m.tiles16 = (int2*)dm2;
    m.nt16 = t16.n;
    m.tiles32 = (int2*)((int*)dm2 + t16pad);
    m.nt32 = t32.n;
    AttnArgs a{};
    a.qkv = qkv;
    a.ld = ld;
    a.q_off = 0;
    a.k_off = Hq * D;
    a.v_off = (Hq + Hkv) * D;
    a.Hq = Hq;
    a.Hkv = Hkv;
    a.window = window;
    a.scale = 1.f / sqrtf((float)D);
    a.cu = m.cu;
    a.out = out;
    a.ld_out = Hq * D;
    auto run = [&] {
      if (dtype == DT_BF16) attention<bf16>(e, D, a, m);
      else attention<float>(e, D, a, m);
    };
    run();
    CK(cudaStreamSynchronize(e.st));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, e.st));
    for (int i = 0; i < iters; ++i) run();
    CK(cudaEventRecord(e1, e.st));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(ms, e0, e1));
    *ms /= iters;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(qkv);
    cudaFree(out);
    cudaFree(dm);
    cudaFree(dm2);
    if (e.cd) fecd_free(e.cd);
    if (e.cd_meta) cudaFree(e.cd_meta);
    cudaStreamDestroy(e.st);
    e.st = nullptr;
  });
}

// opt-in shared memory per block of the current GPU (B200: 227 KB, RTX PRO 6000 Blackwell: 99 KB)
int fe_smem_optin() { return smem_optin(); }

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
  if (e->cd) fecd_free(e->cd);
  if (e->cd_meta) cudaFree(e->cd_meta);
  if (e->span_buf) cudaFree(e->span_buf);
  for (auto& kv : e->raw) cudaFree(kv.second.p);
  for (void* p : e->owned) cudaFree(p);
  if (e->pin) cudaFreeHost(e->pin);
  if (e->cb) cublasDestroy(e->cb);
  if (e->lt) cublasLtDestroy(e->lt);
  if (e->st) cudaStreamDestroy(e->st);
  delete e;
}

}  // extern "C"
