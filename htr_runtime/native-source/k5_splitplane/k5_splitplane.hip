// V5-W3: K5 16B+4B Split-Plane Load-Time Repack + Aligned global_load_b128 GEMV
// Supports: repack_k5, gemv_k5_sp16 (partials), and gemv_k5_sp16_epi (fused epilogue, epi8=0 & epi8=1).
// Zero extra payload bytes (20 B/lane = 16 B aligned plane + 4 B tail plane), 100% bit-identical to htr_kernels.hip!
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

static constexpr uint32_t GEN_A = 0xDF6ADFu;
static constexpr uint32_t GEN_B = 0x34983u;
typedef _Float16 h2v __attribute__((ext_vector_type(2)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));

__device__ __forceinline__ uint32_t alignbit(uint32_t hi, uint32_t lo, uint32_t sh) {
  return __builtin_amdgcn_alignbit(hi, lo, sh);
}
template <int B>
__device__ __forceinline__ uint32_t ext16(const uint32_t* V) {
  constexpr int w = B >> 5, off = B & 31;
  if constexpr (off <= 16) return (V[w] >> (16 - off)) & 0xFFFFu;
  else return alignbit(V[w], V[w + 1], 48 - off) & 0xFFFFu;
}
__device__ __forceinline__ uint32_t val_f16bits2(uint32_t s0, uint32_t s1) {
  return __builtin_amdgcn_sad_hi_u8(s1 * GEN_A + GEN_B, 0u, __builtin_amdgcn_sad_u8(s0 * GEN_A + GEN_B, 0u, 0x64006400u));
}
__device__ __forceinline__ h2v as_h2(uint32_t u) { return __builtin_bit_cast(h2v, u); }
__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f(__hip_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __hip_bfloat16 from_f<__hip_bfloat16>(float v) { return __float2bfloat16(v); }

template <int MSK> __device__ __forceinline__ float shx(float v) {
  const int iv = __builtin_bit_cast(int, v);
  if constexpr (MSK < 16) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, iv, 0x160 | MSK, 0xF, 0xF, false));
  } else {
    return __builtin_bit_cast(float, (int)__builtin_amdgcn_permlanex16((unsigned)iv, (unsigned)iv, 0x76543210u, 0xfedcba98u, false, false));
  }
}
template <int MSK> __device__ __forceinline__ void fwht_stage(float v[4], int lane) {
  const bool up = (lane & MSK) != 0;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float o = shx<MSK>(v[e]);
    v[e] = up ? (o - v[e]) : (v[e] + o);
  }
}
__device__ __forceinline__ void fwht128_wave(float v[4], int lane) {
  float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
  v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
  fwht_stage<1>(v, lane); fwht_stage<2>(v, lane); fwht_stage<4>(v, lane); fwht_stage<8>(v, lane); fwht_stage<16>(v, lane);
}

// Shared K5 split-plane tile accumulation body (exact match to k_gemv<5, M> & k_e82_gemv_epi<5, M>)
#define K5SP_ACCUM_LOOP(M) \
  for (int j = j0 + g; j < j1; j += 4) { \
    const size_t tid = (size_t)i * nt + j; \
    const u4v w4 = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(p16 + tid * 32 + q * 4)); \
    const uint32_t w_last = __builtin_nontemporal_load(p4 + tid * 8 + q); \
    uint32_t V[6]; \
    V[0] = __builtin_amdgcn_permlane16(w_last, w_last, 0x65432107u, 0xedcba98fu, false, false); \
    V[1] = w4.x; V[2] = w4.y; V[3] = w4.z; V[4] = w4.w; V[5] = w_last; \
    h2v xv[M][8]; \
    _Pragma("unroll") for (int r = 0; r < M; ++r) { \
      const uint4* xp = reinterpret_cast<const uint4*>(xt + (size_t)r * n + j * 16); \
      uint4 u0 = xp[0], u1 = xp[1]; \
      xv[r][0] = as_h2(u0.x); xv[r][1] = as_h2(u0.y); xv[r][2] = as_h2(u0.z); xv[r][3] = as_h2(u0.w); \
      xv[r][4] = as_h2(u1.x); xv[r][5] = as_h2(u1.y); xv[r][6] = as_h2(u1.z); xv[r][7] = as_h2(u1.w); \
    } \
    for (int _dummy = 0; _dummy < 1; ++_dummy) { \
      auto _pair = [&](int u_idx, uint32_t s0, uint32_t s1) { \
        h2v w2 = as_h2(val_f16bits2(s0, s1)); \
        _Pragma("unroll") for (int r = 0; r < M; ++r) { \
          if (u_idx < 16) a0[r] = __builtin_amdgcn_fdot2(w2, xv[r][(u_idx & 15) >> 1], a0[r], false); \
          else a1[r] = __builtin_amdgcn_fdot2(w2, xv[r][(u_idx & 15) >> 1], a1[r], false); \
        } \
      }; \
      _pair(0,  ext16<16 +  1*5>(V), ext16<16 +  2*5>(V)); \
      _pair(2,  ext16<16 +  3*5>(V), ext16<16 +  4*5>(V)); \
      _pair(4,  ext16<16 +  5*5>(V), ext16<16 +  6*5>(V)); \
      _pair(6,  ext16<16 +  7*5>(V), ext16<16 +  8*5>(V)); \
      _pair(8,  ext16<16 +  9*5>(V), ext16<16 + 10*5>(V)); \
      _pair(10, ext16<16 + 11*5>(V), ext16<16 + 12*5>(V)); \
      _pair(12, ext16<16 + 13*5>(V), ext16<16 + 14*5>(V)); \
      _pair(14, ext16<16 + 15*5>(V), ext16<16 + 16*5>(V)); \
      _pair(16, ext16<16 + 17*5>(V), ext16<16 + 18*5>(V)); \
      _pair(18, ext16<16 + 19*5>(V), ext16<16 + 20*5>(V)); \
      _pair(20, ext16<16 + 21*5>(V), ext16<16 + 22*5>(V)); \
      _pair(22, ext16<16 + 23*5>(V), ext16<16 + 24*5>(V)); \
      _pair(24, ext16<16 + 25*5>(V), ext16<16 + 26*5>(V)); \
      _pair(26, ext16<16 + 27*5>(V), ext16<16 + 28*5>(V)); \
      _pair(28, ext16<16 + 29*5>(V), ext16<16 + 30*5>(V)); \
      _pair(30, ext16<16 + 31*5>(V), ext16<16 + 32*5>(V)); \
    } \
  } \
  _Pragma("unroll") for (int r = 0; r < M; ++r) { \
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]); \
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]); \
  }

__global__ void k_repack_k5_sp(const uint32_t* __restrict__ qw, uint32_t* __restrict__ p16, uint32_t* __restrict__ p4, int ntiles) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ntiles * 8) return;
  int t = idx >> 3, q = idx & 7;
  const uint32_t* src = qw + (size_t)t * 40 + q * 5;
  uint32_t* d16 = p16 + (size_t)t * 32 + q * 4;
  d16[0] = src[0]; d16[1] = src[1]; d16[2] = src[2]; d16[3] = src[3];
  p4[(size_t)t * 8 + q] = src[4];
}

template <int M>
__global__ __launch_bounds__(256) void k_gemv_k5_sp16(const uint32_t* __restrict__ p16, const uint32_t* __restrict__ p4,
                                                      const _Float16* __restrict__ xt, float* __restrict__ part,
                                                      int mt, int nt, int nsplit, int tps) {
  const int lane = threadIdx.x & 31;
  const int wave = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (wave >= mt * nsplit) return;
  const int i = wave / nsplit, sp = wave % nsplit;
  const int g = lane >> 3, q = lane & 7, n = nt * 16, m = mt * 16;
  float a0[M], a1[M];
#pragma unroll
  for (int r = 0; r < M; ++r) { a0[r] = 0.f; a1[r] = 0.f; }
  const int j0 = sp * tps, j1 = min(nt, j0 + tps);
  K5SP_ACCUM_LOOP(M)
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
      float* pp = part + ((size_t)sp * M + r) * m + i * 16 + 2 * q;
      pp[0] = a0[r]; pp[1] = a1[r];
    }
  }
}

template <int M, typename TO>
__global__ __launch_bounds__(256) void k_gemv_k5_sp16_epi(
    const uint32_t* __restrict__ p16, const uint32_t* __restrict__ p4, const _Float16* __restrict__ xt,
    float* __restrict__ part, unsigned* __restrict__ counters, const float* __restrict__ xs, const float* __restrict__ tok,
    const __half* __restrict__ scales, float cb, const uint8_t* __restrict__ sv, const TO* __restrict__ bias,
    TO* __restrict__ y, int mt, int nt, int nsplit, int tps) {
  const int lane = threadIdx.x & 31;
  const int wave = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (wave >= mt * nsplit) return;
  const int i = wave / nsplit, sp = wave % nsplit;
  const int g = lane >> 3, q = lane & 7, n = nt * 16, m = mt * 16;
  float a0[M], a1[M];
#pragma unroll
  for (int r = 0; r < M; ++r) { a0[r] = 0.f; a1[r] = 0.f; }
  const int j0 = sp * tps, j1 = min(nt, j0 + tps);
  K5SP_ACCUM_LOOP(M)
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
      float* pp = part + ((size_t)sp * M + r) * m + i * 16 + 2 * q;
      pp[0] = a0[r]; pp[1] = a1[r];
    }
  }
  const int b = i >> 3;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  unsigned old = 0;
  if (lane == 0) old = atomicAdd(&counters[b], 1u);
  old = __shfl(old, 0);
  if (old != (unsigned)(8 * nsplit - 1)) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  const int base = b * 128 + lane * 4;
  for (int r = 0; r < M; ++r) {
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < nsplit; ++k) {
      const float* pp = part + ((size_t)k * M + r) * m + base;
#pragma unroll
      for (int e = 0; e < 4; ++e) v[e] += __builtin_nontemporal_load(pp + e);
    }
    const float c = 1534.0f * xs[r], t = tok[r];
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = (v[e] - c) * __half2float(scales[base + e]) * cb * t;
    fwht128_wave(v, lane);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int row = base + e;
      float o = v[e] * (1.0f / 11.313708498984761f);
      if ((sv[row >> 3] >> (row & 7)) & 1) o = -o;
      if (bias != nullptr) o += to_f(bias[row]);
      y[(size_t)r * m + row] = from_f<TO>(o);
    }
  }
  if (lane == 0) counters[b] = 0u;
}

template <int M, typename TO>
__global__ __launch_bounds__(256) void k_gemv_k5_sp16_epi8(
    const uint32_t* __restrict__ p16, const uint32_t* __restrict__ p4, const _Float16* __restrict__ xt,
    float* __restrict__ part, unsigned* __restrict__ counters, const float* __restrict__ xs, const float* __restrict__ tok,
    const __half* __restrict__ scales, float cb, const uint8_t* __restrict__ sv, const TO* __restrict__ bias,
    TO* __restrict__ y, int mt, int nt, int nsplit, int tps) {
  __shared__ float red[8][M][16];
  __shared__ unsigned is_last;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int wave = blockIdx.x * 8 + w;
  const int i = wave / nsplit, sp = wave % nsplit;
  const int g = lane >> 3, q = lane & 7, n = nt * 16, m = mt * 16, ns8 = nsplit >> 3;
  float a0[M], a1[M];
#pragma unroll
  for (int r = 0; r < M; ++r) { a0[r] = 0.f; a1[r] = 0.f; }
  const int j0 = sp * tps, j1 = min(nt, j0 + tps);
  K5SP_ACCUM_LOOP(M)
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M; ++r) { red[w][r][2 * q] = a0[r]; red[w][r][2 * q + 1] = a1[r]; }
  }
  __syncthreads();
  if (threadIdx.x < M * 16) {
    const int r = threadIdx.x >> 4, c = threadIdx.x & 15;
    float s = 0.f;
#pragma unroll
    for (int ww = 0; ww < 8; ++ww) s += red[ww][r][c];
    part[((size_t)(sp >> 3) * M + r) * m + i * 16 + c] = s;
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();
  const int b = i >> 3;
  if (threadIdx.x == 0) is_last = (atomicAdd(&counters[b], 1u) == (unsigned)(nsplit - 1));
  __syncthreads();
  if (!is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  const int base = b * 128 + lane * 4;
  for (int r = w; r < M; r += 8) {
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < ns8; ++k) {
      const float* pp = part + ((size_t)k * M + r) * m + base;
#pragma unroll
      for (int e = 0; e < 4; ++e) v[e] += __builtin_nontemporal_load(pp + e);
    }
    const float c = 1534.0f * xs[r], t = tok[r];
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = (v[e] - c) * __half2float(scales[base + e]) * cb * t;
    fwht128_wave(v, lane);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int row = base + e;
      float o = v[e] * (1.0f / 11.313708498984761f);
      if ((sv[row >> 3] >> (row & 7)) & 1) o = -o;
      if (bias != nullptr) o += to_f(bias[row]);
      y[(size_t)r * m + row] = from_f<TO>(o);
    }
  }
  if (threadIdx.x == 0) counters[b] = 0u;
}

std::vector<torch::Tensor> repack_k5(torch::Tensor qw) {
  TORCH_CHECK(qw.dim() == 3 && qw.size(2) == 40 && qw.scalar_type() == torch::kInt32 && qw.is_contiguous());
  int mt = qw.size(0), nt = qw.size(1), ntiles = mt * nt;
  auto p16 = torch::empty({mt, nt, 8, 4}, qw.options());
  auto p4 = torch::empty({mt, nt, 8}, qw.options());
  hipLaunchKernelGGL(k_repack_k5_sp, dim3((ntiles * 8 + 255) / 256), dim3(256), 0, at::hip::getCurrentHIPStream(),
                     reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>()),
                     reinterpret_cast<uint32_t*>(p16.data_ptr<int32_t>()),
                     reinterpret_cast<uint32_t*>(p4.data_ptr<int32_t>()), ntiles);
  return {p16, p4};
}

torch::Tensor gemv_k5_sp16(torch::Tensor p16, torch::Tensor p4, torch::Tensor xt, int64_t nsplit) {
  int mt = p16.size(0), nt = p16.size(1), M = xt.size(0);
  TORCH_CHECK(M >= 1 && M <= 8);
  const int tps = ((nt + nsplit - 1) / nsplit + 3) / 4 * 4, ns = (nt + tps - 1) / tps;
  auto part = torch::empty({ns, M, mt * 16}, xt.options().dtype(torch::kFloat32));
  int blocks = (mt * ns + 7) / 8;
  hipStream_t st = at::hip::getCurrentHIPStream();
  auto pp16 = reinterpret_cast<const uint32_t*>(p16.data_ptr<int32_t>());
  auto pp4 = reinterpret_cast<const uint32_t*>(p4.data_ptr<int32_t>());
  auto xp = reinterpret_cast<const _Float16*>(xt.data_ptr<at::Half>());
  auto op = part.data_ptr<float>();
  switch (M) {
#define CASE_M(MM) case MM: hipLaunchKernelGGL((k_gemv_k5_sp16<MM>), dim3(blocks), dim3(256), 0, st, pp16, pp4, xp, op, mt, nt, ns, tps); break;
    CASE_M(1) CASE_M(2) CASE_M(3) CASE_M(4) CASE_M(5) CASE_M(6) CASE_M(7) CASE_M(8)
#undef CASE_M
  }
  return part;
}

template <typename TO>
static void dispatch_epi_t(int M, bool epi8, int blocks, hipStream_t st,
                           const uint32_t* p16, const uint32_t* p4, const _Float16* xp,
                           float* part, unsigned* cnt, const float* xs, const float* tok,
                           const __half* sc, float cb, const uint8_t* sv, const TO* bp, TO* yp,
                           int mt, int nt, int ns, int tps) {
  if (epi8) {
    switch (M) {
#define CE8(MM) case MM: hipLaunchKernelGGL((k_gemv_k5_sp16_epi8<MM, TO>), dim3(blocks), dim3(256), 0, st, p16, p4, xp, part, cnt, xs, tok, sc, cb, sv, bp, yp, mt, nt, ns, tps); break;
      CE8(1) CE8(2) CE8(3) CE8(4) CE8(5) CE8(6) CE8(7) CE8(8)
#undef CE8
    }
  } else {
    switch (M) {
#define CE1(MM) case MM: hipLaunchKernelGGL((k_gemv_k5_sp16_epi<MM, TO>), dim3(blocks), dim3(256), 0, st, p16, p4, xp, part, cnt, xs, tok, sc, cb, sv, bp, yp, mt, nt, ns, tps); break;
      CE1(1) CE1(2) CE1(3) CE1(4) CE1(5) CE1(6) CE1(7) CE1(8)
#undef CE1
    }
  }
}

torch::Tensor gemv_k5_sp16_epi(torch::Tensor p16, torch::Tensor p4, torch::Tensor xt, torch::Tensor xs, torch::Tensor tok,
                               torch::Tensor scales, double cb, torch::Tensor sv, c10::optional<torch::Tensor> bias,
                               torch::Tensor counters, int64_t nsplit, int64_t out_dtype, int64_t epi8) {
  int mt = p16.size(0), nt = p16.size(1), M = xt.size(0), m = mt * 16;
  TORCH_CHECK(M >= 1 && M <= 8 && (m % 128 == 0));
  const int tps = ((nt + nsplit - 1) / nsplit + 3) / 4 * 4, ns = (nt + tps - 1) / tps;
  const bool use8 = (epi8 != 0) && ((ns & 7) == 0);
  auto part = (mt >= 1024 && (mt & 7) == 0 && M <= 8) ? torch::empty({1}, xt.options().dtype(torch::kFloat32)) : torch::empty({use8 ? (ns >> 3) : ns, M, m}, xt.options().dtype(torch::kFloat32));
  auto dt = out_dtype == 0 ? torch::kBFloat16 : (out_dtype == 1 ? torch::kHalf : torch::kFloat32);
  auto y = torch::empty({M, m}, xt.options().dtype(dt));
  int blocks = (mt * ns + 7) / 8;
  hipStream_t st = at::hip::getCurrentHIPStream();
  auto pp16 = reinterpret_cast<const uint32_t*>(p16.data_ptr<int32_t>());
  auto pp4 = reinterpret_cast<const uint32_t*>(p4.data_ptr<int32_t>());
  auto xp = reinterpret_cast<const _Float16*>(xt.data_ptr<at::Half>());
  auto cnt = reinterpret_cast<unsigned*>(counters.data_ptr<int32_t>());
  auto sc = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  if (out_dtype == 0) {
    const __hip_bfloat16* bp = (bias.has_value() && bias->defined()) ? reinterpret_cast<const __hip_bfloat16*>(bias->data_ptr<at::BFloat16>()) : nullptr;
    dispatch_epi_t<__hip_bfloat16>(M, use8, blocks, st, pp16, pp4, xp, part.data_ptr<float>(), cnt, xs.data_ptr<float>(), tok.data_ptr<float>(), sc, (float)cb, sv.data_ptr<uint8_t>(), bp, reinterpret_cast<__hip_bfloat16*>(y.data_ptr<at::BFloat16>()), mt, nt, ns, tps);
  } else if (out_dtype == 1) {
    const __half* bp = (bias.has_value() && bias->defined()) ? reinterpret_cast<const __half*>(bias->data_ptr<at::Half>()) : nullptr;
    dispatch_epi_t<__half>(M, use8, blocks, st, pp16, pp4, xp, part.data_ptr<float>(), cnt, xs.data_ptr<float>(), tok.data_ptr<float>(), sc, (float)cb, sv.data_ptr<uint8_t>(), bp, reinterpret_cast<__half*>(y.data_ptr<at::Half>()), mt, nt, ns, tps);
  } else {
    const float* bp = (bias.has_value() && bias->defined()) ? bias->data_ptr<float>() : nullptr;
    dispatch_epi_t<float>(M, use8, blocks, st, pp16, pp4, xp, part.data_ptr<float>(), cnt, xs.data_ptr<float>(), tok.data_ptr<float>(), sc, (float)cb, sv.data_ptr<uint8_t>(), bp, y.data_ptr<float>(), mt, nt, ns, tps);
  }
  return y;
}


// V5-W3b: ZERO-REPACK K=4 aligned nontemporal global_load_b128 + v_permlane16 GEMV (for 318 MB lm_head & any K=4 dense)!
// Canonical qw [mt, nt, 32] already places lane q's 4 uint32 words at 16B-aligned offset q*4.
// Replaces 5 scalar global_load_b32 with 1 nontemporal global_load_b128 + 1 v_permlane16 (0 extra memory, 0 MALL pollution!).
#define K4B128_ACCUM_LOOP(M) \
  for (int j = j0 + g; j < j1; j += 4) { \
    const uint32_t* tp = qw + ((size_t)i * nt + j) * 32; \
    const u4v w4 = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(tp + q * 4)); \
    uint32_t V[5]; \
    V[0] = __builtin_amdgcn_permlane16(w4.w, w4.w, 0x65432107u, 0xedcba98fu, false, false); \
    V[1] = w4.x; V[2] = w4.y; V[3] = w4.z; V[4] = w4.w; \
    h2v xv[M][4]; \
    _Pragma("unroll") for (int r = 0; r < M; ++r) { \
      const uint4* xp = reinterpret_cast<const uint4*>(xt + (size_t)r * n + j * 16); \
      uint4 u0 = xp[0]; \
      xv[r][0] = as_h2(u0.x); xv[r][1] = as_h2(u0.y); xv[r][2] = as_h2(u0.z); xv[r][3] = as_h2(u0.w); \
    } \
    auto _stagger_pair = [&](int ua, int ub, uint32_t sa0, uint32_t sa1, uint32_t sb0, uint32_t sb1) { \
      h2v wa = as_h2(val_f16bits2(sa0, sa1)); \
      h2v wb = as_h2(val_f16bits2(sb0, sb1)); \
      _Pragma("unroll") for (int r = 0; r < M; ++r) { \
        a0[r] = __builtin_amdgcn_fdot2(wa, xv[r][(ua & 7) >> 1], a0[r], false); \
        a1[r] = __builtin_amdgcn_fdot2(wb, xv[r][(ub & 7) >> 1], a1[r], false); \
      } \
    }; \
    _stagger_pair(0, 18, ext16<16 +  1*4>(V), ext16<16 +  2*4>(V), ext16<16 + 19*4>(V), ext16<16 + 20*4>(V)); \
    _stagger_pair(2, 16, ext16<16 +  3*4>(V), ext16<16 +  4*4>(V), ext16<16 + 17*4>(V), ext16<16 + 18*4>(V)); \
    _stagger_pair(4, 22, ext16<16 +  5*4>(V), ext16<16 +  6*4>(V), ext16<16 + 23*4>(V), ext16<16 + 24*4>(V)); \
    _stagger_pair(6, 20, ext16<16 +  7*4>(V), ext16<16 +  8*4>(V), ext16<16 + 21*4>(V), ext16<16 + 22*4>(V)); \
    _Pragma("unroll") for (int r = 0; r < M; ++r) { \
      const uint4* xp = reinterpret_cast<const uint4*>(xt + (size_t)r * n + j * 16); \
      uint4 u1 = xp[1]; \
      xv[r][0] = as_h2(u1.x); xv[r][1] = as_h2(u1.y); xv[r][2] = as_h2(u1.z); xv[r][3] = as_h2(u1.w); \
    } \
    _stagger_pair(8,  26, ext16<16 +  9*4>(V), ext16<16 + 10*4>(V), ext16<16 + 27*4>(V), ext16<16 + 28*4>(V)); \
    _stagger_pair(10, 24, ext16<16 + 11*4>(V), ext16<16 + 12*4>(V), ext16<16 + 25*4>(V), ext16<16 + 26*4>(V)); \
    _stagger_pair(12, 30, ext16<16 + 13*4>(V), ext16<16 + 14*4>(V), ext16<16 + 31*4>(V), ext16<16 + 32*4>(V)); \
    _stagger_pair(14, 28, ext16<16 + 15*4>(V), ext16<16 + 16*4>(V), ext16<16 + 29*4>(V), ext16<16 + 30*4>(V)); \
  } \
  _Pragma("unroll") for (int r = 0; r < M; ++r) { \
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]); \
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]); \
  }

// v0.2.7 ZERO-SPLIT-K K=4 B128 GEMV+Epilogue (1 WG = 8 waves = 128 output rows = 1 strip-block!)
// For tall K=4 matrices like lm_head (mt=15520 -> 1,940 WGs = 48.5 WGs/CU):
//   - ZERO global part writes/reads (-151.2 MB/step!)
//   - ZERO atomicAdd on counters (-147,440 atomics/step!)
//   - ZERO release/acquire fences!
template <int M, typename TO>
__global__ __launch_bounds__(256) void k_gemv_k4_b128_ns1(
    const uint32_t* __restrict__ qw, const _Float16* __restrict__ xt,
    const float* __restrict__ xs, const float* __restrict__ tok,
    const __half* __restrict__ scales, float cb, const uint8_t* __restrict__ sv, const TO* __restrict__ bias,
    TO* __restrict__ y, int mt, int nt) {
  __shared__ __attribute__((aligned(16))) float ep_lds[M][128];
  const int b = blockIdx.x;
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int i = b * 8 + w;
  if (i >= mt) return;
  const int g = lane >> 3, q = lane & 7, n = nt * 16, m = mt * 16;
  float a0[M], a1[M];
#pragma unroll
  for (int r = 0; r < M; ++r) { a0[r] = 0.f; a1[r] = 0.f; }
  // WAVE5_PIPELINED_K4NS1: prefetch w4_next (4 VGPRs!) BEFORE 120-cycle trellis decode + dual fdot2!
  {
    const uint32_t* tp_base = qw + (size_t)i * nt * 32 + q * 4;
    int j = g;
    if (j < nt) {
      u4v w4_cur = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(tp_base + (size_t)j * 32));
      for (; j < nt; j += 4) {
        u4v w4_nxt = w4_cur;
        if (j + 4 < nt) {
          w4_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(tp_base + (size_t)(j + 4) * 32));
        }
        uint32_t V[5];
        V[0] = __builtin_amdgcn_permlane16(w4_cur.w, w4_cur.w, 0x65432107u, 0xedcba98fu, false, false);
        V[1] = w4_cur.x; V[2] = w4_cur.y; V[3] = w4_cur.z; V[4] = w4_cur.w;
        h2v xv[M][4];
#pragma unroll
        for (int r = 0; r < M; ++r) {
          const uint4* xp = reinterpret_cast<const uint4*>(xt + (size_t)r * n + j * 16);
          uint4 u0 = xp[0];
          xv[r][0] = as_h2(u0.x); xv[r][1] = as_h2(u0.y); xv[r][2] = as_h2(u0.z); xv[r][3] = as_h2(u0.w);
        }
        auto _sp = [&](int ua, int ub, uint32_t sa0, uint32_t sa1, uint32_t sb0, uint32_t sb1) {
          h2v wa = as_h2(val_f16bits2(sa0, sa1));
          h2v wb = as_h2(val_f16bits2(sb0, sb1));
#pragma unroll
          for (int r = 0; r < M; ++r) {
            a0[r] = __builtin_amdgcn_fdot2(wa, xv[r][(ua & 7) >> 1], a0[r], false);
            a1[r] = __builtin_amdgcn_fdot2(wb, xv[r][(ub & 7) >> 1], a1[r], false);
          }
        };
        _sp(0, 18, ext16<16 +  1*4>(V), ext16<16 +  2*4>(V), ext16<16 + 19*4>(V), ext16<16 + 20*4>(V));
        _sp(2, 16, ext16<16 +  3*4>(V), ext16<16 +  4*4>(V), ext16<16 + 17*4>(V), ext16<16 + 18*4>(V));
        _sp(4, 22, ext16<16 +  5*4>(V), ext16<16 +  6*4>(V), ext16<16 + 23*4>(V), ext16<16 + 24*4>(V));
        _sp(6, 20, ext16<16 +  7*4>(V), ext16<16 +  8*4>(V), ext16<16 + 21*4>(V), ext16<16 + 22*4>(V));
#pragma unroll
        for (int r = 0; r < M; ++r) {
          const uint4* xp = reinterpret_cast<const uint4*>(xt + (size_t)r * n + j * 16);
          uint4 u1 = xp[1];
          xv[r][0] = as_h2(u1.x); xv[r][1] = as_h2(u1.y); xv[r][2] = as_h2(u1.z); xv[r][3] = as_h2(u1.w);
        }
        _sp(8,  26, ext16<16 +  9*4>(V), ext16<16 + 10*4>(V), ext16<16 + 27*4>(V), ext16<16 + 28*4>(V));
        _sp(10, 24, ext16<16 + 11*4>(V), ext16<16 + 12*4>(V), ext16<16 + 25*4>(V), ext16<16 + 26*4>(V));
        _sp(12, 30, ext16<16 + 13*4>(V), ext16<16 + 14*4>(V), ext16<16 + 31*4>(V), ext16<16 + 32*4>(V));
        _sp(14, 28, ext16<16 + 15*4>(V), ext16<16 + 16*4>(V), ext16<16 + 29*4>(V), ext16<16 + 30*4>(V));
        w4_cur = w4_nxt;
      }
    }
#pragma unroll
    for (int r = 0; r < M; ++r) {
      a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
      a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
    }
  }
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
      ep_lds[r][w * 16 + 2 * q]     = a0[r];
      ep_lds[r][w * 16 + 2 * q + 1] = a1[r];
    }
  }
  __syncthreads();
  // All M rows (M <= 8) execute their 128-elem FWHT + output write IN PARALLEL across waves 0..M-1!
  if (w < M) {
    const int r = w;
    const int base = b * 128 + lane * 4;
    const float4 s4 = *reinterpret_cast<const float4*>(&ep_lds[r][lane * 4]);
    const float c = 1534.0f * xs[r], t = tok[r];
    float v[4] = {
      (s4.x - c) * __half2float(scales[base + 0]) * cb * t,
      (s4.y - c) * __half2float(scales[base + 1]) * cb * t,
      (s4.z - c) * __half2float(scales[base + 2]) * cb * t,
      (s4.w - c) * __half2float(scales[base + 3]) * cb * t,
    };
    fwht128_wave(v, lane);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int row = base + e;
      float o = v[e] * (1.0f / 11.313708498984761f);
      if ((sv[row >> 3] >> (row & 7)) & 1) o = -o;
      if (bias) o += to_f(bias[row]);
      y[(size_t)r * m + row] = from_f<TO>(o);
    }
  }
}

template <int M, typename TO>
__global__ __launch_bounds__(256) void k_gemv_k4_b128_epi(
    const uint32_t* __restrict__ qw, const _Float16* __restrict__ xt,
    float* __restrict__ part, unsigned* __restrict__ counters, const float* __restrict__ xs, const float* __restrict__ tok,
    const __half* __restrict__ scales, float cb, const uint8_t* __restrict__ sv, const TO* __restrict__ bias,
    TO* __restrict__ y, int mt, int nt, int nsplit, int tps) {
  const int lane = threadIdx.x & 31;
  const int wave = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (wave >= mt * nsplit) return;
  const int i = wave / nsplit, sp = wave % nsplit;
  const int g = lane >> 3, q = lane & 7, n = nt * 16, m = mt * 16;
  float a0[M], a1[M];
#pragma unroll
  for (int r = 0; r < M; ++r) { a0[r] = 0.f; a1[r] = 0.f; }
  const int j0 = sp * tps, j1 = min(nt, j0 + tps);
  K4B128_ACCUM_LOOP(M)
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
      float* pp = part + ((size_t)sp * M + r) * m + i * 16 + 2 * q;
      pp[0] = a0[r]; pp[1] = a1[r];
    }
  }
  const int b = i >> 3;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  unsigned old = 0;
  if (lane == 0) old = atomicAdd(&counters[b], 1u);
  old = __shfl(old, 0);
  if (old != (unsigned)(8 * nsplit - 1)) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  const int base = b * 128 + lane * 4;
  for (int r = 0; r < M; ++r) {
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < nsplit; ++k) {
      const float* pp = part + ((size_t)k * M + r) * m + base;
#pragma unroll
      for (int e = 0; e < 4; ++e) v[e] += __builtin_nontemporal_load(pp + e);
    }
    const float c = 1534.0f * xs[r], t = tok[r];
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = (v[e] - c) * __half2float(scales[base + e]) * cb * t;
    fwht128_wave(v, lane);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int row = base + e;
      float o = v[e] * (1.0f / 11.313708498984761f);
      if ((sv[row >> 3] >> (row & 7)) & 1) o = -o;
      if (bias != nullptr) o += to_f(bias[row]);
      y[(size_t)r * m + row] = from_f<TO>(o);
    }
  }
  if (lane == 0) counters[b] = 0u;
}

template <int M, typename TO>
__global__ __launch_bounds__(256) void k_gemv_k4_b128_epi8(
    const uint32_t* __restrict__ qw, const _Float16* __restrict__ xt,
    float* __restrict__ part, unsigned* __restrict__ counters, const float* __restrict__ xs, const float* __restrict__ tok,
    const __half* __restrict__ scales, float cb, const uint8_t* __restrict__ sv, const TO* __restrict__ bias,
    TO* __restrict__ y, int mt, int nt, int nsplit, int tps) {
  __shared__ float red[8][M][16];
  __shared__ unsigned is_last;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int wave = blockIdx.x * 8 + w;
  const int i = wave / nsplit, sp = wave % nsplit;
  const int g = lane >> 3, q = lane & 7, n = nt * 16, m = mt * 16, ns8 = nsplit >> 3;
  float a0[M], a1[M];
#pragma unroll
  for (int r = 0; r < M; ++r) { a0[r] = 0.f; a1[r] = 0.f; }
  const int j0 = sp * tps, j1 = min(nt, j0 + tps);
  K4B128_ACCUM_LOOP(M)
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M; ++r) { red[w][r][2 * q] = a0[r]; red[w][r][2 * q + 1] = a1[r]; }
  }
  __syncthreads();
  if (threadIdx.x < M * 16) {
    const int r = threadIdx.x >> 4, c = threadIdx.x & 15;
    float s = 0.f;
#pragma unroll
    for (int ww = 0; ww < 8; ++ww) s += red[ww][r][c];
    part[((size_t)(sp >> 3) * M + r) * m + i * 16 + c] = s;
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();
  const int b = i >> 3;
  if (threadIdx.x == 0) is_last = (atomicAdd(&counters[b], 1u) == (unsigned)(nsplit - 1));
  __syncthreads();
  if (!is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  const int base = b * 128 + lane * 4;
  for (int r = w; r < M; r += 8) {
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < ns8; ++k) {
      const float* pp = part + ((size_t)k * M + r) * m + base;
#pragma unroll
      for (int e = 0; e < 4; ++e) v[e] += __builtin_nontemporal_load(pp + e);
    }
    const float c = 1534.0f * xs[r], t = tok[r];
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = (v[e] - c) * __half2float(scales[base + e]) * cb * t;
    fwht128_wave(v, lane);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int row = base + e;
      float o = v[e] * (1.0f / 11.313708498984761f);
      if ((sv[row >> 3] >> (row & 7)) & 1) o = -o;
      if (bias != nullptr) o += to_f(bias[row]);
      y[(size_t)r * m + row] = from_f<TO>(o);
    }
  }
  if (threadIdx.x == 0) counters[b] = 0u;
}

template <typename TO>
static void dispatch_k4_epi_t(int M, bool epi8, int blocks, hipStream_t st,
                              const uint32_t* qw, const _Float16* xp,
                              float* part, unsigned* cnt, const float* xs, const float* tok,
                              const __half* sc, float cb, const uint8_t* sv, const TO* bp, TO* yp,
                              int mt, int nt, int ns, int tps) {
  // v0.2.7: For tall K=4 (mt >= 1024 -> >=128 WGs, e.g. lm_head=1,940 WGs), skip split-K completely!
  if (mt >= 1024 && (mt & 7) == 0 && M >= 1 && M <= 8) {
    const int nblocks = mt >> 3;
    switch (M) {
      case 1: hipLaunchKernelGGL((k_gemv_k4_b128_ns1<1, TO>), dim3(nblocks), dim3(256), 0, st, qw, xp, xs, tok, sc, cb, sv, bp, yp, mt, nt); return;
      case 2: hipLaunchKernelGGL((k_gemv_k4_b128_ns1<2, TO>), dim3(nblocks), dim3(256), 0, st, qw, xp, xs, tok, sc, cb, sv, bp, yp, mt, nt); return;
      case 3: hipLaunchKernelGGL((k_gemv_k4_b128_ns1<3, TO>), dim3(nblocks), dim3(256), 0, st, qw, xp, xs, tok, sc, cb, sv, bp, yp, mt, nt); return;
      case 4: hipLaunchKernelGGL((k_gemv_k4_b128_ns1<4, TO>), dim3(nblocks), dim3(256), 0, st, qw, xp, xs, tok, sc, cb, sv, bp, yp, mt, nt); return;
      case 5: hipLaunchKernelGGL((k_gemv_k4_b128_ns1<5, TO>), dim3(nblocks), dim3(256), 0, st, qw, xp, xs, tok, sc, cb, sv, bp, yp, mt, nt); return;
      case 6: hipLaunchKernelGGL((k_gemv_k4_b128_ns1<6, TO>), dim3(nblocks), dim3(256), 0, st, qw, xp, xs, tok, sc, cb, sv, bp, yp, mt, nt); return;
      case 7: hipLaunchKernelGGL((k_gemv_k4_b128_ns1<7, TO>), dim3(nblocks), dim3(256), 0, st, qw, xp, xs, tok, sc, cb, sv, bp, yp, mt, nt); return;
      case 8: hipLaunchKernelGGL((k_gemv_k4_b128_ns1<8, TO>), dim3(nblocks), dim3(256), 0, st, qw, xp, xs, tok, sc, cb, sv, bp, yp, mt, nt); return;
    }
  }
  if (epi8) {
    switch (M) {
#define K4E8(MM) case MM: hipLaunchKernelGGL((k_gemv_k4_b128_epi8<MM, TO>), dim3(blocks), dim3(256), 0, st, qw, xp, part, cnt, xs, tok, sc, cb, sv, bp, yp, mt, nt, ns, tps); break;
      K4E8(1) K4E8(2) K4E8(3) K4E8(4) K4E8(5) K4E8(6) K4E8(7) K4E8(8)
#undef K4E8
    }
  } else {
    switch (M) {
#define K4E1(MM) case MM: hipLaunchKernelGGL((k_gemv_k4_b128_epi<MM, TO>), dim3(blocks), dim3(256), 0, st, qw, xp, part, cnt, xs, tok, sc, cb, sv, bp, yp, mt, nt, ns, tps); break;
      K4E1(1) K4E1(2) K4E1(3) K4E1(4) K4E1(5) K4E1(6) K4E1(7) K4E1(8)
#undef K4E1
    }
  }
}

torch::Tensor gemv_k4_b128_epi(torch::Tensor qw, torch::Tensor xt, torch::Tensor xs, torch::Tensor tok,
                               torch::Tensor scales, double cb, torch::Tensor sv, c10::optional<torch::Tensor> bias,
                               torch::Tensor counters, int64_t nsplit, int64_t out_dtype, int64_t epi8) {
  TORCH_CHECK(qw.dim() == 3 && qw.size(2) == 32 && qw.scalar_type() == torch::kInt32 && qw.is_contiguous());
  int mt = qw.size(0), nt = qw.size(1), M = xt.size(0), m = mt * 16;
  TORCH_CHECK(M >= 1 && M <= 8 && (m % 128 == 0));
  const int tps = ((nt + nsplit - 1) / nsplit + 3) / 4 * 4, ns = (nt + tps - 1) / tps;
  const bool use8 = (epi8 != 0) && ((ns & 7) == 0);
  auto part = torch::empty({use8 ? (ns >> 3) : ns, M, m}, xt.options().dtype(torch::kFloat32));
  auto dt = out_dtype == 0 ? torch::kBFloat16 : (out_dtype == 1 ? torch::kHalf : torch::kFloat32);
  auto y = torch::empty({M, m}, xt.options().dtype(dt));
  int blocks = (mt * ns + 7) / 8;
  hipStream_t st = at::hip::getCurrentHIPStream();
  auto qwp = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>());
  auto xp = reinterpret_cast<const _Float16*>(xt.data_ptr<at::Half>());
  auto cnt = reinterpret_cast<unsigned*>(counters.data_ptr<int32_t>());
  auto sc = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  if (out_dtype == 0) {
    const __hip_bfloat16* bp = (bias.has_value() && bias->defined()) ? reinterpret_cast<const __hip_bfloat16*>(bias->data_ptr<at::BFloat16>()) : nullptr;
    dispatch_k4_epi_t<__hip_bfloat16>(M, use8, blocks, st, qwp, xp, part.data_ptr<float>(), cnt, xs.data_ptr<float>(), tok.data_ptr<float>(), sc, (float)cb, sv.data_ptr<uint8_t>(), bp, reinterpret_cast<__hip_bfloat16*>(y.data_ptr<at::BFloat16>()), mt, nt, ns, tps);
  } else if (out_dtype == 1) {
    const __half* bp = (bias.has_value() && bias->defined()) ? reinterpret_cast<const __half*>(bias->data_ptr<at::Half>()) : nullptr;
    dispatch_k4_epi_t<__half>(M, use8, blocks, st, qwp, xp, part.data_ptr<float>(), cnt, xs.data_ptr<float>(), tok.data_ptr<float>(), sc, (float)cb, sv.data_ptr<uint8_t>(), bp, reinterpret_cast<__half*>(y.data_ptr<at::Half>()), mt, nt, ns, tps);
  } else {
    const float* bp = (bias.has_value() && bias->defined()) ? bias->data_ptr<float>() : nullptr;
    dispatch_k4_epi_t<float>(M, use8, blocks, st, qwp, xp, part.data_ptr<float>(), cnt, xs.data_ptr<float>(), tok.data_ptr<float>(), sc, (float)cb, sv.data_ptr<uint8_t>(), bp, y.data_ptr<float>(), mt, nt, ns, tps);
  }
  return y;
}

PYBIND11_MODULE(k5_splitplane, m) {
  m.def("repack_k5", &repack_k5);
  m.def("gemv_k5_sp16", &gemv_k5_sp16);
  m.def("gemv_k5_sp16_epi", &gemv_k5_sp16_epi);
  m.def("gemv_k4_b128_epi", &gemv_k4_b128_epi);
}
