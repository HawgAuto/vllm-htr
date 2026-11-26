// Native gfx1151 (RDNA3.5) fused ERNIE-5.5 mHC (Manifold-Constrained Hyper-Connections) kernels.
// Collapses ~55 PyTorch launches per sublayer (mhc_post + fn GEMV + ERNIE RMS + 20-iter Sinkhorn
// + weighted pre-sum + RMSNorm) into 2 Wave32 HIP kernels for decode (T <= 16) and 3 for prefill.
// Uses Wave32 DPP reductions, 16-byte vectorized non-temporal loads for fn (Infinity Cache bypass),
// and register-only 20-iteration Sinkhorn-Knopp.
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <math.h>

typedef float f4v __attribute__((ext_vector_type(4)));
typedef uint32_t u2v __attribute__((ext_vector_type(2)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));
typedef u4v u4va __attribute__((aligned(4)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));

// Native RDNA 3.5 (gfx1151) hardware V_DOT2_F32_BF16 + V_PERM_B32 BF16x2 pack
__device__ __forceinline__ float bf16_dot2(uint32_t a, uint32_t b, float c) {
  return __builtin_amdgcn_fdot2_f32_bf16(
      __builtin_bit_cast(bf16x2, a), __builtin_bit_cast(bf16x2, b), c, false);
}
__device__ __forceinline__ uint32_t f32_to_bf16x2_rne(float a, float b) {
  uint32_t ua = __float_as_uint(a);
  uint32_t ub = __float_as_uint(b);
  ua = ((ua & 0x7fffffffu) > 0x7f800000u) ? (ua | 0x00400000u) : (ua + 0x7fffu + ((ua >> 16) & 1u));
  ub = ((ub & 0x7fffffffu) > 0x7f800000u) ? (ub | 0x00400000u) : (ub + 0x7fffu + ((ub >> 16) & 1u));
  return __builtin_amdgcn_perm(ub, ua, 0x07060302u);
}

template <int MSK>
__device__ __forceinline__ float shx(float v) {
  const int iv = __builtin_bit_cast(int, v);
  if constexpr (MSK < 16) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, iv, 0x160 | MSK, 0xF, 0xF, false));
  } else {
    return __builtin_bit_cast(float, (int)__builtin_amdgcn_permlanex16((unsigned)iv, (unsigned)iv, 0x76543210u, 0xfedcba98u, false, false));
  }
}

__device__ __forceinline__ float wave_sum(float a) {
  a += shx<16>(a);
  a += shx<8>(a);
  a += shx<4>(a);
  a += shx<2>(a);
  a += shx<1>(a);
  return a;
}

__device__ __forceinline__ float bf2f(uint16_t b) {
  return __builtin_bit_cast(float, (uint32_t)b << 16);
}

__device__ __forceinline__ uint16_t f2bf(float f) {
  __hip_bfloat16 b = __float2bfloat16(f);
  return __builtin_bit_cast(uint16_t, b);
}

__device__ __forceinline__ float fast_sigmoid(float x) {
  return __builtin_amdgcn_rcpf(1.0f + exp2f(-1.4426950408889634f * x));
}

// ---------------------------------------------------------------------------
// Kernel 1 (Decode, H = 4096, hc_mult = 4, mix_hc = 24, SPLIT_H = 32 -> 128 cols/WG)
// DO_POST=1: fused mhc_post + residual_out write + partial sum-of-squares + partial fn @ r_cur
// DO_POST=0: mhc_pre only (reads residual directly, no write)
// Grid: (32, T), Block: 256 (8 waves of 32)
// ---------------------------------------------------------------------------
template <int DO_POST>
__global__ __launch_bounds__(256) void k_mhc_post_dot_4096(
    const uint16_t* __restrict__ x,          // [T, 4096] bf16 (only used if DO_POST)
    const uint16_t* __restrict__ res_in,     // [T, 4, 4096] bf16
    const float* __restrict__ post_mix,      // [T, 4] fp32 (only used if DO_POST)
    const float* __restrict__ comb_mix,      // [T, 4, 4] fp32 (only used if DO_POST)
    const float* __restrict__ fn,            // [24, 16384] fp32
    uint16_t* __restrict__ res_out,          // [T, 4, 4096] bf16 (only written if DO_POST)
    float* __restrict__ part                 // [T, 32, 25] fp32 (0..23: dots, 24: ss)
) {
  constexpr int H = 4096;
  constexpr int HC = 4;
  constexpr int C = 128; // cols per WG
  __shared__ __attribute__((aligned(16))) float s_rf[HC][C];
  __shared__ float s_ss_w[HC];

  const int wg = blockIdx.x;
  const int t = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int w = tid >> 5;
  const int h0 = wg * C;

  // Waves 0..3 process HC streams j=0..3 in parallel (4x parallel mhc_post at T=1 + wave-uniform scalar cm/pm loads!)
  if (w < HC) {
    const int c4 = lane * 4;
    const int h = h0 + c4;
    const uint16_t* r_base = res_in + (size_t)t * (HC * H) + h;
    float ss0 = 0.f, ss1 = 0.f;
    if constexpr (DO_POST) {
      const uint2 raw0 = *reinterpret_cast<const uint2*>(r_base + 0 * H);
      const uint2 raw1 = *reinterpret_cast<const uint2*>(r_base + 1 * H);
      const uint2 raw2 = *reinterpret_cast<const uint2*>(r_base + 2 * H);
      const uint2 raw3 = *reinterpret_cast<const uint2*>(r_base + 3 * H);
      const uint2 xraw = *reinterpret_cast<const uint2*>(x + (size_t)t * H + h);
      const float r0[4] = {__uint_as_float((raw0.x & 0xFFFFu) << 16), __uint_as_float(raw0.x & 0xFFFF0000u),
                           __uint_as_float((raw0.y & 0xFFFFu) << 16), __uint_as_float(raw0.y & 0xFFFF0000u)};
      const float r1[4] = {__uint_as_float((raw1.x & 0xFFFFu) << 16), __uint_as_float(raw1.x & 0xFFFF0000u),
                           __uint_as_float((raw1.y & 0xFFFFu) << 16), __uint_as_float(raw1.y & 0xFFFF0000u)};
      const float r2[4] = {__uint_as_float((raw2.x & 0xFFFFu) << 16), __uint_as_float(raw2.x & 0xFFFF0000u),
                           __uint_as_float((raw2.y & 0xFFFFu) << 16), __uint_as_float(raw2.y & 0xFFFF0000u)};
      const float r3[4] = {__uint_as_float((raw3.x & 0xFFFFu) << 16), __uint_as_float(raw3.x & 0xFFFF0000u),
                           __uint_as_float((raw3.y & 0xFFFFu) << 16), __uint_as_float(raw3.y & 0xFFFF0000u)};
      const float xv[4] = {__uint_as_float((xraw.x & 0xFFFFu) << 16), __uint_as_float(xraw.x & 0xFFFF0000u),
                           __uint_as_float((xraw.y & 0xFFFFu) << 16), __uint_as_float(xraw.y & 0xFFFF0000u)};
      const float* pm = post_mix + (size_t)t * 4;
      const float* cm = comb_mix + (size_t)t * 16;
      const float c0 = cm[0 * 4 + w], c1 = cm[1 * 4 + w], c2 = cm[2 * 4 + w], c3 = cm[3 * 4 + w], p = pm[w];
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) v[e] = c0 * r0[e] + c1 * r1[e] + c2 * r2[e] + c3 * r3[e] + p * xv[e];
      uint2 ob;
      ob.x = f32_to_bf16x2_rne(v[0], v[1]);
      ob.y = f32_to_bf16x2_rne(v[2], v[3]);
      uint16_t* ro = res_out + (size_t)t * (HC * H) + h;
      *reinterpret_cast<uint2*>(ro + w * H) = ob;
      *reinterpret_cast<f4v*>(&s_rf[w][c4]) = (f4v){
        __uint_as_float((ob.x & 0xFFFFu) << 16), __uint_as_float(ob.x & 0xFFFF0000u),
        __uint_as_float((ob.y & 0xFFFFu) << 16), __uint_as_float(ob.y & 0xFFFF0000u)
      };
      ss0 = bf16_dot2(ob.x, ob.x, ss0);
      ss1 = bf16_dot2(ob.y, ob.y, ss1);
    } else {
      const uint2 raw = *reinterpret_cast<const uint2*>(r_base + w * H);
      *reinterpret_cast<f4v*>(&s_rf[w][c4]) = (f4v){
        __uint_as_float((raw.x & 0xFFFFu) << 16), __uint_as_float(raw.x & 0xFFFF0000u),
        __uint_as_float((raw.y & 0xFFFFu) << 16), __uint_as_float(raw.y & 0xFFFF0000u)
      };
      ss0 = bf16_dot2(raw.x, raw.x, ss0);
      ss1 = bf16_dot2(raw.y, raw.y, ss1);
    }
    const float ss = wave_sum(ss0 + ss1);
    if (lane == 0) s_ss_w[w] = ss;
  }
  __syncthreads();

  // All 8 waves compute 3 rows of fn each (8 * 3 = 24 rows), vectorized 16B NT loads
  const int m0 = w * 3;
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const f4v rv = *reinterpret_cast<const f4v*>(&s_rf[j][lane * 4]);
    const size_t col_off = (size_t)j * H + h0 + lane * 4;
    const f4v w0 = __builtin_nontemporal_load(reinterpret_cast<const f4v*>(fn + (size_t)(m0 + 0) * (HC * H) + col_off));
    const f4v w1 = __builtin_nontemporal_load(reinterpret_cast<const f4v*>(fn + (size_t)(m0 + 1) * (HC * H) + col_off));
    const f4v w2 = __builtin_nontemporal_load(reinterpret_cast<const f4v*>(fn + (size_t)(m0 + 2) * (HC * H) + col_off));
    acc0 = fmaf(w0.x, rv.x, fmaf(w0.y, rv.y, fmaf(w0.z, rv.z, fmaf(w0.w, rv.w, acc0))));
    acc1 = fmaf(w1.x, rv.x, fmaf(w1.y, rv.y, fmaf(w1.z, rv.z, fmaf(w1.w, rv.w, acc1))));
    acc2 = fmaf(w2.x, rv.x, fmaf(w2.y, rv.y, fmaf(w2.z, rv.z, fmaf(w2.w, rv.w, acc2))));
  }
  acc0 = wave_sum(acc0);
  acc1 = wave_sum(acc1);
  acc2 = wave_sum(acc2);

  float* pout = part + ((size_t)t * 32 + wg) * 25;
  if (lane == 0) {
    pout[m0 + 0] = acc0;
    pout[m0 + 1] = acc1;
    pout[m0 + 2] = acc2;
    if (w == 0) {
      pout[24] = s_ss_w[0] + s_ss_w[1] + s_ss_w[2] + s_ss_w[3];
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel 1B (Tiled Multi-Token Decode & Zero-xf Prefill, BT = 4 / 8 / 16):
//   - All 8 waves run mhc_post in parallel (0 idle waves!) & pack BF16x2 into s_rb[BT][4][64] (4..16 KiB LDS)
//   - All 8 waves read fn[24, 16384] ONCE per BT tokens (saves 903 MiB at T=8, 1.94 GiB at T=16!)
//   - For T > 16 (prefill), NT_FN=0 keeps fn (1.5 MiB) 100% resident in the 32 MiB Infinity Cache
//     and completely eliminates the 134 MB xf FP32 DRAM round-trip (saves 23.1 GB per 2048-tok prefill!)
// Grid: (32, ceil(T / BT)), Block: 256 (8 Wave32s)
// ---------------------------------------------------------------------------
template <int DO_POST, int BT, int NT_FN, int NT_IN = 0>
__global__ __launch_bounds__(256) void k_mhc_post_dot_4096_tiled(
    const uint16_t* __restrict__ x,          // [T, 4096] bf16 (only used if DO_POST)
    const uint16_t* __restrict__ res_in,     // [T, 4, 4096] bf16
    const float* __restrict__ post_mix,      // [T, 4] fp32 (only used if DO_POST)
    const float* __restrict__ comb_mix,      // [T, 4, 4] fp32 (only used if DO_POST)
    const float* __restrict__ fn,            // [24, 16384] fp32
    uint16_t* __restrict__ res_out,          // [T, 4, 4096] bf16 (only written if DO_POST)
    float* __restrict__ part,                // [T, 32, 25] fp32
    int T
) {
  constexpr int H = 4096;
  constexpr int HC = 4;
  constexpr int C = 128;
  __shared__ __attribute__((aligned(16))) uint32_t s_rb[BT][HC][64]; // packed BF16x2 (4..16 KiB)
  __shared__ float s_ss[BT];

  const int wg = blockIdx.x;
  const int t0 = blockIdx.y * BT;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int w = tid >> 5;
  const int h0 = wg * C;
  const int c4 = lane * 4;
  const int h = h0 + c4;

  // Phase 1: All 8 waves compute mhc_post + ss + store packed BF16x2 to LDS in parallel!
#pragma unroll
  for (int bt = w; bt < BT; bt += 8) {
    const int t = t0 + bt;
    if (t < T) {
      const uint16_t* r_base = res_in + (size_t)t * (HC * H) + h;
      auto ld64 = [](const uint16_t* p) -> uint2 {
        if constexpr (NT_IN) {
          uint64_t q = __builtin_nontemporal_load(reinterpret_cast<const uint64_t*>(p));
          return make_uint2((uint32_t)q, (uint32_t)(q >> 32));
        } else {
          return *reinterpret_cast<const uint2*>(p);
        }
      };
      const uint2 raw0 = ld64(r_base + 0 * H);
      const uint2 raw1 = ld64(r_base + 1 * H);
      const uint2 raw2 = ld64(r_base + 2 * H);
      const uint2 raw3 = ld64(r_base + 3 * H);
      float ss0 = 0.f, ss1 = 0.f;
      if constexpr (DO_POST) {
        const float r0[4] = {__uint_as_float((raw0.x & 0xFFFFu) << 16), __uint_as_float(raw0.x & 0xFFFF0000u),
                             __uint_as_float((raw0.y & 0xFFFFu) << 16), __uint_as_float(raw0.y & 0xFFFF0000u)};
        const float r1[4] = {__uint_as_float((raw1.x & 0xFFFFu) << 16), __uint_as_float(raw1.x & 0xFFFF0000u),
                             __uint_as_float((raw1.y & 0xFFFFu) << 16), __uint_as_float(raw1.y & 0xFFFF0000u)};
        const float r2[4] = {__uint_as_float((raw2.x & 0xFFFFu) << 16), __uint_as_float(raw2.x & 0xFFFF0000u),
                             __uint_as_float((raw2.y & 0xFFFFu) << 16), __uint_as_float(raw2.y & 0xFFFF0000u)};
        const float r3[4] = {__uint_as_float((raw3.x & 0xFFFFu) << 16), __uint_as_float(raw3.x & 0xFFFF0000u),
                             __uint_as_float((raw3.y & 0xFFFFu) << 16), __uint_as_float(raw3.y & 0xFFFF0000u)};
        const uint2 xraw = ld64(x + (size_t)t * H + h);
        const float xv[4] = {__uint_as_float((xraw.x & 0xFFFFu) << 16), __uint_as_float(xraw.x & 0xFFFF0000u),
                             __uint_as_float((xraw.y & 0xFFFFu) << 16), __uint_as_float(xraw.y & 0xFFFF0000u)};
        const f4v pmv = *reinterpret_cast<const f4v*>(post_mix + (size_t)t * 4);
        const f4v cm0 = *reinterpret_cast<const f4v*>(comb_mix + (size_t)t * 16 + 0);
        const f4v cm1 = *reinterpret_cast<const f4v*>(comb_mix + (size_t)t * 16 + 4);
        const f4v cm2 = *reinterpret_cast<const f4v*>(comb_mix + (size_t)t * 16 + 8);
        const f4v cm3 = *reinterpret_cast<const f4v*>(comb_mix + (size_t)t * 16 + 12);
        const float pm_a[4] = {pmv.x, pmv.y, pmv.z, pmv.w};
        const float cm_a[4][4] = {
          {cm0.x, cm0.y, cm0.z, cm0.w}, {cm1.x, cm1.y, cm1.z, cm1.w},
          {cm2.x, cm2.y, cm2.z, cm2.w}, {cm3.x, cm3.y, cm3.z, cm3.w},
        };
        uint16_t* ro = res_out + (size_t)t * (HC * H) + h;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float c0 = cm_a[0][j], c1 = cm_a[1][j], c2 = cm_a[2][j], c3 = cm_a[3][j], p = pm_a[j];
          float v[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) v[e] = c0 * r0[e] + c1 * r1[e] + c2 * r2[e] + c3 * r3[e] + p * xv[e];
          uint2 ob;
          ob.x = f32_to_bf16x2_rne(v[0], v[1]);
          ob.y = f32_to_bf16x2_rne(v[2], v[3]);
          *reinterpret_cast<uint2*>(ro + j * H) = ob;
          *reinterpret_cast<uint2*>(&s_rb[bt][j][lane * 2]) = ob;
          ss0 = bf16_dot2(ob.x, ob.x, ss0);
          ss1 = bf16_dot2(ob.y, ob.y, ss1);
        }
      } else {
        *reinterpret_cast<uint2*>(&s_rb[bt][0][lane * 2]) = raw0;
        *reinterpret_cast<uint2*>(&s_rb[bt][1][lane * 2]) = raw1;
        *reinterpret_cast<uint2*>(&s_rb[bt][2][lane * 2]) = raw2;
        *reinterpret_cast<uint2*>(&s_rb[bt][3][lane * 2]) = raw3;
        ss0 = bf16_dot2(raw0.x, raw0.x, ss0); ss1 = bf16_dot2(raw0.y, raw0.y, ss1);
        ss0 = bf16_dot2(raw1.x, raw1.x, ss0); ss1 = bf16_dot2(raw1.y, raw1.y, ss1);
        ss0 = bf16_dot2(raw2.x, raw2.x, ss0); ss1 = bf16_dot2(raw2.y, raw2.y, ss1);
        ss0 = bf16_dot2(raw3.x, raw3.x, ss0); ss1 = bf16_dot2(raw3.y, raw3.y, ss1);
      }
      const float ss = wave_sum(ss0 + ss1);
      if (lane == 0) s_ss[bt] = ss;
    }
  }
  __syncthreads();

  // Phase 2: All 8 waves read fn ONCE and accumulate across all BT tokens!
  const int m0 = w * 3;
  float acc0[BT], acc1[BT], acc2[BT];
#pragma unroll
  for (int bt = 0; bt < BT; ++bt) { acc0[bt] = 0.f; acc1[bt] = 0.f; acc2[bt] = 0.f; }

#pragma unroll 1
  for (int j = 0; j < 4; ++j) {
    const size_t col_off = (size_t)j * H + h;
    const f4v* p0 = reinterpret_cast<const f4v*>(fn + (size_t)(m0 + 0) * (HC * H) + col_off);
    const f4v* p1 = reinterpret_cast<const f4v*>(fn + (size_t)(m0 + 1) * (HC * H) + col_off);
    const f4v* p2 = reinterpret_cast<const f4v*>(fn + (size_t)(m0 + 2) * (HC * H) + col_off);
    const f4v w0 = NT_FN ? __builtin_nontemporal_load(p0) : *p0;
    const f4v w1 = NT_FN ? __builtin_nontemporal_load(p1) : *p1;
    const f4v w2 = NT_FN ? __builtin_nontemporal_load(p2) : *p2;
#pragma unroll
    for (int bt = 0; bt < BT; ++bt) {
      const uint2 rb = *reinterpret_cast<const uint2*>(&s_rb[bt][j][lane * 2]);
      const float r0 = __uint_as_float((rb.x & 0xFFFFu) << 16);
      const float r1 = __uint_as_float(rb.x & 0xFFFF0000u);
      const float r2 = __uint_as_float((rb.y & 0xFFFFu) << 16);
      const float r3 = __uint_as_float(rb.y & 0xFFFF0000u);
      acc0[bt] = fmaf(w0.x, r0, fmaf(w0.y, r1, fmaf(w0.z, r2, fmaf(w0.w, r3, acc0[bt]))));
      acc1[bt] = fmaf(w1.x, r0, fmaf(w1.y, r1, fmaf(w1.z, r2, fmaf(w1.w, r3, acc1[bt]))));
      acc2[bt] = fmaf(w2.x, r0, fmaf(w2.y, r1, fmaf(w2.z, r2, fmaf(w2.w, r3, acc2[bt]))));
      if constexpr (BT >= 8) {
        if ((bt & 3) == 3) __builtin_amdgcn_sched_barrier(0);
      }
    }
    __builtin_amdgcn_sched_barrier(0);
  }

  // Phase 3: Wave reduction & partial write for active tokens
#pragma unroll
  for (int bt = 0; bt < BT; ++bt) {
    const int t = t0 + bt;
    if (t >= T) break;
    const float a0 = wave_sum(acc0[bt]);
    const float a1 = wave_sum(acc1[bt]);
    const float a2 = wave_sum(acc2[bt]);
    if (lane == 0) {
      float* pout = part + ((size_t)t * 32 + wg) * 25;
      pout[m0 + 0] = a0;
      pout[m0 + 1] = a1;
      pout[m0 + 2] = a2;
      if (w == 0) pout[24] = s_ss[bt];
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel 2 (Decode & Prefill epilogue):
// Reduces partials (or reads precomputed dots+ss), runs exact ERNIE RMS + 20-iter Sinkhorn
// in registers, computes weighted pre-sum (rounded to bf16) + optional RMSNorm -> layer_input.
// Grid: (T,), Block: 256 (8 waves)
// ---------------------------------------------------------------------------
template <int FROM_PARTIALS, int HAS_NORM>
__global__ __launch_bounds__(256) void k_mhc_sinkhorn_norm_4096(
    const float* __restrict__ part_or_dots,  // FROM_PARTIALS=1: [T, 32, 25]; =0: [T, 24]
    const float* __restrict__ ss_in,         // FROM_PARTIALS=0: [T]
    const uint16_t* __restrict__ res_cur,    // [T, 4, 4096] bf16
    const float* __restrict__ hc_scale,      // [3] fp32
    const float* __restrict__ hc_base,       // [24] fp32
    const uint16_t* __restrict__ norm_w,     // [4096] bf16 (or nullptr if !HAS_NORM)
    float* __restrict__ post_out,            // [T, 4] fp32
    float* __restrict__ comb_out,            // [T, 4, 4] fp32
    uint16_t* __restrict__ layer_in,         // [T, 4096] bf16
    float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
    float hc_post_mult_value, int sinkhorn_repeat, float norm_eps
) {
  constexpr int H = 4096;
  constexpr int HC = 4;
  __shared__ __attribute__((aligned(16))) float s_part[800]; // 3.2 KiB: 200x coalesced 128-bit load instead of 128 uncoalesced lines!
  __shared__ float s_mix[24];
  __shared__ float s_ss_tot;
  __shared__ float s_pre[4];
  __shared__ float s_norm_ss[8];

  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int w = tid >> 5;

  if constexpr (FROM_PARTIALS) {
    // Cooperatively load all 32*25=800 fp32 partials via ONE coalesced global_load_b128 burst (200 threads x f4v)!
    const f4v* pt4 = reinterpret_cast<const f4v*>(part_or_dots + (size_t)t * (32 * 25));
    if (tid < 200) {
      *reinterpret_cast<f4v*>(&s_part[tid * 4]) = pt4[tid];
    }
    __syncthreads();
    // 8 waves reduce 25 slots across 32 split-WGs purely from LDS (zero uncoalesced global traffic!)
    for (int m = w; m < 25; m += 8) {
      float v = s_part[lane * 25 + m];
      v = wave_sum(v);
      if (lane == 0) {
        if (m < 24) s_mix[m] = v;
        else s_ss_tot = v;
      }
    }
    __syncthreads();
  } else {
    if (tid < 24) s_mix[tid] = part_or_dots[(size_t)t * 24 + tid];
    if (tid == 0) s_ss_tot = ss_in[t];
    __syncthreads();
  }

  // Wave 0 executes exact ERNIE RMS + 4-lane pre/post + 16-lane lockstep DPP Sinkhorn-Knopp (zero LDS, 1-op v_rcp_f32!)
  if (w == 0) {
    const float inv_rms = __builtin_amdgcn_rcpf(sqrtf(s_ss_tot * (1.0f / 16384.0f)) + rms_eps);
    const float sc0 = hc_scale[0];
    const float sc1 = hc_scale[1];
    const float sc2 = hc_scale[2];

    if (lane < 4) {
      float* po = post_out + (size_t)t * 4;
      s_pre[lane] = fast_sigmoid(s_mix[lane] * inv_rms * sc0 + hc_base[lane]) + hc_pre_eps;
      po[lane] = fast_sigmoid(s_mix[4 + lane] * inv_rms * sc1 + hc_base[4 + lane]) * hc_post_mult_value;
    }

    // Map 4x4 matrix onto lanes 0..15 (i = lane >> 2, j = lane & 3); all 32 lanes active for clean DPP16 shx!
    // shx<1>/shx<2> reduce across row j=0..3; shx<4>/shx<8> reduce across col i=0..3
    const int idx16 = lane & 15;
    const float val = s_mix[8 + idx16] * inv_rms * sc2 + hc_base[8 + idx16];
    float mx = fmaxf(val, shx<1>(val));
    mx = fmaxf(mx, shx<2>(mx));
    const float e = exp2f((val - mx) * 1.4426950408889634f);
    float sm = e + shx<1>(e);
    sm += shx<2>(sm);
    float c = e * __builtin_amdgcn_rcpf(sm) + hc_sinkhorn_eps;
    // First col norm (iter 0)
    float cs0 = c + shx<4>(c);
    cs0 += shx<8>(cs0);
    c *= __builtin_amdgcn_rcpf(cs0 + hc_sinkhorn_eps);
    // Remaining sinkhorn_repeat - 1 iterations in 16-way lockstep SIMD with 1-op V_RCP_F32
    for (int it = 1; it < sinkhorn_repeat; ++it) {
      float rs = c + shx<1>(c);
      rs += shx<2>(rs);
      c *= __builtin_amdgcn_rcpf(rs + hc_sinkhorn_eps);
      float cs = c + shx<4>(c);
      cs += shx<8>(cs);
      c *= __builtin_amdgcn_rcpf(cs + hc_sinkhorn_eps);
    }
    if (lane < 16) {
      comb_out[(size_t)t * 16 + lane] = c;
    }
  }
  __syncthreads();

  const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
  const uint16_t* r_base = res_cur + (size_t)t * (HC * H);
  uint16_t* lout = layer_in + (size_t)t * H;

  // 2 contiguous 8-bf16 (16B global_load_b128 / global_store_b128) passes = 8x fewer VMEM ops!
  uint32_t packed_v[2][4];
  float lss0 = 0.f, lss1 = 0.f;
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    const int h = p * 2048 + tid * 8;
    const u4v raw0 = *reinterpret_cast<const u4va*>(r_base + 0 * H + h);
    const u4v raw1 = *reinterpret_cast<const u4va*>(r_base + 1 * H + h);
    const u4v raw2 = *reinterpret_cast<const u4va*>(r_base + 2 * H + h);
    const u4v raw3 = *reinterpret_cast<const u4va*>(r_base + 3 * H + h);
    const uint32_t w0[4] = {raw0.x, raw0.y, raw0.z, raw0.w};
    const uint32_t w1[4] = {raw1.x, raw1.y, raw1.z, raw1.w};
    const uint32_t w2[4] = {raw2.x, raw2.y, raw2.z, raw2.w};
    const uint32_t w3[4] = {raw3.x, raw3.y, raw3.z, raw3.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float s0 = p0 * __uint_as_float((w0[k] & 0xFFFFu) << 16)
                     + p1 * __uint_as_float((w1[k] & 0xFFFFu) << 16)
                     + p2 * __uint_as_float((w2[k] & 0xFFFFu) << 16)
                     + p3 * __uint_as_float((w3[k] & 0xFFFFu) << 16);
      const float s1 = p0 * __uint_as_float(w0[k] & 0xFFFF0000u)
                     + p1 * __uint_as_float(w1[k] & 0xFFFF0000u)
                     + p2 * __uint_as_float(w2[k] & 0xFFFF0000u)
                     + p3 * __uint_as_float(w3[k] & 0xFFFF0000u);
      const uint32_t pk = f32_to_bf16x2_rne(s0, s1);
      packed_v[p][k] = pk;
      if constexpr (HAS_NORM) {
        if ((k & 1) == 0) lss0 = bf16_dot2(pk, pk, lss0);
        else              lss1 = bf16_dot2(pk, pk, lss1);
      }
    }
    if constexpr (!HAS_NORM) {
      *reinterpret_cast<u4va*>(lout + h) = (u4v){packed_v[p][0], packed_v[p][1], packed_v[p][2], packed_v[p][3]};
    }
  }

  if constexpr (HAS_NORM) {
    const float lss = wave_sum(lss0 + lss1);
    if (lane == 0) s_norm_ss[w] = lss;
    __syncthreads();
    const float tot_ss = s_norm_ss[0] + s_norm_ss[1] + s_norm_ss[2] + s_norm_ss[3]
                       + s_norm_ss[4] + s_norm_ss[5] + s_norm_ss[6] + s_norm_ss[7];
    const float rinv = rsqrtf(tot_ss * (1.0f / 4096.0f) + norm_eps);
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int h = p * 2048 + tid * 8;
      const u4v nw = *reinterpret_cast<const u4va*>(norm_w + h);
      const uint32_t nww[4] = {nw.x, nw.y, nw.z, nw.w};
      uint32_t out_w[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float v0 = __uint_as_float((packed_v[p][k] & 0xFFFFu) << 16);
        const float v1 = __uint_as_float(packed_v[p][k] & 0xFFFF0000u);
        const float w0 = __uint_as_float((nww[k] & 0xFFFFu) << 16);
        const float w1 = __uint_as_float(nww[k] & 0xFFFF0000u);
        out_w[k] = f32_to_bf16x2_rne(v0 * rinv * w0, v1 * rinv * w1);
      }
      *reinterpret_cast<u4va*>(lout + h) = (u4v){out_w[0], out_w[1], out_w[2], out_w[3]};
    }
  }
}

// ---------------------------------------------------------------------------
// Prefill helper: fused mhc_post -> writes res_out (bf16), x_f32 (fp32 for SGEMM), and ss (fp32)
// Grid: (T,), Block: 256
// ---------------------------------------------------------------------------
template <int DO_POST>
__global__ __launch_bounds__(256) void k_mhc_post_sq_4096(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ res_in,
    const float* __restrict__ post_mix,
    const float* __restrict__ comb_mix,
    uint16_t* __restrict__ res_out,
    float* __restrict__ x_f32_out,
    float* __restrict__ ss_out
) {
  constexpr int H = 4096;
  constexpr int HC = 4;
  __shared__ float s_ss[8];
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int w = tid >> 5;

  float pm[4], cm[16];
  if constexpr (DO_POST) {
    const f4v pmv = *reinterpret_cast<const f4v*>(post_mix + (size_t)t * 4);
    const f4v cm0 = *reinterpret_cast<const f4v*>(comb_mix + (size_t)t * 16 + 0);
    const f4v cm1 = *reinterpret_cast<const f4v*>(comb_mix + (size_t)t * 16 + 4);
    const f4v cm2 = *reinterpret_cast<const f4v*>(comb_mix + (size_t)t * 16 + 8);
    const f4v cm3 = *reinterpret_cast<const f4v*>(comb_mix + (size_t)t * 16 + 12);
    pm[0] = pmv.x; pm[1] = pmv.y; pm[2] = pmv.z; pm[3] = pmv.w;
    cm[0] = cm0.x; cm[1] = cm0.y; cm[2] = cm0.z; cm[3] = cm0.w;
    cm[4] = cm1.x; cm[5] = cm1.y; cm[6] = cm1.z; cm[7] = cm1.w;
    cm[8] = cm2.x; cm[9] = cm2.y; cm[10] = cm2.z; cm[11] = cm2.w;
    cm[12] = cm3.x; cm[13] = cm3.y; cm[14] = cm3.z; cm[15] = cm3.w;
  }

  const uint16_t* r_base = res_in + (size_t)t * (HC * H);
  uint16_t* ro_base = res_out ? (res_out + (size_t)t * (HC * H)) : nullptr;
  float* xf_base = x_f32_out ? (x_f32_out + (size_t)t * (HC * H)) : nullptr;

  // 2 contiguous 8-bf16 (16B global_load_b128 / global_store_b128) passes = 8x fewer VMEM ops!
  float ss0 = 0.f, ss1 = 0.f;
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    const int h = p * 2048 + tid * 8;
    const u4v raw0 = *reinterpret_cast<const u4va*>(r_base + 0 * H + h);
    const u4v raw1 = *reinterpret_cast<const u4va*>(r_base + 1 * H + h);
    const u4v raw2 = *reinterpret_cast<const u4va*>(r_base + 2 * H + h);
    const u4v raw3 = *reinterpret_cast<const u4va*>(r_base + 3 * H + h);
    const uint32_t rw0[4] = {raw0.x, raw0.y, raw0.z, raw0.w};
    const uint32_t rw1[4] = {raw1.x, raw1.y, raw1.z, raw1.w};
    const uint32_t rw2[4] = {raw2.x, raw2.y, raw2.z, raw2.w};
    const uint32_t rw3[4] = {raw3.x, raw3.y, raw3.z, raw3.w};

    if constexpr (DO_POST) {
      const u4v xraw = *reinterpret_cast<const u4va*>(x + (size_t)t * H + h);
      const uint32_t xw[4] = {xraw.x, xraw.y, xraw.z, xraw.w};
      float r0[8], r1[8], r2[8], r3[8], xv[8];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        r0[2 * k] = __uint_as_float((rw0[k] & 0xFFFFu) << 16); r0[2 * k + 1] = __uint_as_float(rw0[k] & 0xFFFF0000u);
        r1[2 * k] = __uint_as_float((rw1[k] & 0xFFFFu) << 16); r1[2 * k + 1] = __uint_as_float(rw1[k] & 0xFFFF0000u);
        r2[2 * k] = __uint_as_float((rw2[k] & 0xFFFFu) << 16); r2[2 * k + 1] = __uint_as_float(rw2[k] & 0xFFFF0000u);
        r3[2 * k] = __uint_as_float((rw3[k] & 0xFFFFu) << 16); r3[2 * k + 1] = __uint_as_float(rw3[k] & 0xFFFF0000u);
        xv[2 * k] = __uint_as_float((xw[k] & 0xFFFFu) << 16);  xv[2 * k + 1] = __uint_as_float(xw[k] & 0xFFFF0000u);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float c0 = cm[0 * 4 + j], c1 = cm[1 * 4 + j], c2 = cm[2 * 4 + j], c3 = cm[3 * 4 + j], pmj = pm[j];
        uint32_t ob[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float v0 = c0 * r0[2 * k] + c1 * r1[2 * k] + c2 * r2[2 * k] + c3 * r3[2 * k] + pmj * xv[2 * k];
          const float v1 = c0 * r0[2 * k + 1] + c1 * r1[2 * k + 1] + c2 * r2[2 * k + 1] + c3 * r3[2 * k + 1] + pmj * xv[2 * k + 1];
          ob[k] = f32_to_bf16x2_rne(v0, v1);
          if ((k & 1) == 0) ss0 = bf16_dot2(ob[k], ob[k], ss0);
          else              ss1 = bf16_dot2(ob[k], ob[k], ss1);
        }
        *reinterpret_cast<u4va*>(ro_base + j * H + h) = (u4v){ob[0], ob[1], ob[2], ob[3]};
        if (xf_base) {
          *reinterpret_cast<f4v*>(xf_base + j * H + h) = (f4v){
            __uint_as_float((ob[0] & 0xFFFFu) << 16), __uint_as_float(ob[0] & 0xFFFF0000u),
            __uint_as_float((ob[1] & 0xFFFFu) << 16), __uint_as_float(ob[1] & 0xFFFF0000u)
          };
          *reinterpret_cast<f4v*>(xf_base + j * H + h + 4) = (f4v){
            __uint_as_float((ob[2] & 0xFFFFu) << 16), __uint_as_float(ob[2] & 0xFFFF0000u),
            __uint_as_float((ob[3] & 0xFFFFu) << 16), __uint_as_float(ob[3] & 0xFFFF0000u)
          };
        }
      }
    } else {
      const u4v raws[4] = {raw0, raw1, raw2, raw3};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ss0 = bf16_dot2(raws[j].x, raws[j].x, ss0);
        ss1 = bf16_dot2(raws[j].y, raws[j].y, ss1);
        ss0 = bf16_dot2(raws[j].z, raws[j].z, ss0);
        ss1 = bf16_dot2(raws[j].w, raws[j].w, ss1);
        if (xf_base) {
          *reinterpret_cast<f4v*>(xf_base + j * H + h) = (f4v){
            __uint_as_float((raws[j].x & 0xFFFFu) << 16), __uint_as_float(raws[j].x & 0xFFFF0000u),
            __uint_as_float((raws[j].y & 0xFFFFu) << 16), __uint_as_float(raws[j].y & 0xFFFF0000u)
          };
          *reinterpret_cast<f4v*>(xf_base + j * H + h + 4) = (f4v){
            __uint_as_float((raws[j].z & 0xFFFFu) << 16), __uint_as_float(raws[j].z & 0xFFFF0000u),
            __uint_as_float((raws[j].w & 0xFFFFu) << 16), __uint_as_float(raws[j].w & 0xFFFF0000u)
          };
        }
      }
    }
  }
  if (ss_out) {
    const float ss = wave_sum(ss0 + ss1);
    if (lane == 0) s_ss[w] = ss;
    __syncthreads();
    if (tid == 0) {
      ss_out[t] = s_ss[0] + s_ss[1] + s_ss[2] + s_ss[3] + s_ss[4] + s_ss[5] + s_ss[6] + s_ss[7];
    }
  }
}

// ---------------------------------------------------------------------------
// Final head contraction: hc_head_contract [T, 4, 4096] -> [T, 4096]
// Uses rsqrt(mean(x^2) + rms_eps) (eps INSIDE sqrt, matching DeepSeek/vLLM-ERNIE)
// Grid: (T,), Block: 256
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void k_hc_head_contract_4096(
    const uint16_t* __restrict__ res,       // [T, 4, 4096] bf16
    const float* __restrict__ fn,           // [4, 16384] fp32
    const float* __restrict__ scale,        // [1] fp32
    const float* __restrict__ base,         // [4] fp32
    uint16_t* __restrict__ out,             // [T, 4096] bf16
    float rms_eps, float hc_eps
) {
  constexpr int H = 4096;
  constexpr int HC = 4;
  __shared__ float s_dot[4][8];
  __shared__ float s_ss[8];
  __shared__ float s_w[4];

  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int w = tid >> 5;
  const uint16_t* r_base = res + (size_t)t * (HC * H);

  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f, ss0 = 0.f, ss1 = 0.f;
  // 2 contiguous 8-bf16 (16B) passes with 16B NT loads on fn!
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    const int h = p * 2048 + tid * 8;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const u4v raw = *reinterpret_cast<const u4va*>(r_base + j * H + h);
      ss0 = bf16_dot2(raw.x, raw.x, ss0);
      ss1 = bf16_dot2(raw.y, raw.y, ss1);
      ss0 = bf16_dot2(raw.z, raw.z, ss0);
      ss1 = bf16_dot2(raw.w, raw.w, ss1);
      const f4v rv0 = {__uint_as_float((raw.x & 0xFFFFu) << 16), __uint_as_float(raw.x & 0xFFFF0000u),
                       __uint_as_float((raw.y & 0xFFFFu) << 16), __uint_as_float(raw.y & 0xFFFF0000u)};
      const f4v rv1 = {__uint_as_float((raw.z & 0xFFFFu) << 16), __uint_as_float(raw.z & 0xFFFF0000u),
                       __uint_as_float((raw.w & 0xFFFFu) << 16), __uint_as_float(raw.w & 0xFFFF0000u)};
      const size_t col0 = (size_t)j * H + h;
      const size_t col1 = col0 + 4;
      // hc_head_fn is only 256 KB -> keep cached in L2/Infinity Cache across T tokens (saves 512 MB on T=2048 prefill!)
      const f4v f0a = *reinterpret_cast<const f4v*>(fn + (size_t)0 * (HC * H) + col0);
      const f4v f0b = *reinterpret_cast<const f4v*>(fn + (size_t)0 * (HC * H) + col1);
      const f4v f1a = *reinterpret_cast<const f4v*>(fn + (size_t)1 * (HC * H) + col0);
      const f4v f1b = *reinterpret_cast<const f4v*>(fn + (size_t)1 * (HC * H) + col1);
      const f4v f2a = *reinterpret_cast<const f4v*>(fn + (size_t)2 * (HC * H) + col0);
      const f4v f2b = *reinterpret_cast<const f4v*>(fn + (size_t)2 * (HC * H) + col1);
      const f4v f3a = *reinterpret_cast<const f4v*>(fn + (size_t)3 * (HC * H) + col0);
      const f4v f3b = *reinterpret_cast<const f4v*>(fn + (size_t)3 * (HC * H) + col1);
      d0 = fmaf(f0a.x, rv0.x, fmaf(f0a.y, rv0.y, fmaf(f0a.z, rv0.z, fmaf(f0a.w, rv0.w, d0))));
      d0 = fmaf(f0b.x, rv1.x, fmaf(f0b.y, rv1.y, fmaf(f0b.z, rv1.z, fmaf(f0b.w, rv1.w, d0))));
      d1 = fmaf(f1a.x, rv0.x, fmaf(f1a.y, rv0.y, fmaf(f1a.z, rv0.z, fmaf(f1a.w, rv0.w, d1))));
      d1 = fmaf(f1b.x, rv1.x, fmaf(f1b.y, rv1.y, fmaf(f1b.z, rv1.z, fmaf(f1b.w, rv1.w, d1))));
      d2 = fmaf(f2a.x, rv0.x, fmaf(f2a.y, rv0.y, fmaf(f2a.z, rv0.z, fmaf(f2a.w, rv0.w, d2))));
      d2 = fmaf(f2b.x, rv1.x, fmaf(f2b.y, rv1.y, fmaf(f2b.z, rv1.z, fmaf(f2b.w, rv1.w, d2))));
      d3 = fmaf(f3a.x, rv0.x, fmaf(f3a.y, rv0.y, fmaf(f3a.z, rv0.z, fmaf(f3a.w, rv0.w, d3))));
      d3 = fmaf(f3b.x, rv1.x, fmaf(f3b.y, rv1.y, fmaf(f3b.z, rv1.z, fmaf(f3b.w, rv1.w, d3))));
    }
  }
  const float ss = wave_sum(ss0 + ss1);
  d0 = wave_sum(d0); d1 = wave_sum(d1); d2 = wave_sum(d2); d3 = wave_sum(d3);
  if (lane == 0) {
    s_dot[0][w] = d0; s_dot[1][w] = d1; s_dot[2][w] = d2; s_dot[3][w] = d3; s_ss[w] = ss;
  }
  __syncthreads();
  if (tid == 0) {
    float tot_ss = s_ss[0] + s_ss[1] + s_ss[2] + s_ss[3] + s_ss[4] + s_ss[5] + s_ss[6] + s_ss[7];
    float inv_rms = rsqrtf(tot_ss * (1.0f / 16384.0f) + rms_eps);
    float sc = scale[0];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float dj = s_dot[j][0] + s_dot[j][1] + s_dot[j][2] + s_dot[j][3]
               + s_dot[j][4] + s_dot[j][5] + s_dot[j][6] + s_dot[j][7];
      s_w[j] = fast_sigmoid(dj * inv_rms * sc + base[j]) + hc_eps;
    }
  }
  __syncthreads();

  const float w0 = s_w[0], w1 = s_w[1], w2 = s_w[2], w3 = s_w[3];
  uint16_t* o_base = out + (size_t)t * H;
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    const int h = p * 2048 + tid * 8;
    const u4v raw0 = *reinterpret_cast<const u4va*>(r_base + 0 * H + h);
    const u4v raw1 = *reinterpret_cast<const u4va*>(r_base + 1 * H + h);
    const u4v raw2 = *reinterpret_cast<const u4va*>(r_base + 2 * H + h);
    const u4v raw3 = *reinterpret_cast<const u4va*>(r_base + 3 * H + h);
    const uint32_t rw0[4] = {raw0.x, raw0.y, raw0.z, raw0.w};
    const uint32_t rw1[4] = {raw1.x, raw1.y, raw1.z, raw1.w};
    const uint32_t rw2[4] = {raw2.x, raw2.y, raw2.z, raw2.w};
    const uint32_t rw3[4] = {raw3.x, raw3.y, raw3.z, raw3.w};
    uint32_t out_w[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float s0 = w0 * __uint_as_float((rw0[k] & 0xFFFFu) << 16)
                     + w1 * __uint_as_float((rw1[k] & 0xFFFFu) << 16)
                     + w2 * __uint_as_float((rw2[k] & 0xFFFFu) << 16)
                     + w3 * __uint_as_float((rw3[k] & 0xFFFFu) << 16);
      const float s1 = w0 * __uint_as_float(rw0[k] & 0xFFFF0000u)
                     + w1 * __uint_as_float(rw1[k] & 0xFFFF0000u)
                     + w2 * __uint_as_float(rw2[k] & 0xFFFF0000u)
                     + w3 * __uint_as_float(rw3[k] & 0xFFFF0000u);
      out_w[k] = f32_to_bf16x2_rne(s0, s1);
    }
    *reinterpret_cast<u4va*>(o_base + h) = (u4v){out_w[0], out_w[1], out_w[2], out_w[3]};
  }
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------
template <int DO_POST>
static void dispatch_mhc_post_dot(
    int T, hipStream_t stream,
    const uint16_t* xp, const uint16_t* rp, const float* pmp, const float* cmp,
    const float* fnp, uint16_t* rop, float* partp) {
  if (T == 1) {
    hipLaunchKernelGGL(
        (k_mhc_post_dot_4096<DO_POST>), dim3(32, 1), dim3(256), 0, stream,
        xp, rp, pmp, cmp, fnp, rop, partp);
  } else if (T <= 4) {
    hipLaunchKernelGGL(
        (k_mhc_post_dot_4096_tiled<DO_POST, 4, 1>), dim3(32, 1), dim3(256), 0, stream,
        xp, rp, pmp, cmp, fnp, rop, partp, T);
  } else if (T <= 8) {
    hipLaunchKernelGGL(
        (k_mhc_post_dot_4096_tiled<DO_POST, 8, 1>), dim3(32, 1), dim3(256), 0, stream,
        xp, rp, pmp, cmp, fnp, rop, partp, T);
  } else if (T <= 16) {
    hipLaunchKernelGGL(
        (k_mhc_post_dot_4096_tiled<DO_POST, 16, 1>), dim3(32, 1), dim3(256), 0, stream,
        xp, rp, pmp, cmp, fnp, rop, partp, T);
  } else {
    // Prefill T >= 17: NT_FN=0 keeps fn (1.5 MiB) resident in 32 MiB Infinity Cache across all token-tiles!
    hipLaunchKernelGGL(
        (k_mhc_post_dot_4096_tiled<DO_POST, 16, 0, 1>), dim3(32, (T + 15) / 16), dim3(256), 0, stream,
        xp, rp, pmp, cmp, fnp, rop, partp, T);
  }
}

void launch_mhc_fused_post_pre_decode(
    torch::Tensor x, torch::Tensor res_in, torch::Tensor post_mix, torch::Tensor comb_mix,
    torch::Tensor fn, torch::Tensor hc_scale, torch::Tensor hc_base,
    c10::optional<torch::Tensor> norm_w,
    torch::Tensor res_out, torch::Tensor post_out, torch::Tensor comb_out,
    torch::Tensor layer_in, torch::Tensor part_buf,
    double rms_eps, double hc_pre_eps, double hc_sinkhorn_eps,
    double hc_post_mult_value, int64_t sinkhorn_repeat, double norm_eps, bool do_post
) {
  const int T = (int)res_in.size(0);
  if (T == 0) return;
  auto stream = c10::hip::getCurrentHIPStream().stream();
  // MALL-tiled CHUNK=512: with NT_IN=1 on res_in/x, res_out (16 MiB) + fn (1.5 MiB) + part (1.6 MiB)
  // stays 100% resident in 32 MiB Infinity Cache between Kernel 1 and Kernel 2 (1.28-1.36x faster!).
  const int part_cap = (int)part_buf.size(0);
  const int chunk = (T <= 512) ? T : (part_cap < 512 ? part_cap : 512);
  const uint16_t* xp_base = do_post ? reinterpret_cast<const uint16_t*>(x.data_ptr()) : nullptr;
  const uint16_t* rp_base = reinterpret_cast<const uint16_t*>(res_in.data_ptr());
  const float* pm_base = do_post ? post_mix.data_ptr<float>() : nullptr;
  const float* cm_base = do_post ? comb_mix.data_ptr<float>() : nullptr;
  const float* fn_p = fn.data_ptr<float>();
  const float* sc_p = hc_scale.data_ptr<float>();
  const float* bs_p = hc_base.data_ptr<float>();
  const uint16_t* nw_p = norm_w.has_value() ? reinterpret_cast<const uint16_t*>(norm_w->data_ptr()) : nullptr;
  uint16_t* ro_base = do_post ? reinterpret_cast<uint16_t*>(res_out.data_ptr()) : nullptr;
  float* po_base = post_out.data_ptr<float>();
  float* co_base = comb_out.data_ptr<float>();
  uint16_t* li_base = reinterpret_cast<uint16_t*>(layer_in.data_ptr());
  float* part_p = part_buf.data_ptr<float>();

  for (int t0 = 0; t0 < T; t0 += chunk) {
    const int tc = (T - t0 < chunk) ? (T - t0) : chunk;
    const uint16_t* rp = rp_base + (size_t)t0 * 16384;
    uint16_t* ro = do_post ? (ro_base + (size_t)t0 * 16384) : nullptr;
    if (do_post) {
      dispatch_mhc_post_dot<1>(
          tc, stream,
          xp_base + (size_t)t0 * 4096, rp,
          pm_base + (size_t)t0 * 4, cm_base + (size_t)t0 * 16,
          fn_p, ro, part_p);
    } else {
      dispatch_mhc_post_dot<0>(
          tc, stream, nullptr, rp, nullptr, nullptr, fn_p, nullptr, part_p);
    }
    const uint16_t* rcur = do_post ? ro : rp;
    if (nw_p) {
      hipLaunchKernelGGL(
          (k_mhc_sinkhorn_norm_4096<1, 1>), dim3(tc), dim3(256), 0, stream,
          part_p, nullptr, rcur, sc_p, bs_p, nw_p,
          po_base + (size_t)t0 * 4, co_base + (size_t)t0 * 16,
          li_base + (size_t)t0 * 4096,
          (float)rms_eps, (float)hc_pre_eps, (float)hc_sinkhorn_eps,
          (float)hc_post_mult_value, (int)sinkhorn_repeat, (float)norm_eps);
    } else {
      hipLaunchKernelGGL(
          (k_mhc_sinkhorn_norm_4096<1, 0>), dim3(tc), dim3(256), 0, stream,
          part_p, nullptr, rcur, sc_p, bs_p, nullptr,
          po_base + (size_t)t0 * 4, co_base + (size_t)t0 * 16,
          li_base + (size_t)t0 * 4096,
          (float)rms_eps, (float)hc_pre_eps, (float)hc_sinkhorn_eps,
          (float)hc_post_mult_value, (int)sinkhorn_repeat, (float)norm_eps);
    }
  }
}

void launch_mhc_post_sq(
    c10::optional<torch::Tensor> x, torch::Tensor res_in,
    c10::optional<torch::Tensor> post_mix, c10::optional<torch::Tensor> comb_mix,
    c10::optional<torch::Tensor> res_out,
    c10::optional<torch::Tensor> x_f32_out,
    c10::optional<torch::Tensor> ss_out, bool do_post
) {
  const int T = (int)res_in.size(0);
  if (T == 0) return;
  auto stream = c10::hip::getCurrentHIPStream().stream();
  if (do_post) {
    hipLaunchKernelGGL(
        (k_mhc_post_sq_4096<1>), dim3(T), dim3(256), 0, stream,
        reinterpret_cast<const uint16_t*>(x->data_ptr()),
        reinterpret_cast<const uint16_t*>(res_in.data_ptr()),
        post_mix->data_ptr<float>(),
        comb_mix->data_ptr<float>(),
        reinterpret_cast<uint16_t*>(res_out->data_ptr()),
        x_f32_out.has_value() ? x_f32_out->data_ptr<float>() : nullptr,
        ss_out.has_value() ? ss_out->data_ptr<float>() : nullptr);
  } else {
    hipLaunchKernelGGL(
        (k_mhc_post_sq_4096<0>), dim3(T), dim3(256), 0, stream,
        nullptr,
        reinterpret_cast<const uint16_t*>(res_in.data_ptr()),
        nullptr, nullptr, nullptr,
        x_f32_out.has_value() ? x_f32_out->data_ptr<float>() : nullptr,
        ss_out.has_value() ? ss_out->data_ptr<float>() : nullptr);
  }
}

void launch_mhc_sinkhorn_norm_from_dots(
    torch::Tensor dots, torch::Tensor ss, torch::Tensor res_cur,
    torch::Tensor hc_scale, torch::Tensor hc_base,
    c10::optional<torch::Tensor> norm_w,
    torch::Tensor post_out, torch::Tensor comb_out, torch::Tensor layer_in,
    double rms_eps, double hc_pre_eps, double hc_sinkhorn_eps,
    double hc_post_mult_value, int64_t sinkhorn_repeat, double norm_eps
) {
  const int T = (int)res_cur.size(0);
  if (T == 0) return;
  auto stream = c10::hip::getCurrentHIPStream().stream();
  if (norm_w.has_value()) {
    hipLaunchKernelGGL(
        (k_mhc_sinkhorn_norm_4096<0, 1>), dim3(T), dim3(256), 0, stream,
        dots.data_ptr<float>(), ss.data_ptr<float>(),
        reinterpret_cast<const uint16_t*>(res_cur.data_ptr()),
        hc_scale.data_ptr<float>(), hc_base.data_ptr<float>(),
        reinterpret_cast<const uint16_t*>(norm_w->data_ptr()),
        post_out.data_ptr<float>(), comb_out.data_ptr<float>(),
        reinterpret_cast<uint16_t*>(layer_in.data_ptr()),
        (float)rms_eps, (float)hc_pre_eps, (float)hc_sinkhorn_eps,
        (float)hc_post_mult_value, (int)sinkhorn_repeat, (float)norm_eps);
  } else {
    hipLaunchKernelGGL(
        (k_mhc_sinkhorn_norm_4096<0, 0>), dim3(T), dim3(256), 0, stream,
        dots.data_ptr<float>(), ss.data_ptr<float>(),
        reinterpret_cast<const uint16_t*>(res_cur.data_ptr()),
        hc_scale.data_ptr<float>(), hc_base.data_ptr<float>(), nullptr,
        post_out.data_ptr<float>(), comb_out.data_ptr<float>(),
        reinterpret_cast<uint16_t*>(layer_in.data_ptr()),
        (float)rms_eps, (float)hc_pre_eps, (float)hc_sinkhorn_eps,
        (float)hc_post_mult_value, (int)sinkhorn_repeat, (float)norm_eps);
  }
}

void launch_hc_head_contract(
    torch::Tensor res, torch::Tensor fn, torch::Tensor scale, torch::Tensor base,
    torch::Tensor out, double rms_eps, double hc_eps
) {
  const int T = (int)res.size(0);
  if (T == 0) return;
  auto stream = c10::hip::getCurrentHIPStream().stream();
  hipLaunchKernelGGL(
      k_hc_head_contract_4096, dim3(T), dim3(256), 0, stream,
      reinterpret_cast<const uint16_t*>(res.data_ptr()),
      fn.data_ptr<float>(), scale.data_ptr<float>(), base.data_ptr<float>(),
      reinterpret_cast<uint16_t*>(out.data_ptr()),
      (float)rms_eps, (float)hc_eps);
}


