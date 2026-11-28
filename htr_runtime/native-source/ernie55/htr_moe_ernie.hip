// Native gfx1151 (RDNA3.5) HTR-W1 packed routed-MoE decode GEMV + subset dequant for ERNIE-5.5-Flash.
// Incorporates Strix Halo RDNA3.5 ISA optimizations from vllm-htr K6:
//   - Wave32 permlane16 zero-redundant-load V[0] (saves 25% global loads at K=3, 33% at K=2)
//   - Non-temporal streaming loads (__builtin_nontemporal_load) to avoid Infinity Cache thrashing
//   - Tight LDS sizing (NBL template param for bps=2/4/8) for higher CU wave occupancy
//   - ROWX real-row dispatch (mr <= M) so sparse experts never compute padded rows
//   - EPW wave-per-row register FWHT-128 epilogue with 16-byte vector loads/stores (0 LDS barriers)
//   - CMB unrolled 10-slot fused combine across both half-WGs
//   - SPLIT-capable k_moe_plan2 so launch bucket M can be smaller than T
// ERNIE-5.5 latent MoE geometry: hidden = 2048, intermediate = 2048, top-k = 10, 512 experts.
// Activation: SiTU-GLU(gate, up; beta=4.0, linear_beta=25.0).
// Widths supported: K = 2 (DSpark draft), K = 3 (main experts), K = 4, K = 5.
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

#define GEN_A 0xDF6ADFu
#define GEN_B 0x34983u
typedef _Float16 h2v __attribute__((ext_vector_type(2)));
typedef _Float16 h4v __attribute__((ext_vector_type(4)));
typedef float f4v __attribute__((ext_vector_type(4)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));
typedef u4v u4va __attribute__((aligned(4)));

__device__ __forceinline__ uint32_t sad_bytesum(uint32_t x, uint32_t acc) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
  return __builtin_amdgcn_sad_u8(x, 0u, acc);
#else
  return (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24) + acc;
#endif
}
__device__ __forceinline__ uint32_t alignbit(uint32_t hi, uint32_t lo, uint32_t sh) {
  return __builtin_amdgcn_alignbit(hi, lo, sh);
}
template <int B>
__device__ __forceinline__ uint32_t ext16(const uint32_t* V) {
  constexpr int w = B >> 5, off = B & 31;
  if constexpr (off <= 16) return (V[w] >> (16 - off)) & 0xFFFFu;
  else return alignbit(V[w], V[w + 1], 48 - off) & 0xFFFFu;
}
__device__ __forceinline__ uint32_t val_f16bits(uint32_t s) {
  return sad_bytesum(s * GEN_A + GEN_B, 0x6400u);
}
__device__ __forceinline__ uint32_t val_f16bits2(uint32_t a, uint32_t b) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
  return __builtin_amdgcn_sad_hi_u8(b * GEN_A + GEN_B, 0u, __builtin_amdgcn_sad_u8(a * GEN_A + GEN_B, 0u, 0x64006400u));
#else
  return val_f16bits(a) | (val_f16bits(b) << 16);
#endif
}
__device__ __forceinline__ h2v as_h2(uint32_t u) { return __builtin_bit_cast(h2v, u); }
__device__ __forceinline__ void moe_lds_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

template <int MSK> __device__ __forceinline__ float shx(float v) {
  const int iv = __builtin_bit_cast(int, v);
  if constexpr (MSK < 16) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, iv, 0x160 | MSK, 0xF, 0xF, false));
  } else {
    return __builtin_bit_cast(float, (int)__builtin_amdgcn_permlanex16((unsigned)iv, (unsigned)iv, 0x76543210u, 0xfedcba98u, false, false));
  }
}
__device__ __forceinline__ float wave_max(float a) {
  a = fmaxf(a, shx<16>(a)); a = fmaxf(a, shx<8>(a)); a = fmaxf(a, shx<4>(a)); a = fmaxf(a, shx<2>(a)); a = fmaxf(a, shx<1>(a));
  return a;
}
__device__ __forceinline__ float wave_sum(float a) {
  a += shx<16>(a); a += shx<8>(a); a += shx<4>(a); a += shx<2>(a); a += shx<1>(a);
  return a;
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

// ---------------------------------------------------------------- ERNIE-5.5 layout constants
#define MOE_H 2048
#define MOE_I 2048
#define MOE_I_SB 16                 // MOE_I / 128
#define MOE_H_SB 16                 // MOE_H / 128
#define MOE_MAXB 16                 // max 128-blocks per WG
#define SC_STRIDE 6144              // scales per expert: w1 2048 | w3 2048 | w2 2048
#define SV_STRIDE 768               // sv bytes per expert: w1 256 | w3 256 | w2 256
#define SU_STRIDE 512               // su bytes per expert: w1/w3 (shared) 256 | w2 256

struct MoeArgs {
  const uint32_t* qw;   const int64_t* qoff;  const int* kexp;
  const __half* scales; const uint8_t* su;    const uint8_t* sv;
  const int* grp_e;     const int* grp_rows;  const int* grp_n;
  const __hip_bfloat16* x;                     // PH0 input [T, 2048]
  const float* gu;                             // PH1 input [A, M, 4096]
  float* part; unsigned* counters; float* out;
  int bps, ns; float cb;
  int spadk[6];
  int fence_ra;
  const int* slot; const float* wts; const int* ng; unsigned* cnt2; __hip_bfloat16* y; int T, topk;
};

template <int K, int R, int NBL>
__device__ __forceinline__ void tile_loop_k6(const uint32_t* __restrict__ qm, size_t pitch, int i, int b0, int nb, int lane,
                                             _Float16 (*xs_l)[NBL * 128], float (*bscale)[NBL], float* a0, float* a1) {
  const int g = lane >> 3, q = lane & 7;
  const uint32_t* qrow = qm + (size_t)i * pitch;
  const int ntiles = nb * 8;
  // Zero-VRAM cyclic K-tile stagger for K=3 (SPAD3=0): 8 tiles * 96B = 768B = 3*256B -> gcd(3,8)=1 bijection across all 8 LPDDR5X channels!
  const int phase = (K == 3 && ntiles >= 64) ? ((i & 7) * 8) :
                    (K == 3 && ntiles >= 32) ? ((i & 7) * 4) :
                    (K == 2 && ntiles >= 32) ? ((i & 7) * 4) : 0;
  // 2-stage register software prefetch of LPDDR5X NT weights (only 4 VGPRs for K=2,3,4; 5 for K=5):
  // xs_l lives in LDS (lgkmcnt only) -> ZERO vmcnt conflict, overlaps 100% of LPDDR5X latency with Pass 1 & Pass 2!
  u4v raw_cur = {0u, 0u, 0u, 0u};
  uint32_t w5_cur = 0u;
  if (g < ntiles) {
    const int jl0 = ((phase < ntiles) ? phase : 0) + g;
    const uint32_t* tp0 = qrow + (size_t)(b0 * 8 + jl0) * (8 * K);
    if constexpr (K >= 4) {
      raw_cur = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tp0 + q * K));
      if constexpr (K == 5) w5_cur = __builtin_nontemporal_load(tp0 + q * K + 4);
    } else if constexpr (K == 3) {
      if (q < 6) raw_cur = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tp0 + q * 4));
    } else if constexpr (K == 2) {
      if (q < 4) raw_cur = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tp0 + q * 4));
    }
  }

  for (int step = 0; step < ntiles; step += 4) {
    int jl_base = step + phase;
    if (jl_base >= ntiles) jl_base -= ntiles;
    const int jl = jl_base + g;
    const int bl = jl >> 3;
    uint32_t V[K + 1];
    // Unpack prefetched tile into V[1..K] (raw_cur is dead right after unpack!)
    if constexpr (K >= 4) {
      V[1] = raw_cur.x; V[2] = raw_cur.y; V[3] = raw_cur.z; V[4] = raw_cur.w;
      if constexpr (K == 5) V[5] = w5_cur;
      else if constexpr (K > 5) {
        const uint32_t* tp = qrow + (size_t)(b0 * 8 + jl) * (8 * K);
#pragma unroll
        for (int k = 4; k < K; ++k) V[k + 1] = __builtin_nontemporal_load(tp + q * K + k);
      }
    } else if constexpr (K == 3) {
      const int b = lane & ~7;
      const int s0 = (b + ((3 * q) >> 2)) << 2;
      const int s1 = (b + ((3 * q + 2) >> 2)) << 2;
      const uint32_t p0 = (uint32_t)__builtin_amdgcn_ds_bpermute(s0, (int)raw_cur.x);
      const uint32_t p1 = (uint32_t)__builtin_amdgcn_ds_bpermute(s0, (int)raw_cur.y);
      const uint32_t p2 = (uint32_t)__builtin_amdgcn_ds_bpermute(s0, (int)raw_cur.z);
      const uint32_t p3 = (uint32_t)__builtin_amdgcn_ds_bpermute(s0, (int)raw_cur.w);
      const uint32_t q0 = (uint32_t)__builtin_amdgcn_ds_bpermute(s1, (int)raw_cur.x);
      const uint32_t q1 = (uint32_t)__builtin_amdgcn_ds_bpermute(s1, (int)raw_cur.y);
      const int rem = (3 * q) & 3;
      V[1] = (rem == 0) ? p0 : (rem == 1) ? p1 : (rem == 2) ? p2 : p3;
      V[2] = (rem == 0) ? p1 : (rem == 1) ? p2 : (rem == 2) ? p3 : q0;
      V[3] = (rem == 0) ? p2 : (rem == 1) ? p3 : (rem == 2) ? q0 : q1;
    } else if constexpr (K == 2) {
      const int s0 = ((lane & ~7) + (q >> 1)) << 2;
      const uint32_t p0 = (uint32_t)__builtin_amdgcn_ds_bpermute(s0, (int)raw_cur.x);
      const uint32_t p1 = (uint32_t)__builtin_amdgcn_ds_bpermute(s0, (int)raw_cur.y);
      const uint32_t p2 = (uint32_t)__builtin_amdgcn_ds_bpermute(s0, (int)raw_cur.z);
      const uint32_t p3 = (uint32_t)__builtin_amdgcn_ds_bpermute(s0, (int)raw_cur.w);
      V[1] = (q & 1) ? p2 : p0;
      V[2] = (q & 1) ? p3 : p1;
    } else {
      const uint32_t* tp = qrow + (size_t)(b0 * 8 + jl) * (8 * K);
#pragma unroll
      for (int k = 0; k < K; ++k) V[k + 1] = __builtin_nontemporal_load(tp + q * K + k);
    }
    // Lane q's V[0] == lane ((q-1)&7)'s V[K] within each 8-lane group (g = 0..3)
    V[0] = __builtin_amdgcn_permlane16(V[K], V[K], 0x65432107u, 0xedcba98fu, false, false);

    // Issue next tile's LPDDR5X NT weight load BEFORE Pass 1 & Pass 2 (xs_l is in LDS -> 100% overlap!)
    u4v raw_nxt = {0u, 0u, 0u, 0u};
    uint32_t w5_nxt = 0u;
    if (step + 4 < ntiles) {
      int jn_base = step + 4 + phase;
      if (jn_base >= ntiles) jn_base -= ntiles;
      const uint32_t* tpn = qrow + (size_t)(b0 * 8 + jn_base + g) * (8 * K);
      if constexpr (K >= 4) {
        raw_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tpn + q * K));
        if constexpr (K == 5) w5_nxt = __builtin_nontemporal_load(tpn + q * K + 4);
      } else if constexpr (K == 3) {
        if (q < 6) raw_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tpn + q * 4));
      } else if constexpr (K == 2) {
        if (q < 4) raw_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tpn + q * 4));
      }
    }
    __builtin_amdgcn_sched_barrier(0);

    // Half-tile xv[R][4] + bank-staggered (U_lo, U_hi) pairing:
    //   1) Halves xv VGPR pressure (-32 VGPRs at R=8 -> <=96 VGPRs = FULL 16 waves/SIMD!)
    //   2) Staggers xv[r][k] vs xv[r][k^1] across different VGPR banks for V_DUAL_DOT2ACC_F32_F16!
    h2v xv[R][4];
    float d0[R], d1[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const uint4 u0 = reinterpret_cast<const uint4*>(&xs_l[r][jl * 16])[0];
      xv[r][0] = as_h2(u0.x); xv[r][1] = as_h2(u0.y); xv[r][2] = as_h2(u0.z); xv[r][3] = as_h2(u0.w);
      d0[r] = 0.f; d1[r] = 0.f;
    }
#define HTR_DUALM(U0, U1)                                                                      \
    {                                                                                          \
      const h2v w0 = as_h2(val_f16bits2(ext16<16 + ((U0) + 1) * K>(V), ext16<16 + ((U0) + 2) * K>(V))); \
      const h2v w1 = as_h2(val_f16bits2(ext16<16 + ((U1) + 1) * K>(V), ext16<16 + ((U1) + 2) * K>(V))); \
      _Pragma("unroll") for (int r = 0; r < R; ++r) {                                          \
        d0[r] = __builtin_amdgcn_fdot2(w0, xv[r][((U0) & 7) >> 1], d0[r], false);              \
        d1[r] = __builtin_amdgcn_fdot2(w1, xv[r][((U1) & 7) >> 1], d1[r], false);              \
      }                                                                                        \
    }
    HTR_DUALM(0, 18) HTR_DUALM(2, 16) HTR_DUALM(4, 22) HTR_DUALM(6, 20)
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const uint4 u1 = reinterpret_cast<const uint4*>(&xs_l[r][jl * 16])[1];
      xv[r][0] = as_h2(u1.x); xv[r][1] = as_h2(u1.y); xv[r][2] = as_h2(u1.z); xv[r][3] = as_h2(u1.w);
    }
    HTR_DUALM(8, 26) HTR_DUALM(10, 24) HTR_DUALM(12, 30) HTR_DUALM(14, 28)
#undef HTR_DUALM
#pragma unroll
    for (int r = 0; r < R; ++r) { const float s = bscale[r][bl]; a0[r] += s * d0[r]; a1[r] += s * d1[r]; }
    raw_cur = raw_nxt;
    w5_cur = w5_nxt;
  }
}

// Branchless RDNA 3.5 hardware fast_tanhf (v_exp_f32 + v_rcp_f32, max abs err < 1e-7, 6 ops vs ~45 for libm tanhf!)
__device__ __forceinline__ float fast_tanhf(float x) {
  const float ax = fabsf(x);
  const float e = exp2f((-2.0f * 1.4426950408889634f) * ax);
  const float t = (1.0f - e) * __builtin_amdgcn_rcpf(1.0f + e);
  return copysignf(t, x);
}
// ERNIE-5.5 SiTU-GLU: beta * tanh(g / beta) * sigmoid(g) * (linear_beta * tanh(u / linear_beta))
__device__ __forceinline__ float situ_glu_f(float g, float u) {
  float sg = 4.0f * fast_tanhf(g * 0.25f) * __builtin_amdgcn_rcpf(1.0f + exp2f(-1.4426950408889634f * g));
  float su = 25.0f * fast_tanhf(u * 0.04f);
  return sg * su;
}

template <int KF, int M, int PH, int FUSE, int NBL, int NSC>
__global__ __launch_bounds__(256) void k_e82_moe_ernie(MoeArgs a) {
  constexpr int m = PH == 0 ? 2 * MOE_I : MOE_H;
  constexpr int n = PH == 0 ? MOE_H : MOE_I;
  constexpr int nt = n >> 4, MPB = m >> 7;
  __shared__ __attribute__((aligned(16))) _Float16 xs_l[M][NBL * 128];
  __shared__ float bscale[M][NBL];
  __shared__ float bsum[M][NBL];
  __shared__ unsigned is_last;
  __shared__ unsigned is_last2;
  const int gidx = blockIdx.z, sb = blockIdx.x, sp = blockIdx.y;
  const int ns = NSC ? NSC : a.ns;
  const int e = a.grp_e[gidx];
  if (e < 0) return;
  const int K = a.kexp[e];
  if constexpr (KF != 0) { if (K != KF) return; }
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int mr = a.grp_n[gidx];
  int sbl, mat;
  if constexpr (PH == 0) { mat = sb < MOE_I_SB ? 0 : 1; sbl = sb < MOE_I_SB ? sb : sb - MOE_I_SB; }
  else { mat = 2; sbl = sb; }
  const uint32_t* qm = a.qw + a.qoff[(size_t)e * 3 + mat];
  const __half* scl = a.scales + (size_t)e * SC_STRIDE + (mat == 0 ? 0 : mat == 1 ? MOE_I : 2 * MOE_I);
  const uint8_t* svp = a.sv + (size_t)e * SV_STRIDE + (mat == 0 ? 0 : mat == 1 ? (MOE_I / 8) : (2 * MOE_I / 8));
  const uint8_t* sup = a.su + (size_t)e * SU_STRIDE + (PH == 0 ? 0 : (MOE_H / 8));
  const int b0 = sp * a.bps, b1 = min(n >> 7, b0 + a.bps), nb = b1 - b0;
  // ROWX prologue: only transform real rows mr <= M
  for (int it = w; it < mr * nb; it += 8) {
    const int r = it / nb, bl = it % nb, base = (b0 + bl) * 128 + lane * 4;
    float v[4];
    if constexpr (PH == 0) {
      const __hip_bfloat16* xr = a.x + (size_t)a.grp_rows[gidx * M + r] * MOE_H;
      const uint2 raw = *reinterpret_cast<const uint2*>(xr + base);
      v[0] = __uint_as_float((raw.x & 0xFFFFu) << 16);
      v[1] = __uint_as_float(raw.x & 0xFFFF0000u);
      v[2] = __uint_as_float((raw.y & 0xFFFFu) << 16);
      v[3] = __uint_as_float(raw.y & 0xFFFF0000u);
    } else {
      const float* gr = a.gu + ((size_t)gidx * M + r) * (2 * MOE_I);
      const f4v g4 = *reinterpret_cast<const f4v*>(gr + base);
      const f4v u4 = *reinterpret_cast<const f4v*>(gr + MOE_I + base);
      v[0] = situ_glu_f(g4.x, u4.x);
      v[1] = situ_glu_f(g4.y, u4.y);
      v[2] = situ_glu_f(g4.z, u4.z);
      v[3] = situ_glu_f(g4.w, u4.w);
    }
    const unsigned sg4 = (unsigned)sup[base >> 3] >> (base & 7);
    if (sg4 & 1u) v[0] = -v[0];
    if (sg4 & 2u) v[1] = -v[1];
    if (sg4 & 4u) v[2] = -v[2];
    if (sg4 & 8u) v[3] = -v[3];
    fwht128_wave(v, lane);
    float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    am = wave_max(am);
    am *= (1.0f / 11.313708498984761f);
    const float sc = am > 30000.f ? exp2f(ceilf(log2f(am * (1.0f / 30000.f)))) : 1.f;
    const float inv = (1.0f / 11.313708498984761f) * __builtin_amdgcn_rcpf(sc);
    const h4v h4 = {(_Float16)(v[0] * inv), (_Float16)(v[1] * inv),
                    (_Float16)(v[2] * inv), (_Float16)(v[3] * inv)};
    *reinterpret_cast<h4v*>(&xs_l[r][bl * 128 + lane * 4]) = h4;
    float sum = (float)h4.x + (float)h4.y + (float)h4.z + (float)h4.w;
    sum = wave_sum(sum);
    if (lane == 0) { bscale[r][bl] = sc; bsum[r][bl] = sum; }
  }
  moe_lds_barrier();
  const int il = sbl * 8 + w;
  float a0[M], a1[M];
#pragma unroll
  for (int r = 0; r < M; ++r) { a0[r] = 0.f; a1[r] = 0.f; }

  // ROWX dispatch on exact active row count mr
  if constexpr (KF != 0) {
    const size_t pitch = (size_t)nt * 8 * KF + a.spadk[KF];
    switch (mr) {
      case 1: if constexpr (M >= 1) tile_loop_k6<KF, 1, NBL>(qm, pitch, il, b0, nb, lane, xs_l, bscale, a0, a1); break;
      case 2: if constexpr (M >= 2) tile_loop_k6<KF, 2, NBL>(qm, pitch, il, b0, nb, lane, xs_l, bscale, a0, a1); break;
      case 3: if constexpr (M >= 3) tile_loop_k6<KF, 3, NBL>(qm, pitch, il, b0, nb, lane, xs_l, bscale, a0, a1); break;
      case 4: if constexpr (M >= 4) tile_loop_k6<KF, 4, NBL>(qm, pitch, il, b0, nb, lane, xs_l, bscale, a0, a1); break;
      case 5: if constexpr (M >= 5) tile_loop_k6<KF, 5, NBL>(qm, pitch, il, b0, nb, lane, xs_l, bscale, a0, a1); break;
      case 6: if constexpr (M >= 6) tile_loop_k6<KF, 6, NBL>(qm, pitch, il, b0, nb, lane, xs_l, bscale, a0, a1); break;
      case 7: if constexpr (M >= 7) tile_loop_k6<KF, 7, NBL>(qm, pitch, il, b0, nb, lane, xs_l, bscale, a0, a1); break;
      case 8: if constexpr (M >= 8) tile_loop_k6<KF, 8, NBL>(qm, pitch, il, b0, nb, lane, xs_l, bscale, a0, a1); break;
      default: break;
    }
  } else {
    switch (K) {
      case 2: tile_loop_k6<2, M, NBL>(qm, (size_t)nt * 16 + a.spadk[2], il, b0, nb, lane, xs_l, bscale, a0, a1); break;
      case 3: tile_loop_k6<3, M, NBL>(qm, (size_t)nt * 24 + a.spadk[3], il, b0, nb, lane, xs_l, bscale, a0, a1); break;
      case 4: tile_loop_k6<4, M, NBL>(qm, (size_t)nt * 32 + a.spadk[4], il, b0, nb, lane, xs_l, bscale, a0, a1); break;
      case 5: tile_loop_k6<5, M, NBL>(qm, (size_t)nt * 40 + a.spadk[5], il, b0, nb, lane, xs_l, bscale, a0, a1); break;
      default: __builtin_trap();
    }
  }

#pragma unroll
  for (int r = 0; r < M; ++r) {
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
  }
  const int q = lane & 7;
  if ((lane >> 3) == 0) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
      if (r >= mr) continue;
      float off = 0.f;
      for (int bl = 0; bl < nb; ++bl) off += bscale[r][bl] * bsum[r][bl];
      off *= 1534.0f;
      float* pp = a.part + (((size_t)gidx * ns + sp) * M + r) * m + (sb * 8 + w) * 16 + 2 * q;
      *reinterpret_cast<float2*>(pp) = make_float2(a0[r] - off, a1[r] - off);
    }
  }
  if (a.fence_ra) __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent"); else __threadfence();
  moe_lds_barrier();
  unsigned* cnt = a.counters + (size_t)gidx * MPB + sb;
  if (threadIdx.x == 0) is_last = (atomicAdd(cnt, 1u) == (unsigned)(ns - 1));
  moe_lds_barrier();
  if (!is_last) return;
  if (a.fence_ra) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent"); else __threadfence();

  // EPW: Wave32 register FWHT-128 epilogue (wave w owns row r = w, zero LDS barriers, 16-byte NT loads/stores)
  const int t = threadIdx.x;
  if (w < mr) {
    const int r = w;
    f4v s4 = {0.f, 0.f, 0.f, 0.f};
    if constexpr (NSC != 0) {
      f4v pv[NSC ? NSC : 1];
#pragma unroll
      for (int k = 0; k < NSC; ++k) {
        pv[k] = *reinterpret_cast<const f4v*>(
            a.part + (((size_t)gidx * NSC + k) * M + r) * m + sb * 128 + lane * 4);
      }
#pragma unroll
      for (int k = 0; k < NSC; ++k) s4 += pv[k];
    } else {
      for (int k = 0; k < ns; ++k) {
        s4 += *reinterpret_cast<const f4v*>(
            a.part + (((size_t)gidx * ns + k) * M + r) * m + sb * 128 + lane * 4);
      }
    }
    const h4v hs = *reinterpret_cast<const h4v*>(scl + sbl * 128 + lane * 4);
    float v[4] = {
      s4.x * (float)hs.x * a.cb,
      s4.y * (float)hs.y * a.cb,
      s4.z * (float)hs.z * a.cb,
      s4.w * (float)hs.w * a.cb
    };
    fwht128_wave(v, lane);
    const unsigned sg4 = (unsigned)svp[sbl * 16 + (lane >> 1)] >> ((lane & 1) * 4);
    f4v o4;
    o4.x = v[0] * (1.0f / 11.313708498984761f);
    o4.y = v[1] * (1.0f / 11.313708498984761f);
    o4.z = v[2] * (1.0f / 11.313708498984761f);
    o4.w = v[3] * (1.0f / 11.313708498984761f);
    if (sg4 & 1) o4.x = -o4.x;
    if (sg4 & 2) o4.y = -o4.y;
    if (sg4 & 4) o4.z = -o4.z;
    if (sg4 & 8) o4.w = -o4.w;
    *reinterpret_cast<f4v*>(a.out + ((size_t)gidx * M + r) * m + sb * 128 + lane * 4) = o4;
  }
  if (t == 0) *cnt = 0u;

  if constexpr (FUSE != 0 && PH == 1) {
    if (a.fence_ra) __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent"); else __threadfence();
    moe_lds_barrier();
    if (t == 0) is_last2 = (atomicAdd(a.cnt2 + sb, 1u) == (unsigned)(a.ng[0] - 1));
    moe_lds_barrier();
    if (!is_last2) return;
    if (a.fence_ra) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent"); else __threadfence();
    // CMB: unrolled 10-slot fused combine across both 128-thread halves of the WG
    if (a.topk == 10) {
      const int i = sb * 128 + (t & 127);
      for (int tk = t >> 7; tk < a.T; tk += 2) {
        int sl[10]; float wv[10], ov[10];
#pragma unroll
        for (int k = 0; k < 10; ++k) { sl[k] = a.slot[tk * 10 + k]; wv[k] = a.wts[tk * 10 + k]; }
#pragma unroll
        for (int k = 0; k < 10; ++k) ov[k] = a.out[(size_t)(sl[k] < 0 ? 0 : sl[k]) * MOE_H + i];
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < 10; ++k) { const float s2 = s + wv[k] * ov[k]; s = sl[k] >= 0 ? s2 : s; }
        a.y[(size_t)tk * MOE_H + i] = __float2bfloat16(s);
      }
    } else if (t < 128) {
      const int i = sb * 128 + t;
      for (int tk = 0; tk < a.T; ++tk) {
        float s = 0.f;
        for (int k = 0; k < a.topk; ++k) {
          const int sl = a.slot[tk * a.topk + k];
          if (sl >= 0) s += a.wts[tk * a.topk + k] * a.out[(size_t)sl * MOE_H + i];
        }
        a.y[(size_t)tk * MOE_H + i] = __float2bfloat16(s);
      }
    }
    if (t == 0) a.cnt2[sb] = 0u;
  }
}

__global__ void k_moe_plan(const int* __restrict__ ids, int A, int topk, int M, int* __restrict__ grp_e, int* __restrict__ grp_rows,
                           int* __restrict__ grp_n, int* __restrict__ slot) {
  const int a = threadIdx.x;
  if (a >= A) return;
  const int e = ids[a];
  int rank = 0;
  if (e >= 0)
    for (int b = 0; b < a; ++b) rank += (ids[b] == e);
  if (e < 0) { grp_e[a] = -1; grp_n[a] = 0; slot[a] = -1; return; }
  if (rank % M != 0) { grp_e[a] = -1; grp_n[a] = 0; return; }
  int c = 0;
  for (int b = a; b < A && c < M; ++b)
    if (ids[b] == e) { grp_rows[a * M + c] = b / topk; slot[b] = a * M + c; ++c; }
  grp_e[a] = e; grp_n[a] = c;
  for (int c2 = c; c2 < M; ++c2) grp_rows[a * M + c2] = 0;
}

__global__ void k_moe_plan2(const int* __restrict__ ids, int A, int topk, int M, int* __restrict__ grp_e, int* __restrict__ grp_rows,
                            int* __restrict__ grp_n, int* __restrict__ slot, int* __restrict__ ng) {
  __shared__ int s_ng;
  const int a = threadIdx.x;
  if (a == 0) s_ng = 0;
  if (a < A) { grp_e[a] = -1; grp_n[a] = 0; }
  moe_lds_barrier();
  if (a < A) {
    const int e = ids[a];
    int rank = 0;
    if (e >= 0)
      for (int b = 0; b < a; ++b) rank += (ids[b] == e);
    if (e < 0) { slot[a] = -1; }
    else if ((rank & (M - 1)) == 0) {
      const int g = atomicAdd(&s_ng, 1);
      int c = 0;
      for (int b = a; b < A && c < M; ++b)
        if (ids[b] == e) { grp_rows[g * M + c] = b / topk; slot[b] = g * M + c; ++c; }
      grp_e[g] = e; grp_n[g] = c;
      for (int c2 = c; c2 < M; ++c2) grp_rows[g * M + c2] = 0;
    }
  }
  moe_lds_barrier();
  if (a == 0) ng[0] = s_ng;
}

__global__ __launch_bounds__(256) void k_moe_combine(const float* __restrict__ dn, const int* __restrict__ slot,
                                                     const float* __restrict__ wts, int topk, __hip_bfloat16* __restrict__ y) {
  const int tk = blockIdx.x;
  for (int i = threadIdx.x; i < MOE_H; i += 256) {
    float s = 0.f;
    for (int k = 0; k < topk; ++k) {
      const int sl = slot[tk * topk + k];
      if (sl >= 0) s += wts[tk * topk + k] * dn[(size_t)sl * MOE_H + i];
    }
    y[(size_t)tk * MOE_H + i] = __float2bfloat16(s);
  }
}

// ---------------------------------------------------------------- selective subset dequant (prefill)
#define SWZ(r, c) (((r) << 7) | ((c) ^ (((r) & 7) << 4)))
template <int K>
__global__ __launch_bounds__(256) void k_dq_arena_subset(
    const uint32_t* __restrict__ qw, const __half* __restrict__ scales, const float* __restrict__ cbtab,
    const uint8_t* __restrict__ su, const uint8_t* __restrict__ sv, __hip_bfloat16* __restrict__ W,
    const int* __restrict__ active_ids, int mt, int nt, const int64_t* __restrict__ qoff, int role, int pad,
    int s_off, int s_es, int su_off, int su_es, int sv_off, int sv_es, size_t w_es) {
  constexpr int NW = 8 * K, NB = 256 * K;
  __shared__ float L[128 * 128];
  const int bx = blockIdx.x, by = blockIdx.y, out_b = blockIdx.z;
  const int b = active_ids[out_b];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int n = nt * 16;
  qw += qoff[(size_t)b * 3 + role];
  scales += (size_t)b * s_es + s_off;
  su += (size_t)b * su_es + su_off;
  sv += (size_t)b * sv_es + sv_off;
  W += (size_t)out_b * w_es;
  const size_t pitch = (size_t)nt * NW + pad;
  const float inv = 1.0f / 11.313708498984761f;
  for (int c = w; c < 128; c += 8) {
    const int tj = bx * 8 + (c >> 4), cc = c & 15;
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int r = lane * 4 + e, ti = by * 8 + (r >> 4), p = (r & 15) * 16 + cc;
      const uint32_t* tp = qw + (size_t)ti * pitch + (size_t)tj * NW;
      const int b0 = ((p + 1) * K - 16 + NB) % NB, wi = b0 >> 5, off = b0 & 31;
      const uint64_t comb = ((uint64_t)__builtin_nontemporal_load(tp + wi) << 32) | (uint64_t)__builtin_nontemporal_load(tp + (wi + 1) % NW);
      const uint32_t s = (uint32_t)(comb >> (48 - off)) & 0xFFFFu;
      v[e] = cbtab[s] * __half2float(scales[by * 128 + r]);
    }
    fwht128_wave(v, lane);
#pragma unroll
    for (int e = 0; e < 4; ++e) { const int r = lane * 4 + e; L[SWZ(r, c)] = v[e] * inv; }
  }
  __syncthreads();
  for (int r = w; r < 128; r += 8) {
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = L[SWZ(r, lane * 4 + e)];
    fwht128_wave(v, lane);
    const int gr = by * 128 + r;
    const bool nr = (sv[gr >> 3] >> (gr & 7)) & 1;
    __hip_bfloat16 o[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int gc = bx * 128 + lane * 4 + e;
      float x = v[e] * inv;
      if (nr) x = -x;
      if ((su[gc >> 3] >> (gc & 7)) & 1) x = -x;
      o[e] = __float2bfloat16(x);
    }
    *reinterpret_cast<uint2*>(W + (size_t)gr * n + bx * 128 + lane * 4) = *reinterpret_cast<uint2*>(o);
  }
}

// ---------------------------------------------------------------- host entrypoints
template <int KF, int M, int PH, int FUSE>
static void launch_moe_cfg(dim3 grid, hipStream_t st, MoeArgs a) {
  if (a.bps <= 2 && a.ns == 8) {
    hipLaunchKernelGGL((k_e82_moe_ernie<KF, M, PH, FUSE, 2, 8>), grid, dim3(256), 0, st, a);
  } else if (a.bps <= 4 && a.ns == 4) {
    hipLaunchKernelGGL((k_e82_moe_ernie<KF, M, PH, FUSE, 4, 4>), grid, dim3(256), 0, st, a);
  } else if (a.bps <= 8 && a.ns == 2) {
    hipLaunchKernelGGL((k_e82_moe_ernie<KF, M, PH, FUSE, 8, 2>), grid, dim3(256), 0, st, a);
  } else {
    hipLaunchKernelGGL((k_e82_moe_ernie<KF, M, PH, FUSE, MOE_MAXB, 0>), grid, dim3(256), 0, st, a);
  }
}
template <int KF, int M, int PH>
static void launch_moe(dim3 grid, hipStream_t st, MoeArgs a) {
  if constexpr (PH == 1) {
    if (a.y != nullptr) { launch_moe_cfg<KF, M, PH, 1>(grid, st, a); return; }
  }
  launch_moe_cfg<KF, M, PH, 0>(grid, st, a);
}
template <int KF, int PH>
static void moe_dispatch_m(int M, dim3 grid, hipStream_t st, MoeArgs a) {
  switch (M) {
    case 1: launch_moe<KF, 1, PH>(grid, st, a); break;
    case 2: launch_moe<KF, 2, PH>(grid, st, a); break;
    case 4: launch_moe<KF, 4, PH>(grid, st, a); break;
    case 8: launch_moe<KF, 8, PH>(grid, st, a); break;
    default: TORCH_CHECK(false, "M bucket must be 1/2/4/8");
  }
}
template <int PH>
static void moe_dispatch(int kf, int M, dim3 grid, hipStream_t st, MoeArgs a) {
  switch (kf) {
    case 0: moe_dispatch_m<0, PH>(M, grid, st, a); break;
    case 2: moe_dispatch_m<2, PH>(M, grid, st, a); break;
    case 3: moe_dispatch_m<3, PH>(M, grid, st, a); break;
    case 4:
    case 5: moe_dispatch_m<0, PH>(M, grid, st, a); break;
    default: TORCH_CHECK(false, "kf must be 0/2/3/4/5");
  }
}

std::vector<torch::Tensor> moe_plan(torch::Tensor ids, int64_t M) {
  TORCH_CHECK(ids.dim() == 2 && ids.scalar_type() == torch::kInt32 && ids.is_contiguous() && ids.is_cuda());
  const int T = ids.size(0), topk = ids.size(1), A = T * topk;
  TORCH_CHECK(A <= 256 && (M == 1 || M == 2 || M == 4 || M == 8));
  auto o = ids.options();
  auto grp_e = torch::empty({A}, o), grp_rows = torch::empty({A, M}, o), grp_n = torch::empty({A}, o), slot = torch::empty({A}, o);
  hipLaunchKernelGGL(k_moe_plan, dim3(1), dim3(256), 0, at::hip::getCurrentHIPStream(),
                     ids.data_ptr<int>(), A, topk, (int)M, grp_e.data_ptr<int>(),
                     grp_rows.data_ptr<int>(), grp_n.data_ptr<int>(), slot.data_ptr<int>());
  return {grp_e, grp_rows, grp_n, slot};
}

std::vector<torch::Tensor> moe_plan2(torch::Tensor ids, int64_t M) {
  TORCH_CHECK(ids.dim() == 2 && ids.scalar_type() == torch::kInt32 && ids.is_contiguous() && ids.is_cuda());
  const int T = ids.size(0), topk = ids.size(1), A = T * topk;
  TORCH_CHECK(A <= 256 && (M == 1 || M == 2 || M == 4 || M == 8));
  auto o = ids.options();
  auto grp_e = torch::empty({A}, o), grp_rows = torch::empty({A, M}, o), grp_n = torch::empty({A}, o), slot = torch::empty({A}, o);
  auto ng = torch::empty({1}, o);
  hipLaunchKernelGGL(k_moe_plan2, dim3(1), dim3(256), 0, at::hip::getCurrentHIPStream(),
                     ids.data_ptr<int>(), A, topk, (int)M, grp_e.data_ptr<int>(),
                     grp_rows.data_ptr<int>(), grp_n.data_ptr<int>(), slot.data_ptr<int>(), ng.data_ptr<int>());
  return {grp_e, grp_rows, grp_n, slot, ng};
}

void moe_plan2_into(torch::Tensor ids, int64_t M,
                    torch::Tensor grp_e, torch::Tensor grp_rows, torch::Tensor grp_n,
                    torch::Tensor slot, torch::Tensor ng) {
  TORCH_CHECK(ids.dim() == 2 && ids.scalar_type() == torch::kInt32 && ids.is_contiguous() && ids.is_cuda());
  const int T = ids.size(0), topk = ids.size(1), A = T * topk;
  TORCH_CHECK(A <= 256 && (M == 1 || M == 2 || M == 4 || M == 8));
  TORCH_CHECK(grp_e.numel() >= A && grp_rows.numel() >= A * M && grp_n.numel() >= A && slot.numel() >= A && ng.numel() >= 1);
  hipLaunchKernelGGL(k_moe_plan2, dim3(1), dim3(256), 0, at::hip::getCurrentHIPStream(),
                     ids.data_ptr<int>(), A, topk, (int)M, grp_e.data_ptr<int>(),
                     grp_rows.data_ptr<int>(), grp_n.data_ptr<int>(), slot.data_ptr<int>(), ng.data_ptr<int>());
}

void moe_gemv(int64_t phase, int64_t kf, int64_t M, torch::Tensor qw, torch::Tensor qoff, torch::Tensor kexp,
              torch::Tensor scales, torch::Tensor su, torch::Tensor sv, torch::Tensor grp_e, torch::Tensor grp_rows,
              torch::Tensor grp_n, torch::Tensor xin, torch::Tensor part, torch::Tensor counters, torch::Tensor out,
              int64_t bps, double cb, int64_t spad2, int64_t spad3, int64_t spad4, int64_t spad5, int64_t fence_ra) {
  const int A = grp_e.size(0);
  const int m = phase == 0 ? 2 * MOE_I : MOE_H, n = phase == 0 ? MOE_H : MOE_I, nbt = n / 128;
  TORCH_CHECK(bps >= 1 && bps <= MOE_MAXB);
  const int ns = (nbt + bps - 1) / bps;
  TORCH_CHECK(part.numel() >= (int64_t)A * ns * M * m && out.numel() >= (int64_t)A * M * m && counters.numel() >= A * (m / 128));
  MoeArgs a;
  a.qw = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>()); a.qoff = qoff.data_ptr<int64_t>(); a.kexp = kexp.data_ptr<int>();
  a.scales = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()); a.su = su.data_ptr<uint8_t>(); a.sv = sv.data_ptr<uint8_t>();
  a.grp_e = grp_e.data_ptr<int>(); a.grp_rows = grp_rows.data_ptr<int>(); a.grp_n = grp_n.data_ptr<int>();
  a.x = phase == 0 ? reinterpret_cast<const __hip_bfloat16*>(xin.data_ptr<at::BFloat16>()) : nullptr;
  a.gu = phase == 1 ? xin.data_ptr<float>() : nullptr;
  a.part = part.data_ptr<float>(); a.counters = reinterpret_cast<unsigned*>(counters.data_ptr<int32_t>()); a.out = out.data_ptr<float>();
  a.bps = (int)bps; a.ns = ns; a.cb = (float)cb;
  for (int k = 0; k < 6; ++k) a.spadk[k] = 0;
  a.spadk[2] = (int)spad2; a.spadk[3] = (int)spad3; a.spadk[4] = (int)spad4; a.spadk[5] = (int)spad5;
  a.fence_ra = fence_ra ? 1 : 0;
  a.slot = nullptr; a.wts = nullptr; a.ng = nullptr; a.cnt2 = nullptr; a.y = nullptr; a.T = 0; a.topk = 0;
  dim3 grid(m / 128, ns, A);
  auto st = at::hip::getCurrentHIPStream();
  if (phase == 0) moe_dispatch<0>((int)kf, (int)M, grid, st, a); else moe_dispatch<1>((int)kf, (int)M, grid, st, a);
}

torch::Tensor moe_combine(torch::Tensor dn, torch::Tensor slot, torch::Tensor wts) {
  TORCH_CHECK(wts.dim() == 2 && wts.scalar_type() == torch::kFloat32 && wts.is_contiguous());
  const int T = wts.size(0), topk = wts.size(1);
  auto y = torch::empty({T, MOE_H}, dn.options().dtype(torch::kBFloat16));
  hipLaunchKernelGGL(k_moe_combine, dim3(T), dim3(256), 0, at::hip::getCurrentHIPStream(),
                     dn.data_ptr<float>(), slot.data_ptr<int>(), wts.data_ptr<float>(), topk,
                     reinterpret_cast<__hip_bfloat16*>(y.data_ptr<at::BFloat16>()));
  return y;
}

torch::Tensor moe_gemv_dn_fused(int64_t kf, int64_t M, torch::Tensor qw, torch::Tensor qoff, torch::Tensor kexp,
                                torch::Tensor scales, torch::Tensor su, torch::Tensor sv, torch::Tensor grp_e,
                                torch::Tensor grp_rows, torch::Tensor grp_n, torch::Tensor gu, torch::Tensor part,
                                torch::Tensor counters, torch::Tensor dn, int64_t bps, double cb,
                                int64_t spad2, int64_t spad3, int64_t spad4, int64_t spad5, int64_t fence_ra,
                                torch::Tensor slot, torch::Tensor wts, torch::Tensor ng, torch::Tensor cnt2) {
  const int A = grp_e.size(0);
  const int m = MOE_H, n = MOE_I, nbt = n / 128;
  TORCH_CHECK(bps >= 1 && bps <= MOE_MAXB);
  const int ns = (nbt + bps - 1) / bps;
  TORCH_CHECK(part.numel() >= (int64_t)A * ns * M * m && dn.numel() >= (int64_t)A * M * m && counters.numel() >= A * (m / 128));
  TORCH_CHECK(wts.dim() == 2 && wts.scalar_type() == torch::kFloat32 && wts.is_contiguous() && slot.numel() == wts.numel());
  TORCH_CHECK(cnt2.numel() >= m / 128 && cnt2.scalar_type() == torch::kInt32 && ng.numel() == 1);
  const int T = wts.size(0), topk = wts.size(1);
  auto y = torch::empty({T, MOE_H}, dn.options().dtype(torch::kBFloat16));
  MoeArgs a;
  a.qw = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>()); a.qoff = qoff.data_ptr<int64_t>(); a.kexp = kexp.data_ptr<int>();
  a.scales = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()); a.su = su.data_ptr<uint8_t>(); a.sv = sv.data_ptr<uint8_t>();
  a.grp_e = grp_e.data_ptr<int>(); a.grp_rows = grp_rows.data_ptr<int>(); a.grp_n = grp_n.data_ptr<int>();
  a.x = nullptr; a.gu = gu.data_ptr<float>();
  a.part = part.data_ptr<float>(); a.counters = reinterpret_cast<unsigned*>(counters.data_ptr<int32_t>()); a.out = dn.data_ptr<float>();
  a.bps = (int)bps; a.ns = ns; a.cb = (float)cb;
  for (int k = 0; k < 6; ++k) a.spadk[k] = 0;
  a.spadk[2] = (int)spad2; a.spadk[3] = (int)spad3; a.spadk[4] = (int)spad4; a.spadk[5] = (int)spad5;
  a.fence_ra = fence_ra ? 1 : 0;
  a.slot = slot.data_ptr<int>(); a.wts = wts.data_ptr<float>(); a.ng = ng.data_ptr<int>();
  a.cnt2 = reinterpret_cast<unsigned*>(cnt2.data_ptr<int32_t>()); a.y = reinterpret_cast<__hip_bfloat16*>(y.data_ptr<at::BFloat16>());
  a.T = T; a.topk = topk;
  dim3 grid(m / 128, ns, A);
  moe_dispatch<1>((int)kf, (int)M, grid, at::hip::getCurrentHIPStream(), a);
  return y;
}

void dq_arena_subset(torch::Tensor qw, torch::Tensor qoff, torch::Tensor scales, torch::Tensor su, torch::Tensor sv,
                     torch::Tensor cbtab, torch::Tensor active_ids, int64_t K, int64_t role, int64_t pad,
                     int64_t m, int64_t n, int64_t s_off, int64_t s_es, int64_t su_off, int64_t su_es,
                     int64_t sv_off, int64_t sv_es, torch::Tensor out, int64_t w_es, int64_t row_off) {
  const int E_act = active_ids.size(0);
  if (E_act == 0) return;
  TORCH_CHECK(active_ids.is_cuda() && active_ids.is_contiguous() && active_ids.scalar_type() == torch::kInt32);
  TORCH_CHECK(m % 128 == 0 && n % 128 == 0 && role >= 0 && role <= 2 && pad >= 0);
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == torch::kBFloat16 && out.dim() == 3 && out.size(0) >= E_act);
  auto st = at::hip::getCurrentHIPStream();
  const int mt = m / 16, nt = n / 16;
  dim3 grid(nt / 8, mt / 8, E_act);
  const uint32_t* q = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>());
  const __half* sc = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  __hip_bfloat16* o = reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()) + (size_t)row_off * n;
  const float* cb = cbtab.data_ptr<float>();
  const int64_t* qo = qoff.data_ptr<int64_t>();
  const int* aids = active_ids.data_ptr<int>();
  if (K == 2) {
    hipLaunchKernelGGL((k_dq_arena_subset<2>), grid, dim3(256), 0, st, q, sc, cb, su.data_ptr<uint8_t>(), sv.data_ptr<uint8_t>(), o,
                       aids, mt, nt, qo, (int)role, (int)pad, (int)s_off, (int)s_es, (int)su_off, (int)su_es, (int)sv_off, (int)sv_es, w_es);
  } else if (K == 3) {
    hipLaunchKernelGGL((k_dq_arena_subset<3>), grid, dim3(256), 0, st, q, sc, cb, su.data_ptr<uint8_t>(), sv.data_ptr<uint8_t>(), o,
                       aids, mt, nt, qo, (int)role, (int)pad, (int)s_off, (int)s_es, (int)su_off, (int)su_es, (int)sv_off, (int)sv_es, w_es);
  } else if (K == 4) {
    hipLaunchKernelGGL((k_dq_arena_subset<4>), grid, dim3(256), 0, st, q, sc, cb, su.data_ptr<uint8_t>(), sv.data_ptr<uint8_t>(), o,
                       aids, mt, nt, qo, (int)role, (int)pad, (int)s_off, (int)s_es, (int)su_off, (int)su_es, (int)sv_off, (int)sv_es, w_es);
  } else if (K == 5) {
    hipLaunchKernelGGL((k_dq_arena_subset<5>), grid, dim3(256), 0, st, q, sc, cb, su.data_ptr<uint8_t>(), sv.data_ptr<uint8_t>(), o,
                       aids, mt, nt, qo, (int)role, (int)pad, (int)s_off, (int)s_es, (int)su_off, (int)su_es, (int)sv_off, (int)sv_es, w_es);
  } else {
    TORCH_CHECK(false, "K must be 2, 3, 4 or 5");
  }
}


