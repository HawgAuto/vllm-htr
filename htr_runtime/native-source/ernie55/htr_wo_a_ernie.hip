// Native gfx1151 (RDNA3.5) batched 8-group HCA wo_a HTR-W1 (K=5) GEMV + fused sigmoid(o_gate).
// Collapses 8 separate htr::linear calls + torch.cat + torch.sigmoid + multiply (~19 launches/layer,
// 703 launches/token across 37 HCA layers) into 1 Wave32 HIP kernel launch per HCA layer!
// Uses RDNA3.5 permlane16 zero-redundant-load V[0], non-temporal 16B weight loads (Infinity Cache bypass),
// EPW register-only Wave32 FWHT-128 epilogue, and fused sigmoid(o_gate) multiplication.
//
// Shape: G = 8 groups, each [M_OUT = 1024, N_IN = 4096], K = 5 (mt = 64, nt = 256, 40 words/tile).
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
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
  return __builtin_amdgcn_sad_u8(x, 0u, acc);
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
__device__ __forceinline__ float bf2f(uint16_t b) { return __builtin_bit_cast(float, (uint32_t)b << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {
  __hip_bfloat16 b = __float2bfloat16(f);
  return __builtin_bit_cast(uint16_t, b);
}
__device__ __forceinline__ uint32_t f32_to_bf16x2_rne(float a, float b) {
  uint32_t ua = __builtin_bit_cast(uint32_t, a);
  uint32_t ub = __builtin_bit_cast(uint32_t, b);
  ua = ((ua & 0x7fffffffu) > 0x7f800000u) ? (ua | 0x00400000u) : (ua + 0x7fffu + ((ua >> 16) & 1u));
  ub = ((ub & 0x7fffffffu) > 0x7f800000u) ? (ub | 0x00400000u) : (ub + 0x7fffu + ((ub >> 16) & 1u));
  return __builtin_amdgcn_perm(ub, ua, 0x07060302u);
}

// K=5 batched 8-group wo_a GEMV:
//   x:    [T, G, 4096] bf16  (G = 8)
//   qw:   [G, 64, 256, 40] int32 (contiguous)
//   scl:  [G, 1024] fp16
//   su:   [G, 512] uint8
//   sv:   [G, 128] uint8
//   gate: [T, 8192] bf16 (optional; if non-null, output is z * sigmoid(gate))
//   out:  [T, 8192] bf16
//   part: [G, NS, M, 1024] fp32
//   cnt:  [G, 8] uint32 (zero-initialized, self-resetting)
// Grid: (8 /*sb*/, NS, 8 /*group g*/), Block: 256 (8 waves)
template <int M, int NBL, int NS, int HAS_GATE, int SP16>
__global__ __launch_bounds__(256) void k_wo_a_k5_batched(
    const uint32_t* __restrict__ qw,
    const __half* __restrict__ scales,
    const uint8_t* __restrict__ su,
    const uint8_t* __restrict__ sv,
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ gate,
    float* __restrict__ part,
    unsigned* __restrict__ counters,
    uint16_t* __restrict__ out,
    int T, float cb, int spad
) {
  constexpr int K = 5;
  constexpr int G = 8;
  constexpr int M_OUT = 1024;
  constexpr int N_IN = 4096;
  constexpr int MT = 64;
  constexpr int NT = 256;
  constexpr int MPB = 8; // 1024 / 128

  __shared__ __attribute__((aligned(16))) _Float16 xs_l[M][NBL * 128];
  __shared__ float bscale[M][NBL];
  __shared__ float bsum[M][NBL];
  __shared__ unsigned is_last;

  const int sb = blockIdx.x;
  const int sp = blockIdx.y;
  const int grp = blockIdx.z;
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int mr = T;

  const size_t pitch = (size_t)NT * 40 + spad;
  const uint32_t* qm = SP16 ? nullptr : (qw + (size_t)grp * MT * pitch);
  const uint32_t* p16_grp = SP16 ? (qw + (size_t)grp * (MT * NT * 40)) : nullptr;
  const uint32_t* p4_grp  = SP16 ? (p16_grp + (size_t)(MT * NT) * 32) : nullptr;
  const __half* scl = scales + (size_t)grp * M_OUT;
  const uint8_t* sup = su + (size_t)grp * (N_IN / 8);
  const uint8_t* svp = sv + (size_t)grp * (M_OUT / 8);

  const int b0 = sp * NBL;
  const int b1 = min(N_IN >> 7, b0 + NBL);
  const int nb = b1 - b0;

  // Prologue: transform only real rows 0..mr-1 for group grp
  for (int it = w; it < mr * nb; it += 8) {
    const int r = it / nb;
    const int bl = it % nb;
    const int base = (b0 + bl) * 128 + lane * 4;
    const uint16_t* xr = x + ((size_t)r * G + grp) * N_IN;
    const uint2 raw = *reinterpret_cast<const uint2*>(xr + base);
    float v[4] = {
      __uint_as_float((raw.x & 0xFFFFu) << 16),
      __uint_as_float(raw.x & 0xFFFF0000u),
      __uint_as_float((raw.y & 0xFFFFu) << 16),
      __uint_as_float(raw.y & 0xFFFF0000u),
    };
    const unsigned sg4 = (unsigned)sup[base >> 3] >> (base & 7);
    if (sg4 & 1u) v[0] = -v[0];
    if (sg4 & 2u) v[1] = -v[1];
    if (sg4 & 4u) v[2] = -v[2];
    if (sg4 & 8u) v[3] = -v[3];
    fwht128_wave(v, lane);
    float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    am = wave_max(am) * (1.0f / 11.313708498984761f);
    const float sc = am > 30000.f ? exp2f(ceilf(log2f(am * (1.0f / 30000.f)))) : 1.f;
    const float inv = (1.0f / 11.313708498984761f) * __builtin_amdgcn_rcpf(sc);
    const h4v h4 = {(_Float16)(v[0] * inv), (_Float16)(v[1] * inv),
                    (_Float16)(v[2] * inv), (_Float16)(v[3] * inv)};
    *reinterpret_cast<h4v*>(&xs_l[r][bl * 128 + lane * 4]) = h4;
    float sum = (float)h4.x + (float)h4.y + (float)h4.z + (float)h4.w;
    sum = wave_sum(sum);
    if (lane == 0) { bscale[r][bl] = sc; bsum[r][bl] = sum; }
  }
  __syncthreads();

  const int il = sb * 8 + w;
  const int g_lane = lane >> 3, q = lane & 7;
  float a0[M], a1[M];
#pragma unroll
  for (int r = 0; r < M; ++r) { a0[r] = 0.f; a1[r] = 0.f; }

  // Zero-VRAM cyclic K-tile stagger for SP16==0 && spad==0: breaks 8-way LPDDR5X channel conflict on 40960B row pitch!
  const int ntiles = nb * 8;
  const int phase = (spad == 0) ? ((ntiles >= 64) ? ((w & 7) * 8) : ((ntiles >= 32) ? ((w & 7) * 4) : 0)) : 0;

  // 2-stage register software prefetch of K=5 weights (xs_l is in LDS -> 0 vmcnt conflict across BOTH passes!)
  u4v wa_cur = {0u, 0u, 0u, 0u};
  uint32_t w5_cur = 0u;
  {
    const int jl0 = ((phase < ntiles) ? phase : 0) + g_lane;
    const int j0 = b0 * 8 + jl0;
    if constexpr (SP16) {
      const size_t tid0 = (size_t)il * NT + j0;
      wa_cur = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(p16_grp + tid0 * 32 + q * 4));
      w5_cur = __builtin_nontemporal_load(p4_grp + tid0 * 8 + q);
    } else {
      const uint32_t* tp0 = qm + (size_t)il * pitch + (size_t)j0 * 40;
      wa_cur = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tp0 + q * 5));
      w5_cur = __builtin_nontemporal_load(tp0 + q * 5 + 4);
    }
  }

  for (int step = 0; step < ntiles; step += 4) {
    int jl_base = step + phase;
    if (jl_base >= ntiles) jl_base -= ntiles;
    const int jl = jl_base + g_lane;
    const int bl = jl >> 3;
    uint32_t V[6];
    V[0] = __builtin_amdgcn_permlane16(w5_cur, w5_cur, 0x65432107u, 0xedcba98fu, false, false);
    V[1] = wa_cur.x; V[2] = wa_cur.y; V[3] = wa_cur.z; V[4] = wa_cur.w; V[5] = w5_cur;

    // Issue next tile's LPDDR5X NT loads BEFORE Pass 1 & Pass 2 (xs_l uses lgkmcnt only -> 100% overlap!)
    u4v wa_nxt = {0u, 0u, 0u, 0u};
    uint32_t w5_nxt = 0u;
    if (step + 4 < ntiles) {
      int jn_base = step + 4 + phase;
      if (jn_base >= ntiles) jn_base -= ntiles;
      const int jn = b0 * 8 + jn_base + g_lane;
      if constexpr (SP16) {
        const size_t tidn = (size_t)il * NT + jn;
        wa_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(p16_grp + tidn * 32 + q * 4));
        w5_nxt = __builtin_nontemporal_load(p4_grp + tidn * 8 + q);
      } else {
        const uint32_t* tpn = qm + (size_t)il * pitch + (size_t)jn * 40;
        wa_nxt = *reinterpret_cast<const u4va*>(tpn + q * 5);
        w5_nxt = tpn[q * 5 + 4];
      }
    }
    __builtin_amdgcn_sched_barrier(0);

    // Half-tile xv[M][4] + bank-staggered (U_lo, U_hi) pairing:
    //   1) Halves xv VGPR pressure (-16..-32 VGPRs -> M=4,8 drop <=96 VGPRs = FULL 16w/SIMD!)
    //   2) Staggers xv[r][k] vs xv[r][k^1] across different VGPR banks for V_DUAL_DOT2ACC_F32_F16!
    h2v xv[M][4];
    float d0[M], d1[M];
#pragma unroll
    for (int r = 0; r < M; ++r) {
      const uint4 u0 = reinterpret_cast<const uint4*>(&xs_l[r][jl * 16])[0];
      xv[r][0] = as_h2(u0.x); xv[r][1] = as_h2(u0.y); xv[r][2] = as_h2(u0.z); xv[r][3] = as_h2(u0.w);
      d0[r] = 0.f; d1[r] = 0.f;
    }
#define HTR_DUAL5(U0, U1)                                                                      \
    {                                                                                          \
      const h2v w0 = as_h2(val_f16bits2(ext16<16 + ((U0) + 1) * K>(V), ext16<16 + ((U0) + 2) * K>(V))); \
      const h2v w1 = as_h2(val_f16bits2(ext16<16 + ((U1) + 1) * K>(V), ext16<16 + ((U1) + 2) * K>(V))); \
      _Pragma("unroll") for (int r = 0; r < M; ++r) {                                          \
        d0[r] = __builtin_amdgcn_fdot2(w0, xv[r][((U0) & 7) >> 1], d0[r], false);              \
        d1[r] = __builtin_amdgcn_fdot2(w1, xv[r][((U1) & 7) >> 1], d1[r], false);              \
      }                                                                                        \
    }
    HTR_DUAL5(0, 18) HTR_DUAL5(2, 16) HTR_DUAL5(4, 22) HTR_DUAL5(6, 20)
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int r = 0; r < M; ++r) {
      const uint4 u1 = reinterpret_cast<const uint4*>(&xs_l[r][jl * 16])[1];
      xv[r][0] = as_h2(u1.x); xv[r][1] = as_h2(u1.y); xv[r][2] = as_h2(u1.z); xv[r][3] = as_h2(u1.w);
    }
    HTR_DUAL5(8, 26) HTR_DUAL5(10, 24) HTR_DUAL5(12, 30) HTR_DUAL5(14, 28)
#undef HTR_DUAL5
#pragma unroll
    for (int r = 0; r < M; ++r) { const float s = bscale[r][bl]; a0[r] += s * d0[r]; a1[r] += s * d1[r]; }
    wa_cur = wa_nxt;
    w5_cur = w5_nxt;
  }

#pragma unroll
  for (int r = 0; r < M; ++r) {
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
  }
  if (g_lane == 0) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
      if (r >= mr) continue;
      float off = 0.f;
      for (int bl = 0; bl < nb; ++bl) off += bscale[r][bl] * bsum[r][bl];
      off *= 1534.0f;
      float* pp = part + (((size_t)grp * NS + sp) * M + r) * M_OUT + il * 16 + 2 * q;
      *reinterpret_cast<float2*>(pp) = make_float2(a0[r] - off, a1[r] - off);
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();
  unsigned* cnt = counters + (size_t)grp * MPB + sb;
  if (threadIdx.x == 0) is_last = (atomicAdd(cnt, 1u) == (unsigned)(NS - 1));
  __syncthreads();
  if (!is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  // EPW register-only Wave32 FWHT-128 epilogue + optional fused sigmoid(gate) with 8B vectorized R/W
  if (w < mr) {
    const int r = w;
    f4v s4 = {0.f, 0.f, 0.f, 0.f};
    f4v pv[NS];
#pragma unroll
    for (int k = 0; k < NS; ++k) {
      pv[k] = __builtin_nontemporal_load(reinterpret_cast<const f4v*>(
          part + (((size_t)grp * NS + k) * M + r) * M_OUT + sb * 128 + lane * 4));
    }
#pragma unroll
    for (int k = 0; k < NS; ++k) s4 += pv[k];
    const h4v hs = *reinterpret_cast<const h4v*>(scl + sb * 128 + lane * 4);
    float v[4] = {
      s4.x * (float)hs.x * cb,
      s4.y * (float)hs.y * cb,
      s4.z * (float)hs.z * cb,
      s4.w * (float)hs.w * cb
    };
    fwht128_wave(v, lane);
    const unsigned sg4 = (unsigned)svp[sb * 16 + (lane >> 1)] >> ((lane & 1) * 4);
    float o[4] = {
      v[0] * (1.0f / 11.313708498984761f),
      v[1] * (1.0f / 11.313708498984761f),
      v[2] * (1.0f / 11.313708498984761f),
      v[3] * (1.0f / 11.313708498984761f)
    };
    if (sg4 & 1) o[0] = -o[0];
    if (sg4 & 2) o[1] = -o[1];
    if (sg4 & 4) o[2] = -o[2];
    if (sg4 & 8) o[3] = -o[3];

    const size_t out_off = (size_t)r * (G * M_OUT) + (size_t)grp * M_OUT + sb * 128 + lane * 4;
    uint2 out_p;
    if constexpr (HAS_GATE) {
      const uint2 graw = *reinterpret_cast<const uint2*>(gate + out_off);
      const float gfv[4] = {
        __uint_as_float((graw.x & 0xFFFFu) << 16),
        __uint_as_float(graw.x & 0xFFFF0000u),
        __uint_as_float((graw.y & 0xFFFFu) << 16),
        __uint_as_float(graw.y & 0xFFFF0000u),
      };
      float prod[4];
#pragma unroll
      for (int e2 = 0; e2 < 4; ++e2) {
        const float sig = __builtin_amdgcn_rcpf(1.0f + exp2f(-1.4426950408889634f * gfv[e2]));
        const float zb = bf2f(f2bf(o[e2]));
        const float gb = bf2f(f2bf(sig));
        prod[e2] = zb * gb;
      }
      out_p.x = f32_to_bf16x2_rne(prod[0], prod[1]);
      out_p.y = f32_to_bf16x2_rne(prod[2], prod[3]);
    } else {
      out_p.x = f32_to_bf16x2_rne(o[0], o[1]);
      out_p.y = f32_to_bf16x2_rne(o[2], o[3]);
    }
    *reinterpret_cast<uint2*>(out + out_off) = out_p;
  }
  if (threadIdx.x == 0) *cnt = 0u;
}


// v0.2.8: 2-subpass T=9..16 batched 8-group wo_a K=5 GEMV (recovers ~3.4 ms/step at T=12 concurrent B=2!)
// Always uses NS=8, NBL=4 (12 KiB LDS for M_CAP=12, 16 KiB LDS for M_CAP=16 -> 2+ WGs/CU!).
template <int M_CAP, int HAS_GATE, int SP16>
__global__ __launch_bounds__(256) void k_wo_a_k5_batched_big(
    const uint32_t* __restrict__ qw,
    const __half* __restrict__ scales,
    const uint8_t* __restrict__ su,
    const uint8_t* __restrict__ sv,
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ gate,
    float* __restrict__ part,
    unsigned* __restrict__ counters,
    uint16_t* __restrict__ out,
    int T, float cb, int spad
) {
  static_assert(M_CAP > 8 && M_CAP <= 16, "M_CAP must be in 9..16");
  constexpr int R2 = M_CAP - 8;
  constexpr int K = 5;
  constexpr int G = 8;
  constexpr int M_OUT = 1024;
  constexpr int N_IN = 4096;
  constexpr int MT = 64;
  constexpr int NT = 256;
  constexpr int MPB = 8;
  constexpr int NBL = 4;
  constexpr int NS = 8;

  __shared__ __attribute__((aligned(16))) _Float16 xs_l[M_CAP][NBL * 128];
  __shared__ float bscale[M_CAP][NBL];
  __shared__ float bsum[M_CAP][NBL];
  __shared__ unsigned is_last;

  const int sb = blockIdx.x;
  const int sp = blockIdx.y;
  const int grp = blockIdx.z;
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int mr = T;

  const size_t pitch = (size_t)NT * 40 + spad;
  const uint32_t* qm = SP16 ? nullptr : (qw + (size_t)grp * MT * pitch);
  const uint32_t* p16_grp = SP16 ? (qw + (size_t)grp * (MT * NT * 40)) : nullptr;
  const uint32_t* p4_grp  = SP16 ? (p16_grp + (size_t)(MT * NT) * 32) : nullptr;
  const __half* scl = scales + (size_t)grp * M_OUT;
  const uint8_t* sup = su + (size_t)grp * (N_IN / 8);
  const uint8_t* svp = sv + (size_t)grp * (M_OUT / 8);

  const int b0 = sp * NBL;
  const int nb = NBL;

  for (int it = w; it < mr * nb; it += 8) {
    const int r = it >> 2;
    const int bl = it & 3;
    const int base = (b0 + bl) * 128 + lane * 4;
    const uint16_t* xr = x + ((size_t)r * G + grp) * N_IN;
    const uint2 raw = *reinterpret_cast<const uint2*>(xr + base);
    float v[4] = {
      __uint_as_float((raw.x & 0xFFFFu) << 16),
      __uint_as_float(raw.x & 0xFFFF0000u),
      __uint_as_float((raw.y & 0xFFFFu) << 16),
      __uint_as_float(raw.y & 0xFFFF0000u),
    };
    const unsigned sg4 = (unsigned)sup[base >> 3] >> (base & 7);
    if (sg4 & 1u) v[0] = -v[0];
    if (sg4 & 2u) v[1] = -v[1];
    if (sg4 & 4u) v[2] = -v[2];
    if (sg4 & 8u) v[3] = -v[3];
    fwht128_wave(v, lane);
    float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    am = wave_max(am) * (1.0f / 11.313708498984761f);
    const float sc = am > 30000.f ? exp2f(ceilf(log2f(am * (1.0f / 30000.f)))) : 1.f;
    const float inv = (1.0f / 11.313708498984761f) * __builtin_amdgcn_rcpf(sc);
    const h4v h4 = {(_Float16)(v[0] * inv), (_Float16)(v[1] * inv),
                    (_Float16)(v[2] * inv), (_Float16)(v[3] * inv)};
    *reinterpret_cast<h4v*>(&xs_l[r][bl * 128 + lane * 4]) = h4;
    float sum = (float)h4.x + (float)h4.y + (float)h4.z + (float)h4.w;
    sum = wave_sum(sum);
    if (lane == 0) { bscale[r][bl] = sc; bsum[r][bl] = sum; }
  }
  __syncthreads();

  const int il = sb * 8 + w;
  const int g_lane = lane >> 3, q = lane & 7;
  float a0[M_CAP], a1[M_CAP];
#pragma unroll
  for (int r = 0; r < M_CAP; ++r) { a0[r] = 0.f; a1[r] = 0.f; }

  constexpr int ntiles = NBL * 8;
  const int phase = (spad == 0) ? ((w & 7) * 4) : 0;

  u4v wa_cur = {0u, 0u, 0u, 0u};
  uint32_t w5_cur = 0u;
  {
    const int jl0 = phase + g_lane;
    const int j0 = b0 * 8 + jl0;
    if constexpr (SP16) {
      const size_t tid0 = (size_t)il * NT + j0;
      wa_cur = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(p16_grp + tid0 * 32 + q * 4));
      w5_cur = __builtin_nontemporal_load(p4_grp + tid0 * 8 + q);
    } else {
      const uint32_t* tp0 = qm + (size_t)il * pitch + (size_t)j0 * 40;
      wa_cur = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tp0 + q * 5));
      w5_cur = __builtin_nontemporal_load(tp0 + q * 5 + 4);
    }
  }

  for (int step = 0; step < ntiles; step += 4) {
    int jl_base = step + phase;
    if (jl_base >= ntiles) jl_base -= ntiles;
    const int jl = jl_base + g_lane;
    const int bl = jl >> 3;
    uint32_t V[6];
    V[0] = __builtin_amdgcn_permlane16(w5_cur, w5_cur, 0x65432107u, 0xedcba98fu, false, false);
    V[1] = wa_cur.x; V[2] = wa_cur.y; V[3] = wa_cur.z; V[4] = wa_cur.w; V[5] = w5_cur;

    u4v wa_nxt = {0u, 0u, 0u, 0u};
    uint32_t w5_nxt = 0u;
    if (step + 4 < ntiles) {
      int jn_base = step + 4 + phase;
      if (jn_base >= ntiles) jn_base -= ntiles;
      const int jn = b0 * 8 + jn_base + g_lane;
      if constexpr (SP16) {
        const size_t tidn = (size_t)il * NT + jn;
        wa_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(p16_grp + tidn * 32 + q * 4));
        w5_nxt = __builtin_nontemporal_load(p4_grp + tidn * 8 + q);
      } else {
        const uint32_t* tpn = qm + (size_t)il * pitch + (size_t)jn * 40;
        wa_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tpn + q * 5));
        w5_nxt = __builtin_nontemporal_load(tpn + q * 5 + 4);
      }
    }
    __builtin_amdgcn_sched_barrier(0);

    const h2v w0a = as_h2(val_f16bits2(ext16<16 + 1 * K>(V),  ext16<16 + 2 * K>(V)));
    const h2v w1a = as_h2(val_f16bits2(ext16<16 + 19 * K>(V), ext16<16 + 20 * K>(V)));
    const h2v w0b = as_h2(val_f16bits2(ext16<16 + 3 * K>(V),  ext16<16 + 4 * K>(V)));
    const h2v w1b = as_h2(val_f16bits2(ext16<16 + 17 * K>(V), ext16<16 + 18 * K>(V)));
    const h2v w0c = as_h2(val_f16bits2(ext16<16 + 5 * K>(V),  ext16<16 + 6 * K>(V)));
    const h2v w1c = as_h2(val_f16bits2(ext16<16 + 23 * K>(V), ext16<16 + 24 * K>(V)));
    const h2v w0d = as_h2(val_f16bits2(ext16<16 + 7 * K>(V),  ext16<16 + 8 * K>(V)));
    const h2v w1d = as_h2(val_f16bits2(ext16<16 + 21 * K>(V), ext16<16 + 22 * K>(V)));
    {
      h2v xv[8][4];
      float d0[8], d1[8];
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const uint4 u0 = reinterpret_cast<const uint4*>(&xs_l[r][jl * 16])[0];
        xv[r][0] = as_h2(u0.x); xv[r][1] = as_h2(u0.y); xv[r][2] = as_h2(u0.z); xv[r][3] = as_h2(u0.w);
        d0[r] = 0.f; d1[r] = 0.f;
      }
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        d0[r] = __builtin_amdgcn_fdot2(w0a, xv[r][0], d0[r], false);
        d1[r] = __builtin_amdgcn_fdot2(w1a, xv[r][1], d1[r], false);
        d0[r] = __builtin_amdgcn_fdot2(w0b, xv[r][1], d0[r], false);
        d1[r] = __builtin_amdgcn_fdot2(w1b, xv[r][0], d1[r], false);
        d0[r] = __builtin_amdgcn_fdot2(w0c, xv[r][2], d0[r], false);
        d1[r] = __builtin_amdgcn_fdot2(w1c, xv[r][3], d1[r], false);
        d0[r] = __builtin_amdgcn_fdot2(w0d, xv[r][3], d0[r], false);
        d1[r] = __builtin_amdgcn_fdot2(w1d, xv[r][2], d1[r], false);
        const float s = bscale[r][bl];
        a0[r] += s * d0[r]; a1[r] += s * d1[r];
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    {
      h2v xv2[R2][4];
      float d0[R2], d1[R2];
#pragma unroll
      for (int k = 0; k < R2; ++k) {
        const int rr = (8 + k < mr) ? (8 + k) : (mr - 1);
        const uint4 u0 = reinterpret_cast<const uint4*>(&xs_l[rr][jl * 16])[0];
        xv2[k][0] = as_h2(u0.x); xv2[k][1] = as_h2(u0.y); xv2[k][2] = as_h2(u0.z); xv2[k][3] = as_h2(u0.w);
        d0[k] = 0.f; d1[k] = 0.f;
      }
#pragma unroll
      for (int k = 0; k < R2; ++k) {
        d0[k] = __builtin_amdgcn_fdot2(w0a, xv2[k][0], d0[k], false);
        d1[k] = __builtin_amdgcn_fdot2(w1a, xv2[k][1], d1[k], false);
        d0[k] = __builtin_amdgcn_fdot2(w0b, xv2[k][1], d0[k], false);
        d1[k] = __builtin_amdgcn_fdot2(w1b, xv2[k][0], d1[k], false);
        d0[k] = __builtin_amdgcn_fdot2(w0c, xv2[k][2], d0[k], false);
        d1[k] = __builtin_amdgcn_fdot2(w1c, xv2[k][3], d1[k], false);
        d0[k] = __builtin_amdgcn_fdot2(w0d, xv2[k][3], d0[k], false);
        d1[k] = __builtin_amdgcn_fdot2(w1d, xv2[k][2], d1[k], false);
        const int rr = (8 + k < mr) ? (8 + k) : (mr - 1);
        const float s = bscale[rr][bl];
        a0[8 + k] += s * d0[k]; a1[8 + k] += s * d1[k];
      }
    }
    __builtin_amdgcn_sched_barrier(0);

    const h2v w0e = as_h2(val_f16bits2(ext16<16 + 9 * K>(V),  ext16<16 + 10 * K>(V)));
    const h2v w1e = as_h2(val_f16bits2(ext16<16 + 27 * K>(V), ext16<16 + 28 * K>(V)));
    const h2v w0f = as_h2(val_f16bits2(ext16<16 + 11 * K>(V), ext16<16 + 12 * K>(V)));
    const h2v w1f = as_h2(val_f16bits2(ext16<16 + 25 * K>(V), ext16<16 + 26 * K>(V)));
    const h2v w0g = as_h2(val_f16bits2(ext16<16 + 13 * K>(V), ext16<16 + 14 * K>(V)));
    const h2v w1g = as_h2(val_f16bits2(ext16<16 + 31 * K>(V), ext16<16 + 32 * K>(V)));
    const h2v w0h = as_h2(val_f16bits2(ext16<16 + 15 * K>(V), ext16<16 + 16 * K>(V)));
    const h2v w1h = as_h2(val_f16bits2(ext16<16 + 29 * K>(V), ext16<16 + 30 * K>(V)));
    {
      h2v xv[8][4];
      float d0[8], d1[8];
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const uint4 u1 = reinterpret_cast<const uint4*>(&xs_l[r][jl * 16])[1];
        xv[r][0] = as_h2(u1.x); xv[r][1] = as_h2(u1.y); xv[r][2] = as_h2(u1.z); xv[r][3] = as_h2(u1.w);
        d0[r] = 0.f; d1[r] = 0.f;
      }
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        d0[r] = __builtin_amdgcn_fdot2(w0e, xv[r][0], d0[r], false);
        d1[r] = __builtin_amdgcn_fdot2(w1e, xv[r][1], d1[r], false);
        d0[r] = __builtin_amdgcn_fdot2(w0f, xv[r][1], d0[r], false);
        d1[r] = __builtin_amdgcn_fdot2(w1f, xv[r][0], d1[r], false);
        d0[r] = __builtin_amdgcn_fdot2(w0g, xv[r][2], d0[r], false);
        d1[r] = __builtin_amdgcn_fdot2(w1g, xv[r][3], d1[r], false);
        d0[r] = __builtin_amdgcn_fdot2(w0h, xv[r][3], d0[r], false);
        d1[r] = __builtin_amdgcn_fdot2(w1h, xv[r][2], d1[r], false);
        const float s = bscale[r][bl];
        a0[r] += s * d0[r]; a1[r] += s * d1[r];
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    {
      h2v xv2[R2][4];
      float d0[R2], d1[R2];
#pragma unroll
      for (int k = 0; k < R2; ++k) {
        const int rr = (8 + k < mr) ? (8 + k) : (mr - 1);
        const uint4 u1 = reinterpret_cast<const uint4*>(&xs_l[rr][jl * 16])[1];
        xv2[k][0] = as_h2(u1.x); xv2[k][1] = as_h2(u1.y); xv2[k][2] = as_h2(u1.z); xv2[k][3] = as_h2(u1.w);
        d0[k] = 0.f; d1[k] = 0.f;
      }
#pragma unroll
      for (int k = 0; k < R2; ++k) {
        d0[k] = __builtin_amdgcn_fdot2(w0e, xv2[k][0], d0[k], false);
        d1[k] = __builtin_amdgcn_fdot2(w1e, xv2[k][1], d1[k], false);
        d0[k] = __builtin_amdgcn_fdot2(w0f, xv2[k][1], d0[k], false);
        d1[k] = __builtin_amdgcn_fdot2(w1f, xv2[k][0], d1[k], false);
        d0[k] = __builtin_amdgcn_fdot2(w0g, xv2[k][2], d0[k], false);
        d1[k] = __builtin_amdgcn_fdot2(w1g, xv2[k][3], d1[k], false);
        d0[k] = __builtin_amdgcn_fdot2(w0h, xv2[k][3], d0[k], false);
        d1[k] = __builtin_amdgcn_fdot2(w1h, xv2[k][2], d1[k], false);
        const int rr = (8 + k < mr) ? (8 + k) : (mr - 1);
        const float s = bscale[rr][bl];
        a0[8 + k] += s * d0[k]; a1[8 + k] += s * d1[k];
      }
    }

    wa_cur = wa_nxt;
    w5_cur = w5_nxt;
  }

#pragma unroll
  for (int r = 0; r < M_CAP; ++r) {
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
  }
  if (g_lane == 0) {
#pragma unroll
    for (int r = 0; r < M_CAP; ++r) {
      if (r >= mr) continue;
      float off = 0.f;
#pragma unroll
      for (int bl = 0; bl < NBL; ++bl) off += bscale[r][bl] * bsum[r][bl];
      off *= 1534.0f;
      float* pp = part + (((size_t)grp * NS + sp) * M_CAP + r) * M_OUT + il * 16 + 2 * q;
      *reinterpret_cast<float2*>(pp) = make_float2(a0[r] - off, a1[r] - off);
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();
  unsigned* cnt = counters + (size_t)grp * MPB + sb;
  if (threadIdx.x == 0) is_last = (atomicAdd(cnt, 1u) == (unsigned)(NS - 1));
  __syncthreads();
  if (!is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const h4v hs = *reinterpret_cast<const h4v*>(scl + sb * 128 + lane * 4);
  const unsigned sg4 = (unsigned)svp[sb * 16 + (lane >> 1)] >> ((lane & 1) * 4);
  for (int r = w; r < mr; r += 8) {
    f4v s4 = {0.f, 0.f, 0.f, 0.f};
    f4v pv[NS];
#pragma unroll
    for (int k = 0; k < NS; ++k) {
      pv[k] = __builtin_nontemporal_load(reinterpret_cast<const f4v*>(
          part + (((size_t)grp * NS + k) * M_CAP + r) * M_OUT + sb * 128 + lane * 4));
    }
#pragma unroll
    for (int k = 0; k < NS; ++k) s4 += pv[k];
    float v[4] = {
      s4.x * (float)hs.x * cb, s4.y * (float)hs.y * cb,
      s4.z * (float)hs.z * cb, s4.w * (float)hs.w * cb
    };
    fwht128_wave(v, lane);
    float o[4] = {
      v[0] * (1.0f / 11.313708498984761f), v[1] * (1.0f / 11.313708498984761f),
      v[2] * (1.0f / 11.313708498984761f), v[3] * (1.0f / 11.313708498984761f)
    };
    if (sg4 & 1) o[0] = -o[0];
    if (sg4 & 2) o[1] = -o[1];
    if (sg4 & 4) o[2] = -o[2];
    if (sg4 & 8) o[3] = -o[3];

    const size_t out_off = (size_t)r * (G * M_OUT) + (size_t)grp * M_OUT + sb * 128 + lane * 4;
    uint2 out_p;
    if constexpr (HAS_GATE) {
      const uint2 graw = *reinterpret_cast<const uint2*>(gate + out_off);
      const float gfv[4] = {
        __uint_as_float((graw.x & 0xFFFFu) << 16),
        __uint_as_float(graw.x & 0xFFFF0000u),
        __uint_as_float((graw.y & 0xFFFFu) << 16),
        __uint_as_float(graw.y & 0xFFFF0000u),
      };
      float prod[4];
#pragma unroll
      for (int e2 = 0; e2 < 4; ++e2) {
        const float sig = __builtin_amdgcn_rcpf(1.0f + exp2f(-1.4426950408889634f * gfv[e2]));
        const float zb = bf2f(f2bf(o[e2]));
        const float gb = bf2f(f2bf(sig));
        prod[e2] = zb * gb;
      }
      out_p.x = f32_to_bf16x2_rne(prod[0], prod[1]);
      out_p.y = f32_to_bf16x2_rne(prod[2], prod[3]);
    } else {
      out_p.x = f32_to_bf16x2_rne(o[0], o[1]);
      out_p.y = f32_to_bf16x2_rne(o[2], o[3]);
    }
    *reinterpret_cast<uint2*>(out + out_off) = out_p;
  }
  if (threadIdx.x == 0) *cnt = 0u;
}

template <int M, int HAS_GATE, int SP16>
static void dispatch_ns(
    int ns, const uint32_t* qw, const __half* scales, const uint8_t* su, const uint8_t* sv,
    const uint16_t* x, const uint16_t* gate, float* part, unsigned* cnt, uint16_t* out,
    int T, float cb, int spad, hipStream_t stream
) {
  // N_IN / 128 = 32 blocks total
  if (ns == 8) {
    hipLaunchKernelGGL((k_wo_a_k5_batched<M, 4, 8, HAS_GATE, SP16>), dim3(8, 8, 8), dim3(256), 0, stream,
                       qw, scales, su, sv, x, gate, part, cnt, out, T, cb, spad);
  } else if (ns == 4) {
    hipLaunchKernelGGL((k_wo_a_k5_batched<M, 8, 4, HAS_GATE, SP16>), dim3(8, 4, 8), dim3(256), 0, stream,
                       qw, scales, su, sv, x, gate, part, cnt, out, T, cb, spad);
  } else {
    hipLaunchKernelGGL((k_wo_a_k5_batched<M, 16, 2, HAS_GATE, SP16>), dim3(8, 2, 8), dim3(256), 0, stream,
                       qw, scales, su, sv, x, gate, part, cnt, out, T, cb, spad);
  }
}

void launch_wo_a_k5_batched(
    torch::Tensor qw, torch::Tensor scales, torch::Tensor su, torch::Tensor sv,
    torch::Tensor x, c10::optional<torch::Tensor> gate,
    torch::Tensor part, torch::Tensor counters, torch::Tensor out,
    int64_t ns, double cb, int64_t spad, bool sp16 = false
) {
  const int T = (int)x.size(0);
  if (T == 0) return;
  TORCH_CHECK(T <= 16, "launch_wo_a_k5_batched supports T <= 16 (got ", T, ")");
  auto stream = c10::hip::getCurrentHIPStream().stream();
  const uint32_t* qw_p = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>());
  const __half* scl_p = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  const uint8_t* su_p = su.data_ptr<uint8_t>();
  const uint8_t* sv_p = sv.data_ptr<uint8_t>();
  const uint16_t* x_p = reinterpret_cast<const uint16_t*>(x.data_ptr());
  const uint16_t* g_p = gate.has_value() ? reinterpret_cast<const uint16_t*>(gate->data_ptr()) : nullptr;
  float* part_p = part.data_ptr<float>();
  unsigned* cnt_p = reinterpret_cast<unsigned*>(counters.data_ptr<int32_t>());
  uint16_t* out_p = reinterpret_cast<uint16_t*>(out.data_ptr());

#define DISP_M(MM)                                                                                                  \
  if (sp16) {                                                                                                       \
    if (g_p) dispatch_ns<MM, 1, 1>((int)ns, qw_p, scl_p, su_p, sv_p, x_p, g_p, part_p, cnt_p, out_p, T, (float)cb, 0, stream); \
    else     dispatch_ns<MM, 0, 1>((int)ns, qw_p, scl_p, su_p, sv_p, x_p, nullptr, part_p, cnt_p, out_p, T, (float)cb, 0, stream); \
  } else {                                                                                                          \
    if (g_p) dispatch_ns<MM, 1, 0>((int)ns, qw_p, scl_p, su_p, sv_p, x_p, g_p, part_p, cnt_p, out_p, T, (float)cb, (int)spad, stream); \
    else     dispatch_ns<MM, 0, 0>((int)ns, qw_p, scl_p, su_p, sv_p, x_p, nullptr, part_p, cnt_p, out_p, T, (float)cb, (int)spad, stream); \
  }

  if (T == 1) { DISP_M(1); }
  else if (T == 2) { DISP_M(2); }
  else if (T == 3) { DISP_M(3); }
  else if (T == 4) { DISP_M(4); }
  else if (T == 5) { DISP_M(5); }
  else if (T == 6) { DISP_M(6); }
  else if (T == 7) { DISP_M(7); }
  else if (T == 8) { DISP_M(8); }
  else if (T <= 12) {
    if (sp16) {
      if (g_p) hipLaunchKernelGGL((k_wo_a_k5_batched_big<12, 1, 1>), dim3(8, 8, 8), dim3(256), 0, stream, qw_p, scl_p, su_p, sv_p, x_p, g_p, part_p, cnt_p, out_p, T, (float)cb, 0);
      else     hipLaunchKernelGGL((k_wo_a_k5_batched_big<12, 0, 1>), dim3(8, 8, 8), dim3(256), 0, stream, qw_p, scl_p, su_p, sv_p, x_p, nullptr, part_p, cnt_p, out_p, T, (float)cb, 0);
    } else {
      if (g_p) hipLaunchKernelGGL((k_wo_a_k5_batched_big<12, 1, 0>), dim3(8, 8, 8), dim3(256), 0, stream, qw_p, scl_p, su_p, sv_p, x_p, g_p, part_p, cnt_p, out_p, T, (float)cb, (int)spad);
      else     hipLaunchKernelGGL((k_wo_a_k5_batched_big<12, 0, 0>), dim3(8, 8, 8), dim3(256), 0, stream, qw_p, scl_p, su_p, sv_p, x_p, nullptr, part_p, cnt_p, out_p, T, (float)cb, (int)spad);
    }
  } else {
    if (sp16) {
      if (g_p) hipLaunchKernelGGL((k_wo_a_k5_batched_big<16, 1, 1>), dim3(8, 8, 8), dim3(256), 0, stream, qw_p, scl_p, su_p, sv_p, x_p, g_p, part_p, cnt_p, out_p, T, (float)cb, 0);
      else     hipLaunchKernelGGL((k_wo_a_k5_batched_big<16, 0, 1>), dim3(8, 8, 8), dim3(256), 0, stream, qw_p, scl_p, su_p, sv_p, x_p, nullptr, part_p, cnt_p, out_p, T, (float)cb, 0);
    } else {
      if (g_p) hipLaunchKernelGGL((k_wo_a_k5_batched_big<16, 1, 0>), dim3(8, 8, 8), dim3(256), 0, stream, qw_p, scl_p, su_p, sv_p, x_p, g_p, part_p, cnt_p, out_p, T, (float)cb, (int)spad);
      else     hipLaunchKernelGGL((k_wo_a_k5_batched_big<16, 0, 0>), dim3(8, 8, 8), dim3(256), 0, stream, qw_p, scl_p, su_p, sv_p, x_p, nullptr, part_p, cnt_p, out_p, T, (float)cb, (int)spad);
    }
  }
#undef DISP_M
}

// ============================================================================
// Batched 8-Group K=5 WMMA Prefill + Fused sigmoid(gate) for ALL T > 8!
// Replaces BOTH the T=9..64 chunked GEMV loop (8x weight re-read) AND the
// T>64 8-call Python loop + cat + sigmoid + mul (~400 launches/prefill across 37L)!
// Uses: global_load_b128 + permlane16 V[0] + v_sad_hi_u8 + v_wmma_f32_16x16x16_f16_w32
//       + zero-sync wave_barrier + in-WG H128 FWHT + fused sigmoid(gate).
// ============================================================================
typedef _Float16 h16v __attribute__((ext_vector_type(16)));
typedef float f8v __attribute__((ext_vector_type(8)));
struct u2_128 { uint4 a, b; };

// Prologue: x [T, 8, 4096] bf16 -> xt [T, 8, 4096] fp16, tok [T, 8] fp32
// Grid: (T, 8), Block: 256 (8 waves x 4 blocks of 128 = 4096 exact!)
__device__ __forceinline__ void woa_lds_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

__global__ __launch_bounds__(256) void k_wo_a_prologue(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ su,
    _Float16* __restrict__ xt,
    float* __restrict__ tok,
    int T) {
  __shared__ float s_wmax[8];
  const int r = blockIdx.x;
  const int grp = blockIdx.y;
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const uint16_t* xr = x + ((size_t)r * 8 + grp) * 4096;
  const uint8_t* sup = su + (size_t)grp * 512;

  float vals[4][4];
  float am = 0.0f;
#pragma unroll
  for (int bl = 0; bl < 4; ++bl) {
    const int base = (w * 4 + bl) * 128 + lane * 4;
    const uint2 raw = *reinterpret_cast<const uint2*>(xr + base);
    vals[bl][0] = __uint_as_float((raw.x & 0xFFFFu) << 16);
    vals[bl][1] = __uint_as_float(raw.x & 0xFFFF0000u);
    vals[bl][2] = __uint_as_float((raw.y & 0xFFFFu) << 16);
    vals[bl][3] = __uint_as_float(raw.y & 0xFFFF0000u);
    const unsigned sg4 = (unsigned)sup[base >> 3] >> (base & 7);
    if (sg4 & 1u) vals[bl][0] = -vals[bl][0];
    if (sg4 & 2u) vals[bl][1] = -vals[bl][1];
    if (sg4 & 4u) vals[bl][2] = -vals[bl][2];
    if (sg4 & 8u) vals[bl][3] = -vals[bl][3];
    fwht128_wave(vals[bl], lane);
#pragma unroll
    for (int e2 = 0; e2 < 4; ++e2) {
      vals[bl][e2] *= (1.0f / 11.313708498984761f);
      am = fmaxf(am, fabsf(vals[bl][e2]));
    }
  }
  am = wave_max(am);
  if (lane == 0) s_wmax[w] = am;
  __syncthreads();
  float gmax = s_wmax[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) gmax = fmaxf(gmax, s_wmax[i]);
  const float tscale = fmaxf(gmax, 1e-12f) * (1.0f / 32.0f);
  const float inv = __builtin_amdgcn_rcpf(tscale);
  if (threadIdx.x == 0) tok[(size_t)r * 8 + grp] = tscale;

  // Store xt in [G=8, K/64=64, T, 64]: each half-wave (16 lanes x 8B) writes 1 full 128B cache line!
  const int sub_blk = lane >> 4;
  const int col64 = (lane & 15) * 4;
  _Float16* dst_g = xt + (size_t)grp * 64 * (size_t)T * 64 + (size_t)r * 64 + col64;
#pragma unroll
  for (int bl = 0; bl < 4; ++bl) {
    const int jblk = (w * 4 + bl) * 2 + sub_blk;
    h4v h4 = {(_Float16)(vals[bl][0] * inv), (_Float16)(vals[bl][1] * inv),
              (_Float16)(vals[bl][2] * inv), (_Float16)(vals[bl][3] * inv)};
    *reinterpret_cast<h4v*>(dst_g + (size_t)jblk * (size_t)T * 64) = h4;
  }
}

// Per-group 16B+4B split-plane repack: [8, 64, 256, 40] -> same-size [8, (p16 | p4)] (0 extra VRAM!)
__global__ void k_repack_wo_a_sp16(
    const uint32_t* __restrict__ src_qw,
    uint32_t* __restrict__ dst_qw,
    int64_t ntiles_per_grp) {
  const int grp = blockIdx.y;
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ntiles_per_grp * 8) return;
  const int64_t t = idx >> 3;
  const int q = (int)(idx & 7);
  const uint32_t* s_grp = src_qw + (size_t)grp * ntiles_per_grp * 40;
  uint32_t* d_grp = dst_qw + (size_t)grp * ntiles_per_grp * 40;
  const uint32_t* src = s_grp + t * 40 + q * 5;
  const u4v v = *reinterpret_cast<const u4va*>(src);
  *reinterpret_cast<u4v*>(d_grp + t * 32 + q * 4) = v;
  d_grp[ntiles_per_grp * 32 + t * 8 + q] = src[4];
}

torch::Tensor repack_wo_a_sp16(torch::Tensor qw) {
  TORCH_CHECK(qw.is_cuda() && qw.is_contiguous() && qw.dim() == 4 &&
              qw.size(0) == 8 && qw.size(1) == 64 && qw.size(2) == 256 && qw.size(3) == 40 &&
              qw.scalar_type() == torch::kInt32);
  auto out = torch::empty_like(qw);
  const int64_t ntiles_per_grp = 64 * 256;
  const int64_t nthreads = ntiles_per_grp * 8;
  hipLaunchKernelGGL(
      k_repack_wo_a_sp16, dim3((unsigned)((nthreads + 255) / 256), 8), dim3(256), 0,
      c10::hip::getCurrentHIPStream().stream(),
      reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>()),
      reinterpret_cast<uint32_t*>(out.data_ptr<int32_t>()), ntiles_per_grp);
  return out;
}

// Grid: (8 /*sb*/, ceil(T / (NB*16)), 8 /*grp*/), Block: 256 (8 waves)
template <int NB, int HAS_GATE, int SP16>
__global__ __launch_bounds__(256) void k_wo_a_k5_prefill_wmma(
    const uint32_t* __restrict__ qw,       // [8, 64, 256, 40] (or per-group sp16)
    const _Float16* __restrict__ xt,       // [T, 8, 4096]
    const float* __restrict__ tok,         // [T, 8]
    const __half* __restrict__ scales,     // [8, 1024]
    const uint8_t* __restrict__ sv,        // [8, 128]
    const uint16_t* __restrict__ gate,     // [T, 8192] bf16 (optional)
    uint16_t* __restrict__ out,            // [T, 8192] bf16
    int T, float cb, int spad) {
  constexpr int K = 5;
  constexpr int G = 8;
  constexpr int M_OUT = 1024;
  constexpr int N_IN = 4096;
  constexpr int MT = 64;
  constexpr int NT = 256;

  union SharedMem {
    struct {
      uint32_t w_lds[8][4][128];     // 16,384 B
      uint32_t x_lds[4][NB * 16][8]; //  2..16 KiB (cooperative xt staging -> 8x fewer global xt loads + zero vmcnt conflict!)
    } stage;
    float    o_lds[16][128];         //  8,192 B (reused in epilogue!)
  };
  __shared__ __attribute__((aligned(16))) SharedMem smem;

  const int tid = threadIdx.x;
  const int w = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 3;
  const int q = lane & 7;
  const int l16 = lane & 15;
  const int hb = lane >> 4;

  const int sb = blockIdx.x;               // 0..7 (128-row output block)
  const int m0 = blockIdx.y * (NB * 16);   // token block start
  const int grp = blockIdx.z;              // 0..7 group
  const int il = sb * 8 + w;               // 16-row strip owned by wave w

  const size_t pitch = (size_t)NT * 40 + spad;
  const uint32_t* qm = SP16 ? nullptr : (qw + (size_t)grp * MT * pitch + (size_t)il * pitch);
  const uint32_t* p16_grp = SP16 ? (qw + (size_t)grp * (MT * NT * 40)) : nullptr;
  const uint32_t* p4_grp  = SP16 ? (p16_grp + (size_t)(MT * NT) * 32) : nullptr;
  const __half* scl = scales + (size_t)grp * M_OUT;
  const uint8_t* svp = sv + (size_t)grp * (M_OUT / 8);

  f8v acc[NB];
#pragma unroll
  for (int c = 0; c < NB; ++c) acc[c] = f8v{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  // 2-stage register software prefetch of K=5 weights (5 VGPRs, 0 extra LDS):
  // Overlaps 100% of LPDDR5X weight fetch latency with all 4*NB LDS-fed WMMAs!
  u4v wa_cur = {0u, 0u, 0u, 0u};
  uint32_t w5_cur = 0u;
  if constexpr (SP16) {
    const size_t tid0 = (size_t)il * NT + g;
    wa_cur = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(p16_grp + tid0 * 32 + q * 4));
    w5_cur = __builtin_nontemporal_load(p4_grp + tid0 * 8 + q);
  } else {
    const uint32_t* tp0 = qm + (size_t)g * 40;
    wa_cur = *reinterpret_cast<const u4va*>(tp0 + q * 5);
    w5_cur = tp0[q * 5 + 4];
  }

  for (int j = 0; j < NT; j += 4) {
    // 1) Cooperatively stage xt[m0..m0+NB*16-1, grp, j*16..(j+4)*16-1] into x_lds (burst global_load_b128 -> ds_store_b128!)
    constexpr int X_U4_TOTAL = NB * 128;
    constexpr int X_STEPS = (X_U4_TOTAL + 255) / 256;
    uint4 x_tmp[X_STEPS];
#pragma unroll
    for (int s = 0; s < X_STEPS; ++s) {
      const int idx = s * 256 + tid;
      x_tmp[s] = make_uint4(0u, 0u, 0u, 0u);
      if (idx < X_U4_TOTAL) {
        const int half_k = idx & 1;
        const int t_ld = (idx >> 1) & 3;
        const int row_l = idx >> 3;
        const int rr = m0 + row_l;
        if (rr < T) {
          x_tmp[s] = *reinterpret_cast<const uint4*>(xt + (((size_t)grp * 64 + (size_t)(j >> 2)) * (size_t)T + (size_t)rr) * 64 + t_ld * 16 + half_k * 8);
        }
      }
    }
#pragma unroll
    for (int s = 0; s < X_STEPS; ++s) {
      const int idx = s * 256 + tid;
      if (idx < X_U4_TOTAL) {
        const int half_k = idx & 1;
        const int t_ld = (idx >> 1) & 3;
        const int row_l = idx >> 3;
        const int hk_swz = half_k ^ ((row_l >> 2) & 1);
        *reinterpret_cast<uint4*>(&smem.stage.x_lds[t_ld][row_l][hk_swz * 4]) = x_tmp[s];
      }
    }

    // 2) Decode prefetched K=5 weights into w_lds
    uint32_t* L = smem.stage.w_lds[w][g];
    uint32_t V[6];
    V[0] = __builtin_amdgcn_permlane16(w5_cur, w5_cur, 0x65432107u, 0xedcba98fu, false, false);
    V[1] = wa_cur.x; V[2] = wa_cur.y; V[3] = wa_cur.z; V[4] = wa_cur.w; V[5] = w5_cur;
#define PW_W(u) { \
      uint32_t bits = val_f16bits2(ext16<16 + (2 * (u) + 1) * K>(V), ext16<16 + (2 * (u) + 2) * K>(V)); \
      h2v centered = as_h2(bits) - h2v{(_Float16)1534.0f, (_Float16)1534.0f}; \
      L[((u) < 8 ? 2 * q : 2 * q + 1) * 8 + ((u) & 7)] = __builtin_bit_cast(uint32_t, centered); }
    PW_W(0) PW_W(1) PW_W(2) PW_W(3) PW_W(4) PW_W(5) PW_W(6) PW_W(7)
    PW_W(8) PW_W(9) PW_W(10) PW_W(11) PW_W(12) PW_W(13) PW_W(14) PW_W(15)
#undef PW_W
    __syncthreads();

    // 3) Prefetch next K=5 tile (j + 4) BEFORE all 4*NB LDS-only WMMAs (zero vmcnt conflict!)
    u4v wa_nxt = {0u, 0u, 0u, 0u};
    uint32_t w5_nxt = 0u;
    if (j + 4 < NT) {
      const int jjn = j + 4 + g;
      if constexpr (SP16) {
        const size_t tidn = (size_t)il * NT + jjn;
        wa_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(p16_grp + tidn * 32 + q * 4));
        w5_nxt = __builtin_nontemporal_load(p4_grp + tidn * 8 + q);
      } else {
        const uint32_t* tpn = qm + (size_t)jjn * 40;
        wa_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tpn + q * 5));
        w5_nxt = __builtin_nontemporal_load(tpn + q * 5 + 4);
      }
    }
    __builtin_amdgcn_sched_barrier(0);

    // 4) Pure LDS-fed WMMA compute (0 global loads, 0 bounds checks!) while wa_nxt/w5_nxt stream in!
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const uint4* ap = reinterpret_cast<const uint4*>(smem.stage.w_lds[w][t] + l16 * 8);
      u2_128 au{ap[0], ap[1]};
      const h16v a = __builtin_bit_cast(h16v, au);
#pragma unroll
      for (int c = 0; c < NB; ++c) {
        const int rl = c * 16 + l16;
        const int h0 = (rl >> 2) & 1;
        const uint4* bp = reinterpret_cast<const uint4*>(smem.stage.x_lds[t][rl]);
        u2_128 bu{bp[h0], bp[h0 ^ 1]};
        acc[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, __builtin_bit_cast(h16v, bu), acc[c]);
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();
    wa_cur = wa_nxt;
    w5_cur = w5_nxt;
  }

  // In-WG fused H128 FWHT + scales + sv + sigmoid(gate) epilogue (8B coalesced R/W)
  const int base_col = sb * 128 + lane * 4;
  float sc[4];
#pragma unroll
  for (int e2 = 0; e2 < 4; ++e2) sc[e2] = __half2float(scl[base_col + e2]) * cb;
  const unsigned sg4 = (unsigned)svp[sb * 16 + (lane >> 1)] >> ((lane & 1) * 4);

#pragma unroll
  for (int c = 0; c < NB; ++c) {
    if (m0 + c * 16 >= T) break;
    __syncthreads();
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      smem.o_lds[l16][w * 16 + 2 * v + hb] = acc[c][v];
    }
    __syncthreads();

#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      const int rt = w + sub * 8;
      const int r = m0 + c * 16 + rt;
      if (r < T) {
        const float tt = tok[(size_t)r * G + grp];
        float v[4];
#pragma unroll
        for (int e2 = 0; e2 < 4; ++e2) v[e2] = smem.o_lds[rt][lane * 4 + e2] * sc[e2] * tt;
        fwht128_wave(v, lane);
        float o[4] = {
          v[0] * (1.0f / 11.313708498984761f),
          v[1] * (1.0f / 11.313708498984761f),
          v[2] * (1.0f / 11.313708498984761f),
          v[3] * (1.0f / 11.313708498984761f)
        };
        if (sg4 & 1) o[0] = -o[0];
        if (sg4 & 2) o[1] = -o[1];
        if (sg4 & 4) o[2] = -o[2];
        if (sg4 & 8) o[3] = -o[3];
        const size_t out_off = (size_t)r * (G * M_OUT) + (size_t)grp * M_OUT + base_col;
        uint2 out_p;
        if constexpr (HAS_GATE) {
          const uint2 graw = *reinterpret_cast<const uint2*>(gate + out_off);
          const float gfv[4] = {
            __uint_as_float((graw.x & 0xFFFFu) << 16),
            __uint_as_float(graw.x & 0xFFFF0000u),
            __uint_as_float((graw.y & 0xFFFFu) << 16),
            __uint_as_float(graw.y & 0xFFFF0000u),
          };
          float prod[4];
#pragma unroll
          for (int e2 = 0; e2 < 4; ++e2) {
            const float sig = __builtin_amdgcn_rcpf(1.0f + exp2f(-1.4426950408889634f * gfv[e2]));
            const float zb = bf2f(f2bf(o[e2]));
            const float gb = bf2f(f2bf(sig));
            prod[e2] = zb * gb;
          }
          out_p.x = f32_to_bf16x2_rne(prod[0], prod[1]);
          out_p.y = f32_to_bf16x2_rne(prod[2], prod[3]);
        } else {
          out_p.x = f32_to_bf16x2_rne(o[0], o[1]);
          out_p.y = f32_to_bf16x2_rne(o[2], o[3]);
        }
        *reinterpret_cast<uint2*>(out + out_off) = out_p;
      }
    }
  }
}


// ============================================================================
// Wave 9C: 4x4 Outer-Product WMMA + PAD=12 + Single-Reg v_alignbit + lds_barrier()
//   + [G=8, K/64=64, T, 64] xt layout (128B sequential token stride -> 32 KiB/jblk
//     fits 100% in GL0/GL2 across all 8 sb WGs, killing 64 KiB channel collision!)
// Grid: dim3(8 /*sb*/, (T + 255) / 256, 8 /*grp*/), Block: 256 (8 waves)
// ============================================================================
__device__ __forceinline__ int woa_sw_col4(int row, int col4) {
  return (col4 ^ ((row >> 2) & 1)) << 2;
}

// Wave 51: 24 KiB XOR-swizzled STR=8 (4 WGs/WGP = 2 WGs/CU, 16w/CU, V<=192, vSp=0!)
//          + ds_store_b64 + compile-time ping-pong j+=2 + b0,b1/1-a snake reuse (11 LDS loads vs 12)
//          + wave-uniform SGPR bases for w & xt!
template <int HAS_GATE, int SP16>
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8, 8)))
void k_wo_a_k5_prefill_wmma_4x4(
    const uint32_t* __restrict__ qw,       // [8, 64, 256, 40] (or per-group sp16)
    const _Float16* __restrict__ xt,       // [8, 64, T, 64]
    const float* __restrict__ tok,         // [T, 8]
    const __half* __restrict__ scales,     // [8, 1024]
    const uint8_t* __restrict__ sv,        // [8, 128]
    const uint16_t* __restrict__ gate,     // [T, 8192] or nullptr
    uint16_t* __restrict__ out,            // [T, 8192]
    int T, float cb, int spad) {
  constexpr int G = 8;
  constexpr int M_OUT = 1024;
  constexpr int K_DIM = 4096;
  constexpr int MT = M_OUT / 16;           // 64
  constexpr int NT = K_DIM / 16;           // 256
  constexpr int STR = 8;                   // 32 B row stride + XOR swizzle -> 24,576 B total = 4 WGs/WGP!

  union SharedMem {
    struct {
      uint32_t w_lds[2][128][STR];         //  8,192 B
      uint32_t x_lds[2][256][STR];         // 16,384 B
    } stage;                               // 24,576 B TOTAL -> 4 WGs/WGP = 2 WGs/CU!
    float o_lds[32][128];                  // 16,384 B (reused in epilogue)
  };
  __shared__ __attribute__((aligned(16))) SharedMem smem;

  const int tid = threadIdx.x;             // 0..255
  const int sb  = blockIdx.x;              // 0..7
  const int m0  = ((int)gridDim.y - 1 - (int)blockIdx.y) * 256;
  const int grp = blockIdx.z;              // 0..7

  f8v acc[4][4];
#pragma unroll
  for (int ni = 0; ni < 4; ++ni)
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
      acc[ni][mi] = f8v{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  // Wave-uniform SGPR bases for w (0 VGPRs!) and 32-bit offsets for xt!
  const int i_dec_s = __builtin_amdgcn_readfirstlane(sb * 8 + (tid >> 5));
  const size_t w_row_base = (size_t)i_dec_s * NT;
  const uint32_t* w_base0 = SP16
      ? (qw + (size_t)grp * ((size_t)MT * NT * 40))
      : (qw + (size_t)grp * ((size_t)MT * ((size_t)NT * 40 + (size_t)spad)) + (size_t)i_dec_s * ((size_t)NT * 40 + (size_t)spad));
  const uint32_t* p4_grp = qw + (size_t)grp * ((size_t)MT * NT * 40) + (size_t)MT * NT * 32;

  const int rr_0 = m0 + (tid >> 1);
  const int rr_1 = m0 + 128 + (tid >> 1);
  const bool v_rr0 = (rr_0 < T);
  const bool v_rr1 = (rr_1 < T);
  const size_t jblk_stride = (size_t)T * 64;
  const _Float16* xt_g0 = xt + (size_t)grp * 64 * jblk_stride;
  const uint32_t x_off0 = (uint32_t)((v_rr0 ? rr_0 : (T - 1)) * 64 + (tid & 1) * 8);
  const uint32_t x_off1 = (uint32_t)((v_rr1 ? rr_1 : (T - 1)) * 64 + (tid & 1) * 8);

  auto load_w = [&](int j_idx, u4v& w4, uint32_t& wl) {
    const int q = tid & 7;
    if constexpr (SP16) {
      const size_t tid0 = w_row_base + j_idx;
      w4 = *reinterpret_cast<const u4v*>(w_base0 + tid0 * 32 + q * 4);
      wl = p4_grp[tid0 * 8 + q];
    } else {
      const uint32_t* tp0 = w_base0 + (size_t)j_idx * 40 + q * 5;
      w4 = *reinterpret_cast<const u4va*>(tp0);
      wl = tp0[4];
    }
  };
  auto load_x = [&](int j_idx, uint4 x_reg[2]) {
    const _Float16* jb_ptr = xt_g0 + (size_t)(j_idx >> 2) * jblk_stride + (j_idx & 3) * 16;
    uint4 r0 = *reinterpret_cast<const uint4*>(jb_ptr + x_off0);
    uint4 r1 = *reinterpret_cast<const uint4*>(jb_ptr + x_off1);
    x_reg[0] = v_rr0 ? r0 : make_uint4(0u, 0u, 0u, 0u);
    x_reg[1] = v_rr1 ? r1 : make_uint4(0u, 0u, 0u, 0u);
  };
  auto decode_store = [&](int buf, const u4v& w4, uint32_t wl, const uint4 x_reg[2]) {
    const int x_rl0 = tid >> 1;
    const int x_rl1 = 128 + (tid >> 1);
    *reinterpret_cast<uint4*>(&smem.stage.x_lds[buf][x_rl0][woa_sw_col4(x_rl0, tid & 1)]) = x_reg[0];
    *reinterpret_cast<uint4*>(&smem.stage.x_lds[buf][x_rl1][woa_sw_col4(x_rl1, tid & 1)]) = x_reg[1];

    const int w = tid >> 5, lane = tid & 31, q = lane & 7, g = lane >> 3;
    const uint32_t sh_e = (uint32_t)((g * 12 + 11) & 31);
    const uint32_t sh_o = (uint32_t)((g * 12 + 27) & 31);
    const uint32_t v0 = __builtin_amdgcn_permlane16(wl, wl, 0x65432107u, 0xedcba98fu, false, false);
    const uint32_t v1 = w4.x, v2 = w4.y, v3 = w4.z, v4 = w4.w, v5 = wl;
    const uint32_t ea = (g == 0) ? v0 : ((g == 3) ? v2 : v1);
    const uint32_t eb = (g == 0) ? v1 : ((g == 3) ? v3 : v2);
    const uint32_t oa = (g < 2)  ? v3 : ((g == 2) ? v4 : v5);
    const uint32_t ob = (g < 2)  ? v4 : ((g == 2) ? v5 : 0u);
    const uint32_t ehi = __builtin_amdgcn_alignbit(ea, eb, sh_e);
    const uint32_t ohi = __builtin_amdgcn_alignbit(oa, ob, sh_o);
    const h2v bias2 = h2v{(_Float16)1534.0f, (_Float16)1534.0f};
    h2v e0 = as_h2(val_f16bits2((ehi >> 16) & 0xFFFFu, (ehi >> 11) & 0xFFFFu)) - bias2;
    h2v e1 = as_h2(val_f16bits2((ehi >>  6) & 0xFFFFu, (ehi >>  1) & 0xFFFFu)) - bias2;
    h2v o0 = as_h2(val_f16bits2((ohi >> 16) & 0xFFFFu, (ohi >> 11) & 0xFFFFu)) - bias2;
    h2v o1 = as_h2(val_f16bits2((ohi >>  6) & 0xFFFFu, (ohi >>  1) & 0xFFFFu)) - bias2;
    const int r_even = w * 16 + 2 * q;
    const int col2_sw = (g * 2) ^ (((r_even >> 2) & 1) << 2);
    *reinterpret_cast<uint2*>(&smem.stage.w_lds[buf][r_even][col2_sw]) =
        make_uint2(__builtin_bit_cast(uint32_t, e0), __builtin_bit_cast(uint32_t, e1));
    *reinterpret_cast<uint2*>(&smem.stage.w_lds[buf][r_even + 1][col2_sw]) =
        make_uint2(__builtin_bit_cast(uint32_t, o0), __builtin_bit_cast(uint32_t, o1));
  };

  {
    u4v w4_0; uint32_t wl_0; uint4 x_0[2];
    load_w(0, w4_0, wl_0);
    load_x(0, x_0);
    decode_store(0, w4_0, wl_0, x_0);
    woa_lds_barrier();
  }

  auto ld_a = [&](int buf, int ni) -> h16v {
    const int l16 = tid & 15, wn = (tid >> 5) & 1;
    const uint32_t* p = &smem.stage.w_lds[buf][wn * 64 + ni * 16 + l16][0];
    const int c0 = woa_sw_col4(l16, 0), c1 = woa_sw_col4(l16, 1);
    return __builtin_bit_cast(h16v, u2_128{*reinterpret_cast<const uint4*>(p + c0), *reinterpret_cast<const uint4*>(p + c1)});
  };
  auto ld_b = [&](int buf, int mi) -> h16v {
    const int l16 = tid & 15, wm = tid >> 6;
    const uint32_t* p = &smem.stage.x_lds[buf][wm * 64 + mi * 16 + l16][0];
    const int c0 = woa_sw_col4(l16, 0), c1 = woa_sw_col4(l16, 1);
    return __builtin_bit_cast(h16v, u2_128{*reinterpret_cast<const uint4*>(p + c0), *reinterpret_cast<const uint4*>(p + c1)});
  };

  auto step = [&](int buf, int j_nxt, bool has_nxt) {
    u4v w4_n; uint32_t wl_n; uint4 x_n[2];
    if (has_nxt) {
      load_w(j_nxt, w4_n, wl_n);
      load_x(j_nxt, x_n);
    }
    h16v b0 = ld_b(buf, 0), b1 = ld_b(buf, 1);
    h16v at;
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      at = ld_a(buf, ni);
      acc[ni][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(at, b0, acc[ni][0]);
      acc[ni][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(at, b1, acc[ni][1]);
      __builtin_amdgcn_sched_barrier(0);
    }
    b0 = ld_b(buf, 2); b1 = ld_b(buf, 3);
    acc[3][2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(at, b0, acc[3][2]);
    acc[3][3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(at, b1, acc[3][3]);
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int ni = 2; ni >= 0; --ni) {
      at = ld_a(buf, ni);
      acc[ni][2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(at, b0, acc[ni][2]);
      acc[ni][3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(at, b1, acc[ni][3]);
      __builtin_amdgcn_sched_barrier(0);
    }
    if (has_nxt) {
      decode_store(buf ^ 1, w4_n, wl_n, x_n);
      woa_lds_barrier();
    }
  };

#pragma unroll 1
  for (int j = 0; j < NT - 2; j += 2) {
    step(0, j + 1, true);
    step(1, j + 2, true);
  }
  step(0, NT - 1, true);
  step(1, 0, false);

  // Epilogue: 32 rows at a time in o_lds[32][128], bit-exact FWHT-128 + fused sigmoid(gate)
  const int w = tid >> 5, wn = w & 1, wm = w >> 1;
  const int lane = tid & 31, l16 = lane & 15, hb = lane >> 4;
  const __half* scl = scales + (size_t)grp * M_OUT;
  const uint8_t* svp = sv + (size_t)grp * (M_OUT / 8);
  const int base_col = sb * 128 + lane * 4;
  float sc[4];
#pragma unroll
  for (int e2 = 0; e2 < 4; ++e2) sc[e2] = __half2float(scl[base_col + e2]) * cb;
  const unsigned sg4 = (unsigned)svp[sb * 16 + (lane >> 1)] >> ((lane & 1) * 4);
  const int c_base = wn * 64;

#pragma unroll 1
  for (int wm_pair = 0; wm_pair < 2; ++wm_pair) {
    const int wm_lo = wm_pair * 2;
#pragma unroll 1
    for (int mi = 0; mi < 4; ++mi) {
      woa_lds_barrier();
      if (wm == wm_lo || wm == wm_lo + 1) {
        const int o_row = (wm - wm_lo) * 16 + l16;
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          const int col_n = c_base + ni * 16;
          const f8v cur = (mi == 0) ? acc[ni][0]
                        : (mi == 1) ? acc[ni][1]
                        : (mi == 2) ? acc[ni][2]
                                    : acc[ni][3];
#pragma unroll
          for (int v = 0; v < 8; ++v) {
            smem.o_lds[o_row][col_n + 2 * v + hb] = cur[v];
          }
        }
      }
      woa_lds_barrier();

#pragma unroll
      for (int sub = 0; sub < 4; ++sub) {
        const int rt = w + sub * 8;
        const int r = m0 + (wm_lo + (rt >= 16 ? 1 : 0)) * 64 + mi * 16 + (rt & 15);
        if (r < T) {
          const float tt = tok[(size_t)r * G + grp];
          float v[4];
#pragma unroll
          for (int e2 = 0; e2 < 4; ++e2) v[e2] = smem.o_lds[rt][lane * 4 + e2] * sc[e2] * tt;
          fwht128_wave(v, lane);
          float o[4] = {
            v[0] * (1.0f / 11.313708498984761f),
            v[1] * (1.0f / 11.313708498984761f),
            v[2] * (1.0f / 11.313708498984761f),
            v[3] * (1.0f / 11.313708498984761f)
          };
          if (sg4 & 1) o[0] = -o[0];
          if (sg4 & 2) o[1] = -o[1];
          if (sg4 & 4) o[2] = -o[2];
          if (sg4 & 8) o[3] = -o[3];
          const size_t out_off = (size_t)r * (G * M_OUT) + (size_t)grp * M_OUT + base_col;
          uint2 out_p;
          if constexpr (HAS_GATE) {
            const uint2 graw = *reinterpret_cast<const uint2*>(gate + out_off);
            const float gfv[4] = {
              __uint_as_float((graw.x & 0xFFFFu) << 16),
              __uint_as_float(graw.x & 0xFFFF0000u),
              __uint_as_float((graw.y & 0xFFFFu) << 16),
              __uint_as_float(graw.y & 0xFFFF0000u),
            };
            float prod[4];
#pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) {
              const float sig = __builtin_amdgcn_rcpf(1.0f + exp2f(-1.4426950408889634f * gfv[e2]));
              const float zb = bf2f(f2bf(o[e2]));
              const float gb = bf2f(f2bf(sig));
              prod[e2] = zb * gb;
            }
            out_p = make_uint2((uint32_t)f2bf(prod[0]) | ((uint32_t)f2bf(prod[1]) << 16), (uint32_t)f2bf(prod[2]) | ((uint32_t)f2bf(prod[3]) << 16));
          } else {
            out_p = make_uint2((uint32_t)f2bf(o[0]) | ((uint32_t)f2bf(o[1]) << 16), (uint32_t)f2bf(o[2]) | ((uint32_t)f2bf(o[3]) << 16));
          }
          *reinterpret_cast<uint2*>(out + out_off) = out_p;
        }
      }
    }
  }
}

template <int NB, int SP16>
static void dispatch_prefill_nb(
    dim3 grid, hipStream_t stream,
    const uint32_t* qw_p, const _Float16* xt_p, const float* tok_p,
    const __half* scl_p, const uint8_t* sv_p, const uint16_t* g_p,
    uint16_t* out_p, int T, float cb, int spad) {
  if (g_p) hipLaunchKernelGGL((k_wo_a_k5_prefill_wmma<NB, 1, SP16>), grid, dim3(256), 0, stream, qw_p, xt_p, tok_p, scl_p, sv_p, g_p, out_p, T, cb, spad);
  else     hipLaunchKernelGGL((k_wo_a_k5_prefill_wmma<NB, 0, SP16>), grid, dim3(256), 0, stream, qw_p, xt_p, tok_p, scl_p, sv_p, nullptr, out_p, T, cb, spad);
}

void launch_wo_a_k5_prefill_wmma(
    torch::Tensor qw, torch::Tensor scales, torch::Tensor su, torch::Tensor sv,
    torch::Tensor x, c10::optional<torch::Tensor> gate,
    torch::Tensor xt_buf, torch::Tensor tok_buf, torch::Tensor out,
    double cb, int64_t spad, bool sp16 = false) {
  const int T = (int)x.size(0);
  if (T == 0) return;
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 3 && x.size(1) == 8 && x.size(2) == 4096);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  const uint32_t* qw_p = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>());
  const __half* scl_p = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  const uint8_t* su_p = su.data_ptr<uint8_t>();
  const uint8_t* sv_p = sv.data_ptr<uint8_t>();
  const uint16_t* x_p = reinterpret_cast<const uint16_t*>(x.data_ptr());
  const uint16_t* g_p = gate.has_value() ? reinterpret_cast<const uint16_t*>(gate->data_ptr()) : nullptr;
  _Float16* xt_p = reinterpret_cast<_Float16*>(xt_buf.data_ptr<at::Half>());
  float* tok_p = tok_buf.data_ptr<float>();
  uint16_t* out_p = reinterpret_cast<uint16_t*>(out.data_ptr());

  hipLaunchKernelGGL(k_wo_a_prologue, dim3(T, 8), dim3(256), 0, stream, x_p, su_p, xt_p, tok_p, T);
  if (T >= 256) {
    dim3 grid(8, (T + 255) / 256, 8);
    if (sp16) {
      if (g_p) hipLaunchKernelGGL((k_wo_a_k5_prefill_wmma_4x4<1, 1>), grid, dim3(256), 0, stream, qw_p, xt_p, tok_p, scl_p, sv_p, g_p, out_p, T, (float)cb, 0);
      else     hipLaunchKernelGGL((k_wo_a_k5_prefill_wmma_4x4<0, 1>), grid, dim3(256), 0, stream, qw_p, xt_p, tok_p, scl_p, sv_p, nullptr, out_p, T, (float)cb, 0);
    } else {
      if (g_p) hipLaunchKernelGGL((k_wo_a_k5_prefill_wmma_4x4<1, 0>), grid, dim3(256), 0, stream, qw_p, xt_p, tok_p, scl_p, sv_p, g_p, out_p, T, (float)cb, (int)spad);
      else     hipLaunchKernelGGL((k_wo_a_k5_prefill_wmma_4x4<0, 0>), grid, dim3(256), 0, stream, qw_p, xt_p, tok_p, scl_p, sv_p, nullptr, out_p, T, (float)cb, (int)spad);
    }
  } else if (T <= 32) {
    dim3 grid(8, (T + 31) / 32, 8);
    if (sp16) dispatch_prefill_nb<2, 1>(grid, stream, qw_p, xt_p, tok_p, scl_p, sv_p, g_p, out_p, T, (float)cb, 0);
    else      dispatch_prefill_nb<2, 0>(grid, stream, qw_p, xt_p, tok_p, scl_p, sv_p, g_p, out_p, T, (float)cb, (int)spad);
  } else {
    dim3 grid(8, (T + 63) / 64, 8);
    if (sp16) dispatch_prefill_nb<4, 1>(grid, stream, qw_p, xt_p, tok_p, scl_p, sv_p, g_p, out_p, T, (float)cb, 0);
    else      dispatch_prefill_nb<4, 0>(grid, stream, qw_p, xt_p, tok_p, scl_p, sv_p, g_p, out_p, T, (float)cb, (int)spad);
  }
}




// ============================================================================
// Wave 10A: Fused VHA [64,64] @ O [T,64,512] + su + FWHT-128 + [8,64,T,64] xt prologue
//   - Eliminates 134.2 MiB intermediate BF16 tensor (268.4 MiB DRAM R+W/layer x 37L = -9.93 GB!)
//   - Grid: dim3(4 /*gp=0..3, 16 heads = 2 groups/WG*/, T), Block: 256 (8 waves)
//   - All 4 gp WGs for token t share O[t, 64, 512] (64 KiB) in 4 MiB GL2 -> read ONCE from DRAM!
//   - 100% full 16x16x16 BF16 WMMA utilization, 20.3 KiB LDS -> 2 WGs/CU = 16 waves/CU!
// ============================================================================
typedef short vha_bf16x16 __attribute__((ext_vector_type(16)));
typedef uint32_t vha_u8v __attribute__((ext_vector_type(8)));

__global__ __launch_bounds__(256) void k_vha_wo_a_prologue(
    const uint16_t* __restrict__ M_ptr,    // [64, 64] bf16 contiguous
    const uint16_t* __restrict__ o_ptr,    // [T, >=64, 512] bf16 (stride o_s0, o_s1, 1)
    const uint8_t* __restrict__ su,        // [8, 512]
    _Float16* __restrict__ xt,             // [8, 64, T, 64] fp16
    float* __restrict__ tok,               // [T, 8] fp32
    int T, int64_t o_s0, int64_t o_s1) {
  __shared__ __attribute__((aligned(16))) uint16_t M_lds[16][72]; // 2,304 B (0 bank conflicts!)
  __shared__ __attribute__((aligned(16))) union {
    uint16_t O_lds[512][18];                                      // 18,432 B
    uint16_t Y_lds[16][512];                                      // 16,384 B (reused!)
  } smem;
  __shared__ float s_wmax[8];

  const int gp = blockIdx.x;               // 0..3 (heads gp*16 .. gp*16+15, groups 2*gp, 2*gp+1)
  const int t  = blockIdx.y;               // 0..T-1
  const int tid = threadIdx.x;             // 0..255
  const int w = tid >> 5;                  // 0..7 (each wave owns 64 cols = 4 N-tiles)
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hb = lane >> 4;
  const int h0 = gp * 16;

  // 1. Load M[h0..h0+15, 0..63] into M_lds (128 threads x 8 bf16 = 1024 exact!)
  if (tid < 128) {
    const int mr = tid >> 3;
    const int mc = (tid & 7) * 8;
    *reinterpret_cast<u4va*>(&M_lds[mr][mc]) = *reinterpret_cast<const u4va*>(M_ptr + (h0 + mr) * 64 + mc);
  }

  // 2. Stream O[t, 0..63, 0..511] in 4 K-steps of 16x512 with 2-stage software prefetch
  const int kr_ld = tid >> 4;              // 0..15
  const int sub_ld = tid & 15;             // 0..15
  const uint16_t* o_tok = o_ptr + (int64_t)t * o_s0;

  f8v acc[4];
#pragma unroll
  for (int ni = 0; ni < 4; ++ni) acc[ni] = f8v{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  u4v o_cur[4];
  {
    const uint16_t* src0 = o_tok + (int64_t)kr_ld * o_s1 + sub_ld * 8;
#pragma unroll
    for (int c = 0; c < 4; ++c) o_cur[c] = *reinterpret_cast<const u4va*>(src0 + c * 128);
  }

#pragma unroll
  for (int k = 0; k < 4; ++k) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int col0 = c * 128 + sub_ld * 8;
      const uint32_t w0 = o_cur[c].x, w1 = o_cur[c].y, w2 = o_cur[c].z, w3 = o_cur[c].w;
      smem.O_lds[col0 + 0][kr_ld] = (uint16_t)(w0 & 0xFFFFu);
      smem.O_lds[col0 + 1][kr_ld] = (uint16_t)(w0 >> 16);
      smem.O_lds[col0 + 2][kr_ld] = (uint16_t)(w1 & 0xFFFFu);
      smem.O_lds[col0 + 3][kr_ld] = (uint16_t)(w1 >> 16);
      smem.O_lds[col0 + 4][kr_ld] = (uint16_t)(w2 & 0xFFFFu);
      smem.O_lds[col0 + 5][kr_ld] = (uint16_t)(w2 >> 16);
      smem.O_lds[col0 + 6][kr_ld] = (uint16_t)(w3 & 0xFFFFu);
      smem.O_lds[col0 + 7][kr_ld] = (uint16_t)(w3 >> 16);
    }
    woa_lds_barrier();

    u4v o_nxt[4];
    if (k < 3) {
      const uint16_t* srcn = o_tok + (int64_t)((k + 1) * 16 + kr_ld) * o_s1 + sub_ld * 8;
#pragma unroll
      for (int c = 0; c < 4; ++c) o_nxt[c] = *reinterpret_cast<const u4va*>(srcn + c * 128);
    }
    __builtin_amdgcn_sched_barrier(0);

    const u4v a0 = *reinterpret_cast<const u4va*>(&M_lds[l16][k * 16 + 0]);
    const u4v a1 = *reinterpret_cast<const u4va*>(&M_lds[l16][k * 16 + 8]);
    const vha_bf16x16 af = __builtin_bit_cast(vha_bf16x16, (vha_u8v){a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w});
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int col_l = w * 64 + ni * 16 + l16;
      const u4v b0 = *reinterpret_cast<const u4va*>(&smem.O_lds[col_l][0]);
      const u4v b1 = *reinterpret_cast<const u4va*>(&smem.O_lds[col_l][8]);
      const vha_bf16x16 bf = __builtin_bit_cast(vha_bf16x16, (vha_u8v){b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w});
      acc[ni] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af, bf, acc[ni]);
    }
    woa_lds_barrier();
    if (k < 3) {
#pragma unroll
      for (int c = 0; c < 4; ++c) o_cur[c] = o_nxt[c];
    }
  }

  // 3. Round VHA result to BF16 in dead smem.Y_lds[16][512] (matches torch.matmul BF16 output!)
#pragma unroll
  for (int ni = 0; ni < 4; ++ni) {
    const int col = w * 64 + ni * 16 + l16;
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      smem.Y_lds[2 * ei + hb][col] = f2bf(acc[ni][ei]);
    }
  }
  woa_lds_barrier();

  // 4. Run su + FWHT-128 + absmax + [8, 64, T, 64] coalesced 128B store directly from LDS for both groups!
  const int sub_blk = lane >> 4;
  const int col64 = (lane & 15) * 4;
#pragma unroll 1
  for (int gi = 0; gi < 2; ++gi) {
    const int grp = 2 * gp + gi;
    const uint16_t* xr = &smem.Y_lds[gi * 8][0];
    const uint8_t* sup = su + (size_t)grp * 512;
    float vals[4][4];
    float am = 0.0f;
#pragma unroll
    for (int bl = 0; bl < 4; ++bl) {
      const int base = (w * 4 + bl) * 128 + lane * 4;
      const uint2 raw = *reinterpret_cast<const uint2*>(xr + base);
      vals[bl][0] = __uint_as_float((raw.x & 0xFFFFu) << 16);
      vals[bl][1] = __uint_as_float(raw.x & 0xFFFF0000u);
      vals[bl][2] = __uint_as_float((raw.y & 0xFFFFu) << 16);
      vals[bl][3] = __uint_as_float(raw.y & 0xFFFF0000u);
      const unsigned sg4 = (unsigned)sup[base >> 3] >> (base & 7);
      if (sg4 & 1u) vals[bl][0] = -vals[bl][0];
      if (sg4 & 2u) vals[bl][1] = -vals[bl][1];
      if (sg4 & 4u) vals[bl][2] = -vals[bl][2];
      if (sg4 & 8u) vals[bl][3] = -vals[bl][3];
      fwht128_wave(vals[bl], lane);
#pragma unroll
      for (int e2 = 0; e2 < 4; ++e2) {
        vals[bl][e2] *= (1.0f / 11.313708498984761f);
        am = fmaxf(am, fabsf(vals[bl][e2]));
      }
    }
    am = wave_max(am);
    if (lane == 0) s_wmax[w] = am;
    woa_lds_barrier();
    float gmax = s_wmax[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) gmax = fmaxf(gmax, s_wmax[i]);
    const float tscale = fmaxf(gmax, 1e-12f) * (1.0f / 32.0f);
    const float inv = __builtin_amdgcn_rcpf(tscale);
    if (tid == 0) tok[(size_t)t * 8 + grp] = tscale;
    _Float16* dst_g = xt + (size_t)grp * 64 * (size_t)T * 64 + (size_t)t * 64 + col64;
#pragma unroll
    for (int bl = 0; bl < 4; ++bl) {
      const int jblk = (w * 4 + bl) * 2 + sub_blk;
      h4v h4 = {(_Float16)(vals[bl][0] * inv), (_Float16)(vals[bl][1] * inv),
                (_Float16)(vals[bl][2] * inv), (_Float16)(vals[bl][3] * inv)};
      *reinterpret_cast<h4v*>(dst_g + (size_t)jblk * (size_t)T * 64) = h4;
    }
    if (gi == 0) woa_lds_barrier();
  }
}

void launch_wo_a_vha_k5_prefill_wmma(
    torch::Tensor qw, torch::Tensor scales, torch::Tensor su, torch::Tensor sv,
    torch::Tensor vha_m, torch::Tensor o_in, c10::optional<torch::Tensor> gate,
    torch::Tensor xt_buf, torch::Tensor tok_buf, torch::Tensor out,
    double cb, int64_t spad, bool sp16 = false) {
  const int T = (int)o_in.size(0);
  if (T == 0) return;
  TORCH_CHECK(vha_m.is_cuda() && vha_m.is_contiguous() && vha_m.dim() == 2 && vha_m.size(0) == 64 && vha_m.size(1) == 64 && vha_m.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(o_in.is_cuda() && o_in.dim() == 3 && o_in.size(1) >= 64 && o_in.size(2) == 512 && o_in.stride(2) == 1 && o_in.scalar_type() == torch::kBFloat16);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  const uint32_t* qw_p = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>());
  const __half* scl_p = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  const uint8_t* su_p = su.data_ptr<uint8_t>();
  const uint8_t* sv_p = sv.data_ptr<uint8_t>();
  const uint16_t* m_p = reinterpret_cast<const uint16_t*>(vha_m.data_ptr());
  const uint16_t* o_p = reinterpret_cast<const uint16_t*>(o_in.data_ptr());
  const uint16_t* g_p = gate.has_value() ? reinterpret_cast<const uint16_t*>(gate->data_ptr()) : nullptr;
  _Float16* xt_p = reinterpret_cast<_Float16*>(xt_buf.data_ptr<at::Half>());
  float* tok_p = tok_buf.data_ptr<float>();
  uint16_t* out_p = reinterpret_cast<uint16_t*>(out.data_ptr());

  hipLaunchKernelGGL(k_vha_wo_a_prologue, dim3(4, T), dim3(256), 0, stream,
                     m_p, o_p, su_p, xt_p, tok_p, T, o_in.stride(0), o_in.stride(1));
  if (T >= 256) {
    dim3 grid(8, (T + 255) / 256, 8);
    if (sp16) {
      if (g_p) hipLaunchKernelGGL((k_wo_a_k5_prefill_wmma_4x4<1, 1>), grid, dim3(256), 0, stream, qw_p, xt_p, tok_p, scl_p, sv_p, g_p, out_p, T, (float)cb, 0);
      else     hipLaunchKernelGGL((k_wo_a_k5_prefill_wmma_4x4<0, 1>), grid, dim3(256), 0, stream, qw_p, xt_p, tok_p, scl_p, sv_p, nullptr, out_p, T, (float)cb, 0);
    } else {
      if (g_p) hipLaunchKernelGGL((k_wo_a_k5_prefill_wmma_4x4<1, 0>), grid, dim3(256), 0, stream, qw_p, xt_p, tok_p, scl_p, sv_p, g_p, out_p, T, (float)cb, (int)spad);
      else     hipLaunchKernelGGL((k_wo_a_k5_prefill_wmma_4x4<0, 0>), grid, dim3(256), 0, stream, qw_p, xt_p, tok_p, scl_p, sv_p, nullptr, out_p, T, (float)cb, (int)spad);
    }
  } else if (T <= 32) {
    dim3 grid(8, (T + 31) / 32, 8);
    if (sp16) dispatch_prefill_nb<2, 1>(grid, stream, qw_p, xt_p, tok_p, scl_p, sv_p, g_p, out_p, T, (float)cb, 0);
    else      dispatch_prefill_nb<2, 0>(grid, stream, qw_p, xt_p, tok_p, scl_p, sv_p, g_p, out_p, T, (float)cb, (int)spad);
  } else {
    dim3 grid(8, (T + 63) / 64, 8);
    if (sp16) dispatch_prefill_nb<4, 1>(grid, stream, qw_p, xt_p, tok_p, scl_p, sv_p, g_p, out_p, T, (float)cb, 0);
    else      dispatch_prefill_nb<4, 0>(grid, stream, qw_p, xt_p, tok_p, scl_p, sv_p, g_p, out_p, T, (float)cb, (int)spad);
  }
}
