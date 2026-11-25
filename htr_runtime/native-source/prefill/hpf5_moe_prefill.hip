// hpf5_moe_prefill.hip — HTR v0.2.6 fused routed-MoE prefill for gfx1151.
// Eliminates BOTH Y1 ([P,1280] BF16) + k_hpf3_act<1> AND Y2 ([P,2560] BF16) + inv + reduce!
//   1. k_hpf5_gemm0_fused: computes gate & up in ONE WG (halves A1 reads!), fuses SiLU(gate)*up +
//      su sign + FWHT128 directly in registers -> writes A2 [P,640] FP16 + atomicAdd S2 [P].
//   2. k_hpf5_gemm1_fused: computes down GEMM -> fuses w[sid]*yv atomicAdd directly into out_f32 [T,2560].
//   3. Supports both BMW=64 (4 waves, MAX_NBLK=4) and BMW=128 (8 waves, MAX_NBLK=8) via bmw arg.
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

typedef _Float16 f16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));

#define BM 16
#define BN 128
#define BKS 32
#define RS 20
#define HK 4
#define NWT (8 * HK)
#define GEN_A 0xDF6ADFu
#define GEN_B 0x34983u
#define VBIAS 1534.0f
#define INV128 (1.0f / 11.313708498984761f)

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
template <int MSK> __device__ __forceinline__ void lane_stage8(float v[8], int lane) {
  const bool up = (lane & MSK) != 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float o = shx<MSK>(v[j]);
    v[j] = up ? (o - v[j]) : (v[j] + o);
  }
}
template <int BIT> __device__ __forceinline__ void reg_stage8(float v[8]) {
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    if (!(j & BIT)) { const float a = v[j], b = v[j | BIT]; v[j] = a + b; v[j | BIT] = a - b; }
  }
}
__device__ __forceinline__ void fwht128_reg8(float v[8], int lane) {
  reg_stage8<1>(v); reg_stage8<2>(v); reg_stage8<4>(v);
  lane_stage8<1>(v, lane); lane_stage8<2>(v, lane); lane_stage8<4>(v, lane); lane_stage8<8>(v, lane);
}
__device__ __forceinline__ int sw4(int n, int ch) { return n * RS + 4 * (ch ^ ((n >> 4) & 3)); }
__device__ __forceinline__ uint16_t f32_to_bf16_rne(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ _Float16 f16sat(float f, unsigned& sat) {
  if (f > 65504.f) { f = 65504.f; ++sat; } else if (f < -65504.f) { f = -65504.f; ++sat; }
  return (_Float16)f;
}

template <int M> __device__ __forceinline__ uint32_t win(uint32_t hi, uint32_t mid, uint32_t lo) {
  if constexpr (M <= 3) return __builtin_amdgcn_perm(mid, lo, (uint32_t)M | ((uint32_t)(M + 1) << 8) | 0x0c0c0000u);
  else return __builtin_amdgcn_perm(hi, mid, (uint32_t)(M - 4) | ((uint32_t)(M - 3) << 8) | 0x0c0c0000u);
}
__device__ __forceinline__ uint32_t tv16x2(uint32_t a, uint32_t b) {
  return __builtin_amdgcn_sad_hi_u8(b * GEN_A + GEN_B, 0u, __builtin_amdgcn_sad_u8(a * GEN_A + GEN_B, 0u, 0x64006400u));
}
template <int I> __device__ __forceinline__ uint32_t sym(uint32_t hi, uint32_t mid, uint32_t lo, uint32_t hi4, uint32_t mid4, uint32_t lo4) {
  if constexpr ((I & 1) == 0) return win<I / 2>(hi, mid, lo);
  else return win<I / 2>(hi4, mid4, lo4);
}
__device__ __forceinline__ void decode_row(uint32_t hi, uint32_t mid, uint32_t lo, uint32_t o[8]) {
  const uint32_t lo4 = __builtin_amdgcn_alignbit(mid, lo, 4), mid4 = __builtin_amdgcn_alignbit(hi, mid, 4), hi4 = hi >> 4;
#define HPF5_PAIR(JJ) \
  { o[JJ] = tv16x2(sym<15 - 2 * (JJ)>(hi, mid, lo, hi4, mid4, lo4), sym<14 - 2 * (JJ)>(hi, mid, lo, hi4, mid4, lo4)); }
  HPF5_PAIR(0) HPF5_PAIR(1) HPF5_PAIR(2) HPF5_PAIR(3) HPF5_PAIR(4) HPF5_PAIR(5) HPF5_PAIR(6) HPF5_PAIR(7)
#undef HPF5_PAIR
}

// ------------------------------------------------------------------ 0: plan
__global__ void k_hpf5_plan(const int32_t* __restrict__ expert_ids, const int32_t* __restrict__ ntpp,
                            int32_t* __restrict__ grp_y, int32_t* __restrict__ grp_nblk, int32_t* __restrict__ grp_cnt,
                            int max_nblk) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const int nbt = ntpp[0] / BM;
  int g = 0, y = 0;
  while (y < nbt) {
    const int e = expert_ids[y];
    if (e < 0) { ++y; continue; }
    int nb = 1;
    while (nb < max_nblk && (y + nb) < nbt && expert_ids[y + nb] == e) ++nb;
    grp_y[g] = y;
    grp_nblk[g] = nb;
    ++g;
    y += nb;
  }
  grp_cnt[0] = g;
}

// ------------------------------------------------------------------ 1: act0 (x -> A1, S1) + zero S2
__global__ __launch_bounds__(128) void k_hpf5_act0(const __hip_bfloat16* __restrict__ src, const int32_t* __restrict__ sorted_ids,
                                                   const int32_t* __restrict__ expert_ids, const uint8_t* __restrict__ su, int topk,
                                                   int num_valid, int P, _Float16* __restrict__ dst, float* __restrict__ S1,
                                                   float* __restrict__ S2, unsigned* __restrict__ nsat) {
  constexpr int D = 2560, NB = D / 128;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int p = blockIdx.x * 4 + w;
  if (p >= P) return;
  if (lane == 0) S2[p] = 0.f;
  const int sid = sorted_ids[p], e = expert_ids[p / BM];
  if (!(sid < num_valid && e >= 0)) return;
  _Float16* out = dst + (size_t)p * D;
  const uint8_t* s8 = su + (size_t)e * 400;
  unsigned sat = 0;
  float vpart[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) vpart[i] = 0.f;
#pragma unroll
  for (int blk = 0; blk < NB; ++blk) {
    const int c0 = blk * 128 + 4 * lane;
    const uint2 raw = *reinterpret_cast<const uint2*>(src + (size_t)(sid / topk) * D + c0);
    const uint32_t r2[2] = {raw.x, raw.y};
    float v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const uint32_t bits = (r2[k >> 1] >> (16 * (k & 1))) & 0xFFFFu;
      v[k] = __uint_as_float(bits << 16);
      const int c = c0 + k;
      if ((s8[c >> 3] >> (c & 7)) & 1) v[k] = -v[k];
    }
    fwht128_wave(v, lane);
    _Float16 h[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      h[k] = f16sat(v[k] * INV128, sat);
      vpart[blk & 7] += (float)h[k];
    }
    uint2 o;
    o.x = (uint32_t)__builtin_bit_cast(uint16_t, h[0]) | ((uint32_t)__builtin_bit_cast(uint16_t, h[1]) << 16);
    o.y = (uint32_t)__builtin_bit_cast(uint16_t, h[2]) | ((uint32_t)__builtin_bit_cast(uint16_t, h[3]) << 16);
    *reinterpret_cast<uint2*>(out + c0) = o;
  }
  float t = 0.f;
#pragma unroll
  for (int vw = 0; vw < 8; ++vw) {
    float part = vpart[vw];
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) part += __shfl_xor(part, m, 32);
    t += part;
  }
  if (lane == 0) S1[p] = t;
  if (sat) atomicAdd(nsat, sat);
}

// ------------------------------------------------------------------ 2: FUSED gemm0 (gate + up + SiLU + su + FWHT128 -> A2, S2)
struct Gemm0Args {
  const _Float16* A1; const float* S1;
  const uint32_t* qw; const int64_t* qoff;
  const __half* scales; const uint8_t* su; const uint8_t* sv;
  const int32_t* sorted_ids; const int32_t* expert_ids;
  const int32_t* grp_y; const int32_t* grp_nblk; const int32_t* grp_cnt;
  _Float16* A2; float* S2; unsigned* nsat;
  int num_valid, pad; float cb;
};

template <int MAX_NBLK>
__global__ __launch_bounds__(MAX_NBLK * 32) void k_hpf5_gemm0_fused(Gemm0Args a) {
  constexpr int BMW = MAX_NBLK * BM;
  constexpr int NTHR = MAX_NBLK * 32;
  constexpr int K = 2560, NT = K / 16, NS = K / BKS;
  __shared__ __attribute__((aligned(16))) uint32_t Bs[2][BN * RS];
  __shared__ int32_t s_tok[BMW];

  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, l16 = lane & 15;
  const int g = blockIdx.y, nb = blockIdx.x;  // nb in 0..4
  if (g >= a.grp_cnt[0]) return;
  const int y = a.grp_y[g];
  const int nblk = a.grp_nblk[g];
  const int e = a.expert_ids[y];
  if (e < 0) return;

  if (tid < BMW) s_tok[tid] = tid < BM * nblk ? a.sorted_ids[y * BM + tid] : a.num_valid;
  __syncthreads();

  const size_t pitch = (size_t)NT * NWT + a.pad;
  const uint32_t* qm_g = a.qw + a.qoff[(size_t)e * 3 + 0] + (size_t)nb * 8 * pitch;
  const uint32_t* qm_u = a.qw + a.qoff[(size_t)e * 3 + 1] + (size_t)nb * 8 * pitch;

  const bool wave_active = s_tok[wave * 16] < a.num_valid;
  const size_t prow = (size_t)y * BM + wave * 16;
  const _Float16* arow = a.A1 + (prow + l16) * K;

  // B loader helper: handles both NTHR=128 (2 halves/thread) and NTHR=256 (1 half/thread)
  auto stageB = [&](int buf, const uint32_t* qm, int k0) {
    uint32_t* bs = Bs[buf];
    if constexpr (NTHR == 256) {
      const int n2 = tid & 127, h2i = tid >> 7, r = n2 & 15;
      const uint32_t* tile = qm + (size_t)(n2 >> 4) * pitch + (size_t)(k0 / 16 + h2i) * NWT;
      const u32x2 pr = *(const u32x2*)(tile + 2 * r);
      uint32_t o[8];
      decode_row(tile[(2 * r + NWT - 1) % NWT], pr[0], pr[1], o);
      *(u32x4*)(&bs[sw4(n2, 2 * h2i)]) = (u32x4){o[0], o[1], o[2], o[3]};
      *(u32x4*)(&bs[sw4(n2, 2 * h2i + 1)]) = (u32x4){o[4], o[5], o[6], o[7]};
    } else {
      const int n2 = tid, r = n2 & 15;
      const uint32_t* strip = qm + (size_t)(n2 >> 4) * pitch;
#pragma unroll 1
      for (int h2i = 0; h2i < 2; ++h2i) {
        const uint32_t* tile = strip + (size_t)(k0 / 16 + h2i) * NWT;
        const u32x2 pr = *(const u32x2*)(tile + 2 * r);
        uint32_t o[8];
        decode_row(tile[(2 * r + NWT - 1) % NWT], pr[0], pr[1], o);
        *(u32x4*)(&bs[sw4(n2, 2 * h2i)]) = (u32x4){o[0], o[1], o[2], o[3]};
        *(u32x4*)(&bs[sw4(n2, 2 * h2i + 1)]) = (u32x4){o[4], o[5], o[6], o[7]};
      }
    }
  };

  auto loadA = [&](u32x4* dst, int k0) {
    if (wave_active) {
#pragma unroll
      for (int c = 0; c < 4; ++c) dst[c] = *(const u32x4*)(arow + k0 + 8 * c);
    }
  };

  f32x8 acc_g[8], acc_u[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    acc_g[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};
    acc_u[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};
  }

  u32x4 aCur[4], aNxt[4];
  loadA(aCur, 0);
  stageB(0, qm_g, 0);
  __syncthreads();

#pragma unroll 1
  for (int s = 0; s < NS; ++s) {
    // (a) Stage B_up(s) into Bs[1] while computing WMMA(aCur, B_gate(s) in Bs[0])
    stageB(1, qm_u, s * BKS);
    if (wave_active) {
      const uint32_t* bs = Bs[0];
#pragma unroll
      for (int kc = 0; kc < 2; ++kc) {
        u32x4 x0 = aCur[2 * kc], x1 = aCur[2 * kc + 1];
        f16x16 af = __builtin_bit_cast(f16x16, (u32x8){x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]});
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int n = 16 * j + l16;
          u32x4 y0 = *(const u32x4*)(&bs[sw4(n, 2 * kc)]);
          u32x4 y1 = *(const u32x4*)(&bs[sw4(n, 2 * kc + 1)]);
          f16x16 bf = __builtin_bit_cast(f16x16, (u32x8){y0[0], y0[1], y0[2], y0[3], y1[0], y1[1], y1[2], y1[3]});
          acc_g[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, acc_g[j]);
        }
      }
    }
    __syncthreads();

    // (b) Prefetch A(s+1) + Stage B_gate(s+1) into Bs[0] while computing WMMA(aCur, B_up(s) in Bs[1])
    if (s + 1 < NS) {
      loadA(aNxt, (s + 1) * BKS);
      stageB(0, qm_g, (s + 1) * BKS);
    }
    if (wave_active) {
      const uint32_t* bs = Bs[1];
#pragma unroll
      for (int kc = 0; kc < 2; ++kc) {
        u32x4 x0 = aCur[2 * kc], x1 = aCur[2 * kc + 1];
        f16x16 af = __builtin_bit_cast(f16x16, (u32x8){x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]});
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int n = 16 * j + l16;
          u32x4 y0 = *(const u32x4*)(&bs[sw4(n, 2 * kc)]);
          u32x4 y1 = *(const u32x4*)(&bs[sw4(n, 2 * kc + 1)]);
          f16x16 bf = __builtin_bit_cast(f16x16, (u32x8){y0[0], y0[1], y0[2], y0[3], y1[0], y1[1], y1[2], y1[3]});
          acc_u[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, acc_u[j]);
        }
      }
    }
    if (s + 1 < NS) {
#pragma unroll
      for (int c = 0; c < 4; ++c) aCur[c] = aNxt[c];
      __syncthreads();
    }
  }

  if (!wave_active) return;

  // FUSED EPILOGUE: gate FWHT128 + up FWHT128 + SiLU(gate)*up + su_down + FWHT128 -> A2 [P,640] + S2 [P]
  const int row0 = nb * 128;
  const __half* scl_g = a.scales + (size_t)e * 3840 + row0;
  const __half* scl_u = a.scales + (size_t)e * 3840 + 640 + row0;
  const uint8_t* svp_g = a.sv + (size_t)e * 480 + (row0 >> 3);
  const uint8_t* svp_u = a.sv + (size_t)e * 480 + 80 + (row0 >> 3);
  const uint8_t* sup_d = a.su + (size_t)e * 400 + 320 + (row0 >> 3);

  float sc_g[8], sc_u[8];
  uint32_t neg_g = 0, neg_u = 0, neg_su = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    sc_g[j] = __half2float(scl_g[16 * j + l16]) * a.cb;
    sc_u[j] = __half2float(scl_u[16 * j + l16]) * a.cb;
    neg_g  |= ((uint32_t)(svp_g[2 * j + (l16 >> 3)] >> (l16 & 7)) & 1u) << j;
    neg_u  |= ((uint32_t)(svp_u[2 * j + (l16 >> 3)] >> (l16 & 7)) & 1u) << j;
    neg_su |= ((uint32_t)(sup_d[2 * j + (l16 >> 3)] >> (l16 & 7)) & 1u) << j;
  }

  const int hrow = lane >> 4;
  unsigned sat = 0;
#pragma unroll 1
  for (int ei = 0; ei < 8; ++ei) {
    const int rl = wave * 16 + 2 * ei + hrow;
    if (s_tok[rl] >= a.num_valid) continue;
    const size_t ridx = prow + 2 * ei + hrow;
    const float bias = VBIAS * a.S1[ridx];
    float vg[8], vu[8], vh[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      vg[j] = (acc_g[j][ei] - bias) * sc_g[j];
      vu[j] = (acc_u[j][ei] - bias) * sc_u[j];
    }
    fwht128_reg8(vg, lane);
    fwht128_reg8(vu, lane);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float gv = vg[j] * INV128; if ((neg_g >> j) & 1u) gv = -gv;
      float uv = vu[j] * INV128; if ((neg_u >> j) & 1u) uv = -uv;
      float hv = (gv / (1.f + __expf(-gv))) * uv;
      if ((neg_su >> j) & 1u) hv = -hv;
      vh[j] = hv;
    }
    fwht128_reg8(vh, lane);
    _Float16* a2r = a.A2 + ridx * 640 + row0 + l16;
    float psum = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      _Float16 hf = f16sat(vh[j] * INV128, sat);
      a2r[16 * j] = hf;
      psum += (float)hf;
    }
    // Reduce psum across the 16 lanes of this half-wave (l16 = 0..15)
    psum += shx<8>(psum);
    psum += shx<4>(psum);
    psum += shx<2>(psum);
    psum += shx<1>(psum);
    if (l16 == 0) atomicAdd(&a.S2[ridx], psum);
  }
  if (sat) atomicAdd(a.nsat, sat);
}

// ------------------------------------------------------------------ 3: FUSED gemm1 (down + weighted atomicAdd -> out_f32)
struct Gemm1Args {
  const _Float16* A2; const float* S2;
  const uint32_t* qw; const int64_t* qoff;
  const __half* scales; const uint8_t* sv;
  const int32_t* sorted_ids; const int32_t* expert_ids;
  const int32_t* grp_y; const int32_t* grp_nblk; const int32_t* grp_cnt;
  const float* w; float* out_f32;
  int topk, num_valid, pad; float cb;
};

template <int MAX_NBLK>
__global__ __launch_bounds__(MAX_NBLK * 32) void k_hpf5_gemm1_fused(Gemm1Args a) {
  constexpr int BMW = MAX_NBLK * BM;
  constexpr int NTHR = MAX_NBLK * 32;
  constexpr int N = 2560, K = 640, NT = K / 16, NS = K / BKS;
  __shared__ __attribute__((aligned(16))) uint32_t Bs[2][BN * RS];
  __shared__ int32_t s_tok[BMW];

  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, l16 = lane & 15;
  const int g = blockIdx.y, nb = blockIdx.x;  // nb in 0..19
  if (g >= a.grp_cnt[0]) return;
  const int y = a.grp_y[g];
  const int nblk = a.grp_nblk[g];
  const int e = a.expert_ids[y];
  if (e < 0) return;

  if (tid < BMW) s_tok[tid] = tid < BM * nblk ? a.sorted_ids[y * BM + tid] : a.num_valid;
  __syncthreads();

  const size_t pitch = (size_t)NT * NWT + a.pad;
  const uint32_t* qm = a.qw + a.qoff[(size_t)e * 3 + 2] + (size_t)nb * 8 * pitch;
  const bool wave_active = s_tok[wave * 16] < a.num_valid;
  const size_t prow = (size_t)y * BM + wave * 16;
  const _Float16* arow = a.A2 + (prow + l16) * K;

  auto loadA = [&](u32x4* dst, int k0) {
    if (wave_active) {
#pragma unroll
      for (int c = 0; c < 4; ++c) dst[c] = *(const u32x4*)(arow + k0 + 8 * c);
    }
  };
  auto stageB = [&](int buf, int k0) {
    uint32_t* bs = Bs[buf];
    if constexpr (NTHR == 256) {
      const int n2 = tid & 127, h2i = tid >> 7, r = n2 & 15;
      const uint32_t* tile = qm + (size_t)(n2 >> 4) * pitch + (size_t)(k0 / 16 + h2i) * NWT;
      const u32x2 pr = *(const u32x2*)(tile + 2 * r);
      uint32_t o[8];
      decode_row(tile[(2 * r + NWT - 1) % NWT], pr[0], pr[1], o);
      *(u32x4*)(&bs[sw4(n2, 2 * h2i)]) = (u32x4){o[0], o[1], o[2], o[3]};
      *(u32x4*)(&bs[sw4(n2, 2 * h2i + 1)]) = (u32x4){o[4], o[5], o[6], o[7]};
    } else {
      const int n2 = tid, r = n2 & 15;
      const uint32_t* strip = qm + (size_t)(n2 >> 4) * pitch;
#pragma unroll 1
      for (int h2i = 0; h2i < 2; ++h2i) {
        const uint32_t* tile = strip + (size_t)(k0 / 16 + h2i) * NWT;
        const u32x2 pr = *(const u32x2*)(tile + 2 * r);
        uint32_t o[8];
        decode_row(tile[(2 * r + NWT - 1) % NWT], pr[0], pr[1], o);
        *(u32x4*)(&bs[sw4(n2, 2 * h2i)]) = (u32x4){o[0], o[1], o[2], o[3]};
        *(u32x4*)(&bs[sw4(n2, 2 * h2i + 1)]) = (u32x4){o[4], o[5], o[6], o[7]};
      }
    }
  };

  f32x8 acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  u32x4 aCur[4], aNxt[4];
  loadA(aCur, 0);
  stageB(0, 0);
  __syncthreads();

#pragma unroll 1
  for (int s = 0; s < NS; ++s) {
    const int buf = s & 1;
    if (s + 1 < NS) {
      loadA(aNxt, (s + 1) * BKS);
      stageB(buf ^ 1, (s + 1) * BKS);
    }
    if (wave_active) {
      const uint32_t* bs = Bs[buf];
#pragma unroll
      for (int kc = 0; kc < 2; ++kc) {
        u32x4 x0 = aCur[2 * kc], x1 = aCur[2 * kc + 1];
        f16x16 af = __builtin_bit_cast(f16x16, (u32x8){x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]});
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int n = 16 * j + l16;
          u32x4 y0 = *(const u32x4*)(&bs[sw4(n, 2 * kc)]);
          u32x4 y1 = *(const u32x4*)(&bs[sw4(n, 2 * kc + 1)]);
          f16x16 bf = __builtin_bit_cast(f16x16, (u32x8){y0[0], y0[1], y0[2], y0[3], y1[0], y1[1], y1[2], y1[3]});
          acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, acc[j]);
        }
      }
    }
    if (s + 1 < NS) {
#pragma unroll
      for (int c = 0; c < 4; ++c) aCur[c] = aNxt[c];
      __syncthreads();
    }
  }

  if (!wave_active) return;

  const int row0 = nb * 128;
  const __half* scl = a.scales + (size_t)e * 3840 + 1280 + row0;
  const uint8_t* svp = a.sv + (size_t)e * 480 + 160 + (row0 >> 3);
  float sc[8];
  uint32_t neg = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    sc[j] = __half2float(scl[16 * j + l16]) * a.cb;
    neg |= ((uint32_t)(svp[2 * j + (l16 >> 3)] >> (l16 & 7)) & 1u) << j;
  }
  const int hrow = lane >> 4;
#pragma unroll 1
  for (int ei = 0; ei < 8; ++ei) {
    const int rl = wave * 16 + 2 * ei + hrow;
    const int sid = s_tok[rl];
    if (sid >= a.num_valid) continue;
    const float wt = a.w[sid];
    if (wt == 0.f) continue;
    const float bias = VBIAS * a.S2[prow + 2 * ei + hrow];
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = (acc[j][ei] - bias) * sc[j];
    fwht128_reg8(v, lane);
    const int tok = sid / a.topk;
    float* dst = a.out_f32 + (size_t)tok * N + row0 + l16;
    const float w_inv = wt * INV128;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float yv = v[j] * w_inv;
      if ((neg >> j) & 1u) yv = -yv;
      atomicAdd(&dst[16 * j], yv);
    }
  }
}

// ------------------------------------------------------------------ 4: f32 -> bf16 + zero out_f32
__global__ __launch_bounds__(256) void k_f32_to_bf16(float* __restrict__ in_f32, __hip_bfloat16* __restrict__ out_bf16, int n4) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n4) return;
  const float4 v = reinterpret_cast<const float4*>(in_f32)[i];
  uint2 o;
  o.x = (uint32_t)f32_to_bf16_rne(v.x) | ((uint32_t)f32_to_bf16_rne(v.y) << 16);
  o.y = (uint32_t)f32_to_bf16_rne(v.z) | ((uint32_t)f32_to_bf16_rne(v.w) << 16);
  reinterpret_cast<uint2*>(out_bf16)[i] = o;
}

torch::Tensor hpf5_forward(torch::Tensor x, torch::Tensor sorted_ids, torch::Tensor expert_ids, torch::Tensor ntpp, torch::Tensor w,
                           torch::Tensor qw, torch::Tensor qoff, torch::Tensor scales, torch::Tensor su, torch::Tensor sv, double cb,
                           int64_t K, int64_t pad, torch::Tensor A1, torch::Tensor A2,
                           torch::Tensor S1, torch::Tensor S2, torch::Tensor out_f32, torch::Tensor nsat,
                           torch::Tensor grp_y, torch::Tensor grp_nblk, torch::Tensor grp_cnt, int64_t bmw) {
  TORCH_CHECK(K == HK, "this build decodes K = 4 only");
  TORCH_CHECK(bmw == 64 || bmw == 128, "bmw must be 64 or 128");
  const int T = x.size(0), topk = w.size(1), nv = T * topk, P = sorted_ids.numel();
  const int max_blocks = P / BM;
  const int max_nblk = (int)(bmw / BM);
  auto st = at::cuda::getCurrentCUDAStream();
  const int32_t* sid = sorted_ids.data_ptr<int32_t>();
  const int32_t* eid = expert_ids.data_ptr<int32_t>();
  unsigned* ns = reinterpret_cast<unsigned*>(nsat.data_ptr<int32_t>());
  float* s1p = S1.data_ptr<float>();
  float* s2p = S2.data_ptr<float>();
  _Float16* a1 = reinterpret_cast<_Float16*>(A1.data_ptr<at::Half>());
  _Float16* a2 = reinterpret_cast<_Float16*>(A2.data_ptr<at::Half>());

  hipLaunchKernelGGL(k_hpf5_plan, dim3(1), dim3(32), 0, st, eid, ntpp.data_ptr<int32_t>(),
                     grp_y.data_ptr<int32_t>(), grp_nblk.data_ptr<int32_t>(), grp_cnt.data_ptr<int32_t>(), max_nblk);

  const int pg = (P + 3) / 4;
  hipLaunchKernelGGL(k_hpf5_act0, dim3(pg), dim3(128), 0, st,
                     reinterpret_cast<const __hip_bfloat16*>(x.data_ptr<at::BFloat16>()),
                     sid, eid, su.data_ptr<uint8_t>(), topk, nv, P, a1, s1p, s2p, ns);

  // Zero out_f32 [T, 2560] asynchronously on stream
  (void)hipMemsetAsync(out_f32.data_ptr<float>(), 0, (size_t)T * 2560 * sizeof(float), st);

  Gemm0Args a0;
  a0.A1 = a1; a0.S1 = s1p;
  a0.qw = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>()); a0.qoff = qoff.data_ptr<int64_t>();
  a0.scales = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  a0.su = su.data_ptr<uint8_t>(); a0.sv = sv.data_ptr<uint8_t>();
  a0.sorted_ids = sid; a0.expert_ids = eid;
  a0.grp_y = grp_y.data_ptr<int32_t>(); a0.grp_nblk = grp_nblk.data_ptr<int32_t>(); a0.grp_cnt = grp_cnt.data_ptr<int32_t>();
  a0.A2 = a2; a0.S2 = s2p; a0.nsat = ns;
  a0.num_valid = nv; a0.pad = (int)pad; a0.cb = (float)cb;

  Gemm1Args a1_args;
  a1_args.A2 = a2; a1_args.S2 = s2p;
  a1_args.qw = a0.qw; a1_args.qoff = a0.qoff;
  a1_args.scales = a0.scales; a1_args.sv = a0.sv;
  a1_args.sorted_ids = sid; a1_args.expert_ids = eid;
  a1_args.grp_y = a0.grp_y; a1_args.grp_nblk = a0.grp_nblk; a1_args.grp_cnt = a0.grp_cnt;
  a1_args.w = w.data_ptr<float>(); a1_args.out_f32 = out_f32.data_ptr<float>();
  a1_args.topk = topk; a1_args.num_valid = nv; a1_args.pad = (int)pad; a1_args.cb = (float)cb;

  if (bmw == 64) {
    hipLaunchKernelGGL((k_hpf5_gemm0_fused<4>), dim3(5, max_blocks), dim3(128), 0, st, a0);
    hipLaunchKernelGGL((k_hpf5_gemm1_fused<4>), dim3(20, max_blocks), dim3(128), 0, st, a1_args);
  } else {
    hipLaunchKernelGGL((k_hpf5_gemm0_fused<8>), dim3(5, max_blocks), dim3(256), 0, st, a0);
    hipLaunchKernelGGL((k_hpf5_gemm1_fused<8>), dim3(20, max_blocks), dim3(256), 0, st, a1_args);
  }

  auto out = torch::empty({T, 2560}, x.options());
  const int n4 = (T * 2560) / 4;
  hipLaunchKernelGGL(k_f32_to_bf16, dim3((n4 + 255) / 256), dim3(256), 0, st,
                     out_f32.data_ptr<float>(), reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()), n4);
  return out;
}

PYBIND11_MODULE(hpf5_moe_prefill, m) {
  m.def("hpf5_forward", &hpf5_forward);
}
