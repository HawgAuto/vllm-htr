// Native gfx1151 (RDNA3.5) HPF5-ERNIE fused WMMA + SAD_HI + Register-FWHT128 routed-MoE prefill kernel.
// Zero BF16 weight materialization: reads packed HTR-W1 K=3 (or K=2 for DSpark) weights directly into LDS
// via 3-stage double-buffered software pipeline, decodes pairs with RDNA3.5 v_sad_u8 + v_sad_hi_u8 (SAD_HI),
// multiplies with Wave32 WMMA (__builtin_amdgcn_wmma_f32_16x16x16_f16_w32), and runs 100% in-register
// FWHT-128 epilogue.
//
// ERNIE-5.5 latent MoE geometry: H = 2048, I = 2048, E = 512, topk = 10, SiTU-GLU(beta=4.0, linear_beta=25.0).
// Widths supported: K = 3 (main experts, NWT = 24), K = 2 (DSpark draft experts, NWT = 16).
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

typedef _Float16 f16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef u32x2 u32x2a __attribute__((aligned(4)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));

#define MOE_H 2048
#define MOE_I 2048
#define SC_STRIDE 6144   // w1 2048 | w3 2048 | w2 2048
#define SV_STRIDE 768    // w1 256  | w3 256  | w2 256
#define SU_STRIDE 512    // w1/w3 256 | w2 256
#define BM 16            // moe_align block size (sorted rows per block)
#define BN 128           // output columns per WG
#define BKS 32           // K-step per pipeline stage (2x 16x16 tiles)
#define RS 20            // LDS row stride in uint32s (16 uint32s = 32 f16s + 4 pad)
#define GEN_A 0xDF6ADFu
#define GEN_B 0x34983u
#define VBIAS 1534.0f
#define INV128 (1.0f / 11.313708498984761f)

#ifndef HPF_ERNIE_WPE
#define HPF_ERNIE_WPE 8
#endif

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
__device__ __forceinline__ int sw4(int n, int ch) { return n * RS + 4 * (ch ^ ((n >> 4) & 3)); }

__device__ __forceinline__ uint16_t f32_to_bf16_rne(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ uint32_t f32_to_bf16x2_rne(float a, float b) {
  uint32_t ua = __float_as_uint(a);
  uint32_t ub = __float_as_uint(b);
  ua = ((ua & 0x7fffffffu) > 0x7f800000u) ? (ua | 0x00400000u) : (ua + 0x7fffu + ((ua >> 16) & 1u));
  ub = ((ub & 0x7fffffffu) > 0x7f800000u) ? (ub | 0x00400000u) : (ub + 0x7fffu + ((ub >> 16) & 1u));
  return __builtin_amdgcn_perm(ub, ua, 0x07060302u);
}
__device__ __forceinline__ _Float16 f16sat(float f, unsigned& sat) {
  if (f > 65504.f) { f = 65504.f; ++sat; } else if (f < -65504.f) { f = -65504.f; ++sat; }
  return (_Float16)f;
}
// Branchless RDNA 3.5 hardware fast_tanhf (v_exp_f32 + v_rcp_f32, max abs err < 1e-7, 6 ops vs ~45 for libm tanhf!)
__device__ __forceinline__ float fast_tanhf(float x) {
  const float ax = fabsf(x);
  const float e = exp2f((-2.0f * 1.4426950408889634f) * ax);
  const float t = (1.0f - e) * __builtin_amdgcn_rcpf(1.0f + e);
  return copysignf(t, x);
}
__device__ __forceinline__ float situ_glu_f(float g, float u) {
  float sg = 4.0f * fast_tanhf(g * 0.25f) * __builtin_amdgcn_rcpf(1.0f + exp2f(-1.4426950408889634f * g));
  float su = 25.0f * fast_tanhf(u * 0.04f);
  return sg * su;
}

// SAD_HI pair decode: val_f16bits(a) | (val_f16bits(b) << 16)
__device__ __forceinline__ uint32_t tv16x2(uint32_t a, uint32_t b) {
  return __builtin_amdgcn_sad_hi_u8(b * GEN_A + GEN_B, 0u, __builtin_amdgcn_sad_u8(a * GEN_A + GEN_B, 0u, 0x64006400u));
}

template <int B>
__device__ __forceinline__ uint32_t ext16(const uint32_t* V) {
  constexpr int w = B >> 5, off = B & 31;
  if constexpr (off <= 16) return (V[w] >> (16 - off)) & 0xFFFFu;
  else return __builtin_amdgcn_alignbit(V[w], V[w + 1], 48 - off) & 0xFFFFu;
}

// Decode one 16-element tile row r into 8 packed f16 pairs (v + 1534)
// Uses RDNA 3.5 v_permlane16_b32 (shift -1 mod 16 within each half-wave) for zero redundant weight loads!
template <int K>
__device__ __forceinline__ void decode_row_k(const uint32_t* br, int r, uint32_t o[8]) {
  if constexpr (K == 3) {
    const uint32_t w0 = br[0], w1 = br[1];
    // Lane r gets lane ((r + 15) & 15)'s w1: for even r=2*kk, lane 2*kk-1 holds tile[(3*kk+23)%24] in w1!
    const uint32_t prev_w1 = __builtin_amdgcn_permlane16(w1, w1, 0x6543210Fu, 0xEDCBA987u, false, false);
    uint32_t V[3];
    if (r & 1) {
      V[0] = w0 >> 16;
      V[1] = __builtin_amdgcn_alignbit(w0, w1, 16);
      V[2] = w1 << 16;
    } else {
      V[0] = prev_w1; V[1] = w0; V[2] = w1;
    }
#define DEC3_PAIR(JJ) o[JJ] = tv16x2(ext16<19 + 6 * (JJ)>(V), ext16<22 + 6 * (JJ)>(V));
    DEC3_PAIR(0) DEC3_PAIR(1) DEC3_PAIR(2) DEC3_PAIR(3)
    DEC3_PAIR(4) DEC3_PAIR(5) DEC3_PAIR(6) DEC3_PAIR(7)
#undef DEC3_PAIR
  } else if constexpr (K == 2) {
    const uint32_t w0 = br[0];
    const uint32_t prev_w0 = __builtin_amdgcn_permlane16(w0, w0, 0x6543210Fu, 0xEDCBA987u, false, false);
    const uint32_t V[2] = {prev_w0, w0};
#define DEC2_PAIR(JJ) o[JJ] = tv16x2(ext16<18 + 4 * (JJ)>(V), ext16<20 + 4 * (JJ)>(V));
    DEC2_PAIR(0) DEC2_PAIR(1) DEC2_PAIR(2) DEC2_PAIR(3)
    DEC2_PAIR(4) DEC2_PAIR(5) DEC2_PAIR(6) DEC2_PAIR(7)
#undef DEC2_PAIR
  }
}

// ------------------------------------------------------------------ 0: fast single-WG moe_align_block_size (E <= 512, BM = 16)
// Parallel 512-thread two-level prefix scan + parallel expert_ids/grp emit (eliminates 512-iter serial thread-0 loop!)
__device__ __forceinline__ int scan512_incl(int val, int32_t* s_buf, int32_t* s_wave, int tid) {
  const int w = tid >> 5;
  const int lane = tid & 31;
  s_buf[tid] = val;
  __syncthreads();
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int add = (lane >= d) ? s_buf[tid - d] : 0;
    __syncthreads();
    val += add;
    s_buf[tid] = val;
    __syncthreads();
  }
  if (lane == 31) s_wave[w] = val;
  __syncthreads();
#pragma unroll
  for (int d = 1; d < 16; d <<= 1) {
    int wv = 0;
    if (tid < 16) {
      wv = s_wave[tid] + ((tid >= d) ? s_wave[tid - d] : 0);
    }
    __syncthreads();
    if (tid < 16) s_wave[tid] = wv;
    __syncthreads();
  }
  const int w_off = (w > 0) ? s_wave[w - 1] : 0;
  return val + w_off;
}

__global__ __launch_bounds__(512) void k_moe_align_ernie(
    const int32_t* __restrict__ topk_ids, int nv, int E,
    int32_t* __restrict__ sorted_ids, int32_t* __restrict__ expert_ids, int32_t* __restrict__ ntpp,
    int32_t* __restrict__ grp_y, int32_t* __restrict__ grp_nblk, int32_t* __restrict__ grp_cnt, int max_nblk) {
  __shared__ int32_t s_cnt[512];
  __shared__ int32_t s_cur[512];
  __shared__ int32_t s_scan[512];
  __shared__ int32_t s_wblk[16];
  __shared__ int32_t s_wgrp[16];
  const int tid = threadIdx.x;
  s_cnt[tid] = 0;
  __syncthreads();

  for (int i = tid; i < nv; i += 512) {
    const int e = topk_ids[i];
    if (e >= 0 && e < E) atomicAdd(&s_cnt[e], 1);
  }
  __syncthreads();

  // Parallel 512-expert scan over block counts (nb) and group counts (ng)
  const int c = (tid < E) ? s_cnt[tid] : 0;
  const int nb = (c + BM - 1) / BM;
  const int ng = (nb + max_nblk - 1) / max_nblk;
  const int blk_incl = scan512_incl(nb, s_scan, s_wblk, tid);
  const int blk_excl = blk_incl - nb;
  const int grp_incl = scan512_incl(ng, s_scan, s_wgrp, tid);
  const int grp_excl = grp_incl - ng;

  const int total_blocks = s_wblk[15];
  const int total_grps = s_wgrp[15];
  if (tid < E) {
    s_cur[tid] = blk_excl * BM;
    for (int k = 0; k < nb; ++k) expert_ids[blk_excl + k] = tid;
    int rem = nb, y = blk_excl, g = grp_excl;
    while (rem > 0) {
      const int take = rem < max_nblk ? rem : max_nblk;
      grp_y[g] = y;
      grp_nblk[g] = take;
      ++g;
      y += take;
      rem -= take;
    }
  }
  if (tid == 0) {
    ntpp[0] = total_blocks * BM;
    grp_cnt[0] = total_grps;
  }
  __syncthreads();

  const int total_padded = total_blocks * BM;
  for (int i = tid; i < total_padded; i += 512) sorted_ids[i] = nv;
  __syncthreads();

  for (int i = tid; i < nv; i += 512) {
    const int e = topk_ids[i];
    if (e >= 0 && e < E) {
      const int pos = atomicAdd(&s_cur[e], 1);
      sorted_ids[pos] = i;
    }
  }
}

// ------------------------------------------------------------------ 1/3: activation prologues (1 wave per sorted row)
// MODE 0: src = x [T, 2048] bf16 -> A1 [P, 2048] f16, S [P] f32
// MODE 1: src = Y1 [P, 4096] bf16 (gate|up) -> SiTU-GLU -> A2 [P, 2048] f16, S [P] f32
template <int MODE>
__global__ __launch_bounds__(128) void k_hpf_ernie_act(
    const __hip_bfloat16* __restrict__ src,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ ntpp,
    const uint8_t* __restrict__ su,
    int topk, int nv,
    _Float16* __restrict__ A, float* __restrict__ S, unsigned* __restrict__ nsat) {
  constexpr int KDIM = (MODE == 0) ? MOE_H : MOE_I; // 2048
  constexpr int NBLK = KDIM / 128;                  // 16
  const int p = blockIdx.x * 4 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int P_act = ntpp[0];
  if (p >= P_act) return;
  const int sid = sorted_ids[p];
  const int e = expert_ids[p / BM];
  if (sid >= nv || e < 0) return;

  const uint8_t* s8 = su + (size_t)e * SU_STRIDE + (MODE == 0 ? 0 : (MOE_H / 8));
  _Float16* out = A + (size_t)p * KDIM;
  const int tok = sid / topk;
  unsigned sat = 0;
  float vsum = 0.f;

#pragma unroll 1
  for (int blk = 0; blk < NBLK; ++blk) {
    const int c0 = blk * 128 + lane * 4;
    float v[4];
    if constexpr (MODE == 0) {
      const uint2 raw = *reinterpret_cast<const uint2*>(src + (size_t)tok * MOE_H + c0);
      v[0] = __uint_as_float((raw.x & 0xFFFFu) << 16);
      v[1] = __uint_as_float(raw.x & 0xFFFF0000u);
      v[2] = __uint_as_float((raw.y & 0xFFFFu) << 16);
      v[3] = __uint_as_float(raw.y & 0xFFFF0000u);
    } else {
      const uint64_t qg = __builtin_nontemporal_load(reinterpret_cast<const uint64_t*>(src + (size_t)p * (2 * MOE_I) + c0));
      const uint64_t qu = __builtin_nontemporal_load(reinterpret_cast<const uint64_t*>(src + (size_t)p * (2 * MOE_I) + MOE_I + c0));
      const uint2 rg = make_uint2((uint32_t)qg, (uint32_t)(qg >> 32));
      const uint2 ru = make_uint2((uint32_t)qu, (uint32_t)(qu >> 32));
      v[0] = situ_glu_f(__uint_as_float((rg.x & 0xFFFFu) << 16), __uint_as_float((ru.x & 0xFFFFu) << 16));
      v[1] = situ_glu_f(__uint_as_float(rg.x & 0xFFFF0000u),     __uint_as_float(ru.x & 0xFFFF0000u));
      v[2] = situ_glu_f(__uint_as_float((rg.y & 0xFFFFu) << 16), __uint_as_float((ru.y & 0xFFFFu) << 16));
      v[3] = situ_glu_f(__uint_as_float(rg.y & 0xFFFF0000u),     __uint_as_float(ru.y & 0xFFFF0000u));
    }
    const unsigned sg4 = (unsigned)s8[c0 >> 3] >> (c0 & 7);
    if (sg4 & 1u) v[0] = -v[0];
    if (sg4 & 2u) v[1] = -v[1];
    if (sg4 & 4u) v[2] = -v[2];
    if (sg4 & 8u) v[3] = -v[3];
    fwht128_wave(v, lane);
    _Float16 h[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      h[k] = f16sat(v[k] * INV128, sat);
      vsum += (float)h[k];
    }
    uint2 o;
    o.x = (uint32_t)__builtin_bit_cast(uint16_t, h[0]) | ((uint32_t)__builtin_bit_cast(uint16_t, h[1]) << 16);
    o.y = (uint32_t)__builtin_bit_cast(uint16_t, h[2]) | ((uint32_t)__builtin_bit_cast(uint16_t, h[3]) << 16);
    *reinterpret_cast<uint2*>(out + c0) = o;
  }
  // Single 5-step Wave32 DPP reduction (8x fewer wave shuffles than 8x separate reductions!)
  vsum += shx<16>(vsum);
  vsum += shx<8>(vsum);
  vsum += shx<4>(vsum);
  vsum += shx<2>(vsum);
  vsum += shx<1>(vsum);
  if (lane == 0) S[p] = vsum;
  if (sat) atomicAdd(nsat, sat);
}

// ------------------------------------------------------------------ 2/4: WMMA grouped GEMM with SAD_HI + register FWHT-128 epilogue
struct ErnieGemmArgs {
  const _Float16* A; const float* S;
  const uint32_t* qw; const int64_t* qoff;
  const __half* scales; const uint8_t* sv;
  const int32_t* sorted_ids; const int32_t* expert_ids;
  const int32_t* grp_y; const int32_t* grp_nblk; const int32_t* grp_cnt;
  __hip_bfloat16* Y;
  int num_valid, pad; float cb;
};

template <int K, int PH, int MAX_NBLK, int NT_LOAD>
__global__ __launch_bounds__(MAX_NBLK * 32) __attribute__((amdgpu_waves_per_eu(HPF_ERNIE_WPE)))
void k_hpf_ernie_gemm(ErnieGemmArgs a) {
  constexpr int N = (PH == 0) ? (2 * MOE_I) : MOE_H;  // 4096 or 2048
  constexpr int KDIM = (PH == 0) ? MOE_H : MOE_I;     // 2048
  constexpr int NT = KDIM / 16;                       // 128
  constexpr int NS = KDIM / BKS;                      // 64
  constexpr int NWT = 8 * K;                          // 24 (K=3) or 16 (K=2)
  constexpr int LBMW = MAX_NBLK * BM;
  constexpr int NTHR = MAX_NBLK * 32;
  constexpr int B_ITERS = 256 / NTHR;                 // 1 for BMW=128, 2 for BMW=64, 4 for BMW=32
  constexpr int W_PER_ROW = (K == 3) ? 2 : 1;         // permlane16 eliminates redundant cross-row weight loads!

  __shared__ __attribute__((aligned(16))) uint32_t Bs[2][BN * RS];
  __shared__ int32_t s_tok[LBMW];
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, l16 = lane & 15;
  const int g = blockIdx.y, nb = blockIdx.x;
  if (g >= a.grp_cnt[0]) return;
  const int y = a.grp_y[g];
  const int nblk = a.grp_nblk[g];
  const int e = a.expert_ids[y];
  if (e < 0) return;

  for (int i = tid; i < LBMW; i += NTHR) {
    s_tok[i] = (i < BM * nblk) ? a.sorted_ids[y * BM + i] : a.num_valid;
  }
  __syncthreads();

  const int role = (PH == 0) ? (nb < (MOE_I / BN) ? 0 : 1) : 2;
  const int sblk = (PH == 0) ? (nb < (MOE_I / BN) ? nb : nb - (MOE_I / BN)) : nb;
  const size_t pitch = (size_t)NT * NWT + a.pad;
  const uint32_t* qm = a.qw + a.qoff[(size_t)e * 3 + role] + (size_t)sblk * 8 * pitch;

  const bool wave_active = s_tok[wave * 16] < a.num_valid;
  const size_t prow = (size_t)y * BM + wave * 16;
  const _Float16* arow = a.A + (prow + l16) * KDIM;

  // Map thread tid across 256 virtual B-loaders: each loader handles (n2 in 0..127, h2i in 0..1)
  auto loadA = [&](u32x4* dst, int k0) {
    if (wave_active) {
#pragma unroll
      for (int c = 0; c < 4; ++c) dst[c] = *(const u32x4*)(arow + k0 + 8 * c);
    }
  };
  auto loadB = [&](uint32_t* br, int k0) {
#pragma unroll
    for (int bi = 0; bi < B_ITERS; ++bi) {
      const int vtid = tid + bi * NTHR;
      const int n2 = vtid & 127, r = n2 & 15, h2i = vtid >> 7;
      const uint32_t* tile = qm + (size_t)(n2 >> 4) * pitch + (size_t)(k0 / 16 + h2i) * NWT;
      if constexpr (K == 3) {
        // 100% branchless single global_load_b64 (5x fewer VMEM ops than divergent 5x b32!)
        const int kk = r >> 1;
        const u32x2a* p2 = reinterpret_cast<const u32x2a*>(tile + 3 * kk + (r & 1));
        const u32x2 pair = NT_LOAD ? __builtin_nontemporal_load(p2) : *p2;
        br[bi * 2 + 0] = pair[0];
        br[bi * 2 + 1] = pair[1];
      } else if constexpr (K == 2) {
        // Single global_load_b32 (2x fewer VMEM ops; prev lane word fetched via v_permlane16_b32!)
        br[bi] = NT_LOAD ? __builtin_nontemporal_load(tile + r) : tile[r];
      }
    }
  };
  auto storeB = [&](int buf, const uint32_t* br) {
    uint32_t* bs = Bs[buf];
#pragma unroll
    for (int bi = 0; bi < B_ITERS; ++bi) {
      const int vtid = tid + bi * NTHR;
      const int n2 = vtid & 127, r = n2 & 15, h2i = vtid >> 7;
      uint32_t o[8];
      decode_row_k<K>(&br[bi * W_PER_ROW], r, o);
      *(u32x4*)(&bs[sw4(n2, 2 * h2i)]) = (u32x4){o[0], o[1], o[2], o[3]};
      *(u32x4*)(&bs[sw4(n2, 2 * h2i + 1)]) = (u32x4){o[4], o[5], o[6], o[7]};
      if constexpr (B_ITERS > 1) __builtin_amdgcn_sched_barrier(0);
    }
  };

  u32x4 aX[4], aY[4];
  uint32_t bX[B_ITERS * W_PER_ROW], bY[B_ITERS * W_PER_ROW], bZ[B_ITERS * W_PER_ROW];
  f32x8 acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  auto step = [&](int s, int buf, const u32x4* cur, u32x4* nx, const uint32_t* b1, uint32_t* b2) {
    if (s + 1 < NS) loadA(nx, (s + 1) * BKS);
    if (s + 2 < NS) loadB(b2, (s + 2) * BKS);
    if (wave_active) {
      const uint32_t* bs = Bs[buf];
#pragma unroll
      for (int kc = 0; kc < 2; ++kc) {
        u32x4 x0 = cur[2 * kc], x1 = cur[2 * kc + 1];
        f16x16 af = __builtin_bit_cast(f16x16, (u32x8){x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]});
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int n = 16 * j + l16;
          u32x4 y0 = *(const u32x4*)(&bs[sw4(n, 2 * kc)]);
          u32x4 y1 = *(const u32x4*)(&bs[sw4(n, 2 * kc + 1)]);
          f16x16 bf = __builtin_bit_cast(f16x16, (u32x8){y0[0], y0[1], y0[2], y0[3], y1[0], y1[1], y1[2], y1[3]});
          acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, acc[j]);
          __builtin_amdgcn_sched_barrier(0);
        }
      }
    }
    if (s + 1 < NS) storeB(buf ^ 1, b1);
    __syncthreads();
  };

  loadA(aX, 0); loadB(bX, 0);
  if (NS > 1) loadB(bY, BKS);
  storeB(0, bX);
  __syncthreads();
  for (int s = 0; s < NS; s += 6) {
    step(s + 0, 0, aX, aY, bY, bZ);
    if (s + 1 < NS) step(s + 1, 1, aY, aX, bZ, bX);
    if (s + 2 < NS) step(s + 2, 0, aX, aY, bX, bY);
    if (s + 3 < NS) step(s + 3, 1, aY, aX, bY, bZ);
    if (s + 4 < NS) step(s + 4, 0, aX, aY, bZ, bX);
    if (s + 5 < NS) step(s + 5, 1, aY, aX, bX, bY);
  }
  if (!wave_active) return;

  const int row0 = nb * 128;
  const int soff = (PH == 0) ? 0 : (2 * MOE_I);
  const int svoff = (PH == 0) ? 0 : (2 * MOE_I / 8);
  const __half* scl = a.scales + (size_t)e * SC_STRIDE + soff + row0;
  const uint8_t* svp = a.sv + (size_t)e * SV_STRIDE + svoff + (row0 >> 3);
  float sc[8]; uint32_t neg = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    sc[j] = __half2float(scl[16 * j + l16]) * a.cb;
    neg |= ((uint32_t)(svp[2 * j + (l16 >> 3)] >> (l16 & 7)) & 1u) << j;
  }
  const int hrow = lane >> 4;
  // Zero-barrier wave-private 512 B staging slice in dead Bs[0] (8 waves * 512 B = 4 KiB << 20 KiB)!
  // Keeps if (!wave_active) return; 100% intact and replaces 64x scattered 2B RFO stores with 256B u32x4!
  uint16_t* eb = reinterpret_cast<uint16_t*>(&Bs[0][0]) + wave * 256;
#pragma unroll
  for (int ei = 0; ei < 8; ++ei) {
    const int rl = wave * 16 + 2 * ei + hrow;
    const float bias = VBIAS * a.S[prow + 2 * ei + hrow];
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = (acc[j][ei] - bias) * sc[j];
    reg_stage8<1>(v); reg_stage8<2>(v); reg_stage8<4>(v);
    lane_stage8<1>(v, lane); lane_stage8<2>(v, lane); lane_stage8<4>(v, lane); lane_stage8<8>(v, lane);
    if (s_tok[rl] < a.num_valid) {
      uint16_t* dst_l = eb + hrow * 128 + l16;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float yv = v[j] * INV128;
        if ((neg >> j) & 1u) yv = -yv;
        dst_l[16 * j] = f32_to_bf16_rne(yv);
      }
      __builtin_amdgcn_wave_barrier();
      const u32x4 v4 = *reinterpret_cast<const u32x4*>(eb + hrow * 128 + l16 * 8);
      *reinterpret_cast<u32x4*>(reinterpret_cast<uint16_t*>(a.Y) + (prow + 2 * ei + hrow) * (size_t)N + row0 + l16 * 8) = v4;
    }
  }
}

// ------------------------------------------------------------------ 5: inverse scatter + weighted reduce (256 threads x 8 cols = 2048)
__global__ __launch_bounds__(256) void k_hpf_ernie_inv(
    const int32_t* __restrict__ sorted_ids, const int32_t* __restrict__ ntpp, int nv, int32_t* __restrict__ inv) {
  const int p = blockIdx.x * 256 + threadIdx.x;
  if (p < ntpp[0]) {
    const int s = sorted_ids[p];
    if (s < nv) inv[s] = p;
  }
}

__global__ __launch_bounds__(256) void k_hpf_ernie_reduce(
    const __hip_bfloat16* __restrict__ Y2, const int32_t* __restrict__ inv,
    const float* __restrict__ w, int topk, __hip_bfloat16* __restrict__ out) {
  const int t = blockIdx.x, i0 = threadIdx.x * 8;  // 256 threads x 8 cols = 2048
  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = 0.f;
  uint4 rows[10];
  float ws[10];
#pragma unroll
  for (int s = 0; s < 10; ++s) {
    ws[s] = s < topk ? w[t * topk + s] : 0.f;
    if (ws[s] != 0.f) rows[s] = *reinterpret_cast<const uint4*>(Y2 + (size_t)inv[t * topk + s] * MOE_H + i0);
    else rows[s] = make_uint4(0, 0, 0, 0);
  }
#pragma unroll
  for (int s = 0; s < 10; ++s) {
    if (ws[s] != 0.f) {
      const uint32_t r4[4] = {rows[s].x, rows[s].y, rows[s].z, rows[s].w};
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float y = __uint_as_float(((r4[j >> 1] >> (16 * (j & 1))) & 0xFFFFu) << 16);
        acc[j] = __builtin_fmaf(ws[s], y, acc[j]);
      }
    }
  }
  uint4 o;
  o.x = f32_to_bf16x2_rne(acc[0], acc[1]);
  o.y = f32_to_bf16x2_rne(acc[2], acc[3]);
  o.z = f32_to_bf16x2_rne(acc[4], acc[5]);
  o.w = f32_to_bf16x2_rne(acc[6], acc[7]);
  *reinterpret_cast<uint4*>(out + (size_t)t * MOE_H + i0) = o;
}


// v0.2.8 RDNA3.5 2-N-Half (1x4 for bmw=32 -> 2x occupancy; 2x4 for bmw=64/128 -> 2x WMMA/LDS!)
// Keeps SAD_HI + v_permlane16 + 3-stage B-pipeline + 100% exact FWHT-128 epilogue (64-pt reg + 1 butterfly via dead Bs LDS).
template <int MSK> __device__ __forceinline__ void lane_stage4(float v[4], int lane) {
  const bool up = (lane & MSK) != 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float o = shx<MSK>(v[j]);
    v[j] = up ? (o - v[j]) : (v[j] + o);
  }
}
template <int BIT> __device__ __forceinline__ void reg_stage4(float v[4]) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    if (!(j & BIT)) { const float a = v[j], b = v[j | BIT]; v[j] = a + b; v[j | BIT] = a - b; }
  }
}

// MT = M-tiles per wave (1 for bmw=32 -> 4 waves = 32x128; 2 for bmw=64 -> 4 waves = 64x128; 2 for bmw=128 -> 8 waves = 128x128)
// NWM = M-wave-groups (2 for bmw=32/64 -> 4 waves total; 4 for bmw=128 -> 8 waves total)
template <int K, int PH, int NWM, int MT, int NT_LOAD>
__global__ __launch_bounds__(NWM * 2 * 32)
void k_hpf_ernie_gemm_n2(ErnieGemmArgs a) {
  constexpr int N = (PH == 0) ? (2 * MOE_I) : MOE_H;
  constexpr int KDIM = (PH == 0) ? MOE_H : MOE_I;
  constexpr int NT = KDIM / 16;
  constexpr int NS = KDIM / BKS;
  constexpr int NWT = 8 * K;
  constexpr int LBMW = NWM * MT * BM;
  constexpr int NWAVES = NWM * 2;
  constexpr int NTHR = NWAVES * 32;
  constexpr int B_ITERS = 256 / NTHR;
  constexpr int W_PER_ROW = (K == 3) ? 2 : 1;

  __shared__ __attribute__((aligned(16))) uint32_t Bs[2][BN * RS];
  __shared__ int32_t s_tok[LBMW];
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, l16 = lane & 15;
  const int wm = wave >> 1;      // 0 .. NWM-1
  const int wn = wave & 1;       // 0 .. 1 (0: cols 0..63, 1: cols 64..127)
  const int g = blockIdx.y, nb = blockIdx.x;
  if (g >= a.grp_cnt[0]) return;
  const int y = a.grp_y[g];
  const int nblk = a.grp_nblk[g];
  const int e = a.expert_ids[y];
  if (e < 0) return;

  for (int i = tid; i < LBMW; i += NTHR) {
    s_tok[i] = (i < BM * nblk) ? a.sorted_ids[y * BM + i] : a.num_valid;
  }
  __syncthreads();

  const int role = (PH == 0) ? (nb < (MOE_I / BN) ? 0 : 1) : 2;
  const int sblk = (PH == 0) ? (nb < (MOE_I / BN) ? nb : nb - (MOE_I / BN)) : nb;
  const size_t pitch = (size_t)NT * NWT + a.pad;
  const uint32_t* qm = a.qw + a.qoff[(size_t)e * 3 + role] + (size_t)sblk * 8 * pitch;

  bool m_act[MT];
  const _Float16* arow[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    const int mb = wm * MT + m;
    m_act[m] = s_tok[mb * 16] < a.num_valid;
    arow[m] = a.A + ((size_t)y * BM + mb * 16 + l16) * KDIM;
  }

  auto loadA = [&](u32x4 dst[MT][4], int k0) {
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (m_act[m]) {
#pragma unroll
        for (int c = 0; c < 4; ++c) dst[m][c] = *(const u32x4*)(arow[m] + k0 + 8 * c);
      }
    }
  };
  auto loadB = [&](uint32_t* br, int k0) {
#pragma unroll
    for (int bi = 0; bi < B_ITERS; ++bi) {
      const int vtid = tid + bi * NTHR;
      const int n2 = vtid & 127, r = n2 & 15, h2i = vtid >> 7;
      const uint32_t* tile = qm + (size_t)(n2 >> 4) * pitch + (size_t)(k0 / 16 + h2i) * NWT;
      if constexpr (K == 3) {
        const int kk = r >> 1;
        const u32x2a* p2 = reinterpret_cast<const u32x2a*>(tile + 3 * kk + (r & 1));
        const u32x2 pair = NT_LOAD ? __builtin_nontemporal_load(p2) : *p2;
        br[bi * 2 + 0] = pair[0];
        br[bi * 2 + 1] = pair[1];
      } else if constexpr (K == 2) {
        br[bi] = NT_LOAD ? __builtin_nontemporal_load(tile + r) : tile[r];
      }
    }
  };
  auto storeB = [&](int buf, const uint32_t* br) {
    uint32_t* bs = Bs[buf];
#pragma unroll
    for (int bi = 0; bi < B_ITERS; ++bi) {
      const int vtid = tid + bi * NTHR;
      const int n2 = vtid & 127, r = n2 & 15, h2i = vtid >> 7;
      uint32_t o[8];
      decode_row_k<K>(&br[bi * W_PER_ROW], r, o);
      *(u32x4*)(&bs[sw4(n2, 2 * h2i)]) = (u32x4){o[0], o[1], o[2], o[3]};
      *(u32x4*)(&bs[sw4(n2, 2 * h2i + 1)]) = (u32x4){o[4], o[5], o[6], o[7]};
      if constexpr (B_ITERS > 1) __builtin_amdgcn_sched_barrier(0);
    }
  };

  // Explicit register buffers (ZERO dynamic indexing -> ZERO scratch ops!)
  u32x4 aX[MT][4], aY[MT][4];
  uint32_t bX[B_ITERS * W_PER_ROW], bY[B_ITERS * W_PER_ROW], bZ[B_ITERS * W_PER_ROW];
  f32x8 acc[MT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[m][j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  const int n_base = wn * 64;
  auto step = [&](int s, int buf, const u32x4 curA[MT][4], u32x4 nxtA[MT][4], const uint32_t* b1, uint32_t* b2) {
    if (s + 1 < NS) loadA(nxtA, (s + 1) * BKS);
    if (s + 2 < NS) loadB(b2, (s + 2) * BKS);
    if (m_act[0]) {
      const uint32_t* bs = Bs[buf];
#pragma unroll
      for (int kc = 0; kc < 2; ++kc) {
        f16x16 af[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          const u32x4 x0 = curA[m][2 * kc], x1 = curA[m][2 * kc + 1];
          af[m] = __builtin_bit_cast(f16x16, (u32x8){x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]});
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int n = n_base + 16 * j + l16;
          const u32x4 y0 = *(const u32x4*)(&bs[sw4(n, 2 * kc)]);
          const u32x4 y1 = *(const u32x4*)(&bs[sw4(n, 2 * kc + 1)]);
          const f16x16 bf = __builtin_bit_cast(f16x16, (u32x8){y0[0], y0[1], y0[2], y0[3], y1[0], y1[1], y1[2], y1[3]});
#pragma unroll
          for (int m = 0; m < MT; ++m) {
            if (m == 0 || m_act[m]) {
              acc[m][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[m], bf, acc[m][j]);
            }
          }
          __builtin_amdgcn_sched_barrier(0);
        }
      }
    }
    if (s + 1 < NS) storeB(buf ^ 1, b1);
    __syncthreads();
  };

  loadA(aX, 0); loadB(bX, 0);
  if (NS > 1) loadB(bY, BKS);
  storeB(0, bX);
  __syncthreads();
  // Unroll by 6 = LCM(2 for aX/aY & buf, 3 for bX/bY/bZ) -> 100% static register addressing!
  for (int s = 0; s < NS; s += 6) {
    step(s + 0, 0, aX, aY, bY, bZ);
    if (s + 1 < NS) step(s + 1, 1, aY, aX, bZ, bX);
    if (s + 2 < NS) step(s + 2, 0, aX, aY, bX, bY);
    if (s + 3 < NS) step(s + 3, 1, aY, aX, bY, bZ);
    if (s + 4 < NS) step(s + 4, 0, aX, aY, bZ, bX);
    if (s + 5 < NS) step(s + 5, 1, aY, aX, bX, bY);
  }

  // Exact FWHT-128 epilogue: 64-pt in-register + 1 butterfly (col ^ 64) via dead Bs LDS!
  const int row0 = nb * 128;
  const int soff = (PH == 0) ? 0 : (2 * MOE_I);
  const int svoff = (PH == 0) ? 0 : (2 * MOE_I / 8);
  const __half* scl = a.scales + (size_t)e * SC_STRIDE + soff + row0 + n_base;
  const uint8_t* svp = a.sv + (size_t)e * SV_STRIDE + svoff + ((row0 + n_base) >> 3);
  float sc[4]; uint32_t neg = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    sc[j] = __half2float(scl[16 * j + l16]) * a.cb;
    neg |= ((uint32_t)(svp[2 * j + (l16 >> 3)] >> (l16 & 7)) & 1u) << j;
  }
  const int hrow = lane >> 4;
  float* ep = reinterpret_cast<float*>(&Bs[0][0]); // 5120 floats available (we use NWAVES*32*4 <= 1024 floats!)
  const int my_ep = (wave * 32 + lane) * 4;
  const int peer_ep = ((wave ^ 1) * 32 + lane) * 4;

#pragma unroll
  for (int m = 0; m < MT; ++m) {
    const int mb = wm * MT + m;
    const size_t prow = (size_t)y * BM + mb * 16;
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      const int rl = mb * 16 + 2 * ei + hrow;
      float v[4] = {0.f, 0.f, 0.f, 0.f};
      if (m_act[m]) {
        const float bias = VBIAS * a.S[prow + 2 * ei + hrow];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = (acc[m][j][ei] - bias) * sc[j];
        reg_stage4<1>(v); reg_stage4<2>(v);
        lane_stage4<1>(v, lane); lane_stage4<2>(v, lane); lane_stage4<4>(v, lane); lane_stage4<8>(v, lane);
      }
      *reinterpret_cast<float4*>(ep + my_ep) = make_float4(v[0], v[1], v[2], v[3]);
      __syncthreads();
      const float4 p4 = *reinterpret_cast<const float4*>(ep + peer_ep);
      __syncthreads();
      if (m_act[m] && s_tok[rl] < a.num_valid) {
        const float pv[4] = {p4.x, p4.y, p4.z, p4.w};
        uint16_t* eb = reinterpret_cast<uint16_t*>(&Bs[0][0]) + 2048; // byte +4096..20480 = 16,384 B (holds 64x128 BF16!)
        uint16_t* dst_lds = eb + (wm * 16 + 2 * ei + hrow) * 128 + n_base + l16;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float yv = ((wn == 0) ? (v[j] + pv[j]) : (pv[j] - v[j])) * INV128;
          if ((neg >> j) & 1u) yv = -yv;
          dst_lds[16 * j] = f32_to_bf16_rne(yv);
        }
      }
    }
    __syncthreads();
    // ONE single coalesced u32x4 (16B x 16 threads = 256B = 4 full 64B lines) pass for all 32 rows!
    {
      uint16_t* eb = reinterpret_cast<uint16_t*>(&Bs[0][0]) + 2048;
      const int sr_base = tid >> 4;
      const int sc_off  = (tid & 15) * 8;
#pragma unroll
      for (int r0 = 0; r0 < NWM * 16; r0 += (NTHR >> 4)) {
        const int r_loc = r0 + sr_base;
        const int s_wm = r_loc >> 4;
        const int s_mb = s_wm * MT + m;
        const int s_rl = s_mb * 16 + (r_loc & 15);
        if (s_tok[s_rl] < a.num_valid) {
          const size_t s_prow = (size_t)y * BM + s_rl;
          const u32x4 v4 = *reinterpret_cast<const u32x4*>(eb + r_loc * 128 + sc_off);
          *reinterpret_cast<u32x4*>(reinterpret_cast<uint16_t*>(a.Y) + s_prow * (size_t)N + row0 + sc_off) = v4;
        }
      }
    }
    if constexpr (MT > 1) __syncthreads();
  }
}

template <int K, int PH>
static void dispatch_gemm(int bmw, int nt_load, int max_g, hipStream_t st, ErnieGemmArgs a) {
  constexpr int N = (PH == 0) ? (2 * MOE_I) : MOE_H;
  if (bmw == 32) {
    if (nt_load) hipLaunchKernelGGL((k_hpf_ernie_gemm_n2<K, PH, 2, 1, 1>), dim3(N / BN, max_g), dim3(128), 0, st, a);
    else         hipLaunchKernelGGL((k_hpf_ernie_gemm_n2<K, PH, 2, 1, 0>), dim3(N / BN, max_g), dim3(128), 0, st, a);
  } else if (bmw == 64) {
    if (nt_load) hipLaunchKernelGGL((k_hpf_ernie_gemm_n2<K, PH, 4, 1, 1>), dim3(N / BN, max_g), dim3(256), 0, st, a);
    else         hipLaunchKernelGGL((k_hpf_ernie_gemm_n2<K, PH, 4, 1, 0>), dim3(N / BN, max_g), dim3(256), 0, st, a);
  } else {
    if (nt_load) hipLaunchKernelGGL((k_hpf_ernie_gemm<K, PH, 8, 1>), dim3(N / BN, max_g), dim3(256), 0, st, a);
    else         hipLaunchKernelGGL((k_hpf_ernie_gemm<K, PH, 8, 0>), dim3(N / BN, max_g), dim3(256), 0, st, a);
  }
}

torch::Tensor hpf5_ernie_forward(
    torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
    torch::Tensor qw, torch::Tensor qoff, torch::Tensor scales, torch::Tensor su, torch::Tensor sv,
    double cb, int64_t K, int64_t pad, int64_t E,
    torch::Tensor sorted_ids, torch::Tensor expert_ids, torch::Tensor ntpp,
    torch::Tensor grp_y, torch::Tensor grp_nblk, torch::Tensor grp_cnt,
    torch::Tensor A_buf, torch::Tensor Y_buf, torch::Tensor S_buf, torch::Tensor inv_buf, torch::Tensor nsat,
    int64_t bmw, int64_t nt_load) {
  TORCH_CHECK(K == 2 || K == 3, "hpf5_ernie supports K=2 and K=3");
  TORCH_CHECK(bmw == 32 || bmw == 64 || bmw == 128, "bmw must be 32, 64, or 128");
  TORCH_CHECK(E <= 512, "E must be <= 512");
  const int T = (int)x.size(0);
  const int topk = (int)topk_ids.size(1);
  const int nv = T * topk;
  const int Pmax = (int)sorted_ids.numel();
  const int max_blocks = Pmax / BM;
  const int max_nblk = (int)(bmw / BM);
  TORCH_CHECK(Pmax >= nv + (int)E * (BM - 1), "scratch Pmax too small");

  auto st = c10::hip::getCurrentHIPStream().stream();
  int32_t* sid = sorted_ids.data_ptr<int32_t>();
  int32_t* eid = expert_ids.data_ptr<int32_t>();
  int32_t* ntpp_p = ntpp.data_ptr<int32_t>();
  int32_t* gy_p = grp_y.data_ptr<int32_t>();
  int32_t* gn_p = grp_nblk.data_ptr<int32_t>();
  int32_t* gc_p = grp_cnt.data_ptr<int32_t>();
  _Float16* Ap = reinterpret_cast<_Float16*>(A_buf.data_ptr<at::Half>());
  __hip_bfloat16* Yp = reinterpret_cast<__hip_bfloat16*>(Y_buf.data_ptr<at::BFloat16>());
  float* Sp = S_buf.data_ptr<float>();
  int32_t* inv_p = inv_buf.data_ptr<int32_t>();
  unsigned* ns_p = reinterpret_cast<unsigned*>(nsat.data_ptr<int32_t>());

  // 0. Sort + pack into BM=16 blocks + run-length group into <= bmw
  hipLaunchKernelGGL(k_moe_align_ernie, dim3(1), dim3(512), 0, st,
                     topk_ids.data_ptr<int32_t>(), nv, (int)E,
                     sid, eid, ntpp_p, gy_p, gn_p, gc_p, max_nblk);

  const int pg = (Pmax + 3) / 4;
  const int tight_g = (int)E + (nv + (int)bmw - 1) / (int)bmw;
  const int max_g = tight_g < max_blocks ? tight_g : max_blocks;

  ErnieGemmArgs a;
  a.A = Ap; a.S = Sp;
  a.qw = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>());
  a.qoff = qoff.data_ptr<int64_t>();
  a.scales = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  a.sv = sv.data_ptr<uint8_t>();
  a.sorted_ids = sid; a.expert_ids = eid;
  a.grp_y = gy_p; a.grp_nblk = gn_p; a.grp_cnt = gc_p;
  a.Y = Yp;
  a.num_valid = nv; a.pad = (int)pad; a.cb = (float)cb;

  // 1. Act0: x -> A1 [P, 2048] f16 + S
  hipLaunchKernelGGL((k_hpf_ernie_act<0>), dim3(pg), dim3(128), 0, st,
                     reinterpret_cast<const __hip_bfloat16*>(x.data_ptr<at::BFloat16>()),
                     sid, eid, ntpp_p, su.data_ptr<uint8_t>(), topk, nv, Ap, Sp, ns_p);

  // 2. Gemm0: A1 @ W13 -> Y1 [P, 4096] bf16
  if (K == 3) dispatch_gemm<3, 0>((int)bmw, (int)nt_load, max_g, st, a);
  else        dispatch_gemm<2, 0>((int)bmw, (int)nt_load, max_g, st, a);

  // 3. Act1: Y1 -> SiTU-GLU -> A2 (reuses A_buf!) [P, 2048] f16 + S
  hipLaunchKernelGGL((k_hpf_ernie_act<1>), dim3(pg), dim3(128), 0, st,
                     Yp, sid, eid, ntpp_p, su.data_ptr<uint8_t>(), topk, nv, Ap, Sp, ns_p);

  // 4. Gemm1: A2 @ W2 -> Y2 (reuses first half of Y_buf!) [P, 2048] bf16
  if (K == 3) dispatch_gemm<3, 1>((int)bmw, (int)nt_load, max_g, st, a);
  else        dispatch_gemm<2, 1>((int)bmw, (int)nt_load, max_g, st, a);

  // 5. Inv + weighted reduce -> out [T, 2048] bf16
  auto out = torch::empty({T, MOE_H}, x.options());
  hipLaunchKernelGGL(k_hpf_ernie_inv, dim3((Pmax + 255) / 256), dim3(256), 0, st, sid, ntpp_p, nv, inv_p);
  hipLaunchKernelGGL(k_hpf_ernie_reduce, dim3(T), dim3(256), 0, st,
                     Yp, inv_p, topk_weights.data_ptr<float>(), topk,
                     reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()));
  return out;
}

PYBIND11_MODULE(hpf5_ernie, m) {
  m.def("hpf5_ernie_forward", &hpf5_ernie_forward, "HPF5 fused WMMA+SAD_HI MoE prefill for ERNIE-5.5 (K=2/3)");
}


