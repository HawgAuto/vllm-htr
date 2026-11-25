// htr_dense_prefill_v2.hip — High-BM (BM=128/64, BN=128, BKS=32) 3-stage pipelined SAD_HI + Wave32 WMMA
// fused on-the-fly K4/K5 dense prefill kernel for gfx1151 (RDNA 3.5).
// Eliminates E.dequant_w BF16 materialization + beats hipBLASLt on M = 65..16384.
// Exact identity: y = sv . H128(scale * cb * (Q_int . xt - 1534 * xs)) / sqrt(128) * tok + bias.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

typedef _Float16 f16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));

#define BN 128
#define BKS 32
#define RS 20
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
template <int MSK> __device__ __forceinline__ void lane_stage8(float v[8], int lane) {
  const bool up = (lane & MSK) != 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float o = shx<MSK>(v[j]);
    v[j] = up ? (o - v[j]) : (v[j] + o);
  }
}
template <int STEP> __device__ __forceinline__ void reg_stage8(float v[8]) {
#pragma unroll
  for (int j = 0; j < 8; j += 2 * STEP) {
#pragma unroll
    for (int k = 0; k < STEP; ++k) {
      const float a = v[j + k], b = v[j + k + STEP];
      v[j + k] = a + b;
      v[j + k + STEP] = a - b;
    }
  }
}
__device__ __forceinline__ int sw4(int n, int c4) { return n * RS + ((c4 ^ (n & 3)) << 2); }
__device__ __forceinline__ uint16_t f32_to_bf16_rne(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ uint32_t tv16x2(uint32_t a, uint32_t b) {
  return __builtin_amdgcn_sad_hi_u8(b * GEN_A + GEN_B, 0u, __builtin_amdgcn_sad_u8(a * GEN_A + GEN_B, 0u, 0x64006400u));
}

template <int M> __device__ __forceinline__ uint32_t win4(uint32_t hi, uint32_t mid, uint32_t lo) {
  if constexpr (M <= 3) return __builtin_amdgcn_perm(mid, lo, (uint32_t)M | ((uint32_t)(M + 1) << 8) | 0x0c0c0000u);
  else return __builtin_amdgcn_perm(hi, mid, (uint32_t)(M - 4) | ((uint32_t)(M - 3) << 8) | 0x0c0c0000u);
}
template <int I> __device__ __forceinline__ uint32_t sym4(uint32_t hi, uint32_t mid, uint32_t lo, uint32_t hi4, uint32_t mid4, uint32_t lo4) {
  if constexpr ((I & 1) == 0) return win4<I / 2>(hi, mid, lo);
  else return win4<I / 2>(hi4, mid4, lo4);
}
__device__ __forceinline__ void decode_row_k4(uint32_t hi, uint32_t mid, uint32_t lo, uint32_t o[8]) {
  const uint32_t lo4 = __builtin_amdgcn_alignbit(mid, lo, 4), mid4 = __builtin_amdgcn_alignbit(hi, mid, 4), hi4 = hi >> 4;
#define D4_PAIR(JJ) o[JJ] = tv16x2(sym4<15 - 2 * (JJ)>(hi, mid, lo, hi4, mid4, lo4), sym4<14 - 2 * (JJ)>(hi, mid, lo, hi4, mid4, lo4));
  D4_PAIR(0) D4_PAIR(1) D4_PAIR(2) D4_PAIR(3) D4_PAIR(4) D4_PAIR(5) D4_PAIR(6) D4_PAIR(7)
#undef D4_PAIR
}

template <int B> __device__ __forceinline__ uint32_t ext16_k5(const uint32_t V[4]) {
  constexpr int w = B >> 5, off = B & 31;
  if constexpr (off <= 16) return (V[w] >> (16 - off)) & 0xFFFFu;
  else return __builtin_amdgcn_alignbit(V[w], V[w + 1], 48 - off) & 0xFFFFu;
}
__device__ __forceinline__ void decode_row_k5(const uint32_t V[4], int odd, uint32_t o[8]) {
  if (odd == 0) {
#define D5E_PAIR(JJ) o[JJ] = tv16x2(ext16_k5<21 + 10 * (JJ)>(V), ext16_k5<26 + 10 * (JJ)>(V));
    D5E_PAIR(0) D5E_PAIR(1) D5E_PAIR(2) D5E_PAIR(3) D5E_PAIR(4) D5E_PAIR(5) D5E_PAIR(6) D5E_PAIR(7)
#undef D5E_PAIR
  } else {
#define D5O_PAIR(JJ) o[JJ] = tv16x2(ext16_k5<5 + 10 * (JJ)>(V), ext16_k5<10 + 10 * (JJ)>(V));
    D5O_PAIR(0) D5O_PAIR(1) D5O_PAIR(2) D5O_PAIR(3) D5O_PAIR(4) D5O_PAIR(5) D5O_PAIR(6) D5O_PAIR(7)
#undef D5O_PAIR
  }
}

template <int K, int MAX_NBLK>
__device__ __forceinline__ void dense_prefill_impl(
    const uint32_t* __restrict__ qw,
    const _Float16* __restrict__ xt,
    const float* __restrict__ xs,
    const float* __restrict__ tok,
    const __half* __restrict__ scales,
    const uint8_t* __restrict__ sv,
    const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ Y,
    int M, int mt, int nt, float cb) {

  constexpr int NWT = 8 * K;
  constexpr int LBMW = MAX_NBLK * 16;
  constexpr int NTHR = MAX_NBLK * 32;
  constexpr int B_ITERS = 256 / NTHR;
  constexpr int NW = (K == 4) ? 3 : 4;

  __shared__ __attribute__((aligned(16))) uint32_t Bs[2][BN * RS];

  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int nb = blockIdx.x;
  const int m0 = blockIdx.y * LBMW;
  const int N_out = mt * 16;
  const int K_in = nt * 16;
  const int NS = nt >> 1;

  const int wrow = m0 + wave * 16;
  const bool wave_active = (wrow < M);
  const int a_row_idx = (wrow + l16 < M) ? (wrow + l16) : (M - 1);
  const _Float16* arow = xt + (size_t)a_row_idx * K_in;

  const int n2 = tid & 127;
  const int r = n2 & 15;
  const int q = r >> 1;
  const int odd = r & 1;
  const size_t strip_pitch = (size_t)nt * NWT;
  const uint32_t* strip = qw + ((size_t)nb * 8 + (n2 >> 4)) * strip_pitch;

  u32x4 aX[4], aY[4], aZ[4];
  uint32_t bX[B_ITERS * NW], bY[B_ITERS * NW], bZ[B_ITERS * NW];

  auto loadA = [&](u32x4* dst, int k0) {
    if (wave_active) {
#pragma unroll
      for (int c = 0; c < 4; ++c) dst[c] = *(const u32x4*)(arow + k0 + 8 * c);
    }
  };

  auto loadB = [&](uint32_t* br, int k0) {
#pragma unroll
    for (int bi = 0; bi < B_ITERS; ++bi) {
      const int h2i = (B_ITERS == 1) ? (tid >> 7) : bi;
      const uint32_t* tile = strip + (size_t)((k0 >> 4) + h2i) * NWT;
      if constexpr (K == 4) {
        const u32x2 pr = __builtin_nontemporal_load((const u32x2*)(tile + 2 * r));
        br[bi * 3 + 0] = __builtin_nontemporal_load(tile + (2 * r + 31) % 32);
        br[bi * 3 + 1] = pr[0];
        br[bi * 3 + 2] = pr[1];
      } else {
        if (odd == 0) {
          br[bi * 4 + 0] = __builtin_nontemporal_load(tile + (q * 5 + 39) % 40);
          br[bi * 4 + 1] = __builtin_nontemporal_load(tile + q * 5 + 0);
          br[bi * 4 + 2] = __builtin_nontemporal_load(tile + q * 5 + 1);
          br[bi * 4 + 3] = __builtin_nontemporal_load(tile + q * 5 + 2);
        } else {
          br[bi * 4 + 0] = __builtin_nontemporal_load(tile + q * 5 + 2);
          br[bi * 4 + 1] = __builtin_nontemporal_load(tile + q * 5 + 3);
          br[bi * 4 + 2] = __builtin_nontemporal_load(tile + q * 5 + 4);
          br[bi * 4 + 3] = 0u;
        }
      }
    }
  };

  auto storeB = [&](int buf, const uint32_t* br) {
    uint32_t* bs = Bs[buf];
#pragma unroll
    for (int bi = 0; bi < B_ITERS; ++bi) {
      const int h2i = (B_ITERS == 1) ? (tid >> 7) : bi;
      uint32_t o[8];
      if constexpr (K == 4) {
        decode_row_k4(br[bi * 3 + 0], br[bi * 3 + 1], br[bi * 3 + 2], o);
      } else {
        uint32_t V[4] = {br[bi * 4 + 0], br[bi * 4 + 1], br[bi * 4 + 2], br[bi * 4 + 3]};
        decode_row_k5(V, odd, o);
      }
      *(u32x4*)(&bs[sw4(n2, 2 * h2i)]) = (u32x4){o[0], o[1], o[2], o[3]};
      *(u32x4*)(&bs[sw4(n2, 2 * h2i + 1)]) = (u32x4){o[4], o[5], o[6], o[7]};
    }
  };

  f32x8 acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  auto step = [&](int s, const u32x4* cur, const uint32_t* b1, u32x4* nx, uint32_t* b2) {
    const int buf = s & 1;
    if (s + 2 < NS) { loadA(nx, (s + 2) * BKS); loadB(b2, (s + 2) * BKS); }
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
        }
      }
    }
    if (s + 1 < NS) storeB(buf ^ 1, b1);
    __syncthreads();
  };

  loadA(aX, 0); loadB(bX, 0);
  if (NS > 1) { loadA(aY, BKS); loadB(bY, BKS); }
  storeB(0, bX);
  __syncthreads();

  for (int s = 0; s < NS; s += 3) {
    step(s, aX, bY, aZ, bZ);
    if (s + 1 < NS) step(s + 1, aY, bZ, aX, bX);
    if (s + 2 < NS) step(s + 2, aZ, bX, aY, bY);
  }

  if (!wave_active) return;

  const int row0 = nb * 128;
  const __half* scl = scales + row0;
  const uint8_t* svp = sv + (row0 >> 3);
  float sc[8];
  float b_add[8];
  uint32_t neg = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    sc[j] = __half2float(scl[16 * j + l16]) * cb;
    neg |= ((uint32_t)(svp[2 * j + (l16 >> 3)] >> (l16 & 7)) & 1u) << j;
    b_add[j] = (bias != nullptr) ? __uint_as_float(((uint32_t)bias[row0 + 16 * j + l16]) << 16) : 0.0f;
  }

  const int hrow = lane >> 4;
#pragma unroll
  for (int ei = 0; ei < 8; ++ei) {
    const int gr = wrow + 2 * ei + hrow;
    if (gr < M) {
      const float xbias = VBIAS * xs[gr];
      const float tscale = INV128 * tok[gr];
      float v[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) v[j] = (acc[j][ei] - xbias) * sc[j];
      reg_stage8<1>(v); reg_stage8<2>(v); reg_stage8<4>(v);
      lane_stage8<1>(v, lane); lane_stage8<2>(v, lane); lane_stage8<4>(v, lane); lane_stage8<8>(v, lane);
      uint16_t* yr = Y + (size_t)gr * N_out + row0 + l16;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float yv = v[j] * tscale;
        if ((neg >> j) & 1u) yv = -yv;
        yr[16 * j] = f32_to_bf16_rne(yv + b_add[j]);
      }
    }
  }
}

extern "C" __global__ __launch_bounds__(128) __attribute__((amdgpu_waves_per_eu(8)))
void dense_prefill_k4_bm64(
    const uint32_t* qw, const _Float16* xt, const float* xs, const float* tok,
    const __half* scales, const uint8_t* sv, const uint16_t* bias, uint16_t* Y,
    int M, int mt, int nt, float cb) {
  dense_prefill_impl<4, 4>(qw, xt, xs, tok, scales, sv, bias, Y, M, mt, nt, cb);
}

extern "C" __global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8)))
void dense_prefill_k4_bm128(
    const uint32_t* qw, const _Float16* xt, const float* xs, const float* tok,
    const __half* scales, const uint8_t* sv, const uint16_t* bias, uint16_t* Y,
    int M, int mt, int nt, float cb) {
  dense_prefill_impl<4, 8>(qw, xt, xs, tok, scales, sv, bias, Y, M, mt, nt, cb);
}

extern "C" __global__ __launch_bounds__(128) __attribute__((amdgpu_waves_per_eu(8)))
void dense_prefill_k5_bm64(
    const uint32_t* qw, const _Float16* xt, const float* xs, const float* tok,
    const __half* scales, const uint8_t* sv, const uint16_t* bias, uint16_t* Y,
    int M, int mt, int nt, float cb) {
  dense_prefill_impl<5, 4>(qw, xt, xs, tok, scales, sv, bias, Y, M, mt, nt, cb);
}

extern "C" __global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8)))
void dense_prefill_k5_bm128(
    const uint32_t* qw, const _Float16* xt, const float* xs, const float* tok,
    const __half* scales, const uint8_t* sv, const uint16_t* bias, uint16_t* Y,
    int M, int mt, int nt, float cb) {
  dense_prefill_impl<5, 8>(qw, xt, xs, tok, scales, sv, bias, Y, M, mt, nt, cb);
}
