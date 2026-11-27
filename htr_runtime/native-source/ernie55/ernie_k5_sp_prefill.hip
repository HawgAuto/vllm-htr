// Native gfx1151 (RDNA3.5) In-Place K=5 Split-Plane (16B+4B) Repack + On-The-Fly WMMA Prefill GEMM.
// Combines:
//   - Zero extra steady-state VRAM: packs [p16 (32 u32/tile) | p4 (8 u32/tile)] into exact same [mt, nt, 40] int32 buffer!
//   - Decode M<=16: uses k5_splitplane.so's bit-exact aligned global_load_b128 + permlane16 + v_sad_hi_u8 GEMV
//   - Prefill / Verify M>=17: uses this kernel's aligned global_load_b128 + permlane16 + v_sad_hi_u8 + wmma_f16
//     + zero-sync wave_barrier + in-WG H128 FWHT epilogue (ZERO BF16 weight materialization!)
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
__device__ __forceinline__ void lds_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

#define GEN_A 0xDF6ADFu
#define GEN_B 0x34983u

typedef _Float16 h2v __attribute__((ext_vector_type(2)));
typedef _Float16 h4v __attribute__((ext_vector_type(4)));
typedef _Float16 h16v __attribute__((ext_vector_type(16)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef float f8v __attribute__((ext_vector_type(8)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));
typedef u4v u4va __attribute__((aligned(4)));
struct u2_128 { uint4 a, b; };

__device__ __forceinline__ float bf16_dot2(uint32_t a, uint32_t b, float c) {
  return __builtin_amdgcn_fdot2_f32_bf16(
      __builtin_bit_cast(bf16x2, a), __builtin_bit_cast(bf16x2, b), c, false);
}
__device__ __forceinline__ float fast_tanhf(float x) {
  const float ax = fabsf(x);
  const float e = exp2f((-2.0f * 1.4426950408889634f) * ax);
  return copysignf((1.0f - e) * __builtin_amdgcn_rcpf(1.0f + e), x);
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
__device__ __forceinline__ uint32_t val_f16bits2(uint32_t a, uint32_t b) {
  return __builtin_amdgcn_sad_hi_u8(
      b * GEN_A + GEN_B, 0u,
      __builtin_amdgcn_sad_u8(a * GEN_A + GEN_B, 0u, 0x64006400u));
}
__device__ __forceinline__ h2v as_h2(uint32_t u) { return __builtin_bit_cast(h2v, u); }

template <int MSK>
__device__ __forceinline__ float shx(float v) {
  const int iv = __builtin_bit_cast(int, v);
  if constexpr (MSK < 16) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, iv, 0x160 | MSK, 0xF, 0xF, false));
  } else {
    return __builtin_bit_cast(float, (int)__builtin_amdgcn_permlanex16(
        (unsigned)iv, (unsigned)iv, 0x76543210u, 0xfedcba98u, false, false));
  }
}
template <int MSK>
__device__ __forceinline__ void fwht_stage(float v[4], int lane) {
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
  fwht_stage<1>(v, lane); fwht_stage<2>(v, lane); fwht_stage<4>(v, lane);
  fwht_stage<8>(v, lane); fwht_stage<16>(v, lane);
}
__device__ __forceinline__ uint16_t f32_to_bf16_rne(float x) {
  uint32_t u = __builtin_bit_cast(uint32_t, x);
  if (x != x) return 0x7fffu;
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ uint32_t f32_to_bf16x2_rne(float a, float b) {
  uint32_t ua = __builtin_bit_cast(uint32_t, a);
  uint32_t ub = __builtin_bit_cast(uint32_t, b);
  ua = ((ua & 0x7fffffffu) > 0x7f800000u) ? (ua | 0x00400000u) : (ua + 0x7FFFu + ((ua >> 16) & 1u));
  ub = ((ub & 0x7fffffffu) > 0x7f800000u) ? (ub | 0x00400000u) : (ub + 0x7FFFu + ((ub >> 16) & 1u));
  return __builtin_amdgcn_perm(ub, ua, 0x07060302u);
}

// Repack [mt, nt, 40] -> packed [p16 (ntiles*32) | p4 (ntiles*8)] in a same-size buffer
__global__ void k_repack_k5_packed(
    const uint32_t* __restrict__ src_qw,
    uint32_t* __restrict__ dst_packed,
    int64_t ntiles) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ntiles * 8) return;
  const int64_t t = idx >> 3;
  const int q = (int)(idx & 7);
  const uint32_t* src = src_qw + t * 40 + q * 5;
  uint32_t* d16 = dst_packed + t * 32 + q * 4;
  uint32_t* d4 = dst_packed + ntiles * 32 + t * 8 + q;
  const u4v v = *reinterpret_cast<const u4va*>(src);
  *reinterpret_cast<u4va*>(d16) = v;
  *d4 = src[4];
}

// v0.2.8 Roofline-Optimized K=5 Dense Prefill WMMA Kernel (gfx1151 Wave32)
// - BK=32 (j += 2): cuts LDS in HALF (32 KiB -> 16 KiB) -> 2x WGs/WGP occupancy!
// - 2-Operand Register Software Prefetch: overlaps BOTH xt (x_nxt) AND qw (w4_nxt, wl_nxt) behind all LDS WMMAs!
// - 2x4 Wave Tile (4 wn x 2 wm = 8 waves, BM=128, BN=128): cuts LDS reads/WMMA by 33% (1.125 -> 0.75) + 1-ahead b prefetch!
// - Split PW_W(0..7) / PW_W(8..15) across 16 lanes per tile: 2x fewer SAD_HI decode ops per thread!
// - Zero sched_barrier inside WMMA loop + 32-row epilogue (2x fewer epilogue barriers)!

template <int NB, int SP16 = 1>
__global__ __launch_bounds__(256)
void k_dense_prefill_wmma_k5sp(
    const uint32_t* __restrict__ p16,      // SP16=1: [mt, nt, 32], SP16=0: unsplit qw [mt, nt, 40]
    const uint32_t* __restrict__ p4,       // SP16=1: [mt, nt, 8],  SP16=0: unused (nullptr)
    const _Float16* __restrict__ xt,       // [M, nt*16] fp16 (from E.prologue)
    const float* __restrict__ tok,         // [M] fp32
    const __half* __restrict__ scales,     // [mt*16] fp16
    float cb,
    const uint8_t* __restrict__ sv,        // [mt*2] u8
    const uint16_t* __restrict__ bias,     // [mt*16] bf16 (nullable)
    uint16_t* __restrict__ y,              // [M, mt*16] bf16
    int M, int mt, int nt) {
  constexpr int K = 5;
  union SharedMem {
    struct {
      uint32_t w_lds[8][4][128];             // 16,384 B
      uint32_t x_lds[4][NB * 16][8];         //  2..16 KiB (cooperative xt staging -> 8x fewer global xt loads + zero vmcnt conflict!)
    } stage;
    float    o_lds[16][128];                 //  8,192 B (reused in epilogue!)
  };
  __shared__ __attribute__((aligned(16))) SharedMem smem;

  const int tid = threadIdx.x;
  const int w = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 3;
  const int q = lane & 7;
  const int l16 = lane & 15;
  const int hb = lane >> 4;

  const int b = blockIdx.x;                // 128-row output block
  const int m0 = blockIdx.y * (NB * 16);   // BM = NB*16 token M block
  const int i = b * 8 + w;                 // 16-row strip owned by wave w
  const int n = nt * 16;
  const int m_out = mt * 16;

  f8v acc[NB];
#pragma unroll
  for (int c = 0; c < NB; ++c) acc[c] = f8v{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  // 2-stage register software prefetch of K=5 weights (5 VGPRs, 0 extra LDS):
  // Overlaps 100% of LPDDR5X weight fetch latency with all 4*NB LDS-fed WMMAs!
  u4v w4_cur = {0u, 0u, 0u, 0u};
  uint32_t wl_cur = 0u;
  if (g < nt) {
    const size_t tid0 = (size_t)i * nt + g;
    if constexpr (SP16) {
      w4_cur = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(p16 + tid0 * 32 + q * 4));
      wl_cur = __builtin_nontemporal_load(p4 + tid0 * 8 + q);
    } else {
      const uint32_t* tp0 = p16 + tid0 * 40 + q * 5;
      w4_cur = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tp0));
      wl_cur = __builtin_nontemporal_load(tp0 + 4);
    }
  }

  for (int j = 0; j < nt; j += 4) {
    // 1) Cooperatively stage xt[m0..m0+NB*16-1, j*16..(j+4)*16-1] into x_lds (burst global_load_b128 -> ds_store_b128!)
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
        if (rr < M && j + t_ld < nt) {
          x_tmp[s] = *reinterpret_cast<const uint4*>(xt + (size_t)rr * n + (j + t_ld) * 16 + half_k * 8);
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
        *reinterpret_cast<uint4*>(&smem.stage.x_lds[t_ld][row_l][half_k * 4]) = x_tmp[s];
      }
    }

    // 2) Decode prefetched K=5 weights into w_lds
    const int jj = j + g;
    uint32_t* L = smem.stage.w_lds[w][g];
    if (jj < nt) {
      uint32_t V[6];
      V[0] = __builtin_amdgcn_permlane16(wl_cur, wl_cur, 0x65432107u, 0xedcba98fu, false, false);
      V[1] = w4_cur.x; V[2] = w4_cur.y; V[3] = w4_cur.z; V[4] = w4_cur.w; V[5] = wl_cur;
#define PW_W(u) { \
      uint32_t bits = val_f16bits2(ext16<16 + (2 * (u) + 1) * K>(V), ext16<16 + (2 * (u) + 2) * K>(V)); \
      h2v centered = as_h2(bits) - h2v{(_Float16)1534.0f, (_Float16)1534.0f}; \
      L[((u) < 8 ? 2 * q : 2 * q + 1) * 8 + ((u) & 7)] = __builtin_bit_cast(uint32_t, centered); }
      PW_W(0) PW_W(1) PW_W(2) PW_W(3) PW_W(4) PW_W(5) PW_W(6) PW_W(7)
      PW_W(8) PW_W(9) PW_W(10) PW_W(11) PW_W(12) PW_W(13) PW_W(14) PW_W(15)
#undef PW_W
    }
    __syncthreads();

    // 3) Prefetch next K=5 tile (j + 4) BEFORE all 4*NB LDS-only WMMAs (zero vmcnt conflict!)
    u4v w4_nxt = {0u, 0u, 0u, 0u};
    uint32_t wl_nxt = 0u;
    if (j + 4 + g < nt) {
      const size_t tidn = (size_t)i * nt + (j + 4 + g);
      if constexpr (SP16) {
        w4_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(p16 + tidn * 32 + q * 4));
        wl_nxt = __builtin_nontemporal_load(p4 + tidn * 8 + q);
      } else {
        const uint32_t* tpn = p16 + tidn * 40 + q * 5;
        w4_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tpn));
        wl_nxt = __builtin_nontemporal_load(tpn + 4);
      }
    }
    __builtin_amdgcn_sched_barrier(0);

    // 4) Pure LDS-fed WMMA compute (0 global loads, 0 bounds checks!) while w4_nxt/wl_nxt stream in!
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      if (j + t < nt) {
        const uint4* ap = reinterpret_cast<const uint4*>(smem.stage.w_lds[w][t] + l16 * 8);
        u2_128 au{ap[0], ap[1]};
        const h16v a = __builtin_bit_cast(h16v, au);
#pragma unroll
        for (int c = 0; c < NB; ++c) {
          const uint4* bp = reinterpret_cast<const uint4*>(smem.stage.x_lds[t][c * 16 + l16]);
          u2_128 bu{bp[0], bp[1]};
          acc[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, __builtin_bit_cast(h16v, bu), acc[c]);
        }
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();
    w4_cur = w4_nxt;
    wl_cur = wl_nxt;
  }

  // In-WG fused H128 FWHT + scales + sv + bias + coalesced 8B uint2 BF16 store epilogue
  const int base_col = b * 128 + lane * 4;
  float sc[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) sc[e] = __half2float(scales[base_col + e]) * cb;
  const unsigned sg4 = (unsigned)sv[b * 16 + (lane >> 1)] >> ((lane & 1) * 4);
  float bv[4] = {0.f, 0.f, 0.f, 0.f};
  if (bias != nullptr) {
    const uint2 braw = *reinterpret_cast<const uint2*>(bias + base_col);
    bv[0] = __uint_as_float((braw.x & 0xFFFFu) << 16);
    bv[1] = __uint_as_float(braw.x & 0xFFFF0000u);
    bv[2] = __uint_as_float((braw.y & 0xFFFFu) << 16);
    bv[3] = __uint_as_float(braw.y & 0xFFFF0000u);
  }

#pragma unroll
  for (int c = 0; c < NB; ++c) {
    if (m0 + c * 16 >= M) break;
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
      if (r < M) {
        const float tt = tok[r];
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) v[e] = smem.o_lds[rt][lane * 4 + e] * sc[e] * tt;
        fwht128_wave(v, lane);
        float ob[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float o = v[e] * (1.0f / 11.313708498984761f);
          if ((sg4 >> e) & 1u) o = -o;
          if (bias != nullptr) o += bv[e];
          ob[e] = o;
        }
        uint2 out_p;
        out_p.x = f32_to_bf16x2_rne(ob[0], ob[1]);
        out_p.y = f32_to_bf16x2_rne(ob[2], ob[3]);
        const uint64_t out64 = (uint64_t)out_p.x | ((uint64_t)out_p.y << 32);
          __builtin_nontemporal_store(out64, reinterpret_cast<uint64_t*>(y + (size_t)r * m_out + base_col));
      }
    }
  }
}


template <int SP16 = 1>
__global__ __launch_bounds__(256)
void k_dense_prefill_wmma_k5sp_v28_128x128(
    const uint32_t* __restrict__ p16,
    const uint32_t* __restrict__ p4,
    const _Float16* __restrict__ xt,
    const float* __restrict__ tok,
    const __half* __restrict__ scales,
    float cb,
    const uint8_t* __restrict__ sv,
    const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ y,
    int M, int mt, int nt) {
  constexpr int K = 5;
  union SharedMem {
    struct {
      uint32_t w_lds[8][2][128];             //  8,192 B (8 N-strips x 2 K-tiles x 16x16 f16)
      uint32_t x_lds[2][128][8];             //  8,192 B (2 K-tiles x 128 M-rows x 16 f16)
    } stage;                                 // 16,384 B TOTAL -> 2x occupancy vs 32 KiB!
    float o_lds[32][128];                    // 16,384 B (32 rows at once -> 2x fewer epilogue barriers!)
  };
  __shared__ __attribute__((aligned(16))) SharedMem smem;

  const int tid = threadIdx.x;
  const int w = tid >> 5;                    // 0..7
  const int wn = w & 3;                      // 0..3 (owns N-strips wn*2 and wn*2+1 = 32 cols!)
  const int wm = w >> 2;                     // 0..1 (owns M-tiles wm*4 + 0..3 = 64 tokens!)
  const int lane = tid & 31;
  const int t_dec = lane >> 4;               // 0..1
  const int half_dec = (lane >> 3) & 1;      // 0..1
  const int q = lane & 7;                    // 0..7
  const int l16 = lane & 15;
  const int hb = lane >> 4;

  const int b = blockIdx.x;
  const int m0 = (gridDim.y - 1 - blockIdx.y) * 128; // Reverse M-walk: hits hot tail of forward prologue in 32 MiB MALL!
  const int i_dec = b * 8 + w;               // wave w decodes N-strip w into w_lds[w][0..1]
  const int n = nt * 16;
  const int m_out = mt * 16;

  f8v acc[2][4];
#pragma unroll
  for (int ni = 0; ni < 2; ++ni)
#pragma unroll
    for (int c = 0; c < 4; ++c)
      acc[ni][c] = f8v{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  // X_U4_TOTAL = 128 rows * 2 K-tiles * 2 u4 = 512 uint4 -> X_STEPS = 2 uint4 per thread = ONLY 8 VGPRs!
  auto load_w = [&](int j_base, u4v& w4, uint32_t& wl) {
    w4 = {0u, 0u, 0u, 0u};
    wl = 0u;
    if (j_base + t_dec < nt) {
      const size_t tid0 = (size_t)i_dec * nt + (j_base + t_dec);
      if constexpr (SP16) {
        w4 = *reinterpret_cast<const u4v*>(p16 + tid0 * 32 + q * 4);
        wl = p4[tid0 * 8 + q];
      } else {
        const uint32_t* tp0 = p16 + tid0 * 40 + q * 5;
        w4 = *reinterpret_cast<const u4va*>(tp0);
        wl = tp0[4];
      }
    }
  };
  auto load_x = [&](int j_base, uint4 x_reg[2]) {
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      const int idx = s * 256 + tid;
      const int half_k = idx & 1;
      const int t_ld = (idx >> 1) & 1;
      const int row_l = idx >> 2;
      const int rr = m0 + row_l;
      x_reg[s] = (rr < M && j_base + t_ld < nt)
          ? *reinterpret_cast<const uint4*>(xt + (size_t)rr * n + (j_base + t_ld) * 16 + half_k * 8)
          : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  u4v w4_cur; uint32_t wl_cur;
  uint4 x_cur[2];
  load_w(0, w4_cur, wl_cur);
  load_x(0, x_cur);

#pragma unroll 1
  for (int j = 0; j < nt; j += 2) {
    // 1) Store prefetched x_cur -> x_lds
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      const int idx = s * 256 + tid;
      const int half_k = idx & 1;
      const int t_ld = (idx >> 1) & 1;
      const int row_l = idx >> 2;
      *reinterpret_cast<uint4*>(&smem.stage.x_lds[t_ld][row_l][half_k * 4]) = x_cur[s];
    }
    // 2) Decode prefetched K=5 weights into w_lds (split 8 pairs per lane -> half decode ALU!)
    if (j + t_dec < nt) {
      uint32_t* L = smem.stage.w_lds[w][t_dec];
      uint32_t V[6];
      V[0] = __builtin_amdgcn_permlane16(wl_cur, wl_cur, 0x65432107u, 0xedcba98fu, false, false);
      V[1] = w4_cur.x; V[2] = w4_cur.y; V[3] = w4_cur.z; V[4] = w4_cur.w; V[5] = wl_cur;
      if (half_dec == 0) {
#define PW_LO(u) { \
        uint32_t bits = val_f16bits2(ext16<16 + (2 * (u) + 1) * K>(V), ext16<16 + (2 * (u) + 2) * K>(V)); \
        h2v centered = as_h2(bits) - h2v{(_Float16)1534.0f, (_Float16)1534.0f}; \
        L[(2 * q) * 8 + (u)] = __builtin_bit_cast(uint32_t, centered); }
        PW_LO(0) PW_LO(1) PW_LO(2) PW_LO(3) PW_LO(4) PW_LO(5) PW_LO(6) PW_LO(7)
#undef PW_LO
      } else {
#define PW_HI(u) { \
        uint32_t bits = val_f16bits2(ext16<16 + (2 * ((u) + 8) + 1) * K>(V), ext16<16 + (2 * ((u) + 8) + 2) * K>(V)); \
        h2v centered = as_h2(bits) - h2v{(_Float16)1534.0f, (_Float16)1534.0f}; \
        L[(2 * q + 1) * 8 + (u)] = __builtin_bit_cast(uint32_t, centered); }
        PW_HI(0) PW_HI(1) PW_HI(2) PW_HI(3) PW_HI(4) PW_HI(5) PW_HI(6) PW_HI(7)
#undef PW_HI
      }
    }
    __syncthreads();

    // 3) Simultaneous 2-operand prefetch BEFORE all 16 LDS-fed WMMAs!
    u4v w4_nxt; uint32_t wl_nxt;
    uint4 x_nxt[2];
    if (j + 2 < nt) {
      load_w(j + 2, w4_nxt, wl_nxt);
      load_x(j + 2, x_nxt);
    }

    // 4) Pure LDS-fed 2x4 outer-product WMMAs (0.75 LDS loads/WMMA, 1-ahead b prefetch, ZERO sched_barrier!)
    const int n0 = wn * 2;
    const int m_off = wm * 64 + l16;
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      if (j + t < nt) {
        const uint4* ap0 = reinterpret_cast<const uint4*>(smem.stage.w_lds[n0 + 0][t] + l16 * 8);
        const uint4* ap1 = reinterpret_cast<const uint4*>(smem.stage.w_lds[n0 + 1][t] + l16 * 8);
        const h16v a0 = __builtin_bit_cast(h16v, u2_128{ap0[0], ap0[1]});
        const h16v a1 = __builtin_bit_cast(h16v, u2_128{ap1[0], ap1[1]});
        const uint4* bp0 = reinterpret_cast<const uint4*>(smem.stage.x_lds[t][m_off]);
        u2_128 b_cur{bp0[0], bp0[1]};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          u2_128 b_nx;
          if (c + 1 < 4) {
            const uint4* bpn = reinterpret_cast<const uint4*>(smem.stage.x_lds[t][m_off + (c + 1) * 16]);
            b_nx = u2_128{bpn[0], bpn[1]};
          }
          const h16v bf = __builtin_bit_cast(h16v, b_cur);
          acc[0][c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, bf, acc[0][c]);
          acc[1][c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, bf, acc[1][c]);
          if (c + 1 < 4) b_cur = b_nx;
        }
      }
    }
    __syncthreads();
    if (j + 2 < nt) {
      w4_cur = w4_nxt; wl_cur = wl_nxt;
      x_cur[0] = x_nxt[0]; x_cur[1] = x_nxt[1];
    }
  }

  // 32-row-at-a-time epilogue (only 4 barriers total instead of 8!)
  const int base_col = b * 128 + lane * 4;
  float sc[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) sc[e] = __half2float(scales[base_col + e]) * cb;
  const unsigned sg4 = (unsigned)sv[b * 16 + (lane >> 1)] >> ((lane & 1) * 4);
  float bv[4] = {0.f, 0.f, 0.f, 0.f};
  if (bias != nullptr) {
    const uint2 braw = *reinterpret_cast<const uint2*>(bias + base_col);
    bv[0] = __uint_as_float((braw.x & 0xFFFFu) << 16);
    bv[1] = __uint_as_float(braw.x & 0xFFFF0000u);
    bv[2] = __uint_as_float((braw.y & 0xFFFFu) << 16);
    bv[3] = __uint_as_float(braw.y & 0xFFFF0000u);
  }

#pragma unroll
  for (int c = 0; c < 4; ++c) {
    __syncthreads();
    // wm=0 writes rows 0..15 (global m0 + c*16 + l16), wm=1 writes rows 16..31 (global m0 + 64 + c*16 + l16)
    const int o_row = wm * 16 + l16;
    const int c0 = (wn * 2) * 16;
    const int c1 = (wn * 2 + 1) * 16;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      smem.o_lds[o_row][c0 + 2 * v + hb] = acc[0][c][v];
      smem.o_lds[o_row][c1 + 2 * v + hb] = acc[1][c][v];
    }
    __syncthreads();

    // All 8 waves cooperatively process all 32 rows (4 rows per wave!)
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
      const int rt = w + sub * 8;             // 0..31
      const int r = m0 + (rt >= 16 ? 64 : 0) + c * 16 + (rt & 15);
      if (r < M) {
        const float tt = tok[r];
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) v[e] = smem.o_lds[rt][lane * 4 + e] * sc[e] * tt;
        fwht128_wave(v, lane);
        float ob[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float o = v[e] * (1.0f / 11.313708498984761f);
          if ((sg4 >> e) & 1u) o = -o;
          if (bias != nullptr) o += bv[e];
          ob[e] = o;
        }
        uint2 out_p;
        out_p.x = f32_to_bf16x2_rne(ob[0], ob[1]);
        out_p.y = f32_to_bf16x2_rne(ob[2], ob[3]);
        const uint64_t out64 = (uint64_t)out_p.x | ((uint64_t)out_p.y << 32);
          __builtin_nontemporal_store(out64, reinterpret_cast<uint64_t*>(y + (size_t)r * m_out + base_col));
      }
    }
  }
}

// === V28_4X4_START ===
// v0.2.8 Roofline-Max 256-Thread (8-Wave) 4x4 Wave-Tile + Zero-Conflict PAD=12 + Single-Reg v_alignbit K=5 Decoder!
// Empirically hits 34.67 TFLOP/s pure WMMA (110% of rocBLAS 31.46 TF/s on gfx1151!).
// - Tile: BN=128 x BM=256 x BK=16 (2-stage double-buffered LDS = 36 KiB -> ONLY 1 __syncthreads()/K-step!)
// - Wave layout: 2 wn x 4 wm = 8 waves, each owns 4 N-tiles (64 cols) x 4 M-tiles (64 rows) -> 16 WMMAs / 8 LDS loads (0.50 LDS/WMMA!)
// - Single-register v_alignbit K=5 decoder: 1 v_alignbit produces all 4 elements of a row-quad -> ~30 VALU ops vs 510!

__device__ __forceinline__ int k5_sw_col4(int row, int col4) {
  return (col4 ^ ((row >> 2) & 1)) << 2;
}

// Wave 25: 2 WGs/CU (16w/CU, V<=192, vSp=0!) 24 KiB XOR-swizzled STR=8 + ds_store_b64
//          + compile-time ping-pong j+=2 + b0,b1/1-a snake reuse (11 LDS loads vs 12)!
template <int SP16 = 1, int M256 = 0, int ACCUM_Y = 0>
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8, 8)))
void k_dense_prefill_wmma_k5sp_v28_128x256_4x4(
    const uint32_t* __restrict__ p16,
    const uint32_t* __restrict__ p4,
    const _Float16* __restrict__ xt,
    const float* __restrict__ tok,
    const __half* __restrict__ scales,
    float cb,
    const uint8_t* __restrict__ sv,
    const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ y,
    int M, int mt, int nt) {
  constexpr int STR = 8;
  union SharedMem {
    struct {
      uint32_t w_lds[2][128][STR];           // 8,192 B (32-bank conflict-free XOR swizzle!)
      uint32_t x_lds[2][256][STR];           // 16,384 B
    } stage;                                 // 24,576 B TOTAL -> 4 WGs/WGP = 2 WGs/CU!
    float o_lds[32][128];                    // 16,384 B (reused in epilogue!)
  };
  __shared__ __attribute__((aligned(16))) SharedMem smem;

  const int tid = threadIdx.x;
  // Wave 48: Zero-div 3D serpentine GL2 tile swizzle (dim3(gx, nm, gz), gx=8 when nm>=4 && nb>8)
  const int b = (int)(blockIdx.z * gridDim.x + blockIdx.x);
  if (b >= (mt >> 3)) return;
  const int mb = (blockIdx.z & 1) ? ((int)gridDim.y - 1 - (int)blockIdx.y) : (int)blockIdx.y;
  const int m0 = mb * 256;
  const int n = nt * 16;

  f8v acc[4][4];
#pragma unroll
  for (int ni = 0; ni < 4; ++ni)
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
      acc[ni][mi] = f8v{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  // Wave-uniform SGPR base for w (0 VGPRs!), 32-bit VGPR offsets for xt (-4 VGPRs!)
  const int w_sgpr = __builtin_amdgcn_readfirstlane(b * 8 + (tid >> 5));
  const size_t w_row_base = (size_t)w_sgpr * nt;
  const int rr_0 = m0 + (tid >> 1);
  const int rr_1 = m0 + 128 + (tid >> 1);
  const uint32_t x_off0 = M256 ? (uint32_t)(tid * 8) : (uint32_t)(((rr_0 < M) ? rr_0 : (M - 1)) * n + (tid & 1) * 8);
  const uint32_t x_off1 = M256 ? (uint32_t)(2048 + tid * 8) : (uint32_t)(((rr_1 < M) ? rr_1 : (M - 1)) * n + (tid & 1) * 8);
  const _Float16* xt_mb = M256 ? (xt + (size_t)(m0 >> 8) * ((size_t)nt * 4096)) : xt;
  const bool v_rr0 = (rr_0 < M);
  const bool v_rr1 = (rr_1 < M);

  auto load_w = [&](int j_idx, u4v& w4, uint32_t& wl) {
    const int q = tid & 7;
    const size_t tid0 = w_row_base + j_idx;
    if constexpr (SP16) {
      w4 = *reinterpret_cast<const u4v*>(p16 + tid0 * 32 + q * 4);
      wl = p4[tid0 * 8 + q];
    } else {
      const uint32_t* tp0 = p16 + tid0 * 40 + q * 5;
      w4 = *reinterpret_cast<const u4va*>(tp0);
      wl = tp0[4];
    }
  };
  auto load_x = [&](int j_idx, uint4 x_reg[2]) {
    if constexpr (M256) {
      const _Float16* xb = xt_mb + (size_t)j_idx * 4096;
      uint4 r0 = *reinterpret_cast<const uint4*>(xb + x_off0);
      uint4 r1 = *reinterpret_cast<const uint4*>(xb + x_off1);
      x_reg[0] = v_rr0 ? r0 : make_uint4(0u, 0u, 0u, 0u);
      x_reg[1] = v_rr1 ? r1 : make_uint4(0u, 0u, 0u, 0u);
    } else {
      const uint32_t jo = (uint32_t)(j_idx * 16);
      uint4 r0 = *reinterpret_cast<const uint4*>(xt + (size_t)(x_off0 + jo));
      uint4 r1 = *reinterpret_cast<const uint4*>(xt + (size_t)(x_off1 + jo));
      x_reg[0] = v_rr0 ? r0 : make_uint4(0u, 0u, 0u, 0u);
      x_reg[1] = v_rr1 ? r1 : make_uint4(0u, 0u, 0u, 0u);
    }
  };
  auto decode_store = [&](int buf, const u4v& w4, uint32_t wl, const uint4 x_reg[2]) {
    const int x_rl0 = tid >> 1;
    const int x_rl1 = 128 + (tid >> 1);
    *reinterpret_cast<uint4*>(&smem.stage.x_lds[buf][x_rl0][k5_sw_col4(x_rl0, tid & 1)]) = x_reg[0];
    *reinterpret_cast<uint4*>(&smem.stage.x_lds[buf][x_rl1][k5_sw_col4(x_rl1, tid & 1)]) = x_reg[1];

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

  u4v w4_0; uint32_t wl_0; uint4 x_0[2];
  load_w(0, w4_0, wl_0); load_x(0, x_0);
  decode_store(0, w4_0, wl_0, x_0);
  lds_barrier();

  auto ld_a = [&](int buf, int ni) -> h16v {
    const int l16 = tid & 15, wn = (tid >> 5) & 1;
    const uint32_t* p = &smem.stage.w_lds[buf][wn * 64 + ni * 16 + l16][0];
    const int c0 = k5_sw_col4(l16, 0), c1 = k5_sw_col4(l16, 1);
    return __builtin_bit_cast(h16v, u2_128{*reinterpret_cast<const uint4*>(p + c0), *reinterpret_cast<const uint4*>(p + c1)});
  };
  auto ld_b = [&](int buf, int mi) -> h16v {
    const int l16 = tid & 15, wm = tid >> 6;
    const uint32_t* p = &smem.stage.x_lds[buf][wm * 64 + mi * 16 + l16][0];
    const int c0 = k5_sw_col4(l16, 0), c1 = k5_sw_col4(l16, 1);
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
    // Reuse at (= a3) across halves with ZERO ds_load!
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
      lds_barrier();
    }
  };

  // nt = K/16 is always a multiple of 8 (HTR FWHT-128) -> always even and >= 8!
  // Zero post-loop branch keeps acc[4][4] 100% in VGPRs (V<=192, vSp=0, scrOps=0)!
#pragma unroll 1
  for (int j = 0; j < nt - 2; j += 2) {
    step(0, j + 1, true);
    step(1, j + 2, true);
  }
  step(0, nt - 1, true);
  step(1, 0, false);

  // Epilogue: 32 rows at a time in o_lds[32][128]
  const int w = tid >> 5, wn = w & 1, wm = w >> 1;
  const int lane = tid & 31, l16 = lane & 15, hb = lane >> 4;
  const int m_out = mt * 16;
  const int base_col = b * 128 + lane * 4;
  float sc[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) sc[e] = __half2float(scales[base_col + e]) * cb;
  const unsigned sg4 = (unsigned)sv[b * 16 + (lane >> 1)] >> ((lane & 1) * 4);
  float bv[4] = {0.f, 0.f, 0.f, 0.f};
  if (bias != nullptr) {
    const uint2 braw = *reinterpret_cast<const uint2*>(bias + base_col);
    bv[0] = __uint_as_float((braw.x & 0xFFFFu) << 16);
    bv[1] = __uint_as_float(braw.x & 0xFFFF0000u);
    bv[2] = __uint_as_float((braw.y & 0xFFFFu) << 16);
    bv[3] = __uint_as_float(braw.y & 0xFFFF0000u);
  }

  const int c_base = wn * 64;
#pragma unroll 1
  for (int wm_pair = 0; wm_pair < 2; ++wm_pair) {
    const int wm_lo = wm_pair * 2;
#pragma unroll 1
    for (int mi = 0; mi < 4; ++mi) {
      lds_barrier();
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
      lds_barrier();

#pragma unroll
      for (int sub = 0; sub < 4; ++sub) {
        const int rt = w + sub * 8;
        const int r = m0 + (wm_lo + (rt >= 16 ? 1 : 0)) * 64 + mi * 16 + (rt & 15);
        if (r < M) {
          const float tt = tok[r];
          float v[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) v[e] = smem.o_lds[rt][lane * 4 + e] * sc[e] * tt;
          fwht128_wave(v, lane);
          float ob[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            float o = v[e] * (1.0f / 11.313708498984761f);
            if ((sg4 >> e) & 1u) o = -o;
            if (bias != nullptr) o += bv[e];
            ob[e] = o;
          }
          uint2 out_p;
          out_p.x = f32_to_bf16x2_rne(ob[0], ob[1]);
          out_p.y = f32_to_bf16x2_rne(ob[2], ob[3]);
          uint64_t* dst64 = reinterpret_cast<uint64_t*>(y + (size_t)r * m_out + base_col);
          if constexpr (ACCUM_Y) {
            const uint64_t prev64 = __builtin_nontemporal_load(dst64);
            const uint32_t px = (uint32_t)prev64, py = (uint32_t)(prev64 >> 32);
            const float s0 = __uint_as_float((out_p.x & 0xFFFFu) << 16) + __uint_as_float((px & 0xFFFFu) << 16);
            const float s1 = __uint_as_float(out_p.x & 0xFFFF0000u)     + __uint_as_float(px & 0xFFFF0000u);
            const float s2 = __uint_as_float((out_p.y & 0xFFFFu) << 16) + __uint_as_float((py & 0xFFFFu) << 16);
            const float s3 = __uint_as_float(out_p.y & 0xFFFF0000u)     + __uint_as_float(py & 0xFFFF0000u);
            out_p.x = f32_to_bf16x2_rne(s0, s1);
            out_p.y = f32_to_bf16x2_rne(s2, s3);
          }
          const uint64_t out64 = (uint64_t)out_p.x | ((uint64_t)out_p.y << 32);
          __builtin_nontemporal_store(out64, dst64);
        }
      }
    }
  }
}

// === V28_4X4_END ===

static inline dim3 k5sp_grid_3d_swz(int nb, int nm) {
  const int gx = (nm >= 4 && nb > 8) ? 8 : nb;
  return dim3(gx, nm, (nb + gx - 1) / gx);
}

torch::Tensor repack_k5_packed(torch::Tensor qw) {
  TORCH_CHECK(qw.is_cuda() && qw.is_contiguous() && qw.dim() == 3 && qw.size(2) == 40 && qw.scalar_type() == torch::kInt32);
  const int64_t ntiles = qw.size(0) * qw.size(1);
  auto out = torch::empty_like(qw);
  auto st = c10::hip::getCurrentHIPStream().stream();
  const int64_t nthreads = ntiles * 8;
  hipLaunchKernelGGL(k_repack_k5_packed, dim3((unsigned)((nthreads + 255) / 256)), dim3(256), 0, st,
                     reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>()),
                     reinterpret_cast<uint32_t*>(out.data_ptr<int32_t>()), ntiles);
  return out;
}


torch::Tensor dense_prefill_wmma_sp16(
    torch::Tensor p16, torch::Tensor p4, torch::Tensor xt, torch::Tensor tok,
    torch::Tensor scales, double cb, torch::Tensor sv,
    c10::optional<torch::Tensor> bias, int64_t bm) {
  TORCH_CHECK(p16.is_cuda() && p16.is_contiguous() && p16.dim() == 3 && p16.size(2) == 32);
  TORCH_CHECK(p4.is_cuda() && p4.is_contiguous() && p4.dim() == 3 && p4.size(2) == 8);
  const bool is_m256 = (xt.dim() == 4);
  TORCH_CHECK(xt.is_cuda() && xt.is_contiguous() && (xt.dim() == 2 || is_m256) && xt.scalar_type() == torch::kHalf);
  const int mt = (int)p16.size(0), nt = (int)p16.size(1);
  const int M = (int)tok.size(0);
  if (is_m256) {
    TORCH_CHECK((mt % 8 == 0) && xt.size(0) == (M + 255) / 256 && xt.size(1) == nt && xt.size(2) == 256 && xt.size(3) == 16);
  } else {
    TORCH_CHECK((mt % 8 == 0) && xt.size(0) == M && xt.size(1) == nt * 16);
  }
  auto y = torch::empty({M, mt * 16}, xt.options().dtype(torch::kBFloat16));
  if (M == 0) return y;
  auto st = c10::hip::getCurrentHIPStream().stream();
  auto p16p = reinterpret_cast<const uint32_t*>(p16.data_ptr<int32_t>());
  auto p4p = reinterpret_cast<const uint32_t*>(p4.data_ptr<int32_t>());
  auto xp = reinterpret_cast<const _Float16*>(xt.data_ptr<at::Half>());
  auto tp = tok.data_ptr<float>();
  auto sp = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  auto svp = sv.data_ptr<uint8_t>();
  const uint16_t* bp = (bias.has_value() && bias->defined())
      ? reinterpret_cast<const uint16_t*>(bias->data_ptr<at::BFloat16>()) : nullptr;
  auto yp = reinterpret_cast<uint16_t*>(y.data_ptr<at::BFloat16>());
  if (is_m256) {
    hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp_v28_128x256_4x4<1, 1>), k5sp_grid_3d_swz(mt / 8, (M + 255) / 256), dim3(256), 0, st,
                       p16p, p4p, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
    return y;
  }
  if (bm <= 0) bm = (M >= 256 && mt > 64) ? 256 : (M >= 128 ? 128 : (M <= 16 ? 16 : (M <= 32 ? 32 : 64)));
  if (bm == 256) {
    hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp_v28_128x256_4x4<1, 0>), k5sp_grid_3d_swz(mt / 8, (M + 255) / 256), dim3(256), 0, st,
                       p16p, p4p, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else if (bm == 128) {
    hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp_v28_128x128<1>), dim3(mt / 8, (M + 127) / 128, 1), dim3(256), 0, st,
                       p16p, p4p, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else if (bm == 16) {
    hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp<1, 1>), dim3(mt / 8, (M + 15) / 16, 1), dim3(256), 0, st,
                       p16p, p4p, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else if (bm == 32) {
    hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp<2, 1>), dim3(mt / 8, (M + 31) / 32, 1), dim3(256), 0, st,
                       p16p, p4p, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else {
    hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp<4, 1>), dim3(mt / 8, (M + 63) / 64, 1), dim3(256), 0, st,
                       p16p, p4p, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  }
  return y;
}

torch::Tensor dense_prefill_wmma_unsplit(
    torch::Tensor qw, torch::Tensor xt, torch::Tensor tok,
    torch::Tensor scales, double cb, torch::Tensor sv,
    c10::optional<torch::Tensor> bias, int64_t bm,
    c10::optional<torch::Tensor> accum_out = c10::nullopt) {
  TORCH_CHECK(qw.is_cuda() && qw.is_contiguous() && qw.dim() == 3 && qw.size(2) == 40);
  const bool is_m256 = (xt.dim() == 4);
  TORCH_CHECK(xt.is_cuda() && xt.is_contiguous() && (xt.dim() == 2 || is_m256) && xt.scalar_type() == torch::kHalf);
  const int mt = (int)qw.size(0), nt = (int)qw.size(1);
  const int M = (int)tok.size(0);
  if (is_m256) {
    TORCH_CHECK((mt % 8 == 0) && xt.size(0) == (M + 255) / 256 && xt.size(1) == nt && xt.size(2) == 256 && xt.size(3) == 16);
  } else {
    TORCH_CHECK((mt % 8 == 0) && xt.size(0) == M && xt.size(1) == nt * 16);
  }
  const bool has_acc = accum_out.has_value() && accum_out->defined();
  if (has_acc) {
    TORCH_CHECK(accum_out->is_cuda() && accum_out->is_contiguous() && accum_out->dim() == 2 &&
                accum_out->size(0) == M && accum_out->size(1) == mt * 16 && accum_out->scalar_type() == torch::kBFloat16);
  }
  if (M == 0) return has_acc ? *accum_out : torch::empty({0, mt * 16}, xt.options().dtype(torch::kBFloat16));
  auto st = c10::hip::getCurrentHIPStream().stream();
  auto qwp = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>());
  auto xp = reinterpret_cast<const _Float16*>(xt.data_ptr<at::Half>());
  auto tp = tok.data_ptr<float>();
  auto sp = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  auto svp = sv.data_ptr<uint8_t>();
  const uint16_t* bp = (bias.has_value() && bias->defined())
      ? reinterpret_cast<const uint16_t*>(bias->data_ptr<at::BFloat16>()) : nullptr;
  if (is_m256) {
    torch::Tensor y = has_acc ? *accum_out : torch::empty({M, mt * 16}, xt.options().dtype(torch::kBFloat16));
    auto yp = reinterpret_cast<uint16_t*>(y.data_ptr<at::BFloat16>());
    if (has_acc) {
      hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp_v28_128x256_4x4<0, 1, 1>), k5sp_grid_3d_swz(mt / 8, (M + 255) / 256), dim3(256), 0, st,
                         qwp, nullptr, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
    } else {
      hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp_v28_128x256_4x4<0, 1, 0>), k5sp_grid_3d_swz(mt / 8, (M + 255) / 256), dim3(256), 0, st,
                         qwp, nullptr, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
    }
    return y;
  }
  auto y = torch::empty({M, mt * 16}, xt.options().dtype(torch::kBFloat16));
  auto yp = reinterpret_cast<uint16_t*>(y.data_ptr<at::BFloat16>());
  if (bm <= 0) bm = (M >= 256 && mt > 64) ? 256 : (M >= 128 ? 128 : (M <= 16 ? 16 : (M <= 32 ? 32 : 64)));
  if (bm == 256) {
    hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp_v28_128x256_4x4<0, 0>), k5sp_grid_3d_swz(mt / 8, (M + 255) / 256), dim3(256), 0, st,
                       qwp, nullptr, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else if (bm == 128) {
    hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp_v28_128x128<0>), dim3(mt / 8, (M + 127) / 128, 1), dim3(256), 0, st,
                       qwp, nullptr, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else if (bm == 16) {
    hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp<1, 0>), dim3(mt / 8, (M + 15) / 16, 1), dim3(256), 0, st,
                       qwp, nullptr, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else if (bm == 32) {
    hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp<2, 0>), dim3(mt / 8, (M + 31) / 32, 1), dim3(256), 0, st,
                       qwp, nullptr, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else {
    hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp<4, 0>), dim3(mt / 8, (M + 63) / 64, 1), dim3(256), 0, st,
                       qwp, nullptr, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  }
  if (has_acc) {
    accum_out->add_(y);
    return *accum_out;
  }
  return y;
}

// ============================================================================
// Zero-Partial, Zero-Atomic, Zero-Fence Single-Pass K=5 Split-Plane GEMV (ns=1)
// Designed for tall ERNIE-5.5 K=5 matrices where mt/8 already saturates all 40 CUs:
//   - wq_b   [ 32768 x 1024] (37x/tok = 777 MB): 256 WGs (6.4 WGs/CU), saves 39..233 MB partial traffic!
//   - lm_head[201216 x 4096] (515 MB/tok + draft): 1572 WGs (39.3 WGs/CU), saves 25.8 MB partial traffic + 50K atomics!
// Exact M=1,2,4,7,8 templates (M=7 matches ERNIE-5.5 DSpark 7-token draft logits with zero wasted row!)
// ============================================================================
template <int M_CAP>
__global__ __launch_bounds__(256)
void k_gemv_k5_sp16_ns1(
    const uint32_t* __restrict__ p16,      // [mt, nt, 32]
    const uint32_t* __restrict__ p4,       // [mt, nt, 8]
    const _Float16* __restrict__ xt,       // [M, nt*16] fp16 (L1/L2 cached across WGs!)
    const float* __restrict__ xs,          // [M] fp32
    const float* __restrict__ tok,         // [M] fp32
    const __half* __restrict__ scales,     // [mt*16] fp16
    float cb,
    const uint8_t* __restrict__ sv,        // [mt*2] u8
    const uint16_t* __restrict__ bias,     // [mt*16] bf16 (nullable)
    uint16_t* __restrict__ y,              // [M, mt*16] bf16
    int M_act, int mt, int nt) {
  constexpr int K = 5;
  __shared__ __attribute__((aligned(16))) float o_lds[M_CAP][128]; // 0.5..4.0 KiB!

  const int tid = threadIdx.x;
  const int w = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 3;
  const int q = lane & 7;
  const int b = blockIdx.x;
  const int i = b * 8 + w;
  const int n = nt * 16;
  const int m_out = mt * 16;

  float a0[M_CAP], a1[M_CAP];
#pragma unroll
  for (int r = 0; r < M_CAP; ++r) { a0[r] = 0.f; a1[r] = 0.f; }

  // 2-stage register software prefetch of K=5 split-plane weights:
  // Wave-uniform 64-bit SGPR bases + 32-bit uint32_t VGPR offsets -> unlocks RDNA3.5 SADDR mode (global_load v, voff, s[base])!
  const uint32_t* p16_base = p16 + (size_t)i * nt * 32;
  const uint32_t* p4_base  = p4  + (size_t)i * nt * 8;
  u4v w4_cur = {0u, 0u, 0u, 0u};
  uint32_t wl_cur = 0u;
  if (g < nt) {
    const uint32_t o0 = (uint32_t)(g * 8 + q);
    w4_cur = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(p16_base + (o0 << 2)));
    wl_cur = __builtin_nontemporal_load(p4_base + o0);
  }

  for (int j = g; j < nt; j += 4) {
    uint32_t V[6];
    V[0] = __builtin_amdgcn_permlane16(wl_cur, wl_cur, 0x65432107u, 0xedcba98fu, false, false);
    V[1] = w4_cur.x; V[2] = w4_cur.y; V[3] = w4_cur.z; V[4] = w4_cur.w; V[5] = wl_cur;

    // Half-tile xv[M_CAP][4] + bank-staggered (U_lo, U_hi) pairing:
    //   1) Halves xv VGPR pressure + 32-bit SADDR offsets -> <=96 VGPRs (FULL 16w/SIMD at M=1..7/8!)
    //   2) Staggers xv[r][k] vs xv[r][k^1] across different VGPR banks to unlock V_DUAL_DOT2ACC_F32_F16!
    const uint32_t xj_off = (uint32_t)(j * 16);
    h2v xv[M_CAP][4];
#pragma unroll
    for (int r = 0; r < M_CAP; ++r) {
      const uint32_t rr = (uint32_t)((r < M_act) ? r : 0);
      const uint4 u0 = reinterpret_cast<const uint4*>(xt + rr * (uint32_t)n + xj_off)[0];
      xv[r][0] = as_h2(u0.x); xv[r][1] = as_h2(u0.y); xv[r][2] = as_h2(u0.z); xv[r][3] = as_h2(u0.w);
    }
#define NS1_DUAL(U0, U1) { \
      const h2v w0 = as_h2(val_f16bits2(ext16<16 + ((U0) + 1) * K>(V), ext16<16 + ((U0) + 2) * K>(V))); \
      const h2v w1 = as_h2(val_f16bits2(ext16<16 + ((U1) + 1) * K>(V), ext16<16 + ((U1) + 2) * K>(V))); \
      _Pragma("unroll") for (int r = 0; r < M_CAP; ++r) { \
        a0[r] = __builtin_amdgcn_fdot2(w0, xv[r][((U0) & 7) >> 1], a0[r], false); \
        a1[r] = __builtin_amdgcn_fdot2(w1, xv[r][((U1) & 7) >> 1], a1[r], false); \
      } }
    NS1_DUAL(0, 18) NS1_DUAL(2, 16) NS1_DUAL(4, 22) NS1_DUAL(6, 20)
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int r = 0; r < M_CAP; ++r) {
      const uint32_t rr = (uint32_t)((r < M_act) ? r : 0);
      const uint4 u1 = reinterpret_cast<const uint4*>(xt + rr * (uint32_t)n + xj_off)[1];
      xv[r][0] = as_h2(u1.x); xv[r][1] = as_h2(u1.y); xv[r][2] = as_h2(u1.z); xv[r][3] = as_h2(u1.w);
    }
    u4v w4_nxt = {0u, 0u, 0u, 0u};
    uint32_t wl_nxt = 0u;
    if (j + 4 < nt) {
      const uint32_t on = (uint32_t)((j + 4) * 8 + q);
      w4_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(p16_base + (on << 2)));
      wl_nxt = __builtin_nontemporal_load(p4_base + on);
    }
    __builtin_amdgcn_sched_barrier(0);
    NS1_DUAL(8, 26) NS1_DUAL(10, 24) NS1_DUAL(12, 30) NS1_DUAL(14, 28)
#undef NS1_DUAL
    w4_cur = w4_nxt;
    wl_cur = wl_nxt;
  }

  // Wave-local reduction across the 4 g-groups (ZERO global partials, ZERO atomics, ZERO fences!)
#pragma unroll
  for (int r = 0; r < M_CAP; ++r) {
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
  }
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M_CAP; ++r) {
      if (r < M_act) {
        const float off = 1534.0f * xs[r];
        o_lds[r][w * 16 + 2 * q]     = a0[r] - off;
        o_lds[r][w * 16 + 2 * q + 1] = a1[r] - off;
      }
    }
  }
  __syncthreads();

  // Direct in-WG Wave32 FWHT-128 + scales + sv + bias + coalesced 8B uint2 BF16 store
  if (w < M_act) {
    const int base_col = b * 128 + lane * 4;
    const float tt = tok[w] * cb;
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      v[e] = o_lds[w][lane * 4 + e] * __half2float(scales[base_col + e]) * tt;
    }
    fwht128_wave(v, lane);
    const unsigned sg4 = (unsigned)sv[b * 16 + (lane >> 1)] >> ((lane & 1) * 4);
    float ob[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float o = v[e] * (1.0f / 11.313708498984761f);
      if ((sg4 >> e) & 1u) o = -o;
      if (bias != nullptr) o += __uint_as_float(((uint32_t)bias[base_col + e]) << 16);
      ob[e] = o;
    }
    uint2 out_p;
    out_p.x = f32_to_bf16x2_rne(ob[0], ob[1]);
    out_p.y = f32_to_bf16x2_rne(ob[2], ob[3]);
    *reinterpret_cast<uint2*>(y + (size_t)w * m_out + base_col) = out_p;
  }
}

torch::Tensor gemv_k5_sp16_ns1(
    torch::Tensor p16, torch::Tensor p4, torch::Tensor xt, torch::Tensor xs, torch::Tensor tok,
    torch::Tensor scales, double cb, torch::Tensor sv, c10::optional<torch::Tensor> bias) {
  TORCH_CHECK(p16.is_cuda() && p16.is_contiguous() && p16.dim() == 3 && p16.size(2) == 32);
  TORCH_CHECK(p4.is_cuda() && p4.is_contiguous() && p4.dim() == 3 && p4.size(2) == 8);
  TORCH_CHECK(xt.is_cuda() && xt.is_contiguous() && xt.dim() == 2 && xt.scalar_type() == torch::kHalf);
  const int mt = (int)p16.size(0), nt = (int)p16.size(1);
  const int M = (int)xt.size(0);
  TORCH_CHECK((mt % 8 == 0) && xt.size(1) == nt * 16 && M >= 1 && M <= 8);
  auto y = torch::empty({M, mt * 16}, xt.options().dtype(torch::kBFloat16));
  auto st = c10::hip::getCurrentHIPStream().stream();
  auto p16p = reinterpret_cast<const uint32_t*>(p16.data_ptr<int32_t>());
  auto p4p = reinterpret_cast<const uint32_t*>(p4.data_ptr<int32_t>());
  auto xp = reinterpret_cast<const _Float16*>(xt.data_ptr<at::Half>());
  auto xsp = xs.data_ptr<float>();
  auto tp = tok.data_ptr<float>();
  auto sp = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  auto svp = sv.data_ptr<uint8_t>();
  const uint16_t* bp = (bias.has_value() && bias->defined())
      ? reinterpret_cast<const uint16_t*>(bias->data_ptr<at::BFloat16>()) : nullptr;
  auto yp = reinterpret_cast<uint16_t*>(y.data_ptr<at::BFloat16>());
  const dim3 grid(mt / 8);
  if (M == 1) {
    hipLaunchKernelGGL((k_gemv_k5_sp16_ns1<1>), grid, dim3(256), 0, st, p16p, p4p, xp, xsp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else if (M == 2) {
    hipLaunchKernelGGL((k_gemv_k5_sp16_ns1<2>), grid, dim3(256), 0, st, p16p, p4p, xp, xsp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else if (M <= 4) {
    hipLaunchKernelGGL((k_gemv_k5_sp16_ns1<4>), grid, dim3(256), 0, st, p16p, p4p, xp, xsp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else if (M <= 7) {
    hipLaunchKernelGGL((k_gemv_k5_sp16_ns1<7>), grid, dim3(256), 0, st, p16p, p4p, xp, xsp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else {
    hipLaunchKernelGGL((k_gemv_k5_sp16_ns1<8>), grid, dim3(256), 0, st, p16p, p4p, xp, xsp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  }
  return y;
}

// ============================================================================
// Fused ERNIE-5.5 Prologues for HTR K=5 Linears (eliminates 90 launches/step + 728 MB/prefill!)
//   1. situ_and_mul_htr_prologue: gate_up [T, 2*D] bf16 -> SiTU-GLU(fast_tanhf) -> HTR prologue (xt, xs, tok)
//      Covers 42 shared_experts.down_proj + 3 DSpark shared_experts + 1 layer-0 MLP (46x/step!)
//   2. rmsnorm_htr_prologue:      x [T, D] bf16 -> RMSNorm(v_dot2_f32_bf16) -> HTR prologue (xt, xs, tok)
//      Covers 42 routed MoE latent_norm->fc2_latent_proj + 3 DSpark layers (45x/step!)
// Exact bit-compatible rounding with (SituAndMul / RMSNorm -> BF16 -> htr_kernels k_prologue1p)!
// ============================================================================
__device__ __forceinline__ float wave_max_f(float a) {
  a = fmaxf(a, shx<16>(a)); a = fmaxf(a, shx<8>(a)); a = fmaxf(a, shx<4>(a)); a = fmaxf(a, shx<2>(a)); a = fmaxf(a, shx<1>(a));
  return a;
}
__device__ __forceinline__ float wave_sum_f(float a) {
  a += shx<16>(a); a += shx<8>(a); a += shx<4>(a); a += shx<2>(a); a += shx<1>(a);
  return a;
}

template <int NB, int M256 = 0>
__global__ __launch_bounds__(256) void k_situ_and_mul_htr_prologue(
    const uint16_t* __restrict__ gate_up, // [T, 2*D] bf16 (stride gu_s0)
    const uint8_t* __restrict__ su,       // [D/8] u8
    _Float16* __restrict__ xt,            // [T, D] fp16
    float* __restrict__ xs,               // [T] fp32
    float* __restrict__ tok,              // [T] fp32
    int64_t gu_s0, int D, float beta, float inv_beta, float linear_beta, float inv_lbeta) {
  __shared__ float red[8];
  __shared__ float s_scale;
  const int row = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int nblk = D >> 7;
  const uint16_t* gr = gate_up + (int64_t)row * gu_s0;
  const uint16_t* ur = gr + D;

  float v[NB][4];
  float amax = 0.f;
#pragma unroll
  for (int t = 0; t < NB; ++t) {
    const int blk = w + 8 * t;
    if (blk < nblk) {
      const int base = blk * 128 + lane * 4;
      const uint2 rg = *reinterpret_cast<const uint2*>(gr + base);
      const uint2 ru = *reinterpret_cast<const uint2*>(ur + base);
      const float gv[4] = {
        __uint_as_float((rg.x & 0xFFFFu) << 16), __uint_as_float(rg.x & 0xFFFF0000u),
        __uint_as_float((rg.y & 0xFFFFu) << 16), __uint_as_float(rg.y & 0xFFFF0000u),
      };
      const float uv[4] = {
        __uint_as_float((ru.x & 0xFFFFu) << 16), __uint_as_float(ru.x & 0xFFFF0000u),
        __uint_as_float((ru.y & 0xFFFFu) << 16), __uint_as_float(ru.y & 0xFFFF0000u),
      };
      const unsigned sg4 = (unsigned)su[base >> 3] >> (base & 7);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float g = gv[e], u = uv[e];
        const float sg = beta * fast_tanhf(g * inv_beta) * __builtin_amdgcn_rcpf(1.0f + exp2f(-1.4426950408889634f * g));
        const float su_v = (linear_beta > 0.0f) ? (linear_beta * fast_tanhf(u * inv_lbeta)) : u;
        const float y = __uint_as_float(((uint32_t)f32_to_bf16_rne(sg * su_v)) << 16);
        amax = fmaxf(amax, fabsf(y));
        v[t][e] = ((sg4 >> e) & 1u) ? -y : y;
      }
    }
  }
  amax = wave_max_f(amax);
  if (lane == 0) red[w] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = red[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) m = fmaxf(m, red[k]);
    const float bound = m * 11.4f;
    s_scale = bound > 30000.f ? exp2f(ceilf(log2f(bound * (1.0f / 30000.f)))) : 1.f;
    tok[row] = s_scale;
  }
  __syncthreads();
  const float inv = (1.0f / 11.313708498984761f) * __builtin_amdgcn_rcpf(s_scale);
  if constexpr (M256) {
    const int nt = D >> 4;
    const int mb = row >> 8;
    const int r_loc = row & 255;
    _Float16* xt_row = xt + (size_t)mb * ((size_t)nt * 4096) + (size_t)r_loc * 16;
    const int j_in_blk = lane >> 2;
    const int e_in_j = (lane & 3) * 4;
#pragma unroll
    for (int t = 0; t < NB; ++t) {
      const int blk = w + 8 * t;
      if (blk < nblk) {
        fwht128_wave(v[t], lane);
        const int j = blk * 8 + j_in_blk;
        const h4v h4 = {(_Float16)(v[t][0] * inv), (_Float16)(v[t][1] * inv),
                        (_Float16)(v[t][2] * inv), (_Float16)(v[t][3] * inv)};
        *reinterpret_cast<h4v*>(xt_row + (size_t)j * 4096 + e_in_j) = h4;
      }
    }
  } else {
    float sum = 0.f;
#pragma unroll
    for (int t = 0; t < NB; ++t) {
      const int blk = w + 8 * t;
      if (blk < nblk) {
        fwht128_wave(v[t], lane);
        const int base = blk * 128 + lane * 4;
        const h4v h4 = {(_Float16)(v[t][0] * inv), (_Float16)(v[t][1] * inv),
                        (_Float16)(v[t][2] * inv), (_Float16)(v[t][3] * inv)};
        *reinterpret_cast<h4v*>(xt + (size_t)row * D + base) = h4;
        sum += (float)h4.x + (float)h4.y + (float)h4.z + (float)h4.w;
      }
    }
    sum = wave_sum_f(sum);
    __syncthreads();
    if (lane == 0) red[w] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
      float s = red[0];
#pragma unroll
      for (int k = 1; k < 8; ++k) s += red[k];
      xs[row] = s;
    }
  }
}

std::vector<torch::Tensor> situ_and_mul_htr_prologue(
    torch::Tensor gate_up, torch::Tensor su, double beta, double linear_beta) {
  TORCH_CHECK(gate_up.is_cuda() && gate_up.dim() == 2 && gate_up.scalar_type() == torch::kBFloat16 && gate_up.stride(1) == 1);
  const int T = (int)gate_up.size(0);
  const int D = (int)(gate_up.size(1) / 2);
  TORCH_CHECK((gate_up.size(1) == 2 * D) && (D % 128 == 0) && D <= 14336);
  const bool use_m256 = (T >= 256 && D >= 1024);
  auto xt = use_m256
      ? torch::empty({(T + 255) / 256, D / 16, 256, 16}, gate_up.options().dtype(torch::kHalf))
      : torch::empty({T, D}, gate_up.options().dtype(torch::kHalf));
  auto xs = use_m256
      ? torch::empty({0}, gate_up.options().dtype(torch::kFloat32))
      : torch::empty({T}, gate_up.options().dtype(torch::kFloat32));
  auto tok = torch::empty({T}, gate_up.options().dtype(torch::kFloat32));
  if (T == 0) return {xt, xs, tok};
  const int nblk = D >> 7;
  const int nb = (nblk + 7) / 8;
  auto st = c10::hip::getCurrentHIPStream().stream();
  const uint16_t* gup = reinterpret_cast<const uint16_t*>(gate_up.data_ptr());
  const uint8_t* sup = su.data_ptr<uint8_t>();
  _Float16* xtp = reinterpret_cast<_Float16*>(xt.data_ptr<at::Half>());
  float* xsp = use_m256 ? nullptr : xs.data_ptr<float>();
  float* tokp = tok.data_ptr<float>();
  const float bf = (float)beta, ibf = 1.0f / bf;
  const float lbf = (float)linear_beta, ilbf = (lbf > 0.0f) ? (1.0f / lbf) : 0.0f;
  if (use_m256) {
    if (nb <= 2) hipLaunchKernelGGL((k_situ_and_mul_htr_prologue<2, 1>), dim3(T), dim3(256), 0, st, gup, sup, xtp, xsp, tokp, gate_up.stride(0), D, bf, ibf, lbf, ilbf);
    else if (nb <= 4) hipLaunchKernelGGL((k_situ_and_mul_htr_prologue<4, 1>), dim3(T), dim3(256), 0, st, gup, sup, xtp, xsp, tokp, gate_up.stride(0), D, bf, ibf, lbf, ilbf);
    else if (nb <= 8) hipLaunchKernelGGL((k_situ_and_mul_htr_prologue<8, 1>), dim3(T), dim3(256), 0, st, gup, sup, xtp, xsp, tokp, gate_up.stride(0), D, bf, ibf, lbf, ilbf);
    else if (nb <= 12) hipLaunchKernelGGL((k_situ_and_mul_htr_prologue<12, 1>), dim3(T), dim3(256), 0, st, gup, sup, xtp, xsp, tokp, gate_up.stride(0), D, bf, ibf, lbf, ilbf);
    else hipLaunchKernelGGL((k_situ_and_mul_htr_prologue<14, 1>), dim3(T), dim3(256), 0, st, gup, sup, xtp, xsp, tokp, gate_up.stride(0), D, bf, ibf, lbf, ilbf);
  } else if (nb <= 2) {
    hipLaunchKernelGGL((k_situ_and_mul_htr_prologue<2, 0>), dim3(T), dim3(256), 0, st, gup, sup, xtp, xsp, tokp, gate_up.stride(0), D, bf, ibf, lbf, ilbf);
  } else if (nb <= 4) {
    hipLaunchKernelGGL((k_situ_and_mul_htr_prologue<4, 0>), dim3(T), dim3(256), 0, st, gup, sup, xtp, xsp, tokp, gate_up.stride(0), D, bf, ibf, lbf, ilbf);
  } else if (nb <= 8) {
    hipLaunchKernelGGL((k_situ_and_mul_htr_prologue<8, 0>), dim3(T), dim3(256), 0, st, gup, sup, xtp, xsp, tokp, gate_up.stride(0), D, bf, ibf, lbf, ilbf);
  } else if (nb <= 12) {
    hipLaunchKernelGGL((k_situ_and_mul_htr_prologue<12, 0>), dim3(T), dim3(256), 0, st, gup, sup, xtp, xsp, tokp, gate_up.stride(0), D, bf, ibf, lbf, ilbf);
  } else {
    hipLaunchKernelGGL((k_situ_and_mul_htr_prologue<14, 0>), dim3(T), dim3(256), 0, st, gup, sup, xtp, xsp, tokp, gate_up.stride(0), D, bf, ibf, lbf, ilbf);
  }
  return {xt, xs, tok};
}

template <int NB, int M256 = 0>
__global__ __launch_bounds__(256) void k_rmsnorm_htr_prologue(
    const uint16_t* __restrict__ x,       // [T, D] bf16 (stride x_s0)
    const uint16_t* __restrict__ norm_w,  // [D] bf16
    const uint8_t* __restrict__ su,       // [D/8] u8
    _Float16* __restrict__ xt,            // [T, D] fp16
    float* __restrict__ xs,               // [T] fp32
    float* __restrict__ tok,              // [T] fp32
    int64_t x_s0, int D, float eps) {
  __shared__ float red[8];
  __shared__ float s_rrms;
  __shared__ float s_scale;
  const int row = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int nblk = D >> 7;
  const uint16_t* xr = x + (int64_t)row * x_s0;

  uint2 xraw[NB];
  float ss0 = 0.f, ss1 = 0.f;
#pragma unroll
  for (int t = 0; t < NB; ++t) {
    const int blk = w + 8 * t;
    if (blk < nblk) {
      const int base = blk * 128 + lane * 4;
      xraw[t] = *reinterpret_cast<const uint2*>(xr + base);
      ss0 = bf16_dot2(xraw[t].x, xraw[t].x, ss0);
      ss1 = bf16_dot2(xraw[t].y, xraw[t].y, ss1);
    }
  }
  const float wss = wave_sum_f(ss0 + ss1);
  if (lane == 0) red[w] = wss;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = red[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) s += red[k];
    s_rrms = rsqrtf(s * __builtin_amdgcn_rcpf((float)D) + eps);
  }
  __syncthreads();
  const float rrms = s_rrms;

  float v[NB][4];
  float amax = 0.f;
#pragma unroll
  for (int t = 0; t < NB; ++t) {
    const int blk = w + 8 * t;
    if (blk < nblk) {
      const int base = blk * 128 + lane * 4;
      const uint2 rw = *reinterpret_cast<const uint2*>(norm_w + base);
      const float xv[4] = {
        __uint_as_float((xraw[t].x & 0xFFFFu) << 16), __uint_as_float(xraw[t].x & 0xFFFF0000u),
        __uint_as_float((xraw[t].y & 0xFFFFu) << 16), __uint_as_float(xraw[t].y & 0xFFFF0000u),
      };
      const float wv[4] = {
        __uint_as_float((rw.x & 0xFFFFu) << 16), __uint_as_float(rw.x & 0xFFFF0000u),
        __uint_as_float((rw.y & 0xFFFFu) << 16), __uint_as_float(rw.y & 0xFFFF0000u),
      };
      const unsigned sg4 = (unsigned)su[base >> 3] >> (base & 7);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float y = __uint_as_float(((uint32_t)f32_to_bf16_rne(xv[e] * rrms * wv[e])) << 16);
        amax = fmaxf(amax, fabsf(y));
        v[t][e] = ((sg4 >> e) & 1u) ? -y : y;
      }
    }
  }
  amax = wave_max_f(amax);
  __syncthreads();
  if (lane == 0) red[w] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = red[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) m = fmaxf(m, red[k]);
    const float bound = m * 11.4f;
    s_scale = bound > 30000.f ? exp2f(ceilf(log2f(bound * (1.0f / 30000.f)))) : 1.f;
    tok[row] = s_scale;
  }
  __syncthreads();
  const float inv = (1.0f / 11.313708498984761f) * __builtin_amdgcn_rcpf(s_scale);
  if constexpr (M256) {
    const int nt = D >> 4;
    const int mb = row >> 8;
    const int r_loc = row & 255;
    _Float16* xt_row = xt + (size_t)mb * ((size_t)nt * 4096) + (size_t)r_loc * 16;
    const int j_in_blk = lane >> 2;
    const int e_in_j = (lane & 3) * 4;
#pragma unroll
    for (int t = 0; t < NB; ++t) {
      const int blk = w + 8 * t;
      if (blk < nblk) {
        fwht128_wave(v[t], lane);
        const int j = blk * 8 + j_in_blk;
        const h4v h4 = {(_Float16)(v[t][0] * inv), (_Float16)(v[t][1] * inv),
                        (_Float16)(v[t][2] * inv), (_Float16)(v[t][3] * inv)};
        *reinterpret_cast<h4v*>(xt_row + (size_t)j * 4096 + e_in_j) = h4;
      }
    }
  } else {
    float sum = 0.f;
#pragma unroll
    for (int t = 0; t < NB; ++t) {
      const int blk = w + 8 * t;
      if (blk < nblk) {
        fwht128_wave(v[t], lane);
        const int base = blk * 128 + lane * 4;
        const h4v h4 = {(_Float16)(v[t][0] * inv), (_Float16)(v[t][1] * inv),
                        (_Float16)(v[t][2] * inv), (_Float16)(v[t][3] * inv)};
        *reinterpret_cast<h4v*>(xt + (size_t)row * D + base) = h4;
        sum += (float)h4.x + (float)h4.y + (float)h4.z + (float)h4.w;
      }
    }
    sum = wave_sum_f(sum);
    __syncthreads();
    if (lane == 0) red[w] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
      float s = red[0];
#pragma unroll
      for (int k = 1; k < 8; ++k) s += red[k];
      xs[row] = s;
    }
  }
}

std::vector<torch::Tensor> rmsnorm_htr_prologue(
    torch::Tensor x, torch::Tensor norm_w, torch::Tensor su, double eps) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == torch::kBFloat16 && x.stride(1) == 1);
  TORCH_CHECK(norm_w.is_cuda() && norm_w.scalar_type() == torch::kBFloat16 && norm_w.is_contiguous());
  const int T = (int)x.size(0);
  const int D = (int)x.size(1);
  TORCH_CHECK((D % 128 == 0) && D <= 8192);
  const bool use_m256 = (T >= 256 && D >= 1024);
  auto xt = use_m256
      ? torch::empty({(T + 255) / 256, D / 16, 256, 16}, x.options().dtype(torch::kHalf))
      : torch::empty({T, D}, x.options().dtype(torch::kHalf));
  auto xs = use_m256
      ? torch::empty({0}, x.options().dtype(torch::kFloat32))
      : torch::empty({T}, x.options().dtype(torch::kFloat32));
  auto tok = torch::empty({T}, x.options().dtype(torch::kFloat32));
  if (T == 0) return {xt, xs, tok};
  const int nb = ((D >> 7) + 7) / 8;
  auto st = c10::hip::getCurrentHIPStream().stream();
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
  const uint16_t* wp = reinterpret_cast<const uint16_t*>(norm_w.data_ptr());
  const uint8_t* sup = su.data_ptr<uint8_t>();
  _Float16* xtp = reinterpret_cast<_Float16*>(xt.data_ptr<at::Half>());
  float* xsp = use_m256 ? nullptr : xs.data_ptr<float>();
  float* tokp = tok.data_ptr<float>();
  if (use_m256) {
    if (nb <= 2) hipLaunchKernelGGL((k_rmsnorm_htr_prologue<2, 1>), dim3(T), dim3(256), 0, st, xp, wp, sup, xtp, xsp, tokp, x.stride(0), D, (float)eps);
    else if (nb <= 4) hipLaunchKernelGGL((k_rmsnorm_htr_prologue<4, 1>), dim3(T), dim3(256), 0, st, xp, wp, sup, xtp, xsp, tokp, x.stride(0), D, (float)eps);
    else hipLaunchKernelGGL((k_rmsnorm_htr_prologue<8, 1>), dim3(T), dim3(256), 0, st, xp, wp, sup, xtp, xsp, tokp, x.stride(0), D, (float)eps);
  } else if (nb <= 2) {
    hipLaunchKernelGGL((k_rmsnorm_htr_prologue<2, 0>), dim3(T), dim3(256), 0, st, xp, wp, sup, xtp, xsp, tokp, x.stride(0), D, (float)eps);
  } else if (nb <= 4) {
    hipLaunchKernelGGL((k_rmsnorm_htr_prologue<4, 0>), dim3(T), dim3(256), 0, st, xp, wp, sup, xtp, xsp, tokp, x.stride(0), D, (float)eps);
  } else {
    hipLaunchKernelGGL((k_rmsnorm_htr_prologue<8, 0>), dim3(T), dim3(256), 0, st, xp, wp, sup, xtp, xsp, tokp, x.stride(0), D, (float)eps);
  }
  return {xt, xs, tok};
}

// 3. sigmoid_mul_htr_prologue: o [T, D] bf16 * sigmoid(gate [T, D] bf16) -> HTR prologue (xt, xs, tok)
//    Covers all 6 DSA layers' wo_b(o.flatten(1) * torch.sigmoid(o_gate(q_c))) (-12 launches/step, -384 MB/prefill!)
template <int NB, int M256 = 0>
__global__ __launch_bounds__(256) void k_sigmoid_mul_htr_prologue(
    const uint16_t* __restrict__ o_in,    // [T, D] bf16 (stride o_s0)
    const uint16_t* __restrict__ g_in,    // [T, D] bf16 (stride g_s0)
    const uint8_t* __restrict__ su,       // [D/8] u8
    _Float16* __restrict__ xt,            // [T, D] fp16
    float* __restrict__ xs,               // [T] fp32
    float* __restrict__ tok,              // [T] fp32
    int64_t o_s0, int64_t g_s0, int D) {
  __shared__ float red[8];
  __shared__ float s_scale;
  const int row = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int nblk = D >> 7;
  const uint16_t* orow = o_in + (int64_t)row * o_s0;
  const uint16_t* grow = g_in + (int64_t)row * g_s0;

  float v[NB][4];
  float amax = 0.f;
#pragma unroll
  for (int t = 0; t < NB; ++t) {
    const int blk = w + 8 * t;
    if (blk < nblk) {
      const int base = blk * 128 + lane * 4;
      const uint2 ro = *reinterpret_cast<const uint2*>(orow + base);
      const uint2 rg = *reinterpret_cast<const uint2*>(grow + base);
      const float ov[4] = {
        __uint_as_float((ro.x & 0xFFFFu) << 16), __uint_as_float(ro.x & 0xFFFF0000u),
        __uint_as_float((ro.y & 0xFFFFu) << 16), __uint_as_float(ro.y & 0xFFFF0000u),
      };
      const float gv[4] = {
        __uint_as_float((rg.x & 0xFFFFu) << 16), __uint_as_float(rg.x & 0xFFFF0000u),
        __uint_as_float((rg.y & 0xFFFFu) << 16), __uint_as_float(rg.y & 0xFFFF0000u),
      };
      const unsigned sg4 = (unsigned)su[base >> 3] >> (base & 7);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float sig_b = __uint_as_float(((uint32_t)f32_to_bf16_rne(__builtin_amdgcn_rcpf(1.0f + exp2f(-1.4426950408889634f * gv[e])))) << 16);
        const float y = __uint_as_float(((uint32_t)f32_to_bf16_rne(ov[e] * sig_b)) << 16);
        amax = fmaxf(amax, fabsf(y));
        v[t][e] = ((sg4 >> e) & 1u) ? -y : y;
      }
    }
  }
  amax = wave_max_f(amax);
  if (lane == 0) red[w] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = red[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) m = fmaxf(m, red[k]);
    const float bound = m * 11.4f;
    s_scale = bound > 30000.f ? exp2f(ceilf(log2f(bound * (1.0f / 30000.f)))) : 1.f;
    tok[row] = s_scale;
  }
  __syncthreads();
  const float inv = (1.0f / 11.313708498984761f) * __builtin_amdgcn_rcpf(s_scale);
  if constexpr (M256) {
    const int nt = D >> 4;
    const int mb = row >> 8;
    const int r_loc = row & 255;
    _Float16* xt_row = xt + (size_t)mb * ((size_t)nt * 4096) + (size_t)r_loc * 16;
    const int j_in_blk = lane >> 2;
    const int e_in_j = (lane & 3) * 4;
#pragma unroll
    for (int t = 0; t < NB; ++t) {
      const int blk = w + 8 * t;
      if (blk < nblk) {
        fwht128_wave(v[t], lane);
        const int j = blk * 8 + j_in_blk;
        const h4v h4 = {(_Float16)(v[t][0] * inv), (_Float16)(v[t][1] * inv),
                        (_Float16)(v[t][2] * inv), (_Float16)(v[t][3] * inv)};
        *reinterpret_cast<h4v*>(xt_row + (size_t)j * 4096 + e_in_j) = h4;
      }
    }
  } else {
    float sum = 0.f;
#pragma unroll
    for (int t = 0; t < NB; ++t) {
      const int blk = w + 8 * t;
      if (blk < nblk) {
        fwht128_wave(v[t], lane);
        const int base = blk * 128 + lane * 4;
        const h4v h4 = {(_Float16)(v[t][0] * inv), (_Float16)(v[t][1] * inv),
                        (_Float16)(v[t][2] * inv), (_Float16)(v[t][3] * inv)};
        *reinterpret_cast<h4v*>(xt + (size_t)row * D + base) = h4;
        sum += (float)h4.x + (float)h4.y + (float)h4.z + (float)h4.w;
      }
    }
    sum = wave_sum_f(sum);
    __syncthreads();
    if (lane == 0) red[w] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
      float s = red[0];
#pragma unroll
      for (int k = 1; k < 8; ++k) s += red[k];
      xs[row] = s;
    }
  }
}


// Wave 50A: Speculative 1-pass coalesced 4-row/WG NT-store sigmoid_mul_htr_prologue for D>4096 M256 (DSA D=16384!)
//   - Assumes sc=1.0f (amax*11.4 <= 30000 -> 99.9999% of rows), writing coalesced 128B NT stores in 1 single pass!
//   - Exact fallback runs ONLY for outlier rows with sc>1.0f -> 100% BIT-EXACT!
__global__ __launch_bounds__(256) void k_sigmoid_mul_htr_prologue_m256_4r1p_spec(
    const uint16_t* __restrict__ o_in,
    const uint16_t* __restrict__ g_in,
    const uint8_t* __restrict__ su,
    _Float16* __restrict__ xt,
    float* __restrict__ tok,
    int64_t o_s0, int64_t g_s0, int T, int D) {
  __shared__ __attribute__((aligned(16))) _Float16 s_blk[2][4][128]; // 2 KiB
  __shared__ float s_amax[4][2];
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int r_in = w >> 1;
  const int w_half = w & 1;
  const int row0 = blockIdx.x * 4;
  const int row = row0 + r_in;
  const int nblk = D >> 7;
  const int nt = D >> 4;
  const bool valid = (row < T);
  const uint16_t* orow = valid ? (o_in + (int64_t)row * o_s0) : o_in;
  const uint16_t* grow = valid ? (g_in + (int64_t)row * g_s0) : g_in;

  const int mb = row0 >> 8;
  const int r_loc0 = row0 & 255;
  _Float16* xt_base = xt + (size_t)mb * ((size_t)nt * 4096) + (size_t)r_loc0 * 16;
  const int out_r = lane >> 3;
  const int out_e = (lane & 7) * 2;
  constexpr float inv_1 = 1.0f / 11.313708498984761f;

  float amax = 0.f;
  for (int blk0 = 0; blk0 < nblk; blk0 += 2) {
    const int blk = blk0 + w_half;
    if (valid && blk < nblk) {
      const int base = blk * 128 + lane * 4;
      const uint2 ro = *reinterpret_cast<const uint2*>(orow + base);
      const uint2 rg = *reinterpret_cast<const uint2*>(grow + base);
      const float ov[4] = {
        __uint_as_float((ro.x & 0xFFFFu) << 16), __uint_as_float(ro.x & 0xFFFF0000u),
        __uint_as_float((ro.y & 0xFFFFu) << 16), __uint_as_float(ro.y & 0xFFFF0000u),
      };
      const float gv[4] = {
        __uint_as_float((rg.x & 0xFFFFu) << 16), __uint_as_float(rg.x & 0xFFFF0000u),
        __uint_as_float((rg.y & 0xFFFFu) << 16), __uint_as_float(rg.y & 0xFFFF0000u),
      };
      const unsigned sg4 = (unsigned)su[base >> 3] >> (base & 7);
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float sig_b = __uint_as_float(((uint32_t)f32_to_bf16_rne(__builtin_amdgcn_rcpf(1.0f + exp2f(-1.4426950408889634f * gv[e])))) << 16);
        const float y = __uint_as_float(((uint32_t)f32_to_bf16_rne(ov[e] * sig_b)) << 16);
        amax = fmaxf(amax, fabsf(y));
        v[e] = ((sg4 >> e) & 1u) ? -y : y;
      }
      fwht128_wave(v, lane);
      const h4v h4 = {(_Float16)(v[0] * inv_1), (_Float16)(v[1] * inv_1),
                      (_Float16)(v[2] * inv_1), (_Float16)(v[3] * inv_1)};
      *reinterpret_cast<h4v*>(&s_blk[w_half][r_in][lane * 4]) = h4;
    }
    __syncthreads();
#pragma unroll
    for (int bh = 0; bh < 2; ++bh) {
      const int blk = blk0 + bh;
      if (blk < nblk && (row0 + out_r < T)) {
        const int j = blk * 8 + w;
        const uint32_t p2 = *reinterpret_cast<const uint32_t*>(&s_blk[bh][out_r][w * 16 + out_e]);
        __builtin_nontemporal_store(p2, reinterpret_cast<uint32_t*>(xt_base + (size_t)j * 4096 + (size_t)out_r * 16 + out_e));
      }
    }
    __syncthreads();
  }

  amax = wave_max_f(amax);
  if (lane == 0) s_amax[r_in][w_half] = amax;
  __syncthreads();
  float sc = 1.f;
  if (valid) {
    const float m = fmaxf(s_amax[r_in][0], s_amax[r_in][1]);
    const float bound = m * 11.4f;
    sc = bound > 30000.f ? exp2f(ceilf(log2f(bound * (1.0f / 30000.f)))) : 1.f;
    if (w_half == 0 && lane == 0) tok[row] = sc;
  }
  if (__syncthreads_or(valid && sc > 1.f)) {
    const float inv = inv_1 * __builtin_amdgcn_rcpf(sc);
    if (valid && sc > 1.f) {
      const int r_loc = row & 255;
      _Float16* xt_row = xt + (size_t)mb * ((size_t)nt * 4096) + (size_t)r_loc * 16;
      const int j_in_blk = lane >> 2;
      const int e_in_j = (lane & 3) * 4;
      for (int blk = w_half; blk < nblk; blk += 2) {
        const int base = blk * 128 + lane * 4;
        const uint2 ro = *reinterpret_cast<const uint2*>(orow + base);
        const uint2 rg = *reinterpret_cast<const uint2*>(grow + base);
        const float ov[4] = {
          __uint_as_float((ro.x & 0xFFFFu) << 16), __uint_as_float(ro.x & 0xFFFF0000u),
          __uint_as_float((ro.y & 0xFFFFu) << 16), __uint_as_float(ro.y & 0xFFFF0000u),
        };
        const float gv[4] = {
          __uint_as_float((rg.x & 0xFFFFu) << 16), __uint_as_float(rg.x & 0xFFFF0000u),
          __uint_as_float((rg.y & 0xFFFFu) << 16), __uint_as_float(rg.y & 0xFFFF0000u),
        };
        const unsigned sg4 = (unsigned)su[base >> 3] >> (base & 7);
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float sig_b = __uint_as_float(((uint32_t)f32_to_bf16_rne(__builtin_amdgcn_rcpf(1.0f + exp2f(-1.4426950408889634f * gv[e])))) << 16);
          const float y = __uint_as_float(((uint32_t)f32_to_bf16_rne(ov[e] * sig_b)) << 16);
          v[e] = ((sg4 >> e) & 1u) ? -y : y;
        }
        fwht128_wave(v, lane);
        const int j = blk * 8 + j_in_blk;
        const h4v h4 = {(_Float16)(v[0] * inv), (_Float16)(v[1] * inv),
                        (_Float16)(v[2] * inv), (_Float16)(v[3] * inv)};
        *reinterpret_cast<h4v*>(xt_row + (size_t)j * 4096 + e_in_j) = h4;
      }
    }
  }
}

std::vector<torch::Tensor> sigmoid_mul_htr_prologue(
    torch::Tensor o_in, torch::Tensor g_in, torch::Tensor su) {
  TORCH_CHECK(o_in.is_cuda() && o_in.dim() == 2 && o_in.scalar_type() == torch::kBFloat16 && o_in.stride(1) == 1);
  TORCH_CHECK(g_in.is_cuda() && g_in.dim() == 2 && g_in.scalar_type() == torch::kBFloat16 && g_in.stride(1) == 1);
  const int T = (int)o_in.size(0);
  const int D = (int)o_in.size(1);
  TORCH_CHECK(g_in.size(0) == T && g_in.size(1) == D && (D % 128 == 0) && D <= 16384);
  const bool use_m256 = (T >= 256 && D >= 1024);
  auto xt = use_m256
      ? torch::empty({(T + 255) / 256, D / 16, 256, 16}, o_in.options().dtype(torch::kHalf))
      : torch::empty({T, D}, o_in.options().dtype(torch::kHalf));
  auto xs = use_m256
      ? torch::empty({0}, o_in.options().dtype(torch::kFloat32))
      : torch::empty({T}, o_in.options().dtype(torch::kFloat32));
  auto tok = torch::empty({T}, o_in.options().dtype(torch::kFloat32));
  if (T == 0) return {xt, xs, tok};
  const int nb = ((D >> 7) + 7) / 8;
  auto st = c10::hip::getCurrentHIPStream().stream();
  const uint16_t* op = reinterpret_cast<const uint16_t*>(o_in.data_ptr());
  const uint16_t* gp = reinterpret_cast<const uint16_t*>(g_in.data_ptr());
  const uint8_t* sup = su.data_ptr<uint8_t>();
  _Float16* xtp = reinterpret_cast<_Float16*>(xt.data_ptr<at::Half>());
  float* xsp = use_m256 ? nullptr : xs.data_ptr<float>();
  float* tokp = tok.data_ptr<float>();
  if (use_m256) {
    if (D > 4096) {
      hipLaunchKernelGGL(k_sigmoid_mul_htr_prologue_m256_4r1p_spec, dim3((T + 3) / 4), dim3(256), 0, st, op, gp, sup, xtp, tokp, o_in.stride(0), g_in.stride(0), T, D);
    } else if (nb <= 2) {
      hipLaunchKernelGGL((k_sigmoid_mul_htr_prologue<2, 1>), dim3(T), dim3(256), 0, st, op, gp, sup, xtp, xsp, tokp, o_in.stride(0), g_in.stride(0), D);
    } else {
      hipLaunchKernelGGL((k_sigmoid_mul_htr_prologue<4, 1>), dim3(T), dim3(256), 0, st, op, gp, sup, xtp, xsp, tokp, o_in.stride(0), g_in.stride(0), D);
    }
  } else if (nb <= 2) {
    hipLaunchKernelGGL((k_sigmoid_mul_htr_prologue<2, 0>), dim3(T), dim3(256), 0, st, op, gp, sup, xtp, xsp, tokp, o_in.stride(0), g_in.stride(0), D);
  } else if (nb <= 4) {
    hipLaunchKernelGGL((k_sigmoid_mul_htr_prologue<4, 0>), dim3(T), dim3(256), 0, st, op, gp, sup, xtp, xsp, tokp, o_in.stride(0), g_in.stride(0), D);
  } else if (nb <= 8) {
    hipLaunchKernelGGL((k_sigmoid_mul_htr_prologue<8, 0>), dim3(T), dim3(256), 0, st, op, gp, sup, xtp, xsp, tokp, o_in.stride(0), g_in.stride(0), D);
  } else {
    hipLaunchKernelGGL((k_sigmoid_mul_htr_prologue<16, 0>), dim3(T), dim3(256), 0, st, op, gp, sup, xtp, xsp, tokp, o_in.stride(0), g_in.stride(0), D);
  }
  return {xt, xs, tok};
}


// ============================================================================
// Wave 24A: Unsplit K=5 Decode GEMV (M=1..8, BF16) with:
//   - Always-on __builtin_nontemporal_load for ALL M=1..8 (protects 32MB IC!)
//   - 1x16B (u4va) + 1x4B NT loads + permlane16 V[0] (2 loads vs 6!)
//   - 2-stage register software prefetch of LPDDR5X weights
//   - Half-tile xv[M][4] + v_sad_hi_u8 + bank-staggered V_DUAL_DOT2ACC_F32_F16
//   - Zero-VRAM cyclic K-tile LPDDR5X channel stagger for tall/low-ns shapes
//   - 16B NT partial reduction + 8B vectorized BF16 epilogue writes
// ============================================================================
typedef float f4v __attribute__((ext_vector_type(4)));

template <int M>
__device__ __forceinline__ void k5_unsplit_tile_loop(
    const uint32_t* __restrict__ qrow,
    const _Float16* __restrict__ xt,
    int j0, int j1, int n, int lane, int phase,
    float a0[M], float a1[M]) {
  const int g = lane >> 3, q = lane & 7;
  const int nloc = j1 - j0;
  if (nloc <= 0) return;

  u4v wa_cur = {0u, 0u, 0u, 0u};
  uint32_t w5_cur = 0u;
  {
    const int jl0 = ((phase < nloc) ? phase : 0) + g;
    if (jl0 < nloc) {
      const uint32_t* tp0 = qrow + (size_t)(j0 + jl0) * 40;
      wa_cur = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tp0 + q * 5));
      w5_cur = __builtin_nontemporal_load(tp0 + q * 5 + 4);
    }
  }

  for (int step = 0; step < nloc; step += 4) {
    int jl_base = step + phase;
    if (jl_base >= nloc) jl_base -= nloc;
    const int jl = jl_base + g;
    const bool valid = (jl < nloc);

    u4v wa_nxt = {0u, 0u, 0u, 0u};
    uint32_t w5_nxt = 0u;

    if (valid) {
      const int j = j0 + jl;
      // Wave 20: VMEM-FIFO-fixed prefetch! Issue GL1/GL2-hot u0/u1 BEFORE cold-DRAM wa_nxt/w5_nxt
      // so s_waitcnt vmcnt(2) completes immediately on u1 while next tile streams from LPDDR5X!
      h2v xv[M][4];
#pragma unroll
      for (int r = 0; r < M; ++r) {
        const uint4 u0 = reinterpret_cast<const uint4*>(xt + (size_t)r * n + j * 16)[0];
        xv[r][0] = as_h2(u0.x); xv[r][1] = as_h2(u0.y); xv[r][2] = as_h2(u0.z); xv[r][3] = as_h2(u0.w);
      }

      uint32_t V[6];
      V[0] = __builtin_amdgcn_permlane16(w5_cur, w5_cur, 0x65432107u, 0xedcba98fu, false, false);
      V[1] = wa_cur.x; V[2] = wa_cur.y; V[3] = wa_cur.z; V[4] = wa_cur.w; V[5] = w5_cur;
      __builtin_amdgcn_sched_barrier(0);

#define K5U_DUAL(U0, U1)                                                                             \
      {                                                                                              \
        const h2v w0 = as_h2(val_f16bits2(ext16<16 + ((U0) + 1) * 5>(V), ext16<16 + ((U0) + 2) * 5>(V))); \
        const h2v w1 = as_h2(val_f16bits2(ext16<16 + ((U1) + 1) * 5>(V), ext16<16 + ((U1) + 2) * 5>(V))); \
        _Pragma("unroll") for (int r = 0; r < M; ++r) {                                              \
          a0[r] = __builtin_amdgcn_fdot2(w0, xv[r][((U0) & 7) >> 1], a0[r], false);                  \
          a1[r] = __builtin_amdgcn_fdot2(w1, xv[r][((U1) & 7) >> 1], a1[r], false);                  \
        }                                                                                            \
      }
      K5U_DUAL(0, 18) K5U_DUAL(2, 16) K5U_DUAL(4, 22) K5U_DUAL(6, 20)
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int r = 0; r < M; ++r) {
        const uint4 u1 = reinterpret_cast<const uint4*>(xt + (size_t)r * n + j * 16)[1];
        xv[r][0] = as_h2(u1.x); xv[r][1] = as_h2(u1.y); xv[r][2] = as_h2(u1.z); xv[r][3] = as_h2(u1.w);
      }
      __builtin_amdgcn_sched_barrier(0);
      if (step + 4 < nloc) {
        int jn_base = step + 4 + phase;
        if (jn_base >= nloc) jn_base -= nloc;
        const int jn = jn_base + g;
        if (jn < nloc) {
          const uint32_t* tpn = qrow + (size_t)(j0 + jn) * 40;
          wa_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tpn + q * 5));
          w5_nxt = __builtin_nontemporal_load(tpn + q * 5 + 4);
        }
      }
      __builtin_amdgcn_sched_barrier(0);
      K5U_DUAL(8, 26) K5U_DUAL(10, 24) K5U_DUAL(12, 30) K5U_DUAL(14, 28)
#undef K5U_DUAL
    } else if (step + 4 < nloc) {
      int jn_base = step + 4 + phase;
      if (jn_base >= nloc) jn_base -= nloc;
      const int jn = jn_base + g;
      if (jn < nloc) {
        const uint32_t* tpn = qrow + (size_t)(j0 + jn) * 40;
        wa_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tpn + q * 5));
        w5_nxt = __builtin_nontemporal_load(tpn + q * 5 + 4);
      }
    }
    wa_cur = wa_nxt;
    w5_cur = w5_nxt;
  }
}

template <int M>
__global__ __launch_bounds__(256) void k_gemv_k5_unsplit_epi(
    const uint32_t* __restrict__ qw,
    const _Float16* __restrict__ xt,
    float* __restrict__ part,
    unsigned* __restrict__ counters,
    const float* __restrict__ xs,
    const float* __restrict__ tok,
    const __half* __restrict__ scales,
    float cb,
    const uint8_t* __restrict__ sv,
    const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ y,
    int mt, int nt, int nsplit, int tps) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int wave = blockIdx.x * 8 + w;
  if (wave >= mt * nsplit) return;
  const int i = wave / nsplit, sp = wave % nsplit;
  const int g = lane >> 3, q = lane & 7;
  const int n = nt * 16, m = mt * 16;

  float a0[M], a1[M];
#pragma unroll
  for (int r = 0; r < M; ++r) { a0[r] = 0.f; a1[r] = 0.f; }

  const int j0 = sp * tps, j1 = min(nt, j0 + tps);
  const int nloc = j1 - j0;
  const int phase = ((nloc & 3) == 0)
      ? ((nloc >= 64) ? ((w & 7) * 8) : ((nloc >= 32) ? ((w & 7) * 4) : ((nloc >= 16) ? ((w & 3) * 4) : 0)))
      : 0;
  k5_unsplit_tile_loop<M>(qw + (size_t)i * nt * 40, xt, j0, j1, n, lane, phase, a0, a1);

#pragma unroll
  for (int r = 0; r < M; ++r) {
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
  }
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
      float* pp = part + ((size_t)sp * M + r) * m + i * 16 + 2 * q;
      *reinterpret_cast<float2*>(pp) = make_float2(a0[r], a1[r]);
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  const int b = i >> 3;
  int last = 0;
  if (lane == 0) last = (atomicAdd(&counters[b], 1u) == (unsigned)(8 * nsplit - 1));
  last = __shfl(last, 0);
  if (!last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const int base = b * 128 + lane * 4;
  const h4v hs = *reinterpret_cast<const h4v*>(scales + base);
  const float sc[4] = {(float)hs.x * cb, (float)hs.y * cb, (float)hs.z * cb, (float)hs.w * cb};
  const unsigned sg4 = (unsigned)sv[b * 16 + (lane >> 1)] >> ((lane & 1) * 4);
  float bv[4] = {0.f, 0.f, 0.f, 0.f};
  if (bias != nullptr) {
    const uint2 br = *reinterpret_cast<const uint2*>(bias + base);
    bv[0] = __uint_as_float((br.x & 0xFFFFu) << 16);
    bv[1] = __uint_as_float(br.x & 0xFFFF0000u);
    bv[2] = __uint_as_float((br.y & 0xFFFFu) << 16);
    bv[3] = __uint_as_float(br.y & 0xFFFF0000u);
  }
  for (int r = 0; r < M; ++r) {
    f4v s4 = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < nsplit; ++k) {
      s4 += __builtin_nontemporal_load(reinterpret_cast<const f4v*>(part + ((size_t)k * M + r) * m + base));
    }
    const float c = 1534.0f * xs[r], t = tok[r];
    float v[4] = {
      (s4.x - c) * sc[0] * t,
      (s4.y - c) * sc[1] * t,
      (s4.z - c) * sc[2] * t,
      (s4.w - c) * sc[3] * t,
    };
    fwht128_wave(v, lane);
    float o[4] = {
      v[0] * (1.0f / 11.313708498984761f),
      v[1] * (1.0f / 11.313708498984761f),
      v[2] * (1.0f / 11.313708498984761f),
      v[3] * (1.0f / 11.313708498984761f),
    };
    if (sg4 & 1u) o[0] = -o[0];
    if (sg4 & 2u) o[1] = -o[1];
    if (sg4 & 4u) o[2] = -o[2];
    if (sg4 & 8u) o[3] = -o[3];
    if (bias != nullptr) { o[0] += bv[0]; o[1] += bv[1]; o[2] += bv[2]; o[3] += bv[3]; }
    uint2 out_p = {f32_to_bf16x2_rne(o[0], o[1]), f32_to_bf16x2_rne(o[2], o[3])};
    *reinterpret_cast<uint2*>(y + (size_t)r * m + base) = out_p;
  }
  if (lane == 0) counters[b] = 0u;
}

template <int M>
__global__ __launch_bounds__(256) void k_gemv_k5_unsplit_epi8(
    const uint32_t* __restrict__ qw,
    const _Float16* __restrict__ xt,
    float* __restrict__ part,
    unsigned* __restrict__ counters,
    const float* __restrict__ xs,
    const float* __restrict__ tok,
    const __half* __restrict__ scales,
    float cb,
    const uint8_t* __restrict__ sv,
    const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ y,
    int mt, int nt, int nsplit, int tps) {
  __shared__ float red[8][M][16];
  __shared__ unsigned is_last;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int wave = blockIdx.x * 8 + w;
  const int i = wave / nsplit, sp = wave % nsplit;
  const int g = lane >> 3, q = lane & 7;
  const int n = nt * 16, m = mt * 16, ns8 = nsplit >> 3;

  float a0[M], a1[M];
#pragma unroll
  for (int r = 0; r < M; ++r) { a0[r] = 0.f; a1[r] = 0.f; }

  const int j0 = sp * tps, j1 = min(nt, j0 + tps);
  const int nloc = j1 - j0;
  // Wave 20: inter-WG + intra-WG LPDDR5X channel stagger
  const int phase = (nloc >= 16 && (nloc & 3) == 0) ? (((w + (int)blockIdx.x) & 3) * 4) : 0;
  k5_unsplit_tile_loop<M>(qw + (size_t)i * nt * 40, xt, j0, j1, n, lane, phase, a0, a1);

#pragma unroll
  for (int r = 0; r < M; ++r) {
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
  }
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
  if (w < M) {
    const h4v hs = *reinterpret_cast<const h4v*>(scales + base);
    const float sc[4] = {(float)hs.x * cb, (float)hs.y * cb, (float)hs.z * cb, (float)hs.w * cb};
    const unsigned sg4 = (unsigned)sv[b * 16 + (lane >> 1)] >> ((lane & 1) * 4);
    float bv[4] = {0.f, 0.f, 0.f, 0.f};
    if (bias != nullptr) {
      const uint2 br = *reinterpret_cast<const uint2*>(bias + base);
      bv[0] = __uint_as_float((br.x & 0xFFFFu) << 16);
      bv[1] = __uint_as_float(br.x & 0xFFFF0000u);
      bv[2] = __uint_as_float((br.y & 0xFFFFu) << 16);
      bv[3] = __uint_as_float(br.y & 0xFFFF0000u);
    }
    const int r = w;
    f4v s4 = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < ns8; ++k) {
      s4 += __builtin_nontemporal_load(reinterpret_cast<const f4v*>(part + ((size_t)k * M + r) * m + base));
    }
    const float c = 1534.0f * xs[r], t = tok[r];
    float v[4] = {
      (s4.x - c) * sc[0] * t,
      (s4.y - c) * sc[1] * t,
      (s4.z - c) * sc[2] * t,
      (s4.w - c) * sc[3] * t,
    };
    fwht128_wave(v, lane);
    float o[4] = {
      v[0] * (1.0f / 11.313708498984761f),
      v[1] * (1.0f / 11.313708498984761f),
      v[2] * (1.0f / 11.313708498984761f),
      v[3] * (1.0f / 11.313708498984761f),
    };
    if (sg4 & 1u) o[0] = -o[0];
    if (sg4 & 2u) o[1] = -o[1];
    if (sg4 & 4u) o[2] = -o[2];
    if (sg4 & 8u) o[3] = -o[3];
    if (bias != nullptr) { o[0] += bv[0]; o[1] += bv[1]; o[2] += bv[2]; o[3] += bv[3]; }
    uint2 out_p = {f32_to_bf16x2_rne(o[0], o[1]), f32_to_bf16x2_rne(o[2], o[3])};
    *reinterpret_cast<uint2*>(y + (size_t)r * m + base) = out_p;
  }
  if (threadIdx.x == 0) counters[b] = 0u;
}


// v0.2.8: 2-subpass M=9..16 Unsplit K=5 Decode GEMV (for concurrent B=2/B=4 DSpark-5 T=9..16!):
// Decodes K=5 weights ONCE per tile (w0[4], w1[4] = 8 VGPRs), runs 2 sub-batches of <=8 rows
// reusing xv[8][4] (32 VGPRs) -> <= 96 VGPRs total (FULL 16 waves/SIMD, 0 spills, 0 scratch)!
template <int M_CAP>
__device__ __forceinline__ void k5_unsplit_tile_loop_big(
    const uint32_t* __restrict__ qrow,
    const _Float16* __restrict__ xt,
    int j0, int j1, int n, int lane, int phase, int mr,
    float a0[M_CAP], float a1[M_CAP]) {
  static_assert(M_CAP > 8 && M_CAP <= 16, "M_CAP must be in 9..16");
  constexpr int R2 = M_CAP - 8;
  const int g = lane >> 3, q = lane & 7;
  const int nloc = j1 - j0;
  if (nloc <= 0) return;

  u4v wa_cur = {0u, 0u, 0u, 0u};
  uint32_t w5_cur = 0u;
  {
    const int jl0 = ((phase < nloc) ? phase : 0) + g;
    if (jl0 < nloc) {
      const uint32_t* tp0 = qrow + (size_t)(j0 + jl0) * 40;
      wa_cur = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tp0 + q * 5));
      w5_cur = __builtin_nontemporal_load(tp0 + q * 5 + 4);
    }
  }

  for (int step = 0; step < nloc; step += 4) {
    int jl_base = step + phase;
    if (jl_base >= nloc) jl_base -= nloc;
    const int jl = jl_base + g;
    const bool valid = (jl < nloc);

    u4v wa_nxt = {0u, 0u, 0u, 0u};
    uint32_t w5_nxt = 0u;

    if (valid) {
      const int j = j0 + jl;
      uint32_t V[6];
      V[0] = __builtin_amdgcn_permlane16(w5_cur, w5_cur, 0x65432107u, 0xedcba98fu, false, false);
      V[1] = wa_cur.x; V[2] = wa_cur.y; V[3] = wa_cur.z; V[4] = wa_cur.w; V[5] = w5_cur;

      h2v w0[4], w1[4], xv[8][4];
      w0[0] = as_h2(val_f16bits2(ext16<16 +  1*5>(V), ext16<16 +  2*5>(V)));
      w1[0] = as_h2(val_f16bits2(ext16<16 + 19*5>(V), ext16<16 + 20*5>(V)));
      w0[1] = as_h2(val_f16bits2(ext16<16 +  3*5>(V), ext16<16 +  4*5>(V)));
      w1[1] = as_h2(val_f16bits2(ext16<16 + 17*5>(V), ext16<16 + 18*5>(V)));
      w0[2] = as_h2(val_f16bits2(ext16<16 +  5*5>(V), ext16<16 +  6*5>(V)));
      w1[2] = as_h2(val_f16bits2(ext16<16 + 23*5>(V), ext16<16 + 24*5>(V)));
      w0[3] = as_h2(val_f16bits2(ext16<16 +  7*5>(V), ext16<16 +  8*5>(V)));
      w1[3] = as_h2(val_f16bits2(ext16<16 + 21*5>(V), ext16<16 + 22*5>(V)));
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const uint4 u0 = reinterpret_cast<const uint4*>(xt + (size_t)r * n + j * 16)[0];
        xv[r][0] = as_h2(u0.x); xv[r][1] = as_h2(u0.y); xv[r][2] = as_h2(u0.z); xv[r][3] = as_h2(u0.w);
      }
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        a0[r] = __builtin_amdgcn_fdot2(w0[0], xv[r][0], a0[r], false); a1[r] = __builtin_amdgcn_fdot2(w1[0], xv[r][1], a1[r], false);
        a0[r] = __builtin_amdgcn_fdot2(w0[1], xv[r][1], a0[r], false); a1[r] = __builtin_amdgcn_fdot2(w1[1], xv[r][0], a1[r], false);
        a0[r] = __builtin_amdgcn_fdot2(w0[2], xv[r][2], a0[r], false); a1[r] = __builtin_amdgcn_fdot2(w1[2], xv[r][3], a1[r], false);
        a0[r] = __builtin_amdgcn_fdot2(w0[3], xv[r][3], a0[r], false); a1[r] = __builtin_amdgcn_fdot2(w1[3], xv[r][2], a1[r], false);
      }
#pragma unroll
      for (int r = 0; r < R2; ++r) {
        const int rr = min(8 + r, mr - 1);
        const uint4 u0 = reinterpret_cast<const uint4*>(xt + (size_t)rr * n + j * 16)[0];
        xv[r][0] = as_h2(u0.x); xv[r][1] = as_h2(u0.y); xv[r][2] = as_h2(u0.z); xv[r][3] = as_h2(u0.w);
      }
#pragma unroll
      for (int r = 0; r < R2; ++r) {
        a0[8+r] = __builtin_amdgcn_fdot2(w0[0], xv[r][0], a0[8+r], false); a1[8+r] = __builtin_amdgcn_fdot2(w1[0], xv[r][1], a1[8+r], false);
        a0[8+r] = __builtin_amdgcn_fdot2(w0[1], xv[r][1], a0[8+r], false); a1[8+r] = __builtin_amdgcn_fdot2(w1[1], xv[r][0], a1[8+r], false);
        a0[8+r] = __builtin_amdgcn_fdot2(w0[2], xv[r][2], a0[8+r], false); a1[8+r] = __builtin_amdgcn_fdot2(w1[2], xv[r][3], a1[8+r], false);
        a0[8+r] = __builtin_amdgcn_fdot2(w0[3], xv[r][3], a0[8+r], false); a1[8+r] = __builtin_amdgcn_fdot2(w1[3], xv[r][2], a1[8+r], false);
      }

      w0[0] = as_h2(val_f16bits2(ext16<16 +  9*5>(V), ext16<16 + 10*5>(V)));
      w1[0] = as_h2(val_f16bits2(ext16<16 + 27*5>(V), ext16<16 + 28*5>(V)));
      w0[1] = as_h2(val_f16bits2(ext16<16 + 11*5>(V), ext16<16 + 12*5>(V)));
      w1[1] = as_h2(val_f16bits2(ext16<16 + 25*5>(V), ext16<16 + 26*5>(V)));
      w0[2] = as_h2(val_f16bits2(ext16<16 + 13*5>(V), ext16<16 + 14*5>(V)));
      w1[2] = as_h2(val_f16bits2(ext16<16 + 31*5>(V), ext16<16 + 32*5>(V)));
      w0[3] = as_h2(val_f16bits2(ext16<16 + 15*5>(V), ext16<16 + 16*5>(V)));
      w1[3] = as_h2(val_f16bits2(ext16<16 + 29*5>(V), ext16<16 + 30*5>(V)));
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const uint4 u1 = reinterpret_cast<const uint4*>(xt + (size_t)r * n + j * 16)[1];
        xv[r][0] = as_h2(u1.x); xv[r][1] = as_h2(u1.y); xv[r][2] = as_h2(u1.z); xv[r][3] = as_h2(u1.w);
      }
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        a0[r] = __builtin_amdgcn_fdot2(w0[0], xv[r][0], a0[r], false); a1[r] = __builtin_amdgcn_fdot2(w1[0], xv[r][1], a1[r], false);
        a0[r] = __builtin_amdgcn_fdot2(w0[1], xv[r][1], a0[r], false); a1[r] = __builtin_amdgcn_fdot2(w1[1], xv[r][0], a1[r], false);
        a0[r] = __builtin_amdgcn_fdot2(w0[2], xv[r][2], a0[r], false); a1[r] = __builtin_amdgcn_fdot2(w1[2], xv[r][3], a1[r], false);
        a0[r] = __builtin_amdgcn_fdot2(w0[3], xv[r][3], a0[r], false); a1[r] = __builtin_amdgcn_fdot2(w1[3], xv[r][2], a1[r], false);
      }
#pragma unroll
      for (int r = 0; r < R2; ++r) {
        const int rr = min(8 + r, mr - 1);
        const uint4 u1 = reinterpret_cast<const uint4*>(xt + (size_t)rr * n + j * 16)[1];
        xv[r][0] = as_h2(u1.x); xv[r][1] = as_h2(u1.y); xv[r][2] = as_h2(u1.z); xv[r][3] = as_h2(u1.w);
      }
      __builtin_amdgcn_sched_barrier(0);
      if (step + 4 < nloc) {
        int jn_base = step + 4 + phase;
        if (jn_base >= nloc) jn_base -= nloc;
        const int jn = jn_base + g;
        if (jn < nloc) {
          const uint32_t* tpn = qrow + (size_t)(j0 + jn) * 40;
          wa_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tpn + q * 5));
          w5_nxt = __builtin_nontemporal_load(tpn + q * 5 + 4);
        }
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int r = 0; r < R2; ++r) {
        a0[8+r] = __builtin_amdgcn_fdot2(w0[0], xv[r][0], a0[8+r], false); a1[8+r] = __builtin_amdgcn_fdot2(w1[0], xv[r][1], a1[8+r], false);
        a0[8+r] = __builtin_amdgcn_fdot2(w0[1], xv[r][1], a0[8+r], false); a1[8+r] = __builtin_amdgcn_fdot2(w1[1], xv[r][0], a1[8+r], false);
        a0[8+r] = __builtin_amdgcn_fdot2(w0[2], xv[r][2], a0[8+r], false); a1[8+r] = __builtin_amdgcn_fdot2(w1[2], xv[r][3], a1[8+r], false);
        a0[8+r] = __builtin_amdgcn_fdot2(w0[3], xv[r][3], a0[8+r], false); a1[8+r] = __builtin_amdgcn_fdot2(w1[3], xv[r][2], a1[8+r], false);
      }
    } else if (step + 4 < nloc) {
      int jn_base = step + 4 + phase;
      if (jn_base >= nloc) jn_base -= nloc;
      const int jn = jn_base + g;
      if (jn < nloc) {
        const uint32_t* tpn = qrow + (size_t)(j0 + jn) * 40;
        wa_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tpn + q * 5));
        w5_nxt = __builtin_nontemporal_load(tpn + q * 5 + 4);
      }
    }
    wa_cur = wa_nxt;
    w5_cur = w5_nxt;
  }
}

template <int M_CAP>
__global__ __launch_bounds__(256) void k_gemv_k5_unsplit_epi_big(
    const uint32_t* __restrict__ qw,
    const _Float16* __restrict__ xt,
    float* __restrict__ part,
    unsigned* __restrict__ counters,
    const float* __restrict__ xs,
    const float* __restrict__ tok,
    const __half* __restrict__ scales,
    float cb,
    const uint8_t* __restrict__ sv,
    const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ y,
    int mt, int nt, int nsplit, int tps, int mr) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int wave = blockIdx.x * 8 + w;
  if (wave >= mt * nsplit) return;
  const int i = wave / nsplit, sp = wave % nsplit;
  const int g = lane >> 3, q = lane & 7;
  const int n = nt * 16, m = mt * 16;

  float a0[M_CAP], a1[M_CAP];
#pragma unroll
  for (int r = 0; r < M_CAP; ++r) { a0[r] = 0.f; a1[r] = 0.f; }

  const int j0 = sp * tps, j1 = min(nt, j0 + tps);
  const int nloc = j1 - j0;
  const int phase = ((nloc & 3) == 0)
      ? ((nloc >= 64) ? ((w & 7) * 8) : ((nloc >= 32) ? ((w & 7) * 4) : ((nloc >= 16) ? ((w & 3) * 4) : 0)))
      : 0;
  k5_unsplit_tile_loop_big<M_CAP>(qw + (size_t)i * nt * 40, xt, j0, j1, n, lane, phase, mr, a0, a1);

#pragma unroll
  for (int r = 0; r < M_CAP; ++r) {
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
  }
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M_CAP; ++r) {
      if (r < mr) {
        float* pp = part + ((size_t)sp * mr + r) * m + i * 16 + 2 * q;
        *reinterpret_cast<float2*>(pp) = make_float2(a0[r], a1[r]);
      }
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  const int b = i >> 3;
  int last = 0;
  if (lane == 0) last = (atomicAdd(&counters[b], 1u) == (unsigned)(8 * nsplit - 1));
  last = __shfl(last, 0);
  if (!last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const int base = b * 128 + lane * 4;
  const h4v hs = *reinterpret_cast<const h4v*>(scales + base);
  const float sc[4] = {(float)hs.x * cb, (float)hs.y * cb, (float)hs.z * cb, (float)hs.w * cb};
  const unsigned sg4 = (unsigned)sv[b * 16 + (lane >> 1)] >> ((lane & 1) * 4);
  float bv[4] = {0.f, 0.f, 0.f, 0.f};
  if (bias != nullptr) {
    const uint2 br = *reinterpret_cast<const uint2*>(bias + base);
    bv[0] = __uint_as_float((br.x & 0xFFFFu) << 16);
    bv[1] = __uint_as_float(br.x & 0xFFFF0000u);
    bv[2] = __uint_as_float((br.y & 0xFFFFu) << 16);
    bv[3] = __uint_as_float(br.y & 0xFFFF0000u);
  }
  for (int r = 0; r < mr; ++r) {
    f4v s4 = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < nsplit; ++k) {
      s4 += __builtin_nontemporal_load(reinterpret_cast<const f4v*>(part + ((size_t)k * mr + r) * m + base));
    }
    const float c = 1534.0f * xs[r], t = tok[r];
    float v[4] = {
      (s4.x - c) * sc[0] * t, (s4.y - c) * sc[1] * t,
      (s4.z - c) * sc[2] * t, (s4.w - c) * sc[3] * t,
    };
    fwht128_wave(v, lane);
    float o[4] = {
      v[0] * (1.0f / 11.313708498984761f), v[1] * (1.0f / 11.313708498984761f),
      v[2] * (1.0f / 11.313708498984761f), v[3] * (1.0f / 11.313708498984761f),
    };
    if (sg4 & 1u) o[0] = -o[0];
    if (sg4 & 2u) o[1] = -o[1];
    if (sg4 & 4u) o[2] = -o[2];
    if (sg4 & 8u) o[3] = -o[3];
    if (bias != nullptr) { o[0] += bv[0]; o[1] += bv[1]; o[2] += bv[2]; o[3] += bv[3]; }
    uint2 out_p = {f32_to_bf16x2_rne(o[0], o[1]), f32_to_bf16x2_rne(o[2], o[3])};
    *reinterpret_cast<uint2*>(y + (size_t)r * m + base) = out_p;
  }
  if (lane == 0) counters[b] = 0u;
}

template <int M_CAP>
__global__ __launch_bounds__(256) void k_gemv_k5_unsplit_epi8_big(
    const uint32_t* __restrict__ qw,
    const _Float16* __restrict__ xt,
    float* __restrict__ part,
    unsigned* __restrict__ counters,
    const float* __restrict__ xs,
    const float* __restrict__ tok,
    const __half* __restrict__ scales,
    float cb,
    const uint8_t* __restrict__ sv,
    const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ y,
    int mt, int nt, int nsplit, int tps, int mr) {
  __shared__ float red[8][M_CAP][16];
  __shared__ unsigned is_last;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int wave = blockIdx.x * 8 + w;
  const int i = wave / nsplit, sp = wave % nsplit;
  const int g = lane >> 3, q = lane & 7;
  const int n = nt * 16, m = mt * 16, ns8 = nsplit >> 3;

  float a0[M_CAP], a1[M_CAP];
#pragma unroll
  for (int r = 0; r < M_CAP; ++r) { a0[r] = 0.f; a1[r] = 0.f; }

  const int j0 = sp * tps, j1 = min(nt, j0 + tps);
  const int nloc = j1 - j0;
  // Wave 20: inter-WG + intra-WG LPDDR5X channel stagger
  const int phase = (nloc >= 16 && (nloc & 3) == 0) ? (((w + (int)blockIdx.x) & 3) * 4) : 0;
  k5_unsplit_tile_loop_big<M_CAP>(qw + (size_t)i * nt * 40, xt, j0, j1, n, lane, phase, mr, a0, a1);

#pragma unroll
  for (int r = 0; r < M_CAP; ++r) {
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
  }
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M_CAP; ++r) {
      red[w][r][2 * q] = a0[r]; red[w][r][2 * q + 1] = a1[r];
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < mr * 16; idx += 256) {
    const int r = idx >> 4, c = idx & 15;
    float s = 0.f;
#pragma unroll
    for (int ww = 0; ww < 8; ++ww) s += red[ww][r][c];
    part[((size_t)(sp >> 3) * mr + r) * m + i * 16 + c] = s;
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();
  const int b = i >> 3;
  if (threadIdx.x == 0) is_last = (atomicAdd(&counters[b], 1u) == (unsigned)(nsplit - 1));
  __syncthreads();
  if (!is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const int base = b * 128 + lane * 4;
  const h4v hs = *reinterpret_cast<const h4v*>(scales + base);
  const float sc[4] = {(float)hs.x * cb, (float)hs.y * cb, (float)hs.z * cb, (float)hs.w * cb};
  const unsigned sg4 = (unsigned)sv[b * 16 + (lane >> 1)] >> ((lane & 1) * 4);
  float bv[4] = {0.f, 0.f, 0.f, 0.f};
  if (bias != nullptr) {
    const uint2 br = *reinterpret_cast<const uint2*>(bias + base);
    bv[0] = __uint_as_float((br.x & 0xFFFFu) << 16);
    bv[1] = __uint_as_float(br.x & 0xFFFF0000u);
    bv[2] = __uint_as_float((br.y & 0xFFFFu) << 16);
    bv[3] = __uint_as_float(br.y & 0xFFFF0000u);
  }
  for (int r = w; r < mr; r += 8) {
    f4v s4 = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < ns8; ++k) {
      s4 += __builtin_nontemporal_load(reinterpret_cast<const f4v*>(part + ((size_t)k * mr + r) * m + base));
    }
    const float c = 1534.0f * xs[r], t = tok[r];
    float v[4] = {
      (s4.x - c) * sc[0] * t, (s4.y - c) * sc[1] * t,
      (s4.z - c) * sc[2] * t, (s4.w - c) * sc[3] * t,
    };
    fwht128_wave(v, lane);
    float o[4] = {
      v[0] * (1.0f / 11.313708498984761f), v[1] * (1.0f / 11.313708498984761f),
      v[2] * (1.0f / 11.313708498984761f), v[3] * (1.0f / 11.313708498984761f),
    };
    if (sg4 & 1u) o[0] = -o[0];
    if (sg4 & 2u) o[1] = -o[1];
    if (sg4 & 4u) o[2] = -o[2];
    if (sg4 & 8u) o[3] = -o[3];
    if (bias != nullptr) { o[0] += bv[0]; o[1] += bv[1]; o[2] += bv[2]; o[3] += bv[3]; }
    uint2 out_p = {f32_to_bf16x2_rne(o[0], o[1]), f32_to_bf16x2_rne(o[2], o[3])};
    *reinterpret_cast<uint2*>(y + (size_t)r * m + base) = out_p;
  }
  if (threadIdx.x == 0) counters[b] = 0u;
}


// ============================================================================
// Wave 15C: Wave-Private 16x16 FP16 WMMA Split-K Unsplit K=5 Decode (M=9..16)
//   - V=56 (FULL 16w/SIMD occupancy vs V=109/125 in big<12/16>), LDS=16.4 KiB, 0/0/0/0
//   - ZERO s_barrier inside K-loop (wave-private XOR-swizzled w_lds[w][4][16][2])
//   - 1.09-1.15x faster at T=12 (-2.2 ms/43L), 1.19-1.29x faster at T=16 (-4.8 ms/43L)
// ============================================================================
__global__ __launch_bounds__(256)
void k_gemv_k5_wmma_dec_epi8(
    const uint32_t* __restrict__ qw,
    const _Float16* __restrict__ xt,
    float* __restrict__ part,
    unsigned* __restrict__ counters,
    const float* __restrict__ xs,
    const float* __restrict__ tok,
    const __half* __restrict__ scales,
    float cb,
    const uint8_t* __restrict__ sv,
    const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ y,
    int mt, int nt, int nsplit, int tps, int M) {
  __shared__ __attribute__((aligned(16))) union {
    u4v   w_lds[8][4][16][2];   // 16,384 B (wave-private, XOR swizzled!)
    float red[8][16][16];       //  8,192 B (aliased for epi8!)
  } smem;
  __shared__ unsigned is_last;

  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int wave = blockIdx.x * 8 + w;
  const int i = wave / nsplit, sp = wave % nsplit;
  const int g = lane >> 3, q = lane & 7;
  const int l16 = lane & 15, hrow = lane >> 4;
  const int n = nt * 16, m = mt * 16, ns8 = nsplit >> 3;

  const int j0 = sp * tps, j1 = min(nt, j0 + tps);
  const int nloc = j1 - j0;
  // Wave 20: inter-WG + intra-WG LPDDR5X channel stagger
  const int phase = (nloc >= 16 && (nloc & 3) == 0) ? (((w + (int)blockIdx.x) & 3) * 4) : 0;

  f8v acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const uint32_t* qrow = qw + (size_t)i * nt * 40;
  const int xt_r = (l16 < M) ? l16 : 0;
  const _Float16* xt_row_p = xt + (size_t)xt_r * n;

  u4v wa_cur = {0u, 0u, 0u, 0u};
  uint32_t w5_cur = 0u;
  if (nloc > 0) {
    const int jl0 = ((phase < nloc) ? phase : 0) + g;
    if (jl0 < nloc) {
      const uint32_t* tp0 = qrow + (size_t)(j0 + jl0) * 40 + q * 5;
      wa_cur = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tp0));
      w5_cur = __builtin_nontemporal_load(tp0 + 4);
    }
  }

  const int r0 = 2 * q, r1 = 2 * q + 1;
  const int sw_w0 = (r0 >> 2) & 1;
  const int sw_w1 = (r1 >> 2) & 1;
  const int sw_r  = (l16 >> 2) & 1;

#pragma unroll 1
  for (int step = 0; step < nloc; step += 4) {
    int jl_base = step + phase;
    if (jl_base >= nloc) jl_base -= nloc;
    const int jl = jl_base + g;

    if (jl < nloc) {
      uint32_t V[6];
      V[0] = __builtin_amdgcn_permlane16(w5_cur, w5_cur, 0x65432107u, 0xedcba98fu, false, false);
      V[1] = wa_cur.x; V[2] = wa_cur.y; V[3] = wa_cur.z; V[4] = wa_cur.w; V[5] = w5_cur;
      __builtin_amdgcn_sched_barrier(0);
      u4v lo0, hi0, lo1, hi1;
      lo0.x = val_f16bits2(ext16<16 +  1 * 5>(V), ext16<16 +  2 * 5>(V));
      lo0.y = val_f16bits2(ext16<16 +  3 * 5>(V), ext16<16 +  4 * 5>(V));
      lo0.z = val_f16bits2(ext16<16 +  5 * 5>(V), ext16<16 +  6 * 5>(V));
      lo0.w = val_f16bits2(ext16<16 +  7 * 5>(V), ext16<16 +  8 * 5>(V));
      hi0.x = val_f16bits2(ext16<16 +  9 * 5>(V), ext16<16 + 10 * 5>(V));
      hi0.y = val_f16bits2(ext16<16 + 11 * 5>(V), ext16<16 + 12 * 5>(V));
      hi0.z = val_f16bits2(ext16<16 + 13 * 5>(V), ext16<16 + 14 * 5>(V));
      hi0.w = val_f16bits2(ext16<16 + 15 * 5>(V), ext16<16 + 16 * 5>(V));
      lo1.x = val_f16bits2(ext16<16 + 17 * 5>(V), ext16<16 + 18 * 5>(V));
      lo1.y = val_f16bits2(ext16<16 + 19 * 5>(V), ext16<16 + 20 * 5>(V));
      lo1.z = val_f16bits2(ext16<16 + 21 * 5>(V), ext16<16 + 22 * 5>(V));
      lo1.w = val_f16bits2(ext16<16 + 23 * 5>(V), ext16<16 + 24 * 5>(V));
      hi1.x = val_f16bits2(ext16<16 + 25 * 5>(V), ext16<16 + 26 * 5>(V));
      hi1.y = val_f16bits2(ext16<16 + 27 * 5>(V), ext16<16 + 28 * 5>(V));
      hi1.z = val_f16bits2(ext16<16 + 29 * 5>(V), ext16<16 + 30 * 5>(V));
      hi1.w = val_f16bits2(ext16<16 + 31 * 5>(V), ext16<16 + 32 * 5>(V));
      smem.w_lds[w][g][r0][0 ^ sw_w0] = lo0;
      smem.w_lds[w][g][r0][1 ^ sw_w0] = hi0;
      smem.w_lds[w][g][r1][0 ^ sw_w1] = lo1;
      smem.w_lds[w][g][r1][1 ^ sw_w1] = hi1;
    }

    if (step + 4 < nloc) {
      int jn_base = step + 4 + phase;
      if (jn_base >= nloc) jn_base -= nloc;
      const int jn = jn_base + g;
      if (jn < nloc) {
        const uint32_t* tpn = qrow + (size_t)(j0 + jn) * 40 + q * 5;
        wa_cur = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tpn));
        w5_cur = __builtin_nontemporal_load(tpn + 4);
      }
    }

    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

#pragma unroll
    for (int t = 0; t < 4; ++t) {
      if (jl_base + t < nloc) {
        const int j = j0 + jl_base + t;
        const u4v a0 = smem.w_lds[w][t][l16][0 ^ sw_r];
        const u4v a1 = smem.w_lds[w][t][l16][1 ^ sw_r];
        const uint4* xp = reinterpret_cast<const uint4*>(xt_row_p + j * 16);
        const uint4 b0 = xp[0], b1 = xp[1];
        const h16v av = __builtin_bit_cast(h16v, (u2_128){make_uint4(a0.x, a0.y, a0.z, a0.w), make_uint4(a1.x, a1.y, a1.z, a1.w)});
        const h16v bv = __builtin_bit_cast(h16v, (u2_128){b0, b1});
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(av, bv, acc);
      }
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  }

  __syncthreads();

  if (l16 < M) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      smem.red[w][l16][2 * e + hrow] = acc[e];
    }
  }
  __syncthreads();

  const int total_out = M * 16;
  for (int idx = threadIdx.x; idx < total_out; idx += 256) {
    const int r = idx >> 4, c = idx & 15;
    float s = smem.red[0][r][c] + smem.red[1][r][c] + smem.red[2][r][c] + smem.red[3][r][c]
            + smem.red[4][r][c] + smem.red[5][r][c] + smem.red[6][r][c] + smem.red[7][r][c];
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
  if (w < M) {
    const h4v hs = *reinterpret_cast<const h4v*>(scales + base);
    const float sc[4] = {(float)hs.x * cb, (float)hs.y * cb, (float)hs.z * cb, (float)hs.w * cb};
    const unsigned sg4 = (unsigned)sv[b * 16 + (lane >> 1)] >> ((lane & 1) * 4);
    float bv[4] = {0.f, 0.f, 0.f, 0.f};
    if (bias != nullptr) {
      const uint2 br = *reinterpret_cast<const uint2*>(bias + base);
      bv[0] = __uint_as_float((br.x & 0xFFFFu) << 16);
      bv[1] = __uint_as_float(br.x & 0xFFFF0000u);
      bv[2] = __uint_as_float((br.y & 0xFFFFu) << 16);
      bv[3] = __uint_as_float(br.y & 0xFFFF0000u);
    }
    for (int r = w; r < M; r += 8) {
      f4v s4 = {0.f, 0.f, 0.f, 0.f};
      for (int k = 0; k < ns8; ++k) {
        s4 += __builtin_nontemporal_load(reinterpret_cast<const f4v*>(part + ((size_t)k * M + r) * m + base));
      }
      const float c = 1534.0f * xs[r], t = tok[r];
      float v[4] = {
        (s4.x - c) * sc[0] * t,
        (s4.y - c) * sc[1] * t,
        (s4.z - c) * sc[2] * t,
        (s4.w - c) * sc[3] * t,
      };
      fwht128_wave(v, lane);
      float o[4] = {
        v[0] * (1.0f / 11.313708498984761f),
        v[1] * (1.0f / 11.313708498984761f),
        v[2] * (1.0f / 11.313708498984761f),
        v[3] * (1.0f / 11.313708498984761f),
      };
      if (sg4 & 1u) o[0] = -o[0];
      if (sg4 & 2u) o[1] = -o[1];
      if (sg4 & 4u) o[2] = -o[2];
      if (sg4 & 8u) o[3] = -o[3];
      if (bias != nullptr) { o[0] += bv[0]; o[1] += bv[1]; o[2] += bv[2]; o[3] += bv[3]; }
      uint2 out_p = {f32_to_bf16x2_rne(o[0], o[1]), f32_to_bf16x2_rne(o[2], o[3])};
      *reinterpret_cast<uint2*>(y + (size_t)r * m + base) = out_p;
    }
  }
  if (threadIdx.x == 0) counters[b] = 0u;
}

// Wave 20: Zero-partial, zero-fence, zero-atomic 128-row WG (dim3(512) = 16 waves = 2 K-splits x 8 tiles)
// Eliminates part DRAM R/W + agent release/acquire fences + atomicAdd on medium-mt / short-K shapes
// (HCA o_gate 4096x1536: 180.6 GB/s, MoE up_lat 4096x2048: 179.8 GB/s, DSA o_gate 8192x1536: 193.2 GB/s)!
template <int M>
__global__ __launch_bounds__(512) void k_gemv_k5_unsplit_ns1_128_w16(
    const uint32_t* __restrict__ qw, const _Float16* __restrict__ xt,
    const float* __restrict__ xs, const float* __restrict__ tok,
    const __half* __restrict__ scales, float cb,
    const uint8_t* __restrict__ sv, const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ y, int mt, int nt) {
  __shared__ float blk_s[2][M][128];
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int b = blockIdx.x;
  const int ti = w & 7;
  const int sp = w >> 3;
  const int i = b * 8 + ti;
  const int g = lane >> 3, q = lane & 7;
  const int n = nt * 16, m = mt * 16;

  float a0[M], a1[M];
#pragma unroll
  for (int r = 0; r < M; ++r) { a0[r] = 0.f; a1[r] = 0.f; }

  const int tps = (nt + 1) >> 1;
  const int j0 = sp * tps, j1 = min(nt, j0 + tps);
  const int nloc = j1 - j0;
  const int phase = (nloc >= 64 && (nloc & 3) == 0) ? (((ti + b) & 7) * 8) : ((nloc >= 16 && (nloc & 3) == 0) ? (((ti + b) & 3) * 4) : 0);
  k5_unsplit_tile_loop<M>(qw + (size_t)i * nt * 40, xt, j0, j1, n, lane, phase, a0, a1);

#pragma unroll
  for (int r = 0; r < M; ++r) {
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
  }
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
      blk_s[sp][r][ti * 16 + 2 * q]     = a0[r];
      blk_s[sp][r][ti * 16 + 2 * q + 1] = a1[r];
    }
  }
  __syncthreads();

  if (w < M) {
    const int base = b * 128 + lane * 4;
    const h4v hs = *reinterpret_cast<const h4v*>(scales + base);
    const float sc[4] = {(float)hs.x * cb, (float)hs.y * cb, (float)hs.z * cb, (float)hs.w * cb};
    const unsigned sg4 = (unsigned)sv[b * 16 + (lane >> 1)] >> ((lane & 1) * 4);
    float bv[4] = {0.f, 0.f, 0.f, 0.f};
    if (bias != nullptr) {
      const uint2 br = *reinterpret_cast<const uint2*>(bias + base);
      bv[0] = __uint_as_float((br.x & 0xFFFFu) << 16); bv[1] = __uint_as_float(br.x & 0xFFFF0000u);
      bv[2] = __uint_as_float((br.y & 0xFFFFu) << 16); bv[3] = __uint_as_float(br.y & 0xFFFF0000u);
    }
    const int r = w;
    f4v s4 = *reinterpret_cast<const f4v*>(&blk_s[0][r][lane * 4])
           + *reinterpret_cast<const f4v*>(&blk_s[1][r][lane * 4]);
    const float c = 1534.0f * xs[r], t = tok[r];
    float v[4] = { (s4.x - c) * sc[0] * t, (s4.y - c) * sc[1] * t, (s4.z - c) * sc[2] * t, (s4.w - c) * sc[3] * t };
    fwht128_wave(v, lane);
    float o[4] = {
      v[0] * (1.0f / 11.313708498984761f), v[1] * (1.0f / 11.313708498984761f),
      v[2] * (1.0f / 11.313708498984761f), v[3] * (1.0f / 11.313708498984761f),
    };
    if (sg4 & 1u) o[0] = -o[0]; if (sg4 & 2u) o[1] = -o[1];
    if (sg4 & 4u) o[2] = -o[2]; if (sg4 & 8u) o[3] = -o[3];
    if (bias != nullptr) { o[0] += bv[0]; o[1] += bv[1]; o[2] += bv[2]; o[3] += bv[3]; }
    *reinterpret_cast<uint2*>(y + (size_t)r * m + base) = {f32_to_bf16x2_rne(o[0], o[1]), f32_to_bf16x2_rne(o[2], o[3])};
  }
}

// Wave 21: Zero-partial, zero-fence, zero-atomic 128-row WGs for concurrent B=2 decode (M=9..16)
//   - w8_big  (dim3(256), 0 K-splits): tall shapes (lm_head, DSA q_b, DSA o_gate, L0 gu)
//   - w16_big (dim3(512), 2 K-splits in LDS): medium-mt / narrow-K (HCA o_gate, MoE up_lat, wq_b)
// Uses gcd(5,8)=1 8-channel LPDDR5X bijection stagger (((ti+b)&7)*8) + VMEM-FIFO prefetch!
// Wave 29: Zero-partial, zero-atomic, zero-fence 128-row WG (8 waves = 256 thr) for M=1..8 (B=1 decode!)
//          Uses 8-ch LPDDR5X bijection stagger + VMEM-FIFO prefetch (k5_unsplit_tile_loop<M>), V<=88, vSp=0!
template <int M>
__global__ __launch_bounds__(256) void k_gemv_k5_unsplit_ns1_128_w8(
    const uint32_t* __restrict__ qw, const _Float16* __restrict__ xt,
    const float* __restrict__ xs, const float* __restrict__ tok,
    const __half* __restrict__ scales, float cb,
    const uint8_t* __restrict__ sv, const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ y, int mt, int nt) {
  __shared__ float o_lds[M][128];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, b = blockIdx.x;
  const int i = b * 8 + w, g = lane >> 3, q = lane & 7;
  const int n = nt * 16, m = mt * 16;

  float a0[M], a1[M];
#pragma unroll
  for (int r = 0; r < M; ++r) { a0[r] = 0.f; a1[r] = 0.f; }

  const int phase = (nt >= 64 && (nt & 3) == 0)
      ? (((w + b) & 7) * 8)
      : ((nt >= 16 && (nt & 3) == 0) ? (((w + b) & 3) * 4) : 0);
  k5_unsplit_tile_loop<M>(qw + (size_t)i * nt * 40, xt, 0, nt, n, lane, phase, a0, a1);

#pragma unroll
  for (int r = 0; r < M; ++r) {
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
  }
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
      o_lds[r][w * 16 + 2 * q]     = a0[r];
      o_lds[r][w * 16 + 2 * q + 1] = a1[r];
    }
  }
  __syncthreads();

  if (w < M) {
    const int base = b * 128 + lane * 4;
    const float xsm = xs[w] * 1534.0f, tt = tok[w];
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e)
      v[e] = (o_lds[w][lane * 4 + e] - xsm) * __half2float(scales[base + e]) * cb * tt;
    fwht128_wave(v, lane);
    const unsigned sg4 = (unsigned)sv[b * 16 + (lane >> 1)] >> ((lane & 1) * 4);
    float o[4] = {
      v[0] * (1.0f / 11.313708498984761f), v[1] * (1.0f / 11.313708498984761f),
      v[2] * (1.0f / 11.313708498984761f), v[3] * (1.0f / 11.313708498984761f),
    };
    if (sg4 & 1u) o[0] = -o[0]; if (sg4 & 2u) o[1] = -o[1];
    if (sg4 & 4u) o[2] = -o[2]; if (sg4 & 8u) o[3] = -o[3];
    if (bias != nullptr) {
      const uint2 braw = *reinterpret_cast<const uint2*>(bias + base);
      o[0] += __uint_as_float((braw.x & 0xFFFFu) << 16);
      o[1] += __uint_as_float(braw.x & 0xFFFF0000u);
      o[2] += __uint_as_float((braw.y & 0xFFFFu) << 16);
      o[3] += __uint_as_float(braw.y & 0xFFFF0000u);
    }
    *reinterpret_cast<uint2*>(y + (size_t)w * m + base) = {f32_to_bf16x2_rne(o[0], o[1]), f32_to_bf16x2_rne(o[2], o[3])};
  }
}

template <int M_CAP>
__global__ __launch_bounds__(256) void k_gemv_k5_unsplit_ns1_128_w8_big(
    const uint32_t* __restrict__ qw, const _Float16* __restrict__ xt,
    const float* __restrict__ xs, const float* __restrict__ tok,
    const __half* __restrict__ scales, float cb,
    const uint8_t* __restrict__ sv, const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ y, int mt, int nt, int mr) {
  __shared__ float o_lds[M_CAP][128];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, b = blockIdx.x;
  const int i = b * 8 + w, g = lane >> 3, q = lane & 7;
  const int n = nt * 16, m = mt * 16;

  float a0[M_CAP], a1[M_CAP];
#pragma unroll
  for (int r = 0; r < M_CAP; ++r) { a0[r] = 0.f; a1[r] = 0.f; }

  const int phase = (nt >= 64 && (nt & 3) == 0)
      ? (((w + b) & 7) * 8)
      : ((nt >= 16 && (nt & 3) == 0) ? (((w + b) & 3) * 4) : 0);
  k5_unsplit_tile_loop_big<M_CAP>(qw + (size_t)i * nt * 40, xt, 0, nt, n, lane, phase, mr, a0, a1);

#pragma unroll
  for (int r = 0; r < M_CAP; ++r) {
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
  }
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M_CAP; ++r) {
      if (r < mr) {
        o_lds[r][w * 16 + 2 * q]     = a0[r];
        o_lds[r][w * 16 + 2 * q + 1] = a1[r];
      }
    }
  }
  __syncthreads();

  const int base = b * 128 + lane * 4;
  const h4v hs = *reinterpret_cast<const h4v*>(scales + base);
  const float sc[4] = {(float)hs.x * cb, (float)hs.y * cb, (float)hs.z * cb, (float)hs.w * cb};
  const unsigned sg4 = (unsigned)sv[b * 16 + (lane >> 1)] >> ((lane & 1) * 4);
  float bv[4] = {0.f, 0.f, 0.f, 0.f};
  if (bias != nullptr) {
    const uint2 br = *reinterpret_cast<const uint2*>(bias + base);
    bv[0] = __uint_as_float((br.x & 0xFFFFu) << 16); bv[1] = __uint_as_float(br.x & 0xFFFF0000u);
    bv[2] = __uint_as_float((br.y & 0xFFFFu) << 16); bv[3] = __uint_as_float(br.y & 0xFFFF0000u);
  }
  for (int r = w; r < mr; r += 8) {
    const f4v s4 = *reinterpret_cast<const f4v*>(&o_lds[r][lane * 4]);
    const float c = 1534.0f * xs[r], t = tok[r];
    float v[4] = { (s4.x - c) * sc[0] * t, (s4.y - c) * sc[1] * t, (s4.z - c) * sc[2] * t, (s4.w - c) * sc[3] * t };
    fwht128_wave(v, lane);
    float o[4] = {
      v[0] * (1.0f / 11.313708498984761f), v[1] * (1.0f / 11.313708498984761f),
      v[2] * (1.0f / 11.313708498984761f), v[3] * (1.0f / 11.313708498984761f),
    };
    if (sg4 & 1u) o[0] = -o[0]; if (sg4 & 2u) o[1] = -o[1];
    if (sg4 & 4u) o[2] = -o[2]; if (sg4 & 8u) o[3] = -o[3];
    if (bias != nullptr) { o[0] += bv[0]; o[1] += bv[1]; o[2] += bv[2]; o[3] += bv[3]; }
    *reinterpret_cast<uint2*>(y + (size_t)r * m + base) = {f32_to_bf16x2_rne(o[0], o[1]), f32_to_bf16x2_rne(o[2], o[3])};
  }
}

template <int M_CAP>
__global__ __launch_bounds__(512) void k_gemv_k5_unsplit_ns1_128_w16_big(
    const uint32_t* __restrict__ qw, const _Float16* __restrict__ xt,
    const float* __restrict__ xs, const float* __restrict__ tok,
    const __half* __restrict__ scales, float cb,
    const uint8_t* __restrict__ sv, const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ y, int mt, int nt, int mr) {
  __shared__ float blk_s[2][M_CAP][128];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, b = blockIdx.x;
  const int ti = w & 7, sp = w >> 3, i = b * 8 + ti, g = lane >> 3, q = lane & 7;
  const int n = nt * 16, m = mt * 16;

  float a0[M_CAP], a1[M_CAP];
#pragma unroll
  for (int r = 0; r < M_CAP; ++r) { a0[r] = 0.f; a1[r] = 0.f; }

  const int tps = (nt + 1) >> 1, j0 = sp * tps, j1 = min(nt, j0 + tps), nloc = j1 - j0;
  const int phase = (nloc >= 64 && (nloc & 3) == 0)
      ? (((ti + b) & 7) * 8)
      : ((nloc >= 16 && (nloc & 3) == 0) ? (((ti + b) & 3) * 4) : 0);
  k5_unsplit_tile_loop_big<M_CAP>(qw + (size_t)i * nt * 40, xt, j0, j1, n, lane, phase, mr, a0, a1);

#pragma unroll
  for (int r = 0; r < M_CAP; ++r) {
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
  }
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M_CAP; ++r) {
      if (r < mr) {
        blk_s[sp][r][ti * 16 + 2 * q]     = a0[r];
        blk_s[sp][r][ti * 16 + 2 * q + 1] = a1[r];
      }
    }
  }
  __syncthreads();

  if (w < mr) {
    const int base = b * 128 + lane * 4;
    const h4v hs = *reinterpret_cast<const h4v*>(scales + base);
    const float sc[4] = {(float)hs.x * cb, (float)hs.y * cb, (float)hs.z * cb, (float)hs.w * cb};
    const unsigned sg4 = (unsigned)sv[b * 16 + (lane >> 1)] >> ((lane & 1) * 4);
    float bv[4] = {0.f, 0.f, 0.f, 0.f};
    if (bias != nullptr) {
      const uint2 br = *reinterpret_cast<const uint2*>(bias + base);
      bv[0] = __uint_as_float((br.x & 0xFFFFu) << 16); bv[1] = __uint_as_float(br.x & 0xFFFF0000u);
      bv[2] = __uint_as_float((br.y & 0xFFFFu) << 16); bv[3] = __uint_as_float(br.y & 0xFFFF0000u);
    }
    const f4v s4 = *reinterpret_cast<const f4v*>(&blk_s[0][w][lane * 4])
                 + *reinterpret_cast<const f4v*>(&blk_s[1][w][lane * 4]);
    const float c = 1534.0f * xs[w], t = tok[w];
    float v[4] = { (s4.x - c) * sc[0] * t, (s4.y - c) * sc[1] * t, (s4.z - c) * sc[2] * t, (s4.w - c) * sc[3] * t };
    fwht128_wave(v, lane);
    float o[4] = {
      v[0] * (1.0f / 11.313708498984761f), v[1] * (1.0f / 11.313708498984761f),
      v[2] * (1.0f / 11.313708498984761f), v[3] * (1.0f / 11.313708498984761f),
    };
    if (sg4 & 1u) o[0] = -o[0]; if (sg4 & 2u) o[1] = -o[1];
    if (sg4 & 4u) o[2] = -o[2]; if (sg4 & 8u) o[3] = -o[3];
    if (bias != nullptr) { o[0] += bv[0]; o[1] += bv[1]; o[2] += bv[2]; o[3] += bv[3]; }
    *reinterpret_cast<uint2*>(y + (size_t)w * m + base) = {f32_to_bf16x2_rne(o[0], o[1]), f32_to_bf16x2_rne(o[2], o[3])};
  }
}

static torch::Tensor g_k5u_part_buf;
static float* get_k5u_part_ptr(int64_t need, const torch::Tensor& ref) {
  if (!g_k5u_part_buf.defined() || g_k5u_part_buf.numel() < need || g_k5u_part_buf.device() != ref.device()) {
    // 4M floats = 16 MiB covers all ERNIE-5.5 shapes up to M=16 (including lm_head 201216) without HIP-graph realloc!
    g_k5u_part_buf = torch::empty({std::max<int64_t>(need, 4 * 1024 * 1024)}, ref.options().dtype(torch::kFloat32));
  }
  return g_k5u_part_buf.data_ptr<float>();
}

torch::Tensor gemv_k5_unsplit_epi(
    torch::Tensor qw, torch::Tensor xt, torch::Tensor xs, torch::Tensor tok,
    torch::Tensor scales, double cb, torch::Tensor sv,
    c10::optional<torch::Tensor> bias, torch::Tensor counters,
    int64_t nsplit, int64_t use_e8) {
  TORCH_CHECK(qw.is_cuda() && qw.is_contiguous() && qw.dim() == 3 && qw.size(2) == 40);
  TORCH_CHECK(xt.is_cuda() && xt.is_contiguous() && xt.dim() == 2 && xt.scalar_type() == torch::kHalf);
  const int mt = (int)qw.size(0), nt = (int)qw.size(1);
  const int M = (int)xt.size(0);
  const int m = mt * 16;
  TORCH_CHECK(M >= 1 && M <= 16 && (mt % 8) == 0 && xt.size(1) == nt * 16);

  auto y = torch::empty({M, m}, qw.options().dtype(torch::kBFloat16));
  auto st = c10::hip::getCurrentHIPStream().stream();

  const uint32_t* qwp = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>());
  const _Float16* xtp = reinterpret_cast<const _Float16*>(xt.data_ptr<at::Half>());
  const float* xsp = xs.data_ptr<float>();
  const float* tokp = tok.data_ptr<float>();
  const __half* sclp = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  const uint8_t* svp = sv.data_ptr<uint8_t>();
  const uint16_t* bp = bias.has_value() ? reinterpret_cast<const uint16_t*>(bias->data_ptr()) : nullptr;
  uint16_t* yp = reinterpret_cast<uint16_t*>(y.data_ptr());

  // Wave 29: Zero-partial, zero-fence, zero-atomic 128-row WG for B=1 decode (M=1..8):
  //   - w16: mt in 128..256 (nt in 32..256 or nt>=768), DSA o_gate (512x96), lm_head (>=8192x256)
  //   - w8 : tall/short-K shapes HCA wq_b (2048x64), DSA q_b (1024x96)
  if (M <= 8 && mt >= 128 && nt >= 32) {
    const bool use_w16_s =
        (mt <= 256 && (nt <= 256 || nt >= 768)) ||
        (mt <= 512 && nt <= 128) ||
        (mt >= 8192 && nt == 256);
    const bool use_w8_s = (!use_w16_s) && (mt >= 1024 && mt <= 4096 && nt >= 64 && nt <= 128);
    if (use_w16_s) {
      dim3 g128(mt >> 3);
#define LAUNCH_K5U_NS1(MM) \
      hipLaunchKernelGGL((k_gemv_k5_unsplit_ns1_128_w16<MM>), g128, dim3(512), 0, st, \
                         qwp, xtp, xsp, tokp, sclp, (float)cb, svp, bp, yp, mt, nt)
      switch (M) {
        case 1: LAUNCH_K5U_NS1(1); break; case 2: LAUNCH_K5U_NS1(2); break;
        case 3: LAUNCH_K5U_NS1(3); break; case 4: LAUNCH_K5U_NS1(4); break;
        case 5: LAUNCH_K5U_NS1(5); break; case 6: LAUNCH_K5U_NS1(6); break;
        case 7: LAUNCH_K5U_NS1(7); break; case 8: LAUNCH_K5U_NS1(8); break;
      }
#undef LAUNCH_K5U_NS1
      return y;
    } else if (use_w8_s) {
      dim3 g128(mt >> 3);
#define LAUNCH_K5U_NS1_W8(MM) \
      hipLaunchKernelGGL((k_gemv_k5_unsplit_ns1_128_w8<MM>), g128, dim3(256), 0, st, \
                         qwp, xtp, xsp, tokp, sclp, (float)cb, svp, bp, yp, mt, nt)
      switch (M) {
        case 1: LAUNCH_K5U_NS1_W8(1); break; case 2: LAUNCH_K5U_NS1_W8(2); break;
        case 3: LAUNCH_K5U_NS1_W8(3); break; case 4: LAUNCH_K5U_NS1_W8(4); break;
        case 5: LAUNCH_K5U_NS1_W8(5); break; case 6: LAUNCH_K5U_NS1_W8(6); break;
        case 7: LAUNCH_K5U_NS1_W8(7); break; case 8: LAUNCH_K5U_NS1_W8(8); break;
      }
#undef LAUNCH_K5U_NS1_W8
      return y;
    }
  }
  // Wave 21+29: Zero-partial, zero-fence, zero-atomic 128-row WG for concurrent B=2 decode (M=9..16):
  //   - w16_big: ALL mt in 128..256 (MoE_fc1_lat, HCA_kv_a, SH_gate_up, HCA_DSA_wo_b, HCA_o_gate, MoE_fc2, L0_down) + wq_b (2048x64)
  //   - w8_big : DSA o_gate (512x96), DSA q_b (1024x96), L0 gu (1792x256), lm_head (12576x256)
  if (M >= 9 && M <= 16 && mt >= 128 && nt >= 32) {
    const bool use_w16 = (mt <= 256) || (mt >= 2048 && nt == 64);
    const bool use_w8  = (!use_w16) && (mt >= 512 && nt >= 64);
    if (use_w16) {
      dim3 g128(mt >> 3);
      if (M <= 12) {
        hipLaunchKernelGGL((k_gemv_k5_unsplit_ns1_128_w16_big<12>), g128, dim3(512), 0, st,
                           qwp, xtp, xsp, tokp, sclp, (float)cb, svp, bp, yp, mt, nt, M);
      } else {
        hipLaunchKernelGGL((k_gemv_k5_unsplit_ns1_128_w16_big<16>), g128, dim3(512), 0, st,
                           qwp, xtp, xsp, tokp, sclp, (float)cb, svp, bp, yp, mt, nt, M);
      }
      return y;
    } else if (use_w8) {
      dim3 g128(mt >> 3);
      if (M <= 12) {
        hipLaunchKernelGGL((k_gemv_k5_unsplit_ns1_128_w8_big<12>), g128, dim3(256), 0, st,
                           qwp, xtp, xsp, tokp, sclp, (float)cb, svp, bp, yp, mt, nt, M);
      } else {
        hipLaunchKernelGGL((k_gemv_k5_unsplit_ns1_128_w8_big<16>), g128, dim3(256), 0, st,
                           qwp, xtp, xsp, tokp, sclp, (float)cb, svp, bp, yp, mt, nt, M);
      }
      return y;
    }
  }

  const int ns = std::max(1, std::min((int)nsplit, nt));
  const int tps = (nt + ns - 1) / ns;
  const int ns_eff = (nt + tps - 1) / tps;
  const bool e8 = (use_e8 != 0) && ((ns_eff % 8) == 0);
  const int ns_part = e8 ? (ns_eff >> 3) : ns_eff;

  float* pp = get_k5u_part_ptr((int64_t)ns_part * M * m, qw);
  unsigned* cntp = reinterpret_cast<unsigned*>(counters.data_ptr<int32_t>());

  dim3 grid = e8 ? dim3(mt * (ns_eff >> 3)) : dim3((mt * ns_eff + 7) / 8);
#define LAUNCH_K5U(MM)                                                                                         \
  if (e8) hipLaunchKernelGGL((k_gemv_k5_unsplit_epi8<MM>), grid, dim3(256), 0, st,                             \
                             qwp, xtp, pp, cntp, xsp, tokp, sclp, (float)cb, svp, bp, yp, mt, nt, ns_eff, tps); \
  else    hipLaunchKernelGGL((k_gemv_k5_unsplit_epi<MM>), grid, dim3(256), 0, st,                              \
                             qwp, xtp, pp, cntp, xsp, tokp, sclp, (float)cb, svp, bp, yp, mt, nt, ns_eff, tps);
  switch (M) {
    case 1: LAUNCH_K5U(1); break;
    case 2: LAUNCH_K5U(2); break;
    case 3: LAUNCH_K5U(3); break;
    case 4: LAUNCH_K5U(4); break;
    case 5: LAUNCH_K5U(5); break;
    case 6: LAUNCH_K5U(6); break;
    case 7: LAUNCH_K5U(7); break;
    case 8: LAUNCH_K5U(8); break;
    case 9: case 10: case 11: case 12:
      // Wave 20: VMEM-FIFO-fixed big<12> beats k_gemv_k5_wmma_dec_epi8 across ALL 11 ERNIE-5.5 shapes (-1.27 ms/step)!
      if (e8) hipLaunchKernelGGL((k_gemv_k5_unsplit_epi8_big<12>), grid, dim3(256), 0, st,
                                 qwp, xtp, pp, cntp, xsp, tokp, sclp, (float)cb, svp, bp, yp, mt, nt, ns_eff, tps, M);
      else    hipLaunchKernelGGL((k_gemv_k5_unsplit_epi_big<12>), grid, dim3(256), 0, st,
                                 qwp, xtp, pp, cntp, xsp, tokp, sclp, (float)cb, svp, bp, yp, mt, nt, ns_eff, tps, M);
      break;
    default:
      if (e8 && nt >= 32) {
        int ns_w = (mt <= 256 && nt >= 256 && nt <= 512) ? 16 : 8;
        if (ns_w > nt) ns_w = (nt / 8) * 8;
        if (ns_w < 8) ns_w = 8;
        const int tps_w = (nt + ns_w - 1) / ns_w;
        const dim3 grid_w((mt * ns_w + 7) / 8);
        hipLaunchKernelGGL(k_gemv_k5_wmma_dec_epi8, grid_w, dim3(256), 0, st,
                           qwp, xtp, pp, cntp, xsp, tokp, sclp, (float)cb, svp, bp, yp, mt, nt, ns_w, tps_w, M);
      } else {
        if (e8) hipLaunchKernelGGL((k_gemv_k5_unsplit_epi8_big<16>), grid, dim3(256), 0, st,
                                   qwp, xtp, pp, cntp, xsp, tokp, sclp, (float)cb, svp, bp, yp, mt, nt, ns_eff, tps, M);
        else    hipLaunchKernelGGL((k_gemv_k5_unsplit_epi_big<16>), grid, dim3(256), 0, st,
                                   qwp, xtp, pp, cntp, xsp, tokp, sclp, (float)cb, svp, bp, yp, mt, nt, ns_eff, tps, M);
      }
      break;
  }
#undef LAUNCH_K5U
  return y;
}


// ============================================================================
// Wave 31: Direct M256-blocked HTR prologue [ceil(T/256), D/16, 256, 16] fp16
// Eliminates power-of-2 row-stride L1/GL2 set thrashing in 128x256_4x4 WMMA
// for D >= 4096, and skips unused xs reduction!
// ============================================================================
template <int NB>
__global__ __launch_bounds__(256) void k_prologue_b16_m256_1r(
    const uint16_t* __restrict__ x,       // [T, D] bf16
    const uint8_t* __restrict__ su,       // [D/8] u8
    _Float16* __restrict__ xt,            // [M_blk, nt, 256, 16] fp16
    float* __restrict__ tok,              // [T] fp32
    int64_t x_s0, int D) {
  __shared__ float red[8];
  __shared__ float s_scale;
  const int row = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int nblk = D >> 7;
  const int nt = D >> 4;
  const uint16_t* xr = x + (int64_t)row * x_s0;

  float v[NB][4];
  float amax = 0.f;
#pragma unroll
  for (int t = 0; t < NB; ++t) {
    const int blk = w + 8 * t;
    if (blk < nblk) {
      const int base = blk * 128 + lane * 4;
      const uint2 rx = *reinterpret_cast<const uint2*>(xr + base);
      const float xv[4] = {
        __uint_as_float((rx.x & 0xFFFFu) << 16), __uint_as_float(rx.x & 0xFFFF0000u),
        __uint_as_float((rx.y & 0xFFFFu) << 16), __uint_as_float(rx.y & 0xFFFF0000u),
      };
      const unsigned sg4 = (unsigned)su[base >> 3] >> (base & 7);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        amax = fmaxf(amax, fabsf(xv[e]));
        v[t][e] = ((sg4 >> e) & 1u) ? -xv[e] : xv[e];
      }
    }
  }
  amax = wave_max_f(amax);
  if (lane == 0) red[w] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = red[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) m = fmaxf(m, red[k]);
    const float bound = m * 11.4f;
    s_scale = bound > 30000.f ? exp2f(ceilf(log2f(bound * (1.0f / 30000.f)))) : 1.f;
    tok[row] = s_scale;
  }
  __syncthreads();
  const float inv = (1.0f / 11.313708498984761f) * __builtin_amdgcn_rcpf(s_scale);
  const int mb = row >> 8;
  const int r_loc = row & 255;
  _Float16* xt_row = xt + (size_t)mb * ((size_t)nt * 4096) + (size_t)r_loc * 16;
  const int j_in_blk = lane >> 2;
  const int e_in_j = (lane & 3) * 4;
#pragma unroll
  for (int t = 0; t < NB; ++t) {
    const int blk = w + 8 * t;
    if (blk < nblk) {
      fwht128_wave(v[t], lane);
      const int j = blk * 8 + j_in_blk;
      const h4v h4 = {(_Float16)(v[t][0] * inv), (_Float16)(v[t][1] * inv),
                      (_Float16)(v[t][2] * inv), (_Float16)(v[t][3] * inv)};
      *reinterpret_cast<h4v*>(xt_row + (size_t)j * 4096 + e_in_j) = h4;
    }
  }
}

__global__ __launch_bounds__(256) void k_prologue_b16_m256_4r2p(
    const uint16_t* __restrict__ x,       // [T, D] bf16
    const uint8_t* __restrict__ su,       // [D/8] u8
    _Float16* __restrict__ xt,            // [M_blk, nt, 256, 16] fp16
    float* __restrict__ tok,              // [T] fp32
    int64_t x_s0, int T, int D) {
  __shared__ __attribute__((aligned(16))) _Float16 s_blk[2][4][128]; // 2 KiB
  __shared__ float s_amax[4][2];
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int r_in = w >> 1;
  const int w_half = w & 1;
  const int row0 = blockIdx.x * 4;
  const int row = row0 + r_in;
  const int nblk = D >> 7;
  const int nt = D >> 4;
  const bool valid = (row < T);
  const uint16_t* xr = valid ? (x + (int64_t)row * x_s0) : x;

  float amax = 0.f;
  if (valid) {
    const int n256 = D >> 8;
    for (int b2 = w_half; b2 < n256; b2 += 2) {
      const u4v rx = *reinterpret_cast<const u4va*>(xr + b2 * 256 + lane * 8);
      auto f_abs_pair = [](uint32_t p) -> float {
        return fmaxf(fabsf(__uint_as_float((p & 0xFFFFu) << 16)), fabsf(__uint_as_float(p & 0xFFFF0000u)));
      };
      amax = fmaxf(amax, fmaxf(fmaxf(f_abs_pair(rx.x), f_abs_pair(rx.y)),
                               fmaxf(f_abs_pair(rx.z), f_abs_pair(rx.w))));
    }
    if ((D & 255) != 0 && w_half == 0) {
      const uint2 rx = *reinterpret_cast<const uint2*>(xr + n256 * 256 + lane * 4);
      amax = fmaxf(amax, fmaxf(fabsf(__uint_as_float((rx.x & 0xFFFFu) << 16)), fabsf(__uint_as_float(rx.x & 0xFFFF0000u))));
      amax = fmaxf(amax, fmaxf(fabsf(__uint_as_float((rx.y & 0xFFFFu) << 16)), fabsf(__uint_as_float(rx.y & 0xFFFF0000u))));
    }
  }
  amax = wave_max_f(amax);
  if (lane == 0) s_amax[r_in][w_half] = amax;
  __syncthreads();
  float sc = 1.f;
  if (valid) {
    const float m = fmaxf(s_amax[r_in][0], s_amax[r_in][1]);
    const float bound = m * 11.4f;
    sc = bound > 30000.f ? exp2f(ceilf(log2f(bound * (1.0f / 30000.f)))) : 1.f;
    if (w_half == 0 && lane == 0) tok[row] = sc;
  }
  const float inv = (1.0f / 11.313708498984761f) * __builtin_amdgcn_rcpf(sc);

  const int mb = row0 >> 8;
  const int r_loc0 = row0 & 255;
  _Float16* xt_base = xt + (size_t)mb * ((size_t)nt * 4096) + (size_t)r_loc0 * 16;
  const int out_r = lane >> 3;
  const int out_e = (lane & 7) * 2;

  for (int blk0 = 0; blk0 < nblk; blk0 += 2) {
    const int blk = blk0 + w_half;
    if (valid && blk < nblk) {
      const int base = blk * 128 + lane * 4;
      const uint2 rx = *reinterpret_cast<const uint2*>(xr + base);
      float v[4] = {
        __uint_as_float((rx.x & 0xFFFFu) << 16), __uint_as_float(rx.x & 0xFFFF0000u),
        __uint_as_float((rx.y & 0xFFFFu) << 16), __uint_as_float(rx.y & 0xFFFF0000u),
      };
      const unsigned sg4 = (unsigned)su[base >> 3] >> (base & 7);
#pragma unroll
      for (int e = 0; e < 4; ++e) if ((sg4 >> e) & 1u) v[e] = -v[e];
      fwht128_wave(v, lane);
      const h4v h4 = {(_Float16)(v[0] * inv), (_Float16)(v[1] * inv),
                      (_Float16)(v[2] * inv), (_Float16)(v[3] * inv)};
      *reinterpret_cast<h4v*>(&s_blk[w_half][r_in][lane * 4]) = h4;
    }
    __syncthreads();
    {
      const int j0 = blk0 * 8 + w;
      const uint32_t p0 = *reinterpret_cast<const uint32_t*>(&s_blk[0][out_r][w * 16 + out_e]);
      *reinterpret_cast<uint32_t*>(xt_base + (size_t)j0 * 4096 + out_r * 16 + out_e) = p0;
    }
    if (blk0 + 1 < nblk) {
      const int j1 = (blk0 + 1) * 8 + w;
      const uint32_t p1 = *reinterpret_cast<const uint32_t*>(&s_blk[1][out_r][w * 16 + out_e]);
      *reinterpret_cast<uint32_t*>(xt_base + (size_t)j1 * 4096 + out_r * 16 + out_e) = p1;
    }
    __syncthreads();
  }
}

std::vector<torch::Tensor> prologue_b16_m256(torch::Tensor x, torch::Tensor su) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == torch::kBFloat16 && x.stride(1) == 1);
  const int T = (int)x.size(0);
  const int D = (int)x.size(1);
  TORCH_CHECK((D % 128 == 0) && D > 0);
  const int m_blks = (T + 255) / 256;
  const int nt = D >> 4;
  auto xt = torch::empty({m_blks, nt, 256, 16}, x.options().dtype(torch::kHalf));
  auto tok = torch::empty({T}, x.options().dtype(torch::kFloat32));
  if (T == 0) return {xt, tok};
  auto st = c10::hip::getCurrentHIPStream().stream();
  auto xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
  auto sup = su.data_ptr<uint8_t>();
  auto xtp = reinterpret_cast<_Float16*>(xt.data_ptr<at::Half>());
  auto tokp = tok.data_ptr<float>();
  const int nb = ((D >> 7) + 7) / 8;
  if (D <= 4096) {
    if (nb <= 2) hipLaunchKernelGGL((k_prologue_b16_m256_1r<2>), dim3(T), dim3(256), 0, st, xp, sup, xtp, tokp, x.stride(0), D);
    else hipLaunchKernelGGL((k_prologue_b16_m256_1r<4>), dim3(T), dim3(256), 0, st, xp, sup, xtp, tokp, x.stride(0), D);
  } else {
    hipLaunchKernelGGL(k_prologue_b16_m256_4r2p, dim3((T + 3) / 4), dim3(256), 0, st, xp, sup, xtp, tokp, x.stride(0), T, D);
  }
  return {xt, tok};
}

PYBIND11_MODULE(ernie_k5_sp_prefill, m) {
  m.def("repack_k5_packed", &repack_k5_packed);
  m.def("dense_prefill_wmma_sp16", &dense_prefill_wmma_sp16);
  m.def("dense_prefill_wmma_unsplit", &dense_prefill_wmma_unsplit,
        py::arg("qw"), py::arg("xt"), py::arg("tok"),
        py::arg("scales"), py::arg("cb"), py::arg("sv"),
        py::arg("bias"), py::arg("bm"),
        py::arg("accum_out") = c10::nullopt);
  m.attr("SUPPORTS_UNSPLIT_ACCUM_OUT") = true;
  m.def("gemv_k5_sp16_ns1", &gemv_k5_sp16_ns1);
  m.def("situ_and_mul_htr_prologue", &situ_and_mul_htr_prologue);
  m.def("rmsnorm_htr_prologue", &rmsnorm_htr_prologue);
  m.def("sigmoid_mul_htr_prologue", &sigmoid_mul_htr_prologue);
  m.def("gemv_k5_unsplit_epi", &gemv_k5_unsplit_epi);
  m.def("prologue_b16_m256", &prologue_b16_m256);
}

