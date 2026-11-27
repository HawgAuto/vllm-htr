// Native gfx1151 (RDNA3.5) Wave32 WMMA HIP kernels for ERNIE-5.5 DSA Attention (6 layers):
//   1. ernie_dsa_sparse_mla:
//      Exact D=576 (NOPE=512, ROPE=64), H=64, BF16 Q & KV, FP32 attn_sink, direct global_idx [T, W].
//      - Uses hardware Wave32 BF16 WMMA (__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32)
//      - Zero 576->1024 power-of-2 padding waste, zero ragged index compaction pass
//      - Split-KV (S=8) for decode (T<=8) to saturate all 40 CUs; S=1 single-pass for prefill (T>8)
//      - Outputs [T, 64, 512] BF16 directly (no wasted 64 rope output dims)
//   2. ernie_fp8_mqa_logits_prefill:
//      Replaces fp8_mqa_logits_torch fallback on gfx1151 (saves 1+ GiB [H,M,N] temp & ~100x faster).
//      - Exact E4M3FN / E4M3FNUZ -> FP16 LUT decode + Wave32 FP16 WMMA over 64 heads x 128 dim
//      - Early-skips out-of-range causal [cu_ks, cu_ke) 16-token blocks
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <math.h>

typedef short bf16x16 __attribute__((ext_vector_type(16)));
typedef _Float16 f16x16 __attribute__((ext_vector_type(16)));
typedef float f4v __attribute__((ext_vector_type(4)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));
typedef u4v u4va __attribute__((aligned(4)));
typedef uint32_t u8v __attribute__((ext_vector_type(8)));

__device__ __forceinline__ uint32_t f32_to_bf16x2_rne(float a, float b) {
  uint32_t ua = __float_as_uint(a);
  uint32_t ub = __float_as_uint(b);
  ua = ((ua & 0x7fffffffu) > 0x7f800000u) ? (ua | 0x00400000u) : (ua + 0x7fffu + ((ua >> 16) & 1u));
  ub = ((ub & 0x7fffffffu) > 0x7f800000u) ? (ub | 0x00400000u) : (ub + 0x7fffu + ((ub >> 16) & 1u));
  return __builtin_amdgcn_perm(ub, ua, 0x07060302u);
}

#define DSA_NOPE 512
#define DSA_ROPE 64
#define DSA_DIM 576
#define DSA_STRIDE 584   // +8 bf16 pad (16B aligned, avoids LDS bank conflicts)
#define LOG2E 1.4426950408889634f
#define NEG_INF (-3.4028234663852886e38f)

template <int MSK> __device__ __forceinline__ float shx_f(float v) {
  const int iv = __builtin_bit_cast(int, v);
  if constexpr (MSK < 16) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, iv, 0x160 | MSK, 0xF, 0xF, false));
  } else {
    return __builtin_bit_cast(float, (int)__builtin_amdgcn_permlanex16((unsigned)iv, (unsigned)iv, 0x76543210u, 0xfedcba98u, false, false));
  }
}
__device__ __forceinline__ float halfwave_max16(float v) {
  v = fmaxf(v, shx_f<8>(v));
  v = fmaxf(v, shx_f<4>(v));
  v = fmaxf(v, shx_f<2>(v));
  v = fmaxf(v, shx_f<1>(v));
  return v;
}
__device__ __forceinline__ float halfwave_sum16(float v) {
  v += shx_f<8>(v);
  v += shx_f<4>(v);
  v += shx_f<2>(v);
  v += shx_f<1>(v);
  return v;
}
__device__ __forceinline__ uint16_t f32_to_bf16_rne(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

// ================================================================== 1: DSA Sparse MLA (D=576, H=64)
// Grid: (T, 4 /*head groups of 16*/, NUM_SPLITS), Block: 128 (4 Wave32s)
// Each wave w in 0..3 owns:
//   - Q@K^T: K-slice [144*w .. 144*w + 143] (9 WMMA steps, 4*144 = 576 exact)
//   - P@V  : V-cols  [128*w .. 128*w + 127] (8 WMMA accumulators = 64 VGPRs, zero spills)
template <int NUM_SPLITS, int NT_KV>
__global__ __launch_bounds__(128) void k_ernie_dsa_sparse_mla(
    const uint16_t* __restrict__ q_ptr,      // [T, 64, 576] bf16
    const uint16_t* __restrict__ kv_ptr,     // [total_rows, 576] bf16 (stride kv_s0)
    const int32_t* __restrict__ idx_ptr,     // [T, W] int32 (-1 = invalid)
    const float* __restrict__ sink_ptr,      // [64] fp32 (or nullptr)
    uint16_t* __restrict__ out_ptr,          // [T, 64, 512] bf16 (used when NUM_SPLITS == 1)
    float* __restrict__ part_o,              // [T, 64, NUM_SPLITS, 512] fp32 (NUM_SPLITS > 1)
    float* __restrict__ part_m,              // [T, 64, NUM_SPLITS] fp32
    float* __restrict__ part_l,              // [T, 64, NUM_SPLITS] fp32
    int64_t q_s0, int64_t q_s1, int64_t kv_s0, int64_t idx_s0, int64_t out_s0, int64_t out_s1,
    int total_kv_rows, int W, float scale) {
  __shared__ __attribute__((aligned(16))) uint16_t KV_lds[16][DSA_STRIDE]; // 18.25 KiB
  __shared__ float S_part[4][16][16];                                       // 4.00 KiB
  __shared__ uint16_t P_lds[16][16];                                        // 0.50 KiB
  __shared__ int32_t s_slots[16];
  __shared__ uint8_t s_valid[16];
  __shared__ float s_alpha[16];
  __shared__ float s_m[16];
  __shared__ float s_l[16];

  const int t = blockIdx.x;
  const int hg = blockIdx.y;        // 0..3 -> heads [16*hg .. 16*hg+15]
  const int sp = blockIdx.z;        // 0..NUM_SPLITS-1
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;
  const int h0 = hg * 16;

  if (tid < 16) {
    s_m[tid] = NEG_INF;
    s_l[tid] = 0.0f;
  }
  __syncthreads();

  f32x8 o_acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) o_acc[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  const int32_t* idx_row = idx_ptr + (int64_t)t * idx_s0;
  const uint16_t* q_base = q_ptr + (int64_t)t * q_s0 + (int64_t)(h0 + l16) * q_s1;

  // Partition W into 16-token blocks across NUM_SPLITS
  const int total_blks = (W + 15) / 16;
  const int blks_per_sp = (total_blks + NUM_SPLITS - 1) / NUM_SPLITS;
  const int blk_start = sp * blks_per_sp;
  const int blk_end = min(total_blks, blk_start + blks_per_sp);

  for (int blk = blk_start; blk < blk_end; ++blk) {
    const int k_base = blk * 16;
    if (tid < 16) {
      const int c = k_base + tid;
      int32_t sl = (c < W) ? idx_row[c] : -1;
      if (sl < 0 || sl >= total_kv_rows) sl = -1;
      s_slots[tid] = sl;
      s_valid[tid] = (sl >= 0) ? 1 : 0;
    }
    __syncthreads();

    // Fast skip if all 16 slots are invalid (common at short contexts!)
    unsigned any_v = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) any_v |= s_valid[i];
    if (!any_v) continue;

    // 1. Cooperatively load 16 x 576 bf16 KV rows into LDS (72 u4v loads per row, 128 threads -> 3x 3-load bursts!)
    // #pragma unroll 1 keeps only 1 burst (12 VGPRs) live at a time -> stays <=112 VGPRs (13w/SIMD) with 3x MLP!
#pragma unroll 1
    for (int g3 = 0; g3 < 3; ++g3) {
      u4v vals[3];
      int krs[3], kc8s[3];
#pragma unroll
      for (int s = 0; s < 3; ++s) {
        const int vidx = (g3 * 3 + s) * 128 + tid; // 0..1151
        const int kr = vidx / 72;                  // 0..15
        const int kc8 = vidx - kr * 72;            // 0..71
        krs[s] = kr;
        kc8s[s] = kc8;
        const int32_t sl = s_slots[kr];
        vals[s] = (u4v){0u, 0u, 0u, 0u};
        if (sl >= 0) {
          const u4va* src = reinterpret_cast<const u4va*>(kv_ptr + (int64_t)sl * kv_s0 + kc8 * 8);
          if constexpr (NT_KV) vals[s] = __builtin_nontemporal_load(src);
          else vals[s] = *src;
        }
      }
#pragma unroll
      for (int s = 0; s < 3; ++s) {
        *reinterpret_cast<u4va*>(&KV_lds[krs[s]][kc8s[s] * 8]) = vals[s];
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();

    // 2. Q @ K^T: each wave accumulates 9 WMMA steps of 16 dims (wave*144 .. wave*144 + 143)
    // Unroll-1 keeps q0/q1/k0/k1 to 16 VGPRs instead of 9x=144 VGPRs -> unlocks higher CU wave occupancy!
    f32x8 s_acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const int d_start = wave * 144;
#pragma unroll 1
    for (int step = 0; step < 9; ++step) {
      const int d = d_start + step * 16;
      const u4v q0 = *reinterpret_cast<const u4va*>(q_base + d);
      const u4v q1 = *reinterpret_cast<const u4va*>(q_base + d + 8);
      const bf16x16 qf = __builtin_bit_cast(bf16x16, (u8v){q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w});
      const u4v k0 = *reinterpret_cast<const u4va*>(&KV_lds[l16][d]);
      const u4v k1 = *reinterpret_cast<const u4va*>(&KV_lds[l16][d + 8]);
      const bf16x16 kf = __builtin_bit_cast(bf16x16, (u8v){k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w});
      s_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(qf, kf, s_acc);
    }
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      S_part[wave][2 * ei + hrow][l16] = s_acc[ei];
    }
    __syncthreads();

    // 3. Wave 0 reduces S_part across 4 waves + runs online softmax in registers via DPP!
    if (wave == 0) {
      const bool vld = (s_valid[l16] != 0);
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) {
        const int hm = 2 * ei + hrow;
        float sc = S_part[0][hm][l16] + S_part[1][hm][l16] + S_part[2][hm][l16] + S_part[3][hm][l16];
        sc = vld ? (sc * scale) : NEG_INF;
        const float bmax = halfwave_max16(sc);
        const float mold = s_m[hm];
        const float mnew = fmaxf(mold, bmax);
        const float alpha = (mold > -1e30f) ? exp2f((mold - mnew) * LOG2E) : 0.0f;
        const float pv = (vld && mnew > -1e30f) ? exp2f((sc - mnew) * LOG2E) : 0.0f;
        const float bsum = halfwave_sum16(pv);
        P_lds[hm][l16] = f32_to_bf16_rne(pv);
        if (l16 == 0) {
          s_alpha[hm] = alpha;
          s_m[hm] = mnew;
          s_l[hm] = s_l[hm] * alpha + bsum;
        }
      }
    }
    __syncthreads();

    // 4. Rescale o_acc & accumulate P @ V (wave w owns V cols 128*w .. 128*w + 127)
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      const float a_ei = s_alpha[2 * ei + hrow];
#pragma unroll
      for (int j = 0; j < 8; ++j) o_acc[j][ei] *= a_ei;
    }

    const u4v p0 = *reinterpret_cast<const u4va*>(&P_lds[l16][0]);
    const u4v p1 = *reinterpret_cast<const u4va*>(&P_lds[l16][8]);
    const bf16x16 pf = __builtin_bit_cast(bf16x16, (u8v){p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w});
    const int v_col0 = wave * 128;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int vc = v_col0 + j * 16 + l16;
      bf16x16 vf;
#pragma unroll
      for (int kr = 0; kr < 16; ++kr) vf[kr] = (short)KV_lds[kr][vc];
      o_acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf, vf, o_acc[j]);
      __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();
  }

  // 5. Epilogue
  const int v_col0 = wave * 128;
  if constexpr (NUM_SPLITS == 1) {
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      const int hm = 2 * ei + hrow;
      float m = s_m[hm];
      float l = s_l[hm];
      float inv_l = 0.0f;
      if (sink_ptr != nullptr) {
        const float sk = sink_ptr[h0 + hm];
        const float mf = fmaxf(m, sk);
        const float lf = (l > 0.0f ? l * exp2f((m - mf) * LOG2E) : 0.0f) + exp2f((sk - mf) * LOG2E);
        const float w_acc = (l > 0.0f) ? exp2f((m - mf) * LOG2E) : 0.0f;
        inv_l = (lf > 0.0f) ? (w_acc * __builtin_amdgcn_rcpf(lf)) : 0.0f;
      } else {
        inv_l = (l > 0.0f) ? __builtin_amdgcn_rcpf(l) : 0.0f;
      }
      uint16_t* dst = out_ptr + (int64_t)t * out_s0 + (int64_t)(h0 + hm) * out_s1 + v_col0 + l16;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        dst[j * 16] = f32_to_bf16_rne(o_acc[j][ei] * inv_l);
      }
    }
  } else {
    if (wave == 0 && l16 == 0) {
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) {
        const int hm = 2 * ei + hrow;
        const int64_t off = ((int64_t)t * 64 + (h0 + hm)) * NUM_SPLITS + sp;
        part_m[off] = s_m[hm];
        part_l[off] = s_l[hm];
      }
    }
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      const int hm = 2 * ei + hrow;
      float* dst = part_o + (((int64_t)t * 64 + (h0 + hm)) * NUM_SPLITS + sp) * DSA_NOPE + v_col0 + l16;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        dst[j * 16] = o_acc[j][ei];
      }
    }
  }
}

// Split-KV reduce for NUM_SPLITS > 1: Grid = (T, 64), Block = 128 (4 dims/thread = 128-bit f4v loads + v_perm_b32 uint2 store!)
template <int NUM_SPLITS>
__global__ __launch_bounds__(128) void k_ernie_dsa_split_reduce(
    const float* __restrict__ part_o,
    const float* __restrict__ part_m,
    const float* __restrict__ part_l,
    const float* __restrict__ sink_ptr,
    uint16_t* __restrict__ out_ptr,
    int64_t out_s0, int64_t out_s1) {
  const int t = blockIdx.x;
  const int h = blockIdx.y;
  const int d0 = threadIdx.x * 4;
  const int64_t base_ml = ((int64_t)t * 64 + h) * NUM_SPLITS;

  float ms[NUM_SPLITS], ls[NUM_SPLITS];
  float m_max = NEG_INF;
#pragma unroll
  for (int s = 0; s < NUM_SPLITS; ++s) {
    ms[s] = part_m[base_ml + s];
    ls[s] = part_l[base_ml + s];
    m_max = fmaxf(m_max, ms[s]);
  }
  if (sink_ptr != nullptr) m_max = fmaxf(m_max, sink_ptr[h]);

  float l_tot = 0.0f;
  float ws[NUM_SPLITS];
#pragma unroll
  for (int s = 0; s < NUM_SPLITS; ++s) {
    ws[s] = (ls[s] > 0.0f) ? exp2f((ms[s] - m_max) * LOG2E) : 0.0f;
    l_tot += ls[s] * ws[s];
  }
  if (sink_ptr != nullptr) l_tot += exp2f((sink_ptr[h] - m_max) * LOG2E);
  const float inv_l = (l_tot > 0.0f) ? __builtin_amdgcn_rcpf(l_tot) : 0.0f;

  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
  for (int s = 0; s < NUM_SPLITS; ++s) {
    if (ws[s] > 0.0f) {
      const f4v po = *reinterpret_cast<const f4v*>(part_o + (base_ml + s) * DSA_NOPE + d0);
      a0 = __builtin_fmaf(ws[s], po.x, a0);
      a1 = __builtin_fmaf(ws[s], po.y, a1);
      a2 = __builtin_fmaf(ws[s], po.z, a2);
      a3 = __builtin_fmaf(ws[s], po.w, a3);
    }
  }
  uint2 out_p;
  out_p.x = f32_to_bf16x2_rne(a0 * inv_l, a1 * inv_l);
  out_p.y = f32_to_bf16x2_rne(a2 * inv_l, a3 * inv_l);
  *reinterpret_cast<uint2*>(out_ptr + (int64_t)t * out_s0 + (int64_t)h * out_s1 + d0) = out_p;
}

// v0.2.8 Roofline-Max All-64-Heads 8-Wave DSA Prefill Attention (BF16 + FP8 + NVFP4!)
// Grid: (T, 1, 1), Block: 256 (8 Wave32s = 4 head-groups x 2 D-halves)
// - 4x fewer global KV loads & FP8/NVFP4 dequants (all 64 heads share ONE KV_lds + VT_lds!)
// - Zero ds_load_u16 in P@V (128-bit VT_lds loads + 1-ahead prefetch + zero sched_barrier!)
// - 4x parallel online softmax across all 4 head-groups simultaneously!

// Fast NaN-free BF16x2 RNE pack for finite quantized dequant values (5 ops vs 14!)
__device__ __forceinline__ uint32_t f32_to_bf16x2_fast(float a, float b) {
  uint32_t ua = __float_as_uint(a);
  uint32_t ub = __float_as_uint(b);
  ua += 0x7fffu + ((ua >> 16) & 1u);
  ub += 0x7fffu + ((ub >> 16) & 1u);
  return __builtin_amdgcn_perm(ub, ua, 0x07060302u);
}

// Register-only branchless FP4 E2M1 -> FP32 (0 LDS loads, 0 bank conflicts!)
// E2M1 nib = (s<<3)|(e<<1)|m -> {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} * (-1)^s
__device__ __forceinline__ float e2m1_reg_to_f32(uint32_t nib) {
  const uint32_t s = (nib & 8u) << 28;
  const uint32_t mag = nib & 7u;
  // Exact FP32 bits for mag 0..7:
  // 0->0x00000000, 1->0x3f000000, 2->0x3f800000, 3->0x3fc00000,
  // 4->0x40000000, 5->0x40400000, 6->0x40800000, 7->0x40c00000
  // For mag>=2: bits = (0x3f000000 + ((mag >> 1) << 23)) | ((mag & 1) << 22)
  // Wait: mag=2->0x3f800000, mag=3->0x3fc00000, mag=4->0x40000000, mag=5->0x40400000, mag=6->0x40800000, mag=7->0x40c00000!
  // Check mag=1: 0x3f000000! Formula: exp_off = (mag >> 1) -> 0x3f000000 + (mag>>1)*0x00800000 + (mag&1)*0x00400000 for mag>=2,
  // and for mag==1: 0x3f000000, mag==0: 0!
  const uint32_t hi = (0x3f000000u + (mag << 22));
  const uint32_t bits = (mag < 2u) ? (mag ? 0x3f000000u : 0u) : hi;
  return __uint_as_float(s | bits);
}

#define DSA_VT_STRIDE 18  // 36B = 9 banks (odd -> low bank conflict, 18 KiB!)

__global__ __launch_bounds__(256) void k_ernie_dsa_sparse_mla_prefill_v28(
    const uint16_t* __restrict__ q_ptr,      // [T, 64, 576] bf16
    const uint16_t* __restrict__ kv_ptr,     // [total_rows, 576] bf16
    const int32_t* __restrict__ idx_ptr,     // [T, W] int32 (-1 = invalid)
    const float* __restrict__ sink_ptr,      // [64] fp32 (or nullptr)
    uint16_t* __restrict__ out_ptr,          // [T, 64, 512] bf16
    int64_t q_s0, int64_t q_s1, int64_t kv_s0, int64_t idx_s0, int64_t out_s0, int64_t out_s1,
    int total_kv_rows, int W, float scale) {
  __shared__ __attribute__((aligned(16))) uint16_t KV_lds[16][DSA_STRIDE];      // 18,688 B
  __shared__ __attribute__((aligned(16))) uint16_t VT_lds[DSA_NOPE][DSA_VT_STRIDE]; // 18,432 B
  __shared__ float S_part[4][16][16];                                            //  4,096 B
  __shared__ __attribute__((aligned(16))) uint16_t P_lds[4][16][16];             //  2,048 B
  __shared__ int32_t s_slots[16];
  __shared__ uint8_t s_valid[16];
  __shared__ float s_alpha[64];
  __shared__ float s_m[64];
  __shared__ float s_l[64];

  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int wh = wave >> 1;   // 0..3 -> heads [wh*16 .. wh*16+15]
  const int wd = wave & 1;    // 0..1 -> K-half (0..287 / 288..575), V-half (0..255 / 256..511)
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;
  const int h0 = wh * 16;

  if (tid < 64) {
    s_m[tid] = NEG_INF;
    s_l[tid] = 0.0f;
  }
  __syncthreads();

  f32x8 o_acc[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) o_acc[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  const int32_t* idx_row = idx_ptr + (int64_t)t * idx_s0;
  const uint16_t* q_base = q_ptr + (int64_t)t * q_s0 + (int64_t)(h0 + l16) * q_s1;
  const int kd_start = wd * 288;
  const int vd_start = wd * 256;
  const int total_blks = (W + 15) >> 4;

#pragma unroll 1
  for (int blk = 0; blk < total_blks; ++blk) {
    const int k_base = blk * 16;
    if (tid < 16) {
      const int c = k_base + tid;
      int32_t sl = (c < W) ? idx_row[c] : -1;
      if (sl < 0 || sl >= total_kv_rows) sl = -1;
      s_slots[tid] = sl;
      s_valid[tid] = (sl >= 0) ? 1 : 0;
    }
    __syncthreads();
    unsigned any_v = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) any_v |= s_valid[i];
    if (!any_v) continue;

    // 1. Load 16 x 576 BF16 KV into KV_lds + VT_lds (1152 uint4s across 256 threads = 4.5 loads/thread, 0 divOps!)
    {
      u4v vals[5];
      int krs[5], kc8s[5];
#pragma unroll
      for (int s = 0; s < 5; ++s) {
        const int vidx = s * 256 + tid;
        vals[s] = (u4v){0u, 0u, 0u, 0u};
        krs[s] = 0; kc8s[s] = 0;
        if (vidx < 1152) {
          // Exact integer div-by-72 for vidx in 0..1151 without v_div: (vidx>>3) in 0..143, /9 = (*7282)>>16
          const int kr = (((vidx >> 3) * 7282) >> 16);
          const int kc8 = vidx - kr * 72;
          krs[s] = kr; kc8s[s] = kc8;
          const int32_t sl = s_slots[kr];
          if (sl >= 0) {
            vals[s] = *reinterpret_cast<const u4va*>(kv_ptr + (int64_t)sl * kv_s0 + kc8 * 8);
          }
        }
      }
#pragma unroll
      for (int s = 0; s < 5; ++s) {
        const int vidx = s * 256 + tid;
        if (vidx < 1152) {
          const int kr = krs[s];
          const int col0 = kc8s[s] * 8;
          *reinterpret_cast<u4va*>(&KV_lds[kr][col0]) = vals[s];
          if (col0 < DSA_NOPE) {
            const uint32_t w0 = vals[s].x, w1 = vals[s].y, w2 = vals[s].z, w3 = vals[s].w;
            VT_lds[col0 + 0][kr] = (uint16_t)(w0 & 0xFFFFu);
            VT_lds[col0 + 1][kr] = (uint16_t)(w0 >> 16);
            VT_lds[col0 + 2][kr] = (uint16_t)(w1 & 0xFFFFu);
            VT_lds[col0 + 3][kr] = (uint16_t)(w1 >> 16);
            VT_lds[col0 + 4][kr] = (uint16_t)(w2 & 0xFFFFu);
            VT_lds[col0 + 5][kr] = (uint16_t)(w2 >> 16);
            VT_lds[col0 + 6][kr] = (uint16_t)(w3 & 0xFFFFu);
            VT_lds[col0 + 7][kr] = (uint16_t)(w3 >> 16);
          }
        }
      }
    }
    __syncthreads();

    // 2. Q @ K^T: 18 WMMA steps per wave (288 dims/wave), 1-ahead prefetch, ZERO sched_barrier!
    f32x8 s_acc = {0, 0, 0, 0, 0, 0, 0, 0};
    u4v q0_c = *reinterpret_cast<const u4va*>(q_base + kd_start);
    u4v q1_c = *reinterpret_cast<const u4va*>(q_base + kd_start + 8);
    u4v k0_c = *reinterpret_cast<const u4va*>(&KV_lds[l16][kd_start]);
    u4v k1_c = *reinterpret_cast<const u4va*>(&KV_lds[l16][kd_start + 8]);
#pragma unroll 1
    for (int step = 0; step < 18; ++step) {
      u4v q0_n = {0u, 0u, 0u, 0u}, q1_n = {0u, 0u, 0u, 0u};
      u4v k0_n = {0u, 0u, 0u, 0u}, k1_n = {0u, 0u, 0u, 0u};
      if (step + 1 < 18) {
        const int dn = kd_start + (step + 1) * 16;
        q0_n = *reinterpret_cast<const u4va*>(q_base + dn);
        q1_n = *reinterpret_cast<const u4va*>(q_base + dn + 8);
        k0_n = *reinterpret_cast<const u4va*>(&KV_lds[l16][dn]);
        k1_n = *reinterpret_cast<const u4va*>(&KV_lds[l16][dn + 8]);
      }
      const bf16x16 qf = __builtin_bit_cast(bf16x16, (u8v){q0_c.x, q0_c.y, q0_c.z, q0_c.w, q1_c.x, q1_c.y, q1_c.z, q1_c.w});
      const bf16x16 kf = __builtin_bit_cast(bf16x16, (u8v){k0_c.x, k0_c.y, k0_c.z, k0_c.w, k1_c.x, k1_c.y, k1_c.z, k1_c.w});
      s_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(qf, kf, s_acc);
      q0_c = q0_n; q1_c = q1_n; k0_c = k0_n; k1_c = k1_n;
    }

    if (wd == 1) {
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) S_part[wh][2 * ei + hrow][l16] = s_acc[ei];
    }
    __syncthreads();

    // 3. 4x-Parallel Online Softmax
    if (wd == 0) {
      const bool vld = (s_valid[l16] != 0);
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) {
        const int hm = 2 * ei + hrow;
        const int h_loc = wh * 16 + hm;
        float sc = s_acc[ei] + S_part[wh][hm][l16];
        sc = vld ? (sc * scale) : NEG_INF;
        const float bmax = halfwave_max16(sc);
        const float mold = s_m[h_loc];
        const float mnew = fmaxf(mold, bmax);
        const float alpha = (mold > -1e30f) ? exp2f((mold - mnew) * LOG2E) : 0.0f;
        const float pv = (vld && mnew > -1e30f) ? exp2f((sc - mnew) * LOG2E) : 0.0f;
        const float bsum = halfwave_sum16(pv);
        P_lds[wh][hm][l16] = f32_to_bf16_rne(pv);
        if (l16 == 0) {
          s_alpha[h_loc] = alpha;
          s_m[h_loc] = mnew;
          s_l[h_loc] = s_l[h_loc] * alpha + bsum;
        }
      }
    }
    __syncthreads();

    // 4. Rescale o_acc[0..15] & P @ V via 128-bit VT_lds loads (1-ahead prefetch, ZERO sched_barrier!)
    float al[8];
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) al[ei] = s_alpha[wh * 16 + 2 * ei + hrow];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) o_acc[j][ei] *= al[ei];
    }

    const u4v p0 = *reinterpret_cast<const u4va*>(&P_lds[wh][l16][0]);
    const u4v p1 = *reinterpret_cast<const u4va*>(&P_lds[wh][l16][8]);
    const bf16x16 pf = __builtin_bit_cast(bf16x16, (u8v){p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w});

    u4v v0_c = *reinterpret_cast<const u4va*>(&VT_lds[vd_start + l16][0]);
    u4v v1_c = *reinterpret_cast<const u4va*>(&VT_lds[vd_start + l16][8]);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      u4v v0_n = {0u, 0u, 0u, 0u}, v1_n = {0u, 0u, 0u, 0u};
      if (j + 1 < 16) {
        const int vc_n = vd_start + (j + 1) * 16 + l16;
        v0_n = *reinterpret_cast<const u4va*>(&VT_lds[vc_n][0]);
        v1_n = *reinterpret_cast<const u4va*>(&VT_lds[vc_n][8]);
      }
      const bf16x16 vf = __builtin_bit_cast(bf16x16, (u8v){v0_c.x, v0_c.y, v0_c.z, v0_c.w, v1_c.x, v1_c.y, v1_c.z, v1_c.w});
      o_acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf, vf, o_acc[j]);
      v0_c = v0_n; v1_c = v1_n;
    }
    __syncthreads();
  }

  // 5. Epilogue
  if (tid < 64) {
    float m = s_m[tid];
    float l = s_l[tid];
    float inv_l = 0.0f;
    if (sink_ptr != nullptr) {
      const float sk = sink_ptr[tid];
      const float mf = fmaxf(m, sk);
      const float lf = (l > 0.0f ? l * exp2f((m - mf) * LOG2E) : 0.0f) + exp2f((sk - mf) * LOG2E);
      const float w_acc = (l > 0.0f) ? exp2f((m - mf) * LOG2E) : 0.0f;
      inv_l = (lf > 0.0f) ? (w_acc * __builtin_amdgcn_rcpf(lf)) : 0.0f;
    } else {
      inv_l = (l > 0.0f) ? __builtin_amdgcn_rcpf(l) : 0.0f;
    }
    s_alpha[tid] = inv_l;
  }
  __syncthreads();

#pragma unroll
  for (int ei = 0; ei < 8; ++ei) {
    const int hm = 2 * ei + hrow;
    const float inv_l = s_alpha[h0 + hm];
    uint16_t* dst = out_ptr + (int64_t)t * out_s0 + (int64_t)(h0 + hm) * out_s1 + vd_start + l16;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      dst[j * 16] = f32_to_bf16_rne(o_acc[j][ei] * inv_l);
    }
  }
}


torch::Tensor ernie_dsa_sparse_mla(
    torch::Tensor q, torch::Tensor kv, torch::Tensor indices,
    c10::optional<torch::Tensor> attn_sink, double scale,
    c10::optional<torch::Tensor> part_o_opt,
    c10::optional<torch::Tensor> part_m_opt,
    c10::optional<torch::Tensor> part_l_opt,
    int64_t num_splits) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 3 && q.size(1) == 64 && q.size(2) == DSA_DIM && q.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(kv.is_cuda() && kv.dim() == 2 && kv.size(1) == DSA_DIM && kv.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(indices.is_cuda() && indices.dim() == 2 && indices.scalar_type() == torch::kInt32);
  const int T = (int)q.size(0);
  const int W = (int)indices.size(1);
  auto out = torch::empty({T, 64, DSA_NOPE}, q.options());
  if (T == 0) return out;
  auto st = c10::hip::getCurrentHIPStream().stream();
  const uint16_t* qp = reinterpret_cast<const uint16_t*>(q.data_ptr());
  const uint16_t* kvp = reinterpret_cast<const uint16_t*>(kv.data_ptr());
  const int32_t* ip = indices.data_ptr<int32_t>();
  const float* sp = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
  uint16_t* op = reinterpret_cast<uint16_t*>(out.data_ptr());

  if (num_splits <= 1 || W <= 32) {
    if (T >= 32) {
      hipLaunchKernelGGL(k_ernie_dsa_sparse_mla_prefill_v28, dim3(T, 1, 1), dim3(256), 0, st,
                         qp, kvp, ip, sp, op,
                         q.stride(0), q.stride(1), kv.stride(0), indices.stride(0), out.stride(0), out.stride(1),
                         (int)kv.size(0), W, (float)scale);
    } else {
      hipLaunchKernelGGL((k_ernie_dsa_sparse_mla<1, 0>), dim3(T, 4, 1), dim3(128), 0, st,
                         qp, kvp, ip, sp, op, nullptr, nullptr, nullptr,
                         q.stride(0), q.stride(1), kv.stride(0), indices.stride(0), out.stride(0), out.stride(1),
                         (int)kv.size(0), W, (float)scale);
    }
  } else {
    constexpr int S = 8;
    torch::Tensor po = part_o_opt.has_value() ? *part_o_opt : torch::empty({T, 64, S, DSA_NOPE}, q.options().dtype(torch::kFloat32));
    torch::Tensor pm = part_m_opt.has_value() ? *part_m_opt : torch::empty({T, 64, S}, q.options().dtype(torch::kFloat32));
    torch::Tensor pl = part_l_opt.has_value() ? *part_l_opt : torch::empty({T, 64, S}, q.options().dtype(torch::kFloat32));
    hipLaunchKernelGGL((k_ernie_dsa_sparse_mla<S, 1>), dim3(T, 4, S), dim3(128), 0, st,
                       qp, kvp, ip, sp, op, po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(),
                       q.stride(0), q.stride(1), kv.stride(0), indices.stride(0), out.stride(0), out.stride(1),
                       (int)kv.size(0), W, (float)scale);
    hipLaunchKernelGGL((k_ernie_dsa_split_reduce<S>), dim3(T, 64), dim3(128), 0, st,
                       po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(), sp, op,
                       out.stride(0), out.stride(1));
  }
  return out;
}

// ================================================================== 2: FP8 MQA Logits Prefill (H=64, D=128)
// Exact E4M3FN (OCP) or E4M3FNUZ -> FP16 conversion helper (computed once into 256-entry LDS LUT)
__device__ __forceinline__ _Float16 fp8_byte_to_f16(uint32_t b, int is_fnuz) {
  const uint32_t sign = (b >> 7) & 1u;
  const uint32_t exp = (b >> 3) & 0xFu;
  const uint32_t man = b & 0x7u;
  float val = 0.0f;
  if (is_fnuz) {
    if (exp == 0) {
      val = (sign && man == 0) ? NAN : ldexpf((float)man, -10);
    } else {
      val = ldexpf(1.0f + (float)man * 0.125f, (int)exp - 8);
    }
    if (sign && !(exp == 0 && man == 0)) val = -val;
  } else {
    if (exp == 15 && man == 7) {
      val = NAN;
    } else if (exp == 0) {
      val = ldexpf((float)man, -9);
      if (sign) val = -val;
    } else {
      val = ldexpf(1.0f + (float)man * 0.125f, (int)exp - 7);
      if (sign) val = -val;
    }
  }
  return (_Float16)val;
}

// Tiled Register-Q + Cached-K + 2-Stage Software-Pipelined FP8 MQA Logits Prefill (6x/prefill!):
// Decodes q_fp8[m, 64, 128] ONCE into registers across all K tiles in this split (128x fewer Q reads & LUT decodes at T=2048!),
// keeps k_fp8 (256 KiB @ T=2048) 100% cached in L2/Infinity Cache, and overlaps K fetches behind all 8 LDS-fed WMMAs!
// Grid: (n_splits, M), Block: 128 (4 Wave32s, 16 heads/wave = all 64 heads!)
__global__ __launch_bounds__(128) void k_ernie_fp8_mqa_logits_prefill(
    const uint8_t* __restrict__ q_fp8,       // [M, 64, 128]
    const uint8_t* __restrict__ k_fp8,       // [N, 128]
    const float* __restrict__ k_scale,       // [N]
    const float* __restrict__ weights,       // [M, 64]
    const int32_t* __restrict__ cu_ks,       // [M]
    const int32_t* __restrict__ cu_ke,       // [M]
    float* __restrict__ out_logits,          // [M, N]
    int64_t q_s0, int64_t q_s1, int64_t k_s0, int64_t w_s0,
    int N, int n_splits, int is_fnuz) {
  __shared__ _Float16 fp8_lut[256];
  __shared__ __attribute__((aligned(16))) _Float16 K_lds[16][136]; // 4.25 KiB
  __shared__ float s_wsum[4][16];
  __shared__ float s_weights[64];
  __shared__ float s_kscale[16];

  const int split = blockIdx.x;
  const int m = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;

  const int n_tiles = (N + 15) >> 4;
  const int tiles_per_sp = (n_tiles + n_splits - 1) / n_splits;
  const int t_lo = split * tiles_per_sp;
  const int t_hi = min(n_tiles, t_lo + tiles_per_sp);
  if (t_lo >= t_hi) return;

  const int n_lo = t_lo << 4;
  const int n_hi = min(N, t_hi << 4);
  const int ks_c = max(0, min(N, cu_ks[m]));
  const int ke_c = max(ks_c, min(N, cu_ke[m]));
  float* out_row = out_logits + (int64_t)m * N;

  const int act_t_lo = max(t_lo, ks_c >> 4);
  const int act_t_hi = min(t_hi, (ke_c + 15) >> 4);
  if (act_t_lo >= act_t_hi) {
    for (int gn = n_lo + tid; gn < n_hi; gn += 128) out_row[gn] = -INFINITY;
    return;
  }
  // Fill out-of-range causal prefix & suffix in this split with -INFINITY in parallel
  const int pre_end = min(n_hi, act_t_lo << 4);
  for (int gn = n_lo + tid; gn < pre_end; gn += 128) out_row[gn] = -INFINITY;
  const int suf_start = max(n_lo, act_t_hi << 4);
  for (int gn = suf_start + tid; gn < n_hi; gn += 128) out_row[gn] = -INFINITY;

  // Build 256-entry FP8->FP16 LUT ONCE + cache weights[m, 0..63] ONCE in LDS
  fp8_lut[tid] = fp8_byte_to_f16((uint32_t)tid, is_fnuz);
  fp8_lut[tid + 128] = fp8_byte_to_f16((uint32_t)(tid + 128), is_fnuz);
  if (tid < 64) s_weights[tid] = weights[(int64_t)m * w_s0 + tid];
  __syncthreads();

  // Decode Q[m, h0..h0+15, 0..127] ONCE into registers (2x burst avoids VGPR spike -> <=112 VGPRs = 13w/SIMD!)
  const int h0 = wave * 16;
  const uint8_t* qp = q_fp8 + (int64_t)m * q_s0 + (int64_t)(h0 + l16) * q_s1;
  f16x16 q_reg[8];
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const u4v raw0 = *reinterpret_cast<const u4va*>(qp + (pair * 2 + 0) * 16);
    const u4v raw1 = *reinterpret_cast<const u4va*>(qp + (pair * 2 + 1) * 16);
    const u4v raws[2] = {raw0, raw1};
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      const uint32_t ws[4] = {raws[s].x, raws[s].y, raws[s].z, raws[s].w};
      f16x16 qf;
#pragma unroll
      for (int w4 = 0; w4 < 4; ++w4) {
        const uint32_t x = ws[w4];
        qf[w4 * 4 + 0] = fp8_lut[x & 0xFFu];
        qf[w4 * 4 + 1] = fp8_lut[(x >> 8) & 0xFFu];
        qf[w4 * 4 + 2] = fp8_lut[(x >> 16) & 0xFFu];
        qf[w4 * 4 + 3] = fp8_lut[(x >> 24) & 0xFFu];
      }
      q_reg[pair * 2 + s] = qf;
    }
    __builtin_amdgcn_sched_barrier(0);
  }

  const int kr = tid >> 3;         // 0..15
  const int kc16 = (tid & 7) * 16; // 0, 16, ..., 112

  // Cached k_fp8 fetch (stays 100% resident in L2/Infinity Cache across all M queries!)
  auto fetch_k = [&](int ti, u4v& out_raw, float& out_sc) {
    const int n0 = ti << 4;
    const int gn = n0 + kr;
    if (gn >= ks_c && gn < ke_c) {
      out_raw = *reinterpret_cast<const u4va*>(k_fp8 + (int64_t)gn * k_s0 + kc16);
    } else {
      out_raw = (u4v){0u, 0u, 0u, 0u};
    }
    if (tid < 16) {
      const int gns = n0 + tid;
      out_sc = (gns >= ks_c && gns < ke_c) ? k_scale[gns] : 0.0f;
    } else {
      out_sc = 0.0f;
    }
  };

  u4v raw_cur = {0u, 0u, 0u, 0u};
  float sc_cur = 0.0f;
  fetch_k(act_t_lo, raw_cur, sc_cur);

  for (int ti = act_t_lo; ti < act_t_hi; ++ti) {
    const int n0 = ti << 4;
    const int gn = n0 + kr;
    if (gn >= ks_c && gn < ke_c) {
      const uint32_t ws[4] = {raw_cur.x, raw_cur.y, raw_cur.z, raw_cur.w};
      uint32_t pk[8];
#pragma unroll
      for (int w4 = 0; w4 < 4; ++w4) {
        const uint32_t x = ws[w4];
        const uint32_t h0b = (uint32_t)__builtin_bit_cast(uint16_t, fp8_lut[x & 0xFFu]);
        const uint32_t h1b = (uint32_t)__builtin_bit_cast(uint16_t, fp8_lut[(x >> 8) & 0xFFu]);
        const uint32_t h2b = (uint32_t)__builtin_bit_cast(uint16_t, fp8_lut[(x >> 16) & 0xFFu]);
        const uint32_t h3b = (uint32_t)__builtin_bit_cast(uint16_t, fp8_lut[(x >> 24) & 0xFFu]);
        pk[w4 * 2 + 0] = h0b | (h1b << 16);
        pk[w4 * 2 + 1] = h2b | (h3b << 16);
      }
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16 + 0]) = (u4v){pk[0], pk[1], pk[2], pk[3]};
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16 + 8]) = (u4v){pk[4], pk[5], pk[6], pk[7]};
    } else {
      const u4v z = {0u, 0u, 0u, 0u};
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16]) = z;
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16 + 8]) = z;
    }
    if (tid < 16) s_kscale[tid] = sc_cur;
    __syncthreads();

    // Overlap next tile's k_fp8 + k_scale fetch behind all 8 LDS-fed WMMAs + reduction!
    u4v raw_nxt = {0u, 0u, 0u, 0u};
    float sc_nxt = 0.0f;
    if (ti + 1 < act_t_hi) fetch_k(ti + 1, raw_nxt, sc_nxt);
    __builtin_amdgcn_sched_barrier(0);

    f32x8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int step = 0; step < 8; ++step) {
      const int d = step * 16;
      const u4v k0 = *reinterpret_cast<const u4va*>(&K_lds[l16][d]);
      const u4v k1 = *reinterpret_cast<const u4va*>(&K_lds[l16][d + 8]);
      const f16x16 kf = __builtin_bit_cast(f16x16, (u8v){k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w});
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(q_reg[step], kf, acc);
      __builtin_amdgcn_sched_barrier(0);
    }

    const float kscl = s_kscale[l16];
    float sum_h = 0.0f;
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      const float sc = fmaxf(acc[ei] * kscl, 0.0f);
      sum_h = __builtin_fmaf(sc, s_weights[h0 + 2 * ei + hrow], sum_h);
    }
    sum_h += shx_f<16>(sum_h);
    if (hrow == 0) s_wsum[wave][l16] = sum_h;
    __syncthreads();

    if (tid < 16) {
      const int gn_out = n0 + tid;
      if (gn_out < N) {
        if (gn_out >= ks_c && gn_out < ke_c) {
          out_row[gn_out] = s_wsum[0][tid] + s_wsum[1][tid] + s_wsum[2][tid] + s_wsum[3][tid];
        } else {
          out_row[gn_out] = -INFINITY;
        }
      }
    }
    __syncthreads();
    raw_cur = raw_nxt;
    sc_cur = sc_nxt;
  }
}

// FlashIndexer H=8 FP8 MQA Logits Prefill (v0.2.8 Wave 11A):
//   - Block=256 (8 Wave32s), M_BLK=16 (2 query tokens x 8 heads = 16 Q-rows/wave -> 100% WMMA!)
//   - N_TILE=32 (2 K-tiles/iter, 16 tokens share every K_lds[32][136] -> 4x fewer K loads & FP8 decodes)
//   - Ping-pong kf0/kf1 register overlap across 16 WMMAs (zero lgkmcnt(0) stalls!)
//   - Weights hoisted into VGPRs once; full 32-lane 128B coalesced output stores; GL2-resident dim3(m_blocks, n_splits)
template <int NT_OUT>
__global__ __launch_bounds__(256, 2) void k_ernie_fp8_mqa_logits_prefill_h8(
    const uint8_t* __restrict__ q_fp8,       // [M, 8, 128]
    const uint8_t* __restrict__ k_fp8,       // [N, 128]
    const float* __restrict__ k_scale,       // [N]
    const float* __restrict__ weights,       // [M, 8]
    const int32_t* __restrict__ cu_ks,       // [M]
    const int32_t* __restrict__ cu_ke,       // [M]
    float* __restrict__ out_logits,          // [M, N]
    int64_t q_s0, int64_t q_s1, int64_t k_s0, int64_t w_s0,
    int M, int N, int n_splits, int is_fnuz) {
  __shared__ _Float16 fp8_lut[256];
  __shared__ __attribute__((aligned(16))) _Float16 K_lds[32][136]; // 8.5 KiB
  __shared__ float s_kscale[32];
  __shared__ int s_ks[16];
  __shared__ int s_ke[16];
  __shared__ int s_blk_lo;
  __shared__ int s_blk_hi;

  const int m_base = blockIdx.x * 16;
  const int split = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;   // 0..7
  const int lane = tid & 31;   // 0..31
  const int l16 = lane & 15;   // 0..15
  const int hrow = lane >> 4;  // 0..1
  const int m0 = m_base + wave * 2;
  const int m1 = m0 + 1;
  const bool m0_valid = (m0 < M);
  const bool m1_valid = (m1 < M);

  const int n_tiles = (N + 31) >> 5;
  const int tiles_per_sp = (n_tiles + n_splits - 1) / n_splits;
  const int t_lo = split * tiles_per_sp;
  const int t_hi = min(n_tiles, t_lo + tiles_per_sp);
  if (t_lo >= t_hi) return;

  const int n_lo = t_lo << 5;
  const int n_hi = min(N, t_hi << 5);

  fp8_lut[tid] = fp8_byte_to_f16((uint32_t)tid, is_fnuz);

  if (tid < 16) {
    const int m_i = m_base + tid;
    if (m_i < M) {
      int ks_v = max(0, min(N, cu_ks[m_i]));
      int ke_v = max(ks_v, min(N, cu_ke[m_i]));
      s_ks[tid] = ks_v;
      s_ke[tid] = ke_v;
    } else {
      s_ks[tid] = N;
      s_ke[tid] = N;
    }
  }
  __syncthreads();

  if (tid == 0) {
    int blo = t_hi;
    int bhi = t_lo;
    for (int w = 0; w < 16; ++w) {
      if (m_base + w < M && s_ks[w] < s_ke[w]) {
        int alo = max(t_lo, s_ks[w] >> 5);
        int ahi = min(t_hi, (s_ke[w] + 31) >> 5);
        if (alo < ahi) {
          if (alo < blo) blo = alo;
          if (ahi > bhi) bhi = ahi;
        }
      }
    }
    s_blk_lo = blo;
    s_blk_hi = bhi;
  }
  __syncthreads();

  const int blk_lo = s_blk_lo;
  const int blk_hi = s_blk_hi;
  const int ks_m0 = s_ks[wave * 2 + 0];
  const int ke_m0 = s_ke[wave * 2 + 0];
  const int ks_m1 = s_ks[wave * 2 + 1];
  const int ke_m1 = s_ke[wave * 2 + 1];
  float* out_row0 = m0_valid ? (out_logits + (int64_t)m0 * N) : nullptr;
  float* out_row1 = m1_valid ? (out_logits + (int64_t)m1 * N) : nullptr;

  auto store_out = [](float* ptr, float val) {
    if constexpr (NT_OUT) {
      __builtin_nontemporal_store(val, ptr);
    } else {
      *ptr = val;
    }
  };

  if (blk_lo >= blk_hi) {
    if (m0_valid) {
      for (int gn = n_lo + lane; gn < n_hi; gn += 32) store_out(out_row0 + gn, -INFINITY);
    }
    if (m1_valid) {
      for (int gn = n_lo + lane; gn < n_hi; gn += 32) store_out(out_row1 + gn, -INFINITY);
    }
    return;
  }
  {
    const int pre_end = min(n_hi, blk_lo << 5);
    const int suf_start = max(n_lo, blk_hi << 5);
    if (m0_valid) {
      for (int gn = n_lo + lane; gn < pre_end; gn += 32) store_out(out_row0 + gn, -INFINITY);
      for (int gn = suf_start + lane; gn < n_hi; gn += 32) store_out(out_row0 + gn, -INFINITY);
    }
    if (m1_valid) {
      for (int gn = n_lo + lane; gn < pre_end; gn += 32) store_out(out_row1 + gn, -INFINITY);
      for (int gn = suf_start + lane; gn < n_hi; gn += 32) store_out(out_row1 + gn, -INFINITY);
    }
  }

  // Hoist per-wave weights for m0 & m1 into 8 VGPRs ONCE (zero LDS reads in loop!)
  float w0[4], w1[4];
#pragma unroll
  for (int ei = 0; ei < 4; ++ei) {
    const int h = 2 * ei + hrow;
    w0[ei] = m0_valid ? weights[(int64_t)m0 * w_s0 + h] : 0.0f;
    w1[ei] = m1_valid ? weights[(int64_t)m1 * w_s0 + h] : 0.0f;
  }

  // Decode Q[m0..m1, 0..7, 0..127] ONCE into registers (l16<8 -> m0, l16>=8 -> m1)
  const int m_q = (l16 < 8) ? m0 : m1;
  const int h_q = l16 & 7;
  const bool q_valid = (m_q < M);
  f16x16 q_reg[8];
  if (q_valid) {
    const uint8_t* qp = q_fp8 + (int64_t)m_q * q_s0 + (int64_t)h_q * q_s1;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const u4v raw0 = *reinterpret_cast<const u4va*>(qp + (pair * 2 + 0) * 16);
      const u4v raw1 = *reinterpret_cast<const u4va*>(qp + (pair * 2 + 1) * 16);
      const u4v raws[2] = {raw0, raw1};
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        const uint32_t ws[4] = {raws[s].x, raws[s].y, raws[s].z, raws[s].w};
        f16x16 qf;
#pragma unroll
        for (int w4 = 0; w4 < 4; ++w4) {
          const uint32_t x = ws[w4];
          qf[w4 * 4 + 0] = fp8_lut[x & 0xFFu];
          qf[w4 * 4 + 1] = fp8_lut[(x >> 8) & 0xFFu];
          qf[w4 * 4 + 2] = fp8_lut[(x >> 16) & 0xFFu];
          qf[w4 * 4 + 3] = fp8_lut[(x >> 24) & 0xFFu];
        }
        q_reg[pair * 2 + s] = qf;
      }
      __builtin_amdgcn_sched_barrier(0);
    }
  } else {
    const f16x16 zf = {};
#pragma unroll
    for (int s = 0; s < 8; ++s) q_reg[s] = zf;
  }

  const int kr = tid >> 3;         // 0..31 (exact 1 row per 8 threads!)
  const int kc16 = (tid & 7) * 16; // 0, 16, ..., 112
  const int blk_n_lo = blk_lo << 5;
  const int blk_n_hi = min(N, blk_hi << 5);

  auto fetch_k = [&](int ti, u4v& out_raw, float& out_sc) {
    const int n0 = ti << 5;
    const int gn = n0 + kr;
    if (gn >= blk_n_lo && gn < blk_n_hi) {
      out_raw = *reinterpret_cast<const u4va*>(k_fp8 + (int64_t)gn * k_s0 + kc16);
    } else {
      out_raw = (u4v){0u, 0u, 0u, 0u};
    }
    if (tid < 32) {
      const int gns = n0 + tid;
      out_sc = (gns >= blk_n_lo && gns < blk_n_hi) ? k_scale[gns] : 0.0f;
    } else {
      out_sc = 0.0f;
    }
  };

  auto load_k16 = [](const _Float16* row_ptr, int d) -> f16x16 {
    const u4v k0 = *reinterpret_cast<const u4va*>(row_ptr + d);
    const u4v k1 = *reinterpret_cast<const u4va*>(row_ptr + d + 8);
    return __builtin_bit_cast(f16x16, (u8v){k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w});
  };

  u4v raw_cur = {0u, 0u, 0u, 0u};
  float sc_cur = 0.0f;
  fetch_k(blk_lo, raw_cur, sc_cur);

  for (int ti = blk_lo; ti < blk_hi; ++ti) {
    const int n0 = ti << 5;
    const int gn = n0 + kr;
    if (gn >= blk_n_lo && gn < blk_n_hi) {
      const uint32_t ws[4] = {raw_cur.x, raw_cur.y, raw_cur.z, raw_cur.w};
      uint32_t pk[8];
#pragma unroll
      for (int w4 = 0; w4 < 4; ++w4) {
        const uint32_t x = ws[w4];
        const uint32_t h0b = (uint32_t)__builtin_bit_cast(uint16_t, fp8_lut[x & 0xFFu]);
        const uint32_t h1b = (uint32_t)__builtin_bit_cast(uint16_t, fp8_lut[(x >> 8) & 0xFFu]);
        const uint32_t h2b = (uint32_t)__builtin_bit_cast(uint16_t, fp8_lut[(x >> 16) & 0xFFu]);
        const uint32_t h3b = (uint32_t)__builtin_bit_cast(uint16_t, fp8_lut[(x >> 24) & 0xFFu]);
        pk[w4 * 2 + 0] = h0b | (h1b << 16);
        pk[w4 * 2 + 1] = h2b | (h3b << 16);
      }
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16 + 0]) = (u4v){pk[0], pk[1], pk[2], pk[3]};
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16 + 8]) = (u4v){pk[4], pk[5], pk[6], pk[7]};
    } else {
      const u4v z = {0u, 0u, 0u, 0u};
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16]) = z;
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16 + 8]) = z;
    }
    if (tid < 32) s_kscale[tid] = sc_cur;
    __syncthreads();

    u4v raw_nxt = {0u, 0u, 0u, 0u};
    float sc_nxt = 0.0f;
    if (ti + 1 < blk_hi) fetch_k(ti + 1, raw_nxt, sc_nxt);
    __builtin_amdgcn_sched_barrier(0);

    f32x8 acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    f32x8 acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
    f16x16 kf0 = load_k16(&K_lds[l16][0], 0);
#pragma unroll
    for (int step = 0; step < 8; ++step) {
      const int d = step * 16;
      f16x16 kf1 = load_k16(&K_lds[16 + l16][0], d);
      acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(q_reg[step], kf0, acc0);
      if (step + 1 < 8) {
        kf0 = load_k16(&K_lds[l16][0], d + 16);
      }
      acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(q_reg[step], kf1, acc1);
    }

    const float ks0 = s_kscale[l16];
    const float ks1 = s_kscale[16 + l16];
    float s0a = 0.0f, s1a = 0.0f, s0b = 0.0f, s1b = 0.0f;
#pragma unroll
    for (int ei = 0; ei < 4; ++ei) {
      s0a = __builtin_fmaf(fmaxf(acc0[ei] * ks0, 0.0f), w0[ei], s0a);
      s1a = __builtin_fmaf(fmaxf(acc0[ei + 4] * ks0, 0.0f), w1[ei], s1a);
      s0b = __builtin_fmaf(fmaxf(acc1[ei] * ks1, 0.0f), w0[ei], s0b);
      s1b = __builtin_fmaf(fmaxf(acc1[ei + 4] * ks1, 0.0f), w1[ei], s1b);
    }
    s0a += shx_f<16>(s0a);
    s1a += shx_f<16>(s1a);
    s0b += shx_f<16>(s0b);
    s1b += shx_f<16>(s1b);

    const int gn_out = n0 + lane;
    if (gn_out < N) {
      if (m0_valid) {
        const float v0 = (hrow == 0) ? s0a : s0b;
        store_out(out_row0 + gn_out, (gn_out >= ks_m0 && gn_out < ke_m0) ? v0 : -INFINITY);
      }
      if (m1_valid) {
        const float v1 = (hrow == 0) ? s1a : s1b;
        store_out(out_row1 + gn_out, (gn_out >= ks_m1 && gn_out < ke_m1) ? v1 : -INFINITY);
      }
    }
    __syncthreads();
    raw_cur = raw_nxt;
    sc_cur = sc_nxt;
  }
}

torch::Tensor ernie_fp8_mqa_logits_prefill(
    torch::Tensor q_fp8, torch::Tensor k_fp8, torch::Tensor k_scale,
    torch::Tensor weights, torch::Tensor cu_ks, torch::Tensor cu_ke) {
  TORCH_CHECK(q_fp8.is_cuda() && q_fp8.dim() == 3 && (q_fp8.size(1) == 64 || q_fp8.size(1) == 8) && q_fp8.size(2) == 128);
  TORCH_CHECK(k_fp8.is_cuda() && k_fp8.dim() == 2 && k_fp8.size(1) == 128);
  const int M = (int)q_fp8.size(0);
  const int H = (int)q_fp8.size(1);
  const int N = (int)k_fp8.size(0);
  auto out = torch::empty({M, N}, weights.options().dtype(torch::kFloat32));
  if (M == 0 || N == 0) return out;
  const int is_fnuz = (q_fp8.scalar_type() == torch::kFloat8_e4m3fnuz) ? 1 : 0;
  auto ks_flat = k_scale.reshape({N}).contiguous();
  auto st = c10::hip::getCurrentHIPStream().stream();
  const int n_tiles = (N + 15) / 16;
  if (H == 8) {
    const int m_blocks = (M + 15) / 16;
    const int n_tiles32 = (N + 31) / 32;
    const int n_splits = max(1, min(n_tiles32, max(max(1, 256 / max(1, m_blocks)), (N + 16383) / 16384)));
    hipLaunchKernelGGL(
        (k_ernie_fp8_mqa_logits_prefill_h8<1>), dim3(m_blocks, n_splits), dim3(256), 0, st,
        reinterpret_cast<const uint8_t*>(q_fp8.data_ptr()),
        reinterpret_cast<const uint8_t*>(k_fp8.data_ptr()),
        ks_flat.data_ptr<float>(),
        weights.data_ptr<float>(),
        cu_ks.data_ptr<int32_t>(),
        cu_ke.data_ptr<int32_t>(),
        out.data_ptr<float>(),
        q_fp8.stride(0), q_fp8.stride(1), k_fp8.stride(0), weights.stride(0),
        M, N, n_splits, is_fnuz);
    return out;
  }
  const int n_splits = max(1, min(n_tiles, max(1, 256 / M)));
  hipLaunchKernelGGL(
      k_ernie_fp8_mqa_logits_prefill, dim3(n_splits, M), dim3(128), 0, st,
      reinterpret_cast<const uint8_t*>(q_fp8.data_ptr()),
      reinterpret_cast<const uint8_t*>(k_fp8.data_ptr()),
      ks_flat.data_ptr<float>(),
      weights.data_ptr<float>(),
      cu_ks.data_ptr<int32_t>(),
      cu_ke.data_ptr<int32_t>(),
      out.data_ptr<float>(),
      q_fp8.stride(0), q_fp8.stride(1), k_fp8.stride(0), weights.stride(0),
      N, n_splits, is_fnuz);
  return out;
}

// ================================================================== 3: FP8 Paged MQA Logits Decode (H=64, D=128)
// Replaces software-emulated Triton tl.dot(fp8, fp8) on gfx1151 (which lacks FP8 WMMA!)
// with 256-entry FP8->FP16 LDS LUT + hardware wmma_f32_16x16x16_f16_w32!
// Supports BOTH SHUFFLE layout (block_size>1, compress_ratio!=4: [bs//16, 8, 16, 16] written by
// indexer_k_quant_and_cache_triton!) and NORMAL flat layout (block_size==1 or C4A).
// Q stays decoded in registers (8 x f16x16 = 32 VGPRs) across ALL KV tiles!
// 100% HIP-graph safe (no host sync, static grid).
template <int IS_SHUFFLE>
__global__ __launch_bounds__(128) void k_ernie_fp8_paged_mqa_logits_decode(
    const uint8_t* __restrict__ q_fp8,       // [B, NEXT_N, 64, 128]
    const uint8_t* __restrict__ kv_flat,     // [num_blocks, stride_blk_bytes] uint8
    const float* __restrict__ weights,       // [B*NEXT_N, 64] fp32
    const int32_t* __restrict__ ctx_lens,    // [B*NEXT_N] or [B] int32
    const int32_t* __restrict__ block_tables,// [B, max_blocks] int32
    float* __restrict__ out_logits,          // [B*NEXT_N, max_model_len] fp32
    int64_t q_sb, int64_t q_sn, int64_t q_sh,
    int64_t kv_s0, int64_t w_s0, int64_t bt_s0, int64_t out_s0,
    int block_size, int max_blocks, int max_model_len,
    int next_n, int n_splits, int ctx_per_row, int is_fnuz) {
  __shared__ _Float16 fp8_lut[256];
  __shared__ __attribute__((aligned(16))) _Float16 K_lds[16][136]; // 4.25 KiB
  __shared__ float s_wsum[4][16];
  __shared__ float s_weights[64]; // 256 B LDS -> saves 8 VGPRs (161/164 -> <=160 = 9w/SIMD!)

  const int row = blockIdx.x;
  const int split = blockIdx.y;
  const int b = row / next_n;
  const int n = row % next_n;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;

  int seq_len = ctx_per_row ? ctx_lens[row] : (ctx_lens[b] - next_n + n + 1);
  if (seq_len < 0) seq_len = 0;
  if (seq_len > max_model_len) seq_len = max_model_len;
  if (split * 16 >= seq_len) return;

  // Build 256-entry FP8->FP16 LUT
  fp8_lut[tid] = fp8_byte_to_f16((uint32_t)tid, is_fnuz);
  fp8_lut[tid + 128] = fp8_byte_to_f16((uint32_t)(tid + 128), is_fnuz);
  __syncthreads();

  // Decode Q[b, n, h0..h0+15, 0..127] ONCE into registers across all KV tiles!
  // Unroll-2 + sched_barrier avoids 32-VGPR 8x global_load_b128 spike -> drops VGPRs 146 -> <=128 (12w/SIMD!)
  const int h0 = wave * 16;
  const uint8_t* qp = q_fp8 + (int64_t)b * q_sb + (int64_t)n * q_sn + (int64_t)(h0 + l16) * q_sh;
  f16x16 q_reg[8];
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const u4v raw0 = *reinterpret_cast<const u4va*>(qp + (pair * 2 + 0) * 16);
    const u4v raw1 = *reinterpret_cast<const u4va*>(qp + (pair * 2 + 1) * 16);
    const u4v raws[2] = {raw0, raw1};
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      const uint32_t ws[4] = {raws[s].x, raws[s].y, raws[s].z, raws[s].w};
      f16x16 qf;
#pragma unroll
      for (int w4 = 0; w4 < 4; ++w4) {
        const uint32_t x = ws[w4];
        qf[w4 * 4 + 0] = fp8_lut[x & 0xFFu];
        qf[w4 * 4 + 1] = fp8_lut[(x >> 8) & 0xFFu];
        qf[w4 * 4 + 2] = fp8_lut[(x >> 16) & 0xFFu];
        qf[w4 * 4 + 3] = fp8_lut[(x >> 24) & 0xFFu];
      }
      q_reg[pair * 2 + s] = qf;
    }
    __builtin_amdgcn_sched_barrier(0);
  }

  // Cache weights[row, 0..63] in 256 B LDS instead of 8 VGPRs/lane to stay <=128 VGPRs (12w/SIMD!)
  if (tid < 64) s_weights[tid] = weights[(int64_t)row * w_s0 + tid];
  const uint32_t scale_byte_off = (uint32_t)(block_size * 128);
  const int kr = tid >> 3;         // 0..15 (tile_block_offset)
  const int ht = tid & 7;          // 0..7  (head_tile_id, 16 bytes each)
  const int kc16 = ht * 16;        // 0, 16, ..., 112
  const uint32_t shuf_lane_off = (uint32_t)(ht * 256 + kr * 16);
  const int stride_kv = n_splits * 16;

  // 2-stage register software prefetch of 16B NT K-cache + FP32 scale across kv_start tiles:
  // Overlaps 100% of LPDDR5X K-tile fetch behind all 8 LDS-fed WMMAs + reduction (zero vmcnt conflict!)
  auto fetch_k_tile = [&](int kv_st, u4v& out_raw, float& out_sc) {
    const int logical_blk = kv_st / block_size;
    const int64_t page = (int64_t)block_tables[(int64_t)b * bt_s0 + logical_blk];
    const uint32_t base_in_blk = (uint32_t)(kv_st - logical_blk * block_size); // multiple of 16!
    const uint8_t* blk_ptr = kv_flat + page * kv_s0;
    const int pos = kv_st + kr;
    if (pos < seq_len) {
      const uint32_t k_off = IS_SHUFFLE
          ? (base_in_blk * 128u + shuf_lane_off)
          : ((base_in_blk + (uint32_t)kr) * 128u + (uint32_t)kc16);
      out_raw = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(blk_ptr + k_off));
    } else {
      out_raw = (u4v){0u, 0u, 0u, 0u};
    }
    if (tid < 16 && (kv_st + tid) < seq_len) {
      out_sc = *reinterpret_cast<const float*>(blk_ptr + scale_byte_off + (base_in_blk + (uint32_t)tid) * 4u);
    } else {
      out_sc = 0.0f;
    }
  };

  int kv_start = split * 16;
  if ((kv_start / block_size) >= max_blocks) return;
  u4v raw_cur = {0u, 0u, 0u, 0u};
  float sc_cur = 0.0f;
  fetch_k_tile(kv_start, raw_cur, sc_cur);

  // Stride across 16-token tiles (block_size is multiple of 16 -> every tile stays in 1 page!)
  for (; kv_start < seq_len && (kv_start / block_size) < max_blocks; kv_start += stride_kv) {
    // Decode prefetched raw_cur (16 x 128 FP8 K bytes) into K_lds
    if (kv_start + kr < seq_len) {
      const uint32_t ws[4] = {raw_cur.x, raw_cur.y, raw_cur.z, raw_cur.w};
      uint32_t pk[8];
#pragma unroll
      for (int w4 = 0; w4 < 4; ++w4) {
        const uint32_t x = ws[w4];
        const uint32_t h0 = (uint32_t)__builtin_bit_cast(uint16_t, fp8_lut[x & 0xFFu]);
        const uint32_t h1 = (uint32_t)__builtin_bit_cast(uint16_t, fp8_lut[(x >> 8) & 0xFFu]);
        const uint32_t h2 = (uint32_t)__builtin_bit_cast(uint16_t, fp8_lut[(x >> 16) & 0xFFu]);
        const uint32_t h3 = (uint32_t)__builtin_bit_cast(uint16_t, fp8_lut[(x >> 24) & 0xFFu]);
        pk[w4 * 2 + 0] = h0 | (h1 << 16);
        pk[w4 * 2 + 1] = h2 | (h3 << 16);
      }
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16 + 0]) = (u4v){pk[0], pk[1], pk[2], pk[3]};
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16 + 8]) = (u4v){pk[4], pk[5], pk[6], pk[7]};
    } else {
      const u4v z = {0u, 0u, 0u, 0u};
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16]) = z;
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16 + 8]) = z;
    }
    __syncthreads();

    // Immediately issue NEXT tile's 16B NT K-cache + scale fetch BEFORE the 8 LDS-fed WMMAs!
    const int kv_next = kv_start + stride_kv;
    u4v raw_nxt = {0u, 0u, 0u, 0u};
    float sc_nxt = 0.0f;
    if (kv_next < seq_len && (kv_next / block_size) < max_blocks) {
      fetch_k_tile(kv_next, raw_nxt, sc_nxt);
    }
    __builtin_amdgcn_sched_barrier(0);

    f32x8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int step = 0; step < 8; ++step) {
      const int d = step * 16;
      const u4v k0 = *reinterpret_cast<const u4va*>(&K_lds[l16][d]);
      const u4v k1 = *reinterpret_cast<const u4va*>(&K_lds[l16][d + 8]);
      const f16x16 kf = __builtin_bit_cast(f16x16, (u8v){k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w});
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(q_reg[step], kf, acc);
      __builtin_amdgcn_sched_barrier(0);
    }

    float sum_h = 0.0f;
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      sum_h = __builtin_fmaf(fmaxf(acc[ei], 0.0f), s_weights[h0 + 2 * ei + hrow], sum_h);
    }
    sum_h += shx_f<16>(sum_h);
    if (hrow == 0) s_wsum[wave][l16] = sum_h;
    __syncthreads();

    if (tid < 16) {
      const int pos = kv_start + tid;
      if (pos < seq_len) {
        const float tot = (s_wsum[0][tid] + s_wsum[1][tid] + s_wsum[2][tid] + s_wsum[3][tid]) * sc_cur;
        out_logits[(int64_t)row * out_s0 + pos] = tot;
      }
    }
    __syncthreads();
    raw_cur = raw_nxt;
    sc_cur = sc_nxt;
  }
}


// ================================================================== 5B-FP8: Native FlashIndexer (H=8) FP8 Paged MQA Logits Decode
// Batches up to 8 verify rows (e.g. ALL 6 DSpark-5 rows!) in ONE WG (4 waves x 2 rows x 8 heads = 16x16x16 WMMA):
//   - Streams K-cache ONCE per request instead of NEXT_N times (6x less DRAM bandwidth at 64K-1M ctx!)
//   - 8x fewer heads (H=8 vs H=64), zero s_wsum LDS barrier, NT K-cache loads, zero v_div!
//   - 100% HIP-graph safe (static grid, zero host sync).
template <int IS_SHUFFLE>
__global__ __launch_bounds__(128) void k_ernie_fp8_paged_mqa_logits_decode_h8(
    const uint8_t* __restrict__ q_fp8,       // [B, NEXT_N, 8, 128]
    const uint8_t* __restrict__ kv_flat,     // [num_blocks, stride_blk_bytes] uint8
    const float* __restrict__ weights,       // [B*NEXT_N, 8] fp32
    const int32_t* __restrict__ ctx_lens,    // [B*NEXT_N] or [B] int32
    const int32_t* __restrict__ block_tables,// [B, max_blocks] int32
    float* __restrict__ out_logits,          // [B*NEXT_N, max_model_len] fp32
    int64_t q_sb, int64_t q_sn, int64_t q_sh,
    int64_t kv_s0, int64_t w_s0, int64_t bt_s0, int64_t out_s0,
    int block_size, int blk_shift, int max_blocks, int max_model_len,
    int next_n, int n_splits, int ctx_per_row, int is_fnuz) {
  __shared__ _Float16 fp8_lut[256];
  __shared__ __attribute__((aligned(16))) _Float16 K_lds[16][136];
  __shared__ float s_weights[8][8];
  __shared__ int32_t s_sl[8];
  __shared__ float s_sc[16];

  const int b = blockIdx.x;
  const int split = blockIdx.y;
  const int chunk = blockIdx.z;
  const int n_base = chunk * 8;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;

  if (tid < 8) {
    const int n_i = n_base + tid;
    int sl = 0;
    if (n_i < next_n) {
      const int r_i = b * next_n + n_i;
      sl = ctx_per_row ? ctx_lens[r_i] : (ctx_lens[b] - next_n + n_i + 1);
      if (sl < 0) sl = 0;
      if (sl > max_model_len) sl = max_model_len;
    }
    s_sl[tid] = sl;
  }
  fp8_lut[tid] = fp8_byte_to_f16((uint32_t)tid, is_fnuz);
  fp8_lut[tid + 128] = fp8_byte_to_f16((uint32_t)(tid + 128), is_fnuz);
  if (tid < 64) {
    const int r_loc = tid >> 3;
    const int h_loc = tid & 7;
    const int n_i = n_base + r_loc;
    s_weights[r_loc][h_loc] = (n_i < next_n) ? weights[(int64_t)(b * next_n + n_i) * w_s0 + h_loc] : 0.0f;
  }
  __syncthreads();

  int max_sl = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) max_sl = max(max_sl, s_sl[i]);
  if (split * 16 >= max_sl) return;

  // Wave w owns 2 rows (w*2 + 0, w*2 + 1) x 8 heads = 16 Q-rows -> 16x16x16 WMMA!
  const int r_q = wave * 2 + (l16 >> 3);
  const int h_q = l16 & 7;
  const int n_q = n_base + r_q;
  const bool q_valid = (n_q < next_n);
  const uint8_t* qp = q_valid ? (q_fp8 + (int64_t)b * q_sb + (int64_t)n_q * q_sn + (int64_t)h_q * q_sh) : nullptr;
  f16x16 q_reg[8];
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const u4v raw0 = q_valid ? *reinterpret_cast<const u4va*>(qp + (pair * 2 + 0) * 16) : (u4v){0u,0u,0u,0u};
    const u4v raw1 = q_valid ? *reinterpret_cast<const u4va*>(qp + (pair * 2 + 1) * 16) : (u4v){0u,0u,0u,0u};
    const u4v raws[2] = {raw0, raw1};
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      const uint32_t ws[4] = {raws[s].x, raws[s].y, raws[s].z, raws[s].w};
      f16x16 qf;
#pragma unroll
      for (int w4 = 0; w4 < 4; ++w4) {
        const uint32_t x = ws[w4];
        qf[w4 * 4 + 0] = q_valid ? fp8_lut[x & 0xFFu] : (_Float16)0.0f;
        qf[w4 * 4 + 1] = q_valid ? fp8_lut[(x >> 8) & 0xFFu] : (_Float16)0.0f;
        qf[w4 * 4 + 2] = q_valid ? fp8_lut[(x >> 16) & 0xFFu] : (_Float16)0.0f;
        qf[w4 * 4 + 3] = q_valid ? fp8_lut[(x >> 24) & 0xFFu] : (_Float16)0.0f;
      }
      q_reg[pair * 2 + s] = qf;
    }
    __builtin_amdgcn_sched_barrier(0);
  }

  const uint32_t scale_byte_off = (uint32_t)(block_size * 128);
  const int kr = tid >> 3;
  const int ht = tid & 7;
  const int kc16 = ht * 16;
  const uint32_t shuf_lane_off = (uint32_t)(ht * 256 + kr * 16);
  const int stride_kv = n_splits * 16;
  const int blk_mask = block_size - 1;

  auto fetch_k_tile = [&](int kv_st, u4v& out_raw, float& out_sc) {
    const int logical_blk = kv_st >> blk_shift;
    const int64_t page = (int64_t)block_tables[(int64_t)b * bt_s0 + logical_blk];
    const uint32_t base_in_blk = (uint32_t)(kv_st & blk_mask);
    const uint8_t* blk_ptr = kv_flat + page * kv_s0;
    if (kv_st + kr < max_sl) {
      const uint32_t k_off = IS_SHUFFLE
          ? (base_in_blk * 128u + shuf_lane_off)
          : ((base_in_blk + (uint32_t)kr) * 128u + (uint32_t)kc16);
      out_raw = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(blk_ptr + k_off));
    } else {
      out_raw = (u4v){0u, 0u, 0u, 0u};
    }
    if (tid < 16 && (kv_st + tid) < max_sl) {
      out_sc = *reinterpret_cast<const float*>(blk_ptr + scale_byte_off + (base_in_blk + (uint32_t)tid) * 4u);
    } else {
      out_sc = 0.0f;
    }
  };

  int kv_start = split * 16;
  if ((kv_start >> blk_shift) >= max_blocks) return;
  u4v raw_cur = {0u, 0u, 0u, 0u};
  float sc_cur = 0.0f;
  fetch_k_tile(kv_start, raw_cur, sc_cur);

  const int n_out0 = n_base + wave * 2 + 0;
  const int n_out1 = n_base + wave * 2 + 1;
  const int sl0 = s_sl[wave * 2 + 0];
  const int sl1 = s_sl[wave * 2 + 1];

  for (; kv_start < max_sl && (kv_start >> blk_shift) < max_blocks; kv_start += stride_kv) {
    if (tid < 16) s_sc[tid] = sc_cur;
    if (kv_start + kr < max_sl) {
      const uint32_t ws[4] = {raw_cur.x, raw_cur.y, raw_cur.z, raw_cur.w};
      uint32_t pk[8];
#pragma unroll
      for (int w4 = 0; w4 < 4; ++w4) {
        const uint32_t x = ws[w4];
        const uint32_t h0 = (uint32_t)__builtin_bit_cast(uint16_t, fp8_lut[x & 0xFFu]);
        const uint32_t h1 = (uint32_t)__builtin_bit_cast(uint16_t, fp8_lut[(x >> 8) & 0xFFu]);
        const uint32_t h2 = (uint32_t)__builtin_bit_cast(uint16_t, fp8_lut[(x >> 16) & 0xFFu]);
        const uint32_t h3 = (uint32_t)__builtin_bit_cast(uint16_t, fp8_lut[(x >> 24) & 0xFFu]);
        pk[w4 * 2 + 0] = h0 | (h1 << 16);
        pk[w4 * 2 + 1] = h2 | (h3 << 16);
      }
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16 + 0]) = (u4v){pk[0], pk[1], pk[2], pk[3]};
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16 + 8]) = (u4v){pk[4], pk[5], pk[6], pk[7]};
    } else {
      const u4v z = {0u, 0u, 0u, 0u};
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16]) = z;
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16 + 8]) = z;
    }
    __syncthreads();

    const int kv_next = kv_start + stride_kv;
    u4v raw_nxt = {0u, 0u, 0u, 0u};
    float sc_nxt = 0.0f;
    if (kv_next < max_sl && (kv_next >> blk_shift) < max_blocks) {
      fetch_k_tile(kv_next, raw_nxt, sc_nxt);
    }
    __builtin_amdgcn_sched_barrier(0);

    f32x8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int step = 0; step < 8; ++step) {
      const int d = step * 16;
      const u4v k0 = *reinterpret_cast<const u4va*>(&K_lds[l16][d]);
      const u4v k1 = *reinterpret_cast<const u4va*>(&K_lds[l16][d + 8]);
      const f16x16 kf = __builtin_bit_cast(f16x16, (u8v){k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w});
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(q_reg[step], kf, acc);
    }

    float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
    for (int ei = 0; ei < 4; ++ei) {
      sum0 = __builtin_fmaf(fmaxf(acc[ei], 0.0f), s_weights[wave * 2 + 0][2 * ei + hrow], sum0);
      sum1 = __builtin_fmaf(fmaxf(acc[ei + 4], 0.0f), s_weights[wave * 2 + 1][2 * ei + hrow], sum1);
    }
    sum0 += shx_f<16>(sum0);
    sum1 += shx_f<16>(sum1);
    if (hrow == 0) {
      const int pos = kv_start + l16;
      const float ks = s_sc[l16];
      if (n_out0 < next_n && pos < sl0) {
        out_logits[(int64_t)(b * next_n + n_out0) * out_s0 + pos] = sum0 * ks;
      }
      if (n_out1 < next_n && pos < sl1) {
        out_logits[(int64_t)(b * next_n + n_out1) * out_s0 + pos] = sum1 * ks;
      }
    }
    __syncthreads();
    raw_cur = raw_nxt;
    sc_cur = sc_nxt;
  }
}

void ernie_fp8_paged_mqa_logits_decode(
    torch::Tensor q_fp8, torch::Tensor kv_cache_fp8, torch::Tensor weights,
    torch::Tensor context_lens, torch::Tensor block_tables,
    torch::Tensor out_logits, int64_t max_model_len, bool is_shuffle) {
  TORCH_CHECK(q_fp8.is_cuda() && q_fp8.dim() == 4 && (q_fp8.size(2) == 64 || q_fp8.size(2) == 8) && q_fp8.size(3) == 128);
  const int B = (int)q_fp8.size(0);
  const int next_n = (int)q_fp8.size(1);
  const int rows = B * next_n;
  if (rows == 0 || max_model_len == 0) return;
  const int block_size = (int)kv_cache_fp8.size(1);
  TORCH_CHECK(block_size % 16 == 0);
  auto kv_flat = kv_cache_fp8.reshape({kv_cache_fp8.size(0), -1});
  auto cl = context_lens.reshape({-1}).contiguous();
  const int ctx_per_row = !(next_n > 1 && cl.numel() == B) ? 1 : 0;
  const int is_fnuz = (q_fp8.scalar_type() == torch::kFloat8_e4m3fnuz) ? 1 : 0;
  const int tiles_cap = ((int)max_model_len + 15) / 16;
  const int n_splits = max(1, min(40, min(tiles_cap, max(1, 512 / rows))));
  auto st = c10::hip::getCurrentHIPStream().stream();
  if (q_fp8.size(2) == 8 && (block_size & (block_size - 1)) == 0) {
    int blk_shift = 0;
    while ((1 << blk_shift) < block_size) ++blk_shift;
    const int n_chunks = (next_n + 7) / 8;
    const int g_x = B * n_chunks;
    const int n_sp = max(1, min(160, min(tiles_cap, max(40, 640 / g_x))));
    if (is_shuffle) {
      hipLaunchKernelGGL(
          (k_ernie_fp8_paged_mqa_logits_decode_h8<1>), dim3(B, n_sp, n_chunks), dim3(128), 0, st,
          reinterpret_cast<const uint8_t*>(q_fp8.data_ptr()),
          kv_flat.data_ptr<uint8_t>(), weights.data_ptr<float>(),
          cl.data_ptr<int32_t>(), block_tables.data_ptr<int32_t>(),
          out_logits.data_ptr<float>(),
          q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2),
          kv_flat.stride(0), weights.stride(0), block_tables.stride(0), out_logits.stride(0),
          block_size, blk_shift, (int)block_tables.size(1), (int)max_model_len,
          next_n, n_sp, ctx_per_row, is_fnuz);
    } else {
      hipLaunchKernelGGL(
          (k_ernie_fp8_paged_mqa_logits_decode_h8<0>), dim3(B, n_sp, n_chunks), dim3(128), 0, st,
          reinterpret_cast<const uint8_t*>(q_fp8.data_ptr()),
          kv_flat.data_ptr<uint8_t>(), weights.data_ptr<float>(),
          cl.data_ptr<int32_t>(), block_tables.data_ptr<int32_t>(),
          out_logits.data_ptr<float>(),
          q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2),
          kv_flat.stride(0), weights.stride(0), block_tables.stride(0), out_logits.stride(0),
          block_size, blk_shift, (int)block_tables.size(1), (int)max_model_len,
          next_n, n_sp, ctx_per_row, is_fnuz);
    }
    return;
  }
  if (is_shuffle) {
    hipLaunchKernelGGL(
        (k_ernie_fp8_paged_mqa_logits_decode<1>), dim3(rows, n_splits), dim3(128), 0, st,
        reinterpret_cast<const uint8_t*>(q_fp8.data_ptr()),
        kv_flat.data_ptr<uint8_t>(), weights.data_ptr<float>(),
        cl.data_ptr<int32_t>(), block_tables.data_ptr<int32_t>(),
        out_logits.data_ptr<float>(),
        q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2),
        kv_flat.stride(0), weights.stride(0), block_tables.stride(0), out_logits.stride(0),
        block_size, (int)block_tables.size(1), (int)max_model_len,
        next_n, n_splits, ctx_per_row, is_fnuz);
  } else {
    hipLaunchKernelGGL(
        (k_ernie_fp8_paged_mqa_logits_decode<0>), dim3(rows, n_splits), dim3(128), 0, st,
        reinterpret_cast<const uint8_t*>(q_fp8.data_ptr()),
        kv_flat.data_ptr<uint8_t>(), weights.data_ptr<float>(),
        cl.data_ptr<int32_t>(), block_tables.data_ptr<int32_t>(),
        out_logits.data_ptr<float>(),
        q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2),
        kv_flat.stride(0), weights.stride(0), block_tables.stride(0), out_logits.stride(0),
        block_size, (int)block_tables.size(1), (int)max_model_len,
        next_n, n_splits, ctx_per_row, is_fnuz);
  }
}


// ================================================================== 4: Quantized DSA Sparse MLA (FP8 656B & NVFP4 352B)
__device__ __forceinline__ float dsa_fp8_ocp_to_f32(uint32_t b) {
  const uint32_t sign = (b & 0x80u) << 24;
  const uint32_t exp = (b >> 3) & 0xFu;
  const uint32_t man = b & 0x7u;
  if (exp == 0u) {
    float v = (float)man * (1.0f / 512.0f);
    return sign ? -v : v;
  }
  if (exp == 15u && man == 7u) return NAN;
  return __uint_as_float(sign | ((exp + 120u) << 23) | (man << 20));
}

// KV_FMT: 1 = fp8_ds_mla (656 B/tok), 2 = nvfp4_ds_mla (352 B/tok)
template <int KV_FMT, int NUM_SPLITS, int NT_KV>
__global__ __launch_bounds__(128) void k_ernie_dsa_sparse_mla_quant(
    const uint16_t* __restrict__ q_ptr,      // [T, 64, 576] bf16
    const uint8_t* __restrict__ kv_ptr,      // [num_blocks, block_size, ROW_BYTES] uint8
    const int32_t* __restrict__ idx_ptr,     // [T, W] int32 (phys_blk * block_size + in_blk, -1=invalid)
    const float* __restrict__ sink_ptr,      // [64] fp32 (or nullptr)
    uint16_t* __restrict__ out_ptr,          // [T, 64, 512] bf16 (NUM_SPLITS == 1)
    float* __restrict__ part_o,              // [T, 64, NUM_SPLITS, 512] fp32
    float* __restrict__ part_m,              // [T, 64, NUM_SPLITS] fp32
    float* __restrict__ part_l,              // [T, 64, NUM_SPLITS] fp32
    int64_t q_s0, int64_t q_s1, int64_t kv_blk_stride, int64_t kv_row_stride,
    int64_t idx_s0, int64_t out_s0, int64_t out_s1,
    int num_blocks, int block_size, int W, float scale) {
  __shared__ __attribute__((aligned(16))) uint16_t KV_lds[16][DSA_STRIDE]; // 18.25 KiB
  __shared__ float S_part[4][16][16];                                       // 4.00 KiB
  __shared__ uint16_t P_lds[16][16];                                        // 0.50 KiB
  __shared__ float fp8_lut[256];                                            // 1.00 KiB
  __shared__ float e2m1_lut[16];                                            // 0.06 KiB
  __shared__ int32_t s_slots[16];
  __shared__ uint8_t s_valid[16];
  __shared__ float s_alpha[16];
  __shared__ float s_m[16];
  __shared__ float s_l[16];

  const int hg = blockIdx.x;
  const int t = blockIdx.y;
  const int sp = blockIdx.z;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;
  const int h0 = hg * 16;

  fp8_lut[tid] = dsa_fp8_ocp_to_f32((uint32_t)tid);
  fp8_lut[tid + 128] = dsa_fp8_ocp_to_f32((uint32_t)(tid + 128));
  if (tid < 16) {
    static constexpr float E2M1_C[16] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    e2m1_lut[tid] = E2M1_C[tid];
    s_m[tid] = NEG_INF;
    s_l[tid] = 0.0f;
  }
  __syncthreads();

  f32x8 o_acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) o_acc[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  const int32_t* idx_row = idx_ptr + (int64_t)t * idx_s0;
  const uint16_t* q_base = q_ptr + (int64_t)t * q_s0 + (int64_t)(h0 + l16) * q_s1;
  bf16x16 q_reg[9];
  const int d_start_q = wave * 144;
#pragma unroll
  for (int step = 0; step < 9; ++step) {
    const int d = d_start_q + step * 16;
    const u4v q0 = *reinterpret_cast<const u4va*>(q_base + d);
    const u4v q1 = *reinterpret_cast<const u4va*>(q_base + d + 8);
    q_reg[step] = __builtin_bit_cast(bf16x16, (u8v){q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w});
  }
  const int total_slots = num_blocks * block_size;
  const int kr = tid >> 3;   // 0..15
  const int sub = tid & 7;   // 0..7

  const int total_blks = (W + 15) / 16;
  const int blks_per_sp = (total_blks + NUM_SPLITS - 1) / NUM_SPLITS;
  const int blk_start = sp * blks_per_sp;
  const int blk_end = min(total_blks, blk_start + blks_per_sp);

  for (int blk = blk_start; blk < blk_end; ++blk) {
    const int k_base = blk * 16;
    if (tid < 16) {
      const int c = k_base + tid;
      int32_t sl = (c < W) ? idx_row[c] : -1;
      if (sl < 0 || sl >= total_slots) sl = -1;
      s_slots[tid] = sl;
      s_valid[tid] = (sl >= 0) ? 1 : 0;
    }
    __syncthreads();

    unsigned any_v = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) any_v |= s_valid[i];
    if (!any_v) continue;

    // 1. Cooperatively dequant 16 quantized KV rows -> BF16 KV_lds[16][584]
    const int32_t sl = s_slots[kr];
    if (sl >= 0) {
      const int phys_blk = (block_size == 64) ? (sl >> 6) : (sl / block_size);
      const int in_blk   = (block_size == 64) ? (sl & 63) : (sl % block_size);
      const uint8_t* row_p = kv_ptr + (int64_t)phys_blk * kv_blk_stride + (int64_t)in_blk * kv_row_stride;

      if constexpr (KV_FMT == 1) {
        const float sc = *reinterpret_cast<const float*>(row_p + 512 + (sub >> 1) * 4);
        const uint8_t* nope_p = row_p + sub * 64;
#pragma unroll 1
        for (int c4 = 0; c4 < 4; ++c4) {
          const u4va* src = reinterpret_cast<const u4va*>(nope_p + c4 * 16);
          const u4v raw = NT_KV ? __builtin_nontemporal_load(src) : *src;
          const uint32_t ws[4] = {raw.x, raw.y, raw.z, raw.w};
          uint32_t p0[4], p1[4];
#pragma unroll
          for (int w = 0; w < 2; ++w) {
            const uint32_t x = ws[w];
            p0[w * 2 + 0] = f32_to_bf16x2_rne(fp8_lut[x & 0xFFu] * sc, fp8_lut[(x >> 8) & 0xFFu] * sc);
            p0[w * 2 + 1] = f32_to_bf16x2_rne(fp8_lut[(x >> 16) & 0xFFu] * sc, fp8_lut[(x >> 24) & 0xFFu] * sc);
          }
#pragma unroll
          for (int w = 0; w < 2; ++w) {
            const uint32_t x = ws[w + 2];
            p1[w * 2 + 0] = f32_to_bf16x2_rne(fp8_lut[x & 0xFFu] * sc, fp8_lut[(x >> 8) & 0xFFu] * sc);
            p1[w * 2 + 1] = f32_to_bf16x2_rne(fp8_lut[(x >> 16) & 0xFFu] * sc, fp8_lut[(x >> 24) & 0xFFu] * sc);
          }
          *reinterpret_cast<u4va*>(&KV_lds[kr][sub * 64 + c4 * 16 + 0]) = (u4v){p0[0], p0[1], p0[2], p0[3]};
          *reinterpret_cast<u4va*>(&KV_lds[kr][sub * 64 + c4 * 16 + 8]) = (u4v){p1[0], p1[1], p1[2], p1[3]};
        }
        const u4va* rsrc = reinterpret_cast<const u4va*>(row_p + 528 + sub * 16);
        *reinterpret_cast<u4va*>(&KV_lds[kr][512 + sub * 8]) = NT_KV ? __builtin_nontemporal_load(rsrc) : *rsrc;
      } else {
        const uint32_t sc4 = *reinterpret_cast<const uint32_t*>(row_p + 320 + sub * 4);
        const uint8_t* fp4_p = row_p + sub * 32;
#pragma unroll 1
        for (int mb = 0; mb < 4; ++mb) {
          const float sc = fp8_lut[(sc4 >> (mb * 8)) & 0xFFu];
          const uint2 raw = *reinterpret_cast<const uint2*>(fp4_p + mb * 8);
          uint32_t p0[4], p1[4];
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            const uint32_t byte0 = (raw.x >> (b * 8)) & 0xFFu;
            const uint32_t byte1 = (raw.y >> (b * 8)) & 0xFFu;
            p0[b] = f32_to_bf16x2_rne(e2m1_lut[byte0 & 0xFu] * sc, e2m1_lut[byte0 >> 4] * sc);
            p1[b] = f32_to_bf16x2_rne(e2m1_lut[byte1 & 0xFu] * sc, e2m1_lut[byte1 >> 4] * sc);
          }
          *reinterpret_cast<u4va*>(&KV_lds[kr][sub * 64 + mb * 16 + 0]) = (u4v){p0[0], p0[1], p0[2], p0[3]};
          *reinterpret_cast<u4va*>(&KV_lds[kr][sub * 64 + mb * 16 + 8]) = (u4v){p1[0], p1[1], p1[2], p1[3]};
        }
        const uint2 rraw = *reinterpret_cast<const uint2*>(row_p + 256 + sub * 8);
        u4v rp;
        rp.x = f32_to_bf16x2_rne(fp8_lut[rraw.x & 0xFFu], fp8_lut[(rraw.x >> 8) & 0xFFu]);
        rp.y = f32_to_bf16x2_rne(fp8_lut[(rraw.x >> 16) & 0xFFu], fp8_lut[(rraw.x >> 24) & 0xFFu]);
        rp.z = f32_to_bf16x2_rne(fp8_lut[rraw.y & 0xFFu], fp8_lut[(rraw.y >> 8) & 0xFFu]);
        rp.w = f32_to_bf16x2_rne(fp8_lut[(rraw.y >> 16) & 0xFFu], fp8_lut[(rraw.y >> 24) & 0xFFu]);
        *reinterpret_cast<u4va*>(&KV_lds[kr][512 + sub * 8]) = rp;
      }
    } else {
      const u4v z = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        *reinterpret_cast<u4va*>(&KV_lds[kr][sub * 64 + c8 * 8]) = z;
      }
      *reinterpret_cast<u4va*>(&KV_lds[kr][512 + sub * 8]) = z;
    }
    __syncthreads();

    // 2. Q @ K^T (Q resident in q_reg[9]!)
    f32x8 s_acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const int d_start = wave * 144;
#pragma unroll
    for (int step = 0; step < 9; ++step) {
      const int d = d_start + step * 16;
      const u4v k0 = *reinterpret_cast<const u4va*>(&KV_lds[l16][d]);
      const u4v k1 = *reinterpret_cast<const u4va*>(&KV_lds[l16][d + 8]);
      const bf16x16 kf = __builtin_bit_cast(bf16x16, (u8v){k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w});
      s_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(q_reg[step], kf, s_acc);
    }
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      S_part[wave][2 * ei + hrow][l16] = s_acc[ei];
    }
    __syncthreads();

    // 3. 4-way parallel S_part reduction + online softmax across all 4 waves (4 heads/wave!)
    {
      const bool vld = (s_valid[l16] != 0);
#pragma unroll
      for (int ei = 0; ei < 2; ++ei) {
        const int hm = wave * 4 + 2 * ei + hrow;
        float sc = S_part[0][hm][l16] + S_part[1][hm][l16] + S_part[2][hm][l16] + S_part[3][hm][l16];
        sc = vld ? (sc * scale) : NEG_INF;
        const float bmax = halfwave_max16(sc);
        const float mold = s_m[hm];
        const float mnew = fmaxf(mold, bmax);
        const float alpha = (mold > -1e30f) ? exp2f((mold - mnew) * LOG2E) : 0.0f;
        const float pv = (vld && mnew > -1e30f) ? exp2f((sc - mnew) * LOG2E) : 0.0f;
        const float bsum = halfwave_sum16(pv);
        P_lds[hm][l16] = f32_to_bf16_rne(pv);
        if (l16 == 0) {
          s_alpha[hm] = alpha;
          s_m[hm] = mnew;
          s_l[hm] = s_l[hm] * alpha + bsum;
        }
      }
    }
    __syncthreads();

    // 4. Rescale o_acc & P @ V
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      const float a_ei = s_alpha[2 * ei + hrow];
#pragma unroll
      for (int j = 0; j < 8; ++j) o_acc[j][ei] *= a_ei;
    }

    const u4v p0 = *reinterpret_cast<const u4va*>(&P_lds[l16][0]);
    const u4v p1 = *reinterpret_cast<const u4va*>(&P_lds[l16][8]);
    const bf16x16 pf = __builtin_bit_cast(bf16x16, (u8v){p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w});
    const int v_col0 = wave * 128;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int vc = v_col0 + j * 16 + l16;
      bf16x16 vf;
#pragma unroll
      for (int kr_i = 0; kr_i < 16; ++kr_i) vf[kr_i] = (short)KV_lds[kr_i][vc];
      o_acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf, vf, o_acc[j]);
      __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();
  }

  // 5. Epilogue
  const int v_col0 = wave * 128;
  if constexpr (NUM_SPLITS == 1) {
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      const int hm = 2 * ei + hrow;
      float m = s_m[hm];
      float l = s_l[hm];
      float inv_l = 0.0f;
      if (sink_ptr != nullptr) {
        const float sk = sink_ptr[h0 + hm];
        const float mf = fmaxf(m, sk);
        const float lf = (l > 0.0f ? l * exp2f((m - mf) * LOG2E) : 0.0f) + exp2f((sk - mf) * LOG2E);
        const float w_acc = (l > 0.0f) ? exp2f((m - mf) * LOG2E) : 0.0f;
        inv_l = (lf > 0.0f) ? (w_acc * __builtin_amdgcn_rcpf(lf)) : 0.0f;
      } else {
        inv_l = (l > 0.0f) ? __builtin_amdgcn_rcpf(l) : 0.0f;
      }
      uint16_t* dst = out_ptr + (int64_t)t * out_s0 + (int64_t)(h0 + hm) * out_s1 + v_col0 + l16;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        dst[j * 16] = f32_to_bf16_rne(o_acc[j][ei] * inv_l);
      }
    }
  } else {
    if (wave == 0 && l16 == 0) {
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) {
        const int hm = 2 * ei + hrow;
        const int64_t off = ((int64_t)t * 64 + (h0 + hm)) * NUM_SPLITS + sp;
        part_m[off] = s_m[hm];
        part_l[off] = s_l[hm];
      }
    }
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      const int hm = 2 * ei + hrow;
      float* dst = part_o + (((int64_t)t * 64 + (h0 + hm)) * NUM_SPLITS + sp) * DSA_NOPE + v_col0 + l16;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        dst[j * 16] = o_acc[j][ei];
      }
    }
  }
}

// Quantized All-64-Heads 8-Wave DSA Prefill (KV_FMT: 1=fp8_ds_mla 656B, 2=nvfp4_ds_mla 352B)
template <int KV_FMT>
__global__ __launch_bounds__(256) void k_ernie_dsa_sparse_mla_quant_prefill_v28(
    const uint16_t* __restrict__ q_ptr,      // [T, 64, 576] bf16
    const uint8_t* __restrict__ kv_ptr,      // [num_blocks, block_size, ROW_BYTES] uint8
    const int32_t* __restrict__ idx_ptr,     // [T, W] int32
    const float* __restrict__ sink_ptr,      // [64] fp32 (or nullptr)
    uint16_t* __restrict__ out_ptr,          // [T, 64, 512] bf16
    int64_t q_s0, int64_t q_s1, int64_t kv_blk_stride, int64_t kv_row_stride,
    int64_t idx_s0, int64_t out_s0, int64_t out_s1,
    int num_blocks, int block_size, int W, float scale) {
  __shared__ __attribute__((aligned(16))) uint16_t KV_lds[16][DSA_STRIDE];         // 18,688 B
  __shared__ __attribute__((aligned(16))) uint16_t VT_lds_g[16][32 * DSA_VT_STRIDE + 2]; // 18,496 B (zero 16-way bank conflicts!)
  __shared__ float S_part[4][16][16];                                               //  4,096 B
  __shared__ __attribute__((aligned(16))) uint16_t P_lds[4][16][16];                //  2,048 B
  __shared__ float fp8_lut[256];                                                    //  1,024 B
  __shared__ float e2m1_lut[16];                                                    //     64 B
  __shared__ int32_t s_slots[16];
  __shared__ uint8_t s_valid[16];
  __shared__ float s_alpha[64];
  __shared__ float s_m[64];
  __shared__ float s_l[64];

  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int wh = wave >> 1;
  const int wd = wave & 1;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;
  const int h0 = wh * 16;

  fp8_lut[tid] = dsa_fp8_ocp_to_f32((uint32_t)tid);
  if (tid < 16) {
    static constexpr float E2M1_C[16] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    e2m1_lut[tid] = E2M1_C[tid];
  }
  if (tid < 64) {
    s_m[tid] = NEG_INF;
    s_l[tid] = 0.0f;
  }
  __syncthreads();

  f32x8 o_acc[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) o_acc[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  const int32_t* idx_row = idx_ptr + (int64_t)t * idx_s0;
  const uint16_t* q_base = q_ptr + (int64_t)t * q_s0 + (int64_t)(h0 + l16) * q_s1;
  const int kd_start = wd * 288;
  const int vd_start = wd * 256;
  const int total_slots = num_blocks * block_size;
  const int kr_ld = tid >> 4;   // 0..15 (16 tokens)
  const int sub16 = tid & 15;   // 0..15 (32 NOPE dims/thread!)
  const int total_blks = (W + 15) >> 4;

#pragma unroll 1
  for (int blk = 0; blk < total_blks; ++blk) {
    const int k_base = blk * 16;
    if (tid < 16) {
      const int c = k_base + tid;
      int32_t sl = (c < W) ? idx_row[c] : -1;
      if (sl < 0 || sl >= total_slots) sl = -1;
      s_slots[tid] = sl;
      s_valid[tid] = (sl >= 0) ? 1 : 0;
    }
    __syncthreads();
    unsigned any_v = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) any_v |= s_valid[i];
    if (!any_v) continue;

    // 1. Dequant 16 quantized KV rows -> BOTH KV_lds & VT_lds (4x fewer dequants vs 4-WG!)
    const int32_t sl = s_slots[kr_ld];
    if (sl >= 0) {
      const int phys_blk = (block_size == 64) ? (sl >> 6) : (sl / block_size);
      const int in_blk   = (block_size == 64) ? (sl & 63) : (sl % block_size);
      const uint8_t* row_p = kv_ptr + (int64_t)phys_blk * kv_blk_stride + (int64_t)in_blk * kv_row_stride;

      if constexpr (KV_FMT == 1) {
        const float sc = *reinterpret_cast<const float*>(row_p + 512 + (sub16 >> 2) * 4);
        const uint8_t* nope_p = row_p + sub16 * 32;
#pragma unroll
        for (int c2 = 0; c2 < 2; ++c2) {
          const u4v raw = *reinterpret_cast<const u4va*>(nope_p + c2 * 16);
          const uint32_t ws[4] = {raw.x, raw.y, raw.z, raw.w};
          uint32_t pk[8];
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            const uint32_t x = ws[w];
            pk[w * 2 + 0] = f32_to_bf16x2_fast(fp8_lut[x & 0xFFu] * sc, fp8_lut[(x >> 8) & 0xFFu] * sc);
            pk[w * 2 + 1] = f32_to_bf16x2_fast(fp8_lut[(x >> 16) & 0xFFu] * sc, fp8_lut[(x >> 24) & 0xFFu] * sc);
          }
          const int col0 = sub16 * 32 + c2 * 16;
          *reinterpret_cast<u4va*>(&KV_lds[kr_ld][col0 + 0]) = (u4v){pk[0], pk[1], pk[2], pk[3]};
          *reinterpret_cast<u4va*>(&KV_lds[kr_ld][col0 + 8]) = (u4v){pk[4], pk[5], pk[6], pk[7]};
          uint16_t* vt_dst = &VT_lds_g[sub16][c2 * 16 * DSA_VT_STRIDE + kr_ld];
#pragma unroll
          for (int p = 0; p < 8; ++p) {
            vt_dst[(p * 2 + 0) * DSA_VT_STRIDE] = (uint16_t)(pk[p] & 0xFFFFu);
            vt_dst[(p * 2 + 1) * DSA_VT_STRIDE] = (uint16_t)(pk[p] >> 16);
          }
        }
        if (sub16 < 8) {
          *reinterpret_cast<u4va*>(&KV_lds[kr_ld][512 + sub16 * 8]) =
              *reinterpret_cast<const u4va*>(row_p + 528 + sub16 * 16);
        }
      } else {
        const uint16_t sc2 = *reinterpret_cast<const uint16_t*>(row_p + 320 + sub16 * 2);
        const uint8_t* fp4_p = row_p + sub16 * 16;
#pragma unroll
        for (int mb = 0; mb < 2; ++mb) {
          const float sc = fp8_lut[(sc2 >> (mb * 8)) & 0xFFu];
          const uint2 raw = *reinterpret_cast<const uint2*>(fp4_p + mb * 8);
          uint32_t pk[8];
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            const uint32_t byte0 = (raw.x >> (b * 8)) & 0xFFu;
            const uint32_t byte1 = (raw.y >> (b * 8)) & 0xFFu;
            pk[b]     = f32_to_bf16x2_fast(e2m1_reg_to_f32(byte0 & 0xFu) * sc, e2m1_reg_to_f32(byte0 >> 4) * sc);
            pk[b + 4] = f32_to_bf16x2_fast(e2m1_reg_to_f32(byte1 & 0xFu) * sc, e2m1_reg_to_f32(byte1 >> 4) * sc);
          }
          const int col0 = sub16 * 32 + mb * 16;
          *reinterpret_cast<u4va*>(&KV_lds[kr_ld][col0 + 0]) = (u4v){pk[0], pk[1], pk[2], pk[3]};
          *reinterpret_cast<u4va*>(&KV_lds[kr_ld][col0 + 8]) = (u4v){pk[4], pk[5], pk[6], pk[7]};
          uint16_t* vt_dst = &VT_lds_g[sub16][mb * 16 * DSA_VT_STRIDE + kr_ld];
#pragma unroll
          for (int p = 0; p < 8; ++p) {
            vt_dst[(p * 2 + 0) * DSA_VT_STRIDE] = (uint16_t)(pk[p] & 0xFFFFu);
            vt_dst[(p * 2 + 1) * DSA_VT_STRIDE] = (uint16_t)(pk[p] >> 16);
          }
        }
        if (sub16 < 8) {
          const uint2 rraw = *reinterpret_cast<const uint2*>(row_p + 256 + sub16 * 8);
          u4v rp;
          rp.x = f32_to_bf16x2_fast(fp8_lut[rraw.x & 0xFFu], fp8_lut[(rraw.x >> 8) & 0xFFu]);
          rp.y = f32_to_bf16x2_fast(fp8_lut[(rraw.x >> 16) & 0xFFu], fp8_lut[(rraw.x >> 24) & 0xFFu]);
          rp.z = f32_to_bf16x2_fast(fp8_lut[rraw.y & 0xFFu], fp8_lut[(rraw.y >> 8) & 0xFFu]);
          rp.w = f32_to_bf16x2_fast(fp8_lut[(rraw.y >> 16) & 0xFFu], fp8_lut[(rraw.y >> 24) & 0xFFu]);
          *reinterpret_cast<u4va*>(&KV_lds[kr_ld][512 + sub16 * 8]) = rp;
        }
      }
    } else {
      const u4v z = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int c4 = 0; c4 < 4; ++c4) {
        *reinterpret_cast<u4va*>(&KV_lds[kr_ld][sub16 * 32 + c4 * 8]) = z;
      }
#pragma unroll
      for (int e = 0; e < 32; ++e) VT_lds_g[sub16][e * DSA_VT_STRIDE + kr_ld] = 0u;
      if (sub16 < 8) *reinterpret_cast<u4va*>(&KV_lds[kr_ld][512 + sub16 * 8]) = z;
    }
    __syncthreads();

    // 2. Q @ K^T: 18 WMMA steps, 1-ahead prefetch, ZERO sched_barrier!
    f32x8 s_acc = {0, 0, 0, 0, 0, 0, 0, 0};
    u4v q0_c = *reinterpret_cast<const u4va*>(q_base + kd_start);
    u4v q1_c = *reinterpret_cast<const u4va*>(q_base + kd_start + 8);
    u4v k0_c = *reinterpret_cast<const u4va*>(&KV_lds[l16][kd_start]);
    u4v k1_c = *reinterpret_cast<const u4va*>(&KV_lds[l16][kd_start + 8]);
#pragma unroll 1
    for (int step = 0; step < 18; ++step) {
      u4v q0_n = {0u, 0u, 0u, 0u}, q1_n = {0u, 0u, 0u, 0u};
      u4v k0_n = {0u, 0u, 0u, 0u}, k1_n = {0u, 0u, 0u, 0u};
      if (step + 1 < 18) {
        const int dn = kd_start + (step + 1) * 16;
        q0_n = *reinterpret_cast<const u4va*>(q_base + dn);
        q1_n = *reinterpret_cast<const u4va*>(q_base + dn + 8);
        k0_n = *reinterpret_cast<const u4va*>(&KV_lds[l16][dn]);
        k1_n = *reinterpret_cast<const u4va*>(&KV_lds[l16][dn + 8]);
      }
      const bf16x16 qf = __builtin_bit_cast(bf16x16, (u8v){q0_c.x, q0_c.y, q0_c.z, q0_c.w, q1_c.x, q1_c.y, q1_c.z, q1_c.w});
      const bf16x16 kf = __builtin_bit_cast(bf16x16, (u8v){k0_c.x, k0_c.y, k0_c.z, k0_c.w, k1_c.x, k1_c.y, k1_c.z, k1_c.w});
      s_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(qf, kf, s_acc);
      q0_c = q0_n; q1_c = q1_n; k0_c = k0_n; k1_c = k1_n;
    }

    if (wd == 1) {
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) S_part[wh][2 * ei + hrow][l16] = s_acc[ei];
    }
    __syncthreads();

    // 3. 4x-Parallel Online Softmax
    if (wd == 0) {
      const bool vld = (s_valid[l16] != 0);
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) {
        const int hm = 2 * ei + hrow;
        const int h_loc = wh * 16 + hm;
        float sc = s_acc[ei] + S_part[wh][hm][l16];
        sc = vld ? (sc * scale) : NEG_INF;
        const float bmax = halfwave_max16(sc);
        const float mold = s_m[h_loc];
        const float mnew = fmaxf(mold, bmax);
        const float alpha = (mold > -1e30f) ? exp2f((mold - mnew) * LOG2E) : 0.0f;
        const float pv = (vld && mnew > -1e30f) ? exp2f((sc - mnew) * LOG2E) : 0.0f;
        const float bsum = halfwave_sum16(pv);
        P_lds[wh][hm][l16] = f32_to_bf16_rne(pv);
        if (l16 == 0) {
          s_alpha[h_loc] = alpha;
          s_m[h_loc] = mnew;
          s_l[h_loc] = s_l[h_loc] * alpha + bsum;
        }
      }
    }
    __syncthreads();

    // 4. Rescale o_acc[0..15] & P @ V via 128-bit VT_lds loads
    float al[8];
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) al[ei] = s_alpha[wh * 16 + 2 * ei + hrow];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) o_acc[j][ei] *= al[ei];
    }

    const u4v p0 = *reinterpret_cast<const u4va*>(&P_lds[wh][l16][0]);
    const u4v p1 = *reinterpret_cast<const u4va*>(&P_lds[wh][l16][8]);
    const bf16x16 pf = __builtin_bit_cast(bf16x16, (u8v){p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w});

    const int vg0 = vd_start >> 5;
    u4v v0_c = *reinterpret_cast<const u4va*>(&VT_lds_g[vg0][l16 * DSA_VT_STRIDE + 0]);
    u4v v1_c = *reinterpret_cast<const u4va*>(&VT_lds_g[vg0][l16 * DSA_VT_STRIDE + 8]);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      u4v v0_n = {0u, 0u, 0u, 0u}, v1_n = {0u, 0u, 0u, 0u};
      if (j + 1 < 16) {
        const int vc_n = vd_start + (j + 1) * 16 + l16;
        const int vgn = vc_n >> 5;
        const int vln = vc_n & 31;
        v0_n = *reinterpret_cast<const u4va*>(&VT_lds_g[vgn][vln * DSA_VT_STRIDE + 0]);
        v1_n = *reinterpret_cast<const u4va*>(&VT_lds_g[vgn][vln * DSA_VT_STRIDE + 8]);
      }
      const bf16x16 vf = __builtin_bit_cast(bf16x16, (u8v){v0_c.x, v0_c.y, v0_c.z, v0_c.w, v1_c.x, v1_c.y, v1_c.z, v1_c.w});
      o_acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf, vf, o_acc[j]);
      v0_c = v0_n; v1_c = v1_n;
    }
    __syncthreads();
  }

  // 5. Epilogue
  if (tid < 64) {
    float m = s_m[tid];
    float l = s_l[tid];
    float inv_l = 0.0f;
    if (sink_ptr != nullptr) {
      const float sk = sink_ptr[tid];
      const float mf = fmaxf(m, sk);
      const float lf = (l > 0.0f ? l * exp2f((m - mf) * LOG2E) : 0.0f) + exp2f((sk - mf) * LOG2E);
      const float w_acc = (l > 0.0f) ? exp2f((m - mf) * LOG2E) : 0.0f;
      inv_l = (lf > 0.0f) ? (w_acc * __builtin_amdgcn_rcpf(lf)) : 0.0f;
    } else {
      inv_l = (l > 0.0f) ? __builtin_amdgcn_rcpf(l) : 0.0f;
    }
    s_alpha[tid] = inv_l;
  }
  __syncthreads();

#pragma unroll
  for (int ei = 0; ei < 8; ++ei) {
    const int hm = 2 * ei + hrow;
    const float inv_l = s_alpha[h0 + hm];
    uint16_t* dst = out_ptr + (int64_t)t * out_s0 + (int64_t)(h0 + hm) * out_s1 + vd_start + l16;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      dst[j * 16] = f32_to_bf16_rne(o_acc[j][ei] * inv_l);
    }
  }
}


// v0.2.8 Pipelined Register-Resident-Q Union-LDS Quantized DSA Prefill (KV_FMT: 1=fp8 656B, 2=nvfp4 352B)
//  - 32H/WG (2 head-groups x 4 D-quarters: Q@K^T 144 dims = 9 WMMAs, P@V 128 dims = 8 WMMAs)
//  - Q[16x144] 100% in VGPRs across all W=2048 tiles (ZERO Q loads in loop!)
//  - Union KV_lds/VT_lds_g -> 27.1 KiB LDS (<32 KiB -> 2 WGs/CU = 16 waves/CU on gfx1151!)
//  - Direct idx_row load (4 barriers/tile instead of 5!) + pipelined raw KV prefetch during P@V WMMAs!
template <int KV_FMT>
__global__ __launch_bounds__(256) void k_ernie_dsa_sparse_mla_quant_qreg_v28(
    const uint16_t* __restrict__ q_ptr,      // [T, 64, 576] bf16
    const uint8_t* __restrict__ kv_ptr,      // [num_blocks, block_size, ROW_BYTES] uint8
    const int32_t* __restrict__ idx_ptr,     // [T, W] int32
    uint16_t* __restrict__ out_ptr,          // [T, 64, 512] bf16
    int64_t q_s0, int64_t q_s1,
    int64_t kv_blk_stride, int64_t kv_row_stride,
    int64_t idx_s0, int64_t out_s0, int64_t out_s1,
    int num_blocks, int block_size, int W, float scale) {
  __shared__ union {
    __attribute__((aligned(16))) uint16_t KV_lds[16][DSA_STRIDE];               // 18,688 B
    __attribute__((aligned(16))) uint16_t VT_lds_g[16][32 * DSA_VT_STRIDE + 2]; // 18,496 B
  } u_kv;
  __shared__ float S_part[2][4][16][16];                                        //  8,192 B (4-way parallel softmax!)
  __shared__ __attribute__((aligned(16))) uint16_t P_lds[2][16][16];            //  1,024 B
  __shared__ float fp8_lut[256];                                                //  1,024 B
  __shared__ float s_alpha[32];
  __shared__ float s_m[32];
  __shared__ float s_l[32];

  const int hg32 = blockIdx.x;
  const int t = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int wh = wave >> 2;     // 0..1
  const int wd = wave & 3;      // 0..3
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;
  const int h_base = hg32 * 32;
  const int h0 = h_base + wh * 16;

  fp8_lut[tid] = dsa_fp8_ocp_to_f32((uint32_t)tid);
  if (tid < 32) {
    s_m[tid] = NEG_INF;
    s_l[tid] = 0.0f;
  }

  // Preload Q[16 x 144] ONCE into VGPRs -> ZERO Q global/L1 loads in 128-tile loop!
  const int kd_start = wd * 144;
  const int vd_start = wd * 128;
  const uint16_t* q_base = q_ptr + (int64_t)t * q_s0 + (int64_t)(h0 + l16) * q_s1 + kd_start;
  bf16x16 q_reg[9];
#pragma unroll
  for (int step = 0; step < 9; ++step) {
    const u4v q0 = *reinterpret_cast<const u4va*>(q_base + step * 16 + 0);
    const u4v q1 = *reinterpret_cast<const u4va*>(q_base + step * 16 + 8);
    q_reg[step] = __builtin_bit_cast(bf16x16, (u8v){q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w});
  }

  f32x8 o_acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) o_acc[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  const int32_t* idx_row = idx_ptr + (int64_t)t * idx_s0;
  const int total_slots = num_blocks * block_size;
  const int kr_ld = tid >> 4;   // 0..15 (16 tokens)
  const int sub16 = tid & 15;   // 0..15 (32 NOPE dims/thread!)
  const int total_blks = (W + 15) >> 4;

  // Compact raw tile storage for pipelining (reused across iterations!)
  struct RawKV {
    int32_t sl;
    u4v raw0, raw1;   // FP8: 32 bytes nope; NVFP4: raw0={fp4_0.x, fp4_0.y, fp4_1.x, fp4_1.y}, raw1.x=sc2
    u4v rope_raw;     // FP8: 8 bf16 rope; NVFP4: rope_raw.xy = 8 fp8 rope
    float sc_fp8;
  };

  auto prefetch_raw = [&](int blk_i, RawKV& dst) {
    const int c = blk_i * 16 + kr_ld;
    int32_t sl = (c < W) ? idx_row[c] : -1;
    if (sl < 0 || sl >= total_slots) sl = -1;
    dst.sl = sl;
    if (sl >= 0) {
      const int phys_blk = (block_size == 64) ? (sl >> 6) : (sl / block_size);
      const int in_blk   = (block_size == 64) ? (sl & 63) : (sl % block_size);
      const uint8_t* row_p = kv_ptr + (int64_t)phys_blk * kv_blk_stride + (int64_t)in_blk * kv_row_stride;
      if constexpr (KV_FMT == 1) {
        dst.sc_fp8 = *reinterpret_cast<const float*>(row_p + 512 + (sub16 >> 2) * 4);
        dst.raw0 = *reinterpret_cast<const u4va*>(row_p + sub16 * 32 + 0);
        dst.raw1 = *reinterpret_cast<const u4va*>(row_p + sub16 * 32 + 16);
        if (sub16 < 8) dst.rope_raw = *reinterpret_cast<const u4va*>(row_p + 528 + sub16 * 16);
      } else {
        dst.raw0 = *reinterpret_cast<const u4va*>(row_p + sub16 * 16);
        dst.raw1.x = (uint32_t)(*reinterpret_cast<const uint16_t*>(row_p + 320 + sub16 * 2));
        if (sub16 < 8) {
          const uint2 rr = *reinterpret_cast<const uint2*>(row_p + 256 + sub16 * 8);
          dst.rope_raw.x = rr.x; dst.rope_raw.y = rr.y;
        }
      }
    }
  };

  RawKV cur_raw;
  if (total_blks > 0) prefetch_raw(0, cur_raw);
  __syncthreads();

#pragma unroll 1
  for (int blk = 0; blk < total_blks; ++blk) {
    // 1. Dequant cur_raw -> u_kv.KV_lds (keep pk_hold[16] in VGPRs for VT_lds_g after Q@K^T!)
    uint32_t pk_hold[16];
    if (cur_raw.sl >= 0) {
      if constexpr (KV_FMT == 1) {
        const float sc = cur_raw.sc_fp8;
        const u4v raws[2] = {cur_raw.raw0, cur_raw.raw1};
#pragma unroll
        for (int c2 = 0; c2 < 2; ++c2) {
          const uint32_t ws[4] = {raws[c2].x, raws[c2].y, raws[c2].z, raws[c2].w};
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            const uint32_t x = ws[w];
            pk_hold[c2 * 8 + w * 2 + 0] = f32_to_bf16x2_fast(fp8_lut[x & 0xFFu] * sc, fp8_lut[(x >> 8) & 0xFFu] * sc);
            pk_hold[c2 * 8 + w * 2 + 1] = f32_to_bf16x2_fast(fp8_lut[(x >> 16) & 0xFFu] * sc, fp8_lut[(x >> 24) & 0xFFu] * sc);
          }
          const int col0 = sub16 * 32 + c2 * 16;
          *reinterpret_cast<u4va*>(&u_kv.KV_lds[kr_ld][col0 + 0]) = (u4v){pk_hold[c2*8+0], pk_hold[c2*8+1], pk_hold[c2*8+2], pk_hold[c2*8+3]};
          *reinterpret_cast<u4va*>(&u_kv.KV_lds[kr_ld][col0 + 8]) = (u4v){pk_hold[c2*8+4], pk_hold[c2*8+5], pk_hold[c2*8+6], pk_hold[c2*8+7]};
        }
        if (sub16 < 8) {
          *reinterpret_cast<u4va*>(&u_kv.KV_lds[kr_ld][512 + sub16 * 8]) = cur_raw.rope_raw;
        }
      } else {
        const uint32_t sc2 = cur_raw.raw1.x;
        const uint32_t rwords[4] = {cur_raw.raw0.x, cur_raw.raw0.y, cur_raw.raw0.z, cur_raw.raw0.w};
#pragma unroll
        for (int mb = 0; mb < 2; ++mb) {
          const float sc = fp8_lut[(sc2 >> (mb * 8)) & 0xFFu];
          const uint32_t rx = rwords[mb * 2 + 0], ry = rwords[mb * 2 + 1];
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            const uint32_t byte0 = (rx >> (b * 8)) & 0xFFu;
            const uint32_t byte1 = (ry >> (b * 8)) & 0xFFu;
            pk_hold[mb * 8 + b]     = f32_to_bf16x2_fast(e2m1_reg_to_f32(byte0 & 0xFu) * sc, e2m1_reg_to_f32(byte0 >> 4) * sc);
            pk_hold[mb * 8 + b + 4] = f32_to_bf16x2_fast(e2m1_reg_to_f32(byte1 & 0xFu) * sc, e2m1_reg_to_f32(byte1 >> 4) * sc);
          }
          const int col0 = sub16 * 32 + mb * 16;
          *reinterpret_cast<u4va*>(&u_kv.KV_lds[kr_ld][col0 + 0]) = (u4v){pk_hold[mb*8+0], pk_hold[mb*8+1], pk_hold[mb*8+2], pk_hold[mb*8+3]};
          *reinterpret_cast<u4va*>(&u_kv.KV_lds[kr_ld][col0 + 8]) = (u4v){pk_hold[mb*8+4], pk_hold[mb*8+5], pk_hold[mb*8+6], pk_hold[mb*8+7]};
        }
        if (sub16 < 8) {
          const uint32_t rx = cur_raw.rope_raw.x, ry = cur_raw.rope_raw.y;
          u4v rp;
          rp.x = f32_to_bf16x2_fast(fp8_lut[rx & 0xFFu], fp8_lut[(rx >> 8) & 0xFFu]);
          rp.y = f32_to_bf16x2_fast(fp8_lut[(rx >> 16) & 0xFFu], fp8_lut[(rx >> 24) & 0xFFu]);
          rp.z = f32_to_bf16x2_fast(fp8_lut[ry & 0xFFu], fp8_lut[(ry >> 8) & 0xFFu]);
          rp.w = f32_to_bf16x2_fast(fp8_lut[(ry >> 16) & 0xFFu], fp8_lut[(ry >> 24) & 0xFFu]);
          *reinterpret_cast<u4va*>(&u_kv.KV_lds[kr_ld][512 + sub16 * 8]) = rp;
        }
      }
    } else {
      const u4v z = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int i = 0; i < 16; ++i) pk_hold[i] = 0u;
#pragma unroll
      for (int c4 = 0; c4 < 4; ++c4) *reinterpret_cast<u4va*>(&u_kv.KV_lds[kr_ld][sub16 * 32 + c4 * 8]) = z;
      if (sub16 < 8) *reinterpret_cast<u4va*>(&u_kv.KV_lds[kr_ld][512 + sub16 * 8]) = z;
    }
    __syncthreads();

    // 2. Q @ K^T: 9 WMMA steps, Q 100% in VGPRs, 1-ahead K_lds prefetch!
    f32x8 s_acc = {0, 0, 0, 0, 0, 0, 0, 0};
    u4v k0_c = *reinterpret_cast<const u4va*>(&u_kv.KV_lds[l16][kd_start]);
    u4v k1_c = *reinterpret_cast<const u4va*>(&u_kv.KV_lds[l16][kd_start + 8]);
#pragma unroll
    for (int step = 0; step < 9; ++step) {
      u4v k0_n = {0u, 0u, 0u, 0u}, k1_n = {0u, 0u, 0u, 0u};
      if (step + 1 < 9) {
        const int dn = kd_start + (step + 1) * 16;
        k0_n = *reinterpret_cast<const u4va*>(&u_kv.KV_lds[l16][dn]);
        k1_n = *reinterpret_cast<const u4va*>(&u_kv.KV_lds[l16][dn + 8]);
      }
      const bf16x16 kf = __builtin_bit_cast(bf16x16, (u8v){k0_c.x, k0_c.y, k0_c.z, k0_c.w, k1_c.x, k1_c.y, k1_c.z, k1_c.w});
      s_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(q_reg[step], kf, s_acc);
      k0_c = k0_n; k1_c = k1_n;
    }

#pragma unroll
    for (int ei = 0; ei < 8; ++ei) S_part[wh][wd][2 * ei + hrow][l16] = s_acc[ei];
    __syncthreads();

    // 3a. Write u_kv.VT_lds_g FIRST (KV_lds is dead!) + hoist NEXT raw KV prefetch BEFORE softmax!
    {
      uint16_t* vt_dst = &u_kv.VT_lds_g[sub16][kr_ld];
#pragma unroll
      for (int p = 0; p < 16; ++p) {
        vt_dst[(p * 2 + 0) * DSA_VT_STRIDE] = (uint16_t)(pk_hold[p] & 0xFFFFu);
        vt_dst[(p * 2 + 1) * DSA_VT_STRIDE] = (uint16_t)(pk_hold[p] >> 16);
      }
    }
    if (blk + 1 < total_blks) {
      prefetch_raw(blk + 1, cur_raw);
    }
    __builtin_amdgcn_sched_barrier(0);

    // 3b. 4-way parallel online softmax across ALL 4 wd waves (2 ei/wave instead of 8 on wd==0!)
    {
      const int c = blk * 16 + l16;
      const int32_t sl_v = (c < W) ? idx_row[c] : -1;
      const bool vld = (sl_v >= 0 && sl_v < total_slots);
#pragma unroll
      for (int e_sub = 0; e_sub < 2; ++e_sub) {
        const int ei = wd * 2 + e_sub;
        const int hm = 2 * ei + hrow;
        const int h_loc = wh * 16 + hm;
        float sc = S_part[wh][0][hm][l16] + S_part[wh][1][hm][l16] + S_part[wh][2][hm][l16] + S_part[wh][3][hm][l16];
        sc = vld ? (sc * scale) : NEG_INF;
        const float bmax = halfwave_max16(sc);
        const float mold = s_m[h_loc];
        const float mnew = fmaxf(mold, bmax);
        const float alpha = (mold > -1e30f) ? exp2f((mold - mnew) * LOG2E) : 0.0f;
        const float pv = (vld && mnew > -1e30f) ? exp2f((sc - mnew) * LOG2E) : 0.0f;
        const float bsum = halfwave_sum16(pv);
        P_lds[wh][hm][l16] = f32_to_bf16_rne(pv);
        if (l16 == 0) {
          s_alpha[h_loc] = alpha;
          s_m[h_loc] = mnew;
          s_l[h_loc] = s_l[h_loc] * alpha + bsum;
        }
      }
    }
    __syncthreads();

    float al[8];
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) al[ei] = s_alpha[wh * 16 + 2 * ei + hrow];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) o_acc[j][ei] *= al[ei];
    }

    const u4v p0 = *reinterpret_cast<const u4va*>(&P_lds[wh][l16][0]);
    const u4v p1 = *reinterpret_cast<const u4va*>(&P_lds[wh][l16][8]);
    const bf16x16 pf = __builtin_bit_cast(bf16x16, (u8v){p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w});

    const int vg0 = vd_start >> 5;
    u4v v0_c = *reinterpret_cast<const u4va*>(&u_kv.VT_lds_g[vg0][l16 * DSA_VT_STRIDE + 0]);
    u4v v1_c = *reinterpret_cast<const u4va*>(&u_kv.VT_lds_g[vg0][l16 * DSA_VT_STRIDE + 8]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      u4v v0_n = {0u, 0u, 0u, 0u}, v1_n = {0u, 0u, 0u, 0u};
      if (j + 1 < 8) {
        const int vc_n = vd_start + (j + 1) * 16 + l16;
        const int vgn = vc_n >> 5;
        v0_n = *reinterpret_cast<const u4va*>(&u_kv.VT_lds_g[vgn][(vc_n & 31) * DSA_VT_STRIDE + 0]);
        v1_n = *reinterpret_cast<const u4va*>(&u_kv.VT_lds_g[vgn][(vc_n & 31) * DSA_VT_STRIDE + 8]);
      }
      const bf16x16 vf = __builtin_bit_cast(bf16x16, (u8v){v0_c.x, v0_c.y, v0_c.z, v0_c.w, v1_c.x, v1_c.y, v1_c.z, v1_c.w});
      o_acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf, vf, o_acc[j]);
      v0_c = v0_n; v1_c = v1_n;
    }
    __syncthreads();
  }

  // Finalize
#pragma unroll
  for (int ei = 0; ei < 8; ++ei) {
    const int hm = 2 * ei + hrow;
    const int h_loc = wh * 16 + hm;
    const float lv = s_l[h_loc];
    const float inv_l = (lv > 0.0f) ? __builtin_amdgcn_rcpf(lv) : 0.0f;
    uint16_t* dst = out_ptr + (int64_t)t * out_s0 + (int64_t)(h0 + hm) * out_s1 + vd_start + l16;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      dst[j * 16] = f32_to_bf16_rne(o_acc[j][ei] * inv_l);
    }
  }
}


// ============================================================================
// v0.2.8 dim3(512) 16-Wave ALL-64-HEADS Quantized DSA Prefill (gfx1151 RDNA 3.5):
//   - 1 WG per token (4 wh x 4 wd = 16 waves = ALL 64 heads -> 2x less KV traffic, 2x less dequant, 2x less VT_lds writes)
//   - Bank-conflict-free swizzled Q_lds[4][4][16][6] (0-way ds_load_b128 conflicts, 0 extra bytes!)
//   - EXEC-safe 1-inst ds_bpermute_b32 NVFP4 E2M1 LUT via 64B e2m1_lds[16]
//   - _Float16 fp8_lut16[256] (512B, 100% bit-exact for OCP FP8 E4M3) + BS64 + sc_vld packing
//   - Anti-LICM zero-hoist on s_acc + per-ei scalar s_alpha + amdgpu_waves_per_eu(8, 8)
//   -> V=186 (NVFP4) / V=188 (FP8), vSp=0, sSp=0, scr=0, LDS<=63.0 KiB -> 2 WGs/WGP = 32 waves/WGP (+33% occ!)
//   -> 1.206x (-90 ms) NVFP4, 1.240x (-107 ms) FP8 at T=4096 W=2048 (6L), 0.000000e+00 bit-exact!
// ============================================================================
template <int KV_FMT, int NUM_SPLITS = 1>
__global__ __launch_bounds__(512) __attribute__((amdgpu_waves_per_eu(8, 8)))
void k_ernie_dsa_sparse_mla_quant_512_v28(
    const uint16_t* __restrict__ q_ptr,
    const uint8_t* __restrict__ kv_ptr,
    const int32_t* __restrict__ idx_ptr,
    const float* __restrict__ sink_ptr,
    uint16_t* __restrict__ out_ptr,
    float* __restrict__ part_o,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    int64_t q_s0, int64_t q_s1,
    int64_t kv_blk_stride, int64_t kv_row_stride,
    int64_t idx_s0, int64_t out_s0, int64_t out_s1,
    int num_blocks, int block_size, int W, float scale) {
  constexpr int Q_LDS_N = 3;
  constexpr int Q_NREG  = 9 - Q_LDS_N;
  constexpr int USE_BPERM = (KV_FMT == 2) ? 1 : 0;

  __shared__ union {
    __attribute__((aligned(16))) uint16_t KV_lds[16][DSA_STRIDE];               // 18,688 B
    __attribute__((aligned(16))) uint16_t VT_lds_g[16][32 * DSA_VT_STRIDE + 2]; // 18,496 B
  } u_kv;
  __shared__ float S_part[4][4][16][16];                                        // 16,384 B
  __shared__ __attribute__((aligned(16))) uint16_t P_lds[4][16][16];            //  2,048 B
  __shared__ _Float16 fp8_lut16[256];                                           //    512 B
  __shared__ uint32_t e2m1_lds[USE_BPERM ? 16 : 1];                             //     64 B
  __shared__ float s_alpha[64];                                                 //    256 B
  __shared__ float s_m[64];                                                     //    256 B
  __shared__ float s_l[64];                                                     //    256 B
  __shared__ __attribute__((aligned(16))) u4v Q_lds[4][4][16][Q_LDS_N * 2];     // 24,576 B

  const int t    = blockIdx.x;
  const int tid  = threadIdx.x;
  const int wave = tid >> 5;
  const int wh   = wave >> 2;
  const int wd   = wave & 3;
  const int lane = tid & 31;
  const int l16  = lane & 15;
  const int hrow = lane >> 4;
  const int h0   = wh * 16;

  if (tid < 256) fp8_lut16[tid] = (_Float16)dsa_fp8_ocp_to_f32((uint32_t)tid);
  if constexpr (USE_BPERM) {
    if (tid < 16) e2m1_lds[tid] = __float_as_uint(e2m1_reg_to_f32((uint32_t)tid));
  }
  if (tid < 64) { s_m[tid] = NEG_INF; s_l[tid] = 0.0f; }

  const int kd_start = wd * 144;
  const int vd_start = wd * 128;
  const uint16_t* q_base = q_ptr + (int64_t)t * q_s0 + (int64_t)(h0 + l16) * q_s1 + kd_start;
  bf16x16 q_reg[Q_NREG];
#pragma unroll
  for (int step = 0; step < Q_NREG; ++step) {
    const u4v q0 = *reinterpret_cast<const u4va*>(q_base + step * 16 + 0);
    const u4v q1 = *reinterpret_cast<const u4va*>(q_base + step * 16 + 8);
    q_reg[step] = __builtin_bit_cast(bf16x16, (u8v){q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w});
  }

  auto q_swz = [&](int c, int r) -> int {
    const int cs = c + ((r >> 2) & 1);
    return (cs >= 6) ? (cs - 6) : cs;
  };
  if (hrow == 0) {
#pragma unroll
    for (int qi = 0; qi < Q_LDS_N; ++qi) {
      const int step = Q_NREG + qi;
      Q_lds[wh][wd][l16][q_swz(qi * 2 + 0, l16)] = *reinterpret_cast<const u4va*>(q_base + step * 16 + 0);
      Q_lds[wh][wd][l16][q_swz(qi * 2 + 1, l16)] = *reinterpret_cast<const u4va*>(q_base + step * 16 + 8);
    }
  }

  f32x8 o_acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) o_acc[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  const int32_t* idx_row = idx_ptr + (int64_t)t * idx_s0;
  const int total_slots = num_blocks * block_size;
  const int kr_ld = tid >> 5;
  const int mb32  = tid & 31;
  const int total_blks = (W + 15) >> 4;
  const int sp = (NUM_SPLITS > 1) ? (int)blockIdx.y : 0;
  const int blk_chunk = (NUM_SPLITS > 1) ? ((total_blks + NUM_SPLITS - 1) / NUM_SPLITS) : total_blks;
  const int blk_lo = (NUM_SPLITS > 1) ? (sp * blk_chunk) : 0;
  const int blk_hi = (NUM_SPLITS > 1) ? min(blk_lo + blk_chunk, total_blks) : total_blks;

  struct RawKV16 {
    u4v raw_fp8;
    uint2 raw_fp4;
    uint32_t sc_vld;
    float sc_fp8;
    u4v rope_fp8;
    uint2 rope_nv4;
  };

  auto prefetch_raw = [&](int blk_i, RawKV16& dst) {
    const int c = blk_i * 16 + kr_ld;
    int32_t sl = (c < W) ? idx_row[c] : -1;
    if (sl < 0 || sl >= total_slots) sl = -1;
    if (sl >= 0) {
      const int phys_blk = sl >> 6;
      const int in_blk   = sl & 63;
      const uint8_t* row_p = kv_ptr + (int64_t)phys_blk * kv_blk_stride + (int64_t)in_blk * kv_row_stride;
      if constexpr (KV_FMT == 1) {
        dst.sc_vld  = 1u;
        dst.sc_fp8  = *reinterpret_cast<const float*>(row_p + 512 + (mb32 >> 3) * 4);
        dst.raw_fp8 = *reinterpret_cast<const u4va*>(row_p + mb32 * 16);
        if (mb32 < 8) dst.rope_fp8 = *reinterpret_cast<const u4va*>(row_p + 528 + mb32 * 16);
      } else {
        dst.raw_fp4 = *reinterpret_cast<const uint2*>(row_p + mb32 * 8);
        dst.sc_vld  = 0x100u | (uint32_t)row_p[320 + mb32];
        if (mb32 < 8) dst.rope_nv4 = *reinterpret_cast<const uint2*>(row_p + 256 + mb32 * 8);
      }
    } else {
      dst.sc_vld = 0u;
    }
  };

  auto decode_e2m1 = [&](uint32_t nib, uint32_t tbl) -> float {
    if constexpr (USE_BPERM) {
      return __uint_as_float(__builtin_amdgcn_ds_bpermute((((uint32_t)lane & 16u) | nib) << 2, tbl));
    } else {
      return e2m1_reg_to_f32(nib);
    }
  };

  RawKV16 cur_raw;
  if (blk_lo < blk_hi) prefetch_raw(blk_lo, cur_raw);
  __syncthreads();

#pragma unroll 1
  for (int blk = blk_lo; blk < blk_hi; ++blk) {
    uint32_t pk_hold[8];
    if (cur_raw.sc_vld) {
      if constexpr (KV_FMT == 1) {
        const float sc = cur_raw.sc_fp8;
        const uint32_t ws[4] = {cur_raw.raw_fp8.x, cur_raw.raw_fp8.y, cur_raw.raw_fp8.z, cur_raw.raw_fp8.w};
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          const uint32_t x = ws[w];
          pk_hold[w * 2 + 0] = f32_to_bf16x2_fast((float)fp8_lut16[x & 0xFFu] * sc, (float)fp8_lut16[(x >> 8) & 0xFFu] * sc);
          pk_hold[w * 2 + 1] = f32_to_bf16x2_fast((float)fp8_lut16[(x >> 16) & 0xFFu] * sc, (float)fp8_lut16[(x >> 24) & 0xFFu] * sc);
          if (w == 1) __builtin_amdgcn_sched_barrier(0);
        }
        const int col0 = mb32 * 16;
        *reinterpret_cast<u4va*>(&u_kv.KV_lds[kr_ld][col0 + 0]) = (u4v){pk_hold[0], pk_hold[1], pk_hold[2], pk_hold[3]};
        *reinterpret_cast<u4va*>(&u_kv.KV_lds[kr_ld][col0 + 8]) = (u4v){pk_hold[4], pk_hold[5], pk_hold[6], pk_hold[7]};
        if (mb32 < 8) *reinterpret_cast<u4va*>(&u_kv.KV_lds[kr_ld][512 + mb32 * 8]) = cur_raw.rope_fp8;
      } else {
        const float sc = (float)fp8_lut16[cur_raw.sc_vld & 0xFFu];
        const uint32_t tbl = e2m1_lds[l16];
        const uint32_t rx = cur_raw.raw_fp4.x, ry = cur_raw.raw_fp4.y;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const uint32_t byte0 = (rx >> (b * 8)) & 0xFFu;
          const uint32_t byte1 = (ry >> (b * 8)) & 0xFFu;
          pk_hold[b]     = f32_to_bf16x2_fast(decode_e2m1(byte0 & 0xFu, tbl) * sc, decode_e2m1(byte0 >> 4, tbl) * sc);
          pk_hold[b + 4] = f32_to_bf16x2_fast(decode_e2m1(byte1 & 0xFu, tbl) * sc, decode_e2m1(byte1 >> 4, tbl) * sc);
          if (b == 1) __builtin_amdgcn_sched_barrier(0);
        }
        const int col0 = mb32 * 16;
        *reinterpret_cast<u4va*>(&u_kv.KV_lds[kr_ld][col0 + 0]) = (u4v){pk_hold[0], pk_hold[1], pk_hold[2], pk_hold[3]};
        *reinterpret_cast<u4va*>(&u_kv.KV_lds[kr_ld][col0 + 8]) = (u4v){pk_hold[4], pk_hold[5], pk_hold[6], pk_hold[7]};
        if (mb32 < 8) {
          const uint32_t rrx = cur_raw.rope_nv4.x, rry = cur_raw.rope_nv4.y;
          u4v rp;
          rp.x = f32_to_bf16x2_fast((float)fp8_lut16[rrx & 0xFFu], (float)fp8_lut16[(rrx >> 8) & 0xFFu]);
          rp.y = f32_to_bf16x2_fast((float)fp8_lut16[(rrx >> 16) & 0xFFu], (float)fp8_lut16[(rrx >> 24) & 0xFFu]);
          rp.z = f32_to_bf16x2_fast((float)fp8_lut16[rry & 0xFFu], (float)fp8_lut16[(rry >> 8) & 0xFFu]);
          rp.w = f32_to_bf16x2_fast((float)fp8_lut16[(rry >> 16) & 0xFFu], (float)fp8_lut16[(rry >> 24) & 0xFFu]);
          *reinterpret_cast<u4va*>(&u_kv.KV_lds[kr_ld][512 + mb32 * 8]) = rp;
        }
      }
    } else {
      const u4v z = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int i = 0; i < 8; ++i) pk_hold[i] = 0u;
      const int col0 = mb32 * 16;
      *reinterpret_cast<u4va*>(&u_kv.KV_lds[kr_ld][col0 + 0]) = z;
      *reinterpret_cast<u4va*>(&u_kv.KV_lds[kr_ld][col0 + 8]) = z;
      if (mb32 < 8) *reinterpret_cast<u4va*>(&u_kv.KV_lds[kr_ld][512 + mb32 * 8]) = z;
    }
    __syncthreads();

    float z0 = 0.0f, z1 = 0.0f, z2 = 0.0f, z3 = 0.0f, z4 = 0.0f, z5 = 0.0f, z6 = 0.0f, z7 = 0.0f;
    asm volatile("" : "+v"(z0), "+v"(z1), "+v"(z2), "+v"(z3), "+v"(z4), "+v"(z5), "+v"(z6), "+v"(z7));
    f32x8 s_acc = {z0, z1, z2, z3, z4, z5, z6, z7};

    u4v k0_c = *reinterpret_cast<const u4va*>(&u_kv.KV_lds[l16][kd_start]);
    u4v k1_c = *reinterpret_cast<const u4va*>(&u_kv.KV_lds[l16][kd_start + 8]);
#pragma unroll
    for (int step = 0; step < 9; ++step) {
      u4v k0_n, k1_n;
      if (step + 1 < 9) {
        const int dn = kd_start + (step + 1) * 16;
        k0_n = *reinterpret_cast<const u4va*>(&u_kv.KV_lds[l16][dn]);
        k1_n = *reinterpret_cast<const u4va*>(&u_kv.KV_lds[l16][dn + 8]);
      }
      const bf16x16 kf = __builtin_bit_cast(bf16x16, (u8v){k0_c.x, k0_c.y, k0_c.z, k0_c.w, k1_c.x, k1_c.y, k1_c.z, k1_c.w});
      if (step < Q_NREG) {
        s_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(q_reg[step], kf, s_acc);
      } else {
        const int qi = step - Q_NREG;
        const u4v q0 = Q_lds[wh][wd][l16][q_swz(qi * 2 + 0, l16)];
        const u4v q1 = Q_lds[wh][wd][l16][q_swz(qi * 2 + 1, l16)];
        const bf16x16 qf = __builtin_bit_cast(bf16x16, (u8v){q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w});
        s_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(qf, kf, s_acc);
      }
      if (step + 1 < 9) { k0_c = k0_n; k1_c = k1_n; }
    }

#pragma unroll
    for (int ei = 0; ei < 8; ++ei) S_part[wh][wd][2 * ei + hrow][l16] = s_acc[ei];
    __syncthreads();

    {
      uint16_t* vt_dst = &u_kv.VT_lds_g[mb32 >> 1][(mb32 & 1) * 16 * DSA_VT_STRIDE + kr_ld];
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        vt_dst[(p * 2 + 0) * DSA_VT_STRIDE] = (uint16_t)(pk_hold[p] & 0xFFFFu);
        vt_dst[(p * 2 + 1) * DSA_VT_STRIDE] = (uint16_t)(pk_hold[p] >> 16);
      }
    }
    if (blk + 1 < blk_hi) {
      prefetch_raw(blk + 1, cur_raw);
    }
    __builtin_amdgcn_sched_barrier(0);

    {
      const int c = blk * 16 + l16;
      const int32_t sl_v = (c < W) ? idx_row[c] : -1;
      const bool vld = (sl_v >= 0 && sl_v < total_slots);
#pragma unroll
      for (int e_sub = 0; e_sub < 2; ++e_sub) {
        const int ei = wd * 2 + e_sub;
        const int hm = 2 * ei + hrow;
        const int h_loc = wh * 16 + hm;
        float sc = S_part[wh][0][hm][l16] + S_part[wh][1][hm][l16] + S_part[wh][2][hm][l16] + S_part[wh][3][hm][l16];
        sc = vld ? (sc * scale) : NEG_INF;
        const float bmax = halfwave_max16(sc);
        const float mold = s_m[h_loc];
        const float mnew = fmaxf(mold, bmax);
        const float alpha = (mold > -1e30f) ? exp2f((mold - mnew) * LOG2E) : 0.0f;
        const float pv = (vld && mnew > -1e30f) ? exp2f((sc - mnew) * LOG2E) : 0.0f;
        const float bsum = halfwave_sum16(pv);
        P_lds[wh][hm][l16] = f32_to_bf16_rne(pv);
        if (l16 == 0) {
          s_alpha[h_loc] = alpha;
          s_m[h_loc] = mnew;
          s_l[h_loc] = s_l[h_loc] * alpha + bsum;
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      const float a = s_alpha[wh * 16 + 2 * ei + hrow];
#pragma unroll
      for (int j = 0; j < 8; ++j) o_acc[j][ei] *= a;
    }

    const u4v p0 = *reinterpret_cast<const u4va*>(&P_lds[wh][l16][0]);
    const u4v p1 = *reinterpret_cast<const u4va*>(&P_lds[wh][l16][8]);
    const bf16x16 pf = __builtin_bit_cast(bf16x16, (u8v){p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w});

    const int vg0 = vd_start >> 5;
    u4v v0_c = *reinterpret_cast<const u4va*>(&u_kv.VT_lds_g[vg0][l16 * DSA_VT_STRIDE + 0]);
    u4v v1_c = *reinterpret_cast<const u4va*>(&u_kv.VT_lds_g[vg0][l16 * DSA_VT_STRIDE + 8]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      u4v v0_n, v1_n;
      if (j + 1 < 8) {
        const int vc_n = vd_start + (j + 1) * 16 + l16;
        const int vgn = vc_n >> 5;
        v0_n = *reinterpret_cast<const u4va*>(&u_kv.VT_lds_g[vgn][(vc_n & 31) * DSA_VT_STRIDE + 0]);
        v1_n = *reinterpret_cast<const u4va*>(&u_kv.VT_lds_g[vgn][(vc_n & 31) * DSA_VT_STRIDE + 8]);
      }
      const bf16x16 vf = __builtin_bit_cast(bf16x16, (u8v){v0_c.x, v0_c.y, v0_c.z, v0_c.w, v1_c.x, v1_c.y, v1_c.z, v1_c.w});
      o_acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf, vf, o_acc[j]);
      if (j + 1 < 8) { v0_c = v0_n; v1_c = v1_n; }
    }
    __syncthreads();
  }

  if constexpr (NUM_SPLITS > 1) {
    if (wd == 0 && l16 == 0) {
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) {
        const int h = h0 + 2 * ei + hrow;
        const int64_t off = ((int64_t)t * 64 + h) * NUM_SPLITS + sp;
        part_m[off] = s_m[h];
        part_l[off] = s_l[h];
      }
    }
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      const int h = h0 + 2 * ei + hrow;
      float* dst = part_o + (((int64_t)t * 64 + h) * NUM_SPLITS + sp) * DSA_NOPE + vd_start + l16;
#pragma unroll
      for (int j = 0; j < 8; ++j) dst[j * 16] = o_acc[j][ei];
    }
  } else {
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      const int hm = 2 * ei + hrow;
      const int h_loc = wh * 16 + hm;
      const float lv = s_l[h_loc];
      float inv_l = (lv > 0.0f) ? __builtin_amdgcn_rcpf(lv) : 0.0f;
      if (sink_ptr != nullptr) {
        const float sk = sink_ptr[h_loc];
        const float mv = s_m[h_loc];
        const float m_fin = fmaxf(mv, sk);
        const float w_o = (lv > 0.0f) ? exp2f((mv - m_fin) * LOG2E) : 0.0f;
        const float w_s = exp2f((sk - m_fin) * LOG2E);
        const float l_tot = lv * w_o + w_s;
        inv_l = (l_tot > 0.0f) ? (__builtin_amdgcn_rcpf(l_tot) * w_o) : 0.0f;
      }
      uint16_t* dst = out_ptr + (int64_t)t * out_s0 + (int64_t)(h0 + hm) * out_s1 + vd_start + l16;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        dst[j * 16] = f32_to_bf16_rne(o_acc[j][ei] * inv_l);
      }
    }
  }
}

torch::Tensor ernie_dsa_sparse_mla_quant(
    torch::Tensor q, torch::Tensor kv_cache, torch::Tensor indices,
    c10::optional<torch::Tensor> attn_sink, double scale,
    c10::optional<torch::Tensor> part_o_opt,
    c10::optional<torch::Tensor> part_m_opt,
    c10::optional<torch::Tensor> part_l_opt,
    int64_t num_splits, int64_t kv_fmt) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 3 && q.size(1) == 64 && q.size(2) == DSA_DIM && q.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(kv_cache.is_cuda() && kv_cache.dim() == 3 && kv_cache.scalar_type() == torch::kUInt8);
  TORCH_CHECK(kv_fmt == 1 || kv_fmt == 2);
  TORCH_CHECK(kv_cache.size(2) == (kv_fmt == 1 ? 656 : 352));
  TORCH_CHECK(indices.is_cuda() && indices.dim() == 2 && indices.scalar_type() == torch::kInt32);
  const int T = (int)q.size(0);
  const int W = (int)indices.size(1);
  auto out = torch::empty({T, 64, DSA_NOPE}, q.options());
  if (T == 0) return out;
  auto st = c10::hip::getCurrentHIPStream().stream();
  const uint16_t* qp = reinterpret_cast<const uint16_t*>(q.data_ptr());
  const uint8_t* kvp = kv_cache.data_ptr<uint8_t>();
  const int32_t* ip = indices.data_ptr<int32_t>();
  const float* sp = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
  uint16_t* op = reinterpret_cast<uint16_t*>(out.data_ptr());
  const int num_blocks = (int)kv_cache.size(0);
  const int block_size = (int)kv_cache.size(1);

  if (num_splits <= 1 || W <= 32) {
    if (T >= 32) {
      if (block_size == 64) {
        if (kv_fmt == 1) {
          hipLaunchKernelGGL((k_ernie_dsa_sparse_mla_quant_512_v28<1, 1>), dim3(T, 1, 1), dim3(512), 0, st,
              qp, kvp, ip, sp, op, nullptr, nullptr, nullptr,
              q.stride(0), q.stride(1), kv_cache.stride(0), kv_cache.stride(1),
              indices.stride(0), out.stride(0), out.stride(1), num_blocks, block_size, W, (float)scale);
        } else {
          hipLaunchKernelGGL((k_ernie_dsa_sparse_mla_quant_512_v28<2, 1>), dim3(T, 1, 1), dim3(512), 0, st,
              qp, kvp, ip, sp, op, nullptr, nullptr, nullptr,
              q.stride(0), q.stride(1), kv_cache.stride(0), kv_cache.stride(1),
              indices.stride(0), out.stride(0), out.stride(1), num_blocks, block_size, W, (float)scale);
        }
        return out;
      }
      if (kv_fmt == 1) {
        hipLaunchKernelGGL((k_ernie_dsa_sparse_mla_quant_qreg_v28<1>), dim3(2, T, 1), dim3(256), 0, st,
            reinterpret_cast<const uint16_t*>(q.data_ptr<at::BFloat16>()),
            kv_cache.data_ptr<uint8_t>(), indices.data_ptr<int32_t>(),
            reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>()),
            q.stride(0), q.stride(1), kv_cache.stride(0), kv_cache.stride(1),
            indices.stride(0), out.stride(0), out.stride(1),
            (int)kv_cache.size(0), (int)kv_cache.size(1), W, (float)scale);
      } else {
        hipLaunchKernelGGL((k_ernie_dsa_sparse_mla_quant_qreg_v28<2>), dim3(2, T, 1), dim3(256), 0, st,
            reinterpret_cast<const uint16_t*>(q.data_ptr<at::BFloat16>()),
            kv_cache.data_ptr<uint8_t>(), indices.data_ptr<int32_t>(),
            reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>()),
            q.stride(0), q.stride(1), kv_cache.stride(0), kv_cache.stride(1),
            indices.stride(0), out.stride(0), out.stride(1),
            (int)kv_cache.size(0), (int)kv_cache.size(1), W, (float)scale);
      }
      return out;
      if (kv_fmt == 1) {
        hipLaunchKernelGGL((k_ernie_dsa_sparse_mla_quant_prefill_v28<1>), dim3(T, 1, 1), dim3(256), 0, st,
                           qp, kvp, ip, sp, op,
                           q.stride(0), q.stride(1), kv_cache.stride(0), kv_cache.stride(1),
                           indices.stride(0), out.stride(0), out.stride(1),
                           num_blocks, block_size, W, (float)scale);
      } else {
        hipLaunchKernelGGL((k_ernie_dsa_sparse_mla_quant_prefill_v28<2>), dim3(T, 1, 1), dim3(256), 0, st,
                           qp, kvp, ip, sp, op,
                           q.stride(0), q.stride(1), kv_cache.stride(0), kv_cache.stride(1),
                           indices.stride(0), out.stride(0), out.stride(1),
                           num_blocks, block_size, W, (float)scale);
      }
    } else if (kv_fmt == 1) {
      hipLaunchKernelGGL((k_ernie_dsa_sparse_mla_quant<1, 1, 0>), dim3(4, T, 1), dim3(128), 0, st,
                         qp, kvp, ip, sp, op, nullptr, nullptr, nullptr,
                         q.stride(0), q.stride(1), kv_cache.stride(0), kv_cache.stride(1),
                         indices.stride(0), out.stride(0), out.stride(1),
                         num_blocks, block_size, W, (float)scale);
    } else {
      hipLaunchKernelGGL((k_ernie_dsa_sparse_mla_quant<2, 1, 0>), dim3(4, T, 1), dim3(128), 0, st,
                         qp, kvp, ip, sp, op, nullptr, nullptr, nullptr,
                         q.stride(0), q.stride(1), kv_cache.stride(0), kv_cache.stride(1),
                         indices.stride(0), out.stride(0), out.stride(1),
                         num_blocks, block_size, W, (float)scale);
    }
  } else {
    auto launch_512_split = [&](auto tag) {
      constexpr int SX = decltype(tag)::value;
      torch::Tensor po = (part_o_opt.has_value() && part_o_opt->numel() >= (int64_t)T * 64 * SX * DSA_NOPE)
          ? *part_o_opt : torch::empty({T, 64, SX, DSA_NOPE}, q.options().dtype(torch::kFloat32));
      torch::Tensor pm = (part_m_opt.has_value() && part_m_opt->numel() >= (int64_t)T * 64 * SX)
          ? *part_m_opt : torch::empty({T, 64, SX}, q.options().dtype(torch::kFloat32));
      torch::Tensor pl = (part_l_opt.has_value() && part_l_opt->numel() >= (int64_t)T * 64 * SX)
          ? *part_l_opt : torch::empty({T, 64, SX}, q.options().dtype(torch::kFloat32));
      if (kv_fmt == 1) {
        hipLaunchKernelGGL((k_ernie_dsa_sparse_mla_quant_512_v28<1, SX>), dim3(T, SX, 1), dim3(512), 0, st,
            qp, kvp, ip, sp, op, po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(),
            q.stride(0), q.stride(1), kv_cache.stride(0), kv_cache.stride(1),
            indices.stride(0), out.stride(0), out.stride(1), num_blocks, block_size, W, (float)scale);
      } else {
        hipLaunchKernelGGL((k_ernie_dsa_sparse_mla_quant_512_v28<2, SX>), dim3(T, SX, 1), dim3(512), 0, st,
            qp, kvp, ip, sp, op, po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(),
            q.stride(0), q.stride(1), kv_cache.stride(0), kv_cache.stride(1),
            indices.stride(0), out.stride(0), out.stride(1), num_blocks, block_size, W, (float)scale);
      }
      hipLaunchKernelGGL((k_ernie_dsa_split_reduce<SX>), dim3(T, 64), dim3(128), 0, st,
                         po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(), sp, op,
                         out.stride(0), out.stride(1));
    };
    if (block_size == 64) {
      // Strix Halo 20-WGP (40 WG slots at 2 WGs/WGP) optimal split-K schedule:
      // T<=2: S=16 fills 16-32 WGPs (2.0-2.2x); T=5..8 & 15..20: S=4 avoids 40-WG wave tail (1.53-2.12x); else S=8 (1.7-2.1x)
      if (T <= 2 && W >= 256 && (!part_o_opt.has_value() || part_o_opt->numel() >= (int64_t)T * 64 * 16 * DSA_NOPE)) {
        launch_512_split(std::integral_constant<int, 16>{});
      } else if ((T >= 5 && T <= 8) || (T >= 15 && T <= 20)) {
        launch_512_split(std::integral_constant<int, 4>{});
      } else {
        launch_512_split(std::integral_constant<int, 8>{});
      }
      return out;
    }
    constexpr int S = 8;
    torch::Tensor po = part_o_opt.has_value() ? *part_o_opt : torch::empty({T, 64, S, DSA_NOPE}, q.options().dtype(torch::kFloat32));
    torch::Tensor pm = part_m_opt.has_value() ? *part_m_opt : torch::empty({T, 64, S}, q.options().dtype(torch::kFloat32));
    torch::Tensor pl = part_l_opt.has_value() ? *part_l_opt : torch::empty({T, 64, S}, q.options().dtype(torch::kFloat32));
    if (kv_fmt == 1) {
      hipLaunchKernelGGL((k_ernie_dsa_sparse_mla_quant<1, S, 1>), dim3(4, T, S), dim3(128), 0, st,
                         qp, kvp, ip, sp, op, po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(),
                         q.stride(0), q.stride(1), kv_cache.stride(0), kv_cache.stride(1),
                         indices.stride(0), out.stride(0), out.stride(1),
                         num_blocks, block_size, W, (float)scale);
    } else {
      hipLaunchKernelGGL((k_ernie_dsa_sparse_mla_quant<2, S, 1>), dim3(4, T, S), dim3(128), 0, st,
                         qp, kvp, ip, sp, op, po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(),
                         q.stride(0), q.stride(1), kv_cache.stride(0), kv_cache.stride(1),
                         indices.stride(0), out.stride(0), out.stride(1),
                         num_blocks, block_size, W, (float)scale);
    }
    hipLaunchKernelGGL((k_ernie_dsa_split_reduce<S>), dim3(T, 64), dim3(128), 0, st,
                       po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(), sp, op,
                       out.stride(0), out.stride(1));
  }
  return out;
}

// ================================================================== 5: FP8-Q x MXFP4-K Paged MQA Logits Decode (68 B/tok!)
__global__ __launch_bounds__(128) void k_ernie_fp8_mxfp4_paged_mqa_logits_decode(
    const uint8_t* __restrict__ q_fp8,       // [B, NEXT_N, 64, 128]
    const uint8_t* __restrict__ kv_ptr,      // [num_blocks, block_size, 68] uint8
    const float* __restrict__ weights,       // [B*NEXT_N, 64] fp32
    const int32_t* __restrict__ ctx_lens,    // [B*NEXT_N] or [B] int32
    const int32_t* __restrict__ block_tables,// [B, max_blocks] int32
    float* __restrict__ out_logits,          // [B*NEXT_N, max_model_len] fp32
    int64_t q_sb, int64_t q_sn, int64_t q_sh,
    int64_t kv_s0, int64_t kv_s1, int64_t w_s0, int64_t bt_s0, int64_t out_s0,
    int block_size, int max_blocks, int max_model_len,
    int next_n, int n_splits, int ctx_per_row, int is_fnuz) {
  __shared__ _Float16 fp8_lut[256];
  __shared__ float e2m1_lut[16];
  __shared__ __attribute__((aligned(16))) _Float16 K_lds[16][136]; // 4.25 KiB
  __shared__ float s_wsum[4][16];
  __shared__ float s_weights[64];

  const int row = blockIdx.x;
  const int split = blockIdx.y;
  const int b = row / next_n;
  const int n = row % next_n;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;

  int seq_len = ctx_per_row ? ctx_lens[row] : (ctx_lens[b] - next_n + n + 1);
  if (seq_len < 0) seq_len = 0;
  if (seq_len > max_model_len) seq_len = max_model_len;
  if (split * 16 >= seq_len) return;

  fp8_lut[tid] = fp8_byte_to_f16((uint32_t)tid, is_fnuz);
  fp8_lut[tid + 128] = fp8_byte_to_f16((uint32_t)(tid + 128), is_fnuz);
  if (tid < 16) {
    static constexpr float E2M1_C[16] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    e2m1_lut[tid] = E2M1_C[tid];
  }
  __syncthreads();

  const int h0 = wave * 16;
  const uint8_t* qp = q_fp8 + (int64_t)b * q_sb + (int64_t)n * q_sn + (int64_t)(h0 + l16) * q_sh;
  f16x16 q_reg[8];
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const u4v raw0 = *reinterpret_cast<const u4va*>(qp + (pair * 2 + 0) * 16);
    const u4v raw1 = *reinterpret_cast<const u4va*>(qp + (pair * 2 + 1) * 16);
    const u4v raws[2] = {raw0, raw1};
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      const uint32_t ws[4] = {raws[s].x, raws[s].y, raws[s].z, raws[s].w};
      f16x16 qf;
#pragma unroll
      for (int w4 = 0; w4 < 4; ++w4) {
        const uint32_t x = ws[w4];
        qf[w4 * 4 + 0] = fp8_lut[x & 0xFFu];
        qf[w4 * 4 + 1] = fp8_lut[(x >> 8) & 0xFFu];
        qf[w4 * 4 + 2] = fp8_lut[(x >> 16) & 0xFFu];
        qf[w4 * 4 + 3] = fp8_lut[(x >> 24) & 0xFFu];
      }
      q_reg[pair * 2 + s] = qf;
    }
    __builtin_amdgcn_sched_barrier(0);
  }

  if (tid < 64) s_weights[tid] = weights[(int64_t)row * w_s0 + tid];
  const int kr = tid >> 3;         // 0..15
  const int ht = tid & 7;          // 0..7
  const int grp = ht >> 1;         // 0..3
  const int kc16 = ht * 16;
  const int stride_kv = n_splits * 16;

  auto fetch_mxfp4_tile = [&](int kv_st, uint2& out_raw, uint32_t& out_sc4, float& out_tok_sc) {
    const int logical_blk = kv_st / block_size;
    const int64_t page = (int64_t)block_tables[(int64_t)b * bt_s0 + logical_blk];
    const int base_in_blk = kv_st - logical_blk * block_size;
    const uint8_t* blk_ptr = kv_ptr + page * kv_s0;
    if (kv_st + kr < seq_len) {
      const uint8_t* rp = blk_ptr + (int64_t)(base_in_blk + kr) * kv_s1;
      out_raw = *reinterpret_cast<const uint2*>(rp + ht * 8);
      out_sc4 = *reinterpret_cast<const uint32_t*>(rp + 64);
    } else {
      out_raw = make_uint2(0u, 0u);
      out_sc4 = 0x7F7F7F7Fu;
    }
    if (tid < 16 && (kv_st + tid) < seq_len) {
      const uint32_t s4 = *reinterpret_cast<const uint32_t*>(blk_ptr + (int64_t)(base_in_blk + tid) * kv_s1 + 64);
      const int e0 = (int)(s4 & 0xFFu), e1 = (int)((s4 >> 8) & 0xFFu);
      const int e2 = (int)((s4 >> 16) & 0xFFu), e3 = (int)((s4 >> 24) & 0xFFu);
      const int me = max(max(e0, e1), max(e2, e3));
      out_tok_sc = __uint_as_float((uint32_t)me << 23);
    } else {
      out_tok_sc = 0.0f;
    }
  };

  int kv_start = split * 16;
  if ((kv_start / block_size) >= max_blocks) return;
  uint2 raw_cur = make_uint2(0u, 0u);
  uint32_t sc4_cur = 0x7F7F7F7Fu;
  float tsc_cur = 0.0f;
  fetch_mxfp4_tile(kv_start, raw_cur, sc4_cur, tsc_cur);

  for (; kv_start < seq_len && (kv_start / block_size) < max_blocks; kv_start += stride_kv) {
    if (kv_start + kr < seq_len) {
      const int e0 = (int)(sc4_cur & 0xFFu), e1 = (int)((sc4_cur >> 8) & 0xFFu);
      const int e2 = (int)((sc4_cur >> 16) & 0xFFu), e3 = (int)((sc4_cur >> 24) & 0xFFu);
      const int me = max(max(e0, e1), max(e2, e3));
      const int eg = (int)((sc4_cur >> (grp * 8)) & 0xFFu);
      const int re = 127 + eg - me;
      const float rsc = (re > 0) ? __uint_as_float((uint32_t)re << 23) : 0.0f;
      uint32_t pk[8];
#pragma unroll
      for (int bi = 0; bi < 4; ++bi) {
        const uint32_t b0 = (raw_cur.x >> (bi * 8)) & 0xFFu;
        const uint32_t b1 = (raw_cur.y >> (bi * 8)) & 0xFFu;
        const uint32_t h0a = (uint32_t)__builtin_bit_cast(uint16_t, (_Float16)(e2m1_lut[b0 & 0xFu] * rsc));
        const uint32_t h1a = (uint32_t)__builtin_bit_cast(uint16_t, (_Float16)(e2m1_lut[b0 >> 4] * rsc));
        const uint32_t h0b = (uint32_t)__builtin_bit_cast(uint16_t, (_Float16)(e2m1_lut[b1 & 0xFu] * rsc));
        const uint32_t h1b = (uint32_t)__builtin_bit_cast(uint16_t, (_Float16)(e2m1_lut[b1 >> 4] * rsc));
        pk[bi] = h0a | (h1a << 16);
        pk[bi + 4] = h0b | (h1b << 16);
      }
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16 + 0]) = (u4v){pk[0], pk[1], pk[2], pk[3]};
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16 + 8]) = (u4v){pk[4], pk[5], pk[6], pk[7]};
    } else {
      const u4v z = {0u, 0u, 0u, 0u};
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16]) = z;
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16 + 8]) = z;
    }
    __syncthreads();

    const int kv_next = kv_start + stride_kv;
    uint2 raw_nxt = make_uint2(0u, 0u);
    uint32_t sc4_nxt = 0x7F7F7F7Fu;
    float tsc_nxt = 0.0f;
    if (kv_next < seq_len && (kv_next / block_size) < max_blocks) {
      fetch_mxfp4_tile(kv_next, raw_nxt, sc4_nxt, tsc_nxt);
    }
    __builtin_amdgcn_sched_barrier(0);

    f32x8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int step = 0; step < 8; ++step) {
      const int d = step * 16;
      const u4v k0 = *reinterpret_cast<const u4va*>(&K_lds[l16][d]);
      const u4v k1 = *reinterpret_cast<const u4va*>(&K_lds[l16][d + 8]);
      const f16x16 kf = __builtin_bit_cast(f16x16, (u8v){k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w});
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(q_reg[step], kf, acc);
      __builtin_amdgcn_sched_barrier(0);
    }

    float sum_h = 0.0f;
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      sum_h = __builtin_fmaf(fmaxf(acc[ei], 0.0f), s_weights[h0 + 2 * ei + hrow], sum_h);
    }
    sum_h += shx_f<16>(sum_h);
    if (hrow == 0) s_wsum[wave][l16] = sum_h;
    __syncthreads();

    if (tid < 16) {
      const int pos = kv_start + tid;
      if (pos < seq_len) {
        const float tot = (s_wsum[0][tid] + s_wsum[1][tid] + s_wsum[2][tid] + s_wsum[3][tid]) * tsc_cur;
        out_logits[(int64_t)row * out_s0 + pos] = tot;
      }
    }
    __syncthreads();
    raw_cur = raw_nxt;
    sc4_cur = sc4_nxt;
    tsc_cur = tsc_nxt;
  }
}


// ================================================================== 5B-MXFP4: Native FlashIndexer (H=8) MXFP4 Paged MQA Logits Decode
typedef uint32_t u2v_a4 __attribute__((ext_vector_type(2), aligned(4)));

__global__ __launch_bounds__(128) void k_ernie_fp8_mxfp4_paged_mqa_logits_decode_h8(
    const uint8_t* __restrict__ q_fp8,       // [B, NEXT_N, 8, 128]
    const uint8_t* __restrict__ kv_ptr,      // [num_blocks, block_size, 68] uint8
    const float* __restrict__ weights,       // [B*NEXT_N, 8] fp32
    const int32_t* __restrict__ ctx_lens,    // [B*NEXT_N] or [B] int32
    const int32_t* __restrict__ block_tables,// [B, max_blocks] int32
    float* __restrict__ out_logits,          // [B*NEXT_N, max_model_len] fp32
    int64_t q_sb, int64_t q_sn, int64_t q_sh,
    int64_t kv_s0, int64_t kv_s1, int64_t w_s0, int64_t bt_s0, int64_t out_s0,
    int block_size, int blk_shift, int max_blocks, int max_model_len,
    int next_n, int n_splits, int ctx_per_row, int is_fnuz) {
  __shared__ _Float16 fp8_lut[256];
  __shared__ float e2m1_lut[16];
  __shared__ __attribute__((aligned(16))) _Float16 K_lds[16][136];
  __shared__ float s_weights[8][8];
  __shared__ int32_t s_sl[8];
  __shared__ float s_tsc[16];

  const int b = blockIdx.x;
  const int split = blockIdx.y;
  const int chunk = blockIdx.z;
  const int n_base = chunk * 8;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;

  if (tid < 16) {
    static constexpr float E2M1_C[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    e2m1_lut[tid] = E2M1_C[tid];
  }
  if (tid < 8) {
    const int n_i = n_base + tid;
    int sl = 0;
    if (n_i < next_n) {
      const int r_i = b * next_n + n_i;
      sl = ctx_per_row ? ctx_lens[r_i] : (ctx_lens[b] - next_n + n_i + 1);
      if (sl < 0) sl = 0;
      if (sl > max_model_len) sl = max_model_len;
    }
    s_sl[tid] = sl;
  }
  fp8_lut[tid] = fp8_byte_to_f16((uint32_t)tid, is_fnuz);
  fp8_lut[tid + 128] = fp8_byte_to_f16((uint32_t)(tid + 128), is_fnuz);
  if (tid < 64) {
    const int r_loc = tid >> 3;
    const int h_loc = tid & 7;
    const int n_i = n_base + r_loc;
    s_weights[r_loc][h_loc] = (n_i < next_n) ? weights[(int64_t)(b * next_n + n_i) * w_s0 + h_loc] : 0.0f;
  }
  __syncthreads();

  int max_sl = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) max_sl = max(max_sl, s_sl[i]);
  if (split * 16 >= max_sl) return;

  const int r_q = wave * 2 + (l16 >> 3);
  const int h_q = l16 & 7;
  const int n_q = n_base + r_q;
  const bool q_valid = (n_q < next_n);
  const uint8_t* qp = q_valid ? (q_fp8 + (int64_t)b * q_sb + (int64_t)n_q * q_sn + (int64_t)h_q * q_sh) : nullptr;
  f16x16 q_reg[8];
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const u4v raw0 = q_valid ? *reinterpret_cast<const u4va*>(qp + (pair * 2 + 0) * 16) : (u4v){0u,0u,0u,0u};
    const u4v raw1 = q_valid ? *reinterpret_cast<const u4va*>(qp + (pair * 2 + 1) * 16) : (u4v){0u,0u,0u,0u};
    const u4v raws[2] = {raw0, raw1};
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      const uint32_t ws[4] = {raws[s].x, raws[s].y, raws[s].z, raws[s].w};
      f16x16 qf;
#pragma unroll
      for (int w4 = 0; w4 < 4; ++w4) {
        const uint32_t x = ws[w4];
        qf[w4 * 4 + 0] = q_valid ? fp8_lut[x & 0xFFu] : (_Float16)0.0f;
        qf[w4 * 4 + 1] = q_valid ? fp8_lut[(x >> 8) & 0xFFu] : (_Float16)0.0f;
        qf[w4 * 4 + 2] = q_valid ? fp8_lut[(x >> 16) & 0xFFu] : (_Float16)0.0f;
        qf[w4 * 4 + 3] = q_valid ? fp8_lut[(x >> 24) & 0xFFu] : (_Float16)0.0f;
      }
      q_reg[pair * 2 + s] = qf;
    }
    __builtin_amdgcn_sched_barrier(0);
  }

  const int kr = tid >> 3;
  const int ht = tid & 7;
  const int grp = ht >> 1;
  const int kc16 = ht * 16;
  const int stride_kv = n_splits * 16;
  const int blk_mask = block_size - 1;

  auto fetch_mxfp4_tile = [&](int kv_st, u2v_a4& out_raw, uint32_t& out_sc4) {
    const int logical_blk = kv_st >> blk_shift;
    const int64_t page = (int64_t)block_tables[(int64_t)b * bt_s0 + logical_blk];
    const int base_in_blk = kv_st & blk_mask;
    const uint8_t* blk_ptr = kv_ptr + page * kv_s0;
    if (kv_st + kr < max_sl) {
      const uint8_t* rp = blk_ptr + (int64_t)(base_in_blk + kr) * kv_s1;
      out_raw = __builtin_nontemporal_load(reinterpret_cast<const u2v_a4*>(rp + ht * 8));
      out_sc4 = __builtin_nontemporal_load(reinterpret_cast<const uint32_t*>(rp + 64));
    } else {
      out_raw = (u2v_a4){0u, 0u};
      out_sc4 = 0x7F7F7F7Fu;
    }
  };

  int kv_start = split * 16;
  if ((kv_start >> blk_shift) >= max_blocks) return;
  u2v_a4 raw_cur = {0u, 0u};
  uint32_t sc4_cur = 0x7F7F7F7Fu;
  fetch_mxfp4_tile(kv_start, raw_cur, sc4_cur);

  const int n_out0 = n_base + wave * 2 + 0;
  const int n_out1 = n_base + wave * 2 + 1;
  const int sl0 = s_sl[wave * 2 + 0];
  const int sl1 = s_sl[wave * 2 + 1];

  for (; kv_start < max_sl && (kv_start >> blk_shift) < max_blocks; kv_start += stride_kv) {
    if (kv_start + kr < max_sl) {
      const int e0 = (int)(sc4_cur & 0xFFu), e1 = (int)((sc4_cur >> 8) & 0xFFu);
      const int e2 = (int)((sc4_cur >> 16) & 0xFFu), e3 = (int)((sc4_cur >> 24) & 0xFFu);
      const int me = max(max(e0, e1), max(e2, e3));
      if (ht == 0) s_tsc[kr] = __uint_as_float((uint32_t)me << 23);
      const int eg = (int)((sc4_cur >> (grp * 8)) & 0xFFu);
      const int re = 127 + eg - me;
      const float rsc = (re > 0) ? __uint_as_float((uint32_t)re << 23) : 0.0f;
      uint32_t pk[8];
#pragma unroll
      for (int bi = 0; bi < 4; ++bi) {
        const uint32_t b0 = (raw_cur[0] >> (bi * 8)) & 0xFFu;
        const uint32_t b1 = (raw_cur[1] >> (bi * 8)) & 0xFFu;
        const uint32_t h0a = (uint32_t)__builtin_bit_cast(uint16_t, (_Float16)(e2m1_lut[b0 & 0xFu] * rsc));
        const uint32_t h1a = (uint32_t)__builtin_bit_cast(uint16_t, (_Float16)(e2m1_lut[b0 >> 4] * rsc));
        const uint32_t h0b = (uint32_t)__builtin_bit_cast(uint16_t, (_Float16)(e2m1_lut[b1 & 0xFu] * rsc));
        const uint32_t h1b = (uint32_t)__builtin_bit_cast(uint16_t, (_Float16)(e2m1_lut[b1 >> 4] * rsc));
        pk[bi] = h0a | (h1a << 16);
        pk[bi + 4] = h0b | (h1b << 16);
      }
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16 + 0]) = (u4v){pk[0], pk[1], pk[2], pk[3]};
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16 + 8]) = (u4v){pk[4], pk[5], pk[6], pk[7]};
    } else {
      if (ht == 0) s_tsc[kr] = 0.0f;
      const u4v z = {0u, 0u, 0u, 0u};
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16]) = z;
      *reinterpret_cast<u4va*>(&K_lds[kr][kc16 + 8]) = z;
    }
    __syncthreads();

    const int kv_next = kv_start + stride_kv;
    u2v_a4 raw_nxt = {0u, 0u};
    uint32_t sc4_nxt = 0x7F7F7F7Fu;
    if (kv_next < max_sl && (kv_next >> blk_shift) < max_blocks) {
      fetch_mxfp4_tile(kv_next, raw_nxt, sc4_nxt);
    }
    __builtin_amdgcn_sched_barrier(0);

    f32x8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int step = 0; step < 8; ++step) {
      const int d = step * 16;
      const u4v k0 = *reinterpret_cast<const u4va*>(&K_lds[l16][d]);
      const u4v k1 = *reinterpret_cast<const u4va*>(&K_lds[l16][d + 8]);
      const f16x16 kf = __builtin_bit_cast(f16x16, (u8v){k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w});
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(q_reg[step], kf, acc);
    }

    float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
    for (int ei = 0; ei < 4; ++ei) {
      sum0 = __builtin_fmaf(fmaxf(acc[ei], 0.0f), s_weights[wave * 2 + 0][2 * ei + hrow], sum0);
      sum1 = __builtin_fmaf(fmaxf(acc[ei + 4], 0.0f), s_weights[wave * 2 + 1][2 * ei + hrow], sum1);
    }
    sum0 += shx_f<16>(sum0);
    sum1 += shx_f<16>(sum1);
    if (hrow == 0) {
      const int pos = kv_start + l16;
      const float ks = s_tsc[l16];
      if (n_out0 < next_n && pos < sl0) {
        out_logits[(int64_t)(b * next_n + n_out0) * out_s0 + pos] = sum0 * ks;
      }
      if (n_out1 < next_n && pos < sl1) {
        out_logits[(int64_t)(b * next_n + n_out1) * out_s0 + pos] = sum1 * ks;
      }
    }
    __syncthreads();
    raw_cur = raw_nxt;
    sc4_cur = sc4_nxt;
  }
}

void ernie_fp8_mxfp4_paged_mqa_logits_decode(
    torch::Tensor q_fp8, torch::Tensor kv_cache_mxfp4, torch::Tensor weights,
    torch::Tensor context_lens, torch::Tensor block_tables,
    torch::Tensor out_logits, int64_t max_model_len) {
  TORCH_CHECK(q_fp8.is_cuda() && q_fp8.dim() == 4 && (q_fp8.size(2) == 64 || q_fp8.size(2) == 8) && q_fp8.size(3) == 128);
  auto kv3 = (kv_cache_mxfp4.dim() == 4 && kv_cache_mxfp4.size(2) == 1)
      ? kv_cache_mxfp4.squeeze(2) : kv_cache_mxfp4;
  TORCH_CHECK(kv3.is_cuda() && kv3.dim() == 3 && kv3.scalar_type() == torch::kUInt8 && kv3.size(2) == 68);
  const int B = (int)q_fp8.size(0);
  const int next_n = (int)q_fp8.size(1);
  const int rows = B * next_n;
  if (rows == 0 || max_model_len == 0) return;
  const int block_size = (int)kv3.size(1);
  TORCH_CHECK(block_size % 16 == 0);
  auto cl = context_lens.reshape({-1}).contiguous();
  const int ctx_per_row = !(next_n > 1 && cl.numel() == B) ? 1 : 0;
  const int is_fnuz = (q_fp8.scalar_type() == torch::kFloat8_e4m3fnuz) ? 1 : 0;
  const int tiles_cap = ((int)max_model_len + 15) / 16;
  const int n_splits = max(1, min(40, min(tiles_cap, max(1, 512 / rows))));
  auto st = c10::hip::getCurrentHIPStream().stream();
  if (q_fp8.size(2) == 8 && (block_size & (block_size - 1)) == 0) {
    int blk_shift = 0;
    while ((1 << blk_shift) < block_size) ++blk_shift;
    const int n_chunks = (next_n + 7) / 8;
    const int g_x = B * n_chunks;
    const int n_sp = max(1, min(160, min(tiles_cap, max(40, 640 / g_x))));
    hipLaunchKernelGGL(
        k_ernie_fp8_mxfp4_paged_mqa_logits_decode_h8, dim3(B, n_sp, n_chunks), dim3(128), 0, st,
        reinterpret_cast<const uint8_t*>(q_fp8.data_ptr()),
        kv3.data_ptr<uint8_t>(), weights.data_ptr<float>(),
        cl.data_ptr<int32_t>(), block_tables.data_ptr<int32_t>(),
        out_logits.data_ptr<float>(),
        q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2),
        kv3.stride(0), kv3.stride(1), weights.stride(0), block_tables.stride(0), out_logits.stride(0),
        block_size, blk_shift, (int)block_tables.size(1), (int)max_model_len,
        next_n, n_sp, ctx_per_row, is_fnuz);
    return;
  }
  hipLaunchKernelGGL(
      k_ernie_fp8_mxfp4_paged_mqa_logits_decode, dim3(rows, n_splits), dim3(128), 0, st,
      reinterpret_cast<const uint8_t*>(q_fp8.data_ptr()),
      kv3.data_ptr<uint8_t>(), weights.data_ptr<float>(),
      cl.data_ptr<int32_t>(), block_tables.data_ptr<int32_t>(),
      out_logits.data_ptr<float>(),
      q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2),
      kv3.stride(0), kv3.stride(1), weights.stride(0), block_tables.stride(0), out_logits.stride(0),
      block_size, (int)block_tables.size(1), (int)max_model_len,
      next_n, n_splits, ctx_per_row, is_fnuz);
}

PYBIND11_MODULE(ernie_dsa_attn, m) {
  m.def("dsa_sparse_mla", &ernie_dsa_sparse_mla);
  m.def("fp8_mqa_logits_prefill", &ernie_fp8_mqa_logits_prefill);
  m.def("fp8_paged_mqa_logits_decode", &ernie_fp8_paged_mqa_logits_decode);
  m.def("dsa_sparse_mla_quant", &ernie_dsa_sparse_mla_quant);
  m.def("fp8_mxfp4_paged_mqa_logits_decode", &ernie_fp8_mxfp4_paged_mqa_logits_decode);
}

