// Native gfx1151 (RDNA3.5) Wave32 WMMA + Split-KV HIP kernel for ERNIE-5.5 HCA Decode (37 main + 3 DSpark layers).
// Replaces Triton _sparse_attn_decode_ragged_kernel on gfx1151 (which ran with 4 WGs idling 90% of CUs,
// padded 448->512, allocated FP32 output + copied to BF16 37x/tok, and omitted fused inverse RoPE!).
//
// Exact ERNIE-5.5 HCA geometry: H = 64, NOPE = 448 (7x64 FP8 + 7 UE8M0 scales), ROPE = 64 (BF16), HEAD_DIM = 512.
//   - Zero 448->512 padding waste, zero per-layer scratch allocations
//   - Fast 1-cycle UE8M0 scale decode ((uint32_t)exp << 23) + 256-entry FP8->FP32 LDS LUT
//   - Hardware Wave32 BF16 WMMA (__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32), 64 VGPR O_acc, ZERO SPILLS
//   - Split-KV (S=8) saturates all 40 CUs across both SWA (main) + C128 compressed (extra) caches
//   - Fused attn_sink + Fused Inverse GPT-J RoPE directly in registers in split-reduce epilogue!
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <math.h>

typedef short bf16x16 __attribute__((ext_vector_type(16)));
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

// Branchless finite-input BF16x2 RNE pack (fp8_lut & ue8m0_to_f32 already map NaN -> 0.0f; saves 64 EXEC-mask SGPR ops!)
__device__ __forceinline__ uint32_t f32_to_bf16x2_rne_finite(float a, float b) {
  uint32_t ua = __float_as_uint(a);
  uint32_t ub = __float_as_uint(b);
  ua += 0x7fffu + ((ua >> 16) & 1u);
  ub += 0x7fffu + ((ub >> 16) & 1u);
  return __builtin_amdgcn_perm(ub, ua, 0x07060302u);
}

struct HcaPassArgs {
  const uint8_t* cache_ptr;
  const int32_t* indices;
  const int32_t* indptr;
  int64_t stride0;
  int num_rows;
  int block_size;
};

#define HCA_NOPE 448
#define HCA_ROPE 64
#define HCA_DIM 512
#define HCA_STRIDE 520   // +8 bf16 pad for zero LDS bank conflicts
#define HCA_ROW_BYTES 576
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
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)0u; // NaN -> 0 matching Triton nan guard
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float fp8_byte_to_f32(uint32_t b, int is_fnuz) {
  const uint32_t sign = (b >> 7) & 1u;
  const uint32_t exp = (b >> 3) & 0xFu;
  const uint32_t man = b & 0x7u;
  float val = 0.0f;
  if (is_fnuz) {
    if (exp == 0) val = (sign && man == 0) ? 0.0f : ldexpf((float)man, -10);
    else val = ldexpf(1.0f + (float)man * 0.125f, (int)exp - 8);
    if (sign && !(exp == 0 && man == 0)) val = -val;
  } else {
    if (exp == 15 && man == 7) val = 0.0f; // NaN -> 0 matching Triton nan guard
    else if (exp == 0) { val = ldexpf((float)man, -9); if (sign) val = -val; }
    else { val = ldexpf(1.0f + (float)man * 0.125f, (int)exp - 7); if (sign) val = -val; }
  }
  return val;
}
__device__ __forceinline__ float ue8m0_to_f32(uint8_t e) {
  if (e == 255u) return 0.0f;
  if (e == 0u) return ldexpf(1.0f, -127);
  return __uint_as_float((uint32_t)e << 23);
}

// Helper: decode 16 KV tokens from an fp8_ds_mla UE8M0 paged cache into KV_lds[16][520]
template <int IS_FNUZ>
__device__ __forceinline__ void load_decode_fp8_kv_block(
    const uint8_t* __restrict__ cache_ptr, int64_t cache_stride0, int block_size,
    const int32_t* s_slots, const uint8_t* s_valid,
    const float* fp8_lut, uint16_t (*KV_lds)[HCA_STRIDE], int tid) {
  // 128 threads cooperatively decode 16 tokens x 512 dims (4 dims/thread? Wait: 16*512 = 8192 -> 64 dims/thread!)
  // Map: 8 threads per token (tid >> 3 = kr in 0..15, sub = tid & 7 in 0..7):
  //   sub = 0..6 -> NOPE block `sub` (64 FP8 bytes + 1 UE8M0 scale byte!)
  //   sub = 7    -> ROPE block (64 BF16 values = 128 bytes!)
  const int kr = tid >> 3;    // 0..15
  const int sub = tid & 7;    // 0..7
  const int32_t sl = s_slots[kr];
  if (sl < 0) {
    u4v z = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      *reinterpret_cast<u4va*>(&KV_lds[kr][sub * 64 + c * 8]) = z;
    }
    return;
  }
  const int blk_i = sl / block_size;
  const int pos_i = sl % block_size;
  const uint8_t* blk_ptr = cache_ptr + (int64_t)blk_i * cache_stride0;
  const uint8_t* tok_data = blk_ptr + (int64_t)pos_i * HCA_ROW_BYTES;

  if (sub < 7) {
    const uint8_t* tok_sc = blk_ptr + (int64_t)block_size * HCA_ROW_BYTES + (int64_t)pos_i * 8;
    const float sc = ue8m0_to_f32(tok_sc[sub]);
    const uint8_t* np = tok_data + sub * 64;
    // 2x 16B NT burst per pair (halves live load VGPRs vs 4x while keeping 2x MLP + sched_barrier!)
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      const u4v raws[2] = {
        __builtin_nontemporal_load(reinterpret_cast<const u4va*>(np + pair * 32 + 0)),
        __builtin_nontemporal_load(reinterpret_cast<const u4va*>(np + pair * 32 + 16)),
      };
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int chunk = pair * 2 + c;
        const u4v raw = raws[c];
        const uint32_t ws[4] = {raw.x, raw.y, raw.z, raw.w};
        uint32_t pk[8];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          const uint32_t x = ws[w];
          pk[w * 2 + 0] = f32_to_bf16x2_rne_finite(fp8_lut[x & 0xFFu] * sc,         fp8_lut[(x >> 8) & 0xFFu] * sc);
          pk[w * 2 + 1] = f32_to_bf16x2_rne_finite(fp8_lut[(x >> 16) & 0xFFu] * sc, fp8_lut[(x >> 24) & 0xFFu] * sc);
        }
        *reinterpret_cast<u4va*>(&KV_lds[kr][sub * 64 + chunk * 16 + 0]) = (u4v){pk[0], pk[1], pk[2], pk[3]};
        *reinterpret_cast<u4va*>(&KV_lds[kr][sub * 64 + chunk * 16 + 8]) = (u4v){pk[4], pk[5], pk[6], pk[7]};
      }
      __builtin_amdgcn_sched_barrier(0);
    }
  } else {
    // sub == 7: 64 BF16 rope values = 128 bytes = 4x (2x uint4 NT burst), guarding NaN -> 0 (avoids 32-VGPR hoist!)
    const uint16_t* rp = reinterpret_cast<const uint16_t*>(tok_data + HCA_NOPE);
#pragma unroll 1
    for (int pair = 0; pair < 4; ++pair) {
      u4v v0 = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(rp + (pair * 2 + 0) * 8));
      u4v v1 = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(rp + (pair * 2 + 1) * 8));
      uint16_t* bp0 = reinterpret_cast<uint16_t*>(&v0);
      uint16_t* bp1 = reinterpret_cast<uint16_t*>(&v1);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        if ((bp0[k] & 0x7FFFu) > 0x7F80u) bp0[k] = 0u;
        if ((bp1[k] & 0x7FFFu) > 0x7F80u) bp1[k] = 0u;
      }
      *reinterpret_cast<u4va*>(&KV_lds[kr][HCA_NOPE + (pair * 2 + 0) * 8]) = v0;
      *reinterpret_cast<u4va*>(&KV_lds[kr][HCA_NOPE + (pair * 2 + 1) * 8]) = v1;
      __builtin_amdgcn_sched_barrier(0);
    }
  }
}

// Grid: (T, num_head_groups, NUM_SPLITS), Block: 128 (4 Wave32s)
template <int NUM_SPLITS, int HAS_EXTRA>
__global__ __launch_bounds__(128) void k_ernie_hca_decode_partial(
    const uint16_t* __restrict__ q_ptr,        // [T, H, 512] bf16
    const uint8_t* __restrict__ main_cache,    // [B_m, BS_m, ...] uint8
    const int32_t* __restrict__ main_indices,  // [nnz_m] int32
    const int32_t* __restrict__ main_indptr,   // [T+1] int32
    const uint8_t* __restrict__ extra_cache,   // [B_e, BS_e, ...] uint8
    const int32_t* __restrict__ extra_indices, // [nnz_e] int32
    const int32_t* __restrict__ extra_indptr,  // [T+1] int32
    float* __restrict__ part_o,                // [T, H, NUM_SPLITS, 512] fp32
    float* __restrict__ part_m,                // [T, H, NUM_SPLITS] fp32
    float* __restrict__ part_l,                // [T, H, NUM_SPLITS] fp32
    int64_t q_s0, int64_t q_s1,
    int64_t main_s0, int64_t extra_s0,
    int main_num_rows, int extra_num_rows,
    int main_bs, int extra_bs,
    int num_heads, float scale, int is_fnuz_main) {
  __shared__ float fp8_lut_main[256];
  __shared__ float fp8_lut_ocp[256];
  __shared__ __attribute__((aligned(16))) uint16_t KV_lds[16][HCA_STRIDE]; // 16.25 KiB
  __shared__ float S_part[4][16][16];                                       // 4.00 KiB
  __shared__ uint16_t P_lds[16][16];                                        // 0.50 KiB
  __shared__ int32_t s_slots[16];
  __shared__ uint8_t s_valid[16];
  __shared__ float s_alpha[16];
  __shared__ float s_m[16];
  __shared__ float s_l[16];
  __shared__ HcaPassArgs s_pass[2]; // 80B LDS -> frees ~24 SGPRs across outer pass loop (0 SGPR spills!)

  const int t = blockIdx.x;
  const int hg = blockIdx.y;
  const int sp = blockIdx.z;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;
  const int h0 = hg * 16;

  // Build LUTs (2 entries/thread) & stash pass metadata in LDS to keep SGPRs < 106
  fp8_lut_main[tid] = fp8_byte_to_f32((uint32_t)tid, is_fnuz_main);
  fp8_lut_main[tid + 128] = fp8_byte_to_f32((uint32_t)(tid + 128), is_fnuz_main);
  if constexpr (HAS_EXTRA) {
    fp8_lut_ocp[tid] = fp8_byte_to_f32((uint32_t)tid, 0);
    fp8_lut_ocp[tid + 128] = fp8_byte_to_f32((uint32_t)(tid + 128), 0);
  }
  if (tid < 16) {
    s_m[tid] = NEG_INF;
    s_l[tid] = 0.0f;
  }
  if (tid == 0) {
    s_pass[0] = {main_cache, main_indices, main_indptr, main_s0, main_num_rows, main_bs};
    if constexpr (HAS_EXTRA) {
      s_pass[1] = {extra_cache, extra_indices, extra_indptr, extra_s0, extra_num_rows, extra_bs};
    }
  }
  __syncthreads();

  f32x8 o_acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) o_acc[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  const bool q_hvalid = (h0 + l16) < num_heads;
  const uint16_t* q_base = q_ptr + (int64_t)t * q_s0 + (int64_t)(q_hvalid ? (h0 + l16) : 0) * q_s1;
  const int d_start = wave * 128;

  // Helper lambda for one 16-token KV block already loaded into KV_lds
  auto process_kv_block = [&]() {
    // 1. Q @ K^T: each wave handles 128 dims (8 WMMA steps; 4*128 = 512 exact!)
    // Unroll-1 + sched_barrier keeps Q/K live range minimal alongside 64-VGPR o_acc[8] -> higher CU occupancy!
    f32x8 s_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 1
    for (int step = 0; step < 8; ++step) {
      const int d = d_start + step * 16;
      u4v q0 = {0u, 0u, 0u, 0u}, q1 = {0u, 0u, 0u, 0u};
      if (q_hvalid) {
        q0 = *reinterpret_cast<const u4va*>(q_base + d);
        q1 = *reinterpret_cast<const u4va*>(q_base + d + 8);
      }
      const bf16x16 qf = __builtin_bit_cast(bf16x16, (u8v){q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w});
      const u4v k0 = *reinterpret_cast<const u4va*>(&KV_lds[l16][d]);
      const u4v k1 = *reinterpret_cast<const u4va*>(&KV_lds[l16][d + 8]);
      const bf16x16 kf = __builtin_bit_cast(bf16x16, (u8v){k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w});
      s_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(qf, kf, s_acc);
      __builtin_amdgcn_sched_barrier(0);
    }
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) S_part[wave][2 * ei + hrow][l16] = s_acc[ei];
    __syncthreads();

    // 2. Online softmax in Wave 0 via DPP
    if (wave == 0) {
      const bool vld = (s_valid[l16] != 0);
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) {
        const int hm = 2 * ei + hrow;
        float sc = S_part[0][hm][l16] + S_part[1][hm][l16] + S_part[2][hm][l16] + S_part[3][hm][l16];
        sc = (vld && (h0 + hm) < num_heads) ? (sc * scale) : NEG_INF;
        const float bmax = halfwave_max16(sc);
        const float mold = s_m[hm];
        const float mnew = fmaxf(mold, bmax);
        const float alpha = (mold > -1e30f) ? exp2f((mold - mnew) * LOG2E) : 0.0f;
        const float pv = (vld && (h0 + hm) < num_heads && mnew > -1e30f) ? exp2f((sc - mnew) * LOG2E) : 0.0f;
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

    // 3. Rescale o_acc & P @ V (wave w owns V cols 128*w .. 128*w + 127)
    float al[8];
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) al[ei] = s_alpha[2 * ei + hrow];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) o_acc[j][ei] *= al[ei];
    }

    const u4v p0 = *reinterpret_cast<const u4va*>(&P_lds[l16][0]);
    const u4v p1 = *reinterpret_cast<const u4va*>(&P_lds[l16][8]);
    const bf16x16 pf = __builtin_bit_cast(bf16x16, (u8v){p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w});
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int vc = d_start + j * 16 + l16;
      bf16x16 vf;
#pragma unroll
      for (int kr = 0; kr < 16; ++kr) vf[kr] = (short)KV_lds[kr][vc];
      o_acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf, vf, o_acc[j]);
      __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();
  };

  // Unified loop over Pass 0 (main SWA cache) and Pass 1 (extra C128 cache if HAS_EXTRA):
  // Emits load_decode_fp8_kv_block + process_kv_block ONCE (halves code size & trims VGPRs for higher CU occupancy!)
  constexpr int NUM_PASSES = HAS_EXTRA ? 2 : 1;
#pragma nounroll
  for (int pass = 0; pass < NUM_PASSES; ++pass) {
    const HcaPassArgs pa = s_pass[pass];
    const float*   c_lut  = (pass == 0) ? fp8_lut_main : fp8_lut_ocp;

    const int p_start = pa.indptr[t];
    const int p_len   = pa.indptr[t + 1] - p_start;
    const int p_chunk = (p_len + NUM_SPLITS - 1) / NUM_SPLITS;
    const int p_lo    = sp * p_chunk;
    const int p_hi    = min(p_lo + p_chunk, p_len);
    for (int k0 = p_lo; k0 < p_hi; k0 += 16) {
      if (tid < 16) {
        const int kp = k0 + tid;
        int32_t sl = (kp < p_hi) ? pa.indices[p_start + kp] : -1;
        if (sl < 0 || sl >= pa.num_rows) sl = -1;
        s_slots[tid] = sl;
        s_valid[tid] = (sl >= 0) ? 1 : 0;
      }
      __syncthreads();
      unsigned any_v = 0;
#pragma unroll
      for (int i = 0; i < 16; ++i) any_v |= s_valid[i];
      if (!any_v) continue;
      load_decode_fp8_kv_block<0>(pa.cache_ptr, pa.stride0, pa.block_size, s_slots, s_valid, c_lut, KV_lds, tid);
      __syncthreads();
      process_kv_block();
    }
  }

  // Store partial (m, l, o_acc)
  if (wave == 0 && l16 == 0) {
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      const int h = h0 + 2 * ei + hrow;
      if (h < num_heads) {
        const int64_t off = ((int64_t)t * num_heads + h) * NUM_SPLITS + sp;
        part_m[off] = s_m[2 * ei + hrow];
        part_l[off] = s_l[2 * ei + hrow];
      }
    }
  }
#pragma unroll
  for (int ei = 0; ei < 8; ++ei) {
    const int h = h0 + 2 * ei + hrow;
    if (h < num_heads) {
      float* dst = part_o + (((int64_t)t * num_heads + h) * NUM_SPLITS + sp) * HCA_DIM + d_start + l16;
#pragma unroll
      for (int j = 0; j < 8; ++j) dst[j * 16] = o_acc[j][ei];
    }
  }
}

// Split-KV reduce with FUSED attn_sink + FUSED inverse GPT-J RoPE in registers!
// Grid = (T, num_heads), Block = 128 (4 dims/thread: 128-bit f4v loads + v_perm_b32 uint2 store = 4x fewer VMEM ops!)
template <int NUM_SPLITS, int FUSE_INV_ROPE, int CS_IS_BF16 = 0>
__global__ __launch_bounds__(128) void k_ernie_hca_split_reduce(
    const float* __restrict__ part_o,
    const float* __restrict__ part_m,
    const float* __restrict__ part_l,
    const float* __restrict__ sink_ptr,
    const int64_t* __restrict__ pos_ptr,
    const void* __restrict__ cos_sin_ptr,
    uint16_t* __restrict__ out_ptr,
    int64_t out_s0, int64_t out_s1, int64_t cs_s0, int num_heads) {
  const int t = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int d0 = tid * 4;
  const int64_t base_ml = ((int64_t)t * num_heads + h) * NUM_SPLITS;

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
  const float inv_l = (l_tot > 0.0f) ? __builtin_amdgcn_rcpf(fmaxf(l_tot, 1.0e-30f)) : 0.0f;

  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
  for (int s = 0; s < NUM_SPLITS; ++s) {
    if (ws[s] > 0.0f) {
      const f4v po = *reinterpret_cast<const f4v*>(part_o + (base_ml + s) * HCA_DIM + d0);
      a0 = __builtin_fmaf(ws[s], po.x, a0);
      a1 = __builtin_fmaf(ws[s], po.y, a1);
      a2 = __builtin_fmaf(ws[s], po.z, a2);
      a3 = __builtin_fmaf(ws[s], po.w, a3);
    }
  }
  float e0 = a0 * inv_l, o0 = a1 * inv_l;
  float e1 = a2 * inv_l, o1 = a3 * inv_l;

  if constexpr (FUSE_INV_ROPE) {
    if (tid >= (HCA_NOPE / 4)) {
      const int k0 = (tid - (HCA_NOPE / 4)) * 2; // 0, 2, ..., 30
      const int64_t pos = pos_ptr[t];
      const int64_t cs_base = pos * cs_s0;
      float c0, s0, c1, s1;
      if constexpr (CS_IS_BF16) {
        const uint16_t* cs_row = reinterpret_cast<const uint16_t*>(cos_sin_ptr) + cs_base;
        c0 = __uint_as_float((uint32_t)cs_row[k0] << 16);
        s0 = __uint_as_float((uint32_t)cs_row[(HCA_ROPE / 2) + k0] << 16);
        c1 = __uint_as_float((uint32_t)cs_row[k0 + 1] << 16);
        s1 = __uint_as_float((uint32_t)cs_row[(HCA_ROPE / 2) + k0 + 1] << 16);
      } else {
        const float* cs_row = reinterpret_cast<const float*>(cos_sin_ptr) + cs_base;
        c0 = cs_row[k0];     s0 = cs_row[(HCA_ROPE / 2) + k0];
        c1 = cs_row[k0 + 1]; s1 = cs_row[(HCA_ROPE / 2) + k0 + 1];
      }
      const float r0 = e0 * c0 + o0 * s0, r1 = o0 * c0 - e0 * s0;
      const float r2 = e1 * c1 + o1 * s1, r3 = o1 * c1 - e1 * s1;
      e0 = r0; o0 = r1; e1 = r2; o1 = r3;
    }
  }
  uint2 out_p;
  out_p.x = f32_to_bf16x2_rne(e0, o0);
  out_p.y = f32_to_bf16x2_rne(e1, o1);
  *reinterpret_cast<uint2*>(out_ptr + (int64_t)t * out_s0 + (int64_t)h * out_s1 + d0) = out_p;
}

// ============================================================================
// v0.2.8 32-Head/WG Q-in-VGPRs + Union-LDS (KV/VT) 2-WG/CU HCA Decode Kernel!
//   - 256 threads (8 Wave32s = 2 head-groups of 16 x 4 D-quarters of 128)
//   - 32H/WG -> 2x fewer KV-cache global reads & 2x fewer FP8 dequants vs v27 (16H/WG)!
//   - 256-thread cooperative FP8 loader (16 threads/token x 32 dims/thread -> 2x MLP!)
//   - Q[16x128] loaded ONCE into q_reg[8] VGPRs -> ZERO Q global/L1 loads in loop!
//   - Union KV_lds[16][520] / VT_lds_g[2][16][264] + fp8_lut -> 27.9 KiB LDS (<32 KiB -> 2 WGs/CU!)
//   - Dynamic NUM_SPLITS (S=8 for T<=2, S=4 for 3<=T<=8, S=2 for T>8) -> 2-4x less part_o traffic!
// ============================================================================
__device__ __forceinline__ void load_decode_fp8_kv_block_256(
    const uint8_t* __restrict__ cache_ptr, int64_t cache_stride0, int block_size,
    const int32_t* s_slots, const float* fp8_lut,
    uint16_t (*KV_lds)[HCA_STRIDE], int tid) {
  // 256 threads = 16 tokens x 16 sub-chunks (32 dims/thread!)
  // sub16 = 0..13 -> NOPE block (sub16>>1) in 0..6, half (sub16&1) in 0..1 (32 FP8 bytes = 2x u4v!)
  // sub16 = 14..15 -> ROPE half (sub16-14) in 0..1 (32 BF16 = 4x u4v!)
  const int kr = tid >> 4;     // 0..15
  const int sub16 = tid & 15;  // 0..15
  const int32_t sl = s_slots[kr];
  if (sl < 0) {
    const u4v z = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      *reinterpret_cast<u4va*>(&KV_lds[kr][sub16 * 32 + c * 8]) = z;
    }
    return;
  }
  // Note: block_size is power-of-2 (e.g. 64) in vLLM paged KV; avoid integer div by using bit ops if pow2,
  // or __builtin_amdgcn_rcpf / exact mul-shift? Wait: isa_audit forbids v_div_* (float/int)!
  // Check: in v27 load_decode_fp8_kv_block, sl / block_size used int div -> wait: block_size is int32!
  // Let us use exact power-of-2 shift when block_size is power-of-2 (always 16/32/64/128 in vLLM!) with zero div!
  const int bs_shift = 31 - __clz(block_size);
  const int blk_i = sl >> bs_shift;
  const int pos_i = sl & (block_size - 1);
  const uint8_t* blk_ptr = cache_ptr + (int64_t)blk_i * cache_stride0;
  const uint8_t* tok_data = blk_ptr + (int64_t)pos_i * HCA_ROW_BYTES;

  if (sub16 < 14) {
    const int sub = sub16 >> 1;
    const int half = sub16 & 1;
    const uint8_t* tok_sc = blk_ptr + (int64_t)block_size * HCA_ROW_BYTES + (int64_t)pos_i * 8;
    const float sc = ue8m0_to_f32(tok_sc[sub]);
    const uint8_t* np = tok_data + sub * 64 + half * 32;
    const u4v r0 = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(np + 0));
    const u4v r1 = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(np + 16));
    const u4v raws[2] = {r0, r1};
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const uint32_t ws[4] = {raws[c].x, raws[c].y, raws[c].z, raws[c].w};
      uint32_t pk[8];
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        const uint32_t x = ws[w];
        pk[w * 2 + 0] = f32_to_bf16x2_rne_finite(fp8_lut[x & 0xFFu] * sc,         fp8_lut[(x >> 8) & 0xFFu] * sc);
        pk[w * 2 + 1] = f32_to_bf16x2_rne_finite(fp8_lut[(x >> 16) & 0xFFu] * sc, fp8_lut[(x >> 24) & 0xFFu] * sc);
      }
      *reinterpret_cast<u4va*>(&KV_lds[kr][sub16 * 32 + c * 16 + 0]) = (u4v){pk[0], pk[1], pk[2], pk[3]};
      *reinterpret_cast<u4va*>(&KV_lds[kr][sub16 * 32 + c * 16 + 8]) = (u4v){pk[4], pk[5], pk[6], pk[7]};
    }
  } else {
    const int rhalf = sub16 - 14;
    const uint16_t* rp = reinterpret_cast<const uint16_t*>(tok_data + HCA_NOPE) + rhalf * 32;
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      u4v v0 = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(rp + (pair * 2 + 0) * 8));
      u4v v1 = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(rp + (pair * 2 + 1) * 8));
      uint16_t* bp0 = reinterpret_cast<uint16_t*>(&v0);
      uint16_t* bp1 = reinterpret_cast<uint16_t*>(&v1);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        if ((bp0[k] & 0x7FFFu) > 0x7F80u) bp0[k] = 0u;
        if ((bp1[k] & 0x7FFFu) > 0x7F80u) bp1[k] = 0u;
      }
      *reinterpret_cast<u4va*>(&KV_lds[kr][HCA_NOPE + rhalf * 32 + (pair * 2 + 0) * 8]) = v0;
      *reinterpret_cast<u4va*>(&KV_lds[kr][HCA_NOPE + rhalf * 32 + (pair * 2 + 1) * 8]) = v1;
    }
  }
}

template <int NUM_SPLITS, int HAS_EXTRA>
__global__ __launch_bounds__(256, 2) void k_ernie_hca_decode_qreg_v28(
    const uint16_t* __restrict__ q_ptr,        // [T, H, 512] bf16
    const uint8_t* __restrict__ main_cache,
    const int32_t* __restrict__ main_indices,
    const int32_t* __restrict__ main_indptr,
    const uint8_t* __restrict__ extra_cache,
    const int32_t* __restrict__ extra_indices,
    const int32_t* __restrict__ extra_indptr,
    float* __restrict__ part_o,                // [T, H, NUM_SPLITS, 512] fp32
    float* __restrict__ part_m,                // [T, H, NUM_SPLITS] fp32
    float* __restrict__ part_l,                // [T, H, NUM_SPLITS] fp32
    int64_t q_s0, int64_t q_s1,
    int64_t main_s0, int64_t extra_s0,
    int main_num_rows, int extra_num_rows,
    int main_bs, int extra_bs,
    int num_heads, float scale, int is_fnuz_main) {
  __shared__ float fp8_lut_main[256];                                        //  1.00 KiB
  __shared__ float fp8_lut_ocp[256];                                         //  1.00 KiB
  __shared__ __attribute__((aligned(16))) union {
    uint16_t KV[16][HCA_STRIDE];      // 16 * 520 * 2 = 16,640 B
    uint16_t VT[512][18];             // 512 * 18 * 2 = 18,432 B (bank-conflict-free!)
  } smem;                                                                    // 16.50 KiB
  __shared__ float S_part[2][4][16][16];                                     //  8.00 KiB
  __shared__ uint16_t P_lds[2][16][16];                                      //  1.00 KiB
  __shared__ int32_t s_slots[16];
  __shared__ uint8_t s_valid[16];
  __shared__ float s_alpha[2][16];
  __shared__ float s_m[2][16];
  __shared__ float s_l[2][16];
  __shared__ HcaPassArgs s_pass[2];
  // Total LDS ~ 27.9 KiB (< 32 KiB -> 2 WGs/CU!)

  const int t = blockIdx.y;
  const int hg2 = blockIdx.x;        // 0..(H/32 - 1), each WG handles 32 heads (2 x 16)
  const int sp = blockIdx.z;
  const int tid = threadIdx.x;       // 0..255
  const int wave = tid >> 5;         // 0..7
  const int mg = wave >> 2;          // 0..1 (head-group of 16)
  const int dq = wave & 3;           // 0..3 (D-quarter of 128)
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;
  const int h0 = (hg2 * 2 + mg) * 16;
  const int d_start = dq * 128;

  // Init LUTs (1 entry/thread) & state
  fp8_lut_main[tid] = fp8_byte_to_f32((uint32_t)tid, is_fnuz_main);
  if constexpr (HAS_EXTRA) {
    fp8_lut_ocp[tid] = fp8_byte_to_f32((uint32_t)tid, 0);
  }
  if (tid < 32) {
    s_m[tid >> 4][tid & 15] = NEG_INF;
    s_l[tid >> 4][tid & 15] = 0.0f;
  }
  if (tid == 0) {
    s_pass[0] = {main_cache, main_indices, main_indptr, main_s0, main_num_rows, main_bs};
    if constexpr (HAS_EXTRA) {
      s_pass[1] = {extra_cache, extra_indices, extra_indptr, extra_s0, extra_num_rows, extra_bs};
    }
  }

  // Load Q[16 x 128] ONCE into VGPRs (q_reg[8] = 64 VGPRs -> ZERO Q global loads in loop!)
  const bool q_hvalid = (h0 + l16) < num_heads;
  const uint16_t* q_base = q_ptr + (int64_t)t * q_s0 + (int64_t)(q_hvalid ? (h0 + l16) : 0) * q_s1 + d_start;
  bf16x16 q_reg[8];
#pragma unroll
  for (int step = 0; step < 8; ++step) {
    u4v q0 = {0u, 0u, 0u, 0u}, q1 = {0u, 0u, 0u, 0u};
    if (q_hvalid) {
      q0 = *reinterpret_cast<const u4va*>(q_base + step * 16 + 0);
      q1 = *reinterpret_cast<const u4va*>(q_base + step * 16 + 8);
    }
    q_reg[step] = __builtin_bit_cast(bf16x16, (u8v){q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w});
  }
  __syncthreads();

  f32x8 o_acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) o_acc[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  constexpr int NUM_PASSES = HAS_EXTRA ? 2 : 1;
#pragma nounroll
  for (int pass = 0; pass < NUM_PASSES; ++pass) {
    const HcaPassArgs pa = s_pass[pass];
    const float* c_lut = (pass == 0) ? fp8_lut_main : fp8_lut_ocp;
    const int p_start = pa.indptr[t];
    const int p_len   = pa.indptr[t + 1] - p_start;
    const int p_chunk = (p_len + NUM_SPLITS - 1) / NUM_SPLITS;
    const int p_lo    = sp * p_chunk;
    const int p_hi    = min(p_lo + p_chunk, p_len);

    for (int k0 = p_lo; k0 < p_hi; k0 += 16) {
      if (tid < 16) {
        const int kp = k0 + tid;
        int32_t sl = (kp < p_hi) ? pa.indices[p_start + kp] : -1;
        if (sl < 0 || sl >= pa.num_rows) sl = -1;
        s_slots[tid] = sl;
        s_valid[tid] = (sl >= 0) ? 1 : 0;
      }
      __syncthreads();
      unsigned any_v = 0;
#pragma unroll
      for (int i = 0; i < 16; ++i) any_v |= s_valid[i];
      if (!any_v) continue;

      // 1. Cooperative 256-thread FP8 decode into smem.KV[16][520]
      load_decode_fp8_kv_block_256(pa.cache_ptr, pa.stride0, pa.block_size, s_slots, c_lut, smem.KV, tid);
      __syncthreads();

      // 2. Q @ K^T using Q 100% from q_reg[8] VGPRs!
      f32x8 s_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int step = 0; step < 8; ++step) {
        const int d = d_start + step * 16;
        const u4v k0v = *reinterpret_cast<const u4va*>(&smem.KV[l16][d]);
        const u4v k1v = *reinterpret_cast<const u4va*>(&smem.KV[l16][d + 8]);
        const bf16x16 kf = __builtin_bit_cast(bf16x16, (u8v){k0v.x, k0v.y, k0v.z, k0v.w, k1v.x, k1v.y, k1v.z, k1v.w});
        s_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(q_reg[step], kf, s_acc);
        __builtin_amdgcn_sched_barrier(0);
      }
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) S_part[mg][dq][2 * ei + hrow][l16] = s_acc[ei];

      // 3. Before overwriting smem.KV with smem.VT, each thread holds 2 cols x 16 rows of KV in VGPRs!
      //    256 threads x 2 cols = 512 cols exact!
      const int v_half = tid >> 7;        // 0..1 (cols 0..255 vs 256..511)
      const int v_c0 = (tid & 127) * 2;   // 0, 2, ..., 254 (even col in half)
      const int v_col = v_half * 256 + v_c0;
      uint32_t vt_tmp[16];
#pragma unroll
      for (int kr = 0; kr < 16; ++kr) {
        vt_tmp[kr] = *reinterpret_cast<const uint32_t*>(&smem.KV[kr][v_col]);
      }
      __syncthreads();

      // 4. Write transposed VT into smem.VT[v_half][kr][v_c0] -> Wait: for WMMA B operand,
      //    each lane l16 needs 16 contiguous bf16s for col = d_start + j*16 + l16 across kr=0..15!
      //    So store VT[v_half][col_in_half][kr] (256 x 16 = 8 KiB per half? 2 * 256 * 16 * 2B = 16 KiB!!).
      //    Wait: smem.VT[2][16][264] is 16,896 B; let us index as smem.VT_c[512][16] (16,384 B) with pad?
      //    Pad by +2 bf16? 512 * 18 * 2 = 18,432 B > 16,896 B -> use smem.VT_g[32][16][16] or simply
      //    store 16-col chunks: VT[d_grp 0..31][c 0..15][k 0..15] -> 32 * 16 * 16 * 2B = 16,384 B, stride 16 bf16 = 32B (8 banks)!
      //    Zero bank conflicts because lane l16 reads row l16 at 32B stride -> 8-bank stride? No, 16 bf16 = 8 u32 = 8 banks -> 2-way;
      //    with stride 18 bf16 = 9 u32 -> 18,432 B fits easily (total LDS = 29.4 KiB < 32 KiB -> still 2 WGs/CU!).
      //    Actually: check smem.VT[2][16][264] -> we can use union { KV[16][520]; uint16_t VT[512][18]; } = 18,432 B!
      //    Total LDS = 18.43 + 8 + 1 + 2 + 0.5 = 29.9 KiB < 32 KiB (2 WGs/CU!!).
      //    Wait: even simpler — store directly using vt_tmp[16] into VT[v_col + 0][kr] and VT[v_col + 1][kr]!
      uint16_t (*VT)[18] = reinterpret_cast<uint16_t (*)[18]>(&smem.KV[0][0]);
#pragma unroll
      for (int kr = 0; kr < 16; ++kr) {
        VT[v_col + 0][kr] = (uint16_t)(vt_tmp[kr] & 0xFFFFu);
        VT[v_col + 1][kr] = (uint16_t)(vt_tmp[kr] >> 16);
      }

      // 4-way parallel online softmax across ALL 4 dq waves (2 ei/wave instead of 8 on dq==0!)
      {
        const bool vld = (s_valid[l16] != 0);
#pragma unroll
        for (int e_sub = 0; e_sub < 2; ++e_sub) {
          const int ei = dq * 2 + e_sub;
          const int hm = 2 * ei + hrow;
          float sc = S_part[mg][0][hm][l16] + S_part[mg][1][hm][l16]
                   + S_part[mg][2][hm][l16] + S_part[mg][3][hm][l16];
          sc = (vld && (h0 + hm) < num_heads) ? (sc * scale) : NEG_INF;
          const float bmax = halfwave_max16(sc);
          const float mold = s_m[mg][hm];
          const float mnew = fmaxf(mold, bmax);
          const float alpha = (mold > -1e30f) ? exp2f((mold - mnew) * LOG2E) : 0.0f;
          const float pv = (vld && (h0 + hm) < num_heads && mnew > -1e30f) ? exp2f((sc - mnew) * LOG2E) : 0.0f;
          const float bsum = halfwave_sum16(pv);
          P_lds[mg][hm][l16] = f32_to_bf16_rne(pv);
          if (l16 == 0) {
            s_alpha[mg][hm] = alpha;
            s_m[mg][hm] = mnew;
            s_l[mg][hm] = s_l[mg][hm] * alpha + bsum;
          }
        }
      }
      __syncthreads();

      // 5. Rescale o_acc & bank-conflict-free vectorized P @ V (u4v loads from VT!)
      float al[8];
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) al[ei] = s_alpha[mg][2 * ei + hrow];
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int ei = 0; ei < 8; ++ei) o_acc[j][ei] *= al[ei];

      const u4v p0 = *reinterpret_cast<const u4va*>(&P_lds[mg][l16][0]);
      const u4v p1 = *reinterpret_cast<const u4va*>(&P_lds[mg][l16][8]);
      const bf16x16 pf = __builtin_bit_cast(bf16x16, (u8v){p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w});
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int vc = d_start + j * 16 + l16;
        // VT[vc][0..15] is 16 contiguous bf16s (32 bytes, 4-byte aligned -> 8x uint32 or 2x u4v!)
        const uint32_t* vp = reinterpret_cast<const uint32_t*>(&VT[vc][0]);
        const bf16x16 vf = __builtin_bit_cast(bf16x16, (u8v){vp[0], vp[1], vp[2], vp[3], vp[4], vp[5], vp[6], vp[7]});
        o_acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf, vf, o_acc[j]);
        __builtin_amdgcn_sched_barrier(0);
      }
      __syncthreads();
    }
  }

  // Store partial (m, l, o_acc)
  if (dq == 0 && l16 == 0) {
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      const int h = h0 + 2 * ei + hrow;
      if (h < num_heads) {
        const int64_t off = ((int64_t)t * num_heads + h) * NUM_SPLITS + sp;
        part_m[off] = s_m[mg][2 * ei + hrow];
        part_l[off] = s_l[mg][2 * ei + hrow];
      }
    }
  }
#pragma unroll
  for (int ei = 0; ei < 8; ++ei) {
    const int h = h0 + 2 * ei + hrow;
    if (h < num_heads) {
      float* dst = part_o + (((int64_t)t * num_heads + h) * NUM_SPLITS + sp) * HCA_DIM + d_start + l16;
#pragma unroll
      for (int j = 0; j < 8; ++j) dst[j * 16] = o_acc[j][ei];
    }
  }
}

void ernie_hca_decode(
    torch::Tensor q,
    torch::Tensor main_cache, torch::Tensor main_indices, torch::Tensor main_indptr,
    c10::optional<torch::Tensor> extra_cache_opt,
    c10::optional<torch::Tensor> extra_indices_opt,
    c10::optional<torch::Tensor> extra_indptr_opt,
    c10::optional<torch::Tensor> attn_sink_opt,
    c10::optional<torch::Tensor> inv_rope_pos_opt,
    c10::optional<torch::Tensor> inv_rope_cs_opt,
    torch::Tensor out,
    torch::Tensor part_o, torch::Tensor part_m, torch::Tensor part_l,
    double scale, bool is_fnuz_main) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 3 && q.size(2) == HCA_DIM && q.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16);
  const int T = (int)q.size(0);
  const int H = (int)q.size(1);
  if (T == 0) return;

  const bool has_extra = extra_cache_opt.has_value() && extra_indices_opt.has_value() && extra_indptr_opt.has_value();
  const bool fuse_rope = inv_rope_pos_opt.has_value() && inv_rope_cs_opt.has_value();
  auto st = c10::hip::getCurrentHIPStream().stream();

  const uint16_t* qp = reinterpret_cast<const uint16_t*>(q.data_ptr());
  const uint8_t* mc = main_cache.data_ptr<uint8_t>();
  const int32_t* mi = main_indices.data_ptr<int32_t>();
  const int32_t* mip = main_indptr.data_ptr<int32_t>();
  const uint8_t* ec = has_extra ? extra_cache_opt->data_ptr<uint8_t>() : mc;
  const int32_t* ei = has_extra ? extra_indices_opt->data_ptr<int32_t>() : mi;
  const int32_t* eip = has_extra ? extra_indptr_opt->data_ptr<int32_t>() : mip;
  const int64_t es0 = has_extra ? extra_cache_opt->stride(0) : main_cache.stride(0);
  const int mbs = (int)main_cache.size(1);
  const int enr = has_extra ? (int)(extra_cache_opt->size(0) * extra_cache_opt->size(1)) : 0;
  const int ebs = has_extra ? (int)extra_cache_opt->size(1) : 1;
  const bool pow2_bs = (mbs > 0 && (mbs & (mbs - 1)) == 0) && (!has_extra || (ebs > 0 && (ebs & (ebs - 1)) == 0));

  const float* sp = attn_sink_opt.has_value() ? attn_sink_opt->data_ptr<float>() : nullptr;
  uint16_t* op = reinterpret_cast<uint16_t*>(out.data_ptr());

  auto run_reduce = [&](auto tag) {
    constexpr int S = decltype(tag)::value;
    if (fuse_rope) {
      const bool cs_bf16 = (inv_rope_cs_opt->scalar_type() == torch::kBFloat16);
      torch::Tensor cs_use = (cs_bf16 || inv_rope_cs_opt->scalar_type() == torch::kFloat32)
          ? *inv_rope_cs_opt : inv_rope_cs_opt->to(torch::kFloat32);
      if (cs_bf16) {
        hipLaunchKernelGGL((k_ernie_hca_split_reduce<S, 1, 1>), dim3(T, H), dim3(128), 0, st,
                           part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
                           sp, inv_rope_pos_opt->data_ptr<int64_t>(), cs_use.data_ptr(),
                           op, out.stride(0), out.stride(1), cs_use.stride(0), H);
      } else {
        hipLaunchKernelGGL((k_ernie_hca_split_reduce<S, 1, 0>), dim3(T, H), dim3(128), 0, st,
                           part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
                           sp, inv_rope_pos_opt->data_ptr<int64_t>(), cs_use.data_ptr(),
                           op, out.stride(0), out.stride(1), cs_use.stride(0), H);
      }
    } else {
      hipLaunchKernelGGL((k_ernie_hca_split_reduce<S, 0, 0>), dim3(T, H), dim3(128), 0, st,
                         part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
                         sp, nullptr, nullptr, op, out.stride(0), out.stride(1), 0, H);
    }
  };

  // v0.2.8 32H/WG Q-in-VGPRs + union KV/VT LDS + dynamic split-K (1.24x @ T=1, 1.31x @ T=6, 1.37x @ T=12!)
  if (pow2_bs && H >= 32) {
    const int hg2 = (H + 31) / 32;
    auto run_qreg = [&](auto tag) {
      constexpr int S = decltype(tag)::value;
      if (has_extra) {
        hipLaunchKernelGGL((k_ernie_hca_decode_qreg_v28<S, 1>), dim3(hg2, T, S), dim3(256), 0, st,
                           qp, mc, mi, mip, ec, ei, eip,
                           part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
                           q.stride(0), q.stride(1), main_cache.stride(0), es0,
                           (int)(main_cache.size(0) * main_cache.size(1)), enr,
                           mbs, ebs, H, (float)scale, is_fnuz_main ? 1 : 0);
      } else {
        hipLaunchKernelGGL((k_ernie_hca_decode_qreg_v28<S, 0>), dim3(hg2, T, S), dim3(256), 0, st,
                           qp, mc, mi, mip, ec, ei, eip,
                           part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
                           q.stride(0), q.stride(1), main_cache.stride(0), es0,
                           (int)(main_cache.size(0) * main_cache.size(1)), enr,
                           mbs, ebs, H, (float)scale, is_fnuz_main ? 1 : 0);
      }
      run_reduce(tag);
    };
    if (T <= 2)      run_qreg(std::integral_constant<int, 8>{});
    else if (T <= 8) run_qreg(std::integral_constant<int, 4>{});
    else             run_qreg(std::integral_constant<int, 2>{});
    return;
  }

  constexpr int S = 8;
  const int hg = (H + 15) / 16;
  if (has_extra) {
    hipLaunchKernelGGL((k_ernie_hca_decode_partial<S, 1>), dim3(T, hg, S), dim3(128), 0, st,
                       qp, mc, mi, mip, ec, ei, eip,
                       part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
                       q.stride(0), q.stride(1), main_cache.stride(0), es0,
                       (int)(main_cache.size(0) * main_cache.size(1)), enr,
                       mbs, ebs, H, (float)scale, is_fnuz_main ? 1 : 0);
  } else {
    hipLaunchKernelGGL((k_ernie_hca_decode_partial<S, 0>), dim3(T, hg, S), dim3(128), 0, st,
                       qp, mc, mi, mip, ec, ei, eip,
                       part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
                       q.stride(0), q.stride(1), main_cache.stride(0), es0,
                       (int)(main_cache.size(0) * main_cache.size(1)), enr,
                       mbs, ebs, H, (float)scale, is_fnuz_main ? 1 : 0);
  }
  run_reduce(std::integral_constant<int, 8>{});
}

// ============================================================================
// HCA Prefill (37 layers, exact D=512 BF16 KV gathered buffer, attn_sink)
// Replaces _rocm_sparse_attn_prefill_triton + build_ragged_indices_from_dense!
//   - S=1 direct write for T > 16 (zero scratch, 4*T WGs, 20.75 KiB LDS)
//   - S=8 split-KV for T <= 16 (saturates all 40 CUs on short chunked prefill)
// ============================================================================
__device__ __forceinline__ void load_bf16_kv_block(
    const uint16_t* __restrict__ kv_ptr, int64_t kv_s0,
    const int32_t* s_slots, uint16_t (*KV_lds)[HCA_STRIDE], int tid) {
  // 128 threads -> 8 threads/token (kr = tid >> 3 in 0..15, sub = tid & 7 in 0..7)
  // Each thread copies 64 bf16 = 8x 128-bit vector loads
  const int kr = tid >> 3;
  const int sub = tid & 7;
  const int32_t sl = s_slots[kr];
  if (sl < 0) {
    u4v z = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      *reinterpret_cast<u4va*>(&KV_lds[kr][sub * 64 + c * 8]) = z;
    }
    return;
  }
  const uint16_t* src = kv_ptr + (int64_t)sl * kv_s0 + sub * 64;
  // 2x 4-load burst (free +0/+16/+32/+48 offsets -> 4x MLP without serialized vmcnt(0) or 32-VGPR spike!)
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const u4v v0 = *reinterpret_cast<const u4va*>(src + (half * 4 + 0) * 8);
    const u4v v1 = *reinterpret_cast<const u4va*>(src + (half * 4 + 1) * 8);
    const u4v v2 = *reinterpret_cast<const u4va*>(src + (half * 4 + 2) * 8);
    const u4v v3 = *reinterpret_cast<const u4va*>(src + (half * 4 + 3) * 8);
    *reinterpret_cast<u4va*>(&KV_lds[kr][sub * 64 + (half * 4 + 0) * 8]) = v0;
    *reinterpret_cast<u4va*>(&KV_lds[kr][sub * 64 + (half * 4 + 1) * 8]) = v1;
    *reinterpret_cast<u4va*>(&KV_lds[kr][sub * 64 + (half * 4 + 2) * 8]) = v2;
    *reinterpret_cast<u4va*>(&KV_lds[kr][sub * 64 + (half * 4 + 3) * 8]) = v3;
    __builtin_amdgcn_sched_barrier(0);
  }
}

// Mode 1: Single-pass direct output (T > 16)
__global__ __launch_bounds__(128) void k_ernie_hca_prefill_s1(
    const uint16_t* __restrict__ q_ptr,      // [T, H, 512] bf16
    const uint16_t* __restrict__ kv_ptr,     // [N_kv, 512] bf16
    const int32_t* __restrict__ indices,     // [T, K_max] int32
    const int32_t* __restrict__ topk_lens,   // [T] int32 or nullptr
    const float* __restrict__ sink_ptr,      // [H] fp32 or nullptr
    uint16_t* __restrict__ out_ptr,          // [T, H, 512] bf16
    int64_t q_s0, int64_t q_s1,
    int64_t kv_s0, int64_t idx_s0,
    int64_t out_s0, int64_t out_s1,
    int num_heads, int num_kv_rows, int K_max, float scale) {
  __shared__ __attribute__((aligned(16))) uint16_t KV_lds[16][HCA_STRIDE]; // 16.25 KiB
  __shared__ float S_part[4][16][16];                                       // 4.00 KiB
  __shared__ uint16_t P_lds[16][16];                                        // 0.50 KiB
  __shared__ int32_t s_slots[16];
  __shared__ uint8_t s_valid[16];
  __shared__ float s_alpha[16];
  __shared__ float s_m[16];
  __shared__ float s_l[16];

  const int t = blockIdx.x;
  const int hg = blockIdx.y;
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

  const bool q_hvalid = (h0 + l16) < num_heads;
  const uint16_t* q_base = q_ptr + (int64_t)t * q_s0 + (int64_t)(q_hvalid ? (h0 + l16) : 0) * q_s1;
  const int d_start = wave * 128;

  int klen = K_max;
  if (topk_lens != nullptr) {
    int tl = topk_lens[t];
    if (tl < 0) tl = 0;
    if (tl < klen) klen = tl;
  }
  const int32_t* idx_row = indices + (int64_t)t * idx_s0;

  for (int k0 = 0; k0 < klen; k0 += 16) {
    if (tid < 16) {
      const int kp = k0 + tid;
      int32_t sl = (kp < klen) ? idx_row[kp] : -1;
      if (sl < 0 || sl >= num_kv_rows) sl = -1;
      s_slots[tid] = sl;
      s_valid[tid] = (sl >= 0) ? 1 : 0;
    }
    __syncthreads();
    unsigned any_v = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) any_v |= s_valid[i];
    if (!any_v) continue;

    load_bf16_kv_block(kv_ptr, kv_s0, s_slots, KV_lds, tid);
    __syncthreads();

    // 1. Q @ K^T (unroll-1 + sched_barrier keeps <=112 VGPRs = 13w/SIMD!)
    f32x8 s_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 1
    for (int step = 0; step < 8; ++step) {
      const int d = d_start + step * 16;
      u4v q0 = {0u, 0u, 0u, 0u}, q1 = {0u, 0u, 0u, 0u};
      if (q_hvalid) {
        q0 = *reinterpret_cast<const u4va*>(q_base + d);
        q1 = *reinterpret_cast<const u4va*>(q_base + d + 8);
      }
      const bf16x16 qf = __builtin_bit_cast(bf16x16, (u8v){q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w});
      const u4v k0v = *reinterpret_cast<const u4va*>(&KV_lds[l16][d]);
      const u4v k1v = *reinterpret_cast<const u4va*>(&KV_lds[l16][d + 8]);
      const bf16x16 kf = __builtin_bit_cast(bf16x16, (u8v){k0v.x, k0v.y, k0v.z, k0v.w, k1v.x, k1v.y, k1v.z, k1v.w});
      s_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(qf, kf, s_acc);
      __builtin_amdgcn_sched_barrier(0);
    }
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) S_part[wave][2 * ei + hrow][l16] = s_acc[ei];
    __syncthreads();

    // 2. Online softmax in Wave 0
    if (wave == 0) {
      const bool vld = (s_valid[l16] != 0);
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) {
        const int hm = 2 * ei + hrow;
        float sc = S_part[0][hm][l16] + S_part[1][hm][l16] + S_part[2][hm][l16] + S_part[3][hm][l16];
        sc = (vld && (h0 + hm) < num_heads) ? (sc * scale) : NEG_INF;
        const float bmax = halfwave_max16(sc);
        const float mold = s_m[hm];
        const float mnew = fmaxf(mold, bmax);
        const float alpha = (mold > -1e30f) ? exp2f((mold - mnew) * LOG2E) : 0.0f;
        const float pv = (vld && (h0 + hm) < num_heads && mnew > -1e30f) ? exp2f((sc - mnew) * LOG2E) : 0.0f;
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

    // 3. Rescale o_acc & P @ V
    float al[8];
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) al[ei] = s_alpha[2 * ei + hrow];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) o_acc[j][ei] *= al[ei];
    }
    const u4v p0 = *reinterpret_cast<const u4va*>(&P_lds[l16][0]);
    const u4v p1 = *reinterpret_cast<const u4va*>(&P_lds[l16][8]);
    const bf16x16 pf = __builtin_bit_cast(bf16x16, (u8v){p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w});
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int vc = d_start + j * 16 + l16;
      bf16x16 vf;
#pragma unroll
      for (int kr = 0; kr < 16; ++kr) vf[kr] = (short)KV_lds[kr][vc];
      o_acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf, vf, o_acc[j]);
      __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();
  }

  // Finalize with optional attn_sink & direct BF16 store
  if (tid < 16) {
    const int h = h0 + tid;
    float m_cur = s_m[tid];
    float l_cur = s_l[tid];
    float scale_o = 0.0f;
    if (h < num_heads) {
      if (sink_ptr != nullptr) {
        const float sv = sink_ptr[h];
        const float m_max = fmaxf(m_cur, sv);
        const float w_cur = (l_cur > 0.0f) ? exp2f((m_cur - m_max) * LOG2E) : 0.0f;
        const float l_tot = l_cur * w_cur + exp2f((sv - m_max) * LOG2E);
        scale_o = (l_tot > 0.0f) ? (w_cur * __builtin_amdgcn_rcpf(fmaxf(l_tot, 1.0e-30f))) : 0.0f;
      } else {
        scale_o = (l_cur > 0.0f) ? __builtin_amdgcn_rcpf(fmaxf(l_cur, 1.0e-30f)) : 0.0f;
      }
    }
    s_alpha[tid] = scale_o;
  }
  __syncthreads();

#pragma unroll
  for (int ei = 0; ei < 8; ++ei) {
    const int hm = 2 * ei + hrow;
    const int h = h0 + hm;
    if (h < num_heads) {
      const float sc_o = s_alpha[hm];
      uint16_t* dst = out_ptr + (int64_t)t * out_s0 + (int64_t)h * out_s1 + d_start + l16;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        dst[j * 16] = f32_to_bf16_rne(o_acc[j][ei] * sc_o);
      }
    }
  }
}

// Mode 2: Split-KV prefill partial (T <= 16) -> reduced via k_ernie_hca_split_reduce<8, 0>
template <int NUM_SPLITS>
__global__ __launch_bounds__(128) void k_ernie_hca_prefill_partial(
    const uint16_t* __restrict__ q_ptr,
    const uint16_t* __restrict__ kv_ptr,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ topk_lens,
    float* __restrict__ part_o,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    int64_t q_s0, int64_t q_s1,
    int64_t kv_s0, int64_t idx_s0,
    int num_heads, int num_kv_rows, int K_max, float scale) {
  __shared__ __attribute__((aligned(16))) uint16_t KV_lds[16][HCA_STRIDE];
  __shared__ float S_part[4][16][16];
  __shared__ uint16_t P_lds[16][16];
  __shared__ int32_t s_slots[16];
  __shared__ uint8_t s_valid[16];
  __shared__ float s_alpha[16];
  __shared__ float s_m[16];
  __shared__ float s_l[16];

  const int t = blockIdx.x;
  const int hg = blockIdx.y;
  const int sp = blockIdx.z;
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

  const bool q_hvalid = (h0 + l16) < num_heads;
  const uint16_t* q_base = q_ptr + (int64_t)t * q_s0 + (int64_t)(q_hvalid ? (h0 + l16) : 0) * q_s1;
  const int d_start = wave * 128;

  int klen = K_max;
  if (topk_lens != nullptr) {
    int tl = topk_lens[t];
    if (tl < 0) tl = 0;
    if (tl < klen) klen = tl;
  }
  const int chunk = (klen + NUM_SPLITS - 1) / NUM_SPLITS;
  const int k_lo = sp * chunk;
  const int k_hi = min(k_lo + chunk, klen);
  const int32_t* idx_row = indices + (int64_t)t * idx_s0;

  for (int k0 = k_lo; k0 < k_hi; k0 += 16) {
    if (tid < 16) {
      const int kp = k0 + tid;
      int32_t sl = (kp < k_hi) ? idx_row[kp] : -1;
      if (sl < 0 || sl >= num_kv_rows) sl = -1;
      s_slots[tid] = sl;
      s_valid[tid] = (sl >= 0) ? 1 : 0;
    }
    __syncthreads();
    unsigned any_v = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) any_v |= s_valid[i];
    if (!any_v) continue;

    load_bf16_kv_block(kv_ptr, kv_s0, s_slots, KV_lds, tid);
    __syncthreads();

    f32x8 s_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 1
    for (int step = 0; step < 8; ++step) {
      const int d = d_start + step * 16;
      u4v q0 = {0u, 0u, 0u, 0u}, q1 = {0u, 0u, 0u, 0u};
      if (q_hvalid) {
        q0 = *reinterpret_cast<const u4va*>(q_base + d);
        q1 = *reinterpret_cast<const u4va*>(q_base + d + 8);
      }
      const bf16x16 qf = __builtin_bit_cast(bf16x16, (u8v){q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w});
      const u4v k0v = *reinterpret_cast<const u4va*>(&KV_lds[l16][d]);
      const u4v k1v = *reinterpret_cast<const u4va*>(&KV_lds[l16][d + 8]);
      const bf16x16 kf = __builtin_bit_cast(bf16x16, (u8v){k0v.x, k0v.y, k0v.z, k0v.w, k1v.x, k1v.y, k1v.z, k1v.w});
      s_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(qf, kf, s_acc);
      __builtin_amdgcn_sched_barrier(0);
    }
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) S_part[wave][2 * ei + hrow][l16] = s_acc[ei];
    __syncthreads();

    if (wave == 0) {
      const bool vld = (s_valid[l16] != 0);
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) {
        const int hm = 2 * ei + hrow;
        float sc = S_part[0][hm][l16] + S_part[1][hm][l16] + S_part[2][hm][l16] + S_part[3][hm][l16];
        sc = (vld && (h0 + hm) < num_heads) ? (sc * scale) : NEG_INF;
        const float bmax = halfwave_max16(sc);
        const float mold = s_m[hm];
        const float mnew = fmaxf(mold, bmax);
        const float alpha = (mold > -1e30f) ? exp2f((mold - mnew) * LOG2E) : 0.0f;
        const float pv = (vld && (h0 + hm) < num_heads && mnew > -1e30f) ? exp2f((sc - mnew) * LOG2E) : 0.0f;
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

    float al[8];
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) al[ei] = s_alpha[2 * ei + hrow];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) o_acc[j][ei] *= al[ei];
    }
    const u4v p0 = *reinterpret_cast<const u4va*>(&P_lds[l16][0]);
    const u4v p1 = *reinterpret_cast<const u4va*>(&P_lds[l16][8]);
    const bf16x16 pf = __builtin_bit_cast(bf16x16, (u8v){p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w});
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int vc = d_start + j * 16 + l16;
      bf16x16 vf;
#pragma unroll
      for (int kr = 0; kr < 16; ++kr) vf[kr] = (short)KV_lds[kr][vc];
      o_acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf, vf, o_acc[j]);
      __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();
  }

  if (wave == 0 && l16 == 0) {
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      const int h = h0 + 2 * ei + hrow;
      if (h < num_heads) {
        const int64_t off = ((int64_t)t * num_heads + h) * NUM_SPLITS + sp;
        part_m[off] = s_m[2 * ei + hrow];
        part_l[off] = s_l[2 * ei + hrow];
      }
    }
  }
#pragma unroll
  for (int ei = 0; ei < 8; ++ei) {
    const int h = h0 + 2 * ei + hrow;
    if (h < num_heads) {
      float* dst = part_o + (((int64_t)t * num_heads + h) * NUM_SPLITS + sp) * HCA_DIM + d_start + l16;
#pragma unroll
      for (int j = 0; j < 8; ++j) dst[j * 16] = o_acc[j][ei];
    }
  }
}

// v0.2.8 Roofline-Max All-64-Heads 8-Wave HCA Prefill (Zero redundant KV loads across 64 heads + VT_lds 128-bit V loads + 4x parallel softmax!)
// Grid: (T, (num_heads + 63) / 64), Block: 256 (8 Wave32s = 4 head-groups x 2 D-halves)
// Each wave owns 16 heads x 256 V-dims (o_acc[16] = 128 VGPRs -> 0 spills, 0 scratch, 4x fewer global KV loads!)
#define HCA_VT_STRIDE 18  // 36B = 9 banks (odd -> low bank conflict on ds_load_b128, 18 KiB total!)

__global__ __launch_bounds__(256) void k_ernie_hca_prefill_s1_v28(
    const uint16_t* __restrict__ q_ptr,      // [T, H, 512] bf16
    const uint16_t* __restrict__ kv_ptr,     // [N_kv, 512] bf16
    const int32_t* __restrict__ indices,     // [T, K_max] int32
    const int32_t* __restrict__ topk_lens,   // [T] int32 or nullptr
    const float* __restrict__ sink_ptr,      // [H] fp32 or nullptr
    uint16_t* __restrict__ out_ptr,          // [T, H, 512] bf16
    int64_t q_s0, int64_t q_s1,
    int64_t kv_s0, int64_t idx_s0,
    int64_t out_s0, int64_t out_s1,
    int num_heads, int num_kv_rows, int K_max, float scale) {
  __shared__ __attribute__((aligned(16))) uint16_t KV_lds[16][HCA_STRIDE];     // 16,640 B
  __shared__ __attribute__((aligned(16))) uint16_t VT_lds[HCA_DIM][HCA_VT_STRIDE]; // 18,432 B
  __shared__ float S_part[4][16][16];                                           //  4,096 B (wd==1 -> wd==0)
  __shared__ __attribute__((aligned(16))) uint16_t P_lds[4][16][16];            //  2,048 B
  __shared__ int32_t s_slots[16];
  __shared__ uint8_t s_valid[16];
  __shared__ float s_alpha[64];
  __shared__ float s_m[64];
  __shared__ float s_l[64];

  const int t = blockIdx.x;
  const int hg64 = blockIdx.y;
  const int tid = threadIdx.x;               // 0..255
  const int wave = tid >> 5;                 // 0..7
  const int wh = wave >> 1;                  // 0..3 (head group of 16)
  const int wd = wave & 1;                   // 0..1 (D-half: 0..255 or 256..511)
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;
  const int h_base = hg64 * 64;
  const int h0 = h_base + wh * 16;

  if (tid < 64) {
    s_m[tid] = NEG_INF;
    s_l[tid] = 0.0f;
  }
  __syncthreads();

  f32x8 o_acc[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) o_acc[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  const bool q_hvalid = (h0 + l16) < num_heads;
  const uint16_t* q_base = q_ptr + (int64_t)t * q_s0 + (int64_t)(q_hvalid ? (h0 + l16) : 0) * q_s1;
  const int d_start = wd * 256;

  int klen = K_max;
  if (topk_lens != nullptr) {
    int tl = topk_lens[t];
    if (tl < 0) tl = 0;
    if (tl < klen) klen = tl;
  }
  const int32_t* idx_row = indices + (int64_t)t * idx_s0;

  // Cooperative KV + VT load mapping: 256 threads = 16 tokens (kr) x 16 sub-threads (sub)
  // Each sub loads 4x uint4 (32 bf16s) at c*128 + sub*8 -> coalesced 256B bursts!
  const int kr_ld = tid >> 4;   // 0..15
  const int sub_ld = tid & 15;  // 0..15

#pragma unroll 1
  for (int k0 = 0; k0 < klen; k0 += 16) {
    if (tid < 16) {
      const int kp = k0 + tid;
      int32_t sl = (kp < klen) ? idx_row[kp] : -1;
      if (sl < 0 || sl >= num_kv_rows) sl = -1;
      s_slots[tid] = sl;
      s_valid[tid] = (sl >= 0) ? 1 : 0;
    }
    __syncthreads();
    unsigned any_v = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) any_v |= s_valid[i];
    if (!any_v) continue;

    // 1. Load 16x512 BF16 KV into BOTH KV_lds (row-major) and VT_lds (col-major) ONCE for all 64 heads!
    {
      const int32_t sl = s_slots[kr_ld];
      u4v v[4];
      if (sl >= 0) {
        const uint16_t* src = kv_ptr + (int64_t)sl * kv_s0 + sub_ld * 8;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          v[c] = *reinterpret_cast<const u4va*>(src + c * 128);
        }
      } else {
        const u4v z = {0u, 0u, 0u, 0u};
#pragma unroll
        for (int c = 0; c < 4; ++c) v[c] = z;
      }
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int col0 = c * 128 + sub_ld * 8;
        *reinterpret_cast<u4va*>(&KV_lds[kr_ld][col0]) = v[c];
        const uint32_t w0 = v[c].x, w1 = v[c].y, w2 = v[c].z, w3 = v[c].w;
        VT_lds[col0 + 0][kr_ld] = (uint16_t)(w0 & 0xFFFFu);
        VT_lds[col0 + 1][kr_ld] = (uint16_t)(w0 >> 16);
        VT_lds[col0 + 2][kr_ld] = (uint16_t)(w1 & 0xFFFFu);
        VT_lds[col0 + 3][kr_ld] = (uint16_t)(w1 >> 16);
        VT_lds[col0 + 4][kr_ld] = (uint16_t)(w2 & 0xFFFFu);
        VT_lds[col0 + 5][kr_ld] = (uint16_t)(w2 >> 16);
        VT_lds[col0 + 6][kr_ld] = (uint16_t)(w3 & 0xFFFFu);
        VT_lds[col0 + 7][kr_ld] = (uint16_t)(w3 >> 16);
      }
    }
    __syncthreads();

    // 2. Q @ K^T: 16 WMMA steps per wave (256 dims/wave), 1-ahead prefetch, ZERO sched_barrier!
    f32x8 s_acc = {0, 0, 0, 0, 0, 0, 0, 0};
    u4v q0_c = {0u, 0u, 0u, 0u}, q1_c = {0u, 0u, 0u, 0u};
    if (q_hvalid) {
      q0_c = *reinterpret_cast<const u4va*>(q_base + d_start);
      q1_c = *reinterpret_cast<const u4va*>(q_base + d_start + 8);
    }
    u4v k0_c = *reinterpret_cast<const u4va*>(&KV_lds[l16][d_start]);
    u4v k1_c = *reinterpret_cast<const u4va*>(&KV_lds[l16][d_start + 8]);
#pragma unroll 1
    for (int step = 0; step < 16; ++step) {
      u4v q0_n = {0u, 0u, 0u, 0u}, q1_n = {0u, 0u, 0u, 0u};
      u4v k0_n = {0u, 0u, 0u, 0u}, k1_n = {0u, 0u, 0u, 0u};
      if (step + 1 < 16) {
        const int dn = d_start + (step + 1) * 16;
        if (q_hvalid) {
          q0_n = *reinterpret_cast<const u4va*>(q_base + dn);
          q1_n = *reinterpret_cast<const u4va*>(q_base + dn + 8);
        }
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

    // 3. 4x-Parallel Online Softmax (all 4 wd==0 waves run simultaneously!)
    if (wd == 0) {
      const bool vld = (s_valid[l16] != 0);
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) {
        const int hm = 2 * ei + hrow;
        const int h_loc = wh * 16 + hm;
        float sc = s_acc[ei] + S_part[wh][hm][l16];
        sc = (vld && (h0 + hm) < num_heads) ? (sc * scale) : NEG_INF;
        const float bmax = halfwave_max16(sc);
        const float mold = s_m[h_loc];
        const float mnew = fmaxf(mold, bmax);
        const float alpha = (mold > -1e30f) ? exp2f((mold - mnew) * LOG2E) : 0.0f;
        const float pv = (vld && (h0 + hm) < num_heads && mnew > -1e30f) ? exp2f((sc - mnew) * LOG2E) : 0.0f;
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

    // 4. Rescale o_acc[0..15] & P @ V via 128-bit VT_lds loads (ZERO ds_load_u16, ZERO sched_barrier!)
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

    // 1-ahead prefetch of VT_lds rows -> hides LDS latency behind back-to-back BF16 WMMAs!
    u4v v0_c = *reinterpret_cast<const u4va*>(&VT_lds[d_start + l16][0]);
    u4v v1_c = *reinterpret_cast<const u4va*>(&VT_lds[d_start + l16][8]);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      u4v v0_n = {0u, 0u, 0u, 0u}, v1_n = {0u, 0u, 0u, 0u};
      if (j + 1 < 16) {
        const int vc_n = d_start + (j + 1) * 16 + l16;
        v0_n = *reinterpret_cast<const u4va*>(&VT_lds[vc_n][0]);
        v1_n = *reinterpret_cast<const u4va*>(&VT_lds[vc_n][8]);
      }
      const bf16x16 vf = __builtin_bit_cast(bf16x16, (u8v){v0_c.x, v0_c.y, v0_c.z, v0_c.w, v1_c.x, v1_c.y, v1_c.z, v1_c.w});
      o_acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf, vf, o_acc[j]);
      v0_c = v0_n; v1_c = v1_n;
    }
    __syncthreads();
  }

  // Finalize with optional attn_sink & direct BF16 store
  if (tid < 64) {
    const int h = h_base + tid;
    float m_cur = s_m[tid];
    float l_cur = s_l[tid];
    float scale_o = 0.0f;
    if (h < num_heads) {
      if (sink_ptr != nullptr) {
        const float sv = sink_ptr[h];
        const float m_max = fmaxf(m_cur, sv);
        const float w_cur = (l_cur > 0.0f) ? exp2f((m_cur - m_max) * LOG2E) : 0.0f;
        const float l_tot = l_cur * w_cur + exp2f((sv - m_max) * LOG2E);
        scale_o = (l_tot > 0.0f) ? (w_cur * __builtin_amdgcn_rcpf(fmaxf(l_tot, 1.0e-30f))) : 0.0f;
      } else {
        scale_o = (l_cur > 0.0f) ? __builtin_amdgcn_rcpf(fmaxf(l_cur, 1.0e-30f)) : 0.0f;
      }
    }
    s_alpha[tid] = scale_o;
  }
  __syncthreads();

#pragma unroll
  for (int ei = 0; ei < 8; ++ei) {
    const int hm = 2 * ei + hrow;
    const int h = h0 + hm;
    if (h < num_heads) {
      const float sc_o = s_alpha[wh * 16 + hm];
      uint16_t* dst = out_ptr + (int64_t)t * out_s0 + (int64_t)h * out_s1 + d_start + l16;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        dst[j * 16] = f32_to_bf16_rne(o_acc[j][ei] * sc_o);
      }
    }
  }
}


// v0.2.8 Pipelined Register-Resident-Q Union-LDS HCA Prefill:
//  - 32H/WG (2 head-groups x 4 D-quarters of 128 dims), Q[16x128] 100% in VGPRs (ZERO Q loads in K loop!)
//  - Union KV_lds/VT_lds_g -> 26 KiB LDS (<32 KiB -> 2 WGs/CU = 16 waves/CU on gfx1151!)
//  - Direct idx_row load (eliminates s_slots barrier: 4 barriers/tile instead of 5!)
//  - Software-pipelined KV prefetch of tile k0+16 issued BEFORE P@V WMMAs (reuses v_hold[4], 0 extra VGPRs!)
__global__ __launch_bounds__(256) void k_ernie_hca_prefill_qreg_pipe_v28(
    const uint16_t* __restrict__ q_ptr,      // [T, H, 512] bf16
    const uint16_t* __restrict__ kv_ptr,     // [N_kv, 512] bf16
    const int32_t* __restrict__ indices,     // [T, K_max] int32
    const int32_t* __restrict__ topk_lens,   // [T] int32 or nullptr
    const float* __restrict__ sink_ptr,      // [H] fp32 or nullptr
    uint16_t* __restrict__ out_ptr,          // [T, H, 512] bf16
    int64_t q_s0, int64_t q_s1,
    int64_t kv_s0, int64_t idx_s0,
    int64_t out_s0, int64_t out_s1,
    int num_heads, int num_kv_rows, int K_max, float scale) {
  __shared__ union {
    __attribute__((aligned(16))) uint16_t KV_lds[16][HCA_STRIDE];
    __attribute__((aligned(16))) uint16_t VT_lds_g[16][32 * HCA_VT_STRIDE + 2];
  } u_kv;
  __shared__ float S_part[2][4][16][16];                                  // 8,192 B (4-way parallel softmax!)
  __shared__ __attribute__((aligned(16))) uint16_t P_lds[2][16][16];      // 1,024 B
  __shared__ float s_alpha[32];
  __shared__ float s_m[32];
  __shared__ float s_l[32];

  const int hg32 = blockIdx.x;
  const int t = blockIdx.y;
  const int tid = threadIdx.x;               // 0..255
  const int wave = tid >> 5;                 // 0..7
  const int wh = wave >> 2;                  // 0..1 (head group of 16)
  const int wd = wave & 3;                   // 0..3 (D-quarter: 0, 128, 256, 384)
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;
  const int h_base = hg32 * 32;
  const int h0 = h_base + wh * 16;

  if (tid < 32) {
    s_m[tid] = NEG_INF;
    s_l[tid] = 0.0f;
  }

  // Preload Q[16 x 128] ONCE into registers -> ZERO Q loads inside K loop!
  const bool q_hvalid = (h0 + l16) < num_heads;
  const uint16_t* q_base = q_ptr + (int64_t)t * q_s0 + (int64_t)(q_hvalid ? (h0 + l16) : 0) * q_s1 + wd * 128;
  bf16x16 q_reg[8];
#pragma unroll
  for (int step = 0; step < 8; ++step) {
    if (q_hvalid) {
      const u4v q0 = *reinterpret_cast<const u4va*>(q_base + step * 16 + 0);
      const u4v q1 = *reinterpret_cast<const u4va*>(q_base + step * 16 + 8);
      q_reg[step] = __builtin_bit_cast(bf16x16, (u8v){q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w});
    } else {
      q_reg[step] = __builtin_bit_cast(bf16x16, (u8v){0u,0u,0u,0u,0u,0u,0u,0u});
    }
  }

  f32x8 o_acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) o_acc[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  const int d_start = wd * 128;
  int klen = K_max;
  if (topk_lens != nullptr) {
    int tl = topk_lens[t];
    if (tl < 0) tl = 0;
    if (tl < klen) klen = tl;
  }
  const int32_t* idx_row = indices + (int64_t)t * idx_s0;
  const int kr_ld = tid >> 4;   // 0..15
  const int sub_ld = tid & 15;  // 0..15

  auto load_kv_tile = [&](int k_base, u4v out_v[4]) {
    const int kp = k_base + kr_ld;
    int32_t sl = (kp < klen) ? idx_row[kp] : -1;
    if (sl >= 0 && sl < num_kv_rows) {
      const uint16_t* src = kv_ptr + (int64_t)sl * kv_s0 + sub_ld * 8;
#pragma unroll
      for (int c = 0; c < 4; ++c) out_v[c] = *reinterpret_cast<const u4va*>(src + c * 128);
    } else {
      const u4v z = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int c = 0; c < 4; ++c) out_v[c] = z;
    }
  };

  // Prologue: prefetch tile 0 while s_m/s_l init finishes!
  u4v v_hold[4];
  if (klen > 0) load_kv_tile(0, v_hold);
  __syncthreads();

#pragma unroll 1
  for (int k0 = 0; k0 < klen; k0 += 16) {
    // 1. Write prefetched v_hold[4] into u_kv.KV_lds
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      *reinterpret_cast<u4va*>(&u_kv.KV_lds[kr_ld][c * 128 + sub_ld * 8]) = v_hold[c];
    }
    __syncthreads();

    // 2. Q @ K^T: 8 WMMA steps, Q 100% in VGPRs, 1-ahead K_lds prefetch!
    f32x8 s_acc = {0, 0, 0, 0, 0, 0, 0, 0};
    u4v k0_c = *reinterpret_cast<const u4va*>(&u_kv.KV_lds[l16][d_start]);
    u4v k1_c = *reinterpret_cast<const u4va*>(&u_kv.KV_lds[l16][d_start + 8]);
#pragma unroll
    for (int step = 0; step < 8; ++step) {
      u4v k0_n = {0u, 0u, 0u, 0u}, k1_n = {0u, 0u, 0u, 0u};
      if (step + 1 < 8) {
        const int dn = d_start + (step + 1) * 16;
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

    // 3a. Write u_kv.VT_lds FIRST (KV_lds is dead!) then IMMEDIATELY prefetch NEXT tile into v_hold
    //     so DRAM/L2 gather overlaps with BOTH 4-way parallel softmax AND all 8 P@V WMMAs!
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int col0 = c * 128 + sub_ld * 8;
      const uint32_t w0 = v_hold[c].x, w1 = v_hold[c].y, w2 = v_hold[c].z, w3 = v_hold[c].w;
      uint16_t* vtg = &u_kv.VT_lds_g[col0 >> 5][(col0 & 31) * HCA_VT_STRIDE + kr_ld];
      vtg[0 * HCA_VT_STRIDE] = (uint16_t)(w0 & 0xFFFFu);
      vtg[1 * HCA_VT_STRIDE] = (uint16_t)(w0 >> 16);
      vtg[2 * HCA_VT_STRIDE] = (uint16_t)(w1 & 0xFFFFu);
      vtg[3 * HCA_VT_STRIDE] = (uint16_t)(w1 >> 16);
      vtg[4 * HCA_VT_STRIDE] = (uint16_t)(w2 & 0xFFFFu);
      vtg[5 * HCA_VT_STRIDE] = (uint16_t)(w2 >> 16);
      vtg[6 * HCA_VT_STRIDE] = (uint16_t)(w3 & 0xFFFFu);
      vtg[7 * HCA_VT_STRIDE] = (uint16_t)(w3 >> 16);
    }
    if (k0 + 16 < klen) {
      load_kv_tile(k0 + 16, v_hold);
    }
    __builtin_amdgcn_sched_barrier(0);

    // 3b. 4-way parallel online softmax across ALL 4 wd waves (2 ei/wave instead of 8 on wd==0!)
    {
      const int kp = k0 + l16;
      const int32_t sl_v = (kp < klen) ? idx_row[kp] : -1;
      const bool vld = (sl_v >= 0 && sl_v < num_kv_rows);
#pragma unroll
      for (int e_sub = 0; e_sub < 2; ++e_sub) {
        const int ei = wd * 2 + e_sub;
        const int hm = 2 * ei + hrow;
        const int h_loc = wh * 16 + hm;
        float sc = S_part[wh][0][hm][l16] + S_part[wh][1][hm][l16] + S_part[wh][2][hm][l16] + S_part[wh][3][hm][l16];
        sc = (vld && (h0 + hm) < num_heads) ? (sc * scale) : NEG_INF;
        const float bmax = halfwave_max16(sc);
        const float mold = s_m[h_loc];
        const float mnew = fmaxf(mold, bmax);
        const float alpha = (mold > -1e30f) ? exp2f((mold - mnew) * LOG2E) : 0.0f;
        const float pv = (vld && (h0 + hm) < num_heads && mnew > -1e30f) ? exp2f((sc - mnew) * LOG2E) : 0.0f;
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

    const int vc0 = d_start + l16;
    u4v v0_c = *reinterpret_cast<const u4va*>(&u_kv.VT_lds_g[vc0 >> 5][(vc0 & 31) * HCA_VT_STRIDE + 0]);
    u4v v1_c = *reinterpret_cast<const u4va*>(&u_kv.VT_lds_g[vc0 >> 5][(vc0 & 31) * HCA_VT_STRIDE + 8]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      u4v v0_n = {0u, 0u, 0u, 0u}, v1_n = {0u, 0u, 0u, 0u};
      if (j + 1 < 8) {
        const int vc_n = d_start + (j + 1) * 16 + l16;
        v0_n = *reinterpret_cast<const u4va*>(&u_kv.VT_lds_g[vc_n >> 5][(vc_n & 31) * HCA_VT_STRIDE + 0]);
        v1_n = *reinterpret_cast<const u4va*>(&u_kv.VT_lds_g[vc_n >> 5][(vc_n & 31) * HCA_VT_STRIDE + 8]);
      }
      const bf16x16 vf = __builtin_bit_cast(bf16x16, (u8v){v0_c.x, v0_c.y, v0_c.z, v0_c.w, v1_c.x, v1_c.y, v1_c.z, v1_c.w});
      o_acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf, vf, o_acc[j]);
      v0_c = v0_n; v1_c = v1_n;
    }
    __syncthreads();
  }

  if (tid < 32) {
    const int h = h_base + tid;
    float m_cur = s_m[tid];
    float l_cur = s_l[tid];
    float scale_o = 0.0f;
    if (h < num_heads) {
      if (sink_ptr != nullptr) {
        const float sv = sink_ptr[h];
        const float m_max = fmaxf(m_cur, sv);
        const float w_cur = (l_cur > 0.0f) ? exp2f((m_cur - m_max) * LOG2E) : 0.0f;
        const float l_tot = l_cur * w_cur + exp2f((sv - m_max) * LOG2E);
        scale_o = (l_tot > 0.0f) ? (w_cur * __builtin_amdgcn_rcpf(fmaxf(l_tot, 1.0e-30f))) : 0.0f;
      } else {
        scale_o = (l_cur > 0.0f) ? __builtin_amdgcn_rcpf(fmaxf(l_cur, 1.0e-30f)) : 0.0f;
      }
    }
    s_alpha[tid] = scale_o;
  }
  __syncthreads();

#pragma unroll
  for (int ei = 0; ei < 8; ++ei) {
    const int hm = 2 * ei + hrow;
    const int h = h0 + hm;
    if (h < num_heads) {
      const float sc_o = s_alpha[wh * 16 + hm];
      uint16_t* dst = out_ptr + (int64_t)t * out_s0 + (int64_t)h * out_s1 + d_start + l16;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        dst[j * 16] = f32_to_bf16_rne(o_acc[j][ei] * sc_o);
      }
    }
  }
}

void ernie_hca_prefill(
    torch::Tensor q,
    torch::Tensor kv,
    torch::Tensor indices,
    c10::optional<torch::Tensor> topk_length_opt,
    c10::optional<torch::Tensor> attn_sink_opt,
    torch::Tensor out,
    c10::optional<torch::Tensor> part_o_opt,
    c10::optional<torch::Tensor> part_m_opt,
    c10::optional<torch::Tensor> part_l_opt,
    double scale) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 3 && q.size(2) == HCA_DIM && q.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(kv.is_cuda() && kv.scalar_type() == torch::kBFloat16 && kv.size(-1) == HCA_DIM);
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16);
  const int T = (int)q.size(0);
  const int H = (int)q.size(1);
  if (T == 0) return;

  auto kv_2d = (kv.dim() == 3) ? kv.squeeze(1) : kv;
  TORCH_CHECK(kv_2d.dim() == 2 && kv_2d.stride(1) == 1);
  auto idx_2d = indices.dim() == 2 ? indices : indices.reshape({T, -1});
  TORCH_CHECK(idx_2d.scalar_type() == torch::kInt32 && idx_2d.stride(1) == 1);

  const int N_kv = (int)kv_2d.size(0);
  const int K_max = (int)idx_2d.size(1);
  const int hg = (H + 15) / 16;
  const int32_t* tl_p = topk_length_opt.has_value() ? topk_length_opt->data_ptr<int32_t>() : nullptr;
  const float* sp = attn_sink_opt.has_value() ? attn_sink_opt->data_ptr<float>() : nullptr;
  const uint16_t* qp = reinterpret_cast<const uint16_t*>(q.data_ptr());
  const uint16_t* kvp = reinterpret_cast<const uint16_t*>(kv_2d.data_ptr());
  uint16_t* op = reinterpret_cast<uint16_t*>(out.data_ptr());
  auto st = c10::hip::getCurrentHIPStream().stream();

  const bool use_split = (T <= 16) && part_o_opt.has_value() && part_m_opt.has_value() && part_l_opt.has_value();
  if (use_split) {
    constexpr int S = 8;
    hipLaunchKernelGGL((k_ernie_hca_prefill_partial<S>), dim3(T, hg, S), dim3(128), 0, st,
                       qp, kvp, idx_2d.data_ptr<int32_t>(), tl_p,
                       part_o_opt->data_ptr<float>(), part_m_opt->data_ptr<float>(), part_l_opt->data_ptr<float>(),
                       q.stride(0), q.stride(1), kv_2d.stride(0), idx_2d.stride(0),
                       H, N_kv, K_max, (float)scale);
    hipLaunchKernelGGL((k_ernie_hca_split_reduce<S, 0>), dim3(T, H), dim3(128), 0, st,
                       part_o_opt->data_ptr<float>(), part_m_opt->data_ptr<float>(), part_l_opt->data_ptr<float>(),
                       sp, nullptr, nullptr, op, out.stride(0), out.stride(1), 0, H);
  } else {
    hipLaunchKernelGGL(
        k_ernie_hca_prefill_qreg_pipe_v28, dim3((H + 31) / 32, T), dim3(256), 0, st,
        qp, kvp, idx_2d.data_ptr<int32_t>(), tl_p, sp, op,
        q.stride(0), q.stride(1), kv_2d.stride(0), idx_2d.stride(0),
        out.stride(0), out.stride(1), H, N_kv, K_max, (float)scale);
  }
}

// ============================================================================
// Vectorized Wave32 UE8M0 K-Cache Dequantize & Gather (replaces Triton 74x/prefill!)
// 128 threads/WG -> 16 tokens/step (8 threads/token: sub 0..6 decode 7x64 FP8 via
// 256-entry LDS LUT + 1-cycle UE8M0 scale; sub 7 copies 64 BF16 RoPE), 128-bit R/W!
// ============================================================================
__global__ __launch_bounds__(128) void k_dequant_gather_ue8m0_kv(
    uint16_t* __restrict__ out_ptr,         // [B, M, 512] bf16
    const uint8_t* __restrict__ k_cache,    // [num_blocks, block_bytes] uint8
    const int32_t* __restrict__ seq_lens,   // [B] int32
    const int32_t* __restrict__ block_table,// [B, max_blocks] int32
    const int32_t* __restrict__ gather_lens,// [B] int32 or nullptr
    int64_t out_s0, int64_t out_s1, int64_t cache_s0, int64_t bt_s0,
    int max_blocks, int block_size, int offset, int is_fnuz) {
  __shared__ float fp8_lut[256];
  const int tid = threadIdx.x;
  fp8_lut[tid] = fp8_byte_to_f32((uint32_t)tid, is_fnuz);
  fp8_lut[tid + 128] = fp8_byte_to_f32((uint32_t)(tid + 128), is_fnuz);
  __syncthreads();

  const int b = blockIdx.y;
  const int seq_len = seq_lens[b];
  const int gather_len = (gather_lens != nullptr) ? gather_lens[b] : seq_len;
  if (gather_len <= 0) return;
  const int start_pos = seq_len - gather_len;

  const int tok_local = tid >> 3; // 0..15
  const int sub = tid & 7;        // 0..7
  const int stride_tok = gridDim.x * 16;
  const int32_t* bt_row = block_table + (int64_t)b * bt_s0;

  for (int i = blockIdx.x * 16 + tok_local; i < gather_len; i += stride_tok) {
    const int pos = start_pos + i;
    if (pos < 0) continue;
    const int blk_in_seq = pos / block_size;
    if (blk_in_seq >= max_blocks) continue;
    const int pos_in_blk = pos % block_size;
    const int64_t phys_blk = (int64_t)bt_row[blk_in_seq];
    const uint8_t* blk_ptr = k_cache + phys_blk * cache_s0;
    const uint8_t* tok_data = blk_ptr + (int64_t)pos_in_blk * HCA_ROW_BYTES;
    uint16_t* dst = out_ptr + (int64_t)b * out_s0 + (int64_t)(offset + i) * out_s1;

    if (sub < 7) {
      const uint8_t* tok_sc = blk_ptr + (int64_t)block_size * HCA_ROW_BYTES + (int64_t)pos_in_blk * 8;
      const float sc = ue8m0_to_f32(tok_sc[sub]);
      const uint8_t* np = tok_data + sub * 64;
      const u4v raws[4] = {
        __builtin_nontemporal_load(reinterpret_cast<const u4va*>(np + 0)),
        __builtin_nontemporal_load(reinterpret_cast<const u4va*>(np + 16)),
        __builtin_nontemporal_load(reinterpret_cast<const u4va*>(np + 32)),
        __builtin_nontemporal_load(reinterpret_cast<const u4va*>(np + 48)),
      };
#pragma unroll
      for (int chunk = 0; chunk < 4; ++chunk) {
        const u4v raw = raws[chunk];
        const uint32_t ws[4] = {raw.x, raw.y, raw.z, raw.w};
        uint32_t pk[8];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          const uint32_t x = ws[w];
          pk[w * 2 + 0] = f32_to_bf16x2_rne_finite(fp8_lut[x & 0xFFu] * sc,         fp8_lut[(x >> 8) & 0xFFu] * sc);
          pk[w * 2 + 1] = f32_to_bf16x2_rne_finite(fp8_lut[(x >> 16) & 0xFFu] * sc, fp8_lut[(x >> 24) & 0xFFu] * sc);
        }
        *reinterpret_cast<u4va*>(dst + sub * 64 + chunk * 16 + 0) = (u4v){pk[0], pk[1], pk[2], pk[3]};
        *reinterpret_cast<u4va*>(dst + sub * 64 + chunk * 16 + 8) = (u4v){pk[4], pk[5], pk[6], pk[7]};
      }
    } else {
      const uint16_t* rp = reinterpret_cast<const uint16_t*>(tok_data + HCA_NOPE);
#pragma unroll
      for (int chunk = 0; chunk < 8; ++chunk) {
        const u4v v = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(rp + chunk * 8));
        *reinterpret_cast<u4va*>(dst + HCA_NOPE + chunk * 8) = v;
      }
    }
  }
}

void ernie_dequant_gather_ue8m0_kv(
    torch::Tensor out, torch::Tensor k_cache, torch::Tensor seq_lens,
    torch::Tensor block_table, c10::optional<torch::Tensor> gather_lens_opt,
    int64_t block_size, int64_t offset, bool is_fnuz) {
  TORCH_CHECK(out.is_cuda() && out.dim() == 3 && out.size(2) == HCA_DIM && out.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(k_cache.is_cuda() && k_cache.scalar_type() == torch::kUInt8);
  const int B = (int)seq_lens.size(0);
  if (B == 0) return;
  auto sl_i32 = seq_lens.scalar_type() == torch::kInt32 ? seq_lens.contiguous() : seq_lens.to(torch::kInt32).contiguous();
  auto bt_i32 = block_table.scalar_type() == torch::kInt32 ? block_table : block_table.to(torch::kInt32);
  torch::Tensor gl_i32;
  const int32_t* gl_p = nullptr;
  if (gather_lens_opt.has_value()) {
    gl_i32 = gather_lens_opt->scalar_type() == torch::kInt32 ? gather_lens_opt->contiguous() : gather_lens_opt->to(torch::kInt32).contiguous();
    gl_p = gl_i32.data_ptr<int32_t>();
  }
  auto st = c10::hip::getCurrentHIPStream().stream();
  hipLaunchKernelGGL(
      k_dequant_gather_ue8m0_kv, dim3(32, B), dim3(128), 0, st,
      reinterpret_cast<uint16_t*>(out.data_ptr()),
      k_cache.data_ptr<uint8_t>(),
      sl_i32.data_ptr<int32_t>(), bt_i32.data_ptr<int32_t>(), gl_p,
      out.stride(0), out.stride(1), k_cache.stride(0), bt_i32.stride(0),
      (int)bt_i32.size(1), (int)block_size, (int)offset, is_fnuz ? 1 : 0);
}

// ============================================================================
// Wave32 Combine TopK + SWA Prefill Indices (replaces torch.full + Triton 37x/prefill!)
// Writes valid topk, valid SWA, and -1 tail in ONE coalesced pass (no torch.full!).
// ============================================================================
template <int HAS_IMAGE>
__global__ __launch_bounds__(128) void k_combine_topk_swa_indices(
    int32_t* __restrict__ combined_indices,
    int64_t comb_s0,
    int32_t* __restrict__ combined_lens,
    const int32_t* __restrict__ topk_indices,
    int64_t topk_s0,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ gather_lens,
    const int32_t* __restrict__ left_visible,
    const int32_t* __restrict__ right_visible,
    int M, int N, int topk, int compress_ratio,
    int window_size, int swa_width, int topk_width, int combined_topk) {
  const int batch_idx = blockIdx.x;
  const int worker_id = blockIdx.y;
  const int num_workers = gridDim.y;
  const int tid = threadIdx.x;

  const int base = query_start_loc[0];
  const int q_start = query_start_loc[batch_idx] - base;
  const int q_end = query_start_loc[batch_idx + 1] - base;
  const int q_len = q_end - q_start;
  const int seq_len = seq_lens[batch_idx];
  const int gather_len = gather_lens[batch_idx];
  const int start_pos = seq_len - q_len;
  const int gather_start = seq_len - gather_len;
  const int batch_m = M * batch_idx;

  for (int tok = q_start + worker_id; tok < q_end; tok += num_workers) {
    const int pos = start_pos + (tok - q_start);
    int topk_len = (pos + 1) / compress_ratio;
    if (topk_len > topk) topk_len = topk;
    int left = 0, right = 0;
    if constexpr (HAS_IMAGE) {
      left = left_visible[tok];
      right = right_visible[tok];
    }
    int left_add = left - (window_size - 1);
    if (left_add < 0) left_add = 0;
    int swa_start = pos - (window_size - 1) - left_add;
    if (swa_start < 0) swa_start = 0;
    if (swa_start < gather_start) swa_start = gather_start;
    int swa_end = pos + right + 1;
    if (swa_end > seq_len) swa_end = seq_len;
    int swa_len = swa_end - swa_start;
    if (swa_len < 0) swa_len = 0;
    const int swa_write = (swa_len < swa_width) ? swa_len : swa_width;
    const int total_write = topk_len + swa_write;
    const int swa_base = batch_m + N + swa_start - gather_start;

    int32_t* out_row = combined_indices + (int64_t)tok * comb_s0;
    const int32_t* src_row = topk_indices + (int64_t)tok * topk_s0;
    for (int j = tid; j < combined_topk; j += 128) {
      int32_t val = -1;
      if (j < topk_len) {
        if (j < topk_width) {
          const int32_t ti = src_row[j];
          if (ti >= 0 && ti < N) val = ti + batch_m;
        }
      } else if (j < total_write) {
        val = swa_base + (j - topk_len);
      }
      out_row[j] = val;
    }
    if (tid == 0) combined_lens[tok] = topk_len + swa_len;
  }
}

std::vector<torch::Tensor> ernie_combine_topk_swa_indices(
    torch::Tensor topk_indices,
    torch::Tensor query_start_loc,
    torch::Tensor seq_lens,
    torch::Tensor gather_lens,
    c10::optional<torch::Tensor> left_visible_opt,
    c10::optional<torch::Tensor> right_visible_opt,
    int64_t window_size, int64_t compress_ratio, int64_t topk,
    int64_t M, int64_t N, int64_t swa_width, int64_t combined_topk) {
  auto tk = topk_indices.reshape({topk_indices.size(0), -1}).contiguous();
  TORCH_CHECK(tk.is_cuda() && tk.scalar_type() == torch::kInt32);
  const int num_tokens = (int)tk.size(0);
  const int num_reqs = (int)seq_lens.size(0);
  auto i32_opt = tk.options().dtype(torch::kInt32);
  auto comb_idx = torch::empty({num_tokens, combined_topk}, i32_opt);
  auto comb_len = torch::empty({num_tokens}, i32_opt);
  if (num_tokens == 0 || num_reqs == 0) return {comb_idx, comb_len};

  auto qsl = query_start_loc.scalar_type() == torch::kInt32 ? query_start_loc.contiguous() : query_start_loc.to(torch::kInt32).contiguous();
  auto sl = seq_lens.scalar_type() == torch::kInt32 ? seq_lens.contiguous() : seq_lens.to(torch::kInt32).contiguous();
  auto gl = gather_lens.scalar_type() == torch::kInt32 ? gather_lens.contiguous() : gather_lens.to(torch::kInt32).contiguous();
  const int workers = std::min(64, std::max(1, num_tokens));
  auto st = c10::hip::getCurrentHIPStream().stream();

  if (left_visible_opt.has_value() && right_visible_opt.has_value()) {
    auto lv = left_visible_opt->scalar_type() == torch::kInt32 ? left_visible_opt->contiguous() : left_visible_opt->to(torch::kInt32).contiguous();
    auto rv = right_visible_opt->scalar_type() == torch::kInt32 ? right_visible_opt->contiguous() : right_visible_opt->to(torch::kInt32).contiguous();
    hipLaunchKernelGGL(
        (k_combine_topk_swa_indices<1>), dim3(num_reqs, workers), dim3(128), 0, st,
        comb_idx.data_ptr<int32_t>(), comb_idx.stride(0), comb_len.data_ptr<int32_t>(),
        tk.data_ptr<int32_t>(), tk.stride(0),
        qsl.data_ptr<int32_t>(), sl.data_ptr<int32_t>(), gl.data_ptr<int32_t>(),
        lv.data_ptr<int32_t>(), rv.data_ptr<int32_t>(),
        (int)M, (int)N, (int)topk, (int)compress_ratio,
        (int)window_size, (int)swa_width, (int)tk.size(1), (int)combined_topk);
  } else {
    hipLaunchKernelGGL(
        (k_combine_topk_swa_indices<0>), dim3(num_reqs, workers), dim3(128), 0, st,
        comb_idx.data_ptr<int32_t>(), comb_idx.stride(0), comb_len.data_ptr<int32_t>(),
        tk.data_ptr<int32_t>(), tk.stride(0),
        qsl.data_ptr<int32_t>(), sl.data_ptr<int32_t>(), gl.data_ptr<int32_t>(),
        nullptr, nullptr,
        (int)M, (int)N, (int)topk, (int)compress_ratio,
        (int)window_size, (int)swa_width, (int)tk.size(1), (int)combined_topk);
  }
  return {comb_idx, comb_len};
}

// ============================================================================
// Wave32 Global TopK Ragged Indices + Indptr (replaces 2 Triton kernels + cumsum!)
// ============================================================================
__global__ __launch_bounds__(128) void k_compute_topk_lens(
    int32_t* __restrict__ topk_lens,
    const int32_t* __restrict__ topk_indices, int64_t topk_s0,
    int topk, const uint8_t* __restrict__ is_valid) {
  __shared__ int32_t s_wsum[4];
  const int tok = blockIdx.x;
  const int tid = threadIdx.x;
  if (!is_valid[tok]) {
    if (tid == 0) topk_lens[tok] = 0;
    return;
  }
  const int32_t* row = topk_indices + (int64_t)tok * topk_s0;
  int32_t cnt = 0;
  for (int j = tid; j < topk; j += 128) {
    cnt += (row[j] >= 0) ? 1 : 0;
  }
  // Wave32 DPP sum
  cnt += __builtin_amdgcn_update_dpp(0, cnt, 0x160 | 1, 0xF, 0xF, false);
  cnt += __builtin_amdgcn_update_dpp(0, cnt, 0x160 | 2, 0xF, 0xF, false);
  cnt += __builtin_amdgcn_update_dpp(0, cnt, 0x160 | 4, 0xF, 0xF, false);
  cnt += __builtin_amdgcn_update_dpp(0, cnt, 0x160 | 8, 0xF, 0xF, false);
  cnt += (int32_t)__builtin_amdgcn_permlanex16((unsigned)cnt, (unsigned)cnt, 0x76543210u, 0xfedcba98u, false, false);
  if ((tid & 31) == 0) s_wsum[tid >> 5] = cnt;
  __syncthreads();
  if (tid == 0) topk_lens[tok] = s_wsum[0] + s_wsum[1] + s_wsum[2] + s_wsum[3];
}

__global__ __launch_bounds__(32) void k_build_indptr_scan(
    const int32_t* __restrict__ lens, int32_t* __restrict__ indptr, int n) {
  // Single-wave chunked inclusive scan (zero sync, <1 us for decode batches)
  const int lane = threadIdx.x;
  int32_t running = 0;
  if (lane == 0) indptr[0] = 0;
  for (int base = 0; base < n; base += 32) {
    const int idx = base + lane;
    int32_t v = (idx < n) ? lens[idx] : 0;
    // Inclusive prefix sum across 32 lanes via __shfl_up
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int32_t prev = __shfl_up(v, off, 32);
      if (lane >= off) v += prev;
    }
    if (idx < n) indptr[idx + 1] = running + v;
    running += __shfl(v, 31, 32);
  }
}

__global__ __launch_bounds__(128) void k_pack_global_topk_ragged(
    int32_t* __restrict__ global_ragged,
    const int32_t* __restrict__ topk_indptr,
    const int32_t* __restrict__ topk_indices, int64_t topk_s0,
    const int32_t* __restrict__ tok_to_req,
    const int32_t* __restrict__ block_table, int64_t bt_s0,
    int block_size, int topk) {
  const int tok = blockIdx.x;
  const int out_start = topk_indptr[tok];
  const int out_len = topk_indptr[tok + 1] - out_start;
  const int off = blockIdx.y * 128 + threadIdx.x;
  if (off >= out_len || off >= topk) return;
  const int req_idx = tok_to_req[tok];
  const int local_idx = topk_indices[(int64_t)tok * topk_s0 + off];
  int32_t slot = -1;
  if (local_idx >= 0) {
    const int blk_idx = local_idx / block_size;
    const int blk_off = local_idx % block_size;
    const int phys = block_table[(int64_t)req_idx * bt_s0 + blk_idx];
    slot = phys * block_size + blk_off;
  }
  global_ragged[out_start + off] = slot;
}

std::vector<torch::Tensor> ernie_compute_global_topk_ragged(
    torch::Tensor topk_indices,
    torch::Tensor token_to_req,
    torch::Tensor block_table,
    int64_t block_size,
    torch::Tensor is_valid_token) {
  auto tk = topk_indices.reshape({topk_indices.size(0), -1}).contiguous();
  TORCH_CHECK(tk.is_cuda() && tk.scalar_type() == torch::kInt32);
  const int num_tokens = (int)tk.size(0);
  const int topk = (int)tk.size(1);
  auto i32_opt = tk.options().dtype(torch::kInt32);
  auto topk_lens = torch::empty({num_tokens}, i32_opt);
  auto topk_indptr = torch::empty({num_tokens + 1}, i32_opt);
  auto global_ragged = torch::empty({num_tokens * topk}, i32_opt);
  auto st = c10::hip::getCurrentHIPStream().stream();
  if (num_tokens == 0) {
    topk_indptr.zero_();
    return {global_ragged, topk_indptr, topk_lens};
  }
  auto valid_u8 = is_valid_token.to(torch::kUInt8).contiguous();
  auto t2r = token_to_req.scalar_type() == torch::kInt32 ? token_to_req.contiguous() : token_to_req.to(torch::kInt32).contiguous();
  auto bt = block_table.scalar_type() == torch::kInt32 ? block_table : block_table.to(torch::kInt32);
  hipLaunchKernelGGL(
      k_compute_topk_lens, dim3(num_tokens), dim3(128), 0, st,
      topk_lens.data_ptr<int32_t>(), tk.data_ptr<int32_t>(), tk.stride(0),
      topk, valid_u8.data_ptr<uint8_t>());
  hipLaunchKernelGGL(
      k_build_indptr_scan, dim3(1), dim3(32), 0, st,
      topk_lens.data_ptr<int32_t>(), topk_indptr.data_ptr<int32_t>(), num_tokens);
  if (topk > 0) {
    hipLaunchKernelGGL(
        k_pack_global_topk_ragged, dim3(num_tokens, (topk + 127) / 128), dim3(128), 0, st,
        global_ragged.data_ptr<int32_t>(), topk_indptr.data_ptr<int32_t>(),
        tk.data_ptr<int32_t>(), tk.stride(0),
        t2r.data_ptr<int32_t>(), bt.data_ptr<int32_t>(), bt.stride(0),
        (int)block_size, topk);
  }
  return {global_ragged, topk_indptr, topk_lens};
}

PYBIND11_MODULE(ernie_hca_attn, m) {
  m.def("hca_decode", &ernie_hca_decode);
  m.def("hca_prefill", &ernie_hca_prefill);
  m.def("dequant_gather_ue8m0_kv", &ernie_dequant_gather_ue8m0_kv);
  m.def("combine_topk_swa_indices", &ernie_combine_topk_swa_indices);
  m.def("compute_global_topk_ragged", &ernie_compute_global_topk_ragged);
}

