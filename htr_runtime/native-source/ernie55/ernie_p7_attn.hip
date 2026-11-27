// ERNIE-5.5 HCA / DSA sparse-prefill attention kernels (gfx1151, Wave32 WMMA), opt-in via ERNIE_P7_ATTN=1. Same contract as ernie_hca_attn.hca_prefill's
// T > 16 path (k_ernie_hca_prefill_s1): q [T, H, 512] bf16, kv [N, 512] bf16 (single KV head, K = V), per-query
// sorted index lists, optional attn_sink, bf16 output.
//
// v1: arithmetic identical to k_ernie_hca_prefill_s1 (same WMMA per 128-dim slice, same S sum order, same online
// softmax, same bf16 P), so the output must be bit-identical. Only data movement changes:
//   - one block per query with 512 threads = 4 head groups x 4 dim slices, so each 16-key tile is loaded once per
//     query instead of once per 16-head group (4x fewer tile loads);
//   - the tile is also written transposed (Vt_lds[dim][key]) so the P.V operand is two 16-byte LDS reads instead of
//     16 scalar reads;
//   - the four head groups run their online softmax in parallel.
// v2a: + next-tile prefetch into registers (no slot barrier). v2b: + softmax rows split over the 4 waves of a group.
// v3: v2b + Q kept in registers (Q was re-read from cache every tile: 64 KB of Q per 16 KB key tile).
// v4: v3 + LDS-only barriers in the loop + index loads one tile further ahead (no exposed global latency per tile).
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef short bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));
typedef u4v u4va __attribute__((aligned(4)));
typedef uint32_t u8v __attribute__((ext_vector_type(8)));

#define HCA_DIM 512
#define HCA_STRIDE 520
#define VT_STRIDE 24
#define NHG 4
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
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)0u;
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

__device__ __forceinline__ void lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Grid: (T), Block: 512 (16 Wave32s: wave = 4 * head_group + dim_slice)
// PREFETCH (v2): the next tile's index + key loads are issued into registers before this tile's compute, which also
//   drops the slot barrier; SPLIT_SM (v2): the 4 waves of a head group each run the softmax for 4 of its 16 rows.
// Q_REG (v3): this wave's 16 x 128 Q fragment lives in 64 VGPRs for the whole query instead of being re-read per tile.
// ABL (timing-only ablations, wrong output): 1 no QK, 2 no PV, 4 no transposed V writes, 8 no key loads, 16 no softmax.
// ABL 32 is a real variant (not bit-exact): QK over two accumulators (even/odd steps) to break the 8-WMMA dependency chain.
// PIPE (v4): in-loop barriers fence LDS only (__syncthreads() also waited for the in-flight key prefetch), and the
//   index for tile k+2 is loaded during tile k, so tile k+1's key loads never wait on a dependent index load.
template <bool PREFETCH, bool SPLIT_SM, bool Q_REG, int ABL = 0, bool PIPE = false>
__global__ __launch_bounds__(512) void k_hca_prefill(
    const uint16_t* __restrict__ q_ptr, const uint16_t* __restrict__ kv_ptr,
    const int32_t* __restrict__ indices, const int32_t* __restrict__ topk_lens,
    const float* __restrict__ sink_ptr, uint16_t* __restrict__ out_ptr,
    int64_t q_s0, int64_t q_s1, int64_t kv_s0, int64_t idx_s0, int64_t out_s0, int64_t out_s1,
    int num_heads, int num_kv_rows, int K_max, float scale) {
  __shared__ __attribute__((aligned(16))) uint16_t KV_lds[16][HCA_STRIDE];   // 16.25 KiB
  __shared__ __attribute__((aligned(16))) uint16_t Vt_lds[HCA_DIM][VT_STRIDE]; // 24.00 KiB
  __shared__ float S_part[NHG][4][16][16];                                    // 16.00 KiB
  __shared__ __attribute__((aligned(16))) uint16_t P_lds[NHG][16][16];       //  2.00 KiB
  __shared__ int32_t s_slots[16];
  __shared__ uint8_t s_valid[16];
  __shared__ float s_alpha[NHG][16];
  __shared__ float s_m[NHG][16];
  __shared__ float s_l[NHG][16];

  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int g = wave >> 2;   // head group
  const int ws = wave & 3;   // 128-dim slice
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;
  const int h0 = g * 16;

  if (tid < NHG * 16) {
    s_m[tid >> 4][tid & 15] = NEG_INF;
    s_l[tid >> 4][tid & 15] = 0.0f;
  }
  __syncthreads();

  f32x8 o_acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) o_acc[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  const bool q_hvalid = (h0 + l16) < num_heads;
  const uint16_t* q_base = q_ptr + (int64_t)t * q_s0 + (int64_t)(q_hvalid ? (h0 + l16) : 0) * q_s1;
  const int d_start = ws * 128;
  bf16x16 q_reg[Q_REG ? 8 : 1];
  if constexpr (Q_REG) {
#pragma unroll
    for (int step = 0; step < 8; ++step) {
      u4v q0 = {0u, 0u, 0u, 0u}, q1 = {0u, 0u, 0u, 0u};
      if (q_hvalid) {
        q0 = *reinterpret_cast<const u4va*>(q_base + d_start + step * 16);
        q1 = *reinterpret_cast<const u4va*>(q_base + d_start + step * 16 + 8);
      }
      q_reg[step] = __builtin_bit_cast(bf16x16, (u8v){q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w});
    }
  }

  int klen = K_max;
  if (topk_lens != nullptr) {
    int tl = topk_lens[t];
    if (tl < 0) tl = 0;
    if (tl < klen) klen = tl;
  }
  const int32_t* idx_row = indices + (int64_t)t * idx_s0;

  // tile loader mapping: a wave covers 8 keys x 4 contiguous 32-byte pieces (one 128-byte line per key), which keeps
  // global reads line-sized and limits the transposed LDS writes to 4-way bank conflicts
  const int ld_kr = (lane & 7) + 8 * (wave & 1);
  const int ld_sub = (lane >> 3) + 4 * (wave >> 1);  // 16-dim piece, 0..31

  u4v pf0 = {0u, 0u, 0u, 0u}, pf1 = {0u, 0u, 0u, 0u};
  int32_t pf_sl = -1;
  int32_t nx_sl = -1;  // PIPE: index of this thread's key in the tile after the one being fetched
  auto load_idx = [&](int kn) {
    const int kp = kn + ld_kr;
    int32_t sl = (kp < klen) ? idx_row[kp] : -1;
    if (sl < 0 || sl >= num_kv_rows) sl = -1;
    return sl;
  };
  auto fetch = [&](int kn) {
    int32_t sl;
    if constexpr (PIPE) {
      sl = nx_sl;
      nx_sl = load_idx(kn + 16);
    } else {
      sl = load_idx(kn);
    }
    pf_sl = sl;
    pf0 = (u4v){0u, 0u, 0u, 0u};
    pf1 = (u4v){0u, 0u, 0u, 0u};
    if (!(ABL & 8) && sl >= 0) {
      const uint16_t* src = kv_ptr + (int64_t)sl * kv_s0 + ld_sub * 16;
      pf0 = *reinterpret_cast<const u4va*>(src);
      pf1 = *reinterpret_cast<const u4va*>(src + 8);
    }
  };
  auto store_tile = [&](u4v v0, u4v v1) {
    *reinterpret_cast<u4va*>(&KV_lds[ld_kr][ld_sub * 16]) = v0;
    *reinterpret_cast<u4va*>(&KV_lds[ld_kr][ld_sub * 16 + 8]) = v1;
    const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    if constexpr (!(ABL & 4))
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      Vt_lds[ld_sub * 16 + 2 * i][ld_kr] = (uint16_t)(w[i] & 0xFFFFu);
      Vt_lds[ld_sub * 16 + 2 * i + 1][ld_kr] = (uint16_t)(w[i] >> 16);
    }
  };
  if constexpr (PIPE) nx_sl = load_idx(0);
  if constexpr (PREFETCH) fetch(0);
  auto bar = [&]() {
    if constexpr (PIPE) lds_barrier();
    else __syncthreads();
  };

  for (int k0 = 0; k0 < klen; k0 += 16) {
    if constexpr (PREFETCH) {
      // an all-invalid tile is computed rather than skipped: P = 0 and alpha = 1 leave o_acc, m and l unchanged
      store_tile(pf0, pf1);
      if (ld_sub == 0) s_valid[ld_kr] = (pf_sl >= 0) ? 1 : 0;
      bar();
      if (k0 + 16 < klen) fetch(k0 + 16);
    } else {
    if (tid < 16) {
      const int kp = k0 + tid;
      int32_t sl = (kp < klen) ? idx_row[kp] : -1;
      if (sl < 0 || sl >= num_kv_rows) sl = -1;
      s_slots[tid] = sl;
      s_valid[tid] = (sl >= 0) ? 1 : 0;
    }
    bar();
    unsigned any_v = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) any_v |= s_valid[i];
    if (!any_v) continue;

    pf_sl = s_slots[ld_kr];
    pf0 = (u4v){0u, 0u, 0u, 0u};
    pf1 = (u4v){0u, 0u, 0u, 0u};
    if (pf_sl >= 0) {
      const uint16_t* src = kv_ptr + (int64_t)pf_sl * kv_s0 + ld_sub * 16;
      pf0 = *reinterpret_cast<const u4va*>(src);
      pf1 = *reinterpret_cast<const u4va*>(src + 8);
    }
    store_tile(pf0, pf1);
    bar();
    }

    // 1. Q @ K^T over this wave's 128-dim slice
    f32x8 s_acc = {0, 0, 0, 0, 0, 0, 0, 0};
    if constexpr (Q_REG) {
      if constexpr (ABL & 32) {
        f32x8 s_odd = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int step = 0; step < 8; ++step) {
          const int d = d_start + step * 16;
          const u4v k0v = *reinterpret_cast<const u4va*>(&KV_lds[l16][d]);
          const u4v k1v = *reinterpret_cast<const u4va*>(&KV_lds[l16][d + 8]);
          const bf16x16 kf = __builtin_bit_cast(bf16x16, (u8v){k0v.x, k0v.y, k0v.z, k0v.w, k1v.x, k1v.y, k1v.z, k1v.w});
          if (step & 1) s_odd = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(q_reg[step], kf, s_odd);
          else s_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(q_reg[step], kf, s_acc);
        }
        s_acc += s_odd;
      } else if constexpr (!(ABL & 1))
#pragma unroll
      for (int step = 0; step < 8; ++step) {
        const int d = d_start + step * 16;
        const u4v k0v = *reinterpret_cast<const u4va*>(&KV_lds[l16][d]);
        const u4v k1v = *reinterpret_cast<const u4va*>(&KV_lds[l16][d + 8]);
        const bf16x16 kf = __builtin_bit_cast(bf16x16, (u8v){k0v.x, k0v.y, k0v.z, k0v.w, k1v.x, k1v.y, k1v.z, k1v.w});
        s_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(q_reg[step], kf, s_acc);
      }
    } else {
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
    }
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) S_part[g][ws][2 * ei + hrow][l16] = s_acc[ei];
    bar();

    // 2. Online softmax: the first wave of each head group (SPLIT_SM: rows 4*ws..4*ws+3 per wave)
    if (!(ABL & 16) && (SPLIT_SM || ws == 0)) {
      const bool vld = (s_valid[l16] != 0);
#pragma unroll
      for (int ei = SPLIT_SM ? 2 * ws : 0; ei < (SPLIT_SM ? 2 * ws + 2 : 8); ++ei) {
        const int hm = 2 * ei + hrow;
        float sc = S_part[g][0][hm][l16] + S_part[g][1][hm][l16] + S_part[g][2][hm][l16] + S_part[g][3][hm][l16];
        sc = (vld && (h0 + hm) < num_heads) ? (sc * scale) : NEG_INF;
        const float bmax = halfwave_max16(sc);
        const float mold = s_m[g][hm];
        const float mnew = fmaxf(mold, bmax);
        const float alpha = (mold > -1e30f) ? exp2f((mold - mnew) * LOG2E) : 0.0f;
        const float pv = (vld && (h0 + hm) < num_heads && mnew > -1e30f) ? exp2f((sc - mnew) * LOG2E) : 0.0f;
        const float bsum = halfwave_sum16(pv);
        P_lds[g][hm][l16] = f32_to_bf16_rne(pv);
        if (l16 == 0) {
          s_alpha[g][hm] = alpha;
          s_m[g][hm] = mnew;
          s_l[g][hm] = s_l[g][hm] * alpha + bsum;
        }
      }
    }
    bar();

    // 3. Rescale o_acc & P @ V (V operand from the transposed copy)
    float al[8];
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) al[ei] = s_alpha[g][2 * ei + hrow];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) o_acc[j][ei] *= al[ei];
    }
    const u4v p0 = *reinterpret_cast<const u4va*>(&P_lds[g][l16][0]);
    const u4v p1 = *reinterpret_cast<const u4va*>(&P_lds[g][l16][8]);
    const bf16x16 pf = __builtin_bit_cast(bf16x16, (u8v){p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w});
    if constexpr (!(ABL & 2))
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int vc = d_start + j * 16 + l16;
      const u4v v0 = *reinterpret_cast<const u4va*>(&Vt_lds[vc][0]);
      const u4v v1 = *reinterpret_cast<const u4va*>(&Vt_lds[vc][8]);
      const bf16x16 vf = __builtin_bit_cast(bf16x16, (u8v){v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w});
      o_acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf, vf, o_acc[j]);
      __builtin_amdgcn_sched_barrier(0);
    }
    bar();
  }

  // Finalize with optional attn_sink & direct BF16 store
  if (tid < NHG * 16) {
    const int gg = tid >> 4, hh = tid & 15;
    const int h = gg * 16 + hh;
    const float m_cur = s_m[gg][hh];
    const float l_cur = s_l[gg][hh];
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
    s_alpha[gg][hh] = scale_o;
  }
  __syncthreads();

#pragma unroll
  for (int ei = 0; ei < 8; ++ei) {
    const int hm = 2 * ei + hrow;
    const int h = h0 + hm;
    if (h < num_heads) {
      const float sc_o = s_alpha[g][hm];
      uint16_t* dst = out_ptr + (int64_t)t * out_s0 + (int64_t)h * out_s1 + d_start + l16;
#pragma unroll
      for (int j = 0; j < 8; ++j) dst[j * 16] = f32_to_bf16_rne(o_acc[j][ei] * sc_o);
    }
  }
}

template <bool PREFETCH, bool SPLIT_SM, bool Q_REG, int ABL = 0, bool PIPE = false>
void hca_prefill(torch::Tensor q, torch::Tensor kv, torch::Tensor indices,
                 c10::optional<torch::Tensor> topk_length_opt, c10::optional<torch::Tensor> attn_sink_opt,
                 torch::Tensor out, double scale) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 3 && q.size(2) == HCA_DIM && q.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(q.size(1) <= NHG * 16, "v1 handles up to 64 heads");
  TORCH_CHECK(kv.is_cuda() && kv.scalar_type() == torch::kBFloat16 && kv.size(-1) == HCA_DIM);
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16);
  const int T = (int)q.size(0);
  const int H = (int)q.size(1);
  if (T == 0) return;
  auto kv_2d = (kv.dim() == 3) ? kv.squeeze(1) : kv;
  TORCH_CHECK(kv_2d.dim() == 2 && kv_2d.stride(1) == 1);
  auto idx_2d = indices.dim() == 2 ? indices : indices.reshape({T, -1});
  TORCH_CHECK(idx_2d.scalar_type() == torch::kInt32 && idx_2d.stride(1) == 1);
  const int32_t* tl_p = topk_length_opt.has_value() ? topk_length_opt->data_ptr<int32_t>() : nullptr;
  const float* sp = attn_sink_opt.has_value() ? attn_sink_opt->data_ptr<float>() : nullptr;
  auto st = c10::hip::getCurrentHIPStream().stream();
  hipLaunchKernelGGL((k_hca_prefill<PREFETCH, SPLIT_SM, Q_REG, ABL, PIPE>), dim3(T), dim3(512), 0, st,
                     reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint16_t*>(kv_2d.data_ptr()),
                     idx_2d.data_ptr<int32_t>(), tl_p, sp, reinterpret_cast<uint16_t*>(out.data_ptr()),
                     q.stride(0), q.stride(1), kv_2d.stride(0), idx_2d.stride(0), out.stride(0), out.stride(1),
                     H, (int)kv_2d.size(0), (int)idx_2d.size(1), (float)scale);
}

// ============================================================================ DSA sparse-MLA core (fp8_ds_mla cache)
// Same contract as ernie_dsa_attn.dsa_sparse_mla_quant's single-pass path (k_ernie_dsa_sparse_mla_quant<1, 1, 0>):
// q [T, 64, 576] bf16, kv_cache [blocks, block_size, 656] uint8 (512 FP8 nope + 4 fp32 scales + 64 bf16 rope),
// indices [T, W] global slots (-1 invalid), optional sink, out [T, 64, 512] bf16. Same v4 structure as the HCA kernel
// (4 head groups share each dequantized 16-key tile, transposed V, split softmax, Q in registers, LDS-only barriers,
// index loads a tile ahead), with the author's FP8 LUT decode, rounding, softmax, all-invalid-tile skip and epilogue.
#define DSA_NOPE 512
#define DSA_DIM 576
#define DSA_STRIDE 584

__device__ __forceinline__ uint32_t f32_to_bf16x2_rne(float a, float b) {
  uint32_t ua = __float_as_uint(a);
  uint32_t ub = __float_as_uint(b);
  ua = ((ua & 0x7fffffffu) > 0x7f800000u) ? (ua | 0x00400000u) : (ua + 0x7fffu + ((ua >> 16) & 1u));
  ub = ((ub & 0x7fffffffu) > 0x7f800000u) ? (ub | 0x00400000u) : (ub + 0x7fffu + ((ub >> 16) & 1u));
  return __builtin_amdgcn_perm(ub, ua, 0x07060302u);
}
__device__ __forceinline__ uint16_t dsa_f32_to_bf16_rne(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
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

// Grid: (T), Block: 512 (16 Wave32s: wave = 4 * head_group + dim_slice). Q_REG holds the 16 x 144 Q fragment in
// 72 VGPRs (251 VGPRs total, 1 block per WGP); without it Q is re-read from cache per tile (2 blocks per WGP).
template <bool Q_REG>
__global__ __launch_bounds__(512) void k_dsa_prefill_v4(
    const uint16_t* __restrict__ q_ptr, const uint8_t* __restrict__ kv_ptr, const int32_t* __restrict__ idx_ptr,
    const float* __restrict__ sink_ptr, uint16_t* __restrict__ out_ptr,
    int64_t q_s0, int64_t q_s1, int64_t kv_blk_stride, int64_t kv_row_stride, int64_t idx_s0, int64_t out_s0,
    int64_t out_s1, int num_blocks, int block_size, int W, float scale) {
  __shared__ __attribute__((aligned(16))) uint16_t KV_lds[16][DSA_STRIDE];     // 18.25 KiB
  __shared__ __attribute__((aligned(16))) uint16_t Vt_lds[DSA_NOPE][VT_STRIDE]; // 24.00 KiB
  __shared__ float S_part[NHG][4][16][16];                                      // 16.00 KiB
  __shared__ __attribute__((aligned(16))) uint16_t P_lds[NHG][16][16];         //  2.00 KiB
  __shared__ float fp8_lut[256];                                                //  1.00 KiB
  __shared__ uint8_t s_valid[16];
  __shared__ float s_alpha[NHG][16];
  __shared__ float s_m[NHG][16];
  __shared__ float s_l[NHG][16];

  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int g = wave >> 2;
  const int ws = wave & 3;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;
  const int h0 = g * 16;

  if (tid < 256) fp8_lut[tid] = dsa_fp8_ocp_to_f32((uint32_t)tid);
  if (tid < NHG * 16) {
    s_m[tid >> 4][tid & 15] = NEG_INF;
    s_l[tid >> 4][tid & 15] = 0.0f;
  }
  __syncthreads();

  f32x8 o_acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) o_acc[j] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  const int32_t* idx_row = idx_ptr + (int64_t)t * idx_s0;
  const uint16_t* q_base = q_ptr + (int64_t)t * q_s0 + (int64_t)(h0 + l16) * q_s1;
  const int d_start = ws * 144;  // QK over 576 dims: 9 WMMA steps per wave
  auto q_frag = [&](int step) {
    const u4v q0 = *reinterpret_cast<const u4va*>(q_base + d_start + step * 16);
    const u4v q1 = *reinterpret_cast<const u4va*>(q_base + d_start + step * 16 + 8);
    return __builtin_bit_cast(bf16x16, (u8v){q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w});
  };
  bf16x16 q_reg[Q_REG ? 9 : 1];
  if constexpr (Q_REG) {
#pragma unroll
    for (int step = 0; step < 9; ++step) q_reg[step] = q_frag(step);
  }
  const int total_slots = num_blocks * block_size;

  // loader: thread (ld_kr, ld_sub) owns FP8 nope dims [16*ld_sub, 16*ld_sub+16) of key ld_kr; ld_sub < 8 also copies
  // bf16 rope dims [512 + 8*ld_sub, +8)
  const int ld_kr = (lane & 7) + 8 * (wave & 1);
  const int ld_sub = (lane >> 3) + 4 * (wave >> 1);
  u4v pf_raw = {0u, 0u, 0u, 0u}, pf_rope = {0u, 0u, 0u, 0u};
  float pf_sc = 0.0f;
  int32_t pf_sl = -1, nx_sl = -1;
  auto load_idx = [&](int kn) {
    const int c = kn + ld_kr;
    int32_t sl = (c < W) ? idx_row[c] : -1;
    if (sl < 0 || sl >= total_slots) sl = -1;
    return sl;
  };
  auto fetch = [&](int kn) {
    const int32_t sl = nx_sl;
    nx_sl = load_idx(kn + 16);
    pf_sl = sl;
    if (sl >= 0) {
      const int phys_blk = (block_size == 64) ? (sl >> 6) : (sl / block_size);
      const int in_blk = (block_size == 64) ? (sl & 63) : (sl % block_size);
      const uint8_t* row_p = kv_ptr + (int64_t)phys_blk * kv_blk_stride + (int64_t)in_blk * kv_row_stride;
      pf_sc = *reinterpret_cast<const float*>(row_p + 512 + (ld_sub >> 3) * 4);
      pf_raw = *reinterpret_cast<const u4va*>(row_p + ld_sub * 16);
      if (ld_sub < 8) pf_rope = *reinterpret_cast<const u4va*>(row_p + 528 + ld_sub * 16);
    }
  };
  auto store_tile = [&]() {
    u4v lo = {0u, 0u, 0u, 0u}, hi = {0u, 0u, 0u, 0u}, rope = {0u, 0u, 0u, 0u};
    if (pf_sl >= 0) {
      const uint32_t w[4] = {pf_raw.x, pf_raw.y, pf_raw.z, pf_raw.w};
      uint32_t pk[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t x = w[i];
        pk[i * 2 + 0] = f32_to_bf16x2_rne(fp8_lut[x & 0xFFu] * pf_sc, fp8_lut[(x >> 8) & 0xFFu] * pf_sc);
        pk[i * 2 + 1] = f32_to_bf16x2_rne(fp8_lut[(x >> 16) & 0xFFu] * pf_sc, fp8_lut[(x >> 24) & 0xFFu] * pf_sc);
      }
      lo = (u4v){pk[0], pk[1], pk[2], pk[3]};
      hi = (u4v){pk[4], pk[5], pk[6], pk[7]};
      rope = pf_rope;
    }
    *reinterpret_cast<u4va*>(&KV_lds[ld_kr][ld_sub * 16]) = lo;
    *reinterpret_cast<u4va*>(&KV_lds[ld_kr][ld_sub * 16 + 8]) = hi;
    if (ld_sub < 8) *reinterpret_cast<u4va*>(&KV_lds[ld_kr][DSA_NOPE + ld_sub * 8]) = rope;
    const uint32_t v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      Vt_lds[ld_sub * 16 + 2 * i][ld_kr] = (uint16_t)(v[i] & 0xFFFFu);
      Vt_lds[ld_sub * 16 + 2 * i + 1][ld_kr] = (uint16_t)(v[i] >> 16);
    }
    if (ld_sub == 0) s_valid[ld_kr] = (pf_sl >= 0) ? 1 : 0;
  };

  nx_sl = load_idx(0);
  fetch(0);
  for (int k_base = 0; k_base < W; k_base += 16) {
    store_tile();
    lds_barrier();
    if (k_base + 16 < W) {
      pf_raw = (u4v){0u, 0u, 0u, 0u};
      pf_rope = (u4v){0u, 0u, 0u, 0u};
      fetch(k_base + 16);
    }
    unsigned any_v = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) any_v |= s_valid[i];
    if (!any_v) {  // the author skips all-invalid tiles: m, l and o_acc stay untouched
      lds_barrier();
      continue;
    }

    f32x8 s_acc = {0, 0, 0, 0, 0, 0, 0, 0};
    if constexpr (Q_REG) {
#pragma unroll
      for (int step = 0; step < 9; ++step) {
        const int d = d_start + step * 16;
        const u4v k0 = *reinterpret_cast<const u4va*>(&KV_lds[l16][d]);
        const u4v k1 = *reinterpret_cast<const u4va*>(&KV_lds[l16][d + 8]);
        const bf16x16 kf = __builtin_bit_cast(bf16x16, (u8v){k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w});
        s_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(q_reg[step], kf, s_acc);
      }
    } else {
#pragma unroll 1
      for (int step = 0; step < 9; ++step) {
        const int d = d_start + step * 16;
        const bf16x16 qf = q_frag(step);
        const u4v k0 = *reinterpret_cast<const u4va*>(&KV_lds[l16][d]);
        const u4v k1 = *reinterpret_cast<const u4va*>(&KV_lds[l16][d + 8]);
        const bf16x16 kf = __builtin_bit_cast(bf16x16, (u8v){k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w});
        s_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(qf, kf, s_acc);
      }
    }
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) S_part[g][ws][2 * ei + hrow][l16] = s_acc[ei];
    lds_barrier();

    {  // softmax: wave ws of the group owns rows 4*ws .. 4*ws + 3
      const bool vld = (s_valid[l16] != 0);
#pragma unroll
      for (int ei = 2 * ws; ei < 2 * ws + 2; ++ei) {
        const int hm = 2 * ei + hrow;
        float sc = S_part[g][0][hm][l16] + S_part[g][1][hm][l16] + S_part[g][2][hm][l16] + S_part[g][3][hm][l16];
        sc = vld ? (sc * scale) : NEG_INF;
        const float bmax = halfwave_max16(sc);
        const float mold = s_m[g][hm];
        const float mnew = fmaxf(mold, bmax);
        const float alpha = (mold > -1e30f) ? exp2f((mold - mnew) * LOG2E) : 0.0f;
        const float pv = (vld && mnew > -1e30f) ? exp2f((sc - mnew) * LOG2E) : 0.0f;
        const float bsum = halfwave_sum16(pv);
        P_lds[g][hm][l16] = dsa_f32_to_bf16_rne(pv);
        if (l16 == 0) {
          s_alpha[g][hm] = alpha;
          s_m[g][hm] = mnew;
          s_l[g][hm] = s_l[g][hm] * alpha + bsum;
        }
      }
    }
    lds_barrier();

#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      const float a_ei = s_alpha[g][2 * ei + hrow];
#pragma unroll
      for (int j = 0; j < 8; ++j) o_acc[j][ei] *= a_ei;
    }
    const u4v p0 = *reinterpret_cast<const u4va*>(&P_lds[g][l16][0]);
    const u4v p1 = *reinterpret_cast<const u4va*>(&P_lds[g][l16][8]);
    const bf16x16 pf = __builtin_bit_cast(bf16x16, (u8v){p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w});
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int vc = ws * 128 + j * 16 + l16;
      const u4v v0 = *reinterpret_cast<const u4va*>(&Vt_lds[vc][0]);
      const u4v v1 = *reinterpret_cast<const u4va*>(&Vt_lds[vc][8]);
      const bf16x16 vf = __builtin_bit_cast(bf16x16, (u8v){v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w});
      o_acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf, vf, o_acc[j]);
      __builtin_amdgcn_sched_barrier(0);
    }
    lds_barrier();
  }

#pragma unroll
  for (int ei = 0; ei < 8; ++ei) {
    const int hm = 2 * ei + hrow;
    const float m = s_m[g][hm];
    const float l = s_l[g][hm];
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
    uint16_t* dst = out_ptr + (int64_t)t * out_s0 + (int64_t)(h0 + hm) * out_s1 + ws * 128 + l16;
#pragma unroll
    for (int j = 0; j < 8; ++j) dst[j * 16] = dsa_f32_to_bf16_rne(o_acc[j][ei] * inv_l);
  }
}

template <bool Q_REG>
torch::Tensor dsa_prefill(torch::Tensor q, torch::Tensor kv_cache, torch::Tensor indices,
                             c10::optional<torch::Tensor> attn_sink, double scale, int64_t kv_fmt) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 3 && q.size(1) == 64 && q.size(2) == DSA_DIM && q.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(kv_fmt == 1, "dsa_prefill handles fp8_ds_mla only");
  TORCH_CHECK(kv_cache.is_cuda() && kv_cache.dim() == 3 && kv_cache.scalar_type() == torch::kUInt8 && kv_cache.size(2) == 656);
  TORCH_CHECK(indices.is_cuda() && indices.dim() == 2 && indices.scalar_type() == torch::kInt32 && indices.stride(1) == 1);
  const int T = (int)q.size(0);
  auto out = torch::empty({T, 64, DSA_NOPE}, q.options());
  if (T == 0) return out;
  auto st = c10::hip::getCurrentHIPStream().stream();
  hipLaunchKernelGGL((k_dsa_prefill_v4<Q_REG>), dim3(T), dim3(512), 0, st,
                     reinterpret_cast<const uint16_t*>(q.data_ptr()), kv_cache.data_ptr<uint8_t>(), indices.data_ptr<int32_t>(),
                     attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr,
                     reinterpret_cast<uint16_t*>(out.data_ptr()),
                     q.stride(0), q.stride(1), kv_cache.stride(0), kv_cache.stride(1), indices.stride(0),
                     out.stride(0), out.stride(1), (int)kv_cache.size(0), (int)kv_cache.size(1), (int)indices.size(1),
                     (float)scale);
  return out;
}

PYBIND11_MODULE(ernie_p7_attn, m) {
  m.def("hca_prefill_v1", &hca_prefill<false, false, false>);
  m.def("hca_prefill_v2b", &hca_prefill<true, true, false>);
  m.def("hca_prefill_v3", &hca_prefill<true, true, true>);
  m.def("hca_prefill_v3c", &hca_prefill<true, true, true, 32>);
  m.def("hca_prefill_v4", &hca_prefill<true, true, true, 0, true>);
  m.def("dsa_prefill_v4", &dsa_prefill<false>);
  m.def("dsa_prefill_v4q", &dsa_prefill<true>);
}
