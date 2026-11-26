// qsa_mqa_scores_v5_v27.hip — Wave 7 gfx1151 Native v_dot2_f32_bf16 + Zero-LDS DPP QSA Indexer Scorer
// Upgrades over qsa_mqa_scores_v5.hip (and massively over qsa_mqa_paged_aot.hsaco which has 40x ds_bpermute + 24x u16!):
//   1. Native RDNA 3.5 v_dot2_f32_bf16 directly on packed uint4 kr[4] and qraw (ZERO unpack8, -8 VGPRs, 4x fewer dot ops!).
//   2. Native 4-op zero-LDS DPP halfwave_sum (v_add_f32_dpp row_xmask:1,2,4,8) -> ZERO ds_swizzle, ZERO ds_bpermute, ZERO lgkmcnt!
//   3. Exact same ABI & kernel symbol `qsa_mqa_scores_v5` (drop-in via HTR_QSA_MQA_V5=1 & HTR_QSA_MQA_V5_HSACO).
#include <hip/hip_runtime.h>
#include <stdint.h>

#define NT 256
#define KT 64
#define NH 4
#define MAX_RPB 32

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef short bf16x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ uint4 ldnt_u4(const uint16_t* p) {
  u32x4 v = __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(p));
  return make_uint4(v[0], v[1], v[2], v[3]);
}

__device__ __forceinline__ float dot2_bf16(uint32_t a, uint32_t b, float acc) {
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2, a), __builtin_bit_cast(bf16x2, b), acc, false);
}

__device__ __forceinline__ float dot8_bf16_u4(const uint4 kv, const uint4 qv) {
  float a0 = dot2_bf16(kv.x, qv.x, 0.0f);
  float a1 = dot2_bf16(kv.y, qv.y, 0.0f);
  a0 = dot2_bf16(kv.z, qv.z, a0);
  a1 = dot2_bf16(kv.w, qv.w, a1);
  return a0 + a1;
}

// 4-op zero-LDS RDNA 3.5 16-lane halfwave sum!
__device__ __forceinline__ float halfwave_sum_dpp(float v) {
  asm volatile(
    "v_add_f32_dpp %0, %0, %0 row_xmask:1 row_mask:0xf bank_mask:0xf fi:1\n\t"
    "v_add_f32_dpp %0, %0, %0 row_xmask:2 row_mask:0xf bank_mask:0xf fi:1\n\t"
    "v_add_f32_dpp %0, %0, %0 row_xmask:4 row_mask:0xf bank_mask:0xf fi:1\n\t"
    "v_add_f32_dpp %0, %0, %0 row_xmask:8 row_mask:0xf bank_mask:0xf fi:1"
    : "+v"(v));
  return v;
}

extern "C" __global__ __launch_bounds__(NT) void qsa_mqa_scores_v5(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ kc,
    const int32_t* __restrict__ table,
    const int32_t* __restrict__ tok2req,
    const int64_t* __restrict__ qpos,
    const int32_t* __restrict__ seqlens,
    int32_t* __restrict__ visible_out,
    float* __restrict__ logits,
    int64_t stride_q_row,
    int64_t stride_q_head,
    int64_t stride_cache_block,
    int64_t stride_cache_token,
    int64_t stride_table_req,
    int64_t stride_logits_row,
    int num_rows,
    int num_columns,
    int num_pages,
    int num_requests,
    int page_size,
    int table_width,
    int compress_ratio,
    float score_divisor,
    int rows_per_block) {
  __shared__ int32_t s_req[MAX_RPB];
  __shared__ int32_t s_vis[MAX_RPB];
  __shared__ int32_t s_ureq[MAX_RPB];
  __shared__ int32_t s_umaxv[MAX_RPB];
  __shared__ int32_t s_nureq;
  __shared__ int32_t s_has_inv;

  const int tid = threadIdx.x;
  const int col0 = blockIdx.x * KT;
  const int rowA = blockIdx.y * rows_per_block;
  const int rowB = min(num_rows, rowA + rows_per_block);
  const int n_r = rowB - rowA;
  if (n_r <= 0) return;

  if (tid < n_r) {
    const int r = rowA + tid;
    const int req = tok2req[r];
    const int64_t pos = qpos[r];
    const bool req_ok = (req >= 0) && (req < num_requests);
    const int sl = req_ok ? seqlens[req] : 0;
    const int64_t a = (pos + 1) / (int64_t)compress_ratio;
    const int64_t b = (int64_t)(sl / compress_ratio);
    const int64_t v = (a < b) ? a : b;
    if (blockIdx.x == 0) visible_out[r] = (int32_t)v;
    const int vis = req_ok ? (v <= 0 ? 0 : (v > 0x7fffffff ? 0x7fffffff : (int)v)) : 0;
    s_req[tid] = req_ok ? req : -1;
    s_vis[tid] = vis;
  }
  __syncthreads();

  if (tid == 0) {
    int nu = 0;
    int has_inv = 0;
    for (int i = 0; i < n_r; ++i) {
      const int rq = s_req[i];
      const int vs = s_vis[i];
      if (rq < 0) {
        has_inv = 1;
        continue;
      }
      int found = -1;
      for (int u = 0; u < nu; ++u) {
        if (s_ureq[u] == rq) { found = u; break; }
      }
      if (found < 0) {
        s_ureq[nu] = rq;
        s_umaxv[nu] = vs;
        ++nu;
      } else {
        if (vs > s_umaxv[found]) s_umaxv[found] = vs;
      }
    }
    s_nureq = nu;
    s_has_inv = has_inv;
  }
  __syncthreads();

  const float neg_inf = -__builtin_inff();

  if (s_has_inv && tid < KT && (col0 + tid) < num_columns) {
    for (int i = 0; i < n_r; ++i) {
      if (s_req[i] < 0) {
        logits[(int64_t)(rowA + i) * stride_logits_row + (col0 + tid)] = neg_inf;
      }
    }
  }

  const int hw = tid >> 4;
  const int sub = tid & 15;
  const int hw_col0 = col0 + hw * 4;
  const int nu = s_nureq;

  for (int u = 0; u < nu; ++u) {
    const int req = s_ureq[u];
    const int max_vis = s_umaxv[u];

    if (col0 >= max_vis) {
      if (tid < KT && (col0 + tid) < num_columns) {
        for (int i = 0; i < n_r; ++i) {
          if (s_req[i] == req) {
            logits[(int64_t)(rowA + i) * stride_logits_row + (col0 + tid)] = neg_inf;
          }
        }
      }
      continue;
    }

    uint4 kr[4];
    uint8_t col_ok = 0;
    const int32_t* req_tab = table + (int64_t)req * stride_table_req;
#pragma unroll
    for (int k_i = 0; k_i < 4; ++k_i) {
      const int c = hw_col0 + k_i;
      bool ok = false;
      if (c < num_columns && c < max_vis) {
        const int lp = c / page_size;
        const int off = c - lp * page_size;
        if (lp < table_width) {
          const int p = req_tab[lp];
          if (p >= 0 && p < num_pages) {
            ok = true;
            kr[k_i] = ldnt_u4(kc + (int64_t)p * stride_cache_block + (int64_t)off * stride_cache_token + sub * 8);
          }
        }
      }
      if (!ok) kr[k_i] = make_uint4(0u, 0u, 0u, 0u);
      if (ok) col_ok |= (uint8_t)(1u << k_i);
    }

    for (int i = 0; i < n_r; ++i) {
      if (s_req[i] != req) continue;
      const int r = rowA + i;
      const int vis_r = s_vis[i];
      float* lr = logits + (int64_t)r * stride_logits_row;

      if (hw_col0 >= vis_r) {
        if (sub < 4 && (hw_col0 + sub) < num_columns) lr[hw_col0 + sub] = neg_inf;
        continue;
      }

      float sc0 = 0.0f, sc1 = 0.0f, sc2 = 0.0f, sc3 = 0.0f;
      const uint16_t* qr = q + (int64_t)r * stride_q_row;
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        const uint4 qraw = *reinterpret_cast<const uint4*>(qr + (int64_t)h * stride_q_head + sub * 8);
        const float d0 = halfwave_sum_dpp(dot8_bf16_u4(kr[0], qraw));
        const float d1 = halfwave_sum_dpp(dot8_bf16_u4(kr[1], qraw));
        const float d2 = halfwave_sum_dpp(dot8_bf16_u4(kr[2], qraw));
        const float d3 = halfwave_sum_dpp(dot8_bf16_u4(kr[3], qraw));
        sc0 += fmaxf(d0, 0.0f);
        sc1 += fmaxf(d1, 0.0f);
        sc2 += fmaxf(d2, 0.0f);
        sc3 += fmaxf(d3, 0.0f);
      }

      if (sub == 0) {
        if (hw_col0 + 0 < num_columns) lr[hw_col0 + 0] = ((hw_col0 + 0 < vis_r) && (col_ok & 1u)) ? (sc0 / score_divisor) : neg_inf;
        if (hw_col0 + 1 < num_columns) lr[hw_col0 + 1] = ((hw_col0 + 1 < vis_r) && (col_ok & 2u)) ? (sc1 / score_divisor) : neg_inf;
        if (hw_col0 + 2 < num_columns) lr[hw_col0 + 2] = ((hw_col0 + 2 < vis_r) && (col_ok & 4u)) ? (sc2 / score_divisor) : neg_inf;
        if (hw_col0 + 3 < num_columns) lr[hw_col0 + 3] = ((hw_col0 + 3 < vis_r) && (col_ok & 8u)) ? (sc3 / score_divisor) : neg_inf;
      }
    }
  }
}
