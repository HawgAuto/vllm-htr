// qsa_mqa_paged_aot.hip v3 — Wave 7 RDNA 3.5 hidden V_DUAL_DOT2ACC_F32_BF16 (VOPD Opcode 13)!
// Pairs heads (0,1) and (2,3) with bank-staggered (k0, k1) operands -> 4 BF16 FMAs / 8 FLOPs per lane per cycle!
// Supports both int32 and int64 query_positions. 0 LDS, 0 spills, 16 waves/SIMD.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));

// Dual-issue BF16 dot2 for two heads simultaneously against shared kv = (k0, k1):
//   a0 += dot2(q00, k0) + dot2(q01, k1)
//   a1 += dot2(q11, k1) + dot2(q10, k0)
// Bank allocation satisfies ALL 4 RDNA 3.5 VOPD rules:
//   - vdst_x (v0) even, vdst_y (v1) odd
//   - vsrcX0 (v4/v5 = bank 0/1) != vsrcY0 (v7/v6 = bank 3/2)
//   - vsrcX1 (v8/v9 = bank 0/1) != vsrcY1 (v9/v8 = bank 1/0)
__device__ __forceinline__ void dual_dot2_bf16_pair(
    u32x2 q0, u32x2 q1, u32x2 kv, float& out0, float& out1) {
  register float a0 asm("v0") = 0.0f;
  register float a1 asm("v1") = 0.0f;
  register uint32_t q00 asm("v4") = q0[0];
  register uint32_t q01 asm("v5") = q0[1];
  register uint32_t q10 asm("v6") = q1[0];
  register uint32_t q11 asm("v7") = q1[1];
  register uint32_t k0  asm("v8") = kv[0];
  register uint32_t k1  asm("v9") = kv[1];
  asm volatile(
    "v_dual_dot2acc_f32_bf16 %0, %2, %6 :: v_dual_dot2acc_f32_bf16 %1, %5, %7\n\t"
    "v_dual_dot2acc_f32_bf16 %0, %3, %7 :: v_dual_dot2acc_f32_bf16 %1, %4, %6\n\t"
    : "+v"(a0), "+v"(a1)
    : "v"(q00), "v"(q01), "v"(q10), "v"(q11), "v"(k0), "v"(k1)
  );
  out0 = a0;
  out1 = a1;
}

__device__ __forceinline__ float warp_sum32(float v) {
  v += __shfl_xor(v, 16, 32);
  v += __shfl_xor(v, 8, 32);
  v += __shfl_xor(v, 4, 32);
  v += __shfl_xor(v, 2, 32);
  v += __shfl_xor(v, 1, 32);
  return v;
}

template <typename PosT>
__device__ __forceinline__ void qsa_mqa_paged_impl(
    const uint16_t* __restrict__ q_ptr,
    const uint16_t* __restrict__ k_cache_ptr,
    const int32_t* __restrict__ page_table_ptr,
    const int32_t* __restrict__ token_to_req_ptr,
    const PosT* __restrict__ query_positions_ptr,
    const int32_t* __restrict__ sequence_lengths_ptr,
    int32_t* __restrict__ visible_blocks_ptr,
    float* __restrict__ logits_ptr,
    int64_t sq_row, int64_t sq_head, int64_t sq_dim,
    int64_t sk_blk, int64_t sk_tok, int64_t sk_dim,
    int64_t st_req, int64_t st_page,
    int64_t sl_row,
    int num_rows, int num_columns, int num_pages, int num_requests,
    float inv_score_divisor,
    int page_size, int page_table_width, int num_heads, int compress_ratio) {
  const int row = blockIdx.x;
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int col_base = blockIdx.y * 32;
  if (row >= num_rows) return;

  const int req = token_to_req_ptr[row];
  const int req_ok = (req >= 0) && (req < num_requests);
  const int qpos = (int)query_positions_ptr[row];
  const int slen = req_ok ? sequence_lengths_ptr[req] : 0;
  const int v1 = (qpos + 1) / compress_ratio;
  const int v2 = slen / compress_ratio;
  const int visible = v1 < v2 ? v1 : v2;

  if (blockIdx.y == 0 && threadIdx.x == 0) {
    visible_blocks_ptr[row] = visible;
  }

  if (!req_ok || col_base >= visible) {
    const int c = col_base + (threadIdx.x & 31);
    if (wave == 0 && c < num_columns) {
      logits_ptr[(int64_t)row * sl_row + c] = -INFINITY;
    }
    return;
  }

  u32x2 qv[4];
  const uint16_t* qr = q_ptr + (int64_t)row * sq_row;
#pragma unroll
  for (int h = 0; h < 4; ++h) {
    if (h < num_heads) {
      const uint16_t* qh = qr + (int64_t)h * sq_head + (int64_t)(lane * 4) * sq_dim;
      qv[h] = (sq_dim == 1) ? *(const u32x2*)qh
                            : (u32x2){(uint32_t)qh[0] | ((uint32_t)qh[sq_dim] << 16),
                                      (uint32_t)qh[2 * sq_dim] | ((uint32_t)qh[3 * sq_dim] << 16)};
    } else {
      qv[h] = {0u, 0u};
    }
  }

#pragma unroll 2
  for (int ci = 0; ci < 8; ++ci) {
    const int col = col_base + wave + ci * 4;
    if (col >= num_columns) break;
    const int lpage = col / page_size;
    const int poff = col - lpage * page_size;
    bool valid = (col < visible) && (lpage < page_table_width);
    int ppage = -1;
    if (valid) {
      ppage = page_table_ptr[(int64_t)req * st_req + (int64_t)lpage * st_page];
      if (ppage < 0 || ppage >= num_pages) valid = false;
    }
    if (!valid) {
      if (lane == 0) logits_ptr[(int64_t)row * sl_row + col] = -INFINITY;
      continue;
    }
    const uint16_t* kp = k_cache_ptr + (int64_t)ppage * sk_blk + (int64_t)poff * sk_tok + (int64_t)(lane * 4) * sk_dim;
    u32x2 kv = (sk_dim == 1) ? *(const u32x2*)kp
                             : (u32x2){(uint32_t)kp[0] | ((uint32_t)kp[sk_dim] << 16),
                                       (uint32_t)kp[2 * sk_dim] | ((uint32_t)kp[3 * sk_dim] << 16)};

    float a0, a1, a2, a3;
    dual_dot2_bf16_pair(qv[0], qv[1], kv, a0, a1);
    dual_dot2_bf16_pair(qv[2], qv[3], kv, a2, a3);
    float d0 = warp_sum32(a0), d1 = warp_sum32(a1), d2 = warp_sum32(a2), d3 = warp_sum32(a3);
    float score = fmaxf(d0, 0.0f) + fmaxf(d1, 0.0f) + fmaxf(d2, 0.0f) + fmaxf(d3, 0.0f);
    if (lane == 0) {
      logits_ptr[(int64_t)row * sl_row + col] = score * inv_score_divisor;
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) qsa_mqa_paged_bf16(
    const uint16_t* __restrict__ q_ptr, const uint16_t* __restrict__ k_cache_ptr,
    const int32_t* __restrict__ page_table_ptr, const int32_t* __restrict__ token_to_req_ptr,
    const int32_t* __restrict__ query_positions_ptr, const int32_t* __restrict__ sequence_lengths_ptr,
    int32_t* __restrict__ visible_blocks_ptr, float* __restrict__ logits_ptr,
    int64_t sq_row, int64_t sq_head, int64_t sq_dim,
    int64_t sk_blk, int64_t sk_tok, int64_t sk_dim,
    int64_t st_req, int64_t st_page, int64_t sl_row,
    int num_rows, int num_columns, int num_pages, int num_requests,
    float inv_score_divisor, int page_size, int page_table_width, int num_heads, int compress_ratio) {
  qsa_mqa_paged_impl<int32_t>(
      q_ptr, k_cache_ptr, page_table_ptr, token_to_req_ptr, query_positions_ptr, sequence_lengths_ptr,
      visible_blocks_ptr, logits_ptr, sq_row, sq_head, sq_dim, sk_blk, sk_tok, sk_dim, st_req, st_page, sl_row,
      num_rows, num_columns, num_pages, num_requests, inv_score_divisor, page_size, page_table_width, num_heads, compress_ratio);
}

extern "C" __global__ void __launch_bounds__(128) qsa_mqa_paged_bf16_i64(
    const uint16_t* __restrict__ q_ptr, const uint16_t* __restrict__ k_cache_ptr,
    const int32_t* __restrict__ page_table_ptr, const int32_t* __restrict__ token_to_req_ptr,
    const int64_t* __restrict__ query_positions_ptr, const int32_t* __restrict__ sequence_lengths_ptr,
    int32_t* __restrict__ visible_blocks_ptr, float* __restrict__ logits_ptr,
    int64_t sq_row, int64_t sq_head, int64_t sq_dim,
    int64_t sk_blk, int64_t sk_tok, int64_t sk_dim,
    int64_t st_req, int64_t st_page, int64_t sl_row,
    int num_rows, int num_columns, int num_pages, int num_requests,
    float inv_score_divisor, int page_size, int page_table_width, int num_heads, int compress_ratio) {
  qsa_mqa_paged_impl<int64_t>(
      q_ptr, k_cache_ptr, page_table_ptr, token_to_req_ptr, query_positions_ptr, sequence_lengths_ptr,
      visible_blocks_ptr, logits_ptr, sq_row, sq_head, sq_dim, sk_blk, sk_tok, sk_dim, st_req, st_page, sl_row,
      num_rows, num_columns, num_pages, num_requests, inv_score_divisor, page_size, page_table_width, num_heads, compress_ratio);
}
