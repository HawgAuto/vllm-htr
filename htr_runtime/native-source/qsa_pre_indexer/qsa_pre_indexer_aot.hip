// qsa_pre_indexer_aot.hip — AOT gfx1151 Wave32 HIP replacement for Triton _qsa_pre_indexer_kernel.
// D=128, rotary_dim=64 (QUARTER=32 = 1 Wave32!): lane in 0..31 owns dims (lane, lane+32, lane+64, lane+96).
// Zero LDS, zero register spills, zero Triton JIT at any token count / chunk size!
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

__device__ __forceinline__ float bf16_to_f32(uint16_t u) {
  return __uint_as_float((uint32_t)u << 16);
}
__device__ __forceinline__ uint16_t f32_to_bf16_rne(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float warp_sum32(float v) {
  v += __shfl_xor(v, 16, 32);
  v += __shfl_xor(v, 8, 32);
  v += __shfl_xor(v, 4, 32);
  v += __shfl_xor(v, 2, 32);
  v += __shfl_xor(v, 1, 32);
  return v;
}

__device__ __forceinline__ void norm_rope_wave32(
    float x0, float x1, float x2, float x3,
    int64_t pos_t, int64_t pos_h, int64_t pos_w,
    const uint16_t* __restrict__ cos_sin_ptr,
    const float* __restrict__ inv_freq_ptr,
    float mscale,
    const uint16_t* __restrict__ norm_w_ptr,
    float eps,
    int is_mrope, int mrope_h, int mrope_w, int table_free,
    uint16_t& o0, uint16_t& o1, uint16_t& o2, uint16_t& o3) {
  const int lane = threadIdx.x & 31;  // 0..31 == QUARTER
  float ssq = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
  float rrms = rsqrtf(warp_sum32(ssq) * (1.0f / 128.0f) + eps);

  float w0 = bf16_to_f32(norm_w_ptr[lane]) + 1.0f;
  float w1 = bf16_to_f32(norm_w_ptr[lane + 32]) + 1.0f;
  float w2 = bf16_to_f32(norm_w_ptr[lane + 64]) + 1.0f;
  float w3 = bf16_to_f32(norm_w_ptr[lane + 96]) + 1.0f;

  // Match Triton: y = (x * rrms * weight).to(bf16) before NeoX RoPE
  float y0 = bf16_to_f32(f32_to_bf16_rne(x0 * rrms * w0));
  float y1 = bf16_to_f32(f32_to_bf16_rne(x1 * rrms * w1));
  o2 = f32_to_bf16_rne(x2 * rrms * w2);
  o3 = f32_to_bf16_rne(x3 * rrms * w3);

  int64_t pos = pos_t;
  if (is_mrope) {
    if ((lane % 3) == 1 && lane <= 3 * mrope_h) pos = pos_h;
    else if ((lane % 3) == 2 && lane <= 3 * mrope_w) pos = pos_w;
  }

  float c_val, s_val;
  if (table_free) {
    float ang = (float)pos * inv_freq_ptr[lane];
    float sc, ss;
    sincosf(ang, &ss, &sc);
    c_val = bf16_to_f32(f32_to_bf16_rne(sc * mscale));
    s_val = bf16_to_f32(f32_to_bf16_rne(ss * mscale));
  } else {
    c_val = bf16_to_f32(cos_sin_ptr[pos * 64 + lane]);
    s_val = bf16_to_f32(cos_sin_ptr[pos * 64 + 32 + lane]);
  }
  o0 = f32_to_bf16_rne(y0 * c_val - y1 * s_val);
  o1 = f32_to_bf16_rne(y1 * c_val + y0 * s_val);
}

// Kernel 1: Q norm + RoPE (grid = (num_tokens, 1, 1), block = (32 * HQ, 1, 1) -> e.g. 128 threads for HQ=4!)
extern "C" __global__ void __launch_bounds__(128) qsa_pre_indexer_q_bf16(
    const uint16_t* __restrict__ q_ptr, int64_t q_stride_token,
    const int64_t* __restrict__ pos_ptr, int64_t pos_stride_axis, int64_t pos_stride_token,
    const uint16_t* __restrict__ cos_sin_ptr, const float* __restrict__ inv_freq_ptr, float mscale,
    const uint16_t* __restrict__ q_norm_w, float eps,
    uint16_t* __restrict__ q_out_ptr, int64_t q_out_stride_token, int64_t q_out_stride_head,
    int num_tokens, int HQ,
    int is_2d_pos, int mrope_h, int mrope_w, int table_free) {
  const int tok = blockIdx.x;
  const int h = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (tok >= num_tokens || h >= HQ) return;

  const uint16_t* qp = q_ptr + (int64_t)tok * q_stride_token + (int64_t)h * 128;
  float x0 = bf16_to_f32(qp[lane]);
  float x1 = bf16_to_f32(qp[lane + 32]);
  float x2 = bf16_to_f32(qp[lane + 64]);
  float x3 = bf16_to_f32(qp[lane + 96]);

  int64_t pt = pos_ptr[(int64_t)tok * pos_stride_token];
  int64_t ph = is_2d_pos ? pos_ptr[pos_stride_axis + (int64_t)tok * pos_stride_token] : pt;
  int64_t pw = is_2d_pos ? pos_ptr[2 * pos_stride_axis + (int64_t)tok * pos_stride_token] : pt;

  uint16_t o0, o1, o2, o3;
  norm_rope_wave32(x0, x1, x2, x3, pt, ph, pw, cos_sin_ptr, inv_freq_ptr, mscale, q_norm_w, eps,
                   is_2d_pos, mrope_h, mrope_w, table_free, o0, o1, o2, o3);

  uint16_t* op = q_out_ptr + (int64_t)tok * q_out_stride_token + (int64_t)h * q_out_stride_head;
  op[lane] = o0; op[lane + 32] = o1; op[lane + 64] = o2; op[lane + 96] = o3;
}

// Kernel 2: K pool + norm + RoPE + state ring commit (grid = (num_k_work, 1, 1), block = (32, 1, 1))
extern "C" __global__ void __launch_bounds__(32) qsa_pre_indexer_k_bf16(
    const uint16_t* __restrict__ k_ptr, int64_t k_stride_token,
    const int64_t* __restrict__ pos_ptr, int64_t pos_stride_axis, int64_t pos_stride_token,
    const uint16_t* __restrict__ cos_sin_ptr, const float* __restrict__ inv_freq_ptr, float mscale,
    const uint16_t* __restrict__ k_norm_w, float eps,
    uint16_t* __restrict__ state_cache_ptr, int64_t sc_stride_blk, int64_t sc_stride_tok,
    const int64_t* __restrict__ state_slots_ptr,
    const int32_t* __restrict__ state_bt_ptr, int64_t sbt_stride_req,
    const int32_t* __restrict__ query_start_loc_ptr,
    const int64_t* __restrict__ logical_pos_ptr,
    const int64_t* __restrict__ comp_slots_ptr,
    const int32_t* __restrict__ k_work_meta_ptr,
    uint16_t* __restrict__ comp_cache_ptr, int64_t cc_stride_blk, int64_t cc_stride_tok,
    int num_tokens, int num_state_blocks, int num_comp_blocks, int num_k_work,
    int compress_ratio, int state_size, int comp_page_size,
    int is_2d_pos, int is_k_mrope, int cache_has_rope_pos,
    int mrope_h, int mrope_w, int table_free) {
  const int pid = blockIdx.x;
  const int lane = threadIdx.x & 31;
  if (pid >= num_k_work) return;

  const int req = k_work_meta_ptr[pid * 2];
  const int work_in_req = k_work_meta_ptr[pid * 2 + 1];
  if (req < 0) return;

  const int q_start = query_start_loc_ptr[req];
  const int q_end = query_start_loc_ptr[req + 1];
  const int q_len = q_end - q_start;
  const int64_t chunk_end = logical_pos_ptr[q_end - 1];
  const int64_t chunk_start = chunk_end - q_len + 1;
  const int64_t num_groups = (chunk_end + 1) / compress_ratio - chunk_start / compress_ratio;

  if (work_in_req < num_groups) {
    const int64_t first_boundary = ((chunk_start + compress_ratio) / compress_ratio) * compress_ratio - 1;
    const int64_t end_pos = first_boundary + (int64_t)work_in_req * compress_ratio;
    const int64_t boundary_tok = q_start + end_pos - chunk_start;
    const bool valid_tok = (boundary_tok >= q_start) && (boundary_tok < q_end) && (boundary_tok < num_tokens);
    const int64_t comp_slot = valid_tok ? comp_slots_ptr[boundary_tok] : -1;
    const bool valid = valid_tok && (comp_slot >= 0) && (comp_slot < (int64_t)num_comp_blocks * comp_page_size);

    const int32_t sb = state_bt_ptr[(int64_t)req * sbt_stride_req];
    const bool sb_valid = (sb >= 0) && (sb < num_state_blocks);
    const int64_t safe_sb = sb >= 0 ? (int64_t)sb : 0;

    float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f, sum3 = 0.f;
    if (valid) {
      for (int g = 0; g < compress_ratio; ++g) {
        const int64_t spos = end_pos - (compress_ratio - 1) + g;
        const uint16_t* src = nullptr;
        if (spos >= chunk_start) {
          const int64_t st = q_start + spos - chunk_start;
          if (st >= q_start && st < q_end && st < num_tokens) {
            src = k_ptr + st * k_stride_token;
          }
        } else if (sb_valid) {
          src = state_cache_ptr + safe_sb * sc_stride_blk + (spos % state_size) * sc_stride_tok;
        }
        if (src) {
          sum0 += bf16_to_f32(src[lane]);
          sum1 += bf16_to_f32(src[lane + 32]);
          sum2 += bf16_to_f32(src[lane + 64]);
          sum3 += bf16_to_f32(src[lane + 96]);
        }
      }
      const float inv_cr = 1.0f / (float)compress_ratio;
      float p0 = bf16_to_f32(f32_to_bf16_rne(sum0 * inv_cr));
      float p1 = bf16_to_f32(f32_to_bf16_rne(sum1 * inv_cr));
      float p2 = bf16_to_f32(f32_to_bf16_rne(sum2 * inv_cr));
      float p3 = bf16_to_f32(f32_to_bf16_rne(sum3 * inv_cr));

      const int64_t first_pos = end_pos - (compress_ratio - 1);
      int64_t pt = first_pos, ph = first_pos, pw = first_pos;
      if (cache_has_rope_pos) {
        if (first_pos >= chunk_start) {
          const int64_t ft = q_start + first_pos - chunk_start;
          if (ft >= q_start && ft < q_end && ft < num_tokens) {
            pt = pos_ptr[ft * pos_stride_token];
            ph = is_2d_pos ? pos_ptr[pos_stride_axis + ft * pos_stride_token] : pt;
            pw = is_2d_pos ? pos_ptr[2 * pos_stride_axis + ft * pos_stride_token] : pt;
          } else { pt = ph = pw = 0; }
        } else if (sb_valid) {
          const int64_t* tail = reinterpret_cast<const int64_t*>(
              state_cache_ptr + safe_sb * sc_stride_blk + (first_pos % state_size) * sc_stride_tok + 128);
          pt = tail[0]; ph = tail[1]; pw = tail[2];
        } else { pt = ph = pw = 0; }
      }

      uint16_t o0, o1, o2, o3;
      norm_rope_wave32(p0, p1, p2, p3, pt, ph, pw, cos_sin_ptr, inv_freq_ptr, mscale, k_norm_w, eps,
                       is_k_mrope, mrope_h, mrope_w, table_free, o0, o1, o2, o3);

      const int64_t cblk = comp_slot / comp_page_size;
      const int64_t crow = comp_slot % comp_page_size;
      uint16_t* dst = comp_cache_ptr + cblk * cc_stride_blk + crow * cc_stride_tok;
      dst[lane] = o0; dst[lane + 32] = o1; dst[lane + 64] = o2; dst[lane + 96] = o3;
    }
  }

  if (work_in_req == 0) {
    __threadfence_block();
    const int n_state = q_len < state_size ? q_len : state_size;
    for (int off = 0; off < n_state; ++off) {
      const int tok = q_end - n_state + off;
      if (tok < q_start || tok >= q_end || tok >= num_tokens) continue;
      const int64_t slot = state_slots_ptr[tok];
      if (slot < 0 || slot >= (int64_t)num_state_blocks * state_size) continue;
      uint16_t* srow = state_cache_ptr + (slot / state_size) * sc_stride_blk + (slot % state_size) * sc_stride_tok;
      const uint16_t* src = k_ptr + (int64_t)tok * k_stride_token;
      srow[lane] = src[lane];
      srow[lane + 32] = src[lane + 32];
      srow[lane + 64] = src[lane + 64];
      srow[lane + 96] = src[lane + 96];
      if (cache_has_rope_pos && lane == 0) {
        int64_t pt = pos_ptr[(int64_t)tok * pos_stride_token];
        int64_t ph = is_2d_pos ? pos_ptr[pos_stride_axis + (int64_t)tok * pos_stride_token] : pt;
        int64_t pw = is_2d_pos ? pos_ptr[2 * pos_stride_axis + (int64_t)tok * pos_stride_token] : pt;
        int64_t* tail = reinterpret_cast<int64_t*>(srow + 128);
        tail[0] = pt; tail[1] = ph; tail[2] = pw;
      }
    }
  }
}
