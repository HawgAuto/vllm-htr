// causal_conv1d_fwd_aot.hip — 128-bit vectorized AOT gfx1151 Wave32 HIP replacement for Triton _causal_conv1d_fwd_kernel(_gs)
// - 8 contiguous BF16 channels/thread via global_load_b128 / global_store_b128 (u32x4)
// - 64 threads = 512 channels/WG, processes 4 consecutive BLOCK_M=8 chunks (32 tokens) per WG
// - RDNA 3.5 single-cycle v_perm_b32 packed BF16x2 RNE conversion (__builtin_amdgcn_perm(ub, ua, 0x07060302u))
// - Full APC/prefix-cache checkpoint & short-sequence conv_state parity, 0 spills, 0 LDS!
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bf16_lo(uint32_t u) { return __uint_as_float((u & 0xffffu) << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ float bf16_to_f32(uint16_t u) { return __uint_as_float(((uint32_t)u) << 16); }

// Single-instruction RDNA 3.5 packed BF16x2 RNE conversion via V_PERM_B32
__device__ __forceinline__ uint32_t f32x2_to_bf16x2_rne(float a, float b) {
  uint32_t ua = __float_as_uint(a);
  uint32_t ub = __float_as_uint(b);
  ua += 0x7fffu + ((ua >> 16) & 1u);
  ub += 0x7fffu + ((ub >> 16) & 1u);
  return __builtin_amdgcn_perm(ub, ua, 0x07060302u);
}
__device__ __forceinline__ float fast_silu(float x) {
  float emx = __builtin_amdgcn_exp2f(-x * 1.4426950408889634f);
  return x / (1.0f + emx);
}

// Grid: (cdiv(dim, 512), cdiv(num_progs, 4), 1), Block: (64, 1, 1) -> 8 BF16s/thread = 512 channels/WG!
extern "C" __global__ void __launch_bounds__(64) causal_conv1d_fwd_w4_bf16_v8(
    const uint16_t* __restrict__ x_ptr,
    const uint16_t* __restrict__ w_ptr,
    const uint16_t* __restrict__ bias_ptr,
    uint16_t* __restrict__ conv_states_ptr,
    const int32_t* __restrict__ cache_indices_ptr,
    const uint8_t* __restrict__ has_initial_states_ptr,
    const int32_t* __restrict__ query_start_loc_ptr,
    const int32_t* __restrict__ batch_ptr,
    const int32_t* __restrict__ token_chunk_offset_ptr,
    const int32_t* __restrict__ block_idx_first_ptr,
    const int32_t* __restrict__ block_idx_last_ptr,
    const int32_t* __restrict__ initial_state_idx_ptr,
    const int32_t* __restrict__ num_computed_tokens_ptr,
    uint16_t* __restrict__ o_ptr,
    int dim,
    int num_cache_lines,
    int num_progs,
    int64_t stride_x_token,
    int64_t stride_w_dim,
    int64_t stride_istate_seq,
    int64_t stride_istate_dim,
    int64_t stride_istate_tok,
    int64_t stride_cache_indices,
    int64_t stride_o_token,
    int B_size,
    int pad_slot_id,
    int null_block_id,
    int flags)
{
  const int f0 = (blockIdx.x * 64 + threadIdx.x) * 8;
  if (f0 >= dim) return;

  const bool HAS_BIAS = (flags & 1) != 0;
  const bool SILU = (flags & 2) != 0;
  const bool IS_APC = (flags & 4) != 0;
  const bool HAS_NULL_BLOCK = (flags & 8) != 0;
  constexpr int BLOCK_M = 8;
  constexpr int STATE_LEN = 3;
  constexpr int GROUP_C = 4;

  float w0[8], w1[8], w2[8], w3[8], bv[8];
  if (stride_w_dim == 4) {
    const u32x4* wp = (const u32x4*)(w_ptr + (int64_t)f0 * 4);
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      u32x4 r = wp[g];
      w0[2*g]   = bf16_lo(r[0]); w1[2*g]   = bf16_hi(r[0]);
      w2[2*g]   = bf16_lo(r[1]); w3[2*g]   = bf16_hi(r[1]);
      w0[2*g+1] = bf16_lo(r[2]); w1[2*g+1] = bf16_hi(r[2]);
      w2[2*g+1] = bf16_lo(r[3]); w3[2*g+1] = bf16_hi(r[3]);
    }
  } else {
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const uint16_t* wp = w_ptr + (int64_t)(f0 + c) * stride_w_dim;
      w0[c] = bf16_to_f32(wp[0]); w1[c] = bf16_to_f32(wp[1]);
      w2[c] = bf16_to_f32(wp[2]); w3[c] = bf16_to_f32(wp[3]);
    }
  }
  if (HAS_BIAS) {
    u32x4 br = *(const u32x4*)(bias_ptr + f0);
#pragma unroll
    for (int g = 0; g < 4; ++g) { bv[2*g] = bf16_lo(br[g]); bv[2*g+1] = bf16_hi(br[g]); }
  } else {
#pragma unroll
    for (int c = 0; c < 8; ++c) bv[c] = 0.0f;
  }

  auto load_x8 = [&](int64_t tok_idx, float dst[8]) {
    u32x4 v = *(const u32x4*)(x_ptr + tok_idx * stride_x_token + f0);
#pragma unroll
    for (int g = 0; g < 4; ++g) { dst[2*g] = bf16_lo(v[g]); dst[2*g+1] = bf16_hi(v[g]); }
  };
  auto store_o8 = [&](int64_t tok_idx, const float src[8]) {
    u32x4 v;
#pragma unroll
    for (int g = 0; g < 4; ++g) v[g] = f32x2_to_bf16x2_rne(src[2*g], src[2*g+1]);
    __builtin_nontemporal_store(v, (u32x4*)(o_ptr + tok_idx * stride_o_token + f0));
  };

  int64_t prev_seq = -1;
  int prev_chunk = -2;
  float c0[8], c1[8], c2[8];
#pragma unroll
  for (int c = 0; c < 8; ++c) { c0[c] = 0.0f; c1[c] = 0.0f; c2[c] = 0.0f; }

  const int prog_base = blockIdx.y * GROUP_C;
  const int prog_end = (prog_base + GROUP_C < num_progs) ? (prog_base + GROUP_C) : num_progs;

  for (int p = prog_base; p < prog_end; ++p) {
    const int64_t idx_seq = (int64_t)batch_ptr[p];
    const int chunk_offset = token_chunk_offset_ptr[p];
    if (idx_seq == pad_slot_id) { prev_seq = -1; continue; }

    const int seq_start = query_start_loc_ptr[idx_seq];
    const int seq_end = query_start_loc_ptr[idx_seq + 1];
    const int seqlen = seq_end - seq_start;

    int n_block_to_fill = 0, current_last_index = 0, conv_state_init_index = 0, current_first_index = 0, last_full_block_token_index = 0;
    if (IS_APC) {
      current_first_index = block_idx_first_ptr[idx_seq];
      current_last_index = block_idx_last_ptr[idx_seq];
      int seq_completed = num_computed_tokens_ptr[idx_seq];
      int seq_completed_off = B_size - (seq_completed % B_size);
      int rem = (seqlen - seq_completed_off) % B_size;
      if (rem < 0) rem += B_size;
      last_full_block_token_index = seq_end - rem;
      if (rem == 0) last_full_block_token_index -= B_size;
      n_block_to_fill = current_last_index - current_first_index;
      conv_state_init_index = initial_state_idx_ptr[idx_seq];
    }

    const int64_t conv_in_coord = (int64_t)cache_indices_ptr[idx_seq * stride_cache_indices + conv_state_init_index];
    if ((HAS_NULL_BLOCK && conv_in_coord == null_block_id) || conv_in_coord == pad_slot_id || conv_in_coord < 0 || conv_in_coord >= num_cache_lines) { prev_seq = -1; continue; }

    const int token_offset = BLOCK_M * chunk_offset;
    const int seg_len = (seqlen - token_offset < BLOCK_M) ? (seqlen - token_offset) : BLOCK_M;
    const bool contiguous = (idx_seq == prev_seq && chunk_offset == prev_chunk + 1 && chunk_offset > 0);

    if (chunk_offset == 0) {
      const bool load_init = (has_initial_states_ptr[idx_seq] != 0);
      uint16_t* cs_in = conv_states_ptr + conv_in_coord * stride_istate_seq + (int64_t)f0 * stride_istate_dim;
      if (load_init) {
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          uint16_t* p_c = cs_in + (int64_t)c * stride_istate_dim;
          c0[c] = bf16_to_f32(p_c[0 * stride_istate_tok]);
          c1[c] = bf16_to_f32(p_c[1 * stride_istate_tok]);
          c2[c] = bf16_to_f32(p_c[2 * stride_istate_tok]);
        }
      } else {
#pragma unroll
        for (int c = 0; c < 8; ++c) { c0[c] = 0.0f; c1[c] = 0.0f; c2[c] = 0.0f; }
      }

      if (STATE_LEN <= seqlen) {
        const int64_t conv_out_coord = (int64_t)cache_indices_ptr[idx_seq * stride_cache_indices + current_last_index];
        if (conv_out_coord >= 0 && conv_out_coord < num_cache_lines && conv_out_coord != pad_slot_id) {
        uint16_t* cs_out = conv_states_ptr + conv_out_coord * stride_istate_seq + (int64_t)f0 * stride_istate_dim;
#pragma unroll
        for (int s = 0; s < STATE_LEN; ++s) {
          const uint16_t* xp = x_ptr + (int64_t)(seq_start + seqlen - STATE_LEN + s) * stride_x_token + f0;
#pragma unroll
          for (int c = 0; c < 8; ++c) cs_out[(int64_t)c * stride_istate_dim + (int64_t)s * stride_istate_tok] = xp[c];
        }
        }
      } else {
        const int val = STATE_LEN - seqlen;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          uint16_t* p_c = cs_in + (int64_t)c * stride_istate_dim;
          uint16_t oldv[3] = {0, 0, 0};
          if (load_init && conv_in_coord < num_cache_lines) {
            oldv[0] = p_c[0 * stride_istate_tok]; oldv[1] = p_c[1 * stride_istate_tok]; oldv[2] = p_c[2 * stride_istate_tok];
          }
#pragma unroll
          for (int s = 0; s < STATE_LEN; ++s) {
            if (s + seqlen < STATE_LEN) p_c[(int64_t)s * stride_istate_tok] = oldv[s + seqlen];
            else p_c[(int64_t)s * stride_istate_tok] = x_ptr[(int64_t)(seq_start + s - val) * stride_x_token + f0 + c];
          }
        }
      }
    } else {
      if (!contiguous) {
        load_x8((int64_t)(seq_start + token_offset - 3), c0);
        load_x8((int64_t)(seq_start + token_offset - 2), c1);
        load_x8((int64_t)(seq_start + token_offset - 1), c2);
      }
      if (IS_APC && (chunk_offset - 1) < n_block_to_fill) {
        int base_tok = last_full_block_token_index - (n_block_to_fill - chunk_offset) * B_size - STATE_LEN;
        const int64_t conv_out_coord = (int64_t)cache_indices_ptr[idx_seq * stride_cache_indices + current_first_index + (chunk_offset - 1)];
        if (conv_out_coord >= 0 && conv_out_coord < num_cache_lines && conv_out_coord != pad_slot_id && (!HAS_NULL_BLOCK || conv_out_coord != null_block_id)) {
          uint16_t* cs_out = conv_states_ptr + conv_out_coord * stride_istate_seq + (int64_t)f0 * stride_istate_dim;
#pragma unroll
          for (int s = 0; s < STATE_LEN; ++s) {
            int t_abs = base_tok + s;
            if (t_abs >= 0 && t_abs < seq_end) {
              const uint16_t* xp = x_ptr + (int64_t)t_abs * stride_x_token + f0;
#pragma unroll
              for (int c = 0; c < 8; ++c) cs_out[(int64_t)c * stride_istate_dim + (int64_t)s * stride_istate_tok] = xp[c];
            } else {
#pragma unroll
              for (int c = 0; c < 8; ++c) cs_out[(int64_t)c * stride_istate_dim + (int64_t)s * stride_istate_tok] = 0;
            }
          }
        }
      }
    }

    for (int t = 0; t < seg_len; ++t) {
      float xv[8], ov[8];
      load_x8((int64_t)(seq_start + token_offset + t), xv);
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        float a = __builtin_fmaf(xv[c], w3[c], __builtin_fmaf(c2[c], w2[c], __builtin_fmaf(c1[c], w1[c], __builtin_fmaf(c0[c], w0[c], bv[c]))));
        c0[c] = c1[c]; c1[c] = c2[c]; c2[c] = xv[c];
        ov[c] = SILU ? fast_silu(a) : a;
      }
      store_o8((int64_t)(seq_start + token_offset + t), ov);
    }

    prev_seq = idx_seq;
    prev_chunk = chunk_offset;
  }
}
