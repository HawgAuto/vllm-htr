// gumbel_sample_aot.hip — AOT gfx1151 Wave32 HIP replacement for Triton _gumbel_sample_kernel.
// Exact bit-for-bit Murmur3 + degree-8 log1p_neg_stable + Wave32 warp-reduce argmax.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ uint32_t murmur3_rotl32(uint32_t v, int s) {
  return (v << s) | (v >> (32 - s));
}
__device__ __forceinline__ uint32_t murmur3_mix(uint32_t h, uint32_t k) {
  k *= 0xCC9E2D51u;
  k = murmur3_rotl32(k, 15);
  k *= 0x1B873593u;
  h ^= k;
  h = murmur3_rotl32(h, 13);
  return h * 5u + 0xE6546B64u;
}
__device__ __forceinline__ uint32_t murmur3_fmix32(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  return h ^ (h >> 16);
}
__device__ __forceinline__ uint32_t murmur3_hash32(int64_t seed, int64_t pos, uint32_t offset) {
  uint32_t h = 0u;
  h = murmur3_mix(h, (uint32_t)((uint64_t)seed & 0xFFFFFFFFull));
  h = murmur3_mix(h, (uint32_t)(((uint64_t)seed >> 32) & 0xFFFFFFFFull));
  h = murmur3_mix(h, (uint32_t)((uint64_t)pos & 0xFFFFFFFFull));
  h = murmur3_mix(h, offset);
  return murmur3_fmix32(h ^ 16u);
}
__device__ __forceinline__ float murmur3_uniform32(int64_t seed, int64_t pos, uint32_t offset) {
  uint32_t r = murmur3_hash32(seed, pos, offset);
  int32_t hi16 = (int32_t)(r >> 16);
  int32_t lo16 = (int32_t)(r & 0xFFFFu);
  return (float)hi16 * 1.52587890625e-05f + ((float)lo16 + 0.5f) * 2.3283064365386963e-10f;
}
__device__ __forceinline__ float log1p_neg_stable(float value) {
  float poly = 1.0f / 8.0f;
  poly = 1.0f / 7.0f + value * poly;
  poly = 1.0f / 6.0f + value * poly;
  poly = 1.0f / 5.0f + value * poly;
  poly = 1.0f / 4.0f + value * poly;
  poly = 1.0f / 3.0f + value * poly;
  poly = 1.0f / 2.0f + value * poly;
  poly = 1.0f + value * poly;
  float series = -value * poly;
  float d = 1.0f - value;
  float direct = logf(d > 5.960464477539063e-08f ? d : 5.960464477539063e-08f);
  return (value < 0.25f) ? series : direct;
}
__device__ __forceinline__ float load_f32_elem(const float* p, int64_t idx) { return p[idx]; }
__device__ __forceinline__ float load_bf16_elem(const uint16_t* p, int64_t idx) {
  return __uint_as_float(((uint32_t)p[idx]) << 16);
}
__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
  uint32_t u = __float_as_uint(f);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// Grid: (num_tokens, num_blocks, 1), Block: (256, 1, 1), BLOCK_SIZE=1024 (4 elems/thread)
// flags: bit0=is_drafting, bit1=apply_temperature, bit2=per_token_col, bit3=logits_is_bf16, bit4=cache_is_bf16
extern "C" __global__ void __launch_bounds__(256) gumbel_sample_block_fp32(
    int64_t* __restrict__ local_argmax_ptr, int64_t local_argmax_stride,
    float* __restrict__ local_max_ptr, int64_t local_max_stride,
    void* __restrict__ logits_cache_ptr, int64_t lc_s0, int64_t lc_s1,
    const int64_t* __restrict__ lc_col_ptr,
    const void* __restrict__ logits_ptr, int64_t logits_stride,
    const int32_t* __restrict__ expanded_idx_mapping_ptr,
    const int64_t* __restrict__ seeds_ptr,
    const int64_t* __restrict__ pos_ptr,
    const float* __restrict__ temp_ptr,
    int vocab_size, int flags) {

  const int64_t token_idx = (int64_t)blockIdx.x;
  const int block_idx = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;

  const bool is_drafting = (flags & 1) != 0;
  const bool apply_temp = (flags & 2) != 0;
  const bool per_token_col = (flags & 4) != 0;
  const bool logits_bf16 = (flags & 8) != 0;
  const bool cache_bf16 = (flags & 16) != 0;

  const int64_t req_state_idx = (int64_t)expanded_idx_mapping_ptr[token_idx];
  const bool is_valid_req = (req_state_idx >= 0);
  const float temp = is_valid_req ? temp_ptr[req_state_idx] : 0.0f;
  const int64_t seed = is_valid_req ? seeds_ptr[req_state_idx] : 0;
  int64_t pos = pos_ptr[token_idx];
  if (is_drafting) pos += (1ll << 30);

  int64_t cache_base = 0;
  const bool write_cache = (logits_cache_ptr != nullptr) && is_valid_req;
  if (write_cache) {
    int64_t col = per_token_col ? lc_col_ptr[token_idx] : lc_col_ptr[0];
    cache_base = req_state_idx * lc_s0 + col * lc_s1;
  }

  float best_val = -INFINITY;
  int best_idx = block_idx * 1024;
  const int base = block_idx * 1024;

#pragma unroll
  for (int k = 0; k < 4; ++k) {
    int idx = base + k * 256 + tid;
    if (idx < vocab_size) {
      float v = logits_bf16
                    ? load_bf16_elem((const uint16_t*)logits_ptr, token_idx * logits_stride + idx)
                    : load_f32_elem((const float*)logits_ptr, token_idx * logits_stride + idx);
      if (write_cache) {
        if (cache_bf16) {
          ((uint16_t*)logits_cache_ptr)[cache_base + idx] = f32_to_bf16(v);
        } else {
          ((float*)logits_cache_ptr)[cache_base + idx] = v;
        }
      }
      if (temp != 0.0f) {
        if (apply_temp) v = v / temp;
        float u = murmur3_uniform32(seed, pos, (uint32_t)idx);
        float noise = -logf(-log1p_neg_stable(u));
        v += noise;
      }
      if (v > best_val || (v == best_val && idx < best_idx)) {
        best_val = v;
        best_idx = idx;
      }
    }
  }

  // Wave32 reduction (tie-break: smaller idx wins)
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
    float ov = __shfl_xor(best_val, off, 32);
    int oi = __shfl_xor(best_idx, off, 32);
    if (ov > best_val || (ov == best_val && oi < best_idx)) {
      best_val = ov;
      best_idx = oi;
    }
  }

  __shared__ float s_val[8];
  __shared__ int s_idx[8];
  if (lane == 0) {
    s_val[wave] = best_val;
    s_idx[wave] = best_idx;
  }
  __syncthreads();

  if (tid == 0) {
    float bv = s_val[0];
    int bi = s_idx[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) {
      if (s_val[w] > bv || (s_val[w] == bv && s_idx[w] < bi)) {
        bv = s_val[w];
        bi = s_idx[w];
      }
    }
    local_argmax_ptr[token_idx * local_argmax_stride + block_idx] = (int64_t)bi;
    local_max_ptr[token_idx * local_max_stride + block_idx] = bv;
  }
}
