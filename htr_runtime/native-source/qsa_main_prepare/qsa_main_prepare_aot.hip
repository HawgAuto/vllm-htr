// qsa_main_prepare_aot.hip — AOT gfx1151 Wave32 HIP replacement for Triton _qsa_main_prepare_kernel.
// Handles D=256, ROT=64 (HALF=32 = 1 Wave32!): lane in 0..31 owns 8 dims (lane + 32*k, k=0..7).
// Zero LDS, zero register spills, 16 waves/SIMD, zero Triton JIT!
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

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

// Grid: ((num_tokens * (HQ + HK) + 3) / 4, 1, 1), Block: (128, 1, 1) -> 4 waves = 4 rows/WG
extern "C" __global__ void __launch_bounds__(128) qsa_main_prepare_d256_rot64(
    const uint16_t* __restrict__ qkv_ptr,
    const uint16_t* __restrict__ q_norm_w,
    const uint16_t* __restrict__ k_norm_w,
    float eps,
    const uint16_t* __restrict__ cos_sin_ptr,
    const float* __restrict__ inv_freq_ptr,
    float mscale,
    const int64_t* __restrict__ pos_ptr,
    int64_t pos_stride_axis, int64_t pos_stride_token,
    uint16_t* __restrict__ q_out_ptr,
    uint16_t* __restrict__ gate_out_ptr,
    uint16_t* __restrict__ cache_ptr,
    int64_t cache_stride_block, int64_t cache_stride_token, int64_t cache_stride_head,
    const int64_t* __restrict__ slots_ptr,
    int num_tokens, int HQ, int HK, int cs_stride, int page_size,
    int is_2d_pos, int mrope_h, int mrope_w, int table_free) {
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;  // 0..31 == HALF (ROT=64)
  const int row = blockIdx.x * 4 + wave;
  const int num_q_rows = num_tokens * HQ;
  const int total_rows = num_q_rows + num_tokens * HK;
  if (row >= total_rows) return;

  const bool is_k = (row >= num_q_rows);
  const int64_t W = (int64_t)2 * (HQ + HK) * 256;
  int64_t token;
  int head;
  int64_t base;
  const uint16_t* w_ptr;
  if (is_k) {
    const int ki = row - num_q_rows;
    token = ki / HK;
    head = ki - (int)token * HK;
    base = token * W + (int64_t)2 * HQ * 256 + (int64_t)head * 256;
    w_ptr = k_norm_w;
  } else {
    token = row / HQ;
    head = row - (int)token * HQ;
    base = token * W + (int64_t)head * 512;
    w_ptr = q_norm_w;
  }

  float x[8];
  float ssq = 0.0f;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    float v = bf16_to_f32(qkv_ptr[base + k * 32 + lane]);
    x[k] = v;
    ssq = __builtin_fmaf(v, v, ssq);
  }
  const float rs = rsqrtf(warp_sum32(ssq) * (1.0f / 256.0f) + eps);

  // RoPE over [0..63]: pair is (lane, lane + 32)
  int64_t pos = pos_ptr[token * pos_stride_token];
  if (is_2d_pos) {
    if ((lane % 3) == 1 && lane < 3 * mrope_h) {
      pos = pos_ptr[pos_stride_axis + token * pos_stride_token];
    } else if ((lane % 3) == 2 && lane < 3 * mrope_w) {
      pos = pos_ptr[2 * pos_stride_axis + token * pos_stride_token];
    }
  }
  float c_val, s_val;
  if (table_free) {
    float ang = (float)pos * inv_freq_ptr[lane];
    float sc, ss;
    sincosf(ang, &ss, &sc);
    c_val = bf16_to_f32(f32_to_bf16_rne(sc * mscale));
    s_val = bf16_to_f32(f32_to_bf16_rne(ss * mscale));
  } else {
    c_val = bf16_to_f32(cos_sin_ptr[pos * cs_stride + lane]);
    s_val = bf16_to_f32(cos_sin_ptr[pos * cs_stride + 32 + lane]);
  }

  const float n0 = x[0] * rs * (bf16_to_f32(w_ptr[lane]) + 1.0f);
  const float n1 = x[1] * rs * (bf16_to_f32(w_ptr[32 + lane]) + 1.0f);
  const uint16_t o0 = f32_to_bf16_rne(n0 * c_val - n1 * s_val);
  const uint16_t o1 = f32_to_bf16_rne(n1 * c_val + n0 * s_val);

  if (is_k) {
    const int64_t slot = slots_ptr[token];
    if (slot < 0) return;
    uint16_t* dst = cache_ptr
        + (slot / page_size) * cache_stride_block
        + (slot % page_size) * cache_stride_token
        + (int64_t)head * cache_stride_head;
    dst[lane] = o0;
    dst[32 + lane] = o1;
#pragma unroll
    for (int k = 2; k < 8; ++k) {
      dst[k * 32 + lane] = f32_to_bf16_rne(x[k] * rs * (bf16_to_f32(w_ptr[k * 32 + lane]) + 1.0f));
    }
    const uint16_t* v_src = qkv_ptr + token * W + (int64_t)(2 * HQ + HK) * 256 + (int64_t)head * 256;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      dst[256 + k * 32 + lane] = v_src[k * 32 + lane];
    }
  } else {
    const int64_t out_off = (token * HQ + head) * 256;
    uint16_t* qo = q_out_ptr + out_off;
    uint16_t* go = gate_out_ptr + out_off;
    qo[lane] = o0;
    qo[32 + lane] = o1;
#pragma unroll
    for (int k = 2; k < 8; ++k) {
      qo[k * 32 + lane] = f32_to_bf16_rne(x[k] * rs * (bf16_to_f32(w_ptr[k * 32 + lane]) + 1.0f));
    }
    const uint16_t* g_src = qkv_ptr + base + 256;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      go[k * 32 + lane] = g_src[k * 32 + lane];
    }
  }
}
