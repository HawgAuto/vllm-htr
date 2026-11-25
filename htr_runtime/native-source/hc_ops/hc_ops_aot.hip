// hc_ops_aot.hip v2 — Wave 7 RDNA 3.5 ISA upgraded AOT HIP replacements for all 5 Triton kernels in hc.py:
//   1. hc_grouped_gemma_rmsnorm_bf16
//   2. hc_silu_bf16
//   3. hc_gate_mix_bf16
//   4. hc_combine_bf16
//   5. hc_combine_norm_bf16
// 128-bit global_load_b128 / global_store_b128 + v_dot2_f32_bf16 + single-cycle v_perm_b32 BF16x2 pack!
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bf16_to_f32(uint16_t u) { return __uint_as_float(((uint32_t)u) << 16); }
__device__ __forceinline__ float bf16_lo(uint32_t u) { return __uint_as_float((u & 0xffffu) << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ uint16_t f32_to_bf16_rne(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ uint32_t f32x2_to_bf16x2_rne(float a, float b) {
  uint32_t ua = __float_as_uint(a);
  uint32_t ub = __float_as_uint(b);
  ua += 0x7fffu + ((ua >> 16) & 1u);
  ub += 0x7fffu + ((ub >> 16) & 1u);
  return __builtin_amdgcn_perm(ub, ua, 0x07060302u);
}
__device__ __forceinline__ float dot2_bf16(uint32_t a, uint32_t b, float acc) {
  float out;
  asm("v_dot2_f32_bf16 %0, %1, %2, %3" : "=v"(out) : "v"(a), "v"(b), "v"(acc));
  return out;
}
__device__ __forceinline__ float fast_sigmoid(float x) {
  return 1.0f / (1.0f + __builtin_amdgcn_exp2f(-x * 1.4426950408889634f));
}
__device__ __forceinline__ float warp_sum32(float v) {
  v += __shfl_xor(v, 16, 32);
  v += __shfl_xor(v, 8, 32);
  v += __shfl_xor(v, 4, 32);
  v += __shfl_xor(v, 2, 32);
  v += __shfl_xor(v, 1, 32);
  return v;
}
__device__ __forceinline__ float block_sum256(float v) {
  __shared__ float s_red[8];
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  v = warp_sum32(v);
  if (lane == 0) s_red[wave] = v;
  __syncthreads();
  if (wave == 0) {
    float x = (lane < 8) ? s_red[lane] : 0.0f;
    x += __shfl_xor(x, 4, 32);
    x += __shfl_xor(x, 2, 32);
    x += __shfl_xor(x, 1, 32);
    if (lane == 0) s_red[0] = x;
  }
  __syncthreads();
  return s_red[0];
}
__device__ __forceinline__ float load_w_f32(const void* w_ptr, int idx, int w_is_bf16) {
  return w_is_bf16 ? bf16_to_f32(((const uint16_t*)w_ptr)[idx]) : ((const float*)w_ptr)[idx];
}

// 1. Grouped Gemma RMSNorm: Grid (N * num_groups, 1, 1), Block (256, 1, 1)
extern "C" __global__ void __launch_bounds__(256) hc_grouped_gemma_rmsnorm_bf16(
    const uint16_t* __restrict__ x_ptr,
    const void* __restrict__ w_ptr,
    uint16_t* __restrict__ y_ptr,
    int64_t stride_x, int64_t stride_y,
    int group_dim, int num_groups, int w_shared, int w_is_bf16, float eps) {
  const int pid = blockIdx.x;
  const int group_id = pid % num_groups;
  const int row = pid / num_groups;
  const int g_off = group_id * group_dim;
  const uint16_t* xr = x_ptr + (int64_t)row * stride_x + g_off;
  uint16_t* yr = y_ptr + (int64_t)row * stride_y + g_off;
  const int w_base = w_shared ? 0 : g_off;

  const bool fast_vec = ((group_dim & 7) == 0) && (((uintptr_t)xr & 15u) == 0u) && (((uintptr_t)yr & 15u) == 0u);
  float sq = 0.0f;
  if (fast_vec) {
    const int n8 = group_dim >> 3;
    const u32x4* x8 = (const u32x4*)xr;
    for (int i = threadIdx.x; i < n8; i += 256) {
      u32x4 v = x8[i];
      sq = dot2_bf16(v[0], v[0], sq);
      sq = dot2_bf16(v[1], v[1], sq);
      sq = dot2_bf16(v[2], v[2], sq);
      sq = dot2_bf16(v[3], v[3], sq);
    }
    float rrms = rsqrtf(block_sum256(sq) / (float)group_dim + eps);
    u32x4* y8 = (u32x4*)yr;
    for (int i = threadIdx.x; i < n8; i += 256) {
      u32x4 v = x8[i], o;
      int d0 = w_base + (i << 3);
      if (w_is_bf16 && ((((uintptr_t)w_ptr + (size_t)d0 * 2u) & 15u) == 0u)) {
        u32x4 wv = *(const u32x4*)((const uint16_t*)w_ptr + d0);
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          float y0 = bf16_lo(v[g]) * rrms; y0 = __builtin_fmaf(y0, bf16_lo(wv[g]), y0);
          float y1 = bf16_hi(v[g]) * rrms; y1 = __builtin_fmaf(y1, bf16_hi(wv[g]), y1);
          o[g] = f32x2_to_bf16x2_rne(y0, y1);
        }
      } else {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          float y0 = bf16_lo(v[g]) * rrms; y0 = __builtin_fmaf(y0, load_w_f32(w_ptr, d0 + 2*g, w_is_bf16), y0);
          float y1 = bf16_hi(v[g]) * rrms; y1 = __builtin_fmaf(y1, load_w_f32(w_ptr, d0 + 2*g + 1, w_is_bf16), y1);
          o[g] = f32x2_to_bf16x2_rne(y0, y1);
        }
      }
      y8[i] = o;
    }
    return;
  }

  for (int d = threadIdx.x; d < group_dim; d += 256) {
    float v = bf16_to_f32(xr[d]);
    sq += v * v;
  }
  float rrms = rsqrtf(block_sum256(sq) / (float)group_dim + eps);
  for (int d = threadIdx.x; d < group_dim; d += 256) {
    float y = bf16_to_f32(xr[d]) * rrms;
    float w = load_w_f32(w_ptr, w_base + d, w_is_bf16);
    y = __builtin_fmaf(y, w, y);
    yr[d] = f32_to_bf16_rne(y);
  }
}

// 2. HC SiLU: Grid (N, 1, 1), Block (256, 1, 1)
extern "C" __global__ void __launch_bounds__(256) hc_silu_bf16(
    const uint16_t* __restrict__ x_ptr,
    uint16_t* __restrict__ y_ptr,
    int64_t stride_x, int64_t stride_y,
    int dim, int hc) {
  const int row = blockIdx.x;
  const float inv_hc = 1.0f / (float)hc;
  const uint16_t* xr = x_ptr + (int64_t)row * stride_x;
  uint16_t* yr = y_ptr + (int64_t)row * stride_y;
  if (((dim & 7) == 0) && (((uintptr_t)xr & 15u) == 0u) && (((uintptr_t)yr & 15u) == 0u)) {
    const int n8 = dim >> 3;
    const u32x4* x8 = (const u32x4*)xr;
    u32x4* y8 = (u32x4*)yr;
    for (int i = threadIdx.x; i < n8; i += 256) {
      u32x4 v = x8[i], o;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        float a = bf16_lo(v[g]) * inv_hc; a *= fast_sigmoid(a);
        float b = bf16_hi(v[g]) * inv_hc; b *= fast_sigmoid(b);
        o[g] = f32x2_to_bf16x2_rne(a, b);
      }
      y8[i] = o;
    }
    return;
  }
  for (int d = threadIdx.x; d < dim; d += 256) {
    float x = bf16_to_f32(xr[d]) * inv_hc;
    yr[d] = f32_to_bf16_rne(x * fast_sigmoid(x));
  }
}

// 3. HC Gate Mix: Grid ((hc_dim + 255) / 256, N, 1), Block (256, 1, 1)
extern "C" __global__ void __launch_bounds__(256) hc_gate_mix_bf16(
    const uint16_t* __restrict__ x_ptr,
    const uint16_t* __restrict__ g_ptr,
    uint16_t* __restrict__ y_ptr,
    int64_t stride_x, int64_t stride_g, int64_t stride_y,
    int hc_dim, int hc) {
  const int d = blockIdx.x * 256 + threadIdx.x;
  const int row = blockIdx.y;
  if (d >= hc_dim) return;
  const uint16_t* xr = x_ptr + (int64_t)row * stride_x + d;
  const uint16_t* gr = g_ptr + (int64_t)row * stride_g + d;
  float acc = 0.0f;
  for (int s = 0; s < hc; ++s) {
    float gv = fast_sigmoid(bf16_to_f32(gr[(int64_t)s * hc_dim]));
    float xv = bf16_to_f32(xr[(int64_t)s * hc_dim]);
    acc = __builtin_fmaf(gv, xv, acc);
  }
  y_ptr[(int64_t)row * stride_y + d] = f32_to_bf16_rne(acc / (float)hc);
}

// 4. HC Combine: Grid ((hc_dim + 255) / 256, N, 1), Block (256, 1, 1)
// Exact signature matching hc.py:338:
//   (block_output, residual, injection_logits [bf16], out, stride_block, stride_res, stride_inj, stride_out, hc_dim, hc_count)
extern "C" __global__ void __launch_bounds__(256) hc_combine_bf16(
    const uint16_t* __restrict__ block_ptr,
    const uint16_t* __restrict__ res_ptr,
    const uint16_t* __restrict__ inj_ptr,
    uint16_t* __restrict__ out_ptr,
    int64_t stride_block, int64_t stride_res, int64_t stride_inj, int64_t stride_out,
    int hc_dim, int hc) {
  const int d = blockIdx.x * 256 + threadIdx.x;
  const int row = blockIdx.y;
  if (d >= hc_dim) return;
  const float inv_hc = 1.0f / (float)hc;
  const float bv = bf16_to_f32(block_ptr[(int64_t)row * stride_block + d]);
  const uint16_t* rr = res_ptr + (int64_t)row * stride_res + d;
  const uint16_t* ir = inj_ptr + (int64_t)row * stride_inj;
  uint16_t* orow = out_ptr + (int64_t)row * stride_out + d;
  for (int s = 0; s < hc; ++s) {
    float inj = 2.0f * fast_sigmoid(bf16_to_f32(ir[s]) * inv_hc);
    float rv = bf16_to_f32(rr[(int64_t)s * hc_dim]);
    orow[(int64_t)s * hc_dim] = f32_to_bf16_rne(__builtin_fmaf(bv, inj, rv));
  }
}

// 5. HC Combine + Norm: Grid (hc_count, N, 1), Block (256, 1, 1)
// Exact signature matching hc.py:482:
//   (block_output, residual, injection_logits [bf16], norm_weight, out, y,
//    stride_block, stride_res, stride_inj, stride_out, stride_y, hc_dim, hc_count, w_shared, w_is_bf16, eps)
extern "C" __global__ void __launch_bounds__(256) hc_combine_norm_bf16(
    const uint16_t* __restrict__ block_ptr,
    const uint16_t* __restrict__ res_ptr,
    const uint16_t* __restrict__ inj_ptr,
    const void* __restrict__ w_ptr,
    uint16_t* __restrict__ out_ptr,
    uint16_t* __restrict__ y_ptr,
    int64_t stride_block, int64_t stride_res, int64_t stride_inj,
    int64_t stride_out, int64_t stride_y,
    int hc_dim, int hc, int w_shared, int w_is_bf16, float eps) {
  const int stream = blockIdx.x;
  const int row = blockIdx.y;
  const int s_off = stream * hc_dim;
  const float inj = 2.0f * fast_sigmoid(bf16_to_f32(inj_ptr[(int64_t)row * stride_inj + stream]) / (float)hc);
  const uint16_t* br = block_ptr + (int64_t)row * stride_block;
  const uint16_t* rr = res_ptr + (int64_t)row * stride_res + s_off;
  uint16_t* orow = out_ptr + (int64_t)row * stride_out + s_off;
  uint16_t* yrow = y_ptr + (int64_t)row * stride_y + s_off;
  const int w_base = w_shared ? 0 : s_off;

  const bool fast_vec = ((hc_dim & 7) == 0) && (((uintptr_t)br & 15u) == 0u) && (((uintptr_t)rr & 15u) == 0u)
                        && (((uintptr_t)orow & 15u) == 0u) && (((uintptr_t)yrow & 15u) == 0u);
  float sq = 0.0f;
  if (fast_vec) {
    const int n8 = hc_dim >> 3;
    const u32x4* b8 = (const u32x4*)br;
    const u32x4* r8 = (const u32x4*)rr;
    u32x4* o8 = (u32x4*)orow;
    for (int i = threadIdx.x; i < n8; i += 256) {
      u32x4 bv = b8[i], rv = r8[i], ov;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        float c0 = __builtin_fmaf(bf16_lo(bv[g]), inj, bf16_lo(rv[g]));
        float c1 = __builtin_fmaf(bf16_hi(bv[g]), inj, bf16_hi(rv[g]));
        ov[g] = f32x2_to_bf16x2_rne(c0, c1);
        sq = dot2_bf16(ov[g], ov[g], sq);
      }
      o8[i] = ov;
    }
    float rrms = rsqrtf(block_sum256(sq) / (float)hc_dim + eps);
    u32x4* y8 = (u32x4*)yrow;
    for (int i = threadIdx.x; i < n8; i += 256) {
      u32x4 ov = o8[i], yv;
      int d0 = w_base + (i << 3);
      if (w_is_bf16 && ((((uintptr_t)w_ptr + (size_t)d0 * 2u) & 15u) == 0u)) {
        u32x4 wv = *(const u32x4*)((const uint16_t*)w_ptr + d0);
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          float y0 = bf16_lo(ov[g]) * rrms; y0 = __builtin_fmaf(y0, bf16_lo(wv[g]), y0);
          float y1 = bf16_hi(ov[g]) * rrms; y1 = __builtin_fmaf(y1, bf16_hi(wv[g]), y1);
          yv[g] = f32x2_to_bf16x2_rne(y0, y1);
        }
      } else {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          float y0 = bf16_lo(ov[g]) * rrms; y0 = __builtin_fmaf(y0, load_w_f32(w_ptr, d0 + 2*g, w_is_bf16), y0);
          float y1 = bf16_hi(ov[g]) * rrms; y1 = __builtin_fmaf(y1, load_w_f32(w_ptr, d0 + 2*g + 1, w_is_bf16), y1);
          yv[g] = f32x2_to_bf16x2_rne(y0, y1);
        }
      }
      y8[i] = yv;
    }
    return;
  }

  for (int d = threadIdx.x; d < hc_dim; d += 256) {
    float c = __builtin_fmaf(bf16_to_f32(br[d]), inj, bf16_to_f32(rr[d]));
    uint16_t b = f32_to_bf16_rne(c);
    orow[d] = b;
    float bf = bf16_to_f32(b);
    sq = __builtin_fmaf(bf, bf, sq);
  }
  float rrms = rsqrtf(block_sum256(sq) / (float)hc_dim + eps);
  for (int d = threadIdx.x; d < hc_dim; d += 256) {
    float v = bf16_to_f32(orow[d]) * rrms;
    float w = load_w_f32(w_ptr, w_base + d, w_is_bf16);
    yrow[d] = f32_to_bf16_rne(__builtin_fmaf(v, w, v));
  }
}
