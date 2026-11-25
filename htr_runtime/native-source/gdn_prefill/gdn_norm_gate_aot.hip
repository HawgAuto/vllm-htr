// gdn_norm_gate_aot.hip — AOT gfx1151 Wave32 HIP replacement for Triton layer_norm_gated_fwd_kernel.
// Specialized for D=128, RMSNorm + Swish/SiLU or Sigmoid gate, BF16 x/g/y, BF16/FP32 weight.
// 1 Wave32 = 1 row of 128 dims (4 BF16s/lane = 64-bit u32x2 coalesced R/W), 0 LDS, 16 w/SIMD!
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ float dot2_bf16(uint32_t a, uint32_t b, float acc) { float out; asm("v_dot2_f32_bf16 %0, %1, %2, %3" : "=v"(out) : "v"(a), "v"(b), "v"(acc)); return out; }
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float bf16_lo(uint32_t u) { return __uint_as_float((u & 0xffffu) << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ uint16_t f32_to_bf16_rne(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ u32x2 pack4_bf16(float f0, float f1, float f2, float f3) {
  uint32_t lo = (uint32_t)f32_to_bf16_rne(f0) | ((uint32_t)f32_to_bf16_rne(f1) << 16);
  uint32_t hi = (uint32_t)f32_to_bf16_rne(f2) | ((uint32_t)f32_to_bf16_rne(f3) << 16);
  return (u32x2){lo, hi};
}
__device__ __forceinline__ float warp_sum32(float v) {
  v += __shfl_xor(v, 16, 32);
  v += __shfl_xor(v, 8, 32);
  v += __shfl_xor(v, 4, 32);
  v += __shfl_xor(v, 2, 32);
  v += __shfl_xor(v, 1, 32);
  return v;
}
__device__ __forceinline__ float gate_act(float g, int is_sigmoid) {
  float s = 1.0f / (1.0f + expf(-g));
  return is_sigmoid ? s : (g * s);
}

// Grid: ((T + 7) / 8, 1, 1), Block: (256, 1, 1) -> 8 waves = 8 rows per WG
// flags: bit0=has_weight, bit1=weight_is_bf16, bit2=is_sigmoid
extern "C" __global__ void __launch_bounds__(256) rms_norm_gated_bf16_d128(
    const uint16_t* __restrict__ x_ptr,
    const uint16_t* __restrict__ g_ptr,
    const void* __restrict__ w_ptr,
    uint16_t* __restrict__ y_ptr,
    float* __restrict__ rstd_ptr,
    int T, int H, int64_t g_stride_n, float eps, int flags) {

  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + wave;
  if (row >= T) return;

  const int d4 = lane * 4;
  const bool has_w = (flags & 1) != 0;
  const bool w_bf16 = (flags & 2) != 0;
  const int is_sig = (flags >> 2) & 1;

  u32x2 xv = *(const u32x2*)(x_ptr + (int64_t)row * 128 + d4);
  float x0 = bf16_lo(xv[0]), x1 = bf16_hi(xv[0]), x2 = bf16_lo(xv[1]), x3 = bf16_hi(xv[1]);
  float var = warp_sum32(dot2_bf16(xv[1], xv[1], dot2_bf16(xv[0], xv[0], 0.0f))) * (1.0f / 128.0f);
  float rstd = rsqrtf(var + eps);
  if (rstd_ptr != nullptr && lane == 0) rstd_ptr[row] = rstd;

  float w0 = 1.0f, w1 = 1.0f, w2 = 1.0f, w3 = 1.0f;
  if (has_w) {
    if (w_bf16) {
      u32x2 wv = *(const u32x2*)((const uint16_t*)w_ptr + d4);
      w0 = bf16_lo(wv[0]); w1 = bf16_hi(wv[0]); w2 = bf16_lo(wv[1]); w3 = bf16_hi(wv[1]);
    } else {
      const float* wp = (const float*)w_ptr + d4;
      w0 = wp[0]; w1 = wp[1]; w2 = wp[2]; w3 = wp[3];
    }
  }

  int64_t g_off = (int64_t)(row / H) * g_stride_n + (int64_t)(row % H) * 128 + d4;
  u32x2 gv = *(const u32x2*)(g_ptr + g_off);
  float g0 = bf16_lo(gv[0]), g1 = bf16_hi(gv[0]), g2 = bf16_lo(gv[1]), g3 = bf16_hi(gv[1]);

  float y0 = (x0 * rstd) * w0 * gate_act(g0, is_sig);
  float y1 = (x1 * rstd) * w1 * gate_act(g1, is_sig);
  float y2 = (x2 * rstd) * w2 * gate_act(g2, is_sig);
  float y3 = (x3 * rstd) * w3 * gate_act(g3, is_sig);

  *(u32x2*)(y_ptr + (int64_t)row * 128 + d4) = pack4_bf16(y0, y1, y2, y3);
}
