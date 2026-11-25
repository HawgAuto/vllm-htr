// gdn_post_conv_aot.hip — AOT gfx1151 Wave32 HIP replacement for Triton _fused_post_conv_kernel.
// Specialized for K=128, V=128, BF16 mixed_qkv/q/k/v/a/b, FP32/BF16 A_log/dt_bias, FP32 g/beta.
// 1 Wave32 = 1 head (4 contiguous BF16s/lane = 64-bit u32x2 coalesced loads/stores), 0 LDS, 16 w/SIMD!
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
__device__ __forceinline__ float load_scalar_f32(const void* p, int64_t idx, int is_bf16) {
  if (is_bf16) return __uint_as_float(((uint32_t)((const uint16_t*)p)[idx]) << 16);
  return ((const float*)p)[idx];
}

// Grid: (L, 1, 1), Block: (256, 1, 1) -> 8 waves per token
// flags: bit0=apply_l2norm, bit1=output_g_exp, bit2=write_v, bit3=a_is_bf16, bit4=b_is_bf16, bit5=alog_is_bf16, bit6=dt_is_bf16
extern "C" __global__ void __launch_bounds__(256) gdn_post_conv_bf16_d128(
    const uint16_t* __restrict__ mixed_qkv_ptr,
    const void* __restrict__ a_ptr,
    const void* __restrict__ b_ptr,
    const void* __restrict__ A_log_ptr,
    const void* __restrict__ dt_bias_ptr,
    uint16_t* __restrict__ q_ptr,
    uint16_t* __restrict__ k_ptr,
    uint16_t* __restrict__ v_ptr,
    float* __restrict__ g_ptr,
    float* __restrict__ beta_ptr,
    int64_t stride_x_tok, int64_t stride_a_tok, int64_t stride_b_tok,
    int64_t stride_q_tok, int64_t stride_k_tok, int64_t stride_v_tok,
    int L, int H, int HV, float l2norm_eps, int flags) {

  const int t = blockIdx.x;
  if (t >= L) return;

  const int tid = threadIdx.x;
  const int wave = tid >> 5;   // 0..7
  const int lane = tid & 31;   // 0..31 -> 4 dims [4*lane .. 4*lane+3]
  const int d4 = lane * 4;
  const int HK = H * 128;

  const bool apply_l2 = (flags & 1) != 0;
  const bool out_g_exp = (flags & 2) != 0;
  const bool write_v = (flags & 4) != 0;
  const int a_bf16 = (flags >> 3) & 1;
  const int b_bf16 = (flags >> 4) & 1;
  const int alog_bf16 = (flags >> 5) & 1;
  const int dt_bf16 = (flags >> 6) & 1;

  const uint16_t* x_row = mixed_qkv_ptr + (int64_t)t * stride_x_tok;
  uint16_t* q_row = q_ptr + (int64_t)t * stride_q_tok;
  uint16_t* k_row = k_ptr + (int64_t)t * stride_k_tok;

  // 1. Q & K heads (8 waves stride across H heads)
  for (int h = wave; h < H; h += 8) {
    u32x2 qv = *(const u32x2*)(x_row + h * 128 + d4);
    u32x2 kv = *(const u32x2*)(x_row + HK + h * 128 + d4);
    if (apply_l2) {
      float q0 = bf16_lo(qv[0]), q1 = bf16_hi(qv[0]), q2 = bf16_lo(qv[1]), q3 = bf16_hi(qv[1]);
      float qsq = warp_sum32(dot2_bf16(qv[1], qv[1], dot2_bf16(qv[0], qv[0], 0.0f)));
      float qinv = rsqrtf(qsq + l2norm_eps);
      qv = pack4_bf16(q0 * qinv, q1 * qinv, q2 * qinv, q3 * qinv);

      float k0 = bf16_lo(kv[0]), k1 = bf16_hi(kv[0]), k2 = bf16_lo(kv[1]), k3 = bf16_hi(kv[1]);
      float ksq = warp_sum32(dot2_bf16(kv[1], kv[1], dot2_bf16(kv[0], kv[0], 0.0f)));
      float kinv = rsqrtf(ksq + l2norm_eps);
      kv = pack4_bf16(k0 * kinv, k1 * kinv, k2 * kinv, k3 * kinv);
    }
    *(u32x2*)(q_row + h * 128 + d4) = qv;
    *(u32x2*)(k_row + h * 128 + d4) = kv;
  }

  // 2. V copy (only when write_v != 0; skipped when DIRECT_GDN_VVIEW is active!)
  if (write_v) {
    const uint16_t* v_src = x_row + 2 * HK;
    uint16_t* v_dst = v_ptr + (int64_t)t * stride_v_tok;
    for (int hv = wave; hv < HV; hv += 8) {
      *(u32x2*)(v_dst + hv * 128 + d4) = *(const u32x2*)(v_src + hv * 128 + d4);
    }
  }

  // 3. Gating scalars g[t, hv] and beta[t, hv]
  for (int hv = tid; hv < HV; hv += 256) {
    float alog = load_scalar_f32(A_log_ptr, hv, alog_bf16);
    float dtb = load_scalar_f32(dt_bias_ptr, hv, dt_bf16);
    float av = load_scalar_f32(a_ptr, (int64_t)t * stride_a_tok + hv, a_bf16);
    float bv = load_scalar_f32(b_ptr, (int64_t)t * stride_b_tok + hv, b_bf16);
    float x = av + dtb;
    float sp = (x > 20.0f) ? x : ((x > 0.0f) ? (x + logf(1.0f + expf(-x))) : logf(1.0f + expf(x)));
    float gv = -expf(alog) * sp;
    if (out_g_exp) gv = expf(gv);
    float betav = 1.0f / (1.0f + expf(-bv));
    int64_t out_idx = (int64_t)t * HV + hv;
    g_ptr[out_idx] = gv;
    beta_ptr[out_idx] = betav;
  }
}
