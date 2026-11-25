// htr_bf16_gemv.hip — Wave 7 RDNA 3.5 (gfx1151) upgraded small-N BF16 GEMV & fused shared_expert_gate!
// - Hidden V_DUAL_DOT2ACC_F32_BF16 (VOPD Opcode 13): 4 BF16 FMAs / 8 FLOPs per lane per cycle for paired rows
// - Native v_dot2_f32_bf16 for M=1 / odd tail rows (replaces 24 scalar ops with 4 dot2 ops)
// - 128-bit u32x4 vectorized in-place gate_scale + single-cycle v_perm_b32 packed BF16x2 RNE conversion
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float bf16_to_f(uint16_t h) {
  return __builtin_bit_cast(float, ((uint32_t)h) << 16);
}
__device__ __forceinline__ float bf16_lo(uint32_t u) {
  return __uint_as_float((u & 0xFFFFu) << 16);
}
__device__ __forceinline__ float bf16_hi(uint32_t u) {
  return __uint_as_float(u & 0xFFFF0000u);
}
__device__ __forceinline__ uint16_t f_to_bf16_rne(float f) {
  uint32_t u = __builtin_bit_cast(uint32_t, f);
  if (f != f) return 0x7fffu;
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ uint32_t f32x2_to_bf16x2_rne(float a, float b) {
  uint32_t ua = __float_as_uint(a);
  uint32_t ub = __float_as_uint(b);
  ua = (a != a) ? 0x7fff0000u : (ua + 0x7fffu + ((ua >> 16) & 1u));
  ub = (b != b) ? 0x7fff0000u : (ub + 0x7fffu + ((ub >> 16) & 1u));
  return __builtin_amdgcn_perm(ub, ua, 0x07060302u);
}

// Single-row native v_dot2_f32_bf16 (4 instructions for 8 BF16 FMAs)
__device__ __forceinline__ float dot2_bf16_u32(uint32_t a, uint32_t b, float acc) {
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2, a), __builtin_bit_cast(bf16x2, b), acc, false);
}
__device__ __forceinline__ float dot8_bf16_single(u32x4 a, u32x4 b, float acc) {
  acc = dot2_bf16_u32(a[0], b[0], acc);
  acc = dot2_bf16_u32(a[1], b[1], acc);
  acc = dot2_bf16_u32(a[2], b[2], acc);
  acc = dot2_bf16_u32(a[3], b[3], acc);
  return acc;
}

// Dual-row Wave 7 V_DUAL_DOT2ACC_F32_BF16 (VOPD Opcode 13): 16 BF16 FMAs in 4 dual-issue instructions!
// Bank-staggered pinned VGPRs mathematically guarantee all 4 RDNA 3.5 VOPD bank/parity rules:
//   vdst_x=v0(even), vdst_y=v1(odd), w=(v4..v7), xa=(v8..v11), xb=(v12..v15)
__device__ __forceinline__ void dual_dot8_bf16_pair(
    u32x4 xa, u32x4 xb, u32x4 wv, float& acc0, float& acc1) {
  register float a0 asm("v0") = acc0;
  register float a1 asm("v1") = acc1;
  register uint32_t w0 asm("v4") = wv.x;
  register uint32_t w1 asm("v5") = wv.y;
  register uint32_t w2 asm("v6") = wv.z;
  register uint32_t w3 asm("v7") = wv.w;
  register uint32_t x0 asm("v8")  = xa.x;
  register uint32_t x1 asm("v9")  = xa.y;
  register uint32_t x2 asm("v10") = xa.z;
  register uint32_t x3 asm("v11") = xa.w;
  register uint32_t y0 asm("v12") = xb.x;
  register uint32_t y1 asm("v13") = xb.y;
  register uint32_t y2 asm("v14") = xb.z;
  register uint32_t y3 asm("v15") = xb.w;
  asm volatile(
    "v_dual_dot2acc_f32_bf16 %0, %6, %2 :: v_dual_dot2acc_f32_bf16 %1, %11, %3\n\t"
    "v_dual_dot2acc_f32_bf16 %0, %7, %3 :: v_dual_dot2acc_f32_bf16 %1, %10, %2\n\t"
    "v_dual_dot2acc_f32_bf16 %0, %8, %4 :: v_dual_dot2acc_f32_bf16 %1, %13, %5\n\t"
    "v_dual_dot2acc_f32_bf16 %0, %9, %5 :: v_dual_dot2acc_f32_bf16 %1, %12, %4\n\t"
    : "+v"(a0), "+v"(a1)
    : "v"(w0), "v"(w1), "v"(w2), "v"(w3),
      "v"(x0), "v"(x1), "v"(x2), "v"(x3),
      "v"(y0), "v"(y1), "v"(y2), "v"(y3)
  );
  acc0 = a0;
  acc1 = a1;
}

__device__ __forceinline__ float w32_sum(float v) {
  v += __shfl_xor(v, 16, 32);
  v += __shfl_xor(v, 8, 32);
  v += __shfl_xor(v, 4, 32);
  v += __shfl_xor(v, 2, 32);
  v += __shfl_xor(v, 1, 32);
  return v;
}

template <int MB>
__device__ __forceinline__ void accum_mb(
    const u32x4* __restrict__ xbase, int k8, int j, int m0, int M, u32x4 wv, float acc[MB]) {
  if constexpr (MB == 1) {
    if (m0 < M) acc[0] = dot8_bf16_single(xbase[(int64_t)m0 * k8 + j], wv, acc[0]);
  } else {
#pragma unroll
    for (int p = 0; p < MB / 2; ++p) {
      int ma = m0 + 2 * p;
      int mb = ma + 1;
      if (mb < M) {
        u32x4 xa = xbase[(int64_t)ma * k8 + j];
        u32x4 xb = xbase[(int64_t)mb * k8 + j];
        dual_dot8_bf16_pair(xa, xb, wv, acc[2 * p], acc[2 * p + 1]);
      } else if (ma < M) {
        acc[2 * p] = dot8_bf16_single(xbase[(int64_t)ma * k8 + j], wv, acc[2 * p]);
      }
    }
  }
}

template <int MB>
__global__ void __launch_bounds__(64) k_bf16_gemv_2w(
    const uint16_t* __restrict__ X, const uint16_t* __restrict__ W,
    uint16_t* __restrict__ Y, int M, int N, int K) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int n = blockIdx.x * 2 + wave, m0 = blockIdx.y * MB;
  if (n >= N) return;
  const int k8 = K >> 3;
  const u32x4* __restrict__ wrow = reinterpret_cast<const u32x4*>(W + (int64_t)n * K);
  const u32x4* __restrict__ xbase = reinterpret_cast<const u32x4*>(X);
  float acc[MB];
#pragma unroll
  for (int i = 0; i < MB; ++i) acc[i] = 0.0f;
  for (int j = lane; j < k8; j += 32) {
    const u32x4 wv = __builtin_nontemporal_load(wrow + j);
    accum_mb<MB>(xbase, k8, j, m0, M, wv, acc);
  }
#pragma unroll
  for (int i = 0; i < MB; ++i) {
    float s = w32_sum(acc[i]);
    int m = m0 + i;
    if (lane == 0 && m < M) Y[(int64_t)m * N + n] = f_to_bf16_rne(s);
  }
}

template <int MB, bool APPLY_SIGMOID>
__global__ void __launch_bounds__(256) k_bf16_gemv_tinyN(
    const uint16_t* __restrict__ X, const uint16_t* __restrict__ W,
    uint16_t* __restrict__ Y, int M, int N, int K) {
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int n = blockIdx.x, m0 = blockIdx.y * MB;
  if (n >= N) return;
  const int k8 = K >> 3;
  const u32x4* __restrict__ wrow = reinterpret_cast<const u32x4*>(W + (int64_t)n * K);
  const u32x4* __restrict__ xbase = reinterpret_cast<const u32x4*>(X);
  float acc[MB];
#pragma unroll
  for (int i = 0; i < MB; ++i) acc[i] = 0.0f;
  for (int j = tid; j < k8; j += 256) {
    const u32x4 wv = __builtin_nontemporal_load(wrow + j);
    accum_mb<MB>(xbase, k8, j, m0, M, wv, acc);
  }
  __shared__ float s_red[8 * MB];
#pragma unroll
  for (int i = 0; i < MB; ++i) {
    float s = w32_sum(acc[i]);
    if (lane == 0) s_red[i * 8 + wave] = s;
  }
  __syncthreads();
  if (tid < MB) {
    int m = m0 + tid;
    if (m < M) {
      float tot = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) tot += s_red[tid * 8 + w];
      if constexpr (APPLY_SIGMOID) {
        float tb = bf16_to_f(f_to_bf16_rne(tot));
        tot = 1.0f / (1.0f + __expf(-tb));
      }
      Y[(int64_t)m * N + n] = f_to_bf16_rne(tot);
    }
  }
}

template <int MB>
__global__ void __launch_bounds__(256) k_bf16_gemv_gate_scale(
    const uint16_t* __restrict__ X,
    const uint16_t* __restrict__ W,
    uint16_t* __restrict__ out,
    int M, int K, int H) {
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int m0 = blockIdx.x * MB;
  const int k8 = K >> 3;
  const u32x4* __restrict__ wrow = reinterpret_cast<const u32x4*>(W);
  const u32x4* __restrict__ xbase = reinterpret_cast<const u32x4*>(X);
  float acc[MB];
#pragma unroll
  for (int i = 0; i < MB; ++i) acc[i] = 0.0f;
  for (int j = tid; j < k8; j += 256) {
    const u32x4 wv = __builtin_nontemporal_load(wrow + j);
    accum_mb<MB>(xbase, k8, j, m0, M, wv, acc);
  }
  __shared__ float s_red[8 * MB];
  __shared__ float s_g[MB];
#pragma unroll
  for (int i = 0; i < MB; ++i) {
    float s = w32_sum(acc[i]);
    if (lane == 0) s_red[i * 8 + wave] = s;
  }
  __syncthreads();
  if (tid < MB) {
    int m = m0 + tid;
    if (m < M) {
      float tot = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) tot += s_red[tid * 8 + w];
      float tb = bf16_to_f(f_to_bf16_rne(tot));
      float sig = 1.0f / (1.0f + __expf(-tb));
      s_g[tid] = bf16_to_f(f_to_bf16_rne(sig));
    }
  }
  __syncthreads();
  // 128-bit u32x4 vectorized in-place scaling + v_perm_b32 packed BF16x2 RNE!
  const bool vec128 = ((H & 7) == 0) && (((uintptr_t)out & 15u) == 0u);
  if (vec128) {
    const int h8 = H >> 3;
    u32x4* __restrict__ out8 = reinterpret_cast<u32x4*>(out);
#pragma unroll
    for (int i = 0; i < MB; ++i) {
      const int m = m0 + i;
      if (m < M) {
        const float g = s_g[i];
        u32x4* row = out8 + (int64_t)m * h8;
        for (int j = tid; j < h8; j += 256) {
          u32x4 v = row[j];
          v.x = f32x2_to_bf16x2_rne(bf16_lo(v.x) * g, bf16_hi(v.x) * g);
          v.y = f32x2_to_bf16x2_rne(bf16_lo(v.y) * g, bf16_hi(v.y) * g);
          v.z = f32x2_to_bf16x2_rne(bf16_lo(v.z) * g, bf16_hi(v.z) * g);
          v.w = f32x2_to_bf16x2_rne(bf16_lo(v.w) * g, bf16_hi(v.w) * g);
          row[j] = v;
        }
      }
    }
  } else {
    const int h2 = H >> 1;
    uint32_t* __restrict__ out2 = reinterpret_cast<uint32_t*>(out);
#pragma unroll
    for (int i = 0; i < MB; ++i) {
      const int m = m0 + i;
      if (m < M) {
        const float g = s_g[i];
        uint32_t* row = out2 + (int64_t)m * h2;
        for (int j = tid; j < h2; j += 256) {
          uint32_t pair = row[j];
          row[j] = f32x2_to_bf16x2_rne(bf16_lo(pair) * g, bf16_hi(pair) * g);
        }
      }
    }
  }
}

torch::Tensor bf16_gemv_smalln(torch::Tensor X, torch::Tensor W) {
  TORCH_CHECK(X.is_cuda() && W.is_cuda() && X.dtype() == torch::kBFloat16 && W.dtype() == torch::kBFloat16);
  TORCH_CHECK(X.is_contiguous() && W.is_contiguous() && X.dim() == 2 && W.dim() == 2);
  const int M = (int)X.size(0), K = (int)X.size(1), N = (int)W.size(0);
  TORCH_CHECK(W.size(1) == K && (K % 8 == 0));
  auto Y = torch::empty({M, N}, X.options());
  if (M == 0 || N == 0) return Y;
  hipStream_t stream = at::hip::getCurrentHIPStream().stream();
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(X.data_ptr());
  const uint16_t* wp = reinterpret_cast<const uint16_t*>(W.data_ptr());
  uint16_t* yp = reinterpret_cast<uint16_t*>(Y.data_ptr());
  if (N < 16) {
    if (M == 1) hipLaunchKernelGGL((k_bf16_gemv_tinyN<1, false>), dim3(N, 1), dim3(256), 0, stream, xp, wp, yp, M, N, K);
    else if (M <= 4) hipLaunchKernelGGL((k_bf16_gemv_tinyN<4, false>), dim3(N, 1), dim3(256), 0, stream, xp, wp, yp, M, N, K);
    else hipLaunchKernelGGL((k_bf16_gemv_tinyN<8, false>), dim3(N, (M + 7) >> 3), dim3(256), 0, stream, xp, wp, yp, M, N, K);
  } else {
    const int gx = (N + 1) >> 1;
    if (M == 1) hipLaunchKernelGGL((k_bf16_gemv_2w<1>), dim3(gx, 1), dim3(64), 0, stream, xp, wp, yp, M, N, K);
    else if (M <= 4) hipLaunchKernelGGL((k_bf16_gemv_2w<4>), dim3(gx, 1), dim3(64), 0, stream, xp, wp, yp, M, N, K);
    else hipLaunchKernelGGL((k_bf16_gemv_2w<8>), dim3(gx, (M + 7) >> 3), dim3(64), 0, stream, xp, wp, yp, M, N, K);
  }
  return Y;
}

torch::Tensor bf16_gemv_gate_sigmoid(torch::Tensor X, torch::Tensor W) {
  TORCH_CHECK(X.is_cuda() && W.is_cuda() && X.dtype() == torch::kBFloat16 && W.dtype() == torch::kBFloat16);
  TORCH_CHECK(X.is_contiguous() && W.is_contiguous() && X.dim() == 2 && W.dim() == 2 && W.size(0) == 1);
  const int M = (int)X.size(0), K = (int)X.size(1);
  TORCH_CHECK(W.size(1) == K && (K % 8 == 0));
  auto Y = torch::empty({M, 1}, X.options());
  if (M == 0) return Y;
  hipStream_t stream = at::hip::getCurrentHIPStream().stream();
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(X.data_ptr());
  const uint16_t* wp = reinterpret_cast<const uint16_t*>(W.data_ptr());
  uint16_t* yp = reinterpret_cast<uint16_t*>(Y.data_ptr());
  if (M == 1) hipLaunchKernelGGL((k_bf16_gemv_tinyN<1, true>), dim3(1, 1), dim3(256), 0, stream, xp, wp, yp, M, 1, K);
  else if (M <= 4) hipLaunchKernelGGL((k_bf16_gemv_tinyN<4, true>), dim3(1, 1), dim3(256), 0, stream, xp, wp, yp, M, 1, K);
  else hipLaunchKernelGGL((k_bf16_gemv_tinyN<8, true>), dim3(1, (M + 7) >> 3), dim3(256), 0, stream, xp, wp, yp, M, 1, K);
  return Y;
}

void bf16_gemv_gate_scale_inplace(torch::Tensor X, torch::Tensor W, torch::Tensor out) {
  TORCH_CHECK(X.is_cuda() && W.is_cuda() && out.is_cuda());
  TORCH_CHECK(X.dtype() == torch::kBFloat16 && W.dtype() == torch::kBFloat16 && out.dtype() == torch::kBFloat16);
  TORCH_CHECK(X.is_contiguous() && W.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(X.dim() == 2 && W.dim() == 2 && out.dim() == 2 && W.size(0) == 1);
  const int M = (int)X.size(0), K = (int)X.size(1), H = (int)out.size(1);
  TORCH_CHECK(W.size(1) == K && out.size(0) == M && (K % 8 == 0) && (H % 2 == 0));
  if (M == 0) return;
  hipStream_t stream = at::hip::getCurrentHIPStream().stream();
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(X.data_ptr());
  const uint16_t* wp = reinterpret_cast<const uint16_t*>(W.data_ptr());
  uint16_t* op = reinterpret_cast<uint16_t*>(out.data_ptr());
  if (M == 1) hipLaunchKernelGGL((k_bf16_gemv_gate_scale<1>), dim3(1), dim3(256), 0, stream, xp, wp, op, M, K, H);
  else if (M <= 4) hipLaunchKernelGGL((k_bf16_gemv_gate_scale<4>), dim3(1), dim3(256), 0, stream, xp, wp, op, M, K, H);
  else hipLaunchKernelGGL((k_bf16_gemv_gate_scale<8>), dim3((M + 7) >> 3), dim3(256), 0, stream, xp, wp, op, M, K, H);
}

PYBIND11_MODULE(htr_bf16_gemv, m) {
  m.def("bf16_gemv_smalln", &bf16_gemv_smalln, "Wave 7 RDNA 3.5 V_DUAL_DOT2ACC_F32_BF16 small-N BF16 GEMV");
  m.def("bf16_gemv_gate_sigmoid", &bf16_gemv_gate_sigmoid, "Wave 7 RDNA 3.5 fused gate + sigmoid");
  m.def("bf16_gemv_gate_scale_inplace", &bf16_gemv_gate_scale_inplace, "Wave 7 RDNA 3.5 fused gate + sigmoid + 128-bit in-place scale");
}
