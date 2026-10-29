// E22 hiphc: manually authored HIP decode kernels (M <= 8) for the GPTQ8-HC-W1 packed hyper-connection (Qwen3.8 / qwen4_exp).
// Wire (gp8hc/wire.py): q u8 [N,K] row-major, scale fp16 / zero u8 per (row, 128-group). Per element the weight is
// w = bf16_rne((q - z) * s) (fp32 math), then dot in fp32 -- same per-element arithmetic as gp8hc v3/v5 Triton.
// sz = packed u32 per (row, group): low 16 bits = fp16 scale bits, bits 16..23 = zero.
//
//   down_silu : y = W_dn . xn  (W_dn [324 | 320, 10240]); y rounded to bf16 like the Linear output; rows < R -> lora = bf16(silu(y/HC))
//               rows R..N-1 -> d[:, n] = y (injection logits, merged-GEMM row stride kept); rows N..NP-1 -> 0.
//   up_gatemix: g = bf16(W_up . lora) (W_up [HC*H, 320]); out[h] = bf16( (sum_s sigmoid(g[s*H+h]) * xn[s*H+h]) / HC )
//   gemv      : plain y = bf16(W . x) for any (N, K % 16 == 0) -- final mixer down / drop-in check.
// Provenance: written for E22 (structure informed by radiance rad_block_hc.h, Apache-2.0; no code copied).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

#define DEV __device__ __forceinline__

DEV float bf2f(uint32_t b16) { return __uint_as_float(b16 << 16); }
DEV float bf16r(float f) {  // round-to-nearest-even to bf16, kept in fp32 (finite inputs)
  uint32_t u = __float_as_uint(f);
  u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
  return __uint_as_float(u);
}
DEV uint16_t f2bf(float f) { return (uint16_t)(__float_as_uint(bf16r(f)) >> 16); }
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
DEV uint4 ldnt(const uint4* p) {  // streaming (nontemporal) 16-byte weight load
#ifdef HIPHC_NO_NT
  return *p;
#else
  u32x4 v = __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(p));
  return make_uint4(v.x, v.y, v.z, v.w);
#endif
}
DEV float h2f(uint32_t h) { return __half2float(__ushort_as_half((unsigned short)h)); }

// wave32 sum via DPP (row_xmask 1,2,4,8) + permlanex16 -- avoids ds_bpermute (see custom-kernel-serving-integration pitfalls)
DEV float wave_sum(float v) {
  v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x161, 0xF, 0xF, false));
  v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x162, 0xF, 0xF, false));
  v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x164, 0xF, 0xF, false));
  v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x168, 0xF, 0xF, false));
  v += __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(v), __float_as_int(v), 0x76543210, 0xfedcba98, false, false));
  return v;
}

// dequantize 16 u8 codes (one uint4) with one (s, z) into fp32 bf16-rounded weights
DEV void deq16(const uint4 q, float s, float zf, float w[16]) {
  const uint32_t qq[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int b = 0; b < 4; ++b) w[i * 4 + b] = bf16r(((float)((qq[i] >> (8 * b)) & 0xFFu) - zf) * s);
  }
}
DEV void ldx16(const uint16_t* __restrict__ p, float x[16]) {  // 16 bf16 activations
  const uint4 a = *reinterpret_cast<const uint4*>(p);
  const uint4 b = *reinterpret_cast<const uint4*>(p + 8);
  const uint32_t v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) { x[2 * i] = bf2f(v[i] & 0xFFFFu); x[2 * i + 1] = bf2f(v[i] >> 16); }
}

// ------------------------------------------------------------------ split-K row GEMV core (long K, few rows)
// WG = KS waves; WG owns RPW consecutive output rows; wave w owns K slice w; lane strides 16-byte chunks.
// EPI 0 = plain bf16 store (O cols, zero pad N..O-1); EPI 1 = down_silu epilogue.
template <int M, int RPW, int KS, int EPI>
__global__ __launch_bounds__(KS * 32) void k_rowgemv(const uint16_t* __restrict__ x, const uint8_t* __restrict__ q,
                                                    const uint32_t* __restrict__ sz, uint16_t* __restrict__ y,
                                                    uint16_t* __restrict__ y2, int N, int K, int O, int R, int ldy,
                                                    int ldy2, float inv_hc) {
  __shared__ float red[KS][RPW][M];
  const int lane = threadIdx.x & 31, wv = threadIdx.x >> 5;
  const int r0 = blockIdx.x * RPW;
  const int NG = (K + 127) >> 7;
  const int chunks = K >> 4;                      // 16-byte chunks per row
  const int cps = (chunks + KS - 1) / KS;         // chunks per slice
  const int c0 = wv * cps, c1 = min(chunks, c0 + cps);
  float acc[RPW][M];
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
  for (int c = c0 + lane; c < c1; c += 32) {
    uint4 qv[RPW];
    uint32_t szv[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const int row = r0 + r;
      if (row < N) {
        qv[r] = ldnt(reinterpret_cast<const uint4*>(q + (size_t)row * K) + c);
        szv[r] = sz[(size_t)row * NG + (c >> 3)];
      } else { qv[r] = make_uint4(0, 0, 0, 0); szv[r] = 0; }
    }
    float xv[M][16];
#pragma unroll
    for (int m = 0; m < M; ++m) ldx16(x + (size_t)m * K + c * 16, xv[m]);
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      float w[16];
      deq16(qv[r], h2f(szv[r] & 0xFFFFu), (float)((szv[r] >> 16) & 0xFFu), w);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float a = acc[r][m];
#pragma unroll
        for (int i = 0; i < 16; ++i) a = fmaf(w[i], xv[m][i], a);
        acc[r][m] = a;
      }
    }
  }
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float s = wave_sum(acc[r][m]);
      if (lane == 0) red[wv][r][m] = s;
    }
  __syncthreads();
  if (threadIdx.x < RPW * M) {
    const int r = threadIdx.x / M, m = threadIdx.x % M, row = r0 + r;
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < KS; ++k) s += red[k][r][m];
    if (row < O) {
      const float yb = row < N ? bf16r(s) : 0.f;  // Linear output is bf16
      if (EPI == 0) {
        y[(size_t)m * ldy + row] = f2bf(yb);
      } else {
        if (row < R) {
          const float t = yb * inv_hc;  // hc_silu: x/HC then silu, fp32
          y[(size_t)m * ldy + row] = f2bf(t / (1.f + __expf(-t)));
        }
        y2[(size_t)m * ldy2 + row] = f2bf(yb);  // full merged row (inject logits at R..R+HC-1, pad zero)
      }
    }
  }
}

// ------------------------------------------------------------------ up + sigmoid gate-mix (short K = 320, HC*H rows)
// WG = HC waves (wave s = stream s), lane = one h; each lane computes the full 320-dot for row s*H + h for all M rows,
// lora (wave-uniform) via scalar loads; streams reduced through LDS.
template <int M, int HC, int KR>
__global__ __launch_bounds__(HC * 32) void k_up_gatemix(const uint16_t* __restrict__ lora, const uint8_t* __restrict__ q,
                                                       const uint32_t* __restrict__ sz, const uint16_t* __restrict__ xn,
                                                       uint16_t* __restrict__ out, int H, int ldl, int ldx, int ldo) {
  constexpr int NG = (KR + 127) / 128, CH = KR / 16;
  __shared__ float part[HC][M][32];
  const int lane = threadIdx.x & 31, s = threadIdx.x >> 5;
  const int h = blockIdx.x * 32 + lane;
  const int row = s * H + h;
  const uint4* qr = reinterpret_cast<const uint4*>(q + (size_t)row * KR);
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;
  uint32_t szv[NG];
#pragma unroll
  for (int g = 0; g < NG; ++g) szv[g] = sz[(size_t)row * NG + g];
  uint4 qv[CH];
#pragma unroll
  for (int c = 0; c < CH; ++c) qv[c] = ldnt(qr + c);
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    const uint32_t zz = szv[c >> 3];
    float w[16];
    deq16(qv[c], h2f(zz & 0xFFFFu), (float)((zz >> 16) & 0xFFu), w);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const uint16_t* lp = lora + (size_t)m * ldl + c * 16;  // wave-uniform
      float a = acc[m];
#pragma unroll
      for (int i = 0; i < 16; ++i) a = fmaf(w[i], bf2f(lp[i]), a);
      acc[m] = a;
    }
  }
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const float g = bf16r(acc[m]);                       // up Linear output is bf16
    const float xv = bf2f(xn[(size_t)m * ldx + row]);
    part[s][m][lane] = xv / (1.f + __expf(-g));          // sigmoid(g) * xn  (fp32)
  }
  __syncthreads();
  if (s == 0) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float a = 0.f;
#pragma unroll
      for (int k = 0; k < HC; ++k) a += part[k][m][lane];
      out[(size_t)m * ldo + h] = f2bf(a / (float)HC);
    }
  }
}

// v2 up variants ---------------------------------------------------------------------------------------------
// VA: same lane=row layout, lora staged once per WG in LDS as fp32 (no scalar bf16 conversion chain).
template <int M, int HC, int KR>
__global__ __launch_bounds__(HC * 32) void k_ugm_lds(const uint16_t* __restrict__ lora, const uint8_t* __restrict__ q,
                                                    const uint32_t* __restrict__ sz, const uint16_t* __restrict__ xn,
                                                    uint16_t* __restrict__ out, int H, int ldl, int ldx, int ldo) {
  constexpr int NG = (KR + 127) / 128, CH = KR / 16;
  __shared__ float lf[M][KR];
  __shared__ float part[HC][M][32];
  const int lane = threadIdx.x & 31, s = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < M * KR; i += HC * 32) lf[i / KR][i % KR] = bf2f(lora[(size_t)(i / KR) * ldl + (i % KR)]);
  const int h = blockIdx.x * 32 + lane;
  const int row = s * H + h;
  const uint4* qr = reinterpret_cast<const uint4*>(q + (size_t)row * KR);
  uint32_t szv[NG];
#pragma unroll
  for (int g = 0; g < NG; ++g) szv[g] = sz[(size_t)row * NG + g];
  uint4 qv[CH];
#pragma unroll
  for (int c = 0; c < CH; ++c) qv[c] = ldnt(qr + c);
  __syncthreads();
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    const uint32_t zz = szv[c >> 3];
    float w[16];
    deq16(qv[c], h2f(zz & 0xFFFFu), (float)((zz >> 16) & 0xFFu), w);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float4* lp = reinterpret_cast<const float4*>(&lf[m][c * 16]);
      float a = acc[m];
#pragma unroll
      for (int i4 = 0; i4 < 4; ++i4) {
        const float4 v = lp[i4];
        a = fmaf(w[4 * i4 + 0], v.x, a); a = fmaf(w[4 * i4 + 1], v.y, a);
        a = fmaf(w[4 * i4 + 2], v.z, a); a = fmaf(w[4 * i4 + 3], v.w, a);
      }
      acc[m] = a;
    }
  }
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const float g = bf16r(acc[m]);
    part[s][m][lane] = bf2f(xn[(size_t)m * ldx + row]) / (1.f + __expf(-g));
  }
  __syncthreads();
  if (s == 0) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float a = 0.f;
#pragma unroll
      for (int k = 0; k < HC; ++k) a += part[k][m][lane];
      out[(size_t)m * ldo + h] = f2bf(a / (float)HC);
    }
  }
}

// VB: coalesced. Wave s = stream s covers RW consecutive rows (RW*KR contiguous bytes), lane strides 16-byte chunks;
// per-chunk partials -> LDS -> per-row sums -> sigmoid gate * xn -> stream mean. WG covers RW h values.
template <int M, int HC, int KR, int RW>
__global__ __launch_bounds__(HC * 32) void k_ugm_co(const uint16_t* __restrict__ lora, const uint8_t* __restrict__ q,
                                                   const uint32_t* __restrict__ sz, const uint16_t* __restrict__ xn,
                                                   uint16_t* __restrict__ out, int H, int ldl, int ldx, int ldo) {
  constexpr int NG = (KR + 127) / 128, CH = KR / 16, TOT = RW * CH, PL = (TOT + 31) / 32;
  __shared__ float lf[M][KR];
  __shared__ float pc[HC][M][TOT];
  __shared__ float mix[HC][M][RW];
  const int lane = threadIdx.x & 31, s = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < M * KR; i += HC * 32) lf[i / KR][i % KR] = bf2f(lora[(size_t)(i / KR) * ldl + (i % KR)]);
  const int h0 = blockIdx.x * RW;
  const size_t row0 = (size_t)s * H + h0;
  const uint4* qb = reinterpret_cast<const uint4*>(q + row0 * KR);
  uint4 qv[PL];
  uint32_t szv[PL];
#pragma unroll
  for (int j = 0; j < PL; ++j) {
    const int c = lane + 32 * j;
    if (c < TOT) { qv[j] = ldnt(qb + c); szv[j] = sz[(row0 + c / CH) * NG + ((c % CH) >> 3)]; }
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < PL; ++j) {
    const int c = lane + 32 * j;
    if (c < TOT) {
      const int cc = c % CH;
      float w[16];
      deq16(qv[j], h2f(szv[j] & 0xFFFFu), (float)((szv[j] >> 16) & 0xFFu), w);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float4* lp = reinterpret_cast<const float4*>(&lf[m][cc * 16]);
        float a = 0.f;
#pragma unroll
        for (int i4 = 0; i4 < 4; ++i4) {
          const float4 v = lp[i4];
          a = fmaf(w[4 * i4 + 0], v.x, a); a = fmaf(w[4 * i4 + 1], v.y, a);
          a = fmaf(w[4 * i4 + 2], v.z, a); a = fmaf(w[4 * i4 + 3], v.w, a);
        }
        pc[s][m][c] = a;
      }
    }
  }
  __syncthreads();
  for (int t = lane; t < M * RW; t += 32) {
    const int m = t / RW, r = t % RW;
    float a = 0.f;
#pragma unroll
    for (int k = 0; k < CH; ++k) a += pc[s][m][r * CH + k];
    const float g = bf16r(a);
    mix[s][m][r] = bf2f(xn[(size_t)m * ldx + row0 + r]) / (1.f + __expf(-g));
  }
  __syncthreads();
  if (s == 0) {
    for (int t = lane; t < M * RW; t += 32) {
      const int m = t / RW, r = t % RW;
      float a = 0.f;
#pragma unroll
      for (int k = 0; k < HC; ++k) a += mix[k][m][r];
      out[(size_t)m * ldo + h0 + r] = f2bf(a / (float)HC);
    }
  }
}

// ------------------------------------------------------------------ host
template <int M, int RPW, int KS, int EPI>
static void launch_rowgemv(const at::Tensor& x, const at::Tensor& q, const at::Tensor& sz, at::Tensor& y, at::Tensor* y2,
                           int O, int R, float inv_hc) {
  const int N = q.size(0), K = q.size(1);
  dim3 grid((O + RPW - 1) / RPW), block(KS * 32);
  hipLaunchKernelGGL((k_rowgemv<M, RPW, KS, EPI>), grid, block, 0, at::cuda::getCurrentCUDAStream(),
                     (const uint16_t*)x.data_ptr(), (const uint8_t*)q.data_ptr(), (const uint32_t*)sz.data_ptr(),
                     (uint16_t*)y.data_ptr(), y2 ? (uint16_t*)y2->data_ptr() : nullptr, N, K, O, R, (int)y.stride(0),
                     y2 ? (int)y2->stride(0) : 0, inv_hc);
}

template <int RPW, int KS, int EPI>
static void dispatch_m(int M, const at::Tensor& x, const at::Tensor& q, const at::Tensor& sz, at::Tensor& y, at::Tensor* y2,
                       int O, int R, float inv_hc) {
  switch (M) {
    case 1: launch_rowgemv<1, RPW, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 2: launch_rowgemv<2, RPW, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 3: launch_rowgemv<3, RPW, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 4: launch_rowgemv<4, RPW, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 5: launch_rowgemv<5, RPW, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 6: launch_rowgemv<6, RPW, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 7: launch_rowgemv<7, RPW, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 8: launch_rowgemv<8, RPW, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    default: TORCH_CHECK(false, "hiphc: M must be 1..8");
  }
}

template <int EPI>
static void dispatch_cfg(int cfg, int M, const at::Tensor& x, const at::Tensor& q, const at::Tensor& sz, at::Tensor& y,
                         at::Tensor* y2, int O, int R, float inv_hc) {
  switch (cfg) {  // (RPW, KS)
    case 0: dispatch_m<1, 4, EPI>(M, x, q, sz, y, y2, O, R, inv_hc); break;
    case 1: dispatch_m<1, 8, EPI>(M, x, q, sz, y, y2, O, R, inv_hc); break;
    case 2: dispatch_m<2, 4, EPI>(M, x, q, sz, y, y2, O, R, inv_hc); break;
    case 3: dispatch_m<2, 8, EPI>(M, x, q, sz, y, y2, O, R, inv_hc); break;
    case 4: dispatch_m<4, 4, EPI>(M, x, q, sz, y, y2, O, R, inv_hc); break;
    case 5: dispatch_m<4, 2, EPI>(M, x, q, sz, y, y2, O, R, inv_hc); break;
    case 6: dispatch_m<1, 2, EPI>(M, x, q, sz, y, y2, O, R, inv_hc); break;
    case 7: dispatch_m<2, 2, EPI>(M, x, q, sz, y, y2, O, R, inv_hc); break;
    case 8: dispatch_m<4, 8, EPI>(M, x, q, sz, y, y2, O, R, inv_hc); break;
    case 9: dispatch_m<8, 4, EPI>(M, x, q, sz, y, y2, O, R, inv_hc); break;
    case 10: dispatch_m<8, 2, EPI>(M, x, q, sz, y, y2, O, R, inv_hc); break;
    case 11: dispatch_m<1, 16, EPI>(M, x, q, sz, y, y2, O, R, inv_hc); break;
    default: TORCH_CHECK(false, "hiphc: unknown rowgemv cfg");
  }
}

static void check_wire(const at::Tensor& x, const at::Tensor& q, const at::Tensor& sz) {
  TORCH_CHECK(x.is_cuda() && q.is_cuda() && sz.is_cuda(), "hiphc: GPU tensors required");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && q.scalar_type() == at::kByte && sz.scalar_type() == at::kInt,
              "hiphc: x bf16, q u8, sz int32");
  TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1 && q.is_contiguous() && sz.is_contiguous(), "hiphc: layout");
  TORCH_CHECK(x.size(1) == q.size(1) && q.size(1) % 16 == 0 && sz.size(0) == q.size(0) && sz.size(1) == (q.size(1) + 127) / 128,
              "hiphc: shape");
  TORCH_CHECK(x.stride(0) == x.size(1), "hiphc: x rows contiguous");
  TORCH_CHECK(((uintptr_t)x.data_ptr() & 15) == 0 && ((uintptr_t)q.data_ptr() & 15) == 0, "hiphc: 16B alignment");
}

// plain y[M,O] = bf16(x @ W^T), rows N..O-1 zero
at::Tensor gemv(at::Tensor x, at::Tensor q, at::Tensor sz, int64_t O, int64_t cfg) {
  check_wire(x, q, sz);
  TORCH_CHECK(O >= q.size(0), "hiphc: O < N");
  auto y = at::empty({x.size(0), O}, x.options());
  dispatch_cfg<0>(cfg, x.size(0), x, q, sz, y, nullptr, O, 0, 1.f);
  return y;
}

// returns (lora_act [M,R], d [M,O]) ; d holds bf16 merged outputs (inject logits at R..R+hc-1)
std::vector<at::Tensor> down_silu(at::Tensor xn, at::Tensor q, at::Tensor sz, int64_t O, int64_t R, int64_t hc, int64_t cfg) {
  check_wire(xn, q, sz);
  TORCH_CHECK(O >= q.size(0) && R <= q.size(0), "hiphc: O/R");
  auto lora = at::empty({xn.size(0), R}, xn.options());
  auto d = at::empty({xn.size(0), O}, xn.options());
  dispatch_cfg<1>(cfg, xn.size(0), xn, q, sz, lora, &d, O, R, 1.f / (float)hc);
  return {lora, d};
}

template <int M>
static void launch_ugm(int var, const at::Tensor& lora, const at::Tensor& q, const at::Tensor& sz, const at::Tensor& xn, at::Tensor& out, int H) {
  auto st = at::cuda::getCurrentCUDAStream();
  const uint16_t* l = (const uint16_t*)lora.data_ptr(); const uint8_t* qq = (const uint8_t*)q.data_ptr();
  const uint32_t* z = (const uint32_t*)sz.data_ptr(); const uint16_t* x = (const uint16_t*)xn.data_ptr(); uint16_t* o = (uint16_t*)out.data_ptr();
  const int a = (int)lora.stride(0), b = (int)xn.stride(0), c = (int)out.stride(0);
  switch (var) {
    case 0: hipLaunchKernelGGL((k_up_gatemix<M, 4, 320>), dim3(H / 32), dim3(128), 0, st, l, qq, z, x, o, H, a, b, c); break;
    case 1: hipLaunchKernelGGL((k_ugm_lds<M, 4, 320>), dim3(H / 32), dim3(128), 0, st, l, qq, z, x, o, H, a, b, c); break;
    case 2: hipLaunchKernelGGL((k_ugm_co<M, 4, 320, 8>), dim3(H / 8), dim3(128), 0, st, l, qq, z, x, o, H, a, b, c); break;
    case 3: hipLaunchKernelGGL((k_ugm_co<M, 4, 320, 16>), dim3(H / 16), dim3(128), 0, st, l, qq, z, x, o, H, a, b, c); break;
    case 4: hipLaunchKernelGGL((k_ugm_co<M, 4, 320, 4>), dim3(H / 4), dim3(128), 0, st, l, qq, z, x, o, H, a, b, c); break;
    default: TORCH_CHECK(false, "hiphc: unknown up variant");
  }
}

at::Tensor up_gatemix(at::Tensor lora, at::Tensor q, at::Tensor sz, at::Tensor xn, int64_t hc, int64_t var) {
  TORCH_CHECK(hc == 4 && q.size(1) == 320, "hiphc: up_gatemix compiled for hc=4, K=320");
  check_wire(lora, q, sz);
  TORCH_CHECK(xn.scalar_type() == at::kBFloat16 && xn.stride(1) == 1 && xn.size(1) == q.size(0) && xn.size(0) == lora.size(0),
              "hiphc: xn");
  const int H = q.size(0) / hc;
  TORCH_CHECK(H % 32 == 0, "hiphc: H % 32");
  auto out = at::empty({lora.size(0), H}, lora.options());
  switch (lora.size(0)) {
    case 1: launch_ugm<1>(var, lora, q, sz, xn, out, H); break;
    case 2: launch_ugm<2>(var, lora, q, sz, xn, out, H); break;
    case 3: launch_ugm<3>(var, lora, q, sz, xn, out, H); break;
    case 4: launch_ugm<4>(var, lora, q, sz, xn, out, H); break;
    case 5: launch_ugm<5>(var, lora, q, sz, xn, out, H); break;
    case 6: launch_ugm<6>(var, lora, q, sz, xn, out, H); break;
    case 7: launch_ugm<7>(var, lora, q, sz, xn, out, H); break;
    case 8: launch_ugm<8>(var, lora, q, sz, xn, out, H); break;
    default: TORCH_CHECK(false, "hiphc: M must be 1..8");
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv", &gemv);
  m.def("down_silu", &down_silu);
  m.def("up_gatemix", &up_gatemix);
}
