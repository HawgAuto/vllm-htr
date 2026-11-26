// Native gfx1151 (RDNA3.5) In-Place K=5 Split-Plane (16B+4B) Repack + On-The-Fly WMMA Prefill GEMM.
// Combines:
//   - Zero extra steady-state VRAM: packs [p16 (32 u32/tile) | p4 (8 u32/tile)] into exact same [mt, nt, 40] int32 buffer!
//   - Decode M<=16: uses k5_splitplane.so's bit-exact aligned global_load_b128 + permlane16 + v_sad_hi_u8 GEMV
//   - Prefill / Verify M>=17: uses this kernel's aligned global_load_b128 + permlane16 + v_sad_hi_u8 + wmma_f16
//     + zero-sync wave_barrier + in-WG H128 FWHT epilogue (ZERO BF16 weight materialization!)
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

#define GEN_A 0xDF6ADFu
#define GEN_B 0x34983u

typedef _Float16 h2v __attribute__((ext_vector_type(2)));
typedef _Float16 h4v __attribute__((ext_vector_type(4)));
typedef _Float16 h16v __attribute__((ext_vector_type(16)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef float f8v __attribute__((ext_vector_type(8)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));
typedef u4v u4va __attribute__((aligned(4)));
struct u2_128 { uint4 a, b; };

__device__ __forceinline__ float bf16_dot2(uint32_t a, uint32_t b, float c) {
  return __builtin_amdgcn_fdot2_f32_bf16(
      __builtin_bit_cast(bf16x2, a), __builtin_bit_cast(bf16x2, b), c, false);
}
__device__ __forceinline__ float fast_tanhf(float x) {
  const float ax = fabsf(x);
  const float e = exp2f((-2.0f * 1.4426950408889634f) * ax);
  return copysignf((1.0f - e) * __builtin_amdgcn_rcpf(1.0f + e), x);
}

__device__ __forceinline__ uint32_t alignbit(uint32_t hi, uint32_t lo, uint32_t sh) {
  return __builtin_amdgcn_alignbit(hi, lo, sh);
}
template <int B>
__device__ __forceinline__ uint32_t ext16(const uint32_t* V) {
  constexpr int w = B >> 5, off = B & 31;
  if constexpr (off <= 16) return (V[w] >> (16 - off)) & 0xFFFFu;
  else return alignbit(V[w], V[w + 1], 48 - off) & 0xFFFFu;
}
__device__ __forceinline__ uint32_t val_f16bits2(uint32_t a, uint32_t b) {
  return __builtin_amdgcn_sad_hi_u8(
      b * GEN_A + GEN_B, 0u,
      __builtin_amdgcn_sad_u8(a * GEN_A + GEN_B, 0u, 0x64006400u));
}
__device__ __forceinline__ h2v as_h2(uint32_t u) { return __builtin_bit_cast(h2v, u); }

template <int MSK>
__device__ __forceinline__ float shx(float v) {
  const int iv = __builtin_bit_cast(int, v);
  if constexpr (MSK < 16) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, iv, 0x160 | MSK, 0xF, 0xF, false));
  } else {
    return __builtin_bit_cast(float, (int)__builtin_amdgcn_permlanex16(
        (unsigned)iv, (unsigned)iv, 0x76543210u, 0xfedcba98u, false, false));
  }
}
template <int MSK>
__device__ __forceinline__ void fwht_stage(float v[4], int lane) {
  const bool up = (lane & MSK) != 0;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float o = shx<MSK>(v[e]);
    v[e] = up ? (o - v[e]) : (v[e] + o);
  }
}
__device__ __forceinline__ void fwht128_wave(float v[4], int lane) {
  float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
  v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
  fwht_stage<1>(v, lane); fwht_stage<2>(v, lane); fwht_stage<4>(v, lane);
  fwht_stage<8>(v, lane); fwht_stage<16>(v, lane);
}
__device__ __forceinline__ uint16_t f32_to_bf16_rne(float x) {
  uint32_t u = __builtin_bit_cast(uint32_t, x);
  if (x != x) return 0x7fffu;
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ uint32_t f32_to_bf16x2_rne(float a, float b) {
  uint32_t ua = __builtin_bit_cast(uint32_t, a);
  uint32_t ub = __builtin_bit_cast(uint32_t, b);
  ua = ((ua & 0x7fffffffu) > 0x7f800000u) ? (ua | 0x00400000u) : (ua + 0x7FFFu + ((ua >> 16) & 1u));
  ub = ((ub & 0x7fffffffu) > 0x7f800000u) ? (ub | 0x00400000u) : (ub + 0x7FFFu + ((ub >> 16) & 1u));
  return __builtin_amdgcn_perm(ub, ua, 0x07060302u);
}

// Repack [mt, nt, 40] -> packed [p16 (ntiles*32) | p4 (ntiles*8)] in a same-size buffer
__global__ void k_repack_k5_packed(
    const uint32_t* __restrict__ src_qw,
    uint32_t* __restrict__ dst_packed,
    int64_t ntiles) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ntiles * 8) return;
  const int64_t t = idx >> 3;
  const int q = (int)(idx & 7);
  const uint32_t* src = src_qw + t * 40 + q * 5;
  uint32_t* d16 = dst_packed + t * 32 + q * 4;
  uint32_t* d4 = dst_packed + ntiles * 32 + t * 8 + q;
  const u4v v = *reinterpret_cast<const u4va*>(src);
  *reinterpret_cast<u4va*>(d16) = v;
  *d4 = src[4];
}

template <int NB, int SP16 = 1>
__global__ __launch_bounds__(256)
void k_dense_prefill_wmma_k5sp(
    const uint32_t* __restrict__ p16,      // SP16=1: [mt, nt, 32], SP16=0: unsplit qw [mt, nt, 40]
    const uint32_t* __restrict__ p4,       // SP16=1: [mt, nt, 8],  SP16=0: unused (nullptr)
    const _Float16* __restrict__ xt,       // [M, nt*16] fp16 (from E.prologue)
    const float* __restrict__ tok,         // [M] fp32
    const __half* __restrict__ scales,     // [mt*16] fp16
    float cb,
    const uint8_t* __restrict__ sv,        // [mt*2] u8
    const uint16_t* __restrict__ bias,     // [mt*16] bf16 (nullable)
    uint16_t* __restrict__ y,              // [M, mt*16] bf16
    int M, int mt, int nt) {
  constexpr int K = 5;
  union SharedMem {
    struct {
      uint32_t w_lds[8][4][128];             // 16,384 B
      uint32_t x_lds[4][NB * 16][8];         //  2..16 KiB (cooperative xt staging -> 8x fewer global xt loads + zero vmcnt conflict!)
    } stage;
    float    o_lds[16][128];                 //  8,192 B (reused in epilogue!)
  };
  __shared__ __attribute__((aligned(16))) SharedMem smem;

  const int tid = threadIdx.x;
  const int w = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 3;
  const int q = lane & 7;
  const int l16 = lane & 15;
  const int hb = lane >> 4;

  const int b = blockIdx.x;                // 128-row output block
  const int m0 = blockIdx.y * (NB * 16);   // BM = NB*16 token M block
  const int i = b * 8 + w;                 // 16-row strip owned by wave w
  const int n = nt * 16;
  const int m_out = mt * 16;

  f8v acc[NB];
#pragma unroll
  for (int c = 0; c < NB; ++c) acc[c] = f8v{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  // 2-stage register software prefetch of K=5 weights (5 VGPRs, 0 extra LDS):
  // Overlaps 100% of LPDDR5X weight fetch latency with all 4*NB LDS-fed WMMAs!
  u4v w4_cur = {0u, 0u, 0u, 0u};
  uint32_t wl_cur = 0u;
  if (g < nt) {
    const size_t tid0 = (size_t)i * nt + g;
    if constexpr (SP16) {
      w4_cur = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(p16 + tid0 * 32 + q * 4));
      wl_cur = __builtin_nontemporal_load(p4 + tid0 * 8 + q);
    } else {
      const uint32_t* tp0 = p16 + tid0 * 40 + q * 5;
      w4_cur = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tp0));
      wl_cur = __builtin_nontemporal_load(tp0 + 4);
    }
  }

  for (int j = 0; j < nt; j += 4) {
    // 1) Cooperatively stage xt[m0..m0+NB*16-1, j*16..(j+4)*16-1] into x_lds (burst global_load_b128 -> ds_store_b128!)
    constexpr int X_U4_TOTAL = NB * 128;
    constexpr int X_STEPS = (X_U4_TOTAL + 255) / 256;
    uint4 x_tmp[X_STEPS];
#pragma unroll
    for (int s = 0; s < X_STEPS; ++s) {
      const int idx = s * 256 + tid;
      x_tmp[s] = make_uint4(0u, 0u, 0u, 0u);
      if (idx < X_U4_TOTAL) {
        const int half_k = idx & 1;
        const int t_ld = (idx >> 1) & 3;
        const int row_l = idx >> 3;
        const int rr = m0 + row_l;
        if (rr < M && j + t_ld < nt) {
          x_tmp[s] = *reinterpret_cast<const uint4*>(xt + (size_t)rr * n + (j + t_ld) * 16 + half_k * 8);
        }
      }
    }
#pragma unroll
    for (int s = 0; s < X_STEPS; ++s) {
      const int idx = s * 256 + tid;
      if (idx < X_U4_TOTAL) {
        const int half_k = idx & 1;
        const int t_ld = (idx >> 1) & 3;
        const int row_l = idx >> 3;
        *reinterpret_cast<uint4*>(&smem.stage.x_lds[t_ld][row_l][half_k * 4]) = x_tmp[s];
      }
    }

    // 2) Decode prefetched K=5 weights into w_lds
    const int jj = j + g;
    uint32_t* L = smem.stage.w_lds[w][g];
    if (jj < nt) {
      uint32_t V[6];
      V[0] = __builtin_amdgcn_permlane16(wl_cur, wl_cur, 0x65432107u, 0xedcba98fu, false, false);
      V[1] = w4_cur.x; V[2] = w4_cur.y; V[3] = w4_cur.z; V[4] = w4_cur.w; V[5] = wl_cur;
#define PW_W(u) { \
      uint32_t bits = val_f16bits2(ext16<16 + (2 * (u) + 1) * K>(V), ext16<16 + (2 * (u) + 2) * K>(V)); \
      h2v centered = as_h2(bits) - h2v{(_Float16)1534.0f, (_Float16)1534.0f}; \
      L[((u) < 8 ? 2 * q : 2 * q + 1) * 8 + ((u) & 7)] = __builtin_bit_cast(uint32_t, centered); }
      PW_W(0) PW_W(1) PW_W(2) PW_W(3) PW_W(4) PW_W(5) PW_W(6) PW_W(7)
      PW_W(8) PW_W(9) PW_W(10) PW_W(11) PW_W(12) PW_W(13) PW_W(14) PW_W(15)
#undef PW_W
    }
    __syncthreads();

    // 3) Prefetch next K=5 tile (j + 4) BEFORE all 4*NB LDS-only WMMAs (zero vmcnt conflict!)
    u4v w4_nxt = {0u, 0u, 0u, 0u};
    uint32_t wl_nxt = 0u;
    if (j + 4 + g < nt) {
      const size_t tidn = (size_t)i * nt + (j + 4 + g);
      if constexpr (SP16) {
        w4_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(p16 + tidn * 32 + q * 4));
        wl_nxt = __builtin_nontemporal_load(p4 + tidn * 8 + q);
      } else {
        const uint32_t* tpn = p16 + tidn * 40 + q * 5;
        w4_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tpn));
        wl_nxt = __builtin_nontemporal_load(tpn + 4);
      }
    }
    __builtin_amdgcn_sched_barrier(0);

    // 4) Pure LDS-fed WMMA compute (0 global loads, 0 bounds checks!) while w4_nxt/wl_nxt stream in!
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      if (j + t < nt) {
        const uint4* ap = reinterpret_cast<const uint4*>(smem.stage.w_lds[w][t] + l16 * 8);
        u2_128 au{ap[0], ap[1]};
        const h16v a = __builtin_bit_cast(h16v, au);
#pragma unroll
        for (int c = 0; c < NB; ++c) {
          const uint4* bp = reinterpret_cast<const uint4*>(smem.stage.x_lds[t][c * 16 + l16]);
          u2_128 bu{bp[0], bp[1]};
          acc[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, __builtin_bit_cast(h16v, bu), acc[c]);
        }
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();
    w4_cur = w4_nxt;
    wl_cur = wl_nxt;
  }

  // In-WG fused H128 FWHT + scales + sv + bias + coalesced 8B uint2 BF16 store epilogue
  const int base_col = b * 128 + lane * 4;
  float sc[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) sc[e] = __half2float(scales[base_col + e]) * cb;
  const unsigned sg4 = (unsigned)sv[b * 16 + (lane >> 1)] >> ((lane & 1) * 4);
  float bv[4] = {0.f, 0.f, 0.f, 0.f};
  if (bias != nullptr) {
    const uint2 braw = *reinterpret_cast<const uint2*>(bias + base_col);
    bv[0] = __uint_as_float((braw.x & 0xFFFFu) << 16);
    bv[1] = __uint_as_float(braw.x & 0xFFFF0000u);
    bv[2] = __uint_as_float((braw.y & 0xFFFFu) << 16);
    bv[3] = __uint_as_float(braw.y & 0xFFFF0000u);
  }

#pragma unroll
  for (int c = 0; c < NB; ++c) {
    if (m0 + c * 16 >= M) break;
    __syncthreads();
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      smem.o_lds[l16][w * 16 + 2 * v + hb] = acc[c][v];
    }
    __syncthreads();

#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      const int rt = w + sub * 8;
      const int r = m0 + c * 16 + rt;
      if (r < M) {
        const float tt = tok[r];
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) v[e] = smem.o_lds[rt][lane * 4 + e] * sc[e] * tt;
        fwht128_wave(v, lane);
        float ob[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float o = v[e] * (1.0f / 11.313708498984761f);
          if ((sg4 >> e) & 1u) o = -o;
          if (bias != nullptr) o += bv[e];
          ob[e] = o;
        }
        uint2 out_p;
        out_p.x = f32_to_bf16x2_rne(ob[0], ob[1]);
        out_p.y = f32_to_bf16x2_rne(ob[2], ob[3]);
        *reinterpret_cast<uint2*>(y + (size_t)r * m_out + base_col) = out_p;
      }
    }
  }
}

torch::Tensor repack_k5_packed(torch::Tensor qw) {
  TORCH_CHECK(qw.is_cuda() && qw.is_contiguous() && qw.dim() == 3 && qw.size(2) == 40 && qw.scalar_type() == torch::kInt32);
  const int64_t ntiles = qw.size(0) * qw.size(1);
  auto out = torch::empty_like(qw);
  auto st = c10::hip::getCurrentHIPStream().stream();
  const int64_t nthreads = ntiles * 8;
  hipLaunchKernelGGL(k_repack_k5_packed, dim3((unsigned)((nthreads + 255) / 256)), dim3(256), 0, st,
                     reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>()),
                     reinterpret_cast<uint32_t*>(out.data_ptr<int32_t>()), ntiles);
  return out;
}

torch::Tensor dense_prefill_wmma_sp16(
    torch::Tensor p16, torch::Tensor p4, torch::Tensor xt, torch::Tensor tok,
    torch::Tensor scales, double cb, torch::Tensor sv,
    c10::optional<torch::Tensor> bias, int64_t bm) {
  TORCH_CHECK(p16.is_cuda() && p16.is_contiguous() && p16.dim() == 3 && p16.size(2) == 32);
  TORCH_CHECK(p4.is_cuda() && p4.is_contiguous() && p4.dim() == 3 && p4.size(2) == 8);
  TORCH_CHECK(xt.is_cuda() && xt.is_contiguous() && xt.dim() == 2 && xt.scalar_type() == torch::kHalf);
  const int mt = (int)p16.size(0), nt = (int)p16.size(1);
  const int M = (int)xt.size(0);
  TORCH_CHECK((mt % 8 == 0) && xt.size(1) == nt * 16);
  auto y = torch::empty({M, mt * 16}, xt.options().dtype(torch::kBFloat16));
  if (M == 0) return y;
  auto st = c10::hip::getCurrentHIPStream().stream();
  auto p16p = reinterpret_cast<const uint32_t*>(p16.data_ptr<int32_t>());
  auto p4p = reinterpret_cast<const uint32_t*>(p4.data_ptr<int32_t>());
  auto xp = reinterpret_cast<const _Float16*>(xt.data_ptr<at::Half>());
  auto tp = tok.data_ptr<float>();
  auto sp = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  auto svp = sv.data_ptr<uint8_t>();
  const uint16_t* bp = (bias.has_value() && bias->defined())
      ? reinterpret_cast<const uint16_t*>(bias->data_ptr<at::BFloat16>()) : nullptr;
  auto yp = reinterpret_cast<uint16_t*>(y.data_ptr<at::BFloat16>());
  if (bm <= 0) bm = (M >= 512) ? 128 : (M <= 16 ? 16 : (M <= 32 ? 32 : 64));
  if (bm == 128) {
    hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp<8, 1>), dim3(mt / 8, (M + 127) / 128, 1), dim3(256), 0, st,
                       p16p, p4p, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else if (bm == 16) {
    hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp<1, 1>), dim3(mt / 8, (M + 15) / 16, 1), dim3(256), 0, st,
                       p16p, p4p, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else if (bm == 32) {
    hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp<2, 1>), dim3(mt / 8, (M + 31) / 32, 1), dim3(256), 0, st,
                       p16p, p4p, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else {
    hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp<4, 1>), dim3(mt / 8, (M + 63) / 64, 1), dim3(256), 0, st,
                       p16p, p4p, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  }
  return y;
}

torch::Tensor dense_prefill_wmma_unsplit(
    torch::Tensor qw, torch::Tensor xt, torch::Tensor tok,
    torch::Tensor scales, double cb, torch::Tensor sv,
    c10::optional<torch::Tensor> bias, int64_t bm) {
  TORCH_CHECK(qw.is_cuda() && qw.is_contiguous() && qw.dim() == 3 && qw.size(2) == 40);
  TORCH_CHECK(xt.is_cuda() && xt.is_contiguous() && xt.dim() == 2 && xt.scalar_type() == torch::kHalf);
  const int mt = (int)qw.size(0), nt = (int)qw.size(1);
  const int M = (int)xt.size(0);
  TORCH_CHECK((mt % 8 == 0) && xt.size(1) == nt * 16);
  auto y = torch::empty({M, mt * 16}, xt.options().dtype(torch::kBFloat16));
  if (M == 0) return y;
  auto st = c10::hip::getCurrentHIPStream().stream();
  auto qwp = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>());
  auto xp = reinterpret_cast<const _Float16*>(xt.data_ptr<at::Half>());
  auto tp = tok.data_ptr<float>();
  auto sp = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  auto svp = sv.data_ptr<uint8_t>();
  const uint16_t* bp = (bias.has_value() && bias->defined())
      ? reinterpret_cast<const uint16_t*>(bias->data_ptr<at::BFloat16>()) : nullptr;
  auto yp = reinterpret_cast<uint16_t*>(y.data_ptr<at::BFloat16>());
  if (bm <= 0) bm = (M >= 512) ? 128 : (M <= 16 ? 16 : (M <= 32 ? 32 : 64));
  if (bm == 128) {
    hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp<8, 0>), dim3(mt / 8, (M + 127) / 128, 1), dim3(256), 0, st,
                       qwp, nullptr, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else if (bm == 16) {
    hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp<1, 0>), dim3(mt / 8, (M + 15) / 16, 1), dim3(256), 0, st,
                       qwp, nullptr, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else if (bm == 32) {
    hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp<2, 0>), dim3(mt / 8, (M + 31) / 32, 1), dim3(256), 0, st,
                       qwp, nullptr, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else {
    hipLaunchKernelGGL((k_dense_prefill_wmma_k5sp<4, 0>), dim3(mt / 8, (M + 63) / 64, 1), dim3(256), 0, st,
                       qwp, nullptr, xp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  }
  return y;
}

// ============================================================================
// Zero-Partial, Zero-Atomic, Zero-Fence Single-Pass K=5 Split-Plane GEMV (ns=1)
// Designed for tall ERNIE-5.5 K=5 matrices where mt/8 already saturates all 40 CUs:
//   - wq_b   [ 32768 x 1024] (37x/tok = 777 MB): 256 WGs (6.4 WGs/CU), saves 39..233 MB partial traffic!
//   - lm_head[201216 x 4096] (515 MB/tok + draft): 1572 WGs (39.3 WGs/CU), saves 25.8 MB partial traffic + 50K atomics!
// Exact M=1,2,4,7,8 templates (M=7 matches ERNIE-5.5 DSpark 7-token draft logits with zero wasted row!)
// ============================================================================
template <int M_CAP>
__global__ __launch_bounds__(256)
void k_gemv_k5_sp16_ns1(
    const uint32_t* __restrict__ p16,      // [mt, nt, 32]
    const uint32_t* __restrict__ p4,       // [mt, nt, 8]
    const _Float16* __restrict__ xt,       // [M, nt*16] fp16 (L1/L2 cached across WGs!)
    const float* __restrict__ xs,          // [M] fp32
    const float* __restrict__ tok,         // [M] fp32
    const __half* __restrict__ scales,     // [mt*16] fp16
    float cb,
    const uint8_t* __restrict__ sv,        // [mt*2] u8
    const uint16_t* __restrict__ bias,     // [mt*16] bf16 (nullable)
    uint16_t* __restrict__ y,              // [M, mt*16] bf16
    int M_act, int mt, int nt) {
  constexpr int K = 5;
  __shared__ __attribute__((aligned(16))) float o_lds[M_CAP][128]; // 0.5..4.0 KiB!

  const int tid = threadIdx.x;
  const int w = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 3;
  const int q = lane & 7;
  const int b = blockIdx.x;
  const int i = b * 8 + w;
  const int n = nt * 16;
  const int m_out = mt * 16;

  float a0[M_CAP], a1[M_CAP];
#pragma unroll
  for (int r = 0; r < M_CAP; ++r) { a0[r] = 0.f; a1[r] = 0.f; }

  // 2-stage register software prefetch of K=5 split-plane weights:
  // Wave-uniform 64-bit SGPR bases + 32-bit uint32_t VGPR offsets -> unlocks RDNA3.5 SADDR mode (global_load v, voff, s[base])!
  const uint32_t* p16_base = p16 + (size_t)i * nt * 32;
  const uint32_t* p4_base  = p4  + (size_t)i * nt * 8;
  u4v w4_cur = {0u, 0u, 0u, 0u};
  uint32_t wl_cur = 0u;
  if (g < nt) {
    const uint32_t o0 = (uint32_t)(g * 8 + q);
    w4_cur = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(p16_base + (o0 << 2)));
    wl_cur = __builtin_nontemporal_load(p4_base + o0);
  }

  for (int j = g; j < nt; j += 4) {
    uint32_t V[6];
    V[0] = __builtin_amdgcn_permlane16(wl_cur, wl_cur, 0x65432107u, 0xedcba98fu, false, false);
    V[1] = w4_cur.x; V[2] = w4_cur.y; V[3] = w4_cur.z; V[4] = w4_cur.w; V[5] = wl_cur;

    // Half-tile xv[M_CAP][4] + bank-staggered (U_lo, U_hi) pairing:
    //   1) Halves xv VGPR pressure + 32-bit SADDR offsets -> <=96 VGPRs (FULL 16w/SIMD at M=1..7/8!)
    //   2) Staggers xv[r][k] vs xv[r][k^1] across different VGPR banks to unlock V_DUAL_DOT2ACC_F32_F16!
    const uint32_t xj_off = (uint32_t)(j * 16);
    h2v xv[M_CAP][4];
#pragma unroll
    for (int r = 0; r < M_CAP; ++r) {
      const uint32_t rr = (uint32_t)((r < M_act) ? r : 0);
      const uint4 u0 = reinterpret_cast<const uint4*>(xt + rr * (uint32_t)n + xj_off)[0];
      xv[r][0] = as_h2(u0.x); xv[r][1] = as_h2(u0.y); xv[r][2] = as_h2(u0.z); xv[r][3] = as_h2(u0.w);
    }
#define NS1_DUAL(U0, U1) { \
      const h2v w0 = as_h2(val_f16bits2(ext16<16 + ((U0) + 1) * K>(V), ext16<16 + ((U0) + 2) * K>(V))); \
      const h2v w1 = as_h2(val_f16bits2(ext16<16 + ((U1) + 1) * K>(V), ext16<16 + ((U1) + 2) * K>(V))); \
      _Pragma("unroll") for (int r = 0; r < M_CAP; ++r) { \
        a0[r] = __builtin_amdgcn_fdot2(w0, xv[r][((U0) & 7) >> 1], a0[r], false); \
        a1[r] = __builtin_amdgcn_fdot2(w1, xv[r][((U1) & 7) >> 1], a1[r], false); \
      } }
    NS1_DUAL(0, 18) NS1_DUAL(2, 16) NS1_DUAL(4, 22) NS1_DUAL(6, 20)
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int r = 0; r < M_CAP; ++r) {
      const uint32_t rr = (uint32_t)((r < M_act) ? r : 0);
      const uint4 u1 = reinterpret_cast<const uint4*>(xt + rr * (uint32_t)n + xj_off)[1];
      xv[r][0] = as_h2(u1.x); xv[r][1] = as_h2(u1.y); xv[r][2] = as_h2(u1.z); xv[r][3] = as_h2(u1.w);
    }
    u4v w4_nxt = {0u, 0u, 0u, 0u};
    uint32_t wl_nxt = 0u;
    if (j + 4 < nt) {
      const uint32_t on = (uint32_t)((j + 4) * 8 + q);
      w4_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(p16_base + (on << 2)));
      wl_nxt = __builtin_nontemporal_load(p4_base + on);
    }
    __builtin_amdgcn_sched_barrier(0);
    NS1_DUAL(8, 26) NS1_DUAL(10, 24) NS1_DUAL(12, 30) NS1_DUAL(14, 28)
#undef NS1_DUAL
    w4_cur = w4_nxt;
    wl_cur = wl_nxt;
  }

  // Wave-local reduction across the 4 g-groups (ZERO global partials, ZERO atomics, ZERO fences!)
#pragma unroll
  for (int r = 0; r < M_CAP; ++r) {
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
  }
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M_CAP; ++r) {
      if (r < M_act) {
        const float off = 1534.0f * xs[r];
        o_lds[r][w * 16 + 2 * q]     = a0[r] - off;
        o_lds[r][w * 16 + 2 * q + 1] = a1[r] - off;
      }
    }
  }
  __syncthreads();

  // Direct in-WG Wave32 FWHT-128 + scales + sv + bias + coalesced 8B uint2 BF16 store
  if (w < M_act) {
    const int base_col = b * 128 + lane * 4;
    const float tt = tok[w] * cb;
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      v[e] = o_lds[w][lane * 4 + e] * __half2float(scales[base_col + e]) * tt;
    }
    fwht128_wave(v, lane);
    const unsigned sg4 = (unsigned)sv[b * 16 + (lane >> 1)] >> ((lane & 1) * 4);
    float ob[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float o = v[e] * (1.0f / 11.313708498984761f);
      if ((sg4 >> e) & 1u) o = -o;
      if (bias != nullptr) o += __uint_as_float(((uint32_t)bias[base_col + e]) << 16);
      ob[e] = o;
    }
    uint2 out_p;
    out_p.x = f32_to_bf16x2_rne(ob[0], ob[1]);
    out_p.y = f32_to_bf16x2_rne(ob[2], ob[3]);
    *reinterpret_cast<uint2*>(y + (size_t)w * m_out + base_col) = out_p;
  }
}

torch::Tensor gemv_k5_sp16_ns1(
    torch::Tensor p16, torch::Tensor p4, torch::Tensor xt, torch::Tensor xs, torch::Tensor tok,
    torch::Tensor scales, double cb, torch::Tensor sv, c10::optional<torch::Tensor> bias) {
  TORCH_CHECK(p16.is_cuda() && p16.is_contiguous() && p16.dim() == 3 && p16.size(2) == 32);
  TORCH_CHECK(p4.is_cuda() && p4.is_contiguous() && p4.dim() == 3 && p4.size(2) == 8);
  TORCH_CHECK(xt.is_cuda() && xt.is_contiguous() && xt.dim() == 2 && xt.scalar_type() == torch::kHalf);
  const int mt = (int)p16.size(0), nt = (int)p16.size(1);
  const int M = (int)xt.size(0);
  TORCH_CHECK((mt % 8 == 0) && xt.size(1) == nt * 16 && M >= 1 && M <= 8);
  auto y = torch::empty({M, mt * 16}, xt.options().dtype(torch::kBFloat16));
  auto st = c10::hip::getCurrentHIPStream().stream();
  auto p16p = reinterpret_cast<const uint32_t*>(p16.data_ptr<int32_t>());
  auto p4p = reinterpret_cast<const uint32_t*>(p4.data_ptr<int32_t>());
  auto xp = reinterpret_cast<const _Float16*>(xt.data_ptr<at::Half>());
  auto xsp = xs.data_ptr<float>();
  auto tp = tok.data_ptr<float>();
  auto sp = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  auto svp = sv.data_ptr<uint8_t>();
  const uint16_t* bp = (bias.has_value() && bias->defined())
      ? reinterpret_cast<const uint16_t*>(bias->data_ptr<at::BFloat16>()) : nullptr;
  auto yp = reinterpret_cast<uint16_t*>(y.data_ptr<at::BFloat16>());
  const dim3 grid(mt / 8);
  if (M == 1) {
    hipLaunchKernelGGL((k_gemv_k5_sp16_ns1<1>), grid, dim3(256), 0, st, p16p, p4p, xp, xsp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else if (M == 2) {
    hipLaunchKernelGGL((k_gemv_k5_sp16_ns1<2>), grid, dim3(256), 0, st, p16p, p4p, xp, xsp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else if (M <= 4) {
    hipLaunchKernelGGL((k_gemv_k5_sp16_ns1<4>), grid, dim3(256), 0, st, p16p, p4p, xp, xsp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else if (M <= 7) {
    hipLaunchKernelGGL((k_gemv_k5_sp16_ns1<7>), grid, dim3(256), 0, st, p16p, p4p, xp, xsp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  } else {
    hipLaunchKernelGGL((k_gemv_k5_sp16_ns1<8>), grid, dim3(256), 0, st, p16p, p4p, xp, xsp, tp, sp, (float)cb, svp, bp, yp, M, mt, nt);
  }
  return y;
}

// ============================================================================
// Fused ERNIE-5.5 Prologues for HTR K=5 Linears (eliminates 90 launches/step + 728 MB/prefill!)
//   1. situ_and_mul_htr_prologue: gate_up [T, 2*D] bf16 -> SiTU-GLU(fast_tanhf) -> HTR prologue (xt, xs, tok)
//      Covers 42 shared_experts.down_proj + 3 DSpark shared_experts + 1 layer-0 MLP (46x/step!)
//   2. rmsnorm_htr_prologue:      x [T, D] bf16 -> RMSNorm(v_dot2_f32_bf16) -> HTR prologue (xt, xs, tok)
//      Covers 42 routed MoE latent_norm->fc2_latent_proj + 3 DSpark layers (45x/step!)
// Exact bit-compatible rounding with (SituAndMul / RMSNorm -> BF16 -> htr_kernels k_prologue1p)!
// ============================================================================
__device__ __forceinline__ float wave_max_f(float a) {
  a = fmaxf(a, shx<16>(a)); a = fmaxf(a, shx<8>(a)); a = fmaxf(a, shx<4>(a)); a = fmaxf(a, shx<2>(a)); a = fmaxf(a, shx<1>(a));
  return a;
}
__device__ __forceinline__ float wave_sum_f(float a) {
  a += shx<16>(a); a += shx<8>(a); a += shx<4>(a); a += shx<2>(a); a += shx<1>(a);
  return a;
}

template <int NB>
__global__ __launch_bounds__(256) void k_situ_and_mul_htr_prologue(
    const uint16_t* __restrict__ gate_up, // [T, 2*D] bf16 (stride gu_s0)
    const uint8_t* __restrict__ su,       // [D/8] u8
    _Float16* __restrict__ xt,            // [T, D] fp16
    float* __restrict__ xs,               // [T] fp32
    float* __restrict__ tok,              // [T] fp32
    int64_t gu_s0, int D, float beta, float inv_beta, float linear_beta, float inv_lbeta) {
  __shared__ float red[8];
  __shared__ float s_scale;
  const int row = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int nblk = D >> 7;
  const uint16_t* gr = gate_up + (int64_t)row * gu_s0;
  const uint16_t* ur = gr + D;

  float v[NB][4];
  float amax = 0.f;
#pragma unroll
  for (int t = 0; t < NB; ++t) {
    const int blk = w + 8 * t;
    if (blk < nblk) {
      const int base = blk * 128 + lane * 4;
      const uint2 rg = *reinterpret_cast<const uint2*>(gr + base);
      const uint2 ru = *reinterpret_cast<const uint2*>(ur + base);
      const float gv[4] = {
        __uint_as_float((rg.x & 0xFFFFu) << 16), __uint_as_float(rg.x & 0xFFFF0000u),
        __uint_as_float((rg.y & 0xFFFFu) << 16), __uint_as_float(rg.y & 0xFFFF0000u),
      };
      const float uv[4] = {
        __uint_as_float((ru.x & 0xFFFFu) << 16), __uint_as_float(ru.x & 0xFFFF0000u),
        __uint_as_float((ru.y & 0xFFFFu) << 16), __uint_as_float(ru.y & 0xFFFF0000u),
      };
      const unsigned sg4 = (unsigned)su[base >> 3] >> (base & 7);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float g = gv[e], u = uv[e];
        const float sg = beta * fast_tanhf(g * inv_beta) * __builtin_amdgcn_rcpf(1.0f + exp2f(-1.4426950408889634f * g));
        const float su_v = (linear_beta > 0.0f) ? (linear_beta * fast_tanhf(u * inv_lbeta)) : u;
        const float y = __uint_as_float(((uint32_t)f32_to_bf16_rne(sg * su_v)) << 16);
        amax = fmaxf(amax, fabsf(y));
        v[t][e] = ((sg4 >> e) & 1u) ? -y : y;
      }
    }
  }
  amax = wave_max_f(amax);
  if (lane == 0) red[w] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = red[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) m = fmaxf(m, red[k]);
    const float bound = m * 11.4f;
    s_scale = bound > 30000.f ? exp2f(ceilf(log2f(bound * (1.0f / 30000.f)))) : 1.f;
    tok[row] = s_scale;
  }
  __syncthreads();
  const float inv = (1.0f / 11.313708498984761f) * __builtin_amdgcn_rcpf(s_scale);
  float sum = 0.f;
#pragma unroll
  for (int t = 0; t < NB; ++t) {
    const int blk = w + 8 * t;
    if (blk < nblk) {
      fwht128_wave(v[t], lane);
      const int base = blk * 128 + lane * 4;
      const h4v h4 = {(_Float16)(v[t][0] * inv), (_Float16)(v[t][1] * inv),
                      (_Float16)(v[t][2] * inv), (_Float16)(v[t][3] * inv)};
      *reinterpret_cast<h4v*>(xt + (size_t)row * D + base) = h4;
      sum += (float)h4.x + (float)h4.y + (float)h4.z + (float)h4.w;
    }
  }
  sum = wave_sum_f(sum);
  __syncthreads();
  if (lane == 0) red[w] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = red[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) s += red[k];
    xs[row] = s;
  }
}

std::vector<torch::Tensor> situ_and_mul_htr_prologue(
    torch::Tensor gate_up, torch::Tensor su, double beta, double linear_beta) {
  TORCH_CHECK(gate_up.is_cuda() && gate_up.dim() == 2 && gate_up.scalar_type() == torch::kBFloat16 && gate_up.stride(1) == 1);
  const int T = (int)gate_up.size(0);
  const int D = (int)(gate_up.size(1) / 2);
  TORCH_CHECK((gate_up.size(1) == 2 * D) && (D % 128 == 0) && D <= 12288);
  auto xt = torch::empty({T, D}, gate_up.options().dtype(torch::kHalf));
  auto xs = torch::empty({T}, gate_up.options().dtype(torch::kFloat32));
  auto tok = torch::empty({T}, gate_up.options().dtype(torch::kFloat32));
  if (T == 0) return {xt, xs, tok};
  const int nblk = D >> 7;
  const int nb = (nblk + 7) / 8;
  auto st = c10::hip::getCurrentHIPStream().stream();
  const uint16_t* gup = reinterpret_cast<const uint16_t*>(gate_up.data_ptr());
  const uint8_t* sup = su.data_ptr<uint8_t>();
  _Float16* xtp = reinterpret_cast<_Float16*>(xt.data_ptr<at::Half>());
  float* xsp = xs.data_ptr<float>();
  float* tokp = tok.data_ptr<float>();
  const float bf = (float)beta, ibf = 1.0f / bf;
  const float lbf = (float)linear_beta, ilbf = (lbf > 0.0f) ? (1.0f / lbf) : 0.0f;
  if (nb <= 2) {
    hipLaunchKernelGGL((k_situ_and_mul_htr_prologue<2>), dim3(T), dim3(256), 0, st, gup, sup, xtp, xsp, tokp, gate_up.stride(0), D, bf, ibf, lbf, ilbf);
  } else if (nb <= 4) {
    hipLaunchKernelGGL((k_situ_and_mul_htr_prologue<4>), dim3(T), dim3(256), 0, st, gup, sup, xtp, xsp, tokp, gate_up.stride(0), D, bf, ibf, lbf, ilbf);
  } else if (nb <= 8) {
    hipLaunchKernelGGL((k_situ_and_mul_htr_prologue<8>), dim3(T), dim3(256), 0, st, gup, sup, xtp, xsp, tokp, gate_up.stride(0), D, bf, ibf, lbf, ilbf);
  } else {
    hipLaunchKernelGGL((k_situ_and_mul_htr_prologue<12>), dim3(T), dim3(256), 0, st, gup, sup, xtp, xsp, tokp, gate_up.stride(0), D, bf, ibf, lbf, ilbf);
  }
  return {xt, xs, tok};
}

template <int NB>
__global__ __launch_bounds__(256) void k_rmsnorm_htr_prologue(
    const uint16_t* __restrict__ x,       // [T, D] bf16 (stride x_s0)
    const uint16_t* __restrict__ norm_w,  // [D] bf16
    const uint8_t* __restrict__ su,       // [D/8] u8
    _Float16* __restrict__ xt,            // [T, D] fp16
    float* __restrict__ xs,               // [T] fp32
    float* __restrict__ tok,              // [T] fp32
    int64_t x_s0, int D, float eps) {
  __shared__ float red[8];
  __shared__ float s_rrms;
  __shared__ float s_scale;
  const int row = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int nblk = D >> 7;
  const uint16_t* xr = x + (int64_t)row * x_s0;

  uint2 xraw[NB];
  float ss0 = 0.f, ss1 = 0.f;
#pragma unroll
  for (int t = 0; t < NB; ++t) {
    const int blk = w + 8 * t;
    if (blk < nblk) {
      const int base = blk * 128 + lane * 4;
      xraw[t] = *reinterpret_cast<const uint2*>(xr + base);
      ss0 = bf16_dot2(xraw[t].x, xraw[t].x, ss0);
      ss1 = bf16_dot2(xraw[t].y, xraw[t].y, ss1);
    }
  }
  const float wss = wave_sum_f(ss0 + ss1);
  if (lane == 0) red[w] = wss;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = red[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) s += red[k];
    s_rrms = rsqrtf(s * __builtin_amdgcn_rcpf((float)D) + eps);
  }
  __syncthreads();
  const float rrms = s_rrms;

  float v[NB][4];
  float amax = 0.f;
#pragma unroll
  for (int t = 0; t < NB; ++t) {
    const int blk = w + 8 * t;
    if (blk < nblk) {
      const int base = blk * 128 + lane * 4;
      const uint2 rw = *reinterpret_cast<const uint2*>(norm_w + base);
      const float xv[4] = {
        __uint_as_float((xraw[t].x & 0xFFFFu) << 16), __uint_as_float(xraw[t].x & 0xFFFF0000u),
        __uint_as_float((xraw[t].y & 0xFFFFu) << 16), __uint_as_float(xraw[t].y & 0xFFFF0000u),
      };
      const float wv[4] = {
        __uint_as_float((rw.x & 0xFFFFu) << 16), __uint_as_float(rw.x & 0xFFFF0000u),
        __uint_as_float((rw.y & 0xFFFFu) << 16), __uint_as_float(rw.y & 0xFFFF0000u),
      };
      const unsigned sg4 = (unsigned)su[base >> 3] >> (base & 7);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float y = __uint_as_float(((uint32_t)f32_to_bf16_rne(xv[e] * rrms * wv[e])) << 16);
        amax = fmaxf(amax, fabsf(y));
        v[t][e] = ((sg4 >> e) & 1u) ? -y : y;
      }
    }
  }
  amax = wave_max_f(amax);
  __syncthreads();
  if (lane == 0) red[w] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = red[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) m = fmaxf(m, red[k]);
    const float bound = m * 11.4f;
    s_scale = bound > 30000.f ? exp2f(ceilf(log2f(bound * (1.0f / 30000.f)))) : 1.f;
    tok[row] = s_scale;
  }
  __syncthreads();
  const float inv = (1.0f / 11.313708498984761f) * __builtin_amdgcn_rcpf(s_scale);
  float sum = 0.f;
#pragma unroll
  for (int t = 0; t < NB; ++t) {
    const int blk = w + 8 * t;
    if (blk < nblk) {
      fwht128_wave(v[t], lane);
      const int base = blk * 128 + lane * 4;
      const h4v h4 = {(_Float16)(v[t][0] * inv), (_Float16)(v[t][1] * inv),
                      (_Float16)(v[t][2] * inv), (_Float16)(v[t][3] * inv)};
      *reinterpret_cast<h4v*>(xt + (size_t)row * D + base) = h4;
      sum += (float)h4.x + (float)h4.y + (float)h4.z + (float)h4.w;
    }
  }
  sum = wave_sum_f(sum);
  __syncthreads();
  if (lane == 0) red[w] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = red[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) s += red[k];
    xs[row] = s;
  }
}

std::vector<torch::Tensor> rmsnorm_htr_prologue(
    torch::Tensor x, torch::Tensor norm_w, torch::Tensor su, double eps) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == torch::kBFloat16 && x.stride(1) == 1);
  TORCH_CHECK(norm_w.is_cuda() && norm_w.scalar_type() == torch::kBFloat16 && norm_w.is_contiguous());
  const int T = (int)x.size(0);
  const int D = (int)x.size(1);
  TORCH_CHECK((D % 128 == 0) && D <= 8192);
  auto xt = torch::empty({T, D}, x.options().dtype(torch::kHalf));
  auto xs = torch::empty({T}, x.options().dtype(torch::kFloat32));
  auto tok = torch::empty({T}, x.options().dtype(torch::kFloat32));
  if (T == 0) return {xt, xs, tok};
  const int nb = ((D >> 7) + 7) / 8;
  auto st = c10::hip::getCurrentHIPStream().stream();
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
  const uint16_t* wp = reinterpret_cast<const uint16_t*>(norm_w.data_ptr());
  const uint8_t* sup = su.data_ptr<uint8_t>();
  _Float16* xtp = reinterpret_cast<_Float16*>(xt.data_ptr<at::Half>());
  float* xsp = xs.data_ptr<float>();
  float* tokp = tok.data_ptr<float>();
  if (nb <= 2) {
    hipLaunchKernelGGL((k_rmsnorm_htr_prologue<2>), dim3(T), dim3(256), 0, st, xp, wp, sup, xtp, xsp, tokp, x.stride(0), D, (float)eps);
  } else if (nb <= 4) {
    hipLaunchKernelGGL((k_rmsnorm_htr_prologue<4>), dim3(T), dim3(256), 0, st, xp, wp, sup, xtp, xsp, tokp, x.stride(0), D, (float)eps);
  } else {
    hipLaunchKernelGGL((k_rmsnorm_htr_prologue<8>), dim3(T), dim3(256), 0, st, xp, wp, sup, xtp, xsp, tokp, x.stride(0), D, (float)eps);
  }
  return {xt, xs, tok};
}

// 3. sigmoid_mul_htr_prologue: o [T, D] bf16 * sigmoid(gate [T, D] bf16) -> HTR prologue (xt, xs, tok)
//    Covers all 6 DSA layers' wo_b(o.flatten(1) * torch.sigmoid(o_gate(q_c))) (-12 launches/step, -384 MB/prefill!)
template <int NB>
__global__ __launch_bounds__(256) void k_sigmoid_mul_htr_prologue(
    const uint16_t* __restrict__ o_in,    // [T, D] bf16 (stride o_s0)
    const uint16_t* __restrict__ g_in,    // [T, D] bf16 (stride g_s0)
    const uint8_t* __restrict__ su,       // [D/8] u8
    _Float16* __restrict__ xt,            // [T, D] fp16
    float* __restrict__ xs,               // [T] fp32
    float* __restrict__ tok,              // [T] fp32
    int64_t o_s0, int64_t g_s0, int D) {
  __shared__ float red[8];
  __shared__ float s_scale;
  const int row = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int nblk = D >> 7;
  const uint16_t* orow = o_in + (int64_t)row * o_s0;
  const uint16_t* grow = g_in + (int64_t)row * g_s0;

  float v[NB][4];
  float amax = 0.f;
#pragma unroll
  for (int t = 0; t < NB; ++t) {
    const int blk = w + 8 * t;
    if (blk < nblk) {
      const int base = blk * 128 + lane * 4;
      const uint2 ro = *reinterpret_cast<const uint2*>(orow + base);
      const uint2 rg = *reinterpret_cast<const uint2*>(grow + base);
      const float ov[4] = {
        __uint_as_float((ro.x & 0xFFFFu) << 16), __uint_as_float(ro.x & 0xFFFF0000u),
        __uint_as_float((ro.y & 0xFFFFu) << 16), __uint_as_float(ro.y & 0xFFFF0000u),
      };
      const float gv[4] = {
        __uint_as_float((rg.x & 0xFFFFu) << 16), __uint_as_float(rg.x & 0xFFFF0000u),
        __uint_as_float((rg.y & 0xFFFFu) << 16), __uint_as_float(rg.y & 0xFFFF0000u),
      };
      const unsigned sg4 = (unsigned)su[base >> 3] >> (base & 7);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float sig_b = __uint_as_float(((uint32_t)f32_to_bf16_rne(__builtin_amdgcn_rcpf(1.0f + exp2f(-1.4426950408889634f * gv[e])))) << 16);
        const float y = __uint_as_float(((uint32_t)f32_to_bf16_rne(ov[e] * sig_b)) << 16);
        amax = fmaxf(amax, fabsf(y));
        v[t][e] = ((sg4 >> e) & 1u) ? -y : y;
      }
    }
  }
  amax = wave_max_f(amax);
  if (lane == 0) red[w] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = red[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) m = fmaxf(m, red[k]);
    const float bound = m * 11.4f;
    s_scale = bound > 30000.f ? exp2f(ceilf(log2f(bound * (1.0f / 30000.f)))) : 1.f;
    tok[row] = s_scale;
  }
  __syncthreads();
  const float inv = (1.0f / 11.313708498984761f) * __builtin_amdgcn_rcpf(s_scale);
  float sum = 0.f;
#pragma unroll
  for (int t = 0; t < NB; ++t) {
    const int blk = w + 8 * t;
    if (blk < nblk) {
      fwht128_wave(v[t], lane);
      const int base = blk * 128 + lane * 4;
      const h4v h4 = {(_Float16)(v[t][0] * inv), (_Float16)(v[t][1] * inv),
                      (_Float16)(v[t][2] * inv), (_Float16)(v[t][3] * inv)};
      *reinterpret_cast<h4v*>(xt + (size_t)row * D + base) = h4;
      sum += (float)h4.x + (float)h4.y + (float)h4.z + (float)h4.w;
    }
  }
  sum = wave_sum_f(sum);
  __syncthreads();
  if (lane == 0) red[w] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = red[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) s += red[k];
    xs[row] = s;
  }
}

std::vector<torch::Tensor> sigmoid_mul_htr_prologue(
    torch::Tensor o_in, torch::Tensor g_in, torch::Tensor su) {
  TORCH_CHECK(o_in.is_cuda() && o_in.dim() == 2 && o_in.scalar_type() == torch::kBFloat16 && o_in.stride(1) == 1);
  TORCH_CHECK(g_in.is_cuda() && g_in.dim() == 2 && g_in.scalar_type() == torch::kBFloat16 && g_in.stride(1) == 1);
  const int T = (int)o_in.size(0);
  const int D = (int)o_in.size(1);
  TORCH_CHECK(g_in.size(0) == T && g_in.size(1) == D && (D % 128 == 0) && D <= 8192);
  auto xt = torch::empty({T, D}, o_in.options().dtype(torch::kHalf));
  auto xs = torch::empty({T}, o_in.options().dtype(torch::kFloat32));
  auto tok = torch::empty({T}, o_in.options().dtype(torch::kFloat32));
  if (T == 0) return {xt, xs, tok};
  const int nb = ((D >> 7) + 7) / 8;
  auto st = c10::hip::getCurrentHIPStream().stream();
  const uint16_t* op = reinterpret_cast<const uint16_t*>(o_in.data_ptr());
  const uint16_t* gp = reinterpret_cast<const uint16_t*>(g_in.data_ptr());
  const uint8_t* sup = su.data_ptr<uint8_t>();
  _Float16* xtp = reinterpret_cast<_Float16*>(xt.data_ptr<at::Half>());
  float* xsp = xs.data_ptr<float>();
  float* tokp = tok.data_ptr<float>();
  if (nb <= 2) {
    hipLaunchKernelGGL((k_sigmoid_mul_htr_prologue<2>), dim3(T), dim3(256), 0, st, op, gp, sup, xtp, xsp, tokp, o_in.stride(0), g_in.stride(0), D);
  } else if (nb <= 4) {
    hipLaunchKernelGGL((k_sigmoid_mul_htr_prologue<4>), dim3(T), dim3(256), 0, st, op, gp, sup, xtp, xsp, tokp, o_in.stride(0), g_in.stride(0), D);
  } else {
    hipLaunchKernelGGL((k_sigmoid_mul_htr_prologue<8>), dim3(T), dim3(256), 0, st, op, gp, sup, xtp, xsp, tokp, o_in.stride(0), g_in.stride(0), D);
  }
  return {xt, xs, tok};
}


// ============================================================================
// Wave 24A: Unsplit K=5 Decode GEMV (M=1..8, BF16) with:
//   - Always-on __builtin_nontemporal_load for ALL M=1..8 (protects 32MB IC!)
//   - 1x16B (u4va) + 1x4B NT loads + permlane16 V[0] (2 loads vs 6!)
//   - 2-stage register software prefetch of LPDDR5X weights
//   - Half-tile xv[M][4] + v_sad_hi_u8 + bank-staggered V_DUAL_DOT2ACC_F32_F16
//   - Zero-VRAM cyclic K-tile LPDDR5X channel stagger for tall/low-ns shapes
//   - 16B NT partial reduction + 8B vectorized BF16 epilogue writes
// ============================================================================
typedef float f4v __attribute__((ext_vector_type(4)));

template <int M>
__device__ __forceinline__ void k5_unsplit_tile_loop(
    const uint32_t* __restrict__ qrow,
    const _Float16* __restrict__ xt,
    int j0, int j1, int n, int lane, int phase,
    float a0[M], float a1[M]) {
  const int g = lane >> 3, q = lane & 7;
  const int nloc = j1 - j0;
  if (nloc <= 0) return;

  u4v wa_cur = {0u, 0u, 0u, 0u};
  uint32_t w5_cur = 0u;
  {
    const int jl0 = ((phase < nloc) ? phase : 0) + g;
    if (jl0 < nloc) {
      const uint32_t* tp0 = qrow + (size_t)(j0 + jl0) * 40;
      wa_cur = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tp0 + q * 5));
      w5_cur = __builtin_nontemporal_load(tp0 + q * 5 + 4);
    }
  }

  for (int step = 0; step < nloc; step += 4) {
    int jl_base = step + phase;
    if (jl_base >= nloc) jl_base -= nloc;
    const int jl = jl_base + g;
    const bool valid = (jl < nloc);

    u4v wa_nxt = {0u, 0u, 0u, 0u};
    uint32_t w5_nxt = 0u;
    if (step + 4 < nloc) {
      int jn_base = step + 4 + phase;
      if (jn_base >= nloc) jn_base -= nloc;
      const int jn = jn_base + g;
      if (jn < nloc) {
        const uint32_t* tpn = qrow + (size_t)(j0 + jn) * 40;
        wa_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(tpn + q * 5));
        w5_nxt = __builtin_nontemporal_load(tpn + q * 5 + 4);
      }
    }

    if (valid) {
      uint32_t V[6];
      V[0] = __builtin_amdgcn_permlane16(w5_cur, w5_cur, 0x65432107u, 0xedcba98fu, false, false);
      V[1] = wa_cur.x; V[2] = wa_cur.y; V[3] = wa_cur.z; V[4] = wa_cur.w; V[5] = w5_cur;
      __builtin_amdgcn_sched_barrier(0);

      const int j = j0 + jl;
      h2v xv[M][4];
#pragma unroll
      for (int r = 0; r < M; ++r) {
        const uint4 u0 = reinterpret_cast<const uint4*>(xt + (size_t)r * n + j * 16)[0];
        xv[r][0] = as_h2(u0.x); xv[r][1] = as_h2(u0.y); xv[r][2] = as_h2(u0.z); xv[r][3] = as_h2(u0.w);
      }
#define K5U_DUAL(U0, U1)                                                                             \
      {                                                                                              \
        const h2v w0 = as_h2(val_f16bits2(ext16<16 + ((U0) + 1) * 5>(V), ext16<16 + ((U0) + 2) * 5>(V))); \
        const h2v w1 = as_h2(val_f16bits2(ext16<16 + ((U1) + 1) * 5>(V), ext16<16 + ((U1) + 2) * 5>(V))); \
        _Pragma("unroll") for (int r = 0; r < M; ++r) {                                              \
          a0[r] = __builtin_amdgcn_fdot2(w0, xv[r][((U0) & 7) >> 1], a0[r], false);                  \
          a1[r] = __builtin_amdgcn_fdot2(w1, xv[r][((U1) & 7) >> 1], a1[r], false);                  \
        }                                                                                            \
      }
      K5U_DUAL(0, 18) K5U_DUAL(2, 16) K5U_DUAL(4, 22) K5U_DUAL(6, 20)
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int r = 0; r < M; ++r) {
        const uint4 u1 = reinterpret_cast<const uint4*>(xt + (size_t)r * n + j * 16)[1];
        xv[r][0] = as_h2(u1.x); xv[r][1] = as_h2(u1.y); xv[r][2] = as_h2(u1.z); xv[r][3] = as_h2(u1.w);
      }
      K5U_DUAL(8, 26) K5U_DUAL(10, 24) K5U_DUAL(12, 30) K5U_DUAL(14, 28)
#undef K5U_DUAL
    }
    wa_cur = wa_nxt;
    w5_cur = w5_nxt;
  }
}

template <int M>
__global__ __launch_bounds__(256) void k_gemv_k5_unsplit_epi(
    const uint32_t* __restrict__ qw,
    const _Float16* __restrict__ xt,
    float* __restrict__ part,
    unsigned* __restrict__ counters,
    const float* __restrict__ xs,
    const float* __restrict__ tok,
    const __half* __restrict__ scales,
    float cb,
    const uint8_t* __restrict__ sv,
    const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ y,
    int mt, int nt, int nsplit, int tps) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int wave = blockIdx.x * 8 + w;
  if (wave >= mt * nsplit) return;
  const int i = wave / nsplit, sp = wave % nsplit;
  const int g = lane >> 3, q = lane & 7;
  const int n = nt * 16, m = mt * 16;

  float a0[M], a1[M];
#pragma unroll
  for (int r = 0; r < M; ++r) { a0[r] = 0.f; a1[r] = 0.f; }

  const int j0 = sp * tps, j1 = min(nt, j0 + tps);
  const int nloc = j1 - j0;
  const int phase = ((nloc & 3) == 0)
      ? ((nloc >= 64) ? ((w & 7) * 8) : ((nloc >= 32) ? ((w & 7) * 4) : ((nloc >= 16) ? ((w & 3) * 4) : 0)))
      : 0;
  k5_unsplit_tile_loop<M>(qw + (size_t)i * nt * 40, xt, j0, j1, n, lane, phase, a0, a1);

#pragma unroll
  for (int r = 0; r < M; ++r) {
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
  }
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
      float* pp = part + ((size_t)sp * M + r) * m + i * 16 + 2 * q;
      *reinterpret_cast<float2*>(pp) = make_float2(a0[r], a1[r]);
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  const int b = i >> 3;
  int last = 0;
  if (lane == 0) last = (atomicAdd(&counters[b], 1u) == (unsigned)(8 * nsplit - 1));
  last = __shfl(last, 0);
  if (!last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const int base = b * 128 + lane * 4;
  const h4v hs = *reinterpret_cast<const h4v*>(scales + base);
  const float sc[4] = {(float)hs.x * cb, (float)hs.y * cb, (float)hs.z * cb, (float)hs.w * cb};
  const unsigned sg4 = (unsigned)sv[b * 16 + (lane >> 1)] >> ((lane & 1) * 4);
  float bv[4] = {0.f, 0.f, 0.f, 0.f};
  if (bias != nullptr) {
    const uint2 br = *reinterpret_cast<const uint2*>(bias + base);
    bv[0] = __uint_as_float((br.x & 0xFFFFu) << 16);
    bv[1] = __uint_as_float(br.x & 0xFFFF0000u);
    bv[2] = __uint_as_float((br.y & 0xFFFFu) << 16);
    bv[3] = __uint_as_float(br.y & 0xFFFF0000u);
  }
  for (int r = 0; r < M; ++r) {
    f4v s4 = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < nsplit; ++k) {
      s4 += __builtin_nontemporal_load(reinterpret_cast<const f4v*>(part + ((size_t)k * M + r) * m + base));
    }
    const float c = 1534.0f * xs[r], t = tok[r];
    float v[4] = {
      (s4.x - c) * sc[0] * t,
      (s4.y - c) * sc[1] * t,
      (s4.z - c) * sc[2] * t,
      (s4.w - c) * sc[3] * t,
    };
    fwht128_wave(v, lane);
    float o[4] = {
      v[0] * (1.0f / 11.313708498984761f),
      v[1] * (1.0f / 11.313708498984761f),
      v[2] * (1.0f / 11.313708498984761f),
      v[3] * (1.0f / 11.313708498984761f),
    };
    if (sg4 & 1u) o[0] = -o[0];
    if (sg4 & 2u) o[1] = -o[1];
    if (sg4 & 4u) o[2] = -o[2];
    if (sg4 & 8u) o[3] = -o[3];
    if (bias != nullptr) { o[0] += bv[0]; o[1] += bv[1]; o[2] += bv[2]; o[3] += bv[3]; }
    uint2 out_p = {f32_to_bf16x2_rne(o[0], o[1]), f32_to_bf16x2_rne(o[2], o[3])};
    *reinterpret_cast<uint2*>(y + (size_t)r * m + base) = out_p;
  }
  if (lane == 0) counters[b] = 0u;
}

template <int M>
__global__ __launch_bounds__(256) void k_gemv_k5_unsplit_epi8(
    const uint32_t* __restrict__ qw,
    const _Float16* __restrict__ xt,
    float* __restrict__ part,
    unsigned* __restrict__ counters,
    const float* __restrict__ xs,
    const float* __restrict__ tok,
    const __half* __restrict__ scales,
    float cb,
    const uint8_t* __restrict__ sv,
    const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ y,
    int mt, int nt, int nsplit, int tps) {
  __shared__ float red[8][M][16];
  __shared__ unsigned is_last;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int wave = blockIdx.x * 8 + w;
  const int i = wave / nsplit, sp = wave % nsplit;
  const int g = lane >> 3, q = lane & 7;
  const int n = nt * 16, m = mt * 16, ns8 = nsplit >> 3;

  float a0[M], a1[M];
#pragma unroll
  for (int r = 0; r < M; ++r) { a0[r] = 0.f; a1[r] = 0.f; }

  const int j0 = sp * tps, j1 = min(nt, j0 + tps);
  const int nloc = j1 - j0;
  // For tps=8, waves w=0..7 already hit w*1280B = w*5*256B (perfect 8/8 channel bijection).
  // For tps=16,32,64, cyclic phase breaks channel aliasing across waves.
  const int phase = ((nloc & 15) == 0 && nloc >= 16)
      ? ((nloc >= 64) ? ((w & 7) * 8) : ((nloc >= 32) ? ((w & 7) * 4) : ((w & 3) * 4)))
      : 0;
  k5_unsplit_tile_loop<M>(qw + (size_t)i * nt * 40, xt, j0, j1, n, lane, phase, a0, a1);

#pragma unroll
  for (int r = 0; r < M; ++r) {
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
  }
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M; ++r) { red[w][r][2 * q] = a0[r]; red[w][r][2 * q + 1] = a1[r]; }
  }
  __syncthreads();
  if (threadIdx.x < M * 16) {
    const int r = threadIdx.x >> 4, c = threadIdx.x & 15;
    float s = 0.f;
#pragma unroll
    for (int ww = 0; ww < 8; ++ww) s += red[ww][r][c];
    part[((size_t)(sp >> 3) * M + r) * m + i * 16 + c] = s;
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();
  const int b = i >> 3;
  if (threadIdx.x == 0) is_last = (atomicAdd(&counters[b], 1u) == (unsigned)(nsplit - 1));
  __syncthreads();
  if (!is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const int base = b * 128 + lane * 4;
  if (w < M) {
    const h4v hs = *reinterpret_cast<const h4v*>(scales + base);
    const float sc[4] = {(float)hs.x * cb, (float)hs.y * cb, (float)hs.z * cb, (float)hs.w * cb};
    const unsigned sg4 = (unsigned)sv[b * 16 + (lane >> 1)] >> ((lane & 1) * 4);
    float bv[4] = {0.f, 0.f, 0.f, 0.f};
    if (bias != nullptr) {
      const uint2 br = *reinterpret_cast<const uint2*>(bias + base);
      bv[0] = __uint_as_float((br.x & 0xFFFFu) << 16);
      bv[1] = __uint_as_float(br.x & 0xFFFF0000u);
      bv[2] = __uint_as_float((br.y & 0xFFFFu) << 16);
      bv[3] = __uint_as_float(br.y & 0xFFFF0000u);
    }
    const int r = w;
    f4v s4 = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < ns8; ++k) {
      s4 += __builtin_nontemporal_load(reinterpret_cast<const f4v*>(part + ((size_t)k * M + r) * m + base));
    }
    const float c = 1534.0f * xs[r], t = tok[r];
    float v[4] = {
      (s4.x - c) * sc[0] * t,
      (s4.y - c) * sc[1] * t,
      (s4.z - c) * sc[2] * t,
      (s4.w - c) * sc[3] * t,
    };
    fwht128_wave(v, lane);
    float o[4] = {
      v[0] * (1.0f / 11.313708498984761f),
      v[1] * (1.0f / 11.313708498984761f),
      v[2] * (1.0f / 11.313708498984761f),
      v[3] * (1.0f / 11.313708498984761f),
    };
    if (sg4 & 1u) o[0] = -o[0];
    if (sg4 & 2u) o[1] = -o[1];
    if (sg4 & 4u) o[2] = -o[2];
    if (sg4 & 8u) o[3] = -o[3];
    if (bias != nullptr) { o[0] += bv[0]; o[1] += bv[1]; o[2] += bv[2]; o[3] += bv[3]; }
    uint2 out_p = {f32_to_bf16x2_rne(o[0], o[1]), f32_to_bf16x2_rne(o[2], o[3])};
    *reinterpret_cast<uint2*>(y + (size_t)r * m + base) = out_p;
  }
  if (threadIdx.x == 0) counters[b] = 0u;
}

torch::Tensor gemv_k5_unsplit_epi(
    torch::Tensor qw, torch::Tensor xt, torch::Tensor xs, torch::Tensor tok,
    torch::Tensor scales, double cb, torch::Tensor sv,
    c10::optional<torch::Tensor> bias, torch::Tensor counters,
    int64_t nsplit, int64_t use_e8) {
  TORCH_CHECK(qw.is_cuda() && qw.is_contiguous() && qw.dim() == 3 && qw.size(2) == 40);
  TORCH_CHECK(xt.is_cuda() && xt.is_contiguous() && xt.dim() == 2 && xt.scalar_type() == torch::kHalf);
  const int mt = (int)qw.size(0), nt = (int)qw.size(1);
  const int M = (int)xt.size(0);
  const int m = mt * 16;
  TORCH_CHECK(M >= 1 && M <= 8 && (mt % 8) == 0 && xt.size(1) == nt * 16);
  const int ns = std::max(1, std::min((int)nsplit, nt));
  const int tps = (nt + ns - 1) / ns;
  const int ns_eff = (nt + tps - 1) / tps;
  const bool e8 = (use_e8 != 0) && ((ns_eff % 8) == 0);
  const int ns_part = e8 ? (ns_eff >> 3) : ns_eff;

  auto part = torch::empty({ns_part, M, m}, qw.options().dtype(torch::kFloat32));
  auto y = torch::empty({M, m}, qw.options().dtype(torch::kBFloat16));
  auto st = c10::hip::getCurrentHIPStream().stream();

  const uint32_t* qwp = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>());
  const _Float16* xtp = reinterpret_cast<const _Float16*>(xt.data_ptr<at::Half>());
  float* pp = part.data_ptr<float>();
  unsigned* cntp = reinterpret_cast<unsigned*>(counters.data_ptr<int32_t>());
  const float* xsp = xs.data_ptr<float>();
  const float* tokp = tok.data_ptr<float>();
  const __half* sclp = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  const uint8_t* svp = sv.data_ptr<uint8_t>();
  const uint16_t* bp = bias.has_value() ? reinterpret_cast<const uint16_t*>(bias->data_ptr()) : nullptr;
  uint16_t* yp = reinterpret_cast<uint16_t*>(y.data_ptr());

  dim3 grid = e8 ? dim3(mt * (ns_eff >> 3)) : dim3((mt * ns_eff + 7) / 8);
#define LAUNCH_K5U(MM)                                                                                         \
  if (e8) hipLaunchKernelGGL((k_gemv_k5_unsplit_epi8<MM>), grid, dim3(256), 0, st,                             \
                             qwp, xtp, pp, cntp, xsp, tokp, sclp, (float)cb, svp, bp, yp, mt, nt, ns_eff, tps); \
  else    hipLaunchKernelGGL((k_gemv_k5_unsplit_epi<MM>), grid, dim3(256), 0, st,                              \
                             qwp, xtp, pp, cntp, xsp, tokp, sclp, (float)cb, svp, bp, yp, mt, nt, ns_eff, tps);
  switch (M) {
    case 1: LAUNCH_K5U(1); break;
    case 2: LAUNCH_K5U(2); break;
    case 3: LAUNCH_K5U(3); break;
    case 4: LAUNCH_K5U(4); break;
    case 5: LAUNCH_K5U(5); break;
    case 6: LAUNCH_K5U(6); break;
    case 7: LAUNCH_K5U(7); break;
    case 8: LAUNCH_K5U(8); break;
  }
#undef LAUNCH_K5U
  return y;
}

PYBIND11_MODULE(ernie_k5_sp_prefill, m) {
  m.def("repack_k5_packed", &repack_k5_packed);
  m.def("dense_prefill_wmma_sp16", &dense_prefill_wmma_sp16);
  m.def("dense_prefill_wmma_unsplit", &dense_prefill_wmma_unsplit);
  m.def("gemv_k5_sp16_ns1", &gemv_k5_sp16_ns1);
  m.def("situ_and_mul_htr_prologue", &situ_and_mul_htr_prologue);
  m.def("rmsnorm_htr_prologue", &rmsnorm_htr_prologue);
  m.def("sigmoid_mul_htr_prologue", &sigmoid_mul_htr_prologue);
  m.def("gemv_k5_unsplit_epi", &gemv_k5_unsplit_epi);
}

