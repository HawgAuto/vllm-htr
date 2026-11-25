// gdn_chunk_o_aot.hip — AOT gfx1151 Wave32 WMMA+LDS replacement for FLA Triton chunk_fwd_kernel_o.
// BT=64, K=128, V=128, supports GQA (H=48, Hg=16 -> hk = h / gqa_ratio), head-fastest grid (H, num_chunks).
// Computes O = ( (Q @ H^T) * exp(g) + tril((Q @ K^T) * exp(g_i - g_j)) @ V_new ) * scale.
// LDS = 26,368 B (< 32 KB -> 2 WGs/CU, 8 waves/SIMD), coalesced u32x4 R/W, zero spills, zero JIT!
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));

#define BT 64
#define D 128
#define HPITCH 72    // 64 + 8 BF16 padding (bank-conflict-free)

__device__ __forceinline__ float bf16_to_f32(uint16_t u) { return __uint_as_float((uint32_t)u << 16); }
__device__ __forceinline__ uint16_t f32_to_bf16_rne(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ bf16x16 as_bf16x16(u32x8 v) { return __builtin_bit_cast(bf16x16, v); }
__device__ __forceinline__ u32x4 nt_load_u32x4(const void* p) { return __builtin_nontemporal_load((const u32x4*)p); }
__device__ __forceinline__ u32x8 load16_bf16(const uint16_t* p) {
  u32x4 a = *(const u32x4*)p;
  u32x4 b = *(const u32x4*)(p + 8);
  return (u32x8){a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
}

// Grid: (H, num_chunks, 1), Block: (256, 1, 1) -> 8 Wave32 waves
extern "C" __global__ void __launch_bounds__(256, 2) gdn_chunk_o_bf16(
    const uint16_t* __restrict__ Q_ptr,      // [T, Hg, 128] BF16
    const uint16_t* __restrict__ K_ptr,      // [T, Hg, 128] BF16
    const uint16_t* __restrict__ V_ptr,      // [T, H, 128] BF16 (v_new)
    const uint16_t* __restrict__ H_ptr,      // [num_chunks, H, 128, 128] BF16 (h[i_tg, h, v, k] or [k, v] — FLA h is [V, K]!)
    const float* __restrict__ G_ptr,         // [T, H] FP32 (g_cumsum)
    uint16_t* __restrict__ O_out,            // [T, H, 128] BF16
    const int32_t* __restrict__ chunk_idx,   // [num_chunks, 2]
    const int32_t* __restrict__ cu_seqlens,  // [num_seqs + 1]
    int64_t sq_t, int64_t sq_h,
    int64_t sk_t, int64_t sk_h,
    int64_t sv_t, int64_t sv_h,
    int64_t sh_c, int64_t sh_h, int64_t sh_v,  // H strides: (chunk, head, v_row, k_col=1)
    int64_t sg_t, int64_t sg_h,
    int64_t so_t, int64_t so_h,
    float scale, int H, int gqa_ratio) {
  // LDS: s_A (9,216 B) + s_B0 (9,216 B) + s_B1 (9,216 B) + s_g (256 B) = 27,904 B (< 32 KB -> 2 WGs/CU!)
  __shared__ __attribute__((aligned(16))) uint16_t s_A[BT * HPITCH];   // 64x64 BF16 causal QK^T * exp(g_i-g_j)
  __shared__ __attribute__((aligned(16))) uint16_t s_B0[BT * HPITCH];  // 64x64 BF16 (Q half / output stage)
  __shared__ __attribute__((aligned(16))) uint16_t s_B1[BT * HPITCH];  // 64x64 BF16 (K half / H tile / V_T)
  __shared__ float s_g[BT];                                            // 256 B

  const int h = blockIdx.x;
  const int hk = h / gqa_ratio;
  const int cid = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int lr = lane & 15;
  const int hb = lane >> 4;

  const int seq_id = chunk_idx[cid * 2];
  const int lchunk = chunk_idx[cid * 2 + 1];
  const int bos = cu_seqlens[seq_id];
  const int eos = cu_seqlens[seq_id + 1];
  const int t0 = bos + lchunk * BT;
  const int clen = (eos - t0) < BT ? (eos - t0) : BT;

  if (tid < 64) {
    s_g[tid] = (tid < clen) ? G_ptr[(int64_t)(t0 + tid) * sg_t + (int64_t)h * sg_h] : 0.0f;
  }

  // ---- STEP 1: Compute causal A[64, 64] = tril(Q @ K^T) * exp(g_i - g_j) in 2 halves of K=128 ----
  constexpr int tile_m[10] = {0, 1, 1, 2, 2, 2, 3, 3, 3, 3};
  constexpr int tile_n[10] = {0, 0, 1, 0, 1, 2, 0, 1, 2, 3};
  f32x8 qk0 = {0, 0, 0, 0, 0, 0, 0, 0};
  f32x8 qk1 = {0, 0, 0, 0, 0, 0, 0, 0};

  // Also accumulate O_state[64, 128] = Q @ H^T in 2 halves of V=128?
  // Better: finish A[64, 64] -> s_A first, then loop vh = 0..1 (64 V-cols each) to compute BOTH
  // (Q @ H[vh]^T) * exp(g) + s_A @ V[vh] in registers and store O[vh] ONCE with coalesced u32x4!
#pragma unroll
  for (int kh = 0; kh < 2; ++kh) {
#pragma unroll
    for (int it = tid; it < BT * 8; it += 256) {
      const int r = it >> 3, c8 = it & 7;
      u32x4 qv = (r < clen) ? *(const u32x4*)(Q_ptr + (int64_t)(t0 + r) * sq_t + (int64_t)hk * sq_h + kh * 64 + c8 * 8) : (u32x4){0, 0, 0, 0};
      u32x4 kv = (r < clen) ? *(const u32x4*)(K_ptr + (int64_t)(t0 + r) * sk_t + (int64_t)hk * sk_h + kh * 64 + c8 * 8) : (u32x4){0, 0, 0, 0};
      *(u32x4*)(s_B0 + r * HPITCH + c8 * 8) = qv;
      *(u32x4*)(s_B1 + r * HPITCH + c8 * 8) = kv;
    }
    __syncthreads();
    {
      const int bm = tile_m[wave], bn = tile_n[wave];
      const uint16_t* rQ = s_B0 + (bm * 16 + lr) * HPITCH;
      const uint16_t* rK = s_B1 + (bn * 16 + lr) * HPITCH;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        qk0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(as_bf16x16(load16_bf16(rQ + k * 16)),
                                                          as_bf16x16(load16_bf16(rK + k * 16)), qk0);
      }
    }
    if (wave < 2) {
      const int bm = tile_m[wave + 8], bn = tile_n[wave + 8];
      const uint16_t* rQ = s_B0 + (bm * 16 + lr) * HPITCH;
      const uint16_t* rK = s_B1 + (bn * 16 + lr) * HPITCH;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        qk1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(as_bf16x16(load16_bf16(rQ + k * 16)),
                                                          as_bf16x16(load16_bf16(rK + k * 16)), qk1);
      }
    }
    __syncthreads();
  }

  // Zero s_A then write causal lower-tri (i >= j) * exp(g_i - g_j) into s_A
#pragma unroll
  for (int it = tid; it < BT * 8; it += 256) {
    const int r = it >> 3, c8 = it & 7;
    *(u32x4*)(s_A + r * HPITCH + c8 * 8) = (u32x4){0, 0, 0, 0};
  }
  __syncthreads();
  {
    const int bm = tile_m[wave], bn = tile_n[wave];
    const int j = bn * 16 + lr;
    const float gj = s_g[j];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int i = bm * 16 + 2 * e + hb;
      if (i >= j && i < clen) {
        s_A[i * HPITCH + j] = f32_to_bf16_rne(qk0[e] * __expf(s_g[i] - gj));
      }
    }
  }
  if (wave < 2) {
    const int bm = tile_m[wave + 8], bn = tile_n[wave + 8];
    const int j = bn * 16 + lr;
    const float gj = s_g[j];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int i = bm * 16 + 2 * e + hb;
      if (i >= j && i < clen) {
        s_A[i * HPITCH + j] = f32_to_bf16_rne(qk1[e] * __expf(s_g[i] - gj));
      }
    }
  }
  __syncthreads();

  // ---- STEP 2: Loop vh = 0..1 (64 V-cols each) -> O_state + A @ V_new ----
  // 4 row-blocks x 4 col-blocks = 16 tiles across 8 waves (2 tiles/wave: bm = (wave>>2) + bi*2, bn = wave&3)
  const uint16_t* h_base = H_ptr + (int64_t)cid * sh_c + (int64_t)h * sh_h;
  const int bn = wave & 3;
  const int col = bn * 16 + lr;

#pragma unroll
  for (int vh = 0; vh < 2; ++vh) {
    f32x8 acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    f32x8 acc1 = {0, 0, 0, 0, 0, 0, 0, 0};

    // 2a: Q @ H[vh*64..vh*64+63, 0..127]^T in 2 halves of K=128
#pragma unroll
    for (int kh = 0; kh < 2; ++kh) {
#pragma unroll
      for (int it = tid; it < BT * 8; it += 256) {
        const int r = it >> 3, c8 = it & 7;
        u32x4 qv = (r < clen) ? *(const u32x4*)(Q_ptr + (int64_t)(t0 + r) * sq_t + (int64_t)hk * sq_h + kh * 64 + c8 * 8) : (u32x4){0, 0, 0, 0};
        u32x4 hv = nt_load_u32x4(h_base + (int64_t)(vh * 64 + r) * sh_v + kh * 64 + c8 * 8);
        *(u32x4*)(s_B0 + r * HPITCH + c8 * 8) = qv;
        *(u32x4*)(s_B1 + r * HPITCH + c8 * 8) = hv;
      }
      __syncthreads();

      const uint16_t* rH = s_B1 + col * HPITCH;
      const int bm0 = (wave >> 2);
      const int bm1 = bm0 + 2;
      const uint16_t* rQ0 = s_B0 + (bm0 * 16 + lr) * HPITCH;
      const uint16_t* rQ1 = s_B0 + (bm1 * 16 + lr) * HPITCH;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        bf16x16 hfk = as_bf16x16(load16_bf16(rH + k * 16));
        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(as_bf16x16(load16_bf16(rQ0 + k * 16)), hfk, acc0);
        acc1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(as_bf16x16(load16_bf16(rQ1 + k * 16)), hfk, acc1);
      }
      __syncthreads();
    }

    // Scale acc0, acc1 by exp(g_i)
    const int bm0 = (wave >> 2);
    const int bm1 = bm0 + 2;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      acc0[e] *= __expf(s_g[bm0 * 16 + 2 * e + hb]);
      acc1[e] *= __expf(s_g[bm1 * 16 + 2 * e + hb]);
    }

    // 2b: Load V_new[64, vh*64..vh*64+63] -> s_B0, transpose -> s_B1[64, 72]
#pragma unroll
    for (int it = tid; it < BT * 8; it += 256) {
      const int r = it >> 3, c8 = it & 7;
      u32x4 vv = (r < clen) ? *(const u32x4*)(V_ptr + (int64_t)(t0 + r) * sv_t + (int64_t)h * sv_h + vh * 64 + c8 * 8) : (u32x4){0, 0, 0, 0};
      *(u32x4*)(s_B0 + r * HPITCH + c8 * 8) = vv;
    }
    __syncthreads();
#pragma unroll
    for (int it = tid; it < BT * 64; it += 256) {
      const int k = it >> 6, n = it & 63;
      s_B1[n * HPITCH + k] = s_B0[k * HPITCH + n];
    }
    __syncthreads();

    // WMMA: acc += s_A @ s_B1^T (only k <= bm!)
    const uint16_t* rVt = s_B1 + col * HPITCH;
    const uint16_t* rA0 = s_A + (bm0 * 16 + lr) * HPITCH;
    const uint16_t* rA1 = s_A + (bm1 * 16 + lr) * HPITCH;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      bf16x16 vfk = as_bf16x16(load16_bf16(rVt + k * 16));
      if (k <= bm0) acc0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(as_bf16x16(load16_bf16(rA0 + k * 16)), vfk, acc0);
      if (k <= bm1) acc1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(as_bf16x16(load16_bf16(rA1 + k * 16)), vfk, acc1);
    }
    __syncthreads();

    // Stage output * scale -> s_B0 -> coalesced u32x4 global stores!
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      s_B0[(bm0 * 16 + 2 * e + hb) * HPITCH + col] = f32_to_bf16_rne(acc0[e] * scale);
      s_B0[(bm1 * 16 + 2 * e + hb) * HPITCH + col] = f32_to_bf16_rne(acc1[e] * scale);
    }
    __syncthreads();
#pragma unroll
    for (int it = tid; it < BT * 8; it += 256) {
      const int r = it >> 3, c8 = it & 7;
      if (r < clen) {
        *(u32x4*)(O_out + (int64_t)(t0 + r) * so_t + (int64_t)h * so_h + vh * 64 + c8 * 8) =
            *(const u32x4*)(s_B0 + r * HPITCH + c8 * 8);
      }
    }
    __syncthreads();
  }
}
