// gdn_wy_fused_aot.hip v2 — High-occupancy (<32KB LDS = 2 WGs/CU), coalesced u32x4 R/W,
// head-fastest grid (H, num_chunks), Wave32 WMMA fusion of FLA GDN Stages 1-4!
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
#define APITCH 65    // 64 + 1 FP32 padding (16,640 B!)

__device__ __forceinline__ float bf16_to_f32(uint16_t u) { return __uint_as_float((uint32_t)u << 16); }
__device__ __forceinline__ uint16_t f32_to_bf16_rne(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ bf16x16 as_bf16x16(u32x8 v) { return __builtin_bit_cast(bf16x16, v); }
__device__ __forceinline__ u32x8 load16_bf16(const uint16_t* p) {
  u32x4 a = *(const u32x4*)p;
  u32x4 b = *(const u32x4*)(p + 8);
  return (u32x8){a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
}

// Grid: (H, num_chunks, 1) [head-fastest for L2/MALL locality!], Block: (256, 1, 1) -> 8 waves
extern "C" __global__ void __launch_bounds__(256, 2) gdn_wy_fused_bf16(
    const uint16_t* __restrict__ K_ptr,
    const uint16_t* __restrict__ V_ptr,
    const float* __restrict__ G_ptr,
    const float* __restrict__ BETA_ptr,
    float* __restrict__ GC_out,
    uint16_t* __restrict__ W_out,
    uint16_t* __restrict__ U_out,
    const int32_t* __restrict__ chunk_idx,
    const int32_t* __restrict__ cu_seqlens,
    int64_t sk_t, int64_t sk_h,
    int64_t sv_t, int64_t sv_h,
    int64_t sg_t, int64_t sg_h,
    int64_t sb_t, int64_t sb_h,
    int64_t sgc_t, int64_t sgc_h,
    int64_t sw_t, int64_t sw_h,
    int64_t su_t, int64_t su_h,
    int H, int gqa_ratio) {
  // Total LDS = 16,640 + 9,216 + 512 + 4,608 = 30,976 B (< 32 KB -> 2 WGs/CU, 16 waves/CU!)
  __shared__ __attribute__((aligned(16))) union {
    float A[BT * APITCH];                   // 16,640 B (Stages 2-3)
    uint16_t B_T[64 * HPITCH];              //  9,216 B (Stage 4: 64-col transposed tile!)
  } u_main;
  __shared__ __attribute__((aligned(16))) uint16_t s_tile[BT * HPITCH];  // 9,216 B (64x64 K/V/out staging!)
  __shared__ __attribute__((aligned(16))) uint16_t s_Ab[BT * HPITCH];    // 9,216 B (A_inv BF16!)
  __shared__ float s_g[BT];                                              //   256 B
  __shared__ float s_beta[BT];                                           //   256 B
  // Wait: 16,640 + 9,216 + 9,216 + 512 = 35,584 B -> let's union s_Ab with nothing?
  // Wait: in Stage 2 we only need s_A (16,640) + s_tile (9,216) = 25,856 B!
  // In Stage 4 we only need s_Ab (9,216) + s_tile (9,216) + B_T (9,216) = 27,648 B!
  // So u_main = union { float A[BT*APITCH]; struct { uint16_t B_T[64*HPITCH]; }; } ->
  // We STILL keep s_Ab separate (9,216) ONLY if <=32KB: wait 16640+9216 = 25856, + 9216 = 35072 (>32768)!
  // How to get <= 32,768 B? Store s_Ab in u_main.A's UPPER HALF? No, Stage 3 needs all of s_A.
  // Make s_tile [64][64] (8,192 B, swizzled!) and s_Ab [64][64] (8,192 B) -> 16,640 + 8,192 + 8,192 = 33,024 (256B over!).
  // EVEN BETTER: APITCH = 64 (16,384 B) and s_tile = 64*64 (8,192 B), s_Ab = 64*64 (8,192 B)?
  // 16384 + 8192 + 8192 = 32,768 B! Wait: put s_g and s_beta in registers or reuse!
  // NO — SIMPLER! After Stage 3 finishes, s_A[64*64] is 16,384 B, and s_Ab[64*64] BF16 is ONLY 8,192 B (HALF of s_A)!
  // So we can convert s_A (FP32, 16KB) -> s_Ab (BF16, 8KB) IN-PLACE into the FIRST 8 KB of u_main,
  // and put B_T (8KB) in the SECOND 8 KB of u_main!!
  // TOTAL LDS = 16,640 (u_main) + 9,216 (s_tile) + 512 (s_g, s_beta) = ONLY 26,368 BYTES (< 32 KB -> 2 WGs/CU GUARANTEED)!!

  // Let's use that exact 26.4 KB layout!
  float* s_A = u_main.A;
  uint16_t* s_Ab_ip = reinterpret_cast<uint16_t*>(u_main.A);             // first 8,192 B (64x64 BF16, stride 64)
  uint16_t* s_BT_ip = reinterpret_cast<uint16_t*>(u_main.A) + 64 * 64;   // next  8,192 B (64x64 BF16, stride 64)

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

  // ---- STAGE 1: chunk_local_cumsum + load s_beta ----
  if (tid < 64) {
    float gv = (tid < clen) ? G_ptr[(int64_t)(t0 + tid) * sg_t + (int64_t)h * sg_h] : 0.0f;
    float bv = (tid < clen) ? BETA_ptr[(int64_t)(t0 + tid) * sb_t + (int64_t)h * sb_h] : 0.0f;
    float x = gv;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      float y = __shfl_up(x, off, 32);
      if (lane >= off) x += y;
    }
    s_g[tid] = x;
    s_beta[tid] = bv;
  }
  __syncthreads();
  if (tid < 64) {
    float gcs = s_g[tid];
    if (tid >= 32) gcs += s_g[31];
    s_g[tid] = gcs;
    if (tid < clen) {
      GC_out[(int64_t)(t0 + tid) * sgc_t + (int64_t)h * sgc_h] = gcs;
    }
  }

  // ---- STAGE 2: K @ K^T in 2 halves of D=128 (d0 = 0, 64) using s_tile[64, 72] ----
  // Accumulate 10 lower-tri 16x16 tiles in registers across the 2 halves!
  constexpr int tile_m[10] = {0, 1, 1, 2, 2, 2, 3, 3, 3, 3};
  constexpr int tile_n[10] = {0, 0, 1, 0, 1, 2, 0, 1, 2, 3};
  f32x8 kkt0 = {0, 0, 0, 0, 0, 0, 0, 0};
  f32x8 kkt1 = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
  for (int dh = 0; dh < 2; ++dh) {
    // Load K[64, dh*64 .. dh*64+63] -> s_tile[64, 72] (512 u32x4 / 256 threads = 2 loads/thread!)
#pragma unroll
    for (int it = tid; it < BT * 8; it += 256) {
      const int r = it >> 3;
      const int c8 = it & 7;
      u32x4 kv = (r < clen)
          ? *(const u32x4*)(K_ptr + (int64_t)(t0 + r) * sk_t + (int64_t)hk * sk_h + dh * 64 + c8 * 8)
          : (u32x4){0, 0, 0, 0};
      *(u32x4*)(s_tile + r * HPITCH + c8 * 8) = kv;
    }
    __syncthreads();

    // Tile 0 (wave 0..7)
    {
      const int bm = tile_m[wave], bn = tile_n[wave];
      const uint16_t* rowA = s_tile + (bm * 16 + lr) * HPITCH;
      const uint16_t* rowB = s_tile + (bn * 16 + lr) * HPITCH;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        u32x8 af = load16_bf16(rowA + k * 16);
        u32x8 bf = load16_bf16(rowB + k * 16);
        kkt0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(as_bf16x16(af), as_bf16x16(bf), kkt0);
      }
    }
    // Tiles 8, 9 (waves 0, 1)
    if (wave < 2) {
      const int bm = tile_m[wave + 8], bn = tile_n[wave + 8];
      const uint16_t* rowA = s_tile + (bm * 16 + lr) * HPITCH;
      const uint16_t* rowB = s_tile + (bn * 16 + lr) * HPITCH;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        u32x8 af = load16_bf16(rowA + k * 16);
        u32x8 bf = load16_bf16(rowB + k * 16);
        kkt1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(as_bf16x16(af), as_bf16x16(bf), kkt1);
      }
    }
    __syncthreads();
  }

  // Store scaled lower-tri A -> s_A
  {
    const int bm = tile_m[wave], bn = tile_n[wave];
    const int j = bn * 16 + lr;
    const float gj = s_g[j];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int i = bm * 16 + 2 * e + hb;
      s_A[i * APITCH + j] = (i > j && i < clen) ? (kkt0[e] * s_beta[i] * __expf(s_g[i] - gj)) : 0.0f;
    }
  }
  if (wave < 2) {
    const int bm = tile_m[wave + 8], bn = tile_n[wave + 8];
    const int j = bn * 16 + lr;
    const float gj = s_g[j];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int i = bm * 16 + 2 * e + hb;
      s_A[i * APITCH + j] = (i > j && i < clen) ? (kkt1[e] * s_beta[i] * __expf(s_g[i] - gj)) : 0.0f;
    }
  }
  __syncthreads();

  // ---- STAGE 3: Exact In-LDS solve_tril: A_inv = (I + A)^-1 ----
  if (wave < 4 && lane < 16) {
    const int b0 = wave * 16;
    const int c = lane;
    s_A[(b0 + c) * APITCH + (b0 + c)] = 1.0f;
    for (int r = 1; r < 16; ++r) {
      if (c < r) {
        float sum = -s_A[(b0 + r) * APITCH + (b0 + c)];
        for (int k = c + 1; k < r; ++k) {
          sum = __builtin_fmaf(-s_A[(b0 + r) * APITCH + (b0 + k)], s_A[(b0 + k) * APITCH + (b0 + c)], sum);
        }
        s_A[(b0 + r) * APITCH + (b0 + c)] = sum;
      } else if (c > r) {
        s_A[(b0 + r) * APITCH + (b0 + c)] = 0.0f;
      }
    }
    if (c > 0) s_A[b0 * APITCH + (b0 + c)] = 0.0f;
  }
  __syncthreads();

  auto mm16 = [&](int dst_m, int dst_n, int a_m, int a_n, int b_m, int b_n, float scale) {
    const int c = lr;
    float out[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int r = 2 * e + hb;
      float s = 0.0f;
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        s = __builtin_fmaf(s_A[(a_m * 16 + r) * APITCH + (a_n * 16 + k)],
                           s_A[(b_m * 16 + k) * APITCH + (b_n * 16 + c)], s);
      }
      out[e] = s * scale;
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) s_A[(dst_m * 16 + 2 * e + hb) * APITCH + (dst_n * 16 + c)] = out[e];
  };
  auto mm16_2 = [&](int dst_m, int dst_n, int a1_m, int a1_n, int b1_m, int b1_n,
                    int a2_m, int a2_n, int b2_m, int b2_n, float scale) {
    const int c = lr;
    float out[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int r = 2 * e + hb;
      float s = 0.0f;
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        s = __builtin_fmaf(s_A[(a1_m * 16 + r) * APITCH + (a1_n * 16 + k)],
                           s_A[(b1_m * 16 + k) * APITCH + (b1_n * 16 + c)], s);
        s = __builtin_fmaf(s_A[(a2_m * 16 + r) * APITCH + (a2_n * 16 + k)],
                           s_A[(b2_m * 16 + k) * APITCH + (b2_n * 16 + c)], s);
      }
      out[e] = s * scale;
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) s_A[(dst_m * 16 + 2 * e + hb) * APITCH + (dst_n * 16 + c)] = out[e];
  };

  if (wave == 0) mm16(0, 1, 1, 1, 1, 0, -1.0f);
  if (wave == 1) mm16(0, 2, 3, 3, 3, 2, -1.0f);
  __syncthreads();
  if (wave == 0) mm16(1, 0, 0, 1, 0, 0, 1.0f);
  if (wave == 1) mm16(3, 2, 0, 2, 2, 2, 1.0f);
  __syncthreads();
  if (wave == 0) mm16(0, 2, 2, 2, 2, 0, -1.0f);
  if (wave == 1) mm16(0, 3, 2, 2, 2, 1, -1.0f);
  if (wave == 2) mm16_2(1, 2, 3, 2, 2, 0, 3, 3, 3, 0, -1.0f);
  if (wave == 3) mm16_2(1, 3, 3, 2, 2, 1, 3, 3, 3, 1, -1.0f);
  __syncthreads();
  if (wave == 0) mm16_2(2, 0, 0, 2, 0, 0, 0, 3, 1, 0, 1.0f);
  if (wave == 1) mm16(2, 1, 0, 3, 1, 1, 1.0f);
  if (wave == 2) mm16_2(3, 0, 1, 2, 0, 0, 1, 3, 1, 0, 1.0f);
  if (wave == 3) mm16(3, 1, 1, 3, 1, 1, 1.0f);
  __syncthreads();

  // Convert s_A (FP32) -> s_tile (temp BF16) -> s_Ab_ip (first 8KB of u_main)!
#pragma unroll
  for (int it = tid; it < BT * BT; it += 256) {
    const int r = it >> 6, c = it & 63;
    s_tile[it] = f32_to_bf16_rne((c <= r) ? s_A[r * APITCH + c] : 0.0f);
  }
  __syncthreads();
#pragma unroll
  for (int it = tid; it < (BT * BT) / 8; it += 256) {
    *(u32x4*)(s_Ab_ip + it * 8) = *(const u32x4*)(s_tile + it * 8);
  }
  __syncthreads();

  // Helper lambda for Stage 4: compute out = s_Ab_ip[64,64] @ (src * row_scale)[64, 128]
  // in 2 halves of 64 cols (dh=0, 1), staging outputs in s_tile for 100% coalesced u32x4 global stores!
  auto run_wy_matmul = [&](const uint16_t* src_ptr, int64_t ss_t, int64_t ss_h, int h_src,
                           uint16_t* dst_ptr, int64_t sd_t, int64_t sd_h, bool use_exp_g) {
#pragma unroll
    for (int dh = 0; dh < 2; ++dh) {
      // Load 64x64 tile -> s_tile[64, 72] (coalesced u32x4)
#pragma unroll
      for (int it = tid; it < BT * 8; it += 256) {
        const int r = it >> 3, c8 = it & 7;
        u32x4 v = (r < clen)
            ? *(const u32x4*)(src_ptr + (int64_t)(t0 + r) * ss_t + (int64_t)h_src * ss_h + dh * 64 + c8 * 8)
            : (u32x4){0, 0, 0, 0};
        *(u32x4*)(s_tile + r * HPITCH + c8 * 8) = v;
      }
      __syncthreads();

      // Scale & transpose s_tile[64, 64] -> s_BT_ip[64, 64]
#pragma unroll
      for (int it = tid; it < BT * 64; it += 256) {
        const int k = it >> 6, n = it & 63;
        float sc = (k < clen) ? (use_exp_g ? (s_beta[k] * __expf(s_g[k])) : s_beta[k]) : 0.0f;
        s_BT_ip[n * 64 + k] = f32_to_bf16_rne(bf16_to_f32(s_tile[k * HPITCH + n]) * sc);
      }
      __syncthreads();

      // 4 row-blocks x 4 col-blocks = 16 tiles across 8 waves (2 tiles/wave: bm = wave/4 + {0,2}, bn = wave%4)
      const int bn = wave & 3;
      const int col = bn * 16 + lr;
      const uint16_t* b_row = s_BT_ip + col * 64;
      u32x8 bf[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) bf[k] = load16_bf16(b_row + k * 16);
#pragma unroll
      for (int bi = 0; bi < 2; ++bi) {
        const int bm = (wave >> 2) + bi * 2;
        f32x8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
        const uint16_t* a_row = s_Ab_ip + (bm * 16 + lr) * 64;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          if (k <= bm) {
            u32x8 af = load16_bf16(a_row + k * 16);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(as_bf16x16(af), as_bf16x16(bf[k]), acc);
          }
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          s_tile[(bm * 16 + 2 * e + hb) * HPITCH + col] = f32_to_bf16_rne(acc[e]);
        }
      }
      __syncthreads();

      // 100% COALESCED 128-bit u32x4 global stores!
#pragma unroll
      for (int it = tid; it < BT * 8; it += 256) {
        const int r = it >> 3, c8 = it & 7;
        if (r < clen) {
          *(u32x4*)(dst_ptr + (int64_t)(t0 + r) * sd_t + (int64_t)h * sd_h + dh * 64 + c8 * 8) =
              *(const u32x4*)(s_tile + r * HPITCH + c8 * 8);
        }
      }
      __syncthreads();
    }
  };

  run_wy_matmul(K_ptr, sk_t, sk_h, hk, W_out, sw_t, sw_h, true);
  run_wy_matmul(V_ptr, sv_t, sv_h, h, U_out, su_t, su_h, false);
}
