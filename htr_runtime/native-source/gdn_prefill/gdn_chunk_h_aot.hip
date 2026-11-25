// gdn_chunk_h_aot.hip v2 — High-occupancy AOT gfx1151 Wave32 WMMA replacement for FLA Stage 5.
// v2 optimizations:
//   - s_k stored transposed [128][72] in LDS -> 8x fewer LDS loads (ds_load_b128 vs 16x ds_load_u16)
//   - u and v_new loaded/stored via coalesced 128-bit u32x4 through LDS
//   - Controlled unrolling -> <=128 VGPRs, 0 spills, 8 waves/SIMD (2 WGs/CU!)
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

typedef __bf16 bf16_t;
typedef bf16_t bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ uint16_t f32_to_bf16_bits(float f) {
  uint32_t u = __float_as_uint(f);
  uint32_t lsb = (u >> 16) & 1u;
  return (uint16_t)((u + 0x7fffu + lsb) >> 16);
}
__device__ __forceinline__ float bf16_bits_to_f32(uint16_t b) {
  return __uint_as_float(((uint32_t)b) << 16);
}

extern "C" __global__ void __launch_bounds__(256, 2) gdn_chunk_h_bf16_k128_v128_bt64(
    const uint16_t* __restrict__ k_ptr,
    const uint16_t* __restrict__ u_ptr,
    const uint16_t* __restrict__ w_ptr,
    uint16_t* __restrict__ v_new_ptr,
    const float* __restrict__ g_ptr,
    uint16_t* __restrict__ h_ptr,
    const void* __restrict__ h0_ptr,
    float* __restrict__ ht_ptr,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ chunk_offsets,
    int T_total, int H, int Hg, int h0_is_bf16, int use_exp2) {

  const int i_v = blockIdx.x;
  const int i_nh = blockIdx.y;
  const int i_n = i_nh / H;
  const int i_h = i_nh - i_n * H;
  const int i_hk = i_h / (H / Hg);
  const int v_base = i_v * 32;

  int bos, eos, NT, boh;
  if (cu_seqlens != nullptr) {
    bos = cu_seqlens[i_n];
    eos = cu_seqlens[i_n + 1];
    int seq_len = eos - bos;
    NT = (seq_len + 63) >> 6;
    boh = chunk_offsets[i_n];
  } else {
    bos = i_n * T_total;
    eos = bos + T_total;
    NT = (T_total + 63) >> 6;
    boh = i_n * NT;
  }
  const int seq_len = eos - bos;

  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int col16 = lane & 15;
  const int half = lane >> 4;

  const int wv = wave >> 2;   // 0..1
  const int wk = wave & 3;    // 0..3
  const int wt = wave >> 1;   // 0..3
  const int wv2 = wave & 1;   // 0..1

  // LDS = 8704 + 18432 + 4608 + 256 = 32,000 B (< 32 KB -> 2 WGs/CU!)
  __shared__ uint16_t s_h[32 * 136];       // 8,704 B
  __shared__ uint16_t s_wk[128 * 72];      // 18,432 B (holds s_w[64][136] in Phase A, s_kT[128][72] in Phase B)
  __shared__ uint16_t s_uv[32 * 72];       // 4,608 B (holds s_u[64][32] in Phase A, s_vT[32][72] in Phase B)
  __shared__ float s_g[64];                // 256 B

  f32x8 h_acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
  f32x8 h_acc1 = {0, 0, 0, 0, 0, 0, 0, 0};

  if (h0_ptr != nullptr) {
    const int64_t h0_base = (int64_t)i_nh * (128 * 128) + (int64_t)v_base * 128;
    const int r_base = wv * 16 + half;
    const int c0 = wk * 32 + col16;
    const int c1 = c0 + 16;
    if (h0_is_bf16) {
      const uint16_t* p0 = (const uint16_t*)h0_ptr + h0_base;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int r = r_base + 2 * i;
        h_acc0[i] = bf16_bits_to_f32(p0[r * 128 + c0]);
        h_acc1[i] = bf16_bits_to_f32(p0[r * 128 + c1]);
      }
    } else {
      const float* p0 = (const float*)h0_ptr + h0_base;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int r = r_base + 2 * i;
        h_acc0[i] = p0[r * 128 + c0];
        h_acc1[i] = p0[r * 128 + c1];
      }
    }
  }

  const int64_t stride_k = (int64_t)Hg * 128;
  const int64_t stride_w = (int64_t)H * 128;
  const int64_t stride_v = (int64_t)H * 128;
  const int64_t stride_h = (int64_t)H * (128 * 128);

  const uint16_t* k_base = k_ptr + ((int64_t)bos * Hg + i_hk) * 128;
  const uint16_t* w_base = w_ptr + ((int64_t)bos * H + i_h) * 128;
  const uint16_t* u_base = u_ptr + ((int64_t)bos * H + i_h) * 128 + v_base;
  uint16_t* vnew_base = (v_new_ptr != nullptr) ? (v_new_ptr + ((int64_t)bos * H + i_h) * 128 + v_base) : nullptr;
  const float* g_base = (g_ptr != nullptr) ? (g_ptr + (int64_t)bos * H + i_h) : nullptr;
  uint16_t* h_base = h_ptr + ((int64_t)boh * H + i_h) * (128 * 128) + (int64_t)v_base * 128;

  for (int i_t = 0; i_t < NT; ++i_t) {
    const int t0 = i_t * 64;
    const int valid_t = (t0 + 64 <= seq_len) ? 64 : (seq_len - t0);

    // 1. Write h_acc -> s_h [32][136]
    {
      const int r_base = wv * 16 + half;
      const int c0 = wk * 32 + col16;
      const int c1 = c0 + 16;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int r = r_base + 2 * i;
        s_h[r * 136 + c0] = f32_to_bf16_bits(h_acc0[i]);
        s_h[r * 136 + c1] = f32_to_bf16_bits(h_acc1[i]);
      }
    }

    // 2. Coalesced load w[64, 128] -> s_wk as [64][136]
    {
      const int r_lane = tid >> 4;
      const int c_vec = (tid & 15) * 8;
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        int r = p * 16 + r_lane;
        u32x4 val = {0, 0, 0, 0};
        if (r < valid_t) {
          val = *(const u32x4*)(w_base + (int64_t)(t0 + r) * stride_w + c_vec);
        }
        *(u32x4*)(&s_wk[r * 136 + c_vec]) = val;
      }
    }

    // 3. Coalesced load u[64, 32] -> s_uv as [64][32] (1 pass: 4 threads/row x 64 rows = 256 threads!)
    {
      const int ur = tid >> 2;        // 0..63
      const int uc = (tid & 3) * 8;   // 0, 8, 16, 24
      u32x4 uval = {0, 0, 0, 0};
      if (ur < valid_t) {
        uval = *(const u32x4*)(u_base + (int64_t)(t0 + ur) * stride_v + uc);
      }
      *(u32x4*)(&s_uv[ur * 32 + uc]) = uval;
    }

    if (g_base != nullptr && tid < 64) {
      s_g[tid] = (tid < valid_t) ? g_base[(int64_t)(t0 + tid) * H] : 0.0f;
    }

    __syncthreads();

    // 4. Coalesced store s_h [32][128] -> global h[i_t]
    {
      uint16_t* h_dst = h_base + (int64_t)i_t * stride_h;
      const int r_lane = tid >> 4;
      const int c_vec = (tid & 15) * 8;
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        int r = p * 16 + r_lane;
        __builtin_nontemporal_store(*(const u32x4*)(&s_h[r * 136 + c_vec]), (u32x4*)(h_dst + r * 128 + c_vec));
      }
    }

    // 5. GEMM 1: v_acc [64, 32] = s_w [64, 128] @ s_h^T [128, 32]
    f32x8 v_acc = {0, 0, 0, 0, 0, 0, 0, 0};
    {
      const int a_row = wt * 16 + col16;
      const int b_row = wv2 * 16 + col16;
#pragma unroll 2
      for (int kk = 0; kk < 128; kk += 16) {
        u32x4 a0 = *(const u32x4*)(&s_wk[a_row * 136 + kk]);
        u32x4 a1 = *(const u32x4*)(&s_wk[a_row * 136 + kk + 8]);
        u32x4 b0 = *(const u32x4*)(&s_h[b_row * 136 + kk]);
        u32x4 b1 = *(const u32x4*)(&s_h[b_row * 136 + kk + 8]);
        bf16x16 af, bf;
        uint32_t* ap = (uint32_t*)&af;
        uint32_t* bp = (uint32_t*)&bf;
        ap[0] = a0[0]; ap[1] = a0[1]; ap[2] = a0[2]; ap[3] = a0[3];
        ap[4] = a1[0]; ap[5] = a1[1]; ap[6] = a1[2]; ap[7] = a1[3];
        bp[0] = b0[0]; bp[1] = b0[1]; bp[2] = b0[2]; bp[3] = b0[3];
        bp[4] = b1[0]; bp[5] = b1[1]; bp[6] = b1[2]; bp[7] = b1[3];
        v_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af, bf, v_acc);
      }
    }

    // 6. Compute b_v = u - v_acc (reading u from s_uv!), write v_new to s_h[0..2047] for coalesced store,
    //    and keep decayed b_v in registers
    float g_last = (g_base != nullptr) ? s_g[valid_t - 1] : 0.0f;
    uint16_t bv_raw[8];
    uint16_t bv_dec[8];
    {
      const int r_base = wt * 16 + half;
      const int c_v = wv2 * 16 + col16;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int r = r_base + 2 * i;
        if (r < valid_t) {
          float u_val = bf16_bits_to_f32(s_uv[r * 32 + c_v]);
          float diff = u_val - v_acc[i];
          bv_raw[i] = f32_to_bf16_bits(diff);
          if (g_base != nullptr) {
            float dg = g_last - s_g[r];
            diff *= (use_exp2 ? exp2f(dg) : __expf(dg));
          }
          bv_dec[i] = f32_to_bf16_bits(diff);
        } else {
          bv_raw[i] = 0;
          bv_dec[i] = 0;
        }
      }
    }

    if (g_base != nullptr) {
      float sc_last = use_exp2 ? exp2f(g_last) : __expf(g_last);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        h_acc0[i] *= sc_last;
        h_acc1[i] *= sc_last;
      }
    }

    __syncthreads();  // Phase A complete!

    // 7. Write bv_raw -> s_h[64][32] for coalesced store of v_new, AND bv_dec -> s_uv[32][72] (transposed!)
    {
      const int r_base = wt * 16 + half;
      const int c_v = wv2 * 16 + col16;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int t = r_base + 2 * i;
        if (vnew_base != nullptr) s_h[t * 32 + c_v] = bv_raw[i];
        s_uv[c_v * 72 + t] = bv_dec[i];
      }
    }

    // 8. Coalesced load k[64, 128] -> store TRANSPOSED into s_wk[128][72] (row = k, col = t)!
    {
      const int r_lane = tid >> 4;
      const int c_vec = (tid & 15) * 8;
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        int t = p * 16 + r_lane;
        u32x4 val = {0, 0, 0, 0};
        if (t < valid_t) {
          val = *(const u32x4*)(k_base + (int64_t)(t0 + t) * stride_k + c_vec);
        }
        const uint16_t* vp = (const uint16_t*)&val;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          s_wk[(c_vec + j) * 72 + t] = vp[j];
        }
      }
    }

    __syncthreads();

    // 9. Coalesced store v_new [64, 32] from s_h -> global v_new (1 pass!)
    if (vnew_base != nullptr) {
      const int ur = tid >> 2;
      const int uc = (tid & 3) * 8;
      if (ur < valid_t) {
        __builtin_nontemporal_store(*(const u32x4*)(&s_h[ur * 32 + uc]), (u32x4*)(vnew_base + (int64_t)(t0 + ur) * stride_v + uc));
      }
    }

    // 10. GEMM 2: h_acc [32, 128] += s_vT [32, 64] @ s_kT^T [64, 128]
    // BOTH A and B fragments are now 100% contiguous 128-bit u32x4 LDS loads!
    {
      const int a_row = wv * 16 + col16;
      const int b_row0 = wk * 32 + col16;
      const int b_row1 = b_row0 + 16;
#pragma unroll 2
      for (int kt = 0; kt < 64; kt += 16) {
        u32x4 a0 = *(const u32x4*)(&s_uv[a_row * 72 + kt]);
        u32x4 a1 = *(const u32x4*)(&s_uv[a_row * 72 + kt + 8]);
        u32x4 b00 = *(const u32x4*)(&s_wk[b_row0 * 72 + kt]);
        u32x4 b01 = *(const u32x4*)(&s_wk[b_row0 * 72 + kt + 8]);
        u32x4 b10 = *(const u32x4*)(&s_wk[b_row1 * 72 + kt]);
        u32x4 b11 = *(const u32x4*)(&s_wk[b_row1 * 72 + kt + 8]);
        bf16x16 af, bf0, bf1;
        uint32_t* ap = (uint32_t*)&af;
        uint32_t* bp0 = (uint32_t*)&bf0;
        uint32_t* bp1 = (uint32_t*)&bf1;
        ap[0] = a0[0]; ap[1] = a0[1]; ap[2] = a0[2]; ap[3] = a0[3];
        ap[4] = a1[0]; ap[5] = a1[1]; ap[6] = a1[2]; ap[7] = a1[3];
        bp0[0] = b00[0]; bp0[1] = b00[1]; bp0[2] = b00[2]; bp0[3] = b00[3];
        bp0[4] = b01[0]; bp0[5] = b01[1]; bp0[6] = b01[2]; bp0[7] = b01[3];
        bp1[0] = b10[0]; bp1[1] = b10[1]; bp1[2] = b10[2]; bp1[3] = b10[3];
        bp1[4] = b11[0]; bp1[5] = b11[1]; bp1[6] = b11[2]; bp1[7] = b11[3];
        h_acc0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af, bf0, h_acc0);
        h_acc1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af, bf1, h_acc1);
      }
    }

    __syncthreads();
  }

  if (ht_ptr != nullptr) {
    float* ht_dst = ht_ptr + (int64_t)i_nh * (128 * 128) + (int64_t)v_base * 128;
    const int r_base = wv * 16 + half;
    const int c0 = wk * 32 + col16;
    const int c1 = c0 + 16;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int r = r_base + 2 * i;
      ht_dst[r * 128 + c0] = h_acc0[i];
      ht_dst[r * 128 + c1] = h_acc1[i];
    }
  }
}
