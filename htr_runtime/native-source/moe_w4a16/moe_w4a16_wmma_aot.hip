// moe_w4a16_wmma_aot.hip — High-throughput double-buffered AOT gfx1151 Wave32 WMMA
// replacement for Triton fused_moe_kernel_gptq_awq (W4A16 symmetric BF16).
// Features:
//  - Coalesced 4-lane-per-row B loads (stride_bk == 1) & cached L2/MALL reuse across M-blocks
//  - Double-buffered s_B[2][64*40] LDS (1 __syncthreads per K=32, overlaps VMEM with WMMA)
//  - Wave-level early skip of all-padding M-tiles (__any(my_a_valid))
//  - Single-cycle v_perm_b32 packed BF16x2 RNE conversion
//  - Supports both int32 and int64 expert_ids via bit 2 of scale_dtype
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ uint16_t f32_to_bf16_rne(float f) {
  uint32_t u = __float_as_uint(f);
  if (f != f) return 0x7fffu;
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ uint32_t pack2_bf16_rne(float a, float b) {
  uint32_t ua = __float_as_uint(a);
  uint32_t ub = __float_as_uint(b);
  ua = (a != a) ? 0x7fff0000u : (ua + 0x7fffu + ((ua >> 16) & 1u));
  ub = (b != b) ? 0x7fff0000u : (ub + 0x7fffu + ((ub >> 16) & 1u));
  return __builtin_amdgcn_perm(ub, ua, 0x07060302u);
}
__device__ __forceinline__ float load_scale(const void* ptr, int64_t off, int flags) {
  const int sd = flags & 3;
  if (sd == 0) return ((const float*)ptr)[off];
  if (sd == 1) return __uint_as_float(((uint32_t)((const uint16_t*)ptr)[off]) << 16);
  return __half2float(((const __half*)ptr)[off]);
}
__device__ __forceinline__ int load_eid(const void* ptr, int idx, int flags) {
  return (flags & 4) ? (int)((const int64_t*)ptr)[idx] : ((const int32_t*)ptr)[idx];
}
__device__ __forceinline__ u32x8 load_a16(const uint16_t* ap, int64_t stride_ak) {
  if (stride_ak == 1) {
    u32x4 x0 = *(const u32x4*)ap;
    u32x4 x1 = *(const u32x4*)(ap + 8);
    return (u32x8){x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]};
  }
  uint16_t tmp[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) tmp[i] = ap[(int64_t)i * stride_ak];
  u32x8 r;
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = (uint32_t)tmp[2 * i] | ((uint32_t)tmp[2 * i + 1] << 16);
  return r;
}

__device__ __forceinline__ void dequant_u32_to_sB(uint32_t w, float sc, uint32_t* dst) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t p = (w >> (i * 8)) & 0xffu;
    float f0 = (float)((int)(p & 0x0fu) - 8) * sc;
    float f1 = (float)((int)(p >> 4) - 8) * sc;
    dst[i] = pack2_bf16_rne(f0, f1);
  }
}

// Fast 128-thread (4 waves) kernel for BM in {16, 32, 64}, BN = 64, BK = 32.
// Uses double-buffered s_B[2][64*40] (10,240 B LDS -> 12 waves/SIMD).
extern "C" __global__ __launch_bounds__(128) void moe_w4a16_sym_bf16_bn64(
    const uint16_t* __restrict__ A_ptr,
    const uint8_t* __restrict__ B_ptr,
    uint16_t* __restrict__ C_ptr,
    const void* __restrict__ B_scale_ptr,
    const float* __restrict__ topk_weights_ptr,
    const int32_t* __restrict__ sorted_token_ids_ptr,
    const void* __restrict__ expert_ids_ptr,
    const int32_t* __restrict__ num_tokens_post_padded_ptr,
    int N, int K, int num_valid_tokens, int top_k, int group_size,
    int64_t stride_am, int64_t stride_ak,
    int64_t stride_be, int64_t stride_bk, int64_t stride_bn,
    int64_t stride_cm, int64_t stride_cn,
    int64_t stride_bse, int64_t stride_bsk, int64_t stride_bsn,
    int BM, int mul_routed_weight, int scale_dtype) {

  const int pid_n = blockIdx.x;
  const int pid_m = blockIdx.y;
  const int ntpp = num_tokens_post_padded_ptr[0];
  if (pid_m * BM >= ntpp) return;

  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int half = lane >> 4;
  const int n_base = pid_n * 64;
  const int off_e = load_eid(expert_ids_ptr, pid_m, scale_dtype);

  __shared__ int32_t s_tok[64];
  if (tid < BM) s_tok[tid] = sorted_token_ids_ptr[pid_m * BM + tid];
  __syncthreads();

  if (off_e == -1) {
    if (wave * 16 < BM) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int m_loc = wave * 16 + 2 * i + half;
        if (m_loc < BM) {
          int tok = s_tok[m_loc];
          if (tok < num_valid_tokens) {
            uint16_t* cr = C_ptr + (int64_t)tok * stride_cm;
#pragma unroll
            for (int j = 0; j < 4; ++j) cr[(int64_t)(n_base + 16 * j + l16) * stride_cn] = 0;
          }
        }
      }
    }
    return;
  }

  const int my_m = wave * 16 + l16;
  const int my_tok = (my_m < BM) ? s_tok[my_m] : num_valid_tokens;
  const bool my_a_valid = (my_tok < num_valid_tokens);
  const bool wave_active = (wave * 16 < BM) && (__any(my_a_valid) != 0);
  const uint16_t* a_row = my_a_valid ? (A_ptr + (int64_t)(my_tok / top_k) * stride_am) : nullptr;

  const uint8_t* b_e = B_ptr + (int64_t)off_e * stride_be;
  const int64_t bs_e_off = (int64_t)off_e * stride_bse;

  __shared__ __attribute__((aligned(16))) uint16_t s_B[2][64 * 40];

  // Coalesced B-loader geometry:
  // If stride_bk == 1 (K-contiguous): 4 consecutive lanes read 16 contiguous bytes of row b_n0 and b_n0+32!
  // If stride_bk != 1 (N-contiguous): 64 lanes read contiguous N rows (b_n = tid & 63, b_half = tid >> 6).
  const bool k_contig = (stride_bk == 1);
  const int b_q = tid & 3;         // 0..3 (4 bytes = 8 nibbles = 8 bf16)
  const int b_n0 = tid >> 2;       // 0..31
  const int bn_nc = tid & 63;      // 0..63
  const int bh_nc = tid >> 6;      // 0..1

  const int n_g0 = n_base + b_n0;
  const int n_g1 = n_base + b_n0 + 32;
  const int n_gnc = n_base + bn_nc;

  int last_kg = -1;
  float sc0 = 0.0f, sc1 = 0.0f, sc_nc = 0.0f;

  auto fetch_b_raw = [&](int k0, uint32_t& rw0, uint32_t& rw1) {
    const int kg = k0 / group_size;
    if (kg != last_kg) {
      last_kg = kg;
      if (k_contig) {
        sc0 = load_scale(B_scale_ptr, bs_e_off + (int64_t)n_g0 * stride_bsn + (int64_t)kg * stride_bsk, scale_dtype);
        sc1 = load_scale(B_scale_ptr, bs_e_off + (int64_t)n_g1 * stride_bsn + (int64_t)kg * stride_bsk, scale_dtype);
      } else {
        sc_nc = load_scale(B_scale_ptr, bs_e_off + (int64_t)n_gnc * stride_bsn + (int64_t)kg * stride_bsk, scale_dtype);
      }
    }
    if (k_contig) {
      const int kb = (k0 >> 1) + b_q * 4;
      rw0 = *(const uint32_t*)(b_e + (int64_t)n_g0 * stride_bn + kb);
      rw1 = *(const uint32_t*)(b_e + (int64_t)n_g1 * stride_bn + kb);
    } else {
      const int kb = (k0 >> 1) + bh_nc * 8;
      uint32_t p0 = 0, p1 = 0;
#pragma unroll
      for (int bi = 0; bi < 4; ++bi) {
        p0 |= ((uint32_t)b_e[(int64_t)(kb + bi) * stride_bk + (int64_t)n_gnc * stride_bn]) << (bi * 8);
        p1 |= ((uint32_t)b_e[(int64_t)(kb + 4 + bi) * stride_bk + (int64_t)n_gnc * stride_bn]) << (bi * 8);
      }
      rw0 = p0; rw1 = p1;
    }
  };

  auto store_b_lds = [&](int buf, uint32_t rw0, uint32_t rw1) {
    if (k_contig) {
      dequant_u32_to_sB(rw0, sc0, (uint32_t*)(s_B[buf] + b_n0 * 40 + b_q * 8));
      dequant_u32_to_sB(rw1, sc1, (uint32_t*)(s_B[buf] + (b_n0 + 32) * 40 + b_q * 8));
    } else {
      uint32_t* dst = (uint32_t*)(s_B[buf] + bn_nc * 40 + bh_nc * 16);
      dequant_u32_to_sB(rw0, sc_nc, dst);
      dequant_u32_to_sB(rw1, sc_nc, dst + 4);
    }
  };

  f32x8 acc[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) acc[j] = {0, 0, 0, 0, 0, 0, 0, 0};

  // Prologue: load k0 = 0 into s_B[0]
  uint32_t cur_w0 = 0, cur_w1 = 0;
  fetch_b_raw(0, cur_w0, cur_w1);
  store_b_lds(0, cur_w0, cur_w1);
  __syncthreads();

  int buf = 0;
  for (int k0 = 0; k0 < K; k0 += 32) {
    const int next_k0 = k0 + 32;
    const int next_buf = buf ^ 1;

    // Issue next B global loads BEFORE WMMA so VMEM overlaps with WMMA compute!
    uint32_t nxt_w0 = 0, nxt_w1 = 0;
    if (next_k0 < K) {
      fetch_b_raw(next_k0, nxt_w0, nxt_w1);
    }

    if (wave_active) {
      u32x8 av0 = my_a_valid ? load_a16(a_row + (int64_t)k0 * stride_ak, stride_ak)
                             : (u32x8){0, 0, 0, 0, 0, 0, 0, 0};
      u32x8 av1 = my_a_valid ? load_a16(a_row + (int64_t)(k0 + 16) * stride_ak, stride_ak)
                             : (u32x8){0, 0, 0, 0, 0, 0, 0, 0};
      bf16x16 af0 = __builtin_bit_cast(bf16x16, av0);
      bf16x16 af1 = __builtin_bit_cast(bf16x16, av1);

#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint16_t* bp = &s_B[buf][(16 * j + l16) * 40];
        u32x4 y0 = *(const u32x4*)bp;
        u32x4 y1 = *(const u32x4*)(bp + 8);
        u32x4 y2 = *(const u32x4*)(bp + 16);
        u32x4 y3 = *(const u32x4*)(bp + 24);
        bf16x16 bf0 = __builtin_bit_cast(bf16x16, (u32x8){y0[0], y0[1], y0[2], y0[3], y1[0], y1[1], y1[2], y1[3]});
        bf16x16 bf1 = __builtin_bit_cast(bf16x16, (u32x8){y2[0], y2[1], y2[2], y2[3], y3[0], y3[1], y3[2], y3[3]});
        acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af0, bf0, acc[j]);
        acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af1, bf1, acc[j]);
      }
    }

    if (next_k0 < K) {
      store_b_lds(next_buf, nxt_w0, nxt_w1);
      __syncthreads();
    }
    buf = next_buf;
  }

  if (wave_active) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int m_loc = wave * 16 + 2 * i + half;
      if (m_loc < BM) {
        int tok = s_tok[m_loc];
        if (tok < num_valid_tokens) {
          float w = mul_routed_weight ? topk_weights_ptr[tok] : 1.0f;
          uint16_t* cr = C_ptr + (int64_t)tok * stride_cm;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            cr[(int64_t)(n_base + 16 * j + l16) * stride_cn] = f32_to_bf16_rne(acc[j][i] * w);
          }
        }
      }
    }
  }
}

// Fast 256-thread (8 waves) kernel for BM = 128, BN = 64, BK = 32 (double-buffered s_B).
extern "C" __global__ __launch_bounds__(256) void moe_w4a16_sym_bf16_bn64_bm128(
    const uint16_t* __restrict__ A_ptr,
    const uint8_t* __restrict__ B_ptr,
    uint16_t* __restrict__ C_ptr,
    const void* __restrict__ B_scale_ptr,
    const float* __restrict__ topk_weights_ptr,
    const int32_t* __restrict__ sorted_token_ids_ptr,
    const void* __restrict__ expert_ids_ptr,
    const int32_t* __restrict__ num_tokens_post_padded_ptr,
    int N, int K, int num_valid_tokens, int top_k, int group_size,
    int64_t stride_am, int64_t stride_ak,
    int64_t stride_be, int64_t stride_bk, int64_t stride_bn,
    int64_t stride_cm, int64_t stride_cn,
    int64_t stride_bse, int64_t stride_bsk, int64_t stride_bsn,
    int BM, int mul_routed_weight, int scale_dtype) {

  const int pid_n = blockIdx.x;
  const int pid_m = blockIdx.y;
  const int ntpp = num_tokens_post_padded_ptr[0];
  if (pid_m * 128 >= ntpp) return;

  const int tid = threadIdx.x;     // 0..255
  const int wave = tid >> 5;       // 0..7
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int half = lane >> 4;
  const int n_base = pid_n * 64;
  const int off_e = load_eid(expert_ids_ptr, pid_m, scale_dtype);

  __shared__ int32_t s_tok[128];
  if (tid < 128) s_tok[tid] = sorted_token_ids_ptr[pid_m * 128 + tid];
  __syncthreads();

  if (off_e == -1) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int m_loc = wave * 16 + 2 * i + half;
      int tok = s_tok[m_loc];
      if (tok < num_valid_tokens) {
        uint16_t* cr = C_ptr + (int64_t)tok * stride_cm;
#pragma unroll
        for (int j = 0; j < 4; ++j) cr[(int64_t)(n_base + 16 * j + l16) * stride_cn] = 0;
      }
    }
    return;
  }

  const int my_m = wave * 16 + l16;
  const int my_tok = s_tok[my_m];
  const bool my_a_valid = (my_tok < num_valid_tokens);
  const bool wave_active = (__any(my_a_valid) != 0);
  const uint16_t* a_row = my_a_valid ? (A_ptr + (int64_t)(my_tok / top_k) * stride_am) : nullptr;

  const uint8_t* b_e = B_ptr + (int64_t)off_e * stride_be;
  const int64_t bs_e_off = (int64_t)off_e * stride_bse;

  __shared__ __attribute__((aligned(16))) uint16_t s_B[2][64 * 40];

  const bool k_contig = (stride_bk == 1);
  // 256 threads = 64 rows x 4 quarters (4 bytes = 8 bf16 per thread)!
  const int b_q = k_contig ? (tid & 3) : (tid >> 6);
  const int b_n = k_contig ? (tid >> 2) : (tid & 63);
  const int n_glob = n_base + b_n;

  int last_kg = -1;
  float sc = 0.0f;

  auto fetch_b_raw = [&](int k0) -> uint32_t {
    const int kg = k0 / group_size;
    if (kg != last_kg) {
      last_kg = kg;
      sc = load_scale(B_scale_ptr, bs_e_off + (int64_t)n_glob * stride_bsn + (int64_t)kg * stride_bsk, scale_dtype);
    }
    const int kb = (k0 >> 1) + b_q * 4;
    if (k_contig) {
      return *(const uint32_t*)(b_e + (int64_t)n_glob * stride_bn + kb);
    } else {
      uint32_t p = 0;
#pragma unroll
      for (int bi = 0; bi < 4; ++bi) {
        p |= ((uint32_t)b_e[(int64_t)(kb + bi) * stride_bk + (int64_t)n_glob * stride_bn]) << (bi * 8);
      }
      return p;
    }
  };

  f32x8 acc[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) acc[j] = {0, 0, 0, 0, 0, 0, 0, 0};

  uint32_t rw = fetch_b_raw(0);
  dequant_u32_to_sB(rw, sc, (uint32_t*)(s_B[0] + b_n * 40 + b_q * 8));
  __syncthreads();

  int buf = 0;
  for (int k0 = 0; k0 < K; k0 += 32) {
    const int next_k0 = k0 + 32;
    const int next_buf = buf ^ 1;
    uint32_t nxt_rw = (next_k0 < K) ? fetch_b_raw(next_k0) : 0u;

    if (wave_active) {
      u32x8 av0 = my_a_valid ? load_a16(a_row + (int64_t)k0 * stride_ak, stride_ak)
                             : (u32x8){0, 0, 0, 0, 0, 0, 0, 0};
      u32x8 av1 = my_a_valid ? load_a16(a_row + (int64_t)(k0 + 16) * stride_ak, stride_ak)
                             : (u32x8){0, 0, 0, 0, 0, 0, 0, 0};
      bf16x16 af0 = __builtin_bit_cast(bf16x16, av0);
      bf16x16 af1 = __builtin_bit_cast(bf16x16, av1);

#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint16_t* bp = &s_B[buf][(16 * j + l16) * 40];
        u32x4 y0 = *(const u32x4*)bp;
        u32x4 y1 = *(const u32x4*)(bp + 8);
        u32x4 y2 = *(const u32x4*)(bp + 16);
        u32x4 y3 = *(const u32x4*)(bp + 24);
        bf16x16 bf0 = __builtin_bit_cast(bf16x16, (u32x8){y0[0], y0[1], y0[2], y0[3], y1[0], y1[1], y1[2], y1[3]});
        bf16x16 bf1 = __builtin_bit_cast(bf16x16, (u32x8){y2[0], y2[1], y2[2], y2[3], y3[0], y3[1], y3[2], y3[3]});
        acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af0, bf0, acc[j]);
        acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af1, bf1, acc[j]);
      }
    }

    if (next_k0 < K) {
      dequant_u32_to_sB(nxt_rw, sc, (uint32_t*)(s_B[next_buf] + b_n * 40 + b_q * 8));
      __syncthreads();
    }
    buf = next_buf;
  }

  if (wave_active) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int m_loc = wave * 16 + 2 * i + half;
      int tok = s_tok[m_loc];
      if (tok < num_valid_tokens) {
        float w = mul_routed_weight ? topk_weights_ptr[tok] : 1.0f;
        uint16_t* cr = C_ptr + (int64_t)tok * stride_cm;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          cr[(int64_t)(n_base + 16 * j + l16) * stride_cn] = f32_to_bf16_rne(acc[j][i] * w);
        }
      }
    }
  }
}
