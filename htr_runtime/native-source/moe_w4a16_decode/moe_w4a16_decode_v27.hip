#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float4 load_scale4(const void* ptr, int64_t idx, int sd_type) {
  if (sd_type == 0) {
    return *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(ptr) + idx);
  } else if (sd_type == 1) {
    const hip_bfloat16* p = reinterpret_cast<const hip_bfloat16*>(ptr) + idx;
    return make_float4(float(p[0]), float(p[1]), float(p[2]), float(p[3]));
  } else {
    const __half* p = reinterpret_cast<const __half*>(ptr) + idx;
    return make_float4(__half2float(p[0]), __half2float(p[1]), __half2float(p[2]), __half2float(p[3]));
  }
}

// High-bandwidth Contiguous-N W4A16 Decode MoE GEMV for gfx1151!
// Exact Quark layout: B[E, N, K/2] uint8 with stride_bn == 1, stride_bk == N (i.e. [E, K/2, N] row-major!)
//                     S[E, N, K/gs] with stride_bsn == 1, stride_bsk == N (i.e. [E, K/gs, N] row-major!)
// - Grid: (N / 128, n_blocks, 1), Block: (128, 1, 1) = 4 Wave32 warps.
// - All 4 warps cover the SAME 128 consecutive N-columns (32 lanes x 4 consecutive N-cols = 128B/wave coalesced!),
//   splitting the K-groups (group_size=128) across the 4 warps (warp w handles groups w, w+4, w+8, ...).
// - Activations A[tok, k] are uniform across all 32 lanes (SGPR broadcast — ZERO LDS needed for A!).
// - At the end of K, the 4 warps reduce via 2 KB LDS and write C[slot, n0..n0+3].
extern "C" __global__ __launch_bounds__(128)
void moe_w4a16_decode_cn_n128(
    const hip_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ B,
    hip_bfloat16* __restrict__ C,
    const void* __restrict__ B_scale,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const void* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    int32_t N, int32_t K, int32_t num_valid_tokens, int32_t top_k, int32_t group_size,
    int64_t stride_am, int64_t stride_ak,
    int64_t stride_be, int64_t stride_bk, int64_t stride_bn,
    int64_t stride_cm, int64_t stride_cn,
    int64_t stride_bse, int64_t stride_bsk, int64_t stride_bsn,
    int32_t bm, int32_t mul_routed_weight, int32_t sd_flags) {
  const int pid_m = blockIdx.y;
  if (pid_m * bm >= num_tokens_post_padded[0]) return;

  const int sd_type = sd_flags & 3;
  const int eid_i64 = (sd_flags >> 2) & 1;
  const int64_t off_exp = eid_i64
      ? reinterpret_cast<const int64_t*>(expert_ids)[pid_m]
      : (int64_t)reinterpret_cast<const int32_t*>(expert_ids)[pid_m];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;   // 0..3 (splits K groups!)
  const int lane = tid & 31;   // 0..31 -> handles 4 consecutive N-cols!
  const int n0 = blockIdx.x * 128 + lane * 4;
  if (n0 >= N) return;

  __shared__ int32_t s_slots[128];
  __shared__ float s_wts[128];
  __shared__ int32_t s_cnt;
  if (tid == 0) s_cnt = 0;
  __syncthreads();

  if (tid < bm) {
    int32_t slot = sorted_token_ids[pid_m * bm + tid];
    if (slot < num_valid_tokens) {
      int pos = atomicAdd(&s_cnt, 1);
      s_slots[pos] = slot;
      s_wts[pos] = (mul_routed_weight && topk_weights) ? topk_weights[slot] : 1.0f;
    }
  }
  __syncthreads();
  const int cnt = s_cnt;
  if (cnt == 0) return;

  if (off_exp < 0) {
    if (warp == 0) {
      for (int r = 0; r < cnt; ++r) {
        int64_t cb = (int64_t)s_slots[r] * stride_cm + (int64_t)n0;
        C[cb] = C[cb + 1] = C[cb + 2] = C[cb + 3] = hip_bfloat16(0.0f);
      }
    }
    return;
  }

  // LDS for 4-warp reduction: 4 warps x 128 N-cols x 2 tokens = 4 KB (tiny!)
  __shared__ float s_red[4][2][128];

  const uint8_t* b_base = B + off_exp * stride_be + (int64_t)n0;
  const int64_t bs_base = off_exp * stride_bse + (int64_t)n0;
  const int n_groups = K / group_size;

  // FAST PATH: single token in block (cnt == 1) — covers ~90%+ of M=1..4 blocks!
  if (cnt == 1) {
    const int tok = s_slots[0] / top_k;
    const hip_bfloat16* a_row = A + (int64_t)tok * stride_am;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int g = warp; g < n_groups; g += 4) {
      const float4 sc = load_scale4(B_scale, bs_base + (int64_t)g * stride_bsk, sd_type);
      float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
      float sum_a = 0.0f;
      const int k_start = g * group_size;
      const int kp_start = k_start >> 1;
      const int kp_end = (k_start + group_size) >> 1;
      const uint8_t* bp = b_base + (int64_t)kp_start * stride_bk;

      #pragma unroll 4
      for (int kp = kp_start; kp < kp_end; ++kp) {
        // Uniform scalar load of 2 bf16 activations -> SGPR broadcast across all 32 lanes!
        const uint32_t ab = *reinterpret_cast<const uint32_t*>(a_row + (kp << 1));
        const float a0 = __uint_as_float((ab & 0xFFFFu) << 16);
        const float a1 = __uint_as_float(ab & 0xFFFF0000u);
        sum_a += a0 + a1;

        // 100% coalesced 128-byte cache-line read across 32 lanes (4 consecutive N-cols per lane)!
        const uint32_t w4 = *reinterpret_cast<const uint32_t*>(bp);
        bp += stride_bk;

        g0 = fmaf((float)((w4      ) & 0xFu), a0, fmaf((float)((w4 >>  4) & 0xFu), a1, g0));
        g1 = fmaf((float)((w4 >>  8) & 0xFu), a0, fmaf((float)((w4 >> 12) & 0xFu), a1, g1));
        g2 = fmaf((float)((w4 >> 16) & 0xFu), a0, fmaf((float)((w4 >> 20) & 0xFu), a1, g2));
        g3 = fmaf((float)((w4 >> 24) & 0xFu), a0, fmaf((float)((w4 >> 28) & 0xFu), a1, g3));
      }
      const float bias = -8.0f * sum_a;
      acc0 = fmaf(g0 + bias, sc.x, acc0);
      acc1 = fmaf(g1 + bias, sc.y, acc1);
      acc2 = fmaf(g2 + bias, sc.z, acc2);
      acc3 = fmaf(g3 + bias, sc.w, acc3);
    }

    s_red[warp][0][lane * 4 + 0] = acc0;
    s_red[warp][0][lane * 4 + 1] = acc1;
    s_red[warp][0][lane * 4 + 2] = acc2;
    s_red[warp][0][lane * 4 + 3] = acc3;
    __syncthreads();

    if (warp == 0) {
      const float wt = mul_routed_weight ? s_wts[0] : 1.0f;
      const int64_t cb = (int64_t)s_slots[0] * stride_cm + (int64_t)n0;
      #pragma unroll
      for (int c = 0; c < 4; ++c) {
        int lc = lane * 4 + c;
        float v = (s_red[0][0][lc] + s_red[1][0][lc]) + (s_red[2][0][lc] + s_red[3][0][lc]);
        C[cb + c] = hip_bfloat16(v * wt);
      }
    }
    return;
  }

  // GENERAL PATH: cnt >= 2 tokens (process 2 tokens at a time -> 2x weight reuse!)
  for (int base_r = 0; base_r < cnt; base_r += 2) {
    const int cur_r = (cnt - base_r < 2) ? 1 : 2;
    const hip_bfloat16* a_r0 = A + (int64_t)(s_slots[base_r] / top_k) * stride_am;
    const hip_bfloat16* a_r1 = (cur_r > 1) ? (A + (int64_t)(s_slots[base_r + 1] / top_k) * stride_am) : a_r0;

    float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;

    for (int g = warp; g < n_groups; g += 4) {
      const float4 sc = load_scale4(B_scale, bs_base + (int64_t)g * stride_bsk, sd_type);
      float g0_0 = 0.0f, g0_1 = 0.0f, g0_2 = 0.0f, g0_3 = 0.0f, sa0 = 0.0f;
      float g1_0 = 0.0f, g1_1 = 0.0f, g1_2 = 0.0f, g1_3 = 0.0f, sa1 = 0.0f;
      const int k_start = g * group_size;
      const int kp_start = k_start >> 1;
      const int kp_end = (k_start + group_size) >> 1;
      const uint8_t* bp = b_base + (int64_t)kp_start * stride_bk;

      #pragma unroll 4
      for (int kp = kp_start; kp < kp_end; ++kp) {
        const uint32_t ab0 = *reinterpret_cast<const uint32_t*>(a_r0 + (kp << 1));
        const float x0 = __uint_as_float((ab0 & 0xFFFFu) << 16);
        const float x1 = __uint_as_float(ab0 & 0xFFFF0000u);
        sa0 += x0 + x1;

        const uint32_t w4 = *reinterpret_cast<const uint32_t*>(bp);
        bp += stride_bk;

        const float w00 = (float)((w4      ) & 0xFu), w01 = (float)((w4 >>  4) & 0xFu);
        const float w10 = (float)((w4 >>  8) & 0xFu), w11 = (float)((w4 >> 12) & 0xFu);
        const float w20 = (float)((w4 >> 16) & 0xFu), w21 = (float)((w4 >> 20) & 0xFu);
        const float w30 = (float)((w4 >> 24) & 0xFu), w31 = (float)((w4 >> 28) & 0xFu);

        g0_0 = fmaf(w00, x0, fmaf(w01, x1, g0_0));
        g0_1 = fmaf(w10, x0, fmaf(w11, x1, g0_1));
        g0_2 = fmaf(w20, x0, fmaf(w21, x1, g0_2));
        g0_3 = fmaf(w30, x0, fmaf(w31, x1, g0_3));

        if (cur_r > 1) {
          const uint32_t ab1 = *reinterpret_cast<const uint32_t*>(a_r1 + (kp << 1));
          const float y0 = __uint_as_float((ab1 & 0xFFFFu) << 16);
          const float y1 = __uint_as_float(ab1 & 0xFFFF0000u);
          sa1 += y0 + y1;
          g1_0 = fmaf(w00, y0, fmaf(w01, y1, g1_0));
          g1_1 = fmaf(w10, y0, fmaf(w11, y1, g1_1));
          g1_2 = fmaf(w20, y0, fmaf(w21, y1, g1_2));
          g1_3 = fmaf(w30, y0, fmaf(w31, y1, g1_3));
        }
      }
      const float b0 = -8.0f * sa0;
      a0_0 = fmaf(g0_0 + b0, sc.x, a0_0);
      a0_1 = fmaf(g0_1 + b0, sc.y, a0_1);
      a0_2 = fmaf(g0_2 + b0, sc.z, a0_2);
      a0_3 = fmaf(g0_3 + b0, sc.w, a0_3);
      if (cur_r > 1) {
        const float b1 = -8.0f * sa1;
        a1_0 = fmaf(g1_0 + b1, sc.x, a1_0);
        a1_1 = fmaf(g1_1 + b1, sc.y, a1_1);
        a1_2 = fmaf(g1_2 + b1, sc.z, a1_2);
        a1_3 = fmaf(g1_3 + b1, sc.w, a1_3);
      }
    }

    s_red[warp][0][lane * 4 + 0] = a0_0;
    s_red[warp][0][lane * 4 + 1] = a0_1;
    s_red[warp][0][lane * 4 + 2] = a0_2;
    s_red[warp][0][lane * 4 + 3] = a0_3;
    if (cur_r > 1) {
      s_red[warp][1][lane * 4 + 0] = a1_0;
      s_red[warp][1][lane * 4 + 1] = a1_1;
      s_red[warp][1][lane * 4 + 2] = a1_2;
      s_red[warp][1][lane * 4 + 3] = a1_3;
    }
    __syncthreads();

    if (warp == 0) {
      const float wt0 = mul_routed_weight ? s_wts[base_r] : 1.0f;
      const int64_t cb0 = (int64_t)s_slots[base_r] * stride_cm + (int64_t)n0;
      #pragma unroll
      for (int c = 0; c < 4; ++c) {
        int lc = lane * 4 + c;
        float v = (s_red[0][0][lc] + s_red[1][0][lc]) + (s_red[2][0][lc] + s_red[3][0][lc]);
        C[cb0 + c] = hip_bfloat16(v * wt0);
      }
      if (cur_r > 1) {
        const float wt1 = mul_routed_weight ? s_wts[base_r + 1] : 1.0f;
        const int64_t cb1 = (int64_t)s_slots[base_r + 1] * stride_cm + (int64_t)n0;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          int lc = lane * 4 + c;
          float v = (s_red[0][1][lc] + s_red[1][1][lc]) + (s_red[2][1][lc] + s_red[3][1][lc]);
          C[cb1 + c] = hip_bfloat16(v * wt1);
        }
      }
    }
    if (base_r + 2 < cnt) __syncthreads();
  }
}
