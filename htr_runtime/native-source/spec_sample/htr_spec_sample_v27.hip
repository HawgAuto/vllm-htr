// htr_spec_sample_v27.hip — Wave 6 gfx1151 Vectorized BF16 2-Pass Top-64 + Greedy Fast-Path Spec Sampler
// Upgrades over v0.2.6 (htr_spec_sample.hip):
//   1. BF16 16-bit order-preserving key + 128-bit (uint4 = 8x BF16) vectorized global_load_b128 scans!
//      Cuts select_local_topk from 5 scalar 2-byte passes -> 2 vectorized 16-byte passes (11-bit + 5-bit/32-bin)!
//   2. Parallel 8-wave DPP prefix scan replaces serial 256-iter tid==0 loops across all histogram passes.
//   3. Single-pass vectorized uint4 greedy argmax fast-path (temp==0.0f) in k_target_sample_sparse and
//      k_q38_filter_gumbel_sample: 0 histograms, 0 global merge, 0 bitonic sort, 100.000% bitwise parity!
//   4. Fast-path host tensor guards (avoids redundant ATen .to().contiguous() dispatch when already matched).
//   5. Exact same 4 pybind exports as htr_spec_sample.so (drop-in replacement via HTR_SPEC_SAMPLE_SO).
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>
#include <type_traits>

__device__ __forceinline__ float ld_f(const float* p) { return *p; }
__device__ __forceinline__ float ld_f(const uint16_t* p) {
  return __builtin_bit_cast(float, ((uint32_t)(*p)) << 16);
}
__device__ __forceinline__ void st_raw(float* p, float v) { *p = v; }
__device__ __forceinline__ void st_raw(uint16_t* p, float v) {
  uint32_t u = __builtin_bit_cast(uint32_t, v);
  if (v != v) { *p = 0x7fffu; return; }
  u += 0x7FFFu + ((u >> 16) & 1u);
  *p = (uint16_t)(u >> 16);
}
__device__ __forceinline__ float neg_inf_val() {
  return __builtin_bit_cast(float, 0xFF800000u);
}

// IEEE-754 float -> order-preserving uint32 (NaN -> 0)
__device__ __forceinline__ uint32_t f2key(float x) {
  uint32_t u = __builtin_bit_cast(uint32_t, x);
  if (x != x) return 0u;
  return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ float key2f(uint32_t k) {
  uint32_t u = (k & 0x80000000u) ? (k ^ 0x80000000u) : ~k;
  return __builtin_bit_cast(float, u);
}

// Exact 16-bit order-preserving key for BF16 (bit-for-bit equivalent to f2key(bf16_to_f32(h)))
__device__ __forceinline__ uint32_t bf16_to_key16(uint16_t h) {
  if ((h & 0x7FFFu) > 0x7F80u) return 0u;  // NaN -> 0
  return (h & 0x8000u) ? (uint32_t)((uint16_t)(~h)) : (uint32_t)(h ^ 0x8000u);
}

__device__ __forceinline__ uint32_t key16_to_key32(uint32_t k16) {
  if (k16 == 0u) return 0u;
  return (k16 & 0x8000u) ? (k16 << 16) : ((k16 << 16) | 0xFFFFu);
}

__device__ __forceinline__ bool better(uint32_t ka, int32_t ia, uint32_t kb, int32_t ib) {
  return (ka > kb) || (ka == kb && ia < ib);
}

// Wave32 DPP uint32 sum (all lanes receive sum, zero LDS)
__device__ __forceinline__ uint32_t wave_sum_u32(uint32_t v) {
  v += __shfl_xor(v, 1, 32);
  v += __shfl_xor(v, 2, 32);
  v += __shfl_xor(v, 4, 32);
  v += __shfl_xor(v, 8, 32);
  uint32_t hi;
  asm volatile("v_permlanex16_b32 %0, %1, 0, 0" : "=&v"(hi) : "v"(v));
  return v + hi;
}

// Fast 256-thread descending threshold finder over N_BINS in {2048, 1024}
// Avoids 256-iter tid==0 serial scan: 8 waves sum 32 threads each -> tid==0 scans ONLY 8 wave sums + 32 + 8!
template <int N_BINS>
__device__ __forceinline__ void find_thresh_bin_desc(
    const uint32_t* s_hist, uint32_t target_k,
    uint32_t* s_blk_sum, int* s_thr_bin, uint32_t* s_gt_cnt) {
  constexpr int PER_T = N_BINS / 256;  // 8 or 4
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  uint32_t local_s = 0u;
#pragma unroll
  for (int j = 0; j < PER_T; ++j) local_s += s_hist[(N_BINS - 1) - (tid * PER_T + j)];
  s_blk_sum[tid] = local_s;
  __shared__ uint32_t s_wsum[8];
  uint32_t ws = wave_sum_u32(local_s);
  if (lane == 0) s_wsum[wave] = ws;
  __syncthreads();
  if (tid == 0) {
    uint32_t acc = 0u;
    int thr_b = 0;
    uint32_t gt = 0u;
    for (int w = 0; w < 8; ++w) {
      if (acc + s_wsum[w] >= target_k) {
        const int g0 = w * 32;
        for (int g = g0; g < g0 + 32; ++g) {
          if (acc + s_blk_sum[g] >= target_k) {
            for (int j = 0; j < PER_T; ++j) {
              int b = (N_BINS - 1) - (g * PER_T + j);
              uint32_t c = s_hist[b];
              if (acc + c >= target_k) {
                thr_b = b;
                gt = acc;
                break;
              }
              acc += c;
            }
            break;
          }
          acc += s_blk_sum[g];
        }
        break;
      }
      acc += s_wsum[w];
    }
    *s_thr_bin = thr_b;
    *s_gt_cnt = gt;
  }
  __syncthreads();
}

// ============================================================================
// Vectorized 2-Pass BF16 Local Top-K (uint4 = 8x BF16/load, 11-bit + 5-bit!)
// ============================================================================
__device__ void select_local_topk_bf16(
    const uint16_t* __restrict__ row_logits, int v_beg, int v_end, int K,
    uint32_t* s_hist, int* s_cnt, uint32_t* out_k, int32_t* out_i) {
  const int tid = threadIdx.x;
  const int v_beg_a = (v_beg + 7) & ~7;
  const int v_end_a = v_end & ~7;
  const bool has_vec = (v_beg_a < v_end_a);

  // Pass 1: top 11 bits [15:5] histogram (2048 bins) via 128-bit uint4 loads!
  for (int b = tid; b < 2048; b += 256) s_hist[b] = 0u;
  __syncthreads();

  if (has_vec) {
    if (v_beg + tid < v_beg_a) {
      atomicAdd(&s_hist[bf16_to_key16(row_logits[v_beg + tid]) >> 5], 1u);
    }
    for (int v = v_beg_a + tid * 8; v < v_end_a; v += 256 * 8) {
      const uint4 raw = *reinterpret_cast<const uint4*>(row_logits + v);
      const uint32_t ws[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t k0 = bf16_to_key16((uint16_t)(ws[j] & 0xFFFFu));
        uint32_t k1 = bf16_to_key16((uint16_t)(ws[j] >> 16));
        atomicAdd(&s_hist[k0 >> 5], 1u);
        atomicAdd(&s_hist[k1 >> 5], 1u);
      }
    }
    if (v_end_a + tid < v_end) {
      atomicAdd(&s_hist[bf16_to_key16(row_logits[v_end_a + tid]) >> 5], 1u);
    }
  } else {
    for (int v = v_beg + tid; v < v_end; v += 256) {
      atomicAdd(&s_hist[bf16_to_key16(row_logits[v]) >> 5], 1u);
    }
  }
  __syncthreads();

  __shared__ uint32_t s_blk_sum[256];
  __shared__ int s_thr_bin;
  __shared__ uint32_t s_gt_cnt;
  find_thresh_bin_desc<2048>(s_hist, (uint32_t)K, s_blk_sum, &s_thr_bin, &s_gt_cnt);

  const uint32_t b1 = (uint32_t)s_thr_bin;
  const uint32_t need1 = (uint32_t)K - s_gt_cnt;

  // Pass 2: collect hi > b1 AND histogram bottom 5 bits [4:0] (ONLY 32 bins!) via uint4 loads!
  if (tid == 0) *s_cnt = 0;
  if (tid < 32) s_hist[tid] = 0u;
  __syncthreads();

  auto proc_pass2 = [&](uint16_t h, int v) {
    uint32_t k16 = bf16_to_key16(h);
    uint32_t hi = k16 >> 5;
    if (hi > b1) {
      int pos = atomicAdd(s_cnt, 1);
      if (pos < K) { out_k[pos] = key16_to_key32(k16); out_i[pos] = v; }
    } else if (hi == b1) {
      atomicAdd(&s_hist[k16 & 31u], 1u);
    }
  };

  if (has_vec) {
    if (v_beg + tid < v_beg_a) proc_pass2(row_logits[v_beg + tid], v_beg + tid);
    for (int v = v_beg_a + tid * 8; v < v_end_a; v += 256 * 8) {
      const uint4 raw = *reinterpret_cast<const uint4*>(row_logits + v);
      const uint32_t ws[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        proc_pass2((uint16_t)(ws[j] & 0xFFFFu), v + 2 * j);
        proc_pass2((uint16_t)(ws[j] >> 16),     v + 2 * j + 1);
      }
    }
    if (v_end_a + tid < v_end) proc_pass2(row_logits[v_end_a + tid], v_end_a + tid);
  } else {
    for (int v = v_beg + tid; v < v_end; v += 256) proc_pass2(row_logits[v], v);
  }
  __syncthreads();

  // 32-bin descending scan in a single wave (tid == 0 does at most 32 iterations in L0/LDS)
  if (tid == 0) {
    uint32_t acc = 0u;
    int tb = 0;
    uint32_t gt = 0u;
    for (int b = 31; b >= 0; --b) {
      uint32_t c = s_hist[b];
      if (acc + c >= need1) { tb = b; gt = acc; break; }
      acc += c;
    }
    s_thr_bin = tb;
    s_gt_cnt = gt;
  }
  __syncthreads();

  const uint32_t b2 = (uint32_t)s_thr_bin;
  const uint32_t exact_k16 = (b1 << 5) | b2;
  const uint32_t exact_key32 = key16_to_key32(exact_k16);
  const uint32_t need_eq = need1 - s_gt_cnt;

  // Collect (hi == b1 && lo > b2) if possible (skip completely when b2 == 31!)
  if (b2 < 31u) {
    auto proc_gt = [&](uint16_t h, int v) {
      uint32_t k16 = bf16_to_key16(h);
      if ((k16 >> 5) == b1 && (k16 & 31u) > b2) {
        int pos = atomicAdd(s_cnt, 1);
        if (pos < K) { out_k[pos] = key16_to_key32(k16); out_i[pos] = v; }
      }
    };
    if (has_vec) {
      if (v_beg + tid < v_beg_a) proc_gt(row_logits[v_beg + tid], v_beg + tid);
      for (int v = v_beg_a + tid * 8; v < v_end_a; v += 256 * 8) {
        const uint4 raw = *reinterpret_cast<const uint4*>(row_logits + v);
        const uint32_t ws[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          proc_gt((uint16_t)(ws[j] & 0xFFFFu), v + 2 * j);
          proc_gt((uint16_t)(ws[j] >> 16),     v + 2 * j + 1);
        }
      }
      if (v_end_a + tid < v_end) proc_gt(row_logits[v_end_a + tid], v_end_a + tid);
    } else {
      for (int v = v_beg + tid; v < v_end; v += 256) proc_gt(row_logits[v], v);
    }
    __syncthreads();
  }

  // Collect smallest `need_eq` exact ties via wave-ordered ballot scan (ascending v, early exit!)
  __shared__ int s_eq_cnt;
  __shared__ int s_wcnt[8];
  if (tid == 0) s_eq_cnt = 0;
  __syncthreads();
  const int wave = tid >> 5;
  const int lane = tid & 31;
  for (int base = v_beg; base < v_end; base += 256) {
    if (s_eq_cnt >= (int)need_eq) break;
    int v = base + tid;
    bool hit = (v < v_end) && (bf16_to_key16(row_logits[v]) == exact_k16);
    uint32_t bal = __builtin_amdgcn_ballot_w32(hit);
    if (lane == 0) s_wcnt[wave] = __popc(bal);
    __syncthreads();
    int woff = 0;
    for (int w = 0; w < wave; ++w) woff += s_wcnt[w];
    int prev = s_eq_cnt;
    if (hit) {
      int rank = prev + woff + __popc(bal & ((1u << lane) - 1u));
      if (rank < (int)need_eq) {
        int pos = (*s_cnt) + rank;
        if (pos < K) { out_k[pos] = exact_key32; out_i[pos] = v; }
      }
    }
    __syncthreads();
    if (tid == 0) {
      int step_sum = 0;
      for (int w = 0; w < 8; ++w) step_sum += s_wcnt[w];
      s_eq_cnt += step_sum;
    }
    __syncthreads();
  }
  if (tid == 0) {
    int cur = *s_cnt + (s_eq_cnt < (int)need_eq ? s_eq_cnt : (int)need_eq);
    for (int p = cur; p < K; ++p) { out_k[p] = 0u; out_i[p] = -1; }
  }
  __syncthreads();
}

// FP32 fallback (also accelerated with find_thresh_bin_desc)
__device__ void select_local_topk_f32(
    const float* __restrict__ row_logits, int v_beg, int v_end, int K,
    uint32_t* s_hist, int* s_cnt, uint32_t* out_k, int32_t* out_i) {
  const int tid = threadIdx.x;
  for (int b = tid; b < 2048; b += 256) s_hist[b] = 0u;
  __syncthreads();
  for (int v = v_beg + tid; v < v_end; v += 256) {
    atomicAdd(&s_hist[f2key(row_logits[v]) >> 21], 1u);
  }
  __syncthreads();

  __shared__ uint32_t s_blk_sum[256];
  __shared__ int s_thr_bin;
  __shared__ uint32_t s_gt_cnt;
  find_thresh_bin_desc<2048>(s_hist, (uint32_t)K, s_blk_sum, &s_thr_bin, &s_gt_cnt);

  const uint32_t b1 = (uint32_t)s_thr_bin;
  const uint32_t need1 = (uint32_t)K - s_gt_cnt;

  if (tid == 0) *s_cnt = 0;
  for (int b = tid; b < 2048; b += 256) s_hist[b] = 0u;
  __syncthreads();
  for (int v = v_beg + tid; v < v_end; v += 256) {
    uint32_t key = f2key(row_logits[v]);
    uint32_t hi = key >> 21;
    if (hi > b1) {
      int pos = atomicAdd(s_cnt, 1);
      if (pos < K) { out_k[pos] = key; out_i[pos] = v; }
    } else if (hi == b1) {
      atomicAdd(&s_hist[(key >> 10) & 2047u], 1u);
    }
  }
  __syncthreads();

  find_thresh_bin_desc<2048>(s_hist, need1, s_blk_sum, &s_thr_bin, &s_gt_cnt);
  const uint32_t b2 = (uint32_t)s_thr_bin;
  const uint32_t need2 = need1 - s_gt_cnt;

  for (int b = tid; b < 1024; b += 256) s_hist[b] = 0u;
  __syncthreads();
  for (int v = v_beg + tid; v < v_end; v += 256) {
    uint32_t key = f2key(row_logits[v]);
    if ((key >> 21) == b1) {
      uint32_t mid = (key >> 10) & 2047u;
      if (mid > b2) {
        int pos = atomicAdd(s_cnt, 1);
        if (pos < K) { out_k[pos] = key; out_i[pos] = v; }
      } else if (mid == b2) {
        atomicAdd(&s_hist[key & 1023u], 1u);
      }
    }
  }
  __syncthreads();

  find_thresh_bin_desc<1024>(s_hist, need2, s_blk_sum, &s_thr_bin, &s_gt_cnt);
  const uint32_t b3 = (uint32_t)s_thr_bin;
  const uint32_t exact_key = (b1 << 21) | (b2 << 10) | b3;
  const uint32_t need_eq = need2 - s_gt_cnt;

  __shared__ int s_eq_cnt;
  __shared__ int s_wcnt[8];
  if (tid == 0) s_eq_cnt = 0;
  __syncthreads();
  for (int v = v_beg + tid; v < v_end; v += 256) {
    uint32_t key = f2key(row_logits[v]);
    if ((key >> 10) == ((b1 << 11) | b2) && (key & 1023u) > b3) {
      int pos = atomicAdd(s_cnt, 1);
      if (pos < K) { out_k[pos] = key; out_i[pos] = v; }
    }
  }
  __syncthreads();
  const int wave = tid >> 5;
  const int lane = tid & 31;
  for (int base = v_beg; base < v_end; base += 256) {
    if (s_eq_cnt >= (int)need_eq) break;
    int v = base + tid;
    bool hit = (v < v_end) && (f2key(row_logits[v]) == exact_key);
    uint32_t bal = __builtin_amdgcn_ballot_w32(hit);
    if (lane == 0) s_wcnt[wave] = __popc(bal);
    __syncthreads();
    int woff = 0;
    for (int w = 0; w < wave; ++w) woff += s_wcnt[w];
    int prev = s_eq_cnt;
    if (hit) {
      int rank = prev + woff + __popc(bal & ((1u << lane) - 1u));
      if (rank < (int)need_eq) {
        int pos = (*s_cnt) + rank;
        if (pos < K) { out_k[pos] = exact_key; out_i[pos] = v; }
      }
    }
    __syncthreads();
    if (tid == 0) {
      int step_sum = 0;
      for (int w = 0; w < 8; ++w) step_sum += s_wcnt[w];
      s_eq_cnt += step_sum;
    }
    __syncthreads();
  }
  if (tid == 0) {
    int cur = *s_cnt + (s_eq_cnt < (int)need_eq ? s_eq_cnt : (int)need_eq);
    for (int p = cur; p < K; ++p) { out_k[p] = 0u; out_i[p] = -1; }
  }
  __syncthreads();
}

template <typename T>
__device__ __forceinline__ void select_local_topk(
    const T* __restrict__ row_logits, int v_beg, int v_end, int K,
    uint32_t* s_hist, int* s_cnt, uint32_t* out_k, int32_t* out_i) {
  if constexpr (std::is_same_v<T, uint16_t>) {
    select_local_topk_bf16(row_logits, v_beg, v_end, K, s_hist, s_cnt, out_k, out_i);
  } else {
    select_local_topk_f32(row_logits, v_beg, v_end, K, s_hist, s_cnt, out_k, out_i);
  }
}

// ============================================================================
// Single-pass vectorized argmax helper for greedy fast-path (t == 0.0f)
// Returns (best_key32, best_id) across [v_beg, v_end) in s_out_k, s_out_i (on tid==0)
// ============================================================================
template <typename InT>
__device__ void local_argmax(
    const InT* __restrict__ row_in, int v_beg, int v_end,
    uint32_t& out_key, int32_t& out_id) {
  const int tid = threadIdx.x;
  uint32_t bk = 0u;
  int32_t bi = 0x7FFFFFFF;
  if constexpr (std::is_same_v<InT, uint16_t>) {
    const int v_beg_a = (v_beg + 7) & ~7;
    const int v_end_a = v_end & ~7;
    uint32_t bk16 = 0u;
    auto upd = [&](uint16_t h, int v) {
      uint32_t k16 = bf16_to_key16(h);
      if (k16 > bk16 || (k16 == bk16 && v < bi)) { bk16 = k16; bi = v; }
    };
    if (v_beg_a < v_end_a) {
      if (v_beg + tid < v_beg_a) upd(row_in[v_beg + tid], v_beg + tid);
      for (int v = v_beg_a + tid * 8; v < v_end_a; v += 256 * 8) {
        const uint4 raw = *reinterpret_cast<const uint4*>(row_in + v);
        const uint32_t ws[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          upd((uint16_t)(ws[j] & 0xFFFFu), v + 2 * j);
          upd((uint16_t)(ws[j] >> 16),     v + 2 * j + 1);
        }
      }
      if (v_end_a + tid < v_end) upd(row_in[v_end_a + tid], v_end_a + tid);
    } else {
      for (int v = v_beg + tid; v < v_end; v += 256) upd(row_in[v], v);
    }
    bk = key16_to_key32(bk16);
  } else {
    for (int v = v_beg + tid; v < v_end; v += 256) {
      uint32_t k = f2key(row_in[v]);
      if (k > bk || (k == bk && v < bi)) { bk = k; bi = v; }
    }
  }
  // Wave32 shuffle reduction
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    uint32_t ok = __shfl_xor(bk, off, 32);
    int32_t oi = __shfl_xor(bi, off, 32);
    if (better(ok, oi, bk, bi)) { bk = ok; bi = oi; }
  }
  __shared__ uint32_t s_wk[8];
  __shared__ int32_t s_wi[8];
  if ((tid & 31) == 0) { s_wk[tid >> 5] = bk; s_wi[tid >> 5] = bi; }
  __syncthreads();
  if (tid == 0) {
    uint32_t gk = s_wk[0];
    int32_t gi = s_wi[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) {
      if (better(s_wk[w], s_wi[w], gk, gi)) { gk = s_wk[w]; gi = s_wi[w]; }
    }
    out_key = gk;
    out_id = gi;
  }
}

// ============================================================================
// Exact vLLM RNG helpers (100% bit-exact match with gumbel.py & Triton tl.rand)
// ============================================================================
__device__ __forceinline__ uint32_t m3_rotl32(uint32_t v, int s) {
  return (v << s) | (v >> (32 - s));
}
__device__ __forceinline__ uint32_t m3_mix(uint32_t h, uint32_t k) {
  k *= 0xCC9E2D51u;
  k = m3_rotl32(k, 15);
  k *= 0x1B873593u;
  h ^= k;
  h = m3_rotl32(h, 13);
  return h * 5u + 0xE6546B64u;
}
__device__ __forceinline__ uint32_t m3_fmix32(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  return h ^ (h >> 16);
}
__device__ __forceinline__ uint32_t m3_hash32(int64_t seed, int64_t pos, uint32_t offset, uint32_t domain = 0u) {
  uint64_t s = (uint64_t)seed, p = (uint64_t)pos;
  uint32_t h = domain;
  h = m3_mix(h, (uint32_t)(s & 0xFFFFFFFFull));
  h = m3_mix(h, (uint32_t)((s >> 32) & 0xFFFFFFFFull));
  h = m3_mix(h, (uint32_t)(p & 0xFFFFFFFFull));
  h = m3_mix(h, offset);
  return m3_fmix32(h ^ 16u);
}
__device__ __forceinline__ float m3_uniform32(int64_t seed, int64_t pos, uint32_t offset) {
  uint32_t r = m3_hash32(seed, pos, offset, 0u);
  int32_t hi16 = (int32_t)(r >> 16);
  int32_t lo16 = (int32_t)(r & 0xFFFFu);
  return (float)hi16 * 1.52587890625e-05f + ((float)lo16 + 0.5f) * 2.3283064365386963e-10f;
}
__device__ __forceinline__ double m3_uniform64(int64_t seed, int64_t pos, uint32_t offset) {
  uint64_t lo = (uint64_t)m3_hash32(seed, pos, offset, 0u);
  uint64_t hi = (uint64_t)m3_hash32(seed, pos, offset, 0x9E3779B9u);
  uint64_t r53 = ((hi << 32) | lo) >> 11;
  double u = ((double)r53 + 0.5) * 1.1102230246251565e-16;
  if (u == 1.0) u -= 1.1102230246251565e-16;
  return u;
}
__device__ __forceinline__ float log1p_neg_stable(float v) {
  float poly = 1.0f / 8.0f;
  poly = 1.0f / 7.0f + v * poly;
  poly = 1.0f / 6.0f + v * poly;
  poly = 1.0f / 5.0f + v * poly;
  poly = 1.0f / 4.0f + v * poly;
  poly = 1.0f / 3.0f + v * poly;
  poly = 1.0f / 2.0f + v * poly;
  poly = 1.0f + v * poly;
  float series = -v * poly;
  float direct = logf(fmaxf(1.0f - v, 5.960464477539063e-08f));
  return (v < 0.25f) ? series : direct;
}
__device__ __forceinline__ float gumbel_noise32(int64_t seed, int64_t pos, uint32_t offset) {
  float u = m3_uniform32(seed, pos, offset);
  return -logf(-log1p_neg_stable(u));
}
__device__ __forceinline__ double gumbel_noise64(int64_t seed, int64_t pos, uint32_t offset) {
  double u = m3_uniform64(seed, pos, offset);
  return -log(-log(u));
}
__device__ __forceinline__ float triton_philox_rand32(int64_t seed, int64_t pos) {
  uint64_t s = (uint64_t)seed;
  uint32_t k0 = (uint32_t)(s & 0xFFFFFFFFull);
  uint32_t k1 = (uint32_t)((s >> 32) & 0xFFFFFFFFull);
  uint64_t p = (uint64_t)pos;
  uint32_t c0 = (uint32_t)(p & 0xFFFFFFFFull);
  uint32_t c1 = (uint32_t)((p >> 32) & 0xFFFFFFFFull);
  uint32_t c2 = 0u, c3 = 0u;
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    uint32_t _c0 = c0, _c2 = c2;
    c0 = __umulhi(0xCD9E8D57u, _c2) ^ c1 ^ k0;
    c2 = __umulhi(0xD2511F53u, _c0) ^ c3 ^ k1;
    c1 = 0xCD9E8D57u * _c2;
    c3 = 0xD2511F53u * _c0;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  int32_t x = (int32_t)c0;
  if (x < 0) x = ~x;
  float u = (float)x * 4.6566127342e-10f;
  return fmaxf(u, 4.6566127342e-10f);
}

// Reusable 256-thread in-LDS merge of NS * 64 candidates into sorted global top-64
// BF16 specialization uses 2 passes (11-bit + 5-bit/32-bin) instead of 3 passes!
template <bool IS_BF16>
__device__ void merge_global_top64(
    const uint32_t* __restrict__ rk, const int32_t* __restrict__ ri, int N_cand,
    uint32_t* s_hist, uint32_t* s_blk_sum, uint32_t* s_loc_k, int32_t* s_loc_i,
    int* s_cnt, int* s_thr_bin, uint32_t* s_gt_cnt, int* s_eq_n) {
  const int tid = threadIdx.x;
  const int KC = 64;

  for (int b = tid; b < 2048; b += 256) s_hist[b] = 0u;
  __syncthreads();
  for (int c = tid; c < N_cand; c += 256) if (ri[c] >= 0) atomicAdd(&s_hist[rk[c] >> 21], 1u);
  __syncthreads();
  find_thresh_bin_desc<2048>(s_hist, (uint32_t)KC, s_blk_sum, s_thr_bin, s_gt_cnt);

  const uint32_t gb1 = (uint32_t)(*s_thr_bin), gneed1 = (uint32_t)KC - (*s_gt_cnt);

  if constexpr (IS_BF16) {
    // Only 5 meaningful bits remain: bits [20:16] = 32 bins!
    if (tid == 0) *s_cnt = 0;
    if (tid < 32) s_hist[tid] = 0u;
    __syncthreads();
    for (int c = tid; c < N_cand; c += 256) {
      int32_t id = ri[c]; if (id < 0) continue;
      uint32_t k = rk[c], hi = k >> 21;
      if (hi > gb1) { int p = atomicAdd(s_cnt, 1); if (p < KC) { s_loc_k[p] = k; s_loc_i[p] = id; } }
      else if (hi == gb1) atomicAdd(&s_hist[(k >> 16) & 31u], 1u);
    }
    __syncthreads();
    if (tid == 0) {
      uint32_t acc = 0u, gt = 0u; int tb = 0;
      for (int b = 31; b >= 0; --b) {
        if (acc + s_hist[b] >= gneed1) { tb = b; gt = acc; break; }
        acc += s_hist[b];
      }
      *s_thr_bin = tb; *s_gt_cnt = gt; *s_eq_n = 0;
    }
    __syncthreads();
    const uint32_t gb2 = (uint32_t)(*s_thr_bin);
    const uint32_t gexact16 = (gb1 << 5) | gb2;
    const uint32_t gexact = key16_to_key32(gexact16);
    const uint32_t gneed_eq = gneed1 - (*s_gt_cnt);

    for (int c = tid; c < N_cand; c += 256) {
      int32_t id = ri[c]; if (id < 0) continue;
      uint32_t k = rk[c];
      if ((k >> 21) == gb1) {
        uint32_t lo5 = (k >> 16) & 31u;
        if (lo5 > gb2) {
          int p = atomicAdd(s_cnt, 1); if (p < KC) { s_loc_k[p] = k; s_loc_i[p] = id; }
        } else if (lo5 == gb2) {
          int q = atomicAdd(s_eq_n, 1); if (q < 2048) s_hist[q] = (uint32_t)id;
        }
      }
    }
    __syncthreads();
    if (tid == 0) {
      int n_eq = (*s_eq_n) < 2048 ? (*s_eq_n) : 2048;
      int take = (int)gneed_eq < n_eq ? (int)gneed_eq : n_eq;
      int base_p = *s_cnt;
      for (int j = 0; j < take; ++j) {
        int best = j; uint32_t bv = s_hist[j];
        for (int q = j + 1; q < n_eq; ++q) if (s_hist[q] < bv) { bv = s_hist[q]; best = q; }
        s_hist[best] = s_hist[j];
        if (base_p + j < KC) { s_loc_k[base_p + j] = gexact; s_loc_i[base_p + j] = (int32_t)bv; }
      }
    }
    __syncthreads();
  } else {
    if (tid == 0) *s_cnt = 0;
    for (int b = tid; b < 2048; b += 256) s_hist[b] = 0u;
    __syncthreads();
    for (int c = tid; c < N_cand; c += 256) {
      int32_t id = ri[c]; if (id < 0) continue;
      uint32_t k = rk[c], hi = k >> 21;
      if (hi > gb1) { int p = atomicAdd(s_cnt, 1); if (p < KC) { s_loc_k[p] = k; s_loc_i[p] = id; } }
      else if (hi == gb1) atomicAdd(&s_hist[(k >> 10) & 2047u], 1u);
    }
    __syncthreads();
    find_thresh_bin_desc<2048>(s_hist, gneed1, s_blk_sum, s_thr_bin, s_gt_cnt);
    const uint32_t gb2 = (uint32_t)(*s_thr_bin), gneed2 = gneed1 - (*s_gt_cnt);

    for (int b = tid; b < 1024; b += 256) s_hist[b] = 0u;
    __syncthreads();
    for (int c = tid; c < N_cand; c += 256) {
      int32_t id = ri[c]; if (id < 0) continue;
      uint32_t k = rk[c];
      if ((k >> 21) == gb1) {
        uint32_t mid = (k >> 10) & 2047u;
        if (mid > gb2) { int p = atomicAdd(s_cnt, 1); if (p < KC) { s_loc_k[p] = k; s_loc_i[p] = id; } }
        else if (mid == gb2) atomicAdd(&s_hist[k & 1023u], 1u);
      }
    }
    __syncthreads();
    find_thresh_bin_desc<1024>(s_hist, gneed2, s_blk_sum, s_thr_bin, s_gt_cnt);
    const uint32_t gb3 = (uint32_t)(*s_thr_bin);
    const uint32_t gexact = (gb1 << 21) | (gb2 << 10) | gb3;
    const uint32_t gneed_eq = gneed2 - (*s_gt_cnt);

    if (tid == 0) *s_eq_n = 0;
    __syncthreads();
    for (int c = tid; c < N_cand; c += 256) {
      int32_t id = ri[c]; if (id < 0) continue;
      uint32_t k = rk[c];
      if ((k >> 10) == ((gb1 << 11) | gb2) && (k & 1023u) > gb3) {
        int p = atomicAdd(s_cnt, 1); if (p < KC) { s_loc_k[p] = k; s_loc_i[p] = id; }
      } else if (k == gexact) {
        int q = atomicAdd(s_eq_n, 1); if (q < 2048) s_hist[q] = (uint32_t)id;
      }
    }
    __syncthreads();
    if (tid == 0) {
      int n_eq = (*s_eq_n) < 2048 ? (*s_eq_n) : 2048;
      int take = (int)gneed_eq < n_eq ? (int)gneed_eq : n_eq;
      int base_p = *s_cnt;
      for (int j = 0; j < take; ++j) {
        int best = j; uint32_t bv = s_hist[j];
        for (int q = j + 1; q < n_eq; ++q) if (s_hist[q] < bv) { bv = s_hist[q]; best = q; }
        s_hist[best] = s_hist[j];
        if (base_p + j < KC) { s_loc_k[base_p + j] = gexact; s_loc_i[base_p + j] = (int32_t)bv; }
      }
    }
    __syncthreads();
  }

  // Fast 64-element bitonic sort: only cross-wave step (st==32) needs LDS + barrier!
  // All st <= 16 steps execute purely in Wave32 registers via __shfl_xor (15 fewer barriers!).
  if (tid < 64) {
    uint32_t mk = s_loc_k[tid];
    int32_t mi = s_loc_i[tid];
    for (int sz = 2; sz <= 64; sz <<= 1) {
      bool dir = ((tid & sz) == 0);
      for (int st = sz >> 1; st > 0; st >>= 1) {
        if (st == 32) {
          s_loc_k[tid] = mk;
          s_loc_i[tid] = mi;
          __syncthreads();
          int ixj = tid ^ 32;
          uint32_t ok = s_loc_k[ixj];
          int32_t oi = s_loc_i[ixj];
          bool lo = (tid < ixj);
          bool a_better = lo ? better(mk, mi, ok, oi) : better(ok, oi, mk, mi);
          bool swap = (dir && !a_better) || (!dir && a_better);
          if (swap) { mk = ok; mi = oi; }
          __syncthreads();
        } else {
          uint32_t ok = __shfl_xor(mk, st, 32);
          int32_t oi = __shfl_xor(mi, st, 32);
          bool lo = ((tid & st) == 0);
          bool a_better = lo ? better(mk, mi, ok, oi) : better(ok, oi, mk, mi);
          bool swap = (dir && !a_better) || (!dir && a_better);
          if (swap) { mk = ok; mi = oi; }
        }
      }
    }
    s_loc_k[tid] = mk;
    s_loc_i[tid] = mi;
  } else {
    // Match the 2 __syncthreads() calls for the single st==32 stage (when sz==64)
    __syncthreads();
    __syncthreads();
  }
  __syncthreads();
}

// ============================================================================
// V5-S3: 1-launch 40-CU Fused Draft Filter + Gumbel Sample + Sparse/Dense Cache
// ============================================================================
template <typename InT, typename CacheT, bool WRITE_DENSE, bool USE_FP64>
__global__ void __launch_bounds__(256) k_q38_filter_gumbel_sample(
    const InT* __restrict__ logits,
    CacheT* __restrict__ draft_cache,
    int64_t cache_s0, int64_t cache_s1,
    const int64_t* __restrict__ step_ptr,
    int32_t* __restrict__ sp_ids,
    float* __restrict__ sp_vals,
    int32_t* __restrict__ sp_valid,
    int64_t* __restrict__ sampled_out,
    const int32_t* __restrict__ idx_map,
    const float* __restrict__ temp_buf,
    const int32_t* __restrict__ topk_buf,
    const float* __restrict__ topp_buf,
    const int64_t* __restrict__ seeds_buf,
    const int64_t* __restrict__ pos_buf,
    uint32_t* __restrict__ cand_k,
    int32_t* __restrict__ cand_i,
    int32_t* __restrict__ cnt,
    int rows, int V, int num_steps, int NS) {
  const int ns = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
  if (row >= rows) return;

  const int req = idx_map[row];
  const int64_t step = step_ptr[0];
  const int base_v = V / NS, rem_v = V % NS;
  const int v_beg = ns * base_v + (ns < rem_v ? ns : rem_v);
  const int v_end = v_beg + base_v + (ns < rem_v ? 1 : 0);
  const InT* row_in = logits + (int64_t)row * V;
  const float ninf = neg_inf_val();

  if (req < 0) {
    if (ns == 0 && tid == 0) sampled_out[row] = 0;
    return;
  }

  const float req_t = temp_buf[req];
  const bool filt = (topk_buf[req] <= 64);
  CacheT* c_row = (WRITE_DENSE && draft_cache != nullptr)
      ? (draft_cache + (int64_t)req * cache_s0 + step * cache_s1) : nullptr;

  // GREEDY FAST-PATH (temp == 0.0f && no dense cache write): 1 single vectorized uint4 argmax pass!
  if (req_t == 0.0f && c_row == nullptr) {
    uint32_t lk = 0u;
    int32_t li = 0;
    local_argmax(row_in, v_beg, v_end, lk, li);
    if (tid == 0) {
      const int64_t c_off = ((int64_t)row * NS + ns) * 64;
      cand_k[c_off] = lk;
      cand_i[c_off] = li;
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
      int old = atomicAdd(&cnt[row], 1);
      if (old == NS - 1) {
        cnt[row] = 0;
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        uint32_t gk = 0u;
        int32_t gi = 0x7FFFFFFF;
        for (int s = 0; s < NS; ++s) {
          const int64_t o = ((int64_t)row * NS + s) * 64;
          uint32_t vk = cand_k[o];
          int32_t vi = cand_i[o];
          if (better(vk, vi, gk, gi)) { gk = vk; gi = vi; }
        }
        sampled_out[row] = (int64_t)gi;
        if (sp_valid != nullptr) sp_valid[(int64_t)req * num_steps + step] = filt ? 1 : 0;
        if (sp_ids != nullptr && filt) {
          const int64_t sp_base = ((int64_t)req * num_steps + step) * 64;
          sp_ids[sp_base] = gi;
          sp_vals[sp_base] = key2f(gk);
          for (int k = 1; k < 64; ++k) { sp_ids[sp_base + k] = -1; sp_vals[sp_base + k] = ninf; }
        }
      }
    }
    return;
  }

  if (!filt) {
    if (ns == 0 && tid == 0 && sp_valid != nullptr) sp_valid[(int64_t)req * num_steps + step] = 0;
    const float t = req_t;
    const int64_t seed = seeds_buf[req];
    const int64_t pos = pos_buf[row] + (1LL << 30);
    double best_sc = -1e300;
    int32_t best_id = 0;
    for (int v = v_beg + tid; v < v_end; v += 256) {
      float lv = ld_f(row_in + v);
      if constexpr (WRITE_DENSE) { if (c_row != nullptr) st_raw(c_row + v, lv); }
      double sc;
      if (t != 0.0f) {
        float lvt = lv / t;
        if constexpr (USE_FP64) sc = (double)lvt + gumbel_noise64(seed, pos, (uint32_t)v);
        else sc = (double)(lvt + gumbel_noise32(seed, pos, (uint32_t)v));
      } else {
        sc = (double)lv;
      }
      if (sc > best_sc || (sc == best_sc && v < best_id)) { best_sc = sc; best_id = v; }
    }
    __shared__ double s_sc[256];
    __shared__ int32_t s_id[256];
    s_sc[tid] = best_sc; s_id[tid] = best_id;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
      if (tid < off) {
        if (s_sc[tid + off] > s_sc[tid] || (s_sc[tid + off] == s_sc[tid] && s_id[tid + off] < s_id[tid])) {
          s_sc[tid] = s_sc[tid + off]; s_id[tid] = s_id[tid + off];
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      const int64_t c_off = ((int64_t)row * NS + ns) * 64;
      uint64_t bits = __builtin_bit_cast(uint64_t, s_sc[0]);
      cand_k[c_off + 0] = (uint32_t)(bits & 0xFFFFFFFFull);
      cand_k[c_off + 1] = (uint32_t)(bits >> 32);
      cand_i[c_off + 0] = s_id[0];
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
      int old = atomicAdd(&cnt[row], 1);
      if (old == NS - 1) {
        cnt[row] = 0;
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        double g_sc = -1e300; int32_t g_id = 0;
        for (int s = 0; s < NS; ++s) {
          const int64_t o = ((int64_t)row * NS + s) * 64;
          uint64_t b = (uint64_t)cand_k[o + 0] | ((uint64_t)cand_k[o + 1] << 32);
          double vsc = __builtin_bit_cast(double, b);
          int32_t vid = cand_i[o + 0];
          if (vsc > g_sc || (vsc == g_sc && vid < g_id)) { g_sc = vsc; g_id = vid; }
        }
        sampled_out[row] = (int64_t)g_id;
      }
    }
    return;
  }

  __shared__ uint32_t s_hist[2048];
  __shared__ uint32_t s_loc_k[64];
  __shared__ int32_t s_loc_i[64];
  __shared__ int s_cnt;

  select_local_topk(row_in, v_beg, v_end, 64, s_hist, &s_cnt, s_loc_k, s_loc_i);

  const int64_t c_off = ((int64_t)row * NS + ns) * 64;
  if (tid < 64) { cand_k[c_off + tid] = s_loc_k[tid]; cand_i[c_off + tid] = s_loc_i[tid]; }

  if constexpr (WRITE_DENSE) {
    if (c_row != nullptr) {
      for (int v = v_beg + tid; v < v_end; v += 256) st_raw(c_row + v, ninf);
    }
  }

  __syncthreads();
  __shared__ int s_is_last;
  if (tid == 0) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    int old = atomicAdd(&cnt[row], 1);
    int last = (old == NS - 1) ? 1 : 0;
    if (last) cnt[row] = 0;
    s_is_last = last;
  }
  __syncthreads();
  if (!s_is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  __shared__ uint32_t s_blk_sum[256];
  __shared__ int s_thr_bin, s_eq_n;
  __shared__ uint32_t s_gt_cnt;
  merge_global_top64<std::is_same_v<InT, uint16_t>>(
      cand_k + (int64_t)row * NS * 64, cand_i + (int64_t)row * NS * 64, NS * 64,
      s_hist, s_blk_sum, s_loc_k, s_loc_i, &s_cnt, &s_thr_bin, &s_gt_cnt, &s_eq_n);

  float* s_vals = reinterpret_cast<float*>(s_hist);
  float* s_scaled = reinterpret_cast<float*>(s_hist + 64);
  float* s_pr = reinterpret_cast<float*>(s_hist + 128);
  uint32_t* s_keep = s_hist + 192;
  int32_t* s_out_id = reinterpret_cast<int32_t*>(s_hist + 256);
  float* s_out_val = reinterpret_cast<float*>(s_hist + 320);

  if (tid == 0) {
    float t = req_t;
    float t_filt = (t < 1e-5f) ? 1.0f : t;
    float inv_t = 1.0f / t_filt;
    int kk = topk_buf[req];
    if (kk < 1) kk = 1;
    if (kk > 64) kk = 64;
    float p = topp_buf[req];
    for (int i = 0; i < 64; ++i) {
      s_vals[i] = key2f(s_loc_k[i]);
      s_scaled[i] = s_vals[i] * inv_t;
    }
    float kth = s_scaled[kk - 1], mx = -3.4e38f;
    for (int i = 0; i < 64; ++i) {
      s_keep[i] = ((s_loc_i[i] >= 0) && (s_scaled[i] >= kth)) ? 1u : 0u;
      if (s_keep[i] && s_scaled[i] > mx) mx = s_scaled[i];
    }
    float sum_e = 0.0f;
    for (int i = 0; i < 64; ++i) {
      if (s_keep[i]) { float e = expf(s_scaled[i] - mx); s_pr[i] = e; sum_e += e; }
      else s_pr[i] = 0.0f;
    }
    float inv_se = (sum_e > 0.0f) ? (1.0f / sum_e) : 0.0f, tail = 0.0f, thresh = 1.0f - p;
    for (int i = 63; i >= 0; --i) {
      tail += s_pr[i] * inv_se;
      bool keep_final = s_keep[i] && !((i > 0) && (tail <= thresh));
      if (keep_final) { s_out_id[i] = s_loc_i[i]; s_out_val[i] = s_vals[i]; }
      else { s_out_id[i] = -1; s_out_val[i] = ninf; }
    }
    if (sp_valid != nullptr) sp_valid[(int64_t)req * num_steps + step] = 1;
  }
  __syncthreads();

  double* s_gsc = reinterpret_cast<double*>(s_hist + 512);
  int32_t* s_gid = reinterpret_cast<int32_t*>(s_hist + 640);
  if (tid < 64) {
    int32_t id = s_out_id[tid];
    float rv = s_out_val[tid];
    if (sp_ids != nullptr) {
      const int64_t sp_off = ((int64_t)req * num_steps + step) * 64 + tid;
      sp_ids[sp_off] = id;
      sp_vals[sp_off] = rv;
    }
    if constexpr (WRITE_DENSE) {
      if (c_row != nullptr && id >= 0) st_raw(c_row + id, rv);
    }
    double sc = -1e300;
    int32_t gid = 0;
    if (id >= 0) {
      const float t = req_t;
      if (t != 0.0f) {
        const int64_t seed = seeds_buf[req];
        const int64_t pos = pos_buf[row] + (1LL << 30);
        float vt = rv / t;
        if constexpr (USE_FP64) sc = (double)vt + gumbel_noise64(seed, pos, (uint32_t)id);
        else sc = (double)(vt + gumbel_noise32(seed, pos, (uint32_t)id));
      } else {
        sc = (double)rv;
      }
      gid = id;
    }
    s_gsc[tid] = sc;
    s_gid[tid] = gid;
  }
  __syncthreads();

  for (int off = 32; off > 0; off >>= 1) {
    if (tid < off) {
      if (s_gsc[tid + off] > s_gsc[tid] || (s_gsc[tid + off] == s_gsc[tid] && s_gid[tid + off] < s_gid[tid])) {
        s_gsc[tid] = s_gsc[tid + off]; s_gid[tid] = s_gid[tid + off];
      }
    }
    __syncthreads();
  }
  if (tid == 0) sampled_out[row] = (int64_t)s_gid[0];
}

// ============================================================================
// V5-S4a: 1-launch 40-CU Sparse Target Sample Filter (ZERO full-vocab write!)
// ============================================================================
template <typename InT>
__global__ void __launch_bounds__(256) k_target_sample_sparse(
    const InT* __restrict__ logits,
    int32_t* __restrict__ tgt_ids,
    float* __restrict__ tgt_vals,
    float* __restrict__ tgt_lse,
    const int32_t* __restrict__ idx_map,
    const float* __restrict__ temp_buf,
    const float* __restrict__ minp_buf,
    const int32_t* __restrict__ topk_buf,
    const float* __restrict__ topp_buf,
    uint32_t* __restrict__ cand_k,
    int32_t* __restrict__ cand_i,
    int32_t* __restrict__ cnt,
    int rows, int V, int NS) {
  const int ns = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
  if (row >= rows) return;

  const int req = idx_map[row];
  const float ninf = neg_inf_val();
  if (req < 0) {
    if (ns == 0) {
      if (tid < 64) {
        tgt_ids[(int64_t)row * 64 + tid] = -1;
        tgt_vals[(int64_t)row * 64 + tid] = ninf;
      }
      if (tid == 0) tgt_lse[row] = ninf;
    }
    return;
  }

  const int base_v = V / NS, rem_v = V % NS;
  const int v_beg = ns * base_v + (ns < rem_v ? ns : rem_v);
  const int v_end = v_beg + base_v + (ns < rem_v ? 1 : 0);
  const InT* row_in = logits + (int64_t)row * V;
  const float req_t = temp_buf[req];

  // GREEDY FAST-PATH (temp == 0.0f): 1 single vectorized uint4 argmax pass!
  if (req_t == 0.0f) {
    uint32_t lk = 0u;
    int32_t li = 0;
    local_argmax(row_in, v_beg, v_end, lk, li);
    if (tid == 0) {
      const int64_t c_off = ((int64_t)row * NS + ns) * 64;
      cand_k[c_off] = lk;
      cand_i[c_off] = li;
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
      int old = atomicAdd(&cnt[row], 1);
      if (old == NS - 1) {
        cnt[row] = 0;
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        uint32_t gk = 0u;
        int32_t gi = 0x7FFFFFFF;
        for (int s = 0; s < NS; ++s) {
          const int64_t o = ((int64_t)row * NS + s) * 64;
          uint32_t vk = cand_k[o];
          int32_t vi = cand_i[o];
          if (better(vk, vi, gk, gi)) { gk = vk; gi = vi; }
        }
        int32_t* o_id = tgt_ids + (int64_t)row * 64;
        float* o_val = tgt_vals + (int64_t)row * 64;
        float gv = key2f(gk);
        o_id[0] = gi;
        o_val[0] = gv;
        for (int i = 1; i < 64; ++i) { o_id[i] = -1; o_val[i] = ninf; }
        tgt_lse[row] = gv;
      }
    }
    return;
  }

  __shared__ uint32_t s_hist[2048];
  __shared__ uint32_t s_loc_k[64];
  __shared__ int32_t s_loc_i[64];
  __shared__ int s_cnt;

  select_local_topk(row_in, v_beg, v_end, 64, s_hist, &s_cnt, s_loc_k, s_loc_i);

  const int64_t c_off = ((int64_t)row * NS + ns) * 64;
  if (tid < 64) { cand_k[c_off + tid] = s_loc_k[tid]; cand_i[c_off + tid] = s_loc_i[tid]; }

  __syncthreads();
  __shared__ int s_is_last;
  if (tid == 0) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    int old = atomicAdd(&cnt[row], 1);
    int last = (old == NS - 1) ? 1 : 0;
    if (last) cnt[row] = 0;
    s_is_last = last;
  }
  __syncthreads();
  if (!s_is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  __shared__ uint32_t s_blk_sum[256];
  __shared__ int s_thr_bin, s_eq_n;
  __shared__ uint32_t s_gt_cnt;
  merge_global_top64<std::is_same_v<InT, uint16_t>>(
      cand_k + (int64_t)row * NS * 64, cand_i + (int64_t)row * NS * 64, NS * 64,
      s_hist, s_blk_sum, s_loc_k, s_loc_i, &s_cnt, &s_thr_bin, &s_gt_cnt, &s_eq_n);

  if (tid == 0) {
    int32_t* o_id = tgt_ids + (int64_t)row * 64;
    float* o_val = tgt_vals + (int64_t)row * 64;
    float t = req_t;
    float* scaled = reinterpret_cast<float*>(s_hist + 64);
    float* pr = reinterpret_cast<float*>(s_hist + 128);
    uint32_t* keep = s_hist + 192;
    bool do_div = (t != 1.0f);
    for (int i = 0; i < 64; ++i) {
      float v = key2f(s_loc_k[i]);
      scaled[i] = do_div ? (v / t) : v;
    }
    float mp = minp_buf[req];
    if (mp > 0.0f) {
      float mx0 = scaled[0];
      if (mx0 > -3.4e38f) {
        float mthr = mx0 + logf(mp);
        for (int i = 0; i < 64; ++i) if (s_loc_i[i] < 0 || scaled[i] < mthr) scaled[i] = ninf;
      }
    }
    int kk = topk_buf[req];
    if (kk < 1) kk = 1;
    if (kk > 64) kk = 64;
    float kth = scaled[kk - 1], mx = -3.4e38f;
    for (int i = 0; i < 64; ++i) {
      keep[i] = ((s_loc_i[i] >= 0) && (scaled[i] >= kth) && (scaled[i] > -3.4e38f)) ? 1u : 0u;
      if (keep[i] && scaled[i] > mx) mx = scaled[i];
    }
    float p = topp_buf[req];
    if (p < 1.0f) {
      float sum_e = 0.0f;
      for (int i = 0; i < 64; ++i) {
        if (keep[i]) { float e = expf(scaled[i] - mx); pr[i] = e; sum_e += e; }
        else pr[i] = 0.0f;
      }
      float inv_se = (sum_e > 0.0f) ? (1.0f / sum_e) : 0.0f, tail = 0.0f, thresh = 1.0f - p;
      for (int i = 63; i >= 0; --i) {
        tail += pr[i] * inv_se;
        if (keep[i] && (i > 0) && (tail <= thresh)) keep[i] = 0u;
      }
    }
    float final_mx = -3.4e38f;
    for (int i = 0; i < 64; ++i) if (keep[i] && scaled[i] > final_mx) final_mx = scaled[i];
    float final_se = 0.0f;
    for (int i = 0; i < 64; ++i) {
      if (keep[i]) {
        o_id[i] = s_loc_i[i]; o_val[i] = scaled[i];
        final_se += expf(scaled[i] - final_mx);
      } else {
        o_id[i] = -1; o_val[i] = ninf;
      }
    }
    tgt_lse[row] = (final_se > 0.0f) ? (final_mx + logf(final_se)) : ninf;
  }
}

// ============================================================================
// V5-S4b: 1-launch 64-thread/req Sparse Rejection + Residual Resample Kernel!
// ============================================================================
template <bool HAS_DRAFT, bool USE_FP64>
__global__ void __launch_bounds__(64) k_sparse_rejection_sample(
    const int32_t* __restrict__ tgt_ids,
    const float* __restrict__ tgt_vals,
    const float* __restrict__ tgt_lse,
    const int32_t* __restrict__ d_ids,
    const float* __restrict__ d_vals,
    const int64_t* __restrict__ draft_toks,
    const int32_t* __restrict__ cu_logits,
    const int64_t* __restrict__ pos_buf,
    const int32_t* __restrict__ idx_map,
    const float* __restrict__ temp_buf,
    const int64_t* __restrict__ seeds_buf,
    int64_t* __restrict__ sampled_out,
    int32_t* __restrict__ num_sampled_out,
    int num_reqs, int num_spec_steps) {
  const int req_idx = blockIdx.x, tid = threadIdx.x;
  if (req_idx >= num_reqs) return;

  const int req_state = idx_map[req_idx];
  const int start_idx = cu_logits[req_idx];
  const int end_idx = cu_logits[req_idx + 1];
  const int num_draft = end_idx - start_idx - 1;
  const int64_t seed = seeds_buf[req_state];
  const float temp = temp_buf[req_state];
  const bool is_greedy = (temp == 0.0f);
  const float ninf = neg_inf_val();
  int64_t* s_out = sampled_out + (int64_t)req_idx * (num_spec_steps + 1);

  __shared__ float s_f[64];
  __shared__ int32_t s_d_id[64];
  __shared__ float s_d_val[64];

  bool verifying = true;
  int accepted = 0;
  float rej_tlse = 0.0f, rej_dlse = 0.0f;

  for (int i = 0; i < num_draft; ++i) {
    const int logit_idx = start_idx + i;
    const int64_t ds = draft_toks[logit_idx + 1];
    const bool valid_d = (ds >= 0);
    if (!is_greedy) verifying = verifying && valid_d;

    if (verifying) {
      if (is_greedy) {
        int64_t argmax_id = (int64_t)tgt_ids[(int64_t)logit_idx * 64 + 0];
        bool acc = valid_d && (argmax_id == ds);
        if (tid == 0) s_out[i] = acc ? ds : argmax_id;
        verifying = acc;
        if (acc) accepted++;
      } else {
        int32_t ti = tgt_ids[(int64_t)logit_idx * 64 + tid];
        float tv = (ti == (int32_t)ds) ? tgt_vals[(int64_t)logit_idx * 64 + tid] : ninf;
        s_f[tid] = tv;
        __syncthreads();
        for (int off = 32; off > 0; off >>= 1) {
          if (tid < off) s_f[tid] = fmaxf(s_f[tid], s_f[tid + off]);
          __syncthreads();
        }
        float t_logit = s_f[0];
        float t_lse = tgt_lse[logit_idx];
        float t_logp = t_logit - t_lse;

        float d_logp = 0.0f, d_lse = 0.0f;
        if constexpr (HAS_DRAFT) {
          const int64_t doff = ((int64_t)req_state * num_spec_steps + i) * 64 + tid;
          int32_t di = d_ids[doff];
          float dv = (di >= 0) ? (d_vals[doff] / temp) : ninf;
          s_f[tid] = dv;
          __syncthreads();
          for (int off = 32; off > 0; off >>= 1) {
            if (tid < off) s_f[tid] = fmaxf(s_f[tid], s_f[tid + off]);
            __syncthreads();
          }
          float d_mx = s_f[0];
          float de = (di >= 0 && d_mx > -3.4e38f) ? expf(dv - d_mx) : 0.0f;
          s_f[tid] = de;
          __syncthreads();
          for (int off = 32; off > 0; off >>= 1) {
            if (tid < off) s_f[tid] += s_f[tid + off];
            __syncthreads();
          }
          d_lse = (s_f[0] > 0.0f) ? (d_mx + logf(s_f[0])) : ninf;

          s_f[tid] = (di == (int32_t)ds) ? dv : ninf;
          __syncthreads();
          for (int off = 32; off > 0; off >>= 1) {
            if (tid < off) s_f[tid] = fmaxf(s_f[tid], s_f[tid + off]);
            __syncthreads();
          }
          d_logp = s_f[0] - d_lse;
        }
        float u = triton_philox_rand32(seed, pos_buf[logit_idx]);
        bool acc = (t_logp > logf(u) + d_logp);
        if (tid == 0) s_out[i] = ds;
        verifying = acc;
        if (acc) {
          accepted++;
        } else {
          rej_tlse = t_lse;
          rej_dlse = d_lse;
        }
      }
    }
  }

  const int res_tok_idx = start_idx + accepted;
  const bool is_bonus = (res_tok_idx == end_idx - 1);
  if (tid == 0) num_sampled_out[req_idx] = accepted + 1;

  if (is_greedy) {
    if (is_bonus && tid == 0) s_out[accepted] = (int64_t)tgt_ids[(int64_t)res_tok_idx * 64 + 0];
    return;
  }

  int64_t rej_draft_tok = is_bonus ? 0 : draft_toks[res_tok_idx + 1];
  bool valid_rej_d = (!is_bonus) && (rej_draft_tok >= 0);

  if constexpr (HAS_DRAFT) {
    if (valid_rej_d) {
      const int64_t doff = ((int64_t)req_state * num_spec_steps + accepted) * 64 + tid;
      int32_t di = d_ids[doff];
      s_d_id[tid] = di;
      s_d_val[tid] = (di >= 0) ? (d_vals[doff] / temp) : ninf;
    }
  }
  __syncthreads();

  int32_t ti = tgt_ids[(int64_t)res_tok_idx * 64 + tid];
  float tv = tgt_vals[(int64_t)res_tok_idx * 64 + tid];
  float res_logit = ninf;
  if (ti >= 0) {
    if (is_bonus || !valid_rej_d) {
      res_logit = tv;
    } else if constexpr (HAS_DRAFT) {
      float dv = ninf;
      for (int j = 0; j < 64; ++j) if (s_d_id[j] == ti) { dv = s_d_val[j]; break; }
      float t_lp = tv - rej_tlse;
      float d_lp = dv - rej_dlse;
      float ratio = expf(d_lp - t_lp);
      res_logit = (ratio < 1.0f) ? (t_lp + log1pf(-ratio)) : ninf;
    } else {
      res_logit = ((int64_t)ti != rej_draft_tok) ? tv : ninf;
    }
  }

  double score = -1e300;
  int32_t best_id = 0;
  if (ti >= 0 && res_logit > -3.4e38f) {
    int64_t p_res = pos_buf[res_tok_idx];
    if constexpr (USE_FP64) score = (double)res_logit + gumbel_noise64(seed, p_res, (uint32_t)ti);
    else score = (double)(res_logit + gumbel_noise32(seed, p_res, (uint32_t)ti));
    best_id = ti;
  }

  __shared__ double s_gsc[64];
  __shared__ int32_t s_gid[64];
  s_gsc[tid] = score;
  s_gid[tid] = best_id;
  __syncthreads();
  for (int off = 32; off > 0; off >>= 1) {
    if (tid < off) {
      if (s_gsc[tid + off] > s_gsc[tid] || (s_gsc[tid + off] == s_gsc[tid] && s_gid[tid + off] < s_gid[tid])) {
        s_gsc[tid] = s_gsc[tid + off]; s_gid[tid] = s_gid[tid + off];
      }
    }
    __syncthreads();
  }
  if (tid == 0) s_out[accepted] = (int64_t)s_gid[0];
}

// ============================================================================
// Host Wrappers (fast-path zero-alloc dtype checks)
// ============================================================================
static inline torch::Tensor as_contig_dtype(const torch::Tensor& t, torch::ScalarType dt) {
  if (t.scalar_type() == dt && t.is_contiguous()) return t;
  return t.to(dt).contiguous();
}

static torch::Tensor g_ck, g_ci, g_cnt;
static void ensure_scratch(const torch::Device& dev, int64_t rows, int64_t ns) {
  int64_t need = rows * ns * 64;
  if (!g_ck.defined() || g_ck.device() != dev || g_ck.numel() < need) {
    int64_t cap = std::max<int64_t>(need, 64 * 40 * 64);
    g_ck = torch::empty({cap}, torch::TensorOptions().dtype(torch::kInt32).device(dev));
    g_ci = torch::empty({cap}, torch::TensorOptions().dtype(torch::kInt32).device(dev));
  }
  if (!g_cnt.defined() || g_cnt.device() != dev || g_cnt.numel() < rows) {
    g_cnt = torch::zeros({std::max<int64_t>(rows, 256)}, torch::TensorOptions().dtype(torch::kInt32).device(dev));
  }
}
static inline int choose_ns(int rows, int V) {
  int ns = std::max(1, std::min(40, (40 + rows - 1) / rows));
  int max_ns = std::max(1, V / 256);
  return std::min(ns, max_ns);
}

torch::Tensor q38_filter_gumbel_sample_native(
    torch::Tensor logits,
    c10::optional<torch::Tensor> draft_cache_opt,
    torch::Tensor draft_step,
    c10::optional<torch::Tensor> sp_ids_opt,
    c10::optional<torch::Tensor> sp_vals_opt,
    c10::optional<torch::Tensor> sp_valid_opt,
    torch::Tensor idx_mapping,
    torch::Tensor temperature,
    torch::Tensor top_k,
    torch::Tensor top_p,
    torch::Tensor seeds,
    torch::Tensor pos,
    int64_t num_steps,
    bool write_dense_cache,
    bool use_fp64) {
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.is_contiguous());
  const int rows = (int)logits.size(0), V = (int)logits.size(1);
  auto sampled = torch::empty({rows}, logits.options().dtype(torch::kInt64));
  if (rows == 0) return sampled;

  auto idx_i32 = as_contig_dtype(idx_mapping, torch::kInt32);
  auto t_f32 = as_contig_dtype(temperature, torch::kFloat32);
  auto k_i32 = as_contig_dtype(top_k, torch::kInt32);
  auto p_f32 = as_contig_dtype(top_p, torch::kFloat32);
  auto s_i64 = as_contig_dtype(seeds, torch::kInt64);
  auto pos_i64 = as_contig_dtype(pos, torch::kInt64);
  auto step_i64 = as_contig_dtype(draft_step, torch::kInt64);

  int32_t* sp_i = sp_ids_opt.has_value() ? sp_ids_opt->data_ptr<int32_t>() : nullptr;
  float* sp_v = sp_vals_opt.has_value() ? sp_vals_opt->data_ptr<float>() : nullptr;
  int32_t* sp_ok = sp_valid_opt.has_value() ? sp_valid_opt->data_ptr<int32_t>() : nullptr;

  int64_t cs0 = 0, cs1 = 0;
  bool has_cache = draft_cache_opt.has_value();
  if (has_cache) { cs0 = draft_cache_opt->stride(0); cs1 = draft_cache_opt->stride(1); }

  const int NS = choose_ns(rows, V);
  ensure_scratch(logits.device(), rows, NS);
  hipStream_t stream = at::hip::getCurrentHIPStream().stream();
  uint32_t* ck = reinterpret_cast<uint32_t*>(g_ck.data_ptr<int32_t>());
  int32_t* ci = g_ci.data_ptr<int32_t>();
  int32_t* cp = g_cnt.data_ptr<int32_t>();

  auto launch = [&](auto in_tag, auto cache_tag, auto wd_tag, auto fp64_tag) {
    using InT = decltype(in_tag);
    using CacheT = decltype(cache_tag);
    constexpr bool WD = decltype(wd_tag)::value;
    constexpr bool FP64 = decltype(fp64_tag)::value;
    CacheT* cptr = (WD && has_cache) ? reinterpret_cast<CacheT*>(draft_cache_opt->data_ptr()) : nullptr;
    k_q38_filter_gumbel_sample<InT, CacheT, WD, FP64><<<dim3(NS, rows), dim3(256), 0, stream>>>(
        reinterpret_cast<const InT*>(logits.data_ptr()), cptr, cs0, cs1,
        step_i64.data_ptr<int64_t>(), sp_i, sp_v, sp_ok, sampled.data_ptr<int64_t>(),
        idx_i32.data_ptr<int32_t>(), t_f32.data_ptr<float>(), k_i32.data_ptr<int32_t>(),
        p_f32.data_ptr<float>(), s_i64.data_ptr<int64_t>(), pos_i64.data_ptr<int64_t>(),
        ck, ci, cp, rows, V, (int)num_steps, NS);
  };

  bool in_bf16 = (logits.dtype() == torch::kBFloat16);
  bool c_bf16 = has_cache ? (draft_cache_opt->dtype() == torch::kBFloat16) : in_bf16;

  if (in_bf16 && c_bf16) {
    if (write_dense_cache) {
      if (use_fp64) { launch(uint16_t{}, uint16_t{}, std::true_type{}, std::true_type{}); }
      else { launch(uint16_t{}, uint16_t{}, std::true_type{}, std::false_type{}); }
    } else {
      if (use_fp64) { launch(uint16_t{}, uint16_t{}, std::false_type{}, std::true_type{}); }
      else { launch(uint16_t{}, uint16_t{}, std::false_type{}, std::false_type{}); }
    }
  } else if (in_bf16 && !c_bf16) {
    if (write_dense_cache) {
      if (use_fp64) { launch(uint16_t{}, float{}, std::true_type{}, std::true_type{}); }
      else { launch(uint16_t{}, float{}, std::true_type{}, std::false_type{}); }
    } else {
      if (use_fp64) { launch(uint16_t{}, float{}, std::false_type{}, std::true_type{}); }
      else { launch(uint16_t{}, float{}, std::false_type{}, std::false_type{}); }
    }
  } else {
    TORCH_CHECK(logits.dtype() == torch::kFloat32);
    if (write_dense_cache) {
      if (use_fp64) { launch(float{}, float{}, std::true_type{}, std::true_type{}); }
      else { launch(float{}, float{}, std::true_type{}, std::false_type{}); }
    } else {
      if (use_fp64) { launch(float{}, float{}, std::false_type{}, std::true_type{}); }
      else { launch(float{}, float{}, std::false_type{}, std::false_type{}); }
    }
  }
  return sampled;
}

std::vector<torch::Tensor> target_sample_sparse_native(
    torch::Tensor logits, torch::Tensor idx_mapping, torch::Tensor temperature,
    torch::Tensor min_p, torch::Tensor top_k, torch::Tensor top_p) {
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.is_contiguous());
  const int rows = (int)logits.size(0), V = (int)logits.size(1);
  auto tgt_ids = torch::empty({rows, 64}, logits.options().dtype(torch::kInt32));
  auto tgt_vals = torch::empty({rows, 64}, logits.options().dtype(torch::kFloat32));
  auto tgt_lse = torch::empty({rows}, logits.options().dtype(torch::kFloat32));
  if (rows == 0) return {tgt_ids, tgt_vals, tgt_lse};

  auto idx_i32 = as_contig_dtype(idx_mapping, torch::kInt32);
  auto t_f32 = as_contig_dtype(temperature, torch::kFloat32);
  auto mp_f32 = as_contig_dtype(min_p, torch::kFloat32);
  auto k_i32 = as_contig_dtype(top_k, torch::kInt32);
  auto p_f32 = as_contig_dtype(top_p, torch::kFloat32);

  const int NS = choose_ns(rows, V);
  ensure_scratch(logits.device(), rows, NS);
  hipStream_t stream = at::hip::getCurrentHIPStream().stream();
  uint32_t* ck = reinterpret_cast<uint32_t*>(g_ck.data_ptr<int32_t>());
  int32_t* ci = g_ci.data_ptr<int32_t>();
  int32_t* cp = g_cnt.data_ptr<int32_t>();

  if (logits.dtype() == torch::kBFloat16) {
    k_target_sample_sparse<uint16_t><<<dim3(NS, rows), dim3(256), 0, stream>>>(
        reinterpret_cast<const uint16_t*>(logits.data_ptr()),
        tgt_ids.data_ptr<int32_t>(), tgt_vals.data_ptr<float>(), tgt_lse.data_ptr<float>(),
        idx_i32.data_ptr<int32_t>(), t_f32.data_ptr<float>(), mp_f32.data_ptr<float>(),
        k_i32.data_ptr<int32_t>(), p_f32.data_ptr<float>(), ck, ci, cp, rows, V, NS);
  } else {
    TORCH_CHECK(logits.dtype() == torch::kFloat32);
    k_target_sample_sparse<float><<<dim3(NS, rows), dim3(256), 0, stream>>>(
        logits.data_ptr<float>(),
        tgt_ids.data_ptr<int32_t>(), tgt_vals.data_ptr<float>(), tgt_lse.data_ptr<float>(),
        idx_i32.data_ptr<int32_t>(), t_f32.data_ptr<float>(), mp_f32.data_ptr<float>(),
        k_i32.data_ptr<int32_t>(), p_f32.data_ptr<float>(), ck, ci, cp, rows, V, NS);
  }
  return {tgt_ids, tgt_vals, tgt_lse};
}

std::vector<torch::Tensor> sparse_rejection_sample_native(
    torch::Tensor tgt_ids, torch::Tensor tgt_vals, torch::Tensor tgt_lse,
    c10::optional<torch::Tensor> d_ids_opt, c10::optional<torch::Tensor> d_vals_opt,
    torch::Tensor draft_sampled, torch::Tensor cu_num_logits, torch::Tensor pos,
    torch::Tensor idx_mapping, torch::Tensor temperature, torch::Tensor seeds,
    int64_t num_spec_steps, bool use_fp64) {
  const int num_reqs = (int)cu_num_logits.size(0) - 1;
  auto sampled = torch::empty({num_reqs, num_spec_steps + 1}, tgt_ids.options().dtype(torch::kInt64));
  auto num_sampled = torch::empty({num_reqs}, tgt_ids.options().dtype(torch::kInt32));
  if (num_reqs == 0) return {sampled, num_sampled};

  auto ds_i64 = as_contig_dtype(draft_sampled, torch::kInt64);
  auto cu_i32 = as_contig_dtype(cu_num_logits, torch::kInt32);
  auto pos_i64 = as_contig_dtype(pos, torch::kInt64);
  auto idx_i32 = as_contig_dtype(idx_mapping, torch::kInt32);
  auto t_f32 = as_contig_dtype(temperature, torch::kFloat32);
  auto s_i64 = as_contig_dtype(seeds, torch::kInt64);

  bool has_d = d_ids_opt.has_value() && d_vals_opt.has_value();
  const int32_t* di = has_d ? d_ids_opt->data_ptr<int32_t>() : nullptr;
  const float* dv = has_d ? d_vals_opt->data_ptr<float>() : nullptr;
  hipStream_t stream = at::hip::getCurrentHIPStream().stream();

  if (has_d) {
    if (use_fp64) {
      k_sparse_rejection_sample<true, true><<<dim3(num_reqs), dim3(64), 0, stream>>>(
          tgt_ids.data_ptr<int32_t>(), tgt_vals.data_ptr<float>(), tgt_lse.data_ptr<float>(),
          di, dv, ds_i64.data_ptr<int64_t>(), cu_i32.data_ptr<int32_t>(), pos_i64.data_ptr<int64_t>(),
          idx_i32.data_ptr<int32_t>(), t_f32.data_ptr<float>(), s_i64.data_ptr<int64_t>(),
          sampled.data_ptr<int64_t>(), num_sampled.data_ptr<int32_t>(), num_reqs, (int)num_spec_steps);
    } else {
      k_sparse_rejection_sample<true, false><<<dim3(num_reqs), dim3(64), 0, stream>>>(
          tgt_ids.data_ptr<int32_t>(), tgt_vals.data_ptr<float>(), tgt_lse.data_ptr<float>(),
          di, dv, ds_i64.data_ptr<int64_t>(), cu_i32.data_ptr<int32_t>(), pos_i64.data_ptr<int64_t>(),
          idx_i32.data_ptr<int32_t>(), t_f32.data_ptr<float>(), s_i64.data_ptr<int64_t>(),
          sampled.data_ptr<int64_t>(), num_sampled.data_ptr<int32_t>(), num_reqs, (int)num_spec_steps);
    }
  } else {
    if (use_fp64) {
      k_sparse_rejection_sample<false, true><<<dim3(num_reqs), dim3(64), 0, stream>>>(
          tgt_ids.data_ptr<int32_t>(), tgt_vals.data_ptr<float>(), tgt_lse.data_ptr<float>(),
          nullptr, nullptr, ds_i64.data_ptr<int64_t>(), cu_i32.data_ptr<int32_t>(), pos_i64.data_ptr<int64_t>(),
          idx_i32.data_ptr<int32_t>(), t_f32.data_ptr<float>(), s_i64.data_ptr<int64_t>(),
          sampled.data_ptr<int64_t>(), num_sampled.data_ptr<int32_t>(), num_reqs, (int)num_spec_steps);
    } else {
      k_sparse_rejection_sample<false, false><<<dim3(num_reqs), dim3(64), 0, stream>>>(
          tgt_ids.data_ptr<int32_t>(), tgt_vals.data_ptr<float>(), tgt_lse.data_ptr<float>(),
          nullptr, nullptr, ds_i64.data_ptr<int64_t>(), cu_i32.data_ptr<int32_t>(), pos_i64.data_ptr<int64_t>(),
          idx_i32.data_ptr<int32_t>(), t_f32.data_ptr<float>(), s_i64.data_ptr<int64_t>(),
          sampled.data_ptr<int64_t>(), num_sampled.data_ptr<int32_t>(), num_reqs, (int)num_spec_steps);
    }
  }
  return {sampled, num_sampled};
}

int64_t spec_cnt_max() { return g_cnt.defined() ? g_cnt.max().item<int64_t>() : 0; }

PYBIND11_MODULE(htr_spec_sample, m) {
  m.def("q38_filter_gumbel_sample_native", &q38_filter_gumbel_sample_native, "Wave6: vectorized BF16 2-pass + greedy fast-path draft filter + Gumbel sample");
  m.def("target_sample_sparse_native", &target_sample_sparse_native, "Wave6: vectorized BF16 2-pass + greedy fast-path sparse top-64 target filter");
  m.def("sparse_rejection_sample_native", &sparse_rejection_sample_native, "V5-S4b: 1-launch 64-thread/req sparse rejection + residual resample");
  m.def("spec_cnt_max", &spec_cnt_max, "Check arrival counter self-reset");
}
