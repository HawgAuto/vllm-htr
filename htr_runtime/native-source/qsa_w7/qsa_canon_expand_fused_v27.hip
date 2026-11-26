// qsa_canon_expand_fused.hip — Clean-slate gfx1151 Wave32 Fused QSA Canonicalize + Expand Megakernel (V5-I2)
// Replaces 5 launches + 4 temp allocs + 2 full DRAM scans of logits per QSA layer (13 layers/step = -52 launches/step!):
//   (_kth + _count2 + _scatter + blocks.copy_ + _expand_qsa_indices_kernel) -> 1 single HIP launch!
// Bitwise identical to direct_qsa_canonical.canonicalize_selected_blocks + expand_qsa_block_indices_cuda,
// including exact lowest-column tie-breaking via single-pass Wave32 ballot compaction + O(log K) parallel merge in LDS!
#include <hip/hip_runtime.h>
#include <stdint.h>

#define NT 256
#define MAX_K 512

// WAVE7_CE_DPP: zero-LDS RDNA 3.5 DPP wave_min & wave_sum_i32
__device__ __forceinline__ float wave_min(float v) {
  asm volatile(
    "v_min_f32_dpp %0, %0, %0 row_xmask:1 row_mask:0xf bank_mask:0xf fi:1\n\t"
    "v_min_f32_dpp %0, %0, %0 row_xmask:2 row_mask:0xf bank_mask:0xf fi:1\n\t"
    "v_min_f32_dpp %0, %0, %0 row_xmask:4 row_mask:0xf bank_mask:0xf fi:1\n\t"
    "v_min_f32_dpp %0, %0, %0 row_xmask:8 row_mask:0xf bank_mask:0xf fi:1"
    : "+v"(v));
  float hi;
  asm volatile("v_permlanex16_b32 %0, %1, 0, 0" : "=&v"(hi) : "v"(v));
  return fminf(v, hi);
}

__device__ __forceinline__ int wave_sum_i32(int v) {
  asm volatile(
    "v_add_nc_u32_dpp %0, %0, %0 row_xmask:1 row_mask:0xf bank_mask:0xf fi:1\n\t"
    "v_add_nc_u32_dpp %0, %0, %0 row_xmask:2 row_mask:0xf bank_mask:0xf fi:1\n\t"
    "v_add_nc_u32_dpp %0, %0, %0 row_xmask:4 row_mask:0xf bank_mask:0xf fi:1\n\t"
    "v_add_nc_u32_dpp %0, %0, %0 row_xmask:8 row_mask:0xf bank_mask:0xf fi:1"
    : "+v"(v));
  int hi;
  asm volatile("v_permlanex16_b32 %0, %1, 0, 0" : "=&v"(hi) : "v"(v));
  return v + hi;
}

__device__ __forceinline__ int lower_bound_lds(const int32_t* arr, int n, int target) {
  int lo = 0, hi = n;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (arr[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

extern "C" __global__ __launch_bounds__(NT) void qsa_canon_expand_fused(
    const int32_t* __restrict__ blocks,           // [rows, K] i32 (K <= 512)
    const float* __restrict__ logits,             // [rows, W] f32
    const int32_t* __restrict__ visible_blocks,   // [rows] i32
    const int64_t* __restrict__ query_positions,  // [rows] i64
    const int32_t* __restrict__ sequence_lengths, // [num_requests] i32
    const int32_t* __restrict__ token_to_req,     // [rows] i32
    int32_t* __restrict__ out,                    // [rows, output_width] i32
    int64_t bs0, int64_t bs1,
    int64_t ls0, int64_t ls1,
    int64_t vs,
    int64_t so0, int64_t so1,
    int rows, int W, int K, int num_requests,
    int compress_ratio, int output_width) {
  __shared__ int32_t s_gt[MAX_K];
  __shared__ int32_t s_eq[MAX_K];
  __shared__ int32_t s_blk[MAX_K];
  __shared__ float s_wmin[8];
  __shared__ int32_t s_wcnt[8];
  __shared__ int32_t s_wgt[8];
  __shared__ int32_t s_weq[8];

  const int row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x;
  const int wv = tid >> 5;
  const int lane = tid & 31;

  const int v_raw = visible_blocks[(int64_t)row * vs];
  const int v = (v_raw <= 0) ? 0 : (v_raw < W ? v_raw : W);
  const float pos_inf = __builtin_inff();
  const float neg_inf = -__builtin_inff();

  // 1. Find kth threshold and nsel over blocks[row, 0..K-1]
  float my_min = pos_inf;
  int my_cnt = 0;
  for (int ki = tid; ki < K; ki += NT) {
    const int sel = blocks[(int64_t)row * bs0 + (int64_t)ki * bs1];
    if (sel >= 0 && sel < v) {
      const float sc = logits[(int64_t)row * ls0 + (int64_t)sel * ls1];
      my_min = fminf(my_min, sc);
      ++my_cnt;
    }
  }
  const float wm = wave_min(my_min);
  const int wc = wave_sum_i32(my_cnt);
  if (lane == 0) {
    s_wmin[wv] = wm;
    s_wcnt[wv] = wc;
  }
  __syncthreads();

  float kth = fminf(fminf(fminf(s_wmin[0], s_wmin[1]), fminf(s_wmin[2], s_wmin[3])),
                    fminf(fminf(s_wmin[4], s_wmin[5]), fminf(s_wmin[6], s_wmin[7])));
  int nsel = ((s_wcnt[0] + s_wcnt[1]) + (s_wcnt[2] + s_wcnt[3]))
           + ((s_wcnt[4] + s_wcnt[5]) + (s_wcnt[6] + s_wcnt[7]));

  // Initialize s_blk to -1
  for (int ki = tid; ki < K; ki += NT) s_blk[ki] = -1;

  // 2. Single-pass coalesced scan of logits[row, 0..v-1] with Wave32 ballot compaction!
  int n_gt = 0;
  int n_eq = 0;
  if (nsel > 0 && v > 0) {
    const float* lrow = logits + (int64_t)row * ls0;
    const uint32_t lane_mask = (1u << lane) - 1u;
    for (int c0 = 0; c0 < v; c0 += NT) {
      const int c = c0 + tid;
      const bool valid = (c < v);
      const float sc = valid ? lrow[(int64_t)c * ls1] : neg_inf;
      const bool is_gt = valid && (sc > kth);
      const bool is_eq = valid && (sc == kth);

      const uint32_t bm_gt = __builtin_amdgcn_ballot_w32(is_gt);
      const uint32_t bm_eq = __builtin_amdgcn_ballot_w32(is_eq);
      if (lane == 0) {
        s_wgt[wv] = __popc(bm_gt);
        s_weq[wv] = __popc(bm_eq);
      }
      __syncthreads();

      int pre_gt = 0, tot_gt = 0;
      int pre_eq = 0, tot_eq = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (j < wv) { pre_gt += s_wgt[j]; pre_eq += s_weq[j]; }
        tot_gt += s_wgt[j];
        tot_eq += s_weq[j];
      }

      if (is_gt) {
        const int pos = n_gt + pre_gt + __popc(bm_gt & lane_mask);
        if (pos < MAX_K) s_gt[pos] = c;
      }
      if (is_eq) {
        const int pos = n_eq + pre_eq + __popc(bm_eq & lane_mask);
        if (pos < MAX_K) s_eq[pos] = c;
      }
      n_gt += tot_gt;
      n_eq += tot_eq;
      __syncthreads();
    }

    // 3. Merge strictly sorted disjoint arrays s_gt[0..ng-1] & s_eq[0..ne-1] into s_blk[0..K-1]
    const int ng = (n_gt < K) ? n_gt : K;
    const int need = (nsel > ng) ? (nsel - ng) : 0;
    const int ne_cap = K - ng;
    const int ne_avail = (n_eq < ne_cap) ? n_eq : ne_cap;
    const int ne = (need < ne_avail) ? need : ne_avail;

    if (ne == 0) {
      for (int idx = tid; idx < ng; idx += NT) s_blk[idx] = s_gt[idx];
    } else {
      for (int idx = tid; idx < ng; idx += NT) {
        const int val = s_gt[idx];
        const int rk = idx + lower_bound_lds(s_eq, ne, val);
        if (rk < K) s_blk[rk] = val;
      }
      for (int idx = tid; idx < ne; idx += NT) {
        const int val = s_eq[idx];
        const int rk = idx + lower_bound_lds(s_gt, ng, val);
        if (rk < K) s_blk[rk] = val;
      }
    }
  }
  __syncthreads();

  // 4. Fused expansion into out[row, 0..output_width-1] (exact _expand_qsa_indices_kernel rules)
  const int64_t qpos = query_positions[row];
  const int req = token_to_req[row];
  const int32_t sl = (req >= 0 && req < num_requests) ? sequence_lengths[req] : 0;
  const int64_t ca = (qpos + 1) / (int64_t)compress_ratio;
  const int64_t cb = (int64_t)(sl / compress_ratio);
  int64_t complete_blocks = (ca < cb) ? ca : cb;
  if (complete_blocks > (int64_t)K) complete_blocks = (int64_t)K;
  if (complete_blocks < 0) complete_blocks = 0;
  const int64_t expanded_count = complete_blocks * (int64_t)compress_ratio;
  const int64_t tail_start = ((qpos + 1) / (int64_t)compress_ratio) * (int64_t)compress_ratio;
  const int64_t tail_count = (qpos + 1) - tail_start;

  int32_t* orow = out + (int64_t)row * so0;
  for (int col = tid; col < output_width; col += NT) {
    const bool is_exp = ((int64_t)col < expanded_count);
    int64_t token;
    if (is_exp) {
      const int br = col / compress_ratio;
      const int off = col - br * compress_ratio;
      const int blk = s_blk[br];
      token = (int64_t)blk * (int64_t)compress_ratio + (int64_t)off;
    } else {
      token = tail_start + ((int64_t)col - expanded_count);
    }
    const int64_t toff = (int64_t)col - expanded_count;
    const bool is_tail = (!is_exp) && (toff >= 0) && (toff < tail_count) && (toff < (int64_t)(compress_ratio - 1));
    const bool ok = (is_exp || is_tail) && (token >= 0) && (token < (int64_t)sl);
    orow[(int64_t)col * so1] = ok ? (int32_t)token : -1;
  }
}
