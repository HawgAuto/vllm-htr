// qsa_cache_ops_aot.hip — AOT gfx1151 Wave32 HIP replacements for Triton
// _store_qsa_rows_kernel and _compress_qsa_groups_kernel in qwen4_exp/amd/ops/qsa.py.
// 0 LDS, 0 spills, 16 waves/SIMD!
#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float bf16_to_f32(uint16_t u) { return __uint_as_float(((uint32_t)u) << 16); }
__device__ __forceinline__ uint16_t f32_to_bf16_rne(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ int64_t load_idx(const void* p, int row, int is_i64) {
  return is_i64 ? ((const int64_t*)p)[row] : (int64_t)((const int32_t*)p)[row];
}

// Grid: ((num_rows + 3) / 4, 1, 1), Block: (128, 1, 1) -> 4 waves = 4 rows/WG
template <typename ElemT>
__device__ __forceinline__ void qsa_store_rows_impl(
    ElemT* __restrict__ cache_ptr,
    const void* __restrict__ slots_ptr,
    const ElemT* __restrict__ rows_ptr,
    int64_t sc_blk, int64_t sc_tok, int64_t sc_dim,
    int64_t sr_row, int64_t sr_dim,
    int num_rows, int num_blocks, int page_size, int width, int slot_is_i64) {
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 4 + wave;
  if (row >= num_rows) return;
  const int64_t slot = load_idx(slots_ptr, row, slot_is_i64);
  if (slot < 0 || slot >= (int64_t)num_blocks * page_size) return;
  const int64_t blk = slot / page_size;
  const int64_t tok = slot - blk * page_size;
  ElemT* dst = cache_ptr + blk * sc_blk + tok * sc_tok;
  const ElemT* src = rows_ptr + (int64_t)row * sr_row;
  for (int d = lane; d < width; d += 32) {
    dst[(int64_t)d * sc_dim] = src[(int64_t)d * sr_dim];
  }
}

extern "C" __global__ void __launch_bounds__(128) qsa_store_rows_bf16(
    uint16_t* cache_ptr, const void* slots_ptr, const uint16_t* rows_ptr,
    int64_t sc_blk, int64_t sc_tok, int64_t sc_dim,
    int64_t sr_row, int64_t sr_dim,
    int num_rows, int num_blocks, int page_size, int width, int slot_is_i64) {
  qsa_store_rows_impl<uint16_t>(
      cache_ptr, slots_ptr, rows_ptr, sc_blk, sc_tok, sc_dim, sr_row, sr_dim,
      num_rows, num_blocks, page_size, width, slot_is_i64);
}

extern "C" __global__ void __launch_bounds__(128) qsa_store_rows_i64(
    int64_t* cache_ptr, const void* slots_ptr, const int64_t* rows_ptr,
    int64_t sc_blk, int64_t sc_tok, int64_t sc_dim,
    int64_t sr_row, int64_t sr_dim,
    int num_rows, int num_blocks, int page_size, int width, int slot_is_i64) {
  qsa_store_rows_impl<int64_t>(
      cache_ptr, slots_ptr, rows_ptr, sc_blk, sc_tok, sc_dim, sr_row, sr_dim,
      num_rows, num_blocks, page_size, width, slot_is_i64);
}

// Grid: ((num_rows + 3) / 4, 1, 1), Block: (128, 1, 1) -> 4 waves = 4 rows/WG
// flags: bit0=load_rope_positions, bit1=logical_pos_is_i64, bit2=comp_slot_is_i64
extern "C" __global__ void __launch_bounds__(128) qsa_compress_groups_bf16(
    const uint16_t* __restrict__ raw_keys_ptr,
    const int64_t* __restrict__ raw_positions_ptr,
    const uint16_t* __restrict__ comp_state_ptr,
    const int64_t* __restrict__ rope_cache_ptr,
    const int32_t* __restrict__ comp_table_ptr,
    const int32_t* __restrict__ token_to_req_ptr,
    const int32_t* __restrict__ query_start_loc_ptr,
    const void* __restrict__ logical_positions_ptr,
    const void* __restrict__ compressed_slots_ptr,
    uint16_t* __restrict__ pooled_ptr,
    int64_t* __restrict__ first_positions_ptr,
    int64_t sr_row, int64_t sr_dim,
    int64_t srp_row, int64_t srp_dim,
    int64_t scs_blk, int64_t scs_tok, int64_t scs_dim,
    int64_t srope_blk, int64_t srope_tok, int64_t srope_dim,
    int64_t sct_req,
    int64_t sp_row, int64_t sp_dim,
    int64_t sfp_row, int64_t sfp_dim,
    int num_rows, int num_cs_blocks, int num_requests,
    int cs_size, int compress_ratio, int head_dim, int flags) {

  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 4 + wave;
  if (row >= num_rows) return;

  const bool load_rope = (flags & 1) != 0;
  const int lpos_i64 = (flags >> 1) & 1;
  const int cslot_i64 = (flags >> 2) & 1;

  const int req = token_to_req_ptr[row];
  const int64_t end_pos = load_idx(logical_positions_ptr, row, lpos_i64);
  const int64_t comp_slot = load_idx(compressed_slots_ptr, row, cslot_i64);
  const bool valid_req = (req >= 0) && (req < num_requests);
  const int q_start = valid_req ? query_start_loc_ptr[req] : 0;
  const int q_end = valid_req ? query_start_loc_ptr[req + 1] : 0;
  const int64_t chunk_start_pos = end_pos - (int64_t)(row - q_start);
  const int cs_blk = valid_req ? comp_table_ptr[(int64_t)req * sct_req] : -1;
  const bool valid_cs_blk = (cs_blk >= 0) && (cs_blk < num_cs_blocks);
  const bool valid_row = valid_req && (row >= q_start) && (row < q_end)
                         && (end_pos >= (int64_t)(compress_ratio - 1)) && (comp_slot >= 0);

  const float inv_cr = 1.0f / (float)compress_ratio;
  for (int d = lane; d < head_dim; d += 32) {
    float acc = 0.0f;
    if (valid_row) {
      for (int g = 0; g < compress_ratio; ++g) {
        int64_t pos = end_pos - (int64_t)(compress_ratio - 1 - g);
        if (pos >= chunk_start_pos) {
          int64_t rr = (int64_t)q_start + pos - chunk_start_pos;
          if (rr >= q_start && rr < q_end && rr < num_rows) {
            acc += bf16_to_f32(raw_keys_ptr[rr * sr_row + (int64_t)d * sr_dim]);
          }
        } else if (valid_cs_blk) {
          int64_t ring = pos % (int64_t)cs_size;
          if (ring < 0) ring += cs_size;
          acc += bf16_to_f32(comp_state_ptr[(int64_t)cs_blk * scs_blk + ring * scs_tok + (int64_t)d * scs_dim]);
        }
      }
    }
    pooled_ptr[(int64_t)row * sp_row + (int64_t)d * sp_dim] = f32_to_bf16_rne(acc * inv_cr);
  }

  if (lane < 3) {
    int64_t first_pos = end_pos - (int64_t)compress_ratio + 1;
    int64_t pval = 0;
    if (valid_row) {
      if (load_rope) {
        if (first_pos >= chunk_start_pos) {
          int64_t rfr = (int64_t)q_start + first_pos - chunk_start_pos;
          if (rfr >= q_start && rfr < q_end && rfr < num_rows) {
            pval = raw_positions_ptr[rfr * srp_row + (int64_t)lane * srp_dim];
          }
        } else if (valid_cs_blk) {
          int64_t ring = first_pos % (int64_t)cs_size;
          if (ring < 0) ring += cs_size;
          pval = rope_cache_ptr[(int64_t)cs_blk * srope_blk + ring * srope_tok + (int64_t)lane * srope_dim];
        }
      } else {
        pval = first_pos;
      }
    }
    first_positions_ptr[(int64_t)row * sfp_row + (int64_t)lane * sfp_dim] = pval;
  }
}
