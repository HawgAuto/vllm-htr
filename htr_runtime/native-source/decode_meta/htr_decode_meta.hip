// htr_decode_meta.hip — Zero-Triton-overhead gfx1151 HIP decode bookkeeping & metadata kernels (v0.2.7)
// Eliminates 18+ Triton launches/step (~3.5 ms/step Python dispatch + bubbles between HIP graph replays):
//   1. build_qsa_metadata (5.5x/step)
//   2. compute_slot_mappings (2.8x/step) & gather_block_tables (1x/step)
//   3. prepare_pos_seq_lens, combine_sampled_and_draft_tokens, get_num_sampled_and_rejected,
//      post_update, expand_idx_mapping (5x/step)
//   4. prepare_prefill_inputs, prepare_decode_inputs (2x/step)
//   5. prepare_rope_positions (1x/step)
//   6. scatter_num_accepted, fill_num_accepted (1x/step)
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

// ============================================================================
// 1. build_qsa_metadata (exact 1:1 parity with _build_qsa_metadata_kernel)
// ============================================================================
__global__ void k_build_qsa_metadata(
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ seq_lens,
    const int64_t* __restrict__ common_slot_mapping,
    const int32_t* __restrict__ block_table,
    int32_t* __restrict__ token_to_req,
    int64_t* __restrict__ logical_positions,
    int32_t* __restrict__ visible_blocks,
    int64_t* __restrict__ slot_mapping,
    int64_t bt_stride0,
    int64_t bt_stride1,
    int num_reqs,
    int num_mapped_tokens,
    int num_tokens,
    int storage_block_size,
    int compress_ratio,
    int circular_buffer_size,
    int num_bt_cols) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_tokens) {
    const bool mapped = (tid < num_mapped_tokens);
    int req_idx = 0;
    if (mapped) {
      int lo = 0, hi = num_reqs - 1;
      while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (query_start_loc[mid] <= tid) lo = mid;
        else hi = mid - 1;
      }
      req_idx = lo;
    }
    int32_t q_start = mapped ? query_start_loc[req_idx] : 0;
    int32_t q_end   = mapped ? query_start_loc[req_idx + 1] : 0;
    int32_t slen    = mapped ? seq_lens[req_idx] : 0;
    int64_t lpos    = mapped ? ((int64_t)slen - (int64_t)(q_end - q_start) + (int64_t)(tid - q_start)) : -1;

    token_to_req[tid] = mapped ? req_idx : 0;
    logical_positions[tid] = lpos;
    int64_t vb_a = (lpos + 1) / compress_ratio;
    int64_t vb_b = (int64_t)slen / compress_ratio;
    int64_t vb = vb_a < vb_b ? vb_a : vb_b;
    if (vb < 0) vb = 0;
    visible_blocks[tid] = (int32_t)vb;

    if (circular_buffer_size > 0) {
      bool valid = mapped && (lpos >= 0) && ((tid + circular_buffer_size) >= q_end) && (num_bt_cols > 0);
      int64_t slot = -1;
      if (valid) {
        int32_t phys = block_table[(int64_t)req_idx * bt_stride0];
        if (phys >= 0) {
          slot = (int64_t)phys * circular_buffer_size + (lpos % circular_buffer_size);
        }
      }
      slot_mapping[tid] = slot;
    } else if (compress_ratio != 1) {
      int64_t cpos = (lpos > 0 ? lpos : 0) / compress_ratio;
      int64_t lblk = cpos / storage_block_size;
      bool valid = mapped && (lpos >= 0) && (((lpos + 1) % compress_ratio) == 0) && (lblk < num_bt_cols);
      int64_t slot = -1;
      if (valid) {
        int32_t phys = block_table[(int64_t)req_idx * bt_stride0 + lblk * bt_stride1];
        if (phys >= 0 && common_slot_mapping[tid] >= 0) {
          slot = (int64_t)phys * storage_block_size + (cpos % storage_block_size);
        }
      }
      slot_mapping[tid] = slot;
    }
  }
}

__global__ void k_build_qsa_work_metadata(
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ seq_lens,
    int32_t* __restrict__ k_work_metadata,
    int num_reqs,
    int max_num_work,
    int compress_ratio) {
  __shared__ int32_t s_ends[1024];
  const int tid = threadIdx.x;
  const int work = blockIdx.x * blockDim.x + tid;

  if (tid == 0) {
    int32_t acc = 0;
    for (int r = 0; r < num_reqs; ++r) {
      int32_t qlen = query_start_loc[r + 1] - query_start_loc[r];
      int32_t slen = seq_lens[r];
      int32_t cstart = slen - qlen;
      int32_t ng = (slen / compress_ratio) - (cstart / compress_ratio);
      int32_t wc = (qlen > 0) ? (ng > 1 ? ng : 1) : 0;
      acc += wc;
      s_ends[r] = acc;
    }
  }
  __syncthreads();

  if (work >= max_num_work) return;
  const int32_t total_work = (num_reqs > 0) ? s_ends[num_reqs - 1] : 0;
  if (work >= total_work) {
    k_work_metadata[work * 2] = -1;
    k_work_metadata[work * 2 + 1] = -1;
    return;
  }
  int lo = 0, hi = num_reqs - 1;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (s_ends[mid] > work) hi = mid;
    else lo = mid + 1;
  }
  int32_t start_w = (lo == 0) ? 0 : s_ends[lo - 1];
  k_work_metadata[work * 2] = lo;
  k_work_metadata[work * 2 + 1] = work - start_w;
}

// ============================================================================
// 2. compute_slot_mappings & gather_block_tables
// ============================================================================
__global__ void k_compute_slot_mappings(
    int max_num_tokens,
    const int32_t* __restrict__ idx_mapping,
    const int32_t* __restrict__ query_start_loc,
    const int64_t* __restrict__ pos,
    const uint64_t* __restrict__ block_table_ptrs,
    const int64_t* __restrict__ block_table_strides,
    const int32_t* __restrict__ block_sizes,
    const int32_t* __restrict__ kernel_block_sizes,
    const bool* __restrict__ slot_mapping_enabled,
    int64_t* __restrict__ slot_mappings,
    int64_t slot_mappings_stride,
    int cp_rank, int cp_size, int cp_interleave,
    int64_t pad_id, int num_reqs) {
  const int group_id = blockIdx.x;
  const int batch_idx = blockIdx.y;
  int64_t* sm_ptr = slot_mappings + (int64_t)group_id * slot_mappings_stride;

  if (batch_idx == num_reqs) {
    int actual_num_tokens = query_start_loc[batch_idx];
    for (int idx = actual_num_tokens + threadIdx.x; idx < max_num_tokens; idx += blockDim.x) {
      sm_ptr[idx] = pad_id;
    }
    return;
  }

  const int32_t* bt_ptr = reinterpret_cast<const int32_t*>(block_table_ptrs[group_id]);
  const int64_t bt_stride = block_table_strides[group_id];
  const int32_t kv_bs = block_sizes[group_id];
  const int32_t k_bs = kernel_block_sizes[group_id];
  const bool enabled = slot_mapping_enabled[group_id];

  const int32_t req_state_idx = idx_mapping[batch_idx];
  const bool is_real = (req_state_idx >= 0);
  const int32_t start_idx = query_start_loc[batch_idx];
  const int32_t end_idx = query_start_loc[batch_idx + 1];

  for (int idx = start_idx + threadIdx.x; idx < end_idx; idx += blockDim.x) {
    if (!enabled || !is_real) {
      sm_ptr[idx] = pad_id;
      continue;
    }
    int64_t p = pos[idx];
    int64_t local_p = p;
    bool is_local = true;
    if (cp_size > 1) {
      int64_t vbs = (int64_t)kv_bs * cp_size;
      int64_t vbi = p / vbs;
      int64_t vbo = p % vbs;
      is_local = (((vbo / cp_interleave) % cp_size) == cp_rank);
      int64_t rounds = vbo / ((int64_t)cp_interleave * cp_size);
      int64_t rem = vbo % cp_interleave;
      local_p = vbi * kv_bs + rounds * cp_interleave + rem;
    }
    if (!is_local) {
      sm_ptr[idx] = pad_id;
    } else {
      int64_t blk_idx = local_p / k_bs;
      int64_t blk_off = local_p % k_bs;
      int32_t blk_num = bt_ptr[(int64_t)req_state_idx * bt_stride + blk_idx];
      sm_ptr[idx] = (int64_t)blk_num * k_bs + blk_off;
    }
  }
}

__global__ void k_gather_block_tables(
    const int32_t* __restrict__ batch_idx_to_req_idx,
    const uint64_t* __restrict__ src_bt_ptrs,
    const uint64_t* __restrict__ dst_bt_ptrs,
    const int64_t* __restrict__ bt_strides,
    const int32_t* __restrict__ num_blocks_ptr,
    int64_t num_blocks_stride,
    int num_reqs) {
  const int group_id = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int64_t stride = bt_strides[group_id];
  int32_t* dst_row = reinterpret_cast<int32_t*>(dst_bt_ptrs[group_id]) + (int64_t)batch_idx * stride;
  if (batch_idx >= num_reqs) {
    for (int64_t i = threadIdx.x; i < stride; i += blockDim.x) dst_row[i] = 0;
    return;
  }
  const int32_t req_idx = batch_idx_to_req_idx[batch_idx];
  const int32_t nblk = num_blocks_ptr[(int64_t)group_id * num_blocks_stride + req_idx];
  const int32_t* src_row = reinterpret_cast<const int32_t*>(src_bt_ptrs[group_id]) + (int64_t)req_idx * stride;
  for (int32_t i = threadIdx.x; i < nblk; i += blockDim.x) dst_row[i] = src_row[i];
}

// ============================================================================
// 3. input_batch bookkeeping kernels
// ============================================================================
__global__ void k_prepare_pos_seq_lens(
    int64_t* __restrict__ pos,
    int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ idx_mapping,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ num_computed_tokens,
    int num_reqs,
    int max_num_reqs) {
  const int req_id = blockIdx.x;
  if (req_id == num_reqs) {
    for (int i = num_reqs + threadIdx.x; i < max_num_reqs; i += blockDim.x) seq_lens[i] = 0;
    return;
  }
  const int32_t ridx = idx_mapping[req_id];
  const int32_t ncomp = num_computed_tokens[ridx];
  const int32_t start = query_start_loc[req_id];
  const int32_t end = query_start_loc[req_id + 1];
  const int32_t qlen = end - start;
  if (threadIdx.x == 0) seq_lens[req_id] = ncomp + qlen;
  for (int i = threadIdx.x; i < qlen; i += blockDim.x) {
    pos[start + i] = (int64_t)ncomp + i;
  }
}

__global__ void k_combine_sampled_and_draft(
    int32_t* __restrict__ input_ids,
    const int32_t* __restrict__ idx_mapping,
    const int64_t* __restrict__ last_sampled_tokens,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ prefill_len,
    const int64_t* __restrict__ draft_tokens,
    int64_t draft_stride,
    const int32_t* __restrict__ cu_num_logits,
    int64_t* __restrict__ logits_indices,
    int num_new_sampled) {
  const int batch_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int32_t ridx = idx_mapping[batch_idx];
  const int32_t cu_start = cu_num_logits[batch_idx];
  const int32_t cu_end = cu_num_logits[batch_idx + 1];
  const int32_t num_logits = cu_end - cu_start;
  const int32_t num_draft = num_logits - num_new_sampled;
  const int32_t q_end = query_start_loc[batch_idx + 1];
  const int32_t logits_start = q_end - num_logits;

  for (int i = tid; i < num_logits; i += blockDim.x) {
    logits_indices[cu_start + i] = (int64_t)(logits_start + i);
  }
  const int32_t slen = seq_lens[batch_idx];
  const int32_t plen = prefill_len[ridx];
  if (slen <= plen) return;

  if (tid == 0 && num_new_sampled > 0 && (slen - num_logits) >= plen) {
    input_ids[logits_start] = (int32_t)last_sampled_tokens[ridx];
  }
  if (num_draft > 0) {
    for (int i = tid; i < num_draft; i += blockDim.x) {
      input_ids[q_end - num_draft + i] = (int32_t)draft_tokens[(int64_t)ridx * draft_stride + i];
    }
  }
}

__global__ void k_get_num_sampled_and_rejected(
    int32_t* __restrict__ num_sampled,
    int32_t* __restrict__ num_rejected,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ cu_num_logits,
    const int32_t* __restrict__ idx_mapping,
    const int32_t* __restrict__ prefill_len,
    int num_reqs) {
  const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= num_reqs) return;
  const int32_t ridx = idx_mapping[batch_idx];
  const bool is_chunk = seq_lens[batch_idx] < prefill_len[ridx];
  int32_t ns = is_chunk ? 0 : num_sampled[batch_idx];
  num_sampled[batch_idx] = ns;
  int32_t nlog = cu_num_logits[batch_idx + 1] - cu_num_logits[batch_idx];
  num_rejected[batch_idx] = is_chunk ? 0 : (nlog - ns);
}

template <typename TokT>
__global__ void k_post_update(
    const int32_t* __restrict__ idx_mapping,
    int32_t* __restrict__ num_computed_tokens,
    int64_t* __restrict__ last_sampled_tokens,
    int32_t* __restrict__ output_bin_counts, int64_t obc_stride,
    const TokT* __restrict__ sampled_tokens, int64_t st_stride,
    const int32_t* __restrict__ num_sampled_ptr,
    const int32_t* __restrict__ num_rejected_ptr,
    const int32_t* __restrict__ query_start_loc,
    int32_t* __restrict__ all_token_ids, int64_t ati_stride,
    int32_t* __restrict__ total_len_ptr,
    const int64_t* __restrict__ broadcast_drafts, int64_t bd_stride,
    int64_t* __restrict__ draft_tokens_out, int64_t dt_stride,
    int num_spec, int num_reqs) {
  const int req_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (req_id >= num_reqs) return;
  const int32_t ridx = idx_mapping[req_id];
  if (ridx < 0) return;

  if (broadcast_drafts != nullptr && draft_tokens_out != nullptr) {
    for (int i = 0; i < num_spec; ++i) {
      draft_tokens_out[(int64_t)ridx * dt_stride + i] = broadcast_drafts[(int64_t)req_id * bd_stride + i];
    }
  }
  const int32_t tlen = total_len_ptr[ridx];
  const int32_t ns = num_sampled_ptr[req_id];
  if (ns > 0) {
    int64_t last_tok = (int64_t)sampled_tokens[(int64_t)req_id * st_stride + ns - 1];
    last_sampled_tokens[ridx] = last_tok;
    total_len_ptr[ridx] = tlen + ns;
  }
  for (int i = 0; i < ns; ++i) {
    int32_t tok = (int32_t)sampled_tokens[(int64_t)req_id * st_stride + i];
    all_token_ids[(int64_t)ridx * ati_stride + tlen + i] = tok;
    if (output_bin_counts != nullptr) {
      output_bin_counts[(int64_t)ridx * obc_stride + tok] += 1;
    }
  }
  int32_t qlen = (query_start_loc != nullptr) ? (query_start_loc[req_id + 1] - query_start_loc[req_id]) : 0;
  int32_t delta = qlen - num_rejected_ptr[req_id];
  if (delta != 0) num_computed_tokens[ridx] += delta;
}

template <typename IdxT>
__global__ void k_expand_idx_mapping(
    const IdxT* __restrict__ idx_mapping,
    IdxT* __restrict__ expanded_idx_mapping,
    IdxT* __restrict__ expanded_local_pos,
    const int32_t* __restrict__ cu_num_logits) {
  const int req_idx = blockIdx.x;
  const int32_t s = cu_num_logits[req_idx];
  const int32_t n = cu_num_logits[req_idx + 1] - s;
  const IdxT ridx = idx_mapping[req_idx];
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    expanded_idx_mapping[s + i] = ridx;
    expanded_local_pos[s + i] = (IdxT)i;
  }
}

// ============================================================================
// 4. speculator prepare_prefill_inputs & prepare_decode_inputs
// ============================================================================
template <typename OutIdxT>
__global__ void k_prepare_prefill_inputs(
    OutIdxT* __restrict__ last_token_indices,
    OutIdxT* __restrict__ draft_current_step,
    int32_t* __restrict__ draft_input_ids,
    int64_t* __restrict__ draft_positions,
    int32_t* __restrict__ draft_query_start_loc,
    int32_t* __restrict__ draft_seq_lens,
    const int32_t* __restrict__ target_input_ids,
    const int64_t* __restrict__ target_positions,
    const int32_t* __restrict__ idx_mapping,
    const int64_t* __restrict__ last_sampled,
    const int32_t* __restrict__ next_prefill_tokens,
    const int32_t* __restrict__ num_sampled,
    const int32_t* __restrict__ num_rejected,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ seq_lens,
    int num_reqs, int max_num_reqs) {
  const int req_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int32_t ridx = idx_mapping[req_idx];
  const int32_t q_start = query_start_loc[req_idx];
  const int32_t q_end = query_start_loc[req_idx + 1];
  const int32_t qlen = (q_end - q_start) - num_rejected[req_idx];
  const int32_t slen = seq_lens[req_idx];
  const int32_t next_tok = (num_sampled[req_idx] > 0) ? (int32_t)last_sampled[ridx] : next_prefill_tokens[ridx];

  for (int i = 1 + tid; i < qlen; i += blockDim.x) {
    draft_input_ids[q_start + i - 1] = target_input_ids[q_start + i];
  }
  for (int i = tid; i < qlen; i += blockDim.x) {
    draft_positions[q_start + i] = target_positions[q_start + i];
  }
  if (tid == 0) {
    int32_t lti = q_start + qlen - 1;
    last_token_indices[req_idx] = lti;
    draft_input_ids[lti] = next_tok;
    draft_query_start_loc[req_idx] = q_start;
    draft_seq_lens[req_idx] = slen;
  }
  if (req_idx == num_reqs - 1) {
    if (tid == 0) draft_current_step[0] = 0;
    for (int i = num_reqs + tid; i < max_num_reqs + 1; i += blockDim.x) draft_query_start_loc[i] = q_end;
    for (int i = num_reqs + tid; i < max_num_reqs; i += blockDim.x) {
      draft_seq_lens[i] = 0;
      last_token_indices[i] = 0;
    }
  }
}

template <typename DraftT>
__global__ void k_prepare_decode_inputs(
    const DraftT* __restrict__ draft_tokens, int64_t dt_stride,
    const int32_t* __restrict__ target_seq_lens,
    const int32_t* __restrict__ num_rejected,
    int32_t* __restrict__ input_ids,
    int64_t* __restrict__ positions,
    int64_t* __restrict__ sample_src_positions,
    int32_t* __restrict__ query_start_loc,
    int32_t* __restrict__ seq_lens,
    int num_reqs, int max_model_len, int max_num_reqs, bool advance_pos) {
  const int req_idx = blockIdx.x;
  const int tid = threadIdx.x;
  if (req_idx == num_reqs) {
    for (int i = tid; i < max_num_reqs + 1; i += blockDim.x) {
      query_start_loc[i] = (i < num_reqs) ? i : num_reqs;
    }
    for (int i = num_reqs + tid; i < max_num_reqs; i += blockDim.x) {
      seq_lens[i] = 0;
    }
    return;
  }
  if (tid == 0) {
    input_ids[req_idx] = (int32_t)draft_tokens[(int64_t)req_idx * dt_stride];
    sample_src_positions[req_idx] += 1;
    int32_t slen = target_seq_lens[req_idx] - num_rejected[req_idx];
    if (advance_pos) {
      int64_t p = positions[req_idx] + 1;
      if (p > max_model_len - 1) p = max_model_len - 1;
      positions[req_idx] = p;
      slen = (slen + 1 < max_model_len) ? (slen + 1) : max_model_len;
    }
    seq_lens[req_idx] = slen;
  }
}

// ============================================================================
// 5. prepare_rope_positions & scatter/fill num_accepted
// ============================================================================
template <typename PrefillT>
__global__ void k_prepare_rope_positions(
    int64_t* __restrict__ positions, int64_t pos_stride,
    const PrefillT* __restrict__ prefill_positions, int64_t pp_stride0, int64_t pp_stride1,
    const PrefillT* __restrict__ prefill_delta,
    const int32_t* __restrict__ idx_mapping,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ prefill_lens,
    const int32_t* __restrict__ num_computed_tokens,
    int num_dims) {
  const int batch_idx = blockIdx.x;
  const int32_t ridx = idx_mapping[batch_idx];
  const int32_t plen = prefill_lens[ridx];
  const int32_t ncomp = num_computed_tokens[ridx];
  const bool is_prefill = (ncomp < plen);
  const int32_t q_start = query_start_loc[batch_idx];
  const int32_t qlen = query_start_loc[batch_idx + 1] - q_start;
  const int64_t delta = prefill_delta[ridx];
  for (int i = threadIdx.x; i < qlen; i += blockDim.x) {
    int64_t orig_pos = (int64_t)ncomp + i;
    for (int j = 0; j < num_dims; ++j) {
      int64_t val = is_prefill
          ? prefill_positions[(int64_t)ridx * pp_stride0 + (int64_t)j * pp_stride1 + orig_pos]
          : (orig_pos + delta);
      positions[(int64_t)j * pos_stride + q_start + i] = val;
    }
  }
}

__global__ void k_scatter_num_accepted(
    const int32_t* __restrict__ idx_mapping,
    const int32_t* __restrict__ num_sampled,
    int32_t* __restrict__ num_accepted,
    int num_reqs) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_reqs) return;
  const int32_t ridx = idx_mapping[row];
  if (ridx < 0) return;
  int32_t v = num_sampled[row];
  num_accepted[ridx] = (v > 1) ? v : 1;
}

__global__ void k_fill_num_accepted(
    const int32_t* __restrict__ idx_mapping,
    int32_t* __restrict__ num_accepted,
    int32_t val, int num_reqs) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_reqs) return;
  const int32_t ridx = idx_mapping[row];
  if (ridx < 0) return;
  num_accepted[ridx] = val;
}

// ============================================================================
// 6. Fused GDN pure-spec-decode metadata builder (replaces ~25 PyTorch ops x 3 groups = ~75 launches & 3.15 ms/step CPU!)
// ============================================================================
template <typename TokIdxT>
__global__ void k_gdn_prepare_pure_spec_decode(
    const int32_t* __restrict__ block_table,
    int64_t bt_stride0, int bt_cols,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ src_num_accepted,
    int32_t* __restrict__ dst_state_indices,
    int64_t dst_si_stride0,
    bool* __restrict__ dst_spec_masks,
    TokIdxT* __restrict__ dst_spec_tok_idx,
    int32_t* __restrict__ dst_spec_qsl,
    int32_t* __restrict__ dst_num_accepted,
    int num_spec_decodes, int batch_size, int num_spec_tokens,
    int n_cols, int block_size, bool align_mode, int32_t null_block_id) {
  const int tid = threadIdx.x;
  const int32_t last_qsl = query_start_loc[num_spec_decodes];

  for (int r = tid; r < batch_size; r += blockDim.x) {
    int32_t* dst_row = dst_state_indices + (int64_t)r * dst_si_stride0;
    if (r < num_spec_decodes) {
      int32_t start_col = 0;
      if (align_mode) {
        int32_t sl = seq_lens[r];
        start_col = (sl > 0) ? ((sl - 1) / block_size) : 0;
      }
      const int32_t* bt_row = block_table + (int64_t)r * bt_stride0;
      for (int c = 0; c < n_cols; ++c) {
        int col = start_col + c;
        dst_row[c] = (col < bt_cols) ? bt_row[col] : null_block_id;
      }
      dst_spec_masks[r] = true;
      dst_num_accepted[r] = src_num_accepted[r];
    } else {
      for (int c = 0; c < n_cols; ++c) dst_row[c] = null_block_id;
      dst_spec_masks[r] = false;
      dst_num_accepted[r] = 1;
    }
  }
  for (int i = tid; i <= batch_size; i += blockDim.x) {
    dst_spec_qsl[i] = (i <= num_spec_decodes) ? query_start_loc[i] : last_qsl;
  }
  for (int t = tid; t < num_spec_tokens; t += blockDim.x) {
    dst_spec_tok_idx[t] = (TokIdxT)t;
  }
}

// ============================================================================
// 7. Final 3 Mamba Align Kernels -> LITERALLY 0.0 Triton launches/step in decode!
//    (preprocess_mamba_align_fused, precopy_mamba_align_fused, postprocess_mamba_fused)
// ============================================================================
__global__ void k_preprocess_mamba_align_fused(
    const int32_t* __restrict__ idx_mapping,
    int32_t* __restrict__ state_idx,
    const int32_t* __restrict__ num_computed_tokens,
    const int32_t* __restrict__ query_start_loc,
    int32_t* __restrict__ num_accepted_tokens,
    int32_t* __restrict__ src_col,
    int32_t* __restrict__ src_off,
    int num_reqs, int mamba_block_size) {
  const int off = blockIdx.x * blockDim.x + threadIdx.x;
  if (off >= num_reqs) return;
  const int32_t ridx = idx_mapping[off];
  const int32_t st = state_idx[ridx];
  const int32_t nacc = num_accepted_tokens[ridx];
  src_col[ridx] = st;
  src_off[ridx] = (nacc > 1) ? (nacc - 1) : 0;
  const int32_t ncomp = num_computed_tokens[ridx];
  const int32_t qlen = query_start_loc[off + 1] - query_start_loc[off];
  const int32_t comp_after = ncomp + qlen;
  const int32_t new_st = (comp_after + mamba_block_size - 1) / mamba_block_size - 1;
  state_idx[ridx] = new_st;
  if (st >= 0 && st != new_st) {
    num_accepted_tokens[ridx] = 1;
  }
}

__device__ __forceinline__ void d_memcpy_u64_tiled(
    uint64_t src_addr, uint64_t dst_addr, int64_t copy_bytes,
    int tile_idx, int num_tiles) {
  const int64_t n_u64 = copy_bytes >> 3;
  const int64_t per_tile = (n_u64 + num_tiles - 1) / num_tiles;
  const int64_t t_start = (int64_t)tile_idx * per_tile;
  const int64_t t_end = (t_start + per_tile < n_u64) ? (t_start + per_tile) : n_u64;
  const uint64_t* sp = reinterpret_cast<const uint64_t*>(src_addr);
  uint64_t* dp = reinterpret_cast<uint64_t*>(dst_addr);
  for (int64_t i = t_start + threadIdx.x; i < t_end; i += blockDim.x) {
    dp[i] = sp[i];
  }
  if (tile_idx == 0) {
    const int64_t tail_off = n_u64 << 3;
    const uint8_t* sb = reinterpret_cast<const uint8_t*>(src_addr + tail_off);
    uint8_t* db = reinterpret_cast<uint8_t*>(dst_addr + tail_off);
    for (int64_t i = threadIdx.x; i < (copy_bytes - tail_off); i += blockDim.x) {
      db[i] = sb[i];
    }
  }
}

__device__ __forceinline__ void d_copy_mamba_state_block(
    int state_idx, int bt_row_idx, int32_t src_col, int32_t dst_col, int32_t token_bias,
    const int64_t* __restrict__ block_table_ptrs, int64_t bt_stride_req,
    const int64_t* __restrict__ state_base_addrs,
    const int64_t* __restrict__ state_block_strides,
    const int32_t* __restrict__ state_elem_sizes,
    const int64_t* __restrict__ state_inner_sizes,
    const int32_t* __restrict__ state_conv_widths,
    const int32_t* __restrict__ state_group_indices,
    const int32_t* __restrict__ state_dim_row_count,
    const int64_t* __restrict__ state_dim_row_stride,
    int tile_idx, bool conv_dim_first, int temporal_tiles) {
  const int32_t grp = state_group_indices[state_idx];
  const int32_t* bt_base = reinterpret_cast<const int32_t*>(block_table_ptrs[grp]) + (int64_t)bt_row_idx * bt_stride_req;
  const uint64_t base_addr = (uint64_t)state_base_addrs[state_idx];
  const int64_t blk_stride = state_block_strides[state_idx];
  const int32_t elem_sz = state_elem_sizes[state_idx];
  const int64_t inner_sz = state_inner_sizes[state_idx];
  const int32_t conv_w = state_conv_widths[state_idx];
  const int64_t dst_blk_id = (int64_t)bt_base[dst_col];
  const uint64_t dst_addr = base_addr + (uint64_t)(dst_blk_id * blk_stride);

  if (conv_w > 0 && conv_dim_first) {
    if (tile_idx > 0) return;
    const int64_t src_blk_id = (int64_t)bt_base[src_col];
    const int32_t dim_rows = state_dim_row_count[state_idx];
    const int64_t row_stride = state_dim_row_stride[state_idx];
    const uint64_t src_blk_addr = base_addr + (uint64_t)(src_blk_id * blk_stride);
    const int32_t n_tok = conv_w - token_bias;
    for (int32_t t = 0; t < n_tok; ++t) {
      for (int32_t r = threadIdx.x; r < dim_rows; r += blockDim.x) {
        uint64_t sa = src_blk_addr + (uint64_t)(r * row_stride + (int64_t)(t + token_bias) * elem_sz);
        uint64_t da = dst_addr + (uint64_t)(r * row_stride + (int64_t)t * elem_sz);
        if (elem_sz == 2) *reinterpret_cast<uint16_t*>(da) = *reinterpret_cast<const uint16_t*>(sa);
        else if (elem_sz == 4) *reinterpret_cast<uint32_t*>(da) = *reinterpret_cast<const uint32_t*>(sa);
        else {
          for (int b = 0; b < elem_sz; ++b) reinterpret_cast<uint8_t*>(da)[b] = reinterpret_cast<const uint8_t*>(sa)[b];
        }
      }
    }
    return;
  }

  if (conv_w > 0) {
    if (tile_idx > 0) return;
    const int64_t src_blk_id = (int64_t)bt_base[src_col];
    const uint64_t src_blk_addr = base_addr + (uint64_t)(src_blk_id * blk_stride);
    const int64_t tok_bytes = inner_sz * (int64_t)elem_sz;
    const int32_t n_tok = conv_w - token_bias;
    if (src_blk_id != dst_blk_id || token_bias == 0) {
      d_memcpy_u64_tiled(src_blk_addr + (uint64_t)((int64_t)token_bias * tok_bytes), dst_addr, (int64_t)n_tok * tok_bytes, 0, 1);
    } else {
      for (int32_t t = 0; t < n_tok; ++t) {
        d_memcpy_u64_tiled(src_blk_addr + (uint64_t)((int64_t)(t + token_bias) * tok_bytes),
                           dst_addr + (uint64_t)((int64_t)t * tok_bytes), tok_bytes, 0, 1);
      }
    }
    return;
  }

  const int64_t actual_src_blk_id = (int64_t)bt_base[src_col + token_bias];
  const uint64_t src_addr = base_addr + (uint64_t)(actual_src_blk_id * blk_stride);
  d_memcpy_u64_tiled(src_addr, dst_addr, inner_sz * (int64_t)elem_sz, tile_idx, temporal_tiles);
}

__global__ void k_precopy_mamba_align_fused(
    const int32_t* __restrict__ mamba_state_idx,
    const int32_t* __restrict__ src_col_ptr,
    const int32_t* __restrict__ token_bias_ptr,
    const int64_t* __restrict__ block_table_ptrs, int64_t bt_stride_req,
    const int64_t* __restrict__ state_base_addrs,
    const int64_t* __restrict__ state_block_strides,
    const int32_t* __restrict__ state_elem_sizes,
    const int64_t* __restrict__ state_inner_sizes,
    const int32_t* __restrict__ state_conv_widths,
    const int32_t* __restrict__ state_group_indices,
    const int32_t* __restrict__ state_dim_row_count,
    const int64_t* __restrict__ state_dim_row_stride,
    const int32_t* __restrict__ idx_mapping,  // nullable
    int num_reqs, bool conv_dim_first, int temporal_tiles) {
  const int batch_idx = blockIdx.x;
  const int state_idx = blockIdx.y;
  const int tile_idx = blockIdx.z;
  if (batch_idx >= num_reqs) return;
  const int32_t req_idx = (idx_mapping != nullptr) ? idx_mapping[batch_idx] : batch_idx;
  if (req_idx < 0) return;
  const int32_t sc = src_col_ptr[req_idx];
  const int32_t dc = mamba_state_idx[req_idx];
  if (sc < 0 || sc == dc) return;  // 95%+ decode fast-exit!
  d_copy_mamba_state_block(
      state_idx, batch_idx, sc, dc, token_bias_ptr[req_idx],
      block_table_ptrs, bt_stride_req, state_base_addrs, state_block_strides,
      state_elem_sizes, state_inner_sizes, state_conv_widths, state_group_indices,
      state_dim_row_count, state_dim_row_stride, tile_idx, conv_dim_first, temporal_tiles);
}

__global__ void k_postprocess_mamba_align_v2(
    const int32_t* __restrict__ num_accepted_snap,
    const int32_t* __restrict__ mamba_state_idx,
    const int32_t* __restrict__ new_num_computed_ptr,
    const int64_t* __restrict__ block_table_ptrs, int64_t bt_stride_req,
    const int64_t* __restrict__ state_base_addrs,
    const int64_t* __restrict__ state_block_strides,
    const int32_t* __restrict__ state_elem_sizes,
    const int64_t* __restrict__ state_inner_sizes,
    const int32_t* __restrict__ state_conv_widths,
    const int32_t* __restrict__ state_group_indices,
    const int32_t* __restrict__ state_dim_row_count,
    const int64_t* __restrict__ state_dim_row_stride,
    int32_t* __restrict__ num_accepted_out,
    const int32_t* __restrict__ idx_mapping,
    int num_reqs, int block_size, bool conv_dim_first, int temporal_tiles) {
  const int batch_idx = blockIdx.x;
  const int state_idx = blockIdx.y;
  const int tile_idx = blockIdx.z;
  if (batch_idx >= num_reqs) return;
  const int32_t req_idx = idx_mapping[batch_idx];
  if (req_idx < 0) return;
  const int32_t nacc = num_accepted_snap[req_idx];
  const int32_t src_blk = mamba_state_idx[req_idx];
  const int32_t new_ncomp = new_num_computed_ptr[req_idx];
  const int32_t running = new_ncomp - nacc + 1;
  const int32_t aligned = (new_ncomp / block_size) * block_size;
  if (aligned < running) return;  // 95%+ decode fast-exit!
  const int32_t bias = aligned - running;
  const int32_t dst_blk = aligned / block_size - 1;
  if (src_blk == dst_blk && state_idx == 0 && tile_idx == 0 && threadIdx.x == 0) {
    num_accepted_out[req_idx] = 1;
  }
  if (src_blk == dst_blk && bias == 0) return;
  d_copy_mamba_state_block(
      state_idx, batch_idx, src_blk, dst_blk, bias,
      block_table_ptrs, bt_stride_req, state_base_addrs, state_block_strides,
      state_elem_sizes, state_inner_sizes, state_conv_widths, state_group_indices,
      state_dim_row_count, state_dim_row_stride, tile_idx, conv_dim_first, temporal_tiles);
}

// ============================================================================
// 8. Wave 5: Fused PLE compute_ngram_ids (1.165 ms/step -> 0.005 ms/step CPU!)
//    & PLE short_conv pure-spec-decode metadata builder (0.850 ms/step -> 0.005 ms/step CPU!)
// ============================================================================
template <typename IdT, typename QslT, typename CtxT>
__global__ void k_ple_compute_ngram_ids(
    const IdT* __restrict__ input_ids,            // [num_tokens]
    const QslT* __restrict__ query_start_loc,     // [num_reqs + 1]
    const CtxT* __restrict__ ngram_context,       // [ctx_rows, ctx_len]
    int64_t ctx_stride0, int ctx_rows, int ctx_len,
    const int64_t* __restrict__ multipliers,      // [ngram_size]
    const int64_t* __restrict__ vocab_sizes,      // [total_heads]
    const int64_t* __restrict__ offsets,          // [total_heads]
    int64_t* __restrict__ out_ngram_ids,          // [num_tokens, total_heads]
    int num_tokens, int num_reqs, int ngram_size, int heads_per_ngram,
    int64_t eos_token_id) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_tokens) return;

  // Binary search: last request r with query_start_loc[r] <= tid (matches searchsorted(right=True) - 1, clamped)
  int lo = 0, hi = num_reqs - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if ((int64_t)query_start_loc[mid] <= (int64_t)tid) lo = mid;
    else hi = mid - 1;
  }
  const int req = lo;
  const int64_t q_start = (int64_t)query_start_loc[req];
  const int64_t col = (int64_t)tid - q_start;
  const int ctx_r = (req < ctx_rows) ? req : (ctx_rows - 1);
  const CtxT* ctx_ptr = ngram_context + (int64_t)ctx_r * ctx_stride0;

  auto get_tok = [&](int64_t p) -> int64_t {
    if (p < (int64_t)ctx_len) return (int64_t)ctx_ptr[p];
    int64_t idx = q_start + (p - (int64_t)ctx_len);
    return (idx < (int64_t)num_tokens) ? (int64_t)input_ids[idx] : eos_token_id;
  };

  const int64_t pos = (int64_t)ctx_len + col;
  int64_t shifted[8];  // ngram_size <= 8 (Qwen3.8 uses ngram_size=3..5)
  shifted[0] = get_tok(pos);
  bool broken = false;
  for (int s = 1; s < ngram_size; ++s) {
    int64_t src = pos - s;
    if (src < 0 || broken) {
      shifted[s] = eos_token_id;
    } else {
      int64_t tk = get_tok(src);
      if (tk == eos_token_id) {
        broken = true;
        shifted[s] = eos_token_id;
      } else {
        shifted[s] = tk;
      }
    }
  }

  const int total_heads = (ngram_size - 1) * heads_per_ngram;
  int64_t* dst = out_ngram_ids + (int64_t)tid * total_heads;
  int64_t mixed = shifted[0] * multipliers[0];
  for (int ng = 2; ng <= ngram_size; ++ng) {
    mixed ^= (shifted[ng - 1] * multipliers[ng - 1]);
    const int base = (ng - 2) * heads_per_ngram;
    for (int h = 0; h < heads_per_ngram; ++h) {
      const int idx = base + h;
      const int64_t sz = vocab_sizes[idx];
      int64_t rem = mixed % sz;
      if (rem != 0 && ((rem < 0) != (sz < 0))) rem += sz;
      dst[idx] = rem + offsets[idx];
    }
  }
}

__global__ void k_ple_prepare_pure_spec_decode(
    const int32_t* __restrict__ block_table,
    int64_t bt_stride0, int bt_cols,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ src_num_accepted,
    int32_t* __restrict__ dst_state_indices,      // [batch_size] i32
    bool* __restrict__ dst_spec_masks,            // [batch_size] bool
    int32_t* __restrict__ dst_spec_tok_idx,       // [num_spec_tokens] i32
    int32_t* __restrict__ dst_spec_qsl,           // [batch_size + 1] i32
    int32_t* __restrict__ dst_num_accepted,       // [batch_size] i32
    int num_spec_decodes, int batch_size, int num_spec_tokens,
    int block_size, bool align_mode, int32_t null_block_id) {
  const int tid = threadIdx.x;
  const int32_t last_qsl = query_start_loc[num_spec_decodes];
  for (int r = tid; r < batch_size; r += blockDim.x) {
    if (r < num_spec_decodes) {
      int32_t col = 0;
      if (align_mode) {
        int32_t sl = seq_lens[r];
        col = (sl > 0) ? ((sl - 1) / block_size) : 0;
      }
      dst_state_indices[r] = (col < bt_cols) ? block_table[(int64_t)r * bt_stride0 + col] : null_block_id;
      dst_spec_masks[r] = true;
      dst_num_accepted[r] = src_num_accepted[r];
    } else {
      dst_state_indices[r] = null_block_id;
      dst_spec_masks[r] = false;
      dst_num_accepted[r] = 1;
    }
  }
  for (int i = tid; i <= batch_size; i += blockDim.x) {
    dst_spec_qsl[i] = (i <= num_spec_decodes) ? query_start_loc[i] : last_qsl;
  }
  for (int t = tid; t < num_spec_tokens; t += blockDim.x) {
    dst_spec_tok_idx[t] = t;
  }
}

// ============================================================================
// PyBind11 Host Wrappers (direct hipLaunchKernelGGL on current HIP stream)
// ============================================================================
static inline hipStream_t cur_stream() {
  return at::hip::getCurrentHIPStream().stream();
}

void build_qsa_metadata_hip(
    torch::Tensor query_start_loc,
    torch::Tensor seq_lens,
    torch::Tensor common_slot_mapping,
    torch::Tensor block_table,
    torch::Tensor token_to_req,
    torch::Tensor logical_positions,
    torch::Tensor visible_blocks,
    torch::Tensor slot_mapping,
    c10::optional<torch::Tensor> k_work_metadata,
    int num_reqs, int num_mapped_tokens, int num_tokens,
    int max_num_work, int storage_block_size, int compress_ratio, int circular_buffer_size) {
  hipStream_t st = cur_stream();
  if (num_tokens > 0) {
    int blocks = (num_tokens + 127) / 128;
    hipLaunchKernelGGL(
        k_build_qsa_metadata, dim3(blocks), dim3(128), 0, st,
        query_start_loc.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        common_slot_mapping.data_ptr<int64_t>(),
        block_table.data_ptr<int32_t>(),
        token_to_req.data_ptr<int32_t>(),
        logical_positions.data_ptr<int64_t>(),
        visible_blocks.data_ptr<int32_t>(),
        slot_mapping.data_ptr<int64_t>(),
        block_table.stride(0), block_table.stride(1),
        num_reqs, num_mapped_tokens, num_tokens,
        storage_block_size, compress_ratio, circular_buffer_size, (int)block_table.size(1));
  }
  if (k_work_metadata.has_value() && max_num_work > 0) {
    TORCH_CHECK(num_reqs <= 1024, "build_qsa_metadata_hip: num_reqs > 1024");
    int wblocks = (max_num_work + 255) / 256;
    hipLaunchKernelGGL(
        k_build_qsa_work_metadata, dim3(wblocks), dim3(256), 0, st,
        query_start_loc.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        k_work_metadata->data_ptr<int32_t>(),
        num_reqs, max_num_work, compress_ratio);
  }
}


static inline torch::Tensor as_i32(const torch::Tensor& t) {
  return (t.scalar_type() == torch::kInt32) ? t : t.to(torch::kInt32);
}

void compute_slot_mappings_hip(
    int max_num_tokens,
    torch::Tensor idx_mapping,
    torch::Tensor query_start_loc,
    torch::Tensor pos,
    torch::Tensor block_table_ptrs,
    torch::Tensor block_table_strides,
    torch::Tensor block_sizes,
    torch::Tensor kernel_block_sizes,
    torch::Tensor slot_mapping_enabled,
    torch::Tensor slot_mappings,
    int cp_rank, int cp_size, int cp_interleave, int64_t pad_id) {
  int num_groups = (int)block_table_ptrs.size(0);
  int num_reqs = (int)idx_mapping.size(0);
  hipLaunchKernelGGL(
      k_compute_slot_mappings, dim3(num_groups, num_reqs + 1), dim3(256), 0, cur_stream(),
      max_num_tokens,
      as_i32(idx_mapping).data_ptr<int32_t>(),
      query_start_loc.data_ptr<int32_t>(),
      pos.data_ptr<int64_t>(),
      reinterpret_cast<const uint64_t*>(block_table_ptrs.data_ptr()),
      block_table_strides.data_ptr<int64_t>(),
      block_sizes.data_ptr<int32_t>(),
      kernel_block_sizes.data_ptr<int32_t>(),
      slot_mapping_enabled.data_ptr<bool>(),
      slot_mappings.data_ptr<int64_t>(),
      slot_mappings.stride(0),
      cp_rank, cp_size, cp_interleave, pad_id, num_reqs);
}

void gather_block_tables_hip(
    torch::Tensor batch_idx_to_req_idx,
    torch::Tensor src_bt_ptrs,
    torch::Tensor dst_bt_ptrs,
    torch::Tensor bt_strides,
    torch::Tensor num_blocks,
    int num_reqs, int num_rows_grid) {
  int num_groups = (int)src_bt_ptrs.size(0);
  hipLaunchKernelGGL(
      k_gather_block_tables, dim3(num_groups, num_rows_grid), dim3(256), 0, cur_stream(),
      as_i32(batch_idx_to_req_idx).data_ptr<int32_t>(),
      reinterpret_cast<const uint64_t*>(src_bt_ptrs.data_ptr()),
      reinterpret_cast<const uint64_t*>(dst_bt_ptrs.data_ptr()),
      bt_strides.data_ptr<int64_t>(),
      num_blocks.data_ptr<int32_t>(),
      num_blocks.stride(0),
      num_reqs);
}

void prepare_pos_seq_lens_hip(
    torch::Tensor idx_mapping,
    torch::Tensor query_start_loc,
    torch::Tensor num_computed_tokens,
    torch::Tensor pos,
    torch::Tensor seq_lens) {
  int num_reqs = (int)idx_mapping.size(0);
  hipLaunchKernelGGL(
      k_prepare_pos_seq_lens, dim3(num_reqs + 1), dim3(256), 0, cur_stream(),
      pos.data_ptr<int64_t>(),
      seq_lens.data_ptr<int32_t>(),
      as_i32(idx_mapping).data_ptr<int32_t>(),
      query_start_loc.data_ptr<int32_t>(),
      num_computed_tokens.data_ptr<int32_t>(),
      num_reqs, (int)seq_lens.size(0));
}

void combine_sampled_and_draft_hip(
    torch::Tensor input_ids,
    torch::Tensor idx_mapping,
    torch::Tensor last_sampled_tokens,
    torch::Tensor query_start_loc,
    torch::Tensor seq_lens,
    torch::Tensor prefill_len,
    torch::Tensor draft_tokens,
    torch::Tensor cu_num_logits,
    torch::Tensor logits_indices,
    int num_new_sampled) {
  int num_reqs = (int)idx_mapping.size(0);
  if (num_reqs == 0) return;
  hipLaunchKernelGGL(
      k_combine_sampled_and_draft, dim3(num_reqs), dim3(64), 0, cur_stream(),
      input_ids.data_ptr<int32_t>(),
      as_i32(idx_mapping).data_ptr<int32_t>(),
      last_sampled_tokens.data_ptr<int64_t>(),
      query_start_loc.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(),
      prefill_len.data_ptr<int32_t>(),
      draft_tokens.data_ptr<int64_t>(),
      draft_tokens.stride(0),
      cu_num_logits.data_ptr<int32_t>(),
      logits_indices.data_ptr<int64_t>(),
      num_new_sampled);
}

void get_num_sampled_and_rejected_hip(
    torch::Tensor num_sampled,
    torch::Tensor num_rejected,
    torch::Tensor seq_lens,
    torch::Tensor cu_num_logits,
    torch::Tensor idx_mapping,
    torch::Tensor prefill_len) {
  int num_reqs = (int)idx_mapping.size(0);
  if (num_reqs == 0) return;
  hipLaunchKernelGGL(
      k_get_num_sampled_and_rejected, dim3((num_reqs + 63) / 64), dim3(64), 0, cur_stream(),
      num_sampled.data_ptr<int32_t>(),
      num_rejected.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(),
      cu_num_logits.data_ptr<int32_t>(),
      as_i32(idx_mapping).data_ptr<int32_t>(),
      prefill_len.data_ptr<int32_t>(),
      num_reqs);
}

void post_update_hip(
    torch::Tensor idx_mapping,
    torch::Tensor num_computed_tokens,
    torch::Tensor last_sampled_tokens,
    c10::optional<torch::Tensor> output_bin_counts,
    torch::Tensor sampled_tokens,
    torch::Tensor num_sampled,
    torch::Tensor num_rejected,
    c10::optional<torch::Tensor> query_start_loc,
    torch::Tensor all_token_ids,
    torch::Tensor total_len,
    c10::optional<torch::Tensor> broadcast_drafts,
    c10::optional<torch::Tensor> draft_tokens_out) {
  int num_reqs = (int)idx_mapping.size(0);
  if (num_reqs == 0) return;
  int32_t* obc_ptr = output_bin_counts.has_value() ? output_bin_counts->data_ptr<int32_t>() : nullptr;
  int64_t obc_s = output_bin_counts.has_value() ? output_bin_counts->stride(0) : 0;
  const int32_t* qsl_ptr = query_start_loc.has_value() ? query_start_loc->data_ptr<int32_t>() : nullptr;
  const int64_t* bd_ptr = broadcast_drafts.has_value() ? broadcast_drafts->data_ptr<int64_t>() : nullptr;
  int64_t bd_s = broadcast_drafts.has_value() ? broadcast_drafts->stride(0) : 0;
  int64_t* dt_ptr = draft_tokens_out.has_value() ? draft_tokens_out->data_ptr<int64_t>() : nullptr;
  int64_t dt_s = draft_tokens_out.has_value() ? draft_tokens_out->stride(0) : 0;
  int num_spec = broadcast_drafts.has_value() ? (int)broadcast_drafts->size(1) : 0;

  if (sampled_tokens.scalar_type() == torch::kInt64) {
    hipLaunchKernelGGL(
        k_post_update<int64_t>, dim3((num_reqs + 63) / 64), dim3(64), 0, cur_stream(),
        as_i32(idx_mapping).data_ptr<int32_t>(), num_computed_tokens.data_ptr<int32_t>(),
        last_sampled_tokens.data_ptr<int64_t>(), obc_ptr, obc_s,
        sampled_tokens.data_ptr<int64_t>(), sampled_tokens.stride(0),
        num_sampled.data_ptr<int32_t>(), num_rejected.data_ptr<int32_t>(),
        qsl_ptr, all_token_ids.data_ptr<int32_t>(), all_token_ids.stride(0),
        total_len.data_ptr<int32_t>(), bd_ptr, bd_s, dt_ptr, dt_s, num_spec, num_reqs);
  } else {
    hipLaunchKernelGGL(
        k_post_update<int32_t>, dim3((num_reqs + 63) / 64), dim3(64), 0, cur_stream(),
        as_i32(idx_mapping).data_ptr<int32_t>(), num_computed_tokens.data_ptr<int32_t>(),
        last_sampled_tokens.data_ptr<int64_t>(), obc_ptr, obc_s,
        sampled_tokens.data_ptr<int32_t>(), sampled_tokens.stride(0),
        num_sampled.data_ptr<int32_t>(), num_rejected.data_ptr<int32_t>(),
        qsl_ptr, all_token_ids.data_ptr<int32_t>(), all_token_ids.stride(0),
        total_len.data_ptr<int32_t>(), bd_ptr, bd_s, dt_ptr, dt_s, num_spec, num_reqs);
  }
}

void expand_idx_mapping_hip(
    torch::Tensor idx_mapping,
    torch::Tensor expanded_idx_mapping,
    torch::Tensor expanded_local_pos,
    torch::Tensor cu_num_logits) {
  int num_reqs = (int)idx_mapping.size(0);
  if (num_reqs == 0) return;
  if (idx_mapping.scalar_type() == torch::kInt64) {
    hipLaunchKernelGGL(
        k_expand_idx_mapping<int64_t>, dim3(num_reqs), dim3(64), 0, cur_stream(),
        idx_mapping.data_ptr<int64_t>(),
        expanded_idx_mapping.data_ptr<int64_t>(),
        expanded_local_pos.data_ptr<int64_t>(),
        cu_num_logits.data_ptr<int32_t>());
  } else {
    hipLaunchKernelGGL(
        k_expand_idx_mapping<int32_t>, dim3(num_reqs), dim3(64), 0, cur_stream(),
        as_i32(idx_mapping).data_ptr<int32_t>(),
        expanded_idx_mapping.data_ptr<int32_t>(),
        expanded_local_pos.data_ptr<int32_t>(),
        cu_num_logits.data_ptr<int32_t>());
  }
}

void prepare_prefill_inputs_hip(
    torch::Tensor last_token_indices,
    torch::Tensor draft_current_step,
    torch::Tensor draft_input_ids,
    torch::Tensor draft_positions,
    torch::Tensor draft_query_start_loc,
    torch::Tensor draft_seq_lens,
    torch::Tensor target_input_ids,
    torch::Tensor target_positions,
    torch::Tensor idx_mapping,
    torch::Tensor last_sampled,
    torch::Tensor next_prefill_tokens,
    torch::Tensor num_sampled,
    torch::Tensor num_rejected,
    torch::Tensor query_start_loc,
    torch::Tensor seq_lens,
    int max_num_reqs) {
  int num_reqs = (int)idx_mapping.size(0);
  if (num_reqs == 0) return;
  auto im32 = as_i32(idx_mapping);
  auto ls64 = (last_sampled.scalar_type() == torch::kInt64) ? last_sampled : last_sampled.to(torch::kInt64);
  if (last_token_indices.scalar_type() == torch::kInt64) {
    hipLaunchKernelGGL(
        k_prepare_prefill_inputs<int64_t>, dim3(num_reqs), dim3(256), 0, cur_stream(),
        last_token_indices.data_ptr<int64_t>(),
        draft_current_step.data_ptr<int64_t>(),
        draft_input_ids.data_ptr<int32_t>(),
        draft_positions.data_ptr<int64_t>(),
        draft_query_start_loc.data_ptr<int32_t>(),
        draft_seq_lens.data_ptr<int32_t>(),
        target_input_ids.data_ptr<int32_t>(),
        target_positions.data_ptr<int64_t>(),
        im32.data_ptr<int32_t>(),
        ls64.data_ptr<int64_t>(),
        next_prefill_tokens.data_ptr<int32_t>(),
        num_sampled.data_ptr<int32_t>(),
        num_rejected.data_ptr<int32_t>(),
        query_start_loc.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        num_reqs, max_num_reqs);
  } else {
    hipLaunchKernelGGL(
        k_prepare_prefill_inputs<int32_t>, dim3(num_reqs), dim3(256), 0, cur_stream(),
        last_token_indices.data_ptr<int32_t>(),
        draft_current_step.data_ptr<int32_t>(),
        draft_input_ids.data_ptr<int32_t>(),
        draft_positions.data_ptr<int64_t>(),
        draft_query_start_loc.data_ptr<int32_t>(),
        draft_seq_lens.data_ptr<int32_t>(),
        target_input_ids.data_ptr<int32_t>(),
        target_positions.data_ptr<int64_t>(),
        im32.data_ptr<int32_t>(),
        ls64.data_ptr<int64_t>(),
        next_prefill_tokens.data_ptr<int32_t>(),
        num_sampled.data_ptr<int32_t>(),
        num_rejected.data_ptr<int32_t>(),
        query_start_loc.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        num_reqs, max_num_reqs);
  }
}

void prepare_decode_inputs_hip(
    torch::Tensor draft_tokens,
    torch::Tensor target_seq_lens,
    torch::Tensor num_rejected,
    torch::Tensor input_ids,
    torch::Tensor positions,
    torch::Tensor sample_src_positions,
    torch::Tensor query_start_loc,
    torch::Tensor seq_lens,
    int max_model_len, int max_num_reqs, bool advance_pos) {
  int num_reqs = (int)draft_tokens.size(0);
  if (draft_tokens.scalar_type() == torch::kInt64) {
    hipLaunchKernelGGL(
        k_prepare_decode_inputs<int64_t>, dim3(num_reqs + 1), dim3(256), 0, cur_stream(),
        draft_tokens.data_ptr<int64_t>(), draft_tokens.stride(0),
        target_seq_lens.data_ptr<int32_t>(), num_rejected.data_ptr<int32_t>(),
        input_ids.data_ptr<int32_t>(), positions.data_ptr<int64_t>(),
        sample_src_positions.data_ptr<int64_t>(), query_start_loc.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(), num_reqs, max_model_len, max_num_reqs, advance_pos);
  } else {
    hipLaunchKernelGGL(
        k_prepare_decode_inputs<int32_t>, dim3(num_reqs + 1), dim3(256), 0, cur_stream(),
        draft_tokens.data_ptr<int32_t>(), draft_tokens.stride(0),
        target_seq_lens.data_ptr<int32_t>(), num_rejected.data_ptr<int32_t>(),
        input_ids.data_ptr<int32_t>(), positions.data_ptr<int64_t>(),
        sample_src_positions.data_ptr<int64_t>(), query_start_loc.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(), num_reqs, max_model_len, max_num_reqs, advance_pos);
  }
}

void prepare_rope_positions_hip(
    torch::Tensor positions,
    torch::Tensor prefill_positions,
    int64_t pp_stride0, int64_t pp_stride1,
    torch::Tensor prefill_delta,
    torch::Tensor idx_mapping,
    torch::Tensor query_start_loc,
    torch::Tensor prefill_lens,
    torch::Tensor num_computed_tokens,
    int num_dims) {
  int num_reqs = (int)idx_mapping.size(0);
  if (num_reqs == 0) return;
  auto im32 = as_i32(idx_mapping);
  if (prefill_positions.scalar_type() == torch::kInt32) {
    auto pd32 = as_i32(prefill_delta);
    hipLaunchKernelGGL(
        k_prepare_rope_positions<int32_t>, dim3(num_reqs), dim3(256), 0, cur_stream(),
        positions.data_ptr<int64_t>(), positions.stride(0),
        prefill_positions.data_ptr<int32_t>(), pp_stride0, pp_stride1,
        pd32.data_ptr<int32_t>(), im32.data_ptr<int32_t>(),
        query_start_loc.data_ptr<int32_t>(), prefill_lens.data_ptr<int32_t>(),
        num_computed_tokens.data_ptr<int32_t>(), num_dims);
  } else {
    auto pd64 = (prefill_delta.scalar_type() == torch::kInt64) ? prefill_delta : prefill_delta.to(torch::kInt64);
    hipLaunchKernelGGL(
        k_prepare_rope_positions<int64_t>, dim3(num_reqs), dim3(256), 0, cur_stream(),
        positions.data_ptr<int64_t>(), positions.stride(0),
        prefill_positions.data_ptr<int64_t>(), pp_stride0, pp_stride1,
        pd64.data_ptr<int64_t>(), im32.data_ptr<int32_t>(),
        query_start_loc.data_ptr<int32_t>(), prefill_lens.data_ptr<int32_t>(),
        num_computed_tokens.data_ptr<int32_t>(), num_dims);
  }
}

void scatter_num_accepted_hip(
    torch::Tensor idx_mapping,
    torch::Tensor num_sampled,
    torch::Tensor num_accepted) {
  int num_reqs = (int)idx_mapping.size(0);
  if (num_reqs == 0) return;
  hipLaunchKernelGGL(
      k_scatter_num_accepted, dim3((num_reqs + 63) / 64), dim3(64), 0, cur_stream(),
      as_i32(idx_mapping).data_ptr<int32_t>(), num_sampled.data_ptr<int32_t>(),
      num_accepted.data_ptr<int32_t>(), num_reqs);
}

void fill_num_accepted_hip(
    torch::Tensor idx_mapping,
    torch::Tensor num_accepted,
    int32_t val) {
  int num_reqs = (int)idx_mapping.size(0);
  if (num_reqs == 0) return;
  hipLaunchKernelGGL(
      k_fill_num_accepted, dim3((num_reqs + 63) / 64), dim3(64), 0, cur_stream(),
      as_i32(idx_mapping).data_ptr<int32_t>(), num_accepted.data_ptr<int32_t>(),
      val, num_reqs);
}

void gdn_prepare_pure_spec_decode_hip(
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor query_start_loc,
    torch::Tensor src_num_accepted,
    torch::Tensor dst_state_indices,
    torch::Tensor dst_spec_masks,
    torch::Tensor dst_spec_tok_idx,
    torch::Tensor dst_spec_qsl,
    torch::Tensor dst_num_accepted,
    int num_spec_decodes, int batch_size, int num_spec_tokens,
    int n_cols, int block_size, bool align_mode, int32_t null_block_id) {
  int bt_cols = (int)block_table.size(1);
  if (dst_spec_tok_idx.scalar_type() == torch::kInt64) {
    hipLaunchKernelGGL(
        k_gdn_prepare_pure_spec_decode<int64_t>, dim3(1), dim3(256), 0, cur_stream(),
        block_table.data_ptr<int32_t>(), block_table.stride(0), bt_cols,
        seq_lens.data_ptr<int32_t>(), query_start_loc.data_ptr<int32_t>(),
        src_num_accepted.data_ptr<int32_t>(),
        dst_state_indices.data_ptr<int32_t>(), dst_state_indices.stride(0),
        dst_spec_masks.data_ptr<bool>(), dst_spec_tok_idx.data_ptr<int64_t>(),
        dst_spec_qsl.data_ptr<int32_t>(), dst_num_accepted.data_ptr<int32_t>(),
        num_spec_decodes, batch_size, num_spec_tokens, n_cols, block_size, align_mode, null_block_id);
  } else {
    hipLaunchKernelGGL(
        k_gdn_prepare_pure_spec_decode<int32_t>, dim3(1), dim3(256), 0, cur_stream(),
        block_table.data_ptr<int32_t>(), block_table.stride(0), bt_cols,
        seq_lens.data_ptr<int32_t>(), query_start_loc.data_ptr<int32_t>(),
        src_num_accepted.data_ptr<int32_t>(),
        dst_state_indices.data_ptr<int32_t>(), dst_state_indices.stride(0),
        dst_spec_masks.data_ptr<bool>(), dst_spec_tok_idx.data_ptr<int32_t>(),
        dst_spec_qsl.data_ptr<int32_t>(), dst_num_accepted.data_ptr<int32_t>(),
        num_spec_decodes, batch_size, num_spec_tokens, n_cols, block_size, align_mode, null_block_id);
  }
}

void preprocess_mamba_align_fused_hip(
    torch::Tensor idx_mapping,
    torch::Tensor state_idx,
    torch::Tensor num_computed_tokens,
    torch::Tensor query_start_loc,
    torch::Tensor num_accepted_tokens,
    torch::Tensor src_col,
    torch::Tensor src_off,
    int num_reqs, int mamba_block_size) {
  if (num_reqs <= 0) return;
  hipLaunchKernelGGL(
      k_preprocess_mamba_align_fused, dim3((num_reqs + 255) / 256), dim3(256), 0, cur_stream(),
      as_i32(idx_mapping).data_ptr<int32_t>(), state_idx.data_ptr<int32_t>(),
      num_computed_tokens.data_ptr<int32_t>(), query_start_loc.data_ptr<int32_t>(),
      num_accepted_tokens.data_ptr<int32_t>(), src_col.data_ptr<int32_t>(),
      src_off.data_ptr<int32_t>(), num_reqs, mamba_block_size);
}

void precopy_mamba_align_fused_hip(
    torch::Tensor state_idx_gpu,
    torch::Tensor src_col_gpu,
    torch::Tensor token_bias_gpu,
    torch::Tensor block_table_ptrs, int64_t bt_stride_req,
    torch::Tensor state_base_addrs,
    torch::Tensor state_block_strides,
    torch::Tensor state_elem_sizes,
    torch::Tensor state_inner_sizes,
    torch::Tensor state_conv_widths,
    torch::Tensor state_group_indices,
    torch::Tensor state_dim_row_count,
    torch::Tensor state_dim_row_stride,
    c10::optional<torch::Tensor> idx_mapping,
    int num_reqs, int total_states, int temporal_tiles, bool conv_dim_first) {
  if (num_reqs <= 0 || total_states <= 0) return;
  torch::Tensor im_holder; const int32_t* im = nullptr; if (idx_mapping.has_value()) { im_holder = as_i32(*idx_mapping); im = im_holder.data_ptr<int32_t>(); }
  hipLaunchKernelGGL(
      k_precopy_mamba_align_fused, dim3(num_reqs, total_states, temporal_tiles), dim3(256), 0, cur_stream(),
      state_idx_gpu.data_ptr<int32_t>(), src_col_gpu.data_ptr<int32_t>(), token_bias_gpu.data_ptr<int32_t>(),
      reinterpret_cast<const int64_t*>(block_table_ptrs.data_ptr()), bt_stride_req,
      reinterpret_cast<const int64_t*>(state_base_addrs.data_ptr()),
      state_block_strides.data_ptr<int64_t>(), state_elem_sizes.data_ptr<int32_t>(),
      state_inner_sizes.data_ptr<int64_t>(), state_conv_widths.data_ptr<int32_t>(),
      state_group_indices.data_ptr<int32_t>(), state_dim_row_count.data_ptr<int32_t>(),
      state_dim_row_stride.data_ptr<int64_t>(), im, num_reqs, conv_dim_first, temporal_tiles);
}

void postprocess_mamba_align_v2_hip(
    torch::Tensor num_accepted_snap,
    torch::Tensor state_idx_gpu,
    torch::Tensor new_num_computed_gpu,
    torch::Tensor block_table_ptrs, int64_t bt_stride_req,
    torch::Tensor state_base_addrs,
    torch::Tensor state_block_strides,
    torch::Tensor state_elem_sizes,
    torch::Tensor state_inner_sizes,
    torch::Tensor state_conv_widths,
    torch::Tensor state_group_indices,
    torch::Tensor state_dim_row_count,
    torch::Tensor state_dim_row_stride,
    torch::Tensor num_accepted_out,
    torch::Tensor idx_mapping,
    int num_reqs, int total_states, int block_size, int temporal_tiles, bool conv_dim_first) {
  if (num_reqs <= 0 || total_states <= 0) return;
  hipLaunchKernelGGL(
      k_postprocess_mamba_align_v2, dim3(num_reqs, total_states, temporal_tiles), dim3(256), 0, cur_stream(),
      num_accepted_snap.data_ptr<int32_t>(), state_idx_gpu.data_ptr<int32_t>(),
      new_num_computed_gpu.data_ptr<int32_t>(),
      reinterpret_cast<const int64_t*>(block_table_ptrs.data_ptr()), bt_stride_req,
      reinterpret_cast<const int64_t*>(state_base_addrs.data_ptr()),
      state_block_strides.data_ptr<int64_t>(), state_elem_sizes.data_ptr<int32_t>(),
      state_inner_sizes.data_ptr<int64_t>(), state_conv_widths.data_ptr<int32_t>(),
      state_group_indices.data_ptr<int32_t>(), state_dim_row_count.data_ptr<int32_t>(),
      state_dim_row_stride.data_ptr<int64_t>(), num_accepted_out.data_ptr<int32_t>(),
      as_i32(idx_mapping).data_ptr<int32_t>(), num_reqs, block_size, conv_dim_first, temporal_tiles);
}

torch::Tensor ple_compute_ngram_ids_hip(
    torch::Tensor input_ids,
    torch::Tensor query_start_loc,
    torch::Tensor ngram_context,
    torch::Tensor multipliers,
    torch::Tensor vocab_sizes,
    torch::Tensor offsets,
    int ngram_size, int heads_per_ngram, int64_t eos_token_id) {
  auto ids_1d = input_ids.reshape(-1).contiguous();
  auto qsl = query_start_loc.contiguous();
  auto ctx = ngram_context;
  const int num_tokens = (int)ids_1d.size(0);
  const int num_reqs = (int)qsl.size(0) - 1;
  const int total_heads = (ngram_size - 1) * heads_per_ngram;
  TORCH_CHECK(ngram_size >= 2 && ngram_size <= 8, "ple_compute_ngram_ids: ngram_size out of range");
  auto out = torch::empty({num_tokens, total_heads},
                          torch::TensorOptions().dtype(torch::kInt64).device(ids_1d.device()));
  if (num_tokens == 0) return out;
  const int blocks = (num_tokens + 127) / 128;
  // Support int32/int64 combinations cleanly without .long() temp copies!
  auto launch = [&](auto id_tag, auto qsl_tag, auto ctx_tag) {
    using IdT = decltype(id_tag);
    using QslT = decltype(qsl_tag);
    using CtxT = decltype(ctx_tag);
    hipLaunchKernelGGL(
        (k_ple_compute_ngram_ids<IdT, QslT, CtxT>), dim3(blocks), dim3(128), 0, cur_stream(),
        ids_1d.data_ptr<IdT>(), qsl.data_ptr<QslT>(), ctx.data_ptr<CtxT>(),
        ctx.stride(0), (int)ctx.size(0), (int)ctx.size(1),
        multipliers.data_ptr<int64_t>(), vocab_sizes.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
        out.data_ptr<int64_t>(), num_tokens, num_reqs, ngram_size, heads_per_ngram, eos_token_id);
  };
  bool id64 = (ids_1d.scalar_type() == torch::kInt64);
  bool q64 = (qsl.scalar_type() == torch::kInt64);
  bool c64 = (ctx.scalar_type() == torch::kInt64);
  if (!id64 && !q64 && c64) launch(int32_t{}, int32_t{}, int64_t{});
  else if (!id64 && !q64 && !c64) launch(int32_t{}, int32_t{}, int32_t{});
  else if (id64 && q64 && c64) launch(int64_t{}, int64_t{}, int64_t{});
  else if (id64 && !q64 && c64) launch(int64_t{}, int32_t{}, int64_t{});
  else if (!id64 && q64 && c64) launch(int32_t{}, int64_t{}, int64_t{});
  else {
    auto i64 = ids_1d.to(torch::kInt64);
    auto qsl64 = qsl.to(torch::kInt64);
    auto ctx64 = ctx.to(torch::kInt64);
    hipLaunchKernelGGL(
        (k_ple_compute_ngram_ids<int64_t, int64_t, int64_t>), dim3(blocks), dim3(128), 0, cur_stream(),
        i64.data_ptr<int64_t>(), qsl64.data_ptr<int64_t>(), ctx64.data_ptr<int64_t>(),
        ctx64.stride(0), (int)ctx64.size(0), (int)ctx64.size(1),
        multipliers.data_ptr<int64_t>(), vocab_sizes.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
        out.data_ptr<int64_t>(), num_tokens, num_reqs, ngram_size, heads_per_ngram, eos_token_id);
  }
  return out;
}

void ple_prepare_pure_spec_decode_hip(
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor query_start_loc,
    torch::Tensor src_num_accepted,
    torch::Tensor dst_state_indices,
    torch::Tensor dst_spec_masks,
    torch::Tensor dst_spec_tok_idx,
    torch::Tensor dst_spec_qsl,
    torch::Tensor dst_num_accepted,
    int num_spec_decodes, int batch_size, int num_spec_tokens,
    int block_size, bool align_mode, int32_t null_block_id) {
  hipLaunchKernelGGL(
      k_ple_prepare_pure_spec_decode, dim3(1), dim3(256), 0, cur_stream(),
      block_table.data_ptr<int32_t>(), block_table.stride(0), (int)block_table.size(1),
      seq_lens.data_ptr<int32_t>(), query_start_loc.data_ptr<int32_t>(),
      src_num_accepted.data_ptr<int32_t>(),
      dst_state_indices.data_ptr<int32_t>(),
      dst_spec_masks.data_ptr<bool>(), dst_spec_tok_idx.data_ptr<int32_t>(),
      dst_spec_qsl.data_ptr<int32_t>(), dst_num_accepted.data_ptr<int32_t>(),
      num_spec_decodes, batch_size, num_spec_tokens, block_size, align_mode, null_block_id);
}

PYBIND11_MODULE(htr_decode_meta, m) {
  m.def("build_qsa_metadata", &build_qsa_metadata_hip);
  m.def("compute_slot_mappings", &compute_slot_mappings_hip);
  m.def("gather_block_tables", &gather_block_tables_hip);
  m.def("prepare_pos_seq_lens", &prepare_pos_seq_lens_hip);
  m.def("combine_sampled_and_draft", &combine_sampled_and_draft_hip);
  m.def("get_num_sampled_and_rejected", &get_num_sampled_and_rejected_hip);
  m.def("post_update", &post_update_hip);
  m.def("expand_idx_mapping", &expand_idx_mapping_hip);
  m.def("prepare_prefill_inputs", &prepare_prefill_inputs_hip);
  m.def("prepare_decode_inputs", &prepare_decode_inputs_hip);
  m.def("prepare_rope_positions", &prepare_rope_positions_hip);
  m.def("scatter_num_accepted", &scatter_num_accepted_hip);
  m.def("fill_num_accepted", &fill_num_accepted_hip);
  m.def("gdn_prepare_pure_spec_decode", &gdn_prepare_pure_spec_decode_hip);
  m.def("preprocess_mamba_align_fused", &preprocess_mamba_align_fused_hip);
  m.def("precopy_mamba_align_fused", &precopy_mamba_align_fused_hip);
  m.def("postprocess_mamba_align_v2", &postprocess_mamba_align_v2_hip);
  m.def("ple_compute_ngram_ids", &ple_compute_ngram_ids_hip);
  m.def("ple_prepare_pure_spec_decode", &ple_prepare_pure_spec_decode_hip);
}
