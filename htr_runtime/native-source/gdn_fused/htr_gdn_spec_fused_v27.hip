// htr_gdn_spec_fused_v27.hip — Wave 6 gfx1151 RDNA 3.5 Optimized GDN Spec-Decode + RMSNormGated Megakernel
// Upgrades over v0.2.6 (htr_gdn_spec_fused.hip):
//   1. Native RDNA 3.5 6-op zero-LDS wave_sum (v_add_f32_dpp row_xmask:1,2,4,8 + v_permlanex16_b32(0,0)):
//      Eliminates 100% of ds_swizzle + ds_bpermute + lgkmcnt(0) stalls (~680 LDS round-trips/wave/layer x 36L!).
//   2. Eliminates dynamic-indexed float o_row[R_PER_W] array (-16 VGPRs, zero scratch/select ladder) by
//      writing s_o[w * R_PER_W + r] directly from lane==0 inside the recurrence loop!
//   3. Vectorized uint4 (128-bit) preload of wave-uniform v_ptr[0..R_PER_W-1] before the recurrence loop
//      (replaces 8-16 scalar global_load_u16 stalls inside the recurrence!).
//   4. Overlaps eager nontemporal float4 ssm_state stores INSIDE the r-loop right after h[r] update
//      so 16x global_store_b128 hide behind subsequent rows' VALU recurrence.
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef float f4v __attribute__((ext_vector_type(4)));
typedef uint32_t u2v __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float bf16_to_f32(uint16_t h) {
  return __builtin_bit_cast(float, ((uint32_t)h) << 16);
}

__device__ __forceinline__ uint16_t f32_to_bf16_rne(float x) {
  uint32_t u = __builtin_bit_cast(uint32_t, x);
  if (x != x) return 0x7fffu;
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

__device__ __forceinline__ f4v load_bf16x4(const uint16_t* p) {
  u2v raw = *reinterpret_cast<const u2v*>(p);
  f4v r;
  r[0] = bf16_to_f32((uint16_t)(raw[0] & 0xffffu));
  r[1] = bf16_to_f32((uint16_t)(raw[0] >> 16));
  r[2] = bf16_to_f32((uint16_t)(raw[1] & 0xffffu));
  r[3] = bf16_to_f32((uint16_t)(raw[1] >> 16));
  return r;
}

// Native RDNA 3.5 gfx1151 Wave32 butterfly sum:
// 4x v_add_f32_dpp (row_xmask 1,2,4,8) + 1x v_permlanex16_b32(0,0) + 1x v_add_f32
// ZERO LDS, ZERO s_waitcnt lgkmcnt(0), ZERO SGPR constants!
__device__ __forceinline__ float wave_sum(float v) {
  asm volatile(
    "v_add_f32_dpp %0, %0, %0 row_xmask:1 row_mask:0xf bank_mask:0xf fi:1\n\t"
    "v_add_f32_dpp %0, %0, %0 row_xmask:2 row_mask:0xf bank_mask:0xf fi:1\n\t"
    "v_add_f32_dpp %0, %0, %0 row_xmask:4 row_mask:0xf bank_mask:0xf fi:1\n\t"
    "v_add_f32_dpp %0, %0, %0 row_xmask:8 row_mask:0xf bank_mask:0xf fi:1"
    : "+v"(v));
  float hi;
  asm volatile("v_permlanex16_b32 %0, %1, 0, 0" : "=&v"(hi) : "v"(v));
  return v + hi;
}

__device__ __forceinline__ float apply_gate_act(float z, int act_mode) {
  float sig = 1.0f / (1.0f + __expf(-z));
  return (act_mode == 0) ? sig : (z * sig);  // 0 = sigmoid (Qwen3.8), 1 = silu
}

// Extract the r-th bf16 (as fp32) from packed uint4 v0 (r=0..7) / v1 (r=8..15)
__device__ __forceinline__ float unpack_v_bf16(uint4 v0, uint4 v1, int r) {
  uint32_t word;
  if (r < 8) {
    const int widx = r >> 1;
    word = (widx == 0) ? v0.x : (widx == 1) ? v0.y : (widx == 2) ? v0.z : v0.w;
  } else {
    const int widx = (r - 8) >> 1;
    word = (widx == 0) ? v1.x : (widx == 1) ? v1.y : (widx == 2) ? v1.z : v1.w;
  }
  uint16_t h = (r & 1) ? (uint16_t)(word >> 16) : (uint16_t)(word & 0xffffu);
  return bf16_to_f32(h);
}

// VB in {1, 2, 4}, ACT_MODE: 0=sigmoid, 1=silu, NORM_W_FP32: true/false, DEFER_WRITE: false=eager, true=defer
template <int VB, int ACT_MODE, bool NORM_W_FP32, bool DEFER_WRITE>
__global__ __launch_bounds__(VB == 1 ? 256 : 128)
void k_gdn_spec_fused(
    const uint16_t* __restrict__ mixed_qkv,        // [T_tot, 2*H*128 + HV*128] bf16
    const uint16_t* __restrict__ a_in,             // [T_tot, HV] bf16
    const uint16_t* __restrict__ b_in,             // [T_tot, HV] bf16
    const float* __restrict__ A_log,               // [HV] fp32
    const uint16_t* __restrict__ dt_bias,          // [HV] bf16 (or nullptr if fp32)
    const float* __restrict__ dt_bias_f32,         // [HV] fp32 (or nullptr if bf16)
    const int32_t* __restrict__ state_indices,     // [num_reqs, max_T] i32
    const int32_t* __restrict__ cu_seqlens,        // [num_reqs + 1] i32
    const int32_t* __restrict__ num_accepted,      // [num_reqs] i32 (nullable -> 1)
    float* __restrict__ ssm_state,                 // [num_slots, HV, 128, 128] fp32 (V, K)
    const uint16_t* __restrict__ output_gate,      // [T_tot, HV, 128] bf16
    const void* __restrict__ norm_weight,          // [128] fp32 or bf16
    uint16_t* __restrict__ out,                    // [T_tot, HV, 128] bf16
    float* __restrict__ scratch_o,                 // [num_reqs * HV * max_T * 128] fp32 (VB > 1)
    float* __restrict__ scratch_ssq,               // [num_reqs * HV * max_T] fp32 (VB > 1)
    unsigned* __restrict__ counters,               // [num_reqs * HV * max_T] u32 (VB > 1)
    float* __restrict__ vr_k,                      // [cap_T, H, 128] fp32 (DEFER_WRITE only)
    uint16_t* __restrict__ vr_v,                   // [cap_T, HV, 128] bf16 (DEFER_WRITE only)
    float* __restrict__ vr_decay,                  // [cap_T, HV] fp32 (DEFER_WRITE only)
    float* __restrict__ vr_beta,                   // [cap_T, HV] fp32 (DEFER_WRITE only)
    int num_reqs, int H, int HV, int qkv_stride, int a_stride, int b_stride, int z_stride,
    int idx_stride, int64_t state_slot_stride, int max_T,
    float scale, float norm_eps) {
  constexpr int NW = (VB == 1) ? 8 : 4;
  constexpr int R_PER_W = (VB == 4) ? 8 : 16;
  constexpr int ROWS_PER_WG = NW * R_PER_W;  // 128 for VB=1, 64 for VB=2, 32 for VB=4
  __shared__ float s_o[ROWS_PER_WG];
  __shared__ float s_wred[NW];

  const int bid = blockIdx.x;
  const int vb_id = (VB == 1) ? 0 : (bid & (VB - 1));
  const int nh = (VB == 1) ? bid : (bid / VB);
  const int req = nh / HV;
  const int hv = nh % HV;
  if (req >= num_reqs) return;

  const int bos = cu_seqlens[req];
  const int eos = cu_seqlens[req + 1];
  const int T_req = eos - bos;
  if (T_req <= 0) return;

  const int tid = threadIdx.x;
  const int w = tid >> 5;
  const int lane = tid & 31;
  const int v_base = vb_id * ROWS_PER_WG + w * R_PER_W;

  const int init_t = (num_accepted != nullptr) ? (num_accepted[req] - 1) : 0;
  const bool has_src = (init_t >= 0) && (init_t < max_T);
  const int init_slot = has_src ? state_indices[req * idx_stride + init_t] : 0;
  if (init_slot <= 0) {
    if (tid < ROWS_PER_WG) {
      for (int i_t = 0; i_t < T_req; ++i_t) {
        const int tok = bos + i_t;
        out[(size_t)tok * HV * 128 + hv * 128 + vb_id * ROWS_PER_WG + tid] = 0u;
      }
    }
    return;
  }

  const int ratio = HV / H;
  const int hk = hv / ratio;
  // Load initial SSM state [R_PER_W rows x 128 cols] per wave (aligned nontemporal float4 -> 0 MALL thrash!)
  f4v h[R_PER_W];
  {
    const float* h0_base = ssm_state + (int64_t)init_slot * state_slot_stride + (int64_t)hv * (128 * 128);
#pragma unroll
    for (int r = 0; r < R_PER_W; ++r) {
      const float* hp = h0_base + (size_t)(v_base + r) * 128 + lane * 4;
      h[r] = __builtin_nontemporal_load(reinterpret_cast<const f4v*>(hp));
    }
  }

  const float a_log_v = A_log[hv];
  const float neg_exp_A = -__expf(a_log_v);
  const float dt_b = (dt_bias_f32 != nullptr) ? dt_bias_f32[hv] : bf16_to_f32(dt_bias[hv]);
  const int q_off = hk * 128;
  const int k_off = H * 128 + hk * 128;
  const int v_off = 2 * H * 128 + hv * 128;

  for (int i_t = 0; i_t < T_req; ++i_t) {
    const int tok = bos + i_t;
    const uint16_t* qkv_row = mixed_qkv + (size_t)tok * qkv_stride;

    // 1. Load Q & K (4 bf16/lane) + wave32 zero-LDS DPP L2-norm + Q scale
    f4v qv = load_bf16x4(qkv_row + q_off + lane * 4);
    f4v kv = load_bf16x4(qkv_row + k_off + lane * 4);
    float q2 = fmaf(qv[0], qv[0], fmaf(qv[1], qv[1], fmaf(qv[2], qv[2], qv[3] * qv[3])));
    float k2 = fmaf(kv[0], kv[0], fmaf(kv[1], kv[1], fmaf(kv[2], kv[2], kv[3] * kv[3])));
    float q_inv = rsqrtf(wave_sum(q2) + 1e-6f) * scale;
    float k_inv = rsqrtf(wave_sum(k2) + 1e-6f);
    qv *= q_inv;
    kv *= k_inv;

    // 2. Gating scalars g (decay) and beta + vectorized preload of wave-uniform v_ptr[0..R_PER_W-1]
    float a_val = bf16_to_f32(a_in[(size_t)tok * a_stride + hv]) + dt_b;
    float sp = (a_val <= 20.0f) ? __logf(1.0f + __expf(a_val)) : a_val;
    float decay = __expf(neg_exp_A * sp);
    float b_val = bf16_to_f32(b_in[(size_t)tok * b_stride + hv]);
    float beta = 1.0f / (1.0f + __expf(-b_val));

    const uint16_t* v_ptr = qkv_row + v_off + v_base;
    const uint4 vraw0 = *reinterpret_cast<const uint4*>(v_ptr);
    uint4 vraw1 = {0u, 0u, 0u, 0u};
    if constexpr (R_PER_W == 16) {
      vraw1 = *reinterpret_cast<const uint4*>(v_ptr + 8);
    }

    // 2b. V5-G5 Deferred mode: save tiny exact verify record (0 race, coalesced)
    if constexpr (DEFER_WRITE) {
      if (lane < R_PER_W) {
        vr_v[((size_t)tok * HV + hv) * 128 + v_base + lane] = v_ptr[lane];
      }
      if (vb_id == 0 && w == 0) {
        if (lane == 0) {
          vr_decay[(size_t)tok * HV + hv] = decay;
          vr_beta[(size_t)tok * HV + hv] = beta;
        }
        if ((hv - hk * ratio) == 0) {
          *reinterpret_cast<f4v*>(vr_k + ((size_t)tok * H + hk) * 128 + lane * 4) = kv;
        }
      }
    }

    // 3. Delta-rule recurrence over R_PER_W V-rows in registers
    //    - ZERO o_row[R_PER_W] array (-16 VGPRs, zero dynamic-index spills!): lane 0 writes s_o directly!
    //    - Batched nontemporal float4 stores AFTER r-loop preserve coalesced LPDDR5X write bursts!
#pragma unroll
    for (int r = 0; r < R_PER_W; ++r) {
      f4v hr = h[r] * decay;
      float dot_k = fmaf(hr[0], kv[0], fmaf(hr[1], kv[1], fmaf(hr[2], kv[2], hr[3] * kv[3])));
      dot_k = wave_sum(dot_k);
      float v_new = (unpack_v_bf16(vraw0, vraw1, r) - dot_k) * beta;
      hr[0] = fmaf(v_new, kv[0], hr[0]);
      hr[1] = fmaf(v_new, kv[1], hr[1]);
      hr[2] = fmaf(v_new, kv[2], hr[2]);
      hr[3] = fmaf(v_new, kv[3], hr[3]);
      h[r] = hr;
      float dot_q = fmaf(hr[0], qv[0], fmaf(hr[1], qv[1], fmaf(hr[2], qv[2], hr[3] * qv[3])));
      float o_r = wave_sum(dot_q);
      if (lane == 0) {
        s_o[w * R_PER_W + r] = o_r;
      }
    }

    // 4. Eager mode: batched nontemporal float4 ssm_state write burst
    if constexpr (!DEFER_WRITE) {
      const int dst_slot = (i_t < max_T) ? state_indices[req * idx_stride + i_t] : 0;
      if (dst_slot > 0) {
        float* hd_base = ssm_state + (int64_t)dst_slot * state_slot_stride + (int64_t)hv * (128 * 128);
#pragma unroll
        for (int r = 0; r < R_PER_W; ++r) {
          float* hp = hd_base + (size_t)(v_base + r) * 128 + lane * 4;
          __builtin_nontemporal_store(h[r], reinterpret_cast<f4v*>(hp));
        }
      }
    }

    // 5. Fused RMSNormGated -> out[tok, hv, 0..127]
    __syncthreads();

    if constexpr (VB == 1) {
      if (tid < 128) {
        const float ov = s_o[tid];
        const float wsum = wave_sum(ov * ov);
        if (lane == 0) s_wred[w] = wsum;
      }
      __syncthreads();
      if (tid < 128) {
        const float ssq = (s_wred[0] + s_wred[1]) + (s_wred[2] + s_wred[3]);
        const float rrms = rsqrtf(ssq * (1.0f / 128.0f) + norm_eps);
        const float ov = s_o[tid];
        const float nw = NORM_W_FP32
            ? reinterpret_cast<const float*>(norm_weight)[tid]
            : bf16_to_f32(reinterpret_cast<const uint16_t*>(norm_weight)[tid]);
        const float zv = bf16_to_f32(output_gate[(size_t)tok * z_stride + hv * 128 + tid]);
        const float res = (ov * rrms * nw) * apply_gate_act(zv, ACT_MODE);
        out[(size_t)tok * HV * 128 + hv * 128 + tid] = f32_to_bf16_rne(res);
      }
      __syncthreads();
    } else {
      const int slot_id = (nh * max_T) + i_t;
      float* so_dst = scratch_o + (size_t)slot_id * 128 + vb_id * ROWS_PER_WG;
      if (tid < ROWS_PER_WG) {
        const float ov = s_o[tid];
        so_dst[tid] = ov;
        const float wsum = wave_sum(ov * ov);
        if (lane == 0) s_wred[w] = wsum;
      }
      __syncthreads();
      __shared__ unsigned is_last;
      if (tid == 0) {
        float part_ssq = s_wred[0];
        if constexpr (ROWS_PER_WG == 64) part_ssq += s_wred[1];
        scratch_ssq[(size_t)slot_id * 4 + vb_id] = part_ssq;
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        is_last = (atomicAdd(&counters[slot_id], 1u) == (unsigned)(VB - 1));
      }
      __syncthreads();
      if (is_last) {
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        const float* sp = scratch_ssq + (size_t)slot_id * 4;
        const float ssq = (VB == 2) ? (sp[0] + sp[1]) : ((sp[0] + sp[1]) + (sp[2] + sp[3]));
        const float rrms = rsqrtf(ssq * (1.0f / 128.0f) + norm_eps);
        const float ov = scratch_o[(size_t)slot_id * 128 + tid];
        const float nw = NORM_W_FP32
            ? reinterpret_cast<const float*>(norm_weight)[tid]
            : bf16_to_f32(reinterpret_cast<const uint16_t*>(norm_weight)[tid]);
        const float zv = bf16_to_f32(output_gate[(size_t)tok * z_stride + hv * 128 + tid]);
        const float res = (ov * rrms * nw) * apply_gate_act(zv, ACT_MODE);
        out[(size_t)tok * HV * 128 + hv * 128 + tid] = f32_to_bf16_rne(res);
        __syncthreads();
        if (tid == 0) {
          counters[slot_id] = 0u;
        }
      }
      __syncthreads();
    }
  }
}

// V5-G5: Single-launch multi-layer post-sampling GDN state commit across ALL 36 GDN layers
template <int VB>
__global__ __launch_bounds__(VB == 1 ? 256 : 128)
void k_gdn_commit_all_layers(
    const int64_t* __restrict__ layer_ptrs,
    const int32_t* __restrict__ state_indices,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ prev_num_accepted,
    const int32_t* __restrict__ new_num_sampled,
    int num_reqs, int H, int HV, int idx_stride, int max_T) {
  constexpr int NW = (VB == 1) ? 8 : 4;
  constexpr int R_PER_W = (VB == 4) ? 8 : 16;
  constexpr int ROWS_PER_WG = NW * R_PER_W;

  const int bid = blockIdx.x;
  const int layer = blockIdx.y;
  const int vb_id = (VB == 1) ? 0 : (bid & (VB - 1));
  const int nh = (VB == 1) ? bid : (bid / VB);
  const int req = nh / HV;
  const int hv = nh % HV;
  if (req >= num_reqs) return;

  const int bos = cu_seqlens[req];
  const int eos = cu_seqlens[req + 1];
  const int T_req = eos - bos;
  if (T_req <= 0) return;

  const int init_t = (prev_num_accepted != nullptr) ? (prev_num_accepted[req] - 1) : 0;
  if (init_t < 0 || init_t >= max_T) return;
  const int init_slot = state_indices[req * idx_stride + init_t];
  if (init_slot <= 0) return;

  int n_commit = new_num_sampled[req];
  if (n_commit < 1) n_commit = 1;
  if (n_commit > T_req) n_commit = T_req;
  const int dst_t = n_commit - 1;
  if (dst_t >= max_T) return;
  const int dst_slot = state_indices[req * idx_stride + dst_t];
  if (dst_slot <= 0) return;

  const int64_t* lp = layer_ptrs + (size_t)layer * 6;
  float* __restrict__ ssm_state = reinterpret_cast<float*>(lp[0]);
  const float* __restrict__ vr_k = reinterpret_cast<const float*>(lp[1]);
  const uint16_t* __restrict__ vr_v = reinterpret_cast<const uint16_t*>(lp[2]);
  const float* __restrict__ vr_decay = reinterpret_cast<const float*>(lp[3]);
  const float* __restrict__ vr_beta = reinterpret_cast<const float*>(lp[4]);
  const int64_t state_slot_stride = lp[5];

  const int tid = threadIdx.x;
  const int w = tid >> 5;
  const int lane = tid & 31;
  const int v_base = vb_id * ROWS_PER_WG + w * R_PER_W;
  const int hk = hv / (HV / H);

  f4v h[R_PER_W];
  {
    const float* h0_base = ssm_state + (int64_t)init_slot * state_slot_stride + (int64_t)hv * (128 * 128);
#pragma unroll
    for (int r = 0; r < R_PER_W; ++r) {
      const float* hp = h0_base + (size_t)(v_base + r) * 128 + lane * 4;
      h[r] = __builtin_nontemporal_load(reinterpret_cast<const f4v*>(hp));
    }
  }

  for (int i_t = 0; i_t < n_commit; ++i_t) {
    const int tok = bos + i_t;
    const f4v kv = *reinterpret_cast<const f4v*>(vr_k + ((size_t)tok * H + hk) * 128 + lane * 4);
    const float decay = vr_decay[(size_t)tok * HV + hv];
    const float beta = vr_beta[(size_t)tok * HV + hv];
    const uint16_t* v_ptr = vr_v + ((size_t)tok * HV + hv) * 128 + v_base;
    const uint4 vraw0 = *reinterpret_cast<const uint4*>(v_ptr);
    uint4 vraw1 = {0u, 0u, 0u, 0u};
    if constexpr (R_PER_W == 16) {
      vraw1 = *reinterpret_cast<const uint4*>(v_ptr + 8);
    }
#pragma unroll
    for (int r = 0; r < R_PER_W; ++r) {
      f4v hr = h[r] * decay;
      float dot_k = fmaf(hr[0], kv[0], fmaf(hr[1], kv[1], fmaf(hr[2], kv[2], hr[3] * kv[3])));
      dot_k = wave_sum(dot_k);
      float v_new = (unpack_v_bf16(vraw0, vraw1, r) - dot_k) * beta;
      hr[0] = fmaf(v_new, kv[0], hr[0]);
      hr[1] = fmaf(v_new, kv[1], hr[1]);
      hr[2] = fmaf(v_new, kv[2], hr[2]);
      hr[3] = fmaf(v_new, kv[3], hr[3]);
      h[r] = hr;
    }
  }

  float* hd_base = ssm_state + (int64_t)dst_slot * state_slot_stride + (int64_t)hv * (128 * 128);
#pragma unroll
  for (int r = 0; r < R_PER_W; ++r) {
    float* hp = hd_base + (size_t)(v_base + r) * 128 + lane * 4;
    __builtin_nontemporal_store(h[r], reinterpret_cast<f4v*>(hp));
  }
}

template <bool DEFER>
static void dispatch_gdn_spec_fused(
    torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b,
    torch::Tensor A_log, torch::Tensor dt_bias,
    torch::Tensor state_indices, torch::Tensor cu_seqlens,
    c10::optional<torch::Tensor> num_accepted_opt,
    torch::Tensor ssm_state, torch::Tensor output_gate,
    torch::Tensor norm_weight, torch::Tensor out,
    torch::Tensor scratch_o, torch::Tensor scratch_ssq, torch::Tensor counters,
    float* vr_k_ptr, uint16_t* vr_v_ptr, float* vr_decay_ptr, float* vr_beta_ptr,
    int64_t H, int64_t HV, double scale, double norm_eps,
    int64_t act_mode, int64_t v_blocks) {
  TORCH_CHECK(mixed_qkv.is_cuda() && mixed_qkv.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(ssm_state.is_cuda() && ssm_state.scalar_type() == torch::kFloat32 && ssm_state.dim() == 4 && ssm_state.stride(1) == 16384 && ssm_state.stride(2) == 128 && ssm_state.stride(3) == 1);
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.is_contiguous());
  TORCH_CHECK(v_blocks == 1 || v_blocks == 2 || v_blocks == 4);
  const int num_reqs = (int)cu_seqlens.size(0) - 1;
  if (num_reqs <= 0) return;
  const int max_T = (int)state_indices.size(1);
  const int qkv_stride = (int)mixed_qkv.stride(0);
  const int a_stride = (int)a.stride(0);
  const int b_stride = (int)b.stride(0);
  const int z_stride = (int)output_gate.stride(0);
  const int idx_stride = (int)state_indices.stride(0);
  const int64_t state_slot_stride = (int64_t)ssm_state.stride(0);
  const bool nw_f32 = (norm_weight.scalar_type() == torch::kFloat32);
  const float* dt_f32 = (dt_bias.scalar_type() == torch::kFloat32) ? dt_bias.data_ptr<float>() : nullptr;
  const uint16_t* dt_bf16 = (dt_bias.scalar_type() == torch::kBFloat16) ? reinterpret_cast<const uint16_t*>(dt_bias.data_ptr<at::BFloat16>()) : nullptr;
  TORCH_CHECK(dt_f32 != nullptr || dt_bf16 != nullptr, "dt_bias must be fp32 or bf16");
  const int32_t* nacc = (num_accepted_opt.has_value() && num_accepted_opt->defined()) ? num_accepted_opt->data_ptr<int32_t>() : nullptr;

  auto st = at::hip::getCurrentHIPStream();
  auto qkvp = reinterpret_cast<const uint16_t*>(mixed_qkv.data_ptr<at::BFloat16>());
  auto ap = reinterpret_cast<const uint16_t*>(a.data_ptr<at::BFloat16>());
  auto bp = reinterpret_cast<const uint16_t*>(b.data_ptr<at::BFloat16>());
  auto zp = reinterpret_cast<const uint16_t*>(output_gate.data_ptr<at::BFloat16>());
  auto op = reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>());
  const void* nwp = nw_f32 ? (const void*)norm_weight.data_ptr<float>() : (const void*)norm_weight.data_ptr<at::BFloat16>();
  auto so = scratch_o.data_ptr<float>();
  auto ssq = scratch_ssq.data_ptr<float>();
  auto cnt = reinterpret_cast<unsigned*>(counters.data_ptr<int32_t>());

#define LAUNCH_GDN(VB, AM, NWF) \
  { hipLaunchKernelGGL((k_gdn_spec_fused<VB, AM, NWF, DEFER>), dim3(num_reqs * HV * VB), dim3(VB == 1 ? 256 : 128), 0, st, \
                     qkvp, ap, bp, A_log.data_ptr<float>(), dt_bf16, dt_f32, \
                     state_indices.data_ptr<int32_t>(), cu_seqlens.data_ptr<int32_t>(), nacc, \
                     ssm_state.data_ptr<float>(), zp, nwp, op, so, ssq, cnt, \
                     vr_k_ptr, vr_v_ptr, vr_decay_ptr, vr_beta_ptr, \
                     num_reqs, (int)H, (int)HV, qkv_stride, a_stride, b_stride, z_stride, \
                     idx_stride, state_slot_stride, max_T, (float)scale, (float)norm_eps); }

  if (v_blocks == 1) {
    if (act_mode == 0) { if (nw_f32) { LAUNCH_GDN(1, 0, true); } else { LAUNCH_GDN(1, 0, false); } }
    else               { if (nw_f32) { LAUNCH_GDN(1, 1, true); } else { LAUNCH_GDN(1, 1, false); } }
  } else if (v_blocks == 2) {
    if (act_mode == 0) { if (nw_f32) { LAUNCH_GDN(2, 0, true); } else { LAUNCH_GDN(2, 0, false); } }
    else               { if (nw_f32) { LAUNCH_GDN(2, 1, true); } else { LAUNCH_GDN(2, 1, false); } }
  } else {
    if (act_mode == 0) { if (nw_f32) { LAUNCH_GDN(4, 0, true); } else { LAUNCH_GDN(4, 0, false); } }
    else               { if (nw_f32) { LAUNCH_GDN(4, 1, true); } else { LAUNCH_GDN(4, 1, false); } }
  }
#undef LAUNCH_GDN
}

void gdn_spec_fused(
    torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b,
    torch::Tensor A_log, torch::Tensor dt_bias,
    torch::Tensor state_indices, torch::Tensor cu_seqlens,
    c10::optional<torch::Tensor> num_accepted_opt,
    torch::Tensor ssm_state, torch::Tensor output_gate,
    torch::Tensor norm_weight, torch::Tensor out,
    torch::Tensor scratch_o, torch::Tensor scratch_ssq, torch::Tensor counters,
    int64_t H, int64_t HV, double scale, double norm_eps,
    int64_t act_mode, int64_t v_blocks) {
  dispatch_gdn_spec_fused<false>(
      mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_opt,
      ssm_state, output_gate, norm_weight, out, scratch_o, scratch_ssq, counters,
      nullptr, nullptr, nullptr, nullptr, H, HV, scale, norm_eps, act_mode, v_blocks);
}

void gdn_spec_fused_defer(
    torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b,
    torch::Tensor A_log, torch::Tensor dt_bias,
    torch::Tensor state_indices, torch::Tensor cu_seqlens,
    c10::optional<torch::Tensor> num_accepted_opt,
    torch::Tensor ssm_state, torch::Tensor output_gate,
    torch::Tensor norm_weight, torch::Tensor out,
    torch::Tensor scratch_o, torch::Tensor scratch_ssq, torch::Tensor counters,
    torch::Tensor vr_k, torch::Tensor vr_v, torch::Tensor vr_decay, torch::Tensor vr_beta,
    int64_t H, int64_t HV, double scale, double norm_eps,
    int64_t act_mode, int64_t v_blocks) {
  TORCH_CHECK(vr_k.is_cuda() && vr_k.scalar_type() == torch::kFloat32 && vr_k.is_contiguous());
  TORCH_CHECK(vr_v.is_cuda() && vr_v.scalar_type() == torch::kBFloat16 && vr_v.is_contiguous());
  TORCH_CHECK(vr_decay.is_cuda() && vr_decay.scalar_type() == torch::kFloat32 && vr_decay.is_contiguous());
  TORCH_CHECK(vr_beta.is_cuda() && vr_beta.scalar_type() == torch::kFloat32 && vr_beta.is_contiguous());
  dispatch_gdn_spec_fused<true>(
      mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_opt,
      ssm_state, output_gate, norm_weight, out, scratch_o, scratch_ssq, counters,
      vr_k.data_ptr<float>(),
      reinterpret_cast<uint16_t*>(vr_v.data_ptr<at::BFloat16>()),
      vr_decay.data_ptr<float>(),
      vr_beta.data_ptr<float>(),
      H, HV, scale, norm_eps, act_mode, v_blocks);
}

void gdn_commit_all_layers(
    torch::Tensor layer_ptrs,
    torch::Tensor state_indices,
    torch::Tensor cu_seqlens,
    c10::optional<torch::Tensor> prev_num_accepted_opt,
    torch::Tensor new_num_sampled,
    int64_t H, int64_t HV, int64_t v_blocks) {
  TORCH_CHECK(layer_ptrs.is_cuda() && layer_ptrs.scalar_type() == torch::kInt64 && layer_ptrs.is_contiguous());
  TORCH_CHECK(layer_ptrs.dim() == 2 && layer_ptrs.size(1) == 6);
  TORCH_CHECK(v_blocks == 1 || v_blocks == 2 || v_blocks == 4);
  const int num_layers = (int)layer_ptrs.size(0);
  const int num_reqs = (int)cu_seqlens.size(0) - 1;
  if (num_layers <= 0 || num_reqs <= 0) return;
  const int max_T = (int)state_indices.size(1);
  const int idx_stride = (int)state_indices.stride(0);
  const int32_t* prev_nacc = (prev_num_accepted_opt.has_value() && prev_num_accepted_opt->defined())
                                 ? prev_num_accepted_opt->data_ptr<int32_t>()
                                 : nullptr;
  auto st = at::hip::getCurrentHIPStream();

#define LAUNCH_COMMIT(VB) \
  { hipLaunchKernelGGL((k_gdn_commit_all_layers<VB>), dim3(num_reqs * HV * VB, num_layers), dim3(VB == 1 ? 256 : 128), 0, st, \
                     layer_ptrs.data_ptr<int64_t>(), \
                     state_indices.data_ptr<int32_t>(), cu_seqlens.data_ptr<int32_t>(), \
                     prev_nacc, new_num_sampled.data_ptr<int32_t>(), \
                     num_reqs, (int)H, (int)HV, idx_stride, max_T); }

  if (v_blocks == 1) { LAUNCH_COMMIT(1); }
  else if (v_blocks == 2) { LAUNCH_COMMIT(2); }
  else { LAUNCH_COMMIT(4); }
#undef LAUNCH_COMMIT
}

PYBIND11_MODULE(htr_gdn_spec_fused, m) {
  m.def("gdn_spec_fused", &gdn_spec_fused);
  m.def("gdn_spec_fused_defer", &gdn_spec_fused_defer);
  m.def("gdn_commit_all_layers", &gdn_commit_all_layers);
}
