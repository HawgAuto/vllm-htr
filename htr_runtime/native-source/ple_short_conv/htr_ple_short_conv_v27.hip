// htr_ple_short_conv_v27.hip — Wave 7 gfx1151 Vectorized PLE Short-Conv + Register-Resident b128 Gate-Norm
// Upgrades over v0.2.6 (htr_ple_short_conv.hip):
//   1. k_ple_gate_norm_fused_2560 (320 threads x 8 BF16 = 2560 exact):
//      - 100% 128-bit uint4 (global_load_b128 / global_store_b128) -> 8x fewer VMEM ops!
//      - Keeps K, Q, and GatedValue in registers across passes -> ZERO re-reads of K, Q, or GV!
//      - Native gfx1151 v_dot2_f32_bf16 for sum(k^2), sum(q^2), dot(kb,qb), sum(gv^2)!
//      - Native 6-op zero-LDS RDNA 3.5 DPP wave_sum (row_xmask:1,2,4,8 + v_permlanex16_b32(0,0))!
//   2. Zero-alloc dtype fast-path in host wrappers (avoids redundant .to().contiguous() dispatch).
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef short bf16x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float bf2f(uint16_t h) {
  return __builtin_bit_cast(float, ((uint32_t)h) << 16);
}

__device__ __forceinline__ uint16_t f2bf_rne(float x) {
  uint32_t u = __builtin_bit_cast(uint32_t, x);
  if (x != x) return 0x7fffu;
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

__device__ __forceinline__ float dot2_bf16(uint32_t a, uint32_t b, float acc) {
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2, a), __builtin_bit_cast(bf16x2, b), acc, false);
}

__device__ __forceinline__ float dot8_bf16(uint4 a, uint4 b) {
  float c0 = dot2_bf16(a.x, b.x, 0.0f);
  float c1 = dot2_bf16(a.y, b.y, 0.0f);
  c0 = dot2_bf16(a.z, b.z, c0);
  c1 = dot2_bf16(a.w, b.w, c1);
  return c0 + c1;
}

__device__ __forceinline__ float w32_sum_dpp(float v) {
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

template <typename StateT>
__device__ __forceinline__ float ld_state(const StateT* p);
template <>
__device__ __forceinline__ float ld_state<uint16_t>(const uint16_t* p) { return bf2f(*p); }
template <>
__device__ __forceinline__ float ld_state<float>(const float* p) { return *p; }

template <typename StateT>
__device__ __forceinline__ void st_state(StateT* p, float v);
template <>
__device__ __forceinline__ void st_state<uint16_t>(uint16_t* p, float v) { *p = f2bf_rne(v); }
template <>
__device__ __forceinline__ void st_state<float>(float* p, float v) { *p = v; }

// Mode A: Decode (1 token per request, dilation=3, K=4, S=9)
template <typename StateT, bool HAS_INIT_PTR>
__global__ __launch_bounds__(256)
void k_ple_short_conv_decode_k4d3(
    const uint16_t* __restrict__ x,
    StateT* __restrict__ conv_state,
    const uint16_t* __restrict__ w,
    const int32_t* __restrict__ state_indices,
    const bool* __restrict__ has_initial_states,
    uint16_t* __restrict__ out,
    int64_t sx_b, int64_t so_b, int64_t sw_d, int64_t sw_k,
    int64_t ss_s, int64_t ss_d, int64_t ss_t,
    int B, int H, int null_block_id) {
  const int d = blockIdx.x * 256 + threadIdx.x;
  const int b = blockIdx.y;
  if (d >= H || b >= B) return;

  const int slot = state_indices[b];
  if (slot == null_block_id) {
    out[(int64_t)b * so_b + d] = 0;
    return;
  }

  const bool use_init = HAS_INIT_PTR ? has_initial_states[b] : true;
  StateT* st = conv_state + (int64_t)slot * ss_s + (int64_t)d * ss_d;

  float h0 = 0.0f, h1 = 0.0f, h2 = 0.0f, h3 = 0.0f, h4 = 0.0f;
  float h5 = 0.0f, h6 = 0.0f, h7 = 0.0f, h8 = 0.0f;
  if (use_init) {
    h0 = bf2f(f2bf_rne(ld_state(st + 0 * ss_t)));
    h1 = bf2f(f2bf_rne(ld_state(st + 1 * ss_t)));
    h2 = bf2f(f2bf_rne(ld_state(st + 2 * ss_t)));
    h3 = bf2f(f2bf_rne(ld_state(st + 3 * ss_t)));
    h4 = bf2f(f2bf_rne(ld_state(st + 4 * ss_t)));
    h5 = bf2f(f2bf_rne(ld_state(st + 5 * ss_t)));
    h6 = bf2f(f2bf_rne(ld_state(st + 6 * ss_t)));
    h7 = bf2f(f2bf_rne(ld_state(st + 7 * ss_t)));
    h8 = bf2f(f2bf_rne(ld_state(st + 8 * ss_t)));
  }
  const float x0 = bf2f(x[(int64_t)b * sx_b + d]);

  const uint16_t* wp = w + (int64_t)d * sw_d;
  const float w0 = bf2f(wp[0 * sw_k]);
  const float w1 = bf2f(wp[1 * sw_k]);
  const float w2 = bf2f(wp[2 * sw_k]);
  const float w3 = bf2f(wp[3 * sw_k]);

  float acc = fmaf(h0, w0, 0.0f);
  acc = fmaf(h3, w1, acc);
  acc = fmaf(h6, w2, acc);
  acc = fmaf(x0, w3, acc);
  float cb = bf2f(f2bf_rne(acc));
  float y = cb / (1.0f + __expf(-cb));
  out[(int64_t)b * so_b + d] = f2bf_rne(y);

  st_state(st + 0 * ss_t, h1);
  st_state(st + 1 * ss_t, h2);
  st_state(st + 2 * ss_t, h3);
  st_state(st + 3 * ss_t, h4);
  st_state(st + 4 * ss_t, h5);
  st_state(st + 5 * ss_t, h6);
  st_state(st + 6 * ss_t, h7);
  st_state(st + 7 * ss_t, h8);
  st_state(st + 8 * ss_t, x0);
}

// Mode B: Spec batched (variable T <= max_len tokens/req, rollback, dilation=3, K=4, S=9)
template <typename StateT>
__global__ __launch_bounds__(256)
void k_ple_short_conv_spec_k4d3(
    const uint16_t* __restrict__ x,
    StateT* __restrict__ conv_state,
    const uint16_t* __restrict__ w,
    const int32_t* __restrict__ state_indices,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ num_accepted,
    uint16_t* __restrict__ out,
    int64_t sx_t, int64_t so_t, int64_t sw_d, int64_t sw_k,
    int64_t ss_s, int64_t ss_d, int64_t ss_t,
    int num_reqs, int N_tok, int H, int max_len, int null_block_id) {
  const int d = blockIdx.x * 256 + threadIdx.x;
  const int b = blockIdx.y;
  if (d >= H) return;

  if (b == num_reqs) {
    const int pad_start = query_start_loc[num_reqs];
    for (int tok = pad_start; tok < N_tok; ++tok) {
      out[(int64_t)tok * so_t + d] = 0;
    }
    return;
  }

  const int bos = query_start_loc[b];
  const int eos = query_start_loc[b + 1];
  const int T = eos - bos;
  if (T <= 0) return;

  const int slot = state_indices[b];
  const bool valid = (slot != null_block_id && slot >= 0);
  const int rb = valid ? (num_accepted[b] - 1) : 0;

  float h0 = 0.0f, h1 = 0.0f, h2 = 0.0f, h3 = 0.0f, h4 = 0.0f;
  float h5 = 0.0f, h6 = 0.0f, h7 = 0.0f, h8 = 0.0f;
  StateT* st = nullptr;
  if (valid) {
    st = conv_state + (int64_t)slot * ss_s + (int64_t)d * ss_d;
    h0 = bf2f(f2bf_rne(ld_state(st + (int64_t)(rb + 0) * ss_t)));
    h1 = bf2f(f2bf_rne(ld_state(st + (int64_t)(rb + 1) * ss_t)));
    h2 = bf2f(f2bf_rne(ld_state(st + (int64_t)(rb + 2) * ss_t)));
    h3 = bf2f(f2bf_rne(ld_state(st + (int64_t)(rb + 3) * ss_t)));
    h4 = bf2f(f2bf_rne(ld_state(st + (int64_t)(rb + 4) * ss_t)));
    h5 = bf2f(f2bf_rne(ld_state(st + (int64_t)(rb + 5) * ss_t)));
    h6 = bf2f(f2bf_rne(ld_state(st + (int64_t)(rb + 6) * ss_t)));
    h7 = bf2f(f2bf_rne(ld_state(st + (int64_t)(rb + 7) * ss_t)));
    h8 = bf2f(f2bf_rne(ld_state(st + (int64_t)(rb + 8) * ss_t)));
    st_state(st + 0 * ss_t, h1);
    st_state(st + 1 * ss_t, h2);
    st_state(st + 2 * ss_t, h3);
    st_state(st + 3 * ss_t, h4);
    st_state(st + 4 * ss_t, h5);
    st_state(st + 5 * ss_t, h6);
    st_state(st + 6 * ss_t, h7);
    st_state(st + 7 * ss_t, h8);
  }

  const uint16_t* wp = w + (int64_t)d * sw_d;
  const float w0 = bf2f(wp[0 * sw_k]);
  const float w1 = bf2f(wp[1 * sw_k]);
  const float w2 = bf2f(wp[2 * sw_k]);
  const float w3 = bf2f(wp[3 * sw_k]);

  for (int t = 0; t < T; ++t) {
    const int tok = bos + t;
    const float xt = (tok < N_tok) ? bf2f(x[(int64_t)tok * sx_t + d]) : 0.0f;
    float acc = fmaf(h0, w0, 0.0f);
    acc = fmaf(h3, w1, acc);
    acc = fmaf(h6, w2, acc);
    acc = fmaf(xt, w3, acc);
    float cb = bf2f(f2bf_rne(acc));
    float y = cb / (1.0f + __expf(-cb));
    if (tok < N_tok) out[(int64_t)tok * so_t + d] = f2bf_rne(y);
    if (valid) st_state(st + (int64_t)(8 + t) * ss_t, xt);
    h0 = h1; h1 = h2; h2 = h3; h3 = h4; h4 = h5; h5 = h6; h6 = h7; h7 = h8; h8 = xt;
  }
}

// ============================================================================
// Wave 7: Register-Resident 320-thread b128 PLE Gate + 3x Grouped Gemma RMSNorm
// D == 2560 exact: 320 threads x 8 BF16 (1x uint4) = 2560!
// ZERO re-reads of key_in, query_in, or gated_val_out!
// ============================================================================
__device__ __forceinline__ uint32_t norm_pack2(uint32_t x2, uint32_t w2, float rrms) {
  float a0 = bf2f((uint16_t)(x2 & 0xFFFFu)) * rrms * (1.0f + bf2f((uint16_t)(w2 & 0xFFFFu)));
  float a1 = bf2f((uint16_t)(x2 >> 16))     * rrms * (1.0f + bf2f((uint16_t)(w2 >> 16)));
  return (uint32_t)f2bf_rne(a0) | ((uint32_t)f2bf_rne(a1) << 16);
}

__device__ __forceinline__ uint32_t scale_pack2(uint32_t v2, float g) {
  float a0 = g * bf2f((uint16_t)(v2 & 0xFFFFu));
  float a1 = g * bf2f((uint16_t)(v2 >> 16));
  return (uint32_t)f2bf_rne(a0) | ((uint32_t)f2bf_rne(a1) << 16);
}

__global__ __launch_bounds__(320)
void k_ple_gate_norm_fused_2560(
    const uint16_t* __restrict__ key_in,
    const uint16_t* __restrict__ query_in,
    const uint16_t* __restrict__ value_in,
    const uint16_t* __restrict__ wk,
    const uint16_t* __restrict__ wq,
    const uint16_t* __restrict__ wc,
    uint16_t* __restrict__ gated_val_out,
    uint16_t* __restrict__ norm_conv_out,
    int T, int HC, float eps) {
  __shared__ float s_k[10];
  __shared__ float s_q[10];
  __shared__ float s_d[10];
  const int tok = blockIdx.x;
  const int hc = blockIdx.y;
  if (tok >= T || hc >= HC) return;

  const int tid = threadIdx.x;  // 0..319
  const int lane = tid & 31;
  const int wv = tid >> 5;      // 0..9

  const size_t hd_off = (size_t)hc * 2560;
  const size_t row_off = (size_t)tok * HC * 2560 + hd_off;
  const int elem = tid * 8;

  // Load K and Q ONCE (1x b128 each) and keep register-resident across Pass 1 & Pass 2!
  const uint4 ku = *reinterpret_cast<const uint4*>(key_in + row_off + elem);
  const uint4 qu = *reinterpret_cast<const uint4*>(query_in + row_off + elem);
  const uint4 wku = *reinterpret_cast<const uint4*>(wk + hd_off + elem);
  const uint4 wqu = *reinterpret_cast<const uint4*>(wq + hd_off + elem);

  float sk = w32_sum_dpp(dot8_bf16(ku, ku));
  float sq = w32_sum_dpp(dot8_bf16(qu, qu));
  if (lane == 0) { s_k[wv] = sk; s_q[wv] = sq; }
  __syncthreads();

  float tot_k = ((s_k[0] + s_k[1]) + (s_k[2] + s_k[3])) + ((s_k[4] + s_k[5]) + (s_k[6] + s_k[7])) + (s_k[8] + s_k[9]);
  float tot_q = ((s_q[0] + s_q[1]) + (s_q[2] + s_q[3])) + ((s_q[4] + s_q[5]) + (s_q[6] + s_q[7])) + (s_q[8] + s_q[9]);
  const float rrms_k = rsqrtf(tot_k * (1.0f / 2560.0f) + eps);
  const float rrms_q = rsqrtf(tot_q * (1.0f / 2560.0f) + eps);

  // Pass 2: normalize K & Q from registers + dot via v_dot2_f32_bf16
  uint4 kbn, qbn;
  kbn.x = norm_pack2(ku.x, wku.x, rrms_k);
  kbn.y = norm_pack2(ku.y, wku.y, rrms_k);
  kbn.z = norm_pack2(ku.z, wku.z, rrms_k);
  kbn.w = norm_pack2(ku.w, wku.w, rrms_k);
  qbn.x = norm_pack2(qu.x, wqu.x, rrms_q);
  qbn.y = norm_pack2(qu.y, wqu.y, rrms_q);
  qbn.z = norm_pack2(qu.z, wqu.z, rrms_q);
  qbn.w = norm_pack2(qu.w, wqu.w, rrms_q);

  float dot = w32_sum_dpp(dot8_bf16(kbn, qbn));
  if (lane == 0) s_d[wv] = dot;
  __syncthreads();

  const float tot_d = ((s_d[0] + s_d[1]) + (s_d[2] + s_d[3])) + ((s_d[4] + s_d[5]) + (s_d[6] + s_d[7])) + (s_d[8] + s_d[9]);
  const float g_raw = tot_d * rsqrtf(2560.0f);
  const float g_sgn = (g_raw > 0.0f) ? 1.0f : ((g_raw < 0.0f) ? -1.0f : 0.0f);
  const float g_act = 1.0f / (1.0f + __expf(-(g_sgn * sqrtf(fmaxf(fabsf(g_raw), 1e-6f)))));
  const float gate_bf = bf2f(f2bf_rne(g_act));

  // Pass 3: load value_in & wc (1x b128 each), compute & store gv_u4, keep gv_u4 in registers!
  const uint4 vu = *reinterpret_cast<const uint4*>(value_in + (size_t)tok * 2560 + elem);
  const uint4 wcu = *reinterpret_cast<const uint4*>(wc + hd_off + elem);
  uint4 gvu;
  gvu.x = scale_pack2(vu.x, gate_bf);
  gvu.y = scale_pack2(vu.y, gate_bf);
  gvu.z = scale_pack2(vu.z, gate_bf);
  gvu.w = scale_pack2(vu.w, gate_bf);
  *reinterpret_cast<uint4*>(gated_val_out + row_off + elem) = gvu;

  float sc = w32_sum_dpp(dot8_bf16(gvu, gvu));
  if (lane == 0) s_k[wv] = sc;
  __syncthreads();

  const float tot_c = ((s_k[0] + s_k[1]) + (s_k[2] + s_k[3])) + ((s_k[4] + s_k[5]) + (s_k[6] + s_k[7])) + (s_k[8] + s_k[9]);
  const float rrms_c = rsqrtf(tot_c * (1.0f / 2560.0f) + eps);

  // Pass 4: normalize directly from register gvu -> 1x b128 store (ZERO re-read of gated_val_out!)
  uint4 ncu;
  ncu.x = norm_pack2(gvu.x, wcu.x, rrms_c);
  ncu.y = norm_pack2(gvu.y, wcu.y, rrms_c);
  ncu.z = norm_pack2(gvu.z, wcu.z, rrms_c);
  ncu.w = norm_pack2(gvu.w, wcu.w, rrms_c);
  *reinterpret_cast<uint4*>(norm_conv_out + row_off + elem) = ncu;
}

// Generic fallback for any D != 2560 (upgraded to zero-LDS DPP wave_sum)
__global__ __launch_bounds__(256)
void k_ple_gate_norm_fused(
    const uint16_t* __restrict__ key_in,
    const uint16_t* __restrict__ query_in,
    const uint16_t* __restrict__ value_in,
    const uint16_t* __restrict__ wk,
    const uint16_t* __restrict__ wq,
    const uint16_t* __restrict__ wc,
    uint16_t* __restrict__ gated_val_out,
    uint16_t* __restrict__ norm_conv_out,
    int T, int HC, int D, float eps) {
  __shared__ float s_k[8];
  __shared__ float s_q[8];
  __shared__ float s_d[8];
  const int tok = blockIdx.x;
  const int hc = blockIdx.y;
  if (tok >= T || hc >= HC) return;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wv = tid >> 5;

  const size_t hd_off = (size_t)hc * D;
  const size_t row_off = (size_t)tok * HC * D + hd_off;
  const uint16_t* kp = key_in + row_off;
  const uint16_t* qp = query_in + row_off;
  const uint16_t* vp = value_in + (size_t)tok * D;
  const uint16_t* wkp = wk + hd_off;
  const uint16_t* wqp = wq + hd_off;
  const uint16_t* wcp = wc + hd_off;
  uint16_t* gvp = gated_val_out + row_off;
  uint16_t* ncp = norm_conv_out + row_off;

  float sk = 0.0f, sq = 0.0f;
  for (int i = tid; i < D; i += 256) {
    float kf = bf2f(kp[i]);
    float qf = bf2f(qp[i]);
    sk = fmaf(kf, kf, sk);
    sq = fmaf(qf, qf, sq);
  }
  sk = w32_sum_dpp(sk);
  sq = w32_sum_dpp(sq);
  if (lane == 0) { s_k[wv] = sk; s_q[wv] = sq; }
  __syncthreads();
  const float tot_k = ((s_k[0] + s_k[1]) + (s_k[2] + s_k[3])) + ((s_k[4] + s_k[5]) + (s_k[6] + s_k[7]));
  const float tot_q = ((s_q[0] + s_q[1]) + (s_q[2] + s_q[3])) + ((s_q[4] + s_q[5]) + (s_q[6] + s_q[7]));
  const float rrms_k = rsqrtf(tot_k / (float)D + eps);
  const float rrms_q = rsqrtf(tot_q / (float)D + eps);

  float dot = 0.0f;
  for (int i = tid; i < D; i += 256) {
    float kf = bf2f(kp[i]) * rrms_k * (1.0f + bf2f(wkp[i]));
    float qf = bf2f(qp[i]) * rrms_q * (1.0f + bf2f(wqp[i]));
    dot = fmaf(bf2f(f2bf_rne(kf)), bf2f(f2bf_rne(qf)), dot);
  }
  dot = w32_sum_dpp(dot);
  if (lane == 0) s_d[wv] = dot;
  __syncthreads();
  const float tot_d = ((s_d[0] + s_d[1]) + (s_d[2] + s_d[3])) + ((s_d[4] + s_d[5]) + (s_d[6] + s_d[7]));
  const float g_raw = tot_d * rsqrtf((float)D);
  const float g_sgn = (g_raw > 0.0f) ? 1.0f : ((g_raw < 0.0f) ? -1.0f : 0.0f);
  const float g_act = 1.0f / (1.0f + __expf(-(g_sgn * sqrtf(fmaxf(fabsf(g_raw), 1e-6f)))));
  const float gate_bf = bf2f(f2bf_rne(g_act));

  float sc = 0.0f;
  for (int i = tid; i < D; i += 256) {
    uint16_t gb = f2bf_rne(gate_bf * bf2f(vp[i]));
    gvp[i] = gb;
    float gf = bf2f(gb);
    sc = fmaf(gf, gf, sc);
  }
  sc = w32_sum_dpp(sc);
  if (lane == 0) s_k[wv] = sc;
  __syncthreads();
  const float tot_c = ((s_k[0] + s_k[1]) + (s_k[2] + s_k[3])) + ((s_k[4] + s_k[5]) + (s_k[6] + s_k[7]));
  const float rrms_c = rsqrtf(tot_c / (float)D + eps);

  for (int i = tid; i < D; i += 256) {
    float nf = bf2f(gvp[i]) * rrms_c * (1.0f + bf2f(wcp[i]));
    ncp[i] = f2bf_rne(nf);
  }
}

static inline torch::Tensor as_contig_dtype(const torch::Tensor& t, torch::ScalarType dt) {
  if (t.scalar_type() == dt && t.is_contiguous()) return t;
  return t.to(dt).contiguous();
}

torch::Tensor ple_short_conv_decode_k4d3(
    torch::Tensor x, torch::Tensor conv_state, torch::Tensor w,
    torch::Tensor state_indices, c10::optional<torch::Tensor> has_initial_states_opt,
    int64_t null_block_id) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.dim() == 2);
  TORCH_CHECK(w.is_cuda() && w.dtype() == torch::kBFloat16 && w.dim() == 2 && w.size(1) == 4);
  const int B = (int)x.size(0);
  const int H = (int)x.size(1);
  auto out = torch::empty_like(x);
  if (B == 0) return out;
  auto sidx = as_contig_dtype(state_indices, torch::kInt32);
  bool has_init = has_initial_states_opt.has_value() && has_initial_states_opt->defined();
  torch::Tensor init_t;
  if (has_init) init_t = as_contig_dtype(*has_initial_states_opt, torch::kBool);
  dim3 grid((H + 255) / 256, B);
  dim3 block(256);
  hipStream_t stream = at::hip::getCurrentHIPStream().stream();
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
  const uint16_t* wp = reinterpret_cast<const uint16_t*>(w.data_ptr());
  uint16_t* op = reinterpret_cast<uint16_t*>(out.data_ptr());
  const int32_t* ip = sidx.data_ptr<int32_t>();
  const bool* hip = has_init ? init_t.data_ptr<bool>() : nullptr;
  if (conv_state.dtype() == torch::kBFloat16) {
    uint16_t* sp = reinterpret_cast<uint16_t*>(conv_state.data_ptr());
    if (has_init)
      k_ple_short_conv_decode_k4d3<uint16_t, true><<<grid, block, 0, stream>>>(
          xp, sp, wp, ip, hip, op, x.stride(0), out.stride(0), w.stride(0), w.stride(1),
          conv_state.stride(0), conv_state.stride(1), conv_state.stride(2), B, H, (int)null_block_id);
    else
      k_ple_short_conv_decode_k4d3<uint16_t, false><<<grid, block, 0, stream>>>(
          xp, sp, wp, ip, nullptr, op, x.stride(0), out.stride(0), w.stride(0), w.stride(1),
          conv_state.stride(0), conv_state.stride(1), conv_state.stride(2), B, H, (int)null_block_id);
  } else {
    TORCH_CHECK(conv_state.dtype() == torch::kFloat32);
    float* sp = conv_state.data_ptr<float>();
    if (has_init)
      k_ple_short_conv_decode_k4d3<float, true><<<grid, block, 0, stream>>>(
          xp, sp, wp, ip, hip, op, x.stride(0), out.stride(0), w.stride(0), w.stride(1),
          conv_state.stride(0), conv_state.stride(1), conv_state.stride(2), B, H, (int)null_block_id);
    else
      k_ple_short_conv_decode_k4d3<float, false><<<grid, block, 0, stream>>>(
          xp, sp, wp, ip, nullptr, op, x.stride(0), out.stride(0), w.stride(0), w.stride(1),
          conv_state.stride(0), conv_state.stride(1), conv_state.stride(2), B, H, (int)null_block_id);
  }
  return out;
}

torch::Tensor ple_short_conv_spec_k4d3(
    torch::Tensor x, torch::Tensor conv_state, torch::Tensor w,
    torch::Tensor state_indices, torch::Tensor query_start_loc,
    torch::Tensor num_accepted, int64_t max_len, int64_t null_block_id) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.dim() == 2);
  TORCH_CHECK(w.is_cuda() && w.dtype() == torch::kBFloat16 && w.dim() == 2 && w.size(1) == 4);
  const int N_tok = (int)x.size(0);
  const int H = (int)x.size(1);
  const int num_reqs = (int)state_indices.numel();
  auto out = torch::empty_like(x);
  if (N_tok == 0) return out;
  auto sidx = as_contig_dtype(state_indices, torch::kInt32);
  auto qsl = as_contig_dtype(query_start_loc, torch::kInt32);
  auto nacc = as_contig_dtype(num_accepted, torch::kInt32);
  dim3 grid((H + 255) / 256, num_reqs + 1);
  dim3 block(256);
  hipStream_t stream = at::hip::getCurrentHIPStream().stream();
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
  const uint16_t* wp = reinterpret_cast<const uint16_t*>(w.data_ptr());
  uint16_t* op = reinterpret_cast<uint16_t*>(out.data_ptr());
  if (conv_state.dtype() == torch::kBFloat16) {
    uint16_t* sp = reinterpret_cast<uint16_t*>(conv_state.data_ptr());
    k_ple_short_conv_spec_k4d3<uint16_t><<<grid, block, 0, stream>>>(
        xp, sp, wp, sidx.data_ptr<int32_t>(), qsl.data_ptr<int32_t>(), nacc.data_ptr<int32_t>(),
        op, x.stride(0), out.stride(0), w.stride(0), w.stride(1),
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        num_reqs, N_tok, H, (int)max_len, (int)null_block_id);
  } else {
    TORCH_CHECK(conv_state.dtype() == torch::kFloat32);
    float* sp = conv_state.data_ptr<float>();
    k_ple_short_conv_spec_k4d3<float><<<grid, block, 0, stream>>>(
        xp, sp, wp, sidx.data_ptr<int32_t>(), qsl.data_ptr<int32_t>(), nacc.data_ptr<int32_t>(),
        op, x.stride(0), out.stride(0), w.stride(0), w.stride(1),
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        num_reqs, N_tok, H, (int)max_len, (int)null_block_id);
  }
  return out;
}

std::vector<torch::Tensor> ple_gate_norm_fused(
    torch::Tensor key_in, torch::Tensor query_in, torch::Tensor value_in,
    torch::Tensor wk, torch::Tensor wq, torch::Tensor wc,
    int64_t hc_count, double eps) {
  TORCH_CHECK(key_in.is_cuda() && query_in.is_cuda() && value_in.is_cuda());
  TORCH_CHECK(key_in.dtype() == torch::kBFloat16 && query_in.dtype() == torch::kBFloat16 && value_in.dtype() == torch::kBFloat16);
  TORCH_CHECK(key_in.is_contiguous() && query_in.is_contiguous() && value_in.is_contiguous());
  const int T = (int)key_in.size(0);
  const int HC = (int)hc_count;
  const int D = (int)value_in.size(1);
  TORCH_CHECK(key_in.size(1) == HC * D && query_in.size(1) == HC * D);
  auto gv = torch::empty_like(key_in);
  auto nc = torch::empty_like(key_in);
  if (T == 0) return {gv, nc};
  hipStream_t stream = at::hip::getCurrentHIPStream().stream();
  if (D == 2560 && wk.is_contiguous() && wq.is_contiguous() && wc.is_contiguous() &&
      (((uintptr_t)key_in.data_ptr() | (uintptr_t)query_in.data_ptr() | (uintptr_t)value_in.data_ptr() |
        (uintptr_t)wk.data_ptr() | (uintptr_t)wq.data_ptr() | (uintptr_t)wc.data_ptr() |
        (uintptr_t)gv.data_ptr() | (uintptr_t)nc.data_ptr()) & 15u) == 0u) {
    hipLaunchKernelGGL(
        k_ple_gate_norm_fused_2560, dim3(T, HC), dim3(320), 0, stream,
        reinterpret_cast<const uint16_t*>(key_in.data_ptr()),
        reinterpret_cast<const uint16_t*>(query_in.data_ptr()),
        reinterpret_cast<const uint16_t*>(value_in.data_ptr()),
        reinterpret_cast<const uint16_t*>(wk.data_ptr()),
        reinterpret_cast<const uint16_t*>(wq.data_ptr()),
        reinterpret_cast<const uint16_t*>(wc.data_ptr()),
        reinterpret_cast<uint16_t*>(gv.data_ptr()),
        reinterpret_cast<uint16_t*>(nc.data_ptr()),
        T, HC, (float)eps);
  } else {
    hipLaunchKernelGGL(
        k_ple_gate_norm_fused, dim3(T, HC), dim3(256), 0, stream,
        reinterpret_cast<const uint16_t*>(key_in.data_ptr()),
        reinterpret_cast<const uint16_t*>(query_in.data_ptr()),
        reinterpret_cast<const uint16_t*>(value_in.data_ptr()),
        reinterpret_cast<const uint16_t*>(wk.data_ptr()),
        reinterpret_cast<const uint16_t*>(wq.data_ptr()),
        reinterpret_cast<const uint16_t*>(wc.data_ptr()),
        reinterpret_cast<uint16_t*>(gv.data_ptr()),
        reinterpret_cast<uint16_t*>(nc.data_ptr()),
        T, HC, D, (float)eps);
  }
  return {gv, nc};
}

PYBIND11_MODULE(htr_ple_short_conv, m) {
  m.def("ple_gate_norm_fused", &ple_gate_norm_fused, "Wave7: register-resident b128 + v_dot2_bf16 + DPP PLE gate norm");
  m.def("ple_short_conv_decode_k4d3", &ple_short_conv_decode_k4d3, "Native Wave32 PLE short-conv decode (K=4, D=3)");
  m.def("ple_short_conv_spec_k4d3", &ple_short_conv_spec_k4d3, "Native Wave32 PLE short-conv spec (K=4, D=3)");
}
