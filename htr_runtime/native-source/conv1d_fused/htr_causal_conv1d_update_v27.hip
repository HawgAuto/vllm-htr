// htr_causal_conv1d_update_v27.hip — Wave 7 gfx1151 Register-Unrolled Zero-Scratch Causal Conv1D (36L/step)
// Upgrades over v0.2.6 (htr_causal_conv1d_update.hip):
//   1. Eliminates dynamic-indexed uint16_t xv_u[8] local array -> 0 B scratch spills guaranteed!
//      Dedicated branchless register unrolled paths for T==1 (MTP0) and T==4 (MTP3), plus safe loop for other T<=8.
//   2. Single 64-bit uint2 load for contiguous weight [dim, 4] (4x fewer weight VMEM ops x 36 layers!).
//   3. Fast __expf() for SiLU activation.
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float bf2f(uint16_t h) {
  return __builtin_bit_cast(float, ((uint32_t)h) << 16);
}

__device__ __forceinline__ uint16_t f2bf_rne(float x) {
  uint32_t u = __builtin_bit_cast(uint32_t, x);
  if (x != x) return 0x7fffu;
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

template <bool SILU>
__device__ __forceinline__ uint16_t conv_step_out(float c0, float c1, float c2, float xt,
                                                  float w0, float w1, float w2, float w3, float b_val) {
  float acc = fmaf(c0, w0, b_val);
  acc = fmaf(c1, w1, acc);
  acc = fmaf(c2, w2, acc);
  acc = fmaf(xt, w3, acc);
  if constexpr (SILU) {
    acc = acc / (1.0f + __expf(-acc));
  }
  return f2bf_rne(acc);
}

template <bool IS_VARLEN, bool IS_SPEC, bool HAS_BIAS, bool SILU>
__global__ __launch_bounds__(256)
void k_causal_conv1d_update_w4(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ w,
    const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ conv_state,
    const int32_t* __restrict__ state_indices,
    const int32_t* __restrict__ num_accepted,
    const int32_t* __restrict__ query_start_loc,
    uint16_t* __restrict__ out,
    int64_t sx_tok, int64_t sx_dim,
    int64_t so_tok, int64_t so_dim,
    int64_t sw_dim, int64_t sw_w,
    int64_t ss_seq, int64_t ss_dim, int64_t ss_tok,
    int64_t s_idx,
    int batch, int dim, int max_qlen, int num_cache_lines, int null_block_id) {
  const int d = blockIdx.x * 256 + threadIdx.x;
  const int b = blockIdx.y;
  if (d >= dim || b >= batch) return;

  const int slot = state_indices[(int64_t)b * s_idx];
  if (slot == null_block_id || slot < 0 || slot >= num_cache_lines) return;

  int64_t bos, T;
  if constexpr (IS_VARLEN) {
    bos = query_start_loc[b];
    T = query_start_loc[b + 1] - bos;
  } else {
    bos = b;
    T = 1;
  }
  if (T <= 0 || T > 8) return;

  const int tok_off = IS_SPEC ? (num_accepted[b] - 1) : 0;
  uint16_t* st_base = conv_state + (int64_t)slot * ss_seq + (int64_t)d * ss_dim;
  const uint16_t h0_u = st_base[(int64_t)(tok_off + 0) * ss_tok];
  const uint16_t h1_u = st_base[(int64_t)(tok_off + 1) * ss_tok];
  const uint16_t h2_u = st_base[(int64_t)(tok_off + 2) * ss_tok];

  // Load 4 weights via 1x 64-bit load when contiguous along width (sw_w == 1)
  const uint16_t* w_base = w + (int64_t)d * sw_dim;
  float w0, w1, w2, w3;
  if (sw_w == 1 && (((uintptr_t)w_base) & 7u) == 0u) {
    uint2 ww = *reinterpret_cast<const uint2*>(w_base);
    w0 = bf2f((uint16_t)(ww.x & 0xFFFFu));
    w1 = bf2f((uint16_t)(ww.x >> 16));
    w2 = bf2f((uint16_t)(ww.y & 0xFFFFu));
    w3 = bf2f((uint16_t)(ww.y >> 16));
  } else {
    w0 = bf2f(w_base[0 * sw_w]);
    w1 = bf2f(w_base[1 * sw_w]);
    w2 = bf2f(w_base[2 * sw_w]);
    w3 = bf2f(w_base[3 * sw_w]);
  }
  const float b_val = HAS_BIAS ? bf2f(bias[d]) : 0.0f;

  const uint16_t* x_base = x + bos * sx_tok + (int64_t)d * sx_dim;
  uint16_t* o_base = out + bos * so_tok + (int64_t)d * so_dim;
  float c0 = bf2f(h0_u), c1 = bf2f(h1_u), c2 = bf2f(h2_u);

  if (T == 1) {
    const uint16_t x0_u = x_base[0];
    st_base[0 * ss_tok] = h1_u;
    st_base[1 * ss_tok] = h2_u;
    st_base[2 * ss_tok] = x0_u;
    o_base[0] = conv_step_out<SILU>(c0, c1, c2, bf2f(x0_u), w0, w1, w2, w3, b_val);
    return;
  }

  if (T == 4) {
    const uint16_t x0_u = x_base[0 * sx_tok];
    const uint16_t x1_u = x_base[1 * sx_tok];
    const uint16_t x2_u = x_base[2 * sx_tok];
    const uint16_t x3_u = x_base[3 * sx_tok];
    st_base[0 * ss_tok] = h1_u;
    st_base[1 * ss_tok] = h2_u;
    st_base[2 * ss_tok] = x0_u;
    st_base[3 * ss_tok] = x1_u;
    st_base[4 * ss_tok] = x2_u;
    st_base[5 * ss_tok] = x3_u;
    const float f0 = bf2f(x0_u), f1 = bf2f(x1_u), f2 = bf2f(x2_u), f3 = bf2f(x3_u);
    o_base[0 * so_tok] = conv_step_out<SILU>(c0, c1, c2, f0, w0, w1, w2, w3, b_val);
    o_base[1 * so_tok] = conv_step_out<SILU>(c1, c2, f0, f1, w0, w1, w2, w3, b_val);
    o_base[2 * so_tok] = conv_step_out<SILU>(c2, f0, f1, f2, w0, w1, w2, w3, b_val);
    o_base[3 * so_tok] = conv_step_out<SILU>(f0, f1, f2, f3, w0, w1, w2, w3, b_val);
    return;
  }

  // Generic T in {2,3,5,6,7,8}: register-only via explicit scalar vars (0 scratch spill!)
  const uint16_t x0_u = x_base[0 * sx_tok];
  const uint16_t x1_u = (T > 1) ? x_base[1 * sx_tok] : 0u;
  const uint16_t x2_u = (T > 2) ? x_base[2 * sx_tok] : 0u;
  const uint16_t x3_u = (T > 3) ? x_base[3 * sx_tok] : 0u;
  const uint16_t x4_u = (T > 4) ? x_base[4 * sx_tok] : 0u;
  const uint16_t x5_u = (T > 5) ? x_base[5 * sx_tok] : 0u;
  const uint16_t x6_u = (T > 6) ? x_base[6 * sx_tok] : 0u;
  const uint16_t x7_u = (T > 7) ? x_base[7 * sx_tok] : 0u;

  st_base[0 * ss_tok] = h1_u;
  st_base[1 * ss_tok] = h2_u;
  st_base[2 * ss_tok] = x0_u;
  if (T > 1) st_base[3 * ss_tok] = x1_u;
  if (T > 2) st_base[4 * ss_tok] = x2_u;
  if (T > 3) st_base[5 * ss_tok] = x3_u;
  if (T > 4) st_base[6 * ss_tok] = x4_u;
  if (T > 5) st_base[7 * ss_tok] = x5_u;
  if (T > 6) st_base[8 * ss_tok] = x6_u;
  if (T > 7) st_base[9 * ss_tok] = x7_u;

  auto emit = [&](int t, uint16_t xu) {
    float fx = bf2f(xu);
    o_base[(int64_t)t * so_tok] = conv_step_out<SILU>(c0, c1, c2, fx, w0, w1, w2, w3, b_val);
    c0 = c1; c1 = c2; c2 = fx;
  };
  emit(0, x0_u);
  if (T > 1) emit(1, x1_u);
  if (T > 2) emit(2, x2_u);
  if (T > 3) emit(3, x3_u);
  if (T > 4) emit(4, x4_u);
  if (T > 5) emit(5, x5_u);
  if (T > 6) emit(6, x6_u);
  if (T > 7) emit(7, x7_u);
}

void causal_conv1d_update_w4(
    torch::Tensor x,
    torch::Tensor conv_state,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    torch::Tensor state_indices,
    c10::optional<torch::Tensor> num_accepted_opt,
    c10::optional<torch::Tensor> query_start_loc_opt,
    torch::Tensor out,
    bool silu_activation,
    int64_t max_qlen,
    int64_t null_block_id) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && conv_state.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(weight.scalar_type() == torch::kBFloat16 && weight.dim() == 2 && weight.size(1) == 4);
  const bool is_varlen = query_start_loc_opt.has_value() && query_start_loc_opt->defined();
  const bool is_spec = num_accepted_opt.has_value() && num_accepted_opt->defined();
  const bool has_bias = bias_opt.has_value() && bias_opt->defined();
  const int batch = (int)state_indices.size(0);
  const int dim = (int)x.size(1);
  if (batch <= 0 || dim <= 0) return;

  auto st = at::hip::getCurrentHIPStream();
  dim3 grid((dim + 255) / 256, batch, 1);
  auto xp = reinterpret_cast<const uint16_t*>(x.data_ptr<at::BFloat16>());
  auto wp = reinterpret_cast<const uint16_t*>(weight.data_ptr<at::BFloat16>());
  auto bp = has_bias ? reinterpret_cast<const uint16_t*>(bias_opt->data_ptr<at::BFloat16>()) : nullptr;
  auto sp = reinterpret_cast<uint16_t*>(conv_state.data_ptr<at::BFloat16>());
  auto ip = state_indices.data_ptr<int32_t>();
  auto nap = is_spec ? num_accepted_opt->data_ptr<int32_t>() : nullptr;
  auto qsl = is_varlen ? query_start_loc_opt->data_ptr<int32_t>() : nullptr;
  auto op = reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>());

#define LAUNCH_C(VL, SP, HB, SI) \
  hipLaunchKernelGGL((k_causal_conv1d_update_w4<VL, SP, HB, SI>), grid, dim3(256), 0, st, \
    xp, wp, bp, sp, ip, nap, qsl, op, \
    (int64_t)x.stride(0), (int64_t)x.stride(1), (int64_t)out.stride(0), (int64_t)out.stride(1), \
    (int64_t)weight.stride(0), (int64_t)weight.stride(1), \
    (int64_t)conv_state.stride(0), (int64_t)conv_state.stride(1), (int64_t)conv_state.stride(2), \
    (int64_t)state_indices.stride(0), batch, dim, (int)max_qlen, (int)conv_state.size(0), (int)null_block_id)

  if (is_varlen) {
    if (is_spec) {
      if (has_bias) { if (silu_activation) LAUNCH_C(true, true, true, true); else LAUNCH_C(true, true, true, false); }
      else          { if (silu_activation) LAUNCH_C(true, true, false, true); else LAUNCH_C(true, true, false, false); }
    } else {
      if (has_bias) { if (silu_activation) LAUNCH_C(true, false, true, true); else LAUNCH_C(true, false, true, false); }
      else          { if (silu_activation) LAUNCH_C(true, false, false, true); else LAUNCH_C(true, false, false, false); }
    }
  } else {
    if (has_bias) { if (silu_activation) LAUNCH_C(false, false, true, true); else LAUNCH_C(false, false, true, false); }
    else          { if (silu_activation) LAUNCH_C(false, false, false, true); else LAUNCH_C(false, false, false, false); }
  }
#undef LAUNCH_C
}

PYBIND11_MODULE(htr_causal_conv1d_update, m) {
  m.def("causal_conv1d_update_w4", &causal_conv1d_update_w4);
}
