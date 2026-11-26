#include <torch/extension.h>

void launch_mhc_fused_post_pre_decode(
    torch::Tensor x, torch::Tensor res_in, torch::Tensor post_mix, torch::Tensor comb_mix,
    torch::Tensor fn, torch::Tensor hc_scale, torch::Tensor hc_base,
    c10::optional<torch::Tensor> norm_w,
    torch::Tensor res_out, torch::Tensor post_out, torch::Tensor comb_out,
    torch::Tensor layer_in, torch::Tensor part_buf,
    double rms_eps, double hc_pre_eps, double hc_sinkhorn_eps,
    double hc_post_mult_value, int64_t sinkhorn_repeat, double norm_eps, bool do_post);

void launch_mhc_post_sq(
    c10::optional<torch::Tensor> x, torch::Tensor res_in,
    c10::optional<torch::Tensor> post_mix, c10::optional<torch::Tensor> comb_mix,
    c10::optional<torch::Tensor> res_out,
    c10::optional<torch::Tensor> x_f32_out,
    c10::optional<torch::Tensor> ss_out, bool do_post);

void launch_mhc_sinkhorn_norm_from_dots(
    torch::Tensor dots, torch::Tensor ss, torch::Tensor res_cur,
    torch::Tensor hc_scale, torch::Tensor hc_base,
    c10::optional<torch::Tensor> norm_w,
    torch::Tensor post_out, torch::Tensor comb_out, torch::Tensor layer_in,
    double rms_eps, double hc_pre_eps, double hc_sinkhorn_eps,
    double hc_post_mult_value, int64_t sinkhorn_repeat, double norm_eps);

void launch_hc_head_contract(
    torch::Tensor res, torch::Tensor fn, torch::Tensor scale, torch::Tensor base,
    torch::Tensor out, double rms_eps, double hc_eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mhc_fused_post_pre_decode", &launch_mhc_fused_post_pre_decode);
  m.def("mhc_post_sq", &launch_mhc_post_sq);
  m.def("mhc_sinkhorn_norm_from_dots", &launch_mhc_sinkhorn_norm_from_dots);
  m.def("hc_head_contract", &launch_hc_head_contract);
}

