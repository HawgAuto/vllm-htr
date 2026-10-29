#include <torch/extension.h>
std::vector<torch::Tensor> moe_plan(torch::Tensor ids, int64_t M);
void moe_gemv(int64_t phase, int64_t kf, int64_t M, torch::Tensor qw, torch::Tensor qoff, torch::Tensor kexp, torch::Tensor scales,
              torch::Tensor su, torch::Tensor sv, torch::Tensor grp_e, torch::Tensor grp_rows, torch::Tensor grp_n,
              torch::Tensor xin, torch::Tensor part, torch::Tensor counters, torch::Tensor out, int64_t bps, double cb,
              int64_t spad3, int64_t spad4, int64_t spad5, int64_t fence_ra);
torch::Tensor moe_combine(torch::Tensor dn, torch::Tensor slot, torch::Tensor wts);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_plan", &moe_plan); m.def("moe_gemv", &moe_gemv); m.def("moe_combine", &moe_combine);
}
