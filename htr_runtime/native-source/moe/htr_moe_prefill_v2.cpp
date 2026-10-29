#include <torch/extension.h>
torch::Tensor hpf2_forward(torch::Tensor x, torch::Tensor sorted_ids, torch::Tensor expert_ids, torch::Tensor ntpp, torch::Tensor w,
                           torch::Tensor qw, torch::Tensor qoff, torch::Tensor scales, torch::Tensor su, torch::Tensor sv, double cb,
                           int64_t K, int64_t pad, torch::Tensor A1, torch::Tensor Y1, torch::Tensor A2, torch::Tensor Y2,
                           torch::Tensor S, torch::Tensor inv, torch::Tensor nsat, int64_t stages);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("hpf2_forward", &hpf2_forward); }
