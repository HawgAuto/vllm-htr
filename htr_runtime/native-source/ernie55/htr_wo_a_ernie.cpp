#include <torch/extension.h>

torch::Tensor repack_wo_a_sp16(torch::Tensor qw);

void launch_wo_a_k5_batched(
    torch::Tensor qw, torch::Tensor scales, torch::Tensor su, torch::Tensor sv,
    torch::Tensor x, c10::optional<torch::Tensor> gate,
    torch::Tensor part, torch::Tensor counters, torch::Tensor out,
    int64_t ns, double cb, int64_t spad, bool sp16 = false);

void launch_wo_a_k5_prefill_wmma(
    torch::Tensor qw, torch::Tensor scales, torch::Tensor su, torch::Tensor sv,
    torch::Tensor x, c10::optional<torch::Tensor> gate,
    torch::Tensor xt_buf, torch::Tensor tok_buf, torch::Tensor out,
    double cb, int64_t spad, bool sp16 = false);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("repack_wo_a_sp16", &repack_wo_a_sp16);
  m.def("wo_a_k5_batched", &launch_wo_a_k5_batched,
        py::arg("qw"), py::arg("scales"), py::arg("su"), py::arg("sv"),
        py::arg("x"), py::arg("gate"), py::arg("part"), py::arg("counters"), py::arg("out"),
        py::arg("ns"), py::arg("cb"), py::arg("spad"), py::arg("sp16") = false);
  m.def("wo_a_k5_prefill_wmma", &launch_wo_a_k5_prefill_wmma,
        py::arg("qw"), py::arg("scales"), py::arg("su"), py::arg("sv"),
        py::arg("x"), py::arg("gate"), py::arg("xt_buf"), py::arg("tok_buf"), py::arg("out"),
        py::arg("cb"), py::arg("spad"), py::arg("sp16") = false);
}
