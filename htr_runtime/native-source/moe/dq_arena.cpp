#include <torch/extension.h>
void dq_arena(torch::Tensor qw, torch::Tensor qoff, torch::Tensor scales, torch::Tensor su, torch::Tensor sv, torch::Tensor cbtab,
              int64_t K, int64_t role, int64_t pad, int64_t m, int64_t n, int64_t s_off, int64_t s_es, int64_t su_off, int64_t su_es,
              int64_t sv_off, int64_t sv_es, torch::Tensor out, int64_t w_es, int64_t row_off);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("dq_arena", &dq_arena); }
