#include <torch/extension.h>
std::vector<torch::Tensor> moe_plan(torch::Tensor ids, int64_t M);
std::vector<torch::Tensor> moe_plan2(torch::Tensor ids, int64_t M);
void moe_plan2_into(torch::Tensor ids, int64_t M,
                    torch::Tensor grp_e, torch::Tensor grp_rows, torch::Tensor grp_n,
                    torch::Tensor slot, torch::Tensor ng);
void moe_gemv(int64_t phase, int64_t kf, int64_t M, torch::Tensor qw, torch::Tensor qoff, torch::Tensor kexp,
              torch::Tensor scales, torch::Tensor su, torch::Tensor sv, torch::Tensor grp_e, torch::Tensor grp_rows,
              torch::Tensor grp_n, torch::Tensor xin, torch::Tensor part, torch::Tensor counters, torch::Tensor out,
              int64_t bps, double cb, int64_t spad2, int64_t spad3, int64_t spad4, int64_t spad5, int64_t fence_ra);
torch::Tensor moe_combine(torch::Tensor dn, torch::Tensor slot, torch::Tensor wts);
torch::Tensor moe_gemv_dn_fused(int64_t kf, int64_t M, torch::Tensor qw, torch::Tensor qoff, torch::Tensor kexp,
                                torch::Tensor scales, torch::Tensor su, torch::Tensor sv, torch::Tensor grp_e,
                                torch::Tensor grp_rows, torch::Tensor grp_n, torch::Tensor gu, torch::Tensor part,
                                torch::Tensor counters, torch::Tensor dn, int64_t bps, double cb,
                                int64_t spad2, int64_t spad3, int64_t spad4, int64_t spad5, int64_t fence_ra,
                                torch::Tensor slot, torch::Tensor wts, torch::Tensor ng, torch::Tensor cnt2);
void dq_arena_subset(torch::Tensor qw, torch::Tensor qoff, torch::Tensor scales, torch::Tensor su, torch::Tensor sv,
                     torch::Tensor cbtab, torch::Tensor active_ids, int64_t K, int64_t role, int64_t pad,
                     int64_t m, int64_t n, int64_t s_off, int64_t s_es, int64_t su_off, int64_t su_es,
                     int64_t sv_off, int64_t sv_es, torch::Tensor out, int64_t w_es, int64_t row_off);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_plan", &moe_plan);
  m.def("moe_plan2", &moe_plan2);
  m.def("moe_plan2_into", &moe_plan2_into);
  m.def("moe_gemv", &moe_gemv);
  m.def("moe_combine", &moe_combine);
  m.def("moe_gemv_dn_fused", &moe_gemv_dn_fused);
  m.def("dq_arena_subset", &dq_arena_subset);
}
