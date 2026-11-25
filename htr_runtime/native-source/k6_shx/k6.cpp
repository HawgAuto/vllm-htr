// Bindings for the K6-A routed-MoE block kernel (v0.2.4 dev/v024-moe-k6).
//
// Same call shape as native-source/moe2/kernel.cpp (moe_plan2 / moe_gemv / moe_gemv_dn_fused) so the Python
// route can swap kernels without touching the arena, the scratch buffers or the capture plan. Three differences,
// all of them properties of the kernel:
//   1. the group count is A + nsh (nsh = 0 unless the shared expert is folded: K6-B), so grp_e/grp_rows/grp_n and
//      the f32 partial/out/counter buffers are indexed to A + nsh and the grid's z extent is A + nsh;
//   2. the launch row bucket M is 1/2/4 only and no longer has to cover the largest group (k6_plan splits an
//      expert with more than M rows into several groups of <= M);
//   3. the split shapes are compile-time (gate|up bps 5 -> ns 4, down bps 3 -> ns 2, the adopted (3,2) down shape).
//      Changing them is a recipe change (rebuild), not a runtime argument; the kernel traps if they disagree.
// The routed-expert arithmetic, arena layout, strip padding, fences and counters are the served ones.
#include <torch/extension.h>

std::vector<torch::Tensor> k6_plan(torch::Tensor ids, int64_t M, int64_t sh_e, int64_t nsh, torch::Tensor x, torch::Tensor wsg);
void k6_gemv(int64_t kf, int64_t M, torch::Tensor qw, torch::Tensor qoff, torch::Tensor kexp, torch::Tensor scales,
             torch::Tensor su, torch::Tensor sv, torch::Tensor grp_e, torch::Tensor grp_rows, torch::Tensor grp_n,
             torch::Tensor xin, torch::Tensor part, torch::Tensor counters, torch::Tensor out, double cb,
             int64_t spad3, int64_t spad4, int64_t spad5, int64_t fence_ra);
torch::Tensor k6_gemv_dn_fused(int64_t kf, int64_t M, torch::Tensor qw, torch::Tensor qoff, torch::Tensor kexp, torch::Tensor scales,
                               torch::Tensor su, torch::Tensor sv, torch::Tensor grp_e, torch::Tensor grp_rows, torch::Tensor grp_n,
                               torch::Tensor gu, torch::Tensor part, torch::Tensor counters, torch::Tensor dn, double cb,
                               int64_t spad3, int64_t spad4, int64_t spad5, int64_t fence_ra, torch::Tensor slot, torch::Tensor wts,
                               torch::Tensor ng, torch::Tensor cnt2, torch::Tensor sg, int64_t sh_g);

void k6_gemv_shx(int64_t kf, int64_t M, torch::Tensor qw, torch::Tensor qoff, torch::Tensor kexp, torch::Tensor scales,
                 torch::Tensor su, torch::Tensor sv, torch::Tensor grp_e, torch::Tensor grp_rows, torch::Tensor grp_n,
                 torch::Tensor xin, torch::Tensor part, torch::Tensor counters, torch::Tensor out, double cb,
                 int64_t spad3, int64_t spad4, int64_t spad5, int64_t fence_ra, int64_t sh_g);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("k6_plan", &k6_plan);
  m.def("k6_gemv", &k6_gemv);
  m.def("k6_gemv_shx", &k6_gemv_shx);
  m.def("k6_gemv_dn_fused", &k6_gemv_dn_fused);
}
