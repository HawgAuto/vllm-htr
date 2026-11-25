// K6-A host side (hand-written, NOT generated). k6.hip includes this file after closing namespace k6a; this file reopens it for the launch templates and then defines the three exported entry points at global scope.
//
// Launch/dispatch for k6_moe plus the three entry points declared in k6.cpp. Shapes are compile-time:
//   gate|up  bps 5 -> ns 4  (nbt = 2560/128 = 20)   NBL 5
//   down     bps 3 -> ns 2  (nbt =  640/128 =  5)   NBL 3     <- the adopted (3,2) split of htr-core K6
// OPT is fixed to the adopted set: ROWX(1) | CMB(32) | EPW(64). SHX(4) stays off in v0.2.4 phase A; the shared
// expert still runs on its own HTR K5 path (see REPORT.md section 2 for what phase B needs).
namespace k6a {

constexpr int K6_OPT = 1 | 32 | 64;
// bps = split width (128-blocks of input per WG), NSC = ceil(nbt/bps) fixed at compile time, NBL = LDS blocks
// reserved per row (>= bps). gate|up: nbt 20 -> 5/4/5. down: nbt 5 -> the adopted (3,2) split, 3/2/3;
// the served (1,5) shape for the T=1 bitwise gate is bps 1 / NSC 5 / NBL 1.
constexpr int K6_BPS_GU = 5, K6_NSC_GU = 4, K6_NBL_GU = 5;
constexpr int K6_BPS_DN = 3, K6_NSC_DN = 2, K6_NBL_DN = 3;

template <int KF, int M, int PH>
static void k6_launch(dim3 grid, hipStream_t st, K6Args A6) {
  constexpr int NBL = PH == 0 ? K6_NBL_GU : K6_NBL_DN;
  constexpr int NSC = PH == 0 ? K6_NSC_GU : K6_NSC_DN;
  const bool use_shx = (A6.sh_g < (int)grid.z);
  if constexpr (PH == 1) {
    TORCH_CHECK(A6.a.y != nullptr, "K6 down phase requires the fused-combine output");
    if (use_shx) {
      hipLaunchKernelGGL((k6_moe<KF, M, PH, 1, K6_OPT | 4, NBL, 2, NSC>), dim3(grid), dim3(256), 0, st, A6);
    } else {
      hipLaunchKernelGGL((k6_moe<KF, M, PH, 1, K6_OPT, NBL, 2, NSC>), dim3(grid), dim3(256), 0, st, A6);
    }
  } else {
    if (use_shx) {
      hipLaunchKernelGGL((k6_moe<KF, M, PH, 0, K6_OPT | 4, NBL, 2, NSC>), dim3(grid), dim3(256), 0, st, A6);
    } else {
      hipLaunchKernelGGL((k6_moe<KF, M, PH, 0, K6_OPT, NBL, 2, NSC>), dim3(grid), dim3(256), 0, st, A6);
    }
  }
}

template <int KF, int PH>
static void k6_dispatch_m(int M, dim3 grid, hipStream_t st, K6Args a) {
  switch (M) {
    case 1: k6_launch<KF, 1, PH>(grid, st, a); break;
    case 2: k6_launch<KF, 2, PH>(grid, st, a); break;
    case 4: k6_launch<KF, 4, PH>(grid, st, a); break;
    default: TORCH_CHECK(false, "K6 M bucket must be 1/2/4");
  }
}

template <int PH>
static void k6_dispatch(int kf, int M, dim3 grid, hipStream_t st, K6Args a) {
  switch (kf) {
    case 0: k6_dispatch_m<0, PH>(M, grid, st, a); break;
    case 3: k6_dispatch_m<3, PH>(M, grid, st, a); break;
    case 4: k6_dispatch_m<4, PH>(M, grid, st, a); break;
    case 5: k6_dispatch_m<5, PH>(M, grid, st, a); break;
    default: TORCH_CHECK(false, "K6 kf must be 0/3/4/5");
  }
}

}  // namespace k6a

using namespace k6a;   // the three entry points below are the module's exported symbols

// ids int32 [T, topk] (contiguous, cuda) -> groups. nsh = 0 for the routed-only build; sh_e is then ignored.
// The group arrays are sized A + nsh (index A..A+nsh-1 belong to the shared expert); slot stays A.
std::vector<torch::Tensor> k6_plan(torch::Tensor ids, int64_t M, int64_t sh_e, int64_t nsh, torch::Tensor x, torch::Tensor wsg) {
  TORCH_CHECK(ids.dim() == 2 && ids.scalar_type() == torch::kInt32 && ids.is_contiguous() && ids.is_cuda());
  const int T = ids.size(0), topk = ids.size(1), A = T * topk;
  TORCH_CHECK(topk == 10 && A <= 256);
  TORCH_CHECK(M == 1 || M == 2 || M == 4);              // K6: the bucket no longer has to cover the largest group
  TORCH_CHECK(nsh >= 0 && nsh <= 8 && (nsh == 0 || (sh_e >= 0 && x.defined() && wsg.defined())));
  if (nsh > 0) {
    TORCH_CHECK(x.dim() == 2 && x.size(1) == MOE_H && x.scalar_type() == torch::kBFloat16 && x.is_contiguous());
    TORCH_CHECK(wsg.numel() == MOE_H && wsg.scalar_type() == torch::kBFloat16 && wsg.is_contiguous());
  }
  const int G = A + (int)nsh;
  auto o = ids.options();
  auto grp_e = torch::empty({G}, o), grp_rows = torch::empty({G, M}, o), grp_n = torch::empty({G}, o), slot = torch::empty({A}, o);
  auto ng = torch::empty({1}, o);
  auto sg = torch::empty({std::max<int64_t>(T, 1)}, ids.options().dtype(torch::kFloat32));
  auto bf = x.defined() ? x : torch::empty({0}, ids.options().dtype(torch::kBFloat16));
  auto wb = wsg.defined() ? wsg : torch::empty({0}, ids.options().dtype(torch::kBFloat16));
  hipLaunchKernelGGL((k6_plan), dim3(1), dim3(256), 0, at::cuda::getCurrentCUDAStream(), ids.data_ptr<int>(), A, topk, (int)M,
                     grp_e.data_ptr<int>(), grp_rows.data_ptr<int>(), grp_n.data_ptr<int>(), slot.data_ptr<int>(), ng.data_ptr<int>(),
                     T, (int)sh_e, (int)nsh, reinterpret_cast<const __hip_bfloat16*>(bf.data_ptr<at::BFloat16>()),
                     reinterpret_cast<const __hip_bfloat16*>(wb.data_ptr<at::BFloat16>()), sg.data_ptr<float>());
  return {grp_e, grp_rows, grp_n, slot, ng, sg};
}

// phase 0: x bf16 [T, 2560] -> out f32 [G, M, 1280]. part/counters preallocated for G groups.
void k6_gemv(int64_t kf, int64_t M, torch::Tensor qw, torch::Tensor qoff, torch::Tensor kexp, torch::Tensor scales,
             torch::Tensor su, torch::Tensor sv, torch::Tensor grp_e, torch::Tensor grp_rows, torch::Tensor grp_n,
             torch::Tensor xin, torch::Tensor part, torch::Tensor counters, torch::Tensor out, double cb,
             int64_t spad3, int64_t spad4, int64_t spad5, int64_t fence_ra) {
  const int G = grp_e.size(0);
  const int m = 2 * MOE_I, n = MOE_H, nbt = n / 128, bps = K6_BPS_GU;
  const int ns = (nbt + bps - 1) / bps;
  TORCH_CHECK(ns == K6_NSC_GU && bps <= K6_NBL_GU);
  TORCH_CHECK(part.numel() >= (int64_t)G * ns * M * m && out.numel() >= (int64_t)G * M * m && counters.numel() >= G * (m / 128));
  TORCH_CHECK(qw.scalar_type() == torch::kInt32 && qoff.scalar_type() == torch::kInt64 && kexp.scalar_type() == torch::kInt32);
  K6Args A6;
  MoeArgs& a = A6.a;
  a.qw = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>()); a.qoff = qoff.data_ptr<int64_t>(); a.kexp = kexp.data_ptr<int>();
  a.scales = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()); a.su = su.data_ptr<uint8_t>(); a.sv = sv.data_ptr<uint8_t>();
  a.grp_e = grp_e.data_ptr<int>(); a.grp_rows = grp_rows.data_ptr<int>(); a.grp_n = grp_n.data_ptr<int>();
  a.x = reinterpret_cast<const __hip_bfloat16*>(xin.data_ptr<at::BFloat16>()); a.gu = nullptr;
  a.part = part.data_ptr<float>(); a.counters = reinterpret_cast<unsigned*>(counters.data_ptr<int32_t>()); a.out = out.data_ptr<float>();
  a.bps = bps; a.ns = ns; a.cb = (float)cb;
  for (int k = 0; k < 6; ++k) a.spadk[k] = 0;
  a.spadk[3] = (int)spad3; a.spadk[4] = (int)spad4; a.spadk[5] = (int)spad5; a.fence_ra = fence_ra ? 1 : 0;
  a.slot = nullptr; a.wts = nullptr; a.ng = nullptr; a.cnt2 = nullptr; a.y = nullptr; a.T = 0; a.topk = 0;
  A6.sh_g = G; A6.sg = nullptr;
  dim3 grid(m / 128, ns, G);
  k6_dispatch<0>((int)kf, (int)M, grid, at::cuda::getCurrentCUDAStream(), A6);
}

// phase 1: gu f32 [G, M, 1280] -> dn f32 [G, M, 2560], then the last arrivers of each strip-block combine
// y[t] = sum_s w[t,s]*dn[slot(t,s)] (+ sg[t]*dn[sh_g*M+t] when the shared expert is folded). Returns y bf16 [T,2560].
torch::Tensor k6_gemv_dn_fused(int64_t kf, int64_t M, torch::Tensor qw, torch::Tensor qoff, torch::Tensor kexp, torch::Tensor scales,
                               torch::Tensor su, torch::Tensor sv, torch::Tensor grp_e, torch::Tensor grp_rows, torch::Tensor grp_n,
                               torch::Tensor gu, torch::Tensor part, torch::Tensor counters, torch::Tensor dn, double cb,
                               int64_t spad3, int64_t spad4, int64_t spad5, int64_t fence_ra, torch::Tensor slot, torch::Tensor wts,
                               torch::Tensor ng, torch::Tensor cnt2, torch::Tensor sg, int64_t sh_g) {
  const int G = grp_e.size(0);
  const int m = MOE_H, n = MOE_I, nbt = n / 128, bps = K6_BPS_DN;
  const int ns = (nbt + bps - 1) / bps;
  TORCH_CHECK(ns == K6_NSC_DN && bps <= K6_NBL_DN);
  TORCH_CHECK(part.numel() >= (int64_t)G * ns * M * m && dn.numel() >= (int64_t)G * M * m && counters.numel() >= G * (m / 128));
  TORCH_CHECK(wts.dim() == 2 && wts.scalar_type() == torch::kFloat32 && wts.is_contiguous() && slot.numel() == wts.numel());
  TORCH_CHECK(cnt2.numel() >= m / 128 && cnt2.scalar_type() == torch::kInt32 && ng.numel() == 1);
  TORCH_CHECK(sh_g == G || (sh_g < G && sg.numel() >= wts.size(0)));
  const int T = wts.size(0), topk = wts.size(1);
  auto y = torch::empty({T, MOE_H}, dn.options().dtype(torch::kBFloat16));
  K6Args A6;
  MoeArgs& a = A6.a;
  a.qw = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>()); a.qoff = qoff.data_ptr<int64_t>(); a.kexp = kexp.data_ptr<int>();
  a.scales = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()); a.su = su.data_ptr<uint8_t>(); a.sv = sv.data_ptr<uint8_t>();
  a.grp_e = grp_e.data_ptr<int>(); a.grp_rows = grp_rows.data_ptr<int>(); a.grp_n = grp_n.data_ptr<int>();
  a.x = nullptr; a.gu = gu.data_ptr<float>();
  a.part = part.data_ptr<float>(); a.counters = reinterpret_cast<unsigned*>(counters.data_ptr<int32_t>()); a.out = dn.data_ptr<float>();
  a.bps = bps; a.ns = ns; a.cb = (float)cb;
  for (int k = 0; k < 6; ++k) a.spadk[k] = 0;
  a.spadk[3] = (int)spad3; a.spadk[4] = (int)spad4; a.spadk[5] = (int)spad5; a.fence_ra = fence_ra ? 1 : 0;
  a.slot = slot.data_ptr<int>(); a.wts = wts.data_ptr<float>(); a.ng = ng.data_ptr<int>();
  a.cnt2 = reinterpret_cast<unsigned*>(cnt2.data_ptr<int32_t>()); a.y = reinterpret_cast<__hip_bfloat16*>(y.data_ptr<at::BFloat16>());
  a.T = T; a.topk = topk;
  A6.sh_g = (int)sh_g;
  A6.sg = sh_g < G ? sg.data_ptr<float>() : nullptr;
  dim3 grid(m / 128, ns, G);
  k6_dispatch<1>((int)kf, (int)M, grid, at::cuda::getCurrentCUDAStream(), A6);
  return y;
}

// phase 0 with explicit sh_g for K6W_SHARED=1 (sh_g == A enables SHX K=5 branch even when KF != 0).
void k6_gemv_shx(int64_t kf, int64_t M, torch::Tensor qw, torch::Tensor qoff, torch::Tensor kexp, torch::Tensor scales,
                 torch::Tensor su, torch::Tensor sv, torch::Tensor grp_e, torch::Tensor grp_rows, torch::Tensor grp_n,
                 torch::Tensor xin, torch::Tensor part, torch::Tensor counters, torch::Tensor out, double cb,
                 int64_t spad3, int64_t spad4, int64_t spad5, int64_t fence_ra, int64_t sh_g) {
  const int G = grp_e.size(0);
  const int m = 2 * MOE_I, n = MOE_H, nbt = n / 128, bps = K6_BPS_GU;
  const int ns = (nbt + bps - 1) / bps;
  TORCH_CHECK(ns == K6_NSC_GU && bps <= K6_NBL_GU);
  TORCH_CHECK(part.numel() >= (int64_t)G * ns * M * m && out.numel() >= (int64_t)G * M * m && counters.numel() >= G * (m / 128));
  TORCH_CHECK(qw.scalar_type() == torch::kInt32 && qoff.scalar_type() == torch::kInt64 && kexp.scalar_type() == torch::kInt32);
  K6Args A6;
  MoeArgs& a = A6.a;
  a.qw = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>()); a.qoff = qoff.data_ptr<int64_t>(); a.kexp = kexp.data_ptr<int>();
  a.scales = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()); a.su = su.data_ptr<uint8_t>(); a.sv = sv.data_ptr<uint8_t>();
  a.grp_e = grp_e.data_ptr<int>(); a.grp_rows = grp_rows.data_ptr<int>(); a.grp_n = grp_n.data_ptr<int>();
  a.x = reinterpret_cast<const __hip_bfloat16*>(xin.data_ptr<at::BFloat16>()); a.gu = nullptr;
  a.part = part.data_ptr<float>(); a.counters = reinterpret_cast<unsigned*>(counters.data_ptr<int32_t>()); a.out = out.data_ptr<float>();
  a.bps = bps; a.ns = ns; a.cb = (float)cb;
  for (int k = 0; k < 6; ++k) a.spadk[k] = 0;
  a.spadk[3] = (int)spad3; a.spadk[4] = (int)spad4; a.spadk[5] = (int)spad5; a.fence_ra = fence_ra ? 1 : 0;
  a.slot = nullptr; a.wts = nullptr; a.ng = nullptr; a.cnt2 = nullptr; a.y = nullptr; a.T = 0; a.topk = 0;
  A6.sh_g = (int)sh_g; A6.sg = nullptr;
  dim3 grid(m / 128, ns, G);
  k6_dispatch<0>((int)kf, (int)M, grid, at::cuda::getCurrentCUDAStream(), A6);
}
