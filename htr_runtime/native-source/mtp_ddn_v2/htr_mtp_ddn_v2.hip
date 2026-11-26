// htr_mtp_ddn_v2.hip — Clean-slate gfx1151 Wave32 MTP W8A16 group-64 dense + fused HC megakernels (V5-M2)
// Replaces:
//   1. ddn_v1.hsaco (uncoalesced 1-lane/row, no split-K, 147-237 VGPRs, cached, M<=4 cliff)
//   2. Stock Triton fused_moe_kernel_gptq_awq 16x32x64 fallback on M>4 (fc_hidden at c>=2!)
//   3. Unfused MTP HC down + inject + hc_silu + up + hc_gate_mix (5 launches -> 2 launches)
// Zero weight repack: consumes resident [N, K] uint8 (offset-128 symmetric) + [N, K/64] fp32 scales.
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <stdint.h>
// WAVE7_MTP_DPP: zero-LDS RDNA 3.5 DPP reductions (0 ds_bpermute, 0 lgkmcnt!)
__device__ __forceinline__ float quad_sum_dpp(float v) {
  asm volatile(
    "v_add_f32_dpp %0, %0, %0 row_xmask:1 row_mask:0xf bank_mask:0xf fi:1\n\t"
    "v_add_f32_dpp %0, %0, %0 row_xmask:2 row_mask:0xf bank_mask:0xf fi:1"
    : "+v"(v));
  return v;
}

__device__ __forceinline__ float stream_sum_dpp(float v) {
  asm volatile(
    "v_add_f32_dpp %0, %0, %0 row_xmask:4 row_mask:0xf bank_mask:0xf fi:1\n\t"
    "v_add_f32_dpp %0, %0, %0 row_xmask:8 row_mask:0xf bank_mask:0xf fi:1"
    : "+v"(v));
  return v;
}

__device__ __forceinline__ float wave_sum_dpp(float v) {
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


__device__ __forceinline__ float bf2f(uint16_t h) {
  return __builtin_bit_cast(float, ((uint32_t)h) << 16);
}

__device__ __forceinline__ uint16_t f2bf_rne(float x) {
  uint32_t u = __builtin_bit_cast(uint32_t, x);
  if (x != x) return 0x7fffu;
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ uint4 nt_load_u4(const void* p) {
  u32x4 v = __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(p));
  return make_uint4(v.x, v.y, v.z, v.w);
}

__device__ __forceinline__ float dot4_u8_f4(uint32_t q, float4 a) {
  float d = (float)(q & 0xFFu) * a.x;
  d = fmaf((float)((q >> 8) & 0xFFu), a.y, d);
  d = fmaf((float)((q >> 16) & 0xFFu), a.z, d);
  d = fmaf((float)(q >> 24), a.w, d);
  return d;
}

__device__ __forceinline__ float dot16_u8_f4(uint4 qv, float4 a0, float4 a1, float4 a2, float4 a3) {
  float d = dot4_u8_f4(qv.x, a0);
  d += dot4_u8_f4(qv.y, a1);
  d += dot4_u8_f4(qv.z, a2);
  d += dot4_u8_f4(qv.w, a3);
  return d;
}

__device__ __forceinline__ float dot16_u8_bf16x16_centered(uint4 qv, const uint16_t* a_ptr) {
  // 2x L0/L1-cached 128-bit loads = 16 BF16 activations (zero LDS, zero s_barrier, zero shfl!)
  const uint4 lo = *reinterpret_cast<const uint4*>(a_ptr);
  const uint4 hi = *reinterpret_cast<const uint4*>(a_ptr + 8);
  float4 a0 = make_float4(__uint_as_float((lo.x & 0xffffu) << 16), __uint_as_float(lo.x & 0xffff0000u),
                          __uint_as_float((lo.y & 0xffffu) << 16), __uint_as_float(lo.y & 0xffff0000u));
  float4 a1 = make_float4(__uint_as_float((lo.z & 0xffffu) << 16), __uint_as_float(lo.z & 0xffff0000u),
                          __uint_as_float((lo.w & 0xffffu) << 16), __uint_as_float(lo.w & 0xffff0000u));
  float4 a2 = make_float4(__uint_as_float((hi.x & 0xffffu) << 16), __uint_as_float(hi.x & 0xffff0000u),
                          __uint_as_float((hi.y & 0xffffu) << 16), __uint_as_float(hi.y & 0xffff0000u));
  float4 a3 = make_float4(__uint_as_float((hi.z & 0xffffu) << 16), __uint_as_float(hi.z & 0xffff0000u),
                          __uint_as_float((hi.w & 0xffffu) << 16), __uint_as_float(hi.w & 0xffff0000u));
  float d = dot16_u8_f4(qv, a0, a1, a2, a3);
  float sa = ((a0.x + a0.y) + (a0.z + a0.w)) + ((a1.x + a1.y) + (a1.z + a1.w))
           + ((a2.x + a2.y) + (a2.z + a2.w)) + ((a3.x + a3.y) + (a3.z + a3.w));
  return fmaf(-128.0f, sa, d);
}


// ============================================================================
// Kernel 1: General W8A16 group-64 symmetric GEMV/Skinny-GEMM (M = 1..16)
// 100% coalesced 4 lanes/row x 16B = 64B cache line per group-64 step!
// Algebraic factoring: s * (dot(q, a) - 128 * sum(a)) -> 0 (q-128) in inner loop!
// Deterministic arrival-counter split-K (0 extra launches, HIP-graph safe).
// ============================================================================
template <int BM_MAX, bool USE_SPLIT_K>
__global__ __launch_bounds__(128)
void k_ddn_gemv_v2(
    const uint16_t* __restrict__ A,     // [M, K] bf16
    const uint8_t* __restrict__ W,      // [N, K] uint8 (offset-128)
    const float* __restrict__ S,        // [N, K/64] fp32
    uint16_t* __restrict__ Y,           // [M, N] bf16
    float* __restrict__ partial,        // [split_k * N * M] fp32 (when USE_SPLIT_K)
    int32_t* __restrict__ cnt,          // [grid_n] i32 (zero-initialized, self-resetting)
    int M, int N, int K, int split_k) {
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int sub = lane & 3;
  const int r_in_w = lane >> 2;
  const int bn = blockIdx.x;
  const int ks = USE_SPLIT_K ? blockIdx.y : 0;
  const int row = bn * 32 + wave * 8 + r_in_w;
  const bool row_ok = (row < N);

  const int num_g = K >> 6;
  int g_beg, g_end;
  if constexpr (USE_SPLIT_K) {
    const int base_g = num_g / split_k;
    const int rem_g = num_g % split_k;
    g_beg = ks * base_g + (ks < rem_g ? ks : rem_g);
    g_end = g_beg + base_g + (ks < rem_g ? 1 : 0);
  } else {
    g_beg = 0;
    g_end = num_g;
  }

  __shared__ int s_is_last;

  float acc[BM_MAX];
  #pragma unroll
  for (int m = 0; m < BM_MAX; ++m) acc[m] = 0.0f;

  if (row_ok && g_beg < g_end) {
    // WAVE5_PIPELINED_DDN: software-pipeline W[g+1] & S[g+1] BEFORE computing g + branchless exact BM!
    const uint8_t* w_row = W + (int64_t)row * K + (sub << 4);
    const float* s_row = S + (int64_t)row * num_g;
    uint4 qv_cur = nt_load_u4(w_row + ((int64_t)g_beg << 6));
    float sc_cur = s_row[g_beg];
    const bool exact_m = (M == BM_MAX);
    for (int g = g_beg; g < g_end; ++g) {
      uint4 qv_nxt = qv_cur;
      float sc_nxt = sc_cur;
      if (g + 1 < g_end) {
        qv_nxt = nt_load_u4(w_row + ((int64_t)(g + 1) << 6));
        sc_nxt = s_row[g + 1];
      }
      const int k_off = (g << 6) + (sub << 4);
      if (exact_m) {
        #pragma unroll
        for (int m = 0; m < BM_MAX; ++m) {
          float d = dot16_u8_bf16x16_centered(qv_cur, A + (int64_t)m * K + k_off);
          acc[m] = fmaf(sc_cur, d, acc[m]);
        }
      } else {
        #pragma unroll
        for (int m = 0; m < BM_MAX; ++m) {
          if (m < M) {
            float d = dot16_u8_bf16x16_centered(qv_cur, A + (int64_t)m * K + k_off);
            acc[m] = fmaf(sc_cur, d, acc[m]);
          }
        }
      }
      qv_cur = qv_nxt;
      sc_cur = sc_nxt;
    }
    // Reduce across the 4 sub-lanes per row
    #pragma unroll
    for (int m = 0; m < BM_MAX; ++m) {
      if (m < M) {
        float v = acc[m];
        v = quad_sum_dpp(v);
        acc[m] = v;
      }
    }
  }

  if constexpr (!USE_SPLIT_K) {
    if (row_ok && sub == 0) {
      #pragma unroll
      for (int m = 0; m < BM_MAX; ++m) {
        if (m < M) Y[(int64_t)m * N + row] = f2bf_rne(acc[m]);
      }
    }
  } else {
    if (row_ok && sub == 0) {
      float* dst = partial + (((int64_t)ks * N + row) * M);
      #pragma unroll
      for (int m = 0; m < BM_MAX; ++m) { if (m < M) dst[m] = acc[m]; }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (tid == 0) {
      int old = atomicAdd(&cnt[bn], 1);
      int last = (old == split_k - 1) ? 1 : 0;
      if (last) cnt[bn] = 0;
      s_is_last = last;
    }
    __syncthreads();
    if (s_is_last) {
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
      if (row_ok && sub == 0) {
        #pragma unroll 1
        for (int m = 0; m < M; ++m) {
          float tot = 0.0f;
          for (int s = 0; s < split_k; ++s) {
            tot += __builtin_nontemporal_load(&partial[(((int64_t)s * N + row) * M) + m]);
          }
          Y[(int64_t)m * N + row] = f2bf_rne(tot);
        }
      }
    }
  }
}

// ============================================================================
// Kernel 2: Fused MTP HC Down [N_down, K] + Inject [N_inj, K] + hc_silu(lora)
// v0.2.7: ZERO s_a LDS, ZERO K-loop barriers (160 barriers/launch -> 0!), agent fences!
// ============================================================================
template <int BM_MAX, bool HAS_INJECT>
__global__ __launch_bounds__(128)
void k_ddn_hc_down_inject_silu_v2(
    const uint16_t* __restrict__ A,        // [M, K] bf16 (K=10240)
    const uint8_t* __restrict__ W_down,    // [N_down, K] uint8 (N_down=320)
    const float* __restrict__ S_down,      // [N_down, K/64] fp32
    const uint8_t* __restrict__ W_inj,     // [N_inj, K] uint8 (N_inj=4, nullable)
    const float* __restrict__ S_inj,       // [N_inj, K/64] fp32 (nullable)
    uint16_t* __restrict__ Y_lora,         // [M, N_down] bf16 (after hc_silu!)
    uint16_t* __restrict__ Y_inj,          // [M, N_inj] bf16 (nullable)
    float* __restrict__ partial,           // [split_k * N_tot * M] fp32
    int32_t* __restrict__ cnt,             // [grid_n] i32 (self-resetting)
    int M, int N_down, int N_inj, int K, int split_k, float inv_hc) {
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int sub = lane & 3;
  const int r_in_w = lane >> 2;
  const int bn = blockIdx.x;
  const int ks = blockIdx.y;
  const int row = bn * 32 + wave * 8 + r_in_w;
  const int N_tot = HAS_INJECT ? (N_down + N_inj) : N_down;
  const bool row_ok = (row < N_tot);
  const bool is_inj = HAS_INJECT && (row >= N_down);

  const int num_g = K >> 6;
  const int base_g = num_g / split_k;
  const int rem_g = num_g % split_k;
  const int g_beg = ks * base_g + (ks < rem_g ? ks : rem_g);
  const int g_end = g_beg + base_g + (ks < rem_g ? 1 : 0);

  __shared__ int s_is_last;

  float acc[BM_MAX];
  #pragma unroll
  for (int m = 0; m < BM_MAX; ++m) acc[m] = 0.0f;

  if (row_ok) {
    const uint8_t* w_row = is_inj
        ? (W_inj + (int64_t)(row - N_down) * K + (sub << 4))
        : (W_down + (int64_t)row * K + (sub << 4));
    const float* s_row = is_inj
        ? (S_inj + (int64_t)(row - N_down) * num_g)
        : (S_down + (int64_t)row * num_g);

    for (int g = g_beg; g < g_end; ++g) {
      const uint4 qv = nt_load_u4(w_row + ((int64_t)g << 6));
      const float sc = s_row[g];
      const int k_off = (g << 6) + (sub << 4);
      #pragma unroll
      for (int m = 0; m < BM_MAX; ++m) {
        if (m < M) {
          float d = dot16_u8_bf16x16_centered(qv, A + (int64_t)m * K + k_off);
          acc[m] = fmaf(sc, d, acc[m]);
        }
      }
    }
    #pragma unroll
    for (int m = 0; m < BM_MAX; ++m) {
      if (m < M) {
        float v = acc[m];
        v = quad_sum_dpp(v);
        acc[m] = v;
      }
    }
  }

  if (row_ok && sub == 0) {
    float* dst = partial + (((int64_t)ks * N_tot + row) * M);
    #pragma unroll
    for (int m = 0; m < BM_MAX; ++m) { if (m < M) dst[m] = acc[m]; }
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();
  if (tid == 0) {
    int old = atomicAdd(&cnt[bn], 1);
    int last = (old == split_k - 1) ? 1 : 0;
    if (last) cnt[bn] = 0;
    s_is_last = last;
  }
  __syncthreads();
  if (s_is_last) {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if (row_ok && sub == 0) {
      #pragma unroll 1
      for (int m = 0; m < M; ++m) {
        float tot = 0.0f;
        for (int s = 0; s < split_k; ++s) {
          tot += __builtin_nontemporal_load(&partial[(((int64_t)s * N_tot + row) * M) + m]);
        }
        if (is_inj) {
          Y_inj[(int64_t)m * N_inj + (row - N_down)] = f2bf_rne(tot);
        } else {
          float x = bf2f(f2bf_rne(tot)) * inv_hc;
          float y = x / (1.0f + __expf(-x));
          Y_lora[(int64_t)m * N_down + row] = f2bf_rne(y);
        }
      }
    }
  }
}

// ============================================================================
// Kernel 3: Fused MTP HC Up [4*H, 320] + hc_gate_mix (H=2560, K=320, 5 groups)
// v0.2.7: ZERO LDS, ZERO s_barrier, 2x b128 L0-cached LORA loads (640B*M)!
// ============================================================================
template <int BM_MAX>
__global__ __launch_bounds__(128)
void k_ddn_hc_up_gatemix_v2(
    const uint16_t* __restrict__ XN,    // [M, 4*H] bf16
    const uint16_t* __restrict__ LORA,  // [M, 320] bf16
    const uint8_t* __restrict__ W_up,   // [4*H, 320] uint8
    const float* __restrict__ S_up,     // [4*H, 5] fp32
    uint16_t* __restrict__ OUT,         // [M, H] bf16
    int M, int H) {
  constexpr int K = 320;
  constexpr int NG = 5;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int sub = lane & 3;
  const int stream = (lane >> 2) & 3;
  const int h_local = (lane >> 4) & 1;
  const int h = blockIdx.x * 8 + wave * 2 + h_local;
  if (h >= H) return;
  const int row = stream * H + h;

  float acc[BM_MAX];
  #pragma unroll
  for (int m = 0; m < BM_MAX; ++m) acc[m] = 0.0f;

  const uint8_t* w_row = W_up + (int64_t)row * K + (sub << 4);
  const float* s_row = S_up + (int64_t)row * NG;

  #pragma unroll
  for (int g = 0; g < NG; ++g) {
    const uint4 qv = nt_load_u4(w_row + (g << 6));
    const float sc = s_row[g];
    const int k_off = (g << 6) + (sub << 4);
    #pragma unroll
    for (int m = 0; m < BM_MAX; ++m) {
      if (m < M) {
        float d = dot16_u8_bf16x16_centered(qv, LORA + (int64_t)m * K + k_off);
        acc[m] = fmaf(sc, d, acc[m]);
      }
    }
  }

  #pragma unroll
  for (int m = 0; m < BM_MAX; ++m) {
    if (m >= M) continue;
    float gval = acc[m];
    gval = quad_sum_dpp(gval);
    float contrib = 0.0f;
    if (sub == 0) {
      float gb = bf2f(f2bf_rne(gval));
      float xv = bf2f(XN[(int64_t)m * (4 * H) + row]);
      contrib = xv / (1.0f + __expf(-gb));
    }
    contrib = stream_sum_dpp(contrib);
    if (sub == 0 && stream == 0) {
      OUT[(int64_t)m * H + h] = f2bf_rne(contrib * 0.25f);
    }
  }
}

// Persistent scratch/counter helpers (per-device, HIP-graph safe)
static torch::Tensor g_partial;
static torch::Tensor g_cnt;

static void ensure_scratch(const torch::Device& dev, int64_t partial_elems, int64_t cnt_elems) {
  if (!g_partial.defined() || g_partial.device() != dev || g_partial.numel() < partial_elems) {
    g_partial = torch::empty({std::max<int64_t>(partial_elems, 1024 * 1024)},
                             torch::TensorOptions().dtype(torch::kFloat32).device(dev));
  }
  if (!g_cnt.defined() || g_cnt.device() != dev || g_cnt.numel() < cnt_elems) {
    g_cnt = torch::zeros({std::max<int64_t>(cnt_elems, 4096)},
                         torch::TensorOptions().dtype(torch::kInt32).device(dev));
  }
}

static int choose_split_k(int N, int K) {
  int num_g = K >> 6;
  int grid_n = (N + 31) >> 5;
  // v0.2.7: target >= 320 WGs (8 WGs/CU across 40 CUs) to saturate 242 GB/s LPDDR5X!
  if (grid_n >= 240) return 1;
  int sk = (320 + grid_n - 1) / grid_n;
  // Keep at least 16 groups (1024 K-elems) per split so epilogue overhead stays tiny
  int max_sk = num_g / 16;
  if (max_sk < 1) max_sk = 1;
  if (sk > max_sk) sk = max_sk;
  if (sk > 40) sk = 40;
  return sk < 1 ? 1 : sk;
}

template <bool USE_SPLIT_K>
static void dispatch_gemv_bm(
    const uint16_t* A, const uint8_t* W, const float* S, uint16_t* Y,
    float* partial, int32_t* cnt, int M, int N, int K, int split_k, hipStream_t stream) {
  dim3 grid((N + 31) >> 5, USE_SPLIT_K ? split_k : 1);
  dim3 block(128);
  if (M <= 1) {
    k_ddn_gemv_v2<1, USE_SPLIT_K><<<grid, block, 0, stream>>>(A, W, S, Y, partial, cnt, M, N, K, split_k);
  } else if (M <= 2) {
    k_ddn_gemv_v2<2, USE_SPLIT_K><<<grid, block, 0, stream>>>(A, W, S, Y, partial, cnt, M, N, K, split_k);
  } else if (M <= 4) {
    k_ddn_gemv_v2<4, USE_SPLIT_K><<<grid, block, 0, stream>>>(A, W, S, Y, partial, cnt, M, N, K, split_k);
  } else if (M <= 8) {
    k_ddn_gemv_v2<8, USE_SPLIT_K><<<grid, block, 0, stream>>>(A, W, S, Y, partial, cnt, M, N, K, split_k);
  } else {
    k_ddn_gemv_v2<16, USE_SPLIT_K><<<grid, block, 0, stream>>>(A, W, S, Y, partial, cnt, M, N, K, split_k);
  }
}

torch::Tensor ddn_gemv_v2(torch::Tensor A, torch::Tensor W, torch::Tensor S) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kBFloat16 && A.is_contiguous() && A.dim() == 2);
  TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kUInt8 && W.is_contiguous() && W.dim() == 2);
  TORCH_CHECK(S.is_cuda() && S.dtype() == torch::kFloat32 && S.is_contiguous() && S.dim() == 2);
  const int M = (int)A.size(0);
  const int K = (int)A.size(1);
  const int N = (int)W.size(0);
  TORCH_CHECK(W.size(1) == K && (K % 64 == 0) && S.size(0) == N && S.size(1) == (K >> 6));
  auto Y = torch::empty({M, N}, A.options());
  if (M == 0) return Y;

  const int split_k = choose_split_k(N, K);
  const int grid_n = (N + 31) >> 5;
  hipStream_t stream = at::hip::getCurrentHIPStream().stream();

  // Process in chunks of <= 16 rows so any M works cleanly!
  const int max_chunk = 16;
  if (split_k > 1) {
    ensure_scratch(A.device(), (int64_t)split_k * N * std::min(M, max_chunk), grid_n);
  }
  const uint16_t* a_ptr = reinterpret_cast<const uint16_t*>(A.data_ptr());
  const uint8_t* w_ptr = W.data_ptr<uint8_t>();
  const float* s_ptr = S.data_ptr<float>();
  uint16_t* y_ptr = reinterpret_cast<uint16_t*>(Y.data_ptr());
  float* p_ptr = (split_k > 1) ? g_partial.data_ptr<float>() : nullptr;
  int32_t* c_ptr = (split_k > 1) ? g_cnt.data_ptr<int32_t>() : nullptr;

  for (int m0 = 0; m0 < M; m0 += max_chunk) {
    int mc = std::min(max_chunk, M - m0);
    if (split_k > 1) {
      dispatch_gemv_bm<true>(a_ptr + (int64_t)m0 * K, w_ptr, s_ptr, y_ptr + (int64_t)m0 * N,
                             p_ptr, c_ptr, mc, N, K, split_k, stream);
    } else {
      dispatch_gemv_bm<false>(a_ptr + (int64_t)m0 * K, w_ptr, s_ptr, y_ptr + (int64_t)m0 * N,
                              nullptr, nullptr, mc, N, K, 1, stream);
    }
  }
  return Y;
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> ddn_hc_down_inject_silu_v2(
    torch::Tensor A,
    torch::Tensor W_down, torch::Tensor S_down,
    c10::optional<torch::Tensor> W_inj_opt, c10::optional<torch::Tensor> S_inj_opt,
    int64_t hc_count) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kBFloat16 && A.is_contiguous() && A.dim() == 2);
  const int M = (int)A.size(0);
  const int K = (int)A.size(1);
  const int N_down = (int)W_down.size(0);
  const bool has_inj = W_inj_opt.has_value() && W_inj_opt->defined();
  const int N_inj = has_inj ? (int)W_inj_opt->size(0) : 0;
  const int N_tot = N_down + N_inj;
  auto Y_lora = torch::empty({M, N_down}, A.options());
  c10::optional<torch::Tensor> Y_inj_opt = c10::nullopt;
  if (has_inj) Y_inj_opt = torch::empty({M, N_inj}, A.options());
  if (M == 0) return {Y_lora, Y_inj_opt};

  const int split_k = choose_split_k(N_tot, K);
  const int grid_n = (N_tot + 31) >> 5;
  const int max_chunk = 16;
  ensure_scratch(A.device(), (int64_t)split_k * N_tot * std::min(M, max_chunk), grid_n);
  hipStream_t stream = at::hip::getCurrentHIPStream().stream();

  const uint16_t* a_ptr = reinterpret_cast<const uint16_t*>(A.data_ptr());
  const uint8_t* wd_ptr = W_down.data_ptr<uint8_t>();
  const float* sd_ptr = S_down.data_ptr<float>();
  const uint8_t* wi_ptr = has_inj ? W_inj_opt->data_ptr<uint8_t>() : nullptr;
  const float* si_ptr = has_inj ? S_inj_opt->data_ptr<float>() : nullptr;
  uint16_t* yl_ptr = reinterpret_cast<uint16_t*>(Y_lora.data_ptr());
  uint16_t* yi_ptr = has_inj ? reinterpret_cast<uint16_t*>(Y_inj_opt->data_ptr()) : nullptr;
  float* p_ptr = g_partial.data_ptr<float>();
  int32_t* c_ptr = g_cnt.data_ptr<int32_t>();
  const float inv_hc = 1.0f / (float)hc_count;

  dim3 grid(grid_n, split_k);
  dim3 block(128);
  for (int m0 = 0; m0 < M; m0 += max_chunk) {
    int mc = std::min(max_chunk, M - m0);
    auto launch = [&]<int BM, bool HI>() {
      k_ddn_hc_down_inject_silu_v2<BM, HI><<<grid, block, 0, stream>>>(
          a_ptr + (int64_t)m0 * K, wd_ptr, sd_ptr, wi_ptr, si_ptr,
          yl_ptr + (int64_t)m0 * N_down, has_inj ? (yi_ptr + (int64_t)m0 * N_inj) : nullptr,
          p_ptr, c_ptr, mc, N_down, N_inj, K, split_k, inv_hc);
    };
    if (has_inj) {
      if (mc <= 1) launch.template operator()<1, true>();
      else if (mc <= 2) launch.template operator()<2, true>();
      else if (mc <= 4) launch.template operator()<4, true>();
      else if (mc <= 8) launch.template operator()<8, true>();
      else launch.template operator()<16, true>();
    } else {
      if (mc <= 1) launch.template operator()<1, false>();
      else if (mc <= 2) launch.template operator()<2, false>();
      else if (mc <= 4) launch.template operator()<4, false>();
      else if (mc <= 8) launch.template operator()<8, false>();
      else launch.template operator()<16, false>();
    }
  }
  return {Y_lora, Y_inj_opt};
}

torch::Tensor ddn_hc_up_gatemix_v2(
    torch::Tensor XN, torch::Tensor LORA, torch::Tensor W_up, torch::Tensor S_up) {
  TORCH_CHECK(XN.is_cuda() && XN.dtype() == torch::kBFloat16 && XN.is_contiguous() && XN.dim() == 2);
  TORCH_CHECK(LORA.is_cuda() && LORA.dtype() == torch::kBFloat16 && LORA.is_contiguous() && LORA.dim() == 2);
  TORCH_CHECK(W_up.is_cuda() && W_up.dtype() == torch::kUInt8 && W_up.is_contiguous() && W_up.dim() == 2);
  TORCH_CHECK(S_up.is_cuda() && S_up.dtype() == torch::kFloat32 && S_up.is_contiguous() && S_up.dim() == 2);
  const int M = (int)XN.size(0);
  const int H = (int)XN.size(1) / 4;
  TORCH_CHECK(LORA.size(0) == M && LORA.size(1) == 320 && W_up.size(0) == 4 * H && W_up.size(1) == 320);
  auto OUT = torch::empty({M, H}, XN.options());
  if (M == 0) return OUT;

  hipStream_t stream = at::hip::getCurrentHIPStream().stream();
  const int max_chunk = 16;
  dim3 grid((H + 7) >> 3);
  dim3 block(128);
  const uint16_t* xn_ptr = reinterpret_cast<const uint16_t*>(XN.data_ptr());
  const uint16_t* lo_ptr = reinterpret_cast<const uint16_t*>(LORA.data_ptr());
  const uint8_t* w_ptr = W_up.data_ptr<uint8_t>();
  const float* s_ptr = S_up.data_ptr<float>();
  uint16_t* o_ptr = reinterpret_cast<uint16_t*>(OUT.data_ptr());

  for (int m0 = 0; m0 < M; m0 += max_chunk) {
    int mc = std::min(max_chunk, M - m0);
    if (mc <= 1) {
      k_ddn_hc_up_gatemix_v2<1><<<grid, block, 0, stream>>>(xn_ptr + (int64_t)m0 * 4 * H, lo_ptr + (int64_t)m0 * 320, w_ptr, s_ptr, o_ptr + (int64_t)m0 * H, mc, H);
    } else if (mc <= 2) {
      k_ddn_hc_up_gatemix_v2<2><<<grid, block, 0, stream>>>(xn_ptr + (int64_t)m0 * 4 * H, lo_ptr + (int64_t)m0 * 320, w_ptr, s_ptr, o_ptr + (int64_t)m0 * H, mc, H);
    } else if (mc <= 4) {
      k_ddn_hc_up_gatemix_v2<4><<<grid, block, 0, stream>>>(xn_ptr + (int64_t)m0 * 4 * H, lo_ptr + (int64_t)m0 * 320, w_ptr, s_ptr, o_ptr + (int64_t)m0 * H, mc, H);
    } else if (mc <= 8) {
      k_ddn_hc_up_gatemix_v2<8><<<grid, block, 0, stream>>>(xn_ptr + (int64_t)m0 * 4 * H, lo_ptr + (int64_t)m0 * 320, w_ptr, s_ptr, o_ptr + (int64_t)m0 * H, mc, H);
    } else {
      k_ddn_hc_up_gatemix_v2<16><<<grid, block, 0, stream>>>(xn_ptr + (int64_t)m0 * 4 * H, lo_ptr + (int64_t)m0 * 320, w_ptr, s_ptr, o_ptr + (int64_t)m0 * H, mc, H);
    }
  }
  return OUT;
}

int64_t ddn_cnt_max() {
  if (!g_cnt.defined()) return 0;
  return g_cnt.max().item<int64_t>();
}


// ============================================================================
// Kernel 4 (V5-M3): Fused GemmaRMSNorm(K) + W8A16 g64 GEMV (fc_embedding)
// Replaces pre_fc_norm_embedding + fc_embedding (2->1 launches, 0% padding, 0 DRAM round-trip).
// ============================================================================
template <int BM_MAX>
__global__ __launch_bounds__(128)
void k_ddn_norm_gemv_v2(
    const uint16_t* __restrict__ A,     // [M, K] bf16 (raw inputs_embeds)
    const uint16_t* __restrict__ NW,    // [K] bf16 (GemmaRMSNorm weight)
    const uint8_t* __restrict__ W,      // [N, K] uint8
    const float* __restrict__ S,        // [N, K/64] fp32
    uint16_t* __restrict__ Y,           // [M, N] bf16
    int M, int N, int K, float eps) {
  constexpr int K_TILE = 128;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int sub = lane & 3;
  const int r_in_w = lane >> 2;
  const int row = blockIdx.x * 32 + wave * 8 + r_in_w;
  const bool row_ok = (row < N);
  const int num_g = K >> 6;

  __shared__ float s_a[BM_MAX][K_TILE];
  __shared__ float s_half[BM_MAX][4];
  __shared__ float s_rrms[BM_MAX];

  // Exact zero-padding RMSNorm denominator across all K elements (2560 = 20 x 128)
  #pragma unroll 1
  for (int m = 0; m < M; ++m) {
    float sq = 0.0f;
    const uint16_t* ap = A + (int64_t)m * K;
    for (int k = tid; k < K; k += 128) {
      float v = bf2f(ap[k]);
      sq = fmaf(v, v, sq);
    }
    sq = wave_sum_dpp(sq);
    if (lane == 0) s_half[m][wave] = sq;
  }
  __syncthreads();
  if (tid < M) {
    float tot = s_half[tid][0] + s_half[tid][1] + s_half[tid][2] + s_half[tid][3];
    s_rrms[tid] = rsqrtf(tot / (float)K + eps);
  }
  __syncthreads();

  float acc[BM_MAX];
  #pragma unroll
  for (int m = 0; m < BM_MAX; ++m) acc[m] = 0.0f;

  const uint8_t* w_row = row_ok ? (W + (int64_t)row * K) : nullptr;
  const float* s_row = row_ok ? (S + (int64_t)row * num_g) : nullptr;

  for (int g0 = 0; g0 < num_g; g0 += 2) {
    const int ng_tile = (g0 + 1 < num_g) ? 2 : 1;
    const int k_base = g0 << 6;
    const bool valid_k = (tid < (ng_tile << 6));
    const float nw_val = valid_k ? (1.0f + bf2f(NW[k_base + tid])) : 0.0f;

    #pragma unroll 1
    for (int m = 0; m < M; ++m) {
      float v = 0.0f;
      if (valid_k) {
        float x = bf2f(A[(int64_t)m * K + k_base + tid]);
        v = bf2f(f2bf_rne(x * s_rrms[m] * nw_val));
      }
      s_a[m][tid] = v;
      v = wave_sum_dpp(v);
      if (lane == 0) s_half[m][wave] = v;
    }
    __syncthreads();

    if (row_ok) {
      #pragma unroll
      for (int lg = 0; lg < 2; ++lg) {
        if (lg >= ng_tile) break;
        const int g = g0 + lg;
        const uint4 qv = nt_load_u4(w_row + ((int64_t)g << 6) + (sub << 4));
        const float sc = s_row[g];
        const int a_off = (lg << 6) + (sub << 4);
        #pragma unroll
        for (int m = 0; m < BM_MAX; ++m) {
          if (m < M) {
            const float4* ap = reinterpret_cast<const float4*>(&s_a[m][a_off]);
            float d = dot16_u8_f4(qv, ap[0], ap[1], ap[2], ap[3]);
            if (sub == 0) {
              float sum_g = s_half[m][lg * 2] + s_half[m][lg * 2 + 1];
              d = fmaf(-128.0f, sum_g, d);
            }
            acc[m] = fmaf(sc, d, acc[m]);
          }
        }
      }
    }
    __syncthreads();
  }

  if (row_ok) {
    #pragma unroll
    for (int m = 0; m < BM_MAX; ++m) {
      if (m < M) {
        float v = acc[m];
        v = quad_sum_dpp(v);
        if (sub == 0) Y[(int64_t)m * N + row] = f2bf_rne(v);
      }
    }
  }
}

// ============================================================================
// Kernel 5 (V5-M3): Fused GemmaRMSNorm(HC*K) + W8A16 g64 fc_hidden + broadcast_add(EMB)
// Replaces 16384-padded Inductor GemmaRMSNorm(10240) + fc_hidden + broadcast_add (3->1 launches!)
// ============================================================================
template <int BM_MAX>
__global__ __launch_bounds__(128)
void k_ddn_norm_hc_gemv_add_v2(
    const uint16_t* __restrict__ H,     // [T, HC*K] bf16 (== [M, K] where M = T*HC)
    const uint16_t* __restrict__ NW,    // [HC*K] bf16 (GemmaRMSNorm(10240) weight)
    const uint16_t* __restrict__ EMB,   // [T, N] bf16 (fc_embedding output to broadcast-add)
    const uint8_t* __restrict__ W,      // [N, K] uint8 (fc_hidden weight)
    const float* __restrict__ S,        // [N, K/64] fp32
    uint16_t* __restrict__ Y,           // [T, HC*N] bf16 (== [M, N])
    int T, int HC, int N, int K, float eps) {
  constexpr int K_TILE = 128;
  const int M = T * HC;
  const int HK = HC * K;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int sub = lane & 3;
  const int r_in_w = lane >> 2;
  const int row = blockIdx.x * 32 + wave * 8 + r_in_w;
  const bool row_ok = (row < N);
  const int num_g = K >> 6;

  __shared__ float s_a[BM_MAX][K_TILE];
  __shared__ float s_half[BM_MAX][4];
  __shared__ float s_rrms[BM_MAX]; // only first T entries used

  // Exact zero-padding RMSNorm(10240) per token t (10240 = 80 x 128, 0% wasted lanes!)
  #pragma unroll 1
  for (int t = 0; t < T; ++t) {
    float sq = 0.0f;
    const uint16_t* hp = H + (int64_t)t * HK;
    for (int k = tid; k < HK; k += 128) {
      float v = bf2f(hp[k]);
      sq = fmaf(v, v, sq);
    }
    sq = wave_sum_dpp(sq);
    if (lane == 0) s_half[t][wave] = sq;
  }
  __syncthreads();
  if (tid < T) {
    float tot = s_half[tid][0] + s_half[tid][1] + s_half[tid][2] + s_half[tid][3];
    s_rrms[tid] = rsqrtf(tot / (float)HK + eps);
  }
  __syncthreads();

  float acc[BM_MAX];
  #pragma unroll
  for (int m = 0; m < BM_MAX; ++m) acc[m] = 0.0f;

  const uint8_t* w_row = row_ok ? (W + (int64_t)row * K) : nullptr;
  const float* s_row = row_ok ? (S + (int64_t)row * num_g) : nullptr;

  for (int g0 = 0; g0 < num_g; g0 += 2) {
    const int ng_tile = (g0 + 1 < num_g) ? 2 : 1;
    const int k_base = g0 << 6;
    const bool valid_k = (tid < (ng_tile << 6));

    #pragma unroll 1
    for (int m = 0; m < M; ++m) {
      const int t = m / HC;
      const int c = m - t * HC;
      float v = 0.0f;
      if (valid_k) {
        float x = bf2f(H[(int64_t)m * K + k_base + tid]);
        float nw = 1.0f + bf2f(NW[(int64_t)c * K + k_base + tid]);
        v = bf2f(f2bf_rne(x * s_rrms[t] * nw));
      }
      s_a[m][tid] = v;
      v = wave_sum_dpp(v);
      if (lane == 0) s_half[m][wave] = v;
    }
    __syncthreads();

    if (row_ok) {
      #pragma unroll
      for (int lg = 0; lg < 2; ++lg) {
        if (lg >= ng_tile) break;
        const int g = g0 + lg;
        const uint4 qv = nt_load_u4(w_row + ((int64_t)g << 6) + (sub << 4));
        const float sc = s_row[g];
        const int a_off = (lg << 6) + (sub << 4);
        #pragma unroll
        for (int m = 0; m < BM_MAX; ++m) {
          if (m < M) {
            const float4* ap = reinterpret_cast<const float4*>(&s_a[m][a_off]);
            float d = dot16_u8_f4(qv, ap[0], ap[1], ap[2], ap[3]);
            if (sub == 0) {
              float sum_g = s_half[m][lg * 2] + s_half[m][lg * 2 + 1];
              d = fmaf(-128.0f, sum_g, d);
            }
            acc[m] = fmaf(sc, d, acc[m]);
          }
        }
      }
    }
    __syncthreads();
  }

  if (row_ok) {
    #pragma unroll
    for (int m = 0; m < BM_MAX; ++m) {
      if (m < M) {
        float v = acc[m];
        v = quad_sum_dpp(v);
        if (sub == 0) {
          const int t = m / HC;
          float fc_b = bf2f(f2bf_rne(v));
          float emb_b = bf2f(EMB[(int64_t)t * N + row]);
          Y[(int64_t)m * N + row] = f2bf_rne(emb_b + fc_b);
        }
      }
    }
  }
}

torch::Tensor ddn_mtp_prefc_fuse_v2(
    torch::Tensor inputs_embeds, torch::Tensor nw_emb, torch::Tensor W_emb, torch::Tensor S_emb,
    torch::Tensor hidden_states, torch::Tensor nw_hid, torch::Tensor W_hid, torch::Tensor S_hid,
    int64_t hc_count, double eps) {
  TORCH_CHECK(inputs_embeds.is_cuda() && inputs_embeds.dtype() == torch::kBFloat16 && inputs_embeds.is_contiguous());
  TORCH_CHECK(hidden_states.is_cuda() && hidden_states.dtype() == torch::kBFloat16 && hidden_states.is_contiguous());
  const int T = (int)inputs_embeds.size(0);
  const int K = (int)inputs_embeds.size(1);
  const int N = (int)W_emb.size(0);
  const int HC = (int)hc_count;
  TORCH_CHECK(hidden_states.size(0) == T && hidden_states.size(1) == HC * K && W_hid.size(0) == N && W_hid.size(1) == K);
  auto out = torch::empty({T, HC * N}, inputs_embeds.options());
  if (T == 0) return out;
  auto emb_out = torch::empty({T, N}, inputs_embeds.options());
  hipStream_t stream = at::hip::getCurrentHIPStream().stream();
  dim3 grid((N + 31) >> 5);
  dim3 block(128);
  const uint16_t* ie_p = reinterpret_cast<const uint16_t*>(inputs_embeds.data_ptr());
  const uint16_t* nwe_p = reinterpret_cast<const uint16_t*>(nw_emb.data_ptr());
  const uint8_t* we_p = W_emb.data_ptr<uint8_t>();
  const float* se_p = S_emb.data_ptr<float>();
  uint16_t* eo_p = reinterpret_cast<uint16_t*>(emb_out.data_ptr());
  const uint16_t* hs_p = reinterpret_cast<const uint16_t*>(hidden_states.data_ptr());
  const uint16_t* nwh_p = reinterpret_cast<const uint16_t*>(nw_hid.data_ptr());
  const uint8_t* wh_p = W_hid.data_ptr<uint8_t>();
  const float* sh_p = S_hid.data_ptr<float>();
  uint16_t* o_p = reinterpret_cast<uint16_t*>(out.data_ptr());
  const float feps = (float)eps;

  // Chunk tokens in blocks of <= 4 (so M = T_chunk * 4 <= 16!)
  const int max_t = 16 / HC;
  for (int t0 = 0; t0 < T; t0 += max_t) {
    int tc = std::min(max_t, T - t0);
    int mc = tc * HC;
    if (tc <= 1) {
      k_ddn_norm_gemv_v2<1><<<grid, block, 0, stream>>>(ie_p + (int64_t)t0 * K, nwe_p, we_p, se_p, eo_p + (int64_t)t0 * N, tc, N, K, feps);
    } else if (tc <= 2) {
      k_ddn_norm_gemv_v2<2><<<grid, block, 0, stream>>>(ie_p + (int64_t)t0 * K, nwe_p, we_p, se_p, eo_p + (int64_t)t0 * N, tc, N, K, feps);
    } else {
      k_ddn_norm_gemv_v2<4><<<grid, block, 0, stream>>>(ie_p + (int64_t)t0 * K, nwe_p, we_p, se_p, eo_p + (int64_t)t0 * N, tc, N, K, feps);
    }
    if (mc <= 4) {
      k_ddn_norm_hc_gemv_add_v2<4><<<grid, block, 0, stream>>>(hs_p + (int64_t)t0 * HC * K, nwh_p, eo_p + (int64_t)t0 * N, wh_p, sh_p, o_p + (int64_t)t0 * HC * N, tc, HC, N, K, feps);
    } else if (mc <= 8) {
      k_ddn_norm_hc_gemv_add_v2<8><<<grid, block, 0, stream>>>(hs_p + (int64_t)t0 * HC * K, nwh_p, eo_p + (int64_t)t0 * N, wh_p, sh_p, o_p + (int64_t)t0 * HC * N, tc, HC, N, K, feps);
    } else {
      k_ddn_norm_hc_gemv_add_v2<16><<<grid, block, 0, stream>>>(hs_p + (int64_t)t0 * HC * K, nwh_p, eo_p + (int64_t)t0 * N, wh_p, sh_p, o_p + (int64_t)t0 * HC * N, tc, HC, N, K, feps);
    }
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ddn_gemv_v2", &ddn_gemv_v2, "MTP W8A16 group-64 coalesced nontemporal GEMV/Skinny-GEMM (M=1..64)");
  m.def("ddn_hc_down_inject_silu_v2", &ddn_hc_down_inject_silu_v2, "Fused MTP HC down+inject+hc_silu");
  m.def("ddn_hc_up_gatemix_v2", &ddn_hc_up_gatemix_v2, "Fused MTP HC up+hc_gate_mix");
  m.def("ddn_cnt_max", &ddn_cnt_max, "Check arrival counter self-reset");
  m.def("ddn_mtp_prefc_fuse_v2", &ddn_mtp_prefc_fuse_v2, "V5-M3 Fused MTP pre_fc_norm + fc_embedding + pre_fc_norm_hidden + fc_hidden + broadcast_add (5->2 launches)");
}
