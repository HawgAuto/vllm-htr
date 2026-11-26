// hiphc_v2.hip — Clean-slate gfx1151 Wave32 Hyper-Connection Decode Kernels (M=1..16)
// Replaces hiphc.hip (which runs 96x/step = 634 MB/step) with:
//   1. Algebraic group-128 factored dequant: sum((q-z)*s*x) = s*sum(q*x) - (z*s)*sum(x)
//      Uses native gfx1151 v_cvt_f32_ubyte0..3, ZERO per-element (q-z)*s, ZERO 4-op software bf16r!
//   2. k_up_gatemix_v2: 4 lanes/row x 8 rows/wave x 5 exact 64B cache-line steps for K=320:
//      - 100% cache-line coalesced nontemporal 16B loads (eliminates k_ugm_lds's 320B-strided 4x BW blowup!)
//      - ZERO uint4 qv[20] (80-VGPR) preload -> streams 1 chunk at a time (~48 VGPRs, 0 spills, 16 waves/SIMD!)
//      - ZERO pc[4][M][160] LDS partial buffer -> 2-op Wave32 DPP reduction across the 4 lanes/row!
//   3. k_down_silu_v2: 8 lanes/group-128 x 4 groups/wave (512B coalesced wave bursts),
//      xsum computed ONCE across RPW rows, native M=1..16 (eliminates chunk8 & M=9..16 cliff!).
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define DEV __device__ __forceinline__

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

DEV float bf2f(uint32_t b16) { return __uint_as_float(b16 << 16); }
DEV float h2f(uint32_t h16) { return __half2float(__ushort_as_half((uint16_t)h16)); }
DEV uint16_t f2bf(float f) {
  uint32_t u = __float_as_uint(f);
  if (f != f) return 0x7fffu;
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

DEV uint4 ldnt(const uint4* p) {
  u32x4 v = __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(p));
  return make_uint4(v[0], v[1], v[2], v[3]);
}

// WAVE6_DPP_PIPE: 6-op zero-LDS RDNA 3.5 wave32 sum + 2-op quad sum
DEV float wave_sum(float v) {
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

DEV float quad_sum_dpp(float v) {
  asm volatile(
    "v_add_f32_dpp %0, %0, %0 row_xmask:2 row_mask:0xf bank_mask:0xf fi:1\n\t"
    "v_add_f32_dpp %0, %0, %0 row_xmask:1 row_mask:0xf bank_mask:0xf fi:1"
    : "+v"(v));
  return v;
}

// Compute dot(q_u8[0..15], x_f32[0..15]) using gfx1151 single-cycle byte-to-f32 converts
DEV float dot_u8x16_f32(uint4 q, const float x[16]) {
  const uint32_t w0 = q.x, w1 = q.y, w2 = q.z, w3 = q.w;
  float a = 0.f;
  a = fmaf((float)(w0 & 0xFFu),         x[0],  a);
  a = fmaf((float)((w0 >> 8) & 0xFFu),  x[1],  a);
  a = fmaf((float)((w0 >> 16) & 0xFFu), x[2],  a);
  a = fmaf((float)(w0 >> 24),           x[3],  a);
  a = fmaf((float)(w1 & 0xFFu),         x[4],  a);
  a = fmaf((float)((w1 >> 8) & 0xFFu),  x[5],  a);
  a = fmaf((float)((w1 >> 16) & 0xFFu), x[6],  a);
  a = fmaf((float)(w1 >> 24),           x[7],  a);
  a = fmaf((float)(w2 & 0xFFu),         x[8],  a);
  a = fmaf((float)((w2 >> 8) & 0xFFu),  x[9],  a);
  a = fmaf((float)((w2 >> 16) & 0xFFu), x[10], a);
  a = fmaf((float)(w2 >> 24),           x[11], a);
  a = fmaf((float)(w3 & 0xFFu),         x[12], a);
  a = fmaf((float)((w3 >> 8) & 0xFFu),  x[13], a);
  a = fmaf((float)((w3 >> 16) & 0xFFu), x[14], a);
  a = fmaf((float)(w3 >> 24),           x[15], a);
  return a;
}

DEV float ldx16_sum(const uint16_t* __restrict__ p, float x[16]) {
  const uint4 a = *reinterpret_cast<const uint4*>(p);
  const uint4 b = *reinterpret_cast<const uint4*>(p + 8);
  const uint32_t v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float x0 = bf2f(v[i] & 0xFFFFu);
    float x1 = bf2f(v[i] >> 16);
    x[2 * i] = x0;
    x[2 * i + 1] = x1;
    s += x0 + x1;
  }
  return s;
}

// =========================================================================
// Kernel 1: k_down_silu_v2 (and gemv when EPI==0)
// M_TILE = 1..4 rows processed per register tile (outer loop over m0 += MT for M up to 16 -> low VGPRs!)
// Actually: keep M_ACT (1..8) or tile M in chunks of 4 when M > 4?
// Even better: template M_BLK in {1,2,3,4,8} and loop m0 in [0, M) by M_BLK, OR single pass for M<=8 and 2-pass M_BLK=8 for M=9..16!
// Wait: looping inside K-loop means we ONLY READ WEIGHTS `q` ONCE for all M=1..16!!
// Let's check VGPRs: for M<=8, acc[RPW=2][M] = 16 VGPRs, x[16] loaded per m = 16 VGPRs (NOT M*16!!).
// YES!! Loading x[16] INSIDE the m-loop (since x is 20KB*M -> 100% L0/L1/L2 resident!) uses ONLY 16 VGPRs instead of 16*M=128 VGPRs!!
// =========================================================================
template <int M, int RPW, int KS, int EPI>
__global__ __launch_bounds__(KS * 32)
void k_rowgemv_v2(const uint16_t* __restrict__ x, const uint8_t* __restrict__ q,
                  const uint32_t* __restrict__ sz, uint16_t* __restrict__ y,
                  uint16_t* __restrict__ y2, int N, int K, int O, int R, int ldy,
                  int ldy2, float inv_hc) {
  __shared__ float red[KS][RPW][M];
  const int lane = threadIdx.x & 31, wv = threadIdx.x >> 5;
  const int r0 = blockIdx.x * RPW;
  const int NG = (K + 127) >> 7;
  const int chunks = K >> 4;
  const int cps = (chunks + KS - 1) / KS;
  const int c0 = wv * cps, c1 = min(chunks, c0 + cps);

  float acc[RPW][M];
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;

  // WAVE5_CLAUSE_LOADS: hoist row bounds outside K-loop so ALL RPW b128+b32 loads issue in 1 parallel s_clause burst!
  int rows[RPW];
  float rmask[RPW];
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
    const int rr = r0 + r;
    rows[r] = (rr < N) ? rr : (N - 1);
    rmask[r] = (rr < N) ? 1.f : 0.f;
  }

  for (int c = c0 + lane; c < c1; c += 32) {
    uint4 qv[RPW];
    uint32_t zzv[RPW];
    float sv[RPW], nzs[RPW];
    const int g = c >> 3;
    // Pure branchless load burst (allows LLVM s_clause across all RPW rows -> zero mid-burst s_waitcnt!)
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      qv[r] = ldnt(reinterpret_cast<const uint4*>(q + (size_t)rows[r] * K) + c);
      zzv[r] = sz[(size_t)rows[r] * NG + g];
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const float s = h2f(zzv[r] & 0xFFFFu) * rmask[r];
      const float z = (float)((zzv[r] >> 16) & 0xFFu);
      sv[r] = s;
      nzs[r] = -(z * s);
    }
    // Load 16 activations ONE m at a time (L1-hot!) -> ONLY 16 VGPRs for x regardless of M=1..16!!
#pragma unroll 1
    for (int m = 0; m < M; ++m) {
      float xv[16];
      const float xs = ldx16_sum(x + (size_t)m * K + c * 16, xv);
#pragma unroll
      for (int r = 0; r < RPW; ++r) {
        const float qd = dot_u8x16_f32(qv[r], xv);
        acc[r][m] = fmaf(sv[r], qd, fmaf(nzs[r], xs, acc[r][m]));
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float s = wave_sum(acc[r][m]);
      if (lane == 0) red[wv][r][m] = s;
    }
  __syncthreads();
  for (int idx = threadIdx.x; idx < RPW * M; idx += KS * 32) {
    const int r = idx / M, m = idx % M, row = r0 + r;
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < KS; ++k) s += red[k][r][m];
    if (row < O) {
      const uint16_t ybf = (row < N) ? f2bf(s) : 0u;
      if (EPI == 0) {
        y[(size_t)m * ldy + row] = ybf;
      } else {
        if (row < R) {
          const float yb = bf2f(ybf);
          const float t = yb * inv_hc;
          y[(size_t)m * ldy + row] = f2bf(t / (1.f + __expf(-t)));
        }
        y2[(size_t)m * ldy2 + row] = ybf;
      }
    }
  }
}

// =========================================================================
// Kernel 2: k_up_gatemix_v2 (HC=4, KR=320)
// 1 WG = 4 waves (wave s = stream s = 0..3), 128 threads total.
// Each wave processes RW=8 consecutive rows h0..h0+7:
//   - 4 consecutive lanes (lane_in_row = lane & 3) cooperate on 1 row (row_in_w = lane >> 2).
//   - 5 steps x 4 lanes x 16 bytes = 5 x 64B EXACT cache lines = 320 bytes/row (100% coalesced!).
//   - step 0,1 -> group 0; step 2,3 -> group 1; step 4 -> group 2 (compile-time constant!).
//   - lora staged in LDS as bf16 (M x 320 x 2B <= 10 KB even at M=16) + precomputed per-step xsum in LDS (M x 20 x 4B)!
//   - 2-op Wave32 DPP reduction across the 4 lanes -> ZERO LDS partial array -> 16 waves/SIMD!
// =========================================================================
template <int M, int HC, int KR>
__global__ __launch_bounds__(HC * 32)
void k_up_gatemix_v2(const uint16_t* __restrict__ lora, const uint8_t* __restrict__ q,
                     const uint32_t* __restrict__ sz, const uint16_t* __restrict__ xn,
                     uint16_t* __restrict__ out, int H, int ldl, int ldx, int ldo) {
  static_assert(HC == 4 && KR == 320, "compiled for HC=4, KR=320");
  constexpr int RW = 8;
  constexpr int CH = KR / 16;  // 20
  constexpr int NG = 3;        // ceil(320/128) = 3

  // Stage lora as fp32 in LDS for M<=8 (<= 10 KB), or bf16 for M>8 (<= 10 KB)
  __shared__ uint16_t s_lora[M][KR];
  __shared__ float s_xsum[M][CH];
  __shared__ float s_mix[HC][M][RW];

  const int tid = threadIdx.x;
  const int s = tid >> 5;
  const int lane = tid & 31;
  const int row_w = lane >> 2;  // 0..7
  const int col4  = lane & 3;   // 0..3 (which 16B chunk of the 64B cache line)

  // Cooperative load of lora[M][320] + compute s_xsum[M][20] (16-elem chunk sums!)
  for (int idx = tid; idx < M * CH; idx += HC * 32) {
    const int m = idx / CH;
    const int c = idx % CH;
    const uint16_t* lp = lora + (size_t)m * ldl + c * 16;
    const uint4 a = *reinterpret_cast<const uint4*>(lp);
    const uint4 b = *reinterpret_cast<const uint4*>(lp + 8);
    *reinterpret_cast<uint4*>(&s_lora[m][c * 16])     = a;
    *reinterpret_cast<uint4*>(&s_lora[m][c * 16 + 8]) = b;
    const uint32_t v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    float xs = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) xs += bf2f(v[i] & 0xFFFFu) + bf2f(v[i] >> 16);
    s_xsum[m][c] = xs;
  }
  __syncthreads();

  const int h = blockIdx.x * RW + row_w;
  const size_t row = (size_t)s * H + h;
  const uint4* qr = reinterpret_cast<const uint4*>(q + row * KR);

  // Preload only the 3 uint32 scale/zero words for this row (3 VGPRs!)
  const uint32_t sz0 = sz[row * NG + 0];
  const uint32_t sz1 = sz[row * NG + 1];
  const uint32_t sz2 = sz[row * NG + 2];
  const float sc[3]  = { h2f(sz0 & 0xFFFFu), h2f(sz1 & 0xFFFFu), h2f(sz2 & 0xFFFFu) };
  const float nzs[3] = {
    -((float)((sz0 >> 16) & 0xFFu)) * sc[0],
    -((float)((sz1 >> 16) & 0xFFu)) * sc[1],
    -((float)((sz2 >> 16) & 0xFFu)) * sc[2]
  };

  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

  // 5 steps x 64B exact cache lines = 320B; WAVE6 software-pipelined qv_next (+4 VGPRs -> overlaps DRAM with M dots!)
  uint4 qv_cur = ldnt(qr + col4);
#pragma unroll 1
  for (int step = 0; step < 5; ++step) {
    const int c = step * 4 + col4;  // 0..19
    const int g = step >> 1;        // 0, 0, 1, 1, 2
    const uint4 qv = qv_cur;
    if (step < 4) qv_cur = ldnt(qr + c + 4);
    const float sv = sc[g];
    const float nv = nzs[g];
#pragma unroll 1
    for (int m = 0; m < M; ++m) {
      float xv[16];
      const uint4 a = *reinterpret_cast<const uint4*>(&s_lora[m][c * 16]);
      const uint4 b = *reinterpret_cast<const uint4*>(&s_lora[m][c * 16 + 8]);
      const uint32_t v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
      for (int i = 0; i < 8; ++i) { xv[2*i] = bf2f(v[i] & 0xFFFFu); xv[2*i+1] = bf2f(v[i] >> 16); }
      const float qd = dot_u8x16_f32(qv, xv);
      acc[m] = fmaf(sv, qd, fmaf(nv, s_xsum[m][c], acc[m]));
    }
  }

  // 4-lane native zero-LDS DPP reduction -> col4 == 0 holds full 320-dot!
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float a = quad_sum_dpp(acc[m]);
    if (col4 == 0) {
      const float g = bf2f(f2bf(a));
      const float xv = bf2f(xn[(size_t)m * ldx + row]);
      s_mix[s][m][row_w] = xv / (1.f + __expf(-g));
    }
  }
  __syncthreads();

  // Stream reduction (s=0..3) -> out[m, h0..h0+7]
  for (int idx = tid; idx < M * RW; idx += HC * 32) {
    const int m = idx / RW;
    const int r = idx % RW;
    const float sum = (s_mix[0][m][r] + s_mix[1][m][r]) + (s_mix[2][m][r] + s_mix[3][m][r]);
    out[(size_t)m * ldo + blockIdx.x * RW + r] = f2bf(sum * 0.25f);
  }
}

// =========================================================================
// Dispatchers (M = 1..16!)
// =========================================================================
static void check_wire(const at::Tensor& x, const at::Tensor& q, const at::Tensor& sz) {
  TORCH_CHECK(x.is_cuda() && q.is_cuda() && sz.is_cuda(), "hiphc_v2: GPU tensors required");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && q.scalar_type() == at::kByte && sz.scalar_type() == at::kInt);
  TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1 && q.is_contiguous() && sz.is_contiguous());
  TORCH_CHECK(x.size(1) == q.size(1) && q.size(1) % 16 == 0);
}

template <int M, int RPW, int KS, int EPI>
static void launch_rg(const at::Tensor& x, const at::Tensor& q, const at::Tensor& sz,
                      at::Tensor& y, at::Tensor* y2, int O, int R, float inv_hc) {
  const int N = q.size(0), K = q.size(1);
  hipLaunchKernelGGL((k_rowgemv_v2<M, RPW, KS, EPI>), dim3((O + RPW - 1) / RPW), dim3(KS * 32), 0,
                     at::hip::getCurrentHIPStream(),
                     (const uint16_t*)x.data_ptr(), (const uint8_t*)q.data_ptr(), (const uint32_t*)sz.data_ptr(),
                     (uint16_t*)y.data_ptr(), y2 ? (uint16_t*)y2->data_ptr() : nullptr,
                     N, K, O, R, (int)y.stride(0), y2 ? (int)y2->stride(0) : 0, inv_hc);
}

template <int KS, int EPI>
static void dispatch_rg_cfg(int cfg, int M, const at::Tensor& x, const at::Tensor& q, const at::Tensor& sz,
                            at::Tensor& y, at::Tensor* y2, int O, int R, float inv_hc) {
  // v0.2.7 zero-spill RPW routing (verified via gfx1151 resource usage audit):
  //   M=1..4: RPW=4 (cfg==4), RPW=2 (cfg==1,2), RPW=1 (cfg==0) -> 0 spills
  //   M=5,6,8: cap at RPW<=2 -> 0 spills
  //   M=7,9..16: RPW=1 -> 0 spills, 16w/SIMD!
  const int rpw = (cfg == 0) ? 1 : (cfg == 4 ? 4 : 2);
  switch (M) {
    case 1:  if (rpw == 4) launch_rg<1,  4, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc);
             else if (rpw == 2) launch_rg<1,  2, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc);
             else launch_rg<1,  1, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 2:  if (rpw == 4) launch_rg<2,  4, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc);
             else if (rpw == 2) launch_rg<2,  2, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc);
             else launch_rg<2,  1, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 3:  if (rpw == 4) launch_rg<3,  4, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc);
             else if (rpw == 2) launch_rg<3,  2, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc);
             else launch_rg<3,  1, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 4:  if (rpw == 4) launch_rg<4,  4, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc);
             else if (rpw == 2) launch_rg<4,  2, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc);
             else launch_rg<4,  1, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 5:  if (rpw >= 2) launch_rg<5,  2, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc);
             else launch_rg<5,  1, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 6:  if (rpw >= 2) launch_rg<6,  2, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc);
             else launch_rg<6,  1, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 7:  launch_rg<7,  1, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 8:  if (rpw >= 2) launch_rg<8,  2, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc);
             else launch_rg<8,  1, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 9:  launch_rg<9,  1, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 10: launch_rg<10, 1, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 11: launch_rg<11, 1, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 12: launch_rg<12, 1, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 13: launch_rg<13, 1, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 14: launch_rg<14, 1, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 15: launch_rg<15, 1, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    case 16: launch_rg<16, 1, KS, EPI>(x, q, sz, y, y2, O, R, inv_hc); break;
    default: TORCH_CHECK(false, "hiphc_v2: M must be 1..16");
  }
}

at::Tensor gemv(at::Tensor x, at::Tensor q, at::Tensor sz, int64_t O, int64_t cfg) {
  check_wire(x, q, sz);
  auto y = at::empty({x.size(0), O}, x.options());
  dispatch_rg_cfg<4, 0>(cfg, x.size(0), x, q, sz, y, nullptr, O, 0, 1.f);
  return y;
}

std::vector<at::Tensor> down_silu(at::Tensor xn, at::Tensor q, at::Tensor sz, int64_t O, int64_t R, int64_t hc, int64_t cfg) {
  check_wire(xn, q, sz);
  auto lora = at::empty({xn.size(0), R}, xn.options());
  auto d = at::empty({xn.size(0), O}, xn.options());
  dispatch_rg_cfg<4, 1>(cfg, xn.size(0), xn, q, sz, lora, &d, O, R, 1.f / (float)hc);
  return {lora, d};
}

template <int M>
static void launch_ugm2(const at::Tensor& lora, const at::Tensor& q, const at::Tensor& sz,
                        const at::Tensor& xn, at::Tensor& out, int H) {
  hipLaunchKernelGGL((k_up_gatemix_v2<M, 4, 320>), dim3(H / 8), dim3(128), 0,
                     at::hip::getCurrentHIPStream(),
                     (const uint16_t*)lora.data_ptr(), (const uint8_t*)q.data_ptr(),
                     (const uint32_t*)sz.data_ptr(), (const uint16_t*)xn.data_ptr(),
                     (uint16_t*)out.data_ptr(), H, (int)lora.stride(0), (int)xn.stride(0), (int)out.stride(0));
}

at::Tensor up_gatemix(at::Tensor lora, at::Tensor q, at::Tensor sz, at::Tensor xn, int64_t hc, int64_t var) {
  TORCH_CHECK(hc == 4 && q.size(1) == 320);
  check_wire(lora, q, sz);
  const int H = q.size(0) / hc;
  TORCH_CHECK(H % 8 == 0);
  auto out = at::empty({lora.size(0), H}, lora.options());
  switch (lora.size(0)) {
    case 1:  launch_ugm2<1>(lora, q, sz, xn, out, H); break;
    case 2:  launch_ugm2<2>(lora, q, sz, xn, out, H); break;
    case 3:  launch_ugm2<3>(lora, q, sz, xn, out, H); break;
    case 4:  launch_ugm2<4>(lora, q, sz, xn, out, H); break;
    case 5:  launch_ugm2<5>(lora, q, sz, xn, out, H); break;
    case 6:  launch_ugm2<6>(lora, q, sz, xn, out, H); break;
    case 7:  launch_ugm2<7>(lora, q, sz, xn, out, H); break;
    case 8:  launch_ugm2<8>(lora, q, sz, xn, out, H); break;
    case 9:  launch_ugm2<9>(lora, q, sz, xn, out, H); break;
    case 10: launch_ugm2<10>(lora, q, sz, xn, out, H); break;
    case 11: launch_ugm2<11>(lora, q, sz, xn, out, H); break;
    case 12: launch_ugm2<12>(lora, q, sz, xn, out, H); break;
    case 13: launch_ugm2<13>(lora, q, sz, xn, out, H); break;
    case 14: launch_ugm2<14>(lora, q, sz, xn, out, H); break;
    case 15: launch_ugm2<15>(lora, q, sz, xn, out, H); break;
    case 16: launch_ugm2<16>(lora, q, sz, xn, out, H); break;
    default: TORCH_CHECK(false, "hiphc_v2: M must be 1..16");
  }
  return out;
}


// =========================================================================
// Kernel 3: k_hc_combine_norm (v0.2.7 gfx1151 128-bit register-resident + v_dot2_f32_bf16 + v_perm_b32)
// Fast path (hc_dim == 2560): Grid=(M, HC), Block=320 (10 waves), 8 bf16/thr = 1x global_load_b128!
//   - Pass 1: 2x global_load_b128 (res, blk) -> fma -> 4x v_perm_b32 -> 1x global_store_b128 (out)
//   - Sum-of-squares: 4x native v_dot2_f32_bf16 directly on packed bf16x2 words in registers!
//   - Pass 2: ZERO re-read of out[]! 1x global_load_b128 (w) -> 4x v_perm_b32 -> 1x global_store_b128 (y)
//   Total: 5x b128 VMEM ops/thr instead of 60x scalar b16 ops/thr (12x fewer VMEM instructions)!
// =========================================================================
typedef short h2_bf16 __attribute__((ext_vector_type(2)));

DEV uint32_t f32x2_to_bf16x2_rne(float a, float b) {
  uint32_t ua = __float_as_uint(a);
  uint32_t ub = __float_as_uint(b);
  ua = (a != a) ? 0x7fff0000u : (ua + 0x7fffu + ((ua >> 16) & 1u));
  ub = (b != b) ? 0x7fff0000u : (ub + 0x7fffu + ((ub >> 16) & 1u));
  return __builtin_amdgcn_perm(ub, ua, 0x07060302u);
}

DEV float bf16x2_sq_acc(uint32_t pk, float acc) {
  h2_bf16 v = __builtin_bit_cast(h2_bf16, pk);
  return __builtin_amdgcn_fdot2_f32_bf16(v, v, acc, false);
}

__global__ __launch_bounds__(320)
void k_hc_combine_norm_2560(const uint16_t* __restrict__ res,
                            const uint16_t* __restrict__ blk,
                            const uint16_t* __restrict__ inj,
                            const uint16_t* __restrict__ w,
                            uint16_t* __restrict__ out,
                            uint16_t* __restrict__ y,
                            int ldp_res, int ldp_blk, int ldp_inj, int ldp_out, int ldp_y,
                            int hc, bool w_shared, float eps) {
  __shared__ float s_wsum[10];
  const int row = blockIdx.x;
  const int stream = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wv = tid >> 5;

  const float inj_raw = bf2f(inj[(size_t)row * ldp_inj + stream]);
  const float inj_s = 2.0f / (1.0f + __expf(-inj_raw / (float)hc));

  const size_t stream_off = (size_t)stream * 2560;
  const uint4 r4 = *reinterpret_cast<const uint4*>(res + (size_t)row * ldp_res + stream_off + tid * 8);
  const uint4 b4 = *reinterpret_cast<const uint4*>(blk + (size_t)row * ldp_blk + tid * 8);
  const uint4 w4 = *reinterpret_cast<const uint4*>((w_shared ? w : (w + stream_off)) + tid * 8);

  auto comb2 = [&](uint32_t rp, uint32_t bp) -> uint32_t {
    float r0 = __uint_as_float((rp & 0xffffu) << 16);
    float r1 = __uint_as_float(rp & 0xffff0000u);
    float b0 = __uint_as_float((bp & 0xffffu) << 16);
    float b1 = __uint_as_float(bp & 0xffff0000u);
    return f32x2_to_bf16x2_rne(fmaf(b0, inj_s, r0), fmaf(b1, inj_s, r1));
  };

  uint4 o4;
  o4.x = comb2(r4.x, b4.x);
  o4.y = comb2(r4.y, b4.y);
  o4.z = comb2(r4.z, b4.z);
  o4.w = comb2(r4.w, b4.w);

  *reinterpret_cast<uint4*>(out + (size_t)row * ldp_out + stream_off + tid * 8) = o4;

  float sq = 0.0f;
  sq = bf16x2_sq_acc(o4.x, sq);
  sq = bf16x2_sq_acc(o4.y, sq);
  sq = bf16x2_sq_acc(o4.z, sq);
  sq = bf16x2_sq_acc(o4.w, sq);

  const float ws = wave_sum(sq);
  if (lane == 0) s_wsum[wv] = ws;
  __syncthreads();
  const float tot_sq = ((s_wsum[0] + s_wsum[1]) + (s_wsum[2] + s_wsum[3]))
                     + ((s_wsum[4] + s_wsum[5]) + (s_wsum[6] + s_wsum[7]))
                     + (s_wsum[8] + s_wsum[9]);
  const float rrms = rsqrtf(tot_sq * (1.0f / 2560.0f) + eps);

  auto norm2 = [&](uint32_t op, uint32_t wp) -> uint32_t {
    float o0 = __uint_as_float((op & 0xffffu) << 16) * rrms;
    float o1 = __uint_as_float(op & 0xffff0000u) * rrms;
    float w0 = __uint_as_float((wp & 0xffffu) << 16);
    float w1 = __uint_as_float(wp & 0xffff0000u);
    return f32x2_to_bf16x2_rne(fmaf(o0, w0, o0), fmaf(o1, w1, o1));
  };

  uint4 y4;
  y4.x = norm2(o4.x, w4.x);
  y4.y = norm2(o4.y, w4.y);
  y4.z = norm2(o4.z, w4.z);
  y4.w = norm2(o4.w, w4.w);
  *reinterpret_cast<uint4*>(y + (size_t)row * ldp_y + stream_off + tid * 8) = y4;
}

__global__ __launch_bounds__(256)
void k_hc_combine_norm(const uint16_t* __restrict__ res,
                       const uint16_t* __restrict__ blk,
                       const uint16_t* __restrict__ inj,
                       const uint16_t* __restrict__ w,
                       uint16_t* __restrict__ out,
                       uint16_t* __restrict__ y,
                       int ldp_res, int ldp_blk, int ldp_inj, int ldp_out, int ldp_y,
                       int hc_dim, int hc, bool w_shared, float eps) {
  __shared__ float s_wsum[8];
  const int row = blockIdx.x;
  const int stream = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wv = tid >> 5;

  const float inj_raw = bf2f(inj[(size_t)row * ldp_inj + stream]);
  const float inj_s = 2.0f / (1.0f + __expf(-inj_raw / (float)hc));

  const size_t stream_off = (size_t)stream * hc_dim;
  const uint16_t* rp = res + (size_t)row * ldp_res + stream_off;
  const uint16_t* bp = blk + (size_t)row * ldp_blk;
  uint16_t* op = out + (size_t)row * ldp_out + stream_off;
  uint16_t* yp = y + (size_t)row * ldp_y + stream_off;
  const uint16_t* wp = w_shared ? w : (w + stream_off);

  float local_sq = 0.0f;
  for (int i = tid; i < hc_dim; i += 256) {
    const float rv = bf2f(rp[i]);
    const float bv = bf2f(bp[i]);
    const uint16_t ob = f2bf(fmaf(bv, inj_s, rv));
    op[i] = ob;
    const float of = bf2f(ob);
    local_sq = fmaf(of, of, local_sq);
  }
  const float ws = wave_sum(local_sq);
  if (lane == 0) s_wsum[wv] = ws;
  __syncthreads();
  const float tot_sq = ((s_wsum[0] + s_wsum[1]) + (s_wsum[2] + s_wsum[3]))
                     + ((s_wsum[4] + s_wsum[5]) + (s_wsum[6] + s_wsum[7]));
  const float rrms = rsqrtf(tot_sq / (float)hc_dim + eps);

  for (int i = tid; i < hc_dim; i += 256) {
    const float of = bf2f(op[i]);
    const float wf = bf2f(wp[i]);
    const float n = of * rrms;
    yp[i] = f2bf(fmaf(n, wf, n));
  }
}

std::vector<at::Tensor> combine_norm(at::Tensor residual, at::Tensor block_output,
                                     at::Tensor injection_logits, at::Tensor norm_weight,
                                     double eps, int64_t hc) {
  const int M = (int)residual.size(0);
  const int DIM = (int)residual.size(1);
  const int hc_dim = DIM / (int)hc;
  TORCH_CHECK(residual.scalar_type() == at::kBFloat16 && block_output.scalar_type() == at::kBFloat16
              && injection_logits.scalar_type() == at::kBFloat16 && norm_weight.scalar_type() == at::kBFloat16);
  auto out = at::empty_like(residual);
  auto y = at::empty_like(residual);
  const bool w_shared = (norm_weight.numel() == hc_dim);
  const bool aligned16 = (hc_dim == 2560)
      && ((residual.stride(0) & 7) == 0) && ((block_output.stride(0) & 7) == 0)
      && ((out.stride(0) & 7) == 0) && ((y.stride(0) & 7) == 0)
      && (((uintptr_t)residual.data_ptr() & 15) == 0) && (((uintptr_t)block_output.data_ptr() & 15) == 0)
      && (((uintptr_t)norm_weight.data_ptr() & 15) == 0);
  if (aligned16) {
    hipLaunchKernelGGL(k_hc_combine_norm_2560, dim3(M, (int)hc), dim3(320), 0,
                       at::hip::getCurrentHIPStream(),
                       (const uint16_t*)residual.data_ptr(), (const uint16_t*)block_output.data_ptr(),
                       (const uint16_t*)injection_logits.data_ptr(), (const uint16_t*)norm_weight.data_ptr(),
                       (uint16_t*)out.data_ptr(), (uint16_t*)y.data_ptr(),
                       (int)residual.stride(0), (int)block_output.stride(0), (int)injection_logits.stride(0),
                       (int)out.stride(0), (int)y.stride(0),
                       (int)hc, w_shared, (float)eps);
  } else {
    hipLaunchKernelGGL(k_hc_combine_norm, dim3(M, (int)hc), dim3(256), 0,
                       at::hip::getCurrentHIPStream(),
                       (const uint16_t*)residual.data_ptr(), (const uint16_t*)block_output.data_ptr(),
                       (const uint16_t*)injection_logits.data_ptr(), (const uint16_t*)norm_weight.data_ptr(),
                       (uint16_t*)out.data_ptr(), (uint16_t*)y.data_ptr(),
                       (int)residual.stride(0), (int)block_output.stride(0), (int)injection_logits.stride(0),
                       (int)out.stride(0), (int)y.stride(0),
                       hc_dim, (int)hc, w_shared, (float)eps);
  }
  return {out, y};
}

PYBIND11_MODULE(hiphc_v2, m) {
  m.def("gemv", &gemv);
  m.def("down_silu", &down_silu);
  m.def("up_gatemix", &up_gatemix);
  m.def("combine_norm", &combine_norm);
}
