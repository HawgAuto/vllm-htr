// qsa_tau_v4.hip — HTR v0.2.6 high-occupancy QSA sparse prefill kernel for gfx1151 (BF16 KV).
// Design vs qsa_tau_v2:
//  * QG = 1 (1 query x 12 Q-heads = 1 WMMA 16x16 N-tile): ZERO compatibility-group multi-pass,
//    ZERO V-masking, ZERO 2-chain PV split, 100% wave utilization on shifted/sparse indices.
//  * BN = 32, NTHR = 128 (4 Wave32 waves), split-D=2 for QK:
//    - Each wave holds only qf[8] (64 VGPRs) + acc[4] (32 VGPRs) -> ZERO register spills!
//    - LDS = ~21 KB -> 3 WGs/CU (12 waves/CU, 24 waves/WGP) vs 1 WG/CU in v2.
//  * V global load + transpose into sVt overlapped with QK softmax max reduction.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));

#define HD 256
#define GH 12
#define BN 32
#define NTHR 128
#define KPITCH (HD + 8)
#define PPITCH 40
#define NTILES 65
#define TOPK 2051

__device__ __forceinline__ bf16x16 as_bf16x16(u32x8 v) { return __builtin_bit_cast(bf16x16, v); }

__device__ __forceinline__ uint32_t bf16_rne(float x) {
  uint32_t u = __builtin_bit_cast(uint32_t, x);
  if (x != x) return 0x7fffu;
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}

__device__ __forceinline__ int k_index(int tok, int pc) { return tok * KPITCH + (pc << 3); }
__device__ __forceinline__ int vt_index32(int d, int tok) {
  int sw = (d >> 1) & 3;
  return d * BN + ((((tok >> 3) ^ sw)) << 3) + (tok & 7);
}
__device__ __forceinline__ int rfl(int x) { return __builtin_amdgcn_readfirstlane(x); }

extern "C" __global__ void __launch_bounds__(NTHR, 6) qsa_tau_v4(
    const uint16_t* __restrict__ Q, const uint16_t* __restrict__ K, const uint16_t* __restrict__ V,
    const int32_t* __restrict__ IDX, const int32_t* __restrict__ TAB, const int32_t* __restrict__ T2R,
    uint16_t* __restrict__ O, int32_t* __restrict__ STATS, int32_t* __restrict__ STATS2,
    int64_t SQ0, int64_t SQ1, int64_t SK0, int64_t SK1, int64_t SK2, int64_t SV0, int64_t SV1, int64_t SV2,
    int64_t SO0, int64_t SO1, int64_t SIDX, int64_t STAB, int NR, int NP, int PAGE, int TW, int NREQ, int SHARE) {
  __shared__ __attribute__((aligned(16))) uint16_t sVt[BN * KPITCH];  // 16,896 B (shared by K and V^T)
  __shared__ float sS[2][16][17];                                      // 2,176 B (wd==1 QK partial)
  __shared__ __attribute__((aligned(16))) uint16_t sP[16 * PPITCH];    // 1,280 B (16 rows x 32 cols)
  __shared__ int64_t sOffK[BN];                                        // 256 B
  __shared__ int64_t sOffV[BN];                                        // 256 B
  __shared__ float sMax[2][16];                                        // 128 B
  __shared__ float sSum[2][2][16];                                     // 256 B
  __shared__ uint32_t sValid;
  __shared__ int sLast;

  const int grow = blockIdx.x, kvhead = blockIdx.y;
  if (grow >= NR) return;

  const int tid = threadIdx.x;
  const int wave = rfl(tid >> 5);
  const int lane = tid & 31, lr = lane & 15, hb = lane >> 4;
  const int wt = wave >> 1, wd = wave & 1;
  const bool active = lr < GH;
  const int head = kvhead * GH + lr;
  const float scale = __builtin_bit_cast(float, 0x3db8aa3bu);  // 256^-0.5 * log2(e)
  const float NEG = -1.0e20f;

  // Prefix bound scan (columns 0..2047)
  if (tid == 0) sLast = -1;
  __syncthreads();
  {
    int last = -1;
    const int32_t* idx_row = IDX + (int64_t)grow * SIDX;
#pragma unroll 4
    for (int c = tid; c < (NTILES - 1) * BN; c += NTHR) {
      if (idx_row[c] >= 0) last = c;
    }
    if (last >= 0) atomicMax(&sLast, last);
  }

  // Load half of Q (128 dims = 8 x 16) per wave -> only 64 VGPRs!
  u32x8 qf[8];
  {
    const uint16_t* qp = Q + (int64_t)grow * SQ0 + (int64_t)head * SQ1 + wd * 128;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      u32x4 a = {0, 0, 0, 0}, b = {0, 0, 0, 0};
      if (active) {
        a = *(const u32x4*)(qp + k * 16);
        b = *(const u32x4*)(qp + k * 16 + 8);
      }
      qf[k] = (u32x8){a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    }
  }
  __syncthreads();

  const int ptiles = (rfl(sLast) + BN) / BN;
  const int req = T2R[grow];
  const bool req_ok = req >= 0 && req < NREQ;
  const int32_t* tab_row = req_ok ? (TAB + (int64_t)req * STAB) : nullptr;
  const int32_t* idx_row = IDX + (int64_t)grow * SIDX;

  float maximum = NEG, norm = 0.0f;
  f32x8 acc[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) acc[t] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  for (int ct = 0; ct <= ptiles; ++ct) {
    const int tile = ct < ptiles ? ct : NTILES - 1;
    // (1) Compute 32 token offsets in wave 0 (PV did not touch sOffK/sOffV)
    if (tid < BN) {
      const int col = tile * BN + tid;
      int64_t ok = -1, ov = -1;
      if (col < TOPK && req_ok) {
        const int logical = idx_row[col];
        if (logical >= 0) {
          const int pg = logical / PAGE;
          const int off = logical - pg * PAGE;
          if (pg < TW) {
            const int physical = tab_row[pg];
            if (physical >= 0 && physical < NP) {
              ok = (int64_t)physical * SK0 + (int64_t)off * SK1 + (int64_t)kvhead * SK2;
              ov = (int64_t)physical * SV0 + (int64_t)off * SV1 + (int64_t)kvhead * SV2;
            }
          }
        }
      }
      sOffK[tid] = ok;
      sOffV[tid] = ov;
      const uint32_t bal = __builtin_amdgcn_ballot_w32(ok >= 0);
      if (tid == 0) sValid = bal;
    }
    __syncthreads();  // protects sOffK/sOffV AND prior tile's sVt reads

    const uint32_t vm = (uint32_t)rfl((int)sValid);
    if (vm == 0u) { __syncthreads(); continue; }

    // (2) Stage K [32 x 256] BF16 -> sVt (8 u32x4 loads per thread)
#pragma unroll 2
    for (int it = tid; it < BN * 32; it += NTHR) {
      const int tok = it >> 5, pc = it & 31;
      const int64_t o = sOffK[tok];
      const u32x4 kv = o >= 0 ? *(const u32x4*)(K + o + pc * 8) : (u32x4){0, 0, 0, 0};
      *(u32x4*)(sVt + k_index(tok, pc)) = kv;
    }
    __syncthreads();

    // (3) QK WMMA: split-D=2 across wd=0 (d=0..127) and wd=1 (d=128..255)
    f32x8 S = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};
    {
      const int tok = wt * 16 + lr;
      const int kbase = wd * 16;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const u32x4 a = *(const u32x4*)(sVt + k_index(tok, kbase + 2 * k));
        const u32x4 b = *(const u32x4*)(sVt + k_index(tok, kbase + 2 * k + 1));
        const u32x8 kf = (u32x8){a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        S = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(as_bf16x16(kf), as_bf16x16(qf[k]), S);
      }
    }
    if (wd == 1) {
#pragma unroll
      for (int e = 0; e < 8; ++e) sS[wt][2 * e + hb][lr] = S[e];
    }
    __syncthreads();  // sS ready, sVt is now FREE for V^T!

    // (4) OVERLAPPED: Stage V^T into sVt (all 4 waves) + Softmax rowmax (wd == 0)
    // Load & transpose V [32 x 256] -> sVt [256 x 32] (256 tiles / 128 threads = 2 iters)
#pragma unroll 1
    for (int it = tid; it < 256; it += NTHR) {
      const int c8 = it & 31, tq = it >> 5;
      u32x4 v[4];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int64_t o = sOffV[tq * 4 + t];
        v[t] = o >= 0 ? *(const u32x4*)(V + o + c8 * 8) : (u32x4){0, 0, 0, 0};
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint32_t sel2 = (j & 1) ? 0x07060302u : 0x05040100u;
        const uint32_t w01 = __builtin_amdgcn_perm(v[1][j >> 1], v[0][j >> 1], sel2);
        const uint32_t w23 = __builtin_amdgcn_perm(v[3][j >> 1], v[2][j >> 1], sel2);
        *(uint2*)(sVt + vt_index32(c8 * 8 + j, tq * 4)) = make_uint2(w01, w23);
      }
    }

    float s[8];
    bool use[8];
    if (wd == 0) {
      const uint32_t myv = (vm >> (wt * 16)) & 0xffffu;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        use[e] = active && (((myv >> (2 * e + hb)) & 1u) != 0u);
        const float sc = (S[e] + sS[wt][2 * e + hb][lr]) * scale;
        s[e] = use[e] ? sc : NEG;
      }
      float m = fmaxf(fmaxf(fmaxf(s[0], s[1]), fmaxf(s[2], s[3])), fmaxf(fmaxf(s[4], s[5]), fmaxf(s[6], s[7])));
      m = fmaxf(m, __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, m), __builtin_bit_cast(int, m),
                                                                            0x76543210, 0xfedcba98, false, false)));
      if (hb == 0) sMax[wt][lr] = m;
    }
    __syncthreads();  // sMax AND sVt(V^T) are both ready!

    // (5) Online softmax update + write sP (wd == 0) + rescale acc (all waves)
    const float newmax = fmaxf(maximum, fmaxf(sMax[0][lr], sMax[1][lr]));
    const float alpha = __builtin_amdgcn_exp2f(maximum - newmax);
    if (wd == 0) {
      float p[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) p[e] = use[e] ? __builtin_amdgcn_exp2f(s[e] - newmax) : 0.0f;
      sSum[wt][hb][lr] = ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]));
#pragma unroll
      for (int e = 0; e < 8; ++e) sP[lr * PPITCH + wt * 16 + 2 * e + hb] = (uint16_t)bf16_rne(p[e]);
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) acc[t] = acc[t] * alpha;
    __syncthreads();  // sP and sSum ready!

    // (6) PV WMMA: each wave owns 64 output dims (4 x 16), 2 token chunks (BN=32)
    const float R0 = sSum[0][0][lr] + sSum[1][0][lr];
    const float R1 = sSum[0][1][lr] + sSum[1][1][lr];
    const float rowsum = hb == 0 ? (R0 + R1) : (R1 + R0);
    norm = __builtin_fmaf(norm, alpha, rowsum);
    maximum = newmax;

#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const u32x4 pa = *(const u32x4*)(sP + lr * PPITCH + c * 16);
      const u32x4 pb = *(const u32x4*)(sP + lr * PPITCH + c * 16 + 8);
      const u32x8 pf = (u32x8){pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w};
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int d = (wave * 4 + t) * 16 + lr;
        const u32x4 a = *(const u32x4*)(sVt + vt_index32(d, c * 16));
        const u32x4 b = *(const u32x4*)(sVt + vt_index32(d, c * 16 + 8));
        const u32x8 vf = (u32x8){a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(as_bf16x16(vf), as_bf16x16(pf), acc[t]);
      }
    }
  }

  if (active) {
    uint16_t* op = O + (int64_t)grow * SO0 + (int64_t)head * SO1;
    const float den = fmaxf(norm, 1.0e-20f);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float val = norm > 0.0f ? acc[t][e] / den : 0.0f;
        op[(wave * 4 + t) * 16 + 2 * e + hb] = (uint16_t)bf16_rne(val);
      }
    }
  }
}
