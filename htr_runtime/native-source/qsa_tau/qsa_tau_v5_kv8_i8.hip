// qsa_tau_v5_kv8_i8.hip — Full-INT8 RDNA 3.5 V_WMMA_I32_16X16X16_IU8 QSA sparse prefill kernel (KV8)!
// - Q quantized ONCE per 128-dim half to symmetric int8 (halves qf VGPRs: 64 -> 32)
// - K_i8 loaded raw (16 bytes/load u32x4, ZERO dequant!) -> v_wmma_i32_16x16x16_iu8 (signed x signed)
// - P * sc_v quantized per-(head,tile) to unsigned uint8 (0..255), V_i8 byte-transposed raw (ZERO dequant!)
//   -> v_wmma_i32_16x16x16_iu8 (signed x unsigned)
// - ~15 KB LDS, 0 spills, 2x WMMA throughput!
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

#define HD 256
#define GH 12
#define BN 32
#define NTHR 128
#define K8PITCH 272   // 256 + 16 B (16B-aligned, bank-conflict-free)
#define V8PITCH 48    // >= 32 B, 16B-aligned -> 12,288 B total for 256 rows
#define NTILES 65
#define TOPK 2051

__device__ __forceinline__ float bf16_lo(uint32_t u) { return __uint_as_float((u & 0xffffu) << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ uint32_t bf16_rne(float x) {
  uint32_t u = __builtin_bit_cast(uint32_t, x);
  if (x != x) return 0x7fffu;
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}
__device__ __forceinline__ uint32_t pack4_i8(float a, float b, float c, float d) {
  uint32_t i0 = (uint32_t)(int8_t)__float2int_rn(a) & 0xffu;
  uint32_t i1 = (uint32_t)(int8_t)__float2int_rn(b) & 0xffu;
  uint32_t i2 = (uint32_t)(int8_t)__float2int_rn(c) & 0xffu;
  uint32_t i3 = (uint32_t)(int8_t)__float2int_rn(d) & 0xffu;
  return i0 | (i1 << 8) | (i2 << 16) | (i3 << 24);
}
__device__ __forceinline__ int rfl(int x) { return __builtin_amdgcn_readfirstlane(x); }

extern "C" __global__ void __launch_bounds__(NTHR, 6) qsa_tau_v5_kv8_i8(
    const uint16_t* __restrict__ Q, const int8_t* __restrict__ K, const int8_t* __restrict__ V,
    const int32_t* __restrict__ IDX, const int32_t* __restrict__ TAB, const int32_t* __restrict__ T2R,
    uint16_t* __restrict__ O, int32_t* __restrict__ STATS, int32_t* __restrict__ STATS2,
    int64_t SQ0, int64_t SQ1, int64_t SK0, int64_t SK1, int64_t SK2, int64_t SV0, int64_t SV1, int64_t SV2,
    int64_t SO0, int64_t SO1, int64_t SIDX, int64_t STAB, int NR, int NP, int PAGE, int TW, int NREQ, int SHARE) {

  // Shared memory (~15.5 KB total!)
  // sBuf holds sK_i8[32][272] (8,704 B) during Q@K^T, then sVt_i8[256][48] (12,288 B) during P@V!
  __shared__ __attribute__((aligned(16))) int8_t sBuf[HD * V8PITCH];
  __shared__ float sS[2][16][17];
  __shared__ __attribute__((aligned(16))) uint8_t sP_u8[16 * 32];
  __shared__ int64_t sOffK[BN];
  __shared__ int64_t sOffV[BN];
  __shared__ float sScK[BN];
  __shared__ float sScV[BN];
  __shared__ float sMax[2][16];
  __shared__ float sSum[2][2][16];
  __shared__ float sPmax[2][16];
  __shared__ float sScP[16];
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
  const float scale = __builtin_bit_cast(float, 0x3db8aa3bu);  // (1/sqrt(256)) * log2(e)
  const float NEG = -1.0e20f;

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

  // Load Q[grow, head, wd*128 .. +127] ONCE and quantize per-(head, half) to symmetric int8!
  i32x4 qf[8];
  float sc_q = 0.0f;
  {
    const uint16_t* qp = Q + (int64_t)grow * SQ0 + (int64_t)head * SQ1 + wd * 128;
    float amax = 0.0f;
    if (active) {
#pragma unroll 1
      for (int k = 0; k < 8; ++k) {
        u32x4 a = *(const u32x4*)(qp + k * 16);
        u32x4 b = *(const u32x4*)(qp + k * 16 + 8);
        amax = fmaxf(amax, fmaxf(fabsf(bf16_lo(a.x)), fabsf(bf16_hi(a.x))));
        amax = fmaxf(amax, fmaxf(fabsf(bf16_lo(a.y)), fabsf(bf16_hi(a.y))));
        amax = fmaxf(amax, fmaxf(fabsf(bf16_lo(a.z)), fabsf(bf16_hi(a.z))));
        amax = fmaxf(amax, fmaxf(fabsf(bf16_lo(a.w)), fabsf(bf16_hi(a.w))));
        amax = fmaxf(amax, fmaxf(fabsf(bf16_lo(b.x)), fabsf(bf16_hi(b.x))));
        amax = fmaxf(amax, fmaxf(fabsf(bf16_lo(b.y)), fabsf(bf16_hi(b.y))));
        amax = fmaxf(amax, fmaxf(fabsf(bf16_lo(b.z)), fabsf(bf16_hi(b.z))));
        amax = fmaxf(amax, fmaxf(fabsf(bf16_lo(b.w)), fabsf(bf16_hi(b.w))));
      }
      sc_q = (amax > 0.0f) ? (amax * (1.0f / 127.0f)) : 0.0f;
      float inv_q = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        u32x4 a = *(const u32x4*)(qp + k * 16);
        u32x4 b = *(const u32x4*)(qp + k * 16 + 8);
        uint32_t p0 = pack4_i8(bf16_lo(a.x)*inv_q, bf16_hi(a.x)*inv_q, bf16_lo(a.y)*inv_q, bf16_hi(a.y)*inv_q);
        uint32_t p1 = pack4_i8(bf16_lo(a.z)*inv_q, bf16_hi(a.z)*inv_q, bf16_lo(a.w)*inv_q, bf16_hi(a.w)*inv_q);
        uint32_t p2 = pack4_i8(bf16_lo(b.x)*inv_q, bf16_hi(b.x)*inv_q, bf16_lo(b.y)*inv_q, bf16_hi(b.y)*inv_q);
        uint32_t p3 = pack4_i8(bf16_lo(b.z)*inv_q, bf16_hi(b.z)*inv_q, bf16_lo(b.w)*inv_q, bf16_hi(b.w)*inv_q);
        qf[k] = (i32x4){(int)p0, (int)p1, (int)p2, (int)p3};
      }
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) qf[k] = (i32x4){0, 0, 0, 0};
    }
  }
  __syncthreads();

  const int ptiles = (rfl(sLast) + BN) / BN;
  const int req = T2R[grow];
  const bool req_ok = req >= 0 && req < NREQ;
  const int32_t* tab_row = req_ok ? (TAB + (int64_t)req * STAB) : nullptr;
  const int32_t* idx_row = IDX + (int64_t)grow * SIDX;
  const float q_scale = sc_q * scale;

  float maximum = NEG, norm = 0.0f;
  f32x8 acc[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) acc[t] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  for (int ct = 0; ct <= ptiles; ++ct) {
    const int tile = ct < ptiles ? ct : NTILES - 1;
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
      sScK[tid] = (ok >= 0) ? *(const float*)(K + ok + HD) : 0.0f;
      sScV[tid] = (ov >= 0) ? *(const float*)(V + ov + HD) : 0.0f;
      const uint32_t bal = __builtin_amdgcn_ballot_w32(ok >= 0);
      if (tid == 0) sValid = bal;
    }
    __syncthreads();

    const uint32_t vm = (uint32_t)rfl((int)sValid);
    if (vm == 0u) { __syncthreads(); continue; }

    // 1. Raw INT8 K load: 32 tokens x 16 chunks of 16 bytes = 512 u32x4 loads (4 per thread, ZERO dequant!)
#pragma unroll
    for (int it = tid; it < BN * 16; it += NTHR) {
      const int tok = it >> 4, c16 = it & 15;
      const int64_t ok = sOffK[tok];
      u32x4 v = (ok >= 0) ? *(const u32x4*)(K + ok + c16 * 16) : (u32x4){0, 0, 0, 0};
      *(u32x4*)(sBuf + tok * K8PITCH + c16 * 16) = v;
    }
    __syncthreads();

    // 2. INT8 WMMA: S_i32 = K_i8 @ Q_i8^T (2x throughput, half LDS loads!)
    i32x8 Si = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
    {
      const int tok = wt * 16 + lr;
      const int kbase = wd * 128;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const i32x4 kf = *(const i32x4*)(sBuf + tok * K8PITCH + kbase + k * 16);
        Si = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, kf, true, qf[k], Si, false);
      }
    }
    if (wd == 1) {
#pragma unroll
      for (int e = 0; e < 8; ++e) sS[wt][2 * e + hb][lr] = (float)Si[e] * q_scale;
    }
    __syncthreads();

    // 3. Raw INT8 V load + byte transpose into sBuf as sVt_i8[256][48] (ZERO dequant!)
    // 128 threads: each thread handles 1 (c16 in 0..15, tq in 0..7) = 16 dims x 4 tokens!
    {
      const int c16 = tid & 15;
      const int tq = tid >> 4;
      u32x4 v[4];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int64_t ov = sOffV[tq * 4 + t];
        v[t] = (ov >= 0) ? *(const u32x4*)(V + ov + c16 * 16) : (u32x4){0, 0, 0, 0};
      }
      // Transpose 4 tokens x 16 bytes -> 16 dims x 4 bytes (uint32_t) via v_perm_b32!
#pragma unroll 1
      for (int w = 0; w < 4; ++w) {
        const uint32_t r0 = v[0][w], r1 = v[1][w], r2 = v[2][w], r3 = v[3][w];
        // Pair (r0, r1) and (r2, r3) for bytes 0..3
        const uint32_t t01_lo = __builtin_amdgcn_perm(r1, r0, 0x05010400u);  // [b0(r0), b0(r1), b1(r0), b1(r1)]
        const uint32_t t23_lo = __builtin_amdgcn_perm(r3, r2, 0x05010400u);  // [b0(r2), b0(r3), b1(r2), b1(r3)]
        const uint32_t t01_hi = __builtin_amdgcn_perm(r1, r0, 0x07030602u);  // [b2(r0), b2(r1), b3(r0), b3(r1)]
        const uint32_t t23_hi = __builtin_amdgcn_perm(r3, r2, 0x07030602u);  // [b2(r2), b2(r3), b3(r2), b3(r3)]
        const uint32_t d0 = __builtin_amdgcn_perm(t23_lo, t01_lo, 0x05040100u);
        const uint32_t d1 = __builtin_amdgcn_perm(t23_lo, t01_lo, 0x07060302u);
        const uint32_t d2 = __builtin_amdgcn_perm(t23_hi, t01_hi, 0x05040100u);
        const uint32_t d3 = __builtin_amdgcn_perm(t23_hi, t01_hi, 0x07060302u);
        const int db = c16 * 16 + w * 4;
        *(uint32_t*)(sBuf + (db + 0) * V8PITCH + tq * 4) = d0;
        *(uint32_t*)(sBuf + (db + 1) * V8PITCH + tq * 4) = d1;
        *(uint32_t*)(sBuf + (db + 2) * V8PITCH + tq * 4) = d2;
        *(uint32_t*)(sBuf + (db + 3) * V8PITCH + tq * 4) = d3;
      }
    }

    // 4. Online softmax + fold sScV[tok] into p[e] -> per-(head,tile) uint8 quantization!
    float s[8];
    bool use[8];
    if (wd == 0) {
      const uint32_t myv = (vm >> (wt * 16)) & 0xffffu;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int tok = wt * 16 + 2 * e + hb;
        use[e] = active && (((myv >> (2 * e + hb)) & 1u) != 0u);
        const float sc = ((float)Si[e] * q_scale + sS[wt][2 * e + hb][lr]) * sScK[tok];
        s[e] = use[e] ? sc : NEG;
      }
      float m = fmaxf(fmaxf(fmaxf(s[0], s[1]), fmaxf(s[2], s[3])), fmaxf(fmaxf(s[4], s[5]), fmaxf(s[6], s[7])));
      m = fmaxf(m, __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, m), __builtin_bit_cast(int, m),
                                                                            0x76543210, 0xfedcba98, false, false)));
      if (hb == 0) sMax[wt][lr] = m;
    }
    __syncthreads();

    const float newmax = fmaxf(maximum, fmaxf(sMax[0][lr], sMax[1][lr]));
    const float alpha = __builtin_amdgcn_exp2f(maximum - newmax);
    float pv[8];
    if (wd == 0) {
      float p[8];
      float pm = 0.0f;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int tok = wt * 16 + 2 * e + hb;
        p[e] = use[e] ? __builtin_amdgcn_exp2f(s[e] - newmax) : 0.0f;
        pv[e] = p[e] * sScV[tok];
        pm = fmaxf(pm, pv[e]);
      }
      sSum[wt][hb][lr] = ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]));
      pm = fmaxf(pm, __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, pm), __builtin_bit_cast(int, pm),
                                                                            0x76543210, 0xfedcba98, false, false)));
      if (hb == 0) sPmax[wt][lr] = pm;
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) acc[t] = acc[t] * alpha;
    __syncthreads();

    if (wd == 0) {
      const float pmax = fmaxf(sPmax[0][lr], sPmax[1][lr]);
      const float inv_p = (pmax > 0.0f) ? (255.0f / pmax) : 0.0f;
      if (wt == 0 && hb == 0) sScP[lr] = (pmax > 0.0f) ? (pmax * (1.0f / 255.0f)) : 0.0f;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        sP_u8[lr * 32 + wt * 16 + 2 * e + hb] = (uint8_t)__float2uint_rn(pv[e] * inv_p);
      }
    }
    __syncthreads();

    const float R0 = sSum[0][0][lr] + sSum[1][0][lr];
    const float R1 = sSum[0][1][lr] + sSum[1][1][lr];
    const float rowsum = hb == 0 ? (R0 + R1) : (R1 + R0);
    norm = __builtin_fmaf(norm, alpha, rowsum);
    maximum = newmax;

    // 5. INT8 WMMA: acc += (V_i8 @ P_u8) * sc_p[lr] (signed x unsigned!)
    const float scp = sScP[lr];
    if (__any(scp > 0.0f)) {
      const i32x4 pf0 = *(const i32x4*)(sP_u8 + lr * 32);
      const i32x4 pf1 = *(const i32x4*)(sP_u8 + lr * 32 + 16);
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int d = (wave * 4 + t) * 16 + lr;
        const i32x4 vf0 = *(const i32x4*)(sBuf + d * V8PITCH);
        const i32x4 vf1 = *(const i32x4*)(sBuf + d * V8PITCH + 16);
        i32x8 ti = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
        ti = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, vf0, false, pf0, ti, false);
        ti = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, vf1, false, pf1, ti, false);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          acc[t][e] = __builtin_fmaf((float)ti[e], scp, acc[t][e]);
        }
      }
    }
    __syncthreads();
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
