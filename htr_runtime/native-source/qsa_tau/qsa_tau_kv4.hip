// EXPERIMENTAL rotated INT4 derivative: parent bitwise qualification DOES NOT APPLY.
// Native GPU numerics, cache admission and model integration remain pending.
// qsa_tau_v2.hip — E159 Candidate B native v2: tile-aligned union with COMPATIBILITY GROUPS + per-slot V masking.
// Parent: hip/qsa_tau_v1.hip (G1D: bitwise 47/47 vs release split-K). Per-tile arithmetic contract unchanged (below).
//
// v1 ran one full K/V staging + score + PV pass per query slot whenever the 4 slots' column tiles were not identical.
// v2 per tile:
//  * Pairwise slot compatibility: slots a, b are incompatible iff some column is valid for both with different K or V
//    address. Slots with any valid column are greedily partitioned (slot order 0..3) into groups of mutually
//    compatible slots; one pass per group (prefix captures: 1 group; sparse block-boundary programs: typically 2).
//  * Group pass: K and V staged ONCE from the group's union addresses (first valid member per column; members agree).
//    Scores use the union K; each row masks its own invalid columns to -1e20 / p = 0 exactly as split-K (score values
//    at masked columns never reach max, p, or the row sum). Rows outside the group: use = false (alpha 1, p 0, no-op).
//  * PV: the V^T fragment is the WMMA A operand shared by the 16 rows (N) of a row-tile, and gfx1151 WMMA results
//    change when a row's p = 0 meets a REAL V value where split-K has V = 0 (E159 G1B). So V is masked in registers to
//    a slot's own validity (bf16 lanes ANDed to 0). A row-tile spans exactly 2 slots (qa, qb). If both are in the group
//    with identical validity -> one chain masked to it; with different validity -> two chains (qa then qb), each masked
//    to its slot and with the other slot's P zeroed (all-zero-P rows are an exact no-op, E159 G1B/G1D); one slot in
//    the group -> one chain masked to it.
//  * Wave index and V-mask words are readfirstlane'd (wave-uniform in fact; scalarizes control, no arithmetic change).
//  * Chain loop starts with an empty asm memory clobber (compiler-only): without it LLVM keeps all 16 V^T fragments
//    live across the 2 chains (289 VGPR spills); with it 32 spills. No ISA-level arithmetic change.
//  * SHARE: 0 singleton groups (= v1 noshare), 1 one group only if all slots identical (= v1 strict),
//    2 compatibility groups (v2), 3 NEGATIVE CONTROL: v2 groups with no V mask and no chain split (loose; must fail).
//
// FAITHFUL-PORT CONTRACT (from qsa_native_v1.hip, derived from Triton's production AMDGCN for _attend):
//  * scores tile (16 tok x 16 rows) = 16 chained v_wmma_f32_16x16x16_bf16 over d-chunks 0..15 from C=0,
//    srcA = keys fragment (M = token), srcB = query fragment (N = row)  [Triton #mma isTranspose=true].
//  * s = acc * scale (fp32 mul, scale = 256^-0.5 * log2(e)); s = use ? s : -1e20.
//  * rowmax exact (any order). alpha = v_exp_f32(maximum - newmax). p = use ? v_exp_f32(s - newmax) : 0.
//  * rowsum: per lane-half a balanced binary tree over 32 leaves ordered (token-tile n = 0..3, vgpr e = 0..7):
//      T_n = ((e0+e1)+(e2+e3)) + ((e4+e5)+(e6+e7));  R_half = (T0+T1) + (T2+T3);  rowsum = R_0 + R_1.
//  * norm = fma(norm, alpha, rowsum).
//  * acc = WMMA chain over token-chunks 0..3 from C = acc * alpha, srcA = V^T fragment (M = d), srcB = P^T (N = row),
//    p converted to bf16 with RNE.
//  * out = norm > 0 ? acc / max(norm, 1e-20) : 0  (IEEE division, correctly rounded), bf16 RNE.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));

#define HD 256
#define GH 12
#define QG 4
#define ROWS 48
#define BN 64
#define NTHR 384
#define PPITCH 72
#define NTILES 33
#define TOPK 2051

__device__ __forceinline__ bf16x16 as_bf16x16(u32x8 v) { return __builtin_bit_cast(bf16x16, v); }

__device__ __forceinline__ uint32_t bf16_rne(float x) {
  uint32_t u = __builtin_bit_cast(uint32_t, x);
  if (x != x) return 0x7fffu;
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}

// KV4 addition: offsets are rotated INT4 elements/bytes. Each vector has an inline
// FP32 scale/zero-point metadata at +HD/2. Reconstruct the same eight BF16 LDS operands.
__device__ __forceinline__ u32x4 load_kv4_bf16(const uint8_t* base, int64_t off, int first) {
  u32x4 out = {0,0,0,0};
  if (off < 0) return out;
  const uint32_t meta = *(const uint32_t*)(base + off + HD/2);
  const float scale = __builtin_bit_cast(float, meta & 0xfffffff0u);
  const int zp = (int)(meta & 15u);
#pragma unroll
  for (int i=0; i<4; ++i) {
    const uint8_t packed = base[off + first/2 + i];
    const uint32_t a = bf16_rne((float)((int)(packed & 15u) - zp) * scale);
    const uint32_t b = bf16_rne((float)((int)(packed >> 4) - zp) * scale);
    out[i] = a | (b << 16);
  }
  return out;
}

#define KPITCH (HD + 8)
__device__ __forceinline__ int k_index(int tok, int pc) { return tok * KPITCH + (pc << 3); }
__device__ __forceinline__ int vt_index(int d, int tok) {
  int sw = (d >> 1) & 7;
  return d * BN + ((((tok >> 3) ^ sw)) << 3) + (tok & 7);
}
__device__ __forceinline__ int rfl(int x) { return __builtin_amdgcn_readfirstlane(x); }

extern "C" __global__ void __launch_bounds__(NTHR) __attribute__((amdgpu_waves_per_eu(6, 6))) qsa_tau_kv4(
    const uint16_t* __restrict__ Q, const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
    const int32_t* __restrict__ IDX, const int32_t* __restrict__ TAB, const int32_t* __restrict__ T2R,
    uint16_t* __restrict__ O, int32_t* __restrict__ STATS, int32_t* __restrict__ STATS2,
    int64_t SQ0, int64_t SQ1, int64_t SK0, int64_t SK1, int64_t SK2, int64_t SV0, int64_t SV1, int64_t SV2,
    int64_t SO0, int64_t SO1, int64_t SIDX, int64_t STAB, int NR, int NP, int PAGE, int TW, int NREQ, int SHARE) {
  __shared__ __attribute__((aligned(16))) uint16_t sVt[BN * (HD + 8)];
  __shared__ __attribute__((aligned(16))) uint16_t sP[ROWS * PPITCH];
  __shared__ int64_t sOffK[QG][BN];
  __shared__ int64_t sOffV[QG][BN];
  __shared__ int64_t sOffKG[BN];
  __shared__ int64_t sOffVG[BN];
  __shared__ uint32_t sValid[QG][2];   // per slot: bit c of word c>>5 = column c valid (split-K rule)
  __shared__ float sMax[4][ROWS];
  __shared__ float sSum[4][2][ROWS];
  __shared__ int sLast[QG];
  __shared__ int sAny[2];
  __shared__ int sMixed[2];
  __shared__ int sInc[2];              // 4x4 incompatibility matrix, bit a*4+b

  const int group = blockIdx.x, kvhead = blockIdx.y;
  const int tid = threadIdx.x, wave = __builtin_amdgcn_readfirstlane(tid >> 5), lane = tid & 31, lr = lane & 15, hb = lane >> 4;
  const int rt = wave >> 2, dq = wave & 3;
  const int rr = rt * 16 + lr;
  const int qslot = rr / GH, hh = rr - qslot * GH;
  const int grow = group * QG + qslot;
  const bool active = grow < NR;
  const int head = kvhead * GH + hh;
  const int qa = (rt * 16) / GH, qb = (rt * 16 + 15) / GH;   // the two slots of this wave's row-tile
  const float scale = __builtin_bit_cast(float, 0x39b8aa3bu);
  const float NEG = -1.0e20f;

  if (tid < QG) sLast[tid] = -1;
  if (tid < 2) { sAny[tid] = 0; sMixed[tid] = 0; sInc[tid] = 0; }
  __syncthreads();
  for (int r = 0; r < QG; ++r) {        // release prefixbound rule per row: last nonnegative prefix column
    const int gr = group * QG + r;
    if (gr >= NR) break;
    int last = -1;
    for (int c = tid; c < (NTILES - 1) * BN; c += NTHR)
      if (IDX[(int64_t)gr * SIDX + c] >= 0) last = c;
    if (last >= 0) atomicMax(&sLast[r], last);
  }

  u32x8 qf[16];
  {
    const uint16_t* qp = Q + (int64_t)grow * SQ0 + (int64_t)head * SQ1;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      u32x4 a = {0, 0, 0, 0}, b = {0, 0, 0, 0};
      if (active) {
        a = *(const u32x4*)(qp + k * 16);
        b = *(const u32x4*)(qp + k * 16 + 8);
      }
      qf[k] = (u32x8){a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    }
  }
  __syncthreads();
  int ptiles = 0;
#pragma unroll
  for (int r = 0; r < QG; ++r) {
    const int t = (sLast[r] + BN) / BN;
    ptiles = t > ptiles ? t : ptiles;
  }

  float maximum = NEG, norm = 0.0f;
  f32x8 acc[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) acc[t] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};
  int one_pass_tiles = 0, multi_pass_tiles = 0, passes = 0, split_rowtiles = 0;

  auto pass = [&](const int gm) {
    // (0) group union addresses
    if (tid < BN) {
      int64_t ok = -1, ov = -1;
#pragma unroll
      for (int r = 0; r < QG; ++r) {
        const int64_t k1 = sOffK[r][tid];
        if (((gm >> r) & 1) && ok < 0 && k1 >= 0) { ok = k1; ov = sOffV[r][tid]; }
      }
      sOffKG[tid] = ok;
      sOffVG[tid] = ov;
    }
    __syncthreads();
    const bool mine = ((gm >> qslot) & 1) != 0;
    const bool ina = ((gm >> qa) & 1) != 0, inb = ((gm >> qb) & 1) != 0;
    const bool rt_on = ina || inb;
#pragma unroll 2
    for (int it = tid; it < BN * 32; it += NTHR) {
      const int tok = it >> 5, pc = it & 31;
      const int64_t o = sOffKG[tok];
      const u32x4 kv = load_kv4_bf16(K, o, pc * 8);
      *(u32x4*)(sVt + k_index(tok, pc)) = kv;
    }
    __syncthreads();
    f32x8 S = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};
    if (rt_on) {
      const int tok = dq * 16 + lr;
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        const u32x4 a = *(const u32x4*)(sVt + k_index(tok, 2 * k));
        const u32x4 b = *(const u32x4*)(sVt + k_index(tok, 2 * k + 1));
        const u32x8 kf = (u32x8){a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        S = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(as_bf16x16(kf), as_bf16x16(qf[k]), S);
      }
    }
    const uint32_t myv = sValid[qslot][dq >> 1] >> ((dq & 1) * 16);   // my row's validity for columns 16dq..16dq+15
    float s[8];
    bool use[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      use[e] = active && mine && (((myv >> (2 * e + hb)) & 1u) != 0u);
      const float sc = S[e] * scale;
      s[e] = use[e] ? sc : NEG;
    }
    float m = fmaxf(fmaxf(fmaxf(s[0], s[1]), fmaxf(s[2], s[3])), fmaxf(fmaxf(s[4], s[5]), fmaxf(s[6], s[7])));
    m = fmaxf(m, __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, m), __builtin_bit_cast(int, m),
                                                                          0x76543210, 0xfedcba98, false, false)));
    if (hb == 0) sMax[dq][rr] = m;
    __syncthreads();
    const float newmax = fmaxf(maximum, fmaxf(fmaxf(sMax[0][rr], sMax[1][rr]), fmaxf(sMax[2][rr], sMax[3][rr])));
    const float alpha = __builtin_amdgcn_exp2f(maximum - newmax);
    float p[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) p[e] = use[e] ? __builtin_amdgcn_exp2f(s[e] - newmax) : 0.0f;
    const float T = ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]));
    sSum[dq][hb][rr] = T;
#pragma unroll
    for (int e = 0; e < 8; ++e) sP[rr * PPITCH + dq * 16 + 2 * e + hb] = (uint16_t)bf16_rne(p[e]);
    for (int it = tid; it < 512; it += NTHR) {
      const int c8 = it & 31, tq = it >> 5;
      u32x4 v[4];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int64_t o = sOffVG[tq * 4 + t];
        v[t] = load_kv4_bf16(V, o, c8 * 8);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint32_t sel2 = (j & 1) ? 0x07060302u : 0x05040100u;
        const uint32_t w01 = __builtin_amdgcn_perm(v[1][j >> 1], v[0][j >> 1], sel2);
        const uint32_t w23 = __builtin_amdgcn_perm(v[3][j >> 1], v[2][j >> 1], sel2);
        *(uint2*)(sVt + vt_index(c8 * 8 + j, tq * 4)) = make_uint2(w01, w23);
      }
    }
    __syncthreads();
    const float R0 = (sSum[0][0][rr] + sSum[1][0][rr]) + (sSum[2][0][rr] + sSum[3][0][rr]);
    const float R1 = (sSum[0][1][rr] + sSum[1][1][rr]) + (sSum[2][1][rr] + sSum[3][1][rr]);
    const float rowsum = hb == 0 ? (R0 + R1) : (R1 + R0);
    norm = __builtin_fmaf(norm, alpha, rowsum);
#pragma unroll
    for (int t = 0; t < 4; ++t) acc[t] = acc[t] * alpha;
    if (rt_on) {
      const uint32_t a0 = (uint32_t)rfl((int)sValid[qa][0]), a1 = (uint32_t)rfl((int)sValid[qa][1]);
      const uint32_t b0 = (uint32_t)rfl((int)sValid[qb][0]), b1 = (uint32_t)rfl((int)sValid[qb][1]);
      const bool loose = SHARE == 3;
      const bool split = !loose && ina && inb && (a0 != b0 || a1 != b1);
      const int nch = split ? 2 : 1;
#pragma unroll 1
      for (int ch = 0; ch < nch; ++ch) {
        asm volatile("" ::: "memory");   // compiler-only fence: re-read P/V^T from LDS per chain (no 128-VGPR hoist)
        const bool useA = split ? (ch == 0) : ina;
        const uint32_t m0 = useA ? a0 : b0, m1 = useA ? a1 : b1;
        const bool zero_me = split && (qslot != (useA ? qa : qb));
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const u32x4 pa = *(const u32x4*)(sP + rr * PPITCH + c * 16);
          const u32x4 pb = *(const u32x4*)(sP + rr * PPITCH + c * 16 + 8);
          u32x8 pf = (u32x8){pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w};
          if (zero_me) pf = (u32x8){0, 0, 0, 0, 0, 0, 0, 0};
          const uint32_t bits = ((c < 2 ? m0 : m1) >> ((c & 1) * 16)) & 0xffffu;
          u32x8 vm;   // wave-uniform mask words, forced scalar (SGPR operands of v_and_b32)
#pragma unroll
          for (int i = 0; i < 8; ++i)
            vm[i] = (uint32_t)rfl((int)(loose ? ~0u : ((((bits >> (2 * i)) & 1u) ? 0x0000ffffu : 0u) |
                                                       (((bits >> (2 * i + 1)) & 1u) ? 0xffff0000u : 0u))));
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            const int d = (dq * 4 + t) * 16 + lr;
            const u32x4 a = *(const u32x4*)(sVt + vt_index(d, c * 16));
            const u32x4 b = *(const u32x4*)(sVt + vt_index(d, c * 16 + 8));
            const u32x8 vf = (u32x8){a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w} & vm;
            acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(as_bf16x16(vf), as_bf16x16(pf), acc[t]);
          }
        }
      }
    }
    if (tid == 0) {
      ++passes;
      for (int r2 = 0; r2 < 3; ++r2) {
        const int xa = (r2 * 16) / GH, xb = (r2 * 16 + 15) / GH;
        if (SHARE != 3 && ((gm >> xa) & 1) && ((gm >> xb) & 1) &&
            (sValid[xa][0] != sValid[xb][0] || sValid[xa][1] != sValid[xb][1]))
          ++split_rowtiles;
      }
    }
    maximum = newmax;
    __syncthreads();
  };

  for (int ct = 0; ct <= ptiles; ++ct) {
    const int tile = ct < ptiles ? ct : NTILES - 1;
    const int buf = ct & 1;
    // (a) split-K column metadata for the 4 slots (waves 0..7 exactly: wave w -> slot w>>1, columns 32(w&1)..+31)
    if (tid < QG * BN) {
      const int r = tid >> 6, c = tid & (BN - 1);
      const int gr = group * QG + r;
      const int col = tile * BN + c;
      int64_t ok = -1, ov = -1;
      if (gr < NR && col < TOPK) {
        const int logical = IDX[(int64_t)gr * SIDX + col];
        const int req = T2R[gr];
        bool valid = req >= 0 && req < NREQ && logical >= 0;
        const int pg = valid ? logical / PAGE : 0;
        const int off = valid ? logical - pg * PAGE : 0;
        valid = valid && pg < TW;
        const int physical = valid ? TAB[(int64_t)req * STAB + pg] : -1;
        valid = valid && physical >= 0 && physical < NP;
        if (valid) {
          ok = (int64_t)physical * SK0 + (int64_t)off * SK1 + (int64_t)kvhead * SK2;
          ov = (int64_t)physical * SV0 + (int64_t)off * SV1 + (int64_t)kvhead * SV2;
        }
      }
      sOffK[r][c] = ok;
      sOffV[r][c] = ov;
      const uint32_t bal = __builtin_amdgcn_ballot_w32(ok >= 0);
      if (lane == 0) {
        sValid[r][c >> 5] = bal;
        if (bal) atomicOr(&sAny[buf], 1 << r);
      }
    }
    __syncthreads();
    // (b) identical / pairwise-compatibility tests; reset the other parity buffer
    if (tid == 0) { sAny[buf ^ 1] = 0; sMixed[buf ^ 1] = 0; sInc[buf ^ 1] = 0; }
    if (tid < BN) {
      int64_t kk[QG], vv[QG];
#pragma unroll
      for (int r = 0; r < QG; ++r) { kk[r] = sOffK[r][tid]; vv[r] = sOffV[r][tid]; }
      bool mixed = false;
#pragma unroll
      for (int r = 1; r < QG; ++r)
        if (group * QG + r < NR) mixed |= (kk[r] != kk[0]) || (vv[r] != vv[0]);
      int inc = 0;
#pragma unroll
      for (int a = 0; a < QG; ++a)
#pragma unroll
        for (int b = a + 1; b < QG; ++b)
          if (kk[a] >= 0 && kk[b] >= 0 && (kk[a] != kk[b] || vv[a] != vv[b])) inc |= (1 << (a * 4 + b)) | (1 << (b * 4 + a));
      if (mixed) atomicOr(&sMixed[buf], 1);
      if (inc) atomicOr(&sInc[buf], inc);
    }
    __syncthreads();
    const int any = rfl(sAny[buf]);
    if (any == 0) continue;
    const int mixed = rfl(sMixed[buf]);
    const int inc = rfl(sInc[buf]);
    // (c) groups, packed 4 bits each (uniform scalar logic)
    int gpack = 0, ng = 0;
    if (SHARE == 1 && mixed == 0) {
      gpack = any; ng = 1;
    } else {
#pragma unroll
      for (int s2 = 0; s2 < QG; ++s2) {
        if (!((any >> s2) & 1)) continue;
        const int row = (SHARE >= 2) ? ((inc >> (4 * s2)) & 15) : 15;   // SHARE 0/1(mixed): nothing compatible
        bool placed = false;
#pragma unroll
        for (int g = 0; g < QG; ++g) {
          const int gm = (gpack >> (4 * g)) & 15;
          if (!placed && g < ng && (row & gm) == 0) { gpack |= (1 << s2) << (4 * g); placed = true; }
        }
        if (!placed) { gpack |= (1 << s2) << (4 * ng); ++ng; }
      }
    }
    if (ng == 1) ++one_pass_tiles; else ++multi_pass_tiles;
#pragma unroll 1
    for (int g = 0; g < ng; ++g) pass((gpack >> (4 * g)) & 15);
  }
  if (tid == 0) {
    const int pid = group * gridDim.y + kvhead;
    if (STATS != nullptr) { STATS[pid * 2] = one_pass_tiles; STATS[pid * 2 + 1] = multi_pass_tiles; }
    if (STATS2 != nullptr) { STATS2[pid * 2] = passes; STATS2[pid * 2 + 1] = split_rowtiles; }
  }
  if (active) {
    uint16_t* op = O + (int64_t)grow * SO0 + (int64_t)head * SO1;
    const float den = fmaxf(norm, 1.0e-20f);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float val = norm > 0.0f ? acc[t][e] / den : 0.0f;
        op[(dq * 4 + t) * 16 + 2 * e + hb] = (uint16_t)bf16_rne(val);
      }
    }
  }
}
