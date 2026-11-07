// Apache-2.0. Source-faithful HIP port of q38spec.q38_filter_logits (E21 v2 draft filter, KC = 64) for gfx1151 wave32.
// Oracle: the installed torch implementation in htr-stack:s5-gdn-wave32 (torch 2.13.0+rocm10.0.0, rocPRIM 4.0.6).
// Every output bit must equal the oracle. Reproduced contract (each part verified against torch in test_components.py):
//  1. torch.topk(logits, 64): stable descending radix order; ties by ascending index.
//     Pinned ROCm sort merges +0/-0 for one row, but the multi-row segmented path orders +0 before -0.
//  2. v = float(vals) / t (IEEE fp32 division), t = (t < 1e-5f) ? 1 : t.
//  3. keep = v >= v[kk-1], kk = clamp(k, 1, 64); vm = keep ? v : -inf.
//  4. softmax(vm): ATen softmax_warp_forward<log2_elements=6, WARP_SIZE=32>: lane l holds vm[l], vm[l+32]; max butterfly
//     (a < b ? b : a) over xor 16..1; e = exp(x - max); sum = (0 + e0) + e1 then butterfly add; pr = sum == 0 ? NaN : e / sum.
//  5. tail = flip(cumsum(flip(pr))): rows == 1 -> rocPRIM single-block scan (rdna3 float config: 256 threads, 4 items,
//     reduce_then_scan); rows 2..32 -> ATen tensor_kernel_scan_innermost_dim Sklansky (>= 64 slots, one pass);
//     rows >= 33 -> two Sklansky-32 passes with carried block total.
//  6. rm = (j > 0) && tail <= 1 - p; keep &= !rm; out = -inf except out[id_j] = vals_j where kept.
//  7. Rows with k > 64 or idx_mapping < 0 are copied unchanged.
// Selection: K1 (NB blocks per row) keeps each block's exact top-64 composite keys (ok << 32 | ~idx); a block takes every
// element >= the 64th largest per-thread maximum (a provable superset), falling back to an exact radix select when that
// superset exceeds CAP. K2 (NW blocks per row) sorts the NB*64 survivors, does steps 2-6 in one wave, writes its slice.
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {
constexpr int KC = 64, NT = 256, MAXM = 64, CAP = 256, MAXNB = 16;
typedef unsigned long long u64;
// Negative controls (test builds only): prove the qualification can see a tie-order or scan-association change.
#ifdef Q38F_NEG_TIE
#define Q38F_IDXKEY(j) ((uint32_t)(j))
#else
#define Q38F_IDXKEY(j) (0xFFFFFFFFu - (uint32_t)(j))
#endif
#define Q38F_KEYIDX(ck) ((int)Q38F_IDXKEY((uint32_t)(ck)))

__device__ __forceinline__ uint32_t okey(uint16_t b, bool merge_zero) {
    uint32_t k = (b & 0x8000u) ? (uint32_t)(uint16_t)~b : (uint32_t)(b ^ 0x8000u);
    return merge_zero && k == 0x7FFFu ? 0x8000u : k;
}
__device__ __forceinline__ uint32_t okey(uint32_t b, bool merge_zero) {
    uint32_t k = (b & 0x80000000u) ? ~b : (b ^ 0x80000000u);
    return merge_zero && k == 0x7FFFFFFFu ? 0x80000000u : k;
}
__device__ __forceinline__ float tofloat(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ __forceinline__ float tofloat(uint32_t b) { return __uint_as_float(b); }
template <typename T> __device__ __forceinline__ T neginf();
template <> __device__ __forceinline__ uint16_t neginf<uint16_t>() { return 0xFF80u; }
template <> __device__ __forceinline__ uint32_t neginf<uint32_t>() { return 0xFF800000u; }

// Block-wide bitonic sort of N keys in LDS, descending.
template <typename K, int N>
__device__ __forceinline__ void bitonic_desc(K* s) {
    for (int kk = 2; kk <= N; kk <<= 1)
        for (int jj = kk >> 1; jj > 0; jj >>= 1) {
            for (int q = threadIdx.x; q < N; q += NT) {
                const int ix = q ^ jj;
                if (ix > q) {
                    const K a = s[q], c = s[ix];
                    const bool desc = (q & kk) == 0;
                    if (desc ? (a < c) : (a > c)) { s[q] = c; s[ix] = a; }
                }
            }
            __syncthreads();
        }
}

__device__ __forceinline__ bool row_filtered(const int* idxmap, const int* topk, int row, int& safe, long long& k) {
    const int i = idxmap[row];
    safe = i < 0 ? 0 : i;
    k = topk[safe];
    return k <= KC && i >= 0;
}

template <typename T>
__global__ __launch_bounds__(NT) void q38f_select(const T* __restrict__ x, long long ls, int V, int L, int NB,
                                                   const int* __restrict__ idxmap, const int* __restrict__ topk,
                                                   u64* __restrict__ cand) {
    const int row = blockIdx.y, b = blockIdx.x, t = threadIdx.x;
    int safe; long long k;
    if (!row_filtered(idxmap, topk, row, safe, k)) return;
    const int start = b * L, end = min(V, start + L);
    const T* xr = x + (long long)row * ls;
    uint32_t ok[MAXM];
    uint32_t mx = 0;
#pragma unroll
    for (int m = 0; m < MAXM; ++m) {
        const int j = start + t + m * NT;
        uint32_t o = 0;
        if (j < end) o = okey(xr[j], gridDim.y == 1);
        ok[m] = o;
        mx = o > mx ? o : mx;
    }
    __shared__ uint32_t smx[NT];
    __shared__ u64 sc[CAP];
    __shared__ unsigned int hist[256];
    __shared__ int cn;
    __shared__ unsigned int sel_d, sel_above;
    smx[t] = mx;  // every thread owns >= 1 valid element (host contract: each slice has >= NT elements)
    if (t == 0) cn = 0;
    __syncthreads();
    bitonic_desc<uint32_t, NT>(smx);
    const uint32_t tau = smx[KC - 1];  // >= 64 distinct elements have ok >= tau, so {ok >= tau} contains the block top-64
#pragma unroll
    for (int m = 0; m < MAXM; ++m) {
        const int j = start + t + m * NT;
        if (j < end && ok[m] >= tau) {
            const int s = atomicAdd(&cn, 1);
            if (s < CAP) sc[s] = ((u64)ok[m] << 32) | (u64)Q38F_IDXKEY(j);
        }
    }
    __syncthreads();
    int n = cn;
    if (n > CAP) {
        // Exact radix select of the 64th largest (unique) composite key; wave-aggregated LDS histogram increments.
        u64 prefix = 0, pmask = 0;
        int need = KC;
        for (int shift = 56; shift >= 0; shift -= 8) {
            hist[t] = 0;
            __syncthreads();
#pragma unroll
            for (int m = 0; m < MAXM; ++m) {
                const int j = start + t + m * NT;
                const u64 ck = ((u64)ok[m] << 32) | (u64)Q38F_IDXKEY(j);
                bool act = (j < end) && ((ck & pmask) == prefix);
                const unsigned int dig = (unsigned int)(ck >> shift) & 0xFFu;
                u64 am = __ballot(act);
                while (am) {
                    const int leader = __ffsll((long long)am) - 1;
                    const unsigned int dl = __shfl(dig, leader, 32);
                    const u64 same = __ballot(act && dig == dl);
                    if ((t & 31) == leader) atomicAdd(&hist[dl], (unsigned int)__popcll(same));
                    if (act && dig == dl) act = false;
                    am &= ~same;
                }
            }
            __syncthreads();
            if (t == 0) {
                unsigned int acc = 0;
                int d = 255;
                for (; d > 0; --d) {
                    if (acc + hist[d] >= (unsigned int)need) break;
                    acc += hist[d];
                }
                sel_d = (unsigned int)d;
                sel_above = acc;
            }
            __syncthreads();
            need -= (int)sel_above;
            prefix |= (u64)sel_d << shift;
            pmask |= (u64)0xFFu << shift;
            __syncthreads();
        }
        if (t == 0) cn = 0;
        __syncthreads();
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            const int j = start + t + m * NT;
            const u64 ck = ((u64)ok[m] << 32) | (u64)Q38F_IDXKEY(j);
            if (j < end && ck >= prefix) {
                const int s = atomicAdd(&cn, 1);
                if (s < CAP) sc[s] = ck;
            }
        }
        __syncthreads();
        n = cn;  // == KC (keys are unique)
    }
    for (int q = n + t; q < CAP; q += NT) sc[q] = 0ull;
    __syncthreads();
    bitonic_desc<u64, CAP>(sc);
    if (t < KC) cand[((long long)row * NB + b) * KC + t] = sc[t];
}

__device__ __forceinline__ void sklansky_inplace(float* buf, int n) {
    // ATen tensor_kernel_scan_innermost_dim round structure; no intra-round hazards, so sequential order is exact.
    for (int m = 0; (1 << m) < n; ++m) {
        const int s = 1 << m;
        for (int tid = 0; tid < n / 2; ++tid) {
            const int a = ((tid >> m) << (m + 1)) | s;
            const int ti = a + (tid % s), si = a - 1;
            buf[ti] = buf[ti] + buf[si];
        }
    }
}

template <typename T>
__global__ __launch_bounds__(NT) void q38f_finish(const T* __restrict__ x, long long ls, T* __restrict__ out, long long os,
                                                   int V, int W, int NB, int rows, const int* __restrict__ idxmap,
                                                   const float* __restrict__ temp, const int* __restrict__ topk,
                                                   const float* __restrict__ topp, const u64* __restrict__ cand) {
    const int row = blockIdx.y, w = blockIdx.x, t = threadIdx.x;
    int safe; long long k;
    const bool filt = row_filtered(idxmap, topk, row, safe, k);
    const int ws = w * W, we = min(V, ws + W);
    const T* xr = x + (long long)row * ls;
    T* orow = out + (long long)row * os;
    if (!filt) {
        for (int q = ws + t; q < we; q += NT) orow[q] = xr[q];
        return;
    }
    __shared__ u64 s[KC * MAXNB];
    __shared__ int sid[KC];
    __shared__ T sval[KC];
    __shared__ float spr[KC], stail[KC];
    __shared__ int skeep[KC];
    for (int q = t; q < KC * MAXNB; q += NT) s[q] = q < NB * KC ? cand[(long long)row * NB * KC + q] : 0ull;
    __syncthreads();
    bitonic_desc<u64, KC * MAXNB>(s);
    if (t < 32) {
        const int l = t;
        const int id0 = Q38F_KEYIDX(s[l]), id1 = Q38F_KEYIDX(s[l + 32]);
        const T b0 = xr[id0], b1 = xr[id1];
        float tt = temp[safe];
        tt = (tt < 1e-5f) ? 1.0f : tt;
        const float v0 = tofloat(b0) / tt, v1 = tofloat(b1) / tt;
        const int kk = (int)(k < 1 ? 1 : (k > KC ? KC : k));
        const float kth = __shfl((kk - 1) >= 32 ? v1 : v0, (kk - 1) & 31, 32);
        const bool keep0 = v0 >= kth, keep1 = v1 >= kth;
        float e0 = keep0 ? v0 : -INFINITY, e1 = keep1 ? v1 : -INFINITY;
        float mxv = e0;
        mxv = mxv > e0 ? mxv : e0;
        mxv = mxv > e1 ? mxv : e1;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float o = __shfl_xor(mxv, off, 32);
            mxv = mxv < o ? o : mxv;
        }
        float sum = 0.0f;
        e0 = expf(e0 - mxv);
        sum += e0;
        e1 = expf(e1 - mxv);
        sum += e1;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float o = __shfl_xor(sum, off, 32);
            sum = sum + o;
        }
        const float qnan = __uint_as_float(0x7FC00000u);
        spr[l] = (sum == 0) ? qnan : e0 / sum;
        spr[l + 32] = (sum == 0) ? qnan : e1 / sum;
        sid[l] = id0; sid[l + 32] = id1;
        sval[l] = b0; sval[l + 32] = b1;
        skeep[l] = keep0; skeep[l + 32] = keep1;
    }
    __syncthreads();
    if (t == 0) {
        float xs[KC], c[KC];
        for (int q = 0; q < KC; ++q) xs[q] = spr[KC - 1 - q];  // flip
#ifdef Q38F_NEG_SEQSCAN
        if (rows == 1) {
            float acc = 0.0f;
            for (int q = 0; q < KC; ++q) { acc = acc + xs[q]; c[q] = acc; }
        } else
#endif
        if (rows == 1) {
            c[0] = xs[0]; c[1] = c[0] + xs[1]; c[2] = c[1] + xs[2]; c[3] = c[2] + xs[3];
            float S = ((xs[0] + xs[1]) + xs[2]) + xs[3];
            for (int g = 1; g < KC / 4; ++g) {
                const float* y = xs + 4 * g;
                c[4 * g] = S + y[0];
                c[4 * g + 1] = c[4 * g] + y[1];
                c[4 * g + 2] = c[4 * g + 1] + y[2];
                c[4 * g + 3] = c[4 * g + 2] + y[3];
                const float Tg = ((y[0] + y[1]) + y[2]) + y[3];
                S = S + Tg;
            }
        } else if (rows <= 32) {
            for (int q = 0; q < KC; ++q) c[q] = xs[q];
            c[0] = c[0] + 0.0f;  // row_buf[0] = binary_op(row_buf[0], block_total = init 0)
            sklansky_inplace(c, KC);
        } else {
            float total = 0.0f;
            for (int base = 0; base < KC; base += 32) {
                for (int q = 0; q < 32; ++q) c[base + q] = xs[base + q];
                c[base] = c[base] + total;
                sklansky_inplace(c + base, 32);
                total = c[base + 31];
            }
        }
        for (int q = 0; q < KC; ++q) stail[q] = c[KC - 1 - q];  // flip back
    }
    __syncthreads();
    if (t < KC) {
        const float thr = 1.0f - topp[safe];
        const bool rm = (t > 0) && (stail[t] <= thr);
        skeep[t] = skeep[t] && !rm;
    }
    for (int q = ws + t; q < we; q += NT) orow[q] = neginf<T>();
    __syncthreads();
    if (t < KC && skeep[t]) {
        const int id = sid[t];
        if (id >= ws && id < we) orow[id] = sval[t];
    }
}
}  // namespace

// dtype: 0 = bf16 (uint16 bits), 1 = fp32. Returns 0 on success, negative on contract violation (caller falls back).
extern "C" int q38f_launch(const void* x, long long ls, void* out, long long os, int rows, int V, int dtype,
                           const int* idxmap, const float* temp, const int* topk, const float* topp, void* cand,
                           int NB, int NW, void* stream) {
    if (rows < 1 || V < KC || NB < 1 || NB > MAXNB || NW < 1 || (dtype != 0 && dtype != 1)) return -1;
    const int L = (V + NB - 1) / NB;
    if (L > NT * MAXM || V - (NB - 1) * L < NT) return -2;  // registers hold a slice; every thread owns an element
    const int W = (V + NW - 1) / NW;
    hipStream_t st = (hipStream_t)stream;
    if (dtype == 0) {
        hipLaunchKernelGGL(q38f_select<uint16_t>, dim3(NB, rows), dim3(NT), 0, st, (const uint16_t*)x, ls, V, L, NB,
                           idxmap, topk, (u64*)cand);
        hipLaunchKernelGGL(q38f_finish<uint16_t>, dim3(NW, rows), dim3(NT), 0, st, (const uint16_t*)x, ls,
                           (uint16_t*)out, os, V, W, NB, rows, idxmap, temp, topk, topp, (const u64*)cand);
    } else {
        hipLaunchKernelGGL(q38f_select<uint32_t>, dim3(NB, rows), dim3(NT), 0, st, (const uint32_t*)x, ls, V, L, NB,
                           idxmap, topk, (u64*)cand);
        hipLaunchKernelGGL(q38f_finish<uint32_t>, dim3(NW, rows), dim3(NT), 0, st, (const uint32_t*)x, ls,
                           (uint32_t*)out, os, V, W, NB, rows, idxmap, temp, topk, topp, (const u64*)cand);
    }
    return (int)hipGetLastError();
}
