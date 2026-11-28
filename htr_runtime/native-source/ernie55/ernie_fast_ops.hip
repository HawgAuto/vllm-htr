// Native gfx1151 (RDNA3.5) Wave32 HIP fast ops for ERNIE-5.5-Flash.
// Replaces 6 Triton / PyTorch overhead ops with zero-JIT Wave32 HIP kernels:
//   1. ernie_fused_route      (42x/tok: split-sigmoid-sum + bias top-k + renorm + scale in 1 Wave32/tok)
//   2. ernie_fwht128          (12x/tok: in-register Wave32 FWHT-128 replacing torch.matmul(x, H) in DSA indexer)
//   3. ernie_merge_swa_topk   (6x/tok: merge indexer top-k + sliding window indices)
//   4. ernie_req_to_global    (6x/tok: request-local -> global KV slot lookup)
//   5. ernie_scatter_dsa_kv   (6x/tok: vectorized 128-bit scatter into BLNHC DSA KV cache)
//   6. ernie_multimax         (1x/tok: vectorized float4 in-place MultiMax over 201216 vocab)
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <math.h>

typedef float f4v __attribute__((ext_vector_type(4)));
typedef f4v f4va __attribute__((aligned(4)));
typedef uint32_t u2va __attribute__((ext_vector_type(2), aligned(2)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef short bf16x16 __attribute__((ext_vector_type(16)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));
typedef uint32_t u8v __attribute__((ext_vector_type(8)));
typedef u4v u4va __attribute__((aligned(4)));

// Native RDNA 3.5 (gfx1151) hardware V_DOT2_F32_BF16 + explicit VOPD V_DUAL_DOT2ACC_F32_BF16!
// Note: gfx1151 has V_DUAL_DOT2ACC_F32_BF16 in VOPD (opcode 13) without a VOP2 v_dot2acc_f32_bf16,
// so LLVM never auto-emits V_DUAL_DOT2ACC_F32_BF16; pinning an even/odd VGPR pair (R_EVEN, R_ODD)
// while W and X are contiguous 128-bit u4v loads mathematically guarantees all VOPD bank rules!
__device__ __forceinline__ float bf16_dot2(uint32_t a, uint32_t b, float c) {
  return __builtin_amdgcn_fdot2_f32_bf16(
      __builtin_bit_cast(bf16x2, a), __builtin_bit_cast(bf16x2, b), c, false);
}
#define BF16_DUAL_DOT2_8_PIN(A0, A1, W, X, R_EVEN, R_ODD) do { \
  asm volatile( \
    "v_dual_dot2acc_f32_bf16 %0, %2, %3 :: v_dual_dot2acc_f32_bf16 %1, %4, %5\n\t" \
    "v_dual_dot2acc_f32_bf16 %0, %6, %7 :: v_dual_dot2acc_f32_bf16 %1, %8, %9" \
    : "+{" R_EVEN "}"(A0), "+{" R_ODD "}"(A1) \
    : "v"((W).x), "v"((X).x), "v"((W).y), "v"((X).y), \
      "v"((W).z), "v"((X).z), "v"((W).w), "v"((X).w) \
  ); \
} while (0)

// Native RDNA 3.5 V_PERM_B32 single-instruction packed BF16x2 RNE conversion (3 ops -> 1 op!)
__device__ __forceinline__ uint32_t f32_to_bf16x2_rne(float a, float b) {
  uint32_t ua = __float_as_uint(a);
  uint32_t ub = __float_as_uint(b);
  ua = ((ua & 0x7fffffffu) > 0x7f800000u) ? (ua | 0x00400000u) : (ua + 0x7fffu + ((ua >> 16) & 1u));
  ub = ((ub & 0x7fffffffu) > 0x7f800000u) ? (ub | 0x00400000u) : (ub + 0x7fffu + ((ub >> 16) & 1u));
  return __builtin_amdgcn_perm(ub, ua, 0x07060302u);
}

template <int MSK> __device__ __forceinline__ float shx_f(float v) {
  const int iv = __builtin_bit_cast(int, v);
  if constexpr (MSK < 16) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, iv, 0x160 | MSK, 0xF, 0xF, false));
  } else {
    return __builtin_bit_cast(float, (int)__builtin_amdgcn_permlanex16((unsigned)iv, (unsigned)iv, 0x76543210u, 0xfedcba98u, false, false));
  }
}
template <int MSK> __device__ __forceinline__ int shx_i(int v) {
  if constexpr (MSK < 16) {
    return __builtin_amdgcn_update_dpp(0, v, 0x160 | MSK, 0xF, 0xF, false);
  } else {
    return (int)__builtin_amdgcn_permlanex16((unsigned)v, (unsigned)v, 0x76543210u, 0xfedcba98u, false, false);
  }
}
template <int MSK> __device__ __forceinline__ void fwht_stage(float v[4], int lane) {
  const bool up = (lane & MSK) != 0;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float o = shx_f<MSK>(v[e]);
    v[e] = up ? (o - v[e]) : (v[e] + o);
  }
}
__device__ __forceinline__ void fwht128_wave(float v[4], int lane) {
  float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
  v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
  fwht_stage<1>(v, lane); fwht_stage<2>(v, lane); fwht_stage<4>(v, lane); fwht_stage<8>(v, lane); fwht_stage<16>(v, lane);
}
__device__ __forceinline__ uint16_t f32_to_bf16_rne(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
template <int CS_IS_BF16>
__device__ __forceinline__ float load_cs1(const void* __restrict__ cs_ptr, int64_t idx) {
  if constexpr (CS_IS_BF16) {
    return __uint_as_float((uint32_t)(reinterpret_cast<const uint16_t*>(cs_ptr)[idx]) << 16);
  } else {
    return reinterpret_cast<const float*>(cs_ptr)[idx];
  }
}
template <int CS_IS_BF16>
__device__ __forceinline__ void load_cs4(const void* __restrict__ cs_ptr, int64_t idx, float out[4]) {
  if constexpr (CS_IS_BF16) {
    const uint2 r = *reinterpret_cast<const uint2*>(reinterpret_cast<const uint16_t*>(cs_ptr) + idx);
    out[0] = __uint_as_float((r.x & 0xFFFFu) << 16);
    out[1] = __uint_as_float(r.x & 0xFFFF0000u);
    out[2] = __uint_as_float((r.y & 0xFFFFu) << 16);
    out[3] = __uint_as_float(r.y & 0xFFFF0000u);
  } else {
    const f4v r = *reinterpret_cast<const f4v*>(reinterpret_cast<const float*>(cs_ptr) + idx);
    out[0] = r.x; out[1] = r.y; out[2] = r.z; out[3] = r.w;
  }
}

// ------------------------------------------------------------------ 1: Fused ERNIE-5.5 Router (E=512, topk=10, 1 Wave32/token)
template <int TOPK, int RENORM, int HAS_PAD>
__global__ __launch_bounds__(32) void k_ernie_fused_route_512(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const float* __restrict__ scale,
    const uint8_t* __restrict__ is_pad,
    float* __restrict__ out_w,
    int32_t* __restrict__ out_ids,
    int64_t logits_stride) {
  constexpr int E = 512;
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  const float* lrow = logits + (int64_t)row * logits_stride;

  float sc[16], sel[16];
  // 4x coalesced 128-bit global_load_b128 per array (12x b128 instead of 48x scalar b32!)
  // Slot s = k*4 + e maps to expert idx = k*128 + lane*4 + e (strictly ascending with s -> exact smallest-index tie-break!)
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int e_base = k * 128 + lane * 4;
    const f4v v0 = *reinterpret_cast<const f4v*>(lrow + e_base);
    const f4v v1 = *reinterpret_cast<const f4v*>(lrow + E + e_base);
    const f4v vb = *reinterpret_cast<const f4v*>(bias + e_base);
    const float a0[4] = {v0.x, v0.y, v0.z, v0.w};
    const float a1[4] = {v1.x, v1.y, v1.z, v1.w};
    const float ab[4] = {vb.x, vb.y, vb.z, vb.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float s = __builtin_amdgcn_rcpf(1.0f + exp2f(-1.4426950408889634f * a0[e]))
                    + __builtin_amdgcn_rcpf(1.0f + exp2f(-1.4426950408889634f * a1[e]));
      sc[k * 4 + e] = s;
      sel[k * 4 + e] = s + ab[e];
    }
    __builtin_amdgcn_sched_barrier(0);
  }

  __shared__ int32_t s_top_i[TOPK];
  __shared__ float s_top_w[TOPK];

  // Keep TOPK loop rolled (7x smaller I-cache footprint) & store top-10 in 80B LDS (frees 20 VGPRs -> <=80 VGPRs = 16w/SIMD!)
#pragma nounroll
  for (int t = 0; t < TOPK; ++t) {
    // Lane-local max (smallest expert index wins ties)
    float bv = sel[0];
    int bi = lane * 4;
    float bs = sc[0];
#pragma unroll
    for (int s = 1; s < 16; ++s) {
      const float v = sel[s];
      if (v > bv) {
        bv = v;
        bi = ((s >> 2) << 7) + lane * 4 + (s & 3);
        bs = sc[s];
      }
    }
    // Wave32 deterministic argmax reduction (smallest index wins ties)
#define WAVE_ARGMAX_STEP(MSK)                                             \
    {                                                                     \
      const float ov = shx_f<MSK>(bv);                                    \
      const int oi = shx_i<MSK>(bi);                                      \
      const float os = shx_f<MSK>(bs);                                    \
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; bs = os; }\
    }
    WAVE_ARGMAX_STEP(16)
    WAVE_ARGMAX_STEP(8)
    WAVE_ARGMAX_STEP(4)
    WAVE_ARGMAX_STEP(2)
    WAVE_ARGMAX_STEP(1)
#undef WAVE_ARGMAX_STEP

    if (lane == 0) {
      s_top_i[t] = bi;
      s_top_w[t] = bs;
    }
    // Mask winning expert in owning lane (exact inverse of idx = (k<<7)|(lane<<2)|e)
    if (((bi & 127) >> 2) == lane) {
      const int win_s = ((bi >> 7) << 2) | (bi & 3);
#pragma unroll
      for (int s = 0; s < 16; ++s) {
        if (s == win_s) sel[s] = -INFINITY;
      }
    }
  }

  // 10-lane parallel SIMD epilogue (zero serial loop: 10-way parallel scale gather + 1 coalesced 40B store each!)
  float top_w = (lane < TOPK) ? s_top_w[lane] : 0.0f;
  float inv_sum = 1.0f;
  if constexpr (RENORM) {
    float sm = top_w;
    sm += shx_f<8>(sm); sm += shx_f<4>(sm); sm += shx_f<2>(sm); sm += shx_f<1>(sm);
    inv_sum = __builtin_amdgcn_rcpf(fmaxf(sm, 1e-20f));
  }
  if (lane < TOPK) {
    bool pad = false;
    if constexpr (HAS_PAD) pad = (is_pad[row] != 0);
    float* wr = out_w + (int64_t)row * TOPK;
    int32_t* ir = out_ids + (int64_t)row * TOPK;
    if (pad) {
      wr[lane] = 0.0f;
      ir[lane] = -1;
    } else {
      const int idx = s_top_i[lane];
      wr[lane] = (top_w * inv_sum) * scale[idx];
      ir[lane] = idx;
    }
  }
}

std::vector<torch::Tensor> ernie_fused_route(
    torch::Tensor logits, torch::Tensor bias, torch::Tensor scale,
    c10::optional<torch::Tensor> is_pad, int64_t topk, bool renormalize) {
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.size(1) == 1024 && logits.scalar_type() == torch::kFloat32);
  TORCH_CHECK(logits.stride(1) == 1 && topk == 10);
  const int T = (int)logits.size(0);
  auto out_w = torch::empty({T, topk}, logits.options());
  auto out_ids = torch::empty({T, topk}, logits.options().dtype(torch::kInt32));
  if (T == 0) return {out_w, out_ids};
  auto st = c10::hip::getCurrentHIPStream().stream();
  const uint8_t* pad_p = is_pad.has_value() ? reinterpret_cast<const uint8_t*>(is_pad->data_ptr()) : nullptr;
  if (renormalize) {
    if (pad_p) hipLaunchKernelGGL((k_ernie_fused_route_512<10, 1, 1>), dim3(T), dim3(32), 0, st,
                                  logits.data_ptr<float>(), bias.data_ptr<float>(), scale.data_ptr<float>(), pad_p,
                                  out_w.data_ptr<float>(), out_ids.data_ptr<int32_t>(), logits.stride(0));
    else       hipLaunchKernelGGL((k_ernie_fused_route_512<10, 1, 0>), dim3(T), dim3(32), 0, st,
                                  logits.data_ptr<float>(), bias.data_ptr<float>(), scale.data_ptr<float>(), nullptr,
                                  out_w.data_ptr<float>(), out_ids.data_ptr<int32_t>(), logits.stride(0));
  } else {
    if (pad_p) hipLaunchKernelGGL((k_ernie_fused_route_512<10, 0, 1>), dim3(T), dim3(32), 0, st,
                                  logits.data_ptr<float>(), bias.data_ptr<float>(), scale.data_ptr<float>(), pad_p,
                                  out_w.data_ptr<float>(), out_ids.data_ptr<int32_t>(), logits.stride(0));
    else       hipLaunchKernelGGL((k_ernie_fused_route_512<10, 0, 0>), dim3(T), dim3(32), 0, st,
                                  logits.data_ptr<float>(), bias.data_ptr<float>(), scale.data_ptr<float>(), nullptr,
                                  out_w.data_ptr<float>(), out_ids.data_ptr<int32_t>(), logits.stride(0));
  }
  return {out_w, out_ids};
}

// ------------------------------------------------------------------ 2: In-register Wave32 FWHT-128 (bf16 [N, 128])
__global__ __launch_bounds__(128) void k_ernie_fwht128_bf16(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ out, int N) {
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= N) return;
  const uint2 raw = *reinterpret_cast<const uint2*>(x + (size_t)row * 128 + lane * 4);
  float v[4];
  v[0] = __uint_as_float((raw.x & 0xFFFFu) << 16);
  v[1] = __uint_as_float(raw.x & 0xFFFF0000u);
  v[2] = __uint_as_float((raw.y & 0xFFFFu) << 16);
  v[3] = __uint_as_float(raw.y & 0xFFFF0000u);
  fwht128_wave(v, lane);
  constexpr float INV = 1.0f / 11.313708498984761f;
  uint2 o;
  o.x = f32_to_bf16x2_rne(v[0] * INV, v[1] * INV);
  o.y = f32_to_bf16x2_rne(v[2] * INV, v[3] * INV);
  *reinterpret_cast<uint2*>(out + (size_t)row * 128 + lane * 4) = o;
}

torch::Tensor ernie_fwht128(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 && x.size(1) == 128 && x.scalar_type() == torch::kBFloat16);
  const int N = (int)x.size(0);
  auto out = torch::empty_like(x);
  if (N == 0) return out;
  hipLaunchKernelGGL(k_ernie_fwht128_bf16, dim3((N + 3) / 4), dim3(128), 0,
                     c10::hip::getCurrentHIPStream().stream(),
                     reinterpret_cast<const uint16_t*>(x.data_ptr()),
                     reinterpret_cast<uint16_t*>(out.data_ptr()), N);
  return out;
}

// ------------------------------------------------------------------ 3: merge_swa_topk
__global__ __launch_bounds__(256) void k_merge_swa_topk(
    const int32_t* __restrict__ topk_ptr, int32_t* __restrict__ out_ptr,
    const int64_t* __restrict__ pos_ptr, int64_t topk_stride, int64_t out_stride,
    int window, int topk, int width) {
  const int row = blockIdx.x;
  const int seq_len = (int)pos_ptr[row] + 1;
  const int indexer_len = min(max(seq_len - window, 0), topk);
  const int swa_len = min(seq_len, window);
  const int32_t* src = topk_ptr + (int64_t)row * topk_stride;
  int32_t* dst = out_ptr + (int64_t)row * out_stride;
  for (int c = threadIdx.x; c < width; c += 256) {
    int32_t val = -1;
    if (c < indexer_len) val = src[c];
    else if (c < indexer_len + swa_len) val = seq_len - swa_len + c - indexer_len;
    dst[c] = val;
  }
}

void ernie_merge_swa_topk(torch::Tensor topk_indices, torch::Tensor out, torch::Tensor positions, int64_t window) {
  const int T = (int)positions.size(0);
  if (T == 0) return;
  TORCH_CHECK(positions.scalar_type() == torch::kInt64);
  hipLaunchKernelGGL(k_merge_swa_topk, dim3(T), dim3(256), 0, c10::hip::getCurrentHIPStream().stream(),
                     topk_indices.data_ptr<int32_t>(), out.data_ptr<int32_t>(), positions.data_ptr<int64_t>(),
                     topk_indices.stride(0), out.stride(0), (int)window, (int)topk_indices.size(1), (int)out.size(1));
}

// ------------------------------------------------------------------ 4: req_local_to_global_slots
__global__ __launch_bounds__(256) void k_req_to_global(
    const int32_t* __restrict__ req_ids, const int32_t* __restrict__ bt,
    const int32_t* __restrict__ local_idx, int32_t* __restrict__ global_idx,
    int64_t bt_stride0, int64_t li_stride0, int64_t gi_stride0,
    int max_blocks, int block_size, int block_stride_rows, int W) {
  const int t = blockIdx.x;
  const int c = blockIdx.y * 256 + threadIdx.x;
  if (c >= W) return;
  const int req = req_ids[t];
  const int tok = local_idx[(int64_t)t * li_stride0 + c];
  int32_t gslot = -1;
  if (tok >= 0) {
    const int blk_i = tok / block_size;
    const int in_blk = tok % block_size;
    if (blk_i < max_blocks) {
      const int phys = bt[(int64_t)req * bt_stride0 + blk_i];
      if (phys >= 0) gslot = phys * block_stride_rows + in_blk;
    }
  }
  global_idx[(int64_t)t * gi_stride0 + c] = gslot;
}

torch::Tensor ernie_req_to_global(
    torch::Tensor local_indices, torch::Tensor req_id_per_token, torch::Tensor block_table,
    int64_t block_size, int64_t block_stride_rows) {
  const int T = (int)local_indices.size(0), W = (int)local_indices.size(1);
  auto out = torch::empty_like(local_indices);
  if (T == 0) return out;
  hipLaunchKernelGGL(k_req_to_global, dim3(T, (W + 255) / 256), dim3(256), 0, c10::hip::getCurrentHIPStream().stream(),
                     req_id_per_token.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(),
                     local_indices.data_ptr<int32_t>(), out.data_ptr<int32_t>(),
                     block_table.stride(0), local_indices.stride(0), out.stride(0),
                     (int)block_table.size(1), (int)block_size, (int)block_stride_rows, W);
  return out;
}

// ------------------------------------------------------------------ 5: scatter_dsa_kv (vectorized 128-bit NT store)
__global__ __launch_bounds__(128) void k_scatter_dsa_kv(
    const uint16_t* __restrict__ kvc, const uint16_t* __restrict__ kpe,
    uint16_t* __restrict__ cache, const int64_t* __restrict__ slot_mapping,
    int64_t kvc_s0, int64_t kpe_s0, int64_t c_s0, int64_t c_s1,
    int block_size, int nope_dim, int rope_dim) {
  const int t = blockIdx.x;
  const int64_t slot = slot_mapping[t];
  if (slot < 0) return;
  const int64_t blk = slot / block_size;
  const int64_t off = slot % block_size;
  uint16_t* dst = cache + blk * c_s0 + off * c_s1;
  const int tid = threadIdx.x;
  // 8 bf16 = 16 bytes per thread (64 threads cover nope_dim=512, 8 threads cover rope_dim=64)
  if (tid * 8 < nope_dim) {
    const u4v v = *reinterpret_cast<const u4va*>(kvc + (int64_t)t * kvc_s0 + tid * 8);
    __builtin_nontemporal_store(v, reinterpret_cast<u4va*>(dst + tid * 8));
  } else if ((tid - 64) >= 0 && (tid - 64) * 8 < rope_dim) {
    const int r0 = (tid - 64) * 8;
    const u4v v = *reinterpret_cast<const u4va*>(kpe + (int64_t)t * kpe_s0 + r0);
    __builtin_nontemporal_store(v, reinterpret_cast<u4va*>(dst + nope_dim + r0));
  }
}

void ernie_scatter_dsa_kv(
    torch::Tensor kv_c, torch::Tensor k_pe_2d, torch::Tensor kv_cache,
    torch::Tensor slot_mapping, int64_t block_size) {
  const int T = (int)min(kv_c.size(0), slot_mapping.size(0));
  if (T == 0 || kv_cache.numel() == 0) return;
  TORCH_CHECK(kv_c.size(1) == 512 && k_pe_2d.size(1) == 64 && kv_c.scalar_type() == torch::kBFloat16);
  hipLaunchKernelGGL(k_scatter_dsa_kv, dim3(T), dim3(128), 0, c10::hip::getCurrentHIPStream().stream(),
                     reinterpret_cast<const uint16_t*>(kv_c.data_ptr()),
                     reinterpret_cast<const uint16_t*>(k_pe_2d.data_ptr()),
                     reinterpret_cast<uint16_t*>(kv_cache.data_ptr()),
                     slot_mapping.data_ptr<int64_t>(),
                     kv_c.stride(0), k_pe_2d.stride(0), kv_cache.stride(0), kv_cache.stride(1),
                     (int)block_size, 512, 64);
}

// ------------------------------------------------------------------ 6: vectorized float4 in-place MultiMax
__global__ __launch_bounds__(256) void k_ernie_multimax(
    float* __restrict__ logits, const float* __restrict__ ranges, const float* __restrict__ ts,
    int64_t stride, int V) {
  const int row = blockIdx.y;
  const int idx4 = (blockIdx.x * 256 + threadIdx.x) * 4;
  if (idx4 >= V) return;
  const float r0 = ranges[0], r1 = ranges[1], r2 = ranges[2], r3 = ranges[3];
  const float t0 = ts[0], t1 = ts[1], t2 = ts[2], t3 = ts[3];
  float* p = logits + (int64_t)row * stride + idx4;
  if (idx4 + 4 <= V) {
    f4v x = *reinterpret_cast<f4v*>(p);
    float v[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float a = fmaxf(r0 - v[k], 0.0f);
      float b = fmaxf(v[k] - r1, 0.0f);
      float c = fmaxf(r2 - v[k], 0.0f);
      float d = fmaxf(v[k] - r3, 0.0f);
      v[k] = v[k] + t0 * a + t1 * b + t2 * (c * c) + t3 * (d * d);
    }
    *reinterpret_cast<f4v*>(p) = (f4v){v[0], v[1], v[2], v[3]};
  } else {
    for (int k = 0; k < V - idx4; ++k) {
      float x = p[k];
      float a = fmaxf(r0 - x, 0.0f);
      float b = fmaxf(x - r1, 0.0f);
      float c = fmaxf(r2 - x, 0.0f);
      float d = fmaxf(x - r3, 0.0f);
      p[k] = x + t0 * a + t1 * b + t2 * (c * c) + t3 * (d * d);
    }
  }
}

void ernie_multimax(torch::Tensor logits, torch::Tensor ranges, torch::Tensor ts) {
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.scalar_type() == torch::kFloat32 && logits.stride(1) == 1);
  const int T = (int)logits.size(0), V = (int)logits.size(1);
  if (T == 0 || V == 0) return;
  hipLaunchKernelGGL(k_ernie_multimax, dim3((V + 1023) / 1024, T), dim3(256), 0, c10::hip::getCurrentHIPStream().stream(),
                     logits.data_ptr<float>(), ranges.data_ptr<float>(), ts.data_ptr<float>(), logits.stride(0), V);
}

// ------------------------------------------------------------------ 7: Rope-only in-place inverse GPT-J RoPE
// Replaces Triton _inverse_rope_gptj_kernel (which copied all 448 unchanged NoPE dims in-place = 87.5% wasted traffic!).
// 8 Wave32s (8 heads/WG -> 8x fewer WGs on prefill + L0/LDS-shared cos/sin!). Touches ONLY [nope_dim : head_dim]!
template <int CS_IS_BF16>
__global__ __launch_bounds__(256) void k_inv_rope_gptj_inplace_64(
    uint16_t* __restrict__ o_ptr,       // [T, H, D] bf16
    const int64_t* __restrict__ pos_ptr,// [T] int64
    const void* __restrict__ cs_ptr,    // [P, 64] fp32 or bf16 (cos[32] | sin[32])
    int64_t s0, int64_t s1, int64_t cs_s0, int H, int nope_dim) {
  __shared__ float s_cs[64];
  const int t = blockIdx.y;
  const int tid = threadIdx.x;
  const int h = blockIdx.x * 8 + (tid >> 5);
  const int k = tid & 31; // 0..31
  if (tid < 64) {
    const int64_t pos = pos_ptr[t];
    s_cs[tid] = load_cs1<CS_IS_BF16>(cs_ptr, pos * cs_s0 + tid);
  }
  __syncthreads();
  if (h >= H) return;
  uint32_t* pair_p = reinterpret_cast<uint32_t*>(o_ptr + (int64_t)t * s0 + (int64_t)h * s1 + nope_dim + 2 * k);
  const uint32_t packed = *pair_p;
  const float a = __uint_as_float((packed & 0xFFFFu) << 16);
  const float b = __uint_as_float(packed & 0xFFFF0000u);
  const float c = s_cs[k];
  const float s = s_cs[32 + k];
  const float r0 = a * c + b * s;
  const float r1 = b * c - a * s;
  *pair_p = f32_to_bf16x2_rne(r0, r1);
}

void ernie_inv_rope_gptj_inplace(
    torch::Tensor o, torch::Tensor positions, torch::Tensor cos_sin_cache, int64_t rope_head_dim) {
  TORCH_CHECK(o.is_cuda() && o.dim() == 3 && o.scalar_type() == torch::kBFloat16 && o.stride(2) == 1);
  TORCH_CHECK(rope_head_dim == 64 && cos_sin_cache.size(-1) == 64);
  const int T = (int)o.size(0);
  const int H = (int)o.size(1);
  if (T == 0 || H == 0) return;
  auto pos_i64 = positions.scalar_type() == torch::kInt64 ? positions : positions.to(torch::kInt64);
  const int nope_dim = (int)o.size(2) - 64;
  auto st = c10::hip::getCurrentHIPStream().stream();
  if (cos_sin_cache.scalar_type() == torch::kBFloat16) {
    hipLaunchKernelGGL((k_inv_rope_gptj_inplace_64<1>), dim3((H + 7) / 8, T), dim3(256), 0, st,
                       reinterpret_cast<uint16_t*>(o.data_ptr()),
                       pos_i64.data_ptr<int64_t>(), cos_sin_cache.data_ptr(),
                       o.stride(0), o.stride(1), cos_sin_cache.stride(0), H, nope_dim);
  } else {
    auto cs_f32 = cos_sin_cache.scalar_type() == torch::kFloat32 ? cos_sin_cache : cos_sin_cache.to(torch::kFloat32);
    hipLaunchKernelGGL((k_inv_rope_gptj_inplace_64<0>), dim3((H + 7) / 8, T), dim3(256), 0, st,
                       reinterpret_cast<uint16_t*>(o.data_ptr()),
                       pos_i64.data_ptr<int64_t>(), cs_f32.data_ptr(),
                       o.stride(0), o.stride(1), cs_f32.stride(0), H, nope_dim);
  }
}

// ------------------------------------------------------------------ 8: Fused save_partial_states (BF16 or FP32 input -> FP32 state_cache)
// Eliminates both kv_score.float() cast (37x/tok) AND Triton SavePartialStatesKernel (37x/tok)!
// 128 threads/token, float4 (128-bit) vectorized stores for head_size=512 (or any multiple of 4).
template <int INPUT_IS_BF16>
__global__ __launch_bounds__(128) void k_save_partial_states(
    const void* __restrict__ kv_raw, int64_t kv_s0,
    const void* __restrict__ sc_raw, int64_t sc_s0,
    const float* __restrict__ ape_ptr, int64_t ape_s0,
    const int64_t* __restrict__ pos_ptr,
    float* __restrict__ state_cache, int64_t cache_s0, int64_t cache_s1,
    const int64_t* __restrict__ slot_mapping,
    int block_size, int head_size, int state_width, int compress_ratio) {
  const int t = blockIdx.x;
  const int64_t slot_id = slot_mapping[t];
  if (slot_id < 0) return;
  const int64_t blk = slot_id / block_size;
  const int64_t off = slot_id % block_size;
  float* dst = state_cache + blk * cache_s0 + off * cache_s1;
  const int64_t pos = pos_ptr[t];
  const int64_t ape_row = ((pos % compress_ratio) + compress_ratio) % compress_ratio;
  const float* ape = ape_ptr + ape_row * ape_s0;

  for (int i4 = threadIdx.x * 4; i4 < head_size; i4 += 128 * 4) {
    f4v kv4, sc4;
    if constexpr (INPUT_IS_BF16) {
      const uint16_t* kp = reinterpret_cast<const uint16_t*>(kv_raw) + (int64_t)t * kv_s0 + i4;
      const uint16_t* sp = reinterpret_cast<const uint16_t*>(sc_raw) + (int64_t)t * sc_s0 + i4;
      const uint2 kr = *reinterpret_cast<const uint2*>(kp);
      const uint2 sr = *reinterpret_cast<const uint2*>(sp);
      kv4 = (f4v){
        __uint_as_float((kr.x & 0xFFFFu) << 16),
        __uint_as_float(kr.x & 0xFFFF0000u),
        __uint_as_float((kr.y & 0xFFFFu) << 16),
        __uint_as_float(kr.y & 0xFFFF0000u),
      };
      sc4 = (f4v){
        __uint_as_float((sr.x & 0xFFFFu) << 16),
        __uint_as_float(sr.x & 0xFFFF0000u),
        __uint_as_float((sr.y & 0xFFFFu) << 16),
        __uint_as_float(sr.y & 0xFFFF0000u),
      };
    } else {
      kv4 = *reinterpret_cast<const f4v*>(reinterpret_cast<const float*>(kv_raw) + (int64_t)t * kv_s0 + i4);
      sc4 = *reinterpret_cast<const f4v*>(reinterpret_cast<const float*>(sc_raw) + (int64_t)t * sc_s0 + i4);
    }
    const f4v ap4 = *reinterpret_cast<const f4v*>(ape + i4);
    *reinterpret_cast<f4v*>(dst + i4) = kv4;
    *reinterpret_cast<f4v*>(dst + state_width + i4) = sc4 + ap4;
  }
}

void ernie_save_partial_states(
    torch::Tensor kv, torch::Tensor score, torch::Tensor ape,
    torch::Tensor positions, torch::Tensor state_cache, torch::Tensor slot_mapping,
    int64_t block_size, int64_t state_width, int64_t compress_ratio) {
  const int T = (int)slot_mapping.size(0);
  if (T == 0) return;
  const int head_size = (int)kv.size(1);
  TORCH_CHECK(head_size % 4 == 0 && kv.stride(1) == 1 && score.stride(1) == 1);
  TORCH_CHECK(ape.scalar_type() == torch::kFloat32 && state_cache.scalar_type() == torch::kFloat32);
  auto pos_i64 = positions.scalar_type() == torch::kInt64 ? positions : positions.to(torch::kInt64);
  auto slot_i64 = slot_mapping.scalar_type() == torch::kInt64 ? slot_mapping : slot_mapping.to(torch::kInt64);
  auto st = c10::hip::getCurrentHIPStream().stream();
  if (kv.scalar_type() == torch::kBFloat16 && score.scalar_type() == torch::kBFloat16) {
    hipLaunchKernelGGL((k_save_partial_states<1>), dim3(T), dim3(128), 0, st,
                       kv.data_ptr(), kv.stride(0), score.data_ptr(), score.stride(0),
                       ape.data_ptr<float>(), ape.stride(0), pos_i64.data_ptr<int64_t>(),
                       state_cache.data_ptr<float>(), state_cache.stride(0), state_cache.stride(1),
                       slot_i64.data_ptr<int64_t>(), (int)block_size, head_size, (int)state_width, (int)compress_ratio);
  } else {
    auto kv_f = kv.scalar_type() == torch::kFloat32 ? kv : kv.to(torch::kFloat32);
    auto sc_f = score.scalar_type() == torch::kFloat32 ? score : score.to(torch::kFloat32);
    hipLaunchKernelGGL((k_save_partial_states<0>), dim3(T), dim3(128), 0, st,
                       kv_f.data_ptr(), kv_f.stride(0), sc_f.data_ptr(), sc_f.stride(0),
                       ape.data_ptr<float>(), ape.stride(0), pos_i64.data_ptr<int64_t>(),
                       state_cache.data_ptr<float>(), state_cache.stride(0), state_cache.stride(1),
                       slot_i64.data_ptr<int64_t>(), (int)block_size, head_size, (int)state_width, (int)compress_ratio);
  }
}

// ------------------------------------------------------------------ 9: DSpark Context KV Insert (RoPE + UE8M0 FP8 Quant + Paged Store)
// Eliminates dummy_q = torch.zeros(n_ctx, 64, 512) allocation + 64-head wasted Q norm/RoPE on every DSpark draft step!
__device__ __forceinline__ uint8_t f32_to_fp8_e4m3fn_ocp(float x) {
  // Clamp to [-448, 448], round-to-nearest-even to OCP E4M3FN (bias=7, max=448)
  uint32_t u = __float_as_uint(x);
  const uint32_t sign = (u >> 31) & 1u;
  float ax = fabsf(x);
  if (!(ax == ax)) return 0x7Fu; // NaN
  if (ax > 448.0f) ax = 448.0f;
  u = __float_as_uint(ax);
  const int exp_f = (int)((u >> 23) & 0xFFu) - 127;
  const uint32_t man_f = u & 0x7FFFFFu;
  uint32_t fp8_exp, fp8_man;
  if (exp_f < -9) {
    // Underflow to 0 (threshold 2^-10)
    return (uint8_t)(sign << 7);
  } else if (exp_f < -6) {
    // Subnormal: value = man / 512, shift = -6 - exp_f in [1..3]
    const uint32_t sig = (1u << 23) | man_f;
    const int shift = (20 + (-6 - exp_f));
    const uint32_t half = 1u << (shift - 1);
    const uint32_t rem = sig & ((1u << shift) - 1u);
    uint32_t q = sig >> shift;
    if (rem > half || (rem == half && (q & 1u))) q += 1u;
    if (q >= 8u) { fp8_exp = 1u; fp8_man = 0u; }
    else { fp8_exp = 0u; fp8_man = q; }
  } else {
    // Normal: exp_f in [-6..8] -> biased exp in [1..15]
    const uint32_t rem = man_f & 0xFFFFFu;
    uint32_t q = man_f >> 20;
    if (rem > 0x80000u || (rem == 0x80000u && (q & 1u))) q += 1u;
    fp8_exp = (uint32_t)(exp_f + 7);
    if (q >= 8u) { fp8_exp += 1u; fp8_man = 0u; }
    else { fp8_man = q; }
    if (fp8_exp > 15u || (fp8_exp == 15u && fp8_man > 6u)) {
      fp8_exp = 15u; fp8_man = 6u; // saturate to 448.0
    }
  }
  return (uint8_t)((sign << 7) | (fp8_exp << 3) | fp8_man);
}

template <int HAS_NORM, int CS_IS_BF16>
__global__ __launch_bounds__(128) void k_insert_context_kv_ue8m0_v2(
    const uint16_t* __restrict__ kv_ptr,    // [T, 512] bf16
    const uint16_t* __restrict__ norm_w,    // [512] bf16 or nullptr
    float norm_eps,
    uint8_t* __restrict__ swa_cache,        // [num_blocks, block_bytes] uint8
    const int64_t* __restrict__ slot_map,   // [T] int64
    const int64_t* __restrict__ pos_ptr,    // [T] int64
    const void* __restrict__ cs_ptr,        // [P, 64] fp32 or bf16
    int64_t kv_s0, int64_t cache_s0, int64_t cs_s0, int block_size) {
  __shared__ float s_wsum[4];
  const int t = blockIdx.x;
  const int64_t slot = slot_map[t];
  if (slot < 0) return;
  const int tid = threadIdx.x; // 0..127 -> 4 elems/thread = 512 exact!
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int col4 = tid * 4;

  const uint16_t* src = kv_ptr + (int64_t)t * kv_s0 + col4;
  const uint2 raw = *reinterpret_cast<const uint2*>(src);
  float v[4];
  v[0] = __uint_as_float((raw.x & 0xFFFFu) << 16);
  v[1] = __uint_as_float(raw.x & 0xFFFF0000u);
  v[2] = __uint_as_float((raw.y & 0xFFFFu) << 16);
  v[3] = __uint_as_float(raw.y & 0xFFFF0000u);

  if constexpr (HAS_NORM) {
    float ss0 = bf16_dot2(raw.x, raw.x, 0.0f);
    float ss1 = bf16_dot2(raw.y, raw.y, 0.0f);
    float ss = ss0 + ss1;
    ss += shx_f<16>(ss);
    ss += shx_f<8>(ss);
    ss += shx_f<4>(ss);
    ss += shx_f<2>(ss);
    ss += shx_f<1>(ss);
    if (lane == 0) s_wsum[wave] = ss;
    __syncthreads();
    const float rrms = rsqrtf((s_wsum[0] + s_wsum[1] + s_wsum[2] + s_wsum[3]) * (1.0f / 512.0f) + norm_eps);
    const uint2 wraw = *reinterpret_cast<const uint2*>(norm_w + col4);
    const float w0 = __uint_as_float((wraw.x & 0xFFFFu) << 16);
    const float w1 = __uint_as_float(wraw.x & 0xFFFF0000u);
    const float w2 = __uint_as_float((wraw.y & 0xFFFFu) << 16);
    const float w3 = __uint_as_float(wraw.y & 0xFFFF0000u);
    v[0] = __uint_as_float(((uint32_t)f32_to_bf16_rne(v[0] * rrms * w0)) << 16);
    v[1] = __uint_as_float(((uint32_t)f32_to_bf16_rne(v[1] * rrms * w1)) << 16);
    v[2] = __uint_as_float(((uint32_t)f32_to_bf16_rne(v[2] * rrms * w2)) << 16);
    v[3] = __uint_as_float(((uint32_t)f32_to_bf16_rne(v[3] * rrms * w3)) << 16);
  }

  const int64_t blk = slot / block_size;
  const int64_t off = slot % block_size;
  uint8_t* blk_ptr = swa_cache + blk * cache_s0;
  uint8_t* tok_data = blk_ptr + off * 576;
  uint8_t* tok_sc = blk_ptr + (int64_t)block_size * 576 + off * 8;

  const int grp = tid >> 4; // 0..6 = 7 exact 64-elem NoPE blocks, 7 = 64 RoPE dims!
  const int sub = tid & 15; // 16 threads x 4 elems = 64 elems (half-wave DPP, ZERO barriers!)
  if (grp < 7) {
    float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    am = fmaxf(am, shx_f<8>(am));
    am = fmaxf(am, shx_f<4>(am));
    am = fmaxf(am, shx_f<2>(am));
    am = fmaxf(am, shx_f<1>(am));
    const float bmax = fmaxf(am, 1e-4f);
    const float exponent = ceilf(log2f(bmax * (1.0f / 448.0f)));
    const float inv_sc = exp2f(-exponent);
    uint8_t ob[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) ob[e] = f32_to_fp8_e4m3fn_ocp(v[e] * inv_sc);
    *reinterpret_cast<uint32_t*>(tok_data + col4) = *reinterpret_cast<const uint32_t*>(ob);
    if (sub == 0) {
      float enc = exponent + 127.0f;
      if (enc < 0.0f) enc = 0.0f;
      if (enc > 255.0f) enc = 255.0f;
      tok_sc[grp] = (uint8_t)enc;
    }
  } else {
    if (sub == 0) tok_sc[7] = 0u;
    const int64_t pos = pos_ptr[t];
    const int64_t cs_base = pos * cs_s0;
    const int p0 = 2 * sub;
    const int p1 = 2 * sub + 1;
    const float c0 = load_cs1<CS_IS_BF16>(cs_ptr, cs_base + p0), s0 = load_cs1<CS_IS_BF16>(cs_ptr, cs_base + 32 + p0);
    const float c1 = load_cs1<CS_IS_BF16>(cs_ptr, cs_base + p1), s1 = load_cs1<CS_IS_BF16>(cs_ptr, cs_base + 32 + p1);
    const float r0 = v[0] * c0 - v[1] * s0;
    const float r1 = v[1] * c0 + v[0] * s0;
    const float r2 = v[2] * c1 - v[3] * s1;
    const float r3 = v[3] * c1 + v[2] * s1;
    uint2 out_r;
    out_r.x = f32_to_bf16x2_rne(r0, r1);
    out_r.y = f32_to_bf16x2_rne(r2, r3);
    *reinterpret_cast<uint2*>(tok_data + 448 + sub * 8) = out_r;
  }
}

void ernie_insert_context_kv_ue8m0(
    torch::Tensor kv, torch::Tensor swa_cache, torch::Tensor slot_mapping,
    torch::Tensor positions, torch::Tensor cos_sin_cache, int64_t block_size,
    c10::optional<torch::Tensor> norm_w_opt = c10::nullopt, double norm_eps = 1e-6) {
  const int T = (int)slot_mapping.size(0);
  if (T == 0) return;
  TORCH_CHECK(kv.is_cuda() && kv.dim() == 2 && kv.size(1) == 512 && kv.scalar_type() == torch::kBFloat16 && kv.stride(1) == 1);
  TORCH_CHECK(swa_cache.is_cuda() && swa_cache.scalar_type() == torch::kUInt8);
  const bool cs_bf16 = (cos_sin_cache.scalar_type() == torch::kBFloat16);
  torch::Tensor cs_use = (cs_bf16 || cos_sin_cache.scalar_type() == torch::kFloat32)
      ? cos_sin_cache : cos_sin_cache.to(torch::kFloat32);
  auto pos_i64 = positions.scalar_type() == torch::kInt64 ? positions : positions.to(torch::kInt64);
  auto slot_i64 = slot_mapping.scalar_type() == torch::kInt64 ? slot_mapping : slot_mapping.to(torch::kInt64);
  auto st = c10::hip::getCurrentHIPStream().stream();
  if (norm_w_opt.has_value()) {
    auto nw = norm_w_opt->scalar_type() == torch::kBFloat16 ? norm_w_opt->contiguous() : norm_w_opt->to(torch::kBFloat16).contiguous();
    if (cs_bf16) {
      hipLaunchKernelGGL((k_insert_context_kv_ue8m0_v2<1, 1>), dim3(T), dim3(128), 0, st,
                         reinterpret_cast<const uint16_t*>(kv.data_ptr()),
                         reinterpret_cast<const uint16_t*>(nw.data_ptr()), (float)norm_eps,
                         swa_cache.data_ptr<uint8_t>(), slot_i64.data_ptr<int64_t>(),
                         pos_i64.data_ptr<int64_t>(), cs_use.data_ptr(),
                         kv.stride(0), swa_cache.stride(0), cs_use.stride(0), (int)block_size);
    } else {
      hipLaunchKernelGGL((k_insert_context_kv_ue8m0_v2<1, 0>), dim3(T), dim3(128), 0, st,
                         reinterpret_cast<const uint16_t*>(kv.data_ptr()),
                         reinterpret_cast<const uint16_t*>(nw.data_ptr()), (float)norm_eps,
                         swa_cache.data_ptr<uint8_t>(), slot_i64.data_ptr<int64_t>(),
                         pos_i64.data_ptr<int64_t>(), cs_use.data_ptr(),
                         kv.stride(0), swa_cache.stride(0), cs_use.stride(0), (int)block_size);
    }
  } else {
    if (cs_bf16) {
      hipLaunchKernelGGL((k_insert_context_kv_ue8m0_v2<0, 1>), dim3(T), dim3(128), 0, st,
                         reinterpret_cast<const uint16_t*>(kv.data_ptr()),
                         nullptr, 0.0f,
                         swa_cache.data_ptr<uint8_t>(), slot_i64.data_ptr<int64_t>(),
                         pos_i64.data_ptr<int64_t>(), cs_use.data_ptr(),
                         kv.stride(0), swa_cache.stride(0), cs_use.stride(0), (int)block_size);
    } else {
      hipLaunchKernelGGL((k_insert_context_kv_ue8m0_v2<0, 0>), dim3(T), dim3(128), 0, st,
                         reinterpret_cast<const uint16_t*>(kv.data_ptr()),
                         nullptr, 0.0f,
                         swa_cache.data_ptr<uint8_t>(), slot_i64.data_ptr<int64_t>(),
                         pos_i64.data_ptr<int64_t>(), cs_use.data_ptr(),
                         kv.stride(0), swa_cache.stride(0), cs_use.stride(0), (int)block_size);
    }
  }
}

// ------------------------------------------------------------------ 10 & 11: Fused DSpark Markov Step (NT-streaming, Infinity-Cache safe!)
// Replaces 7x (markov_w2 @ embed [103MB] + base_logits add + isneginf + multimax + masked_fill_ + argmax)!
//   - Non-temporal global_load_b128 on markov_w2 [V, 256] protects Strix Halo's 32 MB Infinity Cache
//   - Keeps markov_embed [256] hot in LDS (1 KB)
//   - Fuses exact MultiMax (preserving -inf) + Wave32 argmax directly in registers!
__device__ __forceinline__ float apply_multimax_scalar(
    float x, float r0, float r1, float r2, float r3,
    float t0, float t1, float t2, float t3) {
  if (!isfinite(x)) return x; // preserves -inf cleanly without isneginf/masked_fill_ passes!
  const float a = fmaxf(r0 - x, 0.0f);
  const float b = fmaxf(x - r1, 0.0f);
  const float c = fmaxf(r2 - x, 0.0f);
  const float d = fmaxf(x - r3, 0.0f);
  return x + t0 * a + t1 * b + t2 * (c * c) + t3 * (d * d);
}

// Multi-batch tiled (R=1,2,4,8) + 2-row MLP-unrolled (R=1) NT-streaming DSpark Markov Greedy!
// At B=2..8 reads 103MB markov_w2 ONCE across all R requests (saves 721MB at B=2, 2.16GB at B=4 per draft step!)
// At B=1 unrolls 2 vocab rows per iteration so 2x global_load_b128 stay in flight simultaneously to saturate LPDDR5X!
template <int R, int HAS_MM, int BASE_BF16>
__global__ __launch_bounds__(128) void k_dspark_markov_greedy_p1(
    const uint16_t* __restrict__ w2,       // [V, 256] bf16
    const uint16_t* __restrict__ emb,      // [B, 256] bf16
    const void* __restrict__ base_logits,  // [B, V] fp32 or bf16
    const float* __restrict__ ranges,      // [4] fp32 or nullptr
    const float* __restrict__ ts,          // [4] fp32 or nullptr
    float* __restrict__ part_val,          // [B, NUM_BLK] fp32
    int32_t* __restrict__ part_idx,        // [B, NUM_BLK] int32
    int64_t w2_s0, int64_t emb_s0, int64_t base_s0, int B, int V) {
  __shared__ __attribute__((aligned(16))) uint32_t s_emb[R][128]; // R * 512 B packed BF16x2
  __shared__ float s_bv[R][4];
  __shared__ int32_t s_bi[R][4];

  const int b0 = blockIdx.y * R;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;

#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int br = (b0 + r < B) ? (b0 + r) : (B - 1);
    const uint32_t* eb32 = reinterpret_cast<const uint32_t*>(emb + (int64_t)br * emb_s0);
    s_emb[r][tid] = eb32[tid];
  }
  __syncthreads();

  u4v e4[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    e4[r] = *reinterpret_cast<const u4v*>(&s_emb[r][lane * 4]); // 1 ds_load_b128 per batch row!
  }

  float r0 = 0.f, r1 = 0.f, r2 = 0.f, r3 = 0.f, t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
  if constexpr (HAS_MM) {
    r0 = ranges[0]; r1 = ranges[1]; r2 = ranges[2]; r3 = ranges[3];
    t0 = ts[0]; t1 = ts[1]; t2 = ts[2]; t3 = ts[3];
  }

  float best_v[R];
  int32_t best_i[R];
#pragma unroll
  for (int r = 0; r < R; ++r) { best_v[r] = -INFINITY; best_i[r] = 0; }

  // Contiguous wave tile: U_ROWS=4 for R<=4 (4x MLP + free +0/+512/+1024/+1536 offsets, <=89 VGPRs 16w/SIMD!),
  // U_ROWS=1 for R=8 (8x batch reuse, <=80 VGPRs, 0 SGPR spills!)
  constexpr int U_ROWS = (R <= 4) ? 4 : 1;
  const int stride_v = gridDim.x * (4 * U_ROWS);
  int v0 = (blockIdx.x * 4 + wave) * U_ROWS;
  for (; v0 + (U_ROWS - 1) < V; v0 += stride_v) {
    const uint16_t* wp = w2 + (int64_t)v0 * 256 + lane * 8;
    u4v rw[U_ROWS];
#pragma unroll
    for (int u = 0; u < U_ROWS; ++u) {
      rw[u] = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(wp + u * 256));
    }
    float dots[R][U_ROWS];
#pragma unroll
    for (int u = 0; u < U_ROWS; ++u) {
      float a0[R], a1[R];
#pragma unroll
      for (int r = 0; r < R; ++r) { a0[r] = 0.f; a1[r] = 0.f; }
      if constexpr (R > 0) BF16_DUAL_DOT2_8_PIN(a0[0], a1[0], rw[u], e4[0], "v16", "v17");
      if constexpr (R > 1) BF16_DUAL_DOT2_8_PIN(a0[1], a1[1], rw[u], e4[1], "v18", "v19");
      if constexpr (R > 2) BF16_DUAL_DOT2_8_PIN(a0[2], a1[2], rw[u], e4[2], "v20", "v21");
      if constexpr (R > 3) BF16_DUAL_DOT2_8_PIN(a0[3], a1[3], rw[u], e4[3], "v22", "v23");
      if constexpr (R > 4) BF16_DUAL_DOT2_8_PIN(a0[4], a1[4], rw[u], e4[4], "v24", "v25");
      if constexpr (R > 5) BF16_DUAL_DOT2_8_PIN(a0[5], a1[5], rw[u], e4[5], "v26", "v27");
      if constexpr (R > 6) BF16_DUAL_DOT2_8_PIN(a0[6], a1[6], rw[u], e4[6], "v28", "v29");
      if constexpr (R > 7) BF16_DUAL_DOT2_8_PIN(a0[7], a1[7], rw[u], e4[7], "v30", "v31");
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float acc = a0[r] + a1[r];
        acc += shx_f<16>(acc); acc += shx_f<8>(acc); acc += shx_f<4>(acc); acc += shx_f<2>(acc); acc += shx_f<1>(acc);
        dots[r][u] = acc;
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        if ((b0 + r) < B) {
          const int br = b0 + r;
          float bv[U_ROWS];
          if constexpr (U_ROWS == 4) {
            if constexpr (BASE_BF16) {
              const u2va bp2 = *reinterpret_cast<const u2va*>(
                  static_cast<const uint16_t*>(base_logits) + (int64_t)br * base_s0 + v0);
              bv[0] = __uint_as_float((bp2.x & 0xFFFFu) << 16);
              bv[1] = __uint_as_float(bp2.x & 0xFFFF0000u);
              bv[2] = __uint_as_float((bp2.y & 0xFFFFu) << 16);
              bv[3] = __uint_as_float(bp2.y & 0xFFFF0000u);
            } else {
              const f4v brw = *reinterpret_cast<const f4va*>(
                  static_cast<const float*>(base_logits) + (int64_t)br * base_s0 + v0);
              bv[0] = brw.x; bv[1] = brw.y; bv[2] = brw.z; bv[3] = brw.w;
            }
          } else {
            bv[0] = BASE_BF16
                ? __uint_as_float(((uint32_t)(static_cast<const uint16_t*>(base_logits)[(int64_t)br * base_s0 + v0])) << 16)
                : static_cast<const float*>(base_logits)[(int64_t)br * base_s0 + v0];
          }
#pragma unroll
          for (int u = 0; u < U_ROWS; ++u) {
            float lg = bv[u] + dots[r][u];
            if constexpr (HAS_MM) lg = apply_multimax_scalar(lg, r0, r1, r2, r3, t0, t1, t2, t3);
            const int vi = v0 + u;
            if (lg > best_v[r] || (lg == best_v[r] && vi < best_i[r])) { best_v[r] = lg; best_i[r] = vi; }
          }
        }
      }
    }
  }
  if constexpr (U_ROWS > 1) {
    for (int vi = v0; vi < V; ++vi) {
      const u4v raw = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w2 + (int64_t)vi * 256 + lane * 8));
      float a0[R], a1[R];
#pragma unroll
      for (int r = 0; r < R; ++r) { a0[r] = 0.f; a1[r] = 0.f; }
      if constexpr (R > 0) BF16_DUAL_DOT2_8_PIN(a0[0], a1[0], raw, e4[0], "v16", "v17");
      if constexpr (R > 1) BF16_DUAL_DOT2_8_PIN(a0[1], a1[1], raw, e4[1], "v18", "v19");
      if constexpr (R > 2) BF16_DUAL_DOT2_8_PIN(a0[2], a1[2], raw, e4[2], "v20", "v21");
      if constexpr (R > 3) BF16_DUAL_DOT2_8_PIN(a0[3], a1[3], raw, e4[3], "v22", "v23");
      if constexpr (R > 4) BF16_DUAL_DOT2_8_PIN(a0[4], a1[4], raw, e4[4], "v24", "v25");
      if constexpr (R > 5) BF16_DUAL_DOT2_8_PIN(a0[5], a1[5], raw, e4[5], "v26", "v27");
      if constexpr (R > 6) BF16_DUAL_DOT2_8_PIN(a0[6], a1[6], raw, e4[6], "v28", "v29");
      if constexpr (R > 7) BF16_DUAL_DOT2_8_PIN(a0[7], a1[7], raw, e4[7], "v30", "v31");
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float acc = a0[r] + a1[r];
        acc += shx_f<16>(acc); acc += shx_f<8>(acc); acc += shx_f<4>(acc); acc += shx_f<2>(acc); acc += shx_f<1>(acc);
        if (lane == 0 && (b0 + r) < B) {
          const int br = b0 + r;
          const float base_v = BASE_BF16
              ? __uint_as_float(((uint32_t)(static_cast<const uint16_t*>(base_logits)[(int64_t)br * base_s0 + vi])) << 16)
              : static_cast<const float*>(base_logits)[(int64_t)br * base_s0 + vi];
          float lg = base_v + acc;
          if constexpr (HAS_MM) lg = apply_multimax_scalar(lg, r0, r1, r2, r3, t0, t1, t2, t3);
          if (lg > best_v[r] || (lg == best_v[r] && vi < best_i[r])) { best_v[r] = lg; best_i[r] = vi; }
        }
      }
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      s_bv[r][wave] = best_v[r];
      s_bi[r][wave] = best_i[r];
    }
  }
  __syncthreads();
  if (tid < R && (b0 + tid) < B) {
    float bv = s_bv[tid][0];
    int32_t bi = s_bi[tid][0];
#pragma unroll
    for (int k = 1; k < 4; ++k) {
      if (s_bv[tid][k] > bv || (s_bv[tid][k] == bv && s_bi[tid][k] < bi)) {
        bv = s_bv[tid][k];
        bi = s_bi[tid][k];
      }
    }
    part_val[(int64_t)(b0 + tid) * gridDim.x + blockIdx.x] = bv;
    part_idx[(int64_t)(b0 + tid) * gridDim.x + blockIdx.x] = bi;
  }
}

__global__ __launch_bounds__(256) void k_dspark_markov_greedy_p2(
    const float* __restrict__ part_val, const int32_t* __restrict__ part_idx,
    int64_t* __restrict__ out_ids, int num_blk) {
  __shared__ float s_bv[8];
  __shared__ int32_t s_bi[8];
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  float bv = (tid < num_blk) ? part_val[(int64_t)b * num_blk + tid] : -INFINITY;
  int32_t bi = (tid < num_blk) ? part_idx[(int64_t)b * num_blk + tid] : 0;
#define W_AMAX(MSK) { \
  const float ov = shx_f<MSK>(bv); const int oi = shx_i<MSK>(bi); \
  if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } }
  W_AMAX(16) W_AMAX(8) W_AMAX(4) W_AMAX(2) W_AMAX(1)
#undef W_AMAX
  if (lane == 0) { s_bv[wave] = bv; s_bi[wave] = bi; }
  __syncthreads();
  if (tid == 0) {
    float fbv = s_bv[0];
    int32_t fbi = s_bi[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) {
      if (s_bv[k] > fbv || (s_bv[k] == fbv && s_bi[k] < fbi)) {
        fbv = s_bv[k];
        fbi = s_bi[k];
      }
    }
    out_ids[b] = (int64_t)fbi;
  }
}

template <int HAS_MM, int BASE_BF16>
static void dispatch_markov_greedy_p1(
    int B, int num_blk, hipStream_t st,
    const uint16_t* w2p, const uint16_t* ep, const void* blp, const float* rp, const float* tp,
    float* pvp, int32_t* pip, int64_t w2s, int64_t es, int64_t bs, int V) {
  if (B == 1) {
    hipLaunchKernelGGL((k_dspark_markov_greedy_p1<1, HAS_MM, BASE_BF16>), dim3(num_blk, 1), dim3(128), 0, st,
                       w2p, ep, blp, rp, tp, pvp, pip, w2s, es, bs, B, V);
  } else if (B == 2) {
    hipLaunchKernelGGL((k_dspark_markov_greedy_p1<2, HAS_MM, BASE_BF16>), dim3(num_blk, 1), dim3(128), 0, st,
                       w2p, ep, blp, rp, tp, pvp, pip, w2s, es, bs, B, V);
  } else if (B <= 4) {
    hipLaunchKernelGGL((k_dspark_markov_greedy_p1<4, HAS_MM, BASE_BF16>), dim3(num_blk, 1), dim3(128), 0, st,
                       w2p, ep, blp, rp, tp, pvp, pip, w2s, es, bs, B, V);
  } else {
    hipLaunchKernelGGL((k_dspark_markov_greedy_p1<8, HAS_MM, BASE_BF16>), dim3(num_blk, (B + 7) / 8), dim3(128), 0, st,
                       w2p, ep, blp, rp, tp, pvp, pip, w2s, es, bs, B, V);
  }
}

void ernie_dspark_markov_greedy(
    torch::Tensor markov_w2, torch::Tensor markov_embed, torch::Tensor base_logits,
    c10::optional<torch::Tensor> ranges_opt, c10::optional<torch::Tensor> ts_opt,
    torch::Tensor part_val, torch::Tensor part_idx, torch::Tensor out_ids) {
  TORCH_CHECK(markov_w2.is_cuda() && markov_w2.dim() == 2 && markov_w2.size(1) == 256 && markov_w2.scalar_type() == torch::kBFloat16 && markov_w2.is_contiguous());
  TORCH_CHECK(markov_embed.is_cuda() && markov_embed.dim() == 2 && markov_embed.size(1) == 256 && markov_embed.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(base_logits.is_cuda() && base_logits.dim() == 2 &&
              (base_logits.scalar_type() == torch::kFloat32 || base_logits.scalar_type() == torch::kBFloat16));
  const int B = (int)markov_embed.size(0);
  const int V = (int)markov_w2.size(0);
  if (B == 0 || V == 0) return;
  constexpr int NUM_BLK = 256;
  auto st = c10::hip::getCurrentHIPStream().stream();
  const bool has_mm = ranges_opt.has_value() && ts_opt.has_value();
  const bool is_bf16 = (base_logits.scalar_type() == torch::kBFloat16);
  const uint16_t* w2p = reinterpret_cast<const uint16_t*>(markov_w2.data_ptr());
  const uint16_t* ep = reinterpret_cast<const uint16_t*>(markov_embed.data_ptr());
  const void* blp = base_logits.data_ptr();
  const float* rp = has_mm ? ranges_opt->data_ptr<float>() : nullptr;
  const float* tp = has_mm ? ts_opt->data_ptr<float>() : nullptr;
  if (has_mm && is_bf16) {
    dispatch_markov_greedy_p1<1, 1>(B, NUM_BLK, st, w2p, ep, blp, rp, tp, part_val.data_ptr<float>(), part_idx.data_ptr<int32_t>(),
                                    markov_w2.stride(0), markov_embed.stride(0), base_logits.stride(0), V);
  } else if (has_mm) {
    dispatch_markov_greedy_p1<1, 0>(B, NUM_BLK, st, w2p, ep, blp, rp, tp, part_val.data_ptr<float>(), part_idx.data_ptr<int32_t>(),
                                    markov_w2.stride(0), markov_embed.stride(0), base_logits.stride(0), V);
  } else if (is_bf16) {
    dispatch_markov_greedy_p1<0, 1>(B, NUM_BLK, st, w2p, ep, blp, nullptr, nullptr, part_val.data_ptr<float>(), part_idx.data_ptr<int32_t>(),
                                    markov_w2.stride(0), markov_embed.stride(0), base_logits.stride(0), V);
  } else {
    dispatch_markov_greedy_p1<0, 0>(B, NUM_BLK, st, w2p, ep, blp, nullptr, nullptr, part_val.data_ptr<float>(), part_idx.data_ptr<int32_t>(),
                                    markov_w2.stride(0), markov_embed.stride(0), base_logits.stride(0), V);
  }
  hipLaunchKernelGGL(k_dspark_markov_greedy_p2, dim3(B), dim3(256), 0, st,
                     part_val.data_ptr<float>(), part_idx.data_ptr<int32_t>(),
                     out_ids.data_ptr<int64_t>(), NUM_BLK);
}

template <int R, int HAS_MM, int BASE_BF16>
__global__ __launch_bounds__(128) void k_dspark_markov_bias_multimax(
    const uint16_t* __restrict__ w2,
    const uint16_t* __restrict__ emb,
    const void* __restrict__ base_logits,
    const float* __restrict__ ranges,
    const float* __restrict__ ts,
    float* __restrict__ out_logits,
    int64_t w2_s0, int64_t emb_s0, int64_t base_s0, int64_t out_s0, int B, int V) {
  __shared__ __attribute__((aligned(16))) uint32_t s_emb[R][128]; // R * 512 B packed BF16x2
  const int b0 = blockIdx.y * R;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int br = (b0 + r < B) ? (b0 + r) : (B - 1);
    const uint32_t* eb32 = reinterpret_cast<const uint32_t*>(emb + (int64_t)br * emb_s0);
    s_emb[r][tid] = eb32[tid];
  }
  __syncthreads();

  u4v e4[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    e4[r] = *reinterpret_cast<const u4v*>(&s_emb[r][lane * 4]);
  }

  float r0 = 0.f, r1 = 0.f, r2 = 0.f, r3 = 0.f, t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
  if constexpr (HAS_MM) {
    r0 = ranges[0]; r1 = ranges[1]; r2 = ranges[2]; r3 = ranges[3];
    t0 = ts[0]; t1 = ts[1]; t2 = ts[2]; t3 = ts[3];
  }
  // Contiguous wave tile: U_ROWS=4 for R<=4 (4x MLP + f4v store), U_ROWS=1 for R=8 (0 SGPR spills, <=64 VGPRs!)
  constexpr int U_ROWS = (R <= 4) ? 4 : 1;
  const int stride_v = gridDim.x * (4 * U_ROWS);
  int v0 = (blockIdx.x * 4 + wave) * U_ROWS;
  for (; v0 + (U_ROWS - 1) < V; v0 += stride_v) {
    const uint16_t* wp = w2 + (int64_t)v0 * 256 + lane * 8;
    u4v rw[U_ROWS];
#pragma unroll
    for (int u = 0; u < U_ROWS; ++u) {
      rw[u] = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(wp + u * 256));
    }
    float dots[R][U_ROWS];
#pragma unroll
    for (int u = 0; u < U_ROWS; ++u) {
      float a0[R], a1[R];
#pragma unroll
      for (int r = 0; r < R; ++r) { a0[r] = 0.f; a1[r] = 0.f; }
      if constexpr (R > 0) BF16_DUAL_DOT2_8_PIN(a0[0], a1[0], rw[u], e4[0], "v16", "v17");
      if constexpr (R > 1) BF16_DUAL_DOT2_8_PIN(a0[1], a1[1], rw[u], e4[1], "v18", "v19");
      if constexpr (R > 2) BF16_DUAL_DOT2_8_PIN(a0[2], a1[2], rw[u], e4[2], "v20", "v21");
      if constexpr (R > 3) BF16_DUAL_DOT2_8_PIN(a0[3], a1[3], rw[u], e4[3], "v22", "v23");
      if constexpr (R > 4) BF16_DUAL_DOT2_8_PIN(a0[4], a1[4], rw[u], e4[4], "v24", "v25");
      if constexpr (R > 5) BF16_DUAL_DOT2_8_PIN(a0[5], a1[5], rw[u], e4[5], "v26", "v27");
      if constexpr (R > 6) BF16_DUAL_DOT2_8_PIN(a0[6], a1[6], rw[u], e4[6], "v28", "v29");
      if constexpr (R > 7) BF16_DUAL_DOT2_8_PIN(a0[7], a1[7], rw[u], e4[7], "v30", "v31");
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float acc = a0[r] + a1[r];
        acc += shx_f<16>(acc); acc += shx_f<8>(acc); acc += shx_f<4>(acc); acc += shx_f<2>(acc); acc += shx_f<1>(acc);
        dots[r][u] = acc;
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        if ((b0 + r) < B) {
          const int br = b0 + r;
          if constexpr (U_ROWS == 4) {
            float bv[4], og[4];
            if constexpr (BASE_BF16) {
              const u2va bp2 = *reinterpret_cast<const u2va*>(
                  static_cast<const uint16_t*>(base_logits) + (int64_t)br * base_s0 + v0);
              bv[0] = __uint_as_float((bp2.x & 0xFFFFu) << 16);
              bv[1] = __uint_as_float(bp2.x & 0xFFFF0000u);
              bv[2] = __uint_as_float((bp2.y & 0xFFFFu) << 16);
              bv[3] = __uint_as_float(bp2.y & 0xFFFF0000u);
            } else {
              const f4v brw = *reinterpret_cast<const f4va*>(
                  static_cast<const float*>(base_logits) + (int64_t)br * base_s0 + v0);
              bv[0] = brw.x; bv[1] = brw.y; bv[2] = brw.z; bv[3] = brw.w;
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
              float lg = bv[u] + dots[r][u];
              if constexpr (HAS_MM) lg = apply_multimax_scalar(lg, r0, r1, r2, r3, t0, t1, t2, t3);
              og[u] = lg;
            }
            *reinterpret_cast<f4va*>(out_logits + (int64_t)br * out_s0 + v0) = (f4v){og[0], og[1], og[2], og[3]};
          } else {
            const float bv0 = BASE_BF16
                ? __uint_as_float(((uint32_t)(static_cast<const uint16_t*>(base_logits)[(int64_t)br * base_s0 + v0])) << 16)
                : static_cast<const float*>(base_logits)[(int64_t)br * base_s0 + v0];
            float lg = bv0 + dots[r][0];
            if constexpr (HAS_MM) lg = apply_multimax_scalar(lg, r0, r1, r2, r3, t0, t1, t2, t3);
            out_logits[(int64_t)br * out_s0 + v0] = lg;
          }
        }
      }
    }
  }
  if constexpr (U_ROWS > 1) {
    for (int vi = v0; vi < V; ++vi) {
      const u4v raw = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w2 + (int64_t)vi * 256 + lane * 8));
      float a0[R], a1[R];
#pragma unroll
      for (int r = 0; r < R; ++r) { a0[r] = 0.f; a1[r] = 0.f; }
      if constexpr (R > 0) BF16_DUAL_DOT2_8_PIN(a0[0], a1[0], raw, e4[0], "v16", "v17");
      if constexpr (R > 1) BF16_DUAL_DOT2_8_PIN(a0[1], a1[1], raw, e4[1], "v18", "v19");
      if constexpr (R > 2) BF16_DUAL_DOT2_8_PIN(a0[2], a1[2], raw, e4[2], "v20", "v21");
      if constexpr (R > 3) BF16_DUAL_DOT2_8_PIN(a0[3], a1[3], raw, e4[3], "v22", "v23");
      if constexpr (R > 4) BF16_DUAL_DOT2_8_PIN(a0[4], a1[4], raw, e4[4], "v24", "v25");
      if constexpr (R > 5) BF16_DUAL_DOT2_8_PIN(a0[5], a1[5], raw, e4[5], "v26", "v27");
      if constexpr (R > 6) BF16_DUAL_DOT2_8_PIN(a0[6], a1[6], raw, e4[6], "v28", "v29");
      if constexpr (R > 7) BF16_DUAL_DOT2_8_PIN(a0[7], a1[7], raw, e4[7], "v30", "v31");
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float acc = a0[r] + a1[r];
        acc += shx_f<16>(acc); acc += shx_f<8>(acc); acc += shx_f<4>(acc); acc += shx_f<2>(acc); acc += shx_f<1>(acc);
        if (lane == 0 && (b0 + r) < B) {
          const int br = b0 + r;
          const float base_v = BASE_BF16
              ? __uint_as_float(((uint32_t)(static_cast<const uint16_t*>(base_logits)[(int64_t)br * base_s0 + vi])) << 16)
              : static_cast<const float*>(base_logits)[(int64_t)br * base_s0 + vi];
          float lg = base_v + acc;
          if constexpr (HAS_MM) lg = apply_multimax_scalar(lg, r0, r1, r2, r3, t0, t1, t2, t3);
          out_logits[(int64_t)br * out_s0 + vi] = lg;
        }
      }
    }
  }
}

template <int HAS_MM, int BASE_BF16>
static void dispatch_markov_bias_mm(
    int B, hipStream_t st,
    const uint16_t* w2p, const uint16_t* ep, const void* blp, const float* rp, const float* tp,
    float* op, int64_t w2s, int64_t es, int64_t bs, int64_t os, int V) {
  if (B == 1) {
    hipLaunchKernelGGL((k_dspark_markov_bias_multimax<1, HAS_MM, BASE_BF16>), dim3(256, 1), dim3(128), 0, st,
                       w2p, ep, blp, rp, tp, op, w2s, es, bs, os, B, V);
  } else if (B == 2) {
    hipLaunchKernelGGL((k_dspark_markov_bias_multimax<2, HAS_MM, BASE_BF16>), dim3(256, 1), dim3(128), 0, st,
                       w2p, ep, blp, rp, tp, op, w2s, es, bs, os, B, V);
  } else if (B <= 4) {
    hipLaunchKernelGGL((k_dspark_markov_bias_multimax<4, HAS_MM, BASE_BF16>), dim3(256, 1), dim3(128), 0, st,
                       w2p, ep, blp, rp, tp, op, w2s, es, bs, os, B, V);
  } else {
    hipLaunchKernelGGL((k_dspark_markov_bias_multimax<8, HAS_MM, BASE_BF16>), dim3(256, (B + 7) / 8), dim3(128), 0, st,
                       w2p, ep, blp, rp, tp, op, w2s, es, bs, os, B, V);
  }
}

void ernie_dspark_markov_bias_multimax(
    torch::Tensor markov_w2, torch::Tensor markov_embed, torch::Tensor base_logits,
    c10::optional<torch::Tensor> ranges_opt, c10::optional<torch::Tensor> ts_opt,
    torch::Tensor out_logits) {
  TORCH_CHECK(markov_w2.is_cuda() && markov_w2.dim() == 2 && markov_w2.size(1) == 256 && markov_w2.scalar_type() == torch::kBFloat16 && markov_w2.is_contiguous());
  TORCH_CHECK(base_logits.is_cuda() && base_logits.dim() == 2 &&
              (base_logits.scalar_type() == torch::kFloat32 || base_logits.scalar_type() == torch::kBFloat16));
  const int B = (int)markov_embed.size(0);
  const int V = (int)markov_w2.size(0);
  if (B == 0 || V == 0) return;
  auto st = c10::hip::getCurrentHIPStream().stream();
  const bool has_mm = ranges_opt.has_value() && ts_opt.has_value();
  const bool is_bf16 = (base_logits.scalar_type() == torch::kBFloat16);
  const uint16_t* w2p = reinterpret_cast<const uint16_t*>(markov_w2.data_ptr());
  const uint16_t* ep = reinterpret_cast<const uint16_t*>(markov_embed.data_ptr());
  const void* blp = base_logits.data_ptr();
  const float* rp = has_mm ? ranges_opt->data_ptr<float>() : nullptr;
  const float* tp = has_mm ? ts_opt->data_ptr<float>() : nullptr;
  if (has_mm && is_bf16) {
    dispatch_markov_bias_mm<1, 1>(B, st, w2p, ep, blp, rp, tp, out_logits.data_ptr<float>(),
                                  markov_w2.stride(0), markov_embed.stride(0), base_logits.stride(0), out_logits.stride(0), V);
  } else if (has_mm) {
    dispatch_markov_bias_mm<1, 0>(B, st, w2p, ep, blp, rp, tp, out_logits.data_ptr<float>(),
                                  markov_w2.stride(0), markov_embed.stride(0), base_logits.stride(0), out_logits.stride(0), V);
  } else if (is_bf16) {
    dispatch_markov_bias_mm<0, 1>(B, st, w2p, ep, blp, nullptr, nullptr, out_logits.data_ptr<float>(),
                                  markov_w2.stride(0), markov_embed.stride(0), base_logits.stride(0), out_logits.stride(0), V);
  } else {
    dispatch_markov_bias_mm<0, 0>(B, st, w2p, ep, blp, nullptr, nullptr, out_logits.data_ptr<float>(),
                                  markov_w2.stride(0), markov_embed.stride(0), base_logits.stride(0), out_logits.stride(0), V);
  }
}

// ------------------------------------------------------------------ 12: Fused Q (1024) + KV (512) RMSNorm (40x/tok: 37 HCA + 3 DSpark!)
// Replaces Triton _FUSED_Q_KV_RMSNORM_KERNEL with zero-JIT 128-bit vectorized Wave32 HIP!
// Grid: (T, 2) -> by=0 norms qr [T, 1024] (128 threads x 8 bf16), by=1 norms kv [T, 512] (64 active threads x 8 bf16)
__global__ __launch_bounds__(128) void k_fused_q_kv_rmsnorm_1024_512(
    const uint16_t* __restrict__ qr_in, int64_t qr_s0,
    const uint16_t* __restrict__ kv_in, int64_t kv_s0,
    uint16_t* __restrict__ qr_out, int64_t qro_s0,
    uint16_t* __restrict__ kv_out, int64_t kvo_s0,
    const uint16_t* __restrict__ qw,
    const uint16_t* __restrict__ kvw,
    float eps) {
  __shared__ float s_wsum[4];
  const int t = blockIdx.x;
  const int is_kv = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;

  const int D = is_kv ? 512 : 1024;
  const int n_active = is_kv ? 64 : 128;
  u4v raw = {0u, 0u, 0u, 0u};
  float ss = 0.0f;
  if (tid < n_active) {
    const uint16_t* src = is_kv ? (kv_in + (int64_t)t * kv_s0 + tid * 8)
                                : (qr_in + (int64_t)t * qr_s0 + tid * 8);
    raw = *reinterpret_cast<const u4va*>(src);
    float ss0 = bf16_dot2(raw.x, raw.x, 0.0f);
    float ss1 = bf16_dot2(raw.y, raw.y, 0.0f);
    ss0 = bf16_dot2(raw.z, raw.z, ss0);
    ss1 = bf16_dot2(raw.w, raw.w, ss1);
    ss = ss0 + ss1;
  }
  ss += shx_f<16>(ss);
  ss += shx_f<8>(ss);
  ss += shx_f<4>(ss);
  ss += shx_f<2>(ss);
  ss += shx_f<1>(ss);
  if (lane == 0) s_wsum[wave] = ss;
  __syncthreads();
  const float tot = is_kv ? (s_wsum[0] + s_wsum[1]) : (s_wsum[0] + s_wsum[1] + s_wsum[2] + s_wsum[3]);
  const float rrms = rsqrtf(tot * (1.0f / (float)D) + eps);

  if (tid < n_active) {
    const uint16_t* wp = (is_kv ? kvw : qw) + tid * 8;
    const u4v wraw = *reinterpret_cast<const u4va*>(wp);
    const uint32_t xws[4] = {raw.x, raw.y, raw.z, raw.w};
    const uint32_t wws[4] = {wraw.x, wraw.y, wraw.z, wraw.w};
    uint32_t out_w[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float x0 = __uint_as_float((xws[k] & 0xFFFFu) << 16);
      const float x1 = __uint_as_float(xws[k] & 0xFFFF0000u);
      const float w0 = __uint_as_float((wws[k] & 0xFFFFu) << 16);
      const float w1 = __uint_as_float(wws[k] & 0xFFFF0000u);
      out_w[k] = f32_to_bf16x2_rne(x0 * rrms * w0, x1 * rrms * w1);
    }
    uint16_t* dst = is_kv ? (kv_out + (int64_t)t * kvo_s0 + tid * 8)
                          : (qr_out + (int64_t)t * qro_s0 + tid * 8);
    *reinterpret_cast<u4va*>(dst) = (u4v){out_w[0], out_w[1], out_w[2], out_w[3]};
  }
}

std::vector<torch::Tensor> ernie_fused_q_kv_rmsnorm(
    torch::Tensor qr, torch::Tensor kv, torch::Tensor qw, torch::Tensor kvw, double eps) {
  TORCH_CHECK(qr.is_cuda() && qr.dim() == 2 && qr.size(1) == 1024 && qr.scalar_type() == torch::kBFloat16 && qr.stride(1) == 1);
  TORCH_CHECK(kv.is_cuda() && kv.dim() == 2 && kv.size(1) == 512 && kv.scalar_type() == torch::kBFloat16 && kv.stride(1) == 1);
  const int T = (int)qr.size(0);
  auto qr_out = torch::empty({T, 1024}, qr.options());
  auto kv_out = torch::empty({T, 512}, kv.options());
  if (T == 0) return {qr_out, kv_out};
  hipLaunchKernelGGL(
      k_fused_q_kv_rmsnorm_1024_512, dim3(T, 2), dim3(128), 0, c10::hip::getCurrentHIPStream().stream(),
      reinterpret_cast<const uint16_t*>(qr.data_ptr()), qr.stride(0),
      reinterpret_cast<const uint16_t*>(kv.data_ptr()), kv.stride(0),
      reinterpret_cast<uint16_t*>(qr_out.data_ptr()), qr_out.stride(0),
      reinterpret_cast<uint16_t*>(kv_out.data_ptr()), kv_out.stride(0),
      reinterpret_cast<const uint16_t*>(qw.data_ptr()),
      reinterpret_cast<const uint16_t*>(kvw.data_ptr()), (float)eps);
  return {qr_out, kv_out};
}

// ------------------------------------------------------------------ 13: Fused C128 Compress + RMSNorm + UE8M0 FP8 Quant + RoPE + Store
// Replaces Triton _fused_kv_compress_norm_rope_insert_sparse_attn (37x/tok inside FULL HIP graphs & boundary steps!)
//   - Instant early exit on non-boundary tokens (127/128 decode steps!)
//   - Single-pass online softmax over 128 history rows (2x less state_cache traffic!)
//   - 128 threads = 7 half-waves (exact 64-elem UE8M0 blocks via DPP!) + 1 half-wave (64 RoPE dims = 32 pairs!)
template <int CS_IS_BF16>
__global__ __launch_bounds__(128) void k_compress_norm_rope_store_c128(
    const float* __restrict__ state_cache, int64_t sc_s0, int64_t sc_s1,
    const int32_t* __restrict__ tok_to_req,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ slot_mapping,
    const int32_t* __restrict__ block_table, int64_t bt_s0,
    int state_block_size, int state_width,
    const float* __restrict__ rms_w_f32, const uint16_t* __restrict__ rms_w_bf16, float rms_eps,
    const void* __restrict__ cos_sin_ptr, int64_t cs_s0,
    uint8_t* __restrict__ k_cache, int64_t kv_blk_stride,
    const int64_t* __restrict__ kv_slot_mapping, int kv_block_size) {
  __shared__ int64_t s_row_off[128];
  __shared__ float s_wsum[4];

  const int tok = blockIdx.x;
  const int64_t slot_id = slot_mapping[tok];
  if (slot_id < 0) return;
  const int64_t position = positions[tok];
  if (((position + 1) & 127) != 0) return;
  const int64_t kv_slot = kv_slot_mapping[tok];
  if (kv_slot < 0) return;

  const int tid = threadIdx.x; // 0..127
  const int wave = tid >> 5;
  const int lane = tid & 31;

  // Precompute all 128 history row offsets in state_cache (1 thread per history token)
  const int req_idx = tok_to_req[tok];
  const int64_t start_pos = position - 127;
  const int64_t p_hist = start_pos + tid;
  if (p_hist >= 0) {
    const int64_t bi = p_hist / state_block_size;
    const int64_t bo = p_hist % state_block_size;
    const int64_t phys = (int64_t)block_table[(int64_t)req_idx * bt_s0 + bi];
    s_row_off[tid] = phys * sc_s0 + bo * sc_s1;
  } else {
    s_row_off[tid] = -1;
  }
  __syncthreads();

  // Pass 1: Single-pass online softmax + weighted sum over 128 history rows (4 cols/thread = 512 exact!)
  const int col4 = tid * 4;
  float m[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  float l[4] = {0.f, 0.f, 0.f, 0.f};
  float num[4] = {0.f, 0.f, 0.f, 0.f};
  for (int k = 0; k < 128; ++k) {
    const int64_t roff = s_row_off[k];
    if (roff < 0) continue;
    const f4v sc4 = *reinterpret_cast<const f4v*>(state_cache + roff + state_width + col4);
    const f4v kv4 = *reinterpret_cast<const f4v*>(state_cache + roff + col4);
    const float sv[4] = {sc4.x, sc4.y, sc4.z, sc4.w};
    const float kvv[4] = {kv4.x, kv4.y, kv4.z, kv4.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float mn = fmaxf(m[e], sv[e]);
      const float al = (m[e] > -1e30f) ? __expf(m[e] - mn) : 0.0f;
      const float wt = __expf(sv[e] - mn);
      l[e] = l[e] * al + wt;
      num[e] = __builtin_fmaf(wt, kvv[e], num[e] * al);
      m[e] = mn;
    }
  }

  float ckv[4];
  float ss = 0.0f;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    ckv[e] = (l[e] > 0.0f) ? (num[e] * __builtin_amdgcn_rcpf(l[e])) : 0.0f;
    ss = __builtin_fmaf(ckv[e], ckv[e], ss);
  }

  // Pass 2: RMSNorm over 512 dims
  ss += shx_f<16>(ss);
  ss += shx_f<8>(ss);
  ss += shx_f<4>(ss);
  ss += shx_f<2>(ss);
  ss += shx_f<1>(ss);
  if (lane == 0) s_wsum[wave] = ss;
  __syncthreads();
  const float rrms = rsqrtf((s_wsum[0] + s_wsum[1] + s_wsum[2] + s_wsum[3]) * (1.0f / 512.0f) + rms_eps);

  float normed[4];
  if (rms_w_f32 != nullptr) {
    const f4v rw = *reinterpret_cast<const f4v*>(rms_w_f32 + col4);
    normed[0] = ckv[0] * rrms * rw.x;
    normed[1] = ckv[1] * rrms * rw.y;
    normed[2] = ckv[2] * rrms * rw.z;
    normed[3] = ckv[3] * rrms * rw.w;
  } else {
    const uint2 rw = *reinterpret_cast<const uint2*>(rms_w_bf16 + col4);
    normed[0] = ckv[0] * rrms * __uint_as_float((rw.x & 0xFFFFu) << 16);
    normed[1] = ckv[1] * rrms * __uint_as_float(rw.x & 0xFFFF0000u);
    normed[2] = ckv[2] * rrms * __uint_as_float((rw.y & 0xFFFFu) << 16);
    normed[3] = ckv[3] * rrms * __uint_as_float(rw.y & 0xFFFF0000u);
  }

  // Pass 3: KV cache output pointers
  const int64_t kv_blk = kv_slot / kv_block_size;
  const int64_t kv_off = kv_slot % kv_block_size;
  uint8_t* blk_ptr = k_cache + kv_blk * kv_blk_stride;
  uint8_t* fp8_ptr = blk_ptr + kv_off * 576;
  uint8_t* scale_ptr = blk_ptr + (int64_t)kv_block_size * 576 + kv_off * 8;

  const int grp = tid >> 4; // 0..7 (0..6 = 7 exact 64-elem NoPE blocks, 7 = 64 RoPE dims!)
  const int sub = tid & 15; // 0..15 (16 threads x 4 elems = 64 elems!)
  if (grp < 7) {
    // Cast fp32 -> bf16 -> fp32 matching Triton reference before UE8M0 quant
    float qv[4];
    float am = 0.0f;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      qv[e] = __uint_as_float(((uint32_t)f32_to_bf16_rne(normed[e])) << 16);
      am = fmaxf(am, fabsf(qv[e]));
    }
    // 16-lane half-wave DPP absmax (zero LDS!)
    am = fmaxf(am, shx_f<8>(am));
    am = fmaxf(am, shx_f<4>(am));
    am = fmaxf(am, shx_f<2>(am));
    am = fmaxf(am, shx_f<1>(am));
    const float bmax = fmaxf(am, 1e-4f);
    const float exponent = ceilf(log2f(bmax * (1.0f / 448.0f)));
    const float inv_sc = exp2f(-exponent);
    uint8_t out_b[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) out_b[e] = f32_to_fp8_e4m3fn_ocp(qv[e] * inv_sc);
    *reinterpret_cast<uint32_t*>(fp8_ptr + col4) = *reinterpret_cast<const uint32_t*>(out_b);
    if (sub == 0) {
      float enc = exponent + 127.0f;
      if (enc < 0.0f) enc = 0.0f;
      if (enc > 255.0f) enc = 255.0f;
      scale_ptr[grp] = (uint8_t)enc;
    }
  } else {
    // grp == 7: 64 RoPE dims (sub in 0..15 -> 2 pairs: 2*sub and 2*sub+1)
    if (sub == 0) scale_ptr[7] = 0u;
    const int64_t comp_pos = (position / 128) * 128;
    const int64_t cs_base = comp_pos * cs_s0;
    const int p0 = 2 * sub;
    const int p1 = 2 * sub + 1;
    const float c0 = load_cs1<CS_IS_BF16>(cos_sin_ptr, cs_base + p0), s0 = load_cs1<CS_IS_BF16>(cos_sin_ptr, cs_base + 32 + p0);
    const float c1 = load_cs1<CS_IS_BF16>(cos_sin_ptr, cs_base + p1), s1 = load_cs1<CS_IS_BF16>(cos_sin_ptr, cs_base + 32 + p1);
    const float ne0 = normed[0] * c0 - normed[1] * s0;
    const float no0 = normed[1] * c0 + normed[0] * s0;
    const float ne1 = normed[2] * c1 - normed[3] * s1;
    const float no1 = normed[3] * c1 + normed[2] * s1;
    uint2 out_r;
    out_r.x = f32_to_bf16x2_rne(ne0, no0);
    out_r.y = f32_to_bf16x2_rne(ne1, no1);
    *reinterpret_cast<uint2*>(fp8_ptr + 448 + sub * 8) = out_r;
  }
}

void ernie_compress_norm_rope_store_c128(
    torch::Tensor state_cache, int64_t num_actual,
    torch::Tensor token_to_req, torch::Tensor positions, torch::Tensor slot_mapping,
    torch::Tensor block_table, int64_t state_block_size, int64_t state_width,
    torch::Tensor rms_w, double rms_eps, torch::Tensor cos_sin_cache,
    torch::Tensor k_cache, torch::Tensor kv_slot_mapping, int64_t kv_block_size) {
  if (num_actual <= 0) return;
  TORCH_CHECK(state_cache.is_cuda() && state_cache.scalar_type() == torch::kFloat32);
  TORCH_CHECK(k_cache.is_cuda() && k_cache.scalar_type() == torch::kUInt8);
  auto req_i32 = token_to_req.scalar_type() == torch::kInt32 ? token_to_req.contiguous() : token_to_req.to(torch::kInt32).contiguous();
  auto pos_i64 = positions.scalar_type() == torch::kInt64 ? positions.contiguous() : positions.to(torch::kInt64).contiguous();
  auto sm_i64 = slot_mapping.scalar_type() == torch::kInt64 ? slot_mapping.contiguous() : slot_mapping.to(torch::kInt64).contiguous();
  auto kvsm_i64 = kv_slot_mapping.scalar_type() == torch::kInt64 ? kv_slot_mapping.contiguous() : kv_slot_mapping.to(torch::kInt64).contiguous();
  auto bt_i32 = block_table.scalar_type() == torch::kInt32 ? block_table : block_table.to(torch::kInt32);
  const bool cs_bf16 = (cos_sin_cache.scalar_type() == torch::kBFloat16);
  torch::Tensor cs_use = (cs_bf16 || cos_sin_cache.scalar_type() == torch::kFloat32)
      ? cos_sin_cache : cos_sin_cache.to(torch::kFloat32);
  const float* rw_f = (rms_w.scalar_type() == torch::kFloat32) ? rms_w.data_ptr<float>() : nullptr;
  const uint16_t* rw_b = (rms_w.scalar_type() == torch::kBFloat16) ? reinterpret_cast<const uint16_t*>(rms_w.data_ptr()) : nullptr;
  torch::Tensor rw_f_tmp;
  if (rw_f == nullptr && rw_b == nullptr) {
    rw_f_tmp = rms_w.to(torch::kFloat32).contiguous();
    rw_f = rw_f_tmp.data_ptr<float>();
  }
  auto st = c10::hip::getCurrentHIPStream().stream();
  if (cs_bf16) {
    hipLaunchKernelGGL(
        (k_compress_norm_rope_store_c128<1>), dim3((unsigned)num_actual), dim3(128), 0, st,
        state_cache.data_ptr<float>(), state_cache.stride(0), state_cache.stride(1),
        req_i32.data_ptr<int32_t>(), pos_i64.data_ptr<int64_t>(), sm_i64.data_ptr<int64_t>(),
        bt_i32.data_ptr<int32_t>(), bt_i32.stride(0), (int)state_block_size, (int)state_width,
        rw_f, rw_b, (float)rms_eps, cs_use.data_ptr(), cs_use.stride(0),
        k_cache.data_ptr<uint8_t>(), k_cache.stride(0), kvsm_i64.data_ptr<int64_t>(), (int)kv_block_size);
  } else {
    hipLaunchKernelGGL(
        (k_compress_norm_rope_store_c128<0>), dim3((unsigned)num_actual), dim3(128), 0, st,
        state_cache.data_ptr<float>(), state_cache.stride(0), state_cache.stride(1),
        req_i32.data_ptr<int32_t>(), pos_i64.data_ptr<int64_t>(), sm_i64.data_ptr<int64_t>(),
        bt_i32.data_ptr<int32_t>(), bt_i32.stride(0), (int)state_block_size, (int)state_width,
        rw_f, rw_b, (float)rms_eps, cs_use.data_ptr(), cs_use.stride(0),
        k_cache.data_ptr<uint8_t>(), k_cache.stride(0), kvsm_i64.data_ptr<int64_t>(), (int)kv_block_size);
  }
}

// ------------------------------------------------------------------ 14: Fused FWHT-128 + UE8M0 FP8 Quant + Indexer Weights Scale
// Replaces 3 kernels (fwht128 + Triton per_token_group_quant_fp8 + weights * q_scale * scale)
// in Ernie5_5IndexerROCm.forward (6x/tok), eliminating the [T, 64, 128] BF16 round-trip!
__global__ __launch_bounds__(128) void k_fwht128_ue8m0_quant_weights(
    const uint16_t* __restrict__ q_in,       // [T, 64, 128] bf16 (contiguous)
    const uint16_t* __restrict__ w_in,       // [T, 64] bf16 (stride w_s0)
    uint8_t* __restrict__ q_fp8_out,         // [T, 64, 128] fp8 (as uint8)
    float* __restrict__ w_out,               // [T, 64] fp32
    int64_t w_s0, int T, float weights_scale) {
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5); // row in 0 .. T*64 - 1
  const int lane = threadIdx.x & 31;
  if (row >= T * 64) return;
  const int t = row >> 6;
  const int h = row & 63;

  const uint2 raw = *reinterpret_cast<const uint2*>(q_in + (size_t)row * 128 + lane * 4);
  float v[4];
  v[0] = __uint_as_float((raw.x & 0xFFFFu) << 16);
  v[1] = __uint_as_float(raw.x & 0xFFFF0000u);
  v[2] = __uint_as_float((raw.y & 0xFFFFu) << 16);
  v[3] = __uint_as_float(raw.y & 0xFFFF0000u);
  fwht128_wave(v, lane);

  constexpr float INV = 1.0f / 11.313708498984761f;
  float am = 0.0f;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    v[e] *= INV;
    // Match BF16 intermediate rounding before quant
    v[e] = __uint_as_float(((uint32_t)f32_to_bf16_rne(v[e])) << 16);
    am = fmaxf(am, fabsf(v[e]));
  }
  am = fmaxf(am, shx_f<16>(am));
  am = fmaxf(am, shx_f<8>(am));
  am = fmaxf(am, shx_f<4>(am));
  am = fmaxf(am, shx_f<2>(am));
  am = fmaxf(am, shx_f<1>(am));
  const float exponent = ceilf(log2f(fmaxf(am, 1e-4f) * (1.0f / 448.0f)));
  const float scale = exp2f(exponent);
  const float inv_sc = exp2f(-exponent);

  uint8_t ob[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) ob[e] = f32_to_fp8_e4m3fn_ocp(v[e] * inv_sc);
  *reinterpret_cast<uint32_t*>(q_fp8_out + (size_t)row * 128 + lane * 4) = *reinterpret_cast<const uint32_t*>(ob);
  if (lane == 0) {
    const float wf = __uint_as_float(((uint32_t)w_in[(int64_t)t * w_s0 + h]) << 16);
    w_out[(int64_t)t * 64 + h] = wf * scale * weights_scale;
  }
}

std::vector<torch::Tensor> ernie_fwht128_ue8m0_quant_weights(
    torch::Tensor q_bf16, torch::Tensor weights_bf16, double weights_scale) {
  TORCH_CHECK(q_bf16.is_cuda() && q_bf16.is_contiguous() && q_bf16.dim() == 3 && q_bf16.size(1) == 64 && q_bf16.size(2) == 128 && q_bf16.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(weights_bf16.is_cuda() && weights_bf16.dim() == 2 && weights_bf16.size(1) == 64 && weights_bf16.scalar_type() == torch::kBFloat16 && weights_bf16.stride(1) == 1);
  const int T = (int)q_bf16.size(0);
  auto q_fp8 = torch::empty({T, 64, 128}, q_bf16.options().dtype(torch::kFloat8_e4m3fn));
  auto w_out = torch::empty({T, 64}, q_bf16.options().dtype(torch::kFloat32));
  if (T == 0) return {q_fp8, w_out};
  const int total_rows = T * 64;
  hipLaunchKernelGGL(
      k_fwht128_ue8m0_quant_weights, dim3((total_rows + 3) / 4), dim3(128), 0, c10::hip::getCurrentHIPStream().stream(),
      reinterpret_cast<const uint16_t*>(q_bf16.data_ptr()),
      reinterpret_cast<const uint16_t*>(weights_bf16.data_ptr()),
      reinterpret_cast<uint8_t*>(q_fp8.data_ptr()),
      w_out.data_ptr<float>(), weights_bf16.stride(0), T, (float)weights_scale);
  return {q_fp8, w_out};
}

// ------------------------------------------------------------------ 15: DSA Indexer K Quant & Cache (replaces Triton indexer_k_quant_and_cache_triton 6x/tok!)
// 1 Wave32 per token (32 lanes x 4 elems = 128 dims). Exact SHUFFLE ([bs//16, 8, 16, 16]) or NORMAL layout!
template <int IS_SHUFFLE, int USE_UE8M0>
__global__ __launch_bounds__(32) void k_indexer_k_quant_and_cache_128(
    const uint16_t* __restrict__ k_in,       // [T, 128] bf16
    uint8_t* __restrict__ kv_cache_flat,     // [num_blocks, stride_blk] uint8
    const int64_t* __restrict__ slot_map,    // [T] int64
    int64_t k_s0, int64_t cache_s0, int block_size) {
  const int t = blockIdx.x;
  const int64_t slot = slot_map[t];
  if (slot < 0) return;
  const int lane = threadIdx.x; // 0..31 -> elems [lane*4 .. lane*4+3]
  const uint2 raw = *reinterpret_cast<const uint2*>(k_in + (int64_t)t * k_s0 + lane * 4);
  float v[4];
  v[0] = __uint_as_float((raw.x & 0xFFFFu) << 16);
  v[1] = __uint_as_float(raw.x & 0xFFFF0000u);
  v[2] = __uint_as_float((raw.y & 0xFFFFu) << 16);
  v[3] = __uint_as_float(raw.y & 0xFFFF0000u);
  float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  am = fmaxf(am, shx_f<16>(am));
  am = fmaxf(am, shx_f<8>(am));
  am = fmaxf(am, shx_f<4>(am));
  am = fmaxf(am, shx_f<2>(am));
  am = fmaxf(am, shx_f<1>(am));
  float scale = fmaxf(am, 1e-4f) * (1.0f / 448.0f);
  if constexpr (USE_UE8M0) {
    scale = exp2f(ceilf(log2f(scale)));
  }
  const float inv_sc = __builtin_amdgcn_rcpf(scale);
  uint8_t ob[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) ob[e] = f32_to_fp8_e4m3fn_ocp(v[e] * inv_sc);

  const int64_t blk = slot / block_size;
  const int64_t off = slot % block_size;
  uint8_t* blk_ptr = kv_cache_flat + blk * cache_s0;
  if constexpr (IS_SHUFFLE) {
    const int tb_id = (int)(off >> 4);
    const int tb_off = (int)(off & 15);
    const int ht_id = lane >> 2;         // 0..7 (each head tile = 16 bytes = 4 lanes)
    const int ht_off = (lane & 3) * 4;   // 0, 4, 8, 12
    uint8_t* dst = blk_ptr + (int64_t)tb_id * 16 * 128 + (int64_t)ht_id * 256 + (int64_t)tb_off * 16 + ht_off;
    *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<const uint32_t*>(ob);
  } else {
    *reinterpret_cast<uint32_t*>(blk_ptr + off * 128 + lane * 4) = *reinterpret_cast<const uint32_t*>(ob);
  }
  if (lane == 0) {
    *reinterpret_cast<float*>(blk_ptr + (int64_t)block_size * 128 + off * 4) = scale;
  }
}

void ernie_indexer_k_quant_and_cache(
    torch::Tensor k, torch::Tensor kv_cache, torch::Tensor slot_mapping,
    int64_t block_size, bool use_ue8m0) {
  const int T = (int)slot_mapping.size(0);
  if (T == 0) return;
  TORCH_CHECK(k.is_cuda() && k.dim() == 2 && k.size(1) == 128 && k.scalar_type() == torch::kBFloat16 && k.stride(1) == 1);
  auto kv_flat = kv_cache.view({kv_cache.size(0), -1});
  auto sm_i64 = slot_mapping.scalar_type() == torch::kInt64 ? slot_mapping.contiguous() : slot_mapping.to(torch::kInt64).contiguous();
  auto st = c10::hip::getCurrentHIPStream().stream();
  const bool is_shuffle = (block_size > 1);
  if (is_shuffle && use_ue8m0) {
    hipLaunchKernelGGL((k_indexer_k_quant_and_cache_128<1, 1>), dim3(T), dim3(32), 0, st,
                       reinterpret_cast<const uint16_t*>(k.data_ptr()), kv_flat.data_ptr<uint8_t>(),
                       sm_i64.data_ptr<int64_t>(), k.stride(0), kv_flat.stride(0), (int)block_size);
  } else if (is_shuffle) {
    hipLaunchKernelGGL((k_indexer_k_quant_and_cache_128<1, 0>), dim3(T), dim3(32), 0, st,
                       reinterpret_cast<const uint16_t*>(k.data_ptr()), kv_flat.data_ptr<uint8_t>(),
                       sm_i64.data_ptr<int64_t>(), k.stride(0), kv_flat.stride(0), (int)block_size);
  } else if (use_ue8m0) {
    hipLaunchKernelGGL((k_indexer_k_quant_and_cache_128<0, 1>), dim3(T), dim3(32), 0, st,
                       reinterpret_cast<const uint16_t*>(k.data_ptr()), kv_flat.data_ptr<uint8_t>(),
                       sm_i64.data_ptr<int64_t>(), k.stride(0), kv_flat.stride(0), (int)block_size);
  } else {
    hipLaunchKernelGGL((k_indexer_k_quant_and_cache_128<0, 0>), dim3(T), dim3(32), 0, st,
                       reinterpret_cast<const uint16_t*>(k.data_ptr()), kv_flat.data_ptr<uint8_t>(),
                       sm_i64.data_ptr<int64_t>(), k.stride(0), kv_flat.stride(0), (int)block_size);
  }
}

// ------------------------------------------------------------------ 16: CP Gather Indexer K Quant Cache (6x/prefill, 128-bit R/W!)
// Replaces Triton _cp_gather_indexer_quant_cache_kernel for D=128 in BOTH SHUFFLE ([bs//16, 8, 16, 16]) and NORMAL layouts!
// 128 threads/WG = 16 tokens/WG, 8 threads/token -> each thread copies one exact 16-byte tile via global_load_b128/store_b128!
template <int IS_SHUFFLE>
__global__ __launch_bounds__(128) void k_cp_gather_indexer_k_quant_cache_128(
    const uint8_t* __restrict__ kv_cache_flat, // [num_blocks, stride_blk] uint8
    uint8_t* __restrict__ k_fp8_out,           // [num_tokens, 128] uint8
    float* __restrict__ k_scale_out,           // [num_tokens] fp32
    const int32_t* __restrict__ block_table,   // [num_batches, bt_s0] int32
    const int32_t* __restrict__ cu_seqlen,     // [num_batches + 1] int32
    const int32_t* __restrict__ token_to_seq,  // [num_tokens] int32
    int64_t cache_s0, int64_t bt_s0,
    int num_tokens, int num_batches, int bt_width, int num_blocks, int block_size) {
  const int tid = threadIdx.x;
  const int tok_local = tid >> 3; // 0..15
  const int ht = tid & 7;         // 0..7 (8 x 16B = 128B exact!)
  const int t = blockIdx.x * 16 + tok_local;
  if (t >= num_tokens) return;

  const int batch_id = token_to_seq[t];
  if (batch_id < 0 || batch_id >= num_batches) return;
  const int batch_start = cu_seqlen[batch_id];
  const int batch_end = cu_seqlen[batch_id + 1];
  if (t < batch_start || t >= batch_end) return;

  const int batch_off = t - batch_start;
  const int bt_id = batch_off / block_size;
  const int blk_off = batch_off % block_size;
  bool valid_blk = (bt_id >= 0 && bt_id < bt_width && blk_off >= 0 && blk_off < block_size);
  int phys_blk = -1;
  if (valid_blk) {
    phys_blk = block_table[(int64_t)batch_id * bt_s0 + bt_id];
    valid_blk = (phys_blk >= 0 && phys_blk < num_blocks);
  }
  if (!valid_blk) {
    if (ht == 0) k_scale_out[t] = 0.0f;
    return;
  }

  const uint8_t* blk_ptr = kv_cache_flat + (int64_t)phys_blk * cache_s0;
  if (ht == 0) {
    k_scale_out[t] = *reinterpret_cast<const float*>(blk_ptr + (int64_t)block_size * 128 + (int64_t)blk_off * 4);
  }
  const uint8_t* src;
  if constexpr (IS_SHUFFLE) {
    const int tb_id = blk_off >> 4;
    const int tb_off = blk_off & 15;
    src = blk_ptr + (int64_t)tb_id * 2048 + (int64_t)ht * 256 + (int64_t)tb_off * 16;
  } else {
    src = blk_ptr + (int64_t)blk_off * 128 + ht * 16;
  }
  const u4v v = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(src));
  *reinterpret_cast<u4va*>(k_fp8_out + (int64_t)t * 128 + ht * 16) = v;
}

void ernie_cp_gather_indexer_k_quant_cache(
    torch::Tensor k_cache, torch::Tensor k_fp8, torch::Tensor k_fp8_scale,
    torch::Tensor block_table, torch::Tensor cu_seqlen, torch::Tensor token_to_seq,
    int64_t block_size, bool is_shuffle) {
  const int num_tokens = (int)k_fp8.size(0);
  if (num_tokens == 0) return;
  TORCH_CHECK(k_cache.is_cuda() && k_fp8.is_cuda() && k_fp8.dim() == 2 && k_fp8.size(1) == 128 && k_fp8.is_contiguous());
  const int num_blocks = (int)k_cache.size(0);
  const int num_batches = (int)cu_seqlen.size(0) - 1;
  auto kv_flat = k_cache.view({num_blocks, -1});
  auto sc_f32 = k_fp8_scale.view(torch::kFloat32).reshape({-1});
  auto bt = block_table.scalar_type() == torch::kInt32 ? block_table : block_table.to(torch::kInt32);
  auto cu = cu_seqlen.scalar_type() == torch::kInt32 ? cu_seqlen.contiguous() : cu_seqlen.to(torch::kInt32).contiguous();
  auto t2s = token_to_seq.scalar_type() == torch::kInt32 ? token_to_seq.contiguous() : token_to_seq.to(torch::kInt32).contiguous();
  auto st = c10::hip::getCurrentHIPStream().stream();
  const dim3 grid((num_tokens + 15) / 16);
  if (is_shuffle) {
    hipLaunchKernelGGL(
        (k_cp_gather_indexer_k_quant_cache_128<1>), grid, dim3(128), 0, st,
        kv_flat.data_ptr<uint8_t>(), reinterpret_cast<uint8_t*>(k_fp8.data_ptr()),
        sc_f32.data_ptr<float>(), bt.data_ptr<int32_t>(), cu.data_ptr<int32_t>(), t2s.data_ptr<int32_t>(),
        kv_flat.stride(0), bt.stride(0), num_tokens, num_batches, (int)bt.size(1), num_blocks, (int)block_size);
  } else {
    hipLaunchKernelGGL(
        (k_cp_gather_indexer_k_quant_cache_128<0>), grid, dim3(128), 0, st,
        kv_flat.data_ptr<uint8_t>(), reinterpret_cast<uint8_t*>(k_fp8.data_ptr()),
        sc_f32.data_ptr<float>(), bt.data_ptr<int32_t>(), cu.data_ptr<int32_t>(), t2s.data_ptr<int32_t>(),
        kv_flat.stride(0), bt.stride(0), num_tokens, num_batches, (int)bt.size(1), num_blocks, (int)block_size);
  }
}

// ------------------------------------------------------------------ 17: Wave32 BF16 WMMA VHA Head Mixing (46x/tok: 37 HCA D=512 + 6 DSA D=128 + 3 DSpark D=512!)
// Computes out[t, 64, D] = M[64, 64] @ O[t, 64, D] in BF16 (D multiple of 64) with hardware wmma_f32_16x16x16_bf16_w32.
// Grid: (T, 4, D / 64), 4.25 KB LDS (transposed O_lds[64][18] -> 2x ds_load_b128/WMMA), zero rocBLAS overhead!
__global__ __launch_bounds__(128) void k_vha_mix_bf16_64x512(
    const uint16_t* __restrict__ M_ptr, // [64, 64] bf16 contiguous
    const uint16_t* __restrict__ O_ptr, // [T, 64, D] bf16
    uint16_t* __restrict__ out_ptr,     // [T, 64, D] bf16 contiguous
    int64_t o_s0, int64_t o_s1, int D) {
  __shared__ __attribute__((aligned(16))) uint16_t M_lds[16][64]; // 2.00 KiB
  // Transposed O_lds[64][18] (2.25 KiB, 36B stride = 9 banks -> zero bank conflicts for 128-bit ds_load_b128!)
  // Cuts per-WMMA VHA LDS reads from 16x scalar ds_load_u16 -> 2x ds_load_b128 (8x fewer LDS ops across 46 layers/tok!)
  __shared__ __attribute__((aligned(16))) uint16_t O_lds[64][18];

  const int t = blockIdx.x;
  const int h0 = blockIdx.y * 16; // 0, 16, 32, 48
  const int c0 = blockIdx.z * 64; // 0, 64, ..., D - 64
  const int tid = threadIdx.x;    // 0..127
  const int wave = tid >> 5;      // 0..3 (each wave owns 16 output cols)
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;
  const int mr = tid >> 3;        // 0..15
  const int mc = (tid & 7) * 8;   // 0, 8, ..., 56

  // Load 16x64 tile of M once (128 threads x 8 bf16 = 1024 exact!)
  *reinterpret_cast<u4va*>(&M_lds[mr][mc]) = *reinterpret_cast<const u4va*>(M_ptr + (h0 + mr) * 64 + mc);

  f32x8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const uint16_t* o_tok = O_ptr + (int64_t)t * o_s0;

  // 2-stage software prefetch of O_ptr[k+1] behind LDS-fed WMMA (zero vmcnt conflict across all 46 layers/tok!)
  u4v oraw_cur = *reinterpret_cast<const u4va*>(o_tok + (int64_t)mr * o_s1 + c0 + mc);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint32_t ow[4] = {oraw_cur.x, oraw_cur.y, oraw_cur.z, oraw_cur.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      O_lds[mc + 2 * e + 0][mr] = (uint16_t)(ow[e] & 0xFFFFu);
      O_lds[mc + 2 * e + 1][mr] = (uint16_t)(ow[e] >> 16);
    }
    __syncthreads();

    u4v oraw_nxt = {0u, 0u, 0u, 0u};
    if (k < 3) {
      oraw_nxt = *reinterpret_cast<const u4va*>(o_tok + (int64_t)((k + 1) * 16 + mr) * o_s1 + c0 + mc);
    }
    __builtin_amdgcn_sched_barrier(0);

    const u4v a0 = *reinterpret_cast<const u4va*>(&M_lds[l16][k * 16]);
    const u4v a1 = *reinterpret_cast<const u4va*>(&M_lds[l16][k * 16 + 8]);
    const bf16x16 af = __builtin_bit_cast(bf16x16, (u8v){a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w});

    const int col_l = wave * 16 + l16;
    const u4v b0 = *reinterpret_cast<const u4va*>(&O_lds[col_l][0]);
    const u4v b1 = *reinterpret_cast<const u4va*>(&O_lds[col_l][8]);
    const bf16x16 bf = __builtin_bit_cast(bf16x16, (u8v){b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w});
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af, bf, acc);
    __syncthreads();
    oraw_cur = oraw_nxt;
  }

  uint16_t* dst = out_ptr + (int64_t)t * ((int64_t)64 * D) + c0 + wave * 16 + l16;
#pragma unroll
  for (int ei = 0; ei < 8; ++ei) {
    const int h = h0 + 2 * ei + hrow;
    dst[(int64_t)h * D] = f32_to_bf16_rne(acc[ei]);
  }
}

torch::Tensor ernie_vha_mix_bf16(torch::Tensor M, torch::Tensor O) {
  TORCH_CHECK(M.is_cuda() && M.is_contiguous() && M.dim() == 2 && M.size(0) == 64 && M.size(1) == 64 && M.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(O.is_cuda() && O.dim() == 3 && O.size(1) == 64 && (O.size(2) % 64 == 0) && O.scalar_type() == torch::kBFloat16 && O.stride(2) == 1);
  const int T = (int)O.size(0);
  const int D = (int)O.size(2);
  auto out = torch::empty({T, 64, D}, O.options());
  if (T == 0) return out;
  hipLaunchKernelGGL(
      k_vha_mix_bf16_64x512, dim3(T, 4, D / 64), dim3(128), 0, c10::hip::getCurrentHIPStream().stream(),
      reinterpret_cast<const uint16_t*>(M.data_ptr()),
      reinterpret_cast<const uint16_t*>(O.data_ptr()),
      reinterpret_cast<uint16_t*>(out.data_ptr()),
      O.stride(0), O.stride(1), D);
  return out;
}

// ------------------------------------------------------------------ 18: Fused DSA Indexer Q: NeoX Partial RoPE(64) + FWHT-128 + UE8M0 Quant + Weights Scale (6x/tok!)
// Eliminates the entire [T, 64, 128] BF16 RoPE round-trip! Uses 1-cycle DPP shx_f<8> to swap dims [0:32] <-> [32:64] in registers!
template <int CS_IS_BF16>
__global__ __launch_bounds__(128) void k_rope_neox64_fwht128_ue8m0_quant_weights(
    const uint16_t* __restrict__ q_in,       // [T, 64, 128] bf16 contiguous
    const uint16_t* __restrict__ w_in,       // [T, 64] bf16
    const int64_t* __restrict__ pos_ptr,     // [T] int64
    const void* __restrict__ cs_ptr,         // [max_pos, 64] fp32 or bf16 (cos[0:32] | sin[0:32])
    uint8_t* __restrict__ q_fp8_out,         // [T, 64, 128] fp8
    float* __restrict__ w_out,               // [T, 64] fp32
    int64_t w_s0, int64_t cs_s0, int T, float weights_scale) {
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= T * 64) return;
  const int t = row >> 6;
  const int h = row & 63;

  const uint2 raw = *reinterpret_cast<const uint2*>(q_in + (size_t)row * 128 + lane * 4);
  float v[4];
  v[0] = __uint_as_float((raw.x & 0xFFFFu) << 16);
  v[1] = __uint_as_float(raw.x & 0xFFFF0000u);
  v[2] = __uint_as_float((raw.y & 0xFFFFu) << 16);
  v[3] = __uint_as_float(raw.y & 0xFFFF0000u);

  // NeoX partial RoPE on first 64 dims: lanes 0..7 ([0:32]) pair with lanes 8..15 ([32:64]) via 1-cycle shx_f<8>!
  float p[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) p[e] = shx_f<8>(v[e]);
  if (lane < 16) {
    const int64_t pos = pos_ptr[t];
    const int64_t cs_base = pos * cs_s0;
    const int base = (lane & 7) * 4;
    float cv[4], sv[4];
    load_cs4<CS_IS_BF16>(cs_ptr, cs_base + base, cv);
    load_cs4<CS_IS_BF16>(cs_ptr, cs_base + 32 + base, sv);
    const bool second_half = (lane >= 8);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float rot = second_half ? (v[e] * cv[e] + p[e] * sv[e]) : (v[e] * cv[e] - p[e] * sv[e]);
      v[e] = __uint_as_float(((uint32_t)f32_to_bf16_rne(rot)) << 16);
    }
  }

  fwht128_wave(v, lane);

  constexpr float INV = 1.0f / 11.313708498984761f;
  float am = 0.0f;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    v[e] = __uint_as_float(((uint32_t)f32_to_bf16_rne(v[e] * INV)) << 16);
    am = fmaxf(am, fabsf(v[e]));
  }
  am = fmaxf(am, shx_f<16>(am));
  am = fmaxf(am, shx_f<8>(am));
  am = fmaxf(am, shx_f<4>(am));
  am = fmaxf(am, shx_f<2>(am));
  am = fmaxf(am, shx_f<1>(am));
  const float exponent = ceilf(log2f(fmaxf(am, 1e-4f) * (1.0f / 448.0f)));
  const float scale = exp2f(exponent);
  const float inv_sc = exp2f(-exponent);

  uint8_t ob[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) ob[e] = f32_to_fp8_e4m3fn_ocp(v[e] * inv_sc);
  *reinterpret_cast<uint32_t*>(q_fp8_out + (size_t)row * 128 + lane * 4) = *reinterpret_cast<const uint32_t*>(ob);
  if (lane == 0) {
    const float wf = __uint_as_float(((uint32_t)w_in[(int64_t)t * w_s0 + h]) << 16);
    w_out[(int64_t)t * 64 + h] = wf * scale * weights_scale;
  }
}

std::vector<torch::Tensor> ernie_rope_neox64_fwht128_ue8m0_quant_weights(
    torch::Tensor q_bf16, torch::Tensor weights_bf16,
    torch::Tensor positions, torch::Tensor cos_sin_cache, double weights_scale) {
  TORCH_CHECK(q_bf16.is_cuda() && q_bf16.is_contiguous() && q_bf16.dim() == 3 && q_bf16.size(1) == 64 && q_bf16.size(2) == 128 && q_bf16.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(weights_bf16.is_cuda() && weights_bf16.dim() == 2 && weights_bf16.size(1) == 64 && weights_bf16.scalar_type() == torch::kBFloat16 && weights_bf16.stride(1) == 1);
  const int T = (int)q_bf16.size(0);
  auto q_fp8 = torch::empty({T, 64, 128}, q_bf16.options().dtype(torch::kFloat8_e4m3fn));
  auto w_out = torch::empty({T, 64}, q_bf16.options().dtype(torch::kFloat32));
  if (T == 0) return {q_fp8, w_out};
  auto pos_i64 = positions.scalar_type() == torch::kInt64 ? positions.contiguous() : positions.to(torch::kInt64).contiguous();
  const bool cs_bf16 = (cos_sin_cache.scalar_type() == torch::kBFloat16);
  torch::Tensor cs_use = (cs_bf16 || cos_sin_cache.scalar_type() == torch::kFloat32)
      ? cos_sin_cache : cos_sin_cache.to(torch::kFloat32);
  const int total_rows = T * 64;
  auto st = c10::hip::getCurrentHIPStream().stream();
  if (cs_bf16) {
    hipLaunchKernelGGL(
        (k_rope_neox64_fwht128_ue8m0_quant_weights<1>), dim3((total_rows + 3) / 4), dim3(128), 0, st,
        reinterpret_cast<const uint16_t*>(q_bf16.data_ptr()),
        reinterpret_cast<const uint16_t*>(weights_bf16.data_ptr()),
        pos_i64.data_ptr<int64_t>(), cs_use.data_ptr(),
        reinterpret_cast<uint8_t*>(q_fp8.data_ptr()),
        w_out.data_ptr<float>(), weights_bf16.stride(0), cs_use.stride(0), T, (float)weights_scale);
  } else {
    hipLaunchKernelGGL(
        (k_rope_neox64_fwht128_ue8m0_quant_weights<0>), dim3((total_rows + 3) / 4), dim3(128), 0, st,
        reinterpret_cast<const uint16_t*>(q_bf16.data_ptr()),
        reinterpret_cast<const uint16_t*>(weights_bf16.data_ptr()),
        pos_i64.data_ptr<int64_t>(), cs_use.data_ptr(),
        reinterpret_cast<uint8_t*>(q_fp8.data_ptr()),
        w_out.data_ptr<float>(), weights_bf16.stride(0), cs_use.stride(0), T, (float)weights_scale);
  }
  return {q_fp8, w_out};
}

// ------------------------------------------------------------------ 19: Fused DSA Indexer K: LayerNorm(128) + NeoX Partial RoPE(64) + FWHT-128 (6x/tok!)
// Collapses 3 kernels (k_norm + rotary_emb + fwht128) into 1 Wave32-per-token register-fused kernel (0 LDS, 0 barriers!)
template <int CS_IS_BF16>
__global__ __launch_bounds__(32) void k_layernorm_rope_neox64_fwht128(
    const uint16_t* __restrict__ k_in,       // [T, 128] bf16
    const uint16_t* __restrict__ ln_w,       // [128] bf16
    const uint16_t* __restrict__ ln_b,       // [128] bf16 or nullptr
    const int64_t* __restrict__ pos_ptr,     // [T] int64
    const void* __restrict__ cs_ptr,         // [max_pos, 64] fp32 or bf16
    uint16_t* __restrict__ k_out,            // [T, 128] bf16 contiguous
    int64_t k_s0, int64_t cs_s0, float eps) {
  const int t = blockIdx.x;
  const int lane = threadIdx.x;
  const uint2 raw = *reinterpret_cast<const uint2*>(k_in + (int64_t)t * k_s0 + lane * 4);
  float v[4];
  v[0] = __uint_as_float((raw.x & 0xFFFFu) << 16);
  v[1] = __uint_as_float(raw.x & 0xFFFF0000u);
  v[2] = __uint_as_float((raw.y & 0xFFFFu) << 16);
  v[3] = __uint_as_float(raw.y & 0xFFFF0000u);

  // LayerNorm(128) in FP32 via Wave32 DPP
  float s = v[0] + v[1] + v[2] + v[3];
  s += shx_f<16>(s); s += shx_f<8>(s); s += shx_f<4>(s); s += shx_f<2>(s); s += shx_f<1>(s);
  const float mean = s * (1.0f / 128.0f);
  float vv = 0.0f;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    v[e] -= mean;
    vv = __builtin_fmaf(v[e], v[e], vv);
  }
  vv += shx_f<16>(vv); vv += shx_f<8>(vv); vv += shx_f<4>(vv); vv += shx_f<2>(vv); vv += shx_f<1>(vv);
  const float rstd = rsqrtf(vv * (1.0f / 128.0f) + eps);

  const uint2 wraw = *reinterpret_cast<const uint2*>(ln_w + lane * 4);
  const float wv[4] = {
    __uint_as_float((wraw.x & 0xFFFFu) << 16),
    __uint_as_float(wraw.x & 0xFFFF0000u),
    __uint_as_float((wraw.y & 0xFFFFu) << 16),
    __uint_as_float(wraw.y & 0xFFFF0000u),
  };
  float bv[4] = {0.f, 0.f, 0.f, 0.f};
  if (ln_b != nullptr) {
    const uint2 braw = *reinterpret_cast<const uint2*>(ln_b + lane * 4);
    bv[0] = __uint_as_float((braw.x & 0xFFFFu) << 16);
    bv[1] = __uint_as_float(braw.x & 0xFFFF0000u);
    bv[2] = __uint_as_float((braw.y & 0xFFFFu) << 16);
    bv[3] = __uint_as_float(braw.y & 0xFFFF0000u);
  }
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    v[e] = __uint_as_float(((uint32_t)f32_to_bf16_rne(v[e] * rstd * wv[e] + bv[e])) << 16);
  }

  // NeoX partial RoPE on first 64 dims
  float p[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) p[e] = shx_f<8>(v[e]);
  if (lane < 16) {
    const int64_t pos = pos_ptr[t];
    const int64_t cs_base = pos * cs_s0;
    const int base = (lane & 7) * 4;
    float cv[4], sv[4];
    load_cs4<CS_IS_BF16>(cs_ptr, cs_base + base, cv);
    load_cs4<CS_IS_BF16>(cs_ptr, cs_base + 32 + base, sv);
    const bool second_half = (lane >= 8);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float rot = second_half ? (v[e] * cv[e] + p[e] * sv[e]) : (v[e] * cv[e] - p[e] * sv[e]);
      v[e] = __uint_as_float(((uint32_t)f32_to_bf16_rne(rot)) << 16);
    }
  }

  fwht128_wave(v, lane);
  constexpr float INV = 1.0f / 11.313708498984761f;
  uint2 o;
  o.x = f32_to_bf16x2_rne(v[0] * INV, v[1] * INV);
  o.y = f32_to_bf16x2_rne(v[2] * INV, v[3] * INV);
  *reinterpret_cast<uint2*>(k_out + (int64_t)t * 128 + lane * 4) = o;
}

torch::Tensor ernie_layernorm_rope_neox64_fwht128(
    torch::Tensor k_bf16, torch::Tensor ln_w, c10::optional<torch::Tensor> ln_b_opt,
    torch::Tensor positions, torch::Tensor cos_sin_cache, double eps) {
  TORCH_CHECK(k_bf16.is_cuda() && k_bf16.dim() == 2 && k_bf16.size(1) == 128 && k_bf16.scalar_type() == torch::kBFloat16 && k_bf16.stride(1) == 1);
  const int T = (int)k_bf16.size(0);
  auto out = torch::empty({T, 128}, k_bf16.options());
  if (T == 0) return out;
  auto w_bf16 = ln_w.scalar_type() == torch::kBFloat16 ? ln_w.contiguous() : ln_w.to(torch::kBFloat16).contiguous();
  torch::Tensor b_bf16;
  const uint16_t* bp = nullptr;
  if (ln_b_opt.has_value()) {
    b_bf16 = ln_b_opt->scalar_type() == torch::kBFloat16 ? ln_b_opt->contiguous() : ln_b_opt->to(torch::kBFloat16).contiguous();
    bp = reinterpret_cast<const uint16_t*>(b_bf16.data_ptr());
  }
  auto pos_i64 = positions.scalar_type() == torch::kInt64 ? positions.contiguous() : positions.to(torch::kInt64).contiguous();
  const bool cs_bf16 = (cos_sin_cache.scalar_type() == torch::kBFloat16);
  torch::Tensor cs_use = (cs_bf16 || cos_sin_cache.scalar_type() == torch::kFloat32)
      ? cos_sin_cache : cos_sin_cache.to(torch::kFloat32);
  auto st = c10::hip::getCurrentHIPStream().stream();
  if (cs_bf16) {
    hipLaunchKernelGGL(
        (k_layernorm_rope_neox64_fwht128<1>), dim3(T), dim3(32), 0, st,
        reinterpret_cast<const uint16_t*>(k_bf16.data_ptr()),
        reinterpret_cast<const uint16_t*>(w_bf16.data_ptr()), bp,
        pos_i64.data_ptr<int64_t>(), cs_use.data_ptr(),
        reinterpret_cast<uint16_t*>(out.data_ptr()),
        k_bf16.stride(0), cs_use.stride(0), (float)eps);
  } else {
    hipLaunchKernelGGL(
        (k_layernorm_rope_neox64_fwht128<0>), dim3(T), dim3(32), 0, st,
        reinterpret_cast<const uint16_t*>(k_bf16.data_ptr()),
        reinterpret_cast<const uint16_t*>(w_bf16.data_ptr()), bp,
        pos_i64.data_ptr<int64_t>(), cs_use.data_ptr(),
        reinterpret_cast<uint16_t*>(out.data_ptr()),
        k_bf16.stride(0), cs_use.stride(0), (float)eps);
  }
  return out;
}

// ------------------------------------------------------------------ 20: NT-Streaming BF16 GEMV (575 MB/step, protects 32MB Infinity Cache!)
// Streams W [N, K] bf16 via non-temporal global_load_b128 (8 bf16s/lane) while X [T, K] stays hot in L1/L2!
// Covers ERNIE-5.5 gate (336MB), DSpark main_proj (101MB), DSA indexer.wq_b (101MB), kv_a_proj (28MB), wk_weights (9MB)!
template <int R, int OUT_BF16>
__global__ __launch_bounds__(128) void k_bf16_gemv_nt(
    const uint16_t* __restrict__ x, int64_t x_s0,
    const uint16_t* __restrict__ w, int64_t w_s0,
    void* __restrict__ out, int N, int K) {
  const int n = blockIdx.x * 4 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (n >= N) return;
  const uint16_t* w_row = w + (int64_t)n * w_s0;

  float acc0[R], acc1[R];
#pragma unroll
  for (int r = 0; r < R; ++r) { acc0[r] = 0.0f; acc1[r] = 0.0f; }

#define BF16_GEMV_STEP(WVAL, KOFF) do { \
    if constexpr (R > 0) { const u4v x0 = *reinterpret_cast<const u4va*>(x + (int64_t)0 * x_s0 + k + (KOFF)); BF16_DUAL_DOT2_8_PIN(acc0[0], acc1[0], (WVAL), x0, "v16", "v17"); } \
    if constexpr (R > 1) { const u4v x1 = *reinterpret_cast<const u4va*>(x + (int64_t)1 * x_s0 + k + (KOFF)); BF16_DUAL_DOT2_8_PIN(acc0[1], acc1[1], (WVAL), x1, "v18", "v19"); } \
    if constexpr (R > 2) { const u4v x2 = *reinterpret_cast<const u4va*>(x + (int64_t)2 * x_s0 + k + (KOFF)); BF16_DUAL_DOT2_8_PIN(acc0[2], acc1[2], (WVAL), x2, "v20", "v21"); } \
    if constexpr (R > 3) { const u4v x3 = *reinterpret_cast<const u4va*>(x + (int64_t)3 * x_s0 + k + (KOFF)); BF16_DUAL_DOT2_8_PIN(acc0[3], acc1[3], (WVAL), x3, "v22", "v23"); } \
    if constexpr (R > 4) { const u4v x4 = *reinterpret_cast<const u4va*>(x + (int64_t)4 * x_s0 + k + (KOFF)); BF16_DUAL_DOT2_8_PIN(acc0[4], acc1[4], (WVAL), x4, "v24", "v25"); } \
    if constexpr (R > 5) { const u4v x5 = *reinterpret_cast<const u4va*>(x + (int64_t)5 * x_s0 + k + (KOFF)); BF16_DUAL_DOT2_8_PIN(acc0[5], acc1[5], (WVAL), x5, "v26", "v27"); } \
    if constexpr (R > 6) { const u4v x6 = *reinterpret_cast<const u4va*>(x + (int64_t)6 * x_s0 + k + (KOFF)); BF16_DUAL_DOT2_8_PIN(acc0[6], acc1[6], (WVAL), x6, "v28", "v29"); } \
    if constexpr (R > 7) { const u4v x7 = *reinterpret_cast<const u4va*>(x + (int64_t)7 * x_s0 + k + (KOFF)); BF16_DUAL_DOT2_8_PIN(acc0[7], acc1[7], (WVAL), x7, "v30", "v31"); } \
  } while (0)

  // 4x MLP-unrolled main loop (100% of ERNIE-5.5 shapes have K % 1024 == 0: 1024, 4096, 12288!)
  // Uses free 13-bit hardware offsets (+0, +512B, +1024B, +1536B) and puts 4 NT global_load_b128s in flight!
  int k = lane * 8;
  for (; k + 768 < K; k += 1024) {
    if constexpr (R == 1) {
      const u4v w0 = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_row + k));
      const u4v x0 = *reinterpret_cast<const u4va*>(x + k);
      const u4v w1 = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_row + k + 256));
      const u4v x1 = *reinterpret_cast<const u4va*>(x + k + 256);
      const u4v w2 = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_row + k + 512));
      const u4v x2 = *reinterpret_cast<const u4va*>(x + k + 512);
      const u4v w3 = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_row + k + 768));
      const u4v x3 = *reinterpret_cast<const u4va*>(x + k + 768);
      BF16_DUAL_DOT2_8_PIN(acc0[0], acc1[0], w0, x0, "v16", "v17");
      BF16_DUAL_DOT2_8_PIN(acc0[0], acc1[0], w1, x1, "v16", "v17");
      BF16_DUAL_DOT2_8_PIN(acc0[0], acc1[0], w2, x2, "v16", "v17");
      BF16_DUAL_DOT2_8_PIN(acc0[0], acc1[0], w3, x3, "v16", "v17");
    } else {
      const u4v w0 = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_row + k));
      const u4v w1 = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_row + k + 256));
      const u4v w2 = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_row + k + 512));
      const u4v w3 = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_row + k + 768));
      BF16_GEMV_STEP(w0, 0);
      BF16_GEMV_STEP(w1, 256);
      BF16_GEMV_STEP(w2, 512);
      BF16_GEMV_STEP(w3, 768);
    }
  }
  for (; k < K; k += 256) {
    const u4v wraw = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_row + k));
    BF16_GEMV_STEP(wraw, 0);
  }
#undef BF16_GEMV_STEP
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float a = acc0[r] + acc1[r];
    a += shx_f<16>(a); a += shx_f<8>(a); a += shx_f<4>(a); a += shx_f<2>(a); a += shx_f<1>(a);
    if (lane == 0) {
      if constexpr (OUT_BF16) {
        static_cast<uint16_t*>(out)[(int64_t)r * N + n] = f32_to_bf16_rne(a);
      } else {
        static_cast<float*>(out)[(int64_t)r * N + n] = a;
      }
    }
  }
}

// Wave 19B: k_bf16_gemv_nt_xlds512_v28 (T = 1..16, maxV <= 75, 0/0/0/0, single-launch sk=1!)
//   - Stages 512 BF16 (64 u4v) per token in x_lds[R][64] (4x fewer X VMEM loads, 2x NT b128 W MLP!)
//   - Intra-WG 4-channel 256B LPDDR5X stagger (wave w starts at u4v (w*16 + lane) & 63) + inter-WG block stagger
//     -> breaks 8-way power-of-2 channel collisions on K=2048 (Router) & K=4096 (main_proj, kv_a, wk_weights)!
//   - Supports T = 1..16 (B=1 & B=2 DSpark!), single kernel launch, zero atomics.
template <int R, int OUT_BF16>
__global__ __launch_bounds__(128) void k_bf16_gemv_nt_xlds512_v28(
    const uint16_t* __restrict__ x, int64_t x_s0,
    const uint16_t* __restrict__ w, int64_t w_s0,
    void* __restrict__ out, int N, int K) {
  __shared__ __attribute__((aligned(16))) u4v x_lds[R][64];

  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int n = blockIdx.x * 4 + wave;
  const bool vn = (n < N);

  const uint16_t* w_row = w + (int64_t)(vn ? n : (N - 1)) * w_s0;

  float acc0[R], acc1[R];
#pragma unroll
  for (int r = 0; r < R; ++r) { acc0[r] = 0.0f; acc1[r] = 0.0f; }

  const int u0 = (wave * 16 + lane) & 63;
  const int u1 = (wave * 16 + 32 + lane) & 63;
  const int k_512 = K & ~511;
  const int n512 = k_512 >> 9;
  const int blk_ph = (n512 >= 4) ? (((int)blockIdx.x & 3) * 512)
                   : ((n512 >= 2) ? (((int)blockIdx.x & 1) * 512) : 0);

  int ks = 0;
  for (; ks < k_512; ks += 512) {
    int k_blk = ks + blk_ph;
    if (k_blk >= k_512) k_blk -= k_512;
    for (int idx = tid; idx < R * 64; idx += 128) {
      const int rr = idx >> 6;
      const int cc = idx & 63;
      x_lds[rr][cc] = *reinterpret_cast<const u4va*>(x + (int64_t)rr * x_s0 + k_blk + cc * 8);
    }
    const u4v wv0 = vn ? __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_row + k_blk + u0 * 8)) : (u4v){0u,0u,0u,0u};
    const u4v wv1 = vn ? __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_row + k_blk + u1 * 8)) : (u4v){0u,0u,0u,0u};
    __syncthreads();
#define _XL512_STEP2(IDX, R0, R1) \
    if constexpr (R > (IDX)) { \
      const u4v xv0 = x_lds[IDX][u0]; \
      BF16_DUAL_DOT2_8_PIN(acc0[IDX], acc1[IDX], wv0, xv0, R0, R1); \
      const u4v xv1 = x_lds[IDX][u1]; \
      BF16_DUAL_DOT2_8_PIN(acc0[IDX], acc1[IDX], wv1, xv1, R0, R1); \
    }
    _XL512_STEP2(0,  "v16", "v17"); _XL512_STEP2(1,  "v18", "v19");
    _XL512_STEP2(2,  "v20", "v21"); _XL512_STEP2(3,  "v22", "v23");
    _XL512_STEP2(4,  "v24", "v25"); _XL512_STEP2(5,  "v26", "v27");
    _XL512_STEP2(6,  "v28", "v29"); _XL512_STEP2(7,  "v30", "v31");
    _XL512_STEP2(8,  "v32", "v33"); _XL512_STEP2(9,  "v34", "v35");
    _XL512_STEP2(10, "v36", "v37"); _XL512_STEP2(11, "v38", "v39");
    _XL512_STEP2(12, "v40", "v41"); _XL512_STEP2(13, "v42", "v43");
    _XL512_STEP2(14, "v44", "v45"); _XL512_STEP2(15, "v46", "v47");
#undef _XL512_STEP2
    __syncthreads();
  }
  for (; ks < K; ks += 256) {
    for (int idx = tid; idx < R * 32; idx += 128) {
      const int rr = idx >> 5;
      const int cc = idx & 31;
      x_lds[rr][cc] = *reinterpret_cast<const u4va*>(x + (int64_t)rr * x_s0 + ks + cc * 8);
    }
    const u4v wv0 = vn ? __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_row + ks + lane * 8)) : (u4v){0u,0u,0u,0u};
    __syncthreads();
#define _XL512_STEPT(IDX, R0, R1) \
    if constexpr (R > (IDX)) { \
      const u4v xv0 = x_lds[IDX][lane]; \
      BF16_DUAL_DOT2_8_PIN(acc0[IDX], acc1[IDX], wv0, xv0, R0, R1); \
    }
    _XL512_STEPT(0,  "v16", "v17"); _XL512_STEPT(1,  "v18", "v19");
    _XL512_STEPT(2,  "v20", "v21"); _XL512_STEPT(3,  "v22", "v23");
    _XL512_STEPT(4,  "v24", "v25"); _XL512_STEPT(5,  "v26", "v27");
    _XL512_STEPT(6,  "v28", "v29"); _XL512_STEPT(7,  "v30", "v31");
    _XL512_STEPT(8,  "v32", "v33"); _XL512_STEPT(9,  "v34", "v35");
    _XL512_STEPT(10, "v36", "v37"); _XL512_STEPT(11, "v38", "v39");
    _XL512_STEPT(12, "v40", "v41"); _XL512_STEPT(13, "v42", "v43");
    _XL512_STEPT(14, "v44", "v45"); _XL512_STEPT(15, "v46", "v47");
#undef _XL512_STEPT
    __syncthreads();
  }

  if (!vn) return;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float a = acc0[r] + acc1[r];
    a += shx_f<16>(a); a += shx_f<8>(a); a += shx_f<4>(a); a += shx_f<2>(a); a += shx_f<1>(a);
    if (lane == 0) {
      if constexpr (OUT_BF16) {
        static_cast<uint16_t*>(out)[(int64_t)r * N + n] = f32_to_bf16_rne(a);
      } else {
        static_cast<float*>(out)[(int64_t)r * N + n] = a;
      }
    }
  }
}

template <int OUT_BF16>
static void dispatch_bf16_gemv_xlds512(
    int T, dim3 grid, hipStream_t st,
    const uint16_t* xp, int64_t xs0, const uint16_t* wp, int64_t ws0,
    void* op, int N, int K) {
  switch (T) {
    case 1:  hipLaunchKernelGGL((k_bf16_gemv_nt_xlds512_v28<1,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 2:  hipLaunchKernelGGL((k_bf16_gemv_nt_xlds512_v28<2,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 3:  hipLaunchKernelGGL((k_bf16_gemv_nt_xlds512_v28<3,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 4:  hipLaunchKernelGGL((k_bf16_gemv_nt_xlds512_v28<4,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 5:  hipLaunchKernelGGL((k_bf16_gemv_nt_xlds512_v28<5,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 6:  hipLaunchKernelGGL((k_bf16_gemv_nt_xlds512_v28<6,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 7:  hipLaunchKernelGGL((k_bf16_gemv_nt_xlds512_v28<7,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 8:  hipLaunchKernelGGL((k_bf16_gemv_nt_xlds512_v28<8,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 9:  hipLaunchKernelGGL((k_bf16_gemv_nt_xlds512_v28<9,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 10: hipLaunchKernelGGL((k_bf16_gemv_nt_xlds512_v28<10, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 11: hipLaunchKernelGGL((k_bf16_gemv_nt_xlds512_v28<11, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 12: hipLaunchKernelGGL((k_bf16_gemv_nt_xlds512_v28<12, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 13: hipLaunchKernelGGL((k_bf16_gemv_nt_xlds512_v28<13, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 14: hipLaunchKernelGGL((k_bf16_gemv_nt_xlds512_v28<14, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 15: hipLaunchKernelGGL((k_bf16_gemv_nt_xlds512_v28<15, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 16: hipLaunchKernelGGL((k_bf16_gemv_nt_xlds512_v28<16, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    default: TORCH_CHECK(false, "T must be 1..16");
  }
}

// Wave 70: Zero-in-loop-barrier multi-wave-per-row BF16 GEMV (1r4w & 2r2w) for gfx1151!
//   - Eliminates 2x __syncthreads() (buffer_gl0_inv) per 512-K step -> X stays 100% in 32 KiB L0 cache!
//   - 1r4w (grid=N, 4 waves/row) & 2r2w (grid=(N+1)/2, 2 waves/row) saturate all 40 CUs on N=192..2048
//     and hit 233 GB/s (LPDDR5X roofline) on DSpark main_proj 4096x12288 (T<=8)!
//   - Only ONE LDS-only barrier (s_waitcnt lgkmcnt(0); s_barrier) at final reduction (<= 256 B LDS).
__device__ __forceinline__ void fops_lds_barrier_v70() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

template <int R, int OUT_BF16>
__global__ __launch_bounds__(128) void k_bf16_gemv_1r4w_v28(
    const uint16_t* __restrict__ x, int64_t x_s0,
    const uint16_t* __restrict__ w, int64_t w_s0,
    void* __restrict__ out, int N, int K) {
  __shared__ float s_part[4][R];
  const int n = blockIdx.x;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const uint16_t* w_row = w + (int64_t)n * w_s0;

  const int k_1024 = K & ~1023;
  const int n1024 = k_1024 >> 10;
  const int ph = (n1024 >= 4) ? ((n & 3) << 10) : ((n1024 >= 2) ? ((n & 1) << 10) : 0);

  float acc0[R], acc1[R];
#pragma unroll
  for (int r = 0; r < R; ++r) { acc0[r] = 0.0f; acc1[r] = 0.0f; }

  for (int ks = 0; ks < k_1024; ks += 1024) {
    int kb = ks + ph;
    if (kb >= k_1024) kb -= k_1024;
    const int ko = kb + wave * 256 + lane * 8;
    const u4v wv = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_row + ko));
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const u4v xv = *reinterpret_cast<const u4va*>(x + (int64_t)r * x_s0 + ko);
      acc0[r] = bf16_dot2(wv.x, xv.x, acc0[r]);
      acc1[r] = bf16_dot2(wv.y, xv.y, acc1[r]);
      acc0[r] = bf16_dot2(wv.z, xv.z, acc0[r]);
      acc1[r] = bf16_dot2(wv.w, xv.w, acc1[r]);
    }
  }
  for (int ks = k_1024 + wave * 256; ks < K; ks += 1024) {
    const int ko = ks + lane * 8;
    const u4v wv = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_row + ko));
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const u4v xv = *reinterpret_cast<const u4va*>(x + (int64_t)r * x_s0 + ko);
      acc0[r] = bf16_dot2(wv.x, xv.x, acc0[r]);
      acc1[r] = bf16_dot2(wv.y, xv.y, acc1[r]);
      acc0[r] = bf16_dot2(wv.z, xv.z, acc0[r]);
      acc1[r] = bf16_dot2(wv.w, xv.w, acc1[r]);
    }
  }

#pragma unroll
  for (int r = 0; r < R; ++r) {
    float a = acc0[r] + acc1[r];
    a += __shfl_xor(a, 16, 32);
    a += __shfl_xor(a, 8, 32);
    a += __shfl_xor(a, 4, 32);
    a += __shfl_xor(a, 2, 32);
    a += __shfl_xor(a, 1, 32);
    if (lane == 0) s_part[wave][r] = a;
  }
  fops_lds_barrier_v70();
  if (tid < R) {
    const float tot = s_part[0][tid] + s_part[1][tid] + s_part[2][tid] + s_part[3][tid];
    if constexpr (OUT_BF16) {
      reinterpret_cast<uint16_t*>(out)[(int64_t)tid * N + n] = f32_to_bf16_rne(tot);
    } else {
      reinterpret_cast<float*>(out)[(int64_t)tid * N + n] = tot;
    }
  }
}

template <int R, int OUT_BF16>
__global__ __launch_bounds__(128) void k_bf16_gemv_2r2w_v28(
    const uint16_t* __restrict__ x, int64_t x_s0,
    const uint16_t* __restrict__ w, int64_t w_s0,
    void* __restrict__ out, int N, int K) {
  __shared__ float s_part[2][2][R];
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int rloc = wave >> 1;
  const int wsub = wave & 1;
  const int n = blockIdx.x * 2 + rloc;
  const bool vn = (n < N);
  const uint16_t* w_row = w + (int64_t)(vn ? n : (N - 1)) * w_s0;

  const int k_512 = K & ~511;
  const int n512 = k_512 >> 9;
  const int ph = (n512 >= 4) ? ((n & 3) << 9) : ((n512 >= 2) ? ((n & 1) << 9) : 0);

  float acc0[R], acc1[R];
#pragma unroll
  for (int r = 0; r < R; ++r) { acc0[r] = 0.0f; acc1[r] = 0.0f; }

  if (vn) {
    for (int ks = 0; ks < k_512; ks += 512) {
      int kb = ks + ph;
      if (kb >= k_512) kb -= k_512;
      const int ko = kb + wsub * 256 + lane * 8;
      const u4v wv = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_row + ko));
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const u4v xv = *reinterpret_cast<const u4va*>(x + (int64_t)r * x_s0 + ko);
        acc0[r] = bf16_dot2(wv.x, xv.x, acc0[r]);
        acc1[r] = bf16_dot2(wv.y, xv.y, acc1[r]);
        acc0[r] = bf16_dot2(wv.z, xv.z, acc0[r]);
        acc1[r] = bf16_dot2(wv.w, xv.w, acc1[r]);
      }
    }
    for (int ks = k_512 + wsub * 256; ks < K; ks += 512) {
      const int ko = ks + lane * 8;
      const u4v wv = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_row + ko));
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const u4v xv = *reinterpret_cast<const u4va*>(x + (int64_t)r * x_s0 + ko);
        acc0[r] = bf16_dot2(wv.x, xv.x, acc0[r]);
        acc1[r] = bf16_dot2(wv.y, xv.y, acc1[r]);
        acc0[r] = bf16_dot2(wv.z, xv.z, acc0[r]);
        acc1[r] = bf16_dot2(wv.w, xv.w, acc1[r]);
      }
    }
  }

#pragma unroll
  for (int r = 0; r < R; ++r) {
    float a = acc0[r] + acc1[r];
    a += __shfl_xor(a, 16, 32);
    a += __shfl_xor(a, 8, 32);
    a += __shfl_xor(a, 4, 32);
    a += __shfl_xor(a, 2, 32);
    a += __shfl_xor(a, 1, 32);
    if (lane == 0) s_part[rloc][wsub][r] = a;
  }
  fops_lds_barrier_v70();
  if (vn && wsub == 0 && lane < R) {
    const float tot = s_part[rloc][0][lane] + s_part[rloc][1][lane];
    if constexpr (OUT_BF16) {
      reinterpret_cast<uint16_t*>(out)[(int64_t)lane * N + n] = f32_to_bf16_rne(tot);
    } else {
      reinterpret_cast<float*>(out)[(int64_t)lane * N + n] = tot;
    }
  }
}

template <int OUT_BF16>
static void dispatch_bf16_gemv_1r4w(
    int T, dim3 grid, hipStream_t st,
    const uint16_t* xp, int64_t xs0, const uint16_t* wp, int64_t ws0,
    void* op, int N, int K) {
  switch (T) {
    case 1:  hipLaunchKernelGGL((k_bf16_gemv_1r4w_v28<1,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 2:  hipLaunchKernelGGL((k_bf16_gemv_1r4w_v28<2,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 3:  hipLaunchKernelGGL((k_bf16_gemv_1r4w_v28<3,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 4:  hipLaunchKernelGGL((k_bf16_gemv_1r4w_v28<4,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 5:  hipLaunchKernelGGL((k_bf16_gemv_1r4w_v28<5,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 6:  hipLaunchKernelGGL((k_bf16_gemv_1r4w_v28<6,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 7:  hipLaunchKernelGGL((k_bf16_gemv_1r4w_v28<7,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 8:  hipLaunchKernelGGL((k_bf16_gemv_1r4w_v28<8,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 9:  hipLaunchKernelGGL((k_bf16_gemv_1r4w_v28<9,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 10: hipLaunchKernelGGL((k_bf16_gemv_1r4w_v28<10, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 11: hipLaunchKernelGGL((k_bf16_gemv_1r4w_v28<11, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 12: hipLaunchKernelGGL((k_bf16_gemv_1r4w_v28<12, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 13: hipLaunchKernelGGL((k_bf16_gemv_1r4w_v28<13, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 14: hipLaunchKernelGGL((k_bf16_gemv_1r4w_v28<14, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 15: hipLaunchKernelGGL((k_bf16_gemv_1r4w_v28<15, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 16: hipLaunchKernelGGL((k_bf16_gemv_1r4w_v28<16, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    default: TORCH_CHECK(false, "T must be 1..16");
  }
}

template <int OUT_BF16>
static void dispatch_bf16_gemv_2r2w(
    int T, dim3 grid, hipStream_t st,
    const uint16_t* xp, int64_t xs0, const uint16_t* wp, int64_t ws0,
    void* op, int N, int K) {
  switch (T) {
    case 1:  hipLaunchKernelGGL((k_bf16_gemv_2r2w_v28<1,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 2:  hipLaunchKernelGGL((k_bf16_gemv_2r2w_v28<2,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 3:  hipLaunchKernelGGL((k_bf16_gemv_2r2w_v28<3,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 4:  hipLaunchKernelGGL((k_bf16_gemv_2r2w_v28<4,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 5:  hipLaunchKernelGGL((k_bf16_gemv_2r2w_v28<5,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 6:  hipLaunchKernelGGL((k_bf16_gemv_2r2w_v28<6,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 7:  hipLaunchKernelGGL((k_bf16_gemv_2r2w_v28<7,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 8:  hipLaunchKernelGGL((k_bf16_gemv_2r2w_v28<8,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 9:  hipLaunchKernelGGL((k_bf16_gemv_2r2w_v28<9,  OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 10: hipLaunchKernelGGL((k_bf16_gemv_2r2w_v28<10, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 11: hipLaunchKernelGGL((k_bf16_gemv_2r2w_v28<11, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 12: hipLaunchKernelGGL((k_bf16_gemv_2r2w_v28<12, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 13: hipLaunchKernelGGL((k_bf16_gemv_2r2w_v28<13, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 14: hipLaunchKernelGGL((k_bf16_gemv_2r2w_v28<14, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 15: hipLaunchKernelGGL((k_bf16_gemv_2r2w_v28<15, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    case 16: hipLaunchKernelGGL((k_bf16_gemv_2r2w_v28<16, OUT_BF16>), grid, dim3(128), 0, st, xp, xs0, wp, ws0, op, N, K); break;
    default: TORCH_CHECK(false, "T must be 1..16");
  }
}

torch::Tensor ernie_bf16_gemv_nt(torch::Tensor x, torch::Tensor w, bool out_bf16 = false) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.dim() == 2 && w.dim() == 2);
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16);
  const int T = (int)x.size(0);
  const int K = (int)x.size(1);
  const int N = (int)w.size(0);
  TORCH_CHECK(w.size(1) == K && (K % 256) == 0 && x.stride(1) == 1 && w.stride(1) == 1 && T >= 1 && T <= 16);
  auto st = c10::hip::getCurrentHIPStream().stream();
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
  const uint16_t* wp = reinterpret_cast<const uint16_t*>(w.data_ptr());

  // Wave 70: zero-in-loop-barrier multi-wave-per-row L0-cached X kernels:
  //   - Small/medium N (N<=2048, K>=1024): 1r4w (N<=384 or T<=6) / 2r2w (T>6, N>384) -> 1.20-2.20x!
  //   - Wide K (K>=8192, T<=8, e.g. DSpark main_proj 4096x12288): 1r4w -> 233 GB/s (1.10-1.19x)!
  if ((N <= 2048 && K >= 1024) || (K >= 8192 && T <= 8)) {
    const bool use_2r2w = (N <= 2048 && N > 384 && T > 6);
    if (out_bf16) {
      auto out = torch::empty({T, N}, x.options());
      if (use_2r2w) dispatch_bf16_gemv_2r2w<1>(T, dim3((N + 1) / 2), st, xp, x.stride(0), wp, w.stride(0), out.data_ptr(), N, K);
      else          dispatch_bf16_gemv_1r4w<1>(T, dim3(N), st, xp, x.stride(0), wp, w.stride(0), out.data_ptr(), N, K);
      return out;
    } else {
      auto out = torch::empty({T, N}, x.options().dtype(torch::kFloat32));
      if (use_2r2w) dispatch_bf16_gemv_2r2w<0>(T, dim3((N + 1) / 2), st, xp, x.stride(0), wp, w.stride(0), out.data_ptr(), N, K);
      else          dispatch_bf16_gemv_1r4w<0>(T, dim3(N), st, xp, x.stride(0), wp, w.stride(0), out.data_ptr(), N, K);
      return out;
    }
  }

  const dim3 grid((N + 3) / 4);
  // T==1 on non-pow2-K tall shapes (e.g. DSA idx wq_b 8192x1536): 4x MLP direct kernel peaks at 230 GB/s!
  if (T == 1 && N >= 4096 && (K & 2047) != 0) {
    if (out_bf16) {
      auto out = torch::empty({T, N}, x.options());
      hipLaunchKernelGGL((k_bf16_gemv_nt<1, 1>), grid, dim3(128), 0, st, xp, x.stride(0), wp, w.stride(0), out.data_ptr(), N, K);
      return out;
    } else {
      auto out = torch::empty({T, N}, x.options().dtype(torch::kFloat32));
      hipLaunchKernelGGL((k_bf16_gemv_nt<1, 0>), grid, dim3(128), 0, st, xp, x.stride(0), wp, w.stride(0), out.data_ptr(), N, K);
      return out;
    }
  }

  if (out_bf16) {
    auto out = torch::empty({T, N}, x.options());
    dispatch_bf16_gemv_xlds512<1>(T, grid, st, xp, x.stride(0), wp, w.stride(0), out.data_ptr(), N, K);
    return out;
  } else {
    auto out = torch::empty({T, N}, x.options().dtype(torch::kFloat32));
    dispatch_bf16_gemv_xlds512<0>(T, grid, st, xp, x.stride(0), wp, w.stride(0), out.data_ptr(), N, K);
    return out;
  }
}

// ------------------------------------------------------------------ 21 & 22: Fused NT-Streaming DSA MLA Absorption (6x/tok = 100.7 MB/step!)
// Replaces 2x rocBLAS torch.bmm + 4x transpose + torch.cat + contiguous + copy_ per DSA layer (54 ops -> 12 NT kernels!)
//   21. dsa_q_absorb_cat: q_abs[T, 64, 576] = [q_nope[T, 64, 128] @ w_uk_rm[64, 512, 128]^T, q_pe[T, 64, 64]]
//   22. dsa_uv_proj:      out[T, 64, 128]   = attn_latent[T, 64, 512] @ w_uv_rm[64, 128, 512]^T
// Both stream 16B NT weights into VOPD V_DUAL_DOT2ACC_F32_BF16 -> ZERO Infinity Cache eviction, ZERO extra VRAM!
template <int R>
__global__ __launch_bounds__(128) void k_dsa_q_absorb_cat(
    const uint16_t* __restrict__ q_nope,   // [T, 64, 128] bf16 (strides qn_s0, qn_s1)
    const uint16_t* __restrict__ q_pe,     // [T, 64, 64]  bf16 (strides qp_s0, qp_s1)
    const uint16_t* __restrict__ w_uk_rm,  // [64, 512, 128] bf16 contiguous
    uint16_t* __restrict__ q_abs_out,      // [T, 64, 576] bf16 contiguous
    int64_t qn_s0, int64_t qn_s1, int64_t qp_s0, int64_t qp_s1) {
  const int h = blockIdx.x;    // 0..63
  const int nb = blockIdx.y;   // 0..15 (32 output dims per WG -> 16*32 = 512 exact!)
  const int tid = threadIdx.x; // 0..127 -> 8 half-waves of 16 lanes (16 lanes x 8 bf16 = 128 exact!)
  const int hw = tid >> 4;     // 0..7
  const int l16 = tid & 15;    // 0..15

  // Load q_nope[0..R-1, h, l16*8 .. l16*8+7] ONCE into registers across 4 output rows!
  u4v x[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    x[r] = *reinterpret_cast<const u4va*>(q_nope + (int64_t)r * qn_s0 + (int64_t)h * qn_s1 + l16 * 8);
  }

  const int n_base = nb * 32 + hw * 4;
  const uint16_t* w_head = w_uk_rm + (int64_t)h * (512 * 128) + (int64_t)n_base * 128 + l16 * 8;
  // Issue all 4 NT weight loads up front (free offsets +0/+256B/+512B/+768B -> 4x MLP with s_waitcnt vmcnt(3..0)!)
  const u4v w_pf[4] = {
    __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_head)),
    __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_head + 128)),
    __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_head + 256)),
    __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_head + 384)),
  };
#pragma unroll
  for (int ni = 0; ni < 4; ++ni) {
    const int n = n_base + ni;
    const u4v wraw = w_pf[ni];
    float a0[R], a1[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { a0[r] = 0.0f; a1[r] = 0.0f; }
    if constexpr (R > 0) BF16_DUAL_DOT2_8_PIN(a0[0], a1[0], wraw, x[0], "v16", "v17");
    if constexpr (R > 1) BF16_DUAL_DOT2_8_PIN(a0[1], a1[1], wraw, x[1], "v18", "v19");
    if constexpr (R > 2) BF16_DUAL_DOT2_8_PIN(a0[2], a1[2], wraw, x[2], "v20", "v21");
    if constexpr (R > 3) BF16_DUAL_DOT2_8_PIN(a0[3], a1[3], wraw, x[3], "v22", "v23");
    if constexpr (R > 4) BF16_DUAL_DOT2_8_PIN(a0[4], a1[4], wraw, x[4], "v24", "v25");
    if constexpr (R > 5) BF16_DUAL_DOT2_8_PIN(a0[5], a1[5], wraw, x[5], "v26", "v27");
    if constexpr (R > 6) BF16_DUAL_DOT2_8_PIN(a0[6], a1[6], wraw, x[6], "v28", "v29");
    if constexpr (R > 7) BF16_DUAL_DOT2_8_PIN(a0[7], a1[7], wraw, x[7], "v30", "v31");
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float acc = a0[r] + a1[r];
      acc += shx_f<8>(acc);
      acc += shx_f<4>(acc);
      acc += shx_f<2>(acc);
      acc += shx_f<1>(acc);
      if (l16 == 0) {
        q_abs_out[((int64_t)r * 64 + h) * 576 + n] = f32_to_bf16_rne(acc);
      }
    }
  }

  // Copy q_pe[0..R-1, h, 0..63] into q_abs_out[..., 512..575] inside nb==0 (8 threads/token x 8 bf16 = 64 exact!)
  if (nb == 0 && tid < R * 8) {
    const int r = tid >> 3;
    const int c8 = (tid & 7) * 8;
    const u4v pv = *reinterpret_cast<const u4va*>(q_pe + (int64_t)r * qp_s0 + (int64_t)h * qp_s1 + c8);
    *reinterpret_cast<u4va*>(q_abs_out + ((int64_t)r * 64 + h) * 576 + 512 + c8) = pv;
  }
}

torch::Tensor ernie_dsa_q_absorb_cat(torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor w_uk_rm) {
  TORCH_CHECK(q_nope.is_cuda() && q_pe.is_cuda() && w_uk_rm.is_cuda());
  TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(1) == 64 && q_nope.size(2) == 128 && q_nope.stride(2) == 1);
  TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(1) == 64 && q_pe.size(2) == 64 && q_pe.stride(2) == 1);
  TORCH_CHECK(w_uk_rm.is_contiguous() && w_uk_rm.dim() == 3 && w_uk_rm.size(0) == 64 && w_uk_rm.size(1) == 512 && w_uk_rm.size(2) == 128);
  const int T = (int)q_nope.size(0);
  TORCH_CHECK(T >= 1 && T <= 8);
  auto out = torch::empty({T, 64, 576}, q_nope.options());
  auto st = c10::hip::getCurrentHIPStream().stream();
  const dim3 grid(64, 16);
  const uint16_t* qnp = reinterpret_cast<const uint16_t*>(q_nope.data_ptr());
  const uint16_t* qpp = reinterpret_cast<const uint16_t*>(q_pe.data_ptr());
  const uint16_t* wp = reinterpret_cast<const uint16_t*>(w_uk_rm.data_ptr());
  uint16_t* op = reinterpret_cast<uint16_t*>(out.data_ptr());
  switch (T) {
    case 1: hipLaunchKernelGGL((k_dsa_q_absorb_cat<1>), grid, dim3(128), 0, st, qnp, qpp, wp, op, q_nope.stride(0), q_nope.stride(1), q_pe.stride(0), q_pe.stride(1)); break;
    case 2: hipLaunchKernelGGL((k_dsa_q_absorb_cat<2>), grid, dim3(128), 0, st, qnp, qpp, wp, op, q_nope.stride(0), q_nope.stride(1), q_pe.stride(0), q_pe.stride(1)); break;
    case 3: hipLaunchKernelGGL((k_dsa_q_absorb_cat<3>), grid, dim3(128), 0, st, qnp, qpp, wp, op, q_nope.stride(0), q_nope.stride(1), q_pe.stride(0), q_pe.stride(1)); break;
    case 4: hipLaunchKernelGGL((k_dsa_q_absorb_cat<4>), grid, dim3(128), 0, st, qnp, qpp, wp, op, q_nope.stride(0), q_nope.stride(1), q_pe.stride(0), q_pe.stride(1)); break;
    case 5: hipLaunchKernelGGL((k_dsa_q_absorb_cat<5>), grid, dim3(128), 0, st, qnp, qpp, wp, op, q_nope.stride(0), q_nope.stride(1), q_pe.stride(0), q_pe.stride(1)); break;
    case 6: hipLaunchKernelGGL((k_dsa_q_absorb_cat<6>), grid, dim3(128), 0, st, qnp, qpp, wp, op, q_nope.stride(0), q_nope.stride(1), q_pe.stride(0), q_pe.stride(1)); break;
    case 7: hipLaunchKernelGGL((k_dsa_q_absorb_cat<7>), grid, dim3(128), 0, st, qnp, qpp, wp, op, q_nope.stride(0), q_nope.stride(1), q_pe.stride(0), q_pe.stride(1)); break;
    case 8: hipLaunchKernelGGL((k_dsa_q_absorb_cat<8>), grid, dim3(128), 0, st, qnp, qpp, wp, op, q_nope.stride(0), q_nope.stride(1), q_pe.stride(0), q_pe.stride(1)); break;
    default: TORCH_CHECK(false, "T must be 1..8");
  }
  return out;
}

template <int R>
__global__ __launch_bounds__(128) void k_dsa_uv_proj(
    const uint16_t* __restrict__ attn_latent, // [T, 64, 512] bf16 (strides a_s0, a_s1)
    const uint16_t* __restrict__ w_uv_rm,     // [64, 128, 512] bf16 contiguous
    uint16_t* __restrict__ out_ptr,           // [T, 64, 128] bf16 (strides o_s0, o_s1)
    int64_t a_s0, int64_t a_s1, int64_t o_s0, int64_t o_s1) {
  __shared__ __attribute__((aligned(16))) uint32_t s_x[R][256]; // R * 1 KiB packed BF16x2 (1..8 KiB)
  const int h = blockIdx.x;    // 0..63
  const int nb = blockIdx.y;   // 0..7 (16 output dims per WG -> 8*16 = 128 exact!)
  const int tid = threadIdx.x; // 0..127
  const int wave = tid >> 5;   // 0..3 (4 output dims per wave)
  const int lane = tid & 31;

  // Cooperatively stage attn_latent[0..R-1, h, 0..511] into LDS (128 threads x 2 u32 = 256 u32 = 512 bf16!)
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const uint32_t* src32 = reinterpret_cast<const uint32_t*>(attn_latent + (int64_t)r * a_s0 + (int64_t)h * a_s1);
    s_x[r][tid]       = src32[tid];
    s_x[r][tid + 128] = src32[tid + 128];
  }
  __syncthreads();

  const int n_base = nb * 16 + wave * 4;
  const uint16_t* w_base = w_uv_rm + (int64_t)h * (128 * 512) + (int64_t)n_base * 512 + lane * 8;
  // 2-stage software prefetch across ni=0..3 + 2x MLP per row (s_x is in LDS -> zero vmcnt conflict!)
  u4v w0_cur = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_base));
  u4v w1_cur = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_base + 256));
#define DSA_UV_STEP(WVAL, KU32) do { \
    if constexpr (R > 0) { const u4v x0 = *reinterpret_cast<const u4v*>(&s_x[0][(KU32)]); BF16_DUAL_DOT2_8_PIN(a0[0], a1[0], (WVAL), x0, "v16", "v17"); } \
    if constexpr (R > 1) { const u4v x1 = *reinterpret_cast<const u4v*>(&s_x[1][(KU32)]); BF16_DUAL_DOT2_8_PIN(a0[1], a1[1], (WVAL), x1, "v18", "v19"); } \
    if constexpr (R > 2) { const u4v x2 = *reinterpret_cast<const u4v*>(&s_x[2][(KU32)]); BF16_DUAL_DOT2_8_PIN(a0[2], a1[2], (WVAL), x2, "v20", "v21"); } \
    if constexpr (R > 3) { const u4v x3 = *reinterpret_cast<const u4v*>(&s_x[3][(KU32)]); BF16_DUAL_DOT2_8_PIN(a0[3], a1[3], (WVAL), x3, "v22", "v23"); } \
    if constexpr (R > 4) { const u4v x4 = *reinterpret_cast<const u4v*>(&s_x[4][(KU32)]); BF16_DUAL_DOT2_8_PIN(a0[4], a1[4], (WVAL), x4, "v24", "v25"); } \
    if constexpr (R > 5) { const u4v x5 = *reinterpret_cast<const u4v*>(&s_x[5][(KU32)]); BF16_DUAL_DOT2_8_PIN(a0[5], a1[5], (WVAL), x5, "v26", "v27"); } \
    if constexpr (R > 6) { const u4v x6 = *reinterpret_cast<const u4v*>(&s_x[6][(KU32)]); BF16_DUAL_DOT2_8_PIN(a0[6], a1[6], (WVAL), x6, "v28", "v29"); } \
    if constexpr (R > 7) { const u4v x7 = *reinterpret_cast<const u4v*>(&s_x[7][(KU32)]); BF16_DUAL_DOT2_8_PIN(a0[7], a1[7], (WVAL), x7, "v30", "v31"); } \
  } while (0)

#pragma unroll
  for (int ni = 0; ni < 4; ++ni) {
    u4v w0_nxt = {0u, 0u, 0u, 0u}, w1_nxt = {0u, 0u, 0u, 0u};
    if (ni < 3) {
      w0_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_base + (ni + 1) * 512));
      w1_nxt = __builtin_nontemporal_load(reinterpret_cast<const u4va*>(w_base + (ni + 1) * 512 + 256));
    }
    const int n = n_base + ni;
    float a0[R], a1[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { a0[r] = 0.0f; a1[r] = 0.0f; }
    DSA_UV_STEP(w0_cur, lane * 4);
    DSA_UV_STEP(w1_cur, 128 + lane * 4);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float acc = a0[r] + a1[r];
      acc += shx_f<16>(acc); acc += shx_f<8>(acc); acc += shx_f<4>(acc); acc += shx_f<2>(acc); acc += shx_f<1>(acc);
      if (lane == 0) {
        out_ptr[(int64_t)r * o_s0 + (int64_t)h * o_s1 + n] = f32_to_bf16_rne(acc);
      }
    }
    w0_cur = w0_nxt;
    w1_cur = w1_nxt;
  }
#undef DSA_UV_STEP
}

void ernie_dsa_uv_proj(torch::Tensor attn_latent, torch::Tensor w_uv_rm, torch::Tensor out) {
  TORCH_CHECK(attn_latent.is_cuda() && w_uv_rm.is_cuda() && out.is_cuda());
  TORCH_CHECK(attn_latent.dim() == 3 && attn_latent.size(1) == 64 && attn_latent.size(2) == 512 && attn_latent.stride(2) == 1);
  TORCH_CHECK(w_uv_rm.is_contiguous() && w_uv_rm.dim() == 3 && w_uv_rm.size(0) == 64 && w_uv_rm.size(1) == 128 && w_uv_rm.size(2) == 512);
  TORCH_CHECK(out.dim() == 3 && out.size(1) == 64 && out.size(2) == 128 && out.stride(2) == 1);
  const int T = (int)attn_latent.size(0);
  TORCH_CHECK(T >= 1 && T <= 8);
  auto st = c10::hip::getCurrentHIPStream().stream();
  const dim3 grid(64, 8);
  const uint16_t* ap = reinterpret_cast<const uint16_t*>(attn_latent.data_ptr());
  const uint16_t* wp = reinterpret_cast<const uint16_t*>(w_uv_rm.data_ptr());
  uint16_t* op = reinterpret_cast<uint16_t*>(out.data_ptr());
  switch (T) {
    case 1: hipLaunchKernelGGL((k_dsa_uv_proj<1>), grid, dim3(128), 0, st, ap, wp, op, attn_latent.stride(0), attn_latent.stride(1), out.stride(0), out.stride(1)); break;
    case 2: hipLaunchKernelGGL((k_dsa_uv_proj<2>), grid, dim3(128), 0, st, ap, wp, op, attn_latent.stride(0), attn_latent.stride(1), out.stride(0), out.stride(1)); break;
    case 3: hipLaunchKernelGGL((k_dsa_uv_proj<3>), grid, dim3(128), 0, st, ap, wp, op, attn_latent.stride(0), attn_latent.stride(1), out.stride(0), out.stride(1)); break;
    case 4: hipLaunchKernelGGL((k_dsa_uv_proj<4>), grid, dim3(128), 0, st, ap, wp, op, attn_latent.stride(0), attn_latent.stride(1), out.stride(0), out.stride(1)); break;
    case 5: hipLaunchKernelGGL((k_dsa_uv_proj<5>), grid, dim3(128), 0, st, ap, wp, op, attn_latent.stride(0), attn_latent.stride(1), out.stride(0), out.stride(1)); break;
    case 6: hipLaunchKernelGGL((k_dsa_uv_proj<6>), grid, dim3(128), 0, st, ap, wp, op, attn_latent.stride(0), attn_latent.stride(1), out.stride(0), out.stride(1)); break;
    case 7: hipLaunchKernelGGL((k_dsa_uv_proj<7>), grid, dim3(128), 0, st, ap, wp, op, attn_latent.stride(0), attn_latent.stride(1), out.stride(0), out.stride(1)); break;
    case 8: hipLaunchKernelGGL((k_dsa_uv_proj<8>), grid, dim3(128), 0, st, ap, wp, op, attn_latent.stride(0), attn_latent.stride(1), out.stride(0), out.stride(1)); break;
    default: TORCH_CHECK(false, "T must be 1..8");
  }
}


// ------------------------------------------------------------------ 23: FP8/FP4 DSA KV Scatter + MXFP4 Indexer K Cache
__device__ __forceinline__ float fp8_e4m3fn_ocp_to_f32(uint32_t b) {
  const uint32_t sign = (b & 0x80u) << 24;
  const uint32_t exp = (b >> 3) & 0xFu;
  const uint32_t man = b & 0x7u;
  if (exp == 0u) {
    float v = (float)man * (1.0f / 512.0f);
    return sign ? -v : v;
  }
  if (exp == 15u && man == 7u) return NAN;
  return __uint_as_float(sign | ((exp + 120u) << 23) | (man << 20));
}

__device__ __forceinline__ uint32_t f32_to_fp4_e2m1(float x) {
  const uint32_t sign = (__float_as_uint(x) >> 28) & 8u;
  const float ax = fabsf(x);
  const uint32_t code = (uint32_t)(ax > 0.25f) + (uint32_t)(ax >= 0.75f)
                      + (uint32_t)(ax > 1.25f) + (uint32_t)(ax >= 1.75f)
                      + (uint32_t)(ax > 2.50f) + (uint32_t)(ax >= 3.50f)
                      + (uint32_t)(ax > 5.00f);
  return sign | code;
}

__device__ __forceinline__ float fp4_e2m1_to_f32(uint32_t nib) {
  static constexpr float LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
  };
  return LUT[nib & 0xFu];
}

// 23a: scatter_dsa_kv_fp8 -> official fp8_ds_mla 656 B/tok layout:
//   [0..511]:   512 FP8 E4M3FN NoPE (4 tiles x 128)
//   [512..527]: 4 FP32 scales (1 per 128 NoPE tile)
//   [528..655]: 64 BF16 RoPE (128 B, lossless)
__global__ __launch_bounds__(128) void k_scatter_dsa_kv_fp8(
    const uint16_t* __restrict__ kvc, const uint16_t* __restrict__ kpe,
    uint8_t* __restrict__ cache, const int64_t* __restrict__ slot_mapping,
    int64_t kvc_s0, int64_t kpe_s0, int64_t c_s0, int64_t c_s1,
    int block_size) {
  const int t = blockIdx.x;
  const int64_t slot = slot_mapping[t];
  if (slot < 0) return;
  const int64_t blk = slot / block_size;
  const int64_t off = slot % block_size;
  uint8_t* dst = cache + blk * c_s0 + off * c_s1;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;   // 0..3 (4 tiles of 128)
  const int lane = tid & 31;   // 0..31 (4 elems/lane = 128 elems/wave)

  const uint2 raw = *reinterpret_cast<const uint2*>(kvc + (int64_t)t * kvc_s0 + wave * 128 + lane * 4);
  float v[4];
  v[0] = __uint_as_float((raw.x & 0xFFFFu) << 16);
  v[1] = __uint_as_float(raw.x & 0xFFFF0000u);
  v[2] = __uint_as_float((raw.y & 0xFFFFu) << 16);
  v[3] = __uint_as_float(raw.y & 0xFFFF0000u);
  float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  am = fmaxf(am, shx_f<16>(am));
  am = fmaxf(am, shx_f<8>(am));
  am = fmaxf(am, shx_f<4>(am));
  am = fmaxf(am, shx_f<2>(am));
  am = fmaxf(am, shx_f<1>(am));
  const float scale = fmaxf(am, 1e-6f) * (1.0f / 448.0f);
  const float inv_sc = __builtin_amdgcn_rcpf(scale);
  uint8_t ob[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) ob[e] = f32_to_fp8_e4m3fn_ocp(v[e] * inv_sc);
  *reinterpret_cast<uint32_t*>(dst + wave * 128 + lane * 4) = *reinterpret_cast<const uint32_t*>(ob);
  if (lane == 0) {
    *reinterpret_cast<float*>(dst + 512 + wave * 4) = scale;
  }
  if (tid < 8) {
    const u4v rv = *reinterpret_cast<const u4va*>(kpe + (int64_t)t * kpe_s0 + tid * 8);
    __builtin_nontemporal_store(rv, reinterpret_cast<u4va*>(dst + 528 + tid * 16));
  }
}

void ernie_scatter_dsa_kv_fp8(
    torch::Tensor kv_c, torch::Tensor k_pe_2d, torch::Tensor kv_cache,
    torch::Tensor slot_mapping, int64_t block_size) {
  const int T = (int)min(kv_c.size(0), slot_mapping.size(0));
  if (T == 0 || kv_cache.numel() == 0) return;
  TORCH_CHECK(kv_c.size(1) == 512 && k_pe_2d.size(1) == 64 && kv_c.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8 && kv_cache.size(2) == 656);
  hipLaunchKernelGGL(k_scatter_dsa_kv_fp8, dim3(T), dim3(128), 0, c10::hip::getCurrentHIPStream().stream(),
                     reinterpret_cast<const uint16_t*>(kv_c.data_ptr()),
                     reinterpret_cast<const uint16_t*>(k_pe_2d.data_ptr()),
                     kv_cache.data_ptr<uint8_t>(),
                     slot_mapping.data_ptr<int64_t>(),
                     kv_c.stride(0), k_pe_2d.stride(0), kv_cache.stride(0), kv_cache.stride(1),
                     (int)block_size);
}

// 23b: scatter_dsa_kv_fp4 -> official nvfp4_ds_mla 352 B/tok layout:
//   [0..255]:   512 FP4 E2M1 NoPE packed 2/byte (low nibble=even, high=odd; 32 micro-blocks x 16)
//   [256..319]: 64 FP8 E4M3FN RoPE (unscaled)
//   [320..351]: 32 FP8 E4M3FN scales (1 per 16-elem NoPE micro-block)
__global__ __launch_bounds__(128) void k_scatter_dsa_kv_fp4(
    const uint16_t* __restrict__ kvc, const uint16_t* __restrict__ kpe,
    uint8_t* __restrict__ cache, const int64_t* __restrict__ slot_mapping,
    int64_t kvc_s0, int64_t kpe_s0, int64_t c_s0, int64_t c_s1,
    int block_size) {
  const int t = blockIdx.x;
  const int64_t slot = slot_mapping[t];
  if (slot < 0) return;
  const int64_t blk = slot / block_size;
  const int64_t off = slot % block_size;
  uint8_t* dst = cache + blk * c_s0 + off * c_s1;
  const int tid = threadIdx.x;
  const int grp = tid >> 2;   // 0..31 (32 micro-blocks of 16 elems)
  const int sub = tid & 3;    // 0..3  (4 elems/thread -> 2 packed bytes)

  const uint2 raw = *reinterpret_cast<const uint2*>(kvc + (int64_t)t * kvc_s0 + tid * 4);
  float v[4];
  v[0] = __uint_as_float((raw.x & 0xFFFFu) << 16);
  v[1] = __uint_as_float(raw.x & 0xFFFF0000u);
  v[2] = __uint_as_float((raw.y & 0xFFFFu) << 16);
  v[3] = __uint_as_float(raw.y & 0xFFFF0000u);
  float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  am = fmaxf(am, shx_f<2>(am));
  am = fmaxf(am, shx_f<1>(am));
  const float raw_sc = am * (1.0f / 6.0f);
  uint8_t sc_u8 = f32_to_fp8_e4m3fn_ocp(raw_sc);
  if (am > 0.0f && (sc_u8 & 0x7Fu) == 0u) sc_u8 = 1u;
  const float sc_f = fp8_e4m3fn_ocp_to_f32(sc_u8);
  const float inv_sc = (sc_f > 0.0f) ? __builtin_amdgcn_rcpf(sc_f) : 0.0f;
  const uint32_t n0 = f32_to_fp4_e2m1(v[0] * inv_sc);
  const uint32_t n1 = f32_to_fp4_e2m1(v[1] * inv_sc);
  const uint32_t n2 = f32_to_fp4_e2m1(v[2] * inv_sc);
  const uint32_t n3 = f32_to_fp4_e2m1(v[3] * inv_sc);
  const uint16_t packed = (uint16_t)(n0 | (n1 << 4) | (n2 << 8) | (n3 << 12));
  *reinterpret_cast<uint16_t*>(dst + tid * 2) = packed;
  if (sub == 0) {
    dst[320 + grp] = sc_u8;
  }
  if (tid < 16) {
    const uint2 rraw = *reinterpret_cast<const uint2*>(kpe + (int64_t)t * kpe_s0 + tid * 4);
    float rv[4];
    rv[0] = __uint_as_float((rraw.x & 0xFFFFu) << 16);
    rv[1] = __uint_as_float(rraw.x & 0xFFFF0000u);
    rv[2] = __uint_as_float((rraw.y & 0xFFFFu) << 16);
    rv[3] = __uint_as_float(rraw.y & 0xFFFF0000u);
    uint8_t rb[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) rb[e] = f32_to_fp8_e4m3fn_ocp(rv[e]);
    *reinterpret_cast<uint32_t*>(dst + 256 + tid * 4) = *reinterpret_cast<const uint32_t*>(rb);
  }
}

void ernie_scatter_dsa_kv_fp4(
    torch::Tensor kv_c, torch::Tensor k_pe_2d, torch::Tensor kv_cache,
    torch::Tensor slot_mapping, int64_t block_size) {
  const int T = (int)min(kv_c.size(0), slot_mapping.size(0));
  if (T == 0 || kv_cache.numel() == 0) return;
  TORCH_CHECK(kv_c.size(1) == 512 && k_pe_2d.size(1) == 64 && kv_c.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8 && kv_cache.size(2) == 352);
  hipLaunchKernelGGL(k_scatter_dsa_kv_fp4, dim3(T), dim3(128), 0, c10::hip::getCurrentHIPStream().stream(),
                     reinterpret_cast<const uint16_t*>(kv_c.data_ptr()),
                     reinterpret_cast<const uint16_t*>(k_pe_2d.data_ptr()),
                     kv_cache.data_ptr<uint8_t>(),
                     slot_mapping.data_ptr<int64_t>(),
                     kv_c.stride(0), k_pe_2d.stride(0), kv_cache.stride(0), kv_cache.stride(1),
                     (int)block_size);
}

// 23c: indexer_k_quant_and_cache_mxfp4 -> 68 B/tok flat per-token layout:
//   [0..63]:  128 FP4 E2M1 packed 2/byte (4 groups of 32 elems)
//   [64..67]: 4 UE8M0 exponent bytes (1 per 32 elems, scale = 2^(e - 127))
__global__ __launch_bounds__(32) void k_indexer_k_quant_and_cache_mxfp4_128(
    const uint16_t* __restrict__ k_in,       // [T, 128] bf16
    uint8_t* __restrict__ kv_cache,          // [num_blocks, block_size, 68] uint8
    const int64_t* __restrict__ slot_map,    // [T] int64
    int64_t k_s0, int64_t c_s0, int64_t c_s1, int block_size) {
  const int t = blockIdx.x;
  const int64_t slot = slot_map[t];
  if (slot < 0) return;
  const int lane = threadIdx.x; // 0..31 -> 4 elems/lane = 128 elems
  const int grp = lane >> 3;    // 0..3  (4 groups of 32 elems = 8 lanes/group)
  const int sub = lane & 7;     // 0..7
  const uint2 raw = *reinterpret_cast<const uint2*>(k_in + (int64_t)t * k_s0 + lane * 4);
  float v[4];
  v[0] = __uint_as_float((raw.x & 0xFFFFu) << 16);
  v[1] = __uint_as_float(raw.x & 0xFFFF0000u);
  v[2] = __uint_as_float((raw.y & 0xFFFFu) << 16);
  v[3] = __uint_as_float(raw.y & 0xFFFF0000u);
  float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  am = fmaxf(am, shx_f<4>(am));
  am = fmaxf(am, shx_f<2>(am));
  am = fmaxf(am, shx_f<1>(am));
  const float exp_f = ceilf(log2f(fmaxf(am, 1e-6f) * (1.0f / 6.0f)));
  int e_i = (int)exp_f + 127;
  if (e_i < 1) e_i = 1;
  if (e_i > 254) e_i = 254;
  const float inv_sc = __uint_as_float((uint32_t)(254 - e_i) << 23);
  const uint32_t n0 = f32_to_fp4_e2m1(v[0] * inv_sc);
  const uint32_t n1 = f32_to_fp4_e2m1(v[1] * inv_sc);
  const uint32_t n2 = f32_to_fp4_e2m1(v[2] * inv_sc);
  const uint32_t n3 = f32_to_fp4_e2m1(v[3] * inv_sc);
  const uint16_t packed = (uint16_t)(n0 | (n1 << 4) | (n2 << 8) | (n3 << 12));

  const int64_t blk = slot / block_size;
  const int64_t off = slot % block_size;
  uint8_t* dst = kv_cache + blk * c_s0 + off * c_s1;
  *reinterpret_cast<uint16_t*>(dst + lane * 2) = packed;
  if (sub == 0) {
    dst[64 + grp] = (uint8_t)e_i;
  }
}

void ernie_indexer_k_quant_and_cache_mxfp4(
    torch::Tensor k, torch::Tensor kv_cache, torch::Tensor slot_mapping, int64_t block_size) {
  const int T = (int)slot_mapping.size(0);
  if (T == 0) return;
  TORCH_CHECK(k.is_cuda() && k.dim() == 2 && k.size(1) == 128 && k.scalar_type() == torch::kBFloat16 && k.stride(1) == 1);
  TORCH_CHECK(kv_cache.is_cuda() && kv_cache.scalar_type() == torch::kUInt8 && kv_cache.size(-1) == 68);
  auto sm_i64 = slot_mapping.scalar_type() == torch::kInt64 ? slot_mapping.contiguous() : slot_mapping.to(torch::kInt64).contiguous();
  hipLaunchKernelGGL(
      k_indexer_k_quant_and_cache_mxfp4_128, dim3(T), dim3(32), 0, c10::hip::getCurrentHIPStream().stream(),
      reinterpret_cast<const uint16_t*>(k.data_ptr()),
      kv_cache.data_ptr<uint8_t>(), sm_i64.data_ptr<int64_t>(),
      k.stride(0), kv_cache.stride(0), kv_cache.stride(1), (int)block_size);
}

// 23d: cp_gather_indexer_k_mxfp4_to_fp8 (for prefill MQA logits!)
__global__ __launch_bounds__(32) void k_cp_gather_indexer_k_mxfp4_to_fp8_128(
    const uint8_t* __restrict__ kv_cache,    // [num_blocks, block_size, 68] uint8
    uint8_t* __restrict__ k_fp8_out,         // [num_tokens, 128] uint8
    float* __restrict__ k_scale_out,         // [num_tokens] fp32
    const int32_t* __restrict__ block_table, // [num_batches, bt_s0] int32
    const int32_t* __restrict__ cu_seqlen,   // [num_batches + 1] int32
    const int32_t* __restrict__ token_to_seq,// [num_tokens] int32
    int64_t c_s0, int64_t c_s1, int64_t bt_s0,
    int num_tokens, int num_batches, int bt_width, int num_blocks, int block_size) {
  const int t = blockIdx.x;
  if (t >= num_tokens) return;
  const int lane = threadIdx.x; // 0..31 -> 4 elems/lane
  const int grp = lane >> 3;    // 0..3

  const int batch_id = token_to_seq[t];
  if (batch_id < 0 || batch_id >= num_batches) return;
  const int batch_start = cu_seqlen[batch_id];
  const int batch_end = cu_seqlen[batch_id + 1];
  if (t < batch_start || t >= batch_end) return;

  const int batch_off = t - batch_start;
  const int bt_id = batch_off / block_size;
  const int blk_off = batch_off % block_size;
  bool valid_blk = (bt_id >= 0 && bt_id < bt_width && blk_off >= 0 && blk_off < block_size);
  int phys_blk = -1;
  if (valid_blk) {
    phys_blk = block_table[(int64_t)batch_id * bt_s0 + bt_id];
    valid_blk = (phys_blk >= 0 && phys_blk < num_blocks);
  }
  if (!valid_blk) {
    *reinterpret_cast<uint32_t*>(k_fp8_out + (int64_t)t * 128 + lane * 4) = 0u;
    if (lane == 0) k_scale_out[t] = 0.0f;
    return;
  }

  const uint8_t* src = kv_cache + (int64_t)phys_blk * c_s0 + (int64_t)blk_off * c_s1;
  const uint32_t sc4 = *reinterpret_cast<const uint32_t*>(src + 64);
  const int e0 = (int)(sc4 & 0xFFu);
  const int e1 = (int)((sc4 >> 8) & 0xFFu);
  const int e2 = (int)((sc4 >> 16) & 0xFFu);
  const int e3 = (int)((sc4 >> 24) & 0xFFu);
  const int max_e = max(max(e0, e1), max(e2, e3));
  const int e_g = (int)((sc4 >> (grp * 8)) & 0xFFu);
  const int rel_exp = 127 + 6 + e_g - max_e;
  const float rel_sc = (rel_exp > 0) ? __uint_as_float((uint32_t)rel_exp << 23) : 0.0f;

  const uint16_t packed = *reinterpret_cast<const uint16_t*>(src + lane * 2);
  uint8_t ob[4];
  ob[0] = f32_to_fp8_e4m3fn_ocp(fp4_e2m1_to_f32(packed & 0xFu) * rel_sc);
  ob[1] = f32_to_fp8_e4m3fn_ocp(fp4_e2m1_to_f32((packed >> 4) & 0xFu) * rel_sc);
  ob[2] = f32_to_fp8_e4m3fn_ocp(fp4_e2m1_to_f32((packed >> 8) & 0xFu) * rel_sc);
  ob[3] = f32_to_fp8_e4m3fn_ocp(fp4_e2m1_to_f32((packed >> 12) & 0xFu) * rel_sc);
  *reinterpret_cast<uint32_t*>(k_fp8_out + (int64_t)t * 128 + lane * 4) = *reinterpret_cast<const uint32_t*>(ob);
  if (lane == 0) {
    const int tok_exp = max_e - 6;
    k_scale_out[t] = (tok_exp > 0) ? __uint_as_float((uint32_t)tok_exp << 23) : (exp2f((float)(max_e - 127)) * (1.0f / 64.0f));
  }
}

void ernie_cp_gather_indexer_k_mxfp4_to_fp8(
    torch::Tensor k_cache, torch::Tensor k_fp8, torch::Tensor k_fp8_scale,
    torch::Tensor block_table, torch::Tensor cu_seqlen, torch::Tensor token_to_seq,
    int64_t block_size) {
  const int num_tokens = (int)k_fp8.size(0);
  if (num_tokens == 0) return;
  TORCH_CHECK(k_cache.is_cuda() && k_cache.scalar_type() == torch::kUInt8 && k_cache.size(-1) == 68);
  TORCH_CHECK(k_fp8.is_cuda() && k_fp8.dim() == 2 && k_fp8.size(1) == 128 && k_fp8.is_contiguous());
  auto bt = block_table.scalar_type() == torch::kInt32 ? block_table : block_table.to(torch::kInt32);
  auto cu = cu_seqlen.scalar_type() == torch::kInt32 ? cu_seqlen.contiguous() : cu_seqlen.to(torch::kInt32).contiguous();
  auto t2s = token_to_seq.scalar_type() == torch::kInt32 ? token_to_seq.contiguous() : token_to_seq.to(torch::kInt32).contiguous();
  auto sc_f32 = k_fp8_scale.view(torch::kFloat32).reshape({-1});
  const int num_batches = (int)cu.size(0) - 1;
  const int num_blocks = (int)k_cache.size(0);
  hipLaunchKernelGGL(
      k_cp_gather_indexer_k_mxfp4_to_fp8_128, dim3(num_tokens), dim3(32), 0, c10::hip::getCurrentHIPStream().stream(),
      k_cache.data_ptr<uint8_t>(), reinterpret_cast<uint8_t*>(k_fp8.data_ptr()),
      sc_f32.data_ptr<float>(), bt.data_ptr<int32_t>(), cu.data_ptr<int32_t>(), t2s.data_ptr<int32_t>(),
      k_cache.stride(0), k_cache.stride(1), bt.stride(0),
      num_tokens, num_batches, (int)bt.size(1), num_blocks, (int)block_size);
}


// ============================================================================
// v0.2.8 Fused DSA Prefill Q-Absorption + RoPE Concat & UV Projection (gfx1151 Wave32 WMMA):
//   Supports REAL ERNIE-5.5 DSA dims (D_NOPE=192, D_V=256) and standard MLA dims (128, 128).
// ============================================================================
typedef __bf16 bf16x16_q __attribute__((ext_vector_type(16)));
typedef float f32x8_q __attribute__((ext_vector_type(8)));
typedef uint32_t u4v_q __attribute__((ext_vector_type(4)));
typedef u4v_q u4va_q __attribute__((aligned(16)));
typedef uint32_t u8v_q __attribute__((ext_vector_type(8)));

__device__ __forceinline__ uint16_t quk_f32_to_bf16_rne(float f) {
  uint32_t u = __float_as_uint(f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

template <int D_NOPE, int NT_LOAD, int NT_STORE>
__global__ __launch_bounds__(256, 2) void k_bmm_q_uk_cat_pe_bf16_singleq(
    const uint16_t* __restrict__ q_nope,   // [T, H, D_NOPE] bf16
    const uint16_t* __restrict__ w_uk_rm,  // [H, 512, D_NOPE] bf16 (contiguous)
    const uint16_t* __restrict__ q_pe,     // [T, H, 64] bf16
    uint16_t* __restrict__ q_full,         // [T, H, 576] bf16 (output)
    int64_t qn_s0, int64_t qn_s1,
    int64_t w_s0, int64_t w_s1,
    int64_t qp_s0, int64_t qp_s1,
    int64_t qf_s0, int64_t qf_s1,
    int T) {
  constexpr int Q_STRIDE = D_NOPE + 8;  // 136 (D=128) or 200 (D=192)
  constexpr int BK = (D_NOPE == 192) ? 16 : 32;
  constexpr int W_STRIDE = BK + 8;      // 24 (BK=16) or 40 (BK=32)
  constexpr int E_ROWS = (D_NOPE == 192) ? 16 : 32;
  __shared__ __attribute__((aligned(16))) uint16_t Q_lds[64][Q_STRIDE];
  __shared__ __attribute__((aligned(16))) union {
    uint16_t W[128][W_STRIDE];
    uint16_t E[E_ROWS][136];
  } smem;

  const int h  = blockIdx.x;
  const int t0 = blockIdx.y * 64;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int wm = wave >> 2;
  const int wn = wave & 3;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;

  // 1. Coalesced load Q_nope[64 x D_NOPE] ONCE into Q_lds
  if constexpr (D_NOPE == 128) {
    const int r_base = tid >> 4;
    const int c_off  = (tid & 15) * 8;
#pragma unroll
    for (int ri = 0; ri < 4; ++ri) {
      const int qr = ri * 16 + r_base;
      const int t = t0 + qr;
      if (t < T) {
        const u4va_q* src = reinterpret_cast<const u4va_q*>(q_nope + (int64_t)t * qn_s0 + (int64_t)h * qn_s1 + c_off);
        *reinterpret_cast<u4va_q*>(&Q_lds[qr][c_off]) = NT_LOAD ? __builtin_nontemporal_load(src) : *src;
      } else {
        *reinterpret_cast<u4va_q*>(&Q_lds[qr][c_off]) = (u4v_q){0u, 0u, 0u, 0u};
      }
    }
  } else {
    const int r_base = tid >> 3;       // 0..31
    const int c_sub  = (tid & 7) * 8;  // 0..56
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const int qr = rh * 32 + r_base;
      const int t = t0 + qr;
      if (t < T) {
        const uint16_t* rowp = q_nope + (int64_t)t * qn_s0 + (int64_t)h * qn_s1 + c_sub;
#pragma unroll
        for (int cc = 0; cc < 3; ++cc) {
          const u4va_q* src = reinterpret_cast<const u4va_q*>(rowp + cc * 64);
          *reinterpret_cast<u4va_q*>(&Q_lds[qr][cc * 64 + c_sub]) = NT_LOAD ? __builtin_nontemporal_load(src) : *src;
        }
      } else {
        const u4v_q z = {0u, 0u, 0u, 0u};
#pragma unroll
        for (int cc = 0; cc < 3; ++cc) *reinterpret_cast<u4va_q*>(&Q_lds[qr][cc * 64 + c_sub]) = z;
      }
    }
  }

  // 2. Coalesced copy q_pe[64 x 64] -> q_full[:, h, 512:576] ONCE
  {
    const int pr_base = tid >> 3;
    const int pc_off  = (tid & 7) * 8;
#pragma unroll
    for (int ri = 0; ri < 2; ++ri) {
      const int pr = ri * 32 + pr_base;
      const int t = t0 + pr;
      if (t < T) {
        const u4va_q* ps = reinterpret_cast<const u4va_q*>(q_pe + (int64_t)t * qp_s0 + (int64_t)h * qp_s1 + pc_off);
        const u4v_q v = NT_LOAD ? __builtin_nontemporal_load(ps) : *ps;
        u4va_q* pd = reinterpret_cast<u4va_q*>(q_full + (int64_t)t * qf_s0 + (int64_t)h * qf_s1 + 512 + pc_off);
        if constexpr (NT_STORE) __builtin_nontemporal_store(v, pd);
        else *pd = v;
      }
    }
  }

  const uint16_t* w_head = w_uk_rm + (int64_t)h * w_s0;

#pragma unroll 1
  for (int nb = 0; nb < 4; ++nb) {
    const int n0 = nb * 128;
    f32x8_q acc[2][2];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int n = 0; n < 2; ++n) acc[m][n] = (f32x8_q){0, 0, 0, 0, 0, 0, 0, 0};

    if constexpr (BK == 32) {
      const int wr0 = tid >> 2;
      const int wc  = (tid & 3) * 8;
      const uint16_t* w_base = w_head + (int64_t)(n0 + wr0) * w_s1 + wc;
      u4v_q w0_cur = *reinterpret_cast<const u4va_q*>(w_base);
      u4v_q w1_cur = *reinterpret_cast<const u4va_q*>(w_base + (int64_t)64 * w_s1);
#pragma unroll 1
      for (int k0 = 0; k0 < D_NOPE; k0 += 32) {
        *reinterpret_cast<u4va_q*>(&smem.W[wr0][wc])      = w0_cur;
        *reinterpret_cast<u4va_q*>(&smem.W[wr0 + 64][wc]) = w1_cur;
        __syncthreads();
        u4v_q w0_nxt = {0u, 0u, 0u, 0u}, w1_nxt = {0u, 0u, 0u, 0u};
        if (k0 + 32 < D_NOPE) {
          w0_nxt = *reinterpret_cast<const u4va_q*>(w_base + k0 + 32);
          w1_nxt = *reinterpret_cast<const u4va_q*>(w_base + (int64_t)64 * w_s1 + k0 + 32);
        }
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          const int k_off = ks * 16;
          bf16x16_q af[2], bf[2];
#pragma unroll
          for (int m = 0; m < 2; ++m) {
            const int qr = wm * 32 + m * 16 + l16;
            const u4v_q a0 = *reinterpret_cast<const u4va_q*>(&Q_lds[qr][k0 + k_off + 0]);
            const u4v_q a1 = *reinterpret_cast<const u4va_q*>(&Q_lds[qr][k0 + k_off + 8]);
            af[m] = __builtin_bit_cast(bf16x16_q, (u8v_q){a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w});
          }
#pragma unroll
          for (int n = 0; n < 2; ++n) {
            const int w_r = wn * 32 + n * 16 + l16;
            const u4v_q b0 = *reinterpret_cast<const u4va_q*>(&smem.W[w_r][k_off + 0]);
            const u4v_q b1 = *reinterpret_cast<const u4va_q*>(&smem.W[w_r][k_off + 8]);
            bf[n] = __builtin_bit_cast(bf16x16_q, (u8v_q){b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w});
          }
#pragma unroll
          for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int n = 0; n < 2; ++n)
              acc[m][n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[m], bf[n], acc[m][n]);
        }
        __syncthreads();
        w0_cur = w0_nxt; w1_cur = w1_nxt;
      }
    } else {
      const int wr = tid >> 1;
      const int wc = (tid & 1) * 8;
      const uint16_t* w_base = w_head + (int64_t)(n0 + wr) * w_s1 + wc;
      u4v_q w_cur = *reinterpret_cast<const u4va_q*>(w_base);
#pragma unroll 1
      for (int k0 = 0; k0 < D_NOPE; k0 += 16) {
        *reinterpret_cast<u4va_q*>(&smem.W[wr][wc]) = w_cur;
        __syncthreads();
        u4v_q w_nxt = {0u, 0u, 0u, 0u};
        if (k0 + 16 < D_NOPE) w_nxt = *reinterpret_cast<const u4va_q*>(w_base + k0 + 16);
        bf16x16_q af[2], bf[2];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          const int qr = wm * 32 + m * 16 + l16;
          const u4v_q a0 = *reinterpret_cast<const u4va_q*>(&Q_lds[qr][k0 + 0]);
          const u4v_q a1 = *reinterpret_cast<const u4va_q*>(&Q_lds[qr][k0 + 8]);
          af[m] = __builtin_bit_cast(bf16x16_q, (u8v_q){a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w});
        }
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          const int w_r = wn * 32 + n * 16 + l16;
          const u4v_q b0 = *reinterpret_cast<const u4va_q*>(&smem.W[w_r][0]);
          const u4v_q b1 = *reinterpret_cast<const u4va_q*>(&smem.W[w_r][8]);
          bf[n] = __builtin_bit_cast(bf16x16_q, (u8v_q){b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w});
        }
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
          for (int n = 0; n < 2; ++n)
            acc[m][n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[m], bf[n], acc[m][n]);
        __syncthreads();
        w_cur = w_nxt;
      }
    }

    const int er_base = tid >> 4;
    const int ec_off  = (tid & 15) * 8;
    if constexpr (E_ROWS == 32) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        if (wm == half) {
#pragma unroll
          for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int ei = 0; ei < 8; ++ei) {
              const int r = m * 16 + 2 * ei + hrow;
              smem.E[r][wn * 32 + l16]      = quk_f32_to_bf16_rne(acc[m][0][ei]);
              smem.E[r][wn * 32 + 16 + l16] = quk_f32_to_bf16_rne(acc[m][1][ei]);
            }
        }
        __syncthreads();
#pragma unroll
        for (int ri = 0; ri < 2; ++ri) {
          const int r = ri * 16 + er_base;
          const int t = t0 + half * 32 + r;
          if (t < T) {
            const u4v_q v = *reinterpret_cast<const u4va_q*>(&smem.E[r][ec_off]);
            u4va_q* pd = reinterpret_cast<u4va_q*>(q_full + (int64_t)t * qf_s0 + (int64_t)h * qf_s1 + n0 + ec_off);
            if constexpr (NT_STORE) __builtin_nontemporal_store(v, pd);
            else *pd = v;
          }
        }
        __syncthreads();
      }
    } else {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          if (wm == half) {
#pragma unroll
            for (int ei = 0; ei < 8; ++ei) {
              const int r = 2 * ei + hrow;
              smem.E[r][wn * 32 + l16]      = quk_f32_to_bf16_rne(acc[m][0][ei]);
              smem.E[r][wn * 32 + 16 + l16] = quk_f32_to_bf16_rne(acc[m][1][ei]);
            }
          }
          __syncthreads();
          const int t = t0 + half * 32 + m * 16 + er_base;
          if (t < T) {
            const u4v_q v = *reinterpret_cast<const u4va_q*>(&smem.E[er_base][ec_off]);
            u4va_q* pd = reinterpret_cast<u4va_q*>(q_full + (int64_t)t * qf_s0 + (int64_t)h * qf_s1 + n0 + ec_off);
            if constexpr (NT_STORE) __builtin_nontemporal_store(v, pd);
            else *pd = v;
          }
          __syncthreads();
        }
      }
    }
  }
}

template <int NT_LOAD, int NT_STORE>
__global__ __launch_bounds__(256, 2) void k_bmm_q_uk_cat_pe_bf16_bk64_192(
    const uint16_t* __restrict__ q_nope,
    const uint16_t* __restrict__ w_uk_rm,
    const uint16_t* __restrict__ q_pe,
    uint16_t* __restrict__ q_full,
    int64_t qn_s0, int64_t qn_s1,
    int64_t w_s0, int64_t w_s1,
    int64_t qp_s0, int64_t qp_s1,
    int64_t qf_s0, int64_t qf_s1,
    int T) {
  __shared__ __attribute__((aligned(16))) uint16_t A_lds[64][72];
  __shared__ __attribute__((aligned(16))) union {
    uint16_t W[128][72];
    uint16_t E[64][136];
  } smem;

  const int h  = blockIdx.x;
  const int t0 = blockIdx.y * 64;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int wm = wave >> 2;
  const int wn = wave & 3;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;

  const int r_base = tid >> 3;
  const int c_off  = (tid & 7) * 8;

  {
#pragma unroll
    for (int ri = 0; ri < 2; ++ri) {
      const int pr = ri * 32 + r_base;
      const int t = t0 + pr;
      if (t < T) {
        const u4va_q* ps = reinterpret_cast<const u4va_q*>(q_pe + (int64_t)t * qp_s0 + (int64_t)h * qp_s1 + c_off);
        const u4v_q v = NT_LOAD ? __builtin_nontemporal_load(ps) : *ps;
        u4va_q* pd = reinterpret_cast<u4va_q*>(q_full + (int64_t)t * qf_s0 + (int64_t)h * qf_s1 + 512 + c_off);
        if constexpr (NT_STORE) __builtin_nontemporal_store(v, pd);
        else *pd = v;
      }
    }
  }

  const uint16_t* a_head = q_nope  + (int64_t)h * qn_s1 + c_off;
  const uint16_t* w_head = w_uk_rm + (int64_t)h * w_s0  + c_off;
  const int ta0 = t0 + r_base;
  const int ta1 = t0 + r_base + 32;
  const bool va0 = ta0 < T;
  const bool va1 = ta1 < T;

#pragma unroll 1
  for (int nb = 0; nb < 4; ++nb) {
    const int n0 = nb * 128;
    f32x8_q acc[2][2];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int n = 0; n < 2; ++n) acc[m][n] = (f32x8_q){0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll 1
    for (int k0 = 0; k0 < 192; k0 += 64) {
      u4v_q a0 = {0u, 0u, 0u, 0u}, a1 = {0u, 0u, 0u, 0u};
      if (va0) a0 = *reinterpret_cast<const u4va_q*>(a_head + (int64_t)ta0 * qn_s0 + k0);
      if (va1) a1 = *reinterpret_cast<const u4va_q*>(a_head + (int64_t)ta1 * qn_s0 + k0);
      const u4v_q w0 = *reinterpret_cast<const u4va_q*>(w_head + (int64_t)(n0 + r_base +  0) * w_s1 + k0);
      const u4v_q w1 = *reinterpret_cast<const u4va_q*>(w_head + (int64_t)(n0 + r_base + 32) * w_s1 + k0);
      const u4v_q w2 = *reinterpret_cast<const u4va_q*>(w_head + (int64_t)(n0 + r_base + 64) * w_s1 + k0);
      const u4v_q w3 = *reinterpret_cast<const u4va_q*>(w_head + (int64_t)(n0 + r_base + 96) * w_s1 + k0);
      *reinterpret_cast<u4va_q*>(&A_lds[r_base][c_off])      = a0;
      *reinterpret_cast<u4va_q*>(&A_lds[r_base + 32][c_off]) = a1;
      *reinterpret_cast<u4va_q*>(&smem.W[r_base +  0][c_off]) = w0;
      *reinterpret_cast<u4va_q*>(&smem.W[r_base + 32][c_off]) = w1;
      *reinterpret_cast<u4va_q*>(&smem.W[r_base + 64][c_off]) = w2;
      *reinterpret_cast<u4va_q*>(&smem.W[r_base + 96][c_off]) = w3;
      __syncthreads();

#pragma unroll 1
      for (int ks = 0; ks < 4; ++ks) {
        const int k_off = ks * 16;
        bf16x16_q af[2], bf[2];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          const int ar = wm * 32 + m * 16 + l16;
          const u4v_q x0 = *reinterpret_cast<const u4va_q*>(&A_lds[ar][k_off + 0]);
          const u4v_q x1 = *reinterpret_cast<const u4va_q*>(&A_lds[ar][k_off + 8]);
          af[m] = __builtin_bit_cast(bf16x16_q, (u8v_q){x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w});
        }
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          const int wr = wn * 32 + n * 16 + l16;
          const u4v_q b0 = *reinterpret_cast<const u4va_q*>(&smem.W[wr][k_off + 0]);
          const u4v_q b1 = *reinterpret_cast<const u4va_q*>(&smem.W[wr][k_off + 8]);
          bf[n] = __builtin_bit_cast(bf16x16_q, (u8v_q){b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w});
        }
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
          for (int n = 0; n < 2; ++n)
            acc[m][n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[m], bf[n], acc[m][n]);
        __builtin_amdgcn_sched_barrier(0);
      }
      __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) {
        const int r = wm * 32 + m * 16 + 2 * ei + hrow;
        smem.E[r][wn * 32 + l16]      = quk_f32_to_bf16_rne(acc[m][0][ei]);
        smem.E[r][wn * 32 + 16 + l16] = quk_f32_to_bf16_rne(acc[m][1][ei]);
      }
    __syncthreads();
    const int er_base = tid >> 4;
    const int ec_off  = (tid & 15) * 8;
#pragma unroll
    for (int ri = 0; ri < 4; ++ri) {
      const int r = ri * 16 + er_base;
      const int t = t0 + r;
      if (t < T) {
        const u4v_q v = *reinterpret_cast<const u4va_q*>(&smem.E[r][ec_off]);
        u4va_q* pd = reinterpret_cast<u4va_q*>(q_full + (int64_t)t * qf_s0 + (int64_t)h * qf_s1 + n0 + ec_off);
        if constexpr (NT_STORE) __builtin_nontemporal_store(v, pd);
        else *pd = v;
      }
    }
    __syncthreads();
  }
}


__device__ __forceinline__ u4v_q quk_nt_load_b128(const uint16_t* p) {
  return __builtin_nontemporal_load(reinterpret_cast<const u4va_q*>(p));
}

__device__ __forceinline__ u4v_q quk_shfl_xor16_u4(u4v_q v) {
  u4v_q r;
  r[0] = __shfl_xor(v[0], 16, 32);
  r[1] = __shfl_xor(v[1], 16, 32);
  r[2] = __shfl_xor(v[2], 16, 32);
  r[3] = __shfl_xor(v[3], 16, 32);
  return r;
}

// v0.2.8 Decode Fast-Path (T <= 16) for bmm_q_uk_cat_pe_bf16:
//   Zero-LDS-W, zero-loop-barrier, NT b128 streaming Wave32 WMMA (hits ~212 GB/s LPDDR5X roofline, 1.63-1.74x, bit-exact).
template <int D_NOPE, int NWAVES>
__global__ __launch_bounds__(NWAVES * 32, 4) void k_bmm_q_uk_cat_pe_bf16_decode(
    const uint16_t* __restrict__ q_nope,
    const uint16_t* __restrict__ w_uk_rm,
    const uint16_t* __restrict__ q_pe,
    uint16_t* __restrict__ q_full,
    int64_t qn_s0, int64_t qn_s1,
    int64_t w_s0, int64_t w_s1,
    int64_t qp_s0, int64_t qp_s1,
    int64_t qf_s0, int64_t qf_s1,
    int T) {
  constexpr int NTHR = NWAVES * 32;
  constexpr int BN   = NWAVES * 16;
  constexpr int C4   = D_NOPE / 8;
  __shared__ __attribute__((aligned(16))) uint16_t A_lds[16][D_NOPE + 8];

  const int h    = blockIdx.x;
  const int n0   = blockIdx.y * BN;
  const int tid  = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int l16  = lane & 15;
  const int hrow = lane >> 4;

  if (blockIdx.y == 0) {
    for (int idx = tid; idx < T * 8; idx += NTHR) {
      const int t = idx >> 3;
      const int c = (idx & 7) * 8;
      const u4v_q v = *reinterpret_cast<const u4va_q*>(q_pe + (int64_t)t * qp_s0 + (int64_t)h * qp_s1 + c);
      *reinterpret_cast<u4va_q*>(q_full + (int64_t)t * qf_s0 + (int64_t)h * qf_s1 + 512 + c) = v;
    }
  }

  const uint16_t* a_h = q_nope + (int64_t)h * qn_s1;
#pragma unroll
  for (int idx = tid; idx < 16 * C4; idx += NTHR) {
    const int r = idx / C4;
    const int c = (idx % C4) * 8;
    u4v_q v = {0u, 0u, 0u, 0u};
    if (r < T) v = *reinterpret_cast<const u4va_q*>(a_h + (int64_t)r * qn_s0 + c);
    *reinterpret_cast<u4va_q*>(&A_lds[r][c]) = v;
  }
  __syncthreads();

  const int wn0 = n0 + wave * 16;
  const uint16_t* wr = w_uk_rm + (int64_t)h * w_s0 + (int64_t)(wn0 + l16) * w_s1 + hrow * 16;

  f32x8_q acc = {0, 0, 0, 0, 0, 0, 0, 0};
  u4v_q cw0 = quk_nt_load_b128(wr + 0);
  u4v_q cw1 = quk_nt_load_b128(wr + 8);

#pragma unroll
  for (int k0 = 0; k0 < D_NOPE; k0 += 32) {
    u4v_q nw0, nw1;
    if (k0 + 32 < D_NOPE) {
      nw0 = quk_nt_load_b128(wr + k0 + 32 + 0);
      nw1 = quk_nt_load_b128(wr + k0 + 32 + 8);
    }

    const u4v_q xw0 = quk_shfl_xor16_u4(cw0);
    const u4v_q xw1 = quk_shfl_xor16_u4(cw1);
    const u4v_q b0_0 = hrow ? xw0 : cw0;
    const u4v_q b0_1 = hrow ? xw1 : cw1;
    const u4v_q b1_0 = hrow ? cw0 : xw0;
    const u4v_q b1_1 = hrow ? cw1 : xw1;

    const u4v_q a0_0 = *reinterpret_cast<const u4va_q*>(&A_lds[l16][k0 +  0]);
    const u4v_q a0_1 = *reinterpret_cast<const u4va_q*>(&A_lds[l16][k0 +  8]);
    const u4v_q a1_0 = *reinterpret_cast<const u4va_q*>(&A_lds[l16][k0 + 16]);
    const u4v_q a1_1 = *reinterpret_cast<const u4va_q*>(&A_lds[l16][k0 + 24]);

    const bf16x16_q af0 = __builtin_bit_cast(bf16x16_q, (u8v_q){a0_0.x, a0_0.y, a0_0.z, a0_0.w, a0_1.x, a0_1.y, a0_1.z, a0_1.w});
    const bf16x16_q bf0 = __builtin_bit_cast(bf16x16_q, (u8v_q){b0_0.x, b0_0.y, b0_0.z, b0_0.w, b0_1.x, b0_1.y, b0_1.z, b0_1.w});
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af0, bf0, acc);

    const bf16x16_q af1 = __builtin_bit_cast(bf16x16_q, (u8v_q){a1_0.x, a1_0.y, a1_0.z, a1_0.w, a1_1.x, a1_1.y, a1_1.z, a1_1.w});
    const bf16x16_q bf1 = __builtin_bit_cast(bf16x16_q, (u8v_q){b1_0.x, b1_0.y, b1_0.z, b1_0.w, b1_1.x, b1_1.y, b1_1.z, b1_1.w});
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af1, bf1, acc);

    if (k0 + 32 < D_NOPE) { cw0 = nw0; cw1 = nw1; }
  }

  uint16_t* o_base = q_full + (int64_t)h * qf_s1 + wn0 + l16;
#pragma unroll
  for (int ei = 0; ei < 8; ++ei) {
    const int t = 2 * ei + hrow;
    if (t < T) {
      o_base[(int64_t)t * qf_s0] = quk_f32_to_bf16_rne(acc[ei]);
    }
  }
}


// ============================================================================
// Wave 33: Single-Pass 64x256 DSA BMMs (BK=64, 32-bank XOR swizzle, GL2-swizzled UV_PROJ)
//   - k_bmm_uv_bf16_64x256_swz<1, 1, 8, 8>: reads attn_lat ONCE (eliminates 1.61 GB/6L duplicate reads!),
//     8-head x 8-mt GL2 swizzle pins 2 MiB w_uv in 4 MiB GL2 (0 TLB thrash), NT_A=1 + NT_O=1 -> 1.21x (-5.3..-6.0ms/4K, 0.0 parity!)
//   - k_bmm_q_uk_cat_pe_singleq_64x256_swz<1, 0, 1, 1>: reads q_nope ONCE into Q_lds, only 2 nb passes x 3 K-steps (6 barriers vs 48!)
//     -> 1.20x (-3.7..-4.1ms/4K, 0.0 parity!)
// ============================================================================
__device__ __forceinline__ int bmm_sw32(int row, int col4) { return ((col4 ^ (row & 7)) << 2); }

template <int GH, int GM>
__device__ __forceinline__ void bmm_swizzle_hm(int H, int mt, int& h, int& m_blk) {
  if constexpr (GM <= 1) {
    h = blockIdx.x;
    m_blk = blockIdx.y;
  } else {
    const int lin = blockIdx.y * H + blockIdx.x;
    const int mg = lin / (H * GM);
    const int rem_m = lin - mg * (H * GM);
    const int m_cnt = min(GM, mt - mg * GM);
    const int hg = rem_m / (GH * m_cnt);
    const int rem_h = rem_m - hg * (GH * m_cnt);
    const int h_cnt = min(GH, H - hg * GH);
    const int loc_m = rem_h / h_cnt;
    const int loc_h = rem_h - loc_m * h_cnt;
    h = hg * GH + loc_h;
    m_blk = mg * GM + loc_m;
  }
}

template <int NT_A, int NT_O, int GH, int GM>
__global__ __launch_bounds__(256, 2) void k_bmm_uv_bf16_64x256_swz(
    const uint16_t* __restrict__ attn_lat,
    const uint16_t* __restrict__ w_uv_rm,
    uint16_t* __restrict__ out,
    int64_t a_s0, int64_t a_s1,
    int64_t w_s0, int64_t w_s1,
    int64_t o_s0, int64_t o_s1,
    int T, int H, int mt) {
  __shared__ __attribute__((aligned(16))) union {
    struct {
      uint32_t A[64][32];    // 8 KiB
      uint32_t W[256][32];   // 32 KiB
    } stage;                 // 40 KiB -> 3 WGs/WGP!
    uint16_t E[32][256];     // 16 KiB
  } smem;

  int h, m_blk;
  bmm_swizzle_hm<GH, GM>(H, mt, h, m_blk);
  const int t0 = m_blk * 64;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int wm = wave >> 2;
  const int wn = wave & 3;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;
  const int r_loc = tid >> 3;
  const int c4 = tid & 7;
  const int c8 = c4 * 8;

  const uint16_t* a_head = attn_lat + (int64_t)h * a_s1 + c8;
  const uint16_t* w_head = w_uv_rm  + (int64_t)h * w_s0 + c8;

  auto load_tile = [&](int k0, u4v_q ar[2], u4v_q wr[8]) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int ta = t0 + i * 32 + r_loc;
      if (ta < T) {
        const u4va_q* pa = reinterpret_cast<const u4va_q*>(a_head + (int64_t)ta * a_s0 + k0);
        ar[i] = NT_A ? __builtin_nontemporal_load(pa) : *pa;
      } else {
        ar[i] = (u4v_q){0u, 0u, 0u, 0u};
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      wr[i] = *reinterpret_cast<const u4va_q*>(w_head + (int64_t)(i * 32 + r_loc) * w_s1 + k0);
    }
  };
  auto store_tile = [&](const u4v_q ar[2], const u4v_q wr[8]) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int rr = i * 32 + r_loc;
      *reinterpret_cast<u4va_q*>(&smem.stage.A[rr][bmm_sw32(rr, c4)]) = ar[i];
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int rr = i * 32 + r_loc;
      *reinterpret_cast<u4va_q*>(&smem.stage.W[rr][bmm_sw32(rr, c4)]) = wr[i];
    }
  };

  f32x8_q acc[2][4];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int n = 0; n < 4; ++n) acc[m][n] = (f32x8_q){0,0,0,0,0,0,0,0};

  u4v_q a_cur[2], w_cur[8], a_nxt[2], w_nxt[8];
  load_tile(0, a_cur, w_cur);

#pragma unroll 1
  for (int k0 = 0; k0 < 512; k0 += 64) {
    store_tile(a_cur, w_cur);
    __syncthreads();
    if (k0 + 64 < 512) load_tile(k0 + 64, a_nxt, w_nxt);

#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const int c_lo = ks * 2, c_hi = ks * 2 + 1;
      bf16x16_q af[2];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int ar = wm * 32 + m * 16 + l16;
        const u4v_q a0 = *reinterpret_cast<const u4va_q*>(&smem.stage.A[ar][bmm_sw32(ar, c_lo)]);
        const u4v_q a1 = *reinterpret_cast<const u4va_q*>(&smem.stage.A[ar][bmm_sw32(ar, c_hi)]);
        af[m] = __builtin_bit_cast(bf16x16_q, (u8v_q){a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w});
      }
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const int wr = wn * 64 + n * 16 + l16;
        const u4v_q b0 = *reinterpret_cast<const u4va_q*>(&smem.stage.W[wr][bmm_sw32(wr, c_lo)]);
        const u4v_q b1 = *reinterpret_cast<const u4va_q*>(&smem.stage.W[wr][bmm_sw32(wr, c_hi)]);
        const bf16x16_q bf = __builtin_bit_cast(bf16x16_q, (u8v_q){b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w});
        acc[0][n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[0], bf, acc[0][n]);
        acc[1][n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[1], bf, acc[1][n]);
        __builtin_amdgcn_sched_barrier(0);
      }
    }
    if (k0 + 64 < 512) {
      __syncthreads();
#pragma unroll
      for (int i = 0; i < 2; ++i) a_cur[i] = a_nxt[i];
#pragma unroll
      for (int i = 0; i < 8; ++i) w_cur[i] = w_nxt[i];
    }
  }

  uint16_t* o_head = out + (int64_t)h * o_s1;
#pragma unroll 1
  for (int hf = 0; hf < 2; ++hf) {
    __syncthreads();
    if (wm == hf) {
#pragma unroll
      for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          const int col = wn * 64 + n * 16 + l16;
#pragma unroll
          for (int ei = 0; ei < 8; ++ei) {
            smem.E[m * 16 + 2 * ei + hrow][col] = quk_f32_to_bf16_rne(acc[m][n][ei]);
          }
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int ri = 0; ri < 4; ++ri) {
      const int rl = wave * 4 + ri;
      const int t = t0 + hf * 32 + rl;
      if (t < T) {
        const u4v_q v = *reinterpret_cast<const u4va_q*>(&smem.E[rl][lane * 8]);
        u4va_q* dst = reinterpret_cast<u4va_q*>(o_head + (int64_t)t * o_s0 + lane * 8);
        if constexpr (NT_O) __builtin_nontemporal_store(v, dst);
        else *dst = v;
      }
    }
  }
}

template <int NT_A, int NT_O, int GH, int GM>
__global__ __launch_bounds__(256, 2) void k_bmm_q_uk_cat_pe_singleq_64x256_swz(
    const uint16_t* __restrict__ q_nope,
    const uint16_t* __restrict__ w_uk_rm,
    const uint16_t* __restrict__ q_pe,
    uint16_t* __restrict__ q_full,
    int64_t qn_s0, int64_t qn_s1,
    int64_t w_s0,  int64_t w_s1,
    int64_t qp_s0, int64_t qp_s1,
    int64_t qf_s0, int64_t qf_s1,
    int T, int H, int mt) {
  __shared__ __attribute__((aligned(16))) uint32_t Q_lds[64][96];
  __shared__ __attribute__((aligned(16))) union {
    uint32_t W[256][32];
    uint16_t E[32][256];
  } smem;

  int h, m_blk;
  bmm_swizzle_hm<GH, GM>(H, mt, h, m_blk);
  const int t0 = m_blk * 64;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int wm = wave >> 2;
  const int wn = wave & 3;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;
  const int r32 = tid >> 3;
  const int c4  = tid & 7;
  const int c8  = c4 * 8;

#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    const int qr = rh * 32 + r32;
    const int t = t0 + qr;
    const int sw = bmm_sw32(qr, c4);
    if (t < T) {
      const uint16_t* rowp = q_nope + (int64_t)t * qn_s0 + (int64_t)h * qn_s1 + c8;
#pragma unroll
      for (int cc = 0; cc < 3; ++cc) {
        const u4va_q* src = reinterpret_cast<const u4va_q*>(rowp + cc * 64);
        *reinterpret_cast<u4va_q*>(&Q_lds[qr][cc * 32 + sw]) = NT_A ? __builtin_nontemporal_load(src) : *src;
      }
      const u4va_q* ps = reinterpret_cast<const u4va_q*>(q_pe + (int64_t)t * qp_s0 + (int64_t)h * qp_s1 + c8);
      const u4v_q pv = NT_A ? __builtin_nontemporal_load(ps) : *ps;
      u4va_q* pd = reinterpret_cast<u4va_q*>(q_full + (int64_t)t * qf_s0 + (int64_t)h * qf_s1 + 512 + c8);
      if constexpr (NT_O) __builtin_nontemporal_store(pv, pd); else *pd = pv;
    } else {
      const u4v_q z = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int cc = 0; cc < 3; ++cc) *reinterpret_cast<u4va_q*>(&Q_lds[qr][cc * 32 + sw]) = z;
    }
  }

  const uint16_t* w_head = w_uk_rm + (int64_t)h * w_s0 + c8;
  uint16_t* o_head = q_full + (int64_t)h * qf_s1;

#pragma unroll 1
  for (int nb = 0; nb < 2; ++nb) {
    const int n0 = nb * 256;
    f32x8_q acc[2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int n = 0; n < 4; ++n) acc[m][n] = (f32x8_q){0,0,0,0,0,0,0,0};

    u4v_q w_cur[8], w_nxt[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      w_cur[i] = *reinterpret_cast<const u4va_q*>(w_head + (int64_t)(n0 + i * 32 + r32) * w_s1 + 0);

#pragma unroll 1
    for (int cc = 0; cc < 3; ++cc) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int rr = i * 32 + r32;
        *reinterpret_cast<u4va_q*>(&smem.W[rr][bmm_sw32(rr, c4)]) = w_cur[i];
      }
      __syncthreads();
      if (cc < 2) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          w_nxt[i] = *reinterpret_cast<const u4va_q*>(w_head + (int64_t)(n0 + i * 32 + r32) * w_s1 + (cc + 1) * 64);
      }

#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const int c_lo = ks * 2, c_hi = ks * 2 + 1;
        bf16x16_q af[2];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          const int ar = wm * 32 + m * 16 + l16;
          const u4v_q a0 = *reinterpret_cast<const u4va_q*>(&Q_lds[ar][cc * 32 + bmm_sw32(ar, c_lo)]);
          const u4v_q a1 = *reinterpret_cast<const u4va_q*>(&Q_lds[ar][cc * 32 + bmm_sw32(ar, c_hi)]);
          af[m] = __builtin_bit_cast(bf16x16_q, (u8v_q){a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w});
        }
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          const int wr = wn * 64 + n * 16 + l16;
          const u4v_q b0 = *reinterpret_cast<const u4va_q*>(&smem.W[wr][bmm_sw32(wr, c_lo)]);
          const u4v_q b1 = *reinterpret_cast<const u4va_q*>(&smem.W[wr][bmm_sw32(wr, c_hi)]);
          const bf16x16_q bf = __builtin_bit_cast(bf16x16_q, (u8v_q){b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w});
          acc[0][n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[0], bf, acc[0][n]);
          acc[1][n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[1], bf, acc[1][n]);
          __builtin_amdgcn_sched_barrier(0);
        }
      }
      if (cc < 2) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 8; ++i) w_cur[i] = w_nxt[i];
      }
    }

#pragma unroll 1
    for (int hf = 0; hf < 2; ++hf) {
      __syncthreads();
      if (wm == hf) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
#pragma unroll
          for (int n = 0; n < 4; ++n) {
            const int col = wn * 64 + n * 16 + l16;
#pragma unroll
            for (int ei = 0; ei < 8; ++ei) {
              smem.E[m * 16 + 2 * ei + hrow][col] = quk_f32_to_bf16_rne(acc[m][n][ei]);
            }
          }
        }
      }
      __syncthreads();
#pragma unroll
      for (int ri = 0; ri < 4; ++ri) {
        const int rl = wave * 4 + ri;
        const int t = t0 + hf * 32 + rl;
        if (t < T) {
          const u4v_q v = *reinterpret_cast<const u4va_q*>(&smem.E[rl][lane * 8]);
          u4va_q* dst = reinterpret_cast<u4va_q*>(o_head + (int64_t)t * qf_s0 + n0 + lane * 8);
          if constexpr (NT_O) __builtin_nontemporal_store(v, dst);
          else *dst = v;
        }
      }
    }
    if (nb == 0) __syncthreads();
  }
}

torch::Tensor bmm_q_uk_cat_pe_bf16(
    torch::Tensor q_nope, torch::Tensor w_uk_rm, torch::Tensor q_pe) {
  const int D = (int)q_nope.size(2);
  TORCH_CHECK(q_nope.is_cuda() && q_nope.dim() == 3 && (D == 128 || D == 192) && q_nope.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(w_uk_rm.is_cuda() && w_uk_rm.dim() == 3 && w_uk_rm.size(1) == 512 && w_uk_rm.size(2) == D && w_uk_rm.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(q_pe.is_cuda() && q_pe.dim() == 3 && q_pe.size(2) == 64 && q_pe.scalar_type() == torch::kBFloat16);
  const int T = (int)q_nope.size(0);
  const int H = (int)q_nope.size(1);
  auto q_full = torch::empty({T, H, 576}, q_nope.options());
  if (T == 0) return q_full;
  auto st = c10::hip::getCurrentHIPStream().stream();
  const int mt = (T + 63) / 64;
  auto qn_p = reinterpret_cast<const uint16_t*>(q_nope.data_ptr<at::BFloat16>());
  auto w_p  = reinterpret_cast<const uint16_t*>(w_uk_rm.data_ptr<at::BFloat16>());
  auto qp_p = reinterpret_cast<const uint16_t*>(q_pe.data_ptr<at::BFloat16>());
  auto qf_p = reinterpret_cast<uint16_t*>(q_full.data_ptr<at::BFloat16>());
  if (T <= 16 && q_nope.stride(2) == 1 && w_uk_rm.stride(2) == 1 && q_pe.stride(2) == 1 &&
      ((q_nope.stride(0) | q_nope.stride(1) | w_uk_rm.stride(0) | w_uk_rm.stride(1) |
        q_pe.stride(0) | q_pe.stride(1)) & 7) == 0) {
    if (D == 128) {
      hipLaunchKernelGGL((k_bmm_q_uk_cat_pe_bf16_decode<128, 4>), dim3(H, 512 / 64), dim3(128), 0, st,
          qn_p, w_p, qp_p, qf_p, q_nope.stride(0), q_nope.stride(1),
          w_uk_rm.stride(0), w_uk_rm.stride(1), q_pe.stride(0), q_pe.stride(1),
          q_full.stride(0), q_full.stride(1), T);
    } else {
      hipLaunchKernelGGL((k_bmm_q_uk_cat_pe_bf16_decode<192, 4>), dim3(H, 512 / 64), dim3(128), 0, st,
          qn_p, w_p, qp_p, qf_p, q_nope.stride(0), q_nope.stride(1),
          w_uk_rm.stride(0), w_uk_rm.stride(1), q_pe.stride(0), q_pe.stride(1),
          q_full.stride(0), q_full.stride(1), T);
    }
    return q_full;
  }
  if (D == 128) {
    if (T >= 256 && T <= 1536) {
      hipLaunchKernelGGL((k_bmm_q_uk_cat_pe_bf16_singleq<128, 0, 1>), dim3(H, mt), dim3(256), 0, st,
          qn_p, w_p, qp_p, qf_p, q_nope.stride(0), q_nope.stride(1),
          w_uk_rm.stride(0), w_uk_rm.stride(1), q_pe.stride(0), q_pe.stride(1),
          q_full.stride(0), q_full.stride(1), T);
    } else {
      hipLaunchKernelGGL((k_bmm_q_uk_cat_pe_bf16_singleq<128, 1, 0>), dim3(H, mt), dim3(256), 0, st,
          qn_p, w_p, qp_p, qf_p, q_nope.stride(0), q_nope.stride(1),
          w_uk_rm.stride(0), w_uk_rm.stride(1), q_pe.stride(0), q_pe.stride(1),
          q_full.stride(0), q_full.stride(1), T);
    }
  } else {
    // Empirically optimal gfx1151 policy for D_NOPE=192:
    //   T<=256: bk64 WC (20.2 TF/s, 2.35x); 256<T<=768: bk64 NT (16.6 TF/s, 3.47x); T>768: singleq WC (3.09-3.53x)
    if (T <= 256) {
      hipLaunchKernelGGL((k_bmm_q_uk_cat_pe_bf16_bk64_192<0, 0>), dim3(H, mt), dim3(256), 0, st,
          qn_p, w_p, qp_p, qf_p, q_nope.stride(0), q_nope.stride(1),
          w_uk_rm.stride(0), w_uk_rm.stride(1), q_pe.stride(0), q_pe.stride(1),
          q_full.stride(0), q_full.stride(1), T);
    } else if (T < 512) {
      hipLaunchKernelGGL((k_bmm_q_uk_cat_pe_bf16_bk64_192<0, 1>), dim3(H, mt), dim3(256), 0, st,
          qn_p, w_p, qp_p, qf_p, q_nope.stride(0), q_nope.stride(1),
          w_uk_rm.stride(0), w_uk_rm.stride(1), q_pe.stride(0), q_pe.stride(1),
          q_full.stride(0), q_full.stride(1), T);
    } else {
      hipLaunchKernelGGL((k_bmm_q_uk_cat_pe_singleq_64x256_swz<1, 0, 1, 1>), dim3(H, mt), dim3(256), 0, st,
          qn_p, w_p, qp_p, qf_p, q_nope.stride(0), q_nope.stride(1),
          w_uk_rm.stride(0), w_uk_rm.stride(1), q_pe.stride(0), q_pe.stride(1),
          q_full.stride(0), q_full.stride(1), T, H, mt);
    }
  }
  return q_full;
}

template <int NT_A, int NT_O, int NB_ITERS>
__global__ __launch_bounds__(256, 2) void k_bmm_uv_bf16_v3(
    const uint16_t* __restrict__ attn_lat,
    const uint16_t* __restrict__ w_uv_rm,
    uint16_t* __restrict__ out,
    int64_t a_s0, int64_t a_s1,
    int64_t w_s0, int64_t w_s1,
    int64_t o_s0, int64_t o_s1,
    int T) {
  __shared__ __attribute__((aligned(16))) uint16_t A_lds[64][72];
  __shared__ __attribute__((aligned(16))) union {
    uint16_t W[128][72];
    uint16_t E[64][136];
  } smem;

  const int h  = blockIdx.x;
  const int t0 = blockIdx.y * 64;
  const int nb_base = (NB_ITERS == 1) ? blockIdx.z : 0;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int wm = wave >> 2;
  const int wn = wave & 3;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int hrow = lane >> 4;

  const int r_base = tid >> 3;
  const int c_off  = (tid & 7) * 8;

  const uint16_t* a_head = attn_lat + (int64_t)h * a_s1 + c_off;
  const uint16_t* w_head = w_uv_rm  + (int64_t)h * w_s0 + c_off;
  const int ta0 = t0 + r_base;
  const int ta1 = t0 + r_base + 32;
  const bool va0 = ta0 < T;
  const bool va1 = ta1 < T;

#pragma unroll 1
  for (int nbi = 0; nbi < NB_ITERS; ++nbi) {
    const int n0 = (nb_base + nbi) * 128;
    f32x8_q acc[2][2];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int n = 0; n < 2; ++n) acc[m][n] = (f32x8_q){0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll 1
    for (int k0 = 0; k0 < 512; k0 += 64) {
      {
        u4v_q a0 = {0u, 0u, 0u, 0u}, a1 = {0u, 0u, 0u, 0u};
        if (va0) {
          const u4va_q* p = reinterpret_cast<const u4va_q*>(a_head + (int64_t)ta0 * a_s0 + k0);
          a0 = NT_A ? __builtin_nontemporal_load(p) : *p;
        }
        if (va1) {
          const u4va_q* p = reinterpret_cast<const u4va_q*>(a_head + (int64_t)ta1 * a_s0 + k0);
          a1 = NT_A ? __builtin_nontemporal_load(p) : *p;
        }
        const u4v_q w0 = *reinterpret_cast<const u4va_q*>(w_head + (int64_t)(n0 + r_base +  0) * w_s1 + k0);
        const u4v_q w1 = *reinterpret_cast<const u4va_q*>(w_head + (int64_t)(n0 + r_base + 32) * w_s1 + k0);
        const u4v_q w2 = *reinterpret_cast<const u4va_q*>(w_head + (int64_t)(n0 + r_base + 64) * w_s1 + k0);
        const u4v_q w3 = *reinterpret_cast<const u4va_q*>(w_head + (int64_t)(n0 + r_base + 96) * w_s1 + k0);
        *reinterpret_cast<u4va_q*>(&A_lds[r_base][c_off])      = a0;
        *reinterpret_cast<u4va_q*>(&A_lds[r_base + 32][c_off]) = a1;
        *reinterpret_cast<u4va_q*>(&smem.W[r_base +  0][c_off]) = w0;
        *reinterpret_cast<u4va_q*>(&smem.W[r_base + 32][c_off]) = w1;
        *reinterpret_cast<u4va_q*>(&smem.W[r_base + 64][c_off]) = w2;
        *reinterpret_cast<u4va_q*>(&smem.W[r_base + 96][c_off]) = w3;
      }
      __syncthreads();

#pragma unroll 1
      for (int ks = 0; ks < 4; ++ks) {
        const int k_off = ks * 16;
        bf16x16_q af[2], bf[2];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          const int ar = wm * 32 + m * 16 + l16;
          const u4v_q a0 = *reinterpret_cast<const u4va_q*>(&A_lds[ar][k_off + 0]);
          const u4v_q a1 = *reinterpret_cast<const u4va_q*>(&A_lds[ar][k_off + 8]);
          af[m] = __builtin_bit_cast(bf16x16_q, (u8v_q){a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w});
        }
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          const int wr = wn * 32 + n * 16 + l16;
          const u4v_q b0 = *reinterpret_cast<const u4va_q*>(&smem.W[wr][k_off + 0]);
          const u4v_q b1 = *reinterpret_cast<const u4va_q*>(&smem.W[wr][k_off + 8]);
          bf[n] = __builtin_bit_cast(bf16x16_q, (u8v_q){b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w});
        }
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
          for (int n = 0; n < 2; ++n)
            acc[m][n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[m], bf[n], acc[m][n]);
        __builtin_amdgcn_sched_barrier(0);
      }
      __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int ei = 0; ei < 8; ++ei) {
        const int r = wm * 32 + m * 16 + 2 * ei + hrow;
        smem.E[r][wn * 32 + l16]      = quk_f32_to_bf16_rne(acc[m][0][ei]);
        smem.E[r][wn * 32 + 16 + l16] = quk_f32_to_bf16_rne(acc[m][1][ei]);
      }
    __syncthreads();
    const int er_base = tid >> 4;
    const int ec_off  = (tid & 15) * 8;
#pragma unroll
    for (int ri = 0; ri < 4; ++ri) {
      const int r = ri * 16 + er_base;
      const int t = t0 + r;
      if (t < T) {
        const u4v_q v = *reinterpret_cast<const u4va_q*>(&smem.E[r][ec_off]);
        u4va_q* pd = reinterpret_cast<u4va_q*>(out + (int64_t)t * o_s0 + (int64_t)h * o_s1 + n0 + ec_off);
        if constexpr (NT_O) __builtin_nontemporal_store(v, pd);
        else *pd = v;
      }
    }
    if constexpr (NB_ITERS > 1) __syncthreads();
  }
}


// v0.2.8 Decode Fast-Path (T <= 16) for bmm_uv_bf16:
//   1-Wave32/WG, zero-barrier, 2-ahead NT b128 prefetch WMMA (16.6 KiB A_lds self-throttles to 7 WGs/WGP -> 2.02-2.40x, bit-exact).
__global__ __launch_bounds__(32) void k_bmm_uv_bf16_decode(
    const uint16_t* __restrict__ attn_lat,
    const uint16_t* __restrict__ w_uv_rm,
    uint16_t* __restrict__ out,
    int64_t a_s0, int64_t a_s1,
    int64_t w_s0, int64_t w_s1,
    int64_t o_s0, int64_t o_s1,
    int T) {
  __shared__ __attribute__((aligned(16))) uint16_t A_lds[16][520];

  const int h    = blockIdx.x;
  const int n0   = blockIdx.y * 16;
  const int lane = threadIdx.x;
  const int l16  = lane & 15;
  const int hrow = lane >> 4;

  const uint16_t* a_h = attn_lat + (int64_t)h * a_s1;
  const int tot_u4 = T * 64;
  for (int idx = lane; idx < tot_u4; idx += 32) {
    const int r = idx >> 6;
    const int c = (idx & 63) * 8;
    *reinterpret_cast<u4va_q*>(&A_lds[r][c]) = *reinterpret_cast<const u4va_q*>(a_h + (int64_t)r * a_s0 + c);
  }

  const uint16_t* wr = w_uv_rm + (int64_t)h * w_s0 + (int64_t)(n0 + l16) * w_s1 + hrow * 16;
  const bool va = l16 < T;
  const int ar_idx = va ? l16 : 0;

  f32x8_q acc = {0, 0, 0, 0, 0, 0, 0, 0};
  u4v_q w0_a = quk_nt_load_b128(wr + 0);
  u4v_q w1_a = quk_nt_load_b128(wr + 8);
  u4v_q w0_b = quk_nt_load_b128(wr + 32);
  u4v_q w1_b = quk_nt_load_b128(wr + 40);

#pragma unroll 4
  for (int k0 = 0; k0 < 512; k0 += 32) {
    u4v_q w0_n, w1_n;
    if (k0 + 64 < 512) {
      w0_n = quk_nt_load_b128(wr + k0 + 64 + 0);
      w1_n = quk_nt_load_b128(wr + k0 + 64 + 8);
    }

    const u4v_q xw0 = quk_shfl_xor16_u4(w0_a);
    const u4v_q xw1 = quk_shfl_xor16_u4(w1_a);
    const u4v_q b0_0 = hrow ? xw0 : w0_a;
    const u4v_q b0_1 = hrow ? xw1 : w1_a;
    const u4v_q b1_0 = hrow ? w0_a : xw0;
    const u4v_q b1_1 = hrow ? w1_a : xw1;

    const u4v_q a0_0 = va ? *reinterpret_cast<const u4va_q*>(&A_lds[ar_idx][k0 +  0]) : (u4v_q){0u, 0u, 0u, 0u};
    const u4v_q a0_1 = va ? *reinterpret_cast<const u4va_q*>(&A_lds[ar_idx][k0 +  8]) : (u4v_q){0u, 0u, 0u, 0u};
    const u4v_q a1_0 = va ? *reinterpret_cast<const u4va_q*>(&A_lds[ar_idx][k0 + 16]) : (u4v_q){0u, 0u, 0u, 0u};
    const u4v_q a1_1 = va ? *reinterpret_cast<const u4va_q*>(&A_lds[ar_idx][k0 + 24]) : (u4v_q){0u, 0u, 0u, 0u};

    const bf16x16_q af0 = __builtin_bit_cast(bf16x16_q, (u8v_q){a0_0.x, a0_0.y, a0_0.z, a0_0.w, a0_1.x, a0_1.y, a0_1.z, a0_1.w});
    const bf16x16_q bf0 = __builtin_bit_cast(bf16x16_q, (u8v_q){b0_0.x, b0_0.y, b0_0.z, b0_0.w, b0_1.x, b0_1.y, b0_1.z, b0_1.w});
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af0, bf0, acc);

    const bf16x16_q af1 = __builtin_bit_cast(bf16x16_q, (u8v_q){a1_0.x, a1_0.y, a1_0.z, a1_0.w, a1_1.x, a1_1.y, a1_1.z, a1_1.w});
    const bf16x16_q bf1 = __builtin_bit_cast(bf16x16_q, (u8v_q){b1_0.x, b1_0.y, b1_0.z, b1_0.w, b1_1.x, b1_1.y, b1_1.z, b1_1.w});
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af1, bf1, acc);

    w0_a = w0_b; w1_a = w1_b;
    if (k0 + 64 < 512) { w0_b = w0_n; w1_b = w1_n; }
  }

  uint16_t* o_base = out + (int64_t)h * o_s1 + n0 + l16;
#pragma unroll
  for (int ei = 0; ei < 8; ++ei) {
    const int t = 2 * ei + hrow;
    if (t < T) {
      o_base[(int64_t)t * o_s0] = quk_f32_to_bf16_rne(acc[ei]);
    }
  }
}

void bmm_uv_bf16(
    torch::Tensor attn_latent, torch::Tensor w_uv_rm, torch::Tensor out) {
  const int DV = (int)w_uv_rm.size(1);
  TORCH_CHECK(attn_latent.is_cuda() && attn_latent.dim() == 3 && attn_latent.size(2) >= 512 && attn_latent.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(w_uv_rm.is_cuda() && w_uv_rm.dim() == 3 && (DV == 128 || DV == 256) && w_uv_rm.size(2) == 512 && w_uv_rm.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(out.is_cuda() && out.dim() == 3 && out.size(2) == DV && out.scalar_type() == torch::kBFloat16);
  const int T = (int)attn_latent.size(0);
  const int H = (int)attn_latent.size(1);
  if (T == 0) return;
  auto st = c10::hip::getCurrentHIPStream().stream();
  const int mt = (T + 63) / 64;
  auto a_p = reinterpret_cast<const uint16_t*>(attn_latent.data_ptr<at::BFloat16>());
  auto w_p = reinterpret_cast<const uint16_t*>(w_uv_rm.data_ptr<at::BFloat16>());
  auto o_p = reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>());
  if (T <= 16 && attn_latent.stride(2) == 1 && w_uv_rm.stride(2) == 1 && out.stride(2) == 1 &&
      ((attn_latent.stride(0) | attn_latent.stride(1) | w_uv_rm.stride(0) | w_uv_rm.stride(1)) & 7) == 0) {
    hipLaunchKernelGGL(k_bmm_uv_bf16_decode, dim3(H, DV / 16), dim3(32), 0, st,
        a_p, w_p, o_p, attn_latent.stride(0), attn_latent.stride(1),
        w_uv_rm.stride(0), w_uv_rm.stride(1), out.stride(0), out.stride(1), T);
    return;
  }
  if (DV == 128) {
    hipLaunchKernelGGL((k_bmm_uv_bf16_v3<1, 1, 1>), dim3(H, mt, 1), dim3(256), 0, st,
        a_p, w_p, o_p, attn_latent.stride(0), attn_latent.stride(1),
        w_uv_rm.stride(0), w_uv_rm.stride(1), out.stride(0), out.stride(1), T);
  } else if (T >= 512 && H >= 8 && (H % 8) == 0) {
    // Wave 33: Single-pass 64x256 BK=64 with 8-head x 8-mt GL2 swizzle + NT_A=1 + NT_O=1 (1.21x vs v3, 0.0 parity!)
    hipLaunchKernelGGL((k_bmm_uv_bf16_64x256_swz<1, 1, 8, 8>), dim3(H, mt, 1), dim3(256), 0, st,
        a_p, w_p, o_p, attn_latent.stride(0), attn_latent.stride(1),
        w_uv_rm.stride(0), w_uv_rm.stride(1), out.stride(0), out.stride(1), T, H, mt);
  } else {
    // Fallback for small T (<512) or non-multiple-of-8 H: single-pass 64x256 without swizzle if T>=256
    if (T >= 256) {
      hipLaunchKernelGGL((k_bmm_uv_bf16_64x256_swz<1, 1, 1, 1>), dim3(H, mt, 1), dim3(256), 0, st,
          a_p, w_p, o_p, attn_latent.stride(0), attn_latent.stride(1),
          w_uv_rm.stride(0), w_uv_rm.stride(1), out.stride(0), out.stride(1), T, H, mt);
    } else {
      hipLaunchKernelGGL((k_bmm_uv_bf16_v3<0, 1, 2>), dim3(H, mt, 1), dim3(256), 0, st,
          a_p, w_p, o_p, attn_latent.stride(0), attn_latent.stride(1),
          w_uv_rm.stride(0), w_uv_rm.stride(1), out.stride(0), out.stride(1), T);
    }
  }
}


// =============================================================================
// Wave 27: RDNA 3.5 (gfx1151) BF16 x BF16 -> FP32 WMMA Router GEMM (BK=64)
//   Replaces Tensile BSS MT32x32x64 fallback (9.22 ms/L -> ~1.67 ms/L, 4.0-4.8x!)
//   32-bank conflict-free XOR swizzle (sw32), zero-LDS coalesced 64B NT FP32 store.
// =============================================================================
struct w27_u2_128 { uint4 a; uint4 b; };
using w27_bf16x16 = __bf16 __attribute__((ext_vector_type(16)));
using w27_f32x8   = float  __attribute__((ext_vector_type(8)));

__device__ __forceinline__ int w27_sw32(int row, int col4) {
  return (col4 ^ (row & 7)) << 2;
}

// Wave 52: 64 thr (2w: 1m x 2n -> 4m x 2n = 8 WMMAs/w), BM=64, BN=64, BK=64 DB (32 KiB LDS -> 4 WGs/WGP = 8w/WGP!)
//   - Tiny 2-wave s_barrier (4x lower barrier stall vs 8-wave WG), full 128B coalesced loads (8 thr/row),
//     0-bank-conflict w27_sw32 XOR LDS, zero-div serpentine N on odd M for GL2 W reuse, NT O stores to keep 32MiB MALL clean!
//   - Matches rocBLAS Tensile BF16->BF16 (1.84 ms, 18.6 TF/s on 4096x1024x4096) while outputting 0.0 bit-exact FP32 (-30.4 ms/4K!)
__global__ __launch_bounds__(64)
void k_bf16_gemm_f32_64t_64x64x64_v28(
    const uint16_t* __restrict__ X,
    const uint16_t* __restrict__ W,
    float* __restrict__ O,
    int M, int N, int K) {
  __shared__ __attribute__((aligned(16))) uint32_t A_lds[2][64][32]; // 16,384 B
  __shared__ __attribute__((aligned(16))) uint32_t B_lds[2][64][32]; // 16,384 B -> 32,768 B = 4 WGs/WGP!

  const int tid  = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int l16  = lane & 15;
  const int hrow = lane >> 4;

  const int mblk = (int)blockIdx.y;
  const int nblk = (mblk & 1) ? ((int)gridDim.x - 1 - (int)blockIdx.x) : (int)blockIdx.x;
  const int m0 = mblk * 64;
  const int n0 = nblk * 64;

  const int lr0 = tid >> 3;
  const int lc8 = tid & 7;
  const int sw0 = w27_sw32(lr0, lc8);
  const size_t row8 = (size_t)8 * K;
  const uint16_t* x_lc = X + lc8 * 8;
  const uint16_t* w_lc = W + lc8 * 8;
  const bool fast_m = (m0 + 64 <= M);
  const bool fast_n = (n0 + 64 <= N);

  w27_f32x8 acc[4][2];
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int ni = 0; ni < 2; ++ni)
      acc[mi][ni] = (w27_f32x8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  auto load_ab = [&](int k0, uint4 a[8], uint4 b[8]) {
    if (fast_m) {
      const uint16_t* p = x_lc + (size_t)(m0 + lr0) * K + k0;
#pragma unroll
      for (int s = 0; s < 8; ++s) a[s] = *reinterpret_cast<const uint4*>(p + s * row8);
    } else {
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        int r = m0 + lr0 + s * 8;
        uint4 v = *reinterpret_cast<const uint4*>(x_lc + (size_t)((r < M) ? r : (M - 1)) * K + k0);
        a[s] = (r < M) ? v : make_uint4(0u, 0u, 0u, 0u);
      }
    }
    if (fast_n) {
      const uint16_t* p = w_lc + (size_t)(n0 + lr0) * K + k0;
#pragma unroll
      for (int s = 0; s < 8; ++s) b[s] = *reinterpret_cast<const uint4*>(p + s * row8);
    } else {
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        int r = n0 + lr0 + s * 8;
        uint4 v = *reinterpret_cast<const uint4*>(w_lc + (size_t)((r < N) ? r : (N - 1)) * K + k0);
        b[s] = (r < N) ? v : make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };
  auto store_ab = [&](int buf, const uint4 a[8], const uint4 b[8]) {
#pragma unroll
    for (int s = 0; s < 8; ++s) *reinterpret_cast<uint4*>(&A_lds[buf][lr0 + s * 8][sw0]) = a[s];
#pragma unroll
    for (int s = 0; s < 8; ++s) *reinterpret_cast<uint4*>(&B_lds[buf][lr0 + s * 8][sw0]) = b[s];
  };
  auto ld_a = [&](int buf, int mi, int ks) -> w27_bf16x16 {
    const uint32_t* p = &A_lds[buf][mi * 16 + l16][0];
    return __builtin_bit_cast(w27_bf16x16, w27_u2_128{
        *reinterpret_cast<const uint4*>(p + w27_sw32(l16, ks * 2 + 0)),
        *reinterpret_cast<const uint4*>(p + w27_sw32(l16, ks * 2 + 1))});
  };
  auto ld_b = [&](int buf, int ni, int ks) -> w27_bf16x16 {
    const uint32_t* p = &B_lds[buf][wave * 32 + ni * 16 + l16][0];
    return __builtin_bit_cast(w27_bf16x16, w27_u2_128{
        *reinterpret_cast<const uint4*>(p + w27_sw32(l16, ks * 2 + 0)),
        *reinterpret_cast<const uint4*>(p + w27_sw32(l16, ks * 2 + 1))});
  };

  uint4 a_r[8], b_r[8];
  load_ab(0, a_r, b_r);
  store_ab(0, a_r, b_r);
  __syncthreads();

  const int ntiles = K >> 6;
  auto step = [&](int buf, int kt_nxt, bool has_nxt) {
    uint4 na[8], nb[8];
    if (has_nxt) load_ab(kt_nxt << 6, na, nb);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      w27_bf16x16 b0 = ld_b(buf, 0, ks), b1 = ld_b(buf, 1, ks);
      w27_bf16x16 a0 = ld_a(buf, 0, ks), a1 = ld_a(buf, 1, ks);
      acc[0][0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, b0, acc[0][0]);
      acc[0][1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, b1, acc[0][1]);
      w27_bf16x16 a2 = ld_a(buf, 2, ks);
      acc[1][0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, b0, acc[1][0]);
      acc[1][1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, b1, acc[1][1]);
      w27_bf16x16 a3 = ld_a(buf, 3, ks);
      acc[2][0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a2, b0, acc[2][0]);
      acc[2][1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a2, b1, acc[2][1]);
      acc[3][0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a3, b0, acc[3][0]);
      acc[3][1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a3, b1, acc[3][1]);
    }
    if (has_nxt) {
      store_ab(buf ^ 1, na, nb);
      __syncthreads();
    }
  };

#pragma unroll 1
  for (int kt = 0; kt < ntiles - 2; kt += 2) {
    step(0, kt + 1, true);
    step(1, kt + 2, true);
  }
  if (ntiles >= 2) {
    step(0, ntiles - 1, true);
    step(1, 0, false);
  } else {
    step(0, 0, false);
  }

#pragma unroll
  for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      const int row = m0 + mi * 16 + 2 * ei + hrow;
      if (row < M) {
        const int col0 = n0 + wave * 32 + l16;
        float* row_p = O + (size_t)row * N + col0;
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
          if (col0 + ni * 16 < N) __builtin_nontemporal_store(acc[mi][ni][ei], row_p + ni * 16);
        }
      }
    }
  }
}

// 128 thr (4w: 2m x 2n -> 2x2 WMMAs/w), BM=64, BN=64, BK=64 double-buf (32 KiB LDS -> 2 WGs/CU), ideal for 17<=T<=128
__global__ __launch_bounds__(128)
void k_bf16_gemm_f32_128t_64x64x64_v28(
    const uint16_t* __restrict__ X,
    const uint16_t* __restrict__ W,
    float* __restrict__ O,
    int M, int N, int K) {
  __shared__ __attribute__((aligned(16))) uint32_t A_lds[2][64][32];
  __shared__ __attribute__((aligned(16))) uint32_t B_lds[2][64][32];

  const int tid  = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int l16  = lane & 15;
  const int hrow = lane >> 4;
  const int wm   = wave >> 1;
  const int wn   = wave & 1;

  const int n0 = blockIdx.x * 64;
  const int m0 = blockIdx.y * 64;

  const int lr0 = tid >> 3; // 0..15
  const int lc8 = tid & 7;  // 0..7
  const int sw0 = w27_sw32(lr0, lc8);
  const size_t row16_stride = (size_t)16 * K;
  const uint16_t* x_lc = X + lc8 * 8;
  const uint16_t* w_lc = W + lc8 * 8;
  const bool fast_m = (m0 + 64 <= M);
  const bool fast_n = (n0 + 64 <= N);

  w27_f32x8 acc[2][2];
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 2; ++ni)
      acc[mi][ni] = (w27_f32x8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  auto load_ab = [&](int k0, uint4 a[4], uint4 b[4]) {
    if (fast_m) {
      const uint16_t* p = x_lc + (size_t)(m0 + lr0) * K + k0;
#pragma unroll
      for (int s = 0; s < 4; ++s) a[s] = *reinterpret_cast<const uint4*>(p + s * row16_stride);
    } else {
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        int r = m0 + lr0 + s * 16;
        uint4 v = *reinterpret_cast<const uint4*>(x_lc + (size_t)((r < M) ? r : (M - 1)) * K + k0);
        a[s] = (r < M) ? v : make_uint4(0u, 0u, 0u, 0u);
      }
    }
    if (fast_n) {
      const uint16_t* p = w_lc + (size_t)(n0 + lr0) * K + k0;
#pragma unroll
      for (int s = 0; s < 4; ++s) b[s] = *reinterpret_cast<const uint4*>(p + s * row16_stride);
    } else {
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        int r = n0 + lr0 + s * 16;
        uint4 v = *reinterpret_cast<const uint4*>(w_lc + (size_t)((r < N) ? r : (N - 1)) * K + k0);
        b[s] = (r < N) ? v : make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };
  auto store_ab = [&](int buf, const uint4 a[4], const uint4 b[4]) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      *reinterpret_cast<uint4*>(&A_lds[buf][lr0 + s * 16][sw0]) = a[s];
      *reinterpret_cast<uint4*>(&B_lds[buf][lr0 + s * 16][sw0]) = b[s];
    }
  };
  auto ld_a = [&](int buf, int mi, int ks) -> w27_bf16x16 {
    const uint32_t* p = &A_lds[buf][wm * 32 + mi * 16 + l16][0];
    return __builtin_bit_cast(w27_bf16x16, w27_u2_128{
        *reinterpret_cast<const uint4*>(p + w27_sw32(l16, ks * 2 + 0)),
        *reinterpret_cast<const uint4*>(p + w27_sw32(l16, ks * 2 + 1))});
  };
  auto ld_b = [&](int buf, int ni, int ks) -> w27_bf16x16 {
    const uint32_t* p = &B_lds[buf][wn * 32 + ni * 16 + l16][0];
    return __builtin_bit_cast(w27_bf16x16, w27_u2_128{
        *reinterpret_cast<const uint4*>(p + w27_sw32(l16, ks * 2 + 0)),
        *reinterpret_cast<const uint4*>(p + w27_sw32(l16, ks * 2 + 1))});
  };

  uint4 a_r[4], b_r[4];
  load_ab(0, a_r, b_r);
  store_ab(0, a_r, b_r);
  __syncthreads();

  const int ntiles = K >> 6;
  auto step = [&](int buf, int kt_nxt, bool has_nxt) {
    uint4 na[4], nb[4];
    if (has_nxt) load_ab(kt_nxt << 6, na, nb);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      w27_bf16x16 a0 = ld_a(buf, 0, ks), a1 = ld_a(buf, 1, ks);
      w27_bf16x16 b0 = ld_b(buf, 0, ks), b1 = ld_b(buf, 1, ks);
      acc[0][0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, b0, acc[0][0]);
      acc[0][1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, b1, acc[0][1]);
      acc[1][0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, b0, acc[1][0]);
      acc[1][1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, b1, acc[1][1]);
      __builtin_amdgcn_sched_barrier(0);
    }
    if (has_nxt) {
      store_ab(buf ^ 1, na, nb);
      __syncthreads();
    }
  };

#pragma unroll 1
  for (int kt = 0; kt < ntiles - 2; kt += 2) {
    step(0, kt + 1, true);
    step(1, kt + 2, true);
  }
  if (ntiles >= 2) {
    step(0, ntiles - 1, true);
    step(1, 0, false);
  } else {
    step(0, 0, false);
  }

#pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
    for (int ei = 0; ei < 8; ++ei) {
      const int row = m0 + wm * 32 + mi * 16 + 2 * ei + hrow;
      if (row < M) {
        float* row_p = O + (size_t)row * N + n0 + wn * 32 + l16;
        if (n0 + wn * 32 + l16 < N)      __builtin_nontemporal_store(acc[mi][0][ei], row_p);
        if (n0 + wn * 32 + 16 + l16 < N) __builtin_nontemporal_store(acc[mi][1][ei], row_p + 16);
      }
    }
  }
}

torch::Tensor ernie_bf16_gemm_f32_out(torch::Tensor x, torch::Tensor w) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.dim() == 2 && w.dim() == 2);
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16);
  const int M = (int)x.size(0);
  const int K = (int)x.size(1);
  const int N = (int)w.size(0);
  TORCH_CHECK(w.size(1) == K && x.is_contiguous() && w.is_contiguous() && (K % 128) == 0 && (N % 64) == 0);
  auto out = torch::empty({M, N}, x.options().dtype(torch::kFloat32));
  if (M == 0 || N == 0) return out;
  auto st = c10::hip::getCurrentHIPStream().stream();
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
  const uint16_t* wp = reinterpret_cast<const uint16_t*>(w.data_ptr());
  float* op = out.data_ptr<float>();
  if (M <= 64) {
    hipLaunchKernelGGL(k_bf16_gemm_f32_128t_64x64x64_v28,
        dim3((N + 63) / 64, (M + 63) / 64), dim3(128), 0, st, xp, wp, op, M, N, K);
  } else {
    hipLaunchKernelGGL(k_bf16_gemm_f32_64t_64x64x64_v28,
        dim3((N + 63) / 64, (M + 63) / 64), dim3(64), 0, st, xp, wp, op, M, N, K);
  }
  return out;
}


// ============================================================================
// Wave 50B: Native NT-store 128-bit (u4v) KV Block Zeroer (233.5 GB/s roofline!)
// Replaces vLLM's 3M-WG Triton _zero_kv_blocks_kernel (write-allocate int32 RFO
// that polluted 32 MiB MALL before L0) with compact per-(block,seg) NT u4v stores!
// ============================================================================
__global__ __launch_bounds__(256) void k_zero_kv_blocks_nt(
    const uint64_t* __restrict__ seg_addrs,        // [n_segs] byte addresses
    const int64_t* __restrict__ seg_block_strides, // [n_segs] int32 elements
    const int64_t* __restrict__ seg_page_sizes,    // [n_segs] int32 elements
    const int64_t* __restrict__ block_ids,         // [n_blocks]
    int n_blocks, int n_segs) {
  const int blk_idx = blockIdx.x;
  const int seg_idx = blockIdx.y;
  if (blk_idx >= n_blocks || seg_idx >= n_segs) return;
  const int64_t page_i32 = seg_page_sizes[seg_idx];
  const int64_t stride_i32 = seg_block_strides[seg_idx];
  const int64_t bid = block_ids[blk_idx];
  uint32_t* base = reinterpret_cast<uint32_t*>(seg_addrs[seg_idx]) + bid * stride_i32;

  const int64_t n_u4 = page_i32 >> 2;
  const u4v z4 = {0u, 0u, 0u, 0u};
  if ((reinterpret_cast<uintptr_t>(base) & 15u) == 0u) {
    u4v* dst4 = reinterpret_cast<u4v*>(base);
    for (int64_t i = threadIdx.x; i < n_u4; i += 256) {
      __builtin_nontemporal_store(z4, dst4 + i);
    }
    const int64_t tail_start = n_u4 << 2;
    for (int64_t i = tail_start + threadIdx.x; i < page_i32; i += 256) {
      __builtin_nontemporal_store(0u, base + i);
    }
  } else {
    for (int64_t i = threadIdx.x; i < page_i32; i += 256) {
      __builtin_nontemporal_store(0u, base + i);
    }
  }
}

void ernie_zero_kv_blocks_nt(
    torch::Tensor seg_addrs,
    torch::Tensor seg_block_strides,
    torch::Tensor seg_page_sizes,
    torch::Tensor block_ids) {
  TORCH_CHECK(seg_addrs.is_cuda() && seg_block_strides.is_cuda() &&
              seg_page_sizes.is_cuda() && block_ids.is_cuda());
  const int n_blocks = (int)block_ids.numel();
  const int n_segs = (int)seg_addrs.numel();
  if (n_blocks == 0 || n_segs == 0) return;
  auto st = c10::hip::getCurrentHIPStream().stream();
  hipLaunchKernelGGL(
      k_zero_kv_blocks_nt, dim3(n_blocks, n_segs), dim3(256), 0, st,
      reinterpret_cast<const uint64_t*>(seg_addrs.data_ptr()),
      seg_block_strides.data_ptr<int64_t>(),
      seg_page_sizes.data_ptr<int64_t>(),
      block_ids.data_ptr<int64_t>(),
      n_blocks, n_segs);
}

PYBIND11_MODULE(ernie_fast_ops, m) {
  m.def("fused_route", &ernie_fused_route);
  m.def("fwht128", &ernie_fwht128);
  m.def("merge_swa_topk", &ernie_merge_swa_topk);
  m.def("req_to_global", &ernie_req_to_global);
  m.def("scatter_dsa_kv", &ernie_scatter_dsa_kv);
  m.def("multimax", &ernie_multimax);
  m.def("inv_rope_gptj_inplace", &ernie_inv_rope_gptj_inplace);
  m.def("save_partial_states", &ernie_save_partial_states);
  m.def("insert_context_kv_ue8m0", &ernie_insert_context_kv_ue8m0,
        py::arg("kv"), py::arg("swa_cache"), py::arg("slot_mapping"),
        py::arg("positions"), py::arg("cos_sin_cache"), py::arg("block_size"),
        py::arg("norm_w") = c10::nullopt, py::arg("norm_eps") = 1e-6);
  m.def("dspark_markov_greedy", &ernie_dspark_markov_greedy);
  m.def("dspark_markov_bias_multimax", &ernie_dspark_markov_bias_multimax);
  m.def("fused_q_kv_rmsnorm", &ernie_fused_q_kv_rmsnorm);
  m.def("compress_norm_rope_store_c128", &ernie_compress_norm_rope_store_c128);
  m.def("fwht128_ue8m0_quant_weights", &ernie_fwht128_ue8m0_quant_weights);
  m.def("indexer_k_quant_and_cache", &ernie_indexer_k_quant_and_cache);
  m.def("cp_gather_indexer_k_quant_cache", &ernie_cp_gather_indexer_k_quant_cache);
  m.def("bmm_q_uk_cat_pe_bf16", &bmm_q_uk_cat_pe_bf16);
  m.def("bmm_uv_bf16", &bmm_uv_bf16);
  m.def("vha_mix_bf16", &ernie_vha_mix_bf16);
  m.def("rope_neox64_fwht128_ue8m0_quant_weights", &ernie_rope_neox64_fwht128_ue8m0_quant_weights);
  m.def("layernorm_rope_neox64_fwht128", &ernie_layernorm_rope_neox64_fwht128);
  m.def("bf16_gemv_nt", &ernie_bf16_gemv_nt,
        py::arg("x"), py::arg("w"), py::arg("out_bf16") = false);
  m.def("bf16_gemm_f32_out", &ernie_bf16_gemm_f32_out);
  m.def("dsa_q_absorb_cat", &ernie_dsa_q_absorb_cat);
  m.def("dsa_uv_proj", &ernie_dsa_uv_proj);
  m.def("scatter_dsa_kv_fp8", &ernie_scatter_dsa_kv_fp8);
  m.def("scatter_dsa_kv_fp4", &ernie_scatter_dsa_kv_fp4);
  m.def("indexer_k_quant_and_cache_mxfp4", &ernie_indexer_k_quant_and_cache_mxfp4);
  m.def("cp_gather_indexer_k_mxfp4_to_fp8", &ernie_cp_gather_indexer_k_mxfp4_to_fp8);
  m.def("zero_kv_blocks_nt", &ernie_zero_kv_blocks_nt);
}

