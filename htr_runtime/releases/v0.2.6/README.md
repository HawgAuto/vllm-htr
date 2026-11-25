# vLLM HTR — v0.2.6 Release Notes

- **Image:** `ghcr.io/hawgauto/vllm-htr@sha256:dc11916549c534b207bb45048fe2f9b2177117995931f5aff5b384b274ea0fe8`
- **Alias:** `ghcr.io/hawgauto/vllm-htr:v0.2.6`
- **Base:** `ghcr.io/hawgauto/vllm-htr:v0.2.5`
- **Executable source:** [`00f0b8c0caf7d85fee9232ecf600e0317408b1ab`](https://github.com/HawgAuto/vllm-htr/tree/00f0b8c0caf7d85fee9232ecf600e0317408b1ab)
- **Zero-overlay verification:** `htr_native_policy.verify_installation()` verified at build time (`24` Python files + `23` native binaries) + full zero-overlay E2E gate (`0` mid-inference Triton JIT warnings, `0` corrupted streams, `6/6` `HTR_MOE_SELFCHECK` passing).

## Summary (Plain-English Highlights)

v0.2.6 finishes moving the serving hot path off Triton JIT and onto hand-tuned ahead-of-time (AOT) HIP kernels built specifically for AMD Strix Halo (`gfx1151`), while adding RDNA 3.5 ISA upgrades to the decode stack:

- **2× faster long-context prefill:** 13K-token warm prompts finish in **11.44 s** instead of **23.58 s** (**1,137.7 tok/s** vs `551.9 tok/s`), and cold 13K prompts finish in **14.30 s** (**910.5 tok/s**).
- **Zero Triton compilation stalls mid-serving:** Every hot-path prefill and decode operator now loads prebuilt `.hsaco`/`.so` binaries (`0` mid-inference Triton JIT warnings).
- **Faster multi-request serving:** `c=8` concurrency finishes **1.23× faster** (`19.03 s → 15.51 s`, `66.01 tok/s` wall aggregate) and `c=4 × 4K` long-context concurrency finishes **1.09× faster** (`23.82 s → 21.77 s`, `+7.7%` sum decode).
- **Drop-in container update:** Model weights, tokenizer, configs, and sampling defaults are completely unchanged.

## Verified Zero-Overlay E2E Results (AMD Strix Halo `gfx1151`, `int8_per_token_head`, MTP-3)

| Workload | `v0.2.5` KV8 | **`v0.2.6` Zero-Overlay Gate** | Speedup |
|---|---:|---:|---:|
| **13K warm prefill (`13,018 tok`)** | `23.58 s` (`551.9 tok/s`) | **`11.44 s` (`1,137.7 tok/s`)** | **2.06×** |
| **13K cold prefill (`13,017 tok`)** | `23.58 s` (`551.9 tok/s`) | **`14.30 s` (`910.5 tok/s`)** | **1.65×** |
| **Greedy `c=1` (`128 tok`)** | `45.05 tok/s` | **`45.16 tok/s`** (`236.1 ms` TTFT) | **+0.2%** |
| **`c=4 × 4K` long-ctx (`wall` / `sum_dec`)** | `23.82 s` / `57.25 tok/s` | **`21.77 s` / `61.65 tok/s`** | **1.09× / +7.7%** |
| **`c=8` high-conc (`wall` / `wall_agg`)** | `19.03 s` / `53.80 tok/s` | **`15.51 s` / `66.01 tok/s`** | **1.23×** |
| **MTP-3 spec acceptance** | — | **`63.20%`** (`2.896 tok/step`) | — |
| **Mid-inference Triton JIT warnings** | multiple | **`0`** | **eliminated** |

## INT8 WMMA QSA Prefill (`qsa_tau_v5_kv8_i8`) Parity & Direct A/B

In direct apples-to-apples E2E comparison on the same v0.2.6 stack (`HTR_QSA_TAU_V5_I8=1` default vs `HTR_QSA_TAU_V5_I8=0` BF16 WMMA):

- **13K warm prefill (`13,018 tok`):** `12,336.7 ms` (`1,055.2 tok/s`) → **`11,471.5 ms` (`1,134.8 tok/s`)** (**−865.1 ms / +7.5%**)
- **`c=4 × 4K`:** `25.10 s` (`12,322 ms` avg TTFT) → **`23.27 s` (`11,034 ms` avg TTFT)** (**−1.83 s / −1,288 ms**)
- **`c=8`:** `17.80 s` → **`16.35 s`** (**−1.45 s**)
- **Accuracy & stability:** Kernel cosine similarity **`0.999994`** on normal activations (**`0.99986`** with 40× outliers, **`0.99926`** with 200× attention sinks, `0 NaNs`); **200/200** character exact match on both `greedy_c1` and `ctx12k_c1_warm`; MTP-3 spec acceptance `62.34% → 63.45%` (`2.870 → 2.904 tok/step`).
- **Control:** Enabled by default; pass `-e HTR_QSA_TAU_V5_I8=0` to use BF16 WMMA QSA prefill (`qsa_tau_v4_kv8`) instead.

## Technical Breakdown

1. **Zero-Triton-JIT Hot Path (`20` new zero-spill AOT `gfx1151` Wave32 HIP binaries)**
   - **Full-INT8 WMMA QSA Prefill (`qsa_tau_v5_kv8_i8.hsaco`)**: Uses `v_wmma_i32_16x16x16_iu8` (`225 VGPRs, 0 spills, 16,328 B LDS`), **1.66–1.70× faster** kernel time than `v4_kv8` BF16 WMMA.
   - **AOT GDN & Causal Conv1D Prefill (`causal_conv1d_fwd_aot.hsaco`, `gdn_chunk_h/o/wy/post_conv/norm_gate_aot.hsaco`)**: 128-bit `u32x4` vectorized + single-cycle `v_perm_b32` packed BF16×2 RNE, bit-exact `conv_states`, full APC & strided-1D `cache_indices` (`stride_cache_indices`) fix.
   - **AOT W4A16 MoE (`moe_w4a16_wmma_aot.hsaco`)**: Double-buffered LDS (`s_B[2][64*40]`), coalesced 4-lane cached `B` loads, wave-level padding skip, `int64` `expert_ids` support — **bit-exact (`0.000000`) and 1.04–1.26× faster than warm Triton**.
   - **AOT HyperConnection v2 (`hc_ops_aot.hsaco`)** (`229 GB/s` = 94% LPDDR5X peak), **AOT QSA MQA Paged v3 (`qsa_mqa_paged_aot.hsaco`)** with hidden `v_dual_dot2acc_f32_bf16`, **HPF5 v2 MoE prefill (`hpf5_v2_moe_prefill.so`)**, **AOT dense prefill v2 (`htr_dense_prefill_v2.hsaco`)**, **AOT QSA cache/prepare/indexer ops**, **AOT PLE transpose (`direct_tr12.hsaco`)**, and **AOT Gumbel sampler (`gumbel_sample_aot.hsaco`)**.
2. **Wave 7 RDNA 3.5 (`gfx1151`) ISA Decode Upgrades (`3` upgraded `.so` extensions)**
   - **`htr_bf16_gemv.so`**: Unlocked hidden `v_dual_dot2acc_f32_bf16` (VOPD Opcode 13, `0 → 96`), native `v_dot2_f32_bf16` (`0 → 112`), 128-bit vectorized `gate_scale` + `v_perm_b32` (`0 → 65`); **1.23–1.58× faster**, `0 spills`, `16 w/SIMD`, verified against FP32 oracle across 20 even/odd `M` cases (`<= 1 BF16 ULP`).
   - **`htr_moe_k6_shx.so`** (`v_dual_dot2acc_f32_f16`: `1,524 → 1,696`) and **`htr_kernels.so`** (`6,590 → 7,544`) with half-tile `xv[M][4]` + bank-staggered `(0,18),(2,16),(4,22),(6,20) / (8,26),(10,24),(12,30),(14,28)` VOPD pairing.
