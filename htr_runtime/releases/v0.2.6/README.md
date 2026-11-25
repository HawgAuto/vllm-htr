# vLLM HTR — v0.2.6 Release Notes

- **Image:** `ghcr.io/hawgauto/vllm-htr@sha256:dc11916549c534b207bb45048fe2f9b2177117995931f5aff5b384b274ea0fe8`
- **Alias:** `ghcr.io/hawgauto/vllm-htr:v0.2.6`
- **Base:** `ghcr.io/hawgauto/vllm-htr:v0.2.5`
- **Zero-overlay self-check:** `htr_native_policy.verify_installation()` verified at build time + full zero-overlay E2E gate (`0` mid-inference Triton JIT warnings, `0` corrupted streams, `6/6` `HTR_MOE_SELFCHECK` passing).

## What's New in v0.2.6

1. **Zero-Triton-JIT Hot Path (`20` new zero-spill AOT `gfx1151` Wave32 HIP binaries)**
   - **Full-INT8 WMMA QSA Prefill (`qsa_tau_v5_kv8_i8.hsaco`)**: Uses `v_wmma_i32_16x16x16_iu8` (`225 VGPRs, 0 spills, 16,328 B LDS`), **1.66–1.70× faster** than `v4_kv8` BF16 WMMA (**−865 ms** on 13K warm prefill, `cos=0.999994`, toggleable via `HTR_QSA_TAU_V5_I8=0`).
   - **AOT GDN & Causal Conv1D Prefill (`causal_conv1d_fwd_aot.hsaco`, `gdn_chunk_h/o/wy/post_conv/norm_gate_aot.hsaco`)**: 128-bit `u32x4` vectorized + single-cycle `v_perm_b32` packed BF16×2 RNE, bit-exact `conv_states`, full APC & strided-1D `cache_indices` parity.
   - **AOT W4A16 MoE (`moe_w4a16_wmma_aot.hsaco`)**: Double-buffered LDS (`s_B[2][64*40]`), coalesced 4-lane cached `B` loads, wave-level padding skip, `int64` `expert_ids` support — **bit-exact (`0.000000`) and 1.04–1.26× faster than warm Triton**.
   - **AOT HyperConnection v2 (`hc_ops_aot.hsaco`)** (`229 GB/s` = 94% LPDDR5X peak), **AOT QSA MQA Paged v3 (`qsa_mqa_paged_aot.hsaco`)** with hidden `v_dual_dot2acc_f32_bf16`, **HPF5 v2 MoE prefill (`hpf5_v2_moe_prefill.so`)**, **AOT dense prefill v2 (`htr_dense_prefill_v2.hsaco`)**, **AOT QSA cache/prepare/indexer ops**, **AOT PLE transpose (`direct_tr12.hsaco`)**, and **AOT Gumbel sampler (`gumbel_sample_aot.hsaco`)**.
2. **Wave 7 RDNA 3.5 (`gfx1151`) ISA Decode Upgrades (`3` upgraded `.so` extensions)**
   - **`htr_bf16_gemv.so`**: Unlocked hidden `v_dual_dot2acc_f32_bf16` (VOPD Opcode 13, `0 → 96`), native `v_dot2_f32_bf16` (`0 → 112`), 128-bit vectorized `gate_scale` + `v_perm_b32` (`0 → 65`); **1.23–1.58× faster**, `0 spills`, `16 w/SIMD`.
   - **`htr_moe_k6_shx.so`** (`v_dual_dot2acc_f32_f16`: `1,524 → 1,696`) and **`htr_kernels.so`** (`6,590 → 7,544`) with half-tile `xv[M][4]` + bank-staggered `(0,18),(2,16),(4,22),(6,20) / (8,26),(10,24),(12,30),(14,28)` VOPD pairing.

## Verified Zero-Overlay E2E Results (AMD Strix Halo `gfx1151`, `int8_per_token_head`, MTP-3)

| Workload | `v0.2.5` KV8 | **`v0.2.6` Zero-Overlay Gate** | Speedup |
|---|---:|---:|---:|
| **13K warm prefill (`13,018 tok`)** | `23.58s` (`551.9 tok/s`) | **`11.44s` (`1,137.7 tok/s`)** | **2.06×** |
| **13K cold prefill (`13,017 tok`)** | `23.58s` (`551.9 tok/s`) | **`14.30s` (`910.5 tok/s`)** | **1.65×** |
| **Greedy `c=1` (`128 tok`)** | `45.05 tok/s` | **`45.16 tok/s`** | **+0.2%** |
| **`c=4 × 4K` long-ctx (`wall` / `sum_dec`)** | `23.82s` / `57.25 tok/s` | **`21.77s` / `61.65 tok/s`** | **1.09× / +7.7%** |
| **`c=8` high-conc (`wall` / `wall_agg`)** | `19.03s` / `53.80 tok/s` | **`15.51s` / `66.01 tok/s`** | **1.23×** |
| **Mid-inference Triton JIT warnings** | multiple | **`0`** | eliminated |
