# vLLM HTR — v0.2.7 Release Notes

- **Image (staged locally, not yet pushed):** `ghcr.io/hawgauto/vllm-htr:v0.2.7` (`vllm-htr:v0.2.7`)
- **Base:** `ghcr.io/hawgauto/vllm-htr:v0.2.6`
- **Zero-overlay verification:** Build-time `verify_ernie55_cpu.py` (`8` ERNIE `.so`, `463` Wave32 kernels, `0` spills, `0` divOps, Qwen isolation `_serving_ernie5=False`) + `htr_native_policy.verify_installation()` (`17` upgraded Python files + `12` upgraded Qwen3.8 native binaries) + all 4 live zero-overlay hardware gates on AMD Strix Halo (`gfx1151`).

## Summary (Plain-English Highlights)

v0.2.7 is a unified release for AMD Strix Halo (`gfx1151`) combining **major Qwen3.8 decode + cold-boot speedups** with **full ERNIE-5.5-Flash-Preview-HTR (+ 3L K=2 DSpark speculative decoding)** support in a single zero-overlay container:

1. **2.32× faster Qwen3.8 empty-workspace cold boot (`388.2 s → 167.3 s`, −221 s):**
   - Pre-seeded `module_aiter_core.so` and `libtorch_c_dlpack_addon_torch213-rocm.so` in `/opt/vllm-htr/native-bin/` with shared `/workspace/cache/{aiter-jit,tvm-ffi}` across all serving profiles (eliminates ~257 s cold / ~140 s config-change `aiter` + ~35 s `tvm_ffi` C++ builds).
   - Upgraded `htr_fast_arena_v27.so` with `POSIX_FADV_WILLNEED` lookahead + `POSIX_FADV_DONTNEED` eviction (`HTR_MOE_LOAD_WORKERS=4`, full 62 GB SHA-256 verification on first boot → fast C++ arena loader on warm boots: `92.6 s` warm BF16 boot vs `114.7 s`, `204.3 s` KV8 config-change boot vs `309.8 s`).
2. **Faster Qwen3.8 decode across BF16 and INT8 KV (`int8_per_token_head`):**
   - Wired `moe_w4a16_decode_cn_n128` (`moe_w4a16_decode_v27.hsaco`) into `fused_moe.py` for production `contig_N` W4A16 Quark MTP draft MoE (`PH1` + `PH2`), alongside `htr_decode_meta_v27.so`, `htr_moe_k6_shx.so` (`v27_ph1`), `hiphc_v2.so`, `htr_mtp_ddn_v2.so`, `htr_spec_sample.so`, `htr_ple_short_conv.so`, `qsa_canon_expand_fused.hsaco`, and `qsa_decode_union_kv8.hsaco`.
   - **BF16 KV (`seqs=4, mbt=2048`):** `c=1` avg `+3.6%` (`60.06 tok/s`), `5.8K` `+4.9%` (`62.64 tok/s`), `c=4` stoch `+10.8%` agg / `+12.6%` sum (`69.96 / 107.49 tok/s`), `c=4` greedy `+8.3%` agg / `+5.7%` sum (`87.91 / 112.40 tok/s`).
   - **INT8 KV (`seqs=16, mbt=9600`):** greedy `c=1` `+7.3%` (`48.45 tok/s`), 13K cold prefill `1.18×` (`12.08 s / 1,077.8 tok/s`) + decode `+10.3%` (`60.11 tok/s`), 13K warm decode `+21.5%` (`54.15 tok/s`), faster `c=4×4K` and `c=8`, **`0`** mid-inference Triton JIT warnings.
3. **Unified ERNIE-5.5-Flash-Preview-HTR + 3L K=2 DSpark Serving (Clean 128 GB Strix Halo Boot):**
   - Bundles all 8 zero-spill `gfx1151` Wave32 `.so` binaries (`463` kernels) and the `vllm_ernie_htr` plugin (`Ernie5_5_MoeForCausalLM` + `Ernie5_5_MoeDSparkDraftModel` + `Ernie5_5Renderer`).
   - Automatic model-aware plugin gating in `htr_native_policy.py`: non-Qwen models automatically skip Qwen3.8-only plugins (`q38spec`, `htr_moe`, `gp8hc`, `htr_q38f`, `htr_autotune_pin`) and Qwen `server-argv.json` defaults.
   - **Three critical 128 GB Strix Halo OOM fixes for ERNIE-5.5:**
     - `rocm_aiter_mla_sparse.py`: fixed `gfx1151` `profile_run` decode-logits workspace from **`8.00 GiB → 128 MiB`** at 32K context (`−7.875 GiB`).
     - `weight_utils.py`: per-shard `POSIX_FADV_DONTNEED` for checkpoints `> 32 GiB` eliminates 108 GB Linux page-cache accumulation inside Docker containers.
     - `deepseek_scaling_rope.py` & `vllm_ernie_htr/attention.py`: dynamically caps `cos_sin_cache` materialization to `min(max_position_embeddings, max_model_len + 256)` while keeping YaRN `inv_freq` bit-identical (`−0.97 GiB` permanent VRAM).
   - Fixed `Ernie5_5Renderer` async chat rendering compatibility (`_render_with_timeout`) so both `/v1/completions` and `/v1/chat/completions` work cleanly.

## Verified Zero-Overlay Hardware Gates (`gfx1151`, AMD Strix Halo)

### Qwen3.8-Flash-Next-HTR — Boot & BF16 E2E (`seqs=4, mbt=2048`)

| Metric | `v0.2.6` | **`v0.2.7` Zero-Overlay** | Delta |
|---|---:|---:|---:|
| **Cold empty-workspace boot** (incl. 62 GB SHA-256) | `388.2 s` | **`167.3 s`** | **2.32× faster (−221 s)** |
| **Warm BF16 boot** | `114.7 s` | **`92.6 s`** | **1.24× faster (−22.1 s)** |
| **`c=1` greedy** | `44.62 tok/s` | **`45.25 tok/s`** | **+1.4%** |
| **`c=1` average** | `57.95 tok/s` | **`60.06 tok/s`** | **+3.6%** |
| **`5.8K` context decode** | `59.73 tok/s` | **`62.64 tok/s`** | **+4.9%** |
| **`c=4` stochastic (`agg` / `sum`)** | `63.15 / 95.48 tok/s` | **`69.96 / 107.49 tok/s`** | **+10.8% / +12.6%** |
| **`c=4` greedy (`agg` / `sum`)** | `81.20 / 106.32 tok/s` | **`87.91 / 112.40 tok/s`** | **+8.3% / +5.7%** |

### Qwen3.8-Flash-Next-HTR — INT8 KV (`int8_per_token_head`, `seqs=16, mbt=9600`)

| Metric | `v0.2.6` KV8 | **`v0.2.7` KV8 Zero-Overlay** | Delta |
|---|---:|---:|---:|
| **Config-change boot** | `309.8 s` | **`204.3 s`** | **1.52× faster (−105.5 s)** |
| **Greedy `c=1`** | `45.16 tok/s` | **`48.45 tok/s`** | **+7.3%** |
| **13K cold prefill / decode** | `14.30 s (910.5)` / `54.50` | **`12.08 s (1,077.8)` / `60.11`** | **1.18× / +10.3%** |
| **13K warm prefill / decode** | `11.44 s (1,137.7)` / `44.55` | **`11.69 s (1,113.6)` / `54.15`** | **−2.1% / +21.5%** |
| **`c=4 × 4K` (`wall` / `sum_dec`)** | `21.77 s` / `61.65 tok/s` | **`21.46 s` / `62.82 tok/s`** | **1.01× / +1.9%** |
| **`c=8` (`wall` / `agg` / `sum_dec`)** | `15.51 s` / `66.01` / `90.39` | **`15.23 s` / `67.25` / `91.20`** | **1.02× / +1.9% / +0.9%** |
| **Mid-inference Triton JIT warnings** | `0` | **`0`** | **0** |

### ERNIE-5.5-Flash-Preview-HTR + 3L K=2 DSpark (`num_spec=5`, 32K ctx, `--kv-cache-memory 1395864371`)

| Metric | Measured (`v0.2.7`) |
|---|---:|
| **Total boot to READY** (`43L` target `33.7s` + `3L` draft `1.75s` + CUDAGraph `[1,5,6]` `2s`) | **`85.1 s`** |
| **Model GPU memory** | **`107.0 GiB`** |
| **KV cache capacity** (`1.30 GiB`) | **`34,131 tokens`** (`1.04×` at 32K) |
| **Container CPU RSS / Host RAM available** | **`3.91 GiB` / `6.1 GiB` free** |
| **DSpark mean acceptance length / draft rate** | **`4.61`** (`72.1%` avg; `[0.869, 0.852, 0.721, 0.623, 0.541]`) |
| **`/v1/completions` primes (128 tok, greedy)** | **`60.50 tok/s`** (`2.12 s`) |
| **`/v1/chat/completions` code (128 tok, greedy)** | **`39.16 tok/s`** (`3.27 s`) |
| **`/v1/chat/completions` prose (128 tok, greedy)** | **`31.04 tok/s`** (`4.12 s`) |

> **Host tuning note (128 GB Strix Halo):** On Ubuntu kernels with default CMA (~6 GiB reserved as `free_cma`, unusable by `amdgpu` TTM) and systemd `user-1000.slice` `MemoryMin=3G`, usable pinned GTT caps near ~110 GiB; `htr_runtime/serve_ernie55.sh` uses `--kv-cache-memory 1395864371` (1.30 GiB) by default so ERNIE-5.5 DSpark-5 boots out of the box. Adding `cma=64M` to `GRUB_CMDLINE_LINUX_DEFAULT` unlocks ~6 GiB of additional GPU-usable RAM for larger KV pools.

### ERNIE-5.5-Flash-Preview Quantized KV & Long Context (128 GB Strix Halo)
- **Native Wave32 HIP quantized KV kernels** (`472` total kernels, `0` spills, `0` divOps):
  - `fp8_ds_mla` (`656 B/tok` DSA, `cos=0.99966` vs BF16), `nvfp4_ds_mla` (`352 B/tok` DSA, `cos=0.99550`), and MXFP4 Indexer K cache (`68 B/tok` vs `132 B/tok`, `cos=0.99946`, `1.66x` faster decode).
- **1M Pure FP8 live-verified (`--kv-cache-dtype fp8`, `5.40 GiB` KV):** `1,058,592` GPU KV tokens (`1.01x` @ 1M), `57.62 tok/s` primes, `31.72 tok/s` code, 8K Needle-in-a-Haystack PASS (`30.33s`), DSpark-5 `3.74` mean accept, `2.1 GiB` host RAM free.
- **2M NVFP4 Pool live-verified (`--kv-cache-dtype nvfp4_ds_mla`, `6.55 GiB` KV, `--max-num-seqs 2`):** `2,118,842` GPU KV tokens (`2.02x` @ 1M), `37.6-39.5 tok/s` 2-way parallel, 8K Needle-in-a-Haystack PASS (`27.45s`), DSpark-5 `3.63` mean accept.
- **Allocator & Workspace fixes:** fixed `_max_decode_logits_rows` `4.0 GiB` workspace blowup (`4,096 MiB -> 132 MiB`), fixed `SlidingWindowSpec` anchor inflation (`293 -> 158 KiB/block`), added `empty_cache()` after `profile_run()` in `kv_cache_memory_bytes` path, and bitwise-exact `cos_sin_cache` sharing.
