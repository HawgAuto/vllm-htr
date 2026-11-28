# vLLM HTR v0.2.8 — ERNIE-5.5 2.1× Roofline Prefill, FlashIndexer H=8 & DSpark Filtering

- **Container image:** `ghcr.io/hawgauto/vllm-htr@sha256:356d896546b905ccf89f5611b3376d8cd0a23d82fd0f5e97ecf9aaa5f44e7fbb`
- **Version alias:** `ghcr.io/hawgauto/vllm-htr:v0.2.8`
- **Target hardware:** AMD Strix Halo (`gfx1151`, Radeon 8060S, 128 GB unified LPDDR5X)
- **Scope:** Major ERNIE-5.5-Flash-Preview-HTR prefill, decode, FlashIndexer, DSpark speculation, and parser/launcher release. **Qwen3.8-Flash-Next-HTR `v0.2.7` stack is preserved intact** (`_serving_ernie5=False` isolation verified).

## What's New in v0.2.8

### 1. 686 Native `gfx1151` Wave32 HIP Kernels (0 VGPR spills, 0 SGPR spills, 0 divOps, 0 scratchOps)
- **2.1× Cold Prefill Roofline Rewrite (Zero Extra VRAM):**
  - **HCA Prefill (`ernie_hca_attn.so`):** 3-barrier separate-LDS 2-WG/WGP Wave32 WMMA kernel (`V=174`, `21.8–24.7 TF/s` = **96.5%** of 85W Strix Halo BF16 WMMA peak).
  - **DSA Sparse MLA Prefill (`ernie_dsa_attn.so`):** 2-WG/WGP Wave32 WMMA kernel (`V=188`, `0` spills) supporting both `fp8_ds_mla` and `nvfp4_ds_mla`.
  - **Fused Split-Plane K=5 & `wo_a`+VHA WMMA Prefill (`ernie_k5_sp_prefill.so`, `htr_wo_a_ernie.so`, `hpf5_ernie.so`):** Eliminates dequant-to-BF16 memory spikes while maxing out `gfx1151` WMMA throughput (`128x256_4x4` split-plane K=5, fused `wo_a_vha_k5_prefill_wmma`, `hpf5_ernie` MoE prefill).
  - **Skip Redundant Mixed-Precision KV Zeroing:** Eliminates redundant 3.6 GB/chunk KV block zeroing on ERNIE-5.5 HCA/DSA (strictly scoped; Mamba/GDN zeroing untouched for Qwen3.8).
  - **Shared-K `H=64` Indexer Score Prefill (`ernie_dsa_attn.so`):** `1.24–1.33×` bit-identical speedup for non-FlashIndexer `H=64` fallback (credit `@boxwrench`, folded from #13).
- **FlashIndexer `H=8` Prefill & Decode (`vllm_ernie_htr/flashindexer.py`):**
  - Native `H=64 → H=8` top-8 indexer head selection on real FlashIndexer weights (`~7×` faster indexer prefill, `4.77 ms` @ 16K). Enabled via `--hf-overrides '{"enable_flashindexer": true}'`.
- **Decode & DSpark-5 Speculation Upgrades:**
  - **Zero-in-loop-barrier `1r4w`/`2r2w` BF16 GEMV (`ernie_fast_ops.so`):** `1.32–2.14×` faster BF16 projections (`-0.75 ms/step`, `175–236 GB/s`).
  - **LDS-only barriers & GL2-cached partials (`htr_wo_a_ernie.so`, `htr_moe_ernie.so`):** `woa_lds_barrier()` / `moe_lds_barrier()` + compact `k_moe_plan2`, `0.0` bit-exact.
  - **Filtered DSpark Probabilistic Drafting (`htr_spec_sample.so` + `q38spec`):** Exact `top_k` and full-vocab `logsumexp` `top_p`-only draft filtering with sparse rejection cache, active by default (`ERNIE_DSPARK_DRAFT_FILTER=1`).
- **Built-in `ernie55` / `ernie5` Reasoning & Tool Parsers + Launcher Fixes (#14, #11):**
  - Registered both **`ernie55`** and **`ernie5`** reasoning & tool-call parsers at import time in `vllm/reasoning/__init__.py` and `vllm/tool_parsers/__init__.py` as well as in `vllm_ernie_htr` (no `--reasoning-parser-plugin` required).
  - Enabled `--reasoning-parser ernie55 --enable-auto-tool-choice --tool-call-parser ernie55` by default in `htr_runtime/serve_ernie55.sh` (overridable via `REASONING_PARSER=none` / `TOOL_CALL_PARSER=none`).
  - Auto-detect host numeric `render` GID in `htr_runtime/serve_ernie55.sh` (`--group-add "$(getent group render | cut -d: -f3)"`) alongside `--group-add video` (#11).
  - *Note:* vLLM `0.29.x` exposes parsed `<ernie:think>…</ernie:think>` output on `message.reasoning` / `delta.reasoning` (not the deprecated `reasoning_content`).

## Verified Zero-Overlay Hardware Gate (`gfx1151`, AMD Strix Halo 128 GB)

Measured live zero-overlay on `ghcr.io/hawgauto/vllm-htr:v0.2.8` (`--kv-cache-dtype nvfp4_ds_mla --max-num-batched-tokens 4096 --max-num-seqs 2 --hf-overrides '{"enable_flashindexer": true}' --reasoning-parser ernie55 --enable-auto-tool-choice --tool-call-parser ernie55`, DSpark-5 probabilistic):

| Benchmark | `v0.2.7` | **`v0.2.8` Zero-Overlay** | Speedup / Result |
|---|---:|---:|---|
| **Cold Prefill 8K** | `328.4 tok/s` | **`669.0 tok/s`** (`12.00 s`) | **2.04×** · Needle PASS (`ZETA-4821-COBALT`) |
| **Cold Prefill 16K** | `319.1 tok/s` | **`666.2 tok/s`** (`24.24 s`) | **2.09×** · Needle PASS (`OMEGA-9357-QUARTZ`) |
| **Cold Prefill 32K** | `308.6 tok/s` | **`658.4 tok/s`** (`49.19 s`) | **2.13×** · Needle PASS (`SIGMA-6104-BERYL`) |
| **B=1 `code` (`tk40/tp95`)** | — | **`61.41 tok/s`** dec (`55.89` total) | `mean_acc = 5.35`, `TTFT = 0.43 s` |
| **B=1 `primes` (`tk40/tp95`)** | — | **`60.78 tok/s`** dec (`56.69` total) | `mean_acc = 5.27`, `TTFT = 0.32 s` |
| **B=1 `prose` (`tk40/tp95`)** | — | **`31.44 tok/s`** dec (`29.96` total) | `mean_acc = 2.75`, `TTFT = 0.43 s` |
| **B=1 `code` (`top_p`-only)** | — | **`31.31 tok/s`** dec (`29.85` total) | `mean_acc = 2.84`, `TTFT = 0.43 s` |
| **B=2 Concurrent (`T=12`)** | — | **`42.83 tok/s`** agg (`21.42/user`) | `mean_acc = 3.13`, `0` restarts |
| **Issue #14 (`ernie55` parser)** | unparsed by default | **Blocking & Streaming PASS** | `content == "OK"`, `reasoning` cleanly separated |

## Quick Start

```sh
docker pull ghcr.io/hawgauto/vllm-htr@sha256:356d896546b905ccf89f5611b3376d8cd0a23d82fd0f5e97ecf9aaa5f44e7fbb
```

See [htr_runtime/serve_ernie55.sh](../../serve_ernie55.sh), [ERNIE55-MANIFEST.json](../../ERNIE55-MANIFEST.json), and [V028-NATIVE-MANIFEST.json](../../V028-NATIVE-MANIFEST.json).
