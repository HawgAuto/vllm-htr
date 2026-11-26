# vLLM HTR — v0.2.7

Unified native HTR serving for **AMD Strix Halo / `gfx1151` (Radeon 8060S, 128 GB unified memory)** — serving **both** [**Qwen3.8-Flash-Next-HTR**](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR) (**2.32× faster cold boot**, up to **+21.5% faster decode**) and [**ERNIE-5.5-Flash-Preview-HTR**](https://huggingface.co/hstolte/ERNIE-5.5-Flash-Preview-HTR) (**280B MoE + DSpark-5**, **472 native Wave32 HIP kernels**, **1M Pure FP8** & **2M+ NVFP4 `2.02×` KV pools**, up to **59–60 tok/s**) from a single zero-overlay container image built for models quantized with **[HTR v0.2.0 (Halo Trellis)](https://github.com/HawgAuto/HTR/tree/v0.2.0)**.

[Run v0.2.7](htr_runtime/docs/optimized-serving.md) · [ERNIE-5.5-HTR Model](https://huggingface.co/hstolte/ERNIE-5.5-Flash-Preview-HTR) · [Qwen3.8-HTR Model](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR) · [HTR v0.2.0 Quantizer](https://github.com/HawgAuto/HTR/tree/v0.2.0) · [Release Notes](htr_runtime/releases/v0.2.7/README.md) · [Release History](RELEASES.md)

```sh
docker pull ghcr.io/hawgauto/vllm-htr@sha256:ac7d31a9fcf22da19c412f3d2db2192d163a2e0a06d9a40de7f4697542186e5b
```

Version alias: `ghcr.io/hawgauto/vllm-htr:v0.2.7`. Standard `vllm serve` auto-detects Qwen3.8 vs ERNIE-5.5 and enables every native `gfx1151` Wave32 HIP optimization automatically with zero overlays.

## What's new in v0.2.7

### Unified ERNIE-5.5-Flash-Preview-HTR (280B MoE) + DSpark-5 on 128 GB Strix Halo
- **472 native `gfx1151` Wave32 HIP kernels** across 8 `.so` libraries (`0` VGPR spills, `0` SGPR spills, `0` integer divOps), strictly auto-gated from Qwen3.8 (`_serving_ernie5`).
- **Native Wave32 HIP quantized KV caches:**
  - **1M Pure FP8 (`--kv-cache-dtype fp8`, `5.40 GiB` KV):** **`1,058,592` GPU KV tokens (`1.01×` at 1M)**, `57.6 tok/s` primes, `31.7 tok/s` code, 8K Needle-in-a-Haystack **PASS** (`30.3 s`), DSpark-5 `3.74` mean accept, `2.1 GiB` host RAM free (`cos = 0.99966` vs BF16).
  - **2M+ NVFP4 Pool (`--kv-cache-dtype nvfp4_ds_mla` + MXFP4 Indexer, `6.55 GiB` KV, `--max-num-seqs 2`):** **`2,118,842` GPU KV tokens (`2.02×` concurrency at 1M)**, `37.6–39.5 tok/s` 2-way parallel, `43.5 tok/s` single, 8K Needle-in-a-Haystack **PASS** (`27.5 s`), DSpark-5 `3.63` mean accept (`cos = 0.99550` DSA / `0.99946` MXFP4 Indexer, **1.66× faster** indexer decode).
  - **1M NVFP4 Comfort (`3.40 GiB` KV):** **`1,099,854` tokens (`1.05×` at 1M)**, `59.2 tok/s` primes, `37.1 tok/s` code, ~4 GiB free host RAM.
- **Long-context allocator & workspace fixes:** Eliminated a `4,096 MiB` upstream `_max_decode_logits_rows` workspace blowup (`4,096 → 132 MiB`), fixed `SlidingWindowSpec` KV anchor inflation (`293 → 158 KiB/block`), added per-shard `POSIX_FADV_DONTNEED` streaming weight load, added `empty_cache()` after `profile_run()` in the `--kv-cache-memory-bytes` path, and bitwise-exact `cos_sin_cache` sharing.

### Qwen3.8-Flash-Next-HTR — Fast Boot & Decode Upgrades
- **2.32× faster cold empty-workspace boot:** `388.2 s → 167.3 s` (**−221 s**), warm BF16 boot **1.24× faster** (`114.7 s → 92.6 s`), and KV8 config-change boot **1.52× faster** (`309.8 s → 204.3 s`).
- **Faster decode across BF16 and INT8 KV (`int8_per_token_head`, MTP-3):** KV8 greedy `c=1` **`45.16 → 48.45 tok/s` (+7.3%)**, KV8 13K warm decode **`44.55 → 54.15 tok/s` (+21.5%)**, 13K cold prefill **`14.30 → 12.08 s` (1.18×)**, BF16 `c=4` stochastic **`63.15 / 95.48 → 69.96 / 107.49 tok/s` (+10.8% / +12.6%)**, **`0`** mid-inference Triton JIT warnings.

### Verified Zero-Overlay E2E Gates (AMD Strix Halo `gfx1151`, 128 GB)

#### ERNIE-5.5-Flash-Preview-HTR + DSpark-5 (`1M` Native Context)

| Serving Profile | KV Format | KV Size | GPU KV Tokens | Concurrency @ 1M | Throughput | 8K Needle | DSpark-5 Mean Accept |
|---|---|---:|---:|---:|---|---:|---:|
| **1M Pure FP8** | `fp8` | 5.40 GiB | **1,058,592** | **1.01×** | **57.6 tok/s** primes · **31.7 tok/s** code | **PASS** (`30.3 s`) | **3.74** |
| **2M+ NVFP4 Pool** | `nvfp4_ds_mla` + MXFP4 | 6.55 GiB | **2,118,842** | **2.02×** | **37.6–39.5 tok/s** 2-way · **43.5 tok/s** single | **PASS** (`27.5 s`) | **3.63** |
| **1M NVFP4 Comfort** | `nvfp4_ds_mla` + MXFP4 | 3.40 GiB | **1,099,854** | **1.05×** | **59.2 tok/s** primes · **37.1 tok/s** code | **PASS** | **4.86** |
| **128K Pure FP8** | `fp8` | 1.30 GiB | **162,429** | — | **60.5 tok/s** primes · **39.2 tok/s** code | **PASS** | **4.93** |

#### Qwen3.8-Flash-Next-HTR (`int8_per_token_head`, MTP-3, `seqs=16`)

| Workload | `v0.2.6` KV8 | **`v0.2.7` KV8 Zero-Overlay** | Delta |
|---|---:|---:|---:|
| **Config-change boot** | `309.8 s` | **`204.3 s`** | **1.52× faster (−105.5 s)** |
| **Greedy `c=1`** | `45.16 tok/s` | **`48.45 tok/s`** | **+7.3%** |
| **13K cold prefill / decode** | `14.30 s (910.5)` / `54.50` | **`12.08 s (1,077.8)` / `60.11`** | **1.18× / +10.3%** |
| **13K warm prefill / decode** | `11.44 s (1,137.7)` / `44.55` | **`11.69 s (1,113.6)` / `54.15`** | **−2.1% / +21.5%** |
| **`c=4 × 4K` (`wall` / `sum_dec`)** | `21.77 s` / `61.65 tok/s` | **`21.46 s` / `62.82 tok/s`** | **1.01× / +1.9%** |
| **`c=8` (`wall` / `agg` / `sum_dec`)** | `15.51 s` / `66.01` / `90.39` | **`15.23 s` / `67.25` / `91.20`** | **1.02× / +1.9% / +0.9%** |

> **Strix Halo 128 GB Host Tuning (Ubuntu HWE Kernel 7.0+):** Ubuntu 7.0+ enables `CONFIG_KEXEC_HANDOVER_ENABLE_DEFAULT=y`, silently reserving ~9 GiB as `MIGRATE_CMA` (unusable by ROCm/TTM while `CmaTotal` reports `0 kB`). Add `kho=off` to `GRUB_CMDLINE_LINUX_DEFAULT`, run `sudo update-grub`, and reboot (and optionally lower `user.slice` `MemoryMin=512M`) to unlock the full ~118 GiB GPU unified-memory ceiling for ERNIE-5.5 1M Pure FP8 and 2M+ NVFP4.

Full manifests & notes: [v0.2.7 README](htr_runtime/releases/v0.2.7/README.md) · [V027-NATIVE-MANIFEST.json](htr_runtime/V027-NATIVE-MANIFEST.json) · [ERNIE55-MANIFEST.json](htr_runtime/ERNIE55-MANIFEST.json) · [GitHub Release `htr-v0.2.7`](https://github.com/HawgAuto/vllm-htr/releases/tag/htr-v0.2.7).

## Preserved v0.2.6 release

Native HTR serving for **AMD Strix Halo / gfx1151** — now with **2× faster long-context prefill**, **zero mid-inference Triton JIT pauses**, and **RDNA 3.5 ISA decode upgrades**. All model weights, tokenizers, configs, and sampling defaults are unchanged — just pull the new image.

[Run v0.2.6](htr_runtime/docs/optimized-serving.md) · [Htr5Core model](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR) · [Exact results and release notes](htr_runtime/releases/v0.2.6/README.md) · [Release history](RELEASES.md)

```sh
docker pull ghcr.io/hawgauto/vllm-htr@sha256:dc11916549c534b207bb45048fe2f9b2177117995931f5aff5b384b274ea0fe8
```

Version alias: `ghcr.io/hawgauto/vllm-htr:v0.2.6`. Standard `vllm serve` enables every v0.2.6 optimization automatically via `runtime-env.json` with no extra `-e` flags required.

## What's new in v0.2.6

- **2× faster long-context prefill:** 13K-token warm prefill drops from **23.58 s → 11.44 s** (**551.9 → 1,137.7 tok/s**), and cold 13K prefill reaches **14.30 s** (**910.5 tok/s**, **1.65×** faster).
- **Zero Triton JIT on the serving hot path:** **20 new ahead-of-time (AOT) zero-spill HIP kernels** compiled specifically for `gfx1151` Wave32 replace the remaining Triton hot-path kernels across QSA attention, GDN, `causal_conv1d`, W4A16 MoE, HyperConnection, dense prefill, PLE transpose, and Gumbel sampling (**`0`** mid-inference Triton JIT warnings).
- **Full-INT8 WMMA attention prefill (with instant BF16 fallback):** Uses RDNA 3.5 `v_wmma_i32_16x16x16_iu8` (`qsa_tau_v5_kv8_i8`) for **1.66–1.70× faster** QSA attention prefill (**−865 ms** on 13K prompts) with verified parity (`cos = 0.999994`, exact 200/200-char greedy match, `63.45%` MTP-3 spec acceptance). Pass `-e HTR_QSA_TAU_V5_I8=0` to revert to BF16 WMMA at any time.
- **Faster multi-request concurrency & strided-cache fix:** `c=8` wall time improves **19.03 s → 15.51 s** (**1.23× faster**, `66.01 tok/s` aggregate) and `c=4 × 4K` long-context concurrency improves **23.82 s → 21.77 s** (**+7.7% sum decode** to `61.65 tok/s`), including a bit-exact fix for strided-1D `causal_conv1d` cache indexing under high concurrency.
- **RDNA 3.5 (`gfx1151`) ISA decode upgrades:** Unlocks the hidden dual-issue `v_dual_dot2acc_f32_bf16` instruction (4 BF16 FMAs/lane/cycle), single-cycle `v_perm_b32` BF16 packing, and bank-staggered VOPD pairing across `htr_bf16_gemv.so`, `htr_moe_k6_shx.so`, and `htr_kernels.so`.

### Verified zero-overlay E2E results (`v0.2.6` vs `v0.2.5`)

Measured zero-overlay on exclusive AMD Strix Halo (`gfx1151`), Htr5Core, MTP3, `max-model-len=32768`, `--kv-cache-dtype int8_per_token_head`:

| Workload | `v0.2.5` KV8 | **`v0.2.6` Zero-Overlay Gate** | **Change vs v0.2.5** |
|---|---:|---:|---:|
| `13K` warm prefill (`13,018 tok`) | `551.9 tok/s` (`23.58 s`) | **`1,137.7 tok/s`** (`11.44 s`) | **2.06× faster** |
| `13K` cold prefill (`13,017 tok`) | `551.9 tok/s` (`23.58 s`) | **`910.5 tok/s`** (`14.30 s`) | **1.65× faster** |
| Greedy `c=1` decode (`128 tok`) | `45.05 tok/s` | **`45.16 tok/s`** | **+0.2%** |
| `c=4 × 4K` (`16,702` ctx) — wall / sum decode | `23.82 s` / `57.25 tok/s` | **`21.77 s`** / **`61.65 tok/s`** | **1.09× wall / +7.7% dec** |
| `c=8` high-concurrency — wall / agg (`8×128`) | `19.03 s` / `53.80 tok/s` | **`15.51 s`** / **`66.01 tok/s`** | **1.23× wall** |
| Mid-inference Triton JIT warnings | multiple | **`0`** | **eliminated** |

Executable source: [`00f0b8c0caf7d85fee9232ecf600e0317408b1ab`](https://github.com/HawgAuto/vllm-htr/tree/00f0b8c0caf7d85fee9232ecf600e0317408b1ab).
Full manifest & notes: [v0.2.6 README](htr_runtime/releases/v0.2.6/README.md) · [V026-NATIVE-MANIFEST.json](htr_runtime/V026-NATIVE-MANIFEST.json).

## Preserved v0.2.5 release

Native HTR serving for **AMD Strix Halo / gfx1151**, shipping the full clean-slate
Wave32 HIP/HSACO optimization stack (**18 new native binaries** + **38 runtime updates**):
**K6 routed-MoE `rows=1..16` with zero-realloc shared-expert folding (`T<=4`)**,
**HPF4 MoE prefill**, **nontemporal `global_load_b128` K4 `lm_head`**, **dense WMMA**,
**fused GDN MTP spec decode**, **`hiphc_v2` fused HyperConnection**, **QSA native decode
union (BF16 / INT8 / INT4 KV) & fused canon-expand**, **`ple_gather_v25` + native PLE
short-conv/gate-norm**, **MTP DDN v2/WMMA + BF16 GEMV**, and **native Q38 filter +
Gumbel + sparse rejection sampling**. Weights, tokenizer/configuration and checkpoint
sampling defaults are unchanged.

[Run v0.2.5](htr_runtime/docs/optimized-serving.md) · [Htr5Core model](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR) · [Exact results and release notes](htr_runtime/releases/v0.2.5/README.md) · [Release history](RELEASES.md)

```sh
docker pull ghcr.io/hawgauto/vllm-htr@sha256:daa6b2847bf9678e2de39b8b216105ffe13313e50db889f002f5f918e581d193
```

Version alias: `ghcr.io/hawgauto/vllm-htr:v0.2.5`. Ordinary `vllm serve` receives all
v0.2.5 optimizations by default via `runtime-env.json` with zero extra `-e` flags.

## What v0.2.5 brings

Verified back-to-back on exclusive AMD Strix Halo (`gfx1151`), Htr5Core, MTP3, `max-model-len=32768`, `4 GiB` KV cache:

### BF16 KV (`--max-num-seqs 4`, `--max-num-batched-tokens 3200`)
| Workload | Stock v0.2.4 | **v0.2.5 (ARM 7g)** | **v0.2.5 Zero-Overlay Gate** | **Change vs v0.2.4** |
|---|---:|---:|---:|---:|
| Greedy `c=1` decode (`128 tok`) | 42.21 tok/s (`55.7 ms/st`) | **44.60 tok/s** (`51.8 ms/st`) | 43.05 tok/s | **+5.7% tok/s / +7.6% step** |
| `c=1` decode step latency (`256 tok`) | 58.0–58.4 ms/st | **54.6–55.0 ms/st** | — | **+6.2%–+6.3% step speed** |
| `4K` decode (`5,812` ctx, `128 tok`) | 54.59 tok/s | 59.21 tok/s | **59.56 tok/s** | **+8.5%–+9.1%** |
| `4K` prefill (`5,812 tok`) | 977.7 tok/s | **983.9 tok/s** | 907.0 tok/s | **+0.6%** |
| `c=4` concurrent (`t=0.7`, `4×256`) — sum decode | 71.33 tok/s | 91.61 tok/s | **100.50 tok/s** | **+28.4%–+40.9%** |
| `c=4` concurrent (`t=0.7`, `4×256`) — wall agg | 51.44 tok/s (`19.91 s`) | **69.99 tok/s** (`14.63 s`) | 68.66 tok/s (`14.91 s`) | **+33.5%–+36.1%** |
| `c=4` greedy (`t=0.0`, `4×256`) — sum decode | 80.37 tok/s (`15.59 s`) | 97.94 tok/s (`12.73 s`) | **109.90 tok/s** (`11.68 s`) | **+21.9%–+36.7%** |
| MTP3 acceptance rate | 69.96% (`3.10 tok/st`) | **73.80%** (`3.21 tok/st`) | 71.60% (`3.15 tok/st`) | **+1.6–+3.8 pp** |

### INT8 KV Long-Context & High-Concurrency (`--kv-cache-dtype int8_per_token_head`, `--max-num-seqs 8`)
| Workload | v0.2.4 KV8 | **v0.2.5 KV8 (+ V5-Q4)** | **Change** |
|---|---:|---:|---:|
| Greedy `c=1` decode (`128 tok`) | 43.18 tok/s | **45.05 tok/s** | **+4.3%** |
| `13K` long-context prefill (`13,017 tok`) | 409.6 tok/s (`31.78 s`) | **551.9 tok/s** (`23.58 s`) | **+34.7%** |
| `13K` long-context decode (`13,017` ctx) | 43.39 tok/s | **63.92 tok/s** | **+47.3%** |
| `c=4 × 4K` (`16,702` ctx) — wall time / agg | 33.99 s (`15.06 tok/s`) | **23.82 s** (`21.49 tok/s`) | **+42.7%** |
| `c=8` high-concurrency — sum decode (`8×128`) | 103.96 tok/s | **132.18 tok/s** | **+27.1%** |
| `c=8` high-concurrency — wall agg (`8×128`) | 44.76 tok/s (`22.88 s`) | **53.80 tok/s** (`19.03 s`) | **+20.2%** |

Executable source: [`c427a3851ac6e84177a236c17e9ed38303504814`](https://github.com/HawgAuto/vllm-htr/tree/c427a3851ac6e84177a236c17e9ed38303504814).
Full manifest & notes: [v0.2.5 README](htr_runtime/releases/v0.2.5/README.md) · [V025-NATIVE-MANIFEST.json](htr_runtime/V025-NATIVE-MANIFEST.json).

## Preserved v0.2.4 release

Native HTR serving for **AMD Strix Halo / gfx1151**, with **K6 routed-MoE decode**,
**dense WMMA at eligible 9–15 rows**, and the **PLE zero-row startup repair**.
Weights, tokenizer/configuration and checkpoint sampling defaults are unchanged.

[Run v0.2.4](htr_runtime/docs/optimized-serving.md) · [Htr5Core model](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR) · [Exact results and limitations](htr_runtime/releases/v0.2.4/README.md) · [Release history](RELEASES.md)

```sh
docker pull ghcr.io/hawgauto/vllm-htr@sha256:cbd537589cc7f2e6e9d73940d5da49a9d956ffbfd795277f54b15787284eeab5
```

Version alias: `ghcr.io/hawgauto/vllm-htr:v0.2.4`. Ordinary `vllm serve` receives
the optimizations by default; native flags and explicit disable controls remain
available. No forced evaluation seed, persistent unpacked-weight shadow or
mandatory compiler-cache download. PLE hot rows remain off by default.

## What v0.2.4 brings

The retained selected-stack screens used Htr5Core, MTP3, maximum four sequences,
40,960 configured context and two mirrored visits per arm:

| Single-stream metric | v0.2.3 | Selected v0.2.4 stack | Change |
|---|---:|---:|---:|
| Steady aggregate output | 48.05 tok/s | 55.35 tok/s | +15.19% |
| Mixed prompts, output / full wall | 34.08 tok/s | 36.41 tok/s | +6.82% |
| Mixed prompts, per-request decode p50 | 48.10 tok/s | 53.83 tok/s | +11.91% |

Multi-stream changes varied: steady 2/3/4-stream −5.60% / +2.44% / +4.14%;
mixed-prompt +1.77% / +4.46% / +0.96%. There is no established multi-stream or
prefill win, and two visits do not establish statistical noise bounds.

Separate task screens (262K configured context, INT8 KV, MTP3, thinking on,
temperature1.0/top-p0.95/top-k20, optional2GiB PLE hot rows) recorded HumanEval+
159 → 162 / 164, ToolEval125 / 138 both, and Hermes20 1850 → 1950 / 2000.
These selected-stack performance/quality screens were not rerun on the packaged
image. HumanEval/157 still looped at evaluation seed42 (reproduced twice), but not
at seeds1–3 or101–140; this finite screen is not a population loop-rate estimate.
[All visits, exact conditions and limitations](htr_runtime/releases/v0.2.4/README.md).
The historical throughput targets remain unmet.

Qualification covered five real-model starts (three final-image, two reference),
matched token IDs and logprobs
against the selected stack in MTP0/MTP3, concurrent-request delivery, native
route/self-checks and a documented-command cold-start/tool round trip. Empty
workspace MTP3 readiness took436.1s; first start and new shapes can compile.
This is short-request delivery/packaging qualification, not whole-model bitwise
or maximum-context validation. [Qualification](htr_runtime/releases/v0.2.4/QUALIFICATION.json).

Executable source: [`62cb02413de66042cb11c6dc853922618c368887`](https://github.com/HawgAuto/vllm-htr/tree/62cb02413de66042cb11c6dc853922618c368887).
The release tag adds documentation/evidence only. Historical sections below retain
their original pins and scope; use the current guide above for v0.2.4.

## Preserved v0.2.3 release

Native HTR serving for **AMD Strix Halo / gfx1151**, now with the retained
concurrency stack: **grouped MoE**, **selected dense WMMA**, and **expanded
asynchronous PLE**. This update also repairs the stale quantized-KV binding
manifest that could block the INT8/INT4 compatibility launcher. Model weights,
quantization, supported KV formats and sampling defaults are unchanged.

[Run v0.2.3](htr_runtime/docs/optimized-serving.md) · [Htr5Core model](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR) · [Release history](RELEASES.md)

```sh
docker pull ghcr.io/hawgauto/vllm-htr:v0.2.3-coldstart.1
```

This image is v0.2.3 plus the cold-start fix; the v0.2.3 qualification measurements were made on the v0.2.3 image and were not re-run.

Use native vLLM flags to choose KV dtype/budget, concurrency, context and MTP.
The new paths are on by default with guarded fallback and explicit disable
controls. PLE hot-row caching remains off. No persistent unpacked-weight shadow
or mandatory downloaded compile seed is introduced.

The retained stack was selected through cumulative model screens, not by adding
isolated speedups together. High-concurrency wins do not imply a single-stream
speedup, broad accuracy or newly validated maximum context. The historical
throughput targets remain unmet. Rejected candidates and unfinished concurrency
work are excluded. [Final-image qualification and raw timing rows](htr_runtime/releases/v0.2.3/README.md) document the exact exercised scope.

The following v0.2.2 and earlier sections retain their original measurements,
image pins and qualification scope; use the current guide above for v0.2.3.

## Preserved v0.2.2 release

Native `vllm serve` for the HTR Qwen3.8 runtime on **AMD Strix Halo / gfx1151**. This serving update combines the accepted kernels, optional Octave KV and bounded PLE caching without changing checkpoint weights.

[Run v0.2.2](htr_runtime/docs/native-serving.md) · [Htr5Core model](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR) · [Release](https://github.com/HawgAuto/vllm-htr/releases/tag/htr-v0.2.2) · [Release history](RELEASES.md)

## What changed

- Ordinary **`vllm serve`**, with native control of sampling, context, KV bytes/format, concurrency and prefill budget. No mandatory profile launcher.
- **Three optional Octave KV formats:** `octave_k3v4`, `octave_k3v3`, `octave_k3v3_compact`, with native gfx1151 kernels, deterministic window compaction and the masked-unused-slot NaN fix.
- Accepted sampler/draft improvements, **wave32 GDN**, and **bit-exact QSA preparation**, with packed-KV routing kept separate.
- The accepted **exact HIP draft filter** is on by default; unsupported cases retain the original fallback and `HTR_Q38F=off` remains available.
- Optional **bounded CPU-RAM PLE hot-row cache**, off by default. The large PLE tables remain SSD-backed.
- Persistent per-configuration compiler-cache reuse. No persistent expanded 16-bit weight shadow and no old seed silently reused as a new-image seed.

## Get the image

```sh
docker pull ghcr.io/hawgauto/vllm-htr@sha256:034898ebd182a240fa403ad5ea24bea3f647d933571e43420a27daf9af079d1c
```

Version alias: `ghcr.io/hawgauto/vllm-htr:v0.2.2`. See the [native guide](htr_runtime/docs/native-serving.md) for the complete loopback-only Docker command, API calls, flags and source build. Existing Htr5Core downloads remain valid; no requantization is required.

The native path inherits checkpoint sampling defaults unless you override them. The predecessor's forced temperature1.0/top-p0.95 profile is no longer injected. Thinking/template settings are unchanged.

## Qualification

**18 configurations**, comprising16 new-image loads and2 unchanged controls; **324 requests**, including **90 reconstructed SSE streams**. Five matched pairs retained exact streamed/legacy tokens, text and finish reasons and exact teacher-probe logprobs. PLE off/on, native fresh/seeded startup, tools, all three Octave formats and retained INT8/INT4 KV were exercised. [Complete results and timing rows](htr_runtime/releases/v0.2.2/README.md).

Scope: Htr5Core, one sequence, configured32768 context and2GiB KV. This is not a new broad-accuracy, filled-window, concurrency or multimodal qualification. The native controls remain user-configurable; the tested envelope is not a parser restriction. Earlier context/accuracy results below retain their original image/version scope.

## Historical v0.2.1 context profiles

| KV cache | MTP | Max context | YaRN | KV pool |
|---|---:|---:|---:|---:|
| Scaled 8-bit per-token/head KV (`int8_per_token_head`) | 0 | **2,000,000** | ×8 | 27.75 GiB |
| Scaled 8-bit per-token/head KV (`int8_per_token_head`) | 3 | **2,000,000** | ×8 | 27.75 GiB |
| Rotated 4-bit per-token/head KV (`int4_per_token_head`) | 0 | **4,000,000** | ×16 | 26.875 GiB |
| Rotated 4-bit per-token/head KV (`int4_per_token_head`) | 3 | **3,000,000** | ×16 | 22.375 GiB |

One sequence, prefill chunks2048. Profiles passed allocation, API readiness and generation. BF16 defaults/concurrency remain available separately. [Copyable commands](htr_runtime/docs/QUANTIZED-KV.md).

## Historical v0.2.1 accuracy retention versus BF16 KV

| KV cache | MTP | BF16 prediction agreement | Prediction disagreement | Tie-aware agreement |
|---|---:|---:|---:|---:|
| Scaled 8-bit | 0 | **98.18%** (486/495) | **1.82%** | 99.80% |
| Rotated 4-bit | 0 | **98.18%** (486/495) | **1.82%** | 99.39% |
| Scaled 8-bit | 3 | **98.79%** (489/495) | **1.21%** | 99.80% |
| Rotated 4-bit | 3 | **98.38%** (487/495) | **1.62%** | 99.60% |

These are495-position teacher-forced prediction-agreement measurements, not general answer-accuracy percentages. [Metric definitions, raw counts and logprob deltas](htr_runtime/releases/v0.2.1-qualification.json).

These measurements and capacity recipes belong to the unchanged v0.2.1 image, not a requalification of those capacities on v0.2.2. [Pinned predecessor instructions](htr_runtime/docs/QUANTIZED-KV.md).

## Source and lineage

Executable source: [`9a4917f41101926c8cb23d630018b8e5e4aec44c`](https://github.com/HawgAuto/vllm-htr/tree/9a4917f41101926c8cb23d630018b8e5e4aec44c). Release documentation is a documentation-only successor; the image retains this exact executable source. HTR preserves upstream [vLLM](https://github.com/vllm-project/vllm) ancestry and licenses. Octave is the upstream ROCm Octave work adapted for gfx1151, with the correctness fixes described above.

[HTR format and quantizer](https://github.com/HawgAuto/HTR) · [Methodology](https://github.com/HawgAuto/HTR/blob/main/docs/METHODOLOGY.md) · [Acknowledgments and exlr8 → ExLlamaV3/EXL3 → QTIP lineage](https://github.com/HawgAuto/HTR/blob/main/docs/LINEAGE.md). Model weights retain their Qwen Community License.

[Preserved v0.2.0 qualification and speed rows](htr_runtime/releases/v0.2.0-qualification.md). FP8 KV and unrelated deferred research are not claimed by this release.
