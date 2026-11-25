# vLLM HTR — v0.2.6

Native HTR serving for **AMD Strix Halo / gfx1151** with **zero mid-inference Triton JIT**
across prefill and decode: **20 new zero-spill AOT Wave32 HIP/HSACO kernels**
(full-INT8 `v_wmma_i32_16x16x16_iu8` QSA prefill, 128-bit vectorized `causal_conv1d` & GDN
prefill, double-buffered W4A16 MoE, HPF5 v2 MoE prefill, dense prefill v2, HyperConnection v2,
QSA MQA v3 with hidden `v_dual_dot2acc_f32_bf16`, cache/prepare/indexer/PLE/Gumbel AOT) plus
**3 Wave 7 RDNA 3.5 decode upgrades** (`htr_bf16_gemv.so`, `htr_moe_k6_shx.so`, `htr_kernels.so`).
Weights, tokenizer/configuration and checkpoint sampling defaults are unchanged.

[Run v0.2.6](htr_runtime/docs/optimized-serving.md) · [Htr5Core model](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR) · [Exact results and release notes](htr_runtime/releases/v0.2.6/README.md) · [Release history](RELEASES.md)

```sh
docker pull ghcr.io/hawgauto/vllm-htr@sha256:dc11916549c534b207bb45048fe2f9b2177117995931f5aff5b384b274ea0fe8
```

Version alias: `ghcr.io/hawgauto/vllm-htr:v0.2.6`. Ordinary `vllm serve` receives all
v0.2.6 optimizations by default via `runtime-env.json` with zero extra `-e` flags.

## What v0.2.6 brings

Verified zero-overlay on exclusive AMD Strix Halo (`gfx1151`), Htr5Core, MTP3, `max-model-len=32768`, `--kv-cache-dtype int8_per_token_head`:

| Workload | `v0.2.5` KV8 | **`v0.2.6` Zero-Overlay Gate** | **Change vs v0.2.5** |
|---|---:|---:|---:|
| `13K` warm prefill (`13,018 tok`) | `551.9 tok/s` (`23.58 s`) | **`1,137.7 tok/s`** (`11.44 s`) | **2.06× faster** |
| `13K` cold prefill (`13,017 tok`) | `551.9 tok/s` (`23.58 s`) | **`910.5 tok/s`** (`14.30 s`) | **1.65× faster** |
| Greedy `c=1` decode (`128 tok`) | `45.05 tok/s` | **`45.16 tok/s`** | **+0.2%** |
| `c=4 × 4K` (`16,702` ctx) — wall / sum decode | `23.82 s` / `57.25 tok/s` | **`21.77 s`** / **`61.65 tok/s`** | **1.09× wall / +7.7% dec** |
| `c=8` high-concurrency — wall / agg (`8×128`) | `19.03 s` / `53.80 tok/s` | **`15.51 s`** / **`66.01 tok/s`** | **1.23× wall** |
| Mid-inference Triton JIT warnings | multiple | **`0`** | **eliminated** |

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
