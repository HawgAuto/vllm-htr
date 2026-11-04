# vLLM HTR

**Release v0.2.1 ! 2M, 3M, 4M Max Context enabled!**

[Run the release](htr_runtime/docs/QUANTIZED-KV.md) · [Model / Htr5Core download](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR) · [v0.2.1 release](https://github.com/HawgAuto/vllm-htr/releases/tag/htr-v0.2.1) · [Release history](RELEASES.md)

HTR integrates packed dense/expert kernels, GPTQ8 hyperconnections, SSD-backed PLE, sparse attention and MTP3 into vLLM for AMD Strix Halo / gfx1151. v0.2.1 adds scaled8-bit and RHT-rotated4-bit main-KV serving with fused D256 rotation. This is a serving update; checkpoint weights are unchanged.

## Qualified context profiles

| KV cache | MTP | Max context | YaRN | KV pool |
|---|---:|---:|---:|---:|
| Scaled 8-bit per-token/head KV (`int8_per_token_head`) | 0 | **2,000,000** | ×8 | 27.75 GiB |
| Scaled 8-bit per-token/head KV (`int8_per_token_head`) | 3 | **2,000,000** | ×8 | 27.75 GiB |
| Rotated 4-bit per-token/head KV (`int4_per_token_head`) | 0 | **4,000,000** | ×16 | 26.875 GiB |
| Rotated 4-bit per-token/head KV (`int4_per_token_head`) | 3 | **3,000,000** | ×16 | 22.375 GiB |

One sequence, prefill chunks2048. Profiles passed allocation, API readiness and generation. BF16 defaults/concurrency remain available separately. [Copyable commands](htr_runtime/docs/QUANTIZED-KV.md).

## Accuracy retention versus BF16 KV

| KV cache | MTP | BF16 prediction agreement | Prediction disagreement | Tie-aware agreement |
|---|---:|---:|---:|---:|
| Scaled 8-bit | 0 | **98.18%** (486/495) | **1.82%** | 99.80% |
| Rotated 4-bit | 0 | **98.18%** (486/495) | **1.82%** | 99.39% |
| Scaled 8-bit | 3 | **98.79%** (489/495) | **1.21%** | 99.80% |
| Rotated 4-bit | 3 | **98.38%** (487/495) | **1.62%** | 99.60% |

These are495-position teacher-forced prediction-agreement measurements, not general answer-accuracy percentages. [Metric definitions, raw counts and logprob deltas](htr_runtime/releases/v0.2.1-qualification.json).

## Public image

```sh
docker pull ghcr.io/hawgauto/vllm-htr@sha256:dde8c3ea80e5d0a56e9143894a12622d0c236bab1f6d3d8a10e4ab911c836afb
```

Version alias: `ghcr.io/hawgauto/vllm-htr:v0.2.1`. Preferred sampling is included: temperature1.0, top-p0.95, top-k20, min-p0, presence penalty0, repetition penalty1; thinking remains enabled. No host compiler or additional launcher download is needed. Use the [runtime guide](htr_runtime/docs/QUANTIZED-KV.md) for mounts, cache builder, profile flags and source builds.

## Source and lineage

Executable source: [`d836d40e6deb59e07c50b0db1eb343408278a187`](https://github.com/HawgAuto/vllm-htr/tree/d836d40e6deb59e07c50b0db1eb343408278a187). The release preserves upstream [vLLM](https://github.com/vllm-project/vllm) ancestry and its licenses. Native per-token/head formats retain vLLM's scalar KV wire/rotation conventions and integrate them into HTR's QSA readers; this is not UltraQuant/TurboQuant.

[HTR format and quantizer](https://github.com/HawgAuto/HTR) · [Methodology](https://github.com/HawgAuto/HTR/blob/main/docs/METHODOLOGY.md) · [Acknowledgments and exlr8 → ExLlamaV3/EXL3 → QTIP lineage](https://github.com/HawgAuto/HTR/blob/main/docs/LINEAGE.md). Model weights have their own Qwen Community License.

[Preserved v0.2.0 qualification and historical speed rows](htr_runtime/releases/v0.2.0-qualification.md). FP8 KV and unrelated deferred research are not claimed by this patch release.
