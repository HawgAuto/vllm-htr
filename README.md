# vLLM HTR — v0.2.3

Native HTR serving for **AMD Strix Halo / gfx1151**, now with the retained
concurrency stack: **grouped MoE**, **selected dense WMMA**, and **expanded
asynchronous PLE**. This update also repairs the stale quantized-KV binding
manifest that could block the INT8/INT4 compatibility launcher. Model weights,
quantization, supported KV formats and sampling defaults are unchanged.

[Run v0.2.3](htr_runtime/docs/optimized-serving.md) · [Htr5Core model](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR) · [Release history](RELEASES.md)

```sh
docker pull ghcr.io/hawgauto/vllm-htr:v0.2.3
```

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
