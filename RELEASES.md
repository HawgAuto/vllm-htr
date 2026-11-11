# vLLM HTR release history

## v0.2.3-coldstart.1 — cold start from an empty workspace

v0.2.3 plus the cold-start fix: `htr_native_policy.environment_defaults()` now creates `TMPDIR` and the cache roots, so the first start no longer needs `mkdir -p "$WORKSPACE/tmp"` before `docker run`.

Image `ghcr.io/hawgauto/vllm-htr@sha256:7b7a4688671d945c439c0f934611598975f6fbe67459ca36897b824c17e424a3`; image alias `v0.2.3-coldstart.1`; source commit `879fc13d99a0dea2f05ffce0bc93f11b38e8d490` (= main after `b885a03a56`, i.e. the `htr-v0.2.3` tag plus this fix); same recipe and base as v0.2.3. The `v0.2.3` tag and image are unchanged.

Checked with one cold start on the AMD Strix Halo iGPU from a completely empty workspace using the documented start command and no `mkdir` workaround: healthy after 659 s (11 min 0 s) from container start, one chat request answered (HTTP 200, `42`), clean shutdown, zero `unable to make temporary file`, `Traceback` or `ERROR` lines in either stream; the aiter JIT build completed in 133.1 s and produced a `.so` byte-identical (`65a82a1363c1…`) to the production artifact. `/workspace/tmp` and `/workspace/cache` were created at +0.1 s, before the compile.

Known differences to v0.2.3 (verified against the v0.2.3 image, 411268 paths before / 411284 after — 16 added, 0 removed, 18 changed, provenance-checked byte for byte against source `879fc13d99`): 117 of 120 image layers are shared and the 3 differing layers are the release's own `COPY`/`COPY`/`RUN` positions; the content changes are the fix, its unit test, the v0.2.3 release-note files and the install records; 410 files are mode 664 instead of 644 with identical content (build-environment artifact: 664 build context, `shutil.copytree`); `NATIVE-POLICY-REVISION` changed `6d95ec40…` → `6c1cc60a…` because the revision hash covers the fixed policy file; and `/payload/source/htr_runtime/python/htr_native_policy.py` still holds the older base-image policy file, which is not imported — `import htr_native_policy` resolves to the fixed `/opt/vllm-htr/python` copy. Entrypoint, user, working directory and all 51 environment values are identical to v0.2.3; only the `org.opencontainers.image.revision` label differs.

v0.2.3 performance and capacity qualification was not re-run on this image and no new performance claim is made; the v0.2.3 numbers remain measurements of the v0.2.3 executable bytes.

## v0.2.3 — optimized concurrency stack and KV startup repair

Default-on grouped MoE, selected packed dense WMMA and expanded asynchronous PLE are folded into the native runtime. PLE limits follow MTP0/MTP3 settings; hot-row caching remains off. The INT8/INT4 compatibility binding repair corrects stale hashes without changing KV arithmetic or removing integrity checks. Unsupported shapes retain the original route and explicit controls remain available.

Executable source `ff9897df3d5b5405af6ee2353a10b0fcd8b92394`; image `ghcr.io/hawgauto/vllm-htr@sha256:417cd5d14ce241128379614de7dc128bbc4f236b8c0f3853d6d91c5c4d84b221`; image alias `v0.2.3`; runtime tag `htr-v0.2.3`; model metadata tag `v0.2.3`. Checkpoint weights, tokenizer/configuration and historical releases remain unchanged. Documentation-only successors do not replace the qualified executable bytes.

[Versioned release evidence](htr_runtime/releases/v0.2.3/QUALIFICATION.json) gives the installed-image scope. The MTP0 short Lisbon continuation and returned logprobs match the already-selected winner in both repeats; whole-model bitwise identity to v0.2.2 and broad task accuracy are not claimed. High-concurrency selection does not establish universal or single-stream speedups; historical throughput targets remain unmet. No prebuilt compiler seed is shipped and new shapes may compile.


## v0.2.2 — combined native runtime, Octave KV and bounded PLE

Ordinary `vllm serve` now supplies the accepted sampler/draft improvements, wave32 GDN, bit-exact QSA preparation and default-on exact HIP draft filter. Three optional Octave KV formats add native gfx1151 kernels and deterministic compaction, with a masked-unused-slot NaN repair. Optional bounded PLE hot-row caching is off by default. Sampling, KV/context budgets, concurrency and prefill settings remain native user controls; checkpoint sampling replaces the predecessor's forced response profile. MTP0 and MTP3 remain available.

Executable source `9a4917f41101926c8cb23d630018b8e5e4aec44c`; image `ghcr.io/hawgauto/vllm-htr@sha256:034898ebd182a240fa403ad5ea24bea3f647d933571e43420a27daf9af079d1c`; image alias `v0.2.2`; runtime tag `htr-v0.2.2`; model-card tag `v0.2.2`. Documentation-only release commits do not change these executable bytes. No weights, tokenizer/configuration or historical tags change.

The integration matrix passed18 configurations (16 new-image,2 inherited controls),324 requests and90 reconstructed SSE streams, including five exact matched parity pairs. This is Htr5Core/single-sequence/configured32768-context/2GiB-KV qualification; earlier million-context, concurrency and quality measurements retain their original image scope. EOS-enabled prose probes reached their128-token cap; tool-result turns stopped normally. Persistent compiler-cache reuse is retained; no prebuilt seed is shipped and new request shapes may compile. See the current native guide and versioned qualification evidence.

## v0.2.1 — quantized KV and 2M / 3M / 4M context

Scaled8-bit per-token/head KV (`int8_per_token_head`) supports2M with MTP0 or MTP3. RHT-rotated4-bit per-token/head KV (`int4_per_token_head`) supports4M with MTP0 or3M with MTP3. The retained fusedD256 rotation, native KV selfchecks and source-bound cache profiles are included. Preferred sampling is built into the image; weights and older release tags remain unchanged.

| KV cache | MTP | Max context | YaRN | KV pool |
|---|---:|---:|---:|---:|
| Scaled 8-bit per-token/head KV (`int8_per_token_head`) | 0 | **2,000,000** | ×8 | 27.75 GiB |
| Scaled 8-bit per-token/head KV (`int8_per_token_head`) | 3 | **2,000,000** | ×8 | 27.75 GiB |
| Rotated 4-bit per-token/head KV (`int4_per_token_head`) | 0 | **4,000,000** | ×16 | 26.875 GiB |
| Rotated 4-bit per-token/head KV (`int4_per_token_head`) | 3 | **3,000,000** | ×16 | 22.375 GiB |

| KV cache | MTP | BF16 prediction agreement | Prediction disagreement | Tie-aware agreement |
|---|---:|---:|---:|---:|
| Scaled 8-bit | 0 | **98.18%** (486/495) | **1.82%** | 99.80% |
| Rotated 4-bit | 0 | **98.18%** (486/495) | **1.82%** | 99.39% |
| Scaled 8-bit | 3 | **98.79%** (489/495) | **1.21%** | 99.80% |
| Rotated 4-bit | 3 | **98.38%** (487/495) | **1.62%** | 99.60% |

Accuracy figures are495-position teacher-forced BF16-KV prediction agreement, not broad task accuracy. [Profile/accuracy details](htr_runtime/docs/QUANTIZED-KV.md). Executable source `d836d40e6deb59e07c50b0db1eb343408278a187`; image `ghcr.io/hawgauto/vllm-htr@sha256:dde8c3ea80e5d0a56e9143894a12622d0c236bab1f6d3d8a10e4ab911c836afb`; GitHub tag `htr-v0.2.1`; model tag `v0.2.1`. The four profiles are capacity-qualified by load/readiness/generation.


HTR serving-release versions are shared with the [Qwen3.8-Flash-Next-HTR model card](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR). They are independent of upstream vLLM and the separate HTR quantizer's version numbers.

## v0.1.0 — initial release

This names the existing working, qualified HTR runtime and complete model checkpoint. It is not a release of the serving changes currently under development. Assigning the version does not change model weights, runtime executables or the tested image.

- Runtime source: [`63ba082fbf6d204dba64bfb3803ccf48ad01fff1`](https://github.com/HawgAuto/vllm-htr/tree/63ba082fbf6d204dba64bfb3803ccf48ad01fff1), tagged `htr-v0.1.0`. The original `htr-qsa1600-release-20261028` tag remains unchanged. Later documentation-only commits do not change this qualified source pin.
- Image: `ghcr.io/hawgauto/vllm-htr:v0.1.0`, an alias of `qsa1600-release-20261028`, with the identical tested manifest `sha256:82cad72483626841b0f858606e5d1951b09645bd109a24eeb3ed7241561aba73`.
- Model release: [`hstolte/Qwen3.8-Flash-Next-HTR`, tag `v0.1.0`](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR/tree/v0.1.0). Checkpoint payload unchanged from revision `5ffabdde9d79f48a2dd6c2331140f1936ca6bb16`; all 63 checkpoint files match their manifest sizes and SHA-256 values.
- Shipped profile: AMD gfx1151 / Strix Halo, QSA1600, probabilistic MTP3, BF16 KV, thinking and structured tool calling, SSD-backed PLE, HTR dense/expert kernels and GPTQ8 hyperconnections.
- Qualification: [the existing final-image delivery receipt](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR/blob/v0.1.0/evidence/release-qualification.json) covers text, reasoning, MTP and a structured tool-call/result round trip. Its original controller FAIL and separate PASS_ADJUDICATED explanation are retained. This version label is not a new speed or quality measurement.

### Access and limitations

The runtime source, complete model checkpoint and prebuilt GHCR image are public. The image can be pulled anonymously without a GitHub account, token or `docker login`. The visibility change preserves the tested image digest, runtime code, checkpoint and release tags. See the maintained [image/source-build instructions](htr_runtime/README.md) and [model running guide](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR/blob/main/docs/RUNNING.md); immutable release-tag snapshots retain their original documentation.

QSA1600 is approximate sparse attention. The configured 262,144-token ceiling is not a completed maximum-context test. Final-composite quality and image/video execution are not qualified. No precompiled quick-load seed is shipped in v0.1.0. Historical performance tables remain labeled with their original runtime/configuration.

## v0.2.0 — first serving update

Executable source: [`ad52c7df6d461d653199264f6d5bc216984aea3b`](https://github.com/HawgAuto/vllm-htr/tree/ad52c7df6d461d653199264f6d5bc216984aea3b); release tag `htr-v0.2.0` adds documentation/evidence only. Public image `ghcr.io/hawgauto/vllm-htr:v0.2.0` is pinned to Linux/amd64 manifest `sha256:4c3bbfc5f42dad17c1a3bca2a911502aa0670aff1314e87e11bc900eaff491df`. The model release is [`v0.2.0`](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR/tree/v0.2.0).

Integrated TAU v2 sparse attention, partition4 MoE dispatch, per-call dense-prefill dequantization, sticky deterministic compilation, mode-specific reference GDN geometry and reference MTP RMS reduction. Temporary MoE oracle scratch recovers approximately4.6875GiB ready-state GTT in matched checks; traffic-peak savings were much smaller. Numerical checks remain enabled. Nested runtime source participates in quick-load cache identity.

The final-image diagnostic matched18 complete output pairs, including16 returned-logprob pairs. All four MTP0/MTP3 × one/four-slot workflows passed the public seed builder and cold replay at context1,000,000 / YaRN4 / shared27GiB BF16 KV. Warmups reached32,768 tokens; **no full-million prefill/retrieval or four independent million-token histories is claimed**. New-shape compilation remains possible; no prebuilt seed archive is shipped. The final documented default command separately passed text, reasoning, MTP and structured tool-call/result delivery.

Checkpoint weights, tokenizer and quantization remain unchanged. Preserved predecessor speed/reference rows are explicitly labeled; they are not final-image timings or a general speedup. Known short-prefill overhead and PLE/I/O tails remain. See the [release evidence and raw rows](https://github.com/HawgAuto/vllm-htr/blob/htr-v0.2.0/htr_runtime/releases/v0.2.0-qualification.md) and [running guide](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR/blob/v0.2.0/docs/RUNNING.md).

## Recommended sampling update — 2026-11-03

The maintained launch commands now select temperature **1.0**, top-p **0.95**,
top-k **20**, min-p **0**, presence penalty **0** and repetition penalty **1**,
with thinking unchanged. This promotes the maintainer's preferred live response
profile. Explicit client values still override these server defaults.

A separately downloaded, checksum-pinned `launch_recommended.py` adds native
vLLM generation overrides to the identity-checked v0.2.0 launcher and retains all
runtime/checkpoint validation. The image, original launcher, integrity-bound
`generation_config.json` files, weights, tokenizer and immutable release tags
remain unchanged. The original entrypoint keeps its historical 0.7 / 0.8 defaults.
No new image, requantization, benchmark or Bf16Core GPU qualification is claimed.
Historical measurements and qualification receipts retain their original sampler
settings. Follow the maintained running instructions, not the frozen tag's commands,
to use the new recommendation.

## v0.3.0 — second update, planned

FP8 KV and the other deferred work: memory attribution, batched-token tuning, the bitwise-preserving short-prefill regression repair, and PLE I/O improvements. Scope and results remain subject to their own acceptance checks; these are plans, not shipped capability or measured gains.

Future entries will identify exact source commits, image digests, model revisions, measured serving changes and remaining limitations. Runtime-only updates will explicitly state when weights and quantization are unchanged. Published version tags will not be moved to newer code.
