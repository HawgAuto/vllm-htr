# vLLM HTR release history

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

## v0.2.0 — first update, in development

The work currently underway: clean integrated serving stack, accepted at-call dense-prefill work, BF16 context/capacity qualification, and configurable quick-load tooling. Only qualified results will be added to the published release notes. These changes are not part of v0.1.0.

## v0.3.0 — second update, planned

FP8 KV and the other deferred work: memory attribution, batched-token tuning, the bitwise-preserving short-prefill regression repair, and PLE I/O improvements. Scope and results remain subject to their own acceptance checks; these are plans, not shipped capability or measured gains.

Future entries will identify exact source commits, image digests, model revisions, measured serving changes and remaining limitations. Runtime-only updates will explicitly state when weights and quantization are unchanged. Published version tags will not be moved to newer code.
