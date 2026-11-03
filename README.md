# vLLM HTR

**HTR v0.2.0 — the first serving update for Qwen3.8-Flash-Next-HTR on AMD Strix Halo.**

[Pull or build and run](htr_runtime/README.md) · [Model and download](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR) · [Release notes](https://github.com/HawgAuto/vllm-htr/releases/tag/htr-v0.2.0) · [Release history](RELEASES.md)

This fork integrates the custom HTR weight format with vLLM's serving engine on gfx1151: packed dense/expert kernels, GPTQ8 hyperconnections, SSD-backed PLE embeddings, sparse attention and MTP3. It is not a stock-vLLM installation or a new model quantization.

## Recommended response profile

Use **temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 0 and repetition penalty 1**, with thinking unchanged. The [recommended sampling launcher](htr_runtime/launch_recommended.py) applies these defaults to the existing v0.2.0 image while retaining its integrity checks. Follow the [updated run commands](htr_runtime/README.md#get-the-recommended-sampling-launcher). The original image entrypoint/checkpoint defaults remain 0.7 / 0.8 for compatibility; weights, image digest, historical tests and immutable tags are unchanged. This is a maintainer preference, not a new quality or speed claim.

## Get the tested image

```sh
docker pull ghcr.io/hawgauto/vllm-htr@sha256:4c3bbfc5f42dad17c1a3bca2a911502aa0670aff1314e87e11bc900eaff491df
```

The equivalent version tag is `ghcr.io/hawgauto/vllm-htr:v0.2.0`. Public access requires no GitHub account, token or `docker login`. Use the [runtime guide](htr_runtime/README.md) for model/workspace mounts, AMD device access, the loopback API and optional source builds. The intended host is Linux/gfx1151 with approximately128GB shared memory and fast local SSD storage.

## What this update delivers

- TAU v2 attention on eligible prefill shapes, with bitwise reference checks and split-K fallback.
- Partition4 dispatch for selected small MoE batches, preserving the accepted arithmetic.
- Dense-prefill dequantization per call instead of a persistent expanded BF16 shadow.
- Temporary MoE self-check reference scratch, recovering approximately4.6875GiB at readiness in matched tests without removing numerical checks.
- Mode-specific reference GDN geometry and the reference MTP RMS reduction, including serialized-code validation.
- Source/version/profile-bound compile caches and a public `--build-seed` workflow.
- Four configured-million profiles: MTP0 or MTP3, one or four slots, YaRN4 and a shared27GiB BF16 KV pool.

All four seed-build/cold-replay workflows passed. Full-million prefill and long-position quality were not tested; four slots share the pool, not four million-token allocations. New shapes can still compile. The final documented default command passed text, reasoning and a structured tool-call/result round trip.

[Detailed qualification, memory observations and preserved predecessor speed rows](htr_runtime/releases/v0.2.0-qualification.md) · [Machine-readable evidence](htr_runtime/releases/v0.2.0-qualification.json)

Checkpoint weights and quantization are unchanged. v0.1.0 source/image/model tags remain immutable. FP8 KV, the short-prefill speed repair and PLE/I/O work are deferred to v0.3.0; this release does not claim those gains or broad semantic/multimodal quality.

## Source and lineage

Executable HTR source: [`ad52c7df6d461d653199264f6d5bc216984aea3b`](https://github.com/HawgAuto/vllm-htr/tree/ad52c7df6d461d653199264f6d5bc216984aea3b). The release tag adds documentation and evidence without changing executable bytes. This repository preserves [upstream vLLM](https://github.com/vllm-project/vllm) ancestry; [vLLM documentation](https://docs.vllm.ai/) describes the underlying engine, while the linked HTR guide supplies this model's required runtime.

[HTR format and quantizer](https://github.com/HawgAuto/HTR) · [Methodology](https://github.com/HawgAuto/HTR/blob/main/docs/METHODOLOGY.md) · [Acknowledgments and exlr8 → ExLlamaV3/EXL3 → QTIP lineage](https://github.com/HawgAuto/HTR/blob/main/docs/LINEAGE.md). Upstream licenses and notices are retained. Model weights have their own Qwen Community License; see the model card.
