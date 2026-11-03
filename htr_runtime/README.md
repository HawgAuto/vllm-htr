# vLLM HTR runtime

**HTR v0.2.0 — first serving update.** The existing public [v0.1.0 release](https://github.com/HawgAuto/vllm-htr/releases/tag/htr-v0.1.0) remains available; see [release history](../RELEASES.md).

This is the serving runtime for [Qwen3.8-Flash-Next-HTR](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR), based on the image-bound vLLM fork at `322c106f3409df1eeb29d2c14434b389b36c2b43`. It integrates HTR dense and routed-expert operators, GPTQ8 hyper-connection kernels, SSD-backed PLE reads, the TAU v2 QSA attention kernel, and the serving launcher.

The separate [HTR project](https://github.com/HawgAuto/HTR) documents the quantizer and format. This repository supplies their model-serving integration; it does not replace the upstream vLLM project.

## Get the prebuilt image

The versioned release image is distributed publicly. No GitHub account, token or `docker login` is required. The command below pins the tested Linux/amd64 platform manifest. The readable version alias is `ghcr.io/hawgauto/vllm-htr:v0.2.0`.

```sh
IMAGE=ghcr.io/hawgauto/vllm-htr@sha256:4c3bbfc5f42dad17c1a3bca2a911502aa0670aff1314e87e11bc900eaff491df
docker pull "$IMAGE"
```

Runtime executable source: [`ad52c7df6d461d653199264f6d5bc216984aea3b`](https://github.com/HawgAuto/vllm-htr/tree/ad52c7df6d461d653199264f6d5bc216984aea3b). The `htr-v0.2.0` tag adds release documentation without changing those executable bytes. Release tags are immutable. The image includes the native extensions; no host compiler or separate native-library download is needed. Continue to [Run](#run), or build from source instead. The older `v0.1.0` and `qsa1600-release-20261028` tags remain unchanged and use their own historical instructions.

## Build from source

This is an optional alternative to the prebuilt image. From the repository root:

```sh
docker build --pull=false -f htr_runtime/Dockerfile -t vllm-htr:local .
IMAGE=vllm-htr:local
```

The Dockerfile pins the existing ROCm/PyTorch/vLLM base by digest, compiles all eleven HTR/native extensions and the TAU v2 QSA code object from the included source, and installs the Python runtime. The build needs no GPU. It does not compile against host ROCm or copy private prebuilt libraries from the checkout. `NATIVE-BUILD.json` binds the selected source and Ninja recipes by SHA-256. Recipes retain the accepted math flags: HIPHC decode uses `-ffp-contract=fast`, while HIPHCP prefill uses `-ffp-contract=off`. The TAU v2 code object is built with the exact command recorded in `NATIVE-BUILD.json` and the build fails unless it matches the qualified binary hash. Native compiler work products remain in the build stage, not the serving image.

The generated image includes `native-build-receipt.json` and `installed-files.json`. The launcher checks the installed runtime and native-checkpoint metadata before starting vLLM. Those checks detect drift; they are not signatures against a malicious image publisher.

## Get the recommended sampling launcher

The current recommendation uses **temperature 1.0, top-p 0.95, top-k 20,
min-p 0, presence penalty 0 and repetition penalty 1**, with thinking still
enabled. This is the maintainer-selected response profile, not a vendor-default
claim or a new quality/performance benchmark.

Download this small launcher separately; existing model downloads need no changes:

```sh
LAUNCHER="$PWD/launch_htr_recommended.py"
curl --fail --location \
  https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR/resolve/main/launch_recommended.py \
  --output "$LAUNCHER"
printf '%s  %s\n' 'fcc27d10ef3ccb44c77f9a754f91dd65f22ea5033f2bec13fb4b9d316ac9789d' "$LAUNCHER" | sha256sum --check -
```

The launcher is mounted outside the image's integrity-bound runtime. It checks
that the original v0.2.0 launcher is unchanged, retains every runtime/checkpoint
validation, and adds native vLLM `--override-generation-config` defaults. Explicit
client sampling values still take precedence. It also works with `--cpu-check`,
`--print-command` and `--build-seed`; warmup requests keep their explicit greedy
settings. Thinking/template settings, MTP, context and KV are not changed.

**Compatibility:** the immutable image's original entrypoint and checkpoint
`generation_config.json` files retain temperature 0.7 / top-p 0.8. Do not edit those
files: the original launcher checks them. The commands below select the new
launcher, so their effective defaults are 1.0 / 0.95. Omitting that launcher gives
the historical defaults. No replacement image or prebuilt cache is shipped;
existing performance and builder/replay results retain their original settings.
New compilation remains possible; do not treat old timings as measurements of
this sampling update.

## Run

Requires Linux, an AMD gfx1151 GPU (Strix Halo), approximately 128 GB shared system memory, Docker with AMD device access, and fast local SSD storage. This profile is not for a 16/24 GB discrete GPU and is not qualified for other GPU architectures. Download the complete model directory, including `experts/`, `draft/`, and `artifact-metadata/`.

```sh
MODEL=/absolute/path/to/Qwen3.8-Flash-Next-HTR
WORKSPACE=/absolute/path/to/writable-htr-workspace
mkdir -p "$WORKSPACE"
# The image runs as UID 1000; this directory must be writable by that UID.
docker run --rm --init --pull=never --name vllm-htr \
  --device /dev/kfd --device /dev/dri/renderD128 \
  --group-add "$(stat -c %g /dev/kfd)" \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  --pids-limit 8192 \
  --shm-size 16g --ulimit core=0 \
  --security-opt no-new-privileges:true \
  -p 127.0.0.1:8000:8000 \
  --mount "type=bind,src=$MODEL,dst=/model,readonly" \
  --mount "type=bind,src=$WORKSPACE,dst=/workspace" \
  --mount "type=bind,src=$LAUNCHER,dst=/opt/launch_htr_recommended.py,readonly" \
  --entrypoint /opt/venv/bin/python3 \
  "$IMAGE" /opt/launch_htr_recommended.py
```

Resolve the correct AMD render node on multi-GPU hosts instead of assuming `renderD128`. No Docker memory/swap cap is set in this qualified launch. Do not co-load another large shared-memory workload; reported MemAvailable is an observation, not a reserved budget.

For a no-device model/runtime metadata preflight:

```sh
docker run --rm --network none --pull=never \
  --mount "type=bind,src=$MODEL,dst=/model,readonly" \
  --mount "type=bind,src=$WORKSPACE,dst=/workspace" \
  --mount "type=bind,src=$LAUNCHER,dst=/opt/launch_htr_recommended.py,readonly" \
  --entrypoint /opt/venv/bin/python3 \
  "$IMAGE" /opt/launch_htr_recommended.py --cpu-check
```

The API implements the OpenAI-compatible endpoints under `/v1`, with model name `qwen38-htr`. The host port is loopback-only; add an authenticated reverse proxy before any external exposure. The launcher's container bind defaults to `0.0.0.0`; `--host`, `--port`, and `--served-model-name` are supported overrides.

## Serving profile and levers

Defaults with the recommended sampling launcher:

- MTP speculative decoding with three probabilistic draft tokens.
- Thinking enabled, with the Qwen3 reasoning parser.
- Temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 0 and repetition penalty 1. This maintainer-selected profile overrides the frozen checkpoint defaults; client request values still take precedence.
- Context 262,144 tokens; BF16 KV pool 8 GiB; one sequence; maximum batched tokens 12,288.
- Prefix caching and `align` Mamba state caching retained.
- `FULL_AND_PIECEWISE` graph mode; capture sizes 1–5. No eager-mode substitute.

Levers (flags after `/opt/launch_htr_recommended.py` in the commands above):

| Flag | Default | Notes |
|---|---|---|
| `--mtp {0,3}` | 3 | 0 disables speculative decoding |
| `--max-num-seqs N` | 1 | up to 4 with MTP3, up to 32 with MTP0 |
| `--context N` | 262144 | syntax limit 1,048,576; above 262,144 static YaRN is applied; use measured profiles below, not this syntax limit as a fit guarantee |
| `--yarn-factor F` | auto | 2 up to 524,288, otherwise 4; measured long profiles explicitly pin 4 |
| `--kv-cache-gib G` | 8 | BF16 KV pool size |
| `--max-num-batched-tokens N` | 12288 | prefill chunk and scheduler budget |
| `--cudagraph-sizes a,b,...` | from seqs/MTP | decode graph capture sizes |
| `--no-prefix-caching` | off | |

`--print-command` shows the exact vLLM command, the cache profile and the levers without starting anything. These bounds are parser limits, not qualification of every combination; use the documented default or the measured profiles below, and requalify other geometries.

## Quick load

Each distinct set of compile-relevant levers (everything above except the KV pool size, host and port), runtime source and HTR version gets its own compile/autotune cache under `/workspace/cache/<profile>`. The first start with a new lever set compiles graphs and tunes kernels; later starts with the same levers load that cache. Compiler caches are permitted; they are not expanded model-weight caches. To do the first start ahead of time, add `--build-seed`: the launcher starts the server, sends warm-up requests across prefill lengths (64 tokens to 32K), stops the server, and writes `<cache>/SEED.json`. Keep `/workspace` on a persistent volume and reuse it. Repeat the same profile flags for the subsequent normal start, omitting only `--build-seed`. Fresh-process replay was checked through this public workflow; exact configurations and timings are below. New request shapes can still compile: quick load is not a zero-JIT promise. Do not reuse an older-version seed or copy graphs between profiles.

## Runtime features

- Dense HTR linears: fused GEMV for decode; for prefill each weight is dequantized to BF16 per call and multiplied with the vendor GEMM. No persistent BF16 weight copy is kept.
- QSA sparse attention (`htr_qsa`): the release split-K Triton kernel, or the native TAU v2 kernel for eligible 257–8192-row prefills. TAU v2 is bitwise equal to split-K; the first admitted calls of every process are re-run through split-K and must match bit for bit (`HTR_QSA_TAU=0` disables it).
- Autotune pin (`HTR_AUTOTUNE_PIN`): fixed Triton configs for the FLA prefill kernels and sticky Inductor deterministic mode. A source/signature-bound GDN norm/gate policy (`HTR_GDN_XBLOCK1`) selects XBLOCK=1 for MTP0 or XBLOCK=4 for MTP3, with two warps, before compilation and checks serialized native launchers before loading. The MTP3-only RMS policy preserves `R0_BLOCK=16384`, 32 warps and the reference native code object; it is absent in MTP0. Determinism alone does not choose reference-equivalent reduction geometry. Incompatible old seeds are rejected; regenerate them with this runtime. Concurrent request scheduling can still change batch composition and outputs.
- MoE numerical self-check reference scratch is temporary and released after the check; the check tolerance and ordinary fallback behavior are preserved.
- MoE partition4 (`HTR_MOE_PARTITION4`): 5–47-token MoE steps run on 4-row partitions with a parity-preserving plan; arithmetic unchanged.
- Routed-expert HPF3 prefill, fused decode combine, GPTQ8 hyper-connection kernels, device-side PLE gather, MTP draft filtering.

QSA attention follows the model's own top-k sparse selection; it is not full attention.

## Model packaging

The release contains 23 target safetensors shards, 13 indexed HTR expert shards, and one unchanged draft safetensors shard. Expert metadata sanitation removes worker/device/timing fields without changing retained metadata or tensor dtype, shape, or bytes. Shard consolidation is not requantization. The expert loader validates each serialized chunk against the pinned manifest before deserializing it.

The target is mixed precision: HTR5 dense projections, HTR4 output/routed-expert portions, GPTQ8 hyper-connection portions, and retained higher-precision tensors. A `bf16` compatibility selector in native-checkpoint metadata does not mean the entire model is BF16.

## Verification status

The final image passed18/18 complete old/new output comparisons for the memory-lifetime fix, including16 pairs with identical returned logprobs, plus the default-command text, reasoning and structured tool-call/result gate. The separately labeled predecessor passed six26-output fresh/cached/reference comparisons; those are preserved provenance, not a rerun on this final image. The API check used the image environment without diagnostic overrides, the selected temperature0.7/top-p0.8/top-k20 sampler, and thinking enabled. It returned a parsed `get_weather(city="Paris")` call, then used the supplied 18°C test fixture in the follow-up answer. This is a tool protocol test, not a live weather lookup.

The predecessor image sha256:0c31a8266bb71a7cc8698683c9c52ab86ca65c064f3ac0a625be835e698d1238 used context196608, 6GiB BF16 KV, seqs4, batched12288 and no YaRN. Ready times were310.743s fresh /125.464s cached for MTP0, and380.902s /145.362s for MTP3. Existing cache files were unchanged;8 and16 compiler files were added respectively, with no new tuning records. These times are not measurements of the default seqs1 profile or the separate public-seed profiles. The final-image default MTP3 API profile took631.785s to ready from an empty workspace.

See the [versioned qualification report](releases/v0.2.0-qualification.md), [machine-readable evidence](releases/v0.2.0-qualification.json) and [all84 original speed rows](releases/v0.2.0-speed-rows.csv). The unchanged63 checkpoint files remain hash-verified. A local source build has its own image identity; it does not inherit the prebuilt manifest digest.

## Measured BF16-KV / YaRN ×4 profiles

All four configurations below completed the actual public `--build-seed` workflow and an independent fresh-process cold replay on the released image. Each uses **context1,000,000, static YaRN4, a shared27GiB BF16 KV pool and batched-token budget12,288**. Graphs, prefix caching and Mamba `align` caching stay enabled.

| MTP | Slots | Build leg wall s | Cold ready s | Ready MemAvailable GiB | Short-request minimum GiB | Aggregate KV tokens | Equal-share estimate per slot |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1 | 501.478 | 170.424 | 11.941181 | 11.638920 | 1,020,123 | 1,000,000 |
| 0 | 1 | 429.832 | 170.737 | 14.624199 | 14.339275 | 1,126,738 | 1,000,000 |
| 3 | 4 | 553.750 | 200.821 | 8.963573 | 9.269318 | 1,020,123 | 255,030 |
| 0 | 4 | 486.647 | 170.436 | 13.517921 | 13.481205 | 1,126,738 | 281,684 |

Build-leg wall includes startup, the public warmups and teardown; cold-ready time ends at HTTP readiness. These are not prefill/decode throughput measurements. Equal-share values are allocation estimates capped by the per-request context limit, not tested filled histories. All10 short correctness requests passed; four-slot runs had4 overlapping requests.

These were **uncapped GPU-container launches**: no Docker `--memory`/`--memory-swap` ceiling or artificial free-memory veto. They still required exclusive GPU ownership, correct outputs and clean teardown. Startup/compilation used substantial swap. The tightest build reached2.185791GiB MemAvailable; readiness and short-request minima are different phases, not reserved headroom or a co-loading guarantee.

**Scope:** public warmups reached32,768 prompt tokens. A full-million-token prefill, long-position retrieval/quality and four simultaneous million-token histories were not tested. These are working configured profiles, not a measured hardware maximum.

### Choose a long-profile configuration

In the complete Docker launch command above, replace its final `"$IMAGE" /opt/launch_htr_recommended.py` line with the selected line below. Do not add the old108GiB container cap:

```sh
# MTP3, 1 slot(s)
"$IMAGE" /opt/launch_htr_recommended.py --mtp 3 --max-num-seqs 1 --context 1000000 --yarn-factor 4 --kv-cache-gib 27 --max-num-batched-tokens 12288
# MTP0, 1 slot(s)
"$IMAGE" /opt/launch_htr_recommended.py --mtp 0 --max-num-seqs 1 --context 1000000 --yarn-factor 4 --kv-cache-gib 27 --max-num-batched-tokens 12288
# MTP3, 4 slot(s)
"$IMAGE" /opt/launch_htr_recommended.py --mtp 3 --max-num-seqs 4 --context 1000000 --yarn-factor 4 --kv-cache-gib 27 --max-num-batched-tokens 12288
# MTP0, 4 slot(s)
"$IMAGE" /opt/launch_htr_recommended.py --mtp 0 --max-num-seqs 4 --context 1000000 --yarn-factor 4 --kv-cache-gib 27 --max-num-batched-tokens 12288
```

Append `--build-seed` to the chosen line for its one-time public warmup pass, then start again without that flag using the same writable workspace. Profiles are independently namespaced. Changing a compile-relevant lever selects another cache.

| MTP | Slots | Replay-added compiler files | Of those, new HSACOs | Existing files changed/removed | Tuning files added/changed |
|---|---:|---:|---:|---:|---:|
| 3 | 1 | 40 | 5 | 0 | 0 |
| 0 | 1 | 16 | 2 | 0 | 0 |
| 3 | 4 | 48 | 6 | 0 | 0 |
| 0 | 4 | 40 | 5 | 0 | 0 |

Each replay directly loaded AOT and preserved every existing seed entry. New shape-specific JIT is allowed and disclosed; this is not compilation-free startup. No downloaded prebuilt seed blob is shipped. Use the public builder on your own workspace.


QSA1600 is approximate sparse attention, not parity with full attention. Solo reference identity is scoped to the original qualification roster; concurrent scheduling can change outputs. These checks do not establish broad model-quality, image/video or agent-compatibility qualification; full-million prefill/retrieval was not run. The known short-prefill overhead and PLE/I/O tails are not claimed fixed. FP8 KV, memory attribution and further tuning remain outside v0.2.0.

Run the installed CPU suite without exposing devices:

```sh
docker run --rm --network none --pull=never \
  --entrypoint /opt/venv/bin/python3 \
  -e PYTHONPATH=/opt/vllm-htr/python:/opt/vllm-htr \
  -e HTR_MOE2_SO=/opt/vllm-htr/native-bin/htr_moe2.so \
  -e HTR_HPF3_SO=/opt/vllm-htr/native-bin/htr_moe_prefill_v3.so \
  "$IMAGE" -m unittest discover -s /opt/vllm-htr/tests -v
```

Upstream notices and licenses are retained. See HTR's acknowledgments for the exlr8 → ExLlamaV3/EXL3 → QTIP lineage and QuIP/GPTQ/LDLQ foundations; see the upstream vLLM documentation for the engine itself.
