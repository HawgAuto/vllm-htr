# vLLM HTR runtime — v0.2.3

[Model and download](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR) · [Native serving guide](docs/native-serving.md) · [Older BF16 compatibility guide](docs/RUNNING-v0.2.0.md)

## v0.2.3: selected concurrency stack and KV startup repair

Grouped MoE planning, selected dense WMMA and mode-dependent asynchronous PLE
are runtime defaults. The quantized-KV compatibility manifest is repaired;
weights and sampling defaults remain unchanged. Explicit user overrides win.
[Current launch and source-build guide](docs/optimized-serving.md).

The sections below preserve v0.2.2 and older release history and measurements;
they are not new v0.2.3 performance or capacity claims.

## v0.2.2: combined native runtime

This successor keeps the accepted sampler/draft improvements, wave32 GDN and
bit-exact QSA preparation. It adds all three optional Octave KV formats with
native gfx1151 support and ordered window compaction, optional bounded PLE
hot-row caching, and the accepted exact HIP draft filter (enabled by default).
No model weights change. [Completed v0.2.2 qualification](releases/v0.2.2/README.md) records the bounded matrix and delivery evidence; the historical results below belong to v0.2.1 and are not new measurements.

The entrypoint is now ordinary `vllm serve`: pass `/model` and native vLLM
arguments, not the historical launcher-only `--mtp` or `--context` aliases.
Sampling, KV budget/format, concurrency, prefill budget and context are user
controls. Keep a persistent workspace for compile-cache reuse. See the
[native guide](docs/native-serving.md) for copyable commands, configuration
files, PLE controls and the HIP off-switch.

Build from a clean committed checkout using the pinned v0.2.1 base image:

```sh
python htr_runtime/build_combined.py --output "$PWD/../htr-v022-build" \
  --tag vllm-htr:v0.2.2
```

The builder fetches only its pinned Triton source dependency if a verified local
`--triton-kernels` directory is not supplied, exports the exact Git tree, builds
the native gfx1151 core without GPU access, source-builds the PLE/filter payload,
and binds the resulting image. Run the CPU suite with the checkout mounted at
`/source`: `/opt/venv/bin/python3 /source/htr_runtime/tests/run_native.py`.
The dependency snapshot is used only by CMake; the parent runtime's installed
Triton package is not replaced. Old regression files remain byte-preserved,
with explicit successor tests for intentional version/plugin-policy changes.

## Historical v0.2.1 documentation and measurements

# v0.2.1 — 2M, 3M and 4M context profiles

Use the recommended **Htr5Core** checkpoint on Linux / AMD Strix Halo gfx1151 with 128 GB unified memory and fast SSD storage. No requantization or new weight download is required.

## Qualified profiles

| KV cache | MTP | Max context | YaRN | KV pool |
|---|---:|---:|---:|---:|
| Scaled 8-bit per-token/head KV (`int8_per_token_head`) | 0 | 2,000,000 | ×8 | 27.75 GiB |
| Scaled 8-bit per-token/head KV (`int8_per_token_head`) | 3 | 2,000,000 | ×8 | 27.75 GiB |
| Rotated 4-bit per-token/head KV (`int4_per_token_head`) | 0 | 4,000,000 | ×16 | 26.875 GiB |
| Rotated 4-bit per-token/head KV (`int4_per_token_head`) | 3 | 3,000,000 | ×16 | 22.375 GiB |

All four use one sequence and `--max-num-batched-tokens 2048`. KV pool is the total allocation, including the unchanged recurrent/indexer bookkeeping—not extra memory on top of another KV pool. Leave host memory for model weights, runtime, compilation and the OS. These profiles passed allocation, API readiness and generation; they are capacity-qualified profiles, not filled-window benchmarks. The retained BF16 defaults and concurrency options remain available separately.

## Correct KV names

The actual vLLM names are `int8_per_token_head` and `int4_per_token_head`. The latter uses a randomized Hadamard rotation (RHT), scalar 4-bit quantization, packed nibbles and per-token/head scale/zero-point metadata. We call it **rotated 4-bit per-token/head KV**. It is not “UltraQuant” or a TurboQuant implementation. Only main attention K/V change precision; sparse-attention indexer and recurrent state keep their existing precision. The retained fused D256 rotation performs the same ordered FP32 butterflies as its reference.

## Measured accuracy retention versus BF16 KV

| KV cache | MTP | BF16 prediction agreement | Prediction disagreement | Tie-aware agreement |
|---|---:|---:|---:|---:|
| Scaled 8-bit | 0 | **98.18%** (486/495) | **1.82%** | 99.80% |
| Rotated 4-bit | 0 | **98.18%** (486/495) | **1.82%** | 99.39% |
| Scaled 8-bit | 3 | **98.79%** (489/495) | **1.21%** | 99.80% |
| Rotated 4-bit | 3 | **98.38%** (487/495) | **1.62%** | 99.60% |

Each row compares 495 teacher-forced prompt positions with the same model's BF16-KV reference, separately for MTP0 and MTP3. Strict agreement requires the complete highest-probability token set to match; tie-aware agreement accepts a shared tied top token. “Disagreement” is the measured prediction change, not a general task-accuracy loss. The six functional fixtures passed in every compared cell. [Exact counts, logprob deltas and source binding](releases/v0.2.1-qualification.json).

## Run a profile

```sh
IMAGE=ghcr.io/hawgauto/vllm-htr@sha256:dde8c3ea80e5d0a56e9143894a12622d0c236bab1f6d3d8a10e4ab911c836afb
docker pull "$IMAGE"
MODEL="$PWD/Qwen3.8-Flash-Next-HTR/Htr5Core"
WORKSPACE="$PWD/htr-cache/Htr5Core-v021"
mkdir -p "$WORKSPACE"
htr() {
  docker run --rm --init --pull=never --name vllm-htr \
    --device /dev/kfd --device /dev/dri/renderD128 \
    --group-add "$(stat -c %g /dev/kfd)" \
    --group-add "$(stat -c %g /dev/dri/renderD128)" \
    --pids-limit 8192 --shm-size 16g --ulimit core=0 \
    --security-opt no-new-privileges:true \
    -p 127.0.0.1:8000:8000 \
    --mount "type=bind,src=$MODEL,dst=/model,readonly" \
    --mount "type=bind,src=$WORKSPACE,dst=/workspace" \
    "$IMAGE" "$@"
}

# 2M, scaled 8-bit KV, MTP3 (use --mtp 0 for MTP off)
htr --kv-cache-dtype int8_per_token_head --mtp 3 \
  --context 2000000 --yarn-factor 8 --kv-cache-gib 27.75 \
  --max-num-seqs 1 --max-num-batched-tokens 2048

# 3M, rotated 4-bit KV, MTP3
htr --kv-cache-dtype int4_per_token_head --mtp 3 \
  --context 3000000 --yarn-factor 16 --kv-cache-gib 22.375 \
  --max-num-seqs 1 --max-num-batched-tokens 2048

# 4M, rotated 4-bit KV, MTP off
htr --kv-cache-dtype int4_per_token_head --mtp 0 \
  --context 4000000 --yarn-factor 16 --kv-cache-gib 26.875 \
  --max-num-seqs 1 --max-num-batched-tokens 2048
```

Choose one launch at a time. Add `--cpu-check` for checkpoint/runtime preflight, `--print-command` to inspect the native arguments, or `--build-seed` to build that profile's local compile cache before normal serving. Retain the workspace; dtype, context, YaRN, mode and source changes separate cache profiles. First use of a new shape can still compile. The image entrypoint includes the preferred sampling profile: temperature1.0, top-p0.95, top-k20, min-p0, presence penalty0 and repetition penalty1; thinking remains enabled, and explicit client settings override these defaults.

## Build from source

```sh
git clone https://github.com/HawgAuto/vllm-htr.git
cd vllm-htr
git checkout htr-v0.2.1
docker build --provenance=false \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -f htr_runtime/Dockerfile.v0.2.1 -t vllm-htr:v0.2.1 .
```

The incremental recipe starts from the pinned public v0.2.0 image. It rebuilds only the two small TAU KV readers, with fixed HIP compilation-unit identifiers for reproducible object hashes, and retains all unrelated runtime bindings. Numerical source, native selfchecks and inherited notices remain intact. This is a serving-distribution version, not the upstream vLLM package version.

CPU/source regression entrypoint in an installed image: `/opt/venv/bin/python3 /source/htr_runtime/tests/run_v021.py` with this checkout mounted at `/source`. The old v0.2.0 version-literal test is retained byte-for-byte and explicitly skipped by this versioned runner; its v0.2.1 replacement is tested. Private campaign-invariance and saved-tensor fixtures are not part of this portable test payload.
