# Native HTR serving — v0.2.2

The image's entry point is `vllm serve`. HTR implementation defaults are installed
in the runtime; a special launcher is not required to enable the optimization
stack. Historical `launch_htr.py`, `launch_recommended.py`, and
`launch_quantized.py` are preserved as predecessor evidence, not the native
configuration API. Do not use their frozen profile restrictions to determine
what the native runtime supports.

## User-controlled options

Use ordinary vLLM flags or a vLLM YAML configuration file:

- `--max-num-batched-tokens`: chunked prefill / scheduler token budget.
- `--max-num-seqs`: concurrent sequences.
- `--max-model-len`: maximum context.
- `--kv-cache-memory-bytes`: KV allocation budget.
- `--kv-cache-dtype`: `auto` (BF16 for Htr5Core), `int8_per_token_head`,
  `int4_per_token_head`, `octave_k3v4`, `octave_k3v3`, or
  `octave_k3v3_compact`.
- `--speculative-config null`: MTP off. The inherited default is MTP3;
  other draft counts are not implemented by the numerical tile policy.
- `--generation-config` and `--override-generation-config`: server sampling
  defaults. Requests may supply their own temperature, top-p, top-k, min-p,
  penalties, seed, and other native sampling options. No preferred profile is
  forcibly injected and checkpoint sampling values are not an integrity gate.
- `--compilation-config`, prefix-caching and other native flags remain native
  options; explicit values take precedence over defaults.

PLE hot-row cache is an optional CPU-RAM cache, disabled by default. Set
`HTR_PLE_ROW_CACHE_BYTES` in the container environment to choose the byte budget
(including metadata); nonzero budgets must be at least 65536 bytes. The packaged
cache library is selected before model/plugin imports. Set
`HTR_PLE_ROW_CACHE_STATS=/workspace/ple-cache-stats.bin` for its statistics output.
Memory availability still constrains what fits.

The runtime derives KV kernel selection, MTP numerical tile mode, and automatic
YaRN from the effective native settings. A tested-envelope advisory is not a
refusal and is not proof that an arbitrary configuration fits or has passed
quality/performance testing. Invalid configuration or incompatible numerical
modes are still rejected.

## Optimized draft filter and source build

The exact HIP draft filter is enabled by default (`HTR_Q38F=on`) through the
native plugin allowlist. Set `HTR_Q38F=off` to use the original Torch filter.
Its qualified fast path is gfx1151, the pinned Torch/HIP build, vocabulary
248320, KC64, BF16/FP32 logits and one through eight rows. Unsupported shapes
fall back to the original operator; MTP0 does not call the draft filter.
`HTR_Q38F=verify` is an explicit diagnostic shadow mode, not a performance mode.

The image is source-built over the pinned v0.2.1 runtime, including real gfx1151
Octave kernels, the bounded PLE cache, and a byte-identical build of the accepted
filter. Compilation and model-weight storage are different: retaining compiled
programs does not create persistent unpacked model-weight copies.

## Start the combined image

Use Linux, AMD gfx1151,128GB unified memory and a fast local SSD. Htr5Core is the qualified model; Bf16Core remains a separate research reference. The image runs as UID1000: the workspace must be writable by that UID. Use absolute model/workspace paths and the AMD render node if your device numbering differs. Keep one large shared-memory GPU workload active at a time.

Pull the exact published image first:

```sh
docker pull ghcr.io/hawgauto/vllm-htr@sha256:034898ebd182a240fa403ad5ea24bea3f647d933571e43420a27daf9af079d1c
```

Then run the following in its own shell (or save it as a shell script). `MODEL` and `WORKSPACE` can be supplied as host environment variables; they are bind-mount paths, not container environment overrides.

```bash
#!/usr/bin/env bash
set -euo pipefail
IMAGE=ghcr.io/hawgauto/vllm-htr@sha256:034898ebd182a240fa403ad5ea24bea3f647d933571e43420a27daf9af079d1c
MODEL=${MODEL:-"$PWD/Qwen3.8-Flash-Next-HTR/Htr5Core"}
WORKSPACE=${WORKSPACE:-"$PWD/htr-cache/Htr5Core-v022"}
mkdir -p "$WORKSPACE/tmp"
# The first start compiles a module and needs $WORKSPACE/tmp to exist; this pinned v0.2.2 image does not create the directory in time.
exec docker run --rm --init --pull=never --name vllm-htr \
  --device /dev/kfd --device /dev/dri/renderD128 \
  --group-add "$(stat -c %g /dev/kfd)" \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  --pids-limit 8192 --shm-size 16g --ulimit core=0 \
  --security-opt no-new-privileges:true \
  -p 127.0.0.1:8000:8000 \
  --mount "type=bind,src=$MODEL,dst=/model,readonly" \
  --mount "type=bind,src=$WORKSPACE,dst=/workspace" \
  "$IMAGE" /model --host 0.0.0.0 --port 8000 \
  --max-model-len 32768 --max-num-seqs 1 \
  --max-num-batched-tokens 2048 --kv-cache-memory-bytes 2147483648
```

This command uses BF16 KV, MTP3 and the checkpoint's sampling defaults. Add
`--speculative-config null` for MTP0, or choose any listed KV dtype. Use native
`--config your-config.yaml` for saved options. To enable a 16 MiB PLE cache,
place `-e HTR_PLE_ROW_CACHE_BYTES=16777216` before the image name. Warm the
selected configuration once and reuse its workspace; new shapes may compile.
Only completed qualification receipts establish the tested envelope.

## Configuration reuse / quickload

There is no new quickload subsystem. Save the native arguments or YAML you want,
keep `/workspace` on persistent storage, and warm/use that configuration as in the
model-card workflow. Native default compile-cache paths are separated by the
configuration/runtime identity. Explicit per-library cache roots are respected;
the underlying libraries also validate their artifact keys. Do not distribute
an old seed as qualified for a new image without verifying it in that image.

Example arguments (append after the image name in your normal Docker
command, with model/workspace mounts and ROCm device access):

```
/model --max-model-len 32768 --max-num-seqs 1 \
  --max-num-batched-tokens 2048 --kv-cache-memory-bytes 2147483648 \
  --kv-cache-dtype octave_k3v4 --speculative-config null \
  --override-generation-config '{"temperature":0.7,"top_p":0.9,"top_k":40}'
```

This is a configuration example, not a claim of quality, capacity, or speed for
all settings. Octave is an optional smaller-KV format; matching BF16 speed is not
its acceptance criterion. The published release does not activate a service on your machine. Historical measurements and qualification limits remain separate from user-configurable settings.

## Connect and choose sampling

The API is bound to `http://127.0.0.1:8000/v1`. The native policy serves model ID `qwen38-htr`; discover the actual ID from `/v1/models` if you set `--served-model-name`.

```sh
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-htr","messages":[{"role":"user","content":"Explain how speculative decoding works."}],"max_tokens":512}'
```

Use OpenAI-compatible `tools` normally. Thinking consumes the output-token budget. Add authentication and a reverse proxy before exposing the service beyond loopback.

### Sampling for existing clients

Omitted fields inherit the checkpoint defaults (temperature0.7, top-p0.8, top-k20 for the shipped Htr5Core metadata). Explicit request values take precedence. To retain the previous maintainer-selected profile, add the following native server argument or send its fields in each request:

```sh
--override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}'
```

This is an optional sampling choice, not a checkpoint rewrite or a new quality result. With the OpenAI Python SDK, use `extra_body` for top_k, min_p and repetition_penalty. No change to thinking/template settings is implied.

### Cache migration and historical launchers

For v0.2.2, retain a dedicated workspace, start the desired native configuration, exercise your workload and reuse that workspace on subsequent starts. New shapes can compile. The predecessor's `--build-seed`, `--cpu-check`, `--mtp`, `--context` and `--kv-cache-gib` are launcher flags, not flags for the new native entrypoint. Do not append them to this command. Historical v0.2.0/v0.2.1 builder and million-context recipes remain pinned to their original images; do not reinterpret them as new v0.2.2 capacity measurements.

## Build the public runtime source

The prebuilt image needs no host compiler. For a source build, use Python3.11+ and Docker:

```sh
git clone https://github.com/HawgAuto/vllm-htr.git
cd vllm-htr
git checkout 9a4917f41101926c8cb23d630018b8e5e4aec44c
python3 htr_runtime/build_combined.py --output "$PWD/../htr-v022-build" \
  --tag vllm-htr:v0.2.2
```

The builder uses the digest-pinned v0.2.1 base, verifies its pinned Triton source snapshot, compiles the gfx1151 core without GPU access, and builds the PLE/filter/native-policy payload. It does not replace the installed Triton runtime. Its output must be outside a clean checkout. A rebuilt image has its own identity and is not automatically the published image's qualification.
