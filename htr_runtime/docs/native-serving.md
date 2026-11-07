# Native HTR serving (harvest candidate — qualification in progress)

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
- `--kv-cache-dtype`: `auto`/`bfloat16`, `int8_per_token_head`,
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

```sh
IMAGE=vllm-htr:v0.2.2
MODEL="$PWD/Qwen3.8-Flash-Next-HTR/Htr5Core"
WORKSPACE="$PWD/htr-cache/Htr5Core-v022"
mkdir -p "$WORKSPACE"
docker run --rm --init --pull=never --name vllm-htr \
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

Example arguments (append after the candidate image name in your normal Docker
command, with model/workspace mounts and ROCm device access):

```
/model --max-model-len 32768 --max-num-seqs 1 \
  --max-num-batched-tokens 2048 --kv-cache-memory-bytes 2147483648 \
  --kv-cache-dtype octave_k3v4 --speculative-config null \
  --override-generation-config '{"temperature":0.7,"top_p":0.9,"top_k":40}'
```

This is a configuration example, not a claim of quality, capacity, or speed for
all settings. Octave is an optional smaller-KV format; matching BF16 speed is not
its acceptance criterion. Public tags and production are unchanged during
qualification.
