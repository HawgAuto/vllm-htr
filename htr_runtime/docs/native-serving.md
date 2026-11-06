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
