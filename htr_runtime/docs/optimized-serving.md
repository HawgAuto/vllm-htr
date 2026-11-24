# HTR v0.2.5 optimized serving

This serving update keeps the v0.2.2 engine, model weights, KV formats and native
vLLM interface. It makes the retained concurrency stack the runtime default:

- K6 routed-MoE decode at 2–4 rows, outside the unchanged partition/grouped startup guards.
- Dense WMMA widened to eligible 9–15-row projections.
- Async PLE row limit zero now disables the path instead of allocating a zero-byte buffer; the self-check wait is bounded.
- Grouped expert-major MoE planning at its admitted row counts, using the
  existing native kernels and per-layer bitwise startup checks.
- Selected K5/BF16 dense WMMA projections at the measured shapes and row counts.
  Unsupported shapes retain the original route; this is not a claim that the
  WMMA arithmetic is bitwise identical to every predecessor dense path.
- Asynchronous PLE gathering through 32 rows without speculation, or 16 with
  MTP3. The setting follows the parsed native speculative configuration.
- Repaired quantized-KV binding manifests. The v0.2.2 compatibility launcher
  could reject INT8/INT4 startup because its QSA hashes were stale, even though
  the installed source agreed with both other manifests. The repair changes
  metadata, not quantization or KV arithmetic, and retains every integrity check.

The PLE hot-row cache stays off by default. Rejected grouped dense-output work,
unfinished concurrency changes and persistent unpacked weight shadows are not
part of this release. v0.2.4 improves single-stream decode in the selected-stack
screens; 2–4-stream changes vary by workload, including a −5.60% steady two-stream
result. No multi-stream or prefill win is established. See the [versioned
release evidence](../releases/v0.2.5/README.md) for exact conditions and the known
seed42 looping limitation. The historical throughput targets remain unmet.

## Native controls

Use ordinary `vllm serve`, or the image's equivalent default entry point. Existing
sampling, context, KV budget, concurrency and speculation flags remain native.
Pull the qualified image first (no registry login required):

```sh
docker pull ghcr.io/hawgauto/vllm-htr@sha256:daa6b2847bf9678e2de39b8b216105ffe13313e50db889f002f5f918e581d193
```

Then start it:

```bash
#!/usr/bin/env bash
set -euo pipefail
IMAGE=ghcr.io/hawgauto/vllm-htr@sha256:daa6b2847bf9678e2de39b8b216105ffe13313e50db889f002f5f918e581d193
MODEL=${MODEL:-"$PWD/Qwen3.8-Flash-Next-HTR/Htr5Core"}
WORKSPACE=${WORKSPACE:-"$PWD/htr-cache/Htr5Core-v025"}
mkdir -p "$WORKSPACE"
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

Use absolute MODEL and WORKSPACE paths; WORKSPACE must be writable by UID1000.
The inherited default is MTP3 and BF16 KV. Append `--speculative-config null`
for MTP0. Select `int8_per_token_head`, `int4_per_token_head`, `octave_k3v4`,
`octave_k3v3`, or `octave_k3v3_compact` with `--kv-cache-dtype`.
Do not co-load another large UMA/GPU workload. Keep the workspace for ordinary
compiler-cache reuse; the first start and newly encountered shapes may compile.

Optional environment overrides go before the image name:

- `-e HTR_MOE_K6=0`: disable the new 2–4-row K6 route.
- `-e HTR_DENSE_WMMA_MIN9=0`: restore the 16-row dense WMMA floor.
- `-e HTR_MOE_GROUPED=0`: disable grouped MoE (independent of K6).
- `-e HTR_DENSE_WMMA=0`: retain the predecessor dense route.
- `-e HTR_PLE_ASYNC_MAXN=4`: explicitly choose the asynchronous PLE row limit.
  Zero disables this asynchronous row path. The legacy E59_PLE_DEV_MAXN override
  remains recognized; the new name wins if both are supplied.
- `-e HTR_PLE_ROW_CACHE_BYTES=...`: opt into the existing bounded hot-row cache.

These are runtime controls, not quality/maximum-context guarantees. The release
qualification records specify the exact exercised geometries; other native
settings remain user-controlled rather than being refused for exceeding a test
boundary. The legacy `launch_quantized.py` interface remains a compatibility path
with its historical presets, not the recommended way to configure concurrency.

## Source build

```sh
git clone https://github.com/HawgAuto/vllm-htr.git
cd vllm-htr
git checkout 62cb02413de66042cb11c6dc853922618c368887
python3 htr_runtime/build_optimized.py \
  --output /absolute/path/outside-checkout/htr-build \
  --tag vllm-htr:v0.2.4
```

The builder requires a clean committed tree, the pinned public v0.2.2 base and
Docker. It rebuilds the selected dense and K6 extensions from source using their
bound Ninja recipes and requires byte identity with both qualified native artifacts.
The empty-workspace cold-start fix is included.
All other native libraries are inherited unchanged from the pinned base. It
checks source/native/base/KV manifest consistency before admitting the image.
No host ROCm compiler or GPU access is needed for the build. The historical
`build_combined.py` CLI also forwards to this version's builder.

Tests: `htr_runtime/tests/run_optimized.py` in the pinned image environment;
`htr_runtime/tests/installed/test_release_bindings.py` against installed files.
Historical tests are retained; the additive runner names superseded old-version
assertions explicitly. Build success and CPU checks are not real-model serving
qualification. Release receipts provide that separate evidence.
