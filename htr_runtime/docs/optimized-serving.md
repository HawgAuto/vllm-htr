# HTR v0.2.6 optimized serving

This serving update keeps the model weights, tokenizer, KV formats, and native
`vllm serve` interface unchanged. All **v0.2.6** optimizations are enabled by
default via `runtime-env.json` with zero extra `-e` flags required:

- **Zero-Triton-JIT hot path (`20` new zero-spill AOT `gfx1151` Wave32 HIP kernels):**
  - Full-INT8 WMMA QSA prefill (`qsa_tau_v5_kv8_i8`), delivering **2.06×** faster
    13K warm prefill vs `v0.2.5` (`1,137.7 tok/s` / `11.44 s`).
  - 128-bit vectorized AOT `causal_conv1d` (with strided-1D `cache_indices` fix)
    and AOT GDN chunk prefill (`chunk_h`, `chunk_o`, `wy`, `post_conv`, `norm_gate`).
  - Double-buffered, coalesced AOT W4A16 MoE (`moe_w4a16_wmma_aot`), HPF5 v2 MoE
    prefill, AOT dense prefill v2, HyperConnection v2, QSA MQA v3, QSA cache/prepare/indexer,
    PLE transpose, and Gumbel sampler.
- **Wave 7 RDNA 3.5 (`gfx1151`) ISA decode upgrades (`3` upgraded `.so` extensions):**
  - `htr_bf16_gemv.so` (unlocked hidden `v_dual_dot2acc_f32_bf16` + `v_perm_b32`),
    `htr_moe_k6_shx.so`, and `htr_kernels.so` (bank-staggered VOPD pairing).
- **Inherited v0.2.5 / v0.2.4 / v0.2.3 stack:** K6 routed-MoE `rows=1..16` with
  zero-realloc shared-expert folding, K4 `lm_head`, dense WMMA (`rows>=9`),
  fused GDN MTP spec decode, native QSA decode union (BF16 / INT8 / INT4 KV),
  `ple_gather_v25`, native Q38 draft filter, and async PLE defaults.

See the [v0.2.6 release notes](../releases/v0.2.6/README.md) for exact verified
zero-overlay E2E results and INT8 WMMA parity data.

## Native controls

Use ordinary `vllm serve`, or the image's equivalent default entry point. Existing
sampling, context, KV budget, concurrency, and speculation flags remain native.
Pull the qualified image first (no registry login required):

```sh
docker pull ghcr.io/hawgauto/vllm-htr@sha256:dc11916549c534b207bb45048fe2f9b2177117995931f5aff5b384b274ea0fe8
```

Then start it:

```bash
#!/usr/bin/env bash
set -euo pipefail
IMAGE=ghcr.io/hawgauto/vllm-htr@sha256:dc11916549c534b207bb45048fe2f9b2177117995931f5aff5b384b274ea0fe8
MODEL=${MODEL:-"$PWD/Qwen3.8-Flash-Next-HTR/Htr5Core"}
WORKSPACE=${WORKSPACE:-"$PWD/htr-cache/Htr5Core-v026"}
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

Use absolute `MODEL` and `WORKSPACE` paths; `WORKSPACE` must be writable by UID 1000.
The inherited default is MTP3 and BF16 KV. Append `--speculative-config null`
for MTP0. Select `int8_per_token_head`, `int4_per_token_head`, `octave_k3v4`,
`octave_k3v3`, or `octave_k3v3_compact` with `--kv-cache-dtype` (when using
`int8_per_token_head`, also pass `-e HTR_KV8_ENABLE=1` to activate the full INT8
QSA prefill + decode union stack).
Do not co-load another large UMA/GPU workload. Keep the workspace for ordinary
compiler-cache reuse across restarts.

Optional environment overrides go before the image name:

- `-e HTR_QSA_TAU_V5_I8=0`: switch QSA prefill from full-INT8 WMMA (`v5_kv8_i8`)
  back to BF16 WMMA (`v4_kv8`).
- `-e HTR_KV8_ENABLE=1`: enable the native INT8 KV QSA prefill/decode fast path
  when serving with `--kv-cache-dtype int8_per_token_head`.
- `-e HTR_MOE_K6=0`: disable the K6 routed-MoE path.
- `-e HTR_DENSE_WMMA_MIN9=0`: restore the 16-row dense WMMA floor.
- `-e HTR_MOE_GROUPED=0`: disable grouped MoE (independent of K6).
- `-e HTR_DENSE_WMMA=0`: retain the predecessor dense route.
- `-e HTR_PLE_ASYNC_MAXN=4`: explicitly choose the asynchronous PLE row limit
  (`0` disables the async row path).
- `-e HTR_PLE_ROW_CACHE_BYTES=...`: opt into the bounded CPU-RAM PLE hot-row cache.

## Source build

`v0.2.6` builds cleanly on top of the pinned `ghcr.io/hawgauto/vllm-htr:v0.2.5`
base image via `htr_runtime/Dockerfile.v0.2.6` and `htr_runtime/install_v026.py`,
verifying every Python file and native `.so`/`.hsaco` SHA-256 against
`htr_runtime/V026-NATIVE-MANIFEST.json` via `htr_native_policy.verify_installation()`
at image build time.

```sh
git clone https://github.com/HawgAuto/vllm-htr.git
cd vllm-htr
git checkout htr-v0.2.6
docker build -f htr_runtime/Dockerfile.v0.2.6 -t vllm-htr:v0.2.6 .
```
