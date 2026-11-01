# vLLM HTR runtime

This is the serving runtime for [Qwen3.8-Flash-Next-HTR](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR), based on the image-bound vLLM fork at `322c106f3409df1eeb29d2c14434b389b36c2b43`. It integrates HTR dense and routed-expert operators, GPTQ8 hyper-connection kernels, SSD-backed PLE reads, the TAU v2 QSA attention kernel, and the serving launcher.

The separate [HTR project](https://github.com/HawgAuto/HTR) documents the quantizer and format. This repository supplies their model-serving integration; it does not replace the upstream vLLM project.

## Build from source

From the repository root:

```sh
docker build --pull=false -f htr_runtime/Dockerfile -t vllm-htr:local .
```

The Dockerfile pins the existing ROCm/PyTorch/vLLM base by digest, compiles all eleven HTR/native extensions and the TAU v2 QSA code object from the included source, and installs the Python runtime. The build needs no GPU. It does not compile against host ROCm or copy private prebuilt libraries from the checkout. `NATIVE-BUILD.json` binds the selected source and Ninja recipes by SHA-256. Recipes retain the accepted math flags: HIPHC decode uses `-ffp-contract=fast`, while HIPHCP prefill uses `-ffp-contract=off`. The TAU v2 code object is built with the exact command recorded in `NATIVE-BUILD.json` and the build fails unless it matches the qualified binary hash. Native compiler work products remain in the build stage, not the serving image.

The generated image includes `native-build-receipt.json` and `installed-files.json`. The launcher checks the installed runtime and native-checkpoint metadata before starting vLLM. Those checks detect drift; they are not signatures against a malicious image publisher.

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
  --memory 108g --memory-swap 108g --pids-limit 8192 \
  --shm-size 16g --ulimit core=0 \
  --security-opt no-new-privileges:true \
  -p 127.0.0.1:8000:8000 \
  --mount "type=bind,src=$MODEL,dst=/model,readonly" \
  --mount "type=bind,src=$WORKSPACE,dst=/workspace" \
  vllm-htr:local
```

Resolve the correct AMD render node on multi-GPU hosts instead of assuming `renderD128`. Do not co-load another large shared-memory GPU workload. The limits shown leave a reserve on the intended 128 GB machine; they are not a promise that any other workload will fit alongside it.

For a no-device model/runtime metadata preflight:

```sh
docker run --rm --network none --pull=never \
  --mount "type=bind,src=$MODEL,dst=/model,readonly" \
  --mount "type=bind,src=$WORKSPACE,dst=/workspace" \
  vllm-htr:local --cpu-check
```

The API implements the OpenAI-compatible endpoints under `/v1`, with model name `qwen38-htr`. The host port is loopback-only; add an authenticated reverse proxy before any external exposure. The launcher's container bind defaults to `0.0.0.0`; `--host`, `--port`, and `--served-model-name` are supported overrides.

## Serving profile and levers

Defaults (the release profile):

- MTP speculative decoding with three probabilistic draft tokens.
- Thinking enabled, with the Qwen3 reasoning parser.
- Temperature 0.7, top-p 0.8, top-k 20; neutral penalties. These are the selected release settings, not a claim about vendor defaults.
- Context 262,144 tokens; BF16 KV pool 8 GiB; one sequence; maximum batched tokens 12,288.
- Prefix caching and `align` Mamba state caching retained.
- `FULL_AND_PIECEWISE` graph mode; capture sizes 1–5. No eager-mode substitute.

Levers (launcher flags after the image name):

| Flag | Default | Notes |
|---|---|---|
| `--mtp {0,3}` | 3 | 0 disables speculative decoding |
| `--max-num-seqs N` | 1 | up to 4 with MTP3, up to 32 with MTP0 |
| `--context N` | 262144 | up to 1,048,576; above 262,144 static YaRN is applied |
| `--yarn-factor F` | auto | 2 up to 524,288, otherwise 4 |
| `--kv-cache-gib G` | 8 | BF16 KV pool size |
| `--max-num-batched-tokens N` | 12288 | prefill chunk and scheduler budget |
| `--cudagraph-sizes a,b,...` | from seqs/MTP | decode graph capture sizes |
| `--no-prefix-caching` | off | |

`--print-command` shows the exact vLLM command, the cache profile and the levers without starting anything.

## Quick load

Each distinct set of compile-relevant levers (everything above except the KV pool size, host and port) gets its own compile/autotune cache under `/workspace/cache/<profile>`. The first start with a new lever set compiles graphs and tunes kernels; later starts with the same levers load that cache. To do the first start ahead of time, add `--build-seed`: the launcher starts the server, sends warm-up requests across prefill lengths (64 tokens to 32K), stops the server, and writes `<cache>/SEED.json`. Keep `/workspace` on a persistent volume and reuse it.

## Runtime features

- Dense HTR linears: fused GEMV for decode; for prefill each weight is dequantized to BF16 per call and multiplied with the vendor GEMM. No persistent BF16 weight copy is kept.
- QSA sparse attention (`htr_qsa`): the release split-K Triton kernel, or the native TAU v2 kernel for prefills with more than 256 rows. TAU v2 is bitwise equal to split-K; the first admitted calls of every process are re-run through split-K and must match bit for bit (`HTR_QSA_TAU=0` disables it).
- Autotune pin (`HTR_AUTOTUNE_PIN`): fixed Triton configs for the FLA prefill kernels and Inductor deterministic mode, so independent processes run the same kernels.
- MoE partition4 (`HTR_MOE_PARTITION4`): 5–47-token MoE steps run on 4-row partitions with a parity-preserving plan; arithmetic unchanged.
- Routed-expert HPF3 prefill, fused decode combine, GPTQ8 hyper-connection kernels, device-side PLE gather, MTP draft filtering.

QSA attention follows the model's own top-k sparse selection; it is not full attention.

## Model packaging

The release contains 23 target safetensors shards, 13 indexed HTR expert shards, and one unchanged draft safetensors shard. Expert metadata sanitation removes worker/device/timing fields without changing retained metadata or tensor dtype, shape, or bytes. Shard consolidation is not requantization. The expert loader validates each serialized chunk against the pinned manifest before deserializing it.

The target is mixed precision: HTR5 dense projections, HTR4 output/routed-expert portions, GPTQ8 hyper-connection portions, and retained higher-precision tensors. A `bf16` compatibility selector in native-checkpoint metadata does not mean the entire model is BF16.

## Verification status

Source-bound build and CPU checks are separate from physical GPU/model qualification and publication. Local build or import success alone does not establish real serving, performance, or public availability. Final release evidence must identify the tested image and remote source revision; do not treat this source staging README as a completed qualification receipt.

Run the installed CPU suite without exposing devices:

```sh
docker run --rm --network none --pull=never \
  --entrypoint /opt/venv/bin/python3 \
  -e PYTHONPATH=/opt/vllm-htr/python:/opt/vllm-htr \
  -e HTR_MOE2_SO=/opt/vllm-htr/native-bin/htr_moe2.so \
  -e HTR_HPF3_SO=/opt/vllm-htr/native-bin/htr_moe_prefill_v3.so \
  vllm-htr:local -m unittest discover -s /opt/vllm-htr/tests -v
```

Upstream notices and licenses are retained. See HTR's acknowledgments for the exlr8 → ExLlamaV3/EXL3 → QTIP lineage and QuIP/GPTQ/LDLQ foundations; see the upstream vLLM documentation for the engine itself.
