# vLLM HTR runtime

**HTR release v0.1.0 — initial working runtime.** [Release notes and exact source/image pins](https://github.com/HawgAuto/vllm-htr/releases/tag/htr-v0.1.0) · [Release history](../RELEASES.md).

This is the serving runtime for [Qwen3.8-Flash-Next-HTR](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR), based on the image-bound vLLM fork at `322c106f3409df1eeb29d2c14434b389b36c2b43`. It integrates HTR dense and routed-expert operators, GPTQ8 hyper-connection kernels, SSD-backed PLE reads, and the selected QSA1600/MTP3 configuration.

The separate [HTR project](https://github.com/HawgAuto/HTR) documents the quantizer and format. This repository supplies their model-serving integration; it does not replace the upstream vLLM project.

## Get the prebuilt image

The qualified v0.1.0 image is public and can be pulled anonymously. No GitHub account, token or `docker login` is required.

```sh
IMAGE=ghcr.io/hawgauto/vllm-htr@sha256:82cad72483626841b0f858606e5d1951b09645bd109a24eeb3ed7241561aba73
docker pull "$IMAGE"
```

The readable tags `v0.1.0` and `qsa1600-release-20261028` resolve to this same tested Linux/amd64 manifest. The image includes the native extensions; no host compiler or separate native-library download is needed. Continue to [Run](#run), or build from source instead.

## Build from source

This is an optional alternative to the prebuilt image. From the repository root:

```sh
docker build --pull=false -f htr_runtime/Dockerfile -t vllm-htr:local .
IMAGE=vllm-htr:local
```

The Dockerfile pins the existing ROCm/PyTorch/vLLM base by digest, compiles all eleven HTR/native extensions from the included source, and installs the Python runtime. The build needs no GPU. It does not compile against host ROCm or copy private prebuilt libraries from the checkout. `NATIVE-BUILD.json` binds the selected source and Ninja recipes by SHA-256. Recipes retain the accepted math flags: HIPHC decode uses `-ffp-contract=fast`, while HIPHCP prefill uses `-ffp-contract=off`. Native compiler work products remain in the build stage, not the serving image.

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
  "$IMAGE"
```

Resolve the correct AMD render node on multi-GPU hosts instead of assuming `renderD128`. Do not co-load another large shared-memory GPU workload. The limits shown leave a reserve on the intended 128 GB machine; they are not a promise that any other workload will fit alongside it.

For a no-device model/runtime metadata preflight:

```sh
docker run --rm --network none --pull=never \
  --mount "type=bind,src=$MODEL,dst=/model,readonly" \
  --mount "type=bind,src=$WORKSPACE,dst=/workspace" \
  "$IMAGE" --cpu-check
```

The API implements the OpenAI-compatible endpoints under `/v1`, with model name `qwen38-htr`. The host port is loopback-only; add an authenticated reverse proxy before any external exposure. The launcher's container bind defaults to `0.0.0.0`; `--host`, `--port`, and `--served-model-name` are supported overrides.

## Selected profile

- QSA1600 with probabilistic MTP, three draft tokens.
- Thinking enabled, with the Qwen3 reasoning parser.
- Temperature 0.7, top-p 0.8, top-k 20; neutral penalties. These are the selected release settings, not a claim about vendor defaults.
- Configured context: 262,144 tokens; fixed KV budget: 8 GiB; one sequence; maximum batched tokens: 12,288.
- Prefix caching and `align` Mamba state caching retained.
- `FULL_AND_PIECEWISE` graph mode; capture sizes 1–5. No eager-mode substitute.
- Dedicated writable cache namespace, `htr-qsa1600-v1`. No predecessor compile cache is silently reused. First startup includes compilation/autotuning and can be substantially slower than warm startup.

The configured context ceiling is not evidence of a completed 262K-token quality or capacity test. QSA1600 is an approximate sparse-attention policy; it must not be represented as numerically identical to full attention. Short delivery checks are not new quality rankings or throughput benchmarks.

## Model packaging

The release contains 23 target safetensors shards, 13 indexed HTR expert shards, and one unchanged draft safetensors shard. Expert metadata sanitation removes worker/device/timing fields without changing retained metadata or tensor dtype, shape, or bytes. Shard consolidation is not requantization. The expert loader validates each serialized chunk against the pinned manifest before deserializing it.

The target is mixed precision: HTR5 dense projections, HTR4 output/routed-expert portions, GPTQ8 hyper-connection portions, and retained higher-precision tensors. A `bf16` compatibility selector in native-checkpoint metadata does not mean the entire model is BF16.

## Verification status

The existing v0.1.0 runtime has passed final-image text, reasoning, MTP and structured tool-call/result delivery checks; see the [exact-image receipt and its recorded adjudication](https://huggingface.co/hstolte/Qwen3.8-Flash-Next-HTR/blob/v0.1.0/evidence/release-qualification.json). All 63 checkpoint files are complete and hash-verified. The source, prebuilt GHCR image and pinned HaloLoom build base are public. Anonymous access to the unchanged prebuilt manifest and an image layer has been verified. These are existing qualification results, not a new speed/quality benchmark or validation of the upcoming v0.2.0 changes. A local rebuild has its own image identity and does not inherit the prebuilt manifest digest.

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
