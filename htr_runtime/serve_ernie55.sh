#!/usr/bin/env bash
# Production launcher for ERNIE-5.5-Flash-Preview-HTR on AMD Strix Halo (gfx1151, 128GB).
# Usage:
#   MODE=dspark5  ./serve_ernie55.sh   # Default: 43L + 3L K=2 DSpark (num_spec=5, best all-rounder)
#   MODE=adaptive ./serve_ernie55.sh   # 43L + 3L K=2 DSpark (num_spec=7 + confidence_head adaptive verification)
#   MODE=dspark4  ./serve_ernie55.sh   # 43L + 3L K=2 DSpark (num_spec=4, fastest prose)
#   MODE=dspark7  ./serve_ernie55.sh   # 43L + 3L K=2 DSpark (num_spec=7, peak burst)
#   MODE=target   ./serve_ernie55.sh   # 43L target-only (batch/long-ctx serving)
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/hawgauto/vllm-htr:v0.2.8}"
MODEL="${MODEL:-/home/hawg/models/ERNIE-5.5-Flash-Preview-HTR}"
WORKSPACE="${WORKSPACE:-/home/hawg/vllm-htr-ernie55-workspace}"
MODE="${MODE:-dspark5}"
PORT="${PORT:-8000}"
MEM_LIMIT="${MEM_LIMIT:-117g}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_BATCH="${MAX_BATCH:-2048}"
# 1.30 GiB KV cache (~34,131 tokens, 1.04x at 32K) fits safely under default Linux CMA ceilings (~110 GiB GTT).
# On hosts with GRUB `cma=64M` (unlocking +6 GiB GTT), override KV_MEM or set USE_GPU_UTIL=1 GPU_UTIL=0.89.
KV_MEM="${KV_MEM:-1395864371}"
GPU_UTIL="${GPU_UTIL:-0.865}"

mkdir -p "${WORKSPACE}"

SPEC_ARGS=()
MEM_ARGS=(--kv-cache-memory "${KV_MEM}")
CG_SIZES=""
case "$MODE" in
  dspark5)
    SPEC_ARGS=(--speculative-config "{\"method\":\"dspark\",\"model\":\"${MODEL}/draft\",\"num_speculative_tokens\":5}")
    CG_SIZES="[1,5,6]"
    ;;
  adaptive)
    SPEC_ARGS=(--speculative-config "{\"method\":\"dspark\",\"model\":\"${MODEL}/draft\",\"num_speculative_tokens\":7,\"enable_adaptive_verification\":true}")
    CG_SIZES="[1,2,3,4,5,6,7,8]"
    ;;
  dspark4)
    SPEC_ARGS=(--speculative-config "{\"method\":\"dspark\",\"model\":\"${MODEL}/draft\",\"num_speculative_tokens\":4}")
    CG_SIZES="[1,4,5]"
    ;;
  dspark7)
    SPEC_ARGS=(--speculative-config "{\"method\":\"dspark\",\"model\":\"${MODEL}/draft\",\"num_speculative_tokens\":7}")
    CG_SIZES="[1,7,8]"
    ;;
  target)
    MAX_NUM_SEQS="${MAX_NUM_SEQS_TARGET:-4}"
    MAX_BATCH="${MAX_BATCH_TARGET:-2048}"
    MEM_ARGS=(--gpu-memory-utilization "${GPU_UTIL}")
    CG_SIZES="[1,2,4,8]"
    ;;
  *)
    echo "Unknown MODE=$MODE (expected: dspark5 | adaptive | dspark4 | dspark7 | target)" >&2
    exit 1
    ;;
esac

echo "=== Launching ERNIE-5.5-HTR (MODE=${MODE}, IMAGE=${IMAGE}, MEM=${MEM_LIMIT}, PORT=${PORT}) ==="
free -h

exec docker run --rm --name "ernie55_${MODE}" \
  --memory="${MEM_LIMIT}" --memory-swap="${MEM_LIMIT}" \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --group-add video --ipc=host \
  -p "${PORT}:8000" \
  -v "${WORKSPACE}:/workspace" \
  -v "${MODEL}:${MODEL}:ro" \
  -e TMPDIR=/tmp \
  -e ERNIE_HTR_MOE_SPAD3=0 \
  -e PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-garbage_collection_threshold:0.6}" \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-128}" \
  -e ERNIE_DSPARK_DRAFT_FILTER="${ERNIE_DSPARK_DRAFT_FILTER:-1}" \
  -e ERNIE_FLASHINDEXER_DECODE="${ERNIE_FLASHINDEXER_DECODE:-1}" \
  "${IMAGE}" \
  "${MODEL}" \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_BATCH}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --hf-overrides '{"enable_flashindexer": true}' \
  "${MEM_ARGS[@]}" \
  --compilation-config "{\"mode\":0,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":${CG_SIZES}}" \
  "${SPEC_ARGS[@]}"
