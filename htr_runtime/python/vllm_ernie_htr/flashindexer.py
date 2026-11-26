# SPDX-License-Identifier: Apache-2.0
"""ROCm-native FlashIndexer head-selection gate for ERNIE-5.5-Flash DSA prefill.

Matches Baidu's ``vLLM-ERNIE`` FlashIndexer algorithm without NVIDIA-only
DeepGEMM or Triton JIT dependencies:

- For each DSA layer (layers 7, 14, 21, 28, 35, 42), loads the trained linear
  head-selection gate ``linear_gate.weight`` (BF16 ``[64, 4096]``) and
  ``flashindexer_bias`` (``[64]``) from ``<model>/flashindexer/flashindexer.safetensors``.
- During prefill, selects the top-8 of 64 indexer heads per token:
    gates  = F.linear(hidden, linear_gate.weight).float()   # raw linear, no activation
    scores = gates + flashindexer_bias.float()
    idx    = torch.topk(scores, k=8, dim=-1, sorted=True).indices
    w_sel  = weights.gather(1, idx) * gates.gather(1, idx)
    q_sel  = q.gather(1, idx.unsqueeze(-1).expand(-1, -1, q.shape[-1]))
- Passes the 8-head ``(q_sel, w_sel)`` directly to the native gfx1151 Wave32
  ``k_ernie_fp8_mqa_logits_prefill_h8`` WMMA HIP kernel (4x less K traffic,
  8x fewer WMMA ops).
- Decode stays 64-head (unchanged, matching upstream FlashIndexer).
"""

from __future__ import annotations

import contextlib
import os
import threading
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

FLASHINDEXER_TOPK_HEADS = 8
FLASHINDEXER_WEIGHT_RELPATH = os.path.join("flashindexer", "flashindexer.safetensors")

_TLS = threading.local()


def _ctx_stack() -> list[tuple[Any, torch.Tensor, torch.Tensor]]:
    stack = getattr(_TLS, "stack", None)
    if stack is None:
        stack = []
        _TLS.stack = stack
    return stack


def is_flashindexer_enabled(config: Any) -> bool:
    """Return True if FlashIndexer is enabled via config or env override."""
    env = os.environ.get("ERNIE_FLASHINDEXER")
    if env is not None:
        return env == "1"
    return bool(getattr(config, "enable_flashindexer", False))


def attach_flashindexer_gate(indexer: nn.Module, config: Any) -> None:
    """Attach FlashIndexer gate parameters onto a target DSA ``Ernie5_5IndexerROCm``."""
    hidden_size = int(config.hidden_size)
    n_head = int(config.index_n_heads)
    topk_heads = int(getattr(config, "flashindexer_topk", FLASHINDEXER_TOPK_HEADS))
    if topk_heads <= 0 or topk_heads > n_head:
        raise ValueError(
            f"Invalid flashindexer_topk={topk_heads} for index_n_heads={n_head}"
        )
    dtype = getattr(indexer.wq_b.weight, "dtype", torch.bfloat16)
    device = getattr(indexer.wq_b.weight, "device", None)
    indexer.linear_gate = nn.Linear(
        hidden_size, n_head, bias=False, device=device, dtype=dtype
    )
    indexer.flashindexer_bias = nn.Parameter(
        torch.zeros(n_head, dtype=torch.float32, device=device),
        requires_grad=False,
    )
    indexer.flashindexer_topk = topk_heads
    indexer._flashindexer_loaded_keys = set()


@contextlib.contextmanager
def flashindexer_context(
    indexer: nn.Module,
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
):
    """Bind the active DSA indexer + prefill hidden_states/weights for chunked MQA."""
    stack = _ctx_stack()
    stack.append((indexer, hidden_states, weights))
    try:
        yield
    finally:
        stack.pop()


def get_active_flashindexer() -> tuple[Any, torch.Tensor, torch.Tensor] | None:
    stack = _ctx_stack()
    return stack[-1] if stack else None


def slice_hidden_for_weights(
    base_hidden: torch.Tensor,
    base_weights: torch.Tensor,
    weights_chunk: torch.Tensor,
) -> torch.Tensor:
    """Map a chunked prefill ``weights[start:end]`` view back to ``hidden[start:end]``."""
    n_chunk = weights_chunk.shape[0]
    if (
        n_chunk == base_weights.shape[0]
        and weights_chunk.storage_offset() == base_weights.storage_offset()
    ):
        return base_hidden

    if weights_chunk.untyped_storage().data_ptr() != base_weights.untyped_storage().data_ptr():
        if n_chunk == base_hidden.shape[0]:
            return base_hidden
        raise RuntimeError(
            "FlashIndexer weights chunk does not share storage with base weights "
            f"(chunk={tuple(weights_chunk.shape)}, base={tuple(base_weights.shape)})"
        )

    offset_elems = weights_chunk.storage_offset() - base_weights.storage_offset()
    stride0 = base_weights.stride(0)
    if stride0 <= 0 or offset_elems < 0 or (offset_elems % stride0) != 0:
        raise RuntimeError(
            "FlashIndexer could not infer chunk row offset "
            f"(offset_elems={offset_elems}, stride0={stride0})"
        )
    row_start = offset_elems // stride0
    row_end = row_start + n_chunk
    if row_end > base_hidden.shape[0]:
        raise RuntimeError(
            f"FlashIndexer chunk slice [{row_start}:{row_end}] exceeds hidden rows {base_hidden.shape[0]}"
        )
    return base_hidden[row_start:row_end]


def flashindexer_select_heads_rocm(
    q: torch.Tensor,
    weights: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    hidden: torch.Tensor,
    topk_heads: int = FLASHINDEXER_TOPK_HEADS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-K heads per token using exact Baidu FlashIndexer math."""
    h = hidden.to(gate_weight.dtype) if hidden.dtype != gate_weight.dtype else hidden
    gates = F.linear(h, gate_weight).float()  # [M, H], raw linear (no activation)
    scores = gates + bias.float()             # [M, H]
    idx = torch.topk(scores, k=topk_heads, dim=-1, sorted=True).indices  # [M, K]
    w_sel = weights.gather(1, idx) * gates.gather(1, idx)                # [M, K]
    exp_idx = idx.unsqueeze(-1).expand(-1, -1, q.shape[-1])
    if q.element_size() == 1 and q.dtype != torch.uint8:
        q_sel = q.view(torch.uint8).gather(1, exp_idx).view(q.dtype)     # [M, K, D] FP8-safe on CPU+ROCm
    else:
        q_sel = q.gather(1, exp_idx)                                     # [M, K, D]
    return q_sel.contiguous(), w_sel.contiguous()


def _resolve_flashindexer_safetensors(model_path: str) -> str:
    override = os.environ.get("ERNIE_FLASHINDEXER_WEIGHTS")
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(
                f"ERNIE_FLASHINDEXER_WEIGHTS={override!r} does not exist"
            )
        return override

    local_cand = os.path.join(model_path, FLASHINDEXER_WEIGHT_RELPATH)
    if os.path.isfile(local_cand):
        return local_cand

    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=model_path,
            filename="flashindexer/flashindexer.safetensors",
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"FlashIndexer enabled but weights not found at {local_cand!r} "
            f"(and HF hub lookup failed: {exc!r})"
        ) from exc


def _map_flashindexer_key(raw_key: str) -> str:
    """Map HF checkpoint key to Ernie5_5ForCausalLM parameter name."""
    mapped = raw_key.replace(".self_attn.core_attention.", ".attn.")
    return mapped


def load_flashindexer_weights(model: nn.Module, model_path: str) -> set[str]:
    """Load ``flashindexer/flashindexer.safetensors`` into attached DSA indexers."""
    attached: dict[str, nn.Module] = {}
    for mod_name, mod in model.named_modules():
        if getattr(mod, "linear_gate", None) is not None and getattr(mod, "flashindexer_bias", None) is not None:
            attached[mod_name] = mod

    if not attached:
        return set()

    weights_path = _resolve_flashindexer_safetensors(model_path)
    params_dict = dict(model.named_parameters())

    from safetensors import safe_open

    loaded: set[str] = set()
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for raw_key in f.keys():
            target_key = _map_flashindexer_key(raw_key)
            if target_key not in params_dict:
                # Allow weights with or without leading "model." prefix
                alt_key = (
                    target_key[len("model."):]
                    if target_key.startswith("model.")
                    else f"model.{target_key}"
                )
                if alt_key in params_dict:
                    target_key = alt_key
                else:
                    continue
            param = params_dict[target_key]
            tensor = f.get_tensor(raw_key)
            if tuple(tensor.shape) != tuple(param.shape):
                raise ValueError(
                    f"FlashIndexer shape mismatch for {target_key}: "
                    f"got {tuple(tensor.shape)}, expected {tuple(param.shape)}"
                )
            with torch.no_grad():
                param.copy_(tensor.to(device=param.device, dtype=param.dtype))
            loaded.add(target_key)
            for mod_name, mod in attached.items():
                if target_key.startswith(f"{mod_name}."):
                    mod._flashindexer_loaded_keys.add(target_key[len(mod_name) + 1:])

    # Strict verification: every attached DSA layer must have BOTH gate & bias loaded
    missing: list[str] = []
    for mod_name, mod in sorted(attached.items()):
        got = getattr(mod, "_flashindexer_loaded_keys", set())
        for req in ("linear_gate.weight", "flashindexer_bias"):
            if req not in got:
                missing.append(f"{mod_name}.{req}")
    if missing:
        raise RuntimeError(
            f"FlashIndexer enabled but missing weights from {weights_path!r}: {missing}"
        )

    logger.info(
        "ERNIE-5.5 ROCm FlashIndexer loaded %d parameters across %d DSA layers from %s (topk=%d/64)",
        len(loaded),
        len(attached),
        weights_path,
        next(iter(attached.values())).flashindexer_topk,
    )
    return loaded
