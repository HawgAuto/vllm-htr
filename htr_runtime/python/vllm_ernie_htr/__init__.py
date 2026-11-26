# SPDX-License-Identifier: Apache-2.0
"""vLLM general plugin entry point for ERNIE-5.5-Flash HTR-W1 on ROCm gfx1151."""

from __future__ import annotations

import logging

_registered = False


def register() -> None:
    global _registered
    if _registered:
        return
    _registered = True

    vllm_logger = logging.getLogger("vllm")
    plugin_logger = logging.getLogger(__name__)
    if not plugin_logger.handlers:
        plugin_logger.handlers = list(vllm_logger.handlers)
        plugin_logger.setLevel(vllm_logger.level)
        plugin_logger.propagate = False

    from vllm import ModelRegistry

    # Point ERNIE-5.5 target & DSpark draft to our ROCm HTR-W1 implementations
    ModelRegistry.register_model(
        "Ernie5_5_MoeForCausalLM",
        "vllm_ernie_htr.model:Ernie5_5ForCausalLM",
    )
    ModelRegistry.register_model(
        "Ernie5_5_MoeDSparkDraftModel",
        "vllm_ernie_htr.dspark:Ernie5_5DSparkForCausalLM",
    )

    from .patches import apply_patches

    apply_patches()

