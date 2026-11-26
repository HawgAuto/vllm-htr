# SPDX-License-Identifier: Apache-2.0
"""ROCm gfx1151 + HTR-W1 DSpark draft model for ERNIE-5.5-Flash.

Ports baidu/vLLM-ERNIE's Ernie5_5DSparkModel to ROCm + HTR-W1:
  - Reuses Ernie5_5DecoderLayer (mhc_ernie.so + htr_wo_a_ernie.so + htr_moe_ernie.so K=2 + hpf5_ernie.so K=2)
  - Uses native mhc_ernie.so mhc_post + hc_head_contract
  - Uses per-layer HTR-W1 K5 ReplicatedLinear for context_wkv_projs (loaded from linear_kv_proj)
  - Automatically handles both standalone (model.layers.*) and bundled-style (model.mtp.*) draft checkpoints
  - Shares target's embed_tokens and K5 lm_head (zero duplicate VRAM)
"""

from __future__ import annotations

from collections.abc import Iterable
import os
import re

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.deepseek_v4.amd.dspark import _insert_context_kv
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope

from .attention import fold_vha
from .fast_ops import get_fast_ops
from .mhc import hc_head_contract, mhc_post
from .model import (
    Ernie5_5DecoderLayer,
    Ernie5_5ParallelLMHead,
    ernie5_5_mapper,
    fp32_param,
    topk_buffers,
)
from .patches import patch_dspark_sample_logits

logger = init_logger(__name__)

_DSPARK_RENAMES = {
    "model.feature_projection.weight": "model.main_proj.weight",
    "model.feature_projection.weight_scale_inv": "model.main_proj.weight_scale_inv",
    "model.feature_norm.weight": "model.main_norm.weight",
    "model.confidence_head.projection.weight": "model.confidence_head.proj.weight",
}
_TARGET_OWNED = ("lm_head.", "model.embed_tokens.", "model.lm_head.")
_BUNDLED_PREFIX = "model.mtp."
_DRAFT_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(htr_qweight|htr_scales|htr_su|htr_sv)$"
)


def _enrich_htr_modules_for_dspark(
    vllm_config: VllmConfig, draft_config, layer_offset: int
) -> None:
    """Register vLLM-mapped DSpark draft module prefixes in draft quant_config.modules."""
    qc = vllm_config.quant_config
    if qc is None or not hasattr(qc, "modules") or qc.get_name() != "htr":
        return

    # Pull K-map from draft_config.quantization_config if draft quant_config missed mtp.* keys
    mods: dict[str, int] = qc.modules
    raw_q = getattr(draft_config, "quantization_config", None) or {}
    raw_mods = raw_q.get("modules", {}) if isinstance(raw_q, dict) else {}
    for k, v in raw_mods.items():
        kk = int(v["K"]) if isinstance(v, dict) else int(v)
        mods.setdefault(k, kk)

    extra: dict[str, int] = {}
    for k, v in list(mods.items()):
        kk = int(v["K"]) if isinstance(v, dict) else int(v)
        # Normalize model.mtp.{i}.* or mtp.{i}.* or layers.{i}.* -> layers.{layer_offset + i}.*
        m = k
        if m.startswith("model.mtp."):
            m = m[len("model.mtp."):]
        elif m.startswith("mtp."):
            m = m[len("mtp."):]
        elif m.startswith("model.layers."):
            m = m[len("model.layers."):]
        elif m.startswith("layers."):
            m = m[len("layers."):]
        else:
            continue

        head, sep, rest = m.partition(".")
        if not (sep and head.isdigit()):
            continue
        li = int(head)
        off_idx = layer_offset + li

        # Map canonical checkpoint leaf -> vLLM module name
        r = rest
        r = r.replace("self_attn.core_attention.compressor.", "attn.compressor.")
        r = r.replace("self_attn.gate_proj", "attn.o_gate")
        r = r.replace("self_attn.linear_q_up_proj", "attn.wq_b")
        r = r.replace("self_attn.o_proj", "attn.wo_b")
        r = r.replace("self_attn.linear_o_group_proj.g", "attn.wo_a_groups.")
        r = r.replace("self_attn.", "attn.")
        r = r.replace("block_sparse_moe.shared_experts.w2", "ffn.shared_experts.down_proj")
        r = r.replace("block_sparse_moe.shared_experts.w1", "ffn.shared_experts.gate_proj")
        r = r.replace("block_sparse_moe.shared_experts.w3", "ffn.shared_experts.up_proj")
        r = r.replace("block_sparse_moe.", "ffn.")

        for pfx in (f"layers.{off_idx}", f"model.layers.{off_idx}", f"layers.{li}", f"model.layers.{li}"):
            extra[f"{pfx}.{r}"] = kk
            if r == "attn.linear_q_down_proj" or r == "attn.linear_kv_proj":
                extra[f"{pfx}.attn.fused_wqa_wkv"] = kk
            if r == "ffn.shared_experts.gate_proj" or r == "ffn.shared_experts.up_proj":
                extra[f"{pfx}.ffn.shared_experts.gate_up_proj"] = kk
        if rest == "self_attn.linear_kv_proj":
            extra[f"context_wkv_projs.{li}"] = kk
            extra[f"model.context_wkv_projs.{li}"] = kk

    mods.update(extra)


def _unbundle_dspark_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Strip leading `model.mtp.` into `model.layers.{i}.*` / `model.*`."""
    for name, loaded_weight in weights:
        if name.startswith(_BUNDLED_PREFIX):
            rest = name[len(_BUNDLED_PREFIX):]
            if rest.partition(".")[0].isdigit():
                yield f"model.layers.{rest}", loaded_weight
            else:
                yield f"model.{rest}", loaded_weight
        else:
            yield name, loaded_weight


class Ernie5_5DSparkModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        target_config = vllm_config.model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.num_dspark_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)

        # Unique layer indices past target layers -> uncompressed SWA-only attention
        self.layer_offset = (
            target_config.num_hidden_layers + target_config.num_nextn_predict_layers
        )
        _enrich_htr_modules_for_dspark(vllm_config, config, self.layer_offset)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Shared with target after loading (leave None until load_dspark_model binds target's -> saves 1.65 GiB peak VRAM!)
        self.embed_tokens: VocabParallelEmbedding | None = None
        # feature_projection is kept verbatim in BF16 in HTR draft checkpoints
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=maybe_prefix(prefix, "main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.indexer_topk_buffer, self.swa_topk_buffer = topk_buffers(
            vllm_config.scheduler_config.max_num_batched_tokens, target_config
        )
        self.layers = nn.ModuleList(
            [
                Ernie5_5DecoderLayer(
                    vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.layer_offset + i}"),
                    topk_indices_buffer=self.indexer_topk_buffer,
                    swa_topk_buffer=self.swa_topk_buffer,
                )
                for i in range(self.num_dspark_layers)
            ]
        )
        # Share ONE plain RoPE table across all 3 draft layers (saves 0.27 GiB VRAM!)
        if len(self.layers) > 0:
            a0 = self.layers[0].attn
            shared_rope = build_deepseek_v4_rope(
                config,
                head_dim=a0.head_dim,
                rope_head_dim=a0.rope_head_dim,
                max_position_embeddings=target_config.max_position_embeddings,
                compress_ratio=1,
            )
            try:
                from . import attention as _ernie_att
                _t_cs = getattr(_ernie_att, "_ERNIE5_TARGET_HCA_COS_SIN", None)
                _d_cs = getattr(shared_rope, "cos_sin_cache", None)
                if (
                    _t_cs is not None and _d_cs is not None and _t_cs is not _d_cs
                    and _t_cs.shape == _d_cs.shape and _t_cs.dtype == _d_cs.dtype and _t_cs.device == _d_cs.device
                    and torch.equal(_t_cs[:64], _d_cs[:64]) and torch.equal(_t_cs[-64:], _d_cs[-64:])
                ):
                    shared_rope.cos_sin_cache = _t_cs
            except Exception:
                pass
            for layer in self.layers:
                layer.attn.rotary_emb = layer.attn.indexer_rotary_emb = shared_rope

        # Per-layer HTR-W1 K5 linear_kv_proj for context KV precompute
        self.context_wkv_projs = nn.ModuleList(
            [
                ReplicatedLinear(
                    config.hidden_size,
                    config.head_dim,
                    bias=False,
                    return_bias=False,
                    quant_config=vllm_config.quant_config,
                    prefix=maybe_prefix(prefix, f"context_wkv_projs.{i}"),
                )
                for i in range(self.num_dspark_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = fp32_param(self.hc_mult, hc_dim)
        self.hc_head_base = fp32_param(self.hc_mult)
        self.hc_head_scale = fp32_param(1)

        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            config.vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        if getattr(config, "enable_confidence_head", True):
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size + config.dspark_markov_rank,
                prefix=maybe_prefix(prefix, "confidence_head"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        T = aux_hidden_states.shape[0]
        if (
            1 <= T <= 8
            and os.environ.get("ERNIE_BF16_GEMV_NT", "1") == "1"
            and aux_hidden_states.is_cuda
            and aux_hidden_states.dtype == torch.bfloat16
            and self.main_proj.weight.dtype == torch.bfloat16
            and (aux_hidden_states.shape[1] % 256) == 0
        ):
            fops = get_fast_ops()
            if fops is not None and hasattr(fops, "bf16_gemv_nt"):
                proj = fops.bf16_gemv_nt(aux_hidden_states.contiguous(), self.main_proj.weight, True)
                return self.main_norm(proj)
        return self.main_norm(self.main_proj(aux_hidden_states))

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        fops = get_fast_ops()
        for i, (layer, kv_proj) in enumerate(
            zip(self.layers, self.context_wkv_projs, strict=True)
        ):
            slot_mapping = (
                None if context_slot_mappings is None else context_slot_mappings[i]
            )
            kv = kv_proj(main_x)
            attn = layer.attn
            if slot_mapping is None:
                continue
            swa_cache = attn.swa_cache_layer.kv_cache
            if (
                fops is not None
                and swa_cache.dtype == torch.uint8
                and kv.dtype == torch.bfloat16
                and kv.shape[-1] == 512
                and hasattr(attn.kv_norm, "weight")
            ):
                swa_2d = swa_cache.view(swa_cache.shape[0], -1)
                eps = float(getattr(attn.kv_norm, "variance_epsilon", getattr(attn.kv_norm, "eps", 1e-6)))
                fops.insert_context_kv_ue8m0(
                    kv.contiguous(),
                    swa_2d,
                    slot_mapping,
                    context_positions,
                    attn.rotary_emb.cos_sin_cache,
                    int(attn.swa_cache_layer.block_size),
                    attn.kv_norm.weight,
                    eps,
                )
            else:
                kv = attn.kv_norm(kv)
                _insert_context_kv(attn, kv, context_positions, slot_mapping)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds
        residual = post_mix = res_mix = None
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states, positions, post_mix, res_mix, residual
            )
        hidden_states = mhc_post(hidden_states, residual, post_mix, res_mix)
        return hc_head_contract(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )


class Ernie5_5DSparkForCausalLM(nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.model = Ernie5_5DSparkModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Replaced by target's K5 HTR lm_head after loading (leave None -> saves 1.65 GiB peak VRAM!)
        self.lm_head: Ernie5_5ParallelLMHead | None = None
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.logits_processor.head_dtype = None
        patch_dspark_sample_logits()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.attn.swa_cache_layer.prefix for layer in self.model.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mappings
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Multimax is applied AFTER Markov bias in postprocess_draft_logits
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

    def postprocess_draft_logits(self, logits: torch.Tensor) -> torch.Tensor:
        neg_inf = logits.isneginf()
        return self.lm_head.multimax(logits).masked_fill_(neg_inf, float("-inf"))

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def apply_markov_bias_gathered(
        self,
        markov_embed: torch.Tensor,
        logits: torch.Tensor,
        values: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.markov_head.apply_bias_gathered(
            markov_embed,
            logits,
            values,
            index,
            self.logits_processor.scale,
        )

    def compute_confidence(
        self, head_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def _map_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor, int | None]]:
        for name, loaded_weight in weights:
            if name.startswith(_TARGET_OWNED):
                continue
            if name in _DSPARK_RENAMES:
                rname = _DSPARK_RENAMES[name]
                if "confidence_head." in rname and self.model.confidence_head is None:
                    continue
                yield rname, loaded_weight, None
                continue
            if name == "model.markov_head.markov_w1.weight":
                yield name, loaded_weight.t(), None
                continue
            if ".self_attn.linear_kv_proj." in name:
                layer_idx_s, _, leaf = name.removeprefix("model.layers.").partition(
                    ".self_attn.linear_kv_proj."
                )
                yield f"model.context_wkv_projs.{int(layer_idx_s)}.{leaf}", loaded_weight.detach(), None
            mapped = ernie5_5_mapper._map_name_with_shard(name)
            if mapped is None:
                continue
            mname, shard_id = mapped
            yield mname, loaded_weight, shard_id

    def is_unused_checkpoint_weight(self, name: str) -> bool:
        bundled = getattr(self.config, "dspark_bundled_in_target", False)
        return bundled and not name.startswith(_BUNDLED_PREFIX)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Always normalize model.mtp.* -> model.layers.{i}.* / model.*
        stream = _unbundle_dspark_weights(weights)
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight, shard_id in self._map_weights(stream):
            m_exp = _DRAFT_EXPERT_RE.match(name)
            if m_exp is not None:
                layer_idx = int(m_exp.group(1))
                exp_id = int(m_exp.group(2))
                role_name = m_exp.group(3)
                leaf = m_exp.group(4)
                self.model.layers[layer_idx].ffn.experts.load_expert_tensor(
                    exp_id, role_name, leaf, loaded_weight
                )
                loaded_params.add(name)
                continue

            if name not in params_dict:
                head, _, leaf = name.rpartition(".")
                alt = f"{head}.base_layer.{leaf}"
                if alt in params_dict:
                    name = alt
                else:
                    logger.warning("ERNIE-5.5 DSpark skipping unexpected weight: %s", name)
                    continue

            param = params_dict[name]
            leaf = name.rpartition(".")[2]
            if leaf.startswith("hc_") and leaf.endswith("_fn"):
                param.data.copy_(loaded_weight.t())
            elif leaf in ("hc_attn_scale", "hc_ffn_scale"):
                param.data[shard_id : shard_id + 1].copy_(loaded_weight)
            elif name.endswith(".ffn.gate.weight"):
                rows = loaded_weight.shape[0]
                start = (shard_id or 0) * rows
                param.data[start : start + rows].copy_(loaded_weight.to(param.dtype))
            elif leaf == "attn_sink":
                param.data.copy_(loaded_weight)
            elif shard_id is not None:
                param.weight_loader(param, loaded_weight, shard_id)
            else:
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        if self.model.confidence_head is not None and not any(
            "confidence_head." in n for n in loaded_params
        ):
            self.model.confidence_head = None

        loaded_params.update(
            n for n, _ in self.named_parameters()
            if n.startswith(("lm_head.", "model.embed_tokens."))
        )
        logger.info_once(
            "ERNIE-5.5 HTR DSpark draft model loaded: %d params (K=2 experts, K=5 trunk)",
            len(loaded_params),
        )
        return loaded_params

    def process_weights_after_loading(self) -> None:
        fold_vha(self)

