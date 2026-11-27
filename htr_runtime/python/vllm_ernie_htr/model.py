# SPDX-License-Identifier: Apache-2.0
"""ERNIE-5.5-Flash HTR-W1 model for vLLM-HTR on AMD Strix Halo (gfx1151)."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
import os
import re

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SituAndMul
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import EagleModelMixin, SupportsEagle3, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.triton_utils import tl, triton

from .attention import (
    Ernie5_5DSAAttention,
    Ernie5_5HCAAttention,
    fold_vha,
)
from .mhc import (
    hc_head_contract,
    mhc_fused_post_pre,
    mhc_post,
    mhc_pre,
    mhc_pre_broadcast,
)
from .fast_ops import get_fast_ops
from .moe_ops import Ernie5_5HTRRoutedExperts

logger = init_logger(__name__)

_EXPERT_KEY_RE = re.compile(
    r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(htr_qweight|htr_scales|htr_su|htr_sv)$"
)


def fp32_param(*shape: int) -> nn.Parameter:
    return nn.Parameter(torch.empty(shape, dtype=torch.float32), requires_grad=False)


def topk_buffers(num_tokens: int, config) -> tuple[torch.Tensor, torch.Tensor]:
    indexer = torch.empty(num_tokens, config.index_topk, dtype=torch.int32)
    swa = torch.empty(
        num_tokens, config.index_topk + config.sliding_window, dtype=torch.int32
    )
    return indexer, swa


def _enrich_htr_modules_for_ernie55(vllm_config: VllmConfig) -> None:
    """Add vLLM-mapped ERNIE-5.5 layer prefixes as aliases inside HTRConfig.modules.

    Leaves Qwen3.8 and every other model completely untouched.
    """
    qc = vllm_config.quant_config
    if qc is None or not hasattr(qc, "modules") or qc.get_name() != "htr":
        return
    mods: dict[str, int] = qc.modules
    extra: dict[str, int] = {}
    for k, v in list(mods.items()):
        # Map HF checkpoint canonical keys -> vLLM module prefixes
        m = k
        m = m.replace(".self_attn.core_attention.compressor.", ".attn.compressor.")
        m = m.replace(".self_attn.gate_proj", ".attn.o_gate")
        m = m.replace(".self_attn.linear_q_up_proj", ".attn.wq_b")
        m = m.replace(".self_attn.o_proj", ".attn.wo_b")
        m = m.replace(".self_attn.linear_o_group_proj.g", ".attn.wo_a_groups.")
        m = m.replace(".self_attn.", ".attn.")
        m = m.replace(".block_sparse_moe.shared_experts.w2", ".ffn.shared_experts.down_proj")
        m = m.replace(".block_sparse_moe.shared_experts.w1", ".ffn.shared_experts.gate_proj")
        m = m.replace(".block_sparse_moe.shared_experts.w3", ".ffn.shared_experts.up_proj")
        m = m.replace(".block_sparse_moe.", ".ffn.")
        m = m.replace(".mlp.", ".ffn.")
        extra[m] = v
        # Also keep shared_experts w1/w3 aliases just in case
        if ".ffn.shared_experts.gate_proj" in m:
            extra[m.replace(".gate_proj", ".w1")] = v
        if ".ffn.shared_experts.up_proj" in m:
            extra[m.replace(".up_proj", ".w3")] = v
    mods.update(extra)


def _infer_expert_K(vllm_config: VllmConfig, layer_id: int) -> int:
    qc = vllm_config.quant_config
    if qc is not None and hasattr(qc, "modules"):
        for cand in (
            f"layers.{layer_id}.block_sparse_moe.experts.0.w1",
            f"layers.{layer_id}.ffn.experts.0.w1",
        ):
            if cand in qc.modules:
                return int(qc.modules[cand])
    return 3


# ------------------------------------------------------------------ MultiMax LM Head

@triton.jit
def _relu(x):
    return tl.where(x <= 0, 0.0, x)


@triton.jit
def _multimax_kernel(logits_ptr, ranges_ptr, ts_ptr, stride, V, BLOCK: tl.constexpr):
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < V
    ptrs = logits_ptr + tl.program_id(0).to(tl.int64) * stride + offs
    x = tl.load(ptrs, mask=mask)
    a = _relu(tl.load(ranges_ptr) - x)
    b = _relu(x - tl.load(ranges_ptr + 1))
    c = _relu(tl.load(ranges_ptr + 2) - x)
    d = _relu(x - tl.load(ranges_ptr + 3))
    y = x + tl.load(ts_ptr) * a
    y = y + tl.load(ts_ptr + 1) * b
    y = y + tl.load(ts_ptr + 2) * (c * c)
    y = y + tl.load(ts_ptr + 3) * (d * d)
    tl.store(ptrs, y, mask=mask)


class Ernie5_5ParallelLMHead(ParallelLMHead):
    """HTR-W1 K5 lm_head with learned FP32 MultiMax."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("multimax_ranges", torch.zeros(4, dtype=torch.float32))
        self.register_buffer("multimax_ts", torch.zeros(4, dtype=torch.float32))

    def multimax(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.dtype != torch.float32:
            logits = logits.float()
        r, t = self.multimax_ranges, self.multimax_ts
        if (
            logits.is_cuda
            and logits.dtype == torch.float32
            and logits.dim() == 2
            and logits.stride(1) == 1
        ):
            num_tokens, vocab = logits.shape
            if num_tokens and vocab:
                fops = get_fast_ops()
                if fops is not None:
                    fops.multimax(logits, r, t)
                    return logits
                block = 2048
                _multimax_kernel[(num_tokens, triton.cdiv(vocab, block))](
                    logits,
                    r,
                    t,
                    logits.stride(0),
                    vocab,
                    BLOCK=block,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            return logits
        return (
            logits
            + t[0] * (r[0] - logits).relu()
            + t[1] * (logits - r[1]).relu()
            + t[2] * (r[2] - logits).relu().square()
            + t[3] * (logits - r[3]).relu().square()
        )


# ------------------------------------------------------------------ MLP & Latent MoE

class Ernie5_5MLP(nn.Module):
    """SiTU-GLU feed-forward block (layer 0 dense MLP and shared experts)."""

    def __init__(
        self,
        config,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SituAndMul(
            beta=config.activation_situ_beta,
            linear_beta=config.activation_situ_linear_beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        qw = getattr(self.down_proj, "htr_qweight", None)
        if (
            qw is not None
            and gate_up.is_cuda
            and gate_up.dtype == torch.bfloat16
            and gate_up.dim() == 2
            and gate_up.stride(1) == 1
            and (gate_up.shape[1] // 2) <= 14336
            and ((gate_up.shape[1] // 2) % 128 == 0)
            and os.environ.get("ERNIE_FUSED_PROLOGUE", "1") == "1"
        ):
            from .patches import htr_k5sp_linear_pre_prologued
            if qw.is_cuda and qw.dim() == 3 and qw.shape[2] == 40 and (qw.shape[0] % 8) == 0:
                from .fast_ops import get_k5_sp_prefill
                sp_ext = get_k5_sp_prefill()
                if sp_ext is not None and hasattr(sp_ext, "situ_and_mul_htr_prologue"):
                    lb = -1.0 if self.act_fn.linear_beta is None else float(self.act_fn.linear_beta)
                    xt, xs, tok = sp_ext.situ_and_mul_htr_prologue(
                        gate_up, self.down_proj.htr_su, float(self.act_fn.beta), lb
                    )
                    return htr_k5sp_linear_pre_prologued(
                        xt, xs, tok, qw, self.down_proj.htr_scales, self.down_proj.htr_sv,
                        getattr(self.down_proj, "bias", None),
                    )
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


@triton.jit
def _ernie5_5_fused_route_kernel(
    logits_ptr,
    bias_ptr,
    scale_ptr,
    pad_ptr,
    weights_ptr,
    ids_ptr,
    logits_stride,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_PAD: tl.constexpr,
    RENORM: tl.constexpr,
    HAS_PAD: tl.constexpr,
):
    """Single-warp fused split-sigmoid-sum + bias top-k + renorm + scale router."""
    row = tl.program_id(0)
    offs = tl.arange(0, E)
    kofs = tl.arange(0, TOPK_PAD)
    l0 = tl.load(logits_ptr + row * logits_stride + offs).to(tl.float32)
    l1 = tl.load(logits_ptr + row * logits_stride + E + offs).to(tl.float32)
    scores = (1.0 / (1.0 + tl.exp(-l0))) + (1.0 / (1.0 + tl.exp(-l1)))
    sel = scores + tl.load(bias_ptr + offs).to(tl.float32)
    ids = tl.full((TOPK_PAD,), E, dtype=tl.int32)
    weights = tl.zeros((TOPK_PAD,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        best = tl.max(sel, 0)
        idx = tl.minimum(tl.min(tl.where(sel == best, offs, E), 0), E - 1)
        hit = offs == idx
        ids = tl.where(kofs == k, idx, ids)
        weights = tl.where(kofs == k, tl.sum(tl.where(hit, scores, 0.0), 0), weights)
        sel = tl.where(hit, float("-inf"), sel)
    kmask = kofs < TOPK
    if RENORM:
        total = tl.maximum(tl.sum(weights, 0), 1e-20)
        weights = weights / total
    weights = weights * tl.load(scale_ptr + ids, mask=kmask, other=0.0).to(tl.float32)
    if HAS_PAD:
        is_pad = tl.load(pad_ptr + row) != 0
        weights = tl.where(is_pad, 0.0, weights)
        ids = tl.where(is_pad, -1, ids)
    tl.store(weights_ptr + row * TOPK + kofs, weights, mask=kmask)
    tl.store(ids_ptr + row * TOPK + kofs, ids, mask=kmask)


def _ernie5_5_route(
    logits: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    num_experts: int,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact split-feature sigmoid-sum routing + bias top-k + learnable scale."""
    num_tokens = logits.shape[0]
    is_padding = None
    if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
        ip = get_forward_context().is_padding
        if ip is not None and ip.numel() >= num_tokens:
            is_padding = ip[:num_tokens]

    if (
        num_tokens > 0
        and logits.is_cuda
        and logits.stride(1) == 1
        and num_experts == 512
        and topk == 10
        and logits.dtype == torch.float32
    ):
        fops = get_fast_ops()
        if fops is not None:
            weights, ids = fops.fused_route(logits, bias, scale, is_padding, topk, renormalize)
            return weights, ids
    if (
        num_tokens > 0
        and logits.is_cuda
        and logits.stride(1) == 1
        and triton.next_power_of_2(num_experts) == num_experts
    ):
        weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=logits.device)
        ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=logits.device)
        _ernie5_5_fused_route_kernel[(num_tokens,)](
            logits,
            bias,
            scale,
            is_padding if is_padding is not None else ids,
            weights,
            ids,
            logits.stride(0),
            E=num_experts,
            TOPK=topk,
            TOPK_PAD=triton.next_power_of_2(topk),
            RENORM=renormalize,
            HAS_PAD=is_padding is not None,
            num_warps=1,
        )
        return weights, ids

    scores = logits[:, :num_experts].sigmoid() + logits[:, num_experts:].sigmoid()
    ids = torch.topk(scores + bias, topk, dim=-1, sorted=False).indices
    weights = scores.gather(1, ids)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
    weights = (weights * scale[ids]).float()
    ids = ids.to(torch.int32)
    if is_padding is not None:
        mask = is_padding.unsqueeze(1)
        weights = weights.masked_fill(mask, 0.0)
        ids = ids.masked_fill(mask, -1)
    return weights, ids


class Ernie5_5MoE(nn.Module):
    """Latent MoE with split-feature routing and packed HTR-W1 routed experts."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        expert_K: int = 3,
    ) -> None:
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob

        hidden_size = config.hidden_size
        latent_size = config.moe_latent_size

        gate_dtype = (
            torch.bfloat16
            if os.environ.get("ERNIE_ROUTER_BF16", "1") == "1"
            else torch.float32
        )
        self.gate = GateLinear(
            hidden_size,
            2 * self.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            params_dtype=gate_dtype,
            prefix=f"{prefix}.gate",
        )
        self.gate.routed_scaling_factor_param = nn.Parameter(
            torch.ones(self.n_routed_experts, dtype=torch.float32), requires_grad=False
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.n_routed_experts, dtype=torch.float32), requires_grad=False
        )

        self.shared_experts = Ernie5_5MLP(
            config,
            config.moe_intermediate_size * config.n_shared_experts,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )
        self.fc1_latent_proj = ReplicatedLinear(
            hidden_size,
            latent_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1_latent_proj",
        )
        self.latent_norm = RMSNorm(latent_size, eps=config.rms_norm_eps)
        self.fc2_latent_proj = ReplicatedLinear(
            latent_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2_latent_proj",
        )
        self.experts = Ernie5_5HTRRoutedExperts(
            num_experts=self.n_routed_experts, K=expert_K
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        T = hidden_states.shape[0]
        if (
            1 <= T <= 16
            and os.environ.get("ERNIE_ROUTER_GEMV_NT", "1") == "1"
            and os.environ.get("ERNIE_BF16_GEMV_NT", "1") == "1"
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and self.gate.weight.dtype == torch.bfloat16
            and (hidden_states.shape[1] % 256) == 0
        ):
            fops = get_fast_ops()
            if fops is not None and hasattr(fops, "bf16_gemv_nt"):
                logits = fops.bf16_gemv_nt(hidden_states.contiguous(), self.gate.weight)
            else:
                logits, _ = self.gate(hidden_states)
        elif (
            T > 16
            and os.environ.get("ERNIE_ROUTER_GEMM_WMMA", "1") == "1"
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and self.gate.weight.dtype == torch.bfloat16
            and (hidden_states.shape[1] % 128) == 0
            and (self.gate.weight.shape[0] % 64) == 0
        ):
            fops = get_fast_ops()
            if fops is not None and hasattr(fops, "bf16_gemm_f32_out"):
                logits = fops.bf16_gemm_f32_out(
                    hidden_states.contiguous(), self.gate.weight.contiguous()
                )
            else:
                logits, _ = self.gate(hidden_states)
        else:
            logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids = _ernie5_5_route(
            logits,
            self.e_score_correction_bias,
            self.gate.routed_scaling_factor_param,
            self.n_routed_experts,
            self.top_k,
            self.renormalize,
        )
        routed_in, _ = self.fc1_latent_proj(hidden_states)
        routed_out = self.experts(routed_in, topk_ids, topk_weights)
        qw2 = getattr(self.fc2_latent_proj, "htr_qweight", None)
        fused_fc2 = False
        if (
            qw2 is not None
            and routed_out.is_cuda
            and routed_out.dtype == torch.bfloat16
            and routed_out.dim() == 2
            and routed_out.stride(1) == 1
            and (routed_out.shape[1] % 128 == 0)
            and routed_out.shape[1] <= 8192
            and os.environ.get("ERNIE_FUSED_PROLOGUE", "1") == "1"
        ):
            from .patches import htr_k5sp_linear_pre_prologued
            if qw2.is_cuda and qw2.dim() == 3 and qw2.shape[2] == 40 and (qw2.shape[0] % 8) == 0:
                from .fast_ops import get_k5_sp_prefill
                sp_ext = get_k5_sp_prefill()
                if sp_ext is not None and hasattr(sp_ext, "rmsnorm_htr_prologue"):
                    eps = float(getattr(self.latent_norm, "variance_epsilon", getattr(self.latent_norm, "eps", 1e-6)))
                    xt, xs, tok = sp_ext.rmsnorm_htr_prologue(
                        routed_out, self.latent_norm.weight.data.contiguous(), self.fc2_latent_proj.htr_su, eps
                    )
                    # Wave 47: When M256 blocked-xt prefill is active (xt.dim()==4), compute shared_out first
                    # and fuse (routed_out + shared_out) in-place into fc2_latent_proj WMMA epilogue
                    # (0.0 bit-exact, eliminates 42x [T, 4096] BF16 allocs + 2.82 GB DRAM add traffic = -17ms/4K!)
                    if xt.dim() == 4 and getattr(sp_ext, "SUPPORTS_UNSPLIT_ACCUM_OUT", False):
                        shared_out = self.shared_experts(hidden_states)
                        if shared_out.is_cuda and shared_out.is_contiguous() and shared_out.dtype == torch.bfloat16:
                            return htr_k5sp_linear_pre_prologued(
                                xt, xs, tok, qw2, self.fc2_latent_proj.htr_scales, self.fc2_latent_proj.htr_sv,
                                getattr(self.fc2_latent_proj, "bias", None), accum_out=shared_out,
                            )
                        routed_out = htr_k5sp_linear_pre_prologued(
                            xt, xs, tok, qw2, self.fc2_latent_proj.htr_scales, self.fc2_latent_proj.htr_sv,
                            getattr(self.fc2_latent_proj, "bias", None),
                        )
                        return routed_out + shared_out
                    routed_out = htr_k5sp_linear_pre_prologued(
                        xt, xs, tok, qw2, self.fc2_latent_proj.htr_scales, self.fc2_latent_proj.htr_sv,
                        getattr(self.fc2_latent_proj, "bias", None),
                    )
                    fused_fc2 = True
        if not fused_fc2:
            routed_out = self.latent_norm(routed_out)
            routed_out, _ = self.fc2_latent_proj(routed_out)
        shared_out = self.shared_experts(hidden_states)
        return routed_out + shared_out


# ------------------------------------------------------------------ Decoder Layer & Model

class Ernie5_5DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        swa_topk_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        layer_id = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        if config.is_dsa_layer(layer_id):
            self.attn = Ernie5_5DSAAttention(
                vllm_config,
                prefix=f"{prefix}.attn",
                topk_indices_buffer=topk_indices_buffer,
                swa_topk_buffer=swa_topk_buffer,
            )
        else:
            self.attn = Ernie5_5HCAAttention(
                vllm_config,
                prefix=f"{prefix}.attn",
                topk_indices_buffer=topk_indices_buffer,
            )

        if layer_id < config.first_k_dense_replace:
            self.ffn: nn.Module = Ernie5_5MLP(
                config,
                config.intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.ffn",
            )
        else:
            self.ffn = Ernie5_5MoE(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.ffn",
                expert_K=_infer_expert_K(vllm_config, layer_id),
            )

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = fp32_param(mix_hc, hc_dim)
        self.hc_ffn_fn = fp32_param(mix_hc, hc_dim)
        self.hc_attn_base = fp32_param(mix_hc)
        self.hc_ffn_base = fp32_param(mix_hc)
        self.hc_attn_scale = fp32_param(3)
        self.hc_ffn_scale = fp32_param(3)

    def _mhc_args(self, kind: str) -> tuple:
        return (
            getattr(self, f"hc_{kind}_fn"),
            getattr(self, f"hc_{kind}_scale"),
            getattr(self, f"hc_{kind}_base"),
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        norm_kwargs = dict(
            norm_weight=self.attn_norm.weight.data,
            norm_eps=self.attn_norm.variance_epsilon,
        )
        if residual is None:
            if x.dim() == 2:
                residual, post_mix, res_mix, x = mhc_pre_broadcast(
                    x, *self._mhc_args("attn"), **norm_kwargs
                )
            else:
                residual = x
                post_mix, res_mix, x = mhc_pre(
                    x, *self._mhc_args("attn"), **norm_kwargs
                )
        else:
            residual, post_mix, res_mix, x = mhc_fused_post_pre(
                x, residual, post_mix, res_mix, *self._mhc_args("attn"), **norm_kwargs
            )

        x = self.attn(positions, x, None)

        residual, post_mix, res_mix, x = mhc_fused_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            *self._mhc_args("ffn"),
            norm_weight=self.ffn_norm.weight.data,
            norm_eps=self.ffn_norm.variance_epsilon,
        )
        x = self.ffn(x)
        return x, residual, post_mix, res_mix


def _hc_rules(ckpt: str, kind: str) -> dict[str, tuple[str, int]]:
    return {
        f".{ckpt}_hyper_connection.alpha_{alpha}": (f".hc_{kind}_scale", i)
        for i, alpha in enumerate(("pre", "post", "res"))
    }


ernie5_5_mapper = WeightsMapper(
    orig_to_new_substr={
        ".transformer_layer.": ".",
        ".self_attn.core_attention.softmax_offset": ".attn.attn_sink",
        ".self_attn.core_attention.": ".attn.",
        ".self_attn.gate_proj.": ".attn.o_gate.",
        ".self_attn.vha_postmix_": ".attn.vha.",
        ".self_attn.q_layernorm.": ".attn.q_norm.",
        ".self_attn.kv_layernorm.": ".attn.kv_norm.",
        ".self_attn.linear_q_up_proj.": ".attn.wq_b.",
        ".self_attn.linear_o_group_proj.g": ".attn.wo_a_groups.",
        ".self_attn.o_proj.": ".attn.wo_b.",
        ".self_attn.": ".attn.",
        ".block_sparse_moe.": ".ffn.",
        ".mlp.": ".ffn.",
        ".shared_experts.w2.": ".shared_experts.down_proj.",
        ".input_layernorm.": ".attn_norm.",
        ".post_attention_layernorm.": ".ffn_norm.",
        ".self_attention_hyper_connection.mapping_proj.weight": ".hc_attn_fn",
        ".self_attention_hyper_connection.bias": ".hc_attn_base",
        ".mlp_hyper_connection.mapping_proj.weight": ".hc_ffn_fn",
        ".mlp_hyper_connection.bias": ".hc_ffn_base",
    },
    orig_to_new_stacked={
        ".attn.linear_q_down_proj.": (".attn.fused_wqa_wkv.", 0),
        ".attn.linear_kv_proj.": (".attn.fused_wqa_wkv.", 1),
        ".compressor.linear_wkv.": (".compressor.fused_wkv_wgate.", 0),
        ".compressor.linear_wgate.": (".compressor.fused_wkv_wgate.", 1),
        ".indexer.wk.": (".indexer.wk_weights_proj.", 0),
        ".indexer.weights_proj.": (".indexer.wk_weights_proj.", 1),
        ".shared_experts.w1.": (".shared_experts.gate_up_proj.", 0),
        ".shared_experts.w3.": (".shared_experts.gate_up_proj.", 1),
        ".ffn.gate_proj.": (".ffn.gate_up_proj.", 0),
        ".ffn.up_proj.": (".ffn.gate_up_proj.", 1),
        ".ffn.gate.weight_1": (".ffn.gate.weight", 1),
        **_hc_rules("self_attention", "attn"),
        **_hc_rules("mlp", "ffn"),
    },
    orig_to_new_prefix={
        "model.mhc_contract.": "model.",
        "model.lm_head.": "lm_head.",
    },
)


class Ernie5_5Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        _enrich_htr_modules_for_ernie55(vllm_config)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        self.indexer_topk_buffer, self.swa_topk_buffer = topk_buffers(
            max_num_tokens, config
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Ernie5_5DecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.indexer_topk_buffer,
                swa_topk_buffer=self.swa_topk_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.hc_head_fn = fp32_param(self.hc_mult, self.hc_dim)
        self.hc_head_base = fp32_param(self.hc_mult)
        self.hc_head_scale = fp32_param(1)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                )
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        residual, post_mix, res_mix = None, None, None
        aux_hidden_states: list[torch.Tensor] = []
        aux_recon: torch.Tensor | None = None
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states, positions, post_mix, res_mix, residual
            )
            if idx + 1 in self.aux_hidden_state_layers:
                aux_recon = mhc_post(hidden_states, residual, post_mix, res_mix)
                aux_hidden_states.append(aux_recon.mean(dim=1))
                # Reuse aux_recon directly on the next layer (mhc_pre) instead of recomputing mhc_post!
                hidden_states = aux_recon
                residual = post_mix = res_mix = None
        if residual is not None:
            if self.end_layer in self.aux_hidden_state_layers and aux_recon is not None:
                hidden_states = aux_recon
            else:
                hidden_states = mhc_post(hidden_states, residual, post_mix, res_mix)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = hc_head_contract(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        num_layers = self.config.num_hidden_layers
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        for name, loaded_weight in weights:
            if name.startswith("mtp."):
                continue
            if name.startswith("layers.") and int(name.split(".")[1]) >= num_layers:
                continue
            if is_pp_missing_parameter(name, self):
                continue

            # Route packed HTR routed-expert tensors directly into per-layer GPU arena
            m_exp = _EXPERT_KEY_RE.match(name)
            if m_exp is not None:
                layer_idx = int(m_exp.group(1))
                exp_id = int(m_exp.group(2))
                role_name = m_exp.group(3)
                leaf = m_exp.group(4)
                layer = self.layers[layer_idx]
                layer.ffn.experts.load_expert_tensor(exp_id, role_name, leaf, loaded_weight)
                loaded_params.add(name)
                continue

            shard_id = getattr(loaded_weight, "shard_id", None)
            if name not in params_dict:
                head, _, leaf = name.rpartition(".")
                alt = f"{head}.base_layer.{leaf}"
                if alt in params_dict:
                    name = alt
                else:
                    logger.warning("ERNIE-5.5 HTR skipping unexpected weight: %s", name)
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
                n = loaded_weight.shape[0] // tp_size
                param.data[:n].copy_(loaded_weight[tp_rank * n : (tp_rank + 1) * n])
            elif shard_id is not None:
                param.weight_loader(param, loaded_weight, shard_id)
            else:
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class Ernie5_5ForCausalLM(nn.Module, SupportsPP, SupportsEagle3):
    hf_to_vllm_mapper = ernie5_5_mapper

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_wqa_wkv": ["linear_q_down_proj", "linear_kv_proj"],
        "fused_wkv_wgate": ["linear_wkv", "linear_wgate"],
        "wk_weights_proj": ["wk", "weights_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self._model_path = getattr(vllm_config.model_config, "model", None) or getattr(
            config, "_name_or_path", ""
        )
        self.model = Ernie5_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = Ernie5_5ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        # Use LogitsProcessor with head_dtype=None so HTRLinearMethod runs natively;
        # Ernie5_5ParallelLMHead.multimax promotes logits to FP32 cleanly.
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.logits_processor.head_dtype = None
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model._set_aux_hidden_state_layers(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return tuple(getattr(self.config, "dspark_target_layer_ids", (40, 41, 42)))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None
        return self.lm_head.multimax(logits)

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states, skip_gather=True)
        return self.lm_head.multimax(logits)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Handle lm_head.multimax_{ranges,ts} buffers cleanly alongside AutoWeightsLoader
        lm_buffers = {}
        if isinstance(self.lm_head, Ernie5_5ParallelLMHead):
            lm_buffers = {
                "lm_head.multimax_ranges": self.lm_head.multimax_ranges,
                "lm_head.multimax_ts": self.lm_head.multimax_ts,
            }

        loaded_bufs: set[str] = set()

        def _filtered_weights():
            for orig_name, tensor in weights:
                if orig_name == "model.lm_head.multimax_ranges" and "lm_head.multimax_ranges" in lm_buffers:
                    lm_buffers["lm_head.multimax_ranges"].copy_(tensor)
                    loaded_bufs.add("lm_head.multimax_ranges")
                    continue
                if orig_name == "model.lm_head.multimax_ts" and "lm_head.multimax_ts" in lm_buffers:
                    lm_buffers["lm_head.multimax_ts"].copy_(tensor)
                    loaded_bufs.add("lm_head.multimax_ts")
                    continue
                yield orig_name, tensor

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(_filtered_weights(), mapper=self.hf_to_vllm_mapper)
        from .flashindexer import is_flashindexer_enabled, load_flashindexer_weights
        if is_flashindexer_enabled(self.config):
            loaded_bufs |= load_flashindexer_weights(self, self._model_path)
        return loaded | loaded_bufs

    def process_weights_after_loading(self) -> None:
        fold_vha(self)


