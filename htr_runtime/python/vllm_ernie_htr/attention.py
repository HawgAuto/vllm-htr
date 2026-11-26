# SPDX-License-Identifier: Apache-2.0
"""ROCm gfx1151-native HCA and DSA attention for ERNIE-5.5-Flash (HTR-W1).

- HCA (37/43 layers): subclasses vLLM's built-in ``DeepseekV4ROCMAiterMLAAttention``
  (pure-Triton ``rocm_sparse_attn_decode`` / ``rocm_sparse_attn_prefill`` with
  ``attn_sink``, ``DeepseekV4SWACache``, ``DeepseekCompressor``), adds ERNIE's
  ``VHAPostMix``, ``o_gate`` (K5 HTR), HTR-quantized compressor linear, and
  8-group HTR-W1 ``wo_a_groups`` (``linear_o_group_proj.g0..g7``).
- DSA (6/43 layers): pure-Triton sparse MLA on ``gfx1151`` with ``attn_sink``,
  ``Ernie5_5IndexerROCm`` (``hadacore_transform`` + ``rocm_aiter_sparse_attn_indexer``
  + ``merge_swa_topk``), ``VHAPostMix``, ``o_gate``, and HTR-W1 ``q_a_proj`` /
  ``q_b_proj`` / ``kv_b_proj`` / ``wo_b`` alongside BF16 ``kv_a_proj_with_mqa``.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from typing import Any

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v4.amd.rocm import (
    DeepseekV4ROCMAiterMLAAttention,
    DeepseekV4ROCMAiterMLASparseBackend,
)
from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _rocm_sparse_attn_prefill_triton,
)
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from htr import ops as htr_ops
from htr import wire as htr_wire
from .fast_ops import get_dsa_attn, get_fast_ops, get_split_kv_scratch

logger = init_logger(__name__)

VHA_RANK = 16
DSA_BLOCK_SIZE = 64
INDEXER_QUANT_BLOCK = 128

import os as _os
import sys as _sys

_USE_NATIVE_WOA = _os.environ.get("ERNIE_NATIVE_WOA", "1") == "1"
_USE_VHA_HIP = _os.environ.get("ERNIE_VHA_HIP", "1") == "1"
_WOA_NS = int(_os.environ.get("ERNIE_WOA_NS", "8"))
_WOA_EXT = None
_WOA_EXT_TRIED = False
_WOA_SCRATCH: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_WOA_PREFILL_SCRATCH: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_HAD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
_DSA_SPLIT_SCRATCH: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _dsa_split_scratch(dev: torch.device, T: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return get_split_kv_scratch(dev, T, H=64, S=8, D=512)


def _woa_ext():
    global _WOA_EXT, _WOA_EXT_TRIED
    if not _USE_NATIVE_WOA:
        return None
    if not _WOA_EXT_TRIED:
        _WOA_EXT_TRIED = True
        so_dir = _os.environ.get("ERNIE_HTR_MOE_SO_DIR")
        if so_dir and so_dir not in _sys.path:
            _sys.path.insert(0, so_dir)
        pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
        cands = [
            pkg_dir,
            _os.path.join(_os.path.dirname(pkg_dir), "build"),
            _os.path.join(_os.path.dirname(_os.path.dirname(pkg_dir)), "build"),
        ]
        native_dir = _os.environ.get("HTR_NATIVE_DIR")
        if native_dir:
            cands.append(native_dir)
        for cand in cands:
            if _os.path.isfile(_os.path.join(cand, "htr_wo_a_ernie.so")) and cand not in _sys.path:
                _sys.path.insert(0, cand)
        try:
            import htr_wo_a_ernie
            _WOA_EXT = htr_wo_a_ernie
            print(f"ERNIE_NATIVE_WOA_ACTIVE {htr_wo_a_ernie.__file__}", flush=True)
        except Exception as exc:
            print(f"ERNIE_NATIVE_WOA_FAILED {exc!r}", flush=True)
            _WOA_EXT = None
    return _WOA_EXT


def _woa_scratch(dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = str(dev)
    pair = _WOA_SCRATCH.get(key)
    if pair is None:
        # G=8, max_ns=8, max_M=8, M_OUT=1024 -> 2 MiB fp32; counters: 64 int32
        part = torch.empty(8 * 8 * 8 * 1024, dtype=torch.float32, device=dev)
        cnt = torch.zeros(8 * 8, dtype=torch.int32, device=dev)
        pair = _WOA_SCRATCH[key] = (part, cnt)
    return pair


def _woa_prefill_scratch(dev: torch.device, T: int) -> tuple[torch.Tensor, torch.Tensor]:
    key = str(dev)
    pair = _WOA_PREFILL_SCRATCH.get(key)
    if pair is None:
        # Fixed cap=512 (32 MiB) allocated once -> zero reallocations across all T>8 / graph captures!
        xt = torch.empty((512, 8, 4096), dtype=torch.float16, device=dev)
        tok = torch.empty((512, 8), dtype=torch.float32, device=dev)
        pair = _WOA_PREFILL_SCRATCH[key] = (xt, tok)
    return pair[0][:T], pair[1][:T]


def hadamard_rows(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard transform per row (Wave32 HIP FWHT-128 or cached H)."""
    if x.shape[0] == 0:
        return x
    if x.is_cuda and x.shape[1] == 128 and x.dtype == torch.bfloat16:
        fops = get_fast_ops()
        if fops is not None:
            return fops.fwht128(x.contiguous())
    key = (x.shape[1], str(x.device), x.dtype)
    H = _HAD_CACHE.get(key)
    if H is None:
        H = htr_wire.hadamard(x.shape[1], x.device).to(x.dtype).contiguous()
        _HAD_CACHE[key] = H
    return torch.matmul(x, H)


class VHAPostMix(nn.Module):
    """Head mixing ``o[h] <- sum_g m[h, g] * o[g]`` with ``m = I + V @ U^T``."""

    def __init__(self, num_heads: int, rank: int = VHA_RANK) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        num_local_heads = num_heads // self.tp_size
        self.U = nn.Parameter(torch.zeros(num_local_heads, rank), requires_grad=False)
        self.V = nn.Parameter(torch.zeros(num_local_heads, rank), requires_grad=False)
        for param in (self.U, self.V):
            set_weight_attrs(param, {"weight_loader": self._load_local_heads})
        if self.tp_size == 1:
            self.register_buffer("m", torch.eye(num_heads), persistent=False)
            self._m_bf16: torch.Tensor | None = None

    def _load_local_heads(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        rows = param.shape[0]
        param.data.copy_(loaded[self.tp_rank * rows : (self.tp_rank + 1) * rows])

    def fold(self) -> None:
        if self.tp_size != 1:
            return
        eye = torch.eye(self.U.shape[0], dtype=self.U.dtype, device=self.U.device)
        self.m.copy_(torch.addmm(eye, self.V, self.U.t()))
        self._m_bf16 = self.m.to(torch.bfloat16).contiguous()

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            if o.dtype == torch.bfloat16 and self._m_bf16 is not None:
                if _USE_VHA_HIP and o.is_cuda and o.dim() == 3 and o.shape[1] == 64 and (o.shape[2] % 64 == 0) and o.stride(2) == 1:
                    fops = get_fast_ops()
                    if fops is not None and hasattr(fops, "vha_mix_bf16"):
                        return fops.vha_mix_bf16(self._m_bf16, o)
                return torch.matmul(self._m_bf16, o)
            return torch.matmul(self.m.to(o.dtype), o)
        num_tokens = o.shape[0]
        u = self.U.t().expand(num_tokens, -1, -1)
        z = tensor_model_parallel_all_reduce(torch.bmm(u, o, out_dtype=torch.float32))
        v = self.V.float().expand(num_tokens, -1, -1)
        return torch.bmm(v, z).add_(o).to(o.dtype)


def fold_vha(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, VHAPostMix):
            module.fold()
        elif isinstance(module, Ernie5_5HCAAttention):
            module.pack_wo_a()
    # In-place 16B+4B split-plane repack of ERNIE-5.5 K=5 dense weights (0 extra steady-state VRAM!)
    from .patches import repack_ernie_k5_splitplane_inplace
    repack_ernie_k5_splitplane_inplace(model)


class Ernie5_5Compressor(DeepseekCompressor):
    """Keeps C128 compression safe in piecewise graphs and skips non-boundary compress on ROCm."""

    def forward(self, kv_score, positions, rotary_emb):
        from vllm.config.compilation import CUDAGraphMode
        forward_context = get_forward_context()
        metadata = forward_context.attn_metadata
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and isinstance(metadata, dict):
            name = self.state_cache.prefix
            state = metadata.get(name)
            if state is not None and state.c128_boundary is False:
                metadata[name] = dataclasses.replace(state, c128_boundary=None)
        elif (
            isinstance(metadata, dict)
            and self.head_dim == 512
            and self.compress_ratio == 128
            and forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL
        ):
            state_metadata = metadata.get(self.state_cache.prefix)
            if state_metadata is not None and state_metadata.c128_boundary is False:
                # ROCm fast-path: save_partial_states directly, skip compress_norm_rope_store_triton!
                kv, score = kv_score.split(
                    [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
                )
                state_cache = self.state_cache.kv_cache
                state_width = state_cache.shape[-1] // 2
                fops = get_fast_ops()
                if (
                    fops is not None
                    and kv.is_cuda
                    and self.ape.dtype == torch.float32
                    and state_cache.dtype == torch.float32
                ):
                    fops.save_partial_states(
                        kv, score, self.ape, positions, state_cache,
                        state_metadata.slot_mapping, int(state_metadata.block_size),
                        int(state_width), int(self.compress_ratio),
                    )
                    return
                from vllm.models.deepseek_v4.compressor import _SAVE_PARTIAL_STATES_KERNEL
                _SAVE_PARTIAL_STATES_KERNEL(
                    kv=kv, score=score, ape=self.ape, positions=positions,
                    state_cache=state_cache, slot_mapping=state_metadata.slot_mapping,
                    block_size=state_metadata.block_size, state_width=state_width,
                    compress_ratio=self.compress_ratio, pdl_kwargs={},
                )
                return
        return super().forward(kv_score, positions, rotary_emb)



def _resolve_ernie_dsa_kv_dtype(vllm_config: VllmConfig) -> str:
    """Resolve ERNIE-5.5 6-layer DSA KV cache format: auto (BF16), fp8_ds_mla (656B), nvfp4_ds_mla (352B)."""
    env = _os.environ.get("ERNIE_DSA_KV_DTYPE", "").strip().lower()
    if env in ("bf16", "bfloat16", "auto"):
        return "auto"
    if env in ("fp8", "fp8_e4m3", "fp8_ds_mla"):
        return "fp8_ds_mla"
    if env in ("fp4", "nvfp4", "nvfp4_ds_mla"):
        return "nvfp4_ds_mla"
    cc = vllm_config.cache_config
    raw = str(getattr(cc, "_ernie_orig_cache_dtype", None) or cc.cache_dtype or "auto").lower()
    if raw in ("fp8", "fp8_e4m3", "fp8_ds_mla"):
        return "fp8_ds_mla"
    if raw in ("nvfp4", "nvfp4_ds_mla"):
        return "nvfp4_ds_mla"
    return "auto"


def _resolve_ernie_indexer_uses_mxfp4(vllm_config: VllmConfig) -> bool:
    """Whether ERNIE-5.5 DSA indexer K cache uses 68B MXFP4 (vs 132B FP8)."""
    env = _os.environ.get("ERNIE_INDEXER_KV_DTYPE", "").strip().lower()
    if env in ("mxfp4", "fp4", "nvfp4"):
        return True
    if env in ("fp8", "fp8_e4m3"):
        return False
    ac = getattr(vllm_config, "attention_config", None)
    if ac is not None and str(getattr(ac, "indexer_kv_dtype", "")).lower() == "mxfp4":
        return True
    return _resolve_ernie_dsa_kv_dtype(vllm_config) == "nvfp4_ds_mla"


def _resolve_ernie_hca_c128_block_size(vllm_config: VllmConfig) -> int:
    """Auto-size 37-layer HCA c128 block_size to fit the 6-layer DSA anchor block in 1 group."""
    env = _os.environ.get("ERNIE_HCA_C128_BLOCK_SIZE", "").strip()
    if env:
        bs = int(env)
        if bs >= 256 and bs % 128 == 0:
            return bs
    dsa_dt = _resolve_ernie_dsa_kv_dtype(vllm_config)
    dsa_bpt = 656 if dsa_dt == "fp8_ds_mla" else (352 if dsa_dt == "nvfp4_ds_mla" else 1152)
    idx_bpt = 68 if _resolve_ernie_indexer_uses_mxfp4(vllm_config) else 132
    anchor = 6 * DSA_BLOCK_SIZE * (dsa_bpt + idx_bpt)
    # Largest power-of-2 multiple of 256 whose 37 c128 layers fit inside 1 DSA anchor block
    for bs in (2048, 1024, 512, 256):
        states = bs // 128
        pg = ((states * 584 + 575) // 576) * 576
        if 37 * pg <= anchor:
            return bs
    return 256


class Ernie5_5HCASparseMLABackend(DeepseekV4ROCMAiterMLASparseBackend):
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256, 512, 1024, 2048]


class Ernie5_5HCAAttention(DeepseekV4ROCMAiterMLAAttention):
    backend_cls = Ernie5_5HCASparseMLABackend
    """ERNIE-5.5 HCA layer on ROCm gfx1151 with HTR-W1 trunk weights.

    Reuses ``DeepseekV4ROCMAiterMLAAttention`` (Triton sparse decode/prefill
    with ``attn_sink``, ``DeepseekV4SWACache``, ``DeepseekCompressor``) and adds:
      - HTR-W1 K5 ``compressor.fused_wkv_wgate`` (replaces unquantized default)
      - 8-group HTR-W1 K5 ``wo_a_groups`` (``linear_o_group_proj.g0..g7``)
      - HTR-W1 K5 ``o_gate`` and ``VHAPostMix``
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str, **kwargs) -> None:
        # Ensure all gfx1151 ROCm HIP hooks are active on the GPU worker process
        from .patches import _patch_rocm_paged_mqa_logits_triton
        _patch_rocm_paged_mqa_logits_triton()
        # Preserve original user cache_dtype before DeepseekV4Attention mutates it to fp8_ds_mla,
        # and isolate HCA to fp8_ds_mla so --kv-cache-dtype nvfp4_ds_mla works cleanly for DSA!
        if not hasattr(vllm_config.cache_config, "_ernie_orig_cache_dtype"):
            vllm_config.cache_config._ernie_orig_cache_dtype = vllm_config.cache_config.cache_dtype
        hca_vllm_cfg = copy.copy(vllm_config)
        hca_cc = copy.copy(vllm_config.cache_config)
        hca_cc.cache_dtype = "fp8_ds_mla"
        hca_vllm_cfg.cache_config = hca_cc
        # Disable ROCm side streams so we never touch .weight.T on HTR linears
        kwargs["aux_stream_list"] = None
        super().__init__(hca_vllm_cfg, prefix, **kwargs)
        self.aux_stream_list = None
        quant_config = vllm_config.quant_config

        # Replace unquantized compressor.fused_wkv_wgate with quant-aware HTR linear
        if self.compressor is not None:
            self.compressor.__class__ = Ernie5_5Compressor
            self.compressor.fused_wkv_wgate = MergedColumnParallelLinear(
                self.compressor.hidden_size,
                [
                    self.compressor.coff * self.compressor.head_dim,
                    self.compressor.coff * self.compressor.head_dim,
                ],
                bias=False,
                return_bias=False,
                quant_config=quant_config,
                disable_tp=True,
                prefix=f"{prefix}.compressor.fused_wkv_wgate",
            )

        # Replace single BF16 wo_a with 8 separate K5 HTR group linears
        del self.wo_a
        group_in = self.n_local_heads * self.head_dim // self.n_local_groups
        self.wo_a_groups = nn.ModuleList(
            [
                ReplicatedLinear(
                    group_in,
                    self.o_lora_rank,
                    bias=False,
                    quant_config=quant_config,
                    return_bias=False,
                    prefix=f"{prefix}.wo_a_groups.{g}",
                )
                for g in range(self.n_local_groups)
            ]
        )

        self.o_gate = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.o_gate",
        )
        self.vha = VHAPostMix(self.n_heads)
        self._woa_packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._woa_sp16: bool = False
        try:
            _hca_cs = getattr(getattr(self, "rotary_emb", None), "cos_sin_cache", None)
            if _hca_cs is not None and globals().get("_ERNIE5_TARGET_HCA_COS_SIN") is None:
                globals()["_ERNIE5_TARGET_HCA_COS_SIN"] = _hca_cs
        except Exception:
            pass

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        spec = super().get_kv_cache_spec(vllm_config)
        if spec is not None and self.compress_ratio == 128:
            c128_bs = _resolve_ernie_hca_c128_block_size(vllm_config)
            if c128_bs != spec.block_size:
                spec = dataclasses.replace(spec, block_size=c128_bs)
        return spec

    def pack_wo_a(self) -> None:
        """Stack the 8 HTR-K5 wo_a_groups into contiguous tensors (+ 16B+4B split-plane repack, 0 extra VRAM)."""
        ext = _woa_ext()
        if self._woa_packed is not None or self.n_local_groups != 8 or ext is None:
            return
        if not all(hasattr(g, "htr_qweight") for g in self.wo_a_groups):
            return
        g0 = self.wo_a_groups[0]
        if tuple(g0.htr_qweight.shape) != (64, 256, 40):
            return
        qw = torch.stack([g.htr_qweight.data for g in self.wo_a_groups], dim=0).contiguous()
        sc = torch.stack([g.htr_scales.data for g in self.wo_a_groups], dim=0).contiguous()
        su = torch.stack([g.htr_su.data for g in self.wo_a_groups], dim=0).contiguous()
        sv = torch.stack([g.htr_sv.data for g in self.wo_a_groups], dim=0).contiguous()
        if _os.environ.get("ERNIE_WOA_SP16", "0") == "1" and hasattr(ext, "repack_wo_a_sp16") and qw.is_cuda:
            qw = ext.repack_wo_a_sp16(qw)
            self._woa_sp16 = True
        # Rebind each group's parameters to slices of the stacked buffer (zero duplicate VRAM!)
        for idx, g in enumerate(self.wo_a_groups):
            g.htr_qweight.data = qw[idx]
            g.htr_scales.data = sc[idx]
            g.htr_su.data = su[idx]
            g.htr_sv.data = sv[idx]
        self._woa_packed = (qw, sc, su, sv)

    def _run_parallel_input_projections(self, hidden_states: torch.Tensor):
        qr_kv = self.fused_wqa_wkv(hidden_states)
        if isinstance(qr_kv, tuple):
            qr_kv = qr_kv[0]
        if self.compressor is not None:
            kv_score = self.compressor.fused_wkv_wgate(hidden_states)
            if get_fast_ops() is None:
                kv_score = kv_score.float()
        else:
            kv_score = None
        return qr_kv, kv_score, None, None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self._run_parallel_input_projections(hidden_states)
        )
        qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)
        gate_logits = self.o_gate(qr)

        # Runs compressor + Q norm/RoPE + SWA insert + sparse MLA + inverse RoPE
        self._prepare_and_attn_fn(
            hidden_states,
            qr,
            kv,
            qr_scale,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            o_padded,
        )
        # o_padded is already inverse-RoPE'd by DeepseekV4ROCMAiterMLAAttention.forward_mqa;
        # VHA operates on the head axis so it commutes with inverse RoPE.
        o = self.vha(o_padded[:, : self.n_local_heads, :])
        o_groups = o.reshape(num_tokens, self.n_local_groups, -1)

        if (
            self._woa_packed is not None
            and num_tokens > 0
            and o_groups.dtype == torch.bfloat16
        ):
            ext = _woa_ext()
            qw, sc, su, sv = self._woa_packed
            o_groups = o_groups.contiguous()
            gl = gate_logits.contiguous()
            z_gated = torch.empty(
                (num_tokens, self.n_local_groups * self.o_lora_rank),
                dtype=o_groups.dtype,
                device=o_groups.device,
            )
            if num_tokens <= 8:
                part, cnt = _woa_scratch(o_groups.device)
                ext.wo_a_k5_batched(
                    qw, sc, su, sv, o_groups, gl, part, cnt, z_gated,
                    _WOA_NS, float(htr_ops.CB), 0, self._woa_sp16,
                )
            elif hasattr(ext, "wo_a_k5_prefill_wmma"):
                # Batched 8-group K=5 WMMA + SAD_HI + fused sigmoid(gate) for ALL T > 8!
                for i in range(0, num_tokens, 512):
                    j = min(i + 512, num_tokens)
                    xt_b, tok_b = _woa_prefill_scratch(o_groups.device, j - i)
                    ext.wo_a_k5_prefill_wmma(
                        qw, sc, su, sv, o_groups[i:j], gl[i:j], xt_b, tok_b, z_gated[i:j],
                        float(htr_ops.CB), 0, self._woa_sp16,
                    )
            else:
                part, cnt = _woa_scratch(o_groups.device)
                for i in range(0, num_tokens, 8):
                    j = min(i + 8, num_tokens)
                    ext.wo_a_k5_batched(
                        qw, sc, su, sv, o_groups[i:j], gl[i:j], part, cnt, z_gated[i:j],
                        _WOA_NS, float(htr_ops.CB), 0, self._woa_sp16,
                    )
            return self.wo_b(z_gated)

        gate = torch.sigmoid(gate_logits)
        z = torch.cat(
            [grp(o_groups[:, g, :]) for g, grp in enumerate(self.wo_a_groups)],
            dim=-1,
        )
        return self.wo_b(z * gate)


# ------------------------------------------------------------------ DSA Attention (6 layers)

def packed_block_stride_alignment(vllm_config: VllmConfig) -> int:
    hf_config = vllm_config.model_config.hf_config
    hca_row = hf_config.head_dim * 2
    if _resolve_ernie_dsa_kv_dtype(vllm_config) in ("fp8_ds_mla", "nvfp4_ds_mla"):
        return math.lcm(512, hca_row)
    dsa_row = (hf_config.mla_kv_lora_rank + hf_config.mla_qk_rope_head_dim) * 2
    return math.lcm(512, hca_row, dsa_row)


def _dsa_page_spec(spec: KVCacheSpec, vllm_config: VllmConfig) -> KVCacheSpec:
    return dataclasses.replace(
        spec,
        block_size=DSA_BLOCK_SIZE,
        block_stride_alignment=packed_block_stride_alignment(vllm_config),
    )


def _dsa_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
    return [MultipleOf(DSA_BLOCK_SIZE)]


@triton.jit
def _merge_swa_topk_kernel(
    topk_ptr,
    out_ptr,
    positions_ptr,
    topk_stride,
    out_stride,
    WINDOW: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    seq_len = tl.load(positions_ptr + row).to(tl.int32) + 1
    indexer_len = tl.minimum(tl.maximum(seq_len - WINDOW, 0), TOPK)
    swa_len = tl.minimum(seq_len, WINDOW)
    cols = tl.arange(0, BLOCK)
    is_topk = cols < indexer_len
    topk = tl.load(topk_ptr + row * topk_stride + cols, mask=is_topk, other=-1)
    swa_idx = seq_len - swa_len + cols - indexer_len
    vals = tl.where(is_topk, topk, tl.where(cols < indexer_len + swa_len, swa_idx, -1))
    tl.store(out_ptr + row * out_stride + cols, vals, mask=cols < WIDTH)


def merge_swa_topk(
    topk_indices: torch.Tensor,
    out: torch.Tensor,
    positions: torch.Tensor,
    window: int,
) -> None:
    num_tokens = positions.shape[0]
    if num_tokens == 0:
        return
    topk, width = topk_indices.shape[1], out.shape[1]
    assert width >= topk + window
    if topk_indices.is_cuda and positions.dtype == torch.int64:
        fops = get_fast_ops()
        if fops is not None:
            fops.merge_swa_topk(topk_indices, out, positions, window)
            return
    _merge_swa_topk_kernel[(num_tokens,)](
        topk_indices,
        out,
        positions,
        topk_indices.stride(0),
        out.stride(0),
        WINDOW=window,
        TOPK=topk,
        WIDTH=width,
        BLOCK=triton.next_power_of_2(width),
    )


@triton.jit
def _req_to_global_slots_kernel(
    req_ids_ptr,       # [T] int32
    block_table_ptr,   # [num_reqs, max_blocks] int32
    local_idx_ptr,     # [T, W] int32 (request-local token positions, -1 = invalid)
    global_idx_ptr,    # [T, W] int32 (output flat row indices, -1 = invalid)
    bt_stride0,
    li_stride0,
    gi_stride0,
    max_blocks,
    block_stride_rows,
    BLOCK_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    t = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = cols < WIDTH
    req = tl.load(req_ids_ptr + t)
    tok = tl.load(local_idx_ptr + t * li_stride0 + cols, mask=mask, other=-1)
    valid = mask & (tok >= 0)
    blk_i = tok // BLOCK_SIZE
    in_blk = tok % BLOCK_SIZE
    safe_blk_i = tl.where(valid & (blk_i < max_blocks), blk_i, 0)
    phys_blk = tl.load(
        block_table_ptr + req * bt_stride0 + safe_blk_i,
        mask=valid & (blk_i < max_blocks),
        other=-1,
    )
    gslot = tl.where(valid & (phys_blk >= 0), phys_blk * block_stride_rows + in_blk, -1)
    tl.store(global_idx_ptr + t * gi_stride0 + cols, gslot, mask=mask)


def req_local_to_global_slots(
    local_indices: torch.Tensor,
    req_id_per_token: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    block_stride_rows: int,
) -> torch.Tensor:
    T, W = local_indices.shape
    if T == 0:
        return torch.empty_like(local_indices)
    if local_indices.is_cuda:
        fops = get_fast_ops()
        if fops is not None:
            return fops.req_to_global(local_indices, req_id_per_token, block_table, block_size, block_stride_rows)
    out = torch.empty_like(local_indices)
    block_w = 256
    _req_to_global_slots_kernel[(T, triton.cdiv(W, block_w))](
        req_id_per_token,
        block_table,
        local_indices,
        out,
        block_table.stride(0),
        local_indices.stride(0),
        out.stride(0),
        block_table.shape[1],
        block_stride_rows,
        BLOCK_SIZE=block_size,
        WIDTH=W,
        BLOCK_W=block_w,
    )
    return out


class Ernie5_5IndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    """Excludes the trailing sliding window from the indexer KV range."""

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        spec_config = vllm_config.speculative_config
        if spec_config is not None and spec_config.enable_adaptive_verification:
            return AttentionCGSupport.ALWAYS
        return super().get_cudagraph_support(vllm_config, kv_cache_spec)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        spec_config = self.vllm_config.speculative_config
        if spec_config is not None and spec_config.enable_adaptive_verification:
            self.use_flattening = True
        self.window = self.vllm_config.model_config.hf_config.sliding_window
        self.windowed_seq_lens_buffer = torch.zeros_like(self.decode_seq_lens_buffer)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if metadata.prefill is not None:
            metadata.prefill.chunks = [
                dataclasses.replace(
                    chunk,
                    cu_seqlen_ke=torch.maximum(
                        chunk.cu_seqlen_ke - self.window, chunk.cu_seqlen_ks
                    ),
                )
                for chunk in metadata.prefill.chunks
            ]
        if metadata.decode is not None:
            seq_lens = metadata.decode.seq_lens
            windowed = self.windowed_seq_lens_buffer[: seq_lens.numel()].view(
                seq_lens.shape
            )
            torch.sub(seq_lens, self.window, out=windowed)
            windowed.clamp_min_(0)
            metadata.decode = dataclasses.replace(
                metadata.decode,
                seq_lens=windowed,
            )
        return metadata


class Ernie5_5IndexerBackend(DeepseekV32IndexerBackend):
    get_supported_kernel_block_sizes = staticmethod(_dsa_kernel_block_sizes)

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # ROCm runs adaptive verification through the per-token flattened
        # indexer path, deriving row ownership purely from device decode lengths.
        from vllm.platforms import current_platform
        return (
            current_platform.is_rocm()
            or super().supports_device_cpu_query_lens_mismatch()
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.BLNHC, KVCacheLayout.BLHNC)

    @staticmethod
    def get_builder_cls() -> type[Ernie5_5IndexerMetadataBuilder]:
        return Ernie5_5IndexerMetadataBuilder


class Ernie5_5IndexerCache(DeepseekV32IndexerCache):
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return _dsa_page_spec(super().get_kv_cache_spec(vllm_config), vllm_config)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Ernie5_5IndexerBackend


class Ernie5_5IndexerROCm(nn.Module):
    """Lightning Indexer scoring tokens older than the sliding window on ROCm."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        cache_config: CacheConfig,
        topk_indices_buffer: torch.Tensor | None,
        swa_topk_buffer: torch.Tensor | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.hidden_size = config.hidden_size
        self.rope_dim = config.mla_qk_rope_head_dim
        # Indexer weights are kept verbatim in BF16 in ERNIE-5.5 HTR
        self.wq_b = ReplicatedLinear(
            config.mla_q_lora_rank,
            self.n_head * self.head_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.wq_b",
        )
        self.wk_weights_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=int(getattr(vllm_config.model_config, 'max_model_len', config.max_position_embeddings)) + 256,
            is_neox_style=True,
            rope_parameters={
                **config.mla_rope_parameters,
                "partial_rotary_factor": self.rope_dim / self.head_dim,
            },
        )
        self.weights_scale = self.head_dim**-0.5 * self.n_head**-0.5
        self.topk_indices_buffer = swa_topk_buffer
        # Indexer K cache: 68 B/tok (MXFP4: 64 packed E2M1 + 4 UE8M0) or 132 B/tok (FP8: 128 + 4 FP32)
        self.indexer_uses_mxfp4 = _resolve_ernie_indexer_uses_mxfp4(vllm_config)
        if self.indexer_uses_mxfp4:
            k_cache_head_dim = self.head_dim // 2 + self.head_dim // 32
        else:
            k_cache_head_dim = self.head_dim + self.head_dim // INDEXER_QUANT_BLOCK * 4
        self.k_cache = Ernie5_5IndexerCache(
            head_dim=k_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        try:
            import vllm.v1.attention.ops.rocm_aiter_mla_sparse as _rmod
            _sc = getattr(vllm_config, "scheduler_config", None)
            _sp = getattr(vllm_config, "speculative_config", None)
            _mns = int(getattr(_sc, "max_num_seqs", 0) or 0)
            _ns = int(getattr(_sp, "num_speculative_tokens", 0) or 0)
            if _mns > 0:
                _rmod._ernie5_max_decode_rows = _mns * (1 + _ns)
        except Exception:
            pass
        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            INDEXER_QUANT_BLOCK,
            "ue8m0",
            config.index_topk,
            self.head_dim,
            vllm_config.model_config.max_model_len,
            min(
                get_max_prefill_buffer_size(vllm_config),
                vllm_config.model_config.max_model_len,
            ),
            topk_indices_buffer,
            skip_k_cache_insert=False,
            use_fp4_cache=False,
        )

    def forward(
        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        fops = get_fast_ops()
        T = hidden_states.shape[0]
        use_nt = (
            1 <= T <= 8
            and _os.environ.get("ERNIE_BF16_GEMV_NT", "1") == "1"
            and fops is not None
            and hasattr(fops, "bf16_gemv_nt")
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
        )
        if use_nt and qr.dtype == torch.bfloat16 and self.wq_b.weight.dtype == torch.bfloat16 and (qr.shape[1] % 256) == 0:
            q = fops.bf16_gemv_nt(qr.contiguous(), self.wq_b.weight, True)
        else:
            q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        if use_nt and self.wk_weights_proj.weight.dtype == torch.bfloat16 and (hidden_states.shape[1] % 256) == 0:
            kw = fops.bf16_gemv_nt(hidden_states.contiguous(), self.wk_weights_proj.weight, True)
        else:
            kw, _ = self.wk_weights_proj(hidden_states)
        k, weights = kw.split([self.head_dim, self.n_head], dim=-1)
        cs = getattr(self.rotary_emb, "cos_sin_cache", None)
        if (
            fops is not None
            and cs is not None
            and hasattr(fops, "rope_neox64_fwht128_ue8m0_quant_weights")
            and q.is_cuda
            and q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            and weights.dtype == torch.bfloat16
            and self.n_head == 64
            and self.head_dim == 128
            and self.rope_dim == 64
            and getattr(self.rotary_emb, "is_neox_style", True)
            and hasattr(self.k_norm, "weight")
        ):
            eps = float(getattr(self.k_norm, "eps", getattr(self.k_norm, "variance_epsilon", 1e-6)))
            k = fops.layernorm_rope_neox64_fwht128(
                k, self.k_norm.weight, getattr(self.k_norm, "bias", None),
                positions, cs, eps,
            )
            q, weights = fops.rope_neox64_fwht128_ue8m0_quant_weights(
                q.contiguous(), weights, positions, cs, float(self.weights_scale),
            )
            return self.indexer_op(hidden_states, q, k, weights)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        k = hadamard_rows(k)
        if (
            fops is not None
            and hasattr(fops, "fwht128_ue8m0_quant_weights")
            and q.is_cuda
            and q.dtype == torch.bfloat16
            and weights.dtype == torch.bfloat16
            and self.n_head == 64
            and self.head_dim == 128
        ):
            q, weights = fops.fwht128_ue8m0_quant_weights(
                q.contiguous(), weights, float(self.weights_scale)
            )
            return self.indexer_op(hidden_states, q, k, weights)
        q = hadamard_rows(q.view(-1, self.head_dim))
        q, q_scale = per_token_group_quant_fp8(
            q, INDEXER_QUANT_BLOCK, column_major_scales=False, use_ue8m0=True
        )
        q = q.view(-1, self.n_head, self.head_dim)
        weights = weights.float() * q_scale.view(-1, self.n_head) * self.weights_scale
        return self.indexer_op(hidden_states, q, k, weights)


@dataclasses.dataclass
class Ernie5_5ROCmSparseMLAMetadata(AttentionMetadata):
    num_actual_tokens: int
    slot_mapping: torch.Tensor
    req_id_per_token: torch.Tensor
    block_table_all: torch.Tensor
    block_size: int = DSA_BLOCK_SIZE
    num_decode_tokens: int = 0


class Ernie5_5ROCmSparseMLAMetadataBuilder(
    AttentionMetadataBuilder[Ernie5_5ROCmSparseMLAMetadata]
):
    # Static-shape GPU buffers + device-driven req_id_per_token + native HIP kernels
    # support both uniform AND varlen multi-token FULL HIP graph replay (DSpark adaptive verification!).
    _cudagraph_support = AttentionCGSupport.ALWAYS

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        max_toks = vllm_config.scheduler_config.max_num_batched_tokens
        self._req_id_buf = torch.zeros(max_toks, dtype=torch.int32, device=device)
        self._tok_idx = torch.arange(max_toks, dtype=torch.int32, device=device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Ernie5_5ROCmSparseMLAMetadata:
        num_tokens = common_attn_metadata.num_actual_tokens
        num_reqs = common_attn_metadata.num_reqs
        if num_tokens > 0 and num_reqs > 0:
            qsl_end = common_attn_metadata.query_start_loc[1 : num_reqs + 1]
            req_ids = (
                torch.searchsorted(qsl_end, self._tok_idx[:num_tokens], right=True)
                .clamp_(max=num_reqs - 1)
                .to(torch.int32)
            )
            self._req_id_buf[:num_tokens].copy_(req_ids)
        return Ernie5_5ROCmSparseMLAMetadata(
            num_actual_tokens=num_tokens,
            slot_mapping=common_attn_metadata.slot_mapping,
            req_id_per_token=self._req_id_buf[:num_tokens],
            block_table_all=common_attn_metadata.block_table_tensor,
            block_size=self.kv_cache_spec.block_size,
        )


@triton.jit
def _scatter_dsa_kv_kernel(
    kv_c_ptr,
    k_pe_ptr,
    kv_cache_ptr,
    slot_mapping_ptr,
    kvc_stride0,
    kpe_stride0,
    cache_stride0,
    cache_stride1,
    BLOCK_SIZE: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NOPE_PAD: tl.constexpr,
    ROPE_PAD: tl.constexpr,
):
    """Graph-safe scatter of [kv_c_normed, k_pe] into strided BLNHC KV cache."""
    t = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + t).to(tl.int64)
    if slot < 0:
        return
    blk = slot // BLOCK_SIZE
    off = slot % BLOCK_SIZE
    dst = kv_cache_ptr + blk * cache_stride0 + off * cache_stride1

    n_offs = tl.arange(0, NOPE_PAD)
    n_mask = n_offs < NOPE_DIM
    kvc = tl.load(kv_c_ptr + t * kvc_stride0 + n_offs, mask=n_mask)
    tl.store(dst + n_offs, kvc, mask=n_mask)

    r_offs = tl.arange(0, ROPE_PAD)
    r_mask = r_offs < ROPE_DIM
    kpe = tl.load(k_pe_ptr + t * kpe_stride0 + r_offs, mask=r_mask)
    tl.store(dst + NOPE_DIM + r_offs, kpe, mask=r_mask)


class Ernie5_5ROCmSparseMLAImpl(MLAAttentionImpl[Ernie5_5ROCmSparseMLAMetadata]):
    """Pure-Triton sparse MLA with attn_sink on ROCm gfx1151 (prefill + decode)."""

    is_sparse: bool = True
    supports_dense_mha_prefill: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: nn.Module,
        indexer: Any = None,
        topk_indices_buffer: torch.Tensor | None = None,
        sinks: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_b_proj = kv_b_proj
        self.indexer = indexer
        self.topk_indices_buffer = topk_indices_buffer
        self.sinks = sinks
        self.W_UK_T: torch.Tensor | None = None
        self.W_UV: torch.Tensor | None = None

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        kv_b = self.kv_b_proj
        if hasattr(kv_b, "htr_qweight"):
            # Fast native GPU dequant_w (~0.5 ms, ~28 MiB) instead of pure-PyTorch wire.dequant (~40 s, 1.9 GiB!)
            w = htr_ops.ext().dequant_w(
                kv_b.htr_qweight, kv_b.htr_scales, htr_ops.CB, kv_b.htr_su, kv_b.htr_sv
            ).to(act_dtype)
            # Free packed HTR weight buffers now that W_UK_T/W_UV are materialized
            for attr in ("htr_qweight", "htr_scales", "htr_su", "htr_sv"):
                if hasattr(kv_b, attr):
                    setattr(kv_b, attr, nn.Parameter(torch.empty(0, device=w.device), requires_grad=False))
        else:
            w = kv_b.weight.to(act_dtype)
        w = w.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        w_uk, w_uv = w.split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        # Keep K-contiguous row-major layouts for NT-streaming V_DUAL_DOT2ACC_F32_BF16 decode (0 extra VRAM!);
        # W_UK_T and W_UV are zero-copy transposed views natively handled by rocBLAS strided-batched BMM for prefill!
        self._w_uk_rm = w_uk.transpose(1, 2).contiguous()  # [64, 512, 128]
        self._w_uv_rm = w_uv.contiguous()                  # [64, 128, 512]
        self.W_UK_T = self._w_uk_rm.transpose(1, 2)        # [64, 128, 512] view
        self.W_UV = self._w_uv_rm.transpose(1, 2)          # [64, 512, 128] view

    def _scatter_kv_cache(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> None:
        num_tokens = min(kv_c_normed.shape[0], slot_mapping.shape[0])
        if num_tokens == 0 or kv_cache.numel() == 0:
            return
        k_pe_2d = k_pe[:num_tokens].squeeze(1)
        if kv_c_normed.is_cuda and self.kv_lora_rank == 512 and self.qk_rope_head_dim == 64 and kv_c_normed.dtype == torch.bfloat16:
            fops = get_fast_ops()
            if fops is not None:
                if self.kv_cache_dtype == "fp8_ds_mla" and kv_cache.dtype == torch.uint8:
                    fops.scatter_dsa_kv_fp8(kv_c_normed, k_pe_2d, kv_cache, slot_mapping, block_size)
                    return
                if self.kv_cache_dtype == "nvfp4_ds_mla" and kv_cache.dtype == torch.uint8:
                    fops.scatter_dsa_kv_fp4(kv_c_normed, k_pe_2d, kv_cache, slot_mapping, block_size)
                    return
                fops.scatter_dsa_kv(kv_c_normed, k_pe_2d, kv_cache, slot_mapping, block_size)
                return
        _scatter_dsa_kv_kernel[(num_tokens,)](
            kv_c_normed,
            k_pe_2d,
            kv_cache,
            slot_mapping,
            kv_c_normed.stride(0),
            k_pe_2d.stride(0),
            kv_cache.stride(0),
            kv_cache.stride(1),
            BLOCK_SIZE=block_size,
            NOPE_DIM=self.kv_lora_rank,
            ROPE_DIM=self.qk_rope_head_dim,
            NOPE_PAD=triton.next_power_of_2(self.kv_lora_rank),
            ROPE_PAD=triton.next_power_of_2(self.qk_rope_head_dim),
            num_warps=4,
        )

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        raise NotImplementedError("Handled directly in Ernie5_5ROCmSparseMLAImpl.forward")

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Ernie5_5ROCmSparseMLAMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            output = torch.empty(
                (q.shape[0], self.num_heads * self.v_head_dim),
                dtype=q.dtype,
                device=q.device,
            )
        if attn_metadata is None:
            return output.zero_()

        num_actual = attn_metadata.num_actual_tokens
        if num_actual == 0:
            return output

        # kv_cache may have shape [num_blocks, block_size, head_dim] with padded block stride
        if kv_cache.ndim == 4 and kv_cache.shape[1] == 1:
            kv_cache = kv_cache.squeeze(1)
        block_size = attn_metadata.block_size
        self._scatter_kv_cache(
            k_c_normed, k_pe, kv_cache, attn_metadata.slot_mapping, block_size
        )

        q_act = q[:num_actual]
        q_nope = q_act[..., : self.qk_nope_head_dim]
        q_pe = q_act[..., self.qk_nope_head_dim :]
        fops = get_fast_ops()
        use_nt_absorb = (
            1 <= num_actual <= 8
            and _os.environ.get("ERNIE_BF16_GEMV_NT", "1") == "1"
            and fops is not None
            and hasattr(fops, "dsa_q_absorb_cat")
            and hasattr(self, "_w_uk_rm")
            and q_act.is_cuda
            and q_act.dtype == torch.bfloat16
            and self._w_uk_rm.dtype == torch.bfloat16
            and self.num_heads == 64
            and self.qk_nope_head_dim == 128
            and self.kv_lora_rank == 512
            and self.qk_rope_head_dim == 64
        )
        if use_nt_absorb:
            q_abs = fops.dsa_q_absorb_cat(q_nope, q_pe, self._w_uk_rm)
        else:
            # bmm: [H, T, nope] @ [H, nope, kv_lora] -> [H, T, kv_lora] -> [T, H, kv_lora]
            ql_nope = torch.bmm(q_nope.transpose(0, 1), self.W_UK_T).transpose(0, 1)
            q_abs = torch.cat([ql_nope, q_pe], dim=-1).contiguous()

        dsa_ext = get_dsa_attn()
        local_idx = self.topk_indices_buffer[:num_actual]
        is_quant_dsa = (
            self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")
            and kv_cache.dtype == torch.uint8
        )
        if is_quant_dsa:
            assert dsa_ext is not None and hasattr(dsa_ext, "dsa_sparse_mla_quant"), (
                "Quantized DSA KV cache (fp8_ds_mla / nvfp4_ds_mla) requires native ernie_dsa_attn.so"
            )
            global_idx = req_local_to_global_slots(
                local_idx,
                attn_metadata.req_id_per_token,
                attn_metadata.block_table_all,
                block_size,
                block_size,
            )
            kv_fmt = 1 if self.kv_cache_dtype == "fp8_ds_mla" else 2
            if num_actual <= 32:
                po, pm, pl = _dsa_split_scratch(q_abs.device, num_actual)
                attn_latent = dsa_ext.dsa_sparse_mla_quant(
                    q_abs, kv_cache, global_idx, self.sinks, self.scale, po, pm, pl, 8, kv_fmt
                )
            else:
                attn_latent = dsa_ext.dsa_sparse_mla_quant(
                    q_abs, kv_cache, global_idx, self.sinks, self.scale, None, None, None, 1, kv_fmt
                )
        else:
            row_stride = kv_cache.stride(1)
            assert kv_cache.stride(0) % row_stride == 0, (kv_cache.shape, kv_cache.stride())
            block_stride_rows = kv_cache.stride(0) // row_stride
            head_dim = self.kv_lora_rank + self.qk_rope_head_dim
            total_rows = (kv_cache.shape[0] - 1) * block_stride_rows + kv_cache.shape[1]
            kv_flat = kv_cache.as_strided(
                (total_rows, head_dim),
                (row_stride, kv_cache.stride(2)),
            )
            global_idx = req_local_to_global_slots(
                local_idx,
                attn_metadata.req_id_per_token,
                attn_metadata.block_table_all,
                block_size,
                block_stride_rows,
            )
            if (
                dsa_ext is not None
                and q_abs.is_cuda
                and q_abs.dtype == torch.bfloat16
                and self.num_heads == 64
                and self.kv_lora_rank == 512
                and self.qk_rope_head_dim == 64
            ):
                if num_actual <= 32:
                    po, pm, pl = _dsa_split_scratch(q_abs.device, num_actual)
                    attn_latent = dsa_ext.dsa_sparse_mla(
                        q_abs, kv_flat, global_idx, self.sinks, self.scale, po, pm, pl, 8
                    )
                else:
                    attn_latent = dsa_ext.dsa_sparse_mla(
                        q_abs, kv_flat, global_idx, self.sinks, self.scale, None, None, None, 1
                    )
            else:
                attn_latent = _rocm_sparse_attn_prefill_triton(
                    q=q_abs,
                    kv=kv_flat,
                    indices=global_idx,
                    scale=self.scale,
                    attn_sink=self.sinks,
                    nope_head_dim=self.kv_lora_rank,
                    rope_head_dim=self.qk_rope_head_dim,
                )

        if (
            use_nt_absorb
            and hasattr(fops, "dsa_uv_proj")
            and hasattr(self, "_w_uv_rm")
            and attn_latent.dtype == torch.bfloat16
            and self._w_uv_rm.dtype == torch.bfloat16
            and self.v_head_dim == 128
        ):
            out_view = output[:num_actual].view(num_actual, self.num_heads, self.v_head_dim)
            fops.dsa_uv_proj(attn_latent[..., : self.kv_lora_rank], self._w_uv_rm, out_view)
        else:
            # bmm: [H, T, kv_lora] @ [H, kv_lora, v_dim] -> [H, T, v_dim] -> [T, H, v_dim]
            v_in = attn_latent[..., : self.kv_lora_rank].to(self.W_UV.dtype).transpose(0, 1)
            v_out = torch.bmm(v_in, self.W_UV).transpose(0, 1)
            output[:num_actual].view(num_actual, self.num_heads, self.v_head_dim).copy_(v_out)
        return output


class Ernie5_5ROCmSparseMLABackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_kv_cache_dtypes = [
        "auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_ds_mla", "nvfp4", "nvfp4_ds_mla",
    ]
    get_supported_kernel_block_sizes = staticmethod(_dsa_kernel_block_sizes)

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return True

    @staticmethod
    def get_name() -> str:
        return "ERNIE55_ROCM_SPARSE_MLA"

    @staticmethod
    def get_impl_cls() -> type[Ernie5_5ROCmSparseMLAImpl]:
        return Ernie5_5ROCmSparseMLAImpl

    @staticmethod
    def get_builder_cls() -> type[Ernie5_5ROCmSparseMLAMetadataBuilder]:
        return Ernie5_5ROCmSparseMLAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            return (num_blocks, block_size, 656)
        if cache_dtype_str == "nvfp4_ds_mla":
            return (num_blocks, block_size, 352)
        return (num_blocks, block_size, head_size)

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.BLNHC, KVCacheLayout.BLHNC)


class Ernie5_5MLAAttention(MLAAttention):
    """Sparse MLA layer whose KV cache uses the 64-token DSA page size."""

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return _dsa_page_spec(super().get_kv_cache_spec(vllm_config), vllm_config)

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        self.impl.process_weights_after_loading(act_dtype)

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: torch.Size | None = None,
        output: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if output is None:
            out_shape = (
                output_shape
                if output_shape is not None
                else (q.shape[0], self.num_heads * self.v_head_dim)
            )
            output = torch.empty(out_shape, dtype=q.dtype, device=q.device)
        attn_metadata = get_forward_context().attn_metadata
        layer_meta = (
            attn_metadata.get(self.layer_name)
            if isinstance(attn_metadata, dict)
            else None
        )
        return self.impl.forward(
            self, q, kv_c_normed, k_pe, self.kv_cache, layer_meta, output=output
        )


class Ernie5_5DSAAttention(nn.Module):
    """DSA MLA layer (6/43 layers) on ROCm gfx1151 with HTR-W1 trunk + BF16 kv_a_proj."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        swa_topk_buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = copy.copy(vllm_config.cache_config)
        # Resolve DSA KV cache format: auto (BF16 1152B), fp8_ds_mla (656B), or nvfp4_ds_mla (352B)
        cache_config.cache_dtype = _resolve_ernie_dsa_kv_dtype(vllm_config)

        self.num_heads = config.mla_num_heads
        self.num_local_heads = self.num_heads // get_tensor_model_parallel_world_size()
        self.q_lora_rank = config.mla_q_lora_rank
        self.kv_lora_rank = config.mla_kv_lora_rank
        self.qk_nope_head_dim = config.mla_qk_nope_head_dim
        self.qk_rope_head_dim = config.mla_qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.mla_v_head_dim
        self.window = config.sliding_window

        # Separate q_a_proj (HTR K5) and kv_a_proj_with_mqa (protected BF16!)
        self.q_a_proj = ReplicatedLinear(
            config.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.q_a_proj",
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=None,
            return_bias=False,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.wo_b = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )
        self.o_gate = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.v_head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.o_gate",
        )
        self.vha = VHAPostMix(self.num_heads)
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            max_position=int(getattr(vllm_config.model_config, 'max_model_len', config.max_position_embeddings)) + 256,
            is_neox_style=False,
            rope_parameters=config.mla_rope_parameters,
        )
        # _ernie5_dedup_dsa_rope: share bitwise-identical cos_sin_cache with indexer (-128 MiB @ 1M)
        try:
            _idx_rope = getattr(getattr(self, "indexer", None), "rotary_emb", None)
            _c1 = getattr(self.rotary_emb, "cos_sin_cache", None)
            _c2 = getattr(_idx_rope, "cos_sin_cache", None)
            if (
                _c1 is not None and _c2 is not None and _c1 is not _c2
                and _c1.shape == _c2.shape and _c1.dtype == _c2.dtype and _c1.device == _c2.device
                and torch.equal(_c1[:64], _c2[:64]) and torch.equal(_c1[-64:], _c2[-64:])
            ):
                _idx_rope.cos_sin_cache = _c1
        except Exception:
            pass
        self.attn_sink = nn.Parameter(
            torch.empty(self.num_local_heads, dtype=torch.float32), requires_grad=False
        )

        self.indexer = Ernie5_5IndexerROCm(
            vllm_config,
            config,
            cache_config,
            topk_indices_buffer,
            swa_topk_buffer,
            prefix=f"{prefix}.indexer",
        )
        self.mla_attn = Ernie5_5MLAAttention(
            num_heads=self.num_local_heads,
            scale=self.qk_head_dim**-0.5,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            kv_b_proj=self.kv_b_proj,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_backend=Ernie5_5ROCmSparseMLABackend,
            use_sparse=True,
            indexer=self.indexer,
            topk_indices_buffer=swa_topk_buffer,
            sinks=self.attn_sink,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        q_c = self.q_a_layernorm(self.q_a_proj(hidden_states))
        fops = get_fast_ops()
        if (
            1 <= num_tokens <= 8
            and _os.environ.get("ERNIE_BF16_GEMV_NT", "1") == "1"
            and fops is not None
            and hasattr(fops, "bf16_gemv_nt")
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and self.kv_a_proj_with_mqa.weight.dtype == torch.bfloat16
            and (hidden_states.shape[1] % 256) == 0
        ):
            kv_lora = fops.bf16_gemv_nt(hidden_states.contiguous(), self.kv_a_proj_with_mqa.weight, True)
        else:
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)
        o_gate_raw = self.o_gate(q_c)
        q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], -1)
        kv_c_normed = self.kv_a_layernorm(kv_c)
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe.unsqueeze(1)
        )

        attn_out = torch.empty(
            (num_tokens, self.num_local_heads * self.v_head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._indexer_and_attn(
            hidden_states, q_c, positions, q, kv_c_normed, k_pe, attn_out
        )
        o = self.vha(attn_out.view(num_tokens, self.num_local_heads, self.v_head_dim))
        o_flat = o.flatten(1)
        qw_b = getattr(self.wo_b, "htr_qweight", None)
        if (
            qw_b is not None
            and o_flat.is_cuda
            and o_flat.dtype == torch.bfloat16
            and o_gate_raw.dtype == torch.bfloat16
            and o_flat.stride(1) == 1
            and o_gate_raw.stride(1) == 1
            and o_flat.shape[1] <= 8192
            and (o_flat.shape[1] % 128 == 0)
            and _os.environ.get("ERNIE_FUSED_PROLOGUE", "1") == "1"
        ):
            from .patches import htr_k5sp_linear_pre_prologued
            if qw_b.is_cuda and qw_b.dim() == 3 and qw_b.shape[2] == 40 and (qw_b.shape[0] % 8) == 0:
                from .fast_ops import get_k5_sp_prefill
                sp_ext = get_k5_sp_prefill()
                if sp_ext is not None and hasattr(sp_ext, "sigmoid_mul_htr_prologue"):
                    xt, xs, tok = sp_ext.sigmoid_mul_htr_prologue(
                        o_flat, o_gate_raw, self.wo_b.htr_su
                    )
                    return htr_k5sp_linear_pre_prologued(
                        xt, xs, tok, qw_b, self.wo_b.htr_scales, self.wo_b.htr_sv,
                        getattr(self.wo_b, "bias", None),
                    )
        return self.wo_b(o_flat * torch.sigmoid(o_gate_raw))

    @eager_break_during_capture
    def _indexer_and_attn(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        num_tokens = hidden_states.shape[0]
        self.indexer(hidden_states, q_c, positions)
        merge_swa_topk(
            self.indexer.indexer_op.topk_indices_buffer[:num_tokens],
            self.indexer.topk_indices_buffer[:num_tokens],
            positions[:num_tokens],
            self.window,
        )
        self.mla_attn(q, kv_c_normed, k_pe, output=out)


