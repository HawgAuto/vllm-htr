# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMD ROCm QSA owner with Triton kernels."""

from __future__ import annotations

from typing import Optional, ClassVar, cast

import torch
from torch import nn

from vllm._custom_ops import reshape_and_cache_flash
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    canonicalize_singleton_dim_strides,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.ops import rocm_octave
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

from ..common.qsa_cache import QSAForwardMetadata
from . import model
from .indexer_qsa import QSAIndexer
from .ops.qsa_main_prepare import qsa_main_prepare, supports_main_prepare


import os as _gl_os
_KV8_ENABLED = _gl_os.environ.get("HTR_KV8_ENABLE", "0") == "1"
_KV4_ENABLED = _gl_os.environ.get("HTR_KV4_ENABLE", "0") == "1"

def _kv4_mode(mode):
    return _KV4_ENABLED and mode == "int4_per_token_head"

def _kv8_mode(mode):
    return _KV8_ENABLED and mode == "int8_per_token_head"

_OCTAVE_CACHE_DTYPES: tuple[CacheDType, ...] = (
    "octave_k3v4",
    "octave_k3v3",
    "octave_k3v3_compact",
)
_HTR_CACHE_DTYPES: tuple[CacheDType, ...] = tuple(
    (["int8_per_token_head"] if _KV8_ENABLED else [])
    + (["int4_per_token_head"] if _KV4_ENABLED else [])
)
# Packed formats are written by do_kv_cache_update, never by the fused BF16 prepare.
_PACKED_CACHE_DTYPES: tuple[CacheDType, ...] = (*_OCTAVE_CACHE_DTYPES, *_HTR_CACHE_DTYPES)
_QSA_CACHE_DTYPES: tuple[CacheDType, ...] = (
    "auto",
    "bfloat16",
    *_OCTAVE_CACHE_DTYPES,
    *_HTR_CACHE_DTYPES,
)

_GL_GATE_LATE = _gl_os.environ.get("DIRECT_QSA_GATE_LATE") == "1"
_GF_ON = _gl_os.environ.get('DIRECT_QSA_GATE_FUSE') == '1'   # nb2-gatefuse-v1 (static, trace-time)
_GL_OUT_TAIL = _gl_os.environ.get("DIRECT_QSA_OUT_TAILZERO") == "1"


def _octave_mid_o_capacity(vllm_config: VllmConfig) -> int:
    """Query rows times splits the Octave split-KV partials hold: every split
    up to the cudagraph capture size, and one split per row of a full chunk."""
    capture = vllm_config.compilation_config.max_cudagraph_capture_size or 4
    return max(
        capture * vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph,
        2 * vllm_config.scheduler_config.max_num_batched_tokens,
    )


# Exact window of the Octave QSA cache: fp16 shadow tables (entries, powers of
# two) of every stored token and of each sequence's leading positions; the
# kernels keep the last 64 and the first 4 selected positions exact.
_OCTAVE_WINDOW_ENTRIES = 16384
_OCTAVE_SINK_ENTRIES = 2048


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Layers run one after another, so they share one set of Octave
        # split-KV partials; reserve it before the workspace is locked.
        if kv_cache_spec.kv_quant_mode.is_octave and is_workspace_manager_initialized():
            num_heads = vllm_config.model_config.get_num_attention_heads(
                vllm_config.parallel_config
            )
            numel = (
                _octave_mid_o_capacity(vllm_config)
                * num_heads
                * (kv_cache_spec.head_size + 2)
            )
            current_workspace_manager().get_simultaneous(((numel,), torch.float32))




class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner.

    With an Octave ``kv_cache_dtype`` the main K/V cache holds one packed
    Octave slot per (token, KV head) and sparse attention runs the Octave
    sparse decode kernel over the selected positions.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = list(_QSA_CACHE_DTYPES)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype):
        if _kv8_mode(kv_cache_dtype) or _kv4_mode(kv_cache_dtype):
            return True
        return super().supports_kv_cache_dtype(kv_cache_dtype)

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        from vllm.v1.kv_cache_interface import KVQuantMode
        if _KV4_ENABLED and spec.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            if spec.dtype != torch.uint8 or spec.head_size != 256 or spec.num_kv_heads != 2:
                raise ValueError("HTR INT4 main KV requires TP1 head256 KV2")
            from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
            return TritonAttentionBackend.customize_spec(spec)
        if _KV8_ENABLED and spec.kv_quant_mode == KVQuantMode.INT8_PER_TOKEN_HEAD:
            if spec.dtype != torch.int8 or spec.head_size != 256 or spec.num_kv_heads != 2:
                raise ValueError("HTR INT8 main KV is qualified only for TP1 head256 KV2")
            from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
            return TritonAttentionBackend.customize_spec(spec)
        from vllm.v1.attention.backends.rocm_octave_attn import (
            RocmOctaveAttentionBackend,
        )

        return RocmOctaveAttentionBackend.customize_spec(spec)

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # QSA consumes manager pages directly and does not use FA4 paged attention.
        return [MultipleOf(16)]

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAFlashAttentionImpl]:
        return Qwen4ExpQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel."""

    supports_dcp: bool = False
    supports_pcp: bool = False

    def __init__(self, *args, **kwargs) -> None:
        # This owner uses FlashAttentionImpl only for validated metadata and the
        # stock reshape_and_cache_flash KV write. Its attention result is
        # produced exclusively by forward_qsa's paged Triton sparse kernel, so
        # requiring the unrelated dense flash_attn_varlen_func rejects ROCm
        # installations that have every operation this owner actually calls.
        wanted = kwargs.get("kv_cache_dtype", args[6] if len(args) > 6 else None)
        if _kv8_mode(wanted) or _kv4_mode(wanted):
            # Reuse metadata fields, not FlashAttention's dense quantized reader.
            parent_args, parent_kwargs = list(args), dict(kwargs)
            if len(parent_args) > 6:
                parent_args[6] = "bfloat16"
            else:
                parent_kwargs["kv_cache_dtype"] = "bfloat16"
            super().__init__(*parent_args, **parent_kwargs)
            self.kv_cache_dtype = wanted
            if (self.num_heads, self.num_kv_heads, self.head_size) != (24, 2, 256):
                raise NotImplementedError("HTR INT8 main KV currently requires TP1 Q24/KV2/D256")
        else:
            super().__init__(*args, **kwargs)
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support decode context parallelism"
            )
        if self.kv_cache_dtype not in _QSA_CACHE_DTYPES:
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16, HTR INT8/INT4 or Octave main KV cache"
            )
        self.supports_quant_query_input = False
        self.octave_fmt: rocm_octave.OctaveFormat | None = None
        if self.kv_cache_dtype in _OCTAVE_CACHE_DTYPES:
            self.octave_fmt = rocm_octave.OctaveFormat.from_cache_dtype(
                self.kv_cache_dtype,
                self.head_size,
                rocm_octave.registered_rope_dim(self.head_size),
            )
            self._octave_signs: torch.Tensor | None = None
            self._octave_mid_o: torch.Tensor | None = None
            vllm_config = get_current_vllm_config()
            self._octave_splits = (
                vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
            )
            self._octave_capacity = _octave_mid_o_capacity(vllm_config)
            self._octave_positions: torch.Tensor | None = None
            device = torch.device("cuda", torch.accelerator.current_device_index())
            dtype = vllm_config.model_config.dtype
            row = (self.num_kv_heads, 2 * self.head_size)

            def table(entries: int) -> tuple[torch.Tensor, ...]:
                return (
                    torch.zeros((entries, *row), dtype=dtype, device=device),
                    torch.full((entries,), -1, dtype=torch.int32, device=device),
                    torch.zeros(entries, dtype=torch.int32, device=device),
                )

            self._wtab, self._wtags, self._wlocks = table(_OCTAVE_WINDOW_ENTRIES)
            self._stab, self._stags, self._slocks = table(_OCTAVE_SINK_ENTRIES)

    def _octave_buffers(
        self, num_q: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Sign bits and split-KV partials, from the workspace every layer
        shares (without one, a buffer allocated once, since captured graphs
        hold its address). Larger batches use fewer splits so they fit."""
        if self._octave_signs is None or self._octave_signs.device != device:
            self._octave_signs = rocm_octave.sign_bits(self.head_size).to(device)
        capacity = self._octave_capacity
        splits = max(2, min(self._octave_splits, capacity // max(num_q, 1)))
        row = self.head_size + 2
        numel = capacity * self.num_heads * row
        if is_workspace_manager_initialized():
            (buf,) = current_workspace_manager().get_simultaneous(
                ((numel,), torch.float32)
            )
        else:
            if self._octave_mid_o is None:
                self._octave_mid_o = torch.empty(
                    numel, dtype=torch.float32, device=device
                )
            buf = self._octave_mid_o
        mid_o = buf[: num_q * self.num_heads * splits * row]
        return (
            self._octave_signs,
            mid_o.view(num_q, self.num_heads, splits, row),
            splits,
        )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # QSA's Triton attention needs the native HIP KV scatter, not the
        # separately guarded dense flash-attn varlen package. Preserve the
        # parent paged-cache layout and slot-mapping semantics exactly.
        if self.octave_fmt is not None:
            n = slot_mapping.shape[0]
            if n == 0:
                return
            signs, _, _ = self._octave_buffers(1, key.device)
            torch.ops._C.octave_cache_store(
                key[:n],
                value[:n],
                kv_cache,
                slot_mapping,
                signs,
                signs,
                self.octave_fmt.kernel_code,
            )
            assert self._octave_positions is not None
            torch.ops._C.octave_window_store(
                key[:n],
                value[:n],
                slot_mapping,
                self._octave_positions[:n],
                self._wtab,
                self._wtags,
                self._wlocks,
                self._stab,
                self._stags,
                self._slocks,
            )
            return
        if _kv4_mode(self.kv_cache_dtype):
            from htr_qsa_kv4 import write
            write(kv_cache, key, value, slot_mapping)
            return
        if _kv8_mode(self.kv_cache_dtype):
            from htr_qsa_kv8 import write
            write(kv_cache, key, value, slot_mapping)
            return
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        if _GL_OUT_TAIL:
            # rows [:num_tokens] are zeroed by htr_qsa.attention(zero_fallback=True) before the kernel runs;
            # only the tail keeps the zero-init here.
            output[num_tokens:].zero_()
        else:
            output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        if _kv4_mode(self.kv_cache_dtype):
            from htr_qsa_kv4 import attention
            attention(query[:num_tokens], kv_cache, logical_indices,
                      attn_metadata.block_table, token_to_req, output[:num_tokens])
            return output
        if _kv8_mode(self.kv_cache_dtype):
            from htr_qsa_kv8 import attention
            attention(query[:num_tokens], kv_cache, logical_indices,
                      attn_metadata.block_table, token_to_req, output[:num_tokens])
            return output
        if self.octave_fmt is not None:
            assert self._octave_positions is not None
            signs, mid_o, splits = self._octave_buffers(num_tokens, query.device)
            torch.ops._C.octave_decode_sparse(
                output[:num_tokens],
                query[:num_tokens],
                kv_cache,
                attn_metadata.block_table,
                token_to_req,
                logical_indices,
                mid_o,
                signs,
                signs,
                self.scale,
                splits,
                self.octave_fmt.kernel_code,
                self._octave_positions[:num_tokens],
                self._wtab,
                self._wtags,
                self._stab,
                self._stags,
            )
            return output
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA requires BF16 Q/K/V")

        from htr_qsa import attention as htr_qsa_attention

        htr_qsa_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            output[:num_tokens],
            zero_fallback=_GL_OUT_TAIL,
        )
        return output


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
        if cache_config.cache_dtype not in _QSA_CACHE_DTYPES:
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16, HTR INT8/INT4 or Octave main KV cache"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        import direct_dense_cm as _dcm  # nb2-densecm: column-major o_proj weight (NON-BITWISE, quality-gated; off unless DIRECT_DENSE_CM=1)

        _dcm.install(self.o_proj, f"{prefix}.o_proj")
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and text_only
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if (
            self.kv_cache_dtype not in _OCTAVE_CACHE_DTYPES
            and self.kv_cache_torch_dtype != torch.bfloat16
            and not (_kv8_mode(self.kv_cache_dtype) and self.kv_cache_torch_dtype == torch.int8)
            and not (_kv4_mode(self.kv_cache_dtype) and self.kv_cache_torch_dtype == torch.uint8)
        ):
            raise NotImplementedError("Qwen4Exp QSA requires BF16 cache storage")
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
        self.impl = Qwen4ExpQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        # Exact main prepare writes BF16 K/V; every packed format retains
        # do_kv_cache_update. The fused pre-indexer is independent.
        self.use_fused_qsa_prepare = (
            self.attn_output_gate
            and _gl_os.environ.get("HTR_QSA_MAIN_PREPARE", "1") != "0"
            and supports_main_prepare(
                self.rotary_emb, self.head_dim, self.num_heads, self.num_kv_heads
            )
            and self.kv_cache_dtype not in _PACKED_CACHE_DTYPES
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _run_qsa(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor | None,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        output: torch.Tensor,
        qkv: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> None:
        # query/key/value are None when the fused prepare runs inside the
        # indexer launch, which then writes the gate into ``gate``.
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            if gate is not None:
                gate.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        main_outputs = None
        if qkv is not None:
            if gate is None:
                raise ValueError("QSA main prepare requires a gate buffer")
            if positions.ndim == 2:
                main_positions = positions[:, :num_tokens]
            else:
                main_positions = positions[:num_tokens]
            main_q = qsa_main_prepare(
                qkv[:num_tokens],
                main_positions,
                self.rotary_emb._match_cos_sin_cache_dtype(qkv),
                self.q_norm.weight,
                self.k_norm.weight,
                self.q_norm.variance_epsilon,
                self.kv_cache.transpose(1, 2),
                main_metadata.slot_mapping,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                rotary_dim=int(self.rotary_emb.rotary_dim),
                mrope_section=getattr(self.rotary_emb, "mrope_section", None),
                gate_out=gate,
            )
            main_outputs = (main_q, gate)
        selected = self.indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_tokens],
        )
        if selected.shape != (
            num_tokens,
            self.indexer.output_width,
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        if main_outputs is None:
            assert key is not None and value is not None
            if impl.octave_fmt is not None:
                flat = positions[0] if positions.dim() == 2 else positions
                impl._octave_positions = flat.to(torch.int32)
            impl.do_kv_cache_update(
                self,
                key,
                value,
                self.kv_cache,
                main_metadata.slot_mapping,
            )
        else:
            query, _ = main_outputs
            # The gate is applied to every row outside this op; padded rows
            # must stay finite so they keep the zero output forward_qsa gives.
            assert gate is not None
            gate[num_tokens:].zero_()
        assert query is not None
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        num_tokens = hidden_states.shape[0]
        encoded_layer_name = _encode_layer_name(self.layer_name)
        if self.use_fused_qsa_prepare:
            # Norm/RoPE/gate and the K/V cache write happen inside the op.
            attn_output = qkv.new_empty(num_tokens, self.num_heads, self.head_dim)
            gate = torch.empty_like(attn_output)
            if current_platform.opaque_attention_op():
                torch.ops.vllm.qwen4_exp_qsa_prepare_with_output(
                    hidden_states,
                    positions,
                    qkv,
                    attn_output,
                    gate,
                    encoded_layer_name,
                    _GF_ON,
                )
            else:
                qwen4_exp_qsa_prepare_with_output(
                    hidden_states,
                    positions,
                    qkv,
                    attn_output,
                    gate,
                    encoded_layer_name,
                    _GF_ON,
                )
            flat_output = attn_output.view(num_tokens, -1)
            if not _GF_ON:
                flat_output = flat_output * torch.sigmoid(gate.view(num_tokens, -1))
            output, _ = self.o_proj(flat_output)
            return output

        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        _gf_gate = None
        if (_GF_ON and gate is not None and self.attn_output_gate
                and not getattr(self, "use_fused_qk_norm_rope_gate", False)):
            _gf_gate = qkv[..., : self.q_size * 2].view(num_tokens, self.num_heads, 2 * self.head_dim)[..., self.head_dim:]
        _GF_FUSED_THIS_CALL = _gf_gate is not None
        if current_platform.opaque_attention_op():
            torch.ops.vllm.qwen4_exp_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
                _gf_gate,
            )
        else:
            qwen4_exp_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
                _gf_gate,
            )
        if _GF_FUSED_THIS_CALL:
            flat_output = attn_output.view(num_tokens, -1)   # gate applied exactly once inside the op (v3k epilogue or ref kernel)
        elif (gate is not None and _GL_GATE_LATE and self.attn_output_gate
                and not getattr(self, "use_fused_qk_norm_rope_gate", False)):
            # Same elements as _project_qkv_gate's gate (chunk(q_gate.view(T,H,2D),2)[1]), read as a strided view of qkv
            # after the opaque attention op, so the pre-attention graph piece no longer materializes the reshape-copy.
            g = qkv[..., : self.q_size * 2].view(num_tokens, self.num_heads, 2 * self.head_dim)[..., self.head_dim:]
            flat_output = (attn_output * torch.sigmoid(g)).view(num_tokens, -1)
        else:
            flat_output = attn_output.view(num_tokens, -1)
            if gate is not None:
                flat_output = flat_output * torch.sigmoid(gate)
        output, _ = self.o_proj(flat_output)
        return output


def qwen4_exp_qsa_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    gate: Optional[torch.Tensor] = None,
) -> None:
    """Run the complete QSA state/update/attend transaction."""
    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpQSAAttention):
        raise TypeError(f"{layer_name} is not a Qwen4Exp QSA owner")
    if gate is None:
        layer._run_qsa(
            hidden_states,
            positions,
            query,
            key,
            value,
            output,
        )
        return
    # nb2-gatefuse-v1: this op owns the QSA output gate, applied once per element after attention with the
    # verbatim inductor-body Triton kernel (htr_qsa.gate_apply_).
    from htr_qsa import gate_apply_
    layer._run_qsa(
        hidden_states,
        positions,
        query,
        key,
        value,
        output,
    )
    gate_apply_(output, gate)


direct_register_custom_op(
    op_name="qwen4_exp_qsa_with_output",
    op_func=qwen4_exp_qsa_with_output,
    mutates_args=["output"],
)


def qwen4_exp_qsa_prepare_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    qkv: torch.Tensor,
    output: torch.Tensor,
    gate: torch.Tensor,
    layer_name: LayerNameType,
    apply_gate: bool = False,
) -> None:
    """Run the QSA transaction with the main QK-norm/RoPE/gate fused in."""
    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpQSAAttention):
        raise TypeError(f"{layer_name} is not a Qwen4Exp QSA owner")
    layer._run_qsa(
        hidden_states,
        positions,
        None,
        None,
        None,
        output,
        qkv=qkv,
        gate=gate,
    )
    if apply_gate:
        # nb2-gatefuse-v1: gate applied once, by the same kernel as the unfused path.
        from htr_qsa import gate_apply_

        gate_apply_(output, gate)


direct_register_custom_op(
    op_name="qwen4_exp_qsa_prepare_with_output",
    op_func=qwen4_exp_qsa_prepare_with_output,
    mutates_args=["output", "gate"],
)


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAFlashAttentionBackend",
    "Qwen4ExpQSAFlashAttentionImpl",
    "qwen4_exp_qsa_prepare_with_output",
    "qwen4_exp_qsa_with_output",
]
