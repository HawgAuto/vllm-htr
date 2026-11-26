# SPDX-License-Identifier: Apache-2.0
"""Minimal vLLM hooks for ERNIE-5.5-Flash HTR-W1 + DSpark (leaves Qwen3.8 untouched)."""

from __future__ import annotations

import dataclasses
import functools
import os
from typing import Literal, get_args

from vllm.logger import init_logger

logger = init_logger(__name__)

ERNIE5_MODEL_TYPE = "ernie5_5_moe"
ERNIE5_MTP_MODEL_TYPE = "ernie5_5_moe_mtp"
ERNIE5_MTP_ARCH = "Ernie5_5MoeMTPModel"
ERNIE5_DSPARK_ARCH = "Ernie5_5_MoeDSparkDraftModel"
ERNIE5_TOKENIZER_MODE = "ernie5_5"
_applied = False
_serving_ernie5 = False


def _patch_config_registry() -> None:
    import os
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    from .config import Ernie5_5Config

    orig_init = Ernie5_5Config.__init__
    if not getattr(orig_init, "_ernie5_ple_guard", False):
        @functools.wraps(orig_init)
        def _init_with_ple_guard(self, *args, **kwargs):
            os.environ["VLLM_QWEN4_PLE_SSD"] = "0"
            os.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
            # Enable zero-materialization K=5 dense prefill WMMA strictly for ERNIE-5.5
            # (eliminates ~160 GB BF16 weight round-trip per prefill chunk across 43 layers!)
            if os.environ.get("ERNIE_DENSE_PREFILL_WMMA", "1") == "1":
                try:
                    import htr.ops as _htr_ops
                    _htr_ops.DENSE_PREFILL_WMMA = True
                    _htr_ops.DENSE_PREFILL_WMMA_MIN_M = int(
                        os.environ.get("ERNIE_DENSE_PREFILL_WMMA_MIN_M", "33")
                    )
                    _htr_ops.DENSE_PREFILL_WMMA_MAX_M = int(
                        os.environ.get("ERNIE_DENSE_PREFILL_WMMA_MAX_M", "4096")
                    )
                except Exception:
                    pass
            return orig_init(self, *args, **kwargs)
        _init_with_ple_guard._ernie5_ple_guard = True  # type: ignore[attr-defined]
        Ernie5_5Config.__init__ = _init_with_ple_guard

    _CONFIG_REGISTRY.setdefault(ERNIE5_MODEL_TYPE, Ernie5_5Config)


def _patch_rocm_paged_mqa_logits_triton() -> None:
    """Enable graph-safe Split-KV Triton paged MQA logits on gfx1151 (Strix Halo)."""
    try:
        import vllm.v1.attention.ops.rocm_aiter_mla_sparse as rmod
    except Exception:
        return
    orig = getattr(rmod, "rocm_fp8_paged_mqa_logits", None)
    triton_fn = getattr(rmod, "rocm_fp8_paged_mqa_logits_triton", None)
    if orig is None or triton_fn is None or getattr(orig, "_ernie5_gfx1151_triton", False):
        return

    @functools.wraps(orig)
    def _rocm_fp8_paged_mqa_logits_fast(
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        *,
        compress_ratio: int = 1,
    ):
        if (
            not (getattr(rmod, "_ON_GFX950", False) or getattr(rmod, "_ON_GFX942", False))
            and kv_cache_fp8.shape[1] >= 16
            and kv_cache_fp8.shape[1] % 16 == 0
        ):
            if q_fp8.is_cuda and q_fp8.dim() == 4 and q_fp8.shape[2] == 64 and q_fp8.shape[3] == 128:
                from .fast_ops import get_dsa_attn
                ext = get_dsa_attn()
                if ext is not None and hasattr(ext, "fp8_paged_mqa_logits_decode"):
                    import torch
                    from vllm.v1.worker.workspace import current_workspace_manager
                    B_dec, next_n_dec = q_fp8.shape[0], q_fp8.shape[1]
                    rows = B_dec * next_n_dec
                    if os.environ.get("ERNIE_FLASHINDEXER_DECODE", "1") == "1":
                        from .flashindexer import (
                            flashindexer_select_heads_rocm,
                            get_active_flashindexer,
                            slice_hidden_for_weights,
                        )
                        fi_ctx = get_active_flashindexer()
                        if fi_ctx is not None and weights.shape[-1] == 64 and weights.numel() == rows * 64:
                            fi_indexer, fi_hidden, fi_base_weights = fi_ctx
                            if getattr(fi_indexer, "linear_gate", None) is not None and fi_hidden.shape[0] >= rows:
                                w_2d = weights.reshape(rows, 64)
                                if w_2d.untyped_storage().data_ptr() == fi_base_weights.untyped_storage().data_ptr():
                                    h_dec = slice_hidden_for_weights(fi_hidden, fi_base_weights, w_2d)
                                else:
                                    h_dec = fi_hidden[:rows]
                                q_sel, w_sel = flashindexer_select_heads_rocm(
                                    q_fp8.reshape(rows, 64, 128),
                                    w_2d,
                                    fi_indexer.linear_gate.weight,
                                    fi_indexer.flashindexer_bias,
                                    h_dec,
                                    topk_heads=getattr(fi_indexer, "flashindexer_topk", 8),
                                )
                                q_fp8 = q_sel.view(B_dec, next_n_dec, -1, 128)
                                weights = w_sel
                    (out_logits,) = current_workspace_manager().get_simultaneous(
                        ((rows, max_model_len), torch.float32),
                    )
                    if kv_cache_fp8.shape[-1] == 68 and hasattr(ext, "fp8_mxfp4_paged_mqa_logits_decode"):
                        ext.fp8_mxfp4_paged_mqa_logits_decode(
                            q_fp8, kv_cache_fp8, weights, context_lens, block_tables,
                            out_logits, int(max_model_len),
                        )
                        return out_logits
                    is_shuffle = (kv_cache_fp8.shape[1] > 1) and (compress_ratio != 4)
                    ext.fp8_paged_mqa_logits_decode(
                        q_fp8, kv_cache_fp8, weights, context_lens, block_tables,
                        out_logits, int(max_model_len), bool(is_shuffle),
                    )
                    return out_logits
            if kv_cache_fp8.shape[1] >= 64 and kv_cache_fp8.shape[1] % 64 == 0 and compress_ratio == 4:
                return triton_fn(
                    q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len
                )
        return orig(
            q_fp8,
            kv_cache_fp8,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len,
            compress_ratio=compress_ratio,
        )

    _rocm_fp8_paged_mqa_logits_fast._ernie5_gfx1151_triton = True  # type: ignore[attr-defined]
    rmod.rocm_fp8_paged_mqa_logits = _rocm_fp8_paged_mqa_logits_fast

    _orig_max_dec = getattr(rmod, "_max_decode_logits_rows", None)
    if _orig_max_dec is not None and not getattr(_orig_max_dec, "_ernie5_max_decode_rows_patched", False):
        @functools.wraps(_orig_max_dec)
        def _max_decode_logits_rows_fixed(num_batched_tokens: int) -> int:
            res = _orig_max_dec(num_batched_tokens)
            cached = getattr(rmod, "_ernie5_max_decode_rows", None)
            if cached and cached > 0:
                return min(int(res), int(cached))
            return res
        _max_decode_logits_rows_fixed._ernie5_max_decode_rows_patched = True  # type: ignore[attr-defined]
        rmod._max_decode_logits_rows = _max_decode_logits_rows_fixed

    # Also replace slow fp8_mqa_logits_torch prefill fallback on gfx1151 with native WMMA HIP kernel
    orig_pref = getattr(rmod, "rocm_fp8_mqa_logits", None)
    if orig_pref is not None and not getattr(orig_pref, "_ernie5_gfx1151_hip", False):
        @functools.wraps(orig_pref)
        def _rocm_fp8_mqa_logits_hip(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
            from .flashindexer import (
                flashindexer_select_heads_rocm,
                get_active_flashindexer,
                slice_hidden_for_weights,
            )
            fi_ctx = get_active_flashindexer()
            if fi_ctx is not None and q.dim() == 3 and q.shape[1] == 64:
                fi_indexer, fi_hidden, fi_base_weights = fi_ctx
                if getattr(fi_indexer, "linear_gate", None) is not None:
                    h_chunk = slice_hidden_for_weights(fi_hidden, fi_base_weights, weights)
                    q, weights = flashindexer_select_heads_rocm(
                        q,
                        weights,
                        fi_indexer.linear_gate.weight,
                        fi_indexer.flashindexer_bias,
                        h_chunk,
                        topk_heads=getattr(fi_indexer, "flashindexer_topk", 8),
                    )
            if (
                not (getattr(rmod, "_ON_GFX950", False) or getattr(rmod, "_ON_GFX942", False))
                and q.is_cuda
                and q.dim() == 3
                and q.shape[1] in (64, 8)
                and q.shape[2] == 128
            ):
                from .fast_ops import get_dsa_attn
                ext = get_dsa_attn()
                if ext is not None:
                    k_fp8, k_scale = kv
                    return ext.fp8_mqa_logits_prefill(
                        q, k_fp8, k_scale, weights, cu_seqlen_ks, cu_seqlen_ke
                    )
            return orig_pref(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)

        _rocm_fp8_mqa_logits_hip._ernie5_gfx1151_hip = True  # type: ignore[attr-defined]
        rmod.rocm_fp8_mqa_logits = _rocm_fp8_mqa_logits_hip

    # Also replace Triton indexer_k_quant_and_cache_triton (6x/tok) with Wave32 HIP
    orig_ikq = getattr(rmod, "indexer_k_quant_and_cache_triton", None)
    if orig_ikq is not None and not getattr(orig_ikq, "_ernie5_gfx1151_hip", False):
        @functools.wraps(orig_ikq)
        def _indexer_k_quant_and_cache_hip(
            k, kv_cache, slot_mapping, quant_block_size, scale_fmt,
            block_tile_size=16, head_tile_size=16,
        ):
            import torch
            if (
                not (getattr(rmod, "_ON_GFX950", False) or getattr(rmod, "_ON_GFX942", False))
                and k.is_cuda
                and k.dim() == 2
                and k.shape[1] == 128
                and k.dtype == torch.bfloat16
                and block_tile_size == 16
                and head_tile_size == 16
                and kv_cache.dtype == torch.uint8
            ):
                from .fast_ops import get_fast_ops
                fops = get_fast_ops()
                if fops is not None and hasattr(fops, "indexer_k_quant_and_cache"):
                    if kv_cache.shape[-1] == 68 and hasattr(fops, "indexer_k_quant_and_cache_mxfp4"):
                        fops.indexer_k_quant_and_cache_mxfp4(
                            k, kv_cache, slot_mapping, int(kv_cache.shape[1])
                        )
                        return
                    fops.indexer_k_quant_and_cache(
                        k, kv_cache, slot_mapping, int(kv_cache.shape[1]), scale_fmt == "ue8m0"
                    )
                    return
            return orig_ikq(
                k, kv_cache, slot_mapping, quant_block_size, scale_fmt,
                block_tile_size=block_tile_size, head_tile_size=head_tile_size,
            )

        _indexer_k_quant_and_cache_hip._ernie5_gfx1151_hip = True  # type: ignore[attr-defined]
        rmod.indexer_k_quant_and_cache_triton = _indexer_k_quant_and_cache_hip

    # Also replace Triton cp_gather_indexer_k_quant_cache_triton (6x/prefill) with 128-bit Wave32 HIP
    orig_cpg = getattr(rmod, "cp_gather_indexer_k_quant_cache_triton", None)
    if orig_cpg is not None and not getattr(orig_cpg, "_ernie5_gfx1151_hip", False):
        @functools.wraps(orig_cpg)
        def _cp_gather_indexer_hip(
            k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen, token_to_seq,
            block_tile_size=16, head_tile_size=16, cache_layout=None,
        ):
            if (
                not (getattr(rmod, "_ON_GFX950", False) or getattr(rmod, "_ON_GFX942", False))
                and k_cache.is_cuda
                and k_fp8.is_cuda
                and k_fp8.dim() == 2
                and k_fp8.shape[1] == 128
                and block_tile_size == 16
                and head_tile_size == 16
            ):
                from .fast_ops import get_fast_ops
                fops = get_fast_ops()
                if fops is not None and hasattr(fops, "cp_gather_indexer_k_quant_cache"):
                    bs = int(k_cache.shape[1])
                    if k_cache.shape[-1] == 68 and hasattr(fops, "cp_gather_indexer_k_mxfp4_to_fp8"):
                        fops.cp_gather_indexer_k_mxfp4_to_fp8(
                            k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen, token_to_seq, bs,
                        )
                        return
                    layout = ("NORMAL" if bs == 1 else "SHUFFLE") if cache_layout is None else cache_layout
                    fops.cp_gather_indexer_k_quant_cache(
                        k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen, token_to_seq,
                        bs, layout == "SHUFFLE",
                    )
                    return
            return orig_cpg(
                k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen, token_to_seq,
                block_tile_size=block_tile_size, head_tile_size=head_tile_size,
                cache_layout=cache_layout,
            )

        _cp_gather_indexer_hip._ernie5_gfx1151_hip = True  # type: ignore[attr-defined]
        rmod.cp_gather_indexer_k_quant_cache_triton = _cp_gather_indexer_hip

    # Also route HCA sparse decode (448 nope + 64 rope) to native Wave32 WMMA Split-KV HIP kernel
    # (fixes missing inv-RoPE + 90% idle CUs + 37x redundant FP32 alloc/copy on gfx1151!)
    orig_dec = getattr(rmod, "rocm_sparse_attn_decode", None)
    if orig_dec is not None and not getattr(orig_dec, "_ernie5_gfx1151_hip", False):
        def _get_hca_scratch(dev, T, H):
            from .fast_ops import get_split_kv_scratch
            return get_split_kv_scratch(dev, T, H=H, S=8, D=512)

        @functools.wraps(orig_dec)
        def _rocm_sparse_attn_decode_hip(
            q,
            kv_cache,
            swa_k_cache,
            swa_only,
            topk_indices,
            topk_lens,
            swa_indices,
            swa_lens,
            swa_ragged_indices,
            swa_ragged_indptr,
            topk_ragged_indices,
            topk_ragged_indptr,
            attn_sink,
            scale,
            head_dim,
            nope_head_dim,
            rope_head_dim,
            output,
            extra_cache_nan_free=False,
            adaptive_splits=False,
            inv_rope_positions=None,
            inv_rope_cos_sin_cache=None,
        ):
            import torch
            if (
                not (getattr(rmod, "_ON_GFX950", False) or getattr(rmod, "_ON_GFX942", False))
                and q.is_cuda
                and q.dtype == torch.bfloat16
                and output.dtype == torch.bfloat16
                and swa_k_cache.dtype == torch.uint8
                and head_dim == 512
                and nope_head_dim == 448
                and rope_head_dim == 64
            ):
                from .fast_ops import get_hca_attn
                ext = get_hca_attn()
                if ext is not None:
                    m_idx, m_ip = swa_ragged_indices, swa_ragged_indptr
                    if m_idx is None or m_ip is None:
                        main_in = swa_indices.reshape(swa_indices.shape[0], -1)
                        m_idx, m_ip = rmod.build_ragged_indices_from_dense(
                            main_in,
                            swa_lens if swa_lens is not None else (main_in >= 0).sum(dim=-1, dtype=torch.int32),
                            num_rows=swa_k_cache.shape[0] * swa_k_cache.shape[1],
                        )
                    e_cache = None
                    e_idx, e_ip = None, None
                    if not swa_only and kv_cache is not None:
                        e_cache = kv_cache
                        e_idx, e_ip = topk_ragged_indices, topk_ragged_indptr
                        if (e_idx is None or e_ip is None) and topk_indices is not None:
                            ext_in = topk_indices.reshape(topk_indices.shape[0], -1)
                            e_idx, e_ip = rmod.build_ragged_indices_from_dense(
                                ext_in,
                                topk_lens if topk_lens is not None else (ext_in >= 0).sum(dim=-1, dtype=torch.int32),
                                num_rows=kv_cache.shape[0] * kv_cache.shape[1],
                            )
                    T, H = q.shape[0], q.shape[1]
                    po, pm, pl = _get_hca_scratch(q.device, T, H)
                    sink_arg = None if attn_sink is None else attn_sink[:H].contiguous()
                    is_fnuz = getattr(rmod.current_platform, "is_fp8_fnuz", lambda: False)()
                    ext.hca_decode(
                        q, swa_k_cache, m_idx, m_ip,
                        e_cache, e_idx, e_ip,
                        sink_arg, inv_rope_positions, inv_rope_cos_sin_cache,
                        output, po, pm, pl, float(scale), bool(is_fnuz),
                    )
                    return output.shape[0] if inv_rope_positions is not None else 0
            return orig_dec(
                q, kv_cache, swa_k_cache, swa_only,
                topk_indices, topk_lens, swa_indices, swa_lens,
                swa_ragged_indices, swa_ragged_indptr, topk_ragged_indices, topk_ragged_indptr,
                attn_sink, scale, head_dim, nope_head_dim, rope_head_dim, output,
                extra_cache_nan_free=extra_cache_nan_free, adaptive_splits=adaptive_splits,
                inv_rope_positions=inv_rope_positions, inv_rope_cos_sin_cache=inv_rope_cos_sin_cache,
            )

        _rocm_sparse_attn_decode_hip._ernie5_gfx1151_hip = True  # type: ignore[attr-defined]
        rmod.rocm_sparse_attn_decode = _rocm_sparse_attn_decode_hip

    # Also route HCA sparse prefill (D=512 BF16 KV) to native Wave32 WMMA HIP kernel
    # (replaces build_ragged_indices_from_dense + _sparse_attn_prefill_ragged_kernel + extra out alloc)
    orig_pref_attn = getattr(rmod, "rocm_sparse_attn_prefill", None)
    if orig_pref_attn is not None and not getattr(orig_pref_attn, "_ernie5_gfx1151_hip", False):
        def _get_hca_pref_scratch(dev, T, H):
            from .fast_ops import get_split_kv_scratch
            return get_split_kv_scratch(dev, T, H=H, S=8, D=512)

        @functools.wraps(orig_pref_attn)
        def _rocm_sparse_attn_prefill_hip(
            q,
            kv,
            indices,
            scale,
            head_dim,
            nope_head_dim,
            rope_head_dim,
            output,
            attn_sink=None,
            topk_length=None,
        ):
            import torch
            if (
                not (getattr(rmod, "_ON_GFX950", False) or getattr(rmod, "_ON_GFX942", False))
                and q.is_cuda
                and q.dtype == torch.bfloat16
                and kv.dtype == torch.bfloat16
                and output.dtype == torch.bfloat16
                and head_dim == 512
                and nope_head_dim == 448
                and rope_head_dim == 64
            ):
                from .fast_ops import get_hca_attn
                ext = get_hca_attn()
                if ext is not None:
                    T, H = q.shape[0], q.shape[1]
                    po = pm = pl = None
                    if T <= 16 and T > 0:
                        po, pm, pl = _get_hca_pref_scratch(q.device, T, H)
                    sink_arg = None if attn_sink is None else attn_sink[:H].contiguous()
                    tl_arg = None if topk_length is None else topk_length.to(torch.int32).contiguous()
                    idx_arg = indices.to(torch.int32).contiguous() if (indices.dtype != torch.int32 or indices.stride(-1) != 1) else indices
                    ext.hca_prefill(
                        q, kv, idx_arg, tl_arg, sink_arg, output, po, pm, pl, float(scale)
                    )
                    return output
            return orig_pref_attn(
                q=q, kv=kv, indices=indices, scale=scale, head_dim=head_dim,
                nope_head_dim=nope_head_dim, rope_head_dim=rope_head_dim,
                output=output, attn_sink=attn_sink, topk_length=topk_length,
            )

        _rocm_sparse_attn_prefill_hip._ernie5_gfx1151_hip = True  # type: ignore[attr-defined]
        rmod.rocm_sparse_attn_prefill = _rocm_sparse_attn_prefill_hip

    # Replace in-place inverse GPT-J RoPE Triton kernel (which copied 448 unchanged NoPE dims = 87.5% wasted traffic)
    orig_inv_rope = getattr(rmod, "rocm_inverse_rope_rows_", None)
    if orig_inv_rope is not None and not getattr(orig_inv_rope, "_ernie5_gfx1151_hip", False):
        @functools.wraps(orig_inv_rope)
        def _rocm_inverse_rope_rows_hip(o, positions, cos_sin_cache, rope_head_dim):
            import torch
            if o.shape[0] == 0:
                return
            if (
                not (getattr(rmod, "_ON_GFX950", False) or getattr(rmod, "_ON_GFX942", False))
                and o.is_cuda
                and o.dtype == torch.bfloat16
                and o.stride(-1) == 1
                and rope_head_dim == 64
            ):
                from .fast_ops import get_fast_ops
                fops = get_fast_ops()
                if fops is not None:
                    fops.inv_rope_gptj_inplace(o, positions, cos_sin_cache, int(rope_head_dim))
                    return
            return orig_inv_rope(o, positions, cos_sin_cache, rope_head_dim)

        _rocm_inverse_rope_rows_hip._ernie5_gfx1151_hip = True  # type: ignore[attr-defined]
        rmod.rocm_inverse_rope_rows_ = _rocm_inverse_rope_rows_hip

    # Replace Triton dequantize_and_gather_k_cache (74x/prefill across 37 HCA layers) with Wave32 HIP
    try:
        import vllm.models.deepseek_v4.amd.rocm as dsv4_rocm
        orig_dqg = getattr(dsv4_rocm, "dequantize_and_gather_k_cache", None)
        if orig_dqg is not None and not getattr(orig_dqg, "_ernie5_gfx1151_hip", False):
            @functools.wraps(orig_dqg)
            def _dq_gather_hip(
                out, k_cache, seq_lens, block_table,
                gather_lens=None, block_size=64, offset=0, use_fnuz=False,
            ):
                import torch
                if (
                    not (getattr(rmod, "_ON_GFX950", False) or getattr(rmod, "_ON_GFX942", False))
                    and out.is_cuda
                    and out.dim() == 3
                    and out.shape[2] == 512
                    and out.dtype == torch.bfloat16
                    and k_cache.dtype == torch.uint8
                ):
                    from .fast_ops import get_hca_attn
                    ext = get_hca_attn()
                    if ext is not None and hasattr(ext, "dequant_gather_ue8m0_kv"):
                        ext.dequant_gather_ue8m0_kv(
                            out, k_cache, seq_lens, block_table, gather_lens,
                            int(block_size), int(offset), bool(use_fnuz),
                        )
                        return
                return orig_dqg(
                    out, k_cache, seq_lens=seq_lens, gather_lens=gather_lens,
                    block_table=block_table, block_size=block_size,
                    offset=offset, use_fnuz=use_fnuz,
                )

            _dq_gather_hip._ernie5_gfx1151_hip = True  # type: ignore[attr-defined]
            dsv4_rocm.dequantize_and_gather_k_cache = _dq_gather_hip

        # Replace torch.full + Triton _combine_topk_swa_indices_kernel (37x/prefill) with 1-pass Wave32 HIP
        orig_comb = getattr(dsv4_rocm, "combine_topk_swa_indices", None)
        if orig_comb is not None and not getattr(orig_comb, "_ernie5_gfx1151_hip", False):
            align = int(getattr(dsv4_rocm, "_SPARSE_PREFILL_TOPK_ALIGNMENT", 128))

            @functools.wraps(orig_comb)
            def _combine_topk_swa_hip(
                topk_indices, query_start_loc, seq_lens, gather_lens,
                window_size, compress_ratio, topk, M, N,
                max_image_tokens=0, left_visible=None, right_visible=None,
            ):
                if (left_visible is None) != (right_visible is None):
                    raise ValueError("left_visible and right_visible must be provided together")
                if (
                    not (getattr(rmod, "_ON_GFX950", False) or getattr(rmod, "_ON_GFX942", False))
                    and topk_indices.is_cuda
                ):
                    from .fast_ops import get_hca_attn
                    ext = get_hca_attn()
                    if ext is not None and hasattr(ext, "combine_topk_swa_indices"):
                        swa_width = int(window_size) + int(max_image_tokens)
                        combined_topk = ((int(topk) + swa_width + align - 1) // align) * align
                        out = ext.combine_topk_swa_indices(
                            topk_indices, query_start_loc, seq_lens, gather_lens,
                            left_visible, right_visible,
                            int(window_size), int(compress_ratio), int(topk),
                            int(M), int(N), swa_width, combined_topk,
                        )
                        return out[0], out[1]
                return orig_comb(
                    topk_indices, query_start_loc, seq_lens, gather_lens,
                    window_size, compress_ratio, topk, M, N,
                    max_image_tokens=max_image_tokens,
                    left_visible=left_visible, right_visible=right_visible,
                )

            _combine_topk_swa_hip._ernie5_gfx1151_hip = True  # type: ignore[attr-defined]
            dsv4_rocm.combine_topk_swa_indices = _combine_topk_swa_hip

        # Replace 2 Triton kernels + cumsum in compute_global_topk_ragged_indices_and_indptr (decode metadata)
        orig_gtr = getattr(dsv4_rocm, "compute_global_topk_ragged_indices_and_indptr", None)
        if orig_gtr is not None and not getattr(orig_gtr, "_ernie5_gfx1151_hip", False):
            @functools.wraps(orig_gtr)
            def _compute_global_topk_ragged_hip(
                topk_indices, token_to_req_indices, block_table, block_size, is_valid_token,
            ):
                if (
                    not (getattr(rmod, "_ON_GFX950", False) or getattr(rmod, "_ON_GFX942", False))
                    and topk_indices.is_cuda
                ):
                    from .fast_ops import get_hca_attn
                    ext = get_hca_attn()
                    if ext is not None and hasattr(ext, "compute_global_topk_ragged"):
                        out = ext.compute_global_topk_ragged(
                            topk_indices, token_to_req_indices, block_table,
                            int(block_size), is_valid_token,
                        )
                        return out[0], out[1], out[2]
                return orig_gtr(
                    topk_indices, token_to_req_indices, block_table, block_size, is_valid_token,
                )

            _compute_global_topk_ragged_hip._ernie5_gfx1151_hip = True  # type: ignore[attr-defined]
            dsv4_rocm.compute_global_topk_ragged_indices_and_indptr = _compute_global_topk_ragged_hip
    except Exception:
        pass

    # Also patch DeepseekCompressor's _SAVE_PARTIAL_STATES_KERNEL to native HIP (supports BF16 input directly!)
    try:
        import vllm.models.deepseek_v4.compressor as comp_mod
        orig_sps = getattr(comp_mod, "_SAVE_PARTIAL_STATES_KERNEL", None)
        if orig_sps is not None and not getattr(orig_sps, "_ernie5_gfx1151_hip", False):
            def _save_partial_states_hip(
                *, kv, score, ape, positions, state_cache, slot_mapping,
                block_size, state_width, compress_ratio, pdl_kwargs=None,
            ):
                import torch
                if (
                    not (getattr(rmod, "_ON_GFX950", False) or getattr(rmod, "_ON_GFX942", False))
                    and kv.is_cuda
                    and kv.stride(1) == 1
                    and score.stride(1) == 1
                    and (kv.shape[1] % 4 == 0)
                    and ape.dtype == torch.float32
                    and state_cache.dtype == torch.float32
                ):
                    from .fast_ops import get_fast_ops
                    fops = get_fast_ops()
                    if fops is not None:
                        fops.save_partial_states(
                            kv, score, ape, positions, state_cache, slot_mapping,
                            int(block_size), int(state_width), int(compress_ratio),
                        )
                        return
                return orig_sps(
                    kv=kv.float() if kv.dtype != torch.float32 else kv,
                    score=score.float() if score.dtype != torch.float32 else score,
                    ape=ape, positions=positions, state_cache=state_cache,
                    slot_mapping=slot_mapping, block_size=block_size,
                    state_width=state_width, compress_ratio=compress_ratio,
                    pdl_kwargs=pdl_kwargs or {},
                )

            _save_partial_states_hip._ernie5_gfx1151_hip = True  # type: ignore[attr-defined]
            # Preserve register_warmup attribute if called during model init
            if hasattr(orig_sps, "register_warmup"):
                _save_partial_states_hip.register_warmup = orig_sps.register_warmup  # type: ignore[attr-defined]
            comp_mod._SAVE_PARTIAL_STATES_KERNEL = _save_partial_states_hip

        # Also replace Triton compress_norm_rope_store_triton (37x/tok inside FULL HIP graphs!)
        orig_comp = getattr(comp_mod, "compress_norm_rope_store_triton", None)
        if orig_comp is not None and not getattr(orig_comp, "_ernie5_gfx1151_hip", False):
            @functools.wraps(orig_comp)
            def _compress_norm_rope_store_hip(
                state_cache, num_actual, token_to_req_indices, positions, slot_mapping,
                block_table, block_size, state_width, cos_sin_cache, kv_cache,
                k_cache_metadata, pdl_kwargs, head_dim, rope_head_dim, compress_ratio,
                overlap, use_fp4_cache, rms_norm_weight, rms_norm_eps,
                quant_block, token_stride, scale_dim,
            ):
                import torch
                if (
                    not (getattr(rmod, "_ON_GFX950", False) or getattr(rmod, "_ON_GFX942", False))
                    and head_dim == 512
                    and rope_head_dim == 64
                    and compress_ratio == 128
                    and not overlap
                    and not use_fp4_cache
                    and state_cache.is_cuda
                    and state_cache.dtype == torch.float32
                    and kv_cache.dtype == torch.uint8
                ):
                    from .fast_ops import get_fast_ops
                    fops = get_fast_ops()
                    if fops is not None and hasattr(fops, "compress_norm_rope_store_c128"):
                        fops.compress_norm_rope_store_c128(
                            state_cache, int(num_actual), token_to_req_indices, positions,
                            slot_mapping, block_table, int(block_size), int(state_width),
                            rms_norm_weight, float(rms_norm_eps), cos_sin_cache,
                            kv_cache, k_cache_metadata.slot_mapping, int(kv_cache.shape[1]),
                        )
                        return
                return orig_comp(
                    state_cache=state_cache, num_actual=num_actual,
                    token_to_req_indices=token_to_req_indices, positions=positions,
                    slot_mapping=slot_mapping, block_table=block_table,
                    block_size=block_size, state_width=state_width,
                    cos_sin_cache=cos_sin_cache, kv_cache=kv_cache,
                    k_cache_metadata=k_cache_metadata, pdl_kwargs=pdl_kwargs,
                    head_dim=head_dim, rope_head_dim=rope_head_dim,
                    compress_ratio=compress_ratio, overlap=overlap,
                    use_fp4_cache=use_fp4_cache, rms_norm_weight=rms_norm_weight,
                    rms_norm_eps=rms_norm_eps, quant_block=quant_block,
                    token_stride=token_stride, scale_dim=scale_dim,
                )

            _compress_norm_rope_store_hip._ernie5_gfx1151_hip = True  # type: ignore[attr-defined]
            comp_mod.compress_norm_rope_store_triton = _compress_norm_rope_store_hip

        # Also replace prefill 2-stage Triton compressor (37x/chunk!) with single-pass Wave32 HIP
        orig_comp2 = getattr(comp_mod, "compress_norm_rope_store_two_stage_triton", None)
        if orig_comp2 is not None and not getattr(orig_comp2, "_ernie5_gfx1151_hip", False):
            @functools.wraps(orig_comp2)
            def _compress_norm_rope_store_two_stage_hip(
                state_cache, num_actual, token_to_req_indices, positions, slot_mapping,
                block_table, block_size, state_width, cos_sin_cache, kv_cache,
                k_cache_metadata, pdl_kwargs, head_dim, rope_head_dim, compress_ratio,
                overlap, use_fp4_cache, rms_norm_weight, rms_norm_eps,
                quant_block, token_stride, scale_dim, num_decode_tokens, compress_scratch,
            ):
                import torch
                if (
                    not (getattr(rmod, "_ON_GFX950", False) or getattr(rmod, "_ON_GFX942", False))
                    and head_dim == 512
                    and rope_head_dim == 64
                    and compress_ratio == 128
                    and not overlap
                    and not use_fp4_cache
                    and state_cache.is_cuda
                    and state_cache.dtype == torch.float32
                    and kv_cache.dtype == torch.uint8
                ):
                    from .fast_ops import get_fast_ops
                    fops = get_fast_ops()
                    if fops is not None and hasattr(fops, "compress_norm_rope_store_c128"):
                        fops.compress_norm_rope_store_c128(
                            state_cache, int(num_actual), token_to_req_indices, positions,
                            slot_mapping, block_table, int(block_size), int(state_width),
                            rms_norm_weight, float(rms_norm_eps), cos_sin_cache,
                            kv_cache, k_cache_metadata.slot_mapping, int(kv_cache.shape[1]),
                        )
                        return
                return orig_comp2(
                    state_cache=state_cache, num_actual=num_actual,
                    token_to_req_indices=token_to_req_indices, positions=positions,
                    slot_mapping=slot_mapping, block_table=block_table,
                    block_size=block_size, state_width=state_width,
                    cos_sin_cache=cos_sin_cache, kv_cache=kv_cache,
                    k_cache_metadata=k_cache_metadata, pdl_kwargs=pdl_kwargs,
                    head_dim=head_dim, rope_head_dim=rope_head_dim,
                    compress_ratio=compress_ratio, overlap=overlap,
                    use_fp4_cache=use_fp4_cache, rms_norm_weight=rms_norm_weight,
                    rms_norm_eps=rms_norm_eps, quant_block=quant_block,
                    token_stride=token_stride, scale_dim=scale_dim,
                    num_decode_tokens=num_decode_tokens, compress_scratch=compress_scratch,
                )

            _compress_norm_rope_store_two_stage_hip._ernie5_gfx1151_hip = True  # type: ignore[attr-defined]
            comp_mod.compress_norm_rope_store_two_stage_triton = _compress_norm_rope_store_two_stage_hip
    except Exception:
        pass

    # Replace Triton fused_q_kv_rmsnorm (40x/tok: 37 HCA + 3 DSpark!) with Wave32 HIP
    try:
        import vllm.models.deepseek_v4.attention as dsv4_attn_mod
        orig_qkv_norm = getattr(dsv4_attn_mod, "fused_q_kv_rmsnorm", None)
        if orig_qkv_norm is not None and not getattr(orig_qkv_norm, "_ernie5_gfx1151_hip", False):
            @functools.wraps(orig_qkv_norm)
            def _fused_q_kv_rmsnorm_hip(qr, kv, q_weight, kv_weight, eps):
                import torch
                if (
                    not (getattr(rmod, "_ON_GFX950", False) or getattr(rmod, "_ON_GFX942", False))
                    and qr.is_cuda
                    and qr.dim() == 2
                    and kv.dim() == 2
                    and qr.shape[1] == 1024
                    and kv.shape[1] == 512
                    and qr.dtype == torch.bfloat16
                    and kv.dtype == torch.bfloat16
                    and q_weight.dtype == torch.bfloat16
                    and kv_weight.dtype == torch.bfloat16
                    and qr.stride(-1) == 1
                    and kv.stride(-1) == 1
                ):
                    from .fast_ops import get_fast_ops
                    fops = get_fast_ops()
                    if fops is not None and hasattr(fops, "fused_q_kv_rmsnorm"):
                        out = fops.fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, float(eps))
                        return out[0], out[1]
                return orig_qkv_norm(qr, kv, q_weight, kv_weight, eps)

            _fused_q_kv_rmsnorm_hip._ernie5_gfx1151_hip = True  # type: ignore[attr-defined]
            dsv4_attn_mod.fused_q_kv_rmsnorm = _fused_q_kv_rmsnorm_hip
    except Exception:
        pass


def _patch_breakable_cudagraph() -> None:
    import vllm.config.vllm as vllm_config_mod

    vllm_config_mod.DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES = frozenset(
        vllm_config_mod.DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES
        | {"Ernie5_5_MoeForCausalLM", ERNIE5_DSPARK_ARCH}
    )
    vllm_config_mod.default_breakable_cudagraph_architectures.cache_clear()


def _patch_arch_config_convertors() -> None:
    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS,
        ModelArchConfigConvertorBase,
    )

    class Ernie5_5ModelArchConfigConvertor(ModelArchConfigConvertorBase):
        def is_deepseek_mla(self) -> bool:
            if hasattr(self.hf_text_config, "compress_ratios"):
                return getattr(self.hf_text_config, "head_dim", None) is not None
            return getattr(self.hf_text_config, "kv_lora_rank", None) is not None

    MODEL_ARCH_CONFIG_CONVERTORS.setdefault(
        ERNIE5_MODEL_TYPE, Ernie5_5ModelArchConfigConvertor
    )


def _undo_dspark_rewrite(spec_config) -> None:
    """Restore ERNIE-5.5 DSpark draft arch, mask token & quantization after SpeculativeConfig.__post_init__."""
    from .config import Ernie5_5Config

    draft = spec_config.draft_model_config
    hf_config = draft.hf_config
    if spec_config.method != "dspark" or hf_config.architectures != ["DSparkDraftModel"]:
        return
    if ERNIE5_DSPARK_ARCH not in draft.architectures:
        if not isinstance(hf_config, Ernie5_5Config):
            return
        if hf_config.bundles_dspark_draft:
            hf_config.to_bundled_dspark_draft()
    hf_config.model_type = ERNIE5_MODEL_TYPE
    hf_config.architectures = [ERNIE5_DSPARK_ARCH]
    hf_config.first_k_dense_replace = 0
    if getattr(hf_config, "mask_token_id", None) is None:
        hf_config.mask_token_id = getattr(hf_config, "dspark_noise_token_id", 1024)
    if getattr(hf_config, "enable_confidence_head", None) is None:
        hf_config.enable_confidence_head = True
    # Keep draft's own HTR quantization_config
    draft.quantization = spec_config.quantization
    draft._verify_quantization()


def _patch_speculative_config() -> None:
    import vllm.config.speculative as spec_mod
    from vllm.config.speculative import SpeculativeConfig

    if ERNIE5_MTP_MODEL_TYPE not in get_args(spec_mod.MTPModelTypes):
        spec_mod.MTPModelTypes = Literal[  # type: ignore[misc]
            (*get_args(spec_mod.MTPModelTypes), ERNIE5_MTP_MODEL_TYPE)
        ]

    orig_update_arch = SpeculativeConfig.update_arch_
    if not getattr(orig_update_arch, "_ernie5_dspark", False):
        @functools.wraps(orig_update_arch)
        def update_arch_(self):
            _undo_dspark_rewrite(self)
            orig_update_arch(self)
        update_arch_._ernie5_dspark = True  # type: ignore[attr-defined]
        SpeculativeConfig.update_arch_ = update_arch_


def _patch_dspark_attention_backend() -> None:
    """Ensure DSpark draft model for ernie5_5_moe inherits target attention backend like deepseek_v4."""
    try:
        import vllm.v1.worker.gpu.spec_decode.dspark.utils as dspark_utils
    except Exception:
        return
    orig = getattr(dspark_utils, "_resolve_dspark_attention_backend", None)
    if orig is None or getattr(orig, "_ernie5_dspark", False):
        return

    @functools.wraps(orig)
    def _resolve_wrapped(draft_model_config, spec_backend, target_backend):
        if (
            spec_backend is None
            and getattr(draft_model_config.hf_config, "model_type", None) == ERNIE5_MODEL_TYPE
        ):
            return target_backend
        return orig(draft_model_config, spec_backend, target_backend)

    _resolve_wrapped._ernie5_dspark = True  # type: ignore[attr-defined]
    dspark_utils._resolve_dspark_attention_backend = _resolve_wrapped


def _patch_htr_native_policy_for_dspark() -> None:
    """Allow method='dspark' through htr_native_policy.prepare_engine (Qwen3.8 MTP0/MTP3 untouched)."""
    try:
        import htr_native_policy as hnp
    except Exception:
        return
    orig = getattr(hnp, "prepare_engine", None)
    if orig is None or getattr(orig, "_ernie5_dspark", False):
        return

    @functools.wraps(orig)
    def prepare_engine_wrapped(args):
        spec = args.speculative_config
        if isinstance(spec, dict) and spec.get("method") == "dspark":
            args.speculative_config = None
            try:
                res = orig(args)
            finally:
                args.speculative_config = spec
            return res
        return orig(args)

    prepare_engine_wrapped._ernie5_dspark = True  # type: ignore[attr-defined]
    hnp.prepare_engine = prepare_engine_wrapped


def _dspark_draft_filter_enabled() -> bool:
    import os
    env = os.environ.get("ERNIE_DSPARK_DRAFT_FILTER")
    if env is not None:
        return env == "1"
    return os.environ.get("Q38SPEC_DRAFT_FILTER") == "1"


def _dspark_sample_filtered_or_orig(self, logits, idx_map, sample_pos, step, orig_fn):
    """Apply request top_k / exact top_p-only filtering + sparse cache on DSpark probabilistic draft."""
    import os
    import torch

    if (
        not _dspark_draft_filter_enabled()
        or self.draft_logits is None
        or self._d2t_scatter_index is not None
        or self.draft_watermarker is not None
        or getattr(self, "use_acceptance_estimator", False)
        or logits.dim() != 2
        or logits.dtype not in (torch.bfloat16, torch.float32)
        or logits.shape[1] < 256
    ):
        return orig_fn(self, logits, idx_map, sample_pos, step)

    import q38spec

    tk = getattr(self, "_q38_topk", None)
    tp = getattr(self, "_q38_topp", None)
    if tk is None or tp is None:
        n = self.temperature.shape[0]
        tk = torch.full((n,), 1 << 30, dtype=torch.int32, device=self.temperature.device)
        tp = torch.ones((n,), dtype=torch.float32, device=self.temperature.device)
        self._q38_topk = tk
        self._q38_topp = tp

    if logits.is_cuda and os.environ.get("HTR_Q38_FILTER_GUMBEL_FUSED", "1") == "1":
        max_reqs, num_steps = self.draft_logits.shape[0], self.draft_logits.shape[1]
        sp = getattr(self, "_q38_sparse_bufs", None)
        if sp is None or sp[0].shape[0] != max_reqs or sp[0].shape[1] != num_steps:
            dev = self.draft_logits.device
            sp = (
                torch.full((max_reqs, num_steps, q38spec.KC), -1, dtype=torch.int32, device=dev),
                torch.full((max_reqs, num_steps, q38spec.KC), float("-inf"), dtype=torch.float32, device=dev),
                torch.zeros((max_reqs, num_steps), dtype=torch.int32, device=dev),
            )
            self._q38_sparse_bufs = sp
        q38spec._STATE["draft_sparse"][self.draft_logits.data_ptr()] = sp

        step_cols_i64 = getattr(self, "_step_cols_i64", None)
        if step_cols_i64 is None or step_cols_i64.shape[0] != self._step_cols.shape[0]:
            step_cols_i64 = self._step_cols.to(torch.int64)
            self._step_cols_i64 = step_cols_i64

        write_dense = os.environ.get("HTR_SPARSE_REJECTION_ZERO_DENSE") != "1"
        sampled = q38spec._spec_sample_ext().q38_filter_gumbel_sample_native(
            logits.contiguous(),
            self.draft_logits if write_dense else None,
            step_cols_i64[step],
            sp[0],
            sp[1],
            sp[2],
            idx_map,
            self.temperature,
            tk,
            tp,
            self.seeds,
            sample_pos - 1,
            int(num_steps),
            bool(write_dense),
            bool(self.use_fp64_gumbel),
        )
        n_logged = getattr(self, "_ernie_dspark_filter_logged", 0)
        if n_logged < 2:
            self._ernie_dspark_filter_logged = n_logged + 1
            capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
            logger.info(
                "ERNIE_DSPARK_DRAFT_FILTER fused_v28 rows=%d step=%d capturing=%s write_dense=%s",
                logits.shape[0], int(step), capturing, write_dense,
            )
        return sampled

    # Pure-PyTorch fallback (graph-safe, supports both top_k<=64 and exact top_p-only)
    logits = q38spec.q38_filter_logits(logits, idx_map, self.temperature, tk, tp, in_place=True)
    return orig_fn(self, logits, idx_map, sample_pos, step)


def patch_dspark_sample_logits() -> None:
    """Apply ERNIE-5.5 MultiMax + filtered predictive drafting + fused NT-streaming Markov sampling inside DSparkSpeculator."""
    import os
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    if _dspark_draft_filter_enabled():
        try:
            import q38spec
            q38spec._patch_mrv2()
        except Exception as exc:
            logger.warning("ERNIE_DSPARK_DRAFT_FILTER failed to init q38spec mrv2 buffers: %r", exc)

    orig = DSparkSpeculator._sample_logits
    if not getattr(orig, "_ernie5_draft_logits", False):
        @functools.wraps(orig)
        def _sample_logits(self, logits, idx_map, sample_pos, step):
            postprocess = getattr(self.model, "postprocess_draft_logits", None)
            if postprocess is not None:
                logits = postprocess(logits)
            return _dspark_sample_filtered_or_orig(self, logits, idx_map, sample_pos, step, orig)

        _sample_logits._ernie5_draft_logits = True  # type: ignore[attr-defined]
        DSparkSpeculator._sample_logits = _sample_logits

    # Fuse the 7x sequential Markov loop (markov_w2 @ emb + base_logits + multimax + argmax)
    # with non-temporal streaming loads so the 103 MB markov_w2 doesn't thrash the 32 MB Infinity Cache!
    orig_seq = getattr(DSparkSpeculator, "_sample_sequential", None)
    if (
        orig_seq is not None
        and not getattr(orig_seq, "_ernie5_markov_hip", False)
        and os.environ.get("ERNIE_DSPARK_MARKOV_HIP", "1") == "1"
    ):
        @functools.wraps(orig_seq)
        def _sample_sequential_fast(self, num_reqs: int, head_hidden):
            import torch
            if self._draft_topk is not None or self.use_acceptance_estimator:
                return orig_seq(self, num_reqs, head_hidden)
            mhead = getattr(getattr(self.model, "model", None), "markov_head", None)
            lm_head = getattr(self.model, "lm_head", None)
            w2 = getattr(getattr(mhead, "markov_w2", None), "weight", None)
            if (
                w2 is not None
                and w2.is_cuda
                and w2.dim() == 2
                and w2.shape[1] == 256
                and w2.shape[0] == self.vocab_size
                and w2.dtype == torch.bfloat16
                and hasattr(self.model, "postprocess_draft_logits")
            ):
                from .fast_ops import get_fast_ops
                fops = get_fast_ops()
                if fops is not None and hasattr(fops, "dspark_markov_greedy"):
                    n_spec = self.num_speculative_steps
                    num_sample = num_reqs * n_spec
                    sample_hidden = head_hidden[self.sample_indices[:num_sample]]
                    base_logits = self.model.compute_draft_logits(sample_hidden)
                    vocab_size = base_logits.shape[-1]
                    if base_logits.dtype not in (torch.float32, torch.bfloat16):
                        base_logits = base_logits.float()
                    base_logits = base_logits.view(num_reqs, n_spec, vocab_size)
                    idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
                    sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
                    ranges = getattr(lm_head, "multimax_ranges", None)
                    ts = getattr(lm_head, "multimax_ts", None)
                    if ranges is not None and ranges.dtype != torch.float32:
                        ranges = ranges.float()
                    if ts is not None and ts.dtype != torch.float32:
                        ts = ts.float()
                    confidence_markov_embeds = []
                    prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

                    if self.draft_logits is None:
                        # Greedy fast path: 2 tiny NT-streaming HIP kernels per step, ZERO full-vocab buffers!
                        scratch = getattr(self, "_ernie_markov_scratch", None)
                        if scratch is None or scratch[0].shape[0] < self.max_num_reqs:
                            pv = torch.empty((self.max_num_reqs, 256), dtype=torch.float32, device=w2.device)
                            pi = torch.empty((self.max_num_reqs, 256), dtype=torch.int32, device=w2.device)
                            oi = torch.empty((self.max_num_reqs,), dtype=torch.int64, device=w2.device)
                            scratch = (pv, pi, oi)
                            self._ernie_markov_scratch = scratch
                        pv, pi, oi = scratch[0][:num_reqs], scratch[1][:num_reqs], scratch[2][:num_reqs]
                        for i in range(n_spec):
                            markov_embed = self.model.markov_embed(prev)
                            if self.use_confidence_head:
                                confidence_markov_embeds.append(markov_embed)
                            fops.dspark_markov_greedy(
                                w2, markov_embed.to(torch.bfloat16), base_logits[:, i],
                                ranges, ts, pv, pi, oi,
                            )
                            draft_sampled_i = self.model.map_draft_to_target(oi)
                            self.draft_tokens[:num_reqs, i] = draft_sampled_i
                            prev = draft_sampled_i
                    else:
                        # Probabilistic fast path: fuse markov_w2 @ emb + base_logits + multimax into 1 NT kernel
                        lbuf = getattr(self, "_ernie_markov_logits_buf", None)
                        if lbuf is None or lbuf.shape[0] < self.max_num_reqs or lbuf.shape[1] != vocab_size:
                            lbuf = torch.empty((self.max_num_reqs, vocab_size), dtype=torch.float32, device=w2.device)
                            self._ernie_markov_logits_buf = lbuf
                        out_lg = lbuf[:num_reqs]
                        for i in range(n_spec):
                            markov_embed = self.model.markov_embed(prev)
                            if self.use_confidence_head:
                                confidence_markov_embeds.append(markov_embed)
                            fops.dspark_markov_bias_multimax(
                                w2, markov_embed.to(torch.bfloat16), base_logits[:, i],
                                ranges, ts, out_lg,
                            )
                            # MultiMax already applied inline -> filtered predictive drafting (or orig)
                            draft_sampled_i = _dspark_sample_filtered_or_orig(
                                self, out_lg, idx_map[:, i], sample_pos[:, i], i, orig
                            )
                            self.draft_tokens[:num_reqs, i] = draft_sampled_i
                            prev = draft_sampled_i

                    if self.use_confidence_head:
                        confidence = self.model.compute_confidence(
                            sample_hidden,
                            torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
                        )
                        self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                            num_reqs, n_spec
                        )
                    return
            return orig_seq(self, num_reqs, head_hidden)

        _sample_sequential_fast._ernie5_markov_hip = True  # type: ignore[attr-defined]
        DSparkSpeculator._sample_sequential = _sample_sequential_fast


# ------------------------------------------------------------------ In-Place K=5 Split-Plane (0 Extra VRAM!)
_REPACKED_K5_PTRS: set[int] = set()
_K5_SP_LINEAR_PATCHED = False


_K5SP_CFG: tuple[int, bool, int, int] | None = None
import os as _os_k5u
_USE_K5_UNSPLIT_FAST = _os_k5u.environ.get("ERNIE_K5_UNSPLIT_FAST", "1") == "1" 


def _get_k5sp_cfg() -> tuple[int, bool, int, int]:
    global _K5SP_CFG
    if _K5SP_CFG is None:
        import os
        _K5SP_CFG = (
            int(os.environ.get("ERNIE_K5_SP_WMMA_MIN_M", "9")),
            os.environ.get("ERNIE_K5_SP_NS1", "0") == "1",
            int(os.environ.get("ERNIE_K5_SP_NS1_MIN_MT", "512")),
            int(os.environ.get("ERNIE_K5_SP_M4_MAX_NS", "0")),
        )
    return _K5SP_CFG



def _use_k5_unsplit_gemv(M: int, mt: int) -> bool:
    """Empirically optimal gfx1151 crossover between K=5 unsplit GEMV and WMMA.
    - M == 1: gemv_k5_unsplit_epi beats htr_ops.gemv_epi on small (mt<=128, 1.53x) and lm_head (mt>=4096, 236 GB/s).
    - 2 <= M <= 16: with optimal ns=8/16 (_k5_unsplit_nsplit), gemv_k5_unsplit_epi beats WMMA across ALL
      ERNIE-5.5 shapes (1.9-2.3x faster at T=9..16, saving -20.5 ms/step on B=2 concurrent decode!).
    """
    if M == 1:
        return mt <= 128 or mt >= 4096
    if 2 <= M <= 16:
        return True
    return False


def _k5_unsplit_nsplit(mt: int, nt: int, M: int) -> int:
    """Empirically optimal K-split for gemv_k5_unsplit_epi on gfx1151 (40 CUs, 32 MiB MALL).
    Avoids ns=24/32 partial-buffer & atomic-barrier thrashing (saves -7.75 ms/step @ T=6,
    -18.8 ms/step @ T=12, -27.4 ms/step @ T=16 across all ERNIE-5.5 dense K=5 projections!).
    """
    if nt < 16:
        return max(1, nt)
    if M >= 8:
        if mt == 128 and nt >= 320 and M >= 12:
            return 16
        return 8
    if mt <= 128 or nt <= 96 or (mt > 1792 and M >= 4):
        return 8
    return 16

def htr_k5sp_linear_pre_prologued(xt, xs, tok, qweight, scales, sv, bias=None):
    """Execute an ERNIE-5.5 K=5 linear (unsplit or repacked) using pre-computed (xt, xs, tok)."""
    import torch
    import htr.ops as htr_ops
    M = xt.shape[0]
    mt, nt = qweight.shape[0], qweight.shape[1]
    if M == 0:
        return xt.new_empty((0, mt * 16), dtype=torch.bfloat16)
    wmma_min_m, use_ns1_tall, ns1_min_mt, m4_max_ns = _get_k5sp_cfg()
    from .fast_ops import get_k5_sp_prefill
    sp_ext = get_k5_sp_prefill()
    if qweight.data_ptr() not in _REPACKED_K5_PTRS:
        # Unsplit qw [mt, nt, 40]: zero-repack full-speed decode + concurrent T=9..16 GEMV + v28 4x4 WMMA prefill!
        if (
            _USE_K5_UNSPLIT_FAST
            and (mt % 8) == 0
            and sp_ext is not None
            and hasattr(sp_ext, "gemv_k5_unsplit_epi")
            and _use_k5_unsplit_gemv(M, mt)
        ):
            cnt = htr_ops.counters(xt.device)
            ns = _k5_unsplit_nsplit(mt, nt, M)
            _e8 = int(htr_ops.EPI8 and htr_ops._ns_eff(mt, nt, ns) % 8 == 0)
            return sp_ext.gemv_k5_unsplit_epi(
                qweight, xt, xs, tok, scales, float(htr_ops.CB), sv, bias, cnt, ns, _e8
            )
        if M < wmma_min_m and M <= htr_ops.GEMV_MAX_M:
            cnt = htr_ops.counters(xt.device)
            ns = htr_ops.nsplit_for(mt, nt, M)
            _e8 = int(htr_ops.EPI8 and htr_ops._ns_eff(mt, nt, ns) % 8 == 0)
            return htr_ops.ext().gemv_epi(
                qweight, xt, xs, tok, scales, htr_ops.CB, sv, bias, cnt, ns, 0, _e8
            )
        if sp_ext is not None and hasattr(sp_ext, "dense_prefill_wmma_unsplit") and (mt % 8) == 0:
            return sp_ext.dense_prefill_wmma_unsplit(qweight, xt, tok, scales, float(htr_ops.CB), sv, bias, 0)
        return htr_ops._dpw_ext().dense_prefill_wmma(qweight, xt, tok, scales, htr_ops.CB, sv, bias)
    flat = qweight.view(-1)
    n16 = mt * nt * 32
    p16 = flat[:n16].view(mt, nt, 32)
    p4 = flat[n16:].view(mt, nt, 8)
    if M < wmma_min_m:
        if use_ns1_tall and mt >= ns1_min_mt and M <= 8 and sp_ext is not None and hasattr(sp_ext, "gemv_k5_sp16_ns1"):
            return sp_ext.gemv_k5_sp16_ns1(p16, p4, xt, xs, tok, scales, float(htr_ops.CB), sv, bias)
        cnt = htr_ops.counters(xt.device)
        k5sp = htr_ops._k5sp_ext()
        ns = htr_ops.nsplit_for(mt, nt, M)
        if m4_max_ns > 0 and M >= 4 and mt >= 64 and ns > m4_max_ns:
            ns = m4_max_ns
        _e8 = int(htr_ops.EPI8 and htr_ops._ns_eff(mt, nt, ns) % 8 == 0)
        return k5sp.gemv_k5_sp16_epi(
            p16, p4, xt, xs, tok, scales, htr_ops.CB, sv, bias, cnt, ns, 0, _e8
        )
    return sp_ext.dense_prefill_wmma_sp16(p16, p4, xt, tok, scales, float(htr_ops.CB), sv, bias, 0)


def _patch_htr_linear_for_repacked_k5() -> None:
    """Route in-place repacked ERNIE-5.5 K=5 weights through aligned b128+SAD_HI GEMV (M<=16) & WMMA (M>=17)."""
    global _K5_SP_LINEAR_PATCHED
    if _K5_SP_LINEAR_PATCHED:
        return
    try:
        import htr.ops as htr_ops
    except Exception:
        return
    orig_impl = getattr(htr_ops, "_linear_impl", None)
    if orig_impl is None or getattr(orig_impl, "_ernie5_k5sp_inplace", False):
        _K5_SP_LINEAR_PATCHED = True
        return

    import os
    # Route M >= 9 directly to single-pass split-plane WMMA (NB=1 at M=9..16 reads weights 1x instead of 2x chunked GEMV!)
    wmma_min_m = int(os.environ.get("ERNIE_K5_SP_WMMA_MIN_M", "9"))
    use_ns1_tall = os.environ.get("ERNIE_K5_SP_NS1", "0") == "1"
    # mt >= 512 covers o_gate (46x/step, 64 WGs=1.6 WGs/CU), wq_b (40x), q_b_proj/kv_b_proj (12x), and lm_head (2x)
    ns1_min_mt = int(os.environ.get("ERNIE_K5_SP_NS1_MIN_MT", "512"))
    # Optional cap on split-K ns for M>=4 ERNIE-5.5 mt>=256 linears (0 = default htr_ops.nsplit_for)
    m4_max_ns = int(os.environ.get("ERNIE_K5_SP_M4_MAX_NS", "0"))

    wmma_unsplit_max_m = int(os.environ.get("ERNIE_K5_UNSPLIT_WMMA_MAX_M", "8192"))

    @functools.wraps(orig_impl)
    def _linear_impl_k5sp(x2, qweight, scales, su, sv, bias):
        if qweight.data_ptr() not in _REPACKED_K5_PTRS:
            import torch
            M = x2.shape[0]
            if (
                _serving_ernie5
                and x2.dtype == torch.bfloat16
                and qweight.dim() == 3
                and qweight.shape[2] == 40
                and (qweight.shape[0] % 8) == 0
            ):
                mt = qweight.shape[0]
                if _USE_K5_UNSPLIT_FAST and _use_k5_unsplit_gemv(M, mt):
                    from .fast_ops import get_k5_sp_prefill
                    sp_ext = get_k5_sp_prefill()
                    if sp_ext is not None and hasattr(sp_ext, "gemv_k5_unsplit_epi"):
                        if not getattr(_linear_impl_k5sp, "_w24_logged", False):
                            _linear_impl_k5sp._w24_logged = True
                            print(f"ERNIE_WAVE24_K5_UNSPLIT_ACTIVE first_shape={tuple(qweight.shape)} M={M}", flush=True)
                        nt = qweight.shape[1]
                        xt, xs, tok = htr_ops.ext().prologue(x2, su)
                        cnt = htr_ops.counters(x2.device)
                        ns = htr_ops.nsplit_for(mt, nt, min(M, 8))
                        if nt >= 32:
                            ns = min(32, max(8, ((ns + 7) // 8) * 8))
                        _e8 = int(htr_ops.EPI8 and htr_ops._ns_eff(mt, nt, ns) % 8 == 0)
                        return sp_ext.gemv_k5_unsplit_epi(
                            qweight, xt, xs, tok, scales, float(htr_ops.CB), sv, bias, cnt, ns, _e8
                        )
                if wmma_min_m <= M <= wmma_unsplit_max_m:
                    from .fast_ops import get_k5_sp_prefill
                    sp_ext = get_k5_sp_prefill()
                    if sp_ext is not None and hasattr(sp_ext, "dense_prefill_wmma_unsplit"):
                        xt, xs, tok = htr_ops.ext().prologue(x2, su)
                        return sp_ext.dense_prefill_wmma_unsplit(
                            qweight, xt, tok, scales, float(htr_ops.CB), sv, bias, 0
                        )
            return orig_impl(x2, qweight, scales, su, sv, bias)
        import torch
        M = x2.shape[0]
        mt, nt = qweight.shape[0], qweight.shape[1]
        if M == 0:
            return x2.new_empty((0, mt * 16))
        flat = qweight.view(-1)
        n16 = mt * nt * 32
        p16 = flat[:n16].view(mt, nt, 32)
        p4 = flat[n16:].view(mt, nt, 8)
        E = htr_ops.ext()
        if M < wmma_min_m:
            od = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}[x2.dtype]
            xt, xs, tok = E.prologue(x2, su)
            # Zero-partial, zero-atomic single-pass ns=1 for tall matrices (o_gate 8192x1024, wq_b 32768x1024, q_b/kv_b, lm_head)
            if use_ns1_tall and mt >= ns1_min_mt and M <= 8 and od == 0:
                from .fast_ops import get_k5_sp_prefill
                sp_ext = get_k5_sp_prefill()
                if sp_ext is not None and hasattr(sp_ext, "gemv_k5_sp16_ns1"):
                    return sp_ext.gemv_k5_sp16_ns1(p16, p4, xt, xs, tok, scales, float(htr_ops.CB), sv, bias)
            cnt = htr_ops.counters(x2.device)
            k5sp = htr_ops._k5sp_ext()
            if M <= htr_ops.GEMV_MAX_M:
                ns = htr_ops.nsplit_for(mt, nt, M)
                if m4_max_ns > 0 and M >= 4 and mt >= 64 and ns > m4_max_ns:
                    ns = m4_max_ns
                _e8 = int(htr_ops.EPI8 and htr_ops._ns_eff(mt, nt, ns) % 8 == 0)
                return k5sp.gemv_k5_sp16_epi(
                    p16, p4, xt, xs, tok, scales, htr_ops.CB, sv, bias, cnt, ns, od, _e8
                )
            outs = []
            for r0 in range(0, M, htr_ops.GEMV_MAX_M):
                r1 = min(M, r0 + htr_ops.GEMV_MAX_M)
                ns = htr_ops.nsplit_for(mt, nt, r1 - r0)
                _e8 = int(htr_ops.EPI8 and htr_ops._ns_eff(mt, nt, ns) % 8 == 0)
                outs.append(
                    k5sp.gemv_k5_sp16_epi(
                        p16, p4, xt[r0:r1], xs[r0:r1], tok[r0:r1],
                        scales, htr_ops.CB, sv, bias, cnt, ns, od, _e8,
                    )
                )
            return torch.cat(outs)
        from .fast_ops import get_k5_sp_prefill
        sp_ext = get_k5_sp_prefill()
        x_in = x2 if x2.dtype == torch.bfloat16 else x2.to(torch.bfloat16)
        xt, xs, tok = E.prologue(x_in, su)
        y = sp_ext.dense_prefill_wmma_sp16(p16, p4, xt, tok, scales, float(htr_ops.CB), sv, bias, 0)
        return y if x2.dtype == torch.bfloat16 else y.to(x2.dtype)

    _linear_impl_k5sp._ernie5_k5sp_inplace = True  # type: ignore[attr-defined]
    htr_ops._linear_impl = _linear_impl_k5sp
    _K5_SP_LINEAR_PATCHED = True


def repack_ernie_k5_splitplane_inplace(model) -> None:
    """Repack ERNIE-5.5 K=5 dense weights into 16B+4B split-plane layout in-place (0 extra VRAM)."""
    global _serving_ernie5
    _serving_ernie5 = True
    import os
    import torch
    if os.environ.get("ERNIE_K5_SP_INPLACE", "0") != "1":
        _patch_htr_linear_for_repacked_k5()
        return
        return
    from .fast_ops import get_k5_sp_prefill
    sp_ext = get_k5_sp_prefill()
    if sp_ext is None:
        return
    # Verify k5_splitplane.so is loadable too
    try:
        import htr.ops as htr_ops
        htr_ops._k5sp_ext()
    except Exception as exc:
        logger.warning("ERNIE K5 split-plane disabled (k5_splitplane.so unavailable: %r)", exc)
        return
    _patch_htr_linear_for_repacked_k5()

    # Collect wo_a_groups parameter pointers so we never touch them (htr_wo_a_ernie.so owns them!)
    skip_ptrs: set[int] = set()
    for mod in model.modules():
        if hasattr(mod, "wo_a_groups"):
            for g in mod.wo_a_groups:
                qw = getattr(g, "htr_qweight", None)
                if qw is not None:
                    skip_ptrs.add(qw.data_ptr())

    n_repacked = 0
    total_bytes = 0
    for name, mod in model.named_modules():
        qw = getattr(mod, "htr_qweight", None)
        if qw is None or not isinstance(qw, torch.Tensor):
            continue
        if not qw.is_cuda or qw.dim() != 3 or qw.shape[2] != 40 or qw.dtype != torch.int32:
            continue
        if (qw.shape[0] % 8) != 0:
            continue
        ptr = qw.data_ptr()
        if ptr in skip_ptrs or ptr in _REPACKED_K5_PTRS:
            continue
        packed = sp_ext.repack_k5_packed(qw.data.contiguous())
        qw.data = packed
        _REPACKED_K5_PTRS.add(qw.data_ptr())
        n_repacked += 1
        total_bytes += qw.numel() * 4
    if n_repacked > 0:
        logger.info(
            "ERNIE-5.5 K=5 in-place split-plane repack complete: %d matrices (%.2f GiB, 0.00 GiB extra VRAM)",
            n_repacked,
            total_bytes / (1024 ** 3),
        )


def _register_parsers() -> None:
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    ReasoningParserManager.register_lazy_module(
        "ernie5",
        "vllm_ernie_htr.reasoning_parser",
        "Ernie5ParserReasoningAdapter",
    )
    ToolParserManager.register_lazy_module(
        "ernie5",
        "vllm_ernie_htr.tool_parser",
        "Ernie5ToolParser",
    )


def _register_renderer() -> None:
    from vllm.renderers.registry import RENDERER_REGISTRY, RendererRegistry
    from vllm.tokenizers.registry import TokenizerRegistry

    TokenizerRegistry.register(
        ERNIE5_TOKENIZER_MODE, "vllm.tokenizers.hf", "CachedHfTokenizer"
    )
    RENDERER_REGISTRY.register(
        ERNIE5_TOKENIZER_MODE,
        "vllm_ernie_htr.renderer",
        "Ernie5_5Renderer",
    )

    orig = RendererRegistry.load_renderer
    if getattr(orig, "_ernie5_default_renderer", False):
        return

    @functools.wraps(orig)
    def load_renderer(self, renderer_mode, config, tokenizer):
        global _serving_ernie5
        if config.model_config.hf_config.model_type == ERNIE5_MODEL_TYPE:
            _serving_ernie5 = True
            if renderer_mode == "hf":
                renderer_mode = ERNIE5_TOKENIZER_MODE
        return orig(self, renderer_mode, config, tokenizer)

    load_renderer._ernie5_default_renderer = True  # type: ignore[attr-defined]
    RendererRegistry.load_renderer = load_renderer


def _without_injected_enable_thinking(orig):
    if getattr(orig, "_ernie5_enable_thinking", False):
        return orig

    @functools.wraps(orig)
    def build_chat_params(self, *args, **kwargs):
        params = orig(self, *args, **kwargs)
        template_kwargs = params.chat_template_kwargs
        if (
            not _serving_ernie5
            or "enable_thinking" not in template_kwargs
            or "enable_thinking" in (self.chat_template_kwargs or {})
        ):
            return params
        return dataclasses.replace(
            params,
            chat_template_kwargs={
                k: v for k, v in template_kwargs.items() if k != "enable_thinking"
            },
        )

    build_chat_params._ernie5_enable_thinking = True  # type: ignore[attr-defined]
    return build_chat_params


def _patch_enable_thinking() -> None:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    for cls in (ChatCompletionRequest, ResponsesRequest):
        cls.build_chat_params = _without_injected_enable_thinking(cls.build_chat_params)


def apply_patches() -> None:
    global _applied
    if _applied:
        return
    _applied = True
    _patch_config_registry()
    _patch_breakable_cudagraph()
    _patch_arch_config_convertors()
    _patch_speculative_config()
    _patch_dspark_attention_backend()
    _patch_htr_native_policy_for_dspark()
    _register_parsers()
    _register_renderer()
    _patch_enable_thinking()
    _patch_rocm_paged_mqa_logits_triton()
    if _dspark_draft_filter_enabled():
        try:
            import q38spec
            q38spec._patch_mrv2()
        except Exception as exc:
            logger.warning("ERNIE_DSPARK_DRAFT_FILTER early q38spec init failed: %r", exc)

