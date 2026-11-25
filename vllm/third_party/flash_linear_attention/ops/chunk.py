# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import torch

from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_o import chunk_fwd_o
from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .cumsum import chunk_local_cumsum
from .l2norm import l2norm_fwd
from .solve_tril import solve_tril
from .utils import FLA_CHUNK_SIZE, SUPPRESS_LEVEL, input_guard

import contextlib as _vv_contextlib
import functools as _vv_functools


def _vv_token_strided(t):
    # [1, T, H, V] with dense (H, V) inner dims and only a wider token stride: the DIRECT post_conv v-view layout.
    return (isinstance(t, torch.Tensor) and t.dim() == 4 and t.shape[0] == 1 and t.stride(-1) == 1
            and t.stride(-2) == t.shape[-1] and t.stride(1) >= t.shape[-2] * t.shape[-1])


def _input_guard_vview(fn):
    """utils.input_guard (contiguous() every tensor arg + device ctx) except that v (forward positional arg 3, after ctx)
    is passed through when it is a token-strided view (overlay nb2-gdnvv-ab-v1).  Every other layout is made contiguous."""

    @_vv_functools.wraps(fn)
    def wrapper(*args, **kwargs):
        keep = len(args) > 3 and _vv_token_strided(args[3]) and not args[3].is_contiguous()
        contiguous_args = tuple(
            a if (not isinstance(a, torch.Tensor) or (keep and i == 3)) else a.contiguous() for i, a in enumerate(args)
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }
        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break
        if tensor is not None:
            ctx = torch.accelerator.device_index(tensor.device.index)
        else:
            ctx = _vv_contextlib.nullcontext()
        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper
from .wy_fast import recompute_w_u_fwd



_GDN_WY_AOT = None
def _get_gdn_wy_aot():
    global _GDN_WY_AOT
    if _GDN_WY_AOT is None:
        import os as _os
        if _os.environ.get("HTR_GDN_WY_AOT", "1") == "1":
            for cand in (
                _os.environ.get("HTR_GDN_WY_HSACO", ""),
                _os.path.join(_os.environ.get("HTR_NATIVE_DIR", "/opt/vllm-htr/native-bin"), "gdn_wy_fused_aot.hsaco"),
                "/opt/vllm-htr/native-v026/gdn_wy_fused_aot.hsaco",
            ):
                if cand and _os.path.isfile(cand):
                    import direct_hip_module_v1 as _hm
                    _GDN_WY_AOT = (_hm, _hm.Module(cand))
                    break
        if _GDN_WY_AOT is None:
            _GDN_WY_AOT = False
    return _GDN_WY_AOT


_GDN_O_AOT = None
def _get_gdn_o_aot():
    global _GDN_O_AOT
    if _GDN_O_AOT is None:
        import os as _os
        if _os.environ.get("HTR_GDN_O_AOT", "1") == "1":
            for cand in (
                _os.environ.get("HTR_GDN_O_HSACO", ""),
                _os.path.join(_os.environ.get("HTR_NATIVE_DIR", "/opt/vllm-htr/native-bin"), "gdn_chunk_o_aot.hsaco"),
                "/opt/vllm-htr/native-v026/gdn_chunk_o_aot.hsaco",
            ):
                if cand and _os.path.isfile(cand):
                    import direct_hip_module_v1 as _hm
                    _GDN_O_AOT = (_hm, _hm.Module(cand))
                    break
        if _GDN_O_AOT is None:
            _GDN_O_AOT = False
    return _GDN_O_AOT


_GDN_CHUNK_H_AOT = None
def _get_gdn_chunk_h_aot():
    global _GDN_CHUNK_H_AOT
    if _GDN_CHUNK_H_AOT is None:
        import os as _os
        if _os.environ.get("HTR_GDN_CHUNK_H_AOT", "1") == "1":
            for cand in (
                _os.environ.get("HTR_GDN_CHUNK_H_HSACO", ""),
                _os.path.join(_os.environ.get("HTR_NATIVE_DIR", "/opt/vllm-htr/native-bin"), "gdn_chunk_h_aot.hsaco"),
                "/opt/vllm-htr/native-v026/gdn_chunk_h_aot.hsaco",
            ):
                if cand and _os.path.isfile(cand):
                    import direct_hip_module_v1 as _hm
                    _GDN_CHUNK_H_AOT = (_hm, _hm.Module(cand))
                    break
        if _GDN_CHUNK_H_AOT is None:
            _GDN_CHUNK_H_AOT = False
    return _GDN_CHUNK_H_AOT

def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    core_attn_out: torch.Tensor | None = None,
):
    _wy_aot = _get_gdn_wy_aot()
    if (
        _wy_aot
        and FLA_CHUNK_SIZE == 64
        and k.ndim == 4 and k.shape[0] == 1 and k.shape[3] == 128
        and v.ndim == 4 and v.shape[0] == 1 and v.shape[3] == 128
        and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
        and g.dtype == torch.float32 and beta.dtype == torch.float32
        and k.stride(-1) == 1 and v.stride(-1) == 1
        and (k.data_ptr() & 15) == 0 and (v.data_ptr() & 15) == 0
        and (k.stride(1) & 7) == 0 and (v.stride(1) & 7) == 0
        and cu_seqlens is not None
    ):
        import ctypes as _ct
        from .index import prepare_chunk_indices as _prep_cidx
        _hm, _m = _wy_aot
        I32, I64 = _ct.c_int32, _ct.c_int64
        _H = int(v.shape[2])
        _Hg = int(k.shape[2])
        _gqa = _H // _Hg
        _cu = cu_seqlens.to(torch.int32).contiguous()
        _cidx = (chunk_indices if chunk_indices is not None else _prep_cidx(_cu, 64)).to(torch.int32).contiguous()
        _gc = torch.empty_like(g)
        w = torch.empty_like(v)
        u = torch.empty_like(v)
        _m.launch(
            "gdn_wy_fused_bf16",
            (_H, int(_cidx.shape[0]), 1),
            (256, 1, 1),
            [
                _hm.ptr(k), _hm.ptr(v), _hm.ptr(g), _hm.ptr(beta),
                _hm.ptr(_gc), _hm.ptr(w), _hm.ptr(u),
                _hm.ptr(_cidx), _hm.ptr(_cu),
                I64(k.stride(1)), I64(k.stride(2)),
                I64(v.stride(1)), I64(v.stride(2)),
                I64(g.stride(1)), I64(g.stride(2)),
                I64(beta.stride(1)), I64(beta.stride(2)),
                I64(_gc.stride(1)), I64(_gc.stride(2)),
                I64(w.stride(1)), I64(w.stride(2)),
                I64(u.stride(1)), I64(u.stride(2)),
                I32(_H), I32(_gqa),
            ],
        )
        g = _gc
        A = None
    else:
        g = chunk_local_cumsum(
            g, chunk_size=FLA_CHUNK_SIZE, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
        )
        # obtain WY representation. u is actually the new v.
        A = chunk_scaled_dot_kkt_fwd(
            k=k,
            beta=beta,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            output_dtype=torch.float32,
        )
        A = solve_tril(
            A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype
        )
        w, u = recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            g_cumsum=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
    _ch_aot = _get_gdn_chunk_h_aot()
    if (
        _ch_aot
        and FLA_CHUNK_SIZE == 64
        and k.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and u.dtype == torch.bfloat16
        and (g is None or g.dtype == torch.float32)
        and k.shape[0] == 1
        and k.shape[-1] == 128
        and u.shape[-1] == 128
        and k.is_contiguous()
        and w.is_contiguous()
        and u.is_contiguous()
        and (g is None or g.is_contiguous())
        and (initial_state is None or (initial_state.is_contiguous() and initial_state.dtype in (torch.float32, torch.bfloat16)))
    ):
        import ctypes as _ct
        from .index import prepare_chunk_indices as _pci, prepare_chunk_offsets as _pco
        _hm, _m = _ch_aot
        I32, I64 = _ct.c_int32, _ct.c_int64
        _B, _T, _Hg, _K = k.shape
        _H, _V = u.shape[-2], u.shape[-1]
        if cu_seqlens is not None:
            if chunk_indices is None:
                chunk_indices = _pci(cu_seqlens, 64)
            _N = len(cu_seqlens) - 1
            _NT = len(chunk_indices)
            _cu_i32 = cu_seqlens.to(torch.int32).contiguous()
            _co_i32 = (chunk_offsets if chunk_offsets is not None else _pco(cu_seqlens, 64)).to(torch.int32).contiguous()
        else:
            _N = _B
            _NT = (_T + 63) // 64
            _cu_i32 = None
            _co_i32 = None
        h = k.new_empty(_B, _NT, _H, _V, _K)
        v_new = torch.empty_like(u)
        final_state = k.new_empty(_N, _H, _V, _K, dtype=torch.float32) if output_final_state else None
        with torch.cuda.device(k.device):
            _m.launch(
                "gdn_chunk_h_bf16_k128_v128_bt64",
                (4, _N * _H, 1),
                (256, 1, 1),
                [
                    _hm.ptr(k), _hm.ptr(u), _hm.ptr(w), _hm.ptr(v_new),
                    _hm.ptr(g) if g is not None else I64(0),
                    _hm.ptr(h),
                    _hm.ptr(initial_state) if initial_state is not None else I64(0),
                    _hm.ptr(final_state) if final_state is not None else I64(0),
                    _hm.ptr(_cu_i32) if _cu_i32 is not None else I64(0),
                    _hm.ptr(_co_i32) if _co_i32 is not None else I64(0),
                    I32(_T), I32(_H), I32(_Hg),
                    I32(1 if (initial_state is not None and initial_state.dtype == torch.bfloat16) else 0),
                    I32(0),
                ],
            )
    else:
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
        )
    _o_aot = _get_gdn_o_aot()
    if (
        _o_aot
        and FLA_CHUNK_SIZE == 64
        and q.ndim == 4 and q.shape[0] == 1 and q.shape[3] == 128
        and k.ndim == 4 and k.shape[0] == 1 and k.shape[3] == 128
        and v_new.ndim == 4 and v_new.shape[0] == 1 and v_new.shape[3] == 128
        and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v_new.dtype == torch.bfloat16
        and h.dtype == torch.bfloat16 and g.dtype == torch.float32
        and q.stride(-1) == 1 and k.stride(-1) == 1 and v_new.stride(-1) == 1 and h.stride(-1) == 1
        and (q.data_ptr() & 15) == 0 and (k.data_ptr() & 15) == 0 and (v_new.data_ptr() & 15) == 0 and (h.data_ptr() & 15) == 0
        and (q.stride(1) & 7) == 0 and (k.stride(1) & 7) == 0 and (v_new.stride(1) & 7) == 0 and (h.stride(3) & 7) == 0
        and h.ndim == 5
        and (core_attn_out is None or (core_attn_out.dtype == torch.bfloat16 and core_attn_out.is_contiguous() and core_attn_out.numel() == v_new.numel() and (core_attn_out.data_ptr() & 15) == 0))
        and cu_seqlens is not None
    ):
        import ctypes as _ct
        from .index import prepare_chunk_indices as _prep_cidx
        _hm, _m = _o_aot
        I32, I64, F32 = _ct.c_int32, _ct.c_int64, _ct.c_float
        _H = int(v_new.shape[2])
        _Hg = int(k.shape[2])
        _cu = cu_seqlens.to(torch.int32).contiguous()
        _cidx = (chunk_indices if chunk_indices is not None else _prep_cidx(_cu, 64)).to(torch.int32).contiguous()
        o = core_attn_out.view(v_new.shape) if core_attn_out is not None else torch.empty_like(v_new)
        with torch.cuda.device(q.device):
            _m.launch(
                "gdn_chunk_o_bf16",
                (_H, int(_cidx.shape[0]), 1),
                (256, 1, 1),
                [
                    _hm.ptr(q), _hm.ptr(k), _hm.ptr(v_new), _hm.ptr(h), _hm.ptr(g), _hm.ptr(o),
                    _hm.ptr(_cidx), _hm.ptr(_cu),
                    I64(q.stride(1)), I64(q.stride(2)), I64(k.stride(1)), I64(k.stride(2)),
                    I64(v_new.stride(1)), I64(v_new.stride(2)), I64(h.stride(1)), I64(h.stride(2)), I64(h.stride(3)),
                    I64(g.stride(1)), I64(g.stride(2)), I64(o.stride(-3)), I64(o.stride(-2)),
                    F32(float(scale)), I32(_H), I32(_H // _Hg),
                ],
            )
    else:
        o = chunk_fwd_o(
            q=q,
            k=k,
            v=v_new,
            h=h,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            core_attn_out=core_attn_out,
        )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @_input_guard_vview
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        core_attn_out: torch.Tensor | None = None,
    ):
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        g, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            core_attn_out=core_attn_out,
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        if core_attn_out is not None:
            assert not torch.is_grad_enabled(), (
                "core_attn_out buffer reuse is only supported for inference"
            )
            assert q.dtype == o.dtype, "Incompatible dtype for inplace computation"
        return o.to(q.dtype), final_state


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    core_attn_out: torch.Tensor | None = None,
):
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) Gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            Betas of shape `[B, T, H]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, V, K]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, V, K]`. Default: `False`.
        cu_seqlens (torch.Tensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, V, K]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, V, K, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.int32)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, (
        "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    )
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H]."
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        use_qk_l2norm_in_kernel,
        core_attn_out,
    )
    return o, final_state
