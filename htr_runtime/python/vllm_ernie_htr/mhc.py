# SPDX-License-Identifier: Apache-2.0
"""ROCm-native ERNIE-5.5 mHC (Manifold-Constrained Hyper-Connections) ops.

Uses exact ERNIE-5.5 RMS normalization: ``1 / (sqrt(mean(x^2)) + rms_eps)``
(note: ``+ rms_eps`` is OUTSIDE the square root, matching baidu/vLLM-ERNIE
``mhc_kernels.py`` and our verified ``ernie55_ref.py``).

Automatically dispatches to the native gfx1151 Wave32 HIP extension
``mhc_ernie.so`` when available (toggle with ``ERNIE_NATIVE_MHC=0``).
"""

from __future__ import annotations

import os
import sys
import torch
import torch.nn.functional as F

_USE_NATIVE = os.environ.get("ERNIE_NATIVE_MHC", "1") == "1"
_USE_TILED_PREFILL = os.environ.get("ERNIE_MHC_TILED_PREFILL", "1") == "1"
_EXT = None
_EXT_TRIED = False
_PART_BUF: dict[str, torch.Tensor] = {}


def _ext():
    global _EXT, _EXT_TRIED
    if not _USE_NATIVE:
        return None
    if not _EXT_TRIED:
        _EXT_TRIED = True
        so_dir = os.environ.get("ERNIE_HTR_MOE_SO_DIR")
        if so_dir and so_dir not in sys.path:
            sys.path.insert(0, so_dir)
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        cands = [
            pkg_dir,
            os.path.join(os.path.dirname(pkg_dir), "build"),
            os.path.join(os.path.dirname(os.path.dirname(pkg_dir)), "build"),
        ]
        native_dir = os.environ.get("HTR_NATIVE_DIR")
        if native_dir:
            cands.append(native_dir)
        for cand in cands:
            if os.path.isfile(os.path.join(cand, "mhc_ernie.so")) and cand not in sys.path:
                sys.path.insert(0, cand)
        try:
            import mhc_ernie
            _EXT = mhc_ernie
            print(f"ERNIE_NATIVE_MHC_ACTIVE {mhc_ernie.__file__}", flush=True)
        except Exception as exc:
            print(f"ERNIE_NATIVE_MHC_FAILED {exc!r}", flush=True)
            _EXT = None
    return _EXT


def _part_buf(dev: torch.device, T: int) -> torch.Tensor:
    key = str(dev)
    buf = _PART_BUF.get(key)
    if buf is None or buf.shape[0] < T:
        buf = torch.empty((max(16, T), 32, 25), dtype=torch.float32, device=dev)
        _PART_BUF[key] = buf
    return buf[:T]


def _can_use_native(res: torch.Tensor, norm_weight: torch.Tensor | None) -> bool:
    return (
        _ext() is not None
        and res.is_cuda
        and res.dtype == torch.bfloat16
        and res.shape[-2] == 4
        and res.shape[-1] == 4096
        and (norm_weight is None or norm_weight.dtype == torch.bfloat16)
    )


def _ernie_rms_inv(x_f32: torch.Tensor, rms_eps: float) -> torch.Tensor:
    # Exact ERNIE-5.5 formula: 1 / (sqrt(mean(x^2)) + eps)
    return 1.0 / (x_f32.square().mean(dim=-1, keepdim=True).sqrt() + rms_eps)


def _sinkhorn_knopp(
    logits: torch.Tensor, hc_sinkhorn_eps: float, sinkhorn_repeat: int
) -> torch.Tensor:
    comb = torch.softmax(logits, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    return comb


def _apply_norm(
    x: torch.Tensor, norm_weight: torch.Tensor | None, norm_eps: float
) -> torch.Tensor:
    if norm_weight is None:
        return x
    xf = x.float()
    r = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + norm_eps)
    return (xf * r * norm_weight.float()).to(x.dtype)


def mhc_pre_broadcast(
    x: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    fn_broadcast: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """First layer: expand [T, H] -> [T, hc_mult, H] and compute mHC pre."""
    hidden_size = x.shape[-1]
    hc_mult = fn.shape[1] // hidden_size
    residual = x.unsqueeze(-2).expand(*x.shape[:-1], hc_mult, hidden_size).contiguous()
    post_mix, comb_mix, layer_input = mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    return residual, post_mix, comb_mix, layer_input


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ERNIE-5.5 mHC pre step (FP32 mix math, BF16 residual & layer_input)."""
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    res_flat = residual.reshape(-1, hc_mult, hidden_size).contiguous()
    T = res_flat.shape[0]

    if T == 0:
        return (
            torch.empty(*outer_shape, hc_mult, 1, dtype=torch.float32, device=residual.device),
            torch.empty(*outer_shape, hc_mult, hc_mult, dtype=torch.float32, device=residual.device),
            torch.empty(*outer_shape, hidden_size, dtype=residual.dtype, device=residual.device),
        )

    if _can_use_native(res_flat, norm_weight):
        ext = _ext()
        post_out = torch.empty((T, hc_mult, 1), dtype=torch.float32, device=residual.device)
        comb_out = torch.empty((T, hc_mult, hc_mult), dtype=torch.float32, device=residual.device)
        layer_in = torch.empty((T, hidden_size), dtype=residual.dtype, device=residual.device)
        nw = norm_weight.contiguous() if norm_weight is not None else None
        if T <= 16 or _USE_TILED_PREFILL:
            part = _part_buf(residual.device, T)
            ext.mhc_fused_post_pre_decode(
                res_flat, res_flat, post_out, comb_out,
                fn, hc_scale, hc_base, nw,
                res_flat, post_out, comb_out, layer_in, part,
                rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                sinkhorn_repeat, norm_eps, False,
            )
        else:
            xf = torch.empty((T, hc_mult * hidden_size), dtype=torch.float32, device=residual.device)
            ss = torch.empty((T,), dtype=torch.float32, device=residual.device)
            ext.mhc_post_sq(None, res_flat, None, None, None, xf, ss, False)
            dots = F.linear(xf, fn)
            ext.mhc_sinkhorn_norm_from_dots(
                dots, ss, res_flat, hc_scale, hc_base, nw,
                post_out, comb_out, layer_in,
                rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                sinkhorn_repeat, norm_eps,
            )
        return (
            post_out.view(*outer_shape, hc_mult, 1),
            comb_out.view(*outer_shape, hc_mult, hc_mult),
            layer_in.view(*outer_shape, hidden_size),
        )

    x_f32 = res_flat.reshape(T, hc_mult * hidden_size).float()
    mixes = F.linear(x_f32, fn) * _ernie_rms_inv(x_f32, rms_eps)

    pre_mix = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + hc_pre_eps
    post_mix = (
        torch.sigmoid(mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
        * hc_post_mult_value
    )
    comb_logits = (
        mixes[:, 2 * hc_mult :].view(T, hc_mult, hc_mult) * hc_scale[2]
        + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    )
    comb_mix = _sinkhorn_knopp(comb_logits, hc_sinkhorn_eps, sinkhorn_repeat)

    layer_input = torch.sum(pre_mix.unsqueeze(-1) * res_flat.float(), dim=1).to(residual.dtype)
    layer_input = _apply_norm(layer_input, norm_weight, norm_eps)

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Combine previous sublayer output x [..., H] into residual [..., hc_mult, H]."""
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    res_flat = residual.reshape(-1, hc_mult, hidden_size).contiguous()
    T = res_flat.shape[0]
    if T == 0:
        return torch.empty_like(residual)
    if _can_use_native(res_flat, None) and x.dtype == torch.bfloat16:
        out = torch.empty_like(res_flat)
        _ext().mhc_post_sq(
            x.reshape(T, hidden_size).contiguous(),
            res_flat,
            post_layer_mix.reshape(T, hc_mult).contiguous(),
            comb_res_mix.reshape(T, hc_mult, hc_mult).contiguous(),
            out,
            None,
            None,
            True,
        )
        return out.view(*outer_shape, hc_mult, hidden_size)
    mixed = torch.einsum("...ij,...ih->...jh", comb_res_mix.float(), residual.float())
    post_term = post_layer_mix.float() * x.unsqueeze(-2).float()
    return (mixed + post_term).to(residual.dtype)


def mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    res_flat = residual.reshape(-1, hc_mult, hidden_size).contiguous()
    T = res_flat.shape[0]

    if T == 0:
        return (
            torch.empty_like(residual),
            torch.empty(*outer_shape, hc_mult, 1, dtype=torch.float32, device=residual.device),
            torch.empty(*outer_shape, hc_mult, hc_mult, dtype=torch.float32, device=residual.device),
            torch.empty(*outer_shape, hidden_size, dtype=residual.dtype, device=residual.device),
        )

    if _can_use_native(res_flat, norm_weight) and x.dtype == torch.bfloat16:
        ext = _ext()
        x_flat = x.reshape(T, hidden_size).contiguous()
        pm_in = post_layer_mix.reshape(T, hc_mult).contiguous()
        cm_in = comb_res_mix.reshape(T, hc_mult, hc_mult).contiguous()
        res_out = torch.empty_like(res_flat)
        post_out = torch.empty((T, hc_mult, 1), dtype=torch.float32, device=residual.device)
        comb_out = torch.empty((T, hc_mult, hc_mult), dtype=torch.float32, device=residual.device)
        layer_in = torch.empty((T, hidden_size), dtype=residual.dtype, device=residual.device)
        nw = norm_weight.contiguous() if norm_weight is not None else None
        if T <= 16 or _USE_TILED_PREFILL:
            part = _part_buf(residual.device, T)
            ext.mhc_fused_post_pre_decode(
                x_flat, res_flat, pm_in, cm_in,
                fn, hc_scale, hc_base, nw,
                res_out, post_out, comb_out, layer_in, part,
                rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                sinkhorn_repeat, norm_eps, True,
            )
        else:
            xf = torch.empty((T, hc_mult * hidden_size), dtype=torch.float32, device=residual.device)
            ss = torch.empty((T,), dtype=torch.float32, device=residual.device)
            ext.mhc_post_sq(x_flat, res_flat, pm_in, cm_in, res_out, xf, ss, True)
            dots = F.linear(xf, fn)
            ext.mhc_sinkhorn_norm_from_dots(
                dots, ss, res_out, hc_scale, hc_base, nw,
                post_out, comb_out, layer_in,
                rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                sinkhorn_repeat, norm_eps,
            )
        return (
            res_out.view(*outer_shape, hc_mult, hidden_size),
            post_out.view(*outer_shape, hc_mult, 1),
            comb_out.view(*outer_shape, hc_mult, hc_mult),
            layer_in.view(*outer_shape, hidden_size),
        )

    residual_cur = mhc_post(x, residual, post_layer_mix, comb_res_mix)
    post_mix_cur, comb_mix_cur, layer_input_cur = mhc_pre(
        residual_cur,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur


def hc_head_contract(
    residual: torch.Tensor,
    hc_head_fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Final mHC contraction [T, hc_mult, H] -> [T, H] before final RMSNorm."""
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    res_flat = residual.reshape(-1, hc_mult, hidden_size).contiguous()
    T = res_flat.shape[0]
    if T == 0:
        return torch.empty(*outer_shape, hidden_size, dtype=residual.dtype, device=residual.device)

    if _can_use_native(res_flat, None):
        out = torch.empty((T, hidden_size), dtype=residual.dtype, device=residual.device)
        _ext().hc_head_contract(
            res_flat, hc_head_fn, hc_head_scale, hc_head_base, out, rms_eps, hc_eps
        )
        return out.view(*outer_shape, hidden_size)

    x_f32 = res_flat.reshape(T, hc_mult * hidden_size).float()
    logits = F.linear(x_f32, hc_head_fn) * torch.rsqrt(
        x_f32.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    w = torch.sigmoid(logits * hc_head_scale[0] + hc_head_base) + hc_eps
    out = torch.sum(w.unsqueeze(-1) * res_flat.float(), dim=1).to(residual.dtype)
    return out.view(*outer_shape, hidden_size)


