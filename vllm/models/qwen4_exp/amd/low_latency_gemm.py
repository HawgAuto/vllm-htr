# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp low-latency GEMM hook for AMD ROCm (V5-G4 opt-in HTR_BF16_GEMV=1)."""

import os
import torch
from torch import nn

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

_BF16_GEMV_EXT = None
_BF16_GEMV_OP_REGISTERED = False
_HTR_BF16_GEMV_MAX_M = int(os.environ.get("HTR_BF16_GEMV_MAX_M", "16"))
_HTR_BF16_GEMV_N16_MAX_M = int(os.environ.get("HTR_BF16_GEMV_N16_MAX_M", "4"))
_HTR_BF16_GEMV_MAX_N = int(os.environ.get("HTR_BF16_GEMV_MAX_N", "128"))


def _bf16_gemv_ext():
    global _BF16_GEMV_EXT
    if _BF16_GEMV_EXT is None:
        import importlib.util
        cand = os.environ.get("HTR_BF16_GEMV_SO")
        if not cand:
            for d in (os.environ.get("HTR_NATIVE_DIR", "/opt/vllm-htr/native-bin"),
                      "/opt/vllm-htr/native-proto"):
                p_so = os.path.join(d, "htr_bf16_gemv.so")
                if os.path.isfile(p_so):
                    cand = p_so
                    break
        if not cand or not os.path.isfile(cand):
            raise FileNotFoundError(
                f"HTR_BF16_GEMV=1 requires htr_bf16_gemv.so (checked {cand})"
            )
        spec = importlib.util.spec_from_file_location("htr_bf16_gemv", cand)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _BF16_GEMV_EXT = mod
    return _BF16_GEMV_EXT


def _ensure_bf16_gemv_op() -> None:
    global _BF16_GEMV_OP_REGISTERED
    if _BF16_GEMV_OP_REGISTERED:
        return

    def _run_bf16_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and weight.is_cuda
            and x.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and x.ndim >= 2
            and weight.ndim == 2
            and weight.shape[0] <= _HTR_BF16_GEMV_MAX_N
            and (weight.shape[1] & 7) == 0
        ):
            x2 = x.reshape(-1, x.shape[-1])
            m = x2.shape[0]
            lim = _HTR_BF16_GEMV_N16_MAX_M if weight.shape[0] >= 16 else _HTR_BF16_GEMV_MAX_M
            if 1 <= m <= lim:
                y = _bf16_gemv_ext().bf16_gemv_smalln(
                    x2.contiguous(), weight.contiguous()
                )
                return y.reshape(*x.shape[:-1], weight.shape[0])
        return torch.nn.functional.linear(x, weight)

    def _fake_bf16_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return x.new_empty((*x.shape[:-1], weight.shape[0]))

    direct_register_custom_op(
        op_name="htr_bf16_gemv_smalln",
        op_func=_run_bf16_gemv,
        fake_impl=_fake_bf16_gemv,
    )
    def _run_gate_sig(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        if (x2.is_cuda and weight.is_cuda and x2.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
                and weight.ndim == 2 and weight.shape[0] == 1 and (weight.shape[1] & 7) == 0
                and 1 <= x2.shape[0] <= _HTR_BF16_GEMV_MAX_M):
            return _bf16_gemv_ext().bf16_gemv_gate_sigmoid(x2.contiguous(), weight.contiguous()).reshape(*x.shape[:-1], 1)
        return torch.sigmoid(torch.nn.functional.linear(x, weight))

    def _fake_gate_sig(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return x.new_empty((*x.shape[:-1], 1))

    def _run_gate_scale(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
        x2 = x.reshape(-1, x.shape[-1])
        o2 = out.reshape(-1, out.shape[-1])
        if (x2.is_cuda and weight.is_cuda and o2.is_cuda
                and x2.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16 and o2.dtype == torch.bfloat16
                and weight.ndim == 2 and weight.shape[0] == 1 and (weight.shape[1] & 7) == 0
                and (o2.shape[1] & 1) == 0 and o2.is_contiguous()
                and 1 <= x2.shape[0] <= _HTR_BF16_GEMV_MAX_M):
            _bf16_gemv_ext().bf16_gemv_gate_scale_inplace(x2.contiguous(), weight.contiguous(), o2)
            return
        g = torch.sigmoid(torch.nn.functional.linear(x, weight))
        out.mul_(g)

    direct_register_custom_op(op_name="htr_bf16_gemv_gate_sigmoid", op_func=_run_gate_sig, fake_impl=_fake_gate_sig)
    direct_register_custom_op(op_name="htr_bf16_gemv_gate_scale", op_func=_run_gate_scale, mutates_args=["out"])
    _BF16_GEMV_OP_REGISTERED = True


class HTRBF16SmallNLinearMethod(UnquantizedLinearMethod):
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None:
            return torch.ops.vllm.htr_bf16_gemv_smalln(x, layer.weight)
        return super().apply(layer, x, bias)


def enable_qwen4_exp_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    """Opt-in HTR_BF16_GEMV=1: replace small-N unquantized BF16 linear layers
    (36x GDN in_proj_ba [96,2560], 48x shared_expert_gate [1,2560]) with our
    clean-slate gfx1151 Wave32 nontemporal BF16 GEMV for decode M=1..16."""
    if os.environ.get("HTR_BF16_GEMV") != "1" or dtype != torch.bfloat16:
        return
    _ensure_bf16_gemv_op()
    for child in module.modules():
        if not (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        ):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        n, k = int(weight.shape[0]), int(weight.shape[1])
        if 1 <= n <= _HTR_BF16_GEMV_MAX_N and (k & 7) == 0:
            child.quant_method = HTRBF16SmallNLinearMethod()
