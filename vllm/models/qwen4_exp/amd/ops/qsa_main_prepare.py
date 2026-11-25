# SPDX-License-Identifier: Apache-2.0
"""Bit-exact fused main-attention prepare for the AMD Qwen4Exp QSA layer (HTR s4b).

Replaces, for the main (non-indexer) attention of a QSA layer, the graph
pieces Inductor emits for ``Qwen3NextAttention._project_qkv_gate`` (Q/K
GemmaRMSNorm sum-of-squares reductions, NeoX RoPE, pass-through dims, gate
view) plus ``do_kv_cache_update``'s BF16 ``reshape_and_cache_flash`` scatter,
with one launch.

Unlike upstream #59732 this does NOT live inside the indexer launch and does
NOT round the normalized rotary inputs to BF16 before RoPE. Both choices make
it reproduce HTR v0.2.1's compiled graph bit-for-bit:

* Inductor (custom_ops=none) keeps ``x * rsqrt * (w + 1)`` in fp32 straight
  into the RoPE multiply-adds and rounds once at the store.
* Inductor reduces each 256-wide sum of squares as one ``[XBLOCK=2, 256]``
  tile with ``num_warps=2`` and no loop. This kernel uses the same tile shape,
  load pattern and warp count, so ``tl.sum`` lowers to the same reduction tree.

BF16 main KV cache only (HTR INT8/INT4 caches keep ``do_kv_cache_update``).
"""

import torch

from vllm.triton_utils import tl, triton

XBLOCK = 2
NUM_WARPS = 2


@triton.jit
def _qsa_main_prepare_kernel(
    qkv_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    eps,
    cos_sin_ptr,
    inv_freq_ptr,
    mscale,
    pos_ptr,
    pos_stride_axis,
    pos_stride_token,
    q_out_ptr,
    gate_out_ptr,
    cache_ptr,
    cache_stride_block,
    cache_stride_token,
    cache_stride_head,
    slots_ptr,
    num_tokens,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    CS_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    IS_2D_POSITIONS: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
    XBLOCK: tl.constexpr,
    TABLE_FREE: tl.constexpr,
):
    W: tl.constexpr = 2 * (HQ + HK) * D
    HALF: tl.constexpr = ROT // 2
    pid = tl.program_id(0)
    num_q_rows = num_tokens * HQ
    num_q_programs = num_q_rows // XBLOCK
    is_k = pid >= num_q_programs
    if is_k:
        xnumel = num_tokens * HK
        xindex = (pid - num_q_programs) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        token = (xindex // HK).to(tl.int64)
        head = xindex % HK
        base = token * W + 2 * HQ * D + head * D
        w_ptr = k_norm_weight_ptr
    else:
        xnumel = num_q_rows
        xindex = pid * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        token = (xindex // HQ).to(tl.int64)
        head = xindex % HQ
        base = token * W + head * (2 * D)
        w_ptr = q_norm_weight_ptr
    xmask = xindex < xnumel

    # Sum of squares: same tile, load and accumulate pattern as Inductor's
    # triton_red_fused_*rms_norm* kernels (R0_BLOCK == D, single iteration).
    r0 = tl.arange(0, D)[None, :]
    r0_mask = r0 < D
    acc = tl.full([XBLOCK, D], 0, tl.float32)
    xs = tl.load(
        qkv_ptr + (base + r0), r0_mask & xmask, eviction_policy="evict_first", other=0.0
    ).to(tl.float32)
    acc = tl.where(r0_mask & xmask, acc + xs * xs, acc)
    sumsq = tl.sum(acc, 1)[:, None]
    rs = tl.rsqrt(sumsq / D + eps)

    # Pass-through dims [ROT, D): x * rs * (w + 1), rounded once at the store.
    w_all = tl.load(w_ptr + r0).to(tl.float32) + 1.0
    y = xs * rs * w_all

    # NeoX RoPE over [0, ROT) in fp32 (no intermediate BF16 rounding).
    rot = tl.arange(0, HALF)[None, :]
    x1 = tl.load(qkv_ptr + (base + rot), xmask, other=0.0).to(tl.float32)
    x2 = tl.load(qkv_ptr + (base + HALF + rot), xmask, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + rot).to(tl.float32) + 1.0
    w2 = tl.load(w_ptr + HALF + rot).to(tl.float32) + 1.0
    n1 = x1 * rs * w1
    n2 = x2 * rs * w2
    pos = tl.load(pos_ptr + token * pos_stride_token, xmask, other=0).to(tl.int64)
    if IS_2D_POSITIONS:
        pos_h = tl.load(
            pos_ptr + pos_stride_axis + token * pos_stride_token, xmask, other=0
        ).to(tl.int64)
        pos_w = tl.load(
            pos_ptr + 2 * pos_stride_axis + token * pos_stride_token, xmask, other=0
        ).to(tl.int64)
        is_h = ((rot % 3) == 1) & (rot < 3 * MROPE_H)
        is_w = ((rot % 3) == 2) & (rot < 3 * MROPE_W)
        pos = tl.where(is_w, pos_w, tl.where(is_h, pos_h, pos))
    if TABLE_FREE:
        inv_f = tl.load(inv_freq_ptr + rot, xmask, other=0.0).to(tl.float32)
        ang = pos.to(tl.float32) * inv_f
        cos = (tl.cos(ang) * mscale).to(tl.bfloat16).to(tl.float32)
        sin = (tl.sin(ang) * mscale).to(tl.bfloat16).to(tl.float32)
    else:
        cos = tl.load(cos_sin_ptr + pos * CS_STRIDE + rot, xmask, other=0.0).to(tl.float32)
        sin = tl.load(cos_sin_ptr + pos * CS_STRIDE + HALF + rot, xmask, other=0.0).to(
            tl.float32
        )
    a1 = n1 * cos
    b1 = n2 * sin
    o1 = a1 - b1
    a2 = n2 * cos
    b2 = n1 * sin
    o2 = a2 + b2

    if is_k:
        slot = tl.load(slots_ptr + token, xmask, other=-1).to(tl.int64)
        valid = xmask & (slot >= 0)
        safe = tl.maximum(slot, 0)
        dst = (
            cache_ptr
            + (safe // PAGE) * cache_stride_block
            + (safe % PAGE) * cache_stride_token
            + head * cache_stride_head
        )
        tl.store(dst + r0, y.to(tl.bfloat16), valid & (r0 >= ROT))
        tl.store(dst + rot, o1.to(tl.bfloat16), valid)
        tl.store(dst + HALF + rot, o2.to(tl.bfloat16), valid)
        v_src = qkv_ptr + token * W + 2 * HQ * D + HK * D + head * D
        v = tl.load(v_src + r0, xmask, other=0.0)
        tl.store(dst + D + r0, v, valid)
    else:
        out = (token * HQ + head) * D
        tl.store(q_out_ptr + out + r0, y.to(tl.bfloat16), xmask & (r0 >= ROT))
        tl.store(q_out_ptr + out + rot, o1.to(tl.bfloat16), xmask)
        tl.store(q_out_ptr + out + HALF + rot, o2.to(tl.bfloat16), xmask)
        gate = tl.load(qkv_ptr + (base + D + r0), xmask, other=0.0)
        tl.store(gate_out_ptr + out + r0, gate, xmask)


def supports_main_prepare(
    rotary_emb, head_dim: int, num_heads: int, num_kv_heads: int
) -> bool:
    rotary_dim = int(getattr(rotary_emb, "rotary_dim", 0))
    mrope = getattr(rotary_emb, "mrope_section", None)
    return (
        bool(getattr(rotary_emb, "is_neox_style", False))
        and head_dim in (64, 128, 256, 512)
        and rotary_dim in (32, 64, 128)
        and rotary_dim <= head_dim
        and num_heads % XBLOCK == 0
        and num_kv_heads % XBLOCK == 0
        and (
            not mrope
            or (
                len(mrope) == 3
                and sum(mrope) == rotary_dim // 2
                and bool(getattr(rotary_emb, "mrope_interleaved", False))
            )
        )
    )



_QSA_MAIN_PREP_AOT = None
def _get_qsa_main_prep_aot():
    global _QSA_MAIN_PREP_AOT
    if _QSA_MAIN_PREP_AOT is None:
        import os as _os
        if _os.environ.get("HTR_QSA_MAIN_PREP_AOT", "1") == "1":
            for cand in (
                _os.environ.get("HTR_QSA_MAIN_PREP_HSACO", ""),
                _os.path.join(_os.environ.get("HTR_NATIVE_DIR", "/opt/vllm-htr/native-bin"), "qsa_main_prepare_aot.hsaco"),
                "/opt/vllm-htr/native-v026/qsa_main_prepare_aot.hsaco",
            ):
                if cand and _os.path.isfile(cand):
                    import direct_hip_module_v1 as _hm
                    _QSA_MAIN_PREP_AOT = (_hm, _hm.Module(cand))
                    break
        if _QSA_MAIN_PREP_AOT is None:
            _QSA_MAIN_PREP_AOT = False
    return _QSA_MAIN_PREP_AOT

def qsa_main_prepare(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    eps: float,
    kv_cache_t: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section,
    gate_out: torch.Tensor,
    inv_freq: torch.Tensor | None = None,
    mscale: float = 1.0,
) -> torch.Tensor:
    """Return main Q [T, HQ, D]; write gate into ``gate_out[:T]`` and K/V into the cache.

    ``kv_cache_t`` is ``attn.kv_cache.transpose(1, 2)``: [blocks, page, HK, 2*D].
    """
    num_tokens = qkv.shape[0]
    q_out = qkv.new_empty(num_tokens, num_q_heads, head_dim)
    if num_tokens == 0:
        return q_out
    assert qkv.dtype == torch.bfloat16 and qkv.is_contiguous()
    assert qkv.shape[1] == 2 * (num_q_heads + num_kv_heads) * head_dim
    assert kv_cache_t.dtype == torch.bfloat16
    assert kv_cache_t.shape[2] == num_kv_heads and kv_cache_t.shape[3] == 2 * head_dim
    assert kv_cache_t.stride(-1) == 1
    table_free = cos_sin_cache is None
    if table_free:
        assert inv_freq is not None and inv_freq.numel() == rotary_dim // 2 and inv_freq.is_contiguous()
        cs_arg = inv_freq
        inv_arg = inv_freq
        cs_stride = rotary_dim
    else:
        assert cos_sin_cache.dtype == torch.bfloat16 and cos_sin_cache.is_contiguous()
        assert cos_sin_cache.shape[-1] == rotary_dim
        cs_arg = cos_sin_cache
        inv_arg = cos_sin_cache
        cs_stride = cos_sin_cache.shape[-1]
    assert gate_out.is_contiguous() and gate_out.shape[1:] == (num_q_heads, head_dim)
    assert slot_mapping.shape[0] >= num_tokens
    is_2d = positions.ndim == 2
    if is_2d:
        assert positions.shape[0] == 3 and positions.shape[1] >= num_tokens
        pos_stride_axis, pos_stride_token = positions.stride()
        assert mrope_section
    else:
        pos_stride_axis, pos_stride_token = 0, positions.stride(0)
    section = mrope_section if (is_2d and mrope_section) else (0, 0, 0)
    aot = _get_qsa_main_prep_aot()
    if (
        aot
        and head_dim == 256
        and rotary_dim == 64
        and positions.dtype == torch.int64
        and slot_mapping.dtype == torch.int64
        and q_norm_weight.dtype == torch.bfloat16
        and k_norm_weight.dtype == torch.bfloat16
        and (not table_free or inv_freq.dtype == torch.float32)
    ):
        import ctypes as _ct
        _hm, _m = aot
        I32, I64, F32 = _ct.c_int32, _ct.c_int64, _ct.c_float
        total_rows = num_tokens * (num_q_heads + num_kv_heads)
        _m.launch(
            "qsa_main_prepare_d256_rot64",
            ((total_rows + 3) // 4, 1, 1),
            (128, 1, 1),
            [
                _hm.ptr(qkv), _hm.ptr(q_norm_weight), _hm.ptr(k_norm_weight), F32(float(eps)),
                _hm.ptr(cs_arg), _hm.ptr(inv_arg), F32(float(mscale)),
                _hm.ptr(positions), I64(pos_stride_axis), I64(pos_stride_token),
                _hm.ptr(q_out), _hm.ptr(gate_out), _hm.ptr(kv_cache_t),
                I64(kv_cache_t.stride(0)), I64(kv_cache_t.stride(1)), I64(kv_cache_t.stride(2)),
                _hm.ptr(slot_mapping),
                I32(num_tokens), I32(num_q_heads), I32(num_kv_heads),
                I32(cs_stride), I32(kv_cache_t.shape[1]),
                I32(int(is_2d)), I32(int(section[1])), I32(int(section[2])), I32(int(table_free)),
            ],
        )
        return q_out
    grid = (
        num_tokens * num_q_heads // XBLOCK + num_tokens * num_kv_heads // XBLOCK,
    )
    _qsa_main_prepare_kernel[grid](
        qkv,
        q_norm_weight,
        k_norm_weight,
        eps,
        cs_arg,
        inv_arg,
        float(mscale),
        positions,
        pos_stride_axis,
        pos_stride_token,
        q_out,
        gate_out,
        kv_cache_t,
        kv_cache_t.stride(0),
        kv_cache_t.stride(1),
        kv_cache_t.stride(2),
        slot_mapping,
        num_tokens,
        HQ=num_q_heads,
        HK=num_kv_heads,
        D=head_dim,
        ROT=rotary_dim,
        CS_STRIDE=cs_stride,
        PAGE=kv_cache_t.shape[1],
        IS_2D_POSITIONS=is_2d,
        MROPE_H=section[1],
        MROPE_W=section[2],
        XBLOCK=XBLOCK,
        TABLE_FREE=table_free,
        num_warps=NUM_WARPS,
    )
    return q_out


__all__ = ["qsa_main_prepare", "supports_main_prepare"]
