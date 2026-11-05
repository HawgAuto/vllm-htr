# SPDX-License-Identifier: Apache-2.0
"""RESEARCH (research/v033-octave-kv): Octave sparse decode at the HTR TP1 GQA group.

Qwen3.8-HTR at TP1 is Q24/KV2/D256, GQA group 12. Upstream octave_decode_sparse (WMMA tile
hpk * QG <= 32, window kernel hq / hkv <= 8) only takes groups <= 8 (upstream validates 6).
qsa._octave_decode_sparse runs the unmodified upstream op on head subsets of group <= 8 and
scatters the outputs; attention is independent per query head, so it must match the same
reference upstream's test_sparse_decode_matches_reference uses.
"""

import pytest
import torch

from tests.kernels.attention.test_rocm_octave import (
    BLOCK,
    CACHE_DTYPES,
    HEAD,
    _format,
    _rel,
    _window_tables,
    sq,
)
from tests.kernels.attention.test_rocm_octave import pytestmark  # noqa: F401
from vllm.models.qwen4_exp.amd import model as _model  # noqa: F401  (production import order; qsa<->model cycle)
from vllm.models.qwen4_exp.amd import qsa


def _scenario(cache_dtype, hkv, group, window):
    torch.manual_seed(4)
    dev, fmt = "cuda", _format(cache_dtype)
    signs = sq.sign_bits(HEAD).to(dev)
    lens, max_blocks, nblk = [1500, 2600, 777], 192, 1024
    cache = torch.zeros(nblk, hkv, BLOCK, fmt.slot_bytes, dtype=torch.uint8, device=dev)
    block_table = (
        torch.randperm(nblk, device=dev)[: len(lens) * max_blocks].view(len(lens), max_blocks).int()
    )
    wtab, wtags, wlocks = _window_tables(4096, hkv, dev)
    stab, stags, slocks = _window_tables(256, hkv, dev)
    ks, vs = [], []
    for r, n in enumerate(lens):
        k = torch.randn(n, hkv, HEAD, device=dev).half()
        v = torch.randn_like(k)
        pos = torch.arange(n, device=dev)
        slots = block_table[r, pos // BLOCK].long() * BLOCK + pos % BLOCK
        torch.ops._C.octave_cache_store(k, v, cache, slots, signs, signs, fmt.kernel_code)
        torch.ops._C.octave_window_store(k, v, slots, pos.int(), wtab, wtags, wlocks, stab, stags, slocks)
        ks.append(k)
        vs.append(v)
    if window == "evicted":
        wtags[::2] = -7
        stags[::2] = -7
    q_to_req = torch.tensor([0, 1, 1, 2, 0, 1, 2], dtype=torch.int32, device=dev)
    num_q, topk = q_to_req.numel(), 512
    qpos = torch.tensor([lens[r] - 1 - 37 * i for i, r in enumerate(q_to_req.tolist())],
                        dtype=torch.int32, device=dev)
    indices = torch.full((num_q, topk), -1, dtype=torch.int32, device=dev)
    for i in range(num_q):
        n = int(qpos[i]) + 1
        cand = torch.cat([torch.arange(max(0, n - 80), n, device=dev), torch.arange(4, device=dev),
                          torch.randperm(n, device=dev)[:300]]).unique()
        indices[i, : cand.numel()] = cand.int()
        indices[i] = indices[i][torch.randperm(topk, device=dev)]
    q = torch.randn(num_q, hkv * group, HEAD, device=dev).half()
    mid = torch.empty(num_q, hkv * group, 64, HEAD + 2, dtype=torch.float32, device=dev)
    window_args = () if window == "off" else (qpos, wtab, wtags, stab, stags)
    args = (cache, block_table, q_to_req, indices, mid, signs, signs, HEAD**-0.5, 64, fmt.kernel_code,
            *window_args)

    ref = torch.zeros(num_q, hkv * group, HEAD, device=dev)
    for i in range(num_q):
        r, p = int(q_to_req[i]), int(qpos[i])
        sel = indices[i][indices[i] >= 0].long()
        blocks = block_table[r, sel // BLOCK].long()
        k, v = sq.reference_dequantize(cache[blocks, :, sel % BLOCK], fmt)
        if window != "off":
            slot = blocks * BLOCK + sel % BLOCK
            exact = ((p - sel) < 64) & (wtags[slot % 4096].long() == slot)
            exact |= (sel < 4) & (stags[slot % 256].long() == slot)
            k[exact] = ks[r][sel[exact]].float()
            v[exact] = vs[r][sel[exact]].float()
        k = k.repeat_interleave(group, 1)
        v = v.repeat_interleave(group, 1)
        p_attn = torch.softmax(torch.einsum("hd,thd->ht", q[i].float(), k) * HEAD**-0.5, -1)
        ref[i] = torch.einsum("ht,thd->hd", p_attn, v)
    return q, args, ref


def test_upstream_op_refuses_the_htr_group():
    q, args, _ = _scenario("octave_k3v4", 2, 12, "on")
    with pytest.raises(RuntimeError, match="needs the WMMA tier"):
        torch.ops._C.octave_decode_sparse(torch.empty_like(q), q, *args)


@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES)
@pytest.mark.parametrize("window", ["off", "on", "evicted"])
def test_head_split_matches_reference_at_group_12(cache_dtype, window):
    q, args, ref = _scenario(cache_dtype, 2, 12, window)
    split = qsa._octave_head_split(24, 2, q.device)
    assert split is not None and [t.numel() for t in split] == [12, 12]
    out = torch.full_like(q, float("nan"))
    qsa._octave_decode_sparse(out, q, *args, head_split=split)
    assert not out.isnan().any()
    assert _rel(out, ref) < 2e-2


def test_upstream_group_passes_through_unsplit():
    assert qsa._octave_head_split(12, 2, "cpu") is None
    # bitwise only with the exact window off: upstream's window path is not run-to-run
    # reproducible on gfx1151 (probe: 4 identical calls, max |diff| 1.2e-4; window off: equal)
    q, args, ref = _scenario("octave_k3v4", 2, 6, "off")
    out = torch.empty_like(q)
    qsa._octave_decode_sparse(out, q, *args, head_split=None)
    direct = torch.empty_like(q)
    torch.ops._C.octave_decode_sparse(direct, q, *args)
    assert torch.equal(out, direct)
    assert _rel(out, ref) < 2e-2
    q, args, ref = _scenario("octave_k3v4", 2, 6, "on")
    out = torch.empty_like(q)
    qsa._octave_decode_sparse(out, q, *args, head_split=None)
    assert _rel(out, ref) < 2e-2
