"""Main-KV INT8 wire helpers and explicitly CPU-only numerical references.

Layout matches vLLM's per-token/head cache: [K bytes, FP32 K scale,
V bytes, FP32 V scale]. No indexer/recurrent-state handling lives here.
GPU execution and serving admission are NOT qualified by this module.
"""
from __future__ import annotations
import torch


def content_bytes(head_size: int, cache_dtype: str) -> int:
    if head_size < 16 or head_size & (head_size - 1):
        raise ValueError('head size must be a power of two >= 16')
    if cache_dtype in ('auto', 'bfloat16'):
        return 4 * head_size
    if cache_dtype == 'int8_per_token_head':
        return 2 * (head_size + 4)
    raise ValueError('unsupported main KV cache dtype: ' + cache_dtype)


def cache_views(cache: torch.Tensor, head_size: int):
    """Return K,V [page,token,head,dim] and inline scales, without writes.

    `cache` is logically [page,head,token,content], physically HND or NHD,
    with optional larger page strides and a nonzero aligned storage offset.
    No view is cached: rebinding the allocator cannot retain stale pointers.
    """
    mode = {torch.bfloat16: 'bfloat16', torch.int8: 'int8_per_token_head'}.get(cache.dtype)
    if mode is None:
        raise ValueError('unsupported main KV storage dtype')
    width = content_bytes(head_size, mode) // cache.element_size()
    if cache.ndim != 4 or cache.shape[-1] != width or not all(cache.shape):
        raise ValueError('invalid main KV cache shape')
    if cache.stride(-1) != 1:
        raise ValueError('main KV content must be contiguous')
    # Reject overlapping records, including broadcast strides. Page padding is valid.
    extent = width
    for stride, size in sorted(zip(cache.stride()[:-1], cache.shape[:-1])):
        if size > 1:
            if stride < extent:
                raise ValueError('overlapping main KV records')
            extent += (size - 1) * stride
    if cache.dtype == torch.int8:
        if cache.storage_offset() % 4 or any(s % 4 for s in cache.stride()[:-1]):
            raise ValueError('FP32 inline scales require four-byte alignment')
    transposed = cache.transpose(1, 2)
    if cache.dtype == torch.bfloat16:
        k, v = transposed.split(head_size, dim=-1)
        return k, v, None, None
    k_half, v_half = transposed.split(head_size + 4, dim=-1)
    k = k_half[..., :head_size]
    v = v_half[..., :head_size]
    ks = k_half[..., head_size:].view(torch.float32).squeeze(-1)
    vs = v_half[..., head_size:].view(torch.float32).squeeze(-1)
    return k, v, ks, vs


def _cpu(*tensors):
    if any(t.device.type != 'cpu' for t in tensors):
        raise ValueError('reference functions are CPU-only, not serving fallbacks')


def quantize_cpu(x: torch.Tensor, mode: str = 'int8'):
    """Reference for existing vLLM writer; FP8 is an offline comparison only.

    INT8 uses round-half-away-from-zero (NOT torch.round ties-to-even).
    This reference does not establish GPU reciprocal/rounding bit parity.
    """
    _cpu(x)
    if x.ndim < 1 or x.shape[-1] == 0 or not x.is_floating_point():
        raise ValueError('expected nonempty floating vectors')
    if not torch.isfinite(x).all():
        raise ValueError('reference quantization requires finite input')
    f = x.float()
    if mode not in ('int8', 'fp8_e4m3fn'):
        raise ValueError('unsupported comparison mode')
    maximum = 127. if mode == 'int8' else 448.
    s = (f.abs().amax(dim=-1) / maximum).clamp_min(1e-6)
    q = f * s.reciprocal().unsqueeze(-1)
    if mode == 'int8':
        q = torch.where(q >= 0, q + .5, q - .5).clamp(-128, 127).to(torch.int8)
    else:
        q = q.clamp(-448,448).to(torch.float8_e4m3fn)
    return q, s


def dequantize_cpu(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Reconstruct BF16 operands BEFORE attention dot products."""
    _cpu(q, scales)
    if scales.shape != q.shape[:-1] or scales.dtype != torch.float32:
        raise ValueError('invalid per-token/head scales')
    return (q.float() * scales.unsqueeze(-1)).to(torch.bfloat16)


def write_cpu(cache: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
              slots: torch.Tensor) -> None:
    """Failure-atomic CPU page-writer oracle, never called by serving code."""
    _cpu(cache, key, value, slots)
    if cache.dtype != torch.int8 or key.ndim != 3 or key.shape != value.shape:
        raise ValueError('expected INT8 main cache and equal K/V tensors')
    k, v, ks, vs = cache_views(cache, key.shape[-1])
    assert ks is not None and vs is not None
    if key.shape[1] != k.shape[2] or slots.shape != (key.shape[0],) or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError('invalid write shapes or slot dtype')
    active = slots >= 0
    valid_slots = slots[active].long()
    if (valid_slots >= k.shape[0] * k.shape[1]).any() or valid_slots.unique().numel() != valid_slots.numel():
        raise ValueError('out-of-range or duplicate live slot')
    # Compute all data before mutating storage. Invalid inputs leave canaries intact.
    kq, sk = quantize_cpu(key[active]); vq, sv = quantize_cpu(value[active])
    pages, tokens = valid_slots // k.shape[1], valid_slots % k.shape[1]
    k[pages, tokens] = kq; v[pages, tokens] = vq
    ks[pages, tokens] = sk; vs[pages, tokens] = sv


def sparse_attention_cpu(q, k, v, indices):
    """Float64 sensitivity oracle, NOT the native WMMA/reduction oracle.

    Indices address compact token records; preserve duplicate multiplicity.
    Invalid indices are masked. Return FP64 output and per-row probabilities.
    """
    _cpu(q, k, v, indices)
    if (q.ndim != 3 or k.ndim != 3 or k.shape != v.shape or
            k.shape[1] < 1 or q.shape[1] % k.shape[1] or q.shape[2] != k.shape[2]
            or indices.ndim != 2 or indices.shape[0] != q.shape[0]):
        raise ValueError('invalid attention geometry')
    output = torch.zeros_like(q, dtype=torch.float64)
    probabilities = []
    groups = q.shape[1] // k.shape[1]
    for row in range(q.shape[0]):
        ids = indices[row]
        ids = ids[(ids >= 0) & (ids < k.shape[0])].long()
        if not ids.numel():
            probabilities.append(torch.empty(k.shape[1], groups, 0, dtype=torch.float64))
            continue
        qr = q[row].double().reshape(k.shape[1], groups, q.shape[2])
        kr, vr = k[ids].double(), v[ids].double()
        scores = torch.einsum('hgd,shd->hgs', qr, kr) * (q.shape[2] ** -0.5)
        p = torch.softmax(scores, dim=-1)
        output[row] = torch.einsum('hgs,shd->hgd', p, vr).reshape_as(q[row])
        probabilities.append(p)
    return output, probabilities


def compact_capture_cpu(capture, query_rows):
    """Map retained E157 single-request physical-page captures faithfully.

    Only selected live tokens are gathered; unwritten page padding is never
    quantized or inspected. Saved `out` is deliberately not consumed.
    """
    q, indices = capture['q'], capture['idx']
    _cpu(q, indices, capture['k_pages'], capture['v_pages'])
    rows = torch.as_tensor(query_rows, dtype=torch.int64)
    if rows.ndim != 1 or (rows < 0).any() or (rows >= q.shape[0]).any():
        raise ValueError('invalid captured query rows')
    if (capture['t2r'][rows] != 0).any():
        raise ValueError('only single-request captures are supported')
    selected = indices[rows].long()
    valid = selected >= 0
    logical = selected[valid].unique(sorted=True)
    if (logical >= int(capture['context_len'])).any():
        raise ValueError('captured logical token beyond context')
    page = int(capture['page'])
    lpages = logical // page
    if (lpages >= capture['bt0'].numel()).any():
        raise ValueError('captured logical page beyond table')
    physical = capture['bt0'][lpages].long()
    captured = capture['pages'].long()
    if captured.unique().numel() != captured.numel():
        raise ValueError('duplicate captured page')
    match = physical[:, None] == captured[None, :]
    if not (match.sum(dim=1) == 1).all():
        raise ValueError('missing captured page')
    compact_pages = match.to(torch.int64).argmax(dim=1)
    offsets = logical % page
    keys = capture['k_pages'][compact_pages, offsets]
    values = capture['v_pages'][compact_pages, offsets]
    remapped = torch.full_like(selected, -1)
    remapped[valid] = torch.searchsorted(logical, selected[valid])
    return q[rows], keys, values, remapped
