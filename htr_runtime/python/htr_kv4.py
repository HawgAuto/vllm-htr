"""Rotated scalar INT4 main-KV wire, views, and CPU-only reference.

Matches the installed vLLM int4_per_token_head packing/scale/RHT policy.
Reference dequantization deliberately reconstructs BF16 operands for HTR's
native sparse attention, not the donor dense reader's FP32 split-dot arithmetic.
"""
import torch


def content_bytes(head_size: int) -> int:
    if head_size < 16 or head_size & (head_size - 1):
        raise ValueError('head size must be a power of two >= 16')
    return head_size + 8


def cache_views(cache: torch.Tensor, head_size: int):
    width = content_bytes(head_size)
    if cache.dtype != torch.uint8:
        raise ValueError('INT4 main KV requires uint8 packed storage')
    if cache.ndim != 4 or cache.shape[-1] != width or not all(cache.shape):
        raise ValueError('invalid INT4 main KV shape')
    if cache.stride(-1) != 1:
        raise ValueError('packed content must be contiguous')
    extent = width
    for stride, size in sorted(zip(cache.stride()[:-1], cache.shape[:-1])):
        if size > 1:
            if stride < extent:
                raise ValueError('overlapping INT4 cache records')
            extent += (size - 1) * stride
    if cache.storage_offset() % 4 or any(s % 4 for s in cache.stride()[:-1]):
        raise ValueError('inline scales require four-byte alignment')
    half = head_size // 2
    k_half, v_half = cache.transpose(1, 2).split(half + 4, dim=-1)
    return (k_half[..., :half], v_half[..., :half],
            k_half[..., half:].view(torch.float32).squeeze(-1),
            v_half[..., half:].view(torch.float32).squeeze(-1))


def _cpu(*tensors):
    if any(t.device.type != 'cpu' for t in tensors):
        raise ValueError('CPU references are not serving fallbacks')


def single_rht_cpu(x: torch.Tensor, inverse=False):
    _cpu(x)
    d = x.shape[-1]
    content_bytes(d)
    if not x.is_floating_point():
        raise ValueError('rotation requires floating input')
    gen = torch.Generator(device='cpu').manual_seed(0x9E3779B9)
    signs = (2 * torch.bernoulli(torch.full((d,), .5), generator=gen) - 1)
    signs = signs.to(x.dtype)
    if not inverse:
        x = x * signs
    h = 1
    while h < d:
        z = x.reshape(*x.shape[:-1], d // (2 * h), 2, h)
        a, b = z[..., 0, :], z[..., 1, :]
        x = torch.stack((a + b, a - b), dim=-2).reshape(x.shape)
        h *= 2
    return x * signs if inverse else x


def pack_rotated_cpu(x: torch.Tensor):
    _cpu(x)
    if x.ndim < 1 or not x.is_floating_point():
        raise ValueError('expected floating vectors')
    content_bytes(x.shape[-1])
    if not torch.isfinite(x).all():
        raise ValueError('quantization requires finite input')
    f = x.float()
    lo, hi = f.amin(-1), f.amax(-1)
    scale = ((hi - lo) / 15.0).clamp_min(1e-6)
    z = -lo / scale
    zp = torch.where(z >= 0, z + .5, z - .5).to(torch.int32).clamp(0, 15)
    # Emulate the donor GPU's single-rounded FP32 FMA, not two torch ops.
    values = (f.double() * scale.reciprocal().double().unsqueeze(-1)
              + zp.double().unsqueeze(-1)).float()
    codes = torch.where(values >= 0, values + .5, values - .5)
    codes = codes.to(torch.int32).clamp(0, 15).to(torch.uint8)
    packed = codes[..., ::2] | (codes[..., 1::2] << 4)
    metadata = ((scale.contiguous().view(torch.int32) & -16) | zp)
    return packed, metadata.view(torch.float32)


def quantize_cpu(x: torch.Tensor):
    _cpu(x)
    if x.dtype != torch.bfloat16:
        raise ValueError('main-KV source must be BF16')
    rotated = single_rht_cpu(x.float()).to(x.dtype)
    return pack_rotated_cpu(rotated)


def dequantize_cpu(packed: torch.Tensor, scales: torch.Tensor):
    _cpu(packed, scales)
    if (packed.dtype != torch.uint8 or scales.dtype != torch.float32
            or scales.shape != packed.shape[:-1]):
        raise ValueError('invalid packed INT4 vectors or metadata')
    code = torch.stack((packed & 15, packed >> 4), dim=-1)
    code = code.flatten(-2).float()
    bits = scales.contiguous().view(torch.int32)
    zp = (bits & 15).float()
    scale = (bits & -16).view(torch.float32)
    return ((code - zp.unsqueeze(-1)) * scale.unsqueeze(-1)).bfloat16()


def write_cpu(cache, key, value, slots):
    _cpu(cache, key, value, slots)
    if key.ndim != 3 or key.shape != value.shape:
        raise ValueError('expected equal three-dimensional K/V')
    k, v, ks, vs = cache_views(cache, key.shape[-1])
    if (key.shape[1] != k.shape[2] or slots.shape != (key.shape[0],)
            or slots.dtype not in (torch.int32, torch.int64)):
        raise ValueError('invalid cache/slot geometry')
    active = slots >= 0
    live = slots[active].long()
    if ((live >= k.shape[0] * k.shape[1]).any()
            or live.unique().numel() != live.numel()):
        raise ValueError('out-of-range or duplicate live slot')
    kq, sk = quantize_cpu(key[active])
    vq, sv = quantize_cpu(value[active])
    pages, tokens = live // k.shape[1], live % k.shape[1]
    k[pages, tokens], v[pages, tokens] = kq, vq
    ks[pages, tokens], vs[pages, tokens] = sk, sv
