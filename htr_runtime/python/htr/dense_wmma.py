"""Selected dense WMMA bridge; K5 served dense shapes, BF16 and 16..64 rows only.
The caller is htr::linear; all unsupported shapes use its unchanged incumbent.
Native code is built ahead of time. No compilation or device data reads here.
"""
import importlib.util
import json
import os
from pathlib import Path
import torch

SHAPES = {(640,160), (384,160), (160,40), (40,160), (32,160), (768,160), (1024,160), (832,160), (80,160)}
_EXT = None
_SEEN = set()


def enabled():
    value = os.environ.get('HTR_DENSE_WMMA', '1')
    if value not in ('0', '1'):
        raise ValueError('HTR_DENSE_WMMA must be 0 or 1')
    return value == '1'


def eligible(M, mt, nt, K, dtype):
    return (16 <= M <= 64 and dtype == torch.bfloat16 and K == 5 and (mt,nt) in SHAPES
            and ((mt,nt) != (1024,160) or M == 16))


def nsplit(mt, nt, M):
    if (mt,nt) in ((1024,160), (832,160)):
        return 8
    if (mt,nt) == (640,160):
        return 9
    if (mt,nt) == (768,160):
        return 8
    if (mt,nt) == (160,384):
        return 16
    if (mt,nt) == (160,40):
        return 5 if M <= 16 else 2
    if (mt,nt) in SHAPES:
        return 10
    raise ValueError('unqualified WMMA shape')


def ext(device):
    global _EXT
    if _EXT is None:
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(':')[0]
        if not torch.version.hip or arch != 'gfx1151':
            raise RuntimeError('Dense WMMA is qualified only for gfx1151 HIP')
        path = Path(__file__).resolve().parents[2] / 'native-bin/htr_dense_wmma.so'
        spec = importlib.util.spec_from_file_location('htr_wm', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _EXT = module
    return _EXT


def linear(E, x, qw, scales, su, sv, bias, cb, cnt):
    M, n = x.shape
    mt, nt, words = qw.shape
    if not eligible(M, mt, nt, words//8, x.dtype) or words != 40:
        raise ValueError('Dense WMMA unsupported call')
    # Metadata-only checks: safe in capture, no .item()/synchronization.
    if x.device.type != 'cuda' or n != nt*16:
        raise ValueError('Dense WMMA activation device/width')
    for value, dtype, size in [(x, torch.bfloat16, M*n), (qw, torch.int32, mt*nt*40),
                              (scales, torch.float16, mt*16), (su, torch.uint8, n//8),
                              (sv, torch.uint8, mt*2), (cnt, torch.int32, cnt.numel())]:
        if value.device != x.device or value.dtype != dtype or not value.is_contiguous() or value.numel() != size:
            raise ValueError('Dense WMMA tensor contract')
    if cnt.numel() < mt//8:
        raise ValueError('Dense WMMA counter capacity')
    if bias is not None and (bias.device != x.device or bias.dtype != x.dtype or not bias.is_contiguous() or bias.shape != (mt*16,)):
        raise ValueError('Dense WMMA bias contract')
    native = ext(x.device)
    key = (M,mt,nt)
    if key not in _SEEN:
        _SEEN.add(key)
        print('HTR_DENSE_WMMA_ROUTE '+json.dumps({'M':M,'m':mt*16,'n':nt*16,'K':5,'nsplit':nsplit(mt,nt,M)}), flush=True)
    xt, xl, xs, tok = native.prologue_c11(x, su)
    return native.gemv_wmma(qw, xt, xl, xs, tok, scales, cb, sv, bias, cnt, nsplit(mt,nt,M))
