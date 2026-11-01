"""HTR QSA sparse-attention dispatch for Qwen3.8-Flash-Next-HTR on gfx1151.

Called by vllm/models/qwen4_exp/amd/qsa.py for every QSA attention call. Two kernels:

  split-K   the release Triton kernel vllm.models.qwen4_exp.amd.ops.qsa.qsa_sparse_paged_attention (reference
            numerics; every call that is not admitted below).
  TAU v2    native HIP tile-aligned union kernel qsa_tau_v2 (native-source/qsa_tau/qsa_tau_v2.hip, built by
            build_native.py). Bitwise equal to split-K by construction (operator gates: 47/47 single-request,
            26 page-1600 cases, 35/35 multi-request mixed steps) and faster on long prefills. Admitted only for
            gfx1151 BF16 prefills with more than 256 query rows, outside CUDA-graph capture, on the page-1568 KV
            geometry (MTP off) or the page-1600 KV geometry (MTP on) below, at any KV block count / page-table width.

HTR_QSA_TAU=1 (default) enables TAU v2; 0 sends every call to split-K. The first HTR_QSA_TAU_CHECKS admitted calls
(default 8) and the first HTR_QSA_TAU_MR_CHECKS admitted multi-request calls (default 8) also run split-K into a
NaN-filled buffer and raise unless the two outputs are bitwise identical.

Logs: HTR_QSA_TAU_ACTIVE, HTR_QSA_TAU_FIRST, HTR_QSA_TAU_SELFCHECK, HTR_QSA_TAU_STATS, HTR_QSA_TAU_FINAL.

gate_apply_() is the QSA output gate (out *= sigmoid(gate), in place); its body is the production Inductor kernel
triton_poi_fused_mul_sigmoid_slice_view_0, unchanged.
"""
import atexit
import ctypes
import hashlib
import json
import os
import sys
import time

import torch
import triton
import triton.language as tl

# Fail closed if the files this kernel contract was qualified against change.
QSA_OPS_SHA256 = '6ab20eb2778759530ab9fbb7c4a9b435a169d68da85b2b02a75b158301407990'
HIP_MODULE_SHA256 = '6c2b7fd9a715a144fc3e2cde58525418f229073591905b060a68815c8100780d'
HSACO_SHA256 = 'e5a845c8407cd2ed78560e7b9f83ba5c202a11749914618ded862aab056a355c'
HSACO = os.path.join(os.environ.get('HTR_NATIVE_DIR', '/opt/vllm-htr/native-bin'), 'qsa_tau_v2.hsaco')
FN = 'qsa_tau_v2'
SHARE = 2          # compatibility groups (v2)

ENABLED = os.environ.get('HTR_QSA_TAU', '1') == '1'
CHECKS = int(os.environ.get('HTR_QSA_TAU_CHECKS', '8'))
MR_CHECKS = int(os.environ.get('HTR_QSA_TAU_MR_CHECKS', '8'))

_STATE = {'tau_calls': 0, 'tau_rows': 0, 'release_calls': 0, 'checks': 0, 'rows_min': None, 'rows_max': None,
          'mr_calls': 0, 'mr_checks': 0, 'page1600_calls': 0}
_ARCH = {}
_MOD = {}
_INIT = []


def _log(tag, **kw):
    kw['pid'] = os.getpid()
    print(tag + ' ' + json.dumps(kw, sort_keys=True), file=sys.stderr, flush=True)


def _sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _ops():
    import vllm.models.qwen4_exp.amd.ops.qsa as ops
    return ops


def _init():
    if _INIT:
        return
    import direct_hip_module_v1 as hm
    ops = _ops()
    if _sha(ops.__file__) != QSA_OPS_SHA256:
        raise RuntimeError('HTR_QSA_TAU: unexpected ' + ops.__file__)
    if _sha(hm.__file__) != HIP_MODULE_SHA256:
        raise RuntimeError('HTR_QSA_TAU: unexpected ' + hm.__file__)
    if _sha(HSACO) != HSACO_SHA256:
        raise RuntimeError('HTR_QSA_TAU: unexpected ' + HSACO)
    atexit.register(lambda: _log('HTR_QSA_TAU_FINAL', t_ns=time.time_ns(), **_STATE))
    _log('HTR_QSA_TAU_ACTIVE', checks=CHECKS, mr_checks=MR_CHECKS, qsa_sha256=QSA_OPS_SHA256,
         hsaco_sha256=HSACO_SHA256, share=SHARE, version=2)
    _INIT.append(True)


def _module():
    if 'm' not in _MOD:
        import direct_hip_module_v1 as hm
        m = hm.Module(HSACO)                       # hm.lib() runs torch.cuda.init() first
        libpath = os.path.realpath(hm.lib()._name)
        maps = set()
        with open('/proc/self/maps') as f:
            for line in f:
                parts = line.split()
                p = parts[-1] if len(parts) >= 6 else ''
                if 'libamdhip64' in p:
                    maps.add(os.path.realpath(p))
        if maps != {libpath}:
            raise RuntimeError('HTR_QSA_TAU: HIP runtime mismatch ' + json.dumps({'ctypes': libpath, 'maps': sorted(maps)}))
        _MOD.update(m=m, hm=hm, attrs=m.attrs(FN), lib=libpath)
    return _MOD['m']


def page1600_admitted(q, k, v, idx, bt, t2r, out):
    """Page-1600 KV geometry (the MTP attention block size: vLLM raises the block to 1600 so the attention page covers
    the mamba page). Row bounds as qualified (E159 G1F single request 257..8192 rows, G1G multi-request 257..12288
    rows). The KV block count and page-table width follow the served context / KV budget: the kernel takes both as
    arguments (as on the page-1568 path) and every call is bounded by the storage check below. Additional to the
    general TAU guards in admitted(); the bitwise split-K selfchecks apply to page-1600 calls like any other."""
    n = q.shape[0] if q.ndim == 3 else 0
    nreq = bt.shape[0] if bt.ndim == 2 else 0
    hi = 8192 if nreq == 1 else 12288
    if not (q.is_cuda and nreq >= 1 and 257 <= n <= hi and q.shape == (n, 24, 256) and q.stride() == (6144, 256, 1)):
        return False
    if not (k.ndim == 4 and k.shape == v.shape and k.shape[0] >= 1 and k.shape[1:] == (1600, 2, 256)
            and k.stride() == v.stride() and k.stride()[1:] == (1024, 512, 1) and k.stride(0) >= 1600 * 1024):
        return False
    w = bt.shape[1] if bt.ndim == 2 else 0
    if not (idx.shape == (n, 2051) and idx.stride() == (2051, 1) and w >= 1 and bt.stride() == (w, 1)
            and t2r.shape == (n,) and t2r.stride() == (1,)):
        return False
    if not (q.dtype == k.dtype == v.dtype == torch.bfloat16 and idx.dtype == bt.dtype == t2r.dtype == torch.int32):
        return False
    if out is not None and (out.shape != q.shape or out.stride() != q.stride() or out.dtype != q.dtype):
        return False
    tensors = [q, k, v, idx, bt, t2r] + ([] if out is None else [out])
    for t in tensors:
        if t.device != q.device or t.data_ptr() % 16 or t.storage_offset() < 0:
            return False
        if (t.storage_offset() + sum((d - 1) * s for d, s in zip(t.shape, t.stride())) + 1) * t.element_size() \
                > t.untyped_storage().nbytes():
            return False
    return True


def admitted(q, k, v, idx, bt, t2r, out):
    page1600 = page1600_admitted(q, k, v, idx, bt, t2r, out)
    if not (q.is_cuda and q.ndim == 3 and k.ndim == 4 and v.shape == k.shape and q.shape[0] > 256
            and q.shape[1:] == (24, 256) and (k.shape[1:] == (1568, 2, 256) or page1600) and bt.ndim == 2
            and idx.ndim == 2 and idx.shape == (q.shape[0], 2051) and t2r.shape == (q.shape[0],)):
        return False
    if not (q.dtype == k.dtype == v.dtype == torch.bfloat16 and idx.dtype == bt.dtype == t2r.dtype == torch.int32):
        return False
    if not (q.stride(2) == k.stride(3) == v.stride(3) == 1 and idx.stride(1) == bt.stride(1) == 1
            and t2r.stride(0) == 1 and all(bt.shape) and k.shape[0]):
        return False
    if not all(s % 8 == 0 for s in (q.stride(0), q.stride(1), *k.stride()[:3], *v.stride()[:3])):
        return False
    if not all(t.data_ptr() % 16 == 0 for t in (q, k, v)):
        return False
    if out is not None and (out.shape != q.shape or out.dtype != q.dtype or out.stride(2) != 1
                            or out.stride(0) % 8 or out.stride(1) % 8 or out.data_ptr() % 16):
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    d = q.device.index if q.device.index is not None else torch.cuda.current_device()
    if d not in _ARCH:
        _ARCH[d] = torch.cuda.get_device_properties(d).gcnArchName.split(':')[0]
    return _ARCH[d] == 'gfx1151'


def _launch(q, k, v, idx, bt, t2r, out):
    m = _module()
    hm = _MOD['hm']
    I64, I32 = ctypes.c_int64, ctypes.c_int32
    rows = q.shape[0]
    nprog = (rows + 3) // 4
    with torch.cuda.device(q.device):
        m.launch(FN, (nprog, k.shape[2], 1), (384, 1, 1),
                 [hm.ptr(q), hm.ptr(k), hm.ptr(v), hm.ptr(idx), hm.ptr(bt), hm.ptr(t2r), hm.ptr(out),
                  ctypes.c_void_p(0), ctypes.c_void_p(0),
                  I64(q.stride(0)), I64(q.stride(1)), I64(k.stride(0)), I64(k.stride(1)), I64(k.stride(2)),
                  I64(v.stride(0)), I64(v.stride(1)), I64(v.stride(2)), I64(out.stride(0)), I64(out.stride(1)),
                  I64(idx.stride(0)), I64(bt.stride(0)),
                  I32(rows), I32(k.shape[0]), I32(k.shape[1]), I32(bt.shape[1]), I32(bt.shape[0]), I32(SHARE)])


def sparse_paged_attention(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out=None):
    """Same contract as ops.qsa.qsa_sparse_paged_attention (writes and returns out)."""
    split_k = _ops().qsa_sparse_paged_attention
    if not ENABLED or not admitted(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out):
        _STATE['release_calls'] += 1
        return split_k(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out)
    _init()
    if out is None:
        out = torch.empty_like(q)
    _launch(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out)
    rows = int(q.shape[0])
    _STATE['tau_calls'] += 1
    _STATE['page1600_calls'] += int(k_cache.shape[1] == 1600)
    _STATE['tau_rows'] += rows
    _STATE['rows_min'] = rows if _STATE['rows_min'] is None else min(_STATE['rows_min'], rows)
    _STATE['rows_max'] = rows if _STATE['rows_max'] is None else max(_STATE['rows_max'], rows)
    nreq = int(block_table.shape[0])
    if _STATE['tau_calls'] == 1:
        _log('HTR_QSA_TAU_FIRST', rows=rows, kernel=FN, share=SHARE, hsaco_sha256=HSACO_SHA256, hip_lib=_MOD['lib'],
             attrs=_MOD['attrs'], page_table=list(block_table.shape), nreq=nreq,
             kv_shape=list(k_cache.shape), kv_stride=list(k_cache.stride()), page=int(k_cache.shape[1]))
    multi = nreq > 1
    _STATE['mr_calls'] += multi
    if _STATE['checks'] < CHECKS or (multi and _STATE['mr_checks'] < MR_CHECKS):
        ref = split_k(q, k_cache, v_cache, logical_indices, block_table, token_to_req, torch.full_like(q, float('nan')))
        same = bool(torch.equal(ref.view(torch.int16), out.view(torch.int16)))
        if _STATE['checks'] < CHECKS:
            kind = 'first'
            _STATE['checks'] += 1
        else:
            kind = 'multi'
            _STATE['mr_checks'] += 1
        _log('HTR_QSA_TAU_SELFCHECK', n=_STATE['checks'] + _STATE['mr_checks'], rows=rows, bitwise=same, nreq=nreq,
             kind=kind, page=int(k_cache.shape[1]))
        if not same:
            raise RuntimeError('HTR_QSA_TAU selfcheck: tau v2 differs from split-K')
    n = _STATE['tau_calls']
    if n in (1, 10, 100) or n % 1000 == 0:
        _log('HTR_QSA_TAU_STATS', t_ns=time.time_ns(), **_STATE)
    return out


def attention(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out, *, zero_fallback=False):
    """QSA owner entry point. zero_fallback: the caller skipped zero-initialising out[:rows]
    (DIRECT_QSA_OUT_TAILZERO); zero it here so every path sees the same zero-initialised output."""
    if zero_fallback and out is not None:
        out.zero_()
    return sparse_paged_attention(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out)


@triton.jit
def _gate_kernel(O, G, N, SG0, SG1, XBLOCK: tl.constexpr):
    # body copied from the production inductor kernel triton_poi_fused_mul_sigmoid_slice_view_0 (in place)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < N
    x3 = xindex
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 24)
    x2 = xindex // 6144
    tmp0 = tl.load(O + (x3), xmask).to(tl.float32)
    tmp1 = tl.load(G + (x0 + SG1 * x1 + SG0 * x2), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(O + (x3), tmp3, xmask)


def gate_apply_(o, g):
    assert o.dim() == 3 and tuple(o.shape[1:]) == (24, 256) and o.is_contiguous() and o.dtype == torch.bfloat16
    assert g.shape == o.shape and g.stride(2) == 1 and g.dtype == torch.bfloat16
    n = o.numel()
    if n:
        _gate_kernel[(triton.cdiv(n, 1024),)](o, g, n, g.stride(0), g.stride(1), XBLOCK=1024, num_warps=4)
    return o
