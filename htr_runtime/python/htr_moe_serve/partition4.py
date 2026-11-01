"""HTR MoE partition4 (on when HTR_MOE_PARTITION4=1, the release default).

Wraps htr_moe_serve.routes.forward. For 5..47-token steps on 4-bit (_kf == 4) expert layers it runs the decode
kernels on 4-row partitions with an integer group-parity plan (partition4_plan.py) that preserves the native
kernel's eight-row group-position parity. Native library, packed weights and arithmetic are unchanged; only integer
planning changes. T <= 4, T > 47 and other bit widths keep the original forward.

The parity recipe depends on the compiled htr_moe2.so, so activation fails closed unless the loaded library is the
qualified build.
"""
import hashlib
import json
import os
import sys

MOE2_SHA256 = '583bd74640272a6bbac492e1769e5fbaac82f01e80198872bac27e7fef915cac'
_SEEN = set()


def _sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _log(tag, **kw):
    print(tag + ' ' + json.dumps(kw, sort_keys=True), file=sys.stderr, flush=True)


class _Proxy:
    """Routes namespace whose .forward is the ORIGINAL function (the plan's fallback must not recurse)."""

    def __init__(self, R, orig):
        object.__setattr__(self, '_R', R)
        object.__setattr__(self, 'forward', orig)

    def __getattr__(self, k):
        return getattr(self._R, k)


def install(R):
    if getattr(R, '_htr_partition4', False):
        return R.forward
    so = os.environ.get('HTR_MOE2_SO', '')
    if not so or _sha(so) != MOE2_SHA256:
        raise RuntimeError('HTR_MOE_PARTITION4: htr_moe2.so is not the qualified library')
    from . import partition4_plan
    orig = R.forward
    p4 = partition4_plan.make_partition(_Proxy(R, orig))

    def forward(ar, x, ids, w):
        t = x.shape[0]
        path = 'p4' if (5 <= t <= 47 and ar['_kf'] == 4) else 'native'
        key = (path, t)
        if key not in _SEEN:
            _SEEN.add(key)
            import torch
            _log('HTR_MOE_PARTITION4_FIRST', pid=os.getpid(), path=path, T=t, kf=int(ar['_kf']),
                 capturing=bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()))
        return p4(ar, x, ids, w)

    forward._htr_orig = orig
    R.forward = forward
    R._htr_partition4 = True
    _log('HTR_MOE_PARTITION4_ACTIVE', pid=os.getpid(), moe2_sha256=MOE2_SHA256)
    return forward


def register():
    v = os.environ.get('HTR_MOE_PARTITION4', '1')
    if v == '0':
        return
    if v != '1':
        raise RuntimeError('HTR_MOE_PARTITION4 must be 0 or 1')
    from . import routes as R
    install(R)
