"""Private opt-in q38 draft-filter bridge; original q38spec policy and model graph remain unchanged.
HTR_Q38F=off: no replacement. on: qualified HIP filter. verify: returns HIP while comparing original outputs in every graph.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIB = ROOT / 'libq38f.so'
_PATCHED = False
_OP = None
_SEEN = set()
_VERIFY = {}

def emit(kind, **data):
    print('HTR_Q38F ' + json.dumps(dict(kind=kind, pid=os.getpid(), **data)), file=sys.stderr, flush=True)

def validate_payload():
    manifest = json.loads((ROOT / 'payload.json').read_text())
    for name, sha in manifest['files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise RuntimeError('q38f payload changed: ' + name)
    return manifest

def eligible(x, idx, temp, topk, topp, kc):
    import torch
    if not x.is_cuda or kc != 64 or x.ndim != 2 or x.shape[1] != 248320:
        return False
    if not (1 <= x.shape[0] <= 8) or x.stride(1) != 1 or x.dtype not in (torch.bfloat16, torch.float32):
        return False
    if torch.version.hip != '7.15.26333':
        return False
    if torch.cuda.get_device_properties(x.device).gcnArchName.split(':')[0] != 'gfx1151':
        return False
    ts = (idx, temp, topk, topp)
    if any(t.ndim != 1 or not t.is_contiguous() or t.device != x.device for t in ts):
        return False
    if idx.numel() != x.shape[0] or not (temp.numel() == topk.numel() == topp.numel() > 0):
        return False
    return idx.dtype == topk.dtype == torch.int32 and temp.dtype == topp.dtype == torch.float32

def register():
    global _PATCHED, _OP
    arm = os.environ.get('HTR_Q38F', 'off')
    if arm == 'off' or _PATCHED:
        return
    if arm not in ('on', 'verify'):
        raise RuntimeError('unknown HTR_Q38F arm: ' + arm)
    import torch
    import q38spec
    from .q38f import Q38FilterHIP
    manifest = validate_payload()
    if str(torch.__version__) != manifest['torch']:
        raise RuntimeError('q38f torch oracle version changed')
    original = q38spec.q38_filter_logits
    _OP = Q38FilterHIP(str(LIB), original)

    def filtered(x, idx, temp, topk, topp, kc=64):
        if not eligible(x, idx, temp, topk, topp, kc):
            key = ('fallback', str(x.device), tuple(x.shape), str(x.dtype))
            if key not in _SEEN:
                _SEEN.add(key); emit('fallback', shape=list(x.shape), dtype=str(x.dtype))
            return original(x, idx, temp, topk, topp, kc)
        capturing = torch.cuda.is_current_stream_capturing()
        # Per-stream bridge workspace avoids races between independent graph/stream users.
        stream = torch.cuda.current_stream(x.device).cuda_stream
        key = (x.device, stream)
        if key not in bridges:
            bridges[key] = Q38FilterHIP(str(LIB), original)
        got = bridges[key](x, idx, temp, topk, topp, kc)
        route = (tuple(x.shape), str(x.dtype), capturing)
        if route not in _SEEN:
            _SEEN.add(route); emit('hip', shape=list(x.shape), stride=list(x.stride()), dtype=str(x.dtype),
                                   capturing=capturing, arm=arm, library_sha256=manifest['library_sha256'])
        if arm == 'verify':
            expected = original(x, idx, temp, topk, topp, kc)
            if key not in _VERIFY:
                _VERIFY[key] = torch.zeros(2, dtype=torch.int64, device=x.device)
            word = torch.int16 if x.dtype == torch.bfloat16 else torch.int32
            _VERIFY[key][0].add_(1)
            _VERIFY[key][1].add_((got.view(word) != expected.view(word)).sum())
        return got

    bridges = {}
    q38spec.q38_filter_logits = filtered
    if arm == 'verify':
        from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
        run = CudaGraphManager.run_fullgraph
        calls = [0]
        def checked_graph(self, desc):
            result = run(self, desc)
            counts = [t.cpu().tolist() for t in _VERIFY.values()]
            calls[0] += 1
            if counts:
                total = sum(x[0] for x in counts); bad = sum(x[1] for x in counts)
                if calls[0] <= 10 or calls[0] % 16 == 0 or bad:
                    emit('live_bitcheck', comparisons=total, mismatch_elements=bad,
                         graph_manager=type(self).__name__)
                if bad:
                    raise RuntimeError('q38f live bitwise mismatch: ' + str(bad))
            return result
        CudaGraphManager.run_fullgraph = checked_graph
    _PATCHED = True
    emit('installed', arm=arm, library_sha256=manifest['library_sha256'], oracle=q38spec.__file__)
