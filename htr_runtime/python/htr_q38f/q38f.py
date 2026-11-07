"""ctypes bridge for libq38f.so: drop-in for q38spec.q38_filter_logits(logits, idx_mapping, temperature, top_k, top_p, kc).
Supported: CUDA/HIP, 2-D, bf16 or fp32 logits with unit last stride, kc == 64, int32 idx/top_k, fp32 temperature/top_p
(contiguous), rows <= MAX_ROWS, vocabulary large enough for the 16-block slice contract. Anything else -> fallback
(the caller passes the original torch implementation). Graph-capturable: no host sync, static per-(device, rows)
workspace, output from torch.empty_like (capture pool when capturing).
"""
import ctypes
import torch

MAX_ROWS = 8
NT, MAXM, MAXNB = 256, 64, 16


def _geometry(V):
    nb = max(1, -(-V // (NT * MAXM)))
    if nb > MAXNB:
        return None
    L = -(-V // nb)
    if V - (nb - 1) * L < NT:
        return None
    return nb, 16


class Q38FilterHIP:
    def __init__(self, so_path, fallback):
        self.lib = ctypes.CDLL(so_path)
        f = self.lib.q38f_launch
        f.restype = ctypes.c_int
        vp, ll, i = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int
        f.argtypes = [vp, ll, vp, ll, i, i, i, vp, vp, vp, vp, vp, i, i, vp]
        self.fallback = fallback
        self.ws = {}
        self.calls = {'hip': 0, 'fallback': 0}
        self.poison = False  # test-only: pre-fill the output so any unwritten element fails a bitwise comparison

    def supported(self, logits, idx_mapping, temperature, top_k, top_p, kc):
        if kc != 64 or not logits.is_cuda or logits.dim() != 2 or logits.dtype not in (torch.bfloat16, torch.float32):
            return False
        if logits.stride(1) != 1 or not (1 <= logits.shape[0] <= MAX_ROWS) or _geometry(logits.shape[1]) is None:
            return False
        if idx_mapping.dtype != torch.int32 or top_k.dtype != torch.int32:
            return False
        if temperature.dtype != torch.float32 or top_p.dtype != torch.float32:
            return False
        if idx_mapping.numel() != logits.shape[0]:
            return False
        return all(t.is_cuda and t.is_contiguous() and t.device == logits.device
                   for t in (idx_mapping, temperature, top_k, top_p))

    def __call__(self, logits, idx_mapping, temperature, top_k, top_p, kc=64):
        if not self.supported(logits, idx_mapping, temperature, top_k, top_p, kc):
            self.calls['fallback'] += 1
            return self.fallback(logits, idx_mapping, temperature, top_k, top_p, kc)
        rows, V = logits.shape
        nb, nw = _geometry(V)
        key = (logits.device, rows, nb)
        cand = self.ws.get(key)
        if cand is None:
            cand = torch.empty(rows * nb * 64, dtype=torch.int64, device=logits.device)
            self.ws[key] = cand
        out = torch.empty_like(logits)
        if self.poison:
            (out.view(torch.int16) if out.dtype == torch.bfloat16 else out.view(torch.int32)).fill_(0x7A5A)
        rc = self.lib.q38f_launch(logits.data_ptr(), logits.stride(0), out.data_ptr(), out.stride(0), rows, V,
                                  0 if logits.dtype == torch.bfloat16 else 1, idx_mapping.data_ptr(),
                                  temperature.data_ptr(), top_k.data_ptr(), top_p.data_ptr(), cand.data_ptr(), nb, nw,
                                  ctypes.c_void_p(torch.cuda.current_stream(logits.device).cuda_stream))
        if rc != 0:
            raise RuntimeError(f'q38f_launch failed rc={rc} rows={rows} V={V}')
        self.calls['hip'] += 1
        return out
