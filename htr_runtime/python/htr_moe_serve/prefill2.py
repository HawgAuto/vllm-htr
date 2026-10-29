"""E15-C v2 prefill extension loader (compile-only until the GPU gates in design/PREFILL-KERNEL-DESIGN.md §v2 pass; NOT wired
into routes). Separate from prefill.py so the queued v1 operator test is byte-unchanged."""
import os
from .kernels import CSRC, _bd

_HPF2 = None


def hpf2_ext(extra=()):
    global _HPF2
    if _HPF2 is None and os.environ.get("HTR_NATIVE_DIR"):
        from htr_native import load_native
        _HPF2 = load_native("htr_moe_prefill_v2")
    if _HPF2 is None:
        from torch.utils.cpp_extension import load
        _HPF2 = load('htr_moe_prefill_v2', [os.path.join(CSRC, 'htr_moe_prefill_v2.cpp'), os.path.join(CSRC, 'htr_moe_prefill_v2.cu')],
                     build_directory=_bd('hpf2'),
                     extra_cuda_cflags=['-O3', '--offload-arch=' + os.environ.get('HTR_ARCH', 'gfx1151'), *extra],
                     verbose=bool(os.environ.get('HTR_VERBOSE')))
    return _HPF2
