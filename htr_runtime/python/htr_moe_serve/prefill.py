"""E15-C prefill extension loader (compile-only until the GPU gates in design/PREFILL-KERNEL-DESIGN.md pass; NOT wired into routes)."""
import os
from .kernels import CSRC, _bd

_HPF = None


def hpf_ext(extra=()):
    global _HPF
    if _HPF is None and os.environ.get("HTR_NATIVE_DIR"):
        from htr_native import load_native
        _HPF = load_native("htr_moe_prefill")
    if _HPF is None:
        from torch.utils.cpp_extension import load
        _HPF = load('htr_moe_prefill', [os.path.join(CSRC, 'htr_moe_prefill.cpp'), os.path.join(CSRC, 'htr_moe_prefill.cu')],
                    build_directory=_bd('hpf'),
                    extra_cuda_cflags=['-O3', '--offload-arch=' + os.environ.get('HTR_ARCH', 'gfx1151'), *extra],
                    verbose=bool(os.environ.get('HTR_VERBOSE')))
    return _HPF
