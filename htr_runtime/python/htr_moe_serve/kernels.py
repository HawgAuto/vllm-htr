"""Torch extension loaders (JIT-built once into the e15 build dir; build.sh prebuilds them CPU-only in the pinned image, so a
serving process only loads the cached .so).  htr_moe_serve_k = byte copy of e12 csrc/htr_moe.{cpp,cu} (v2 routed decode kernel,
same flags as e12 moe.ext());  htr_dq_arena = tools/build_dq_arena.py output (eval48 dq_ref with arena addressing)."""
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSRC = os.path.join(HERE, 'csrc')
_MOE = None
_DQ = None


def _bd(name):
    root = os.environ.get('HTR_MOE_SERVE_BUILD') or os.path.join(HERE, 'build')
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    return d


def moe_ext():
    global _MOE
    if _MOE is None and os.environ.get("HTR_NATIVE_DIR"):
        from htr_native import load_native
        _MOE = load_native("htr_moe_serve_k")
    if _MOE is None:
        from torch.utils.cpp_extension import load
        _MOE = load('htr_moe_serve_k', [os.path.join(CSRC, 'htr_moe.cpp'), os.path.join(CSRC, 'htr_moe.cu')], build_directory=_bd('moe'),
                    extra_cuda_cflags=['-O3', '--offload-arch=' + os.environ.get('HTR_ARCH', 'gfx1151'), '-save-temps'],
                    verbose=bool(os.environ.get('HTR_VERBOSE')))
    return _MOE


def dq_ext():
    global _DQ
    if _DQ is None and os.environ.get("HTR_NATIVE_DIR"):
        from htr_native import load_native
        _DQ = load_native("htr_dq_arena")
    if _DQ is None:
        from torch.utils.cpp_extension import load
        _DQ = load('htr_dq_arena', [os.path.join(CSRC, 'dq_arena.cpp'), os.path.join(CSRC, 'dq_arena.cu')], build_directory=_bd('dq'),
                   extra_cuda_cflags=['-O3', '--offload-arch=' + os.environ.get('HTR_ARCH', 'gfx1151')],
                   verbose=bool(os.environ.get('HTR_VERBOSE')))
    return _DQ


# ---- E29: prebuilt fused down+combine module (e29 csrc/htr_moe2.{cpp,cu}; tests/qualify_fuse.py bitwise vs this module's
# kernels); imported from a fixed .so path (no JIT, no build-dir writes) when HTR_MOE2_SO is set.
_MOE2 = None


def moe2_ext():
    global _MOE2
    if _MOE2 is None:
        import importlib.util
        path = os.environ['HTR_MOE2_SO']
        spec = importlib.util.spec_from_file_location('htr_moe2', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MOE2 = mod
    return _MOE2


# ---- E31: prebuilt hpf3 prefill module (e31 csrc/htr_moe_prefill_v3.{cpp,cu}: v2 GEMMs + v3 glue kernels, bitwise == hpf2)
_HPF3 = None


def hpf3_ext():
    global _HPF3
    if _HPF3 is None:
        import importlib.util
        path = os.environ['HTR_HPF3_SO']
        spec = importlib.util.spec_from_file_location('htr_moe_prefill_v3', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _HPF3 = mod
    return _HPF3
