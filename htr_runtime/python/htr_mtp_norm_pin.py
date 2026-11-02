"""Reference-bound MTP RMS reduction geometry for the exact gfx1151 source.

Preserve the native static launcher, generated arithmetic and ABI. Only MTP3
installs this supplemental policy; GDN and target-only serving remain separate.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import sys

NAME = "triton_red_fused__to_copy_abs_add_all_eq_mul_ne_rms_norm_view_2"
SOURCE_SHA256 = "21d53726886832ef4362fc523e8c63fef01b27be56d7559023070f4c681bb1e6"
SIGNATURE = dict(in_ptr0="*bf16", in_ptr1="*bf16", out_ptr1="*i1",
                 out_ptr2="*bf16", xnumel="i32", r0_numel="i32",
                 XBLOCK="constexpr", R0_BLOCK="constexpr")
CONFIG = {"XBLOCK": 1, "R0_BLOCK": 16384}
POLICY = "mtp-rms-reference-r16384-w32-v1"


def _require(condition, field):
    if not condition:
        raise RuntimeError("HTR_MTP_NORM_PIN incompatible " + field)


def _log(event, **fields):
    print("HTR_MTP_NORM_PIN_" + event + " " + json.dumps(
        {"pid": os.getpid(), "policy": POLICY, **fields}, sort_keys=True),
        file=sys.stderr, flush=True)


def select(fn, configs, triton_meta, inductor_meta):
    """Pin only the exact qualified function/signature; preserve all others."""
    if (inductor_meta or {}).get("kernel_name") != NAME:
        return configs
    _require(hashlib.sha256(fn.src.encode()).hexdigest() == SOURCE_SHA256,
             "source")
    _require(triton_meta.get("signature") == SIGNATURE, "signature")
    import triton
    return [triton.Config(CONFIG.copy(), num_warps=32, num_stages=1)]


def validate_result(result):
    """Reject stale serialized geometry before the native module is loaded."""
    if result.kernel.name != NAME:
        return False
    c = result.config
    _require(c.kwargs == CONFIG.copy() and c.num_warps == 32
             and c.num_stages == 1, "geometry (regenerate this runtime's seed)")
    _require(result.compile_meta.get("signature") == SIGNATURE, "signature")
    _require(all(result.compile_meta.get("constants", {}).get(key) == value
                 for key, value in CONFIG.items()), "constants")
    return True


def install():
    """Install in serving and compiler workers when enabled by the launcher."""
    enabled = os.environ.get("HTR_GDN_XBLOCK1", "0")
    if enabled == "0" or os.environ.get("HTR_GDN_MTP", "0") != "3":
        return False
    _require(enabled == "1", "HTR_GDN_XBLOCK1 (expected 0 or 1)")
    import torch._inductor.runtime.triton_heuristics as th
    if getattr(th.cached_autotune, "_htr_mtp_norm_pin", False):
        return True
    original = th.cached_autotune
    static = th.StaticTritonCompileResult.make_launcher
    other = th.TritonCompileResult.make_launcher

    @functools.wraps(original)
    def cached(size_hints, configs, triton_meta, heuristic_type,
               filename=None, inductor_meta=None, **kwargs):
        def native(chosen):
            return original(size_hints, chosen, triton_meta, heuristic_type,
                            filename=filename, inductor_meta=inductor_meta,
                            **kwargs)
        if (inductor_meta or {}).get("kernel_name") != NAME:
            return native(configs)

        def decorate(fn):
            chosen = select(fn, configs, triton_meta, inductor_meta)
            tuner = native(chosen)(fn)
            _require(len(tuner.configs) == 1
                     and tuner.configs[0].kwargs == CONFIG.copy()
                     and tuner.configs[0].num_warps == 32
                     and tuner.configs[0].num_stages == 1,
                     "post-cache geometry")
            _log("COMPILE", name=NAME, source_sha256=SOURCE_SHA256,
                 xblock=1, r0_block=16384, num_warps=32, num_stages=1)
            return tuner
        return decorate

    @functools.wraps(static)
    def guarded_static(result):
        if not validate_result(result):
            return static(result)
        k = result.kernel
        # Native load_kernel consumes both cubin_raw and cubin_path.
        # Resolve through the native cache loader, then hash the actual
        # file it will load BEFORE constructing the ordinary launcher.
        if not k.cubin_path:
            result.reload_cubin_path()
        binary_sha256 = hashlib.sha256(Path(k.cubin_path).read_bytes()).hexdigest()
        launcher = static(result)
        _require(launcher._is_static and k.function is not None
                 and launcher.config.kwargs == CONFIG.copy()
                 and launcher.__globals__["runner"].__self__ is k,
                 "native static launcher")
        _log("LOADED", name=NAME, xblock=1, r0_block=16384, num_warps=k.num_warps,
             shared=k.shared, function=k.function, module=k.module,
             binary_sha256=binary_sha256,
             n_regs=k.n_regs, n_spills=k.n_spills)
        return launcher

    @functools.wraps(other)
    def guarded_other(result):
        _require(result.kernel.name != NAME, "nonstatic target launcher")
        return other(result)

    cached._htr_mtp_norm_pin = True
    th.StaticTritonCompileResult.make_launcher = guarded_static
    th.TritonCompileResult.make_launcher = guarded_other
    th.cached_autotune = cached
    _log("ACTIVE", source_sha256=SOURCE_SHA256, xblock=1, r0_block=16384)
    return True
