"""Source-bound numerical policy for the gfx1151 Qwen GDN norm/gate.

Choose the qualified XBLOCK=1 reduction before native compilation/cache lookup.
The ordinary static launcher and fast path remain unchanged. Serialized kernels
must carry that same geometry; old seeds fail closed instead of silently loading
an XBLOCK=16 binary. No kernel arithmetic or compiler binary is rewritten here.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import sys

NAME = "triton_per_fused__to_copy_add_linear_mean_mul_pow_rsqrt_sigmoid_view_0"
SOURCE_SHA256 = "f1b6431679e13cff672efcd5d79b61721822393536c3a85546045e2feea431e9"
SIGNATURE = dict(in_ptr0="*bf16", in_ptr1="*bf16", in_ptr2="*bf16",
                 out_ptr1="*bf16", ks0="i64", xnumel="i32",
                 r0_numel="i32", XBLOCK="constexpr")
POLICY = "gdn-rms-gate-x1-w2-v1"


def _require(condition, field):
    if not condition:
        raise RuntimeError("HTR_GDN_PIN incompatible " + field)


def _log(event, **fields):
    print("HTR_GDN_PIN_" + event + " " + json.dumps(
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
    return [triton.Config({"XBLOCK": 1}, num_warps=2, num_stages=1)]


def validate_result(result):
    """Reject stale serialized geometry before the native module is loaded."""
    if result.kernel.name != NAME:
        return False
    c = result.config
    _require(c.kwargs == {"XBLOCK": 1} and c.num_warps == 2
             and c.num_stages == 1, "geometry (regenerate this runtime's seed)")
    _require(result.compile_meta.get("signature") == SIGNATURE, "signature")
    _require(result.compile_meta.get("constants", {}).get("XBLOCK") == 1,
             "constants")
    return True


def install():
    """Install in serving and compiler workers when enabled by the launcher."""
    enabled = os.environ.get("HTR_GDN_XBLOCK1", "0")
    if enabled == "0":
        return False
    _require(enabled == "1", "HTR_GDN_XBLOCK1 (expected 0 or 1)")
    import torch._inductor.runtime.triton_heuristics as th
    if getattr(th.cached_autotune, "_htr_gdn_pin", False):
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
                     and tuner.configs[0].kwargs == {"XBLOCK": 1}
                     and tuner.configs[0].num_warps == 2
                     and tuner.configs[0].num_stages == 1,
                     "post-cache geometry")
            _log("COMPILE", name=NAME, source_sha256=SOURCE_SHA256,
                 xblock=1, num_warps=2, num_stages=1)
            return tuner
        return decorate

    @functools.wraps(static)
    def guarded_static(result):
        target = validate_result(result)
        launcher = static(result)
        if target:
            k = result.kernel
            _require(launcher._is_static and k.function is not None
                     and launcher.config.kwargs == {"XBLOCK": 1}
                     and launcher.__globals__["runner"].__self__ is k,
                     "native static launcher")
            raw = k.cubin_raw
            if raw is None:
                raw = Path(k.cubin_path).read_bytes()
            _log("LOADED", name=NAME, xblock=1, num_warps=k.num_warps,
                 shared=k.shared, function=k.function, module=k.module,
                 binary_sha256=hashlib.sha256(raw).hexdigest(),
                 n_regs=k.n_regs, n_spills=k.n_spills)
        return launcher

    @functools.wraps(other)
    def guarded_other(result):
        _require(result.kernel.name != NAME, "nonstatic target launcher")
        return other(result)

    cached._htr_gdn_pin = True
    th.StaticTritonCompileResult.make_launcher = guarded_static
    th.TritonCompileResult.make_launcher = guarded_other
    th.cached_autotune = cached
    _log("ACTIVE", source_sha256=SOURCE_SHA256)
    return True
