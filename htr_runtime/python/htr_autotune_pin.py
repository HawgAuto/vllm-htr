"""HTR autotune pin (vLLM general plugin; on when HTR_AUTOTUNE_PIN=1, the release default).

Startup autotuning picks kernel configurations by timing them, so two processes on the same machine could pick
different configurations and produce different (equally valid) tokens. With the pin, every process runs the same
kernels:
  1. Triton @triton.autotune kernels of vLLM's FLA gated-delta-rule prefill: each pinned kernel's candidate list is
     replaced, on its first call, with exactly ONE config taken from its own existing list (fails closed if the config
     is absent or ambiguous). With one candidate Triton never benchmarks. Pins = per-kernel winner over 7 fresh
     processes on gfx1151 (H=48, K=V=128, BT=64), merge = lowest-median 2-stage config.
  2. Inductor deterministic mode (torch._inductor.config.deterministic / TORCHINDUCTOR_DETERMINISTIC): reductions
     compile with one filtered config and are never re-tuned; only numerics-neutral pointwise launch configs may still
     be benchmarked. Kept on across Dynamo frames (Dynamo's use_deterministic_algorithms restore would otherwise
     clear it after the first compiled frame).
Logs: HTR_AUTOTUNE_PIN_ACTIVE (register), HTR_AUTOTUNE_PIN_APPLIED (first call of each pinned kernel),
HTR_AUTOTUNE_PIN_BENCH (first Triton-runtime benchmark per kernel anywhere in the process).
"""
import json
import os
import sys
import time

FLA = 'vllm.third_party.flash_linear_attention.ops.'
# (module, kernel) -> (kwargs, num_warps, num_stages)
PINS = {
    (FLA + 'chunk_o', 'chunk_fwd_kernel_o'): ({'BK': 32, 'BV': 64}, 4, 2),
    (FLA + 'chunk_delta_h', 'chunk_gated_delta_rule_fwd_kernel_h_blockdim64'): ({'BV': 32}, 4, 2),
    (FLA + 'chunk_scaled_dot_kkt', 'chunk_scaled_dot_kkt_fwd_kernel'): ({'BK': 32}, 2, 3),
    (FLA + 'cumsum', 'chunk_local_cumsum_scalar_kernel'): ({}, 1, 3),
    (FLA + 'solve_tril', 'merge_16x16_to_64x64_inverse_kernel'): ({}, 4, 2),
    (FLA + 'wy_fast', 'recompute_w_u_fwd_kernel'): ({}, 8, 3),
}


def _log(tag, **kw):
    print(tag + ' ' + json.dumps({'pid': os.getpid(), **kw}, sort_keys=True), file=sys.stderr, flush=True)


def select(configs, kwargs, warps, stages):
    """The single existing config matching the pin exactly; raises unless exactly one matches."""
    m = [c for c in configs if dict(c.kwargs) == kwargs and c.num_warps == warps and c.num_stages == stages]
    if len(m) != 1:
        raise RuntimeError('HTR_AUTOTUNE_PIN: pin matches %d configs' % len(m))
    return m[0]


def pin(tuner):
    """Pin one Autotuner in place if it is a pinned kernel. Returns the pin key or None."""
    fn = tuner.base_fn
    key = (fn.__module__, fn.__name__)
    if key not in PINS or getattr(tuner, '_htr_pin', None) is not None:
        return None
    kw, w, s = PINS[key]
    n = len(tuner.configs)
    tuner.configs = [select(tuner.configs, kw, w, s)]
    tuner._htr_pin = key
    _log('HTR_AUTOTUNE_PIN_APPLIED', kernel=fn.__name__, module=fn.__module__, n_candidates=n,
         config={'kwargs': kw, 'num_warps': w, 'num_stages': s}, t_ns=time.time_ns())
    return key


def register():
    v = os.environ.get('HTR_AUTOTUNE_PIN', '1')
    if v == '0':
        return
    if v != '1':
        raise RuntimeError('HTR_AUTOTUNE_PIN must be 0 or 1')
    import htr_gdn_pin
    htr_gdn_pin.install()
    import htr_mtp_norm_pin
    htr_mtp_norm_pin.install()
    from triton.runtime.autotuner import Autotuner
    if getattr(Autotuner.run, '_htr_pin', False):
        return
    os.environ['TORCHINDUCTOR_DETERMINISTIC'] = '1'
    import torch
    import torch._inductor.config as ic
    ic.deterministic = True
    # torch>=2.13: torch.use_deterministic_algorithms(mode) also writes inductor config.deterministic = mode, and
    # Dynamo restores the eager flag (False) with that call after every traced frame. Without this wrapper Inductor
    # deterministic mode is silently dropped after the first compiled frame (the model backbone then compiles with
    # per-process reduction autotuning). The eager flag itself is passed through unchanged.
    orig_uda = torch.use_deterministic_algorithms
    if not getattr(orig_uda, '_htr_pin', False):
        restored = []

        def use_deterministic_algorithms(mode, *, warn_only=False):
            orig_uda(mode, warn_only=warn_only)
            if not ic.deterministic:
                ic.deterministic = True
                if not restored:
                    restored.append(True)
                    _log('HTR_AUTOTUNE_PIN_RESTORE', eager_mode=bool(mode), inductor_deterministic=True)

        use_deterministic_algorithms._htr_pin = True
        use_deterministic_algorithms.__wrapped__ = orig_uda
        torch.use_deterministic_algorithms = use_deterministic_algorithms
    orig_run, orig_bench = Autotuner.run, Autotuner._bench
    benched = set()

    def run(self, *args, **kwargs):
        pin(self)
        return orig_run(self, *args, **kwargs)

    def _bench(self, *args, config, **meta):
        k = (self.base_fn.__module__, self.base_fn.__name__)
        if k not in benched:
            benched.add(k)
            _log('HTR_AUTOTUNE_PIN_BENCH', kernel=k[1], module=k[0], pinned=k in PINS, n_candidates=len(self.configs))
        return orig_bench(self, *args, config=config, **meta)

    run._htr_pin = True
    Autotuner.run, Autotuner._bench = run, _bench
    _log('HTR_AUTOTUNE_PIN_ACTIVE', inductor_deterministic=bool(ic.deterministic),
         pins={k[1]: {'module': k[0], 'kwargs': p[0], 'num_warps': p[1], 'num_stages': p[2]} for k, p in PINS.items()})
