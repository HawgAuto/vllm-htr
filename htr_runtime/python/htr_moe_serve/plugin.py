"""vLLM general plugin: HTR routed experts for Qwen3.8 HaloLoom checkpoints (e15, 2026-10-23).

Selection: QuarkConfig.from_config is wrapped at class level (idempotent). When the checkpoint's quantization_config carries an
"htr_moe" block (and the image's mixed_moe selection produced a QuarkMixedConfig), QuarkMixedConfig.get_quant_method is wrapped
so that exactly the listed trunk routed-expert layers (prefix == <vllm_prefix><L>.mlp.experts, also present in mixed_plans) get
HTRMoEMethod; every other prefix (MTP draft experts, dense, router, HC, PLE) goes to the untouched original selector. Configs
without "htr_moe" are unaffected (same object/type, no spec attached).

HTRMoEMethod subclasses the image's QuarkMixedMoEMethod only to inherit its framework contract and constructor guards (single
TP/EP, K2560/I640/BF16, SiLU, no bias/LoRA/deferred finalize; skip_forward_padding; outer-runner-owned shared experts). It
registers zero-length root entry params whose loader REJECTS any arriving release expert piece (the variant folder's index drops
the trunk expert tensors), and takes its weights from the packed P2 arena (arena.Prefetcher, sha-verified per chunk file).
Forward: routes.forward (decode kernel T<=8, chunked kernel T<=cut, dequant+stock BF16 fused_experts above)."""
import json, os, re, sys, threading, time
import torch

FORMAT = 'HTR-W1-P2-PACKED-1'
SPEC_KEYS = {'wire', 'K', 'dir', 'manifest', 'manifest_sha256', 'layers', 'vllm_prefix', 'spad', 'note'}
_SELECTED = []
_COUNT_FILE = os.environ.get('HTR_MOE_COUNT_FILE') or None
_SELFCHECK = os.environ.get('HTR_MOE_SELFCHECK') == '1'
_COUNTS = {}
_CLOCK = threading.Lock()


class Spec(dict):
    def layer_of(self, prefix):
        vp = self['vllm_prefix']
        if not prefix.startswith(vp):
            return None
        m = re.fullmatch(r'(\d+)\.mlp\.experts', prefix[len(vp):])
        if m is None or int(m.group(1)) not in self['layers']:
            return None
        return int(m.group(1))


def parse_spec(ov):
    if not isinstance(ov, dict) or not set(ov) <= SPEC_KEYS or not (SPEC_KEYS - {'note'}) <= set(ov):
        raise ValueError(f'htr_moe: keys {sorted(ov) if isinstance(ov, dict) else type(ov)} != {sorted(SPEC_KEYS)}')
    if ov['wire'] != FORMAT or ov['K'] not in (3, 4, 5):
        raise ValueError(f'htr_moe: wire/K {ov["wire"]}/{ov["K"]} unsupported')
    for k in ('dir', 'manifest'):
        if not isinstance(ov[k], str) or not os.path.isabs(ov[k]):
            raise ValueError(f'htr_moe: {k} must be an absolute path')
    if not re.fullmatch(r'[0-9a-f]{64}', str(ov['manifest_sha256'])):
        raise ValueError('htr_moe: manifest_sha256')
    L = ov['layers']
    if not isinstance(L, list) or not L or len(set(L)) != len(L) or any(type(x) is not int or not 0 <= x < 48 for x in L):
        raise ValueError('htr_moe: layers must be unique ints in [0, 48)')
    if not isinstance(ov['vllm_prefix'], str) or not ov['vllm_prefix'].endswith('.') or 'mtp' in ov['vllm_prefix']:
        raise ValueError('htr_moe: vllm_prefix')
    if set(ov['spad']) != {'3', '4', '5'} or any(type(v) is not int or v < 0 or v % 4 for v in ov['spad'].values()):
        raise ValueError('htr_moe: spad {"3","4","5"} -> words (multiple of 4)')
    return Spec(ov)


def _log(tag, **kw):
    print(tag + ' ' + json.dumps(dict(pid=os.getpid(), **kw), sort_keys=True), file=sys.stderr, flush=True)


def _count(key):
    if _COUNT_FILE is None:
        return
    with _CLOCK:
        _COUNTS[key] = _COUNTS.get(key, 0) + 1
        n = sum(_COUNTS.values())
        if n % 100 == 1 or n < 400:
            tmp = f'{_COUNT_FILE}.{os.getpid()}.tmp'
            with open(tmp, 'w') as f:
                json.dump(dict(pid=os.getpid(), t=time.time(), selected=len(_SELECTED), counts=_COUNTS), f)
            os.replace(tmp, f'{_COUNT_FILE}.{os.getpid()}.json')


def _method_class():
    from quark_mixed_moe import QuarkMixedMoEMethod, PoolPlan, validate_outer_shared_owner
    from . import arena as A, routes as R

    class HTRMoEMethod(QuarkMixedMoEMethod):
        def __init__(self, moe, L, spec):
            super().__init__(moe, PoolPlan(tuple(range(512)), (), 512), int8_wire=None)
            if moe.experts_per_token != R.TOPK:
                raise ValueError(f'HTR MoE: top-k {moe.experts_per_token} != {R.TOPK}')
            self.L, self.spec, self.ar = L, spec, None
            self._checked = set()
            dev = torch.device(self.moe.device) if not isinstance(self.moe.device, torch.device) else self.moe.device
            if dev.type == 'cuda' and dev.index is None:
                dev = torch.device('cuda', torch.cuda.current_device())
            self._dev = dev
            if dev.type == 'cuda':
                self._pf = A.prefetcher(spec, dev)        # starts loading all spec layers now (overlaps the trunk load)

        def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **attrs):
            if (num_experts, hidden_size, intermediate_size_per_partition, params_dtype) != (512, 2560, 640, torch.bfloat16):
                raise ValueError('wrong HTR MoE layer geometry or dtype')
            if hasattr(layer, 'htr_moe_layer'):
                raise ValueError('weights already created')
            layer.htr_moe_layer = self.L
            for name in ('w13_weight', 'w2_weight', 'w13_weight_scale', 'w2_weight_scale'):
                param = torch.nn.Parameter(torch.empty(0, device=self.moe.device), requires_grad=False)
                param.weight_loader = self.weight_loader; param.quark_parameter_name = name
                layer.register_parameter(name, param)
            layer.orig_dtype = params_dtype; layer.weight_block_size = None

        def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
            raise ValueError(f'HTR MoE L{self.L}: release expert tensor {weight_name} in the checkpoint index; the variant folder '
                             'must drop trunk expert tensors (tools/make_variant.py)')

        def process_weights_after_loading(self, layer):
            if self._ready:
                raise ValueError('already processed')
            t = time.time()
            ar = self._pf.get(self.L)
            m = ar['_meta']
            if m['L'] != self.L or ar['_K'] != self.spec['K'] or int(ar['kexp'].min()) != int(ar['kexp'].max()) or ar['qw'].device != self._dev:
                raise ValueError(f'HTR MoE L{self.L}: arena mismatch {m}')
            self.ar = ar; self._ready = True
            _log('HTR_MOE_LAYER_READY', wait_s=round(time.time() - t, 2), **m)

        def get_fused_moe_quant_config(self, layer):
            if not self._ready:
                raise ValueError('HTR MoE weights not ready')
            return None

        def _selfcheck(self, x, ids, w, route):
            # first eager call per route: kernel route vs dequant+BF16 fused_experts (independent kernel path); rel L2 gate
            key = route
            if key in self._checked or torch.cuda.is_current_stream_capturing():
                return
            if route == 'v2':      # v2 must be checked on >= V2_MIN rows (R.forward on <= 8 rows would take the decode route)
                n = min(x.shape[0], 1024)
                a = R.v2_route(self.ar, x[:n].contiguous(), ids[:n], w[:n]).float()
            else:
                n = min(x.shape[0], R.MAXT)
                a = R.forward(self.ar, x[:n].contiguous(), ids[:n], w[:n]).float()
            try:
                b = R.dequant_route(self.ar, x[:n].contiguous(), ids[:n], w[:n], persistent_scratch=False).float()
                rel = float((a - b).norm() / b.norm().clamp_min(1e-30))
                ok = bool(torch.isfinite(a).all()) and rel <= 3e-2
                _log('HTR_MOE_SELFCHECK', L=self.L, route=route, rows=n, rel=rel, ok=ok,
                     reference_scratch='temporary')
                if not ok:
                    raise RuntimeError(f'HTR MoE selfcheck L{self.L} {route}: rel {rel}')
                self._checked.add(key)
            finally:
                # No graph capture reaches this block. The reference tensors have no
                # global owner; return their inactive blocks to ROCm as well, rather
                # than leaving 4.6875 GiB reserved by the caching allocator. Live
                # fallback/graph allocations are untouched by empty_cache().
                with torch.cuda.device(x.device):
                    torch.cuda.empty_cache()

        def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
            if x.device.type != 'cuda':
                raise ValueError('HTR MoE requires a GPU tensor; no CPU fallback')
            if layer.apply_router_weight_on_input:
                raise ValueError('router-on-input is unsupported')
            validate_outer_shared_owner(shared_experts)        # stock runner owns and adds shared experts; never folded here
            if not self._ready:
                raise ValueError('HTR MoE weights not ready')
            T = x.shape[0]
            if (x.ndim != 2 or x.shape[1] != 2560 or x.dtype != torch.bfloat16 or topk_ids.shape != (T, R.TOPK)
                    or topk_weights.shape != topk_ids.shape or topk_ids.dtype not in (torch.int32, torch.int64)
                    or not topk_weights.dtype.is_floating_point or topk_ids.device != x.device):
                raise ValueError('invalid routing tensor geometry/dtype/device')
            if T == 0:
                return torch.zeros_like(x)
            route = R.route_name(T)
            if not torch.cuda.is_current_stream_capturing():
                from vllm.forward_context import get_forward_context, is_forward_context_available
                pad = get_forward_context().is_padding if is_forward_context_available() else None
                valid = (topk_ids >= 0) & (topk_ids < 512)
                if pad is not None and isinstance(pad, torch.Tensor) and pad.ndim == 1 and pad.shape[0] >= T:
                    valid = valid | ((topk_ids == -1) & pad[:T, None])
                torch._assert_async(valid.all(), 'HTR MoE: invalid global expert ID')
                _count(f'{route}:T{T if T <= 8 else ">8"}')
                if _SELFCHECK:
                    self._selfcheck(x, topk_ids, topk_weights, route)
            else:
                _count(f'capture:{route}:T{T}')
            return R.forward(self.ar, x.contiguous(), topk_ids, topk_weights)

    return HTRMoEMethod


_CLS = []


def method_class():
    if not _CLS:
        _CLS.append(_method_class())
    return _CLS[0]


def _spec_of(cfg):
    """The attached spec, else (engine-core process: the config arrives pickled, from_config never runs there) re-parsed from the
    checkpoint's own quantization_config['htr_moe'] block. None for configs without that block."""
    spec = cfg.__dict__.get('_htr_moe_spec')
    if spec is None:
        qc = cfg.__dict__.get('quant_config')
        ov = qc.get('htr_moe') if isinstance(qc, dict) else None
        if ov is not None:
            spec = parse_spec(ov)
            cfg.__dict__['_htr_moe_spec'] = spec
            _log('HTR_MOE_SPEC', layers=len(spec['layers']), K=spec['K'], dir=spec['dir'], via='quant_config')
    return spec


def _wrap_mixed(cls):
    if getattr(cls.get_quant_method, '_htr_moe', False):
        return
    orig = cls.get_quant_method

    def get_quant_method(self, layer, prefix):
        spec = _spec_of(self)
        L = spec.layer_of(prefix) if spec is not None else None
        if L is None:
            return orig(self, layer, prefix)
        from vllm.model_executor.layers.fused_moe import RoutedExperts
        if prefix not in self.mixed_plans or not isinstance(layer, RoutedExperts):
            raise ValueError(f'htr_moe: {prefix} is not a mixed routed-experts layer of this checkpoint')
        m = method_class()(layer.moe_config, L, spec)
        _SELECTED.append((prefix, L))
        _log('HTR_MOE_SELECTED', prefix=prefix, L=L, n=len(_SELECTED))
        return m
    get_quant_method._htr_moe = True
    get_quant_method._htr_orig = orig
    cls.get_quant_method = get_quant_method


def install():
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
    fc = QuarkConfig.__dict__['from_config']
    if getattr(fc.__func__, '_htr_moe', False):
        return
    orig = fc.__func__

    def from_config(cls, config):
        r = orig(cls, config)
        ov = config.get('htr_moe') if isinstance(config, dict) else None
        if ov is None:
            return r
        spec = parse_spec(ov)
        if type(r).__name__ != 'QuarkMixedConfig':
            raise ValueError('htr_moe requires the mixed_moe selection (QuarkMixedConfig)')
        _wrap_mixed(type(r))
        r.__dict__['_htr_moe_spec'] = spec
        _log('HTR_MOE_SPEC', layers=len(spec['layers']), K=spec['K'], dir=spec['dir'])
        return r
    from_config._htr_moe = True
    from_config._htr_orig = orig
    QuarkConfig.from_config = classmethod(from_config)
    # Wrap the mixed selector at plugin load too: vLLM loads general plugins in every process (API server, engine core, workers),
    # but only the API server calls from_config; the engine core gets the config object pickled. (e15 canary4 13:03 failure.)
    from quark_mixed_moe import QuarkMixedConfig
    _wrap_mixed(QuarkMixedConfig)
    _log('HTR_MOE_PLUGIN', wrapped=bool(getattr(QuarkMixedConfig.get_quant_method, '_htr_moe', False)))


def register():
    """vllm.general_plugins entry point (vllm_site/htr_moe_serve-0.1.0.dist-info). Never import from sitecustomize."""
    install()
    from . import partition4
    partition4.register()
    from . import grouped
    grouped.register()
    register_k6w()      # v0.2.4 K6W: OUTERMOST, installed last so every shipped guard stays native


# ----------------------------------------------------------------------------- v0.2.4 K6W
# K6-A routed-MoE kernel installed as the OUTERMOST wrapper of `routes.forward`.
#
# WHY OUTERMOST (this is the whole point of K6W)
# The shipped runtime stacks two *guarded* wrappers over routes.forward, in this order
# (plugin.register, plugin.py:259-265): partition4.install() then grouped.install().
#   * partition4 captures `orig = R.forward` BEFORE it wraps and routes every internal call
#     through `_Proxy(R, orig)` (partition4.py:29-37, 48, 65). Its 4-row partition calls and its
#     `t <= 4` fallback therefore hit the function that was `routes.forward` at install time.
#   * grouped captures `original = R.forward` (which is partition4's forward), and its startup
#     guard compares `P.forward(...)` == that original against `grouped_plan.forward(...)` == the
#     native expert-major plan (grouped.py:56-65, 91-115). prime_smaller re-runs the same
#     comparison for rows 1..4 with cloned eager prefixes (grouped.py:74-85).
# Putting the K6 branch INSIDE routes.forward (the w7/k6a overlay) makes the *reference* side of
# both guards K6 while the *candidate* side stays native, so every startup guard fails
# (A06: "RuntimeError: Grouped MoE bit mismatch: 24" at layer L=1). Wrapping R.forward after both
# installs leaves routes.forward byte-identical to shipped, so both guards compare native against
# native and pass.
#
# NUMERICS (approved class): K6-A is row-invariant (each row's output == what the kernel produces
# for that row alone), while the shipped decode route is not: a row's bits depend on the launch
# bucket M (bucket(T) = 1/2/4/8) and on its position inside its expert group. Measured on the
# shipped module by the A05d parity gate: 0/2, 0/3, 0/4 and 0/8 rows equal to row-alone at
# uniform routing (w7/k6a-build/evidence/parity/run-20261113T005200/parity.json). K6 at T=2..4 is
# therefore ROUNDING-CHANGING versus shipped, not bit-identical -- user-approved for v0.2.4.
# Default OFF: with HTR_MOE_K6 unset no wrapper is installed and routing is byte-identical.
K6W_ON = os.environ.get('HTR_MOE_K6') == '1'
K6W_SHARED = os.environ.get('HTR_MOE_K6_SHARED') == '1'
_K6W_SCR = {}


def _k6w_rows(spec):
    out = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition('-')
        out.update(range(int(lo), int(hi or lo) + 1))
    if not out or min(out) < 1 or max(out) > 8:
        raise ValueError(f'HTR_MOE_K6_ROWS={spec!r}: row counts must land inside 1..8')
    return frozenset(out)


K6W_ROWS = _k6w_rows(os.environ.get('HTR_MOE_K6_ROWS', '1-8'))


def _k6w_bucket(T):
    """K6 row bucket: k6_plan SPLITs a group larger than M, so the bucket stops at 4."""
    return 1 if T == 1 else 2 if T == 2 else 4


def _k6w_scratch(dev):
    """G = T*topk groups (+ ceil(T/M) shared groups in phase B); part = G*ns*M*m f32, which is
    G*M*5120 for both the gate|up (ns 4, m 1280) and the down (ns 2, m 2560) launch."""
    s = _K6W_SCR.get(str(dev))
    if s is None:
        a, m = 8 * 10, 4
        g = a + (-(-8 // m) if K6W_SHARED else 0)
        f = dict(dtype=torch.float32, device=dev)
        s = _K6W_SCR[str(dev)] = dict(
            gu=torch.empty(g * m * 2 * 640, **f), dn=torch.empty(g * m * 2560, **f),
            part=torch.empty(g * m * 5120, **f),
            cnt_gu=torch.zeros(g * 10, dtype=torch.int32, device=dev), cnt_dn=torch.zeros(g * 20, dtype=torch.int32, device=dev),
            cnt_c2=torch.zeros(20, dtype=torch.int32, device=dev))
    return s


def decode_route_k6(ar, x, ids32, w32):
    """K6-A: plan (1 WG, on device) + gate|up + down-with-fused-combine = 3 launches, like E29.

    T <= 8; x bf16 [T,2560] contiguous, ids32 int32 [T,10] contiguous (-1 = no expert),
    w32 f32 [T,10] contiguous. Shared expert stays on its own K5 path (phase A).
    """
    from .kernels import k6_ext
    from . import routes as R
    T = x.shape[0]
    assert 1 <= T <= 8 and T in K6W_ROWS and ids32.shape == (T, R.TOPK) and w32.shape == (T, R.TOPK)
    E, s, M = k6_ext(), _k6w_scratch(x.device), _k6w_bucket(T)
    A_, nsh = T * R.TOPK, (-(-T // M) if K6W_SHARED else 0)
    # phase A passes x twice: k6_plan never reads x/wsg when nsh == 0
    ge, gr, gn, slot, ng, sg = E.k6_plan(ids32, M, int(ar.get('sh_g', -1)), nsh, x, ar.get('wsg', x))
    kf = ar['_kf']
    sp = R.spadk(ar)
    E.k6_gemv(kf, M, ar['qw'], ar['qoff'], ar['kexp'], ar['scales'], ar['su'], ar['sv'], ge, gr, gn, x, s['part'],
              s['cnt_gu'], s['gu'], R.cbscale(), *sp, R.FENCE_RA)
    return E.k6_gemv_dn_fused(kf, M, ar['qw'], ar['qoff'], ar['kexp'], ar['scales'], ar['su'], ar['sv'], ge, gr, gn, s['gu'],
                              s['part'], s['cnt_dn'], s['dn'], R.cbscale(), *sp, R.FENCE_RA, slot, w32, ng, s['cnt_c2'], sg,
                              A_)   # sh_g = A: first shared group (== the group count in phase A, so never hit)


def register_k6w():
    """Install K6 as the OUTERMOST R.forward wrapper. Call LAST, after partition4 + grouped.

    Only direct model calls with T in HTR_MOE_K6_ROWS reach the K6 route; every call the shipped
    guards make internally goes through the proxies they captured, so it stays native.
    """
    if not K6W_ON:
        return None                     # default off: R.forward is exactly grouped's forward
    from . import routes as R
    if getattr(R, '_htr_k6w', False):
        return R.forward
    if not getattr(R, '_htr_grouped', False):
        # not fatal (an env that disables grouped), but then the guard-vs-native argument does not
        # hold and the caller should know which stack it got
        _log('HTR_MOE_K6W_NO_GROUPED', install=os.environ.get('HTR_MOE_GROUPED', '1'))
    inner = R.forward

    def forward(ar, x, ids, w):
        if x.shape[0] in K6W_ROWS and ar.get('_kf') == 4:
            return decode_route_k6(ar, x, ids.to(torch.int32).contiguous(), w.to(torch.float32).contiguous())
        return inner(ar, x, ids, w)

    forward._htr_orig = inner
    forward._htr_k6w = True
    R.forward = forward
    R._htr_k6w = True
    _log('HTR_MOE_K6W_ACTIVE', rows=sorted(K6W_ROWS), inner=getattr(inner, '__name__', str(inner)),
         partitioned=bool(getattr(R, '_htr_partition4', False)), grouped=bool(getattr(R, '_htr_grouped', False)),
         shared=K6W_SHARED)
    return forward