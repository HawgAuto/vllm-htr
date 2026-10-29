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
            self._checked.add(key)
            if route == 'v2':      # v2 must be checked on >= V2_MIN rows (R.forward on <= 8 rows would take the decode route)
                n = min(x.shape[0], 1024)
                a = R.v2_route(self.ar, x[:n].contiguous(), ids[:n], w[:n]).float()
            else:
                n = min(x.shape[0], R.MAXT)
                a = R.forward(self.ar, x[:n].contiguous(), ids[:n], w[:n]).float()
            b = R.dequant_route(self.ar, x[:n].contiguous(), ids[:n], w[:n]).float()
            rel = float((a - b).norm() / b.norm().clamp_min(1e-30))
            ok = bool(torch.isfinite(a).all()) and rel <= 3e-2
            _log('HTR_MOE_SELFCHECK', L=self.L, route=route, rows=n, rel=rel, ok=ok)
            if not ok:
                raise RuntimeError(f'HTR MoE selfcheck L{self.L} {route}: rel {rel}')

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
