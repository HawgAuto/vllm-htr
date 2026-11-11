"""Native HTR defaults. No framework imports at startup in compiler children.

Serving arguments and request sampling remain native vLLM options. Historical
launchers are retained as records, not required to activate this runtime.
"""
import copy
import dataclasses
import enum
import functools
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path('/opt/vllm-htr')
DEFAULT_OVERRIDES = {'served_model_name': ['qwen38-htr']}
CACHE_KEYS = ('XDG_CACHE_HOME', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR',
              'TORCHINDUCTOR_CACHE_DIR', 'AITER_JIT_DIR', 'TVM_FFI_CACHE_DIR')
NATIVE_CONTEXT = 262144


def environment_defaults(env=None, *, root=ROOT, base_defaults=None):
    env = os.environ if env is None else env
    # Record explicit overrides before filling defaults; child engines inherit
    # this provenance so an automatic value can follow the parsed MTP mode.
    if 'HTR_PLE_ASYNC_MAXN_SOURCE' not in env:
        if 'HTR_PLE_ASYNC_MAXN' in env:
            env['HTR_PLE_ASYNC_MAXN_SOURCE'] = 'user'
        elif 'E59_PLE_DEV_MAXN' in env:
            env['HTR_PLE_ASYNC_MAXN'] = env['E59_PLE_DEV_MAXN']
            env['HTR_PLE_ASYNC_MAXN_SOURCE'] = 'legacy-user'
        else:
            env['HTR_PLE_ASYNC_MAXN_SOURCE'] = 'default'
            env['HTR_PLE_ASYNC_MAXN'] = 'auto'
    if base_defaults is None:
        base_defaults = json.loads(Path('/opt/haloloom/qwen38/data/runtime-env.json').read_text())
    defaults = dict(base_defaults)
    defaults.update(json.loads((root/'runtime-env.json').read_text()))
    defaults.update(DIRECT_W8A16='0', VLLM_ALLOW_LONG_MAX_MODEL_LEN='0',
                    VLLM_PLUGINS='htr,htr_moe,gp8hc,q38spec,htr_autotune_pin,htr_q38f')
    for key, value in defaults.items():
        if key.startswith('DIRECT_QSA_UNION_') or key.startswith('DIRECT_QSA_NATIVE_'):
            continue
        env.setdefault(key, str(value))
    # Toolchains consume these paths before prepare_engine() creates them (the
    # aiter JIT build runs while vLLM parses argv), so materialize the roots here
    # and give a cold workspace a usable TMPDIR and cache from the first child on.
    for key in ('TMPDIR',) + CACHE_KEYS:
        value = env.get(key)
        if value:
            try:
                Path(value).mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
    budget=ple_budget(int(env.get('HTR_PLE_ROW_CACHE_BYTES','0')))
    if budget or env.get('HTR_PLE_CACHE_OBSERVE','0')=='1':
        env['DIRECT_PLE_NATIVE_LIB']=str(root/'experimental/ple_row_cache/libple_row_cache.so')
    return env


def resolve_ple_maxn(mtp, env):
    """Resolve the selected async-PLE envelope from native engine settings."""
    if mtp not in (0, 3):
        raise ValueError('HTR PLE defaults require MTP0 or MTP3')
    source = env.get('HTR_PLE_ASYNC_MAXN_SOURCE', 'default')
    value = env.get('HTR_PLE_ASYNC_MAXN', 'auto')
    if source == 'default' or value == 'auto':
        maximum = 32 if mtp == 0 else 16
    else:
        maximum = int(value)
        if maximum < 0:
            raise ValueError('PLE async row limit must be nonnegative')
    env['HTR_PLE_ASYNC_MAXN'] = str(maximum)
    env['E59_PLE_DEV_MAXN'] = str(maximum)
    return maximum


def bind_ple_maxn(maximum, modules=None):
    """Bind parsed policy before fork or in-process engine initialization."""
    if modules is None:
        import sys
        modules = sys.modules
    ple = modules.get('e59_ple_dev')
    if ple is None or ple.MAXN == maximum:
        return
    if ple._B or ple._S['pending'] is not None:
        raise ValueError('Cannot change the PLE row limit after initialization')
    setattr(ple, 'MAXN', maximum)


def canonical(value):
    if dataclasses.is_dataclass(value): return canonical(dataclasses.asdict(value))
    if isinstance(value, enum.Enum): return canonical(value.value)
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k):canonical(v) for k,v in value.items()}
    if isinstance(value, (set,frozenset)): return sorted((canonical(v) for v in value),key=lambda v:json.dumps(v,sort_keys=True))
    if isinstance(value, (tuple,list)): return [canonical(v) for v in value]
    if value is None or isinstance(value, (str,int,float,bool)): return value
    raise TypeError('Unsupported configuration identity type: '+type(value).__name__)


def cache_identity(config, env, revision):
    payload = {'config': canonical(config), 'environment': dict(sorted(env.items())), 'revision': revision}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()[:24]


def ple_budget(value):
    if isinstance(value, bool) or not math.isfinite(value) or value < 0 or int(value) != value:
        raise ValueError('PLE cache bytes must be a finite nonnegative integer')
    if 0 < value < 65536: raise ValueError('nonzero PLE cache requires at least 65536 bytes')
    return int(value)


def parser_defaults(parser):
    # Parsing our recorded command uses native argument types; set_defaults leaves
    # all explicit CLI/config values authoritative, even upstream-default values.
    argv = json.loads((ROOT/'server-argv.json').read_text())[3:]
    defaults = vars(parser.parse_args(argv))
    defaults.update(copy.deepcopy(DEFAULT_OVERRIDES))
    defaults.pop('model_tag', None)
    cc = defaults['compilation_config']
    cc.cudagraph_capture_sizes = None
    cc.max_cudagraph_capture_size = None
    # Do not inject a preferred sampling profile or validate checkpoint sampling.
    parser.set_defaults(**defaults)
    return parser


def apply_yarn(config, *, overrides, factor):
    def update(obj, values):
        for key,value in values.items():
            old = obj.get(key) if isinstance(obj,dict) else getattr(obj,key,None)
            if isinstance(value,dict) and old is not None: update(old,value)
            elif isinstance(obj,dict): obj[key]=copy.deepcopy(value)
            else: setattr(obj,key,copy.deepcopy(value))
    update(config,overrides)
    if str(config.model_type).startswith('dummy_'): return config
    text = config.get_text_config()
    if config.model_type not in ('qwen4_exp','qwen4_exp_text','qwen4_exp_mtp'):
        raise ValueError('HTR automatic YaRN requires Qwen Flash')
    rope = copy.deepcopy(text.rope_parameters)
    rope.update(rope_type='yarn',factor=factor,original_max_position_embeddings=NATIVE_CONTEXT)
    for obj in (config,text):
        obj.max_position_embeddings=int(NATIVE_CONTEXT*factor)
        obj.original_max_position_embeddings=NATIVE_CONTEXT
        obj.rope_parameters=copy.deepcopy(rope)
        obj.rope_scaling=copy.deepcopy(rope)
    return config


def verify_installation():
    for name, digest in json.loads((ROOT/'installed-files.json').read_text()).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError('HTR runtime integrity mismatch: '+name)
    for name, digest in json.loads((ROOT/'native-policy-bindings.json').read_text()).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError('HTR native binding mismatch: '+name)


def prepare_engine(args):
    verify_installation()
    spec = copy.deepcopy(args.speculative_config)
    mtp = 0 if spec is None else spec.get('num_speculative_tokens',3)
    if mtp == 0: spec = None
    if spec is not None and (mtp != 3 or spec.get('method','mtp') != 'mtp'):
        raise ValueError('HTR numerical pins currently implement MTP0 and MTP3')
    args.speculative_config = spec
    context, seqs, chunk = args.max_model_len, args.max_num_seqs, args.max_num_batched_tokens
    for name,value in [('max_model_len',context),('max_num_seqs',seqs),('max_num_batched_tokens',chunk)]:
        if not isinstance(value,int) or value < 1: raise ValueError(name+' must be positive')
    dtype = args.kv_cache_dtype
    os.environ['HTR_GDN_MTP']=str(mtp)
    ple_maxn = resolve_ple_maxn(mtp, os.environ)
    bind_ple_maxn(ple_maxn)
    os.environ['HTR_KV8_ENABLE']='1' if dtype=='int8_per_token_head' else '0'
    os.environ['HTR_KV4_ENABLE']='1' if dtype=='int4_per_token_head' else '0'
    os.environ['HTR_RHT_D256']='triton' if dtype=='int4_per_token_head' else 'pytorch'
    budget = ple_budget(int(os.environ.get('HTR_PLE_ROW_CACHE_BYTES','0')))
    if budget or os.environ.get('HTR_PLE_CACHE_OBSERVE','0')=='1':
        library=ROOT/'experimental/ple_row_cache/libple_row_cache.so'
        if not library.is_file(): raise FileNotFoundError(library)
        os.environ['DIRECT_PLE_NATIVE_LIB']=str(library)
    cc=args.compilation_config
    if cc.cudagraph_capture_sizes is None:
        top=seqs*(mtp+1)
        ladder=[1,2,3,4,5,6,8,10,12,16,20,24,32,40,48,64,80,96,128]
        cc.cudagraph_capture_sizes=sorted({n for n in ladder if n<=top}|{top})
        if seqs==1 and mtp==3: cc.cudagraph_capture_sizes=[1,2,3,4,5]
    if cc.max_cudagraph_capture_size is None:
        cc.max_cudagraph_capture_size=max(cc.cudagraph_capture_sizes)
    # Hash the unresolved native settings, before replacing overrides by a callable.
    config={key:getattr(args,key) for key in (
        'model','tokenizer','dtype','quantization','max_model_len','max_num_seqs',
        'max_num_batched_tokens','kv_cache_dtype','kv_cache_memory_bytes',
        'enable_prefix_caching','compilation_config','speculative_config','hf_overrides',
        'tensor_parallel_size','pipeline_parallel_size','generation_config','override_generation_config')}
    bound_env={key:value for key,value in os.environ.items() if key.startswith(('HTR_','DIRECT_','GP8HC_','E53_','E59_','E92_','Q38SPEC_')) and key not in ('HTR_RUNTIME_CACHE_ROOT','HTR_PLE_ASYNC_MAXN_SOURCE')}
    revision=(ROOT/'NATIVE-POLICY-REVISION').read_text().strip()
    identity=cache_identity(config,bound_env,revision)
    cache_root=Path(os.environ.get('HTR_CACHE_HOME','/workspace/cache'))/('htr-native-'+identity)
    for key in CACHE_KEYS:
        # Explicit locations are preserved. vLLM/Triton also key their own artifacts.
        original=json.loads(Path('/opt/haloloom/qwen38/data/runtime-env.json').read_text()).get(key)
        if key not in os.environ or os.environ[key]==original:
            os.environ[key]=str(cache_root/key.lower())
    os.environ['HTR_RUNTIME_CACHE_ROOT']=str(cache_root)
    cache_root.mkdir(parents=True,exist_ok=True)
    for name in ('tmp','config','profiles'): Path('/workspace',name).mkdir(parents=True,exist_ok=True)
    for key,filename,data in [('DIRECT_MOE_AB_PHASE','moe-ab-phase.json',{'arm':'wmma5','case':'native-serving'}),
        ('DIRECT_PLE_AB_PHASE','ple-ab-phase.json',{'arm':'native','case':'native-serving'}),
        ('DIRECT_NB_PHASE','nb-phase.json',{'nb':9,'case':'native-serving'})]:
        path=Path('/workspace',filename)
        if os.environ.get(key)==str(path) and not path.exists():
            try:
                with path.open('x') as f: json.dump(data,f)
            except FileExistsError: pass
    overrides=copy.deepcopy(args.hf_overrides)
    if context>NATIVE_CONTEXT and isinstance(overrides,dict):
        rope=overrides.get('text_config',{}).get('rope_parameters',{})
        factor=rope.get('factor') if rope.get('rope_type')=='yarn' else float(2**math.ceil(math.log2(context/NATIVE_CONTEXT)))
        if not math.isfinite(factor) or factor<=1 or factor*NATIVE_CONTEXT<context:
            raise ValueError('Requested YaRN factor does not cover context')
        args.hf_overrides=functools.partial(apply_yarn,overrides=overrides,factor=factor)
        if spec is not None: spec.setdefault('max_model_len',context)
    # Envelope is information, not an admission policy.
    if seqs>1 or chunk!=12288 or context>1048576 or dtype.startswith('octave_'):
        print('HTR_QUALIFICATION_ADVISORY '+json.dumps({'qualification_claim':False,'context':context,'seqs':seqs,'chunk':chunk,'kv':dtype}),flush=True)
    print('HTR_RUNTIME_CONFIG '+json.dumps({'profile':identity,'mtp':mtp,'kv_cache_dtype':dtype,'context':context,'max_num_seqs':seqs,'max_num_batched_tokens':chunk,'ple_cache_bytes':budget,'ple_async_maxn':ple_maxn,'ple_async_maxn_source':os.environ.get('HTR_PLE_ASYNC_MAXN_SOURCE','default')}),flush=True)
