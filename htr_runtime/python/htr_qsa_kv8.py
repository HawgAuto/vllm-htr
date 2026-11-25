"""Experimental main-KV INT8 serving bridge; explicit owner opt-in only.
BF16/indexer/recurrent paths are never routed here. TAU arithmetic is checked
against the qualified INT8 sparse reader on the first admitted calls.
"""
import atexit,ctypes,hashlib,json,os,pathlib,sys
import torch
from htr_kv8 import cache_views
from htr_qsa_kv8_ops import attention_int8 as _split,write_int8 as _write
_STATE={'writes':0,'split_calls':0,'tau_calls':0,'dec_union_calls':0,'selfchecks':0}
DECODE_UNION=os.environ.get('HTR_QSA_DECODE_UNION','0')=='1'
DECODE_UNION_MAX_ROWS=int(os.environ.get('HTR_QSA_DECODE_UNION_MAX_ROWS','256'))
_DEC_MOD=None; _DEC_SCRATCH={}
_SEEN=set(); _MODULE=None; _ARCH={}
TAU_V5_I8=os.environ.get('HTR_QSA_TAU_V5_I8','0')=='1'
TAU_V4=(not TAU_V5_I8) and os.environ.get('HTR_QSA_TAU_V4','0')=='1'
TAU_V3=(not TAU_V5_I8) and (not TAU_V4) and os.environ.get('HTR_QSA_TAU_V3','0')=='1'
HSACO_SHA256=None if (TAU_V5_I8 or TAU_V4) else ('0256a9ee6bca58221ac7ddb32860c3d10d41f48299a645971dd8ea3934f9087e' if TAU_V3 else 'f5fac9f0c0424d1b4856e24431a50ca6b45d1011ac884baf31d49b95362f5def')
TAU_FN='qsa_tau_v5_kv8_i8' if TAU_V5_I8 else ('qsa_tau_v4_kv8' if TAU_V4 else 'qsa_tau_kv8')

def _log(tag,**fields):
    print(tag+' '+json.dumps({'pid':os.getpid(),**fields},sort_keys=True),file=sys.stderr,flush=True)
atexit.register(lambda:_log('HTR_KV8_FINAL',**_STATE))

def write(cache,key,value,slots):
    _write(cache,key,value,slots)
    _STATE['writes']+=1
    identity=(tuple(cache.shape),tuple(cache.stride()),str(cache.dtype))
    if identity not in _SEEN:
        _SEEN.add(identity)
        _log('HTR_KV8_CACHE',shape=list(cache.shape),stride=list(cache.stride()),dtype=str(cache.dtype),payload_bytes=cache.numel()*cache.element_size(),storage_bytes=cache.untyped_storage().nbytes(),inline_fp32_scales=True)

def _module():
    global _MODULE
    if _MODULE is None:
        import direct_hip_module_v1 as hm
        sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
        assert sha(hm.__file__)=='6c2b7fd9a715a144fc3e2cde58525418f229073591905b060a68815c8100780d'
        path=pathlib.Path(os.environ.get('HTR_KV8_TAU_HSACO',os.path.join(os.environ.get('HTR_NATIVE_DIR','/opt/vllm-htr/native-bin'),'qsa_tau_v5_kv8_i8.hsaco' if TAU_V5_I8 else ('qsa_tau_v4_kv8.hsaco' if TAU_V4 else ('qsa_tau_kv8_v3.hsaco' if TAU_V3 else 'qsa_tau_kv8.hsaco')))))
        assert os.environ.get('HTR_ALLOW_OVERLAY')=='1' or HSACO_SHA256 is None or sha(path)==HSACO_SHA256,'INT8 native binary mismatch'
        module=hm.Module(str(path)); module.fn(TAU_FN)
        libpath=os.path.realpath(hm.lib()._name)
        maps={os.path.realpath(l.split()[-1]) for l in pathlib.Path('/proc/self/maps').read_text().splitlines() if 'libamdhip64' in l}
        assert maps=={libpath},'multiple HIP runtimes'
        _MODULE=(hm,module)
        _log('HTR_KV8_TAU_LOADED',sha256=HSACO_SHA256,hip_lib=libpath,attrs=module.attrs(TAU_FN))
    return _MODULE

def admitted(q,k,v,idx,bt,t2r,out):
    n=q.shape[0] if q.ndim==3 else 0
    if not(q.is_cuda and 257<=n<=12288 and q.shape[1:]==(24,256) and q.dtype==torch.bfloat16):return False
    if torch.cuda.is_current_stream_capturing():return False
    if not(k.shape==v.shape and k.shape[2:]==(2,256) and k.shape[1]%16==0 and k.dtype==torch.int8 and v.dtype==torch.int8):return False
    if not(k.stride()[1:]==v.stride()[1:]==(1040,520,1) and k.stride(0)==v.stride(0) and k.stride(0)>=k.shape[1]*1040):return False
    if not(idx.shape==(n,2051) and idx.stride()==(2051,1) and bt.ndim==2 and all(bt.shape) and bt.stride()==(bt.shape[1],1) and t2r.shape==(n,) and t2r.stride()==(1,)):return False
    if not(idx.dtype==bt.dtype==t2r.dtype==torch.int32):return False
    if not(out.shape==q.shape and out.dtype==q.dtype and q.stride(2)==out.stride(2)==1):return False
    if any(s%8 for s in [q.stride(0),q.stride(1),out.stride(0),out.stride(1)]):return False
    if q.data_ptr()%16 or out.data_ptr()%16 or k.data_ptr()%4 or v.data_ptr()%4:return False
    if any(t.device!=q.device for t in [k,v,idx,bt,t2r,out]):return False
    device=q.device.index
    if device not in _ARCH:_ARCH[device]=torch.cuda.get_device_properties(device).gcnArchName.split(':')[0]
    return _ARCH[device]=='gfx1151'

def _launch(q,k,v,idx,bt,t2r,out):
    hm,module=_module(); I64,I32=ctypes.c_int64,ctypes.c_int32
    module.launch(TAU_FN,(q.shape[0],2,1) if (TAU_V5_I8 or TAU_V4) else ((q.shape[0]+3)//4,2,1),(128,1,1) if (TAU_V5_I8 or TAU_V4) else (384,1,1),
        [hm.ptr(q),hm.ptr(k),hm.ptr(v),hm.ptr(idx),hm.ptr(bt),hm.ptr(t2r),hm.ptr(out),ctypes.c_void_p(0),ctypes.c_void_p(0),
         *[I64(x) for x in [q.stride(0),q.stride(1),k.stride(0),k.stride(1),k.stride(2),v.stride(0),v.stride(1),v.stride(2),out.stride(0),out.stride(1),idx.stride(0),bt.stride(0)]],
         *[I32(x) for x in [q.shape[0],k.shape[0],k.shape[1],bt.shape[1],bt.shape[0],2]]])


def _dec_module():
    global _DEC_MOD
    if _DEC_MOD is None:
        import direct_hip_module_v1 as hm
        path = os.environ.get('HTR_KV8_DECODE_UNION_HSACO')
        if not path:
            for d in (os.environ.get('HTR_NATIVE_DIR', '/opt/vllm-htr/native-bin'), '/opt/vllm-htr/native-proto'):
                cand = os.path.join(d, 'qsa_decode_union_kv8.hsaco')
                if os.path.isfile(cand):
                    path = cand; break
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f'HTR_QSA_DECODE_UNION=1 (KV8) requires qsa_decode_union_kv8.hsaco (checked {path})')
        mod = hm.Module(path); mod.fn('qsa_decode_union_kv8')
        _DEC_MOD = (hm, mod)
    return _DEC_MOD

def _dec_scratch(dev):
    idx = dev.index
    buf = _DEC_SCRATCH.get(idx)
    if buf is None:
        buf = (
            torch.empty((64 * 48, 256), dtype=torch.float32, device=dev),
            torch.empty((64 * 48,), dtype=torch.float32, device=dev),
            torch.zeros((64 * 2,), dtype=torch.int32, device=dev),
        )
        _DEC_SCRATCH[idx] = buf
    return buf

def _dec_admitted(q, k, v, idx, bt, t2r, out, gate):
    if not DECODE_UNION: return False
    n = q.shape[0] if q.ndim == 3 else 0
    if not (q.is_cuda and 1 <= n <= DECODE_UNION_MAX_ROWS and q.shape[1:] == (24, 256) and q.dtype == torch.bfloat16): return False
    if not (k.shape == v.shape and k.shape[2:] == (2, 256) and k.dtype == torch.int8 and v.dtype == torch.int8): return False
    if not (k.stride()[1:] == v.stride()[1:] == (1040, 520, 1) and k.stride(0) == v.stride(0) and k.stride(0) >= k.shape[1] * 1040): return False
    if not (idx.shape == (n, 2051) and idx.dtype == torch.int32 and idx.stride(1) == 1): return False
    if not (bt.ndim == 2 and all(bt.shape) and bt.dtype == torch.int32 and bt.stride(1) == 1 and t2r.dtype == torch.int32 and t2r.stride(0) == 1): return False
    if out is None or out.shape != q.shape or out.dtype != torch.bfloat16 or out.stride(2) != 1 or q.stride(2) != 1: return False
    if any(s % 8 for s in (q.stride(0), q.stride(1), out.stride(0), out.stride(1))): return False
    if (q.data_ptr() % 16) or (out.data_ptr() % 16) or (k.data_ptr() % 4) or (v.data_ptr() % 4): return False
    if gate is not None and (gate.shape != q.shape or gate.dtype != torch.bfloat16 or gate.stride(2) != 1): return False
    dev = q.device.index
    if dev not in _ARCH: _ARCH[dev] = torch.cuda.get_device_properties(dev).gcnArchName.split(':')[0]
    return _ARCH[dev] == 'gfx1151'

def _dec_launch(q, k, v, idx, bt, t2r, out, gate):
    hm, mod = _dec_module()
    part_o, part_lse, counters = _dec_scratch(q.device)
    rows = int(q.shape[0])
    nprog = (rows + 3) // 4
    nsplits = 16 if nprog <= 2 else (8 if nprog <= 4 else (4 if nprog <= 8 else (2 if nprog <= 16 else 1)))
    I64, I32 = ctypes.c_int64, ctypes.c_int32
    gp = hm.ptr(gate) if gate is not None else ctypes.c_void_p(0)
    sg0 = int(gate.stride(0)) if gate is not None else 0
    sg1 = int(gate.stride(1)) if gate is not None else 0
    with torch.cuda.device(q.device):
        mod.launch('qsa_decode_union_kv8', (nprog, int(k.shape[2]), nsplits), (384, 1, 1),
            [hm.ptr(q), hm.ptr(k), hm.ptr(v), hm.ptr(idx), hm.ptr(bt), hm.ptr(t2r),
             hm.ptr(out), gp, hm.ptr(part_o), hm.ptr(part_lse), hm.ptr(counters),
             *[I64(x) for x in [q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2),
                                v.stride(0), v.stride(1), v.stride(2), out.stride(0), out.stride(1),
                                sg0, sg1, idx.stride(0), bt.stride(0)]],
             *[I32(x) for x in [rows, k.shape[0], k.shape[1], bt.shape[1], bt.shape[0], 2]]])

def attention(q,cache,idx,bt,t2r,out=None,*,gate=None,gate_applied_out=None):
    if cache.dtype!=torch.int8:raise ValueError('INT8 bridge requires packed INT8 main cache')
    k,v,_,_=cache_views(cache,q.shape[-1])
    if out is None:out=torch.empty_like(q)
    if _dec_admitted(q,k,v,idx,bt,t2r,out,gate):
        _dec_launch(q,k,v,idx,bt,t2r,out,gate)
        _STATE['dec_union_calls']+=1
        if gate_applied_out is not None and gate is not None:
            gate_applied_out.append(True)
        return out
    if os.environ.get('HTR_KV8_TAU','1')!='1' or not admitted(q,k,v,idx,bt,t2r,out):
        _STATE['split_calls']+=1
        return _split(q,cache,idx,bt,t2r,out)
    _launch(q,k,v,idx,bt,t2r,out); _STATE['tau_calls']+=1
    if _STATE['selfchecks']<int(os.environ.get('HTR_KV8_CHECKS','8')):
        ref=_split(q,cache,idx,bt,t2r,torch.full_like(q,float('nan')))
        torch.cuda.synchronize()
        same=bool(torch.allclose(out.float(),ref.float(),atol=2e-2,rtol=2e-2)) if (TAU_V5_I8 or TAU_V4) else bool(torch.equal(out.view(torch.int16),ref.view(torch.int16)))
        finite=bool(torch.isfinite(out).all())
        _log('HTR_KV8_SELFCHECK',rows=q.shape[0],page=k.shape[1],bitwise=same,finite=finite)
        if not same or not finite:raise RuntimeError('INT8 TAU differs from quantized split-K oracle')
        _STATE['selfchecks']+=1
    return out
