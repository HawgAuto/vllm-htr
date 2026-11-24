"""Experimental main-KV INT4 serving bridge; explicit owner opt-in only.
BF16/indexer/recurrent paths are never routed here. TAU arithmetic is checked
against the qualified INT4 sparse reader on the first admitted calls.
"""
import atexit,ctypes,hashlib,json,os,pathlib,sys
import torch
from htr_kv4 import cache_views
from htr_qsa_kv4_ops import attention_rotated as _split,write_int4 as _write,rotate_query,finish_output
_STATE={'writes':0,'split_calls':0,'tau_calls':0,'dec_union_calls':0,'selfchecks':0}
DECODE_UNION=os.environ.get('HTR_QSA_DECODE_UNION','0')=='1'
DECODE_UNION_MAX_ROWS=int(os.environ.get('HTR_QSA_DECODE_UNION_MAX_ROWS','256'))
_DEC_MOD=None; _DEC_SCRATCH={}; _DEC_SIGNS={}
_SEEN=set(); _MODULE=None; _ARCH={}
TAU_V3=os.environ.get('HTR_QSA_TAU_V3','0')=='1'
HSACO_SHA256='0c66077243566f45f94ee353b975348244b8ea5581416c3fa5a2831f074d0743' if TAU_V3 else 'cf3597834cae448fb6f7f15b827152f5847fd76b130140a7c924ef92ba98ba69'

def _log(tag,**fields):
    print(tag+' '+json.dumps({'pid':os.getpid(),**fields},sort_keys=True),file=sys.stderr,flush=True)
atexit.register(lambda:_log('HTR_KV4_FINAL',**_STATE))

def write(cache,key,value,slots):
    _write(cache,key,value,slots)
    _STATE['writes']+=1
    identity=(tuple(cache.shape),tuple(cache.stride()),str(cache.dtype))
    if identity not in _SEEN:
        _SEEN.add(identity)
        _log('HTR_KV4_CACHE',shape=list(cache.shape),stride=list(cache.stride()),dtype=str(cache.dtype),payload_bytes=cache.numel()*cache.element_size(),storage_bytes=cache.untyped_storage().nbytes(),inline_fp32_scales=True)

def _module():
    global _MODULE
    if _MODULE is None:
        import direct_hip_module_v1 as hm
        sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
        assert sha(hm.__file__)=='6c2b7fd9a715a144fc3e2cde58525418f229073591905b060a68815c8100780d'
        path=pathlib.Path(os.environ.get('HTR_KV4_TAU_HSACO',os.path.join(os.environ.get('HTR_NATIVE_DIR','/opt/vllm-htr/native-bin'),'qsa_tau_kv4_v3.hsaco' if TAU_V3 else 'qsa_tau_kv4.hsaco')))
        assert sha(path)==HSACO_SHA256,'INT4 native binary mismatch'
        module=hm.Module(str(path)); module.fn('qsa_tau_kv4')
        libpath=os.path.realpath(hm.lib()._name)
        maps={os.path.realpath(l.split()[-1]) for l in pathlib.Path('/proc/self/maps').read_text().splitlines() if 'libamdhip64' in l}
        assert maps=={libpath},'multiple HIP runtimes'
        _MODULE=(hm,module)
        _log('HTR_KV4_TAU_LOADED',sha256=HSACO_SHA256,hip_lib=libpath,attrs=module.attrs('qsa_tau_kv4'))
    return _MODULE

def admitted(q,k,v,idx,bt,t2r,out):
    n=q.shape[0] if q.ndim==3 else 0
    if not(q.is_cuda and 257<=n<=12288 and q.shape[1:]==(24,256) and q.dtype==torch.bfloat16):return False
    if torch.cuda.is_current_stream_capturing():return False
    if not(k.shape==v.shape and k.shape[2:]==(2,128) and k.shape[1]%16==0 and k.dtype==torch.uint8 and v.dtype==torch.uint8):return False
    if not(k.stride()[1:]==v.stride()[1:]==(528,264,1) and k.stride(0)==v.stride(0) and k.stride(0)>=k.shape[1]*528):return False
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
    module.launch('qsa_tau_kv4',((q.shape[0]+3)//4,2,1),(384,1,1),
        [hm.ptr(q),hm.ptr(k),hm.ptr(v),hm.ptr(idx),hm.ptr(bt),hm.ptr(t2r),hm.ptr(out),ctypes.c_void_p(0),ctypes.c_void_p(0),
         *[I64(x) for x in [q.stride(0),q.stride(1),k.stride(0),k.stride(1),k.stride(2),v.stride(0),v.stride(1),v.stride(2),out.stride(0),out.stride(1),idx.stride(0),bt.stride(0)]],
         *[I32(x) for x in [q.shape[0],k.shape[0],k.shape[1],bt.shape[1],bt.shape[0],2]]])


def _dec_module():
    global _DEC_MOD
    if _DEC_MOD is None:
        import direct_hip_module_v1 as hm
        path = os.environ.get('HTR_KV4_DECODE_UNION_HSACO')
        if not path:
            for d in (os.environ.get('HTR_NATIVE_DIR', '/opt/vllm-htr/native-bin'), '/opt/vllm-htr/native-proto'):
                cand = os.path.join(d, 'qsa_decode_union_kv4.hsaco')
                if os.path.isfile(cand):
                    path = cand; break
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f'HTR_QSA_DECODE_UNION=1 (KV4) requires qsa_decode_union_kv4.hsaco (checked {path})')
        mod = hm.Module(path); mod.fn('rht256_fwd_bf16'); mod.fn('qsa_decode_union_kv4')
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

def _dec_signs(dev):
    idx = dev.index
    sg = _DEC_SIGNS.get(idx)
    if sg is None:
        from vllm.v1.attention.ops.int4_per_token_head import _get_rht_signs
        sg = _get_rht_signs(256, 0, dev).float().contiguous()
        _DEC_SIGNS[idx] = sg
    return sg

def _dec_admitted(q, k, v, idx, bt, t2r, out, gate):
    if not DECODE_UNION: return False
    n = q.shape[0] if q.ndim == 3 else 0
    if not (q.is_cuda and 1 <= n <= DECODE_UNION_MAX_ROWS and q.shape[1:] == (24, 256) and q.dtype == torch.bfloat16): return False
    if not (k.shape == v.shape and k.shape[2:] == (2, 128) and k.dtype == torch.uint8 and v.dtype == torch.uint8): return False
    if not (k.stride()[1:] == v.stride()[1:] == (528, 264, 1) and k.stride(0) == v.stride(0) and k.stride(0) >= k.shape[1] * 528): return False
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
    signs = _dec_signs(q.device)
    rows = int(q.shape[0])
    nprog = (rows + 3) // 4
    nsplits = 16 if nprog <= 2 else (8 if nprog <= 4 else (4 if nprog <= 8 else (2 if nprog <= 16 else 1)))
    I64, I32 = ctypes.c_int64, ctypes.c_int32
    qr = torch.empty_like(q)
    gp = hm.ptr(gate) if gate is not None else ctypes.c_void_p(0)
    sg0 = int(gate.stride(0)) if gate is not None else 0
    sg1 = int(gate.stride(1)) if gate is not None else 0
    with torch.cuda.device(q.device):
        mod.launch('rht256_fwd_bf16', (rows, 24, 1), (32, 1, 1),
            [hm.ptr(q), hm.ptr(signs), hm.ptr(qr),
             I64(q.stride(0)), I64(q.stride(1)), I64(qr.stride(0)), I64(qr.stride(1))])
        mod.launch('qsa_decode_union_kv4', (nprog, int(k.shape[2]), nsplits), (384, 1, 1),
            [hm.ptr(qr), hm.ptr(k), hm.ptr(v), hm.ptr(idx), hm.ptr(bt), hm.ptr(t2r),
             hm.ptr(out), gp, hm.ptr(signs), hm.ptr(part_o), hm.ptr(part_lse), hm.ptr(counters),
             *[I64(x) for x in [qr.stride(0), qr.stride(1), k.stride(0), k.stride(1), k.stride(2),
                                v.stride(0), v.stride(1), v.stride(2), out.stride(0), out.stride(1),
                                sg0, sg1, idx.stride(0), bt.stride(0)]],
             *[I32(x) for x in [rows, k.shape[0], k.shape[1], bt.shape[1], bt.shape[0], 2]]])

def attention(q,cache,idx,bt,t2r,out=None,*,gate=None,gate_applied_out=None):
    if cache.dtype!=torch.uint8:raise ValueError('INT4 bridge requires packed uint8 main cache')
    k,v,_,_=cache_views(cache,q.shape[-1])
    if out is None:out=torch.empty_like(q)
    if _dec_admitted(q,k,v,idx,bt,t2r,out,gate):
        _dec_launch(q,k,v,idx,bt,t2r,out,gate)
        _STATE['dec_union_calls']+=1
        if gate_applied_out is not None and gate is not None:
            gate_applied_out.append(True)
        return out
    qr=rotate_query(q)
    rotated=torch.empty_like(qr)
    if os.environ.get('HTR_KV4_TAU','1')!='1' or not admitted(qr,k,v,idx,bt,t2r,rotated):
        _STATE['split_calls']+=1
        _split(qr,cache,idx,bt,t2r,rotated)
    else:
        _launch(qr,k,v,idx,bt,t2r,rotated); _STATE['tau_calls']+=1
        if _STATE['selfchecks']<int(os.environ.get('HTR_KV4_CHECKS','8')):
            ref=_split(qr,cache,idx,bt,t2r,torch.full_like(qr,float('nan')))
            torch.cuda.synchronize()
            same=bool(torch.equal(rotated.view(torch.int16),ref.view(torch.int16)))
            finite=bool(torch.isfinite(rotated).all())
            _log('HTR_KV4_SELFCHECK',rows=q.shape[0],page=k.shape[1],bitwise=same,finite=finite,oracle='quantized-BF16-rotated-splitK')
            if not same or not finite:raise RuntimeError('INT4 TAU differs from quantized split-K oracle')
            _STATE['selfchecks']+=1
    return finish_output(rotated,out)
