"""Opt-in exact FP32 D256 RHT; all other donor domains stay native."""
import os
_ORIGINAL=None
_COUNTS={'forward':0,'inverse':0,'fallback':0}

def eligible(shape,strides,dtype,is_cuda):
    return (is_cuda and len(shape)==3 and shape[-1]==256 and strides[-1]==1
            and dtype=='float32' and all(v>=0 for v in strides))

def selected_mode(env):
    mode=env.get('HTR_RHT_D256','pytorch')
    if mode not in ('pytorch','triton'):raise ValueError('invalid HTR_RHT_D256: '+mode)
    return mode

def rht256(x,inverse=False):
    import torch
    from vllm.v1.attention.ops.int4_per_token_head import _get_rht_signs
    from htr_d256_kernel import _rht256
    if not eligible(tuple(x.shape),tuple(x.stride()),str(x.dtype).removeprefix('torch.'),x.is_cuda):
        raise ValueError('D256 kernel only admits GPU FP32 rank3 with contiguous features')
    out=torch.empty_like(x)
    if x.numel()==0:return out
    signs=_get_rht_signs(256,0,x.device)
    _rht256[(x.shape[0],x.shape[1])](x,signs,out,x.stride(0),x.stride(1),out.stride(0),out.stride(1),INVERSE=inverse,num_warps=1,num_stages=1,enable_fp_fusion=False)
    return out

def _dispatch(x,inverse=False):
    if eligible(tuple(x.shape),tuple(x.stride()),str(x.dtype).removeprefix('torch.'),x.is_cuda):
        key='inverse' if inverse else 'forward';_COUNTS[key]+=1
        return rht256(x,inverse)
    _COUNTS['fallback']+=1
    return _ORIGINAL(x,inverse)

def install():
    global _ORIGINAL
    if selected_mode(os.environ)=='pytorch':return False
    if _ORIGINAL is not None:return True
    import atexit,hashlib,json,pathlib
    from vllm.v1.attention.ops import int4_per_token_head as donor
    _ORIGINAL=donor.single_rht;donor.single_rht=_dispatch
    print('HTR_D256_RHT_INSTALLED '+json.dumps({'mode':'triton','source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'domain':'FP32/rank3/D256','arithmetic':'same ordered eight butterflies'}),flush=True)
    atexit.register(lambda:print('HTR_D256_RHT_COUNTS '+json.dumps(_COUNTS),flush=True))
    return True
