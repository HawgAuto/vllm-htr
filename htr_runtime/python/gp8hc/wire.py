"""GPTQ8-HC-W1: row-major U8 codes, FP16 scales, U8 zeros, g128.

Reconstruct in FP32, then round weights to BF16 before matmul. Partial
final input groups are native (e.g. K320 has 128+128+64). No act-order.
"""
from dataclasses import dataclass
import os
from pathlib import Path
import torch

WIRE = "GPTQ8-HC-W1"
GROUP = 128

@dataclass
class Packet:
    qweight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    group_size: int = GROUP
    wire: str = WIRE


def validate(p, *, values=True):
    if p.wire != WIRE or p.group_size != GROUP:
        raise ValueError("unsupported GPTQ8 wire/group")
    q,s,z = p.qweight,p.scales,p.zeros
    if q.ndim != 2 or min(q.shape) <= 0:
        raise ValueError("qweight must have a nonempty [out,in] shape")
    if (q.dtype,s.dtype,z.dtype) != (torch.uint8,torch.float16,torch.uint8):
        raise TypeError("wire requires U8/FP16/U8")
    shape=(q.shape[0],(q.shape[1]+GROUP-1)//GROUP)
    if tuple(s.shape)!=shape or tuple(z.shape)!=shape:
        raise ValueError("scale/zero shape mismatch")
    if not(q.is_contiguous() and s.is_contiguous() and z.is_contiguous()):
        raise ValueError("wire tensors must be contiguous")
    if q.device != s.device or q.device != z.device:
        raise ValueError("wire tensors must share a device")
    if values and (not bool(torch.isfinite(s).all()) or not bool((s>0).all())):
        raise ValueError("scales must be positive finite FP16")
    return p


def reconstruct(p):
    validate(p)
    n=p.qweight.shape[1]
    group=torch.arange(n,device=p.qweight.device)//GROUP
    return ((p.qweight.float()-p.zeros[:,group].float())*p.scales[:,group].float()).bfloat16()


def payload_bytes(p):
    validate(p)
    return sum(t.numel()*t.element_size() for t in (p.qweight,p.scales,p.zeros))


@torch.no_grad()
def encode(W,H,*,damp=0.01):
    """The frozen htrq.gptq_group arithmetic, with code/scale retention.

    Input-side and post-wire checks are additive. The oracle remains separate.
    Reject FP16 scale underflow rather than silently change the old algorithm.
    """
    if W.ndim!=2 or min(W.shape)<=0 or H.shape!=(W.shape[1],W.shape[1]):
        raise ValueError("expected W[out,in], H[in,in]")
    if W.device!=H.device or not W.is_floating_point() or not H.is_floating_point():
        raise TypeError("W/H must be floating point on the same device")
    if not bool(torch.isfinite(W).all()) or not bool(torch.isfinite(H).all()):
        raise ValueError("nonfinite producer inputs")
    if damp!=0.01:
        raise ValueError("qualified recipe uses damping 0.01")
    W=W.float().clone();H=H.clone().double()
    dead=torch.diag(H)==0
    H[dead,dead]=1.0;W[:,dead]=0
    H+=damp*torch.diag(H).mean()*torch.eye(H.shape[0],device=H.device,dtype=H.dtype)
    U=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)),upper=True).float()
    m,n=W.shape
    codes=torch.empty(m,n,dtype=torch.uint8,device=W.device)
    scales=torch.empty(m,(n+GROUP-1)//GROUP,dtype=torch.float16,device=W.device)
    zeros=torch.empty_like(scales,dtype=torch.uint8)
    for i1 in range(0,n,GROUP):
        i2=min(i1+GROUP,n)
        W1=W[:,i1:i2].clone();Err1=torch.zeros_like(W1);U1=U[i1:i2,i1:i2]
        lo=W1.min(dim=1,keepdim=True).values.clamp(max=0)
        hi=W1.max(dim=1,keepdim=True).values.clamp(min=0)
        sc=((hi-lo)/255).clamp(min=1e-8).half().float()
        if not bool(torch.isfinite(sc).all()) or not bool((sc>0).all()):
            raise ValueError("legacy FP16 scale underflow/overflow; no wire emitted")
        z=torch.round(-lo/sc).clamp(0,255)
        sc,z=sc[:,0],z[:,0]
        scales[:,i1//GROUP]=sc.half();zeros[:,i1//GROUP]=z.to(torch.uint8)
        for i in range(i2-i1):
            v=W1[:,i]
            code=torch.clamp(torch.round(v/sc)+z,0,255)
            q=(code-z)*sc
            codes[:,i1+i]=code.to(torch.uint8)
            err=(v-q)/U1[i,i]
            W1[:,i:]-=err[:,None]*U1[i,i:][None,:]
            Err1[:,i]=err
        W[:,i2:]-=Err1@U[i1:i2,i2:]
    return validate(Packet(codes,scales,zeros))


def save(path,p):
    from safetensors.torch import save_file
    validate(p)
    path=Path(path);tmp=path.with_name(path.name+f'.partial.{os.getpid()}')
    if path.exists() or tmp.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        save_file({k:getattr(p,k).detach().cpu().contiguous() for k in ('qweight','scales','zeros')},str(tmp),metadata={'wire':WIRE,'group_size':str(GROUP),'weight_dtype':'BF16'})
        os.link(tmp,path)  # atomic non-overwrite publication
    finally:
        if tmp.exists():tmp.unlink()


def load(path,device='cpu'):
    from safetensors import safe_open
    with safe_open(str(path),framework='pt',device=device) as f:
        md=f.metadata()
        if md!={'wire':WIRE,'group_size':str(GROUP),'weight_dtype':'BF16'} or set(f.keys())!={'qweight','scales','zeros'}:
            raise ValueError("invalid serialized wire metadata/tensors")
        return validate(Packet(*(f.get_tensor(k) for k in ('qweight','scales','zeros'))))
