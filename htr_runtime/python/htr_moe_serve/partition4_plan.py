"""Integer group-parity plan for HTR MoE partition4 (see partition4.py).

For 5..47-token steps (native chunk_route covers 9..47; T >= 48 is v2_route and stays native) the decode kernels run
on 4-row partitions. The plan preserves the native eight-row group-position parity, so native library, weights and
arithmetic are unchanged; only integer planning changes.
"""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _plan(IDS,GE,GR,GN,SLOT,NG,START:tl.constexpr,A:tl.constexpr):
    i=tl.arange(0,64)
    e=tl.load(IDS+START*10+i,i<A,other=-1)
    valid=(i<A)&(e>=0)
    same=(e[:,None]==e[None,:])&valid[:,None]&valid[None,:]
    prior=tl.sum((same&(i[None,:]<i[:,None])).to(tl.int32),1)
    first=tl.min(tl.where(same,i[None,:],64),1)
    n=tl.sum(same.to(tl.int32),1)
    if START==0:
        parity=tl.full((64,),0,tl.int32)
    else:
        earlier=tl.load(IDS+i,i<START*10,other=-1)
        parity=tl.sum(((e[:,None]==earlier[None,:])&(i[None,:]<START*10)&valid[:,None]).to(tl.int32),1)&1
    owner=valid&(prior==0)
    groups=tl.sum(owner.to(tl.int32),0)
    # A fully padded half has no last-arriver. One unused expert-zero group
    # makes native fused combine write exact zero; all router slots stay -1.
    dummy=(groups==0)&(i==0)
    pos=prior^parity
    tl.store(GE+i,tl.where(dummy,0,tl.where(owner,e,-1)),i<A)
    tl.store(GN+i,tl.where(dummy,1,tl.where(owner,n+((n&1)*parity),0)),i<A)
    tl.store(SLOT+i,tl.where(valid,first*4+pos,-1),i<A)
    ranks=tl.cumsum(same.to(tl.int32),1)-1
    for r in tl.static_range(4):
        selected=same&(ranks==(r^parity[:,None]))
        row=tl.max(tl.where(selected,i[None,:]//10,0),1)
        tl.store(GR+i*4+r,tl.where(owner,row,0),i<A)
    tl.store(NG,tl.maximum(groups,1))


def make_plan(ids,start,n):
    if ids.ndim!=2 or ids.shape[1]!=10 or not 5<=ids.shape[0]<=8 or start not in (0,4) or n!=min(4,ids.shape[0]-start):
        raise ValueError('partition extent')
    if ids.dtype!=torch.int32 or not ids.is_contiguous():raise ValueError('routing representation')
    a=n*10;kw=dict(device=ids.device,dtype=torch.int32)
    ge=torch.empty(a,**kw);gr=torch.empty((a,4),**kw);gn=torch.empty(a,**kw);slot=torch.empty(a,**kw);ng=torch.empty(1,**kw)
    _plan[(1,)](ids,ge,gr,gn,slot,ng,start,a,num_warps=4)
    return ge,gr,gn,slot,ng


def apply_plan(R,ar,x,w,plan,trace=False):
    E,s=R.moe2_ext(),R._scratch(x.device)
    ge,gr,gn,slot,ng=plan;sp=R.spadk(ar);kf=ar['_kf'];m=4
    E.moe_gemv(0,kf,m,ar['qw'],ar['qoff'],ar['kexp'],ar['scales'],ar['su'],ar['sv'],ge,gr,gn,x,
        s['part'],s['cnt_gu'],s['gu'],R.BPS_GU,R.cbscale(),*sp,R.FENCE_RA)
    y=E.moe_gemv_dn_fused(kf,m,ar['qw'],ar['qoff'],ar['kexp'],ar['scales'],ar['su'],ar['sv'],ge,gr,gn,
        s['gu'],s['part'],s['cnt_dn'],s['dn'],R.BPS_DN,R.cbscale(),*sp,R.FENCE_RA,slot,w,ng,s['cnt_c2'])
    return (y,plan,s) if trace else y


def make_partition(R):
    def partition4(ar,x,ids,w):
        t=x.shape[0]
        if t<=4 or t>47 or ar['_kf']!=4:return R.forward(ar,x,ids,w)
        ids=ids.to(torch.int32).contiguous();w=w.to(torch.float32).contiguous()
        out=torch.empty_like(x)
        for parent in range(0,t,8):
            end=min(parent+8,t)
            if end-parent<=4:
                out[parent:end].copy_(R.forward(ar,x[parent:end],ids[parent:end],w[parent:end]));continue
            chunk_ids=ids[parent:end]
            for start in [0,4]:
                n=min(4,end-parent-start);i=parent+start;j=i+n
                p=make_plan(chunk_ids,start,n)
                out[i:j].copy_(apply_plan(R,ar,x[i:j],w[i:j],p))
        return out
    return partition4
