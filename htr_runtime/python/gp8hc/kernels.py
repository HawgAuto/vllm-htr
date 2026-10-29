"""Manually authored Triton consumers; GPU qualification is independent.

Decode reads packed U8 weights directly with per-g128 affine dequantization.
Large input widths use a bounded FP32 partial-output buffer, never BF16 W.
Prefill dequantizes only the live dot tile. W is rounded to BF16 before dot.

E21 v2 (fork of E18 kernels.py sha 66ff276b): _gemm row count M is a RUNTIME argument (do_not_specialize) instead of tl.constexpr.
E18 v1 JIT-compiled a new prefill kernel for every distinct prompt length (measured +1.43 s TTFT per new length, E21 item 2b).
Per-element math and the K-tile accumulation order (BK) are unchanged. Prefill BM/BN tiles are env-selectable for qualification
(GP8HC_GEMM_BM / GP8HC_GEMM_BN, default 16/32 = v1); BK stays 64 so the reduction order per output element is unchanged.

E21 v3 (fork of v2): decode _gemv put the activation row on grid axis 1, so every row re-read the full packed weight
(M=4 MTP3 verify = 4 x 0.64 GB vs 1.28 GB BF16 -> packed decode slower than BF16 at MTP3, E18 2c). _gemv_mr loads the
q/z/s tile ONCE per program and loops the M rows over it. Per row the tile, w rounding, x load and tl.sum([BN,BK]) are
identical to v2 -> expected bitwise equal (verified by tools/gp8hc_gemv_bench.py). GP8HC_GEMV=v2 restores the v2 decode path.
"""
import os
import torch
import triton as tr
import triton.language as tl

_BM = int(os.environ.get("GP8HC_GEMM_BM", "16"))
_BN = int(os.environ.get("GP8HC_GEMM_BN", "32"))
_NW = int(os.environ.get("GP8HC_GEMM_WARPS", "4"))
_GEMV = os.environ.get("GP8HC_GEMV", "v5")
# v5 (K1c): _gemv_g loads zero/scale ONCE per (row, 128-group) as [BN, G] and broadcasts over the group (v3 gathered them per element:
# 3 loads per weight byte). Per element: same (q-z)*s -> bf16 -> f32, same x; reduction = sum over [BN,G,128] (order may differ from v3's
# [BN,BK] sum -> checked bitwise/maxabs in the sweep). Needs BK % 128 == 0 and K % 128 == 0 (else falls back to v3 _gemv_mr).
# v4 (K1b): decode tile knobs. Default BN=4, BK=1024, warps=4 == v3 exactly. BN only regroups output rows per program
# (per-row math unchanged -> bitwise); BK/warps can change the tl.sum reduction order (checked by maxabs in the sweep).
_GBN = int(os.environ.get("GP8HC_GEMV_BN", "4"))
_GBK = int(os.environ.get("GP8HC_GEMV_BK", "1024"))
_GNW = int(os.environ.get("GP8HC_GEMV_WARPS", "4"))


@tr.jit
def _gemv(X,Q,S,Z,Y,P,K:tl.constexpr,N:tl.constexpr,O:tl.constexpr,M:tl.constexpr,NG:tl.constexpr,
          BK:tl.constexpr,BN:tl.constexpr,SPLIT:tl.constexpr):
    rr=tl.program_id(0)*BN+tl.arange(0,BN)
    row=tl.program_id(1);split=tl.program_id(2)
    kk=split*BK+tl.arange(0,BK)
    q=tl.load(Q+rr[:,None]*K+kk[None,:],(rr[:,None]<N)&(kk[None,:]<K),other=0).to(tl.int32)
    zero=tl.load(Z+rr[:,None]*NG+kk[None,:]//128,(rr[:,None]<N)&(kk[None,:]<K),other=0).to(tl.int32)
    scale=tl.load(S+rr[:,None]*NG+kk[None,:]//128,(rr[:,None]<N)&(kk[None,:]<K),other=1).to(tl.float32)
    w=((q-zero).to(tl.float32)*scale).to(tl.bfloat16).to(tl.float32)
    x=tl.load(X+row*K+kk,kk<K,other=0).to(tl.float32)
    acc=tl.sum(w*x[None,:],1)
    if SPLIT==1:
        tl.store(Y+row*O+rr,acc,rr<O)
    else:
        tl.store(P+(split*M+row)*O+rr,acc,rr<O)


@tr.jit
def _gemv_mr(X,Q,S,Z,Y,P,K:tl.constexpr,N:tl.constexpr,O:tl.constexpr,M:tl.constexpr,NG:tl.constexpr,
          BK:tl.constexpr,BN:tl.constexpr,SPLIT:tl.constexpr):
    rr=tl.program_id(0)*BN+tl.arange(0,BN)
    split=tl.program_id(1)
    kk=split*BK+tl.arange(0,BK)
    q=tl.load(Q+rr[:,None]*K+kk[None,:],(rr[:,None]<N)&(kk[None,:]<K),other=0).to(tl.int32)
    zero=tl.load(Z+rr[:,None]*NG+kk[None,:]//128,(rr[:,None]<N)&(kk[None,:]<K),other=0).to(tl.int32)
    scale=tl.load(S+rr[:,None]*NG+kk[None,:]//128,(rr[:,None]<N)&(kk[None,:]<K),other=1).to(tl.float32)
    w=((q-zero).to(tl.float32)*scale).to(tl.bfloat16).to(tl.float32)
    for row in tl.static_range(M):
        x=tl.load(X+row*K+kk,kk<K,other=0).to(tl.float32)
        acc=tl.sum(w*x[None,:],1)
        if SPLIT==1:
            tl.store(Y+row*O+rr,acc,rr<O)
        else:
            tl.store(P+(split*M+row)*O+rr,acc,rr<O)


@tr.jit
def _gemv_g(X,Q,S,Z,Y,P,K:tl.constexpr,N:tl.constexpr,O:tl.constexpr,M:tl.constexpr,NG:tl.constexpr,
          BK:tl.constexpr,BN:tl.constexpr,SPLIT:tl.constexpr):
    G:tl.constexpr=BK//128
    rr=tl.program_id(0)*BN+tl.arange(0,BN)
    split=tl.program_id(1)
    kk=split*BK+tl.arange(0,BK)
    gg=split*G+tl.arange(0,G)
    rm=rr<N
    q=tl.load(Q+rr[:,None]*K+kk[None,:],rm[:,None]&(kk[None,:]<K),other=0).to(tl.int32)
    zero=tl.load(Z+rr[:,None]*NG+gg[None,:],rm[:,None]&(gg[None,:]<NG),other=0).to(tl.int32)
    scale=tl.load(S+rr[:,None]*NG+gg[None,:],rm[:,None]&(gg[None,:]<NG),other=1).to(tl.float32)
    q3=tl.reshape(q,(BN,G,128))
    w=((q3-zero[:,:,None]).to(tl.float32)*scale[:,:,None]).to(tl.bfloat16).to(tl.float32)
    for row in tl.static_range(M):
        x=tl.load(X+row*K+kk,kk<K,other=0).to(tl.float32)
        acc=tl.sum(tl.sum(w*tl.reshape(x,(1,G,128)),2),1)
        if SPLIT==1:
            tl.store(Y+row*O+rr,acc,rr<O)
        else:
            tl.store(P+(split*M+row)*O+rr,acc,rr<O)


@tr.jit
def _sum_parts(P,Y,TOTAL:tl.constexpr,SPLIT:tl.constexpr,BS:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    sp=tl.arange(0,BS)
    a=tl.load(P+sp[:,None]*TOTAL+i[None,:],(sp[:,None]<SPLIT)&(i[None,:]<TOTAL),other=0)
    tl.store(Y+i,tl.sum(a,0),i<TOTAL)


@tr.jit(do_not_specialize=["M"])
def _gemm(X,Q,S,Z,Y,M,K:tl.constexpr,N:tl.constexpr,O:tl.constexpr,NG:tl.constexpr,
          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rm=tl.program_id(0)*BM+tl.arange(0,BM)
    rn=tl.program_id(1)*BN+tl.arange(0,BN)
    acc=tl.full((BM,BN),0,tl.float32)
    for base in range(tr.cdiv(K,BK)):
        rk=base*BK+tl.arange(0,BK)
        x=tl.load(X+rm[:,None]*K+rk[None,:],(rm[:,None]<M)&(rk[None,:]<K),other=0)
        q=tl.load(Q+rn[None,:]*K+rk[:,None],(rn[None,:]<N)&(rk[:,None]<K),other=0).to(tl.int32)
        z=tl.load(Z+rn[None,:]*NG+rk[:,None]//128,(rn[None,:]<N)&(rk[:,None]<K),other=0).to(tl.int32)
        s=tl.load(S+rn[None,:]*NG+rk[:,None]//128,(rn[None,:]<N)&(rk[:,None]<K),other=1).to(tl.float32)
        w=((q-z).to(tl.float32)*s).to(tl.bfloat16)
        acc=tl.dot(x,w,acc)
    tl.store(Y+rm[:,None]*O+rn[None,:],acc,(rm[:,None]<M)&(rn[None,:]<O))


@tr.jit
def _unpack(Q,S,Z,Y,K:tl.constexpr,N:tl.constexpr,NG:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);r=i//K;k=i%K
    q=tl.load(Q+i,i<N*K,other=0).to(tl.int32)
    z=tl.load(Z+r*NG+k//128,i<N*K,other=0).to(tl.int32)
    s=tl.load(S+r*NG+k//128,i<N*K,other=1).to(tl.float32)
    tl.store(Y+i,((q-z).to(tl.float32)*s).to(tl.bfloat16),i<N*K)


def launch(x,q,s,z,out_features):
    m,k=x.shape;n=q.shape[0];o=out_features;ng=s.shape[1]
    y=torch.empty((m,o),device=x.device,dtype=torch.bfloat16)
    if m==0:return y
    if m<=8:
        bk=(1024 if _GBK==1024 else _GBK) if k>512 else tr.next_power_of_2(k)
        bk=min(bk,tr.next_power_of_2(k))
        split=tr.cdiv(k,bk)
        partial=torch.empty((split,m,o),device=x.device,dtype=torch.float32) if split>1 else y
        if _GEMV=="v2":
            _gemv[(tr.cdiv(o,4),m,split)](x,q,s,z,y,partial,k,n,o,m,ng,bk,4,split,num_warps=4)
        elif _GEMV=="v5" and k%128==0 and bk%128==0:
            _gemv_g[(tr.cdiv(o,_GBN),split)](x,q,s,z,y,partial,k,n,o,m,ng,bk,_GBN,split,num_warps=_GNW)
        else:
            _gemv_mr[(tr.cdiv(o,_GBN),split)](x,q,s,z,y,partial,k,n,o,m,ng,bk,_GBN,split,num_warps=_GNW)
        if split>1:
            _sum_parts[(tr.cdiv(m*o,256),)](partial,y,m*o,split,tr.next_power_of_2(split),256,num_warps=4)
    else:
        _gemm[(tr.cdiv(m,_BM),tr.cdiv(o,_BN))](x,q,s,z,y,m,k,n,o,ng,_BM,_BN,64,num_warps=_NW,num_stages=2)
    return y


def unpack_for_test(q,s,z):
    """Qualification-only reconstruction; never called by the serving op."""
    n,k=q.shape
    y=torch.empty(n,k,device=q.device,dtype=torch.bfloat16)
    _unpack[(tr.cdiv(n*k,256),)](q,s,z,y,k,n,s.shape[1],256,num_warps=4)
    return y
