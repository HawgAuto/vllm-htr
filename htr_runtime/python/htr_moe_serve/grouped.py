"""Exact grouped MoE with native parity checks and guarded fallback."""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
from . import grouped_plan as BP, dispatch as D

_CHECKED=set()
_SEEN=set()
_DEVICES=set()


def log(tag,**values):
    print(tag+' '+json.dumps({'pid':os.getpid(),**values},sort_keys=True),file=sys.stderr,flush=True)


def verify_sources():
    path=Path(__file__).with_name('grouped-bindings.json')
    obj=json.loads(path.read_text())
    if str(Path(__file__).resolve()) not in obj['files']:raise RuntimeError('Grouped MoE bridge missing source binding')
    for p,h in obj['files'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise RuntimeError('Grouped MoE source mismatch: '+p)
    return {'binding_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bound_files':len(obj['files'])}


def verify_device(x):
    if x.device.type!='cuda' or not torch.version.hip:raise RuntimeError('Grouped MoE requires gfx1151 HIP')
    key=str(x.device)
    if key not in _DEVICES:
        if torch.cuda.get_device_properties(x.device).gcnArchName.split(':')[0]!='gfx1151':raise RuntimeError('Grouped MoE requires gfx1151 HIP')
        _DEVICES.add(key)


def metadata(ar,x,ids,w):
    T=x.shape[0]
    if (x.ndim!=2 or x.shape[1]!=2560 or x.dtype!=torch.bfloat16 or not x.is_contiguous()
            or ids.shape!=(T,10) or ids.dtype not in (torch.int32,torch.int64)
            or w.shape!=ids.shape or not w.dtype.is_floating_point or ids.device!=x.device or w.device!=x.device
            or type(ar.get('_meta',{}).get('L')) is not int or not 0<=ar['_meta']['L']<48):
        raise ValueError('Grouped MoE metadata contract')
    verify_device(x)


class Proxy:
    def __init__(self,R,original):self._R=R;self.forward=original
    def __getattr__(self,k):return getattr(self._R,k)


def selfcheck(P,ar,x,ids,w,y,source_T=None):
    live=(ids>=0).any(dim=1);n_live=int(live.sum().item());zero_rows=x.shape[0]-n_live
    if not bool(torch.isfinite(y).all()):raise RuntimeError('Grouped MoE nonfinite candidate')
    repeat=None
    if n_live:
        a=P.forward(ar,x,ids,w).clone();b=P.forward(ar,x,ids,w).clone()
        # Empty router slots have exact empty-sum semantics. Native tails with
        # no groups can be uninitialized; never label these rows native parity.
        a=torch.where(live[:,None],a,torch.zeros_like(a));b=torch.where(live[:,None],b,torch.zeros_like(b))
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):raise RuntimeError('Grouped MoE nonfinite native reference')
        repeat=int((a.view(torch.int16)!=b.view(torch.int16)).sum().item())
        if repeat:raise RuntimeError('Grouped MoE native reference repeat mismatch')
    else:a=torch.zeros_like(y)
    bad=int((a.view(torch.int16)!=y.view(torch.int16)).sum().item())
    if bad:raise RuntimeError('Grouped MoE bit mismatch: '+str(bad))
    log('HTR_MOE_GROUPED_SELFCHECK' if n_live else 'HTR_MOE_GROUPED_EMPTYCHECK',
        L=ar['_meta']['L'],T=x.shape[0],live_rows=n_live,zero_rows=zero_rows,
        reference_repeat_mismatches=repeat,mismatches=bad,finite=True,oracle='native_live_plus_semantic_empty',
        source_T=x.shape[0] if source_T is None else source_T,
        qualification_input='current_eager' if source_T is None else 'eager_live_prefix')
    return n_live


def prime_smaller(P,ar,x,ids,w):
    # Capture warmup may route no experts at smaller shapes. Qualify those
    # shapes with cloned actual eager prefixes, never with fabricated IDs.
    # Run before the returned main result so probe scratch cannot clobber it.
    T=x.shape[0]
    for rows in sorted(D.CANDIDATE_ROWS):
        key=(id(ar),ar['_meta']['L'],rows,str(x.device))
        if rows>=T or key in _CHECKED:continue
        if not bool((ids[:rows]>=0).any()):continue
        xx=x[:rows].clone().contiguous();ii=ids[:rows].clone().contiguous();ww=w[:rows].clone().contiguous()
        yy=D.forward(P,BP,ar,xx,ii,ww)
        if selfcheck(P,ar,xx,ii,ww,yy,source_T=T):_CHECKED.add(key)


def install(R):
    if getattr(R,'_htr_grouped',False):return R.forward
    proof=verify_sources()
    if not getattr(R,'_htr_partition4',False):raise RuntimeError('Grouped MoE requires installed partition4')
    original=R.forward;P=Proxy(R,original)
    def forward(ar,x,ids,w):
        T=x.shape[0]
        # Some backends warm only smaller live batches before capture, whose
        # dummy router rows are padding. Qualify admitted shapes from actual
        # larger eager prefixes before returning the unchanged native result.
        if (T>max(D.CANDIDATE_ROWS) and ar['_kf']==4
                and not torch.cuda.is_current_stream_capturing()
                and (P.BPS_GU,P.BPS_DN,P.FENCE_RA,P.FUSE,P.V2_MIN,P._htr_partition4)==(5,1,1,True,48,True)):
            metadata(ar,x,ids,w)
            prime_smaller(P,ar,x,ids,w)
        if T not in D.CANDIDATE_ROWS or ar['_kf']!=4:return original(ar,x,ids,w)
        if (P.BPS_GU,P.BPS_DN,P.FENCE_RA,P.FUSE,P.V2_MIN,P._htr_partition4)!=(5,1,1,True,48,True):
            raise RuntimeError('Grouped MoE split/partition4 contract')
        metadata(ar,x,ids,w)
        key=(id(ar),ar['_meta']['L'],T,str(x.device));capturing=torch.cuda.is_current_stream_capturing()
        if key not in _CHECKED:
            if capturing:raise RuntimeError('Grouped MoE unchecked capture')
            if not bool((((ids>=0)&(ids<512))|(ids==-1)).all()):raise ValueError('Grouped MoE invalid expert ID')
            if not bool(torch.isfinite(w).all()):raise ValueError('Grouped MoE nonfinite router weight')
            prime_smaller(P,ar,x,ids,w)
        y=D.forward(P,BP,ar,x,ids,w)
        if key not in _CHECKED:
            if selfcheck(P,ar,x,ids,w,y):_CHECKED.add(key)
        seen=(T,capturing)
        if key in _CHECKED and seen not in _SEEN:
            _SEEN.add(seen);log('HTR_MOE_GROUPED_ROUTE',T=T,K=4,capturing=capturing,eager_checked=True)
        return y
    forward._htr_orig=original;R.forward=forward;R._htr_grouped=True
    log('HTR_MOE_GROUPED_ACTIVE',rows=sorted(D.CANDIDATE_ROWS),**proof)
    return forward


def register():
    v=os.environ.get('HTR_MOE_GROUPED','1')
    if v not in ('0','1'):raise ValueError('HTR_MOE_GROUPED must be 0 or 1')
    if v=='0':return
    from . import routes as R
    install(R)
