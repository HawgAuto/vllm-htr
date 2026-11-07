# SPDX-License-Identifier: Apache-2.0
"""Inactive sparse slots must not propagate NaN/Inf through zero PV weights.
The old zero-initialized fixtures could not detect reads of poisoned slot zero.
"""
import pytest
import torch
from vllm import _custom_ops  # noqa: F401
from vllm.platforms import current_platform
from vllm.v1.attention.ops import rocm_octave as sq

pytestmark=pytest.mark.skipif(not current_platform.is_rocm(),reason='ROCm native kernel')


@pytest.mark.parametrize('topk',[97,2051])
@pytest.mark.parametrize('cache_dtype',['octave_k3v4','octave_k3v3','octave_k3v3_compact'])
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('window',[False,True])
@pytest.mark.parametrize('poison',[float('nan'),float('inf')],ids=['nan','inf'])
def test_sparse_masked_slot_scale_is_irrelevant(cache_dtype,dtype,window,poison,topk):
    torch.manual_seed(901)
    device='cuda';n=80;hkv=2;hq=12;head=256;block=16;splits=8
    fmt=sq.OctaveFormat.from_cache_dtype(cache_dtype,head,64)
    cache=torch.zeros(8,hkv,block,fmt.slot_bytes,dtype=torch.uint8,device=device)
    # No selected logical position maps to physical block zero.
    bt=torch.arange(2,7,dtype=torch.int32,device=device).view(1,-1)
    positions=torch.arange(n,device=device)
    slots=bt[0,positions//block].long()*block+positions%block
    k=torch.randn(n,hkv,head,device=device).to(dtype)
    v=torch.randn_like(k);signs=sq.sign_bits(head).to(device)
    torch.ops._C.octave_cache_store(k,v,cache,slots,signs,signs,fmt.kernel_code)
    def table(entries):
        return (torch.zeros(entries,hkv,2*head,dtype=dtype,device=device),
                torch.full((entries,),-1,dtype=torch.int32,device=device),
                torch.zeros(entries,dtype=torch.int32,device=device))
    wt,wtag,wlock=table(256);st,stag,slock=table(128)
    torch.ops._C.octave_window_store(k,v,slots,positions.int(),wt,wtag,wlock,st,stag,slock)
    qpos=torch.tensor([68,79,15],dtype=torch.int32,device=device)
    idx=torch.full((3,topk),-1,dtype=torch.int32,device=device)
    for i,p in enumerate([68,79,15]):idx[i,:p+1]=torch.arange(p+1,device=device)
    req=torch.zeros(3,dtype=torch.int32,device=device)
    q=torch.randn(3,hq,head,device=device).to(dtype)
    mid=torch.empty(3,hq,splits,head+2,dtype=torch.float32,device=device)
    window_args=(qpos,wt,wtag,st,stag) if window else ()
    def run():
        out=torch.full_like(q,float('nan'));mid.fill_(float('nan'))
        torch.ops._C.octave_decode_sparse(out,q,cache,bt,req,idx,mid,signs,signs,head**-.5,splits,fmt.kernel_code,*window_args)
        torch.cuda.synchronize()
        return out
    reference=run()
    assert torch.isfinite(reference).all() and reference.abs().sum()>0
    poison_bytes=torch.full((fmt.num_scales,),poison,dtype=torch.float16,device=device).view(torch.uint8)
    cache[0,:,0,fmt.off_scales:fmt.off_scales+2*fmt.num_scales]=poison_bytes
    actual=run()
    assert torch.isfinite(actual).all(), 'masked slot-zero NaN/Inf scale leaked through zero PV probability'
    assert torch.equal(actual.view(torch.int16),reference.view(torch.int16))
