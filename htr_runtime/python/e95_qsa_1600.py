"""Isolated, default-OFF QSA page1600/table164 extension.
Only captured layout and short-context envelope are supported. Legacy selector
and numeric kernels remain byte-preserved. Other cases retain native fallback.
E136: row floor 256 (== DIRECT_QSA_NATIVE_MIN_TOKENS). Below it the union path fell to Triton _attend, whose
NR (row count) is tl.constexpr -> one JIT per distinct short-prompt length (~0.9 s each, E95d chat72 regression).
Rows <256 now keep the unchanged native QSA fallback exactly like the baseline.
"""
import os

def eligible_mtp1600(q,k_cache,v_cache,logical_indices,block_table,token_to_req,out,*,qualified_unique=False,context_len=None):
 if os.environ.get('E95_QSA_MTP1600')!='1' or os.environ.get('DIRECT_QSA_CAP')!='1':return False
 if qualified_unique is not True or not q.is_cuda or type(context_len) is not int:return False
 if q.ndim!=3 or q.shape[1:]!=(24,256) or not 256<=q.shape[0]<=8192 or not q.shape[0]<=context_len<=8192:return False
 if k_cache.shape!=v_cache.shape or k_cache.shape!=(195,1600,2,256):return False
 if k_cache.stride()!=v_cache.stride() or k_cache.stride()!=(21964800,1024,512,1):return False
 if logical_indices.shape!=(q.shape[0],2051) or logical_indices.stride()!=(2051,1):return False
 if block_table.shape!=(1,164) or block_table.stride()!=(164,1):return False
 if token_to_req.shape!=(q.shape[0],) or token_to_req.stride()!=(1,):return False
 if out is None or out.shape!=q.shape or q.stride()!=out.stride() or q.stride()!=(6144,256,1):return False
 import torch
 if not(q.dtype==k_cache.dtype==v_cache.dtype==out.dtype==torch.bfloat16):return False
 if not(logical_indices.dtype==block_table.dtype==token_to_req.dtype==torch.int32):return False
 ts=(q,k_cache,v_cache,logical_indices,block_table,token_to_req,out)
 if not all(t.device==q.device for t in ts):return False
 if not all(t.data_ptr()%16==0 for t in ts):return False
 for t in ts:
  offset=t.storage_offset()
  if offset<0 or (offset+sum((n-1)*s for n,s in zip(t.shape,t.stride()))+1)*t.element_size()>t.untyped_storage().nbytes():return False
 return torch.cuda.get_device_properties(q.device).gcnArchName.split(':')[0]=='gfx1151'
