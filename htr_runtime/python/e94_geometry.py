GRAPH_HASH='84011e66a53ce9f7d029f64c3781bd2c2223882850f5104325ca1e82a967b1fd'
QSA_HASH='f399f1b3599beb790cdace6d9100d372df97c0ddf8f4f4668b5ddf3c9d3f111d'
"""Opt-in metadata-only diagnostic: no tensor contents, graph or arithmetic changes."""
import os,json,hashlib,functools
from pathlib import Path
_INSTALLED=False

def register():
 global _INSTALLED
 if os.environ.get('E94_GEOMETRY_PROOF')!='1' or _INSTALLED:return
 import torch
 import vllm.v1.worker.gpu.cudagraph_utils as cg
 import vllm.models.qwen4_exp.common.qsa_cache as qc
 for mod,h in [(cg,GRAPH_HASH),(qc,QSA_HASH)]:
  assert hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()==h,(mod.__file__,h)
 def emit(tag,**row):print('E94_GEOMETRY '+json.dumps(dict(tag=tag,pid=os.getpid(),**row),sort_keys=True),flush=True)
 orig=cg.CudaGraphManager.run_fullgraph
 seen=set()
 @functools.wraps(orig)
 def replay(self,desc):
  graph=self.graphs.get(desc)
  result=orig(self,desc)
  assert graph is self.graphs[desc] and isinstance(graph,torch.cuda.CUDAGraph)
  key=(id(self),id(graph),str(desc))
  if key not in seen:
   assert len(seen)<64,'geometry event budget'
   seen.add(key)
   emit('full_replay_return',manager=type(self).__name__,manager_id=id(self),graph_id=id(graph),device=str(self.device),num_tokens=desc.num_tokens,num_reqs=desc.num_reqs,uniform_token_count=desc.uniform_token_count,decode_query_len=self.decode_query_len,graphs_captured=self._graphs_captured,capture_sizes=self.compilation_config.cudagraph_capture_sizes,max_capture=self.compilation_config.max_cudagraph_capture_size,graph_type=type(graph).__module__+'.'+type(graph).__qualname__)
  return result
 cg.CudaGraphManager.run_fullgraph=replay
 oldbind=qc._QSAStateCache.bind_kv_cache
 owners={}
 @functools.wraps(oldbind)
 def bind(self,tensor):
  result=oldbind(self,tensor)
  if True:
   assert sum(owners.values())<256,'geometry binding budget'
   owners[id(self)]=owners.get(id(self),0)+1
   emit('qsa_cache_bound',binding_ordinal=owners[id(self)],owner_type=type(self).__name__,owner_id=id(self),shape=list(self.kv_cache.shape),stride=list(self.kv_cache.stride()),device=str(self.kv_cache.device),dtype=str(self.kv_cache.dtype),storage_bytes=self.kv_cache.untyped_storage().nbytes(),storage_offset=self.kv_cache.storage_offset(),data_ptr=self.kv_cache.data_ptr(),is_cuda=self.kv_cache.is_cuda,compress_ratio=self.compress_ratio,attention_block_size=self.cache_config.block_size)
  return result
 qc._QSAStateCache.bind_kv_cache=bind
 _INSTALLED=True
 emit('installed',graph_source=GRAPH_HASH,qsa_source=QSA_HASH,scope='metadata only; sparse successful original replay-call returns; no tensor-content/state-closure claim')
