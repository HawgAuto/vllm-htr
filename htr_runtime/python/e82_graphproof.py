"""Opt-in graph-node identity + sparse replay receipts; no GPU profiler.
No kernel/argument/stream/graph mutation. keep_graph retains the raw graph;
HIP dot export inspects it, then native instantiate creates the executable.
Both control and candidate load this identical module. Capture disk I/O occurs
before timing; replay I/O only on powers of two, identically for both arms.
"""
import hashlib,json,os,time,ctypes,itertools,re
from pathlib import Path
import torch

ROOT=os.environ.get('E82_GRAPH_PROOF_DIR')
if ROOT and not getattr(torch.cuda.CUDAGraph,'_e82_installed',False):
    root=Path(ROOT);root.mkdir(parents=True,exist_ok=True)
    cls=torch.cuda.CUDAGraph
    serial=itertools.count()
    new,init,end,replay=cls.__new__,cls.__init__,cls.capture_end,cls.replay
    hip=ctypes.CDLL('libamdhip64.so')
    dot=hip.hipGraphDebugDotPrint
    dot.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_uint]
    dot.restype=ctypes.c_int
    def emit(row):
        row.update(pid=os.getpid(),time_ns=time.time_ns())
        with (root/('events-'+str(os.getpid())+'.jsonl')).open('a') as f:
            f.write(json.dumps(row,sort_keys=True)+'\n')
    def graph_new(cls,keep_graph=False):
        return new(cls,True)
    def graph_init(self,keep_graph=False):
        return init(self,True)
    def capture_end(self,*args,**kwargs):
        r=end(self,*args,**kwargs)
        self._e82_capture_id=next(serial)
        dst=root/('graph-%d-%x-%d.dot'%(os.getpid(),id(self),self._e82_capture_id))
        assert not dst.exists(),dst
        rc=dot(self.raw_cuda_graph(),os.fsencode(dst),1)
        assert rc==0,('hipGraphDebugDotPrint',rc)
        self.instantiate()
        data=dst.read_bytes();text=data.decode()
        can=len(re.findall(r'void k_e82_gemv_epi8?<5, 1,',text))+text.count('void k_e82_moe<4, 1,')
        native=len(re.findall(r'void k_gemv_epi8?<5, 1,',text))+text.count('void k_moe<4, 1,')
        self._e82_exec=int(self.raw_cuda_graph_exec());self._e82_count=0
        self._e82_interesting=can>0 or native>0
        emit(dict(event='capture',capture_id=self._e82_capture_id,object_id=id(self),graph_exec=self._e82_exec,dot=str(dst),sha256=hashlib.sha256(data).hexdigest(),candidate_nodes=can,native_nodes=native))
        return r
    def graph_replay(self,*args,**kwargs):
        r=replay(self,*args,**kwargs)
        self._e82_count+=1
        n=self._e82_count
        if self._e82_interesting and n&(n-1)==0:
            emit(dict(event='replay',capture_id=self._e82_capture_id,object_id=id(self),graph_exec=self._e82_exec,count=n))
        return r
    cls.__new__=staticmethod(graph_new);cls.__init__=graph_init;cls.capture_end=capture_end;cls.replay=graph_replay;cls._e82_installed=True
