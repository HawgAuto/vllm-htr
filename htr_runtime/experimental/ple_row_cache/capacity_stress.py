"""Capacity stress of the actual native implementation; no model/GPU claims."""
import ctypes as C
import json
import os
from pathlib import Path
import time
from test_row_cache import load,stats,ROOT

lib=load(ROOT/'libple_row_cache.so')
results=[]
for gib in (2,4,8):
    budget=gib*2**30
    assert lib.ple_cache_configure(budget)==0
    # Sparse memfd provides deterministic zero rows without writing bulk SSD data.
    total=budget//320+100000
    fd=os.memfd_create('ple-cache-capacity',os.MFD_CLOEXEC)
    os.ftruncate(fd,total*320)
    fds=(C.c_int32*1)(fd);base=(C.c_int64*1)(0);rows=(C.c_int64*1)(total)
    out=C.create_string_buffer(16384*320);s=(C.c_int64*3)()
    def rss():
        return int([x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')][0].split()[1])*1024
    before=rss();peak=before;t=time.monotonic()
    one=(C.c_int64*1)(0)
    assert lib.ple_gather_ids(one,1,fds,base,rows,1,total,320,total,out,1,1,s)==0
    cap=stats(lib)['capacity'];end=cap+32768
    assert end<total
    for start in range(0,end,16384):
        n=min(16384,end-start);ids=(C.c_int64*n)(*range(start,start+n))
        assert lib.ple_gather_ids(ids,n,fds,base,rows,1,total,320,total,out,1,1,s)==0
        assert out.raw[:n*320]==bytes(n*320)
        peak=max(peak,rss())
    q=stats(lib)
    assert q['resident_rows']==cap and q['evictions']>0
    assert q['allocated_bytes']<=budget
    assert peak-before<=budget+128*2**20,(gib,peak,before)
    results.append({'budget_gib':gib,'wall_s':time.monotonic()-t,'rss_before':before,'rss_peak':peak,'rss_delta':peak-before,'stats':q,'scope':'CPU-only full-capacity/eviction stress on sparse memfd zero rows; exact source bytes and real row mapping separately tested'})
    print(json.dumps(results[-1]),flush=True)
    lib.ple_cache_configure(0);os.close(fd)
path=ROOT/'capacity-stress-result.json'
with path.open('x') as f:json.dump({'status':'PASS','cases':results},f,indent=2)
