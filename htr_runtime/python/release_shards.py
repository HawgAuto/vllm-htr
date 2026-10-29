"""Lossless, bounded-memory model packaging. No torch or tensor conversion.

Safetensors are streamed by raw tensor range. HTR chunk shards concatenate the
original serialized chunks; an index enables direct reads without extraction.
All output payloads are read back and hashed before a completion receipt.
"""
import hashlib
import json
import math
import os
import struct
from pathlib import Path

BLOCK = 8 * 1024 * 1024
DTYPES = {'BOOL':1,'U8':1,'I8':1,'I16':2,'U16':2,'F16':2,'BF16':2,'I32':4,'U32':4,'F32':4,'I64':8,'U64':8,'F64':8,'F8_E4M3':1,'F8_E5M2':1}


def pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError(f'duplicate JSON key: {key}')
        out[key] = value
    return out


def load_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs)


def save_json(path, value):
    path = Path(path)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write('\n')
        f.flush(); os.fsync(f.fileno())


def safe_path(root, relative):
    p = Path(relative)
    root = Path(root).resolve()
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError(f'unsafe relative path: {relative}')
    out = root/p
    if not out.resolve().is_relative_to(root):
        raise ValueError(f'path escapes root: {relative}')
    return out


def identity(path):
    st = Path(path).stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def header(path):
    path = Path(path)
    with path.open('rb') as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated length')
        n = struct.unpack('<Q', prefix)[0]
        if n > 100_000_000 or n < 2 or n+8 > path.stat().st_size:
            raise ValueError('invalid header length')
        raw = f.read(n)
    h = json.loads(raw, object_pairs_hook=pairs)
    if not isinstance(h, dict): raise ValueError('header must be an object')
    if '__metadata__' in h and (not isinstance(h['__metadata__'],dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in h['__metadata__'].items())):
        raise ValueError('invalid metadata')
    ranges = []
    for name, d in h.items():
        if name == '__metadata__': continue
        if not isinstance(d,dict) or set(d) != {'dtype','shape','data_offsets'} or d['dtype'] not in DTYPES:
            raise ValueError(f'invalid descriptor: {name}')
        shape, offsets = d['shape'], d['data_offsets']
        if not isinstance(shape,list) or any(type(x) is not int or x < 0 for x in shape): raise ValueError('invalid shape')
        if not isinstance(offsets,list) or len(offsets)!=2 or any(type(x) is not int for x in offsets): raise ValueError('invalid offsets')
        a,b = offsets
        if a < 0 or b < a or b+n+8 > path.stat().st_size or b-a != math.prod(shape)*DTYPES[d['dtype']]:
            raise ValueError(f'invalid payload range/size: {name}')
        ranges.append((a,b))
    end=0
    for a,b in sorted(ranges):
        if a != end: raise ValueError('payload overlap or gap')
        end=b
    if end+n+8 != path.stat().st_size: raise ValueError('unreferenced payload bytes')
    return h, n+8


def inventory(root):
    root = Path(root)
    idx = load_json(root/'model.safetensors.index.json')
    mapping = idx['weight_map']; out={}
    for relative in sorted(set(mapping.values())):
        path = safe_path(root,relative); h,base = header(path)
        for name,d in h.items():
            if name=='__metadata__': continue
            if name in out or mapping.get(name) != relative:
                raise ValueError(f'duplicate/unindexed/misbound tensor: {name}')
            a,b=d['data_offsets']
            out[name] = dict(name=name,dtype=d['dtype'],shape=d['shape'],path=str(path),offset=base+a,size=b-a,identity=identity(path))
    if set(out) != set(mapping): raise ValueError('index coverage mismatch')
    if sum(r['size'] for r in out.values()) != idx['metadata']['total_size']: raise ValueError('index total_size mismatch')
    return [out[k] for k in sorted(out)]


def read_tensor(record):
    with open(record['path'],'rb') as f:
        f.seek(record['offset']); return f.read(record['size'])


def stream_range(path,offset,size,dst=None):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        f.seek(offset)
        while size:
            b=f.read(min(BLOCK,size))
            if not b:raise ValueError('short read')
            if dst is not None:dst.write(b)
            h.update(b);size-=len(b)
    return h.hexdigest()


def groups(records,limit):
    if type(limit) is not int or limit <= 0: raise ValueError('positive shard size required')
    out=[];current=[];size=0
    for r in records:
        if current and size+r['size'] > limit:
            out.append(current);current=[];size=0
        current.append(r);size+=r['size']
    if current:out.append(current)
    return out


def repack_safetensors(source,destination,max_bytes=5_000_000_000):
    source,destination=Path(source),Path(destination)
    records=inventory(source)
    metadata={}
    for p in sorted({r['path'] for r in records}):
        m=header(p)[0].get('__metadata__',{})
        for k,v in m.items():
            if k in metadata and metadata[k]!=v:raise ValueError('conflicting file metadata')
            metadata[k]=v
    batches=groups(records,max_bytes)
    destination.mkdir(parents=True,exist_ok=False)
    mapping={};tensors={};shards=[]
    for i,batch in enumerate(batches,1):
        filename=f'model-{i:05d}-of-{len(batches):05d}.safetensors'
        path=destination/filename;h={'__metadata__':metadata};off=0
        for r in batch:
            h[r['name']]={'dtype':r['dtype'],'shape':r['shape'],'data_offsets':[off,off+r['size']]};off+=r['size']
        raw=json.dumps(h,separators=(',',':'),ensure_ascii=False).encode();raw+=b' '*(-len(raw)%8)
        with path.open('xb') as f:
            f.write(struct.pack('<Q',len(raw)));f.write(raw)
            for r in batch:
                if identity(r['path']) != r['identity']: raise ValueError('source identity drift')
                sha=stream_range(r['path'],r['offset'],r['size'],f)
                if identity(r['path']) != r['identity']: raise ValueError('source identity drift')
                mapping[r['name']]=filename
                tensors[r['name']]={'dtype':r['dtype'],'shape':r['shape'],'sha256':sha,'size':r['size']}
            f.flush();os.fsync(f.fileno())
        oh,base=header(path)
        for r in batch:
            a,b=oh[r['name']]['data_offsets']
            if stream_range(path,base+a,b-a)!=tensors[r['name']]['sha256']:raise ValueError('output tensor hash mismatch')
        shards.append({'file':filename,'size':path.stat().st_size,'sha256':stream_range(path,0,path.stat().st_size)})
        print(json.dumps({'phase':'safetensors','shard':i,'total':len(batches),'bytes':path.stat().st_size}),flush=True)
    save_json(destination/'model.safetensors.index.json',{'metadata':{'total_size':sum(r['size'] for r in records)},'weight_map':mapping})
    after=inventory(destination)
    if len(after)!=len(records):raise ValueError('output tensor coverage mismatch')
    return {'tensor_count':len(records),'payload_bytes':sum(r['size'] for r in records),'all_tensor_hashes_match':True,'tensors':tensors,'shards':shards}


def pack_chunks(source,destination,manifest,max_bytes=5_000_000_000):
    source,destination=Path(source),Path(destination)
    records=[]
    for rel,sha in sorted(manifest.items()):
        if not rel.endswith('.pt'):raise ValueError('unexpected expert manifest entry')
        path=safe_path(source,rel)
        records.append({'name':rel,'path':str(path),'sha256':sha,'size':path.stat().st_size,'identity':identity(path)})
    batches=groups(records,max_bytes)
    destination.mkdir(parents=True,exist_ok=False)
    chunks={};shards=[]
    for i,batch in enumerate(batches,1):
        fn=f'experts-{i:05d}-of-{len(batches):05d}.htr';p=destination/fn
        with p.open('xb') as out:
            for r in batch:
                if identity(r['path'])!=r['identity']:raise ValueError('source drift')
                off=out.tell();sha=stream_range(r['path'],0,r['size'],out)
                if sha!=r['sha256'] or identity(r['path'])!=r['identity']:raise ValueError('expert source hash/identity mismatch')
                chunks[r['name']]={'file':fn,'offset':off,'size':r['size'],'sha256':sha}
            out.flush();os.fsync(out.fileno())
        for r in batch:
            entry=chunks[r['name']]
            if stream_range(p,entry['offset'],entry['size'])!=entry['sha256']:raise ValueError('expert readback mismatch')
        shards.append({'file':fn,'size':p.stat().st_size,'sha256':stream_range(p,0,p.stat().st_size)})
        print(json.dumps({'phase':'experts','shard':i,'total':len(batches),'bytes':p.stat().st_size}),flush=True)
    save_json(destination/'chunks.index.json',{'schema':'htr-chunk-shards-v1','chunks':chunks,'shards':shards})
    return {'chunk_count':len(chunks),'all_chunk_hashes_match':True,'payload_bytes':sum(r['size'] for r in records),'shards':shards}


def read_chunk(root,relative,expected_sha256):
    root=Path(root);idx=load_json(root/'chunks.index.json')
    if idx['schema']!='htr-chunk-shards-v1':raise ValueError('unsupported chunk index')
    r=idx['chunks'][relative];p=safe_path(root,r['file'])
    off,size=r['offset'],r['size']
    if type(off) is not int or type(size) is not int or off<0 or not 0<size<=64*1024*1024 or off+size>p.stat().st_size:
        raise ValueError('invalid chunk bounds')
    with p.open('rb') as f:f.seek(off);b=f.read(size)
    if len(b)!=size or hashlib.sha256(b).hexdigest()!=expected_sha256 or r['sha256']!=expected_sha256:
        raise ValueError('chunk hash mismatch')
    return b
