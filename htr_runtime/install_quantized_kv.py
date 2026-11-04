"""Build and bind only the accepted v0.2.1 KV delta over v0.2.0."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT=Path('/opt/vllm-htr')
SITE=Path('/opt/venv/lib/python3.14/site-packages')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    contract=json.loads((ROOT/'quantized-kv-source.json').read_text())
    old=json.loads((ROOT/'installed-files.json').read_text())
    changed=set(contract['runtime_sources'])
    # Do not silently rebind unrelated inherited runtime bytes.
    for name,expected in old.items():
        if name not in changed:assert sha(ROOT/name)==expected,('unrelated runtime drift',name)
    for name,expected in contract['runtime_sources'].items():
        assert sha(ROOT/name)==expected,('release source drift',name)
    owner=SITE/'vllm/models/qwen4_exp/amd/qsa.py'
    assert sha(owner)==contract['owner_sha256']
    baseline=Path('/opt/haloloom/qwen38/data/installed-files.json')
    inherited=json.loads(baseline.read_text())
    hits=0
    for row in inherited['files']:
        if row['installed_path']==str(owner):
            assert row['sha256']==contract['parent_owner_sha256']
            row['sha256']=sha(owner)
            for field in ['size','bytes','size_bytes']:
                if field in row:row[field]=owner.stat().st_size
            hits+=1
        else:assert sha(Path(row['installed_path']))==row['sha256'],row['installed_path']
    assert hits==1
    recipe=next(x for x in json.loads((ROOT/'NATIVE-BUILD.json').read_text())['hsaco'] if x['name']=='qsa_tau_v2')
    compiled={}
    with tempfile.TemporaryDirectory(prefix='htr-kv-build-') as tmp:
        work=Path(tmp)
        for bits,expected in contract['native_sha256'].items():
            name='qsa_tau_kv'+bits
            src=ROOT/'native-source/qsa_tau'/(name+'.hip')
            out=work/(name+'.hsaco')
            subprocess.run([*recipe['command'],'-cuid='+contract['native_cuid'][bits],str(src),'-o',str(out)],cwd=work,check=True,timeout=180)
            if out.read_bytes().startswith(b'__CLANG_OFFLOAD_BUNDLE__'):
                elf=work/(name+'.gfx1151.hsaco')
                subprocess.run(['/opt/rocm/llvm/bin/clang-offload-bundler','--unbundle','--type=o',
                    '--targets=hipv4-amdgcn-amd-amdhsa--gfx1151','--input='+str(out),'--output='+str(elf)],check=True)
                out=elf
            assert out.read_bytes()[:4]==b'\x7fELF' and sha(out)==expected,('accepted native byte identity',bits,sha(out),expected)
            dest=ROOT/'native-bin'/(name+'.hsaco');shutil.copyfile(out,dest)
            compiled[str(dest.relative_to(ROOT))]=expected
    baseline.write_text(json.dumps(inherited,sort_keys=True,indent=2)+'\n')
    bindings={str(p):sha(p) for p in [owner,owner.with_name('indexer_qsa.py'),owner.with_name('qsa_mtp.py'),
        SITE/'vllm/v1/attention/ops/int4_per_token_head.py',SITE/'vllm/v1/attention/backends/triton_attn.py']}
    (ROOT/'quantized-kv-bindings.json').write_text(json.dumps(bindings,sort_keys=True,indent=2)+'\n')
    old.update({name:sha(ROOT/name) for name in changed})
    old.update(compiled)
    for name in ['quantized-kv-source.json','quantized-kv-bindings.json']:old[name]=sha(ROOT/name)
    (ROOT/'installed-files.json').write_text(json.dumps(old,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'status':'ACCEPTED_NATIVE_BYTES_AND_RUNTIME_BOUND','native':compiled,'runtime_files':len(old)},sort_keys=True))

if __name__=='__main__':main()
