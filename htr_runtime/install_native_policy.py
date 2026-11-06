"""Install a bounded source delta over v0.2.1; bind exact installed bytes."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path('/opt/vllm-htr')
SITE=Path('/opt/venv/lib/python3.14/site-packages')
SRC=Path('/payload/source')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
# Verify the parent before changing it; do not paper over unrelated drift.
old=json.loads((ROOT/'installed-files.json').read_text())
for name,digest in old.items():
    assert sha(ROOT/name)==digest,('parent drift',name)
base_manifest=Path('/opt/haloloom/qwen38/data/installed-files.json')
base=json.loads(base_manifest.read_text())
for row in base['files']:
    assert sha(Path(row['installed_path']))==row['sha256'],('base drift',row['installed_path'])
changed=json.loads(Path('/payload/changed.json').read_text())
for relative in changed:
    src=SRC/relative
    if not src.is_file(): continue
    if relative.startswith('vllm/') and src.suffix=='.py':
        dst=SITE/relative; dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
# Preserve inherited binaries and historical launchers; add only new policy and
# the exact source tree/tests required to reproduce this candidate.
shutil.copytree(SRC/'htr_runtime',ROOT,dirs_exist_ok=True)
for name in ('htr_native_policy.py','00_htr_native_defaults.pth'):
    shutil.copy2(ROOT/'python'/name,SITE/name)
new_so=next(Path('/payload/native').glob('_C*.so'))
old_so=list((SITE/'vllm').glob('_C.*so'))
assert len(old_so)==1,old_so
shutil.copy2(new_so,old_so[0])
for row in base['files']:
    p=Path(row['installed_path']);row['sha256']=sha(p)
    for field in ('size','bytes','size_bytes'):
        if field in row:row[field]=p.stat().st_size
base_manifest.write_text(json.dumps(base,indent=2)+'\n')
# Bind runtime and every overlaid vLLM file independently of predecessor policy.
bindings={str(SITE/p):sha(SITE/p) for p in changed if p.startswith('vllm/') and (SITE/p).is_file() and p.endswith('.py')}
bindings[str(old_so[0])]=sha(old_so[0])
for p in [SITE/'htr_native_policy.py',SITE/'00_htr_native_defaults.pth']:
    bindings[str(p)]=sha(p)
(ROOT/'native-policy-bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
revision=hashlib.sha256(json.dumps(bindings,sort_keys=True).encode()).hexdigest()
(ROOT/'NATIVE-POLICY-REVISION').write_text(revision+'\n')
files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='installed-files.json' and '__pycache__' not in p.parts}
(ROOT/'installed-files.json').write_text(json.dumps(files,indent=2)+'\n')
print('HTR_NATIVE_POLICY_INSTALLED',revision,len(bindings))
