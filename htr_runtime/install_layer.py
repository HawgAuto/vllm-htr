"""Install the source-reviewed HTR layer during an image build, never at startup."""
import hashlib
import json
from pathlib import Path
import shutil

root=Path('/opt/vllm-htr')
site=Path('/opt/venv/lib/python3.14/site-packages')
# Retired base-layer QSA union path: the HTR QSA owner calls htr_qsa directly and nothing else imports these.
RETIRED=[site/'direct_qsa_union_adapter_v1.py',site/'direct_qsa_union_kernel_v1.py',site/'direct_qsa_union_build_v1.hsaco']
for p in RETIRED:
    if p.exists():p.unlink()
for p in (root/'python').glob('*.py'):
    shutil.copyfile(p,site/p.name)
manifest_path=Path('/opt/haloloom/qwen38/data/installed-files.json')
manifest=json.loads(manifest_path.read_text())
retired={str(p) for p in RETIRED}
manifest['files']=[row for row in manifest['files'] if row['installed_path'] not in retired]
for row in manifest['files']:
    with Path(row['installed_path']).open('rb') as f:
        row['sha256']=hashlib.file_digest(f,'sha256').hexdigest()
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
files={}
for p in sorted(root.rglob('*')):
    if p.is_file() and p.name!='installed-files.json':
        with p.open('rb') as f:files[str(p.relative_to(root))]=hashlib.file_digest(f,'sha256').hexdigest()
(root/'installed-files.json').write_text(json.dumps(files,indent=2)+'\n')
print('HTR_INSTALLED_FILES',len(files))
