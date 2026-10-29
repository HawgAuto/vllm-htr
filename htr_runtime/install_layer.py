"""Install the source-reviewed HTR layer during an image build, never at startup."""
import hashlib
import json
from pathlib import Path
import shutil

root=Path('/opt/vllm-htr')
site=Path('/opt/venv/lib/python3.14/site-packages')
for p in (root/'python').glob('*.py'):
    shutil.copyfile(p,site/p.name)
manifest_path=Path('/opt/haloloom/qwen38/data/installed-files.json')
manifest=json.loads(manifest_path.read_text())
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
