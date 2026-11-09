"""Install the optimized source payload over the pinned v0.2.2 runtime."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path('/opt/vllm-htr')
SITE = Path('/opt/venv/lib/python3.14/site-packages')
SOURCE = Path('/payload/htr_runtime')
BASE_MANIFEST = Path('/opt/haloloom/qwen38/data/installed-files.json')
sys.path.insert(0, str(SOURCE))
from release_bindings import reconcile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    # The image digest is checked by the builder; verify each inherited binding
    # as well before copying source. Only the known KV metadata defect is allowed.
    installed = json.loads((ROOT/'installed-files.json').read_text())
    native = json.loads((ROOT/'native-policy-bindings.json').read_text())
    base = json.loads(BASE_MANIFEST.read_text())
    for name, digest in installed.items():
        assert sha(ROOT/name) == digest, ('parent runtime drift', name)
    for name, digest in native.items():
        assert sha(name) == digest, ('parent native binding drift', name)
    for row in base['files']:
        assert sha(row['installed_path']) == row['sha256'], row['installed_path']
    kv_path = ROOT/'quantized-kv-bindings.json'
    kv_bytes = kv_path.read_text()
    kv = json.loads(kv_bytes)
    actual = {name:sha(name) for name in kv}
    fix_names = {str(SITE/'vllm/models/qwen4_exp/amd'/name)
                 for name in ('qsa.py','indexer_qsa.py')}
    assert {name for name in kv if kv[name] != actual[name]} == fix_names
    base_bindings = {row['installed_path']:row['sha256'] for row in base['files']}
    assert all(base_bindings[name] == native[name] == actual[name] for name in fix_names)
    repaired = reconcile(kv, actual, native, fix_names)
    for name in sorted(fix_names):
        assert kv_bytes.count(kv[name]) == 1
        kv_bytes = kv_bytes.replace(kv[name], repaired[name])
    assert json.loads(kv_bytes) == repaired
    kv_path.write_text(kv_bytes)
    repair = {'schema':'htr-quantized-kv-manifest-repair-v1',
              'entries':{name:{'before':kv[name],'after':repaired[name]}
                         for name in sorted(fix_names)},
              'runtime_bytes_changed_by_repair':False}

    shutil.copytree(SOURCE, ROOT, dirs_exist_ok=True)
    for name in ('htr_native_policy.py','00_htr_native_defaults.pth'):
        shutil.copyfile(ROOT/'python'/name, SITE/name)
    binary = Path('/payload/native/htr_dense_wmma.so')
    build = json.loads((ROOT/'DENSE-WMMA-BUILD.json').read_text())
    assert sha(binary) == build['required_binary_sha256'], 'dense native identity'
    for name, digest in build['files'].items():
        assert sha(ROOT/name) == digest, name
    shutil.copyfile(binary, ROOT/'native-bin/htr_dense_wmma.so')
    grouped = json.loads((ROOT/'python/htr_moe_serve/grouped-bindings.json').read_text())
    for name, digest in grouped['files'].items():
        assert sha(name) == digest, ('grouped closure drift', name)

    # Only files explicitly supplied by this committed source archive may alter
    # the parent base closure. All other members were verified above and stay put.
    declared = {str(ROOT/p.relative_to(SOURCE)):sha(p)
                for p in SOURCE.rglob('*') if p.is_file()}
    declared.update({str(SITE/name):sha(ROOT/'python'/name)
                     for name in ('htr_native_policy.py','00_htr_native_defaults.pth')})
    for row in base['files']:
        path = row['installed_path']
        if sha(path) != row['sha256']:
            assert declared.get(path) == sha(path), ('undeclared base change',path)
            row['sha256'] = sha(path)
            for field in ('size','bytes','size_bytes'):
                if field in row:
                    row[field] = Path(path).stat().st_size
    BASE_MANIFEST.write_text(json.dumps(base,indent=2)+'\n')
    bindings = {}
    for name, digest in native.items():
        current = sha(name)
        assert current == digest or declared.get(name) == current, ('undeclared native change',name)
        bindings[name] = current
    for directory in (ROOT/'python', ROOT/'native-bin'):
        for path in sorted(directory.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                bindings[str(path)] = sha(path)
    for name in ('htr_native_policy.py','00_htr_native_defaults.pth'):
        bindings[str(SITE/name)] = sha(SITE/name)
    (ROOT/'native-policy-bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
    revision = hashlib.sha256(json.dumps(bindings,sort_keys=True).encode()).hexdigest()
    (ROOT/'NATIVE-POLICY-REVISION').write_text(revision+'\n')
    (ROOT/'KV-MANIFEST-REPAIR.json').write_text(json.dumps(repair,indent=2)+'\n')
    files = {str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*'))
             if p.is_file() and p.name!='installed-files.json' and '__pycache__' not in p.parts}
    (ROOT/'installed-files.json').write_text(json.dumps(files,indent=2)+'\n')
    assert all(sha(name) == digest for name,digest in repaired.items())
    print(json.dumps({'status':'INSTALLED_NOT_GPU_QUALIFIED','version':(ROOT/'VERSION').read_text().strip(),
                      'native_revision':revision,'bindings':len(bindings),'kv_repaired':len(fix_names)}))


if __name__ == '__main__':
    main()
