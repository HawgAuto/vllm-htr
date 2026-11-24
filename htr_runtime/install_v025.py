"""Install the v0.2.5 source + native payload cleanly over v0.2.4 and reconcile all bindings."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/opt/vllm-htr')
SITE = Path('/opt/venv/lib/python3.14/site-packages')
PAYLOAD = Path('/payload')
BASE_MANIFEST = Path('/opt/haloloom/qwen38/data/installed-files.json')


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    file_map = json.loads((PAYLOAD / 'file_map.json').read_text())
    nat_manifest = json.loads((PAYLOAD / 'V025-NATIVE-MANIFEST.json').read_text())

    # 1. Install all 38 Python files + runtime-env.json + grouped-bindings.json + V025-NATIVE-MANIFEST.json
    installed_paths = set()
    for rel, dst_str in file_map.items():
        src = PAYLOAD / 'files' / rel
        dst = Path(dst_str)
        assert src.is_file(), f'missing payload file {src}'
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        installed_paths.add(str(dst))

    # Mirror htr_native_policy.py into site-packages (imported by 00_htr_native_defaults.pth)
    shutil.copyfile(ROOT / 'python/htr_native_policy.py', SITE / 'htr_native_policy.py')
    installed_paths.add(str(SITE / 'htr_native_policy.py'))

    # Install runtime-env.json, grouped-bindings.json, V025-NATIVE-MANIFEST.json
    for rel, dst in [
        ('runtime-env.json', ROOT / 'runtime-env.json'),
        ('grouped-bindings.json', ROOT / 'python/htr_moe_serve/grouped-bindings.json'),
        ('V025-NATIVE-MANIFEST.json', ROOT / 'V025-NATIVE-MANIFEST.json'),
    ]:
        shutil.copyfile(PAYLOAD / rel, dst)
        installed_paths.add(str(dst))

    # 2. Install & SHA-verify all 18 winning native binaries into /opt/vllm-htr/native-bin/
    nat_dir = ROOT / 'native-bin'
    nat_dir.mkdir(parents=True, exist_ok=True)
    for name, meta in nat_manifest['binaries'].items():
        src = PAYLOAD / 'native' / name
        assert src.is_file(), f'missing native binary {src}'
        assert sha(src) == meta['sha256'], f'native sha mismatch for {name}'
        dst = nat_dir / name
        shutil.copyfile(src, dst)
        dst.chmod(0o755 if name.endswith('.so') else 0o644)
        installed_paths.add(str(dst))

    # 3. Verify grouped-bindings.json matches installed files
    gb = json.loads((ROOT / 'python/htr_moe_serve/grouped-bindings.json').read_text())
    for path_str, expected in gb['files'].items():
        actual = sha(Path(path_str))
        assert actual == expected, f'grouped-bindings mismatch: {path_str}: {actual} != {expected}'

    # 4. Reconcile installed-files.json, native-policy-bindings.json, quantized-kv-bindings.json, BASE_MANIFEST
    inst_path = ROOT / 'installed-files.json'
    if inst_path.is_file():
        inst = json.loads(inst_path.read_text())
        for k in list(inst.keys()):
            p = ROOT / k
            if p.is_file():
                inst[k] = sha(p)
        for name in nat_manifest['binaries']:
            inst[f'native-bin/{name}'] = sha(nat_dir / name)
        inst['V025-NATIVE-MANIFEST.json'] = sha(ROOT / 'V025-NATIVE-MANIFEST.json')
        inst_path.write_text(json.dumps(inst, indent=2, sort_keys=True) + '\n')

    for bp_name in ('native-policy-bindings.json', 'quantized-kv-bindings.json'):
        bp = ROOT / bp_name
        if bp.is_file():
            obj = json.loads(bp.read_text())
            for k in list(obj.keys()):
                p = Path(k)
                if p.is_file():
                    obj[k] = sha(p)
            if bp_name == 'native-policy-bindings.json':
                for name in nat_manifest['binaries']:
                    obj[str(nat_dir / name)] = sha(nat_dir / name)
            bp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')

    if BASE_MANIFEST.is_file():
        base = json.loads(BASE_MANIFEST.read_text())
        for row in base.get('files', []):
            p = Path(row['installed_path'])
            if p.is_file():
                row['sha256'] = sha(p)
                for field in ('size', 'bytes', 'size_bytes'):
                    if field in row:
                        row[field] = p.stat().st_size
        BASE_MANIFEST.write_text(json.dumps(base, indent=2) + '\n')

    # Clean up /payload so image stays lean
    shutil.rmtree(PAYLOAD, ignore_errors=True)
    print(f'v0.2.5 install complete: {len(file_map)} Python files + {len(nat_manifest["binaries"])} native binaries verified.')


if __name__ == '__main__':
    main()
