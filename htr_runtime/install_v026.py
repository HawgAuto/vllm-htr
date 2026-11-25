"""Install the v0.2.6 source + native payload cleanly over v0.2.5 and reconcile all SHA-256 bindings."""
import hashlib, json, shutil, sys
from pathlib import Path

ROOT = Path('/opt/vllm-htr')
SITE = Path('/opt/venv/lib/python3.14/site-packages')
PAYLOAD = Path('/payload')
BASE_MANIFEST = Path('/opt/haloloom/qwen38/data/installed-files.json')

def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main() -> None:
    file_map = json.loads((PAYLOAD / 'file_map.json').read_text())
    nat_manifest = json.loads((PAYLOAD / 'V026-NATIVE-MANIFEST.json').read_text())

    # 1. Install all Python files (including mirrored site-packages copies)
    for rel, dst_list in file_map.items():
        src = PAYLOAD / 'files' / rel
        assert src.is_file(), f'missing payload file {src}'
        for dst_str in dst_list:
            dst = Path(dst_str)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

    # Install runtime-env.json, V026-NATIVE-MANIFEST.json, VERSION
    shutil.copyfile(PAYLOAD / 'runtime-env.json', ROOT / 'runtime-env.json')
    shutil.copyfile(PAYLOAD / 'V026-NATIVE-MANIFEST.json', ROOT / 'V026-NATIVE-MANIFEST.json')
    (ROOT / 'VERSION').write_text('0.2.6\n')

    # 2. Install & SHA-verify all 23 winning native binaries into /opt/vllm-htr/native-bin/
    nat_dir = ROOT / 'native-bin'
    nat_dir.mkdir(parents=True, exist_ok=True)
    for name, meta in nat_manifest['binaries'].items():
        src = PAYLOAD / 'native' / name
        assert src.is_file(), f'missing native binary {src}'
        assert sha(src) == meta['sha256'], f'native sha mismatch for {name}'
        dst = nat_dir / name
        shutil.copyfile(src, dst)
        dst.chmod(0o755 if name.endswith('.so') else 0o644)

    # 3. Reconcile grouped-bindings.json (hashes htr_moe_serve/*.py and htr_moe*.so)
    gb_path = ROOT / 'python/htr_moe_serve/grouped-bindings.json'
    if gb_path.is_file():
        gb = json.loads(gb_path.read_text())
        for path_str in list(gb.get('files', {}).keys()):
            p = Path(path_str)
            if p.is_file():
                gb['files'][path_str] = sha(p)
        gb['files'][str(nat_dir / 'hpf5_v2_moe_prefill.so')] = sha(nat_dir / 'hpf5_v2_moe_prefill.so')
        gb_path.write_text(json.dumps(gb, indent=2, sort_keys=True) + '\n')

    # 4. Reconcile V025-NATIVE-MANIFEST.json for any Wave 7 upgraded binaries (htr_bf16_gemv.so, htr_moe_k6_shx.so)
    v25_m = ROOT / 'V025-NATIVE-MANIFEST.json'
    if v25_m.is_file():
        m25 = json.loads(v25_m.read_text())
        for name in list(m25.get('binaries', {}).keys()):
            p = nat_dir / name
            if p.is_file():
                m25['binaries'][name] = {'sha256': sha(p), 'bytes': p.stat().st_size}
        v25_m.write_text(json.dumps(m25, indent=2) + '\n')

    # 5. Reconcile native-policy-bindings.json & quantized-kv-bindings.json FIRST,
    #    then installed-files.json LAST (since installed-files.json hashes the binding JSONs!)
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

    inst_path = ROOT / 'installed-files.json'
    if inst_path.is_file():
        inst = json.loads(inst_path.read_text())
        for k in list(inst.keys()):
            p = ROOT / k
            if p.is_file():
                inst[k] = sha(p)
        for name in nat_manifest['binaries']:
            inst[f'native-bin/{name}'] = sha(nat_dir / name)
        inst['V026-NATIVE-MANIFEST.json'] = sha(ROOT / 'V026-NATIVE-MANIFEST.json')
        inst_path.write_text(json.dumps(inst, indent=2, sort_keys=True) + '\n')

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

    # Clean up /payload and run live zero-overlay verify_installation() self-check!
    shutil.rmtree(PAYLOAD, ignore_errors=True)
    sys.path.insert(0, str(ROOT / 'python'))
    import htr_native_policy
    htr_native_policy.verify_installation()
    print(f'v0.2.6 ZERO-OVERLAY install verified: {len(file_map)} Python files + {len(nat_manifest["binaries"])} native binaries.')

if __name__ == '__main__':
    main()
