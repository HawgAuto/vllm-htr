#!/usr/bin/env python3
"""Rebuild the selected gfx1151 kernels using their source-bound Ninja recipes.

Run inside the pinned Docker base. No GPU access or runtime JIT is required.
The recipes retain the accepted compiler flags, including per-module FP rules.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
EXPECTED = {'htr_kernels', 'htr_moe2', 'htr_moe_prefill_v3', 'htr_moe_serve_k',
            'htr_dq_arena', 'htr_moe_prefill', 'htr_moe_prefill_v2',
            'e59_pledev_ext', 'hiphc', 'hiphcp', 'ple_gather_v2'}


def member(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'non-relative build member: {relative}')
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'escaping build member: {relative}')
    return result


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_files(root, files):
    for relative, expected in files.items():
        if digest(member(root, relative)) != expected:
            raise ValueError(f'input digest mismatch: {relative}')


def validate(root):
    manifest = json.loads((root / 'NATIVE-BUILD.json').read_text())
    if manifest['schema'] != 'htr-native-build-v1' or manifest['target'] != 'gfx1151':
        raise ValueError('unsupported native build manifest')
    names = [r['name'] for r in manifest['modules']]
    if len(names) != len(EXPECTED) or set(names) != EXPECTED:
        raise ValueError('incomplete or duplicated native module set')
    outputs = set()
    for row in manifest['modules']:
        output = member(root, row['output'])
        if output in outputs or output.suffix != '.so':
            raise ValueError('invalid or duplicate native output')
        outputs.add(output)
        for relative in [row['recipe'], *row['sources']]:
            if relative not in manifest['files']:
                raise ValueError(f'unbound build input: {relative}')
    for row in manifest.get('hsaco', []):
        output = member(root, row['output'])
        if output in outputs or output.suffix != '.hsaco' or row['source'] not in manifest['files']:
            raise ValueError('invalid hsaco build row: ' + row['name'])
        outputs.add(output)
    check_files(root, manifest['files'])
    return manifest


def build_hsaco(root, work, row):
    """Code-object kernels loaded at run time by hash. The exact command is part of the contract (hipcc emits a
    different code object without -save-temps), so the output must equal the qualified binary."""
    work.mkdir()
    source = member(root, row['source'])
    # hipcc derives the code object's compilation-unit id from the source path as given, so compile a local copy
    # by its bare file name (the qualified binary was built that way).
    shutil.copyfile(source, work / source.name)
    out = work / (row['name'] + '.hsaco')
    subprocess.run([*row['command'], source.name, '-o', out.name], cwd=work, check=True)
    got = digest(out)
    if got != row['required_sha256']:
        raise ValueError(f"{row['name']}: built {got}, qualified {row['required_sha256']}")
    dest = member(root, row['output'])
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out, dest)
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--work', type=Path, default=Path('/build/htr-native'))
    ap.add_argument('--jobs', type=int, default=2)
    args = ap.parse_args()
    manifest = validate(ROOT)
    if args.check:
        print(json.dumps({'validated_modules': len(manifest['modules']),
                          'validated_hsaco': len(manifest.get('hsaco', [])),
                          'bound_files': len(manifest['files'])}))
        return
    if args.jobs < 1:
        ap.error('--jobs must be positive')
    # A fresh work root prevents reuse of objects from another source closure.
    args.work.mkdir(parents=True, exist_ok=False)
    receipt = {'manifest_sha256': digest(ROOT / 'NATIVE-BUILD.json'),
               'target': manifest['target'], 'outputs': {}}
    for row in manifest['modules']:
        work = args.work / row['name']
        work.mkdir()
        recipe = member(ROOT, row['recipe']).read_text().replace('@SOURCE@', str(ROOT))
        (work / 'build.ninja').write_text(recipe)
        print('BUILD_NATIVE ' + row['name'], flush=True)
        subprocess.run(['ninja', '-C', str(work), '-j', str(args.jobs)], check=True)
        source = work / (row['name'] + '.so')
        dest = member(ROOT, row['output'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        receipt['outputs'][row['output']] = {'sha256': digest(dest), 'bytes': dest.stat().st_size}
    for row in manifest.get('hsaco', []):
        print('BUILD_HSACO ' + row['name'], flush=True)
        dest = build_hsaco(ROOT, args.work / row['name'], row)
        receipt['outputs'][row['output']] = {'sha256': digest(dest), 'bytes': dest.stat().st_size}
    # Do not permit source mutation during a build to silently rebind provenance.
    check_files(ROOT, manifest['files'])
    (ROOT / 'native-build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'built_modules': len(receipt['outputs'])}), flush=True)


if __name__ == '__main__':
    main()
