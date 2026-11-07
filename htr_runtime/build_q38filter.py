"""Build the accepted gfx1151 draft-filter payload from packaged source.

The exact qualified compiler invocation is deliberate: changing contraction,
input basename or save-temps can change the pinned binary. No GPU is needed.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
SOURCE_SHA = '53ce0b476368886c6bc3101a7f944b33209dea3ced643161757bbeff4a7b7fcd'
LIB_SHA = '4c1b758ff040cd3f9bc284e79cd27a358b85f931315cfe2c47283acf39edf4ec'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch
    assert str(torch.__version__) == '2.13.0+rocm10.0.0'
    assert torch.version.hip == '7.15.26333'
    source = ROOT / 'native-source/q38filter/q38_filter_hip.hip'
    assert sha(source) == SOURCE_SHA, 'draft-filter source differs from accepted port'
    destination = ROOT / 'python/htr_q38f'
    with tempfile.TemporaryDirectory(prefix='htr-q38f-', dir=os.environ.get('TMPDIR')) as tmp:
        work = Path(tmp)
        shutil.copy2(source, work / source.name)
        command = ['/opt/rocm/bin/hipcc', '-O3', '--offload-arch=gfx1151',
                   '-ffp-contract=off', '-fPIC', '-shared', '-save-temps=obj',
                   '-o', 'libq38f.so', source.name]
        subprocess.run(command, cwd=work, check=True, timeout=180)
        lib = work / 'libq38f.so'
        assert sha(lib) == LIB_SHA, 'draft-filter build is not byte-identical'
        shutil.copy2(lib, destination / lib.name)
    manifest = {'torch': str(torch.__version__), 'library_sha256': LIB_SHA,
                'qualified_source_sha256': SOURCE_SHA,
                'files': {name: sha(destination / name)
                          for name in ('__init__.py', 'q38f.py', 'libq38f.so')}}
    (destination / 'payload.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('HTR_Q38F_SOURCE_BUILD', json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
