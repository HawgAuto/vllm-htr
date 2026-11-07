"""Reproducible combined HTR image builder; no GPU access or service changes.

Run from a clean committed checkout. Output must be outside the checkout and
must not exist. A pinned dependency snapshot can be supplied for offline builds.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

BASE = 'ghcr.io/hawgauto/vllm-htr@sha256:dde8c3ea80e5d0a56e9143894a12622d0c236bab1f6d3d8a10e4ab911c836afb'
TRITON_COMMIT = '669b31acc1dd1b3fd93286afd5db67f65d9f7557'
TRITON_TREE = '528405b13ae2d62d658e28fe53f1dfba63b6871a216869d0e5ecd4c211a38948'
REPO = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def tree_manifest(root):
    result = {}
    for p in sorted(root.rglob('*')):
        if '__pycache__' in p.parts:
            continue
        if p.is_symlink():
            raise ValueError('symlink in pinned input: ' + str(p))
        if p.is_file():
            result[str(p.relative_to(root))] = sha(p)
    return result


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def write(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--tag', default='vllm-htr:v0.2.2')
    ap.add_argument('--triton-kernels', type=Path)
    args = ap.parse_args()
    out = args.output.resolve()
    assert not out.is_relative_to(REPO), 'output must be outside checkout'
    assert not git('status', '--porcelain'), 'commit reviewed source before building'
    revision = git('rev-parse', 'HEAD')
    out.mkdir(parents=True, exist_ok=False)
    receipt = {'source_commit': revision, 'source_tree': git('rev-parse', 'HEAD^{tree}'),
               'base': BASE, 'tag': args.tag, 'started': time.time(), 'status': 'STARTED'}
    write(out/'inputs.json', receipt)
    context = out/'context'
    source = context/'source'
    source.mkdir(parents=True)
    archive = out/'source.tar'
    subprocess.run(['git', '-C', str(REPO), 'archive', '--format=tar',
                    '--output', str(archive), revision], check=True)
    with tarfile.open(archive) as tf:
        tf.extractall(source, filter='data')
    changed = git('diff', '--name-only', 'htr-v0.2.1', revision).splitlines()
    write(context/'changed.json', changed)
    dep = args.triton_kernels
    if dep is None:
        clone = out/'triton-source'
        subprocess.run(['git', 'init', str(clone)], check=True)
        subprocess.run(['git', '-C', str(clone), 'remote', 'add', 'origin',
                        'https://github.com/ROCm/triton.git'], check=True)
        subprocess.run(['git', '-C', str(clone), 'fetch', '--depth=1', 'origin',
                        TRITON_COMMIT], check=True, timeout=600)
        subprocess.run(['git', '-C', str(clone), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
        dep = clone/'python/triton_kernels/triton_kernels'
    dep = dep.resolve()
    dep_manifest = tree_manifest(dep)
    assert hashlib.sha256(json.dumps(dep_manifest, sort_keys=True).encode()).hexdigest() == TRITON_TREE
    write(out/'triton-inputs.json', {'commit': TRITON_COMMIT, 'tree': TRITON_TREE, 'files': dep_manifest})
    if subprocess.run(['docker', 'image', 'inspect', BASE], stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode:
        subprocess.run(['docker', 'pull', BASE], check=True)
    base = json.loads(subprocess.check_output(['docker', 'image', 'inspect', BASE]))[0]
    write(out/'base-image.json', base)
    build = out/'native-build'
    build.mkdir()
    name = 'htr-v022-native-' + revision[:10] + '-' + str(os.getpid())
    command = ['docker', 'run', '--rm', '--name', name, '--network', 'none',
               '--cpus', '4', '--memory', '16g', '--memory-swap', '16g',
               '--user', f'{os.getuid()}:{os.getgid()}',
               '--mount', f'type=bind,src={source},dst=/source,readonly',
               '--mount', f'type=bind,src={build},dst=/build',
               '--mount', f'type=bind,src={dep},dst=/deps/triton_kernels,readonly',
               '--entrypoint', '/usr/bin/timeout', BASE, '--signal=TERM',
               '--kill-after=30s', '2400', '/bin/bash', '/source/htr_runtime/build_core.sh']
    write(out/'native-argv.json', command)
    try:
        with (out/'native-build.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=2460)
        assert result.returncode == 0, 'native build failed; see native-build.log'
    finally:
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=30)
        assert not subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=^/'+name+'$']).strip()
    native = context/'native'
    native.mkdir()
    libraries = list((build/'cmake').glob('_C*.so'))
    assert len(libraries) == 1
    shutil.copy2(libraries[0], native/libraries[0].name)
    write(out/'native-built.json', {'source_archive_sha256': sha(archive),
                                  'libraries': tree_manifest(native), 'exit_code': result.returncode})
    command = ['docker', 'build', '--provenance=false', '--pull=false', '--network=none',
               '--build-arg', 'BASE='+BASE, '--build-arg', 'SOURCE_REVISION='+revision,
               '-f', str(source/'htr_runtime/Dockerfile.native-policy'), '-t', args.tag, str(context)]
    write(out/'image-argv.json', command)
    with (out/'image-build.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', args.tag]))[0]
    assert image['Config']['Labels']['org.opencontainers.image.revision'] == revision
    assert image['Config']['Entrypoint'] == ['/opt/venv/bin/vllm', 'serve']
    write(out/'image.json', image)
    receipt.update(status='BUILT_NOT_GPU_QUALIFIED', finished=time.time(), image_id=image['Id'],
                   source_archive_sha256=sha(archive), native_libraries=tree_manifest(native))
    write(out/'RESULT.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
