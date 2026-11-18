"""Build HTR v0.2.4 from a committed tree over the immutable public base.

The selected dense and K6 extensions are rebuilt; unchanged native libraries remain
those of the qualified v0.2.2 image. No GPU device or network is used by compilation.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from build_combined import REPO, git, sha, tree_manifest, write

BASE = 'ghcr.io/hawgauto/vllm-htr@sha256:034898ebd182a240fa403ad5ea24bea3f647d933571e43420a27daf9af079d1c'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--tag', default='vllm-htr:v0.2.4')
    args = ap.parse_args()
    out = args.output.resolve()
    assert not out.is_relative_to(REPO), 'output must be outside checkout'
    assert not git('status', '--porcelain'), 'commit reviewed source before building'
    revision = git('rev-parse', 'HEAD')
    out.mkdir(parents=True, exist_ok=False)
    context = out/'context'
    source = context/'source'
    source.mkdir(parents=True)
    archive = out/'source.tar'
    subprocess.run(['git','-C',str(REPO),'archive','--format=tar','--output',str(archive),revision],check=True)
    with tarfile.open(archive) as tf:
        tf.extractall(source,filter='data')
    runtime = source/'htr_runtime'
    version = (runtime/'VERSION').read_text().strip()
    assert version == '0.2.4'
    manifest = json.loads((runtime/'DENSE-WMMA-BUILD.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(runtime/name) == digest, name
    receipt = {'status':'BUILDING','version':version,'source_commit':revision,
               'source_tree':git('rev-parse','HEAD^{tree}'),'source_archive_sha256':sha(archive),
               'base':BASE,'tag':args.tag,'started':time.time()}
    write(out/'INPUTS.json',receipt)
    if subprocess.run(['docker','image','inspect',BASE],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        subprocess.run(['docker','pull',BASE],check=True)
    base = json.loads(subprocess.check_output(['docker','image','inspect',BASE]))[0]
    assert base['Id'] == BASE.split('@')[1], 'unexpected runnable base identity'
    write(out/'base-image.json',base)
    work = out/'native-build'
    work.mkdir()
    shutil.copyfile(runtime/'native-recipes/htr_dense_wmma.ninja.in',work/'build.ninja')
    name = 'htr-v024-native-'+revision[:10]+'-'+str(os.getpid())
    command = ['docker','run','--rm','--name',name,'--network','none',
               '--cpus','2','--memory','8g','--memory-swap','8g',
               '--user',f'{os.getuid()}:{os.getgid()}',
               '--mount',f'type=bind,src={runtime}/native-source/dense-wmma,dst=/source,readonly',
               '--mount',f'type=bind,src={work},dst=/build',
               '--entrypoint','/usr/bin/timeout',BASE,'--signal=TERM','--kill-after=30s',
               '1200','ninja','-C','/build','-j','2']
    write(out/'native-argv.json',command)
    try:
        with (out/'native-build.log').open('x') as log:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1260)
        assert result.returncode == 0, 'native build failed; see native-build.log'
    finally:
        subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=30)
        assert not subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+name+'$']).strip()
    binary = work/'htr_wm.so'
    actual = sha(binary)
    write(out/'native-built.json',{'sha256':actual,'required_sha256':manifest['required_binary_sha256'],
                                  'source_bindings':manifest['files'],'exit_code':result.returncode})
    assert actual == manifest['required_binary_sha256'], 'native rebuild differs from selected arithmetic artifact'
    native = context/'native'
    native.mkdir()
    shutil.copyfile(binary,native/'htr_dense_wmma.so')
    # Preserve the qualified compiler source path: HIP derives code-object IDs
    # from it. Compilation is device-less and uses the same immutable base.
    native_manifest = json.loads((runtime/'NATIVE-BUILD.json').read_text())
    k6 = next(m for m in native_manifest['modules'] if m['name'] == 'htr_moe_k6')
    k6_files = {n:d for n,d in native_manifest['files'].items()
                if n.startswith('native-source/moe2/k6/') or n == k6['recipe']}
    for member, digest in k6_files.items():
        assert sha(runtime/member) == digest, member
    k6work = out/'k6-build'
    k6work.mkdir()
    (k6work/'build.ninja').write_text((runtime/k6['recipe']).read_text().replace('@SOURCE@','/opt/vllm-htr'))
    name = 'htr-v024-k6-'+revision[:10]+'-'+str(os.getpid())
    command = ['docker','run','--rm','--name',name,'--network','none',
               '--cpus','2','--memory','8g','--memory-swap','8g',
               '--user',f'{os.getuid()}:{os.getgid()}',
               '--mount',f'type=bind,src={runtime},dst=/opt/vllm-htr,readonly',
               '--mount',f'type=bind,src={k6work},dst=/build',
               '--entrypoint','/usr/bin/timeout',BASE,'--signal=TERM','--kill-after=30s',
               '1200','ninja','-C','/build','-j','2']
    write(out/'k6-argv.json',command)
    try:
        with (out/'k6-build.log').open('x') as log:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1260)
        assert result.returncode == 0, 'K6 build failed; see k6-build.log'
    finally:
        subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=30)
        assert not subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+name+'$']).strip()
    k6_actual = sha(k6work/'htr_moe_k6.so')
    write(out/'k6-built.json',{'sha256':k6_actual,'required_sha256':k6['accepted_binary_sha256'],
                              'source_bindings':k6_files,'exit_code':result.returncode})
    assert k6_actual == k6['accepted_binary_sha256'], 'K6 rebuild differs from qualified artifact'
    shutil.copyfile(k6work/'htr_moe_k6.so',native/'htr_moe_k6.so')
    command = ['docker','build','--provenance=false','--pull=false','--network=none',
               '--build-arg','BASE='+BASE,'--build-arg','SOURCE_REVISION='+revision,
               '-f',str(runtime/'Dockerfile.optimized'),'-t',args.tag,str(context)]
    write(out/'image-argv.json',command)
    with (out/'image-build.log').open('x') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
    image = json.loads(subprocess.check_output(['docker','image','inspect',args.tag]))[0]
    assert image['Config']['Labels']['org.opencontainers.image.revision'] == revision
    assert image['Config']['Labels']['org.opencontainers.image.version'] == version
    for field in ('Entrypoint','User','WorkingDir','Env'):
        assert image['Config'][field] == base['Config'][field], ('runtime config drift',field)
    assert image['RootFS']['Layers'][:len(base['RootFS']['Layers'])] == base['RootFS']['Layers']
    write(out/'image.json',image)
    receipt.update(status='BUILT_NOT_GPU_QUALIFIED',finished=time.time(),image_id=image['Id'],
                   native_binary_sha256=actual,native_binary_matches_selected=True,
                   k6_binary_sha256=k6_actual,k6_binary_matches_selected=True)
    write(out/'RESULT.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == '__main__':
    main()
