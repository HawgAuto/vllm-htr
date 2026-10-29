#!/usr/bin/env python3
"""QSA1600/MTP3 release profile for Qwen3.8 HTR on gfx1151."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='0.0.0.0')
    p.add_argument('--port',type=int,default=8000)
    p.add_argument('--served-model-name',default='qwen38-htr')
    p.add_argument('--cpu-check',action='store_true')
    p.add_argument('--print-command',action='store_true')
    return p


def command(args):
    argv=json.loads((ROOT/'server-argv.json').read_text())
    argv[0]=sys.executable
    for flag,value in [('--host',args.host),('--port',str(args.port)),('--served-model-name',args.served_model_name)]:
        argv[argv.index(flag)+1]=value
    argv+=['--default-chat-template-kwargs','{"enable_thinking":true}']
    return argv


def main():
    args=parser().parse_args()
    if not 1<=args.port<=65535:raise ValueError('port out of range')
    argv=command(args)
    if args.print_command:
        print(json.dumps(argv,indent=2));return
    # The existing release verifier binds native checkpoint manifests and indices.
    sys.path.insert(0,'/opt/haloloom/qwen38')
    import launch as base
    settings=argparse.Namespace(variant='bf16',mtp=3,yarn_factor=None,context=262144)
    base.configure(settings)
    os.environ.update(json.loads((ROOT/'runtime-env.json').read_text()))
    # Separate writable caches from every predecessor; do not apply incompatible seeds.
    for key in ['XDG_CACHE_HOME','VLLM_CACHE_ROOT','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','AITER_JIT_DIR','TVM_FFI_CACHE_DIR']:
        value=os.environ.get(key,'')
        if value.startswith('/workspace/cache/bf16'):
            os.environ[key]=value.replace('/workspace/cache/bf16','/workspace/cache/htr-qsa1600-v1',1)
    manifest=json.loads((ROOT/'installed-files.json').read_text())
    for relative,expected in manifest.items():
        path=ROOT/relative
        with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=expected:raise ValueError(f'HTR runtime integrity mismatch: {relative}')
    generation=json.loads(Path('/model/generation_config.json').read_text())
    expected=json.loads((ROOT/'sampling.json').read_text())
    for k,v in expected.items():
        if generation.get(k)!=v:raise ValueError(f'model sampling profile mismatch: {k}')
    from validate import verify
    report=verify(Path('/model'),'bf16',3)
    print(json.dumps({'htr_runtime_files':len(manifest),'base_and_model':report,'sampling':expected,'thinking':True,'argv':argv}),flush=True)
    if args.cpu_check:return
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]!='gfx1151':
        raise RuntimeError('This release requires a gfx1151 ROCm GPU')
    os.execv(sys.executable,argv)

if __name__=='__main__':main()
