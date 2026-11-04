#!/usr/bin/env python3
"""HTR v0.2.1: native per-token/head KV and qualified context profiles.

Use --kv-cache-dtype int8_per_token_head or int4_per_token_head.
All existing HTR options and preferred sampling are retained. BF16 is default.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

DTYPES=('auto','int8_per_token_head','int4_per_token_head')
LIMITS={('int8_per_token_head',0):2000000,('int8_per_token_head',3):2000000,
        ('int4_per_token_head',0):4000000,('int4_per_token_head',3):3000000}

def load_release(root):
    spec=importlib.util.spec_from_file_location('htr_recommended',root/'launch_recommended.py')
    if spec is None or spec.loader is None:raise ImportError('HTR sampling launcher missing')
    recommended=importlib.util.module_from_spec(spec);spec.loader.exec_module(recommended)
    base=recommended.load_release(root);recommended.apply_profile(base)
    original_parser,original_check,original_profile=base.parser,base.check,base.profile
    original_command,original_configure,original_verify=base.command,base.configure,base.verify_runtime

    def parser():
        p=original_parser()
        p.add_argument('--kv-cache-dtype',choices=DTYPES,default='auto',
                       help='main attention K/V only; indexer and recurrent precision stay unchanged')
        return p

    def check(ap,args):
        if args.kv_cache_dtype=='auto':return original_check(ap,args)
        if args.context>LIMITS[(args.kv_cache_dtype,args.mtp)]:
            ap.error('context exceeds the released KV/MTP profile limit')
        if args.max_num_seqs!=1 or args.max_num_batched_tokens!=2048:
            ap.error('quantized KV profiles require --max-num-seqs 1 --max-num-batched-tokens 2048')
        if args.context<=base.MAX_CONTEXT:return original_check(ap,args)
        factor=8 if args.kv_cache_dtype=='int8_per_token_head' else 16
        if args.yarn_factor is None:args.yarn_factor=float(factor)
        if args.yarn_factor!=factor:ap.error('this long-context KV profile requires YaRN '+str(factor))
        proxy=copy.copy(args);proxy.context=base.MAX_CONTEXT;proxy.yarn_factor=4.0
        original_check(ap,proxy);args.sizes=proxy.sizes
        if args.context>args.yarn_factor*base.NATIVE_CONTEXT:ap.error('context exceeds YaRN envelope')

    def profile(args):
        name,levers=original_profile(args)
        dtype=args.kv_cache_dtype
        # Dtype and rotation selection are explicit cache axes, not shared BF16 graphs.
        return name+'-kv-'+dtype,dict(levers,main_kv=dtype,rht_d256='triton' if dtype=='int4_per_token_head' else None)

    def command(args):
        v=original_command(args)
        if args.kv_cache_dtype!='auto':v+=['--kv-cache-dtype',args.kv_cache_dtype]
        if args.context>base.MAX_CONTEXT:
            if v[1:3]!=['-m','vllm.entrypoints.openai.api_server']:raise ValueError('unexpected native entry point')
            v[2]='htr_capacity_api'
            if args.mtp:
                i=v.index('--speculative-config')+1;s=json.loads(v[i]);s['max_model_len']=args.context;v[i]=json.dumps(s)
        return v

    def configure(args,name):
        result=original_configure(args,name)
        os.environ['HTR_KV8_ENABLE']='1' if args.kv_cache_dtype=='int8_per_token_head' else '0'
        os.environ['HTR_KV4_ENABLE']='1' if args.kv_cache_dtype=='int4_per_token_head' else '0'
        os.environ['HTR_RHT_D256']='triton' if args.kv_cache_dtype=='int4_per_token_head' else 'pytorch'
        os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN']='0'
        return result

    def verify(args):
        for name,expected in json.loads((root/'quantized-kv-bindings.json').read_text()).items():
            with open(name,'rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=expected:raise ValueError('quantized KV binding mismatch: '+name)
        return original_verify(args)

    base.parser=parser;base.check=check;base.profile=profile;base.command=command
    base.configure=configure;base.verify_runtime=verify
    return base

if __name__=='__main__':load_release(Path(__file__).resolve().parent).main()
