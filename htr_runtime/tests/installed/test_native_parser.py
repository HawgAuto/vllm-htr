"""Installed native parser checks. CPU platform used only for argument parsing."""
import json,os,unittest
from pathlib import Path
import vllm.platforms as platforms
from vllm.platforms.cpu import CpuPlatform
platforms._current_platform=CpuPlatform()
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.launchers.cli_args import make_arg_parser
import htr_native_policy as policy

class InstalledNativeParser(unittest.TestCase):
    def parse(self, argv=()):
        return make_arg_parser(FlexibleArgumentParser()).parse_args(list(argv))

    def test_defaults_and_sampling(self):
        a=self.parse()
        self.assertEqual(a.max_num_batched_tokens,12288)
        self.assertEqual(a.max_num_seqs,1)
        self.assertEqual(a.speculative_config['num_speculative_tokens'],3)
        self.assertEqual(a.override_generation_config,{})
        self.assertEqual(a.generation_config,'auto')

    def test_explicit_overrides_all_formats(self):
        for dtype in ('auto','bfloat16','int8_per_token_head','int4_per_token_head','octave_k3v4','octave_k3v3','octave_k3v3_compact'):
            a=self.parse(['--kv-cache-dtype',dtype,'--max-num-seqs','8','--max-num-batched-tokens','4096',
                '--max-model-len','65536','--speculative-config','null','--generation-config','vllm',
                '--override-generation-config','{"temperature":0.35,"top_p":0.81,"top_k":31}',
                '--no-enable-prefix-caching'])
            self.assertEqual(a.kv_cache_dtype,dtype)
            self.assertEqual((a.max_num_seqs,a.max_num_batched_tokens,a.max_model_len),(8,4096,65536))
            self.assertIsNone(a.speculative_config)
            self.assertEqual(a.generation_config,'vllm')
            self.assertEqual(a.override_generation_config,dict(temperature=.35,top_p=.81,top_k=31))
            self.assertFalse(a.enable_prefix_caching)

    def test_user_config_file_equals_explicit_command(self):
        import tempfile,yaml
        options={'max-num-seqs':2,'max-num-batched-tokens':4096,'max-model-len':65536,'kv-cache-dtype':'octave_k3v4','speculative-config':None,'override-generation-config':{'temperature':0.35,'top_p':0.81}}
        with tempfile.NamedTemporaryFile(mode='w',suffix='.yaml') as f:
            yaml.safe_dump(options,f);f.flush()
            saved=self.parse(['--config',f.name])
        explicit=self.parse(['--max-num-seqs','2','--max-num-batched-tokens','4096','--max-model-len','65536','--kv-cache-dtype','octave_k3v4','--speculative-config','null','--override-generation-config','{"temperature":0.35,"top_p":0.81}'])
        from vllm.engine.arg_utils import AsyncEngineArgs
        a=policy.canonical(AsyncEngineArgs.from_cli_args(saved))
        b=policy.canonical(AsyncEngineArgs.from_cli_args(explicit))
        self.assertEqual(a,b)

    def test_real_compilation_config_identity(self):
        a=self.parse()
        encoded=policy.canonical(a.compilation_config)
        self.assertIsInstance(encoded,dict)
        self.assertEqual(policy.cache_identity(encoded,{},'x'),policy.cache_identity(encoded,{},'x'))

    def test_native_bindings(self):
        policy.verify_installation()
        import vllm._C,torch
        for name in ('octave_cache_store','octave_decode','octave_decode_sparse','octave_window_store','octave_rotate','octave_prefill'):
            self.assertTrue(hasattr(torch.ops._C,name),name)

    def test_child_defaults(self):
        import subprocess,sys
        r=subprocess.run([sys.executable,'-c',"import os; print('DEFAULT_CHECK',os.environ['HTR_QSA_TAU'],os.environ['HTR_GDN_XBLOCK1'])"],capture_output=True,text=True,check=True)
        self.assertIn('DEFAULT_CHECK 1 1',r.stdout)

if __name__=='__main__': unittest.main()
