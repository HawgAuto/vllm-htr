"""Public quantized-KV launch contract; no model/GPU needed for these tests."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class QuantizedLaunch(unittest.TestCase):
    def setUp(self):
        path=ROOT/'launch_quantized.py'
        self.assertTrue(path.is_file(), 'v0.2.1 quantized launcher is missing')
        spec=importlib.util.spec_from_file_location('quantized_release',path)
        self.mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.mod)
        self.base=self.mod.load_release(ROOT)

    def args(self, dtype, context, mtp, pool, factor=None):
        argv=['--kv-cache-dtype',dtype,'--context',str(context),'--mtp',str(mtp),
              '--kv-cache-gib',str(pool),'--max-num-batched-tokens','2048']
        if factor is not None:argv+=['--yarn-factor',str(factor)]
        parser=self.base.parser();args=parser.parse_args(argv);self.base.check(parser,args)
        return args

    def test_all_four_qualified_profiles_build_real_native_arguments(self):
        for dtype,context,mtp,pool,factor in [
            ('int8_per_token_head',2000000,0,27.75,8),
            ('int8_per_token_head',2000000,3,27.75,8),
            ('int4_per_token_head',3000000,3,22.375,16),
            ('int4_per_token_head',4000000,0,26.875,16)]:
            with self.subTest(dtype=dtype,mtp=mtp):
                a=self.args(dtype,context,mtp,pool)
                v=self.base.command(a)
                self.assertEqual(v[2],'htr_capacity_api')
                self.assertEqual(v[v.index('--kv-cache-dtype')+1],dtype)
                self.assertEqual(v[v.index('--max-model-len')+1],str(context))
                self.assertEqual(int(v[v.index('--kv-cache-memory-bytes')+1]),round(pool*2**30))
                self.assertEqual(a.yarn_factor,factor)
                if mtp:
                    s=json.loads(v[v.index('--speculative-config')+1])
                    self.assertEqual((s['method'],s['num_speculative_tokens'],s['max_model_len']),('mtp',3,context))
                else:self.assertNotIn('--speculative-config',v)
                self.assertEqual(json.loads(v[v.index('--override-generation-config')+1])['temperature'],1.0)

    def test_failed_3point5m_mtp3_is_not_a_released_profile(self):
        with self.assertRaises(SystemExit):self.args('int4_per_token_head',3500000,3,26,16)

    def test_int8_and_bf16_limits_are_not_silently_widened(self):
        for dtype,ctx in [('int8_per_token_head',3000000),('auto',2000000)]:
            with self.subTest(dtype=dtype),self.assertRaises(SystemExit):self.args(dtype,ctx,3,27.75,16)

    def test_long_context_keeps_single_slot_chunk_and_envelope_checks(self):
        for extra in [['--max-num-seqs','2'],['--max-num-batched-tokens','12288'],['--yarn-factor','4']]:
            p=self.base.parser();a=p.parse_args(['--kv-cache-dtype','int4_per_token_head','--context','3000000','--max-num-batched-tokens','2048']+extra)
            with self.subTest(extra=extra),self.assertRaises(SystemExit):self.base.check(p,a)

    def test_quantized_and_bf16_compile_profiles_are_distinct(self):
        names=[]
        for dtype in ['auto','int8_per_token_head','int4_per_token_head']:
            a=self.args(dtype,32768,3,2);name,levers=self.base.profile(a)
            self.assertEqual(levers['main_kv'],dtype);names.append(name)
        self.assertEqual(len(set(names)),3)

    def test_current_distribution_version(self):
        self.assertEqual((ROOT/'VERSION').read_text().strip(), '0.2.1')

    def test_default_bf16_argv_stays_on_original_entrypoint(self):
        p=self.base.parser();a=p.parse_args([]);self.base.check(p,a);v=self.base.command(a)
        self.assertEqual(v[2],'vllm.entrypoints.openai.api_server')
        self.assertNotIn('--kv-cache-dtype',v)
        self.assertEqual(a.context,262144)

    def test_yarn_preserves_config_objects_and_mrope_fields(self):
        path=ROOT/'python/htr_capacity_rope.py'
        self.assertTrue(path.is_file())
        spec=importlib.util.spec_from_file_location('capacity_rope',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        class Config:
            model_type='qwen4_exp_text';num_hidden_layers=48;hidden_size=2560;architectures=['Qwen4ExpForCausalLM']
            def __init__(self):self.rope_parameters={'rope_type':'default','mrope_section':[11,11,10],'mrope_interleaved':True,'rope_theta':10000000}
            def get_text_config(self):return self
        c=Config();self.assertIs(m.apply_yarn(c,factor=16),c)
        self.assertEqual(c.rope_parameters['mrope_section'],[11,11,10])
        self.assertEqual(c.max_position_embeddings,16*262144)
        self.assertEqual(c.original_max_position_embeddings,262144)

if __name__=='__main__':unittest.main()
