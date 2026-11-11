"""Successor policy tests; historical launcher tests remain unchanged."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('policy', ROOT/'python/htr_native_policy.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)

class NativePolicy(unittest.TestCase):
    def test_user_environment_wins(self):
        env = {'HTR_MOE_HPF3': '0', 'HTR_PLE_ROW_CACHE_BYTES': '123456'}
        policy.environment_defaults(env, root=ROOT, base_defaults={})
        self.assertEqual(env['HTR_MOE_HPF3'], '0')
        self.assertEqual(env['HTR_PLE_ROW_CACHE_BYTES'], '123456')
        self.assertEqual(env['HTR_QSA_TAU'], '1')

    def test_defaults_materialize_tmpdir_and_cache_roots(self):
        with tempfile.TemporaryDirectory() as base:
            base = Path(base)
            env = {'TMPDIR': str(base/'tmp'), 'AITER_JIT_DIR': str(base/'cache/aiter-jit'),
                   'XDG_CACHE_HOME': str(base/'cache/xdg')}
            policy.environment_defaults(env, root=ROOT, base_defaults={})
            for key in ('TMPDIR', 'AITER_JIT_DIR', 'XDG_CACHE_HOME'):
                self.assertTrue(Path(env[key]).is_dir(), key)
            # An existing directory is left alone and does not raise.
            policy.environment_defaults(env, root=ROOT, base_defaults={})
            self.assertTrue((base/'tmp').is_dir())
            # A path that cannot be created does not abort start-up.
            blocker = base/'file'
            blocker.write_text('')
            env = {'TMPDIR': str(base/'tmp'), 'AITER_JIT_DIR': str(blocker/'aiter-jit')}
            policy.environment_defaults(env, root=ROOT, base_defaults={})
            self.assertFalse((blocker/'aiter-jit').exists())

    def test_identity_separates_configuration(self):
        config = {'kv_cache_dtype': 'auto', 'max_num_seqs': 1, 'max_num_batched_tokens': 2048}
        a = policy.cache_identity(config, {'HTR_PLE_ROW_CACHE_BYTES':'0'}, 'rev1')
        for key, value in [('kv_cache_dtype','octave_k3v4'), ('max_num_seqs',4), ('max_num_batched_tokens',4096)]:
            self.assertNotEqual(a, policy.cache_identity(dict(config, **{key:value}), {'HTR_PLE_ROW_CACHE_BYTES':'0'}, 'rev1'))
        self.assertNotEqual(a, policy.cache_identity(config, {'HTR_PLE_ROW_CACHE_BYTES':'65536'}, 'rev1'))
        self.assertNotEqual(a, policy.cache_identity(config, {'HTR_PLE_ROW_CACHE_BYTES':'0'}, 'rev2'))
        self.assertEqual(a, policy.cache_identity(dict(reversed(list(config.items()))), {'HTR_PLE_ROW_CACHE_BYTES':'0'}, 'rev1'))

    def test_ple_budget(self):
        for value in [-1, 1, float('inf'), float('nan')]:
            with self.assertRaises(ValueError): policy.ple_budget(value)
        self.assertEqual(policy.ple_budget(0), 0)
        self.assertEqual(policy.ple_budget(65536), 65536)

    def test_packaged_kv_capabilities(self):
        import ast
        tree=ast.parse((ROOT.parent/'vllm/models/qwen4_exp/amd/qsa.py').read_text())
        values={n.target.id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id in ('_HTR_CACHE_DTYPES','_OCTAVE_CACHE_DTYPES')}
        self.assertEqual(set(values['_HTR_CACHE_DTYPES']),{'int8_per_token_head','int4_per_token_head'})
        self.assertEqual(set(values['_OCTAVE_CACHE_DTYPES']),{'octave_k3v4','octave_k3v3','octave_k3v3_compact'})

    def test_sampling_not_enforced(self):
        self.assertNotIn('override_generation_config', policy.DEFAULT_OVERRIDES)
        self.assertNotIn('generation_config', policy.DEFAULT_OVERRIDES)

if __name__ == '__main__': unittest.main()
