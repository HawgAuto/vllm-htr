"""Successor composition gates; prior release tests stay byte-preserved."""
import ast
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class CombinedRuntime(unittest.TestCase):
    def test_native_defaults_include_kept_filter_and_preserve_override(self):
        p = ROOT / 'python/htr_native_policy.py'
        self.assertTrue(p.is_file(), 'native policy not folded')
        spec = importlib.util.spec_from_file_location('combined_policy', p)
        policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy)
        env = policy.environment_defaults({}, root=ROOT, base_defaults={})
        self.assertEqual(env.get('HTR_Q38F'), 'on')
        self.assertIn('htr_q38f', env['VLLM_PLUGINS'].split(','))
        env = policy.environment_defaults({'HTR_Q38F': 'off'}, root=ROOT, base_defaults={})
        self.assertEqual(env['HTR_Q38F'], 'off')
        self.assertNotIn('generation_config', policy.DEFAULT_OVERRIDES)
        self.assertNotIn('override_generation_config', policy.DEFAULT_OVERRIDES)

    def test_octave_is_packed_and_never_uses_bf16_main_prepare(self):
        p = ROOT.parent / 'vllm/models/qwen4_exp/amd/qsa.py'
        tree = ast.parse(p.read_text())
        values = {n.target.id: ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                  and n.target.id in ('_OCTAVE_CACHE_DTYPES', '_HTR_CACHE_DTYPES')}
        self.assertEqual(set(values.get('_OCTAVE_CACHE_DTYPES', [])),
                         {'octave_k3v4','octave_k3v3','octave_k3v3_compact'})
        self.assertIn('qsa_main_prepare', p.read_text())
        tests = [ast.unparse(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Attribute) and t.attr == 'use_fused_qsa_prepare' for t in n.targets)]
        self.assertTrue(any('_PACKED_CACHE_DTYPES' in s and 'HTR_QSA_MAIN_PREPARE' in s for s in tests),
                        'main prepare must exclude every packed KV format')
        defs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        self.assertEqual(defs.count('do_kv_cache_update'), 1)

    def test_filter_source_and_plugin_are_in_runtime_payload(self):
        self.assertTrue((ROOT/'native-source/q38filter/q38_filter_hip.hip').is_file())
        entry = ROOT/'python/htr_q38f-0.1.0.dist-info/entry_points.txt'
        self.assertTrue(entry.is_file(), 'kept filter plugin missing')
        self.assertIn('htr_q38f = htr_q38f:register', entry.read_text())
        self.assertTrue((ROOT/'python/htr_q38f/q38f.py').is_file())

    def test_kept_plugin_roster_and_retired_paths(self):
        python = ROOT / 'python'
        for name in ['direct_qsa_union_adapter_v1.py', 'e95_qsa_1600.py', 'e82_graphproof.py', 'e94_geometry.py']:
            self.assertFalse((python / name).exists(), name)
        self.assertNotIn('DQ_CACHE', (python / 'htr/ops.py').read_text())
        self.assertEqual(sorted(p.name for p in python.glob('*.dist-info')),
                         ['gp8hc-0.1.0.dist-info', 'htr_autotune_pin-0.1.0.dist-info',
                          'htr_moe_serve-0.1.0.dist-info', 'htr_q38f-0.1.0.dist-info',
                          'htr_toolbox-0.1.0.dist-info', 'q38spec-0.1.0.dist-info'])

    def test_source_builder_hashes_bytes_and_rejects_symlinks(self):
        import tempfile
        p = ROOT / 'build_combined.py'
        self.assertTrue(p.is_file(), 'portable combined builder missing')
        spec = importlib.util.spec_from_file_location('combined_builder', p)
        assert spec is not None and spec.loader is not None
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'input').write_bytes(b'original')
            first = builder.tree_manifest(root)
            (root/'input').write_bytes(b'changed')
            self.assertNotEqual(first, builder.tree_manifest(root))
            (root/'alias').symlink_to('input')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                builder.tree_manifest(root)

    def test_successor_distribution_version(self):
        self.assertEqual((ROOT/'VERSION').read_text().strip(), '0.2.2')

if __name__ == '__main__':
    unittest.main()
