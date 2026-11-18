"""v0.2.4 release regressions (K6 routed-MoE decode, dense WMMA 9-15 rows, E59 zero-row fix), without GPU access."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class V024Runtime(unittest.TestCase):
    def setUp(self):
        self.policy = module('v024_policy_test', ROOT/'python/htr_native_policy.py')

    def test_release_version(self):
        self.assertEqual((ROOT/'VERSION').read_text().strip(), '0.2.4')

    def test_selected_stack_is_the_native_default(self):
        env = self.policy.environment_defaults({}, root=ROOT, base_defaults={})
        self.assertEqual(env['HTR_MOE_K6'], '1')
        self.assertEqual(env['HTR_MOE_K6_ROWS'], '2-4')
        self.assertEqual(env['HTR_DENSE_WMMA_MIN9'], '1')
        self.assertEqual(env['HTR_DENSE_WMMA'], '1')
        self.assertEqual(env['HTR_MOE_GROUPED'], '1')

    def test_explicit_opt_out_wins(self):
        env = {'HTR_MOE_K6': '0', 'HTR_DENSE_WMMA_MIN9': '0'}
        self.policy.environment_defaults(env, root=ROOT, base_defaults={})
        self.assertEqual(env['HTR_MOE_K6'], '0')
        self.assertEqual(env['HTR_DENSE_WMMA_MIN9'], '0')

    def test_no_launcher_pins_a_sampling_seed(self):
        env = json.loads((ROOT/'runtime-env.json').read_text())
        self.assertFalse([k for k in env if 'SEED' in k.upper()])
        self.assertNotIn('seed', json.loads((ROOT/'sampling.json').read_text()))
        self.assertNotIn('--seed', json.loads((ROOT/'server-argv.json').read_text()).__repr__())

    def test_k6_native_module_is_pinned(self):
        build = json.loads((ROOT/'NATIVE-BUILD.json').read_text())
        k6 = [m for m in build['modules'] if m['name'] == 'htr_moe_k6']
        self.assertEqual(len(k6), 1)
        k6 = k6[0]
        self.assertEqual(k6['output'], 'native-bin/htr_moe_k6.so')
        self.assertEqual(k6['accepted_binary_sha256'],
                         '830f49e0de6ec4a47a642cad661516d0091efe8ce30cac2abb49666e565e8ca0')
        for name in k6['sources'] + [k6['recipe']]:
            self.assertIn(name, build['files'])
        for name, digest in build['files'].items():
            if name.startswith('native-source/moe2/k6/') or name == k6['recipe']:
                self.assertEqual(sha(ROOT/name), digest, name)
        native = module('v024_build_native', ROOT/'build_native.py')
        self.assertIn('htr_moe_k6', native.EXPECTED)

    def test_native_build_closure_with_k6(self):
        # Successor of test_native_build's 11-module literal: same checks over 12 modules.
        native = module('v024_build_native_closure', ROOT/'build_native.py')
        manifest = native.validate(ROOT)
        self.assertEqual(len(manifest['modules']), 12)
        byname = {r['name']: r for r in manifest['modules']}
        self.assertIn('htr_moe_k6', byname)
        self.assertIn('-ffp-contract=off', (ROOT/byname['hiphcp']['recipe']).read_text())
        self.assertIn('-ffp-contract=fast', (ROOT/byname['hiphc']['recipe']).read_text())
        for row in manifest['modules']:
            self.assertNotIn('/home/', (ROOT/row['recipe']).read_text())

    def test_installer_verifies_and_binds_k6(self):
        text = (ROOT/'install_optimized.py').read_text()
        self.assertIn("/payload/native/htr_moe_k6.so", text)
        self.assertIn("'K6 native identity'", text)
        self.assertIn("ROOT/k6['output']", text)

    def test_k6_loader_default_path(self):
        text = (ROOT/'python/htr_moe_serve/kernels.py').read_text()
        self.assertIn("'htr_moe_k6.so'", text)
        self.assertIn("HTR_NATIVE_DIR", text)

    def test_grouped_bindings_cover_k6_python(self):
        grouped = json.loads((ROOT/'python/htr_moe_serve/grouped-bindings.json').read_text())
        for name in ('plugin.py', 'kernels.py'):
            self.assertEqual(grouped['files']['/opt/vllm-htr/python/htr_moe_serve/' + name],
                             sha(ROOT/'python/htr_moe_serve'/name))

    def test_dense_wmma_row_floor(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest('torch not available')
        wmma = module('v024_dense_wmma', ROOT/'python/htr/dense_wmma.py')
        with mock.patch.dict(os.environ, {'HTR_DENSE_WMMA_MIN9': '1'}):
            self.assertEqual(wmma.min_m(), 9)
        with mock.patch.dict(os.environ, {'HTR_DENSE_WMMA_MIN9': '0'}):
            self.assertEqual(wmma.min_m(), 16)
        with mock.patch.dict(os.environ, {'HTR_DENSE_WMMA_MIN9': '2'}):
            with self.assertRaises(ValueError):
                wmma.min_m()

    def test_zero_ple_row_limit_disables_async_path(self):
        env = {'HTR_PLE_ASYNC_MAXN_SOURCE': 'user', 'HTR_PLE_ASYNC_MAXN': '0', 'E59_PLE_DEV': '1'}
        self.assertEqual(self.policy.resolve_ple_maxn(3, env), 0)
        self.assertEqual(env['E59_PLE_DEV'], '0')
        env = {'HTR_PLE_ASYNC_MAXN_SOURCE': 'default', 'HTR_PLE_ASYNC_MAXN': 'auto', 'E59_PLE_DEV': '1'}
        self.assertEqual(self.policy.resolve_ple_maxn(3, env), 16)
        self.assertEqual(env['E59_PLE_DEV'], '1')


if __name__ == '__main__':
    unittest.main()
