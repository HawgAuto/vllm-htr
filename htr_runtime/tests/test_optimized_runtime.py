"""Release defaults and binding-repair regressions, without GPU access."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


class OptimizedRuntime(unittest.TestCase):
    def setUp(self):
        self.policy = module('optimized_policy_test', ROOT/'python/htr_native_policy.py')

    def test_release_version(self):
        self.assertEqual((ROOT/'VERSION').read_text().strip(), '0.2.3')

    def test_winners_are_native_defaults_without_ple_weight_cache(self):
        env = self.policy.environment_defaults({}, root=ROOT, base_defaults={})
        self.assertEqual(env.get('HTR_MOE_GROUPED'), '1')
        self.assertEqual(env.get('HTR_DENSE_WMMA'), '1')
        self.assertEqual(env['HTR_PLE_ROW_CACHE_BYTES'], '0')
        self.assertNotIn('HTR_DENSE_GROUP_C16', env)
        self.assertNotIn('HTR_MOE_C13', env)
        self.assertNotIn('HTR_DENSE_WMMA_C10', env)

    def test_explicit_disables_win(self):
        env = {'HTR_MOE_GROUPED':'0', 'HTR_DENSE_WMMA':'0', 'HTR_PLE_ROW_CACHE_BYTES':'65536'}
        self.policy.environment_defaults(env, root=ROOT, base_defaults={})
        self.assertEqual(env['HTR_MOE_GROUPED'], '0')
        self.assertEqual(env['HTR_DENSE_WMMA'], '0')
        self.assertEqual(env['HTR_PLE_ROW_CACHE_BYTES'], '65536')

    def resolve(self, mtp, env):
        self.assertTrue(callable(getattr(self.policy, 'resolve_ple_maxn', None)),
                        'mode-dependent PLE resolution is missing')
        return self.policy.resolve_ple_maxn(mtp, env)

    def test_ple_default_tracks_parsed_mtp_mode(self):
        for mtp, expected in [(0,32), (3,16)]:
            env = self.policy.environment_defaults({}, root=ROOT, base_defaults={})
            self.assertEqual(self.resolve(mtp, env), expected)
            self.assertEqual(env['E59_PLE_DEV_MAXN'], str(expected))

    def test_ple_user_override_including_zero(self):
        for value in ['0','4','64']:
            env = self.policy.environment_defaults({'HTR_PLE_ASYNC_MAXN':value}, root=ROOT, base_defaults={})
            self.assertEqual(self.resolve(3, env), int(value))
            self.assertEqual(env['E59_PLE_DEV_MAXN'], value)

    def test_legacy_ple_override_is_preserved(self):
        env = self.policy.environment_defaults({'E59_PLE_DEV_MAXN':'8'}, root=ROOT, base_defaults={})
        self.assertEqual(self.resolve(0, env), 8)

    def test_canonical_ple_override_wins_over_legacy_alias(self):
        env = self.policy.environment_defaults({'HTR_PLE_ASYNC_MAXN':'16','E59_PLE_DEV_MAXN':'4'}, root=ROOT, base_defaults={})
        self.assertEqual(self.resolve(0, env), 16)

    def test_ple_invalid_budget_rejected(self):
        for value in ['-1','1.5','nan','']:
            env = self.policy.environment_defaults({'HTR_PLE_ASYNC_MAXN':value}, root=ROOT, base_defaults={})
            self.assertTrue(callable(getattr(self.policy, 'resolve_ple_maxn', None)))
            with self.assertRaises(ValueError):
                self.policy.resolve_ple_maxn(0, env)

    def test_ple_child_inheritance_preserves_mode_default_provenance(self):
        env = self.policy.environment_defaults({}, root=ROOT, base_defaults={})
        self.resolve(0, env)
        self.policy.environment_defaults(env, root=ROOT, base_defaults={})
        self.assertEqual(self.resolve(3, env), 16)

    def test_grouped_payload_is_folded_and_default_on(self):
        p = ROOT/'python/htr_moe_serve/grouped.py'
        self.assertTrue(p.is_file(), 'grouped MoE serving implementation is missing')
        text = p.read_text()
        self.assertIn("os.environ.get('HTR_MOE_GROUPED','1')", text)
        self.assertIn('from . import grouped', (ROOT/'python/htr_moe_serve/plugin.py').read_text())
        self.assertNotIn('c13_serving', text)

    def test_dense_dispatch_payload_is_folded(self):
        p = ROOT/'python/htr/dense_wmma.py'
        self.assertTrue(p.is_file(), 'selected dense WMMA implementation is missing')
        text = p.read_text()
        self.assertIn("os.environ.get('HTR_DENSE_WMMA', '1')", text)
        self.assertIn('from . import dense_wmma', (ROOT/'python/htr/ops.py').read_text())
        self.assertIn('native-bin/htr_dense_wmma.so', text)
        self.assertTrue((ROOT/'native-source/dense-wmma/wm.hip').is_file())

    def bindings(self):
        p = ROOT/'release_bindings.py'
        self.assertTrue(p.is_file(), 'checked binding reconciliation is missing')
        return module('release_bindings_test', p)

    def test_repair_only_corroborated_changed_binding(self):
        b = self.bindings()
        before = {'qsa':'old', 'other':'same'}
        actual = {'qsa':'new', 'other':'same'}
        self.assertEqual(b.reconcile(before, actual, {'qsa':'new'}, {'qsa'}), actual)
        self.assertEqual(before, {'qsa':'old', 'other':'same'})

    def test_binding_corruption_is_not_blessed(self):
        b = self.bindings()
        cases = [({'qsa':'new','other':'bad'}, {'qsa':'new'}, {'qsa'}),
                 ({'qsa':'bad','other':'same'}, {'qsa':'new'}, {'qsa'}),
                 ({'qsa':'new','other':'same'}, {}, {'qsa'}),
                 ({'qsa':'new','other':'same'}, {'qsa':'new'}, set()),
                 ({'qsa':'new'}, {'qsa':'new'}, {'qsa'})]
        for actual, corroborated, allowed in cases:
            with self.assertRaises(ValueError):
                b.reconcile({'qsa':'old','other':'same'}, actual, corroborated, allowed)


if __name__ == '__main__':
    unittest.main()
