"""Public source-build contract, with no container or GPU execution."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


class OptimizedBuilder(unittest.TestCase):
    def load(self):
        path = ROOT/'build_optimized.py'
        self.assertTrue(path.is_file(), 'optimized release builder is missing')
        spec = importlib.util.spec_from_file_location('optimized_builder_test', path)
        assert spec is not None and spec.loader is not None
        obj = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, 'path', [str(ROOT), *sys.path]):
            spec.loader.exec_module(obj)
        return obj

    def test_immutable_shipped_base(self):
        builder = self.load()
        self.assertEqual(builder.BASE, 'ghcr.io/hawgauto/vllm-htr@sha256:034898ebd182a240fa403ad5ea24bea3f647d933571e43420a27daf9af079d1c')

    def test_help_does_not_build(self):
        builder = self.load()
        with mock.patch.object(sys,'argv',['build_optimized.py','--help']), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            builder.main()
        self.assertEqual(stopped.exception.code, 0)

    def test_legacy_entrypoint_delegates_to_current_builder(self):
        builder = self.load()
        import types
        forwarded = mock.Mock()
        legacy = sys.modules['build_combined']
        with mock.patch.dict(sys.modules, {'build_optimized':types.SimpleNamespace(main=forwarded)}):
            try:
                legacy.main()
            except SystemExit as error:
                self.fail('legacy builder parsed old arguments instead of forwarding: '+str(error))
        forwarded.assert_called_once_with()

    def test_build_uses_verbatim_native_recipe_and_bound_sources(self):
        builder = self.load()
        manifest = json.loads((ROOT/'DENSE-WMMA-BUILD.json').read_text())
        self.assertEqual(manifest['target'], 'gfx1151')
        for name, expected in manifest['files'].items():
            self.assertEqual(builder.sha(ROOT/name), expected)
        recipe = (ROOT/'native-recipes/htr_dense_wmma.ninja.in').read_text()
        self.assertIn('cuda_compile /source/wm.hip', recipe)
        self.assertIn('--offload-arch=gfx1151', recipe)


if __name__ == '__main__':
    unittest.main()
