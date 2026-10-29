"""No-device source-bound native build manifest checks."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class NativeBuild(unittest.TestCase):
    def test_complete_sanitized_source_closure_and_selected_math_flags(self):
        self.assertTrue((ROOT/'build_native.py').exists(),'portable native builder missing')
        spec=importlib.util.spec_from_file_location('build_native',ROOT/'build_native.py')
        assert spec is not None and spec.loader is not None
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        manifest=mod.validate(ROOT)
        self.assertEqual(len(manifest['modules']),11)
        byname={r['name']:r for r in manifest['modules']}
        pre=(ROOT/byname['hiphcp']['recipe']).read_text()
        dec=(ROOT/byname['hiphc']['recipe']).read_text()
        self.assertIn('-ffp-contract=off',pre)
        self.assertIn('-ffp-contract=fast',dec)
        for row in manifest['modules']:
            self.assertNotIn('/home/',(ROOT/row['recipe']).read_text())

    def test_changed_source_fails_before_compilation(self):
        self.assertTrue((ROOT/'build_native.py').exists(),'portable native builder missing')
        spec=importlib.util.spec_from_file_location('build_native',ROOT/'build_native.py')
        assert spec is not None and spec.loader is not None
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'bad').write_text('changed')
            with self.assertRaisesRegex(ValueError,'digest'):
                mod.check_files(root,{'bad':'0'*64})

if __name__=='__main__':unittest.main()
