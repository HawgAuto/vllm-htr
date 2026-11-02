"""Seed profiles must notice nested runtime package changes; no GPU imports."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class NestedRuntimeIdentity(unittest.TestCase):
    def test_nested_moe_source_changes_profile_identity(self):
        spec = importlib.util.spec_from_file_location('nested_profile_launcher', ROOT / 'launch_htr.py')
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        ap = m.parser(); args = ap.parse_args([]); m.check(ap, args)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['runtime-env.json', 'server-argv.json']:
                (root / name).write_text('{}')
            nested = root / 'python/htr_moe_serve/routes.py'
            nested.parent.mkdir(parents=True)
            nested.write_text('POLICY = "retained"\n')
            m.ROOT = root
            old = m.profile(args)[0]
            nested.write_text('POLICY = "temporary"\n')
            self.assertNotEqual(old, m.profile(args)[0], 'nested runtime edits must invalidate seeds')

if __name__ == '__main__':
    unittest.main()
