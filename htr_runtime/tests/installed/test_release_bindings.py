"""Installed release integrity, usable on a no-device image."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path('/opt/vllm-htr')
BASE = Path('/opt/haloloom/qwen38/data/installed-files.json')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class InstalledReleaseBindings(unittest.TestCase):
    def test_every_quantized_binding_matches_installed_bytes(self):
        bindings = json.loads((ROOT/'quantized-kv-bindings.json').read_text())
        self.assertGreater(len(bindings), 0)
        stale = {p for p, expected in bindings.items() if sha(p) != expected}
        self.assertEqual(stale, set())

    def test_runtime_and_native_policy_manifests_remain_consistent(self):
        for name, expected in json.loads((ROOT/'installed-files.json').read_text()).items():
            self.assertEqual(sha(ROOT/name), expected, name)
        for name, expected in json.loads((ROOT/'native-policy-bindings.json').read_text()).items():
            self.assertEqual(sha(name), expected, name)
        for row in json.loads(BASE.read_text())['files']:
            self.assertEqual(sha(row['installed_path']), row['sha256'], row['installed_path'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
