"""Release wiring/cache-boundary tests; no model or device initialization."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("htr_seed_policy_launcher", ROOT / "launch_htr.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class ReleaseSeedPolicy(unittest.TestCase):
    def args(self):
        ap = launcher.parser()
        args = ap.parse_args([])
        launcher.check(ap, args)
        return args

    def test_release_version_is_separate_from_upstream_vllm(self):
        self.assertTrue((ROOT / "VERSION").is_file(), "HTR version file missing")
        self.assertEqual((ROOT / "VERSION").read_text().strip(), "0.2.0")

    def test_default_launch_enables_the_source_bound_policy(self):
        env = json.loads((ROOT / "runtime-env.json").read_text())
        self.assertEqual(env.get("HTR_GDN_XBLOCK1"), "1")

    def test_worker_bootstrap_is_part_of_the_install_contract(self):
        bootstrap = ROOT / "python/zy_htr_gdn_pin.pth"
        self.assertTrue(bootstrap.is_file(), "compiler-worker bootstrap missing")
        self.assertEqual(bootstrap.read_text(), "import htr_gdn_pin; htr_gdn_pin.install()\n")
        self.assertIn("*.pth", (ROOT / "install_layer.py").read_text())

    def test_cache_profile_changes_with_runtime_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for n in ["runtime-env.json", "server-argv.json"]:
                shutil.copyfile(ROOT / n, root / n)
            (root / "VERSION").write_text("0.2.0\n")
            (root / "python").mkdir()
            source = root / "python/htr_gdn_pin.py"
            source.write_text("POLICY = 'original'\n")
            with patch.object(launcher, "ROOT", root):
                old = launcher.profile(self.args())[0]
                source.write_text("POLICY = 'changed'\n")
                self.assertNotEqual(old, launcher.profile(self.args())[0])

    def test_cache_profile_changes_with_distribution_version(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for n in ["runtime-env.json", "server-argv.json"]:
                shutil.copyfile(ROOT / n, root / n)
            version = root / "VERSION"
            version.write_text("0.1.0\n")
            with patch.object(launcher, "ROOT", root):
                old = launcher.profile(self.args())[0]
                version.write_text("0.2.0\n")
                self.assertNotEqual(old, launcher.profile(self.args())[0])


if __name__ == "__main__":
    unittest.main()
