"""CPU regression for the recommended front door over the frozen v0.2.0 launcher."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "temperature": 1.0, "top_p": 0.95, "top_k": 20,
    "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
}


def load(path):
    spec = importlib.util.spec_from_file_location("sampling_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def printed(path, flags):
    return json.loads(subprocess.check_output(
        [sys.executable, str(path), *flags, "--print-command"], text=True))


class RecommendedSampling(unittest.TestCase):
    def test_front_door_changes_only_generation_defaults(self):
        # The fallback also lets this regression demonstrate RED on v0.2.0.
        entry = ROOT / "launch_recommended.py"
        if not entry.is_file():
            entry = ROOT / "launch_htr.py"
        for flags in ([], ["--mtp", "0", "--max-num-seqs", "4"],
                      ["--context", "1000000", "--yarn-factor", "4",
                       "--kv-cache-gib", "27", "--build-seed"]):
            with self.subTest(flags=flags):
                original = printed(ROOT / "launch_htr.py", flags)
                actual = printed(entry, flags)
                self.assertIn("--override-generation-config", actual["argv"])
                index = actual["argv"].index("--override-generation-config")
                self.assertEqual(json.loads(actual["argv"][index + 1]), EXPECTED)
                self.assertEqual(actual["argv"][:index], original["argv"])
                self.assertEqual(len(actual["argv"]), index + 2)
                self.assertEqual(actual["profile"], original["profile"])
                self.assertEqual(actual["levers"], original["levers"])

    def test_frozen_launcher_identity_is_required(self):
        module = load(ROOT / "launch_recommended.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "launch_htr.py").write_text("raise AssertionError('untrusted')\n")
            with self.assertRaisesRegex(ValueError, "launcher identity mismatch"):
                module.load_release(root)

    def test_preflight_still_calls_original_checks_and_reports_effective_defaults(self):
        module = load(ROOT / "launch_recommended.py")
        release = module.load_release(ROOT)
        original_check = release.verify_runtime
        with patch.object(release, "verify_runtime", return_value=(123, {"ok": True}, {})) as check:
            module.apply_profile(release)
            self.assertEqual(release.verify_runtime(None), (123, {"ok": True}, EXPECTED))
            check.assert_called_once_with(None)
        release.verify_runtime = original_check
        with patch.object(release, "verify_runtime", side_effect=ValueError("integrity mismatch")):
            module.apply_profile(release)
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                release.verify_runtime(None)


if __name__ == "__main__":
    unittest.main()
