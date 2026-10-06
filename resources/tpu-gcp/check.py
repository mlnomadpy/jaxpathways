"""CPU integration checks. Requires JAX; creates only temporary local files."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

LAUNCHER = Path(__file__).with_name("launch.py")


class LauncherChecks(unittest.TestCase):
    def invoke(self, directory, *args, expected=0):
        result = subprocess.run([sys.executable, str(LAUNCHER), "--run-dir", str(directory), *args],
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_resume_matches_uninterrupted_run_and_keeps_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            full, split = (Path(temporary) / name for name in ("full", "split"))
            self.invoke(full, "--platform", "cpu", "--steps", "8")
            self.invoke(split, "--platform", "cpu", "--steps", "4")
            original = (split / "checkpoint.json").read_bytes()
            self.invoke(split, "--platform", "cpu", "--steps", "8", expected=2)
            self.assertEqual(original, (split / "checkpoint.json").read_bytes())
            self.invoke(split, "--platform", "cpu", "--steps", "8", "--resume")
            self.assertEqual(json.loads((full / "checkpoint.json").read_text()),
                             json.loads((split / "checkpoint.json").read_text()))
            self.assertEqual(original, (split / "history/attempt-0001/checkpoint.json").read_bytes())
            summary = json.loads((split / "summary.json").read_text())
            self.assertEqual(summary["updates"], 4)
            self.assertEqual(summary["last_step"], 8)
            before = (split / "run.json").read_bytes()
            self.invoke(split, "--platform", "cpu", "--steps", "8", "--resume", expected=2)
            self.assertEqual(before, (split / "run.json").read_bytes())

    def test_invalid_options_do_not_start_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "invalid"
            self.invoke(root, expected=2)
            for value in ("nan", "inf", "0", "901"):
                self.invoke(root, "--platform", "cpu", "--timeout", value, expected=2)
            self.invoke(root, "--platform", "cpu", "--resume", expected=2)
            self.assertFalse(root.exists())

    def test_timeout_is_failure_and_retains_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "timeout"
            self.invoke(root, "--platform", "cpu", "--timeout", "0.001", expected=1)
            self.assertEqual(json.loads((root / "run.json").read_text())["status"], "timed_out")


if __name__ == "__main__":
    unittest.main()
