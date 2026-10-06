"""Behavioral checks for provenance and non-destructive run creation."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project"
        shutil.copytree(SOURCE, self.root, ignore=shutil.ignore_patterns("runs", "__pycache__", ".git", ".venv"))

    def run_cli(self, name="baseline-01", config="configs/baseline.json"):
        return subprocess.run(
            [sys.executable, str(self.root / "run.py"), "prepare", "--config", config, "--run-id", name],
            cwd=self.temp.name, capture_output=True, text=True,
        )

    def test_snapshot_survives_changed_inputs_and_refuses_overwrite(self):
        self.assertEqual(self.run_cli().returncode, 0)
        run = self.root / "runs/baseline-01"
        original = (run / "manifest.json").read_bytes()
        manifest = json.loads(original)
        self.assertEqual(manifest["status"], "prepared")
        self.assertEqual(manifest["data_sha256"], hashlib.sha256((run / "data.snapshot").read_bytes()).hexdigest())
        (self.root / "data/tiny.csv").write_text("changed data\n")
        self.assertNotEqual(self.run_cli().returncode, 0)
        self.assertEqual((run / "manifest.json").read_bytes(), original)
        self.assertEqual(self.run_cli("changed-data").returncode, 0)
        changed = json.loads((self.root / "runs/changed-data/manifest.json").read_text())
        self.assertNotEqual(changed["data_sha256"], manifest["data_sha256"])
        for path, expected in manifest["source_sha256"].items():
            self.assertEqual(hashlib.sha256((run / "source" / path).read_bytes()).hexdigest(), expected)

    def test_rejects_invalid_configuration_and_path_escape(self):
        for config in ([], {"seed": 7}, {"seed": 7, "steps": 0, "learning_rate": .01, "data": "data/tiny.csv"},
                       {"seed": 7, "steps": 20, "learning_rate": .01, "data": "../outside.csv"}):
            (self.root / "configs/bad.json").write_text(json.dumps(config))
            self.assertNotEqual(self.run_cli(config="configs/bad.json").returncode, 0)
            self.assertFalse((self.root / "runs/baseline-01").exists())
        self.assertNotEqual(self.run_cli("../escape").returncode, 0)


if __name__ == "__main__":
    unittest.main()
