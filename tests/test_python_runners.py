"""Fast orchestration regressions; fixtures never require the JAX training stack."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_runner(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = io.StringIO()
        self.error = io.StringIO()
        self.enterContext(redirect_stdout(self.output))
        self.enterContext(redirect_stderr(self.error))

    def write_json(self, relative, value):
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(value))
        return destination

    def assert_argument_error(self, runner, arguments, message):
        with self.assertRaises(SystemExit) as raised:
            runner.main(arguments)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(message, self.error.getvalue())


class ProjectRunnerTest(RunnerTest):
    def setUp(self):
        super().setUp()
        self.runner = load_runner("check-projects")
        self.runner.ROOT = self.root
        paths = []
        for project_id in ("first", "second", "third"):
            relative = f"projects/{project_id}/project.json"
            paths.append(relative)
            self.write_json(
                relative,
                {
                    "id": project_id,
                    "referenceCommand": f"python projects/{project_id}/check.py",
                },
            )
            (self.root / "projects" / project_id / "check.py").write_text(
                f'print("checked {project_id}")\n'
            )
        self.write_json("curriculum/projects.json", paths)
        self.canonical = self.write_json(
            "curriculum/project-validation.json", {"existing": True}
        )

    def test_focused_execution_checks_only_requested_projects_and_preserves_full_receipt(
        self,
    ):
        self.runner.main(["--project", "second", "--receipt", "focused/receipt.json"])
        receipt = json.loads((self.root / "focused/receipt.json").read_text())
        self.assertEqual(
            [record["projectId"] for record in receipt["projects"]], ["second"]
        )
        self.assertEqual(receipt["projects"][0]["stdout"], "checked second\n")
        self.assertEqual(json.loads(self.canonical.read_text()), {"existing": True})

    def test_list_can_filter_then_shard_without_execution_or_receipt(self):
        with patch.object(self.runner.subprocess, "run") as run:
            self.runner.main(
                ["--project", "third", "--project", "first", "--shard", "1/2", "--list"]
            )
        run.assert_not_called()
        self.assertEqual(self.output.getvalue(), "third\n")
        self.assertEqual(json.loads(self.canonical.read_text()), {"existing": True})

    def test_partial_execution_requires_receipt(self):
        for arguments in (["--shard", "0/4"], ["--project", "first"]):
            with self.subTest(arguments=arguments), patch.object(
                self.runner.subprocess, "run"
            ) as run:
                self.assert_argument_error(self.runner, arguments, "require --receipt")
                run.assert_not_called()
        self.assertEqual(json.loads(self.canonical.read_text()), {"existing": True})

    def test_partial_receipt_cannot_replace_complete_receipt_through_path_aliases(self):
        alias = self.root / "receipt-alias.json"
        alias.symlink_to(self.canonical)
        for selection in (["--project", "first"], ["--shard", "0/2"]):
            for receipt in (
                "curriculum/project-validation.json",
                str(self.canonical),
                "curriculum/../curriculum/project-validation.json",
                str(alias),
            ):
                with self.subTest(selection=selection, receipt=receipt), patch.object(
                    self.runner.subprocess, "run"
                ) as run:
                    self.assert_argument_error(
                        self.runner,
                        [*selection, "--receipt", receipt],
                        "must use a separate --receipt path",
                    )
                    run.assert_not_called()
        self.assertEqual(json.loads(self.canonical.read_text()), {"existing": True})

    def test_unknown_project_and_invalid_shard_fail_before_execution(self):
        for arguments, message in (
            (["--project", "typo", "--list"], "unknown project IDs"),
            (["--shard", "2/2", "--list"], "0 <= index < count"),
        ):
            with self.subTest(arguments=arguments), patch.object(
                self.runner.subprocess, "run"
            ) as run:
                self.assert_argument_error(self.runner, arguments, message)
                run.assert_not_called()

    def test_streaming_hash_preserves_receipt_digest_and_ignores_bytecode(self):
        directory = self.root / "projects/first"
        (directory / "large.bin").write_bytes(b"x" * (1024 * 1024 + 17))
        (directory / "__pycache__").mkdir()
        (directory / "__pycache__/ignored.pyc").write_bytes(b"ignored")
        expected = hashlib.sha256()
        for source in sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ):
            expected.update(
                str(source.relative_to(self.root)).encode()
                + b"\0"
                + source.read_bytes()
                + b"\0"
            )
        self.assertEqual(self.runner.source_hash(directory), expected.hexdigest())


class LessonRunnerTest(RunnerTest):
    def setUp(self):
        super().setUp()
        self.runner = load_runner("smoke-lessons")
        self.runner.ROOT = self.root
        lessons = []
        for lesson_id in ("one", "two", "three"):
            relative = f"phases/first/{lesson_id}"
            lessons.append(
                {
                    "id": lesson_id,
                    "status": "authored",
                    "path": relative,
                    "contentHash": f"{lesson_id}-current",
                    "content": {},
                    "artifacts": {
                        "scriptSource": f"{relative}/main.py",
                        "notebookSource": f"{relative}/exercise.ipynb",
                    },
                }
            )
            (self.root / relative / "outputs").mkdir(parents=True)
            (self.root / relative / "main.py").write_text('print("lesson result")\n')
        self.write_json(
            "public/curriculum.json",
            {
                "phases": [
                    {"id": "first", "lessons": lessons},
                    {"id": "empty", "lessons": []},
                ]
            },
        )
        self.canonical = self.write_json(
            "curriculum/validation.json",
            {
                "lessons": [
                    {"lessonId": "two", "contentHash": "two-current"},
                    {"lessonId": "three", "contentHash": "three-stale"},
                ]
            },
        )

    def simulate_execution(self, command, **kwargs):
        if command[1] == "-c":
            return subprocess.CompletedProcess(
                command,
                0,
                'lesson result\nCOURSE_RUNTIME:{"backend":"cpu","deviceCount":1}\n',
                "",
            )
        Path(command[-1]).write_text(
            json.dumps(
                {
                    "runtime": {"backend": "cpu", "deviceCount": 1},
                    "notebookCells": [{"cell": 0, "status": "passed"}],
                }
            )
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    def run_lessons(self, arguments):
        packages = {
            "jax": SimpleNamespace(__version__="test"),
            "numpy": SimpleNamespace(__version__="test"),
        }
        with patch.object(
            self.runner.subprocess, "run", side_effect=self.simulate_execution
        ) as run, patch.object(
            self.runner.importlib.metadata, "version", return_value="test"
        ) as version, patch.dict(
            sys.modules, packages
        ):
            self.runner.main(arguments)
        return run, version

    def test_selection_listing_requires_no_scientific_packages_or_execution(self):
        with patch.object(self.runner.subprocess, "run") as run, patch.object(
            self.runner.importlib.metadata, "version"
        ) as version:
            self.runner.main(["--phase", "first", "--lesson", "two", "--list"])
        run.assert_not_called()
        version.assert_not_called()
        self.assertEqual(self.output.getvalue(), "one\ntwo\nthree\n")

    def test_invalid_or_empty_selection_fails_before_execution(self):
        for arguments, message in (
            (["--lesson", "typo"], "unknown or unauthored lesson"),
            (["--phase", "typo"], "unknown phase"),
            (["--phase", "empty"], "no authored lessons"),
        ):
            with self.subTest(arguments=arguments), patch.object(
                self.runner.subprocess, "run"
            ) as run:
                self.assert_argument_error(self.runner, arguments, message)
                run.assert_not_called()
        self.assertEqual(len(json.loads(self.canonical.read_text())["lessons"]), 2)

    def test_explicit_partial_receipt_cannot_replace_complete_receipt_through_path_aliases(
        self,
    ):
        original = self.canonical.read_bytes()
        alias = self.root / "receipt-alias.json"
        alias.symlink_to(self.canonical)
        for receipt in (
            "curriculum/validation.json",
            str(self.canonical),
            "curriculum/../curriculum/validation.json",
            str(alias),
        ):
            with self.subTest(receipt=receipt), patch.object(
                self.runner.subprocess, "run"
            ) as run, patch.object(
                self.runner.importlib.metadata, "version"
            ) as version:
                self.assert_argument_error(
                    self.runner,
                    ["--lesson", "one", "--receipt", receipt],
                    "must use a separate --receipt path",
                )
                run.assert_not_called()
                version.assert_not_called()
        self.assertEqual(self.canonical.read_bytes(), original)

    def test_focused_default_receipt_keeps_only_unchanged_previous_results(self):
        run, _ = self.run_lessons(["--lesson", "one"])
        self.assertEqual(run.call_count, 2)
        receipt = json.loads(self.canonical.read_text())
        self.assertEqual(
            [record["lessonId"] for record in receipt["lessons"]], ["one", "two"]
        )
        self.assertTrue(
            (self.root / "phases/first/one/outputs/execution.json").is_file()
        )
        self.assertFalse(
            (self.root / "phases/first/two/outputs/execution.json").exists()
        )

    def test_separate_receipt_contains_only_selected_results(self):
        original = self.canonical.read_bytes()
        run, _ = self.run_lessons(
            ["--lesson", "one", "--receipt", "focused/receipt.json"]
        )
        self.assertEqual(run.call_count, 2)
        receipt = json.loads((self.root / "focused/receipt.json").read_text())
        self.assertEqual([record["lessonId"] for record in receipt["lessons"]], ["one"])
        self.assertEqual(self.canonical.read_bytes(), original)

    def test_full_run_executes_all_lessons_and_reads_package_versions_once(self):
        run, version = self.run_lessons([])
        self.assertEqual(run.call_count, 6)
        self.assertEqual(version.call_count, 7)
        self.assertEqual(len(json.loads(self.canonical.read_text())["lessons"]), 3)


if __name__ == "__main__":
    unittest.main()
