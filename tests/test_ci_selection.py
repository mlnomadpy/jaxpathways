"""Exercise CI scoping without installing numerical runtimes or invoking training."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ci_selection", ROOT / "scripts/select-ci-checks.py"
)
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)
ALL = set(selection.CHECK_NAMES)


class PathSelectionTests(unittest.TestCase):
    def test_known_categories_select_their_consumers(self):
        cases = (
            ("docs/testing.md", set()),
            ("README.md", set()),
            ("src/scripts/catalog.js", {"browser"}),
            ("browser-tests/reader-print.spec.ts", {"browser"}),
            ("phases/01-arrays/example/lesson.json", {"browser", "lessons"}),
            ("curriculum/projects.json", {"browser", "lessons", "projects"}),
            (
                "resources/tpu-gcp/requirements-multihost.txt",
                {"browser", "lessons", "projects", "tpu"},
            ),
            (
                "docs/evidence/tpu-qualification-2026-10-20/restore.py",
                {"lessons", "tpu"},
            ),
        )
        for path, expected in cases:
            with self.subTest(path=path):
                self.assertEqual(selection.checks_for_path(path), expected)

    def test_specialized_projects_keep_separate_checks(self):
        for project, check in (
            ("reproducible-experiments/tests/loggers_check.py", "loggers"),
            ("bayesian-regression/solution/numpyro_workflow.py", "numpyro"),
        ):
            with self.subTest(project=project):
                self.assertEqual(
                    selection.checks_for_path("projects/" + project),
                    {"browser", "projects", check},
                )
        self.assertEqual(
            selection.checks_for_path("projects/causal-model-300m/training.py"),
            {"browser", "projects"},
        )

    def test_shared_or_unknown_changes_keep_every_check(self):
        for path in (
            "scripts/run-tests.mjs",
            "tests/test_ci_selection.py",
            "tests/suites.mjs",
            ".github/workflows/course-check.yml",
            "package-lock.json",
            "requirements-cpu.txt",
            "tsconfig.domain.json",
            "new-check/tool.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(selection.checks_for_path(path), ALL)


class EventSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.event = self.root / "event.json"
        self.event.write_text(
            json.dumps(
                {"pull_request": {"base": {"sha": "base"}, "head": {"sha": "head"}}}
            )
        )
        self.environment = {
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_EVENT_PATH": str(self.event),
            "FULL_CHECKS": "false",
        }

    def test_pull_request_combines_all_changed_paths_and_handles_nul_names(self):
        paths = b"src/scripts/catalog.js\0projects/scientific-inverse/new test.py\0docs/testing.md\0"
        with patch.object(
            selection.subprocess, "check_output", return_value=paths
        ) as diff:
            self.assertEqual(
                selection.select_checks(self.environment),
                {"browser", "projects"},
            )
        diff.assert_called_once_with(
            ["git", "diff", "--name-only", "-z", "--no-renames", "base", "head"]
        )

    def test_documentation_only_and_empty_diff_skip_specialized_checks(self):
        for paths in (b"docs/testing.md\0", b""):
            with self.subTest(paths=paths), patch.object(
                selection.subprocess, "check_output", return_value=paths
            ):
                self.assertEqual(selection.select_checks(self.environment), set())

    def test_manual_reusable_and_missing_configuration_default_to_full(self):
        cases = (
            {**self.environment, "GITHUB_EVENT_NAME": "workflow_dispatch"},
            {**self.environment, "GITHUB_EVENT_NAME": "workflow_call"},
            {**self.environment, "FULL_CHECKS": "true"},
            {
                key: value
                for key, value in self.environment.items()
                if key != "FULL_CHECKS"
            },
            {},
        )
        for environment in cases:
            with self.subTest(environment=environment), patch.object(
                selection.subprocess, "check_output"
            ) as diff:
                self.assertEqual(selection.select_checks(environment), ALL)
                diff.assert_not_called()

    def test_failed_git_or_invalid_event_runs_every_check(self):
        for error in (
            subprocess.CalledProcessError(1, "git"),
            FileNotFoundError("git"),
            UnicodeError("bad path"),
        ):
            with self.subTest(error=error), patch.object(
                selection.subprocess, "check_output", side_effect=error
            ):
                self.assertEqual(selection.select_checks(self.environment), ALL)
        for data in ("{", "{}", "null"):
            with self.subTest(data=data):
                self.event.write_text(data)
                self.assertEqual(selection.select_checks(self.environment), ALL)
        self.event.unlink()
        self.assertEqual(selection.select_checks(self.environment), ALL)

    def test_main_appends_every_output_with_explicit_true_or_false(self):
        output = self.root / "github-output.txt"
        output.write_text("previous=value\n")
        environment = {**self.environment, "GITHUB_OUTPUT": str(output)}
        with patch.object(
            selection.subprocess,
            "check_output",
            return_value=b"src/scripts/catalog.js\0",
        ), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(selection.main(environment), 0)
        lines = output.read_text().splitlines()
        self.assertEqual(lines[0], "previous=value")
        self.assertEqual(
            lines[1:],
            [
                f"{name}={str(name == 'browser').lower()}"
                for name in selection.CHECK_NAMES
            ],
        )


if __name__ == "__main__":
    unittest.main()
