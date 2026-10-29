"""Select specialized CI checks; site checks always run independently."""

import json
import os
from pathlib import Path
import subprocess

CHECK_NAMES = (
    "browser",
    "lessons",
    "projects",
    "tpu",
    "loggers",
    "numpyro",
)


def checks_for_path(path):
    """Use narrow scopes only for recognized categories with known consumers."""
    if (
        path.startswith(("scripts/", "tests/", ".github/workflows/"))
        or path
        in (
            "package.json",
            "package-lock.json",
            "requirements-cpu.txt",
            "playwright.config.ts",
            "astro.config.mjs",
            "eslint.config.js",
        )
        or path.startswith("tsconfig")
    ):
        return set(CHECK_NAMES)
    if path.startswith("docs/evidence/tpu-qualification-"):
        return {"lessons", "tpu"}
    if path.startswith(("docs/", ".github/ISSUE_TEMPLATE/")) or path in (
        "README.md",
        "CURRICULUM.md",
        "LESSON_TEMPLATE.md",
        ".gitignore",
        ".gitattributes",
        ".prettierignore",
        ".prettierrc.json",
    ):
        return set()
    if path.startswith(("src/", "public/", "browser-tests/", "content/", "skills/")):
        return {"browser"}
    if path.startswith(("phases/", "learning-paths/", "assessments/")):
        return {"browser", "lessons"}
    if path.startswith("curriculum/"):
        return {"browser", "lessons", "projects"}
    if path.startswith("resources/"):
        return {"browser", "lessons", "projects", "tpu"}
    if path.startswith("projects/"):
        selected = {"browser", "projects"}
        for prefix, check in (
            ("reproducible-experiments/", "loggers"),
            ("bayesian-regression/", "numpyro"),
        ):
            if path.startswith("projects/" + prefix):
                selected.add(check)
        return selected
    # New categories need an explicit rule before any checks may be skipped.
    return set(CHECK_NAMES)


def select_checks(environment):
    """Scope direct PR runs and fall back to every check when evidence is missing."""
    if (
        environment.get("GITHUB_EVENT_NAME") != "pull_request"
        or environment.get("FULL_CHECKS") != "false"
    ):
        return set(CHECK_NAMES)
    try:
        event = json.loads(Path(environment["GITHUB_EVENT_PATH"]).read_text())
        paths = (
            subprocess.check_output(
                [
                    "git",
                    "diff",
                    "--name-only",
                    "-z",
                    "--no-renames",
                    event["pull_request"]["base"]["sha"],
                    event["pull_request"]["head"]["sha"],
                ]
            )
            .decode()
            .split("\0")
        )
        selected = set()
        for path in filter(None, paths):
            selected.update(checks_for_path(path))
        return selected
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        subprocess.CalledProcessError,
        UnicodeError,
    ):
        return set(CHECK_NAMES)


def main(environment=None):
    environment = os.environ if environment is None else environment
    selected = select_checks(environment)
    with Path(environment["GITHUB_OUTPUT"]).open("a") as output:
        for name in CHECK_NAMES:
            output.write(f"{name}={str(name in selected).lower()}\n")
    print(
        "Selected specialized checks: "
        + (", ".join(sorted(selected)) or "none; site checks still run")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
