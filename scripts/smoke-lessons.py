"""Execute authored script and notebook companions on CPU; emit a bounded validation receipt."""

import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import importlib.metadata

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lesson",
        action="append",
        default=[],
        help="Lesson ID to run; repeat to select several",
    )
    parser.add_argument(
        "--phase",
        action="append",
        default=[],
        help="Phase ID to run; combined with any --lesson IDs",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List selected authored lesson IDs without executing or writing receipts",
    )
    parser.add_argument(
        "--receipt", type=Path, help="Write a separate receipt containing only this run"
    )
    args = parser.parse_args(argv)
    course = json.loads((ROOT / "public/curriculum.json").read_text())
    authored = [
        lesson
        for phase in course["phases"]
        for lesson in phase["lessons"]
        if lesson["status"] == "authored"
    ]
    unknown_lessons = set(args.lesson) - {lesson["id"] for lesson in authored}
    unknown_phases = set(args.phase) - {phase["id"] for phase in course["phases"]}
    if unknown_lessons:
        parser.error(
            "unknown or unauthored lesson IDs: " + ", ".join(sorted(unknown_lessons))
        )
    if unknown_phases:
        parser.error("unknown phase IDs: " + ", ".join(sorted(unknown_phases)))
    selected = set(args.lesson)
    for phase in course["phases"]:
        if phase["id"] in args.phase:
            selected.update(
                lesson["id"]
                for lesson in phase["lessons"]
                if lesson["status"] == "authored"
            )
    focused = bool(args.lesson or args.phase)
    if focused and not selected:
        parser.error("this selection contains no authored lessons")
    if args.list:
        print(
            "\n".join(
                lesson["id"]
                for lesson in authored
                if not focused or lesson["id"] in selected
            )
        )
        return 0
    validation_path = ROOT / (args.receipt or "curriculum/validation.json")
    if (
        focused
        and args.receipt
        and validation_path.resolve() == (ROOT / "curriculum/validation.json").resolve()
    ):
        parser.error(
            "focused runs must use a separate --receipt path; omit --receipt to merge current results"
        )
    existing_records = {}
    if focused and not args.receipt and validation_path.exists():
        for entry in json.loads(validation_path.read_text()).get("lessons", []):
            existing_records[entry["lessonId"]] = entry

    packages = {
        name: importlib.metadata.version(name)
        for name in [
            "jax",
            "numpy",
            "optax",
            "flax",
            "grain",
            "orbax-checkpoint",
            "matplotlib",
        ]
    }
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    records = []
    ran_count = 0
    for phase in course["phases"]:
        for lesson in phase["lessons"]:
            if lesson["status"] != "authored":
                continue
            if focused and lesson["id"] not in selected:
                previous = existing_records.get(lesson["id"])
                if previous and previous.get("contentHash") == lesson["contentHash"]:
                    records.append(previous)
                continue
            ran_count += 1
            script = ROOT / lesson["artifacts"]["scriptSource"]
            # Notebook cells execute in order with shared state, just as in a fresh kernel.
            runtime_probe = "\nimport jax, json\nprint('COURSE_RUNTIME:' + json.dumps({'backend':jax.default_backend(),'deviceCount':jax.local_device_count()}))"
            observed = []
            notebook_result = None
            script_stdout = ""
            with tempfile.TemporaryDirectory() as temporary:
                notebook_output = Path(temporary) / "execution.json"
                for kind in ["script", "notebook"]:
                    command = (
                        [sys.executable, "-c", script.read_text() + runtime_probe]
                        if kind == "script"
                        else [
                            sys.executable,
                            str(ROOT / "scripts/execute-notebook.py"),
                            str(ROOT / lesson["artifacts"]["notebookSource"]),
                            str(notebook_output),
                        ]
                    )
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    if result.returncode:
                        print(
                            f"FAIL {lesson['id']} ({kind})\n{result.stdout}\n{result.stderr}",
                            file=sys.stderr,
                        )
                        sys.exit(1)
                    if kind == "script":
                        marker = next(
                            line.removeprefix("COURSE_RUNTIME:")
                            for line in result.stdout.splitlines()
                            if line.startswith("COURSE_RUNTIME:")
                        )
                        runtime = json.loads(marker)
                        script_stdout = (
                            "\n".join(
                                line
                                for line in result.stdout.splitlines()
                                if not line.startswith("COURSE_RUNTIME:")
                            )
                            + "\n"
                        )
                    else:
                        notebook_result = json.loads(notebook_output.read_text())
                        runtime = notebook_result["runtime"]
                    expected = (
                        lesson["content"].get("runtime", {}).get("deviceCount", 1)
                    )
                    if (
                        runtime["backend"] != "cpu"
                        or runtime["deviceCount"] != expected
                    ):
                        raise RuntimeError(
                            f"{lesson['id']}: expected CPU with {expected} devices, observed {runtime}"
                        )
                    observed.append(runtime)
            execution = {
                "schemaVersion": 1,
                "lessonId": lesson["id"],
                "contentHash": lesson["contentHash"],
                "executedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "environment": {
                    **observed[0],
                    "python": platform.python_version(),
                    **{name: packages[name] for name in ["jax", "numpy", "matplotlib"]},
                },
                "stdout": script_stdout,
                "notebookCells": notebook_result["notebookCells"],
                "scope": "Recorded instructor reference run; all script assertions and notebook code cells passed. No accelerator performance claim.",
            }
            (ROOT / lesson["path"] / "outputs/execution.json").write_text(
                json.dumps(execution, indent=2) + "\n"
            )
            records.append(
                {
                    "lessonId": lesson["id"],
                    "contentHash": lesson["contentHash"],
                    "script": "passed",
                    "notebookCode": "passed",
                    "runtime": observed[0],
                }
            )
            print(f"PASS {lesson['id']}: script + notebook code", flush=True)
    import jax
    import numpy

    receipt = {
        "backend": "cpu",
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": numpy.__version__,
        "packages": packages,
        "validatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Worked examples and reference solutions executed in fresh processes. No TPU validation or learner assessment.",
        "lessons": records,
    }
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        f"OK: {ran_count} executed ({len(records)} total CPU lessons); {ran_count*2} executions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
