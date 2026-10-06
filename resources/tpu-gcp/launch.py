"""Run the course's small training worker on an explicitly selected local backend.

This launcher never creates or deletes cloud resources. Run it inside the chosen VM.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import shutil
import sys


def duration(value):
    seconds = float(value)
    if not math.isfinite(seconds) or not 0 < seconds <= 900:
        raise argparse.ArgumentTypeError("timeout must be finite, positive and at most 900 seconds")
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("cpu", "tpu"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8, help="total target step, including restored steps")
    parser.add_argument("--timeout", type=duration, default=180., help="child process limit; does not delete a VM")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000:
        parser.error("steps must be between 1 and 10000")
    root = args.run_dir.resolve()
    # Claim the folder before inspecting it, so concurrent invocations cannot overwrite a run.
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = root.with_name(root.name + ".launch-lock")
    try:
        lock.mkdir()
    except FileExistsError:
        parser.error("another launcher owns this run; inspect the lock before retrying")
    try:
        if args.resume:
            try:
                saved = json.loads((root / "checkpoint.json").read_text())
                if type(saved.get("step")) is not int or not 0 <= saved["step"] < args.steps:
                    parser.error("resume requires a target step greater than the saved step")
            except (OSError, ValueError) as error:
                parser.error("resume needs a readable checkpoint: " + str(error))
        elif root.exists() and (not root.is_dir() or any(root.iterdir())):
            parser.error("run folder is not empty; choose a new folder or explicitly resume")

        source = Path(__file__).resolve().parents[2] / "projects/workload-operations/solution/model.py"
        spec = importlib.util.spec_from_file_location("course_workload", source)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        root.mkdir(parents=True, exist_ok=True)
        if args.resume:
            history = root / "history"
            history.mkdir(exist_ok=True)
            attempt = history / f"attempt-{len(list(history.iterdir())) + 1:04d}"
            attempt.mkdir()
            for path in root.iterdir():
                if path.is_file():
                    shutil.copy2(path, attempt / path.name)

        packages = {}
        for name in ("jax", "jaxlib", "numpy", "libtpu"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        worker.atomic_json(root / "environment.json", dict(python=platform.python_version(), packages=packages))
        config = worker.default_config(backend=args.platform, steps=args.steps, resume=args.resume)
        print(f"Starting {args.platform} worker. Events are collected when the worker exits.", flush=True)
        result = worker.launch(root, config, timeout=args.timeout)
        summary = worker.summarize(result)
        worker.atomic_json(root / "summary.json", summary)
        with (root / "events.jsonl").open("w") as stream:
            for event in result["events"]:
                stream.write(json.dumps(event) + "\n")
        print(json.dumps(summary, indent=2))
        if result["status"] != "completed":
            print(result["stderr"], file=sys.stderr)
            return 1
        return 0
    finally:
        lock.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
