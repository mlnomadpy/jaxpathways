"""Run registered reference projects, with optional selection and bounded subprocesses."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def source_hash(directory):
    """Preserve the receipt digest without loading entire artifacts into memory."""
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        action="append",
        default=[],
        help="Project ID to run; repeat to select several",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List selected project IDs without executing or writing receipts",
    )
    parser.add_argument(
        "--shard",
        help="Zero-based shard index/count, applied after selection, for example 0/4",
    )
    parser.add_argument(
        "--receipt",
        help="Output receipt path; focused and sharded runs require a separate path",
    )
    args = parser.parse_args(argv)
    index, count = 0, 1
    if args.shard:
        try:
            index, count = map(int, args.shard.split("/"))
        except ValueError:
            parser.error("--shard must be index/count")
        if count < 1 or not 0 <= index < count:
            parser.error("shard must satisfy 0 <= index < count")
    if (args.shard or args.project) and not args.receipt and not args.list:
        parser.error(
            "focused and sharded runs require --receipt to preserve the complete registry receipt"
        )

    destination = ROOT / (args.receipt or "curriculum/project-validation.json")
    if (
        (args.shard or args.project)
        and not args.list
        and destination.resolve()
        == (ROOT / "curriculum/project-validation.json").resolve()
    ):
        parser.error("focused and sharded runs must use a separate --receipt path")

    manifests = [
        (ROOT / relative, json.loads((ROOT / relative).read_text()))
        for relative in json.loads((ROOT / "curriculum/projects.json").read_text())
    ]
    requested = set(args.project)
    unknown = requested - {manifest["id"] for _, manifest in manifests}
    if unknown:
        parser.error("unknown project IDs: " + ", ".join(sorted(unknown)))
    selected = [
        entry for entry in manifests if not requested or entry[1]["id"] in requested
    ]
    selected = selected[index::count]
    if not selected:
        parser.error("this selection/shard contains no projects")
    if args.list:
        print("\n".join(manifest["id"] for _, manifest in selected))
        return 0

    records = []
    for manifest_path, manifest in selected:
        command = shlex.split(manifest["referenceCommand"])
        if len(command) < 2 or command[0] not in ("python", "python3"):
            raise ValueError(f"Unsupported reference runner for {manifest['id']}")
        script = (ROOT / command[1]).resolve()
        if (
            not script.is_relative_to(manifest_path.parent.resolve())
            or not script.is_file()
        ):
            raise ValueError("Reference checker must be inside its project")
        digest = source_hash(manifest_path.parent)
        result = subprocess.run(
            [sys.executable, *command[1:]],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=240,
        )
        print(result.stdout, end="")
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(f"FAIL reference project: {manifest['id']}")
        records.append(
            dict(
                projectId=manifest["id"],
                sourceHash=digest,
                command=manifest["referenceCommand"],
                stdout=result.stdout,
                status="passed",
            )
        )
    receipt = dict(
        schemaVersion=1,
        executedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope="Registered public reference checkers only; not learner assessment or accelerator validation.",
        projects=records,
    )
    receipt["shard"] = {"index": index, "count": count, "totalProjects": len(manifests)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"PASS: {len(records)} registered project reference suites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
