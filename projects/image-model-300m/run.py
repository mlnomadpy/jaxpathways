"""Strict preflight and explicitly approved runner for the flower-image pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_verifier():
    path = PROJECT / "data" / "verify.py"
    spec = importlib.util.spec_from_file_location("image_pilot_data_verifier", path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load the trusted source-bound data verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_candidate_apis(specification: str) -> tuple[dict, Path]:
    candidate_path = (
        PROJECT / "solution" / "model.py"
        if specification == "solution"
        else Path(specification).expanduser().resolve()
    )
    if candidate_path.is_dir():
        candidate_path = candidate_path / "model.py"
    if not candidate_path.is_file():
        raise FileNotFoundError(f"candidate model source not found: {candidate_path}")
    candidate_dir = candidate_path.parent
    sys.path.insert(0, str(candidate_dir))
    modules = {}
    for public_name, file_name in (
        ("model", candidate_path),
        ("objectives", candidate_dir / "objectives.py"),
        ("augment", candidate_dir / "augment.py"),
        ("training", candidate_dir / "training.py"),
        ("evaluate", candidate_dir / "evaluate.py"),
        ("runner", candidate_dir / "runner.py"),
    ):
        if not file_name.is_file():
            raise FileNotFoundError(f"candidate API file missing: {file_name}")
        sys.modules.pop(public_name, None)
        spec = importlib.util.spec_from_file_location(public_name, file_name)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load candidate API file: {file_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[public_name] = module
        spec.loader.exec_module(module)
        modules[public_name] = module
    return {
        "model": modules["model"],
        "objectives": modules["objectives"],
        "augment": modules["augment"],
        "training": modules["training"],
        "evaluate": modules["evaluate"],
        "runner": modules["runner"],
    }, candidate_path


def _candidate_source_inventory(candidate_directory: Path) -> dict[str, str]:
    """Hash every local Python source available to a selected candidate bundle."""
    root = candidate_directory.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"candidate source directory not found: {root}")
    required = {
        "model.py",
        "objectives.py",
        "augment.py",
        "training.py",
        "evaluate.py",
        "runner.py",
    }
    observed = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("candidate Python source escapes its approved directory")
        observed[path.relative_to(root).as_posix()] = _sha(resolved)
    missing = required - set(observed)
    if missing:
        raise FileNotFoundError(
            f"candidate API source inventory is incomplete: {sorted(missing)}"
        )
    if not observed:
        raise ValueError("candidate source inventory is empty")
    return observed


def _check_versions(recipe: dict) -> dict[str, str]:
    direct = {}
    for line in (PROJECT / "requirements-image.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        direct[name.lower().replace("_", "-")] = version
    observed = {}
    for name, expected in direct.items():
        distribution = {
            "pillow": "Pillow",
            "jaxlib": "jaxlib",
            "ml-dtypes": "ml_dtypes",
        }.get(name, name)
        version = importlib.metadata.version(distribution)
        observed[name] = version
        if version != expected:
            raise RuntimeError(
                f"locked dependency mismatch for {name}: expected {expected}, found {version}"
            )
    if not recipe.get("source_sha256", {}).get("requirements-image.txt"):
        raise ValueError("recipe does not source-bind the direct dependency pins")
    return observed


def _verify_approval(
    approval_path: Path,
    recipe_path: Path,
    data_report: dict,
    implementation_source: Path,
    candidate_source_sha256: dict[str, str],
) -> dict:
    approval = json.loads(approval_path.read_text())
    recipe_sha = _sha(recipe_path)
    expected = {
        "schema": "jaxpathways.image-pilot-optimization-approval.v1",
        "status": "approved",
        "reviewer": "Sol",
        "recipe_sha256": recipe_sha,
        "source_manifest_sha256": data_report["source_manifest_sha256"],
        "implementation_source_sha256": _sha(implementation_source),
        "candidate_api_source_sha256": candidate_source_sha256,
    }
    for key, value in expected.items():
        if approval.get(key) != value:
            raise PermissionError(
                f"optimization approval is missing or stale for {key}"
            )
    if approval.get("no_recipe_changes_since_review") is not True:
        raise PermissionError(
            "optimization approval does not assert an unchanged reviewed recipe"
        )
    return approval


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", required=True, type=Path, help="external materialized v7 flower split"
    )
    parser.add_argument(
        "--archive", required=True, type=Path, help="pinned official upstream archive"
    )
    parser.add_argument(
        "--recipe", type=Path, default=PROJECT / "data" / "recipe-review-v10.json"
    )
    parser.add_argument(
        "--approval", type=Path, help="explicit source-bound Sol approval receipt"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new external directory for run artifacts/checkpoints",
    )
    parser.add_argument(
        "--implementation",
        default="solution",
        help="solution or a candidate model.py/directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the source-bound pipeline journal in the existing output directory",
    )
    args = parser.parse_args()
    recipe_path = args.recipe.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    archive_path = args.archive.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if PROJECT == output_path or PROJECT in output_path.parents:
        raise ValueError("run outputs must be outside the source tree")
    recipe = json.loads(recipe_path.read_text())
    verifier = _load_verifier()
    data_report = verifier.verify_data(
        data_path, recipe_path=recipe_path, archive_path=archive_path
    )
    versions = _check_versions(recipe)
    if not recipe.get("preoptimization_gate", {}).get("optimization_permitted"):
        print(
            json.dumps(
                {
                    "status": "blocked_preoptimization_review",
                    "data_verification": data_report,
                    "runtime_versions": versions,
                    "optimizer_updates": 0,
                    "message": "Sol has not approved this exact recipe/source/data snapshot; verification completed, but no model is initialized, no training batch is loaded, and no optimizer update is applied.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    if args.approval is None:
        raise PermissionError(
            "an explicit source-bound Sol approval receipt is required before optimization"
        )
    candidate_source = (
        PROJECT / "solution" / "model.py"
        if args.implementation == "solution"
        else (
            Path(args.implementation).expanduser().resolve() / "model.py"
            if Path(args.implementation).expanduser().resolve().is_dir()
            else Path(args.implementation).expanduser().resolve()
        )
    )
    candidate_source_sha256 = _candidate_source_inventory(candidate_source.parent)
    required_candidate_files = set(recipe.get("candidate_api_required_files", []))
    if not required_candidate_files <= set(candidate_source_sha256):
        raise PermissionError("recipe candidate API inventory is incomplete")
    approval = _verify_approval(
        args.approval.expanduser().resolve(),
        recipe_path,
        data_report,
        candidate_source,
        candidate_source_sha256,
    )
    api_modules, implementation_source = _load_candidate_apis(args.implementation)
    if (
        _candidate_source_inventory(implementation_source.parent)
        != candidate_source_sha256
    ):
        raise PermissionError("candidate source inventory changed during preflight")
    sys.path.insert(0, str(PROJECT / "solution"))
    from pipeline import execute

    report = execute(
        recipe_path=recipe_path,
        data_directory=data_path,
        archive_path=archive_path,
        output_directory=output_path,
        implementation_source=implementation_source,
        api_modules=api_modules,
        candidate_source_paths={
            name: implementation_source.parent / name
            for name in candidate_source_sha256
        },
        candidate_source_sha256=candidate_source_sha256,
        resume=args.resume,
    )
    report["optimization_approval_sha256"] = _sha(args.approval.expanduser().resolve())
    report["optimization_approval_reviewer"] = approval["reviewer"]
    report["runtime_versions"] = versions
    report["model_card_path"] = "MODEL_CARD.md"
    (output_path / "run-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    from model_card import write_model_card

    write_model_card(output_path / "run-report.json", recipe_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
