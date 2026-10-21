"""Run named lm-evaluation-harness tasks on the frozen custom pilot consumer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager

from data.prepare import LM_EVAL_COMMIT, LM_EVAL_VERSION
from lm_eval_adapter import C02PilotLM

HERE = Path(__file__).resolve().parent
TASK_ROOT = HERE / "data" / "lm_eval_tasks"
TASKS = ("hellaswag", "arc_easy")
TASK_PROTOCOL = {
    "hellaswag": {
        "dataset": "Rowan/hellaswag",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "split": "validation",
    },
    "arc_easy": {
        "dataset": "allenai/ai2_arc",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "split": "test",
    },
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _distribution_identity():
    distribution = importlib.metadata.distribution("lm_eval")
    digest = hashlib.sha256()
    files = []
    record_sha256 = None
    for relative in sorted(distribution.files or [], key=str):
        path = Path(distribution.locate_file(relative))
        if not path.is_file():
            continue
        content_hash = _sha256(path)
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\n")
        if str(relative).endswith(".dist-info/RECORD"):
            record_sha256 = content_hash
        files.append(str(relative))
    return {
        "distribution": "lm_eval",
        "version": distribution.version,
        "installed_file_count": len(files),
        "installed_files_manifest_sha256": digest.hexdigest(),
        "record_file_sha256": record_sha256,
        "distribution_location": str(distribution.locate_file("")),
        "identity_note": "The installed PyPI distribution is identified by its RECORD and installed-file digest. A repository commit declared for task definitions is reported separately and is not inferred from lm_eval's returned git_hash.",
    }


def run(package_dir, tasks, output_path, fewshot=0, limit=None):
    installed = importlib.metadata.version("lm_eval")
    if installed != LM_EVAL_VERSION:
        raise RuntimeError(
            f"the frozen adapter requires lm_eval=={LM_EVAL_VERSION}; installed {installed}"
        )
    if tuple(tasks) != TASKS or fewshot != 0 or limit != 8:
        raise ValueError(
            "the frozen pilot harness run requires tasks=hellaswag,arc_easy, "
            "fewshot=0, and the first 8 records from each pinned split"
        )
    model = C02PilotLM(package_dir)
    task_files = {
        name: {
            "yaml": str(TASK_ROOT / name / f"{name}.yaml"),
            "yaml_sha256": _sha256(TASK_ROOT / name / f"{name}.yaml"),
        }
        for name in TASKS
    }
    task_files["hellaswag"]["preprocess_sha256"] = _sha256(
        TASK_ROOT / "hellaswag" / "utils.py"
    )
    result = simple_evaluate(
        model=model,
        # lm_eval 0.4.13's TaskManager treats any non-list sequence as one
        # spec, so pass a concrete list even though this helper's API accepts
        # an immutable tuple.
        tasks=list(tasks),
        num_fewshot=fewshot,
        batch_size=1,
        limit=limit,
        log_samples=True,
        task_manager=TaskManager(include_path=TASK_ROOT),
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "protocol": {
                    "lm_eval_version": installed,
                    "lm_eval_distribution": _distribution_identity(),
                    "task_source_commit_declared_by_frozen_recipe": LM_EVAL_COMMIT,
                    "lm_eval_runner_sha256": _sha256(Path(__file__)),
                    "adapter_sha256": _sha256(HERE / "lm_eval_adapter.py"),
                    "lm_eval_returned_working_directory_git_hash": getattr(
                        result, "get", lambda *_: None
                    )("git_hash"),
                    "package_manifest_sha256": _sha256(
                        Path(package_dir).resolve() / "manifest.json"
                    ),
                    "fewshot": fewshot,
                    "selection": "indices 0 through 7 from each frozen split",
                    "tasks": TASK_PROTOCOL,
                    "task_files": task_files,
                },
                "result": result,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument(
        "--tasks", required=True, help="comma-separated harness task IDs"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    if args.fewshot < 0 or (args.limit is not None and args.limit < 1):
        parser.error("fewshot must be nonnegative and limit must be positive")
    tasks = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
    if not tasks:
        parser.error("at least one task ID is required")
    run(args.package, tasks, args.output, args.fewshot, args.limit)


if __name__ == "__main__":
    main()
