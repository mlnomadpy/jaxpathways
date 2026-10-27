"""Assemble a source-bound, weights-free C02 author review packet.

The input JSON names explicit local files, their SHA-256 values, and the
accepted source inventory. It is an authoring input, not a learner setup step.
Existing evidence files must match exactly; a changed experiment needs a new
packet directory. Full corpus inputs, checkpoints and tokenizer payloads stay
outside the public packet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_immutable(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"existing evidence differs; use a new packet: {path}")
    path.write_bytes(data)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def build(inputs):
    checked = {}
    for name, binding in inputs["inputs"].items():
        path = Path(binding["path"])
        if sha(path) != binding["sha256"]:
            raise ValueError(f"input bytes changed: {name}")
        checked[name] = path
    document_amendments = {}
    for relative, expected in inputs["accepted_source_files_sha256"].items():
        local = relative.removeprefix("projects/causal-model-300m/")
        path = PROJECT / local
        assert path.resolve().is_relative_to(PROJECT.resolve())
        actual = sha(path)
        if actual != expected:
            if local != "README.md":
                raise ValueError(f"accepted executable source changed: {local}")
            document_amendments[local] = {
                "accepted_sha256": expected,
                "current_sha256": actual,
                "scope": "Explained figures, section headings and optional plotting setup; training and evaluator code unchanged.",
            }
    native = json.loads(checked["native_harness"].read_text())
    for name, binding in native["protocol"]["task_files"].items():
        assert (
            sha(PROJECT / "data/lm_eval_tasks" / name / (name + ".yaml"))
            == binding["yaml_sha256"]
        )
        if "preprocess_sha256" in binding:
            assert (
                sha(PROJECT / "data/lm_eval_tasks" / name / "utils.py")
                == binding["preprocess_sha256"]
            )
    packet = PROJECT / "reference/evidence/c02-pilot"
    for name, relative in inputs["copies"].items():
        target = packet / relative
        if target.resolve().is_relative_to(packet.resolve()) is False:
            raise ValueError("unsafe packet output path")
        write_immutable(target, checked[name].read_bytes())

    raw = [json.loads(line) for line in checked["raw_qa"].read_text().splitlines()]
    assert (
        len(raw) == 768 and len({(r["branch"], r["question_id"]) for r in raw}) == 768
    )
    assert len({r["branch"] for r in raw}) == 6
    metrics = json.loads(checked["metrics"].read_text())
    for branch, aggregate in metrics["branches"].items():
        records = [row for row in raw if row["branch"] == branch]
        assert len(records) == 128
        for metric in ("exact_match", "token_f1"):
            mean = sum(row[metric] for row in records) / len(records)
            assert math.isclose(
                mean, aggregate["custom_squad_qa"][metric], abs_tol=1e-12
            )
    fields = (
        "branch",
        "question_id",
        "generated_token_ids",
        "prediction",
        "finish_reason",
        "exact_match",
        "token_f1",
    )
    projected = [{key: row[key] for key in fields} for row in raw]
    write_immutable(
        packet / "qa-generations-projection.jsonl",
        (
            "\n".join(json.dumps(row, ensure_ascii=True) for row in projected) + "\n"
        ).encode(),
    )

    pythia = json.loads(checked["pythia_qa"].read_text())
    public_pythia = {
        key: value for key, value in pythia.items() if key != "per_question"
    }
    public_pythia["per_question"] = [
        {key: row[key] for key in fields if key != "branch"}
        for row in pythia["per_question"]
    ]
    write_immutable(packet / "pythia-qa-projection.json", encoded(public_pythia))

    harness = json.loads(checked["native_harness"].read_text())
    write_immutable(
        packet / "native-harness-summary.json",
        encoded(
            {
                "scope": "First eight records per pinned task; execution smoke, not benchmark qualification.",
                "protocol": harness["protocol"],
                "task_results": harness["result"]["results"],
                "raw_report_sha256": sha(checked["native_harness"]),
            }
        ),
    )

    retention = {
        "inputs": {name: {"sha256": sha(path)} for name, path in checked.items()},
        "projection": "QA projections omit input prompts, contexts, gold answers and prompt token IDs. They preserve every generated prediction/ID and observed score. Recreate the complete raw records with the pinned dataset preparation and evaluation commands.",
        "not_included": "Downloaded corpus files, fitted tokenizer payload, optimizer/parameter arrays, baseline snapshots and runnable trained consumer weights remain local. The archive provides their reproducible source workflow and identities.",
        "accepted_source_files_sha256": inputs["accepted_source_files_sha256"],
        "document_amendments": document_amendments,
        "quality": "All six custom pilot branches failed the frozen QA floors. Full target and capstone remain planned.",
    }
    write_immutable(packet / "retention.json", encoded(retention))
    sources = {}
    for relative in inputs["source_files"]:
        path = PROJECT / relative
        if not path.resolve().is_relative_to(PROJECT.resolve()) or not path.is_file():
            raise ValueError(f"missing or unsafe source: {relative}")
        sources["projects/causal-model-300m/" + relative] = sha(path)
    public = {
        path.relative_to(PROJECT).as_posix(): sha(path)
        for path in sorted((PROJECT / "reference").rglob("*"))
        if path.is_file() and path.name != "evidence-index.json"
    }
    index = {
        "schema": "jaxpathways.causal-c02.public-evidence-index.v1",
        "public_files": public,
        "source_files_sha256": sources,
        "scope": "Bounded 33,088-parameter CPU lifecycle and failed QA outcome; not 300M training, benchmark qualification or a trained model release.",
        "execution_verification": {
            "path": "reference/evidence/c02-pilot/verification/independent-staged-check.log",
            "sha256": sha(packet / "verification/independent-staged-check.log"),
        },
        "retention": {
            "path": "reference/evidence/c02-pilot/retention.json",
            "sha256": sha(packet / "retention.json"),
        },
    }
    (PROJECT / "reference/evidence-index.json").write_bytes(encoded(index))
    print(f"Indexed {len(sources)} sources and {len(public)} public evidence files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    build(json.loads(parser.parse_args().inputs.read_text()))
