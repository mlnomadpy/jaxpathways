"""Build the public path/hash index for the retained C01 evidence packet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent.parent
INDEX = PROJECT / "reference/evidence-index.json"
AMENDMENT = PROJECT / "reference/evidence/checker-attention-amendment.json"
PLOT_AMENDMENT = PROJECT / "reference/evidence/objective-plot-amendment.json"
GRADIENT_AMENDMENT = PROJECT / "reference/evidence/checker-mlm-gradient-amendment.json"
PACKET = PROJECT / "reference/evidence/final8"
TEACHING = PROJECT / "reference/teaching"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    receipt = read_json(PACKET / "sourcebound-receipt.json")
    amendment = read_json(AMENDMENT)["amendment"]
    plot_amendment = read_json(PLOT_AMENDMENT)["amendment"]
    assert sha256(PACKET / "sourcebound-receipt.json") == (
        "514287dcfd125d3127f93c43d0c20540b6325c9d4d39f4a803ea2fa57aa942b8"
    )
    assert amendment["full_checker_rerun_for_amended_hash"] is False
    current_checker = amendment["current_checker_sha256"]
    gradient_amendment = None
    full_verification = None
    current_verification_paths = []
    if GRADIENT_AMENDMENT.is_file():
        gradient_amendment = read_json(GRADIENT_AMENDMENT)["amendment"]
        assert gradient_amendment["previous_checker_sha256"] == current_checker
        assert gradient_amendment["full_checker_rerun_for_amended_hash"] is True
        current_checker = gradient_amendment["current_checker_sha256"]
        for key in ("full_checker_verification", "negative_verification"):
            binding = gradient_amendment[key]
            relative = Path(binding["path"])
            assert not relative.is_absolute() and ".." not in relative.parts
            assert relative.parts[:2] == ("reference", "evidence")
            path = PROJECT / relative
            assert sha256(path) == binding["sha256"]
            record = read_json(path)
            assert record["checker_sha256"] == current_checker
            if key == "full_checker_verification":
                full_verification = record
                assert record["exit_code"] == 0
                assert record["independent_check"]["status"] == "pass"
                assert record["independent_check"]["stage"] == "all"
                assert record["source_files_sha256"]["model.py"] == sha256(
                    PROJECT / "model.py"
                )
            else:
                assert record["rejected_before_optimization"] is True
                assert record["exit_code"] != 0
            current_verification_paths.append(path)
    assert current_checker == sha256(PROJECT / "tests/check.py")
    assert plot_amendment["full_training_or_evaluation_rerun"] is False
    assert plot_amendment["historical_receipt"]["sha256"] == sha256(
        PACKET / "sourcebound-receipt.json"
    )
    assert plot_amendment["source_change"]["current_sha256"] == sha256(
        PROJECT / "plots.py"
    )

    # Current implementation identity is the historical 19-file inventory,
    # with separately declared checker and plotting-only amendments. Earlier
    # receipts remain historical when a later checker has its own full rerun.
    source_files = {}
    for relative, historical_hash in receipt["source_files_sha256"].items():
        path = PROJECT / relative
        current_hash = sha256(path)
        if relative == "tests/check.py":
            assert current_hash == current_checker
        elif relative == "plots.py":
            assert current_hash == plot_amendment["source_change"]["current_sha256"]
        else:
            assert (
                current_hash == historical_hash
            ), f"source changed after final8: {relative}"
        source_files[f"projects/embedding-model-300m/{relative}"] = current_hash
    assert len(source_files) == 19
    if full_verification is not None:
        expected_sources = {
            relative.removeprefix("projects/embedding-model-300m/"): digest
            for relative, digest in source_files.items()
        }
        assert full_verification["source_files_sha256"] == expected_sources

    public_files = {}
    candidates = [PROJECT / "reference/README.md", AMENDMENT, PLOT_AMENDMENT]
    if gradient_amendment is not None:
        candidates.extend([GRADIENT_AMENDMENT, *current_verification_paths])
    candidates.extend(sorted(PACKET.rglob("*")))
    candidates.extend(sorted(TEACHING.rglob("*")))
    candidates.append(PROJECT / "reference/evidence/cold-load-validation.json")
    for path in candidates:
        if not path.is_file():
            continue
        relative = path.relative_to(PROJECT).as_posix()
        if relative == "reference/evidence-index.json":
            continue
        assert relative.startswith("reference/")
        public_files[relative] = sha256(path)

    retention = read_json(PACKET / "retention-manifest.json")
    teaching = read_json(TEACHING / "manifest.json")
    omitted = {
        f"historical-final8/{relative}": details
        for relative, details in retention["omitted_artifacts"].items()
    }
    index = {
        "schema": "jaxpathways.embedding-c01.public-evidence-index.v1",
        "public_files": dict(sorted(public_files.items())),
        "source_files_sha256": dict(sorted(source_files.items())),
        "historical_receipt": {
            "path": "reference/evidence/final8/sourcebound-receipt.json",
            "sha256": sha256(PACKET / "sourcebound-receipt.json"),
        },
        "checker_amendment": {
            "path": "reference/evidence/checker-attention-amendment.json",
            "sha256": sha256(AMENDMENT),
        },
        "plot_layout_amendment": {
            "path": "reference/evidence/objective-plot-amendment.json",
            "sha256": sha256(PLOT_AMENDMENT),
        },
        "teaching_extension": {
            "manifest": {
                "path": "reference/teaching/manifest.json",
                "sha256": sha256(TEACHING / "manifest.json"),
            },
            "renderer": teaching["renderer"],
            "historical_inputs": teaching["historical_inputs"],
            "outputs": teaching["outputs"],
        },
        "omitted_files": omitted,
        "retained_numeric_derivations": retention["derived_full_rank_files"],
        "historical_run": {
            "name": "final8",
            "checker_at_run_sha256": receipt["source_files_sha256"]["tests/check.py"],
            "result": "Historical final8 passed under the source versions in its immutable receipt. Later checker and plotting executions have separate edition verification records; this historical receipt does not qualify later code.",
            "independent_check_sha256": retention["historical_run"][
                "independent_check_sha256"
            ],
        },
    }
    if gradient_amendment is not None:
        index["mlm_gradient_checker_amendment"] = {
            "path": GRADIENT_AMENDMENT.relative_to(PROJECT).as_posix(),
            "sha256": sha256(GRADIENT_AMENDMENT),
            "current_checker_sha256": current_checker,
        }
    INDEX.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {INDEX.relative_to(ROOT)} with "
        f"{len(public_files)} public files and {len(source_files)} source hashes"
    )


if __name__ == "__main__":
    main()
