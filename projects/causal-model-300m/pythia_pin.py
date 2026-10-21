"""Strict local identity checks for the seven-file Pythia-14M baseline."""

from __future__ import annotations

import hashlib
from pathlib import Path

from prepare import PYTHIA_FILES_SHA256, PYTHIA_REVISION


def verify_pythia_snapshot(
    model_dir: str | Path, expected_files: dict[str, str] | None = None
) -> dict[str, str]:
    """Verify every frozen file before a tokenizer or model loader can run."""
    root = Path(model_dir).resolve()
    pins = PYTHIA_FILES_SHA256 if expected_files is None else expected_files
    if set(pins) != set(PYTHIA_FILES_SHA256):
        raise ValueError(
            "Pythia snapshot verifier requires the exact seven-file pin set"
        )
    actual = {}
    for relative, expected in sorted(pins.items()):
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"Pythia baseline file is missing: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Pythia baseline file hash mismatch: {relative}")
        actual[relative] = digest
    return actual


def snapshot_identity(model_dir: str | Path) -> dict[str, object]:
    return {
        "revision": PYTHIA_REVISION,
        "files_sha256": verify_pythia_snapshot(model_dir),
    }
