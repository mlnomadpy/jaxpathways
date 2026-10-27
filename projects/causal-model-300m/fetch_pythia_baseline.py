"""Fetch and verify only the frozen Pythia-14M baseline files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
if str(HERE / "data") not in sys.path:
    sys.path.insert(0, str(HERE / "data"))
import prepare
from pythia_pin import verify_pythia_snapshot


def fetch(output_dir: str | Path) -> Path:
    output = Path(output_dir).resolve()
    snapshot_download(
        repo_id="EleutherAI/pythia-14m",
        revision=prepare.PYTHIA_REVISION,
        allow_patterns=list(prepare.PYTHIA_FILES_SHA256),
        local_dir=output,
    )
    verify_pythia_snapshot(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fetch(args.output)
    print(f"verified Pythia-14M baseline at {result}")


if __name__ == "__main__":
    main()
