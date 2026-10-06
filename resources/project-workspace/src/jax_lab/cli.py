"""Prepare an auditable run directory using only Python's standard library."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(config_path, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("run-id must use 1–64 letters, digits, underscores or hyphens")
    config_path = ROOT / config_path
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict) or set(config) != {"seed", "learning_rate", "steps", "data"}:
        raise ValueError("config needs exactly seed, learning_rate, steps and data")
    for name in ("seed", "steps"):
        if type(config[name]) is not int or config[name] < (1 if name == "steps" else 0):
            raise ValueError(f"{name} must be a {'positive' if name == 'steps' else 'nonnegative'} integer")
    rate = config["learning_rate"]
    if type(rate) not in (int, float) or not 0 < rate < float("inf"):
        raise ValueError("learning_rate must be finite and positive")
    if not isinstance(config["data"], str):
        raise ValueError("data must be a path relative to this workspace")
    data_path = (ROOT / config["data"]).resolve()
    if not data_path.is_relative_to(ROOT.resolve()):
        raise ValueError("data must stay inside this workspace")
    data_hash = digest(data_path)
    sources = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "run.py"]
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    # Only record this workspace's Git repository, never an enclosing course checkout.
    git = {"commit": None, "dirty": None}
    if (ROOT / ".git").exists():
        try:
            git["commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
            ).strip()
            git["dirty"] = bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip())
        except (OSError, subprocess.CalledProcessError):
            pass
    run = ROOT / "runs" / run_id
    run.parent.mkdir(exist_ok=True)
    run.mkdir(exist_ok=False)  # Reusing a run ID must never overwrite evidence.
    (run / "source" / "src" / "jax_lab").mkdir(parents=True)
    for source in sources:
        destination = run / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    (run / "data.snapshot").write_bytes(data_path.read_bytes())
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "run_id": run_id, "status": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": digest(run / "config.json"), "data_sha256": data_hash,
        "source_sha256": source_hashes, "git": git,
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "command": sys.argv,
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run / "notes.md").write_text(
        "# Run notes\n\nHypothesis:\n\nChanged variable:\n\nEvaluation protocol:\n\n"
        "Observed result (after execution):\n\nDecision and limitations:\n"
    )
    print(f"Prepared runs/{run_id}; no training has run.")
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare", help="validate inputs and snapshot a new run")
    command.add_argument("--config", required=True, help="JSON path relative to the workspace")
    command.add_argument("--run-id", required=True, help="unique, readable run directory name")
    args = parser.parse_args()
    try:
        prepare(args.config, args.run_id)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Cannot prepare run: {error}\n")


if __name__ == "__main__":
    main()
