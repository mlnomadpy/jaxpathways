"""Qualify named, reviewed TPU runners on an existing VM without provisioning."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # The probe exits before a runner starts, releasing its TPU client. Holding
    # a client in this supervisor would prevent a child from acquiring the chip.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import jax
if jax.default_backend() != 'tpu':
    raise RuntimeError('physical TPU qualification requires the TPU backend')
print('TPU_RUNTIME:' + json.dumps({
    'backend': jax.default_backend(),
    'devices': [str(d) for d in jax.devices()],
    'deviceKinds': sorted({d.device_kind for d in jax.devices()}),
    'jax': jax.__version__,
}))
""",
        ],
        cwd=root,
        env={**os.environ, "JAX_PLATFORMS": "tpu"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    (output / "runtime-probe.stdout.txt").write_text(probe.stdout)
    (output / "runtime-probe.stderr.txt").write_text(probe.stderr)
    probe.check_returncode()
    runtime = json.loads(
        next(
            line.removeprefix("TPU_RUNTIME:")
            for line in probe.stdout.splitlines()
            if line.startswith("TPU_RUNTIME:")
        )
    )
    runners = [
        (
            "training",
            "resources/tpu-gcp/launch.py",
            [
                "--platform",
                "tpu",
                "--run-dir",
                str(output / "training"),
                "--steps",
                "4",
            ],
        ),
        (
            "resume",
            "resources/tpu-gcp/launch.py",
            [
                "--platform",
                "tpu",
                "--run-dir",
                str(output / "training"),
                "--steps",
                "8",
                "--resume",
            ],
        ),
        (
            "precision",
            "resources/tpu-gcp/precision_profile.py",
            [
                "--platform",
                "tpu",
                "--run-dir",
                str(output / "precision"),
                "--steps",
                "5",
                "--trace",
            ],
        ),
    ]
    receipt = {
        "schemaVersion": 1,
        "executedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **runtime,
        "python": platform.python_version(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "scope": "Single-host training, resume and precision runners only. Does not qualify all lessons, multi-host collectives or Pallas kernels.",
        "runs": [],
    }
    try:
        for name, source, arguments in runners:
            result = subprocess.run(
                [sys.executable, source, *arguments],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=240,
            )
            (output / f"{name}.stdout.txt").write_text(result.stdout)
            (output / f"{name}.stderr.txt").write_text(result.stderr)
            receipt["runs"].append(
                {
                    "name": name,
                    "source": source,
                    "sha256": hashlib.sha256((root / source).read_bytes()).hexdigest(),
                    "exitCode": result.returncode,
                }
            )
            result.check_returncode()
    finally:
        (output / "execution-tpu.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
