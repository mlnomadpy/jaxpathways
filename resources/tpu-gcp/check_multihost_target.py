"""Check the multi-host target rejects CPU before JAX initialization."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
TARGET = Path(__file__).with_name("multihost_target.py")


def main():
    with tempfile.TemporaryDirectory(prefix="multihost-cpu-refusal-") as temporary:
        output = Path(temporary) / "must-not-exist"
        environment = dict(os.environ, JAX_PLATFORMS="cpu")
        result = subprocess.run(
            [
                sys.executable,
                str(TARGET),
                "--output",
                str(output),
                "--checkpoint",
                "gs://private-bucket/run-prefix",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            raise AssertionError("CPU invocation unexpectedly succeeded")
        if "Set JAX_PLATFORMS=tpu" not in result.stderr:
            raise AssertionError(f"unexpected refusal reason: {result.stderr}")
        if output.exists():
            raise AssertionError("CPU refusal created an output directory")
    print(
        "PASS: multi-host target refuses CPU before initialization or output creation"
    )


if __name__ == "__main__":
    main()
