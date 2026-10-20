"""Check the TPU collective target refuses CPU without creating evidence."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
TARGET = Path(__file__).with_name("collectives_target.py")


def main():
    with tempfile.TemporaryDirectory(prefix="collectives-cpu-refusal-") as temporary:
        output = Path(temporary) / "must-not-exist"
        environment = dict(os.environ, JAX_PLATFORMS="cpu")
        result = subprocess.run(
            [sys.executable, str(TARGET), "--output", str(output)],
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
    print("PASS: collective target refuses CPU before output creation")


if __name__ == "__main__":
    main()
