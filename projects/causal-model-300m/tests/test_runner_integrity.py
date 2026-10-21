"""The stage runner refuses schedule, array or SFT edits before model init."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
DATA = Path("/private/tmp/jaxpathways-c02-datafreeze-luna-20261021")
sys.path.insert(0, str(PROJECT))
import run_pilot


class RunnerIntegrityTests(unittest.TestCase):
    @unittest.skipUnless(DATA.is_dir(), "requires the accepted frozen C02 data packet")
    def test_runner_rejects_changed_schedule_blocks_and_sft_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "prepared"
            shutil.copytree(DATA, copied, copy_function=os.link)
            output = Path(temporary) / "must-not-be-created"
            for relative, mutate in (
                ("pilot-batches.jsonl", self._mutate_schedule),
                ("base-train-blocks.npz", self._mutate_npz),
                ("sft-fit.jsonl", self._mutate_sft),
            ):
                target = copied / relative
                original = DATA / relative
                target.unlink()
                os.link(original, target)
                before = target.read_bytes()
                tampered = mutate(before)
                target.unlink()
                target.write_bytes(tampered)
                with self.assertRaisesRegex(ValueError, "bytes changed"):
                    run_pilot.run(copied, output)
                self.assertFalse(output.exists())
                target.unlink()
                os.link(original, target)

    @staticmethod
    def _mutate_schedule(payload):
        rows = payload.decode().splitlines()
        row = json.loads(rows[0])
        row["block_ids"][0] = "forged:000000"
        rows[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
        return ("\n".join(rows) + "\n").encode()

    @staticmethod
    def _mutate_npz(payload):
        import io
        import numpy as np

        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays["target_ids"][0, 0] = (arrays["target_ids"][0, 0] + 1) % 512
        output = io.BytesIO()
        np.savez_compressed(output, **arrays)
        return output.getvalue()

    @staticmethod
    def _mutate_sft(payload):
        rows = payload.decode().splitlines()
        row = json.loads(rows[0])
        row["target_answer"] = "changed after recipe freeze"
        rows[0] = json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return ("\n".join(rows) + "\n").encode()


if __name__ == "__main__":
    unittest.main()
