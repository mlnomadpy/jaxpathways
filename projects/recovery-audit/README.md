# Resume the complete training cursor

A checkpoint that restores only model weights can still change the next update. Momentum and the random-number state affect training too. This project trains a two-value parameter with a momentum buffer, a random perturbation and a cursor, then checks that resuming produces the same next state as uninterrupted execution.

## Before you start

Complete the state and recovery lessons, especially the lessons on optimizer state, PRNG key ownership and checkpoint identity. Use Python with NumPy from `requirements-cpu.txt`. Edit `projects/recovery-audit/starter/model.py` from the repository root.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python3 projects/recovery-audit/tests/check.py --implementation starter --stage 1
python3 projects/recovery-audit/tests/check.py --implementation starter --stage 2
python3 projects/recovery-audit/tests/check.py --implementation starter --stage 3 --evidence-dir runs/recovery-audit
python3 projects/recovery-audit/tests/check.py --implementation starter --stage 4
```

On Windows PowerShell, activate `.venv\Scripts\Activate.ps1` and use `python` in place of `python3`.

Stage 1 checks that one update changes the weight and advances the cursor. The velocity starts nonzero, so the update must carry existing momentum as well as the current gradient. The random perturbation is sampled from the saved key and a new key is returned in the state.

Stage 2 compares two updates in one process with one update, a saved checkpoint, restore, and the next update. Every field, including velocity, cursor and RNG key, must match exactly for this deterministic NumPy fixture. A weight-only restore or a key that is not advanced will fail this comparison.

Stage 3 rejects a changed learning-rate configuration and altered checkpoint bytes, then writes the diagnosis to `runs/recovery-audit/recovery-audit-evidence.json`. Checkpoints are replaced atomically within the destination directory. This local example does not claim power-loss durability or distributed checkpoint safety. Stage 4 runs all checks. Passing output begins `PASS recovery-audit stage N`.

Without `--evidence-dir`, evidence uses a temporary directory. Reference command: `python3 projects/recovery-audit/tests/check.py --implementation solution --stage 4`.

## Review handoff: explain a changed condition

Change dataset order without changing the filename, then attempt to reuse a saved cursor.

**Keep for review:** The old/new data identities, configuration compatibility decision and next-batch comparison. A cursor is meaningful only within its original ordering contract.

Write your prediction before opening the reference implementation. Distinguish what the existing checker verifies from your proposed extension; record any extension command, actual output and limitation separately. A reviewer should be able to locate the first failed contract from your saved evidence.
