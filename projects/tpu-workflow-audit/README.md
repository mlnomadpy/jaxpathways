# Gate a simulated multi-worker checkpoint publication

A coordinator should publish a checkpoint only when every expected worker has produced a receipt at a common step. Missing workers, duplicate IDs and mismatched progress can make a partial checkpoint look complete. This project checks the gate with in-memory lists. It does not launch workers or contact a TPU VM.

## Before you start

Complete the TPU workflow lessons and understand the difference between a staged command and a distributed run. This simulation needs only the course CPU environment. Edit `projects/tpu-workflow-audit/starter/model.py` from the repository root.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python3 projects/tpu-workflow-audit/tests/check.py --implementation starter --stage 1
python3 projects/tpu-workflow-audit/tests/check.py --implementation starter --stage 2
python3 projects/tpu-workflow-audit/tests/check.py --implementation starter --stage 3 --evidence-dir runs/tpu-workflow-audit
python3 projects/tpu-workflow-audit/tests/check.py --implementation starter --stage 4
```

On Windows PowerShell, activate `.venv\Scripts\Activate.ps1` and use `python` instead of `python3`.

Stage 1 gives expected workers `w0,w1,w2`, but receipts only from `w0,w2`. Publication must be false and `w1` must be listed as missing. When both workers report step 7, the gate can mark the set publishable.

Stage 2 changes the receipt steps to 7 and 6, then tries duplicate worker IDs, duplicate receipts, an unknown worker, a negative step, non-integer steps, NaN and infinity. Unequal progress must block publication; invalid inputs must be rejected. If the gate accepts any of these, inspect receipt identity and step validation before the publication decision.

Stage 3 writes the complete receipt-gate result and its simulation scope to `runs/tpu-workflow-audit/tpu-workflow-audit-evidence.json`. Without `--evidence-dir`, it uses a temporary directory. Stage 4 reruns all stages. Passing output begins `PASS tpu-workflow-audit stage N`.

This tests publication logic on data structures only. It does not prove multi-host startup, network reliability, checkpoint durability or TPU behavior. Reference command: `python3 projects/tpu-workflow-audit/tests/check.py --implementation solution --stage 4`.
