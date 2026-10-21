# Estimate optimizer-state bytes and interpret a scaling rate

A memory estimate is only as complete as its named tensors. This project accounts for weights, FP32 master weights and two FP32 optimizer moments. It then calculates ideal throughput from a supplied single-worker rate and worker count. It does not inspect device memory or measure TPU scaling.

## Before you start

Complete the TPU systems lessons on memory budgets and precision. Know that BF16 weights use two bytes per parameter while FP32 masters and Adam moments use four bytes each. Run from the repository root with `requirements-cpu.txt` installed, and edit `projects/tpu-systems-audit/starter/model.py`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python3 projects/tpu-systems-audit/tests/check.py --implementation starter --stage 1
python3 projects/tpu-systems-audit/tests/check.py --implementation starter --stage 2
python3 projects/tpu-systems-audit/tests/check.py --implementation starter --stage 3 --evidence-dir runs/tpu-systems-audit
python3 projects/tpu-systems-audit/tests/check.py --implementation starter --stage 4
```

On Windows PowerShell, activate `.venv\Scripts\Activate.ps1` and use `python` rather than `python3`.

Stage 1 uses 100 parameters, two bytes per stored weight, four bytes per master weight, and two FP32 moments. The ledger is 200 + 400 + 800 = 1400 bytes. With a supplied single-worker rate of 50 examples/second and four workers, the ideal reference is 200 examples/second. A supplied measured rate of 160 gives a calculated ratio of 0.8.

Stage 2 changes the parameter count to 50 and the supplied rate to 300 with two workers. The state estimate is 700 bytes and the ideal rate is 100, so the ratio is 3.0. Treat this as a ratio of supplied values, not proof of superlinear device speedup. The stage also rejects zero throughput. If the ledger is wrong, check whether moment bytes use four bytes per moment rather than the stored weight dtype.

Stage 3 writes the byte ledger, rate calculation and scope note to `runs/tpu-systems-audit/tpu-systems-audit.json`. The report says that no device memory or TPU scaling was measured. Without `--evidence-dir`, evidence is temporary. Stage 4 runs all checks. Passing output begins `PASS tpu-systems-audit stage N`.

The estimate excludes gradients, activations, communication buffers, temporary copies and allocator overhead. Add those before making a capacity decision. Reference command: `python3 projects/tpu-systems-audit/tests/check.py --implementation solution --stage 4`.

## Review handoff: explain a changed condition

Convert 918 TFLOP/s and 1638 GB/s into a Roofline intersection. Compare a hypothetical 341.3 FLOP/byte workload.

**Keep for review:** About 560.4 FLOP/byte after the factor-of-1000 unit conversion. The hypothetical workload lies below that ideal ridge; actual target bottlenecks, native precision and latency remain to be measured.

Write your prediction before opening the reference implementation. Distinguish what the existing checker verifies from your proposed extension; record any extension command, actual output and limitation separately. A reviewer should be able to locate the first failed contract from your saved evidence.
