# Phase 21: TPU Systems: Generations, Memory, Precision & XProf

Start here.

Step 3 of the 3-part Cloud TPU mini-course. Compare TPU v5e, v5p, and v6e specifications and ridge points, budget training and KV-cache HBM before provisioning, measure BF16 and INT8 numerical error, and capture warmed XProf traces.

After completing this third TPU mini-course, you can plan a candidate accelerator workload and interpret hardware evidence. Before running a program on TPU, review its exact source, dependencies, backend assertion, topology, and data ownership; then qualify that program and configuration with target-specific evidence. Existing receipts do not authorize other accelerator phases by association.

**Prerequisites:** 20: TPU Workflows: Launching Jobs & Checkpointed Experiments.

**Hardware:** CPU numerical & profiler harness; Cloud TPU VM for target XProf traces.

## Study guide: How do you size TPU HBM capacity, bound BF16/INT8 precision error, and capture a clean XProf trace?

Budgeting HBM helps reject workloads that exceed the declared training-state capacity before provisioning. Synchronized warmup reduces compilation and first-use work in the measured interval; inspect the trace for recompilation, transfers and host activity rather than assuming it contains only accelerator execution.

### Check your starting point

A matrix multiplication has arithmetic intensity 170 FLOP/byte on TPU v5e (ridge point 240.5 FLOP/byte). Is it compute-bound or HBM bandwidth-bound?

<details><summary>Compare your reasoning</summary>

Under this ideal roofline model, 170 FLOP/byte is below the TPU v5e ridge of 240.5 FLOP/byte, so the HBM bandwidth ceiling is lower. Changing batch or sequence length can change data reuse and operational intensity; recompute the traffic assumptions and measure the actual workload before claiming a bottleneck or speedup.

</details>

Review: [TPU generations and HBM memory budgeting](01-tpu-generations-and-hbm-memory-budgeting/docs/en.md).

### Build in stages

1. **Compare TPU generations and budget HBM capacity.** Compute the v5e/v5p/v6e ideal ridge points and separate training-state from inference KV-cache bytes. Compare replicated and explicitly sharded per-device ledgers under headroom and temporary/activation assumptions before deriving any conditional minimum chip count.

   Lessons: [TPU generations and HBM memory budgeting](01-tpu-generations-and-hbm-memory-budgeting/docs/en.md).

2. **Audit BF16/INT8 precision and capture a warmed XProf trace.** Compare BF16 operand/requested-result policies and an INT8 weight-dequantization reference against FP32, inspect returned dtypes and retained CPU compiler IR, and capture a warmed *.xplane.pb trace with its backend recorded.

   Lessons: [bfloat16, int8 precision, and XProf profiling on TPU](02-bfloat16-int8-precision-and-xprof-profiling/docs/en.md).

### Try a changed condition

At the stated ideal byte count, a BF16 1024-square matmul has about 341.3 FLOP/byte. Compare v5e’s 240.5 ridge with v6e’s 560.4 ridge.

<details><summary>Compare an approach</summary>

The operation is above v5e’s ideal intersection and below v6e’s. That selects the lower of theoretical ceilings, not measured bottlenecks or a device ranking. Record actual precision, output bytes and traffic assumptions; validate the hypothesis with target timing and a trace before claiming a speedup.

</details>

**Symptom:** A BF16 result differs from a requested FP32 result even when both dots receive the same BF16-rounded operands.

**Check next:** Compare actual returned dtypes and an explicit FP32-result-to-BF16 cast, then inspect compiler evidence on the named backend. A result dtype or larger output error does not prove that every partial sum used that dtype.

### Decide what is ready

Retain the TPU ridge-point and HBM budget tables from tpu-03, plus the BF16/INT8 error summary and *.xplane.pb trace receipt from tpu-06.

### Further work

Use the reviewed execution matrix before staging code on a TPU. CPU-only lesson companions stay on CPU; only source-reviewed targets with matching dependencies and topology should be launched. The retained v5e receipts qualify their named workloads, not every course script or a production model.

## Lesson sequence

### 21.01 TPU generations and HBM memory budgeting

[Read the lesson](01-tpu-generations-and-hbm-memory-budgeting/docs/en.md) · [Run the code](01-tpu-generations-and-hbm-memory-budgeting/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Compare TPU generation tradeoffs, prove per-device replicated versus sharded memory fit, and budget training state separately from inference KV cache.

**Evidence:** Keep aggregate and per-device state ledgers, replication/sharding/headroom assumptions, invalid aggregate-fit counterexamples and separate training-versus-inference capacity decisions.

**Checkpoint:** Under this lesson's ledger, why does mixed-BF16 Adam training require 18 bytes per parameter before activations and temporary buffers are counted?

### 21.02 bfloat16, int8 precision, and XProf profiling on TPU

[Read the lesson](02-bfloat16-int8-precision-and-xprof-profiling/docs/en.md) · [Run the code](02-bfloat16-int8-precision-and-xprof-profiling/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Measure BF16 operand and requested result-dtype error against a finite-precision FP32 JAX reference, inspect CPU dtype and compiler evidence, and retain a warmed XProf trace with its observed backend.

**Evidence:** Keep the returned-dtype/error table, seeded cast comparisons, source-bound CPU compiler IR and captured *.xplane.pb trace receipt. State the observed backend and the limits of each artifact.

**Checkpoint:** Why should you run and synchronize at least one warmup step before entering with jax.profiler.trace(trace_dir): on a TPU VM?

## Phase project

Estimate optimizer-state bytes and calculate supplied-rate scaling efficiency.

**Demonstrate:** Compare published TPU roofline estimates, budget training and KV-cache HBM, compare BF16 result-dtype policies and an INT8 weight-dequantization reference, and retain a warmed profiler trace with its observed backend.

Project status: implemented staged practice · [Open source](../../projects/tpu-systems-audit/README.md). Use stages 1, 2, 3, 4 for this phase. Work in projects/tpu-systems-audit/starter/model.py. Complete stages 1 and 2 for independent calculations and changed-condition diagnosis, then retain the stage 3 evidence with --evidence-dir runs/tpu-systems-audit. Stage 4 integrates the checks. Learner commands explicitly select the starter; --implementation solution is instructor reference execution. This bounded practice does not certify the entire phase or target hardware.

Additional project: [Operate and recover a local JAX workload](../../projects/workload-operations/README.md).

[Primary documentation](https://openxla.org/xprof).
