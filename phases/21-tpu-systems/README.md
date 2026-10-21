# Phase 21: TPU Systems: Generations, Memory, Precision & XProf

Start here.

Step 3 of the 3-part Cloud TPU mini-course. Compare TPU v5e, v5p, and v6e specifications and ridge points, budget training and KV-cache HBM before provisioning, measure BF16 and INT8 numerical error, and capture warmed XProf traces.

After completing this third TPU mini-course, you can plan a candidate accelerator workload and interpret hardware evidence. Before running a program on TPU, review its exact source, dependencies, backend assertion, topology, and data ownership; then qualify that program and configuration with target-specific evidence. Existing receipts do not authorize other accelerator phases by association.

**Prerequisites:** 20: TPU Workflows: Launching Jobs & Checkpointed Experiments.

**Hardware:** CPU numerical & profiler harness; Cloud TPU VM for target XProf traces.

## Study guide: How do you size TPU HBM capacity, bound BF16/INT8 precision error, and capture a clean XProf trace?

Provisioning a TPU slice without budgeting HBM wastes time on out-of-memory crashes, and profiling without warming up measures compiler overhead instead of MXU execution.

### Check your starting point

A matrix multiplication has arithmetic intensity 170 FLOP/byte on TPU v5e (ridge point 240.5 FLOP/byte). Is it compute-bound or HBM bandwidth-bound?

<details><summary>Compare your reasoning</summary>

HBM bandwidth-bound, because its arithmetic intensity (170 FLOP/byte) is below the TPU v5e ridge point (240.5 FLOP/byte). Increasing batch size or sequence length raises operational intensity toward the MXU ceiling.

</details>

Review: [TPU generations and HBM memory budgeting](01-tpu-generations-and-hbm-memory-budgeting/docs/en.md).

### Build in stages

1. **Compare TPU generations and budget HBM capacity.** Compute TPU v5e, v5p, and v6e ridge points, calculate FP32 vs mixed-BF16 training state and KV-cache GiB, and determine the minimum chip count at an 80% HBM budget.

   Lessons: [TPU generations and HBM memory budgeting](01-tpu-generations-and-hbm-memory-budgeting/docs/en.md).

2. **Audit BF16/INT8 precision and capture a warmed XProf trace.** Compare BF16 and symmetric INT8 matmul errors against FP32, verify FP32 accumulation inside jax.lax.dot_general, and capture a warmed *.xplane.pb trace.

   Lessons: [bfloat16, int8 precision, and XProf profiling on TPU](02-bfloat16-int8-precision-and-xprof-profiling/docs/en.md).

### Try a changed condition

At the stated ideal byte count, a BF16 1024-square matmul has about 341.3 FLOP/byte. Compare v5e’s 240.5 ridge with v6e’s 560.4 ridge.

<details><summary>Compare an approach</summary>

The operation is above v5e’s ideal intersection and below v6e’s. That selects the lower of theoretical ceilings, not measured bottlenecks or a device ranking. Record actual precision, output bytes and traffic assumptions; validate the hypothesis with target timing and a trace before claiming a speedup.

</details>

**Symptom:** Accumulating a long bfloat16 dot product directly in bfloat16 produces much larger drift than expected.

**Check next:** Pass preferred_element_type=jnp.float32 to jax.lax.dot_general so the TPU MXU accumulates products in FP32 before any output cast.

### Decide what is ready

Retain the TPU ridge-point and HBM budget tables from tpu-03, plus the BF16/INT8 error summary and *.xplane.pb trace receipt from tpu-06.

### Further work

Use the reviewed execution matrix before staging code on a TPU. CPU-only lesson companions stay on CPU; only source-reviewed targets with matching dependencies and topology should be launched. The retained v5e receipts qualify their named workloads, not every course script or a production model.

## Lesson sequence

### 21.01 TPU generations and HBM memory budgeting

[Read the lesson](01-tpu-generations-and-hbm-memory-budgeting/docs/en.md) · [Run the code](01-tpu-generations-and-hbm-memory-budgeting/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Compare TPU v5e, v5p, and v6e specifications and roofline ridge points, budget FP32 vs mixed-BF16 training state and KV-cache HBM, and size the minimum chip count for a model.

**Evidence:** Keep the TPU ridge-point table, 7B and 1.5B training/KV-cache HBM budgets, and minimum chip count calculation at an 80% HBM ceiling.

**Checkpoint:** Why does mixed-precision Adam training still require ~12 bytes per parameter in HBM even when forward and backward matmuls use 2-byte bfloat16?

### 21.02 bfloat16, int8 precision, and XProf profiling on TPU

[Read the lesson](02-bfloat16-int8-precision-and-xprof-profiling/docs/en.md) · [Run the code](02-bfloat16-int8-precision-and-xprof-profiling/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Measure bfloat16 and symmetric int8 matrix multiplication error against float32, verify float32 accumulator precision, and capture a warmed XProf (*.xplane.pb) trace.

**Evidence:** Keep the BF16 vs INT8 error summary, FP32-vs-BF16 accumulator drift comparison, and captured *.xplane.pb trace receipt.

**Checkpoint:** Why should you run and synchronize at least one warmup step before entering with jax.profiler.trace(trace_dir): on a TPU VM?

## Phase project

Estimate optimizer-state bytes and calculate supplied-rate scaling efficiency.

**Demonstrate:** Compare TPU v5e, v5p, and v6e ridge points, budget training and KV-cache HBM, measure BF16/INT8 matmul error with FP32 accumulation, and capture a warmed XProf trace.

Project status: implemented staged practice · [Open source](../../projects/tpu-systems-audit/README.md). Use stages 1, 2, 3, 4 for this phase. Work in projects/tpu-systems-audit/starter/model.py. Complete stages 1 and 2 for independent calculations and changed-condition diagnosis, then retain the stage 3 evidence with --evidence-dir runs/tpu-systems-audit. Stage 4 integrates the checks. Learner commands explicitly select the starter; --implementation solution is instructor reference execution. This bounded practice does not certify the entire phase or target hardware.

Additional project: [Operate and recover a local JAX workload](../../projects/workload-operations/README.md).

[Primary documentation](https://openxla.org/xprof).
