# Phase 21: TPU Systems: Generations, Memory, Precision & XProf

Start here.

Step 3 of the 3-part Cloud TPU mini-course. Compare TPU v5e, v5p, and v6e specifications and ridge points, budget training and KV-cache HBM before provisioning, measure BF16 and INT8 numerical error, and capture warmed XProf traces.

After completing this third TPU mini-course, you are ready to run any accelerator phase (Transformers, Performance, Distributed, Kernels, Deployment, Operations, Pretraining, Post-training) on a Cloud TPU VM.

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

An XProf trace on your TPU VM shows a 12-second host compile block followed by five 9-millisecond step blocks. How do you get a clean steady-state profile?

<details><summary>Compare an approach</summary>

Run at least one warmup call and call jax.block_until_ready() before entering with jax.profiler.trace(trace_dir): so only steady-state execution is recorded.

</details>

**Symptom:** Accumulating a long bfloat16 dot product directly in bfloat16 produces much larger drift than expected.

**Check next:** Pass preferred_element_type=jnp.float32 to jax.lax.dot_general so the TPU MXU accumulates products in FP32 before any output cast.

### Decide what is ready

Retain the TPU ridge-point and HBM budget tables from tpu-03, plus the BF16/INT8 error summary and *.xplane.pb trace receipt from tpu-06.

### Further work

Apply the 3-part Cloud TPU mini-course across Phases 07, 08, 09, 13, 15, 16, 17, and 18 by staging each lesson script in exercises/ to your TPU VM.

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

A TPU generation, HBM capacity, BF16/INT8 precision, and XProf profiling dossier.

**Demonstrate:** Compare TPU v5e, v5p, and v6e ridge points, budget training and KV-cache HBM, measure BF16/INT8 matmul error with FP32 accumulation, and capture a warmed XProf trace.

Project status: implemented staged practice · [Open source](../../projects/workload-operations/README.md). Use stages 5 for this phase. Compute your HBM budget and test precision/profiling locally first, then capture a warmed XProf trace on your Cloud TPU VM and copy it back with gcloud compute tpus tpu-vm scp --recurse. Copy `projects/workload-operations/starter/model.py` to `projects/workload-operations/my_model.py` and write your code in `projects/workload-operations/my_model.py`. Run `python3 projects/workload-operations/tests/check.py --implementation projects/workload-operations/my_model.py --stage 5` from the top-level folder to verify stage 5.



[Primary documentation](https://openxla.org/xprof).
