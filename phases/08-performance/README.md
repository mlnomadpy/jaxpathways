# Phase 08: Performance diagnosis

Training systems.

Measure completed work, inspect a real profiler trace and relate compiler output to memory traffic. Compare rematerialization choices while separating saved residuals, compiler memory estimates and observed latency.

Keep workload shapes, warm-up policy and synchronized timings with your device report.

**Prerequisites:** 05: Neural network training; 19: TPU Setup: Provisioning & Runtime Verification; 20: TPU Workflows: Launching Jobs & Checkpointed Experiments; 21: TPU Systems: Generations, Memory, Precision & XProf.

**Hardware:** CPU for timing concepts; TPU or GPU for profiling.

## Study guide: What measured evidence identifies the bottleneck?

Optimize a verified computation and a named timing boundary. Trace events, compiler estimates and wall-clock samples answer different questions; connect them before proposing a change.

### Check your starting point

A JAX call returns an array handle almost immediately. Does a timer ending there necessarily measure completed computation?

<details><summary>Compare your reasoning</summary>

No. Synchronize the result inside the measured boundary. Report first-call setup separately from warmed execution and keep inputs, dtype and output checks fixed.

</details>

Review: [Benchmark asynchronous work correctly](01-benchmark-asynchronous-work-correctly/docs/en.md).

### Build in stages

1. **Choose a comparable measurement.** Record warmup, synchronization and sample distributions. Change shapes and host reads separately so you can identify their effects.

   Lessons: [Benchmark asynchronous work correctly](01-benchmark-asynchronous-work-correctly/docs/en.md) · [Diagnose recompilation and host synchronization](02-diagnose-recompilation-and-host-synchronization/docs/en.md).

2. **Explain the evidence.** Locate one annotated operation in a real trace, distinguish nested from additive time and state the memory level behind a roofline estimate.

   Lessons: [Read an XProf trace](03-read-an-xprof-trace/docs/en.md) · [Reason with HLO and the roofline model](04-reason-with-hlo-and-the-roofline-model/docs/en.md).

3. **Test a tradeoff.** Compare rematerialized and ordinary gradients before timing. Distinguish saved residuals, compiler estimates and observed peak memory.

   Lessons: [Memory, rematerialization, and optimization tradeoffs](05-memory-rematerialization-and-optimization-tradeoffs/docs/en.md).

### Try a changed condition

A kernel becomes twice as fast but the full step barely changes. What does the result suggest?

<details><summary>Compare an approach</summary>

First measure the kernel’s original share of the step. Even eliminating a small component has limited total benefit. Also inspect copies, synchronization and changed compilation; a local kernel timer cannot explain the complete pipeline alone.

</details>

**Symptom:** A claimed speedup disappears in repeated runs.

**Check next:** Check equivalent work, compilation reuse, synchronization, sample spread and competing system activity; retain the raw samples.

### Decide what is ready

Use the trace and timing stage of sharded-training. Keep before/after correctness, named measurement boundaries, trace locations and distributions with a justified accept/reject decision.

### Further work

CPU traces do not characterize TPU/GPU communication or memory. Target traces and sustained workload measurements remain required for accelerator claims.

## Lesson sequence

### 08.01 Benchmark asynchronous work correctly

[Read the lesson](01-benchmark-asynchronous-work-correctly/docs/en.md) · [Run the code](01-benchmark-asynchronous-work-correctly/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Warm up a computation and synchronize results before reporting timing.

**Evidence:** Keep benchmark.py, complete observed JSON/sample output, workload shapes/dtypes, numerical tolerance/error, first-call versus warmed boundaries, changed-workload report and repaired timer explanation.

**Checkpoint:** Distinguish call-plus-wait wall time from compiler/kernel time; reproduce a new workload with an independent reference.

### 08.02 Diagnose recompilation and host synchronization

[Read the lesson](02-diagnose-recompilation-and-host-synchronization/docs/en.md) · [Run the code](02-diagnose-recompilation-and-host-synchronization/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Find a shape-changing call or host read and connect it to observed overhead.

**Evidence:** Keep the controlled call ledger, actual trace observations labeled as traces, static/dynamic explanation, observed host-consumption intervals, masked-versus-unmasked counterexample and reproduced/repaired tracing failure.

**Checkpoint:** Explain which argument changes affect specialization; justify masked fixed shapes and host reporting boundaries with numerical evidence.

### 08.03 Read an XProf trace

[Read the lesson](03-read-an-xprof-trace/docs/en.md) · [Run the code](03-read-an-xprof-trace/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Capture an actual warmed CPU trace with named steps. Identify nested intervals and avoid double-counting their durations.

**Evidence:** Keep the actual XPlane and Perfetto paths, named interval durations and nesting check, units and an interpretation separating injected input delay from completed computation.

**Checkpoint:** A step annotation lasts longer than its model annotation. What can this alone establish?

### 08.04 Reason with HLO and the roofline model

[Read the lesson](04-reason-with-hlo-and-the-roofline-model/docs/en.md) · [Run the code](04-reason-with-hlo-and-the-roofline-model/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Locate matrix multiplication and activation in StableHLO. Derive the operation and minimum byte counts for a declared matrix contract.

**Evidence:** Keep StableHLO, synchronized CPU samples and independent output parity. Derive the byte/FLOP model and label the roofline peak and bandwidth as hypothetical.

**Checkpoint:** The lowered program contains a dot operation and a roofline model predicts a high ceiling. What is still needed before claiming high accelerator throughput?

### 08.05 Memory, rematerialization, and optimization tradeoffs

[Read the lesson](05-memory-rematerialization-and-optimization-tradeoffs/docs/en.md) · [Run the code](05-memory-rematerialization-and-optimization-tradeoffs/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Explain why reverse-mode differentiation needs forward intermediates. Compare ordinary and rematerialized gradients with an independent reverse recurrence.

**Evidence:** Keep the plain/rematerialized gradient and HVP comparisons, saved-residual counts, compiler-memory observations and synchronized timings. Explain why fewer residuals do not guarantee lower measured memory.

**Checkpoint:** Rematerialization reduces the number of saved residual descriptions but the compiled temporary-byte estimates are equal. What is the supported conclusion?

## Phase project

A trace-backed before-and-after performance report.

**Demonstrate:** Separate compilation, execution, input stalls, and synchronization before claiming a speedup.

Project status: implemented staged practice · [Open source](../../projects/sharded-training/README.md). Use stages 4 for this phase. This connected project also uses performance and distributed lessons. Work through its prerequisites before the full integration check; return here with the completed evidence. Copy `projects/sharded-training/starter/model.py` to `projects/sharded-training/my_model.py` and write your code in `projects/sharded-training/my_model.py`. Run `python3 projects/sharded-training/tests/check.py --implementation projects/sharded-training/my_model.py --stage 4` from the top-level folder to verify stage 4.



[Primary documentation](https://openxla.org/xprof).
