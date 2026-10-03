# Phase 08: Performance diagnosis

Training systems.

**Prerequisites:** 05: Neural network training.

**Hardware:** CPU for timing concepts; TPU or GPU for profiling.

## Lesson sequence

### 08.01 Benchmark asynchronous work correctly

Status: planned brief.

**Learn and build:** Warm up a computation and synchronize results before reporting timing.

**Evidence:** A reproducible experiment for “Benchmark asynchronous work correctly”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Benchmark asynchronous work correctly”, identify one failure case, and show how you verified the fix.

### 08.02 Diagnose recompilation and host synchronization

Status: planned brief.

**Learn and build:** Find a shape-changing call or host read and connect it to observed overhead.

**Evidence:** A reproducible experiment for “Diagnose recompilation and host synchronization”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Diagnose recompilation and host synchronization”, identify one failure case, and show how you verified the fix.

### 08.03 Read an XProf trace

Status: planned brief.

**Learn and build:** Capture and annotate a trace showing execution and input behavior.

**Evidence:** A reproducible experiment for “Read an XProf trace”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Read an XProf trace”, identify one failure case, and show how you verified the fix.

### 08.04 Reason with HLO and the roofline model

Status: planned brief.

**Learn and build:** Relate a measured bottleneck to arithmetic intensity and program structure.

**Evidence:** A reproducible experiment for “Reason with HLO and the roofline model”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Reason with HLO and the roofline model”, identify one failure case, and show how you verified the fix.

### 08.05 Memory, rematerialization, and optimization tradeoffs

Status: planned brief.

**Learn and build:** Compare memory and execution time while preserving numerical checks.

**Evidence:** A reproducible experiment for “Memory, rematerialization, and optimization tradeoffs”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Memory, rematerialization, and optimization tradeoffs”, identify one failure case, and show how you verified the fix.

## Phase project

A trace-backed before-and-after performance report.

**Demonstrate:** Separate compilation, execution, input stalls, and synchronization before claiming a speedup.

Project status: planned brief.

[Primary documentation](https://openxla.org/xprof).
