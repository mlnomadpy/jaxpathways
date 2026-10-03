# Phase 06: Data & checkpoint recovery

Training systems.

**Prerequisites:** 05: Neural network training.

**Hardware:** CPU first; TPU for system measurements.

## Lesson sequence

### 06.01 Design an input pipeline

Status: planned brief.

**Learn and build:** Measure data preparation and explain the distinction between data order and model state.

**Evidence:** A reproducible experiment for “Design an input pipeline”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Design an input pipeline”, identify one failure case, and show how you verified the fix.

### 06.02 Load, batch, and prefetch with Grain

Status: planned brief.

**Learn and build:** Build a deterministic input pipeline and check example order.

**Evidence:** A reproducible experiment for “Load, batch, and prefetch with Grain”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Load, batch, and prefetch with Grain”, identify one failure case, and show how you verified the fix.

### 06.03 Save model, optimizer, and random state

Status: planned brief.

**Learn and build:** Checkpoint all state required for the next training step with Orbax.

**Evidence:** A reproducible experiment for “Save model, optimizer, and random state”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Save model, optimizer, and random state”, identify one failure case, and show how you verified the fix.

### 06.04 Recover data position and resume

Status: planned brief.

**Learn and build:** Interrupt a run, restore data position where supported, and compare subsequent steps.

**Evidence:** A reproducible experiment for “Recover data position and resume”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Recover data position and resume”, identify one failure case, and show how you verified the fix.

### 06.05 Asynchronous checkpoints and failure boundaries

Status: planned brief.

**Learn and build:** Simulate an interruption and distinguish a started checkpoint from a completed one.

**Evidence:** A reproducible experiment for “Asynchronous checkpoints and failure boundaries”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Asynchronous checkpoints and failure boundaries”, identify one failure case, and show how you verified the fix.

## Phase project

An interrupted run restored with its full state.

**Demonstrate:** Compare resumed and uninterrupted runs under a stated determinism tolerance.

Project status: planned brief.

[Primary documentation](https://orbax.readthedocs.io/en/latest/).
