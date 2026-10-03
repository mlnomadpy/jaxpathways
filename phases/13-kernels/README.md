# Phase 13: Pallas kernels

Specializations.

**Prerequisites:** 09: Distributed training.

**Hardware:** Supported GPU or TPU; target-specific labs.

## Lesson sequence

### 13.01 Pallas grids and BlockSpecs

Status: planned brief.

**Learn and build:** Map a small array operation to a kernel grid and check its result.

**Evidence:** A reproducible experiment for “Pallas grids and BlockSpecs”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Pallas grids and BlockSpecs”, identify one failure case, and show how you verified the fix.

### 13.02 A first TPU kernel

Status: planned brief.

**Learn and build:** Implement and validate a target-specific kernel on supported hardware.

**Evidence:** A reproducible experiment for “A first TPU kernel”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “A first TPU kernel”, identify one failure case, and show how you verified the fix.

### 13.03 Tiling, memory, and pipelining

Status: planned brief.

**Learn and build:** Compare tiling choices and explain their memory and execution tradeoffs.

**Evidence:** A reproducible experiment for “Tiling, memory, and pipelining”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Tiling, memory, and pipelining”, identify one failure case, and show how you verified the fix.

### 13.04 Iterate with correctness and performance evidence

Status: planned brief.

**Learn and build:** Optimize a kernel while checking reference agreement over representative shapes.

**Evidence:** A reproducible experiment for “Iterate with correctness and performance evidence”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Iterate with correctness and performance evidence”, identify one failure case, and show how you verified the fix.

## Phase project

A measured custom kernel with correctness checks.

**Demonstrate:** Compare against a reference across shapes and justify each optimization with evidence.

Project status: planned brief.

[Primary documentation](https://docs.jax.dev/en/latest/pallas/index.html).
