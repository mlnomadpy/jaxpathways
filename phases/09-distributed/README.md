# Phase 09: Distributed training

Specializations.

**Prerequisites:** 06: Data & checkpoint recovery; 08: Performance diagnosis.

**Hardware:** Multiple devices; multi-host labs require a cluster.

## Lesson sequence

### 09.01 Arrays, meshes, and sharding

Status: planned brief.

**Learn and build:** Partition an array over a device mesh and inspect its local shards.

**Evidence:** A reproducible experiment for “Arrays, meshes, and sharding”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Arrays, meshes, and sharding”, identify one failure case, and show how you verified the fix.

### 09.02 A sharded training step

Status: planned brief.

**Learn and build:** Run a training step with explicit data and parameter placement.

**Evidence:** A reproducible experiment for “A sharded training step”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “A sharded training step”, identify one failure case, and show how you verified the fix.

### 09.03 Communication-efficient algorithms

Status: planned brief.

**Learn and build:** Compare two partitioning strategies and account for their communication.

**Evidence:** A reproducible experiment for “Communication-efficient algorithms”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Communication-efficient algorithms”, identify one failure case, and show how you verified the fix.

### 09.04 Resilient distributed training

Status: planned brief.

**Learn and build:** Define a restore protocol for model, optimizer, and data state across workers.

**Evidence:** A reproducible experiment for “Resilient distributed training”: code, environment, observed output, and an explanation of one deliberate change.

**Checkpoint:** Explain your result for “Resilient distributed training”, identify one failure case, and show how you verified the fix.

## Phase project

A resilient sharded training experiment.

**Demonstrate:** Explain partitioning, communication costs, and recovery behavior with measured evidence.

Project status: planned brief.

[Primary documentation](https://docs.jax.dev/en/latest/).
