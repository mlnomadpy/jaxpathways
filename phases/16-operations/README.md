# Phase 16: Workload operations

Specializations.

**Prerequisites:** 09: Distributed training.

**Hardware:** Multi-device runtime; cluster exercises need controlled infrastructure access.

## Lesson sequence

### 16.01 Accelerator jobs and runtime lifecycle

Status: planned brief.

**Learn and build:** Define a job configuration, device requirements, and a restartable entry point.

**Evidence:** An operational artifact for “Accelerator jobs and runtime lifecycle”: configuration, observed workload output, and a tested failure or capacity scenario.

**Checkpoint:** Explain the assumptions behind “Accelerator jobs and runtime lifecycle” and demonstrate the behavior under one changed or failed condition.

### 16.02 Health, logs, and workload observability

Status: planned brief.

**Learn and build:** Record progress, throughput, failures, and checkpoint freshness so a stalled job can be diagnosed.

**Evidence:** An operational artifact for “Health, logs, and workload observability”: configuration, observed workload output, and a tested failure or capacity scenario.

**Checkpoint:** Explain the assumptions behind “Health, logs, and workload observability” and demonstrate the behavior under one changed or failed condition.

### 16.03 Failure recovery and operational runbooks

Status: planned brief.

**Learn and build:** Write and test a runbook for a worker failure using complete state recovery.

**Evidence:** An operational artifact for “Failure recovery and operational runbooks”: configuration, observed workload output, and a tested failure or capacity scenario.

**Checkpoint:** Explain the assumptions behind “Failure recovery and operational runbooks” and demonstrate the behavior under one changed or failed condition.

### 16.04 Changes, rollback, and artifact provenance

Status: planned brief.

**Learn and build:** Track configuration and artifact versions and define a rollback for a failed workload change.

**Evidence:** An operational artifact for “Changes, rollback, and artifact provenance”: configuration, observed workload output, and a tested failure or capacity scenario.

**Checkpoint:** Explain the assumptions behind “Changes, rollback, and artifact provenance” and demonstrate the behavior under one changed or failed condition.

### 16.05 Capacity, utilization, and operating cost

Status: planned brief.

**Learn and build:** Estimate resource needs from measured utilization and workload duration; explain the assumptions.

**Evidence:** An operational artifact for “Capacity, utilization, and operating cost”: configuration, observed workload output, and a tested failure or capacity scenario.

**Checkpoint:** Explain the assumptions behind “Capacity, utilization, and operating cost” and demonstrate the behavior under one changed or failed condition.

## Phase project

An observable, restartable accelerator job with a tested recovery runbook.

**Demonstrate:** Diagnose a failed or stalled job, restore it correctly, and justify its resource plan from recorded measurements.

Project status: planned brief.

[Primary documentation](https://docs.jax.dev/en/latest/).
