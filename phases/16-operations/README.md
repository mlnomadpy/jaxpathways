# Phase 16: Workload operations

Specializations.

Launch and observe real local jobs, detect failures and stalls, restore complete training state, verify artifact quality and rehearse rollback before planning capacity. Connect MLOps data/release gates, LLMOps evaluation/traces and ModelOps ownership/approval to the workload runbook.

Use a bounded failure drill and retain its raw event log. Check state and data identity before resuming; separate measured execution from hypothetical demand and pricing.

**Prerequisites:** 09: Distributed training.

**Hardware:** Executed local CPU processes; accelerator and cluster transfer require separate validation.

## Study guide: Can you detect, explain and safely recover a failed model release?

Operational evidence connects work done to user-visible behavior. Separate worker health, model quality and release ownership so an alert leads to a specific investigation rather than automatic retraining.

### Check your starting point

A worker process is alive but has emitted no progress for a long time. Is it healthy?

<details><summary>Compare your reasoning</summary>

Liveness only establishes that the process exists or responds. Check readiness, last completed work, expected duration and checkpoint freshness before choosing a bounded timeout or recovery action.

</details>

Review: [Health, logs, and workload observability](02-health-logs-and-workload-observability/docs/en.md).

### Build in stages

1. **Observe and recover real work.** Launch a bounded worker, retain structured events, rehearse a failure and verify the next restored update against uninterrupted work.

   Lessons: [Accelerator jobs and runtime lifecycle](01-accelerator-jobs-and-runtime-lifecycle/docs/en.md) · [Health, logs, and workload observability](02-health-logs-and-workload-observability/docs/en.md) · [Failure recovery and operational runbooks](03-failure-recovery-and-operational-runbooks/docs/en.md).

2. **Control change and capacity.** Reject invalid artifacts without changing selection. Revalidate rollback targets and calculate demand, reserve and cost with consistent measured units.

   Lessons: [Changes, rollback, and artifact provenance](04-changes-rollback-and-artifact-provenance/docs/en.md) · [Capacity, utilization, and operating cost](05-capacity-utilization-and-operating-cost/docs/en.md).

3. **Connect MLOps, LLMOps and ModelOps.** Use data and slice-quality gates, trace a versioned text application and bind review to the complete release bundle. Keep evaluator replay separate from live-model quality.

   Lessons: [MLOps: data contracts, CI gates and monitoring](06-mlops-data-contracts-and-release-pipelines/docs/en.md) · [LLMOps: version prompts, evaluate behavior and trace requests](07-llmops-prompts-evaluation-and-traces/docs/en.md) · [ModelOps: own, approve, roll back and retire a model](08-modelops-ownership-approval-and-retirement/docs/en.md).

### Try a changed condition

A new artifact passes a global score, but one required slice fails its threshold. The previous artifact is healthy. What should the release pointer do?

<details><summary>Compare an approach</summary>

Keep the previous accepted artifact active and retain the failed candidate’s evidence. A global average cannot override a required slice gate. Record artifact/config identity and the reason for rejection. Test that a failed promotion leaves the active selection unchanged; rollback must select a verified prior artifact, not simply a filename.

</details>

**Symptom:** A release passes average error while a rare critical slice fails.

**Check next:** Require per-slice counts and quality thresholds. Do not average slice means equally or treat a missing critical slice as a passing zero.

### Decide what is ready

Use workload-operations and engineering-release. Keep a real failure/recovery timeline, failed-gate invariance, a rollback drill, run/model identity and named review responsibilities.

### Further work

Authenticated review, multi-writer promotion, staged cloud rollout, SLO alerting and live-model abuse evaluation still require separate infrastructure and application-specific labs.

## Lesson sequence

### 16.01 Accelerator jobs and runtime lifecycle

[Read the lesson](01-accelerator-jobs-and-runtime-lifecycle/docs/en.md) · [Run the code](01-accelerator-jobs-and-runtime-lifecycle/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Launch and reap a real bounded subprocess with explicit runtime requirements. Distinguish successful completion, worker failure and supervisor timeout.

**Evidence:** Keep actual child exit codes, bounded timeout/termination output and proof that children were reaped. Distinguish launch failure, failed preflight, slow work and a stalled job.

**Checkpoint:** What must a timeout handler do beyond recording a timeout status?

### 16.02 Health, logs, and workload observability

[Read the lesson](02-health-logs-and-workload-observability/docs/en.md) · [Run the code](02-health-logs-and-workload-observability/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Build useful summaries from structured events emitted by a real worker. Define throughput with explicit numerator, denominator and synchronization boundary.

**Evidence:** Keep raw structured events, observed work counts, synchronized timing units and checkpoint lag. Derive job and update throughput with different boundaries.

**Checkpoint:** What does update duty fraction establish in this lab?

### 16.03 Failure recovery and operational runbooks

[Read the lesson](03-failure-recovery-and-operational-runbooks/docs/en.md) · [Run the code](03-failure-recovery-and-operational-runbooks/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Rehearse a real worker failure and resume from an atomic complete-state checkpoint. Verify the next update and every later state against an uninterrupted reference.

**Evidence:** Keep the injected failure timeline, accepted checkpoint identity and full-state comparisons for every resumed update. Record the incompatible and corrupted restore rejections.

**Checkpoint:** Which evidence most directly supports a correct resume?

### 16.04 Changes, rollback, and artifact provenance

[Read the lesson](04-changes-rollback-and-artifact-provenance/docs/en.md) · [Run the code](04-changes-rollback-and-artifact-provenance/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Activate and roll back identified local model bundles, verify consumer predictions and preserve the active model after a rejected selection.

**Evidence:** Keep model/checkpoint identities, observed predictions before activation, after activation and after rollback, and unchanged selection/behavior on rejection.

**Checkpoint:** What does a matching artifact digest establish?

### 16.05 Capacity, utilization, and operating cost

[Read the lesson](05-capacity-utilization-and-operating-cost/docs/en.md) · [Run the code](05-capacity-utilization-and-operating-cost/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Measure end-to-end local service time at more than one workload length. Compute offered load, worker reserve and illustrative cost with consistent units.

**Evidence:** Keep measured job/update samples and independent unit calculations. Label demand, reserve and price assumptions as hypothetical before deriving a resource plan.

**Checkpoint:** Which claim is supported by the measured update duty fraction?

### 16.06 MLOps: data contracts, CI gates and monitoring

[Read the lesson](06-mlops-data-contracts-and-release-pipelines/docs/en.md) · [Run the code](06-mlops-data-contracts-and-release-pipelines/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Evaluate predictions from a reloaded local artifact and apply group-safe, count-weighted slice and release rules.

**Evidence:** Keep reloaded predictions, target/case identities, per-example and slice metrics, independent weighted aggregation and an explicit release decision with leakage and drift caveats.

**Checkpoint:** An input drift alert fired before new labels arrived. What follows?

### 16.07 LLMOps: version prompts, evaluate behavior and trace requests

[Read the lesson](07-llmops-prompts-evaluation-and-traces/docs/en.md) · [Run the code](07-llmops-prompts-evaluation-and-traces/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Trace an executed local retrieval, prompt and answer workflow, pair responses by case ID and diagnose component failures before comparing candidates.

**Evidence:** Keep heterogeneous case IDs, actual local-stage traces and retrieved context, missing/duplicate/out-of-order controls, component diagnoses and a same-case candidate comparison. The local adapter is not an LLM quality benchmark.

**Checkpoint:** The controlled replay suite passes every case. What is established?

### 16.08 ModelOps: own, approve, roll back and retire a model

[Read the lesson](08-modelops-ownership-approval-and-retirement/docs/en.md) · [Run the code](08-modelops-ownership-approval-and-retirement/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Bind approval to a model bundle, verify consumer activation/rollback, and enforce retired-target and fallback policies.

**Evidence:** Keep approval/bundle identity, predictions across activation and rollback, retirement/cache/reload behavior, stale/expired/retired-target failures and an audit trail. Scope is a local single-writer policy fixture.

**Checkpoint:** An approval record contains a reviewer name and a matching hash. Does that authenticate the reviewer?

## Phase project

An observable, restartable local workload.

**Demonstrate:** Diagnose a failed or stalled job, restore it correctly, and justify its resource plan from recorded measurements.

Project status: implemented staged practice · [Open source](../../projects/workload-operations/README.md). Copy `projects/workload-operations/starter/model.py` to `projects/workload-operations/my_model.py` and write your code in `projects/workload-operations/my_model.py`. Run `python3 projects/workload-operations/tests/check.py --implementation projects/workload-operations/my_model.py --stage 1` from the top-level folder to verify all 5 stages.

Additional project: [Ship a tracked and containerized model release](../../projects/engineering-release/README.md).

[Primary documentation](https://docs.jax.dev/en/latest/).
