# Phase 20: TPU Workflows: Launching Jobs & Checkpointed Experiments

Start here.

Step 2 of the 3-part Cloud TPU mini-course. Rehearse checkpointed jobs on CPU, then use only source-reviewed TPU targets with matching backend and dependency contracts. Separate single-host receipts from multi-host discovery and training qualification.

Do not run arbitrary lesson scripts on TPU. `distributed-02` is CPU-only; inspect every other source before considering a target. Existing receipts qualify only their named programs and recorded shapes, environments, and topologies: the one-device v5e1 launcher and precision target, and the four-device single-host v5e collective target. None qualifies multi-host execution or authorizes Transformers, Pallas, or other lesson scripts by association.

**Prerequisites:** 19: TPU Setup: Provisioning & Runtime Verification.

**Hardware:** CPU workflow planner; target execution only for source-reviewed programs with matching backend, dependencies, topology, and data-ownership contracts.

## Study guide: Can you launch a JAX job over SSH, separate XLA warmup from steady-state timing, and resume cleanly after interruption?

Cloud TPU VMs can be preempted and multi-host Pod slices require lockstep execution across all workers. A reliable workflow saves atomic checkpoints, logs JSONL events, and copies artifacts back before teardown.

### Check your starting point

Step 1 of your TPU job takes 14 seconds while steps 2–50 take 11 milliseconds each. How should your launcher record timing?

<details><summary>Compare your reasoning</summary>

Run and synchronize one warmup call before the timed loop and log it as a separate warmup event so XLA compilation does not pollute steady-state step latency.

</details>

Review: [Stage and launch checkpointed TPU jobs](01-stage-and-launch-checkpointed-tpu-jobs/docs/en.md).

### Build in stages

1. **Launch resumable TPU jobs with warmup separation.** Run run_tpu_ready_job, log warmup separately from timed steps, and verify that a checkpoint-resumed run matches an uninterrupted control run bit-for-bit.

   Lessons: [Stage and launch checkpointed TPU jobs](01-stage-and-launch-checkpointed-tpu-jobs/docs/en.md).

2. **Stage course scripts and launch multi-host Pod slices.** Classify `distributed-02` as CPU-only and leave unreviewed scripts blocked. Generate a TPU command only for a source-reviewed target with an explicit backend and pinned dependencies; distinguish `--worker=all` topology discovery from a collective training receipt with per-process data ownership and checked global updates.

   Lessons: [Review scripts before TPU staging](02-run-course-scripts-and-multihost-tpu-slices/docs/en.md).

### Try a changed condition

You move a script from a single-host v5litepod-4 VM to a 4-host v5litepod-16 Pod slice, and it hangs at startup. What is the first thing to check?

<details><summary>Compare an approach</summary>

For topology discovery, launch all hosts with `--worker=all` and initialize JAX before querying devices. This proves process/device discovery only. A training claim also needs explicit per-process inputs, a checked global update, synchronized outputs, and all worker receipts.

</details>

**Symptom:** A resumed run loads the saved model weights `w`, but its next loss diverges from the uninterrupted control run.

**Check next:** Verify that the checkpoint also restored optimizer momentum `v` and step counter `step` so the deterministic batch schedule continues at the exact next step.

### Decide what is ready

Keep the CPU checkpoint-resume comparison from `tpu-02`, the per-script execution matrix and rejected unsafe routes from `tpu-05`, and any target receipt only when its source hash, dependencies, backend, topology, and input ownership match the qualified program. Multi-host qualification remains separate.

### Further work

Continue to Phase 21 (TPU Systems: Generations, Memory, Precision & XProf) to budget HBM, test BF16/INT8 precision, and capture XProf traces.

## Lesson sequence

### 20.01 Stage and launch checkpointed TPU jobs

[Read the lesson](01-stage-and-launch-checkpointed-tpu-jobs/docs/en.md) · [Run the code](01-stage-and-launch-checkpointed-tpu-jobs/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Stage launch.py to a TPU VM, separate first-step XLA compilation warmup from steady-state step timing, and verify that resuming from an atomic checkpoint matches an uninterrupted run.

**Evidence:** Keep events.jsonl (started, warmup, step, checkpoint, restored, completed), checkpoint-latest.json, and matching resumed vs control state hashes.

**Checkpoint:** Why does run_tpu_ready_job execute train_step once and discard its outputs before starting the timed while step < steps loop?

### 20.02 Review scripts before TPU staging

[Read the lesson](02-run-course-scripts-and-multihost-tpu-slices/docs/en.md) · [Run the code](02-run-course-scripts-and-multihost-tpu-slices/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Classify the exact program before staging: keep `distributed-02` on CPU and generate a TPU launch only for a reviewed target whose backend, dependencies, topology, and data ownership are explicit. Distinguish multi-host discovery from a qualified distributed update.

**Evidence:** Keep the execution classification and rejected-route checks. For a target run, retain the exact source/environment hashes, selected backend and devices, data ownership, numerical check, and all worker receipts required by that target. A topology receipt alone is not a training result.

**Checkpoint:** What happens if you launch a JAX script on only worker 0 of a 4-host v5litepod-16 slice instead of passing --worker=all?

## Phase project

Gate simulated multi-worker checkpoint publication on complete receipts.

**Demonstrate:** Classify each exact program before staging. Route CPU-only and unreviewed scripts to CPU or inspection; use a TPU target only when its backend, pinned dependencies, topology, and data ownership are reviewed. Treat --worker=all discovery as topology evidence, not distributed training.

Project status: implemented staged practice · [Open source](../../projects/tpu-workflow-audit/README.md). Use stages 1, 2, 3, 4 for this phase. Work in projects/tpu-workflow-audit/starter/model.py. Complete stages 1 and 2 for independent calculations and changed-condition diagnosis, then retain the stage 3 evidence with --evidence-dir runs/tpu-workflow-audit. Stage 4 integrates the checks. Learner commands explicitly select the starter; --implementation solution is instructor reference execution. This bounded practice does not certify the entire phase or target hardware.

Additional project: [Operate and recover a local JAX workload](../../projects/workload-operations/README.md).

[Primary documentation](https://cloud.google.com/tpu/docs/run-calculation-jax).
