# Phase 20: TPU Workflows: Launching Jobs & Checkpointed Experiments

Start here.

Step 2 of the 3-part Cloud TPU mini-course. Stage scripts with gcloud compute tpus tpu-vm scp, separate XLA compilation warmup from steady-state step timing, recover interrupted runs from atomic checkpoints, and launch single-host or --worker=all Pod slice experiments.

Use the staging pattern from this course to run any lesson script in public/exercises/ on your Cloud TPU VM alongside your local CPU checks.

**Prerequisites:** 19: TPU Setup: Provisioning & Runtime Verification.

**Hardware:** CPU launcher rehearsal; Cloud TPU VM (single-host or --worker=all Pod slice) for target execution.

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

2. **Stage course scripts and launch multi-host Pod slices.** Build scp-in, ssh-run, and scp-out commands for any course script, use --worker=all on multi-host slices, and gate host artifact writes on jax.process_index() == 0.

   Lessons: [Run course scripts and multi-host TPU Pod slices](02-run-course-scripts-and-multihost-tpu-slices/docs/en.md).

### Try a changed condition

You move a script from a single-host v5litepod-4 VM to a 4-host v5litepod-16 Pod slice, and it hangs at startup. What is the first thing to check?

<details><summary>Compare an approach</summary>

Check that you launched the script across all hosts simultaneously using gcloud compute tpus tpu-vm ssh --worker=all and called jax.distributed.initialize() before querying jax.devices().

</details>

**Symptom:** A resumed run loads the saved model weights `w`, but its next loss diverges from the uninterrupted control run.

**Check next:** Verify that the checkpoint also restored optimizer momentum `v` and step counter `step` so the deterministic batch schedule continues at the exact next step.

### Decide what is ready

Retain the resumed vs control events.jsonl and state hashes from tpu-02, plus the single-host and --worker=all command workflows from tpu-05.

### Further work

Continue to Phase 21 (TPU Systems: Generations, Memory, Precision & XProf) to budget HBM, test BF16/INT8 precision, and capture XProf traces.

## Lesson sequence

### 20.01 Stage and launch checkpointed TPU jobs

[Read the lesson](01-stage-and-launch-checkpointed-tpu-jobs/docs/en.md) · [Run the code](01-stage-and-launch-checkpointed-tpu-jobs/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Stage launch.py to a TPU VM, separate first-step XLA compilation warmup from steady-state step timing, and verify that resuming from an atomic checkpoint matches an uninterrupted run.

**Evidence:** Keep events.jsonl (started, warmup, step, checkpoint, restored, completed), checkpoint-latest.json, and matching resumed vs control state hashes.

**Checkpoint:** Why does run_tpu_ready_job execute train_step once and discard its outputs before starting the timed while step < steps loop?

### 20.02 Run course scripts and multi-host TPU Pod slices

[Read the lesson](02-run-course-scripts-and-multihost-tpu-slices/docs/en.md) · [Run the code](02-run-course-scripts-and-multihost-tpu-slices/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Stage any course script in public/exercises/ to a TPU VM, construct single-host and --worker=all multi-host SSH launch commands, and verify jax.distributed.initialize() requirements.

**Evidence:** Keep the generated scp_in_cmd, ssh_run_cmd, and scp_out_cmd workflow for single-host and multi-host slices plus the process-0 checkpoint writer check.

**Checkpoint:** What happens if you launch a JAX script on only worker 0 of a 4-host v5litepod-16 slice instead of passing --worker=all?

## Phase project

A checkpoint-resumable TPU job launcher and multi-host staging workflow.

**Demonstrate:** Stage scripts with gcloud compute tpus tpu-vm scp, separate XLA warmup from timed step loops, verify zero-drift checkpoint resume, and build single-host and --worker=all Pod launch commands.

Project status: implemented staged practice · [Open source](../../projects/workload-operations/README.md). Use stages 3 for this phase. Rehearse checkpoint resume and command generation locally first, then stage launch.py to your Cloud TPU VM and copy back the run directory before deleting the VM. Copy `projects/workload-operations/starter/model.py` to `projects/workload-operations/my_model.py` and write your code in `projects/workload-operations/my_model.py`. Run `python3 projects/workload-operations/tests/check.py --implementation projects/workload-operations/my_model.py --stage 3` from the top-level folder to verify stage 3.



[Primary documentation](https://cloud.google.com/tpu/docs/run-calculation-jax).
