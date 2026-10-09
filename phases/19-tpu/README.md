# Phase 19: TPU Setup: Provisioning & Runtime Verification

Start here.

Step 1 of the 3-part Cloud TPU mini-course. Choose the right TPU slice, provision a Cloud TPU VM (or Colab TPU), bootstrap ~/jax-tpu-lab/.venv with jax[tpu], verify a synchronized pmap runtime receipt, and practice clean teardown.

Complete this short 2-lesson setup course right after 00: Setup & first steps, then continue directly to 20: TPU Workflows and 21: TPU Systems.

**Prerequisites:** 00: Setup & first steps.

**Hardware:** CPU topology/contract verifier; Cloud TPU VM (v5litepod-4 / v6e-4) or Colab TPU for target runs.

## Study guide: Can you provision a Cloud TPU VM, verify that JAX actually executes on its chips, and prove clean teardown?

Every accelerator experiment starts with knowing how many hosts and GiB of HBM your slice provides, verifying that libtpu is active inside an isolated .venv, and deleting billable VMs when finished.

### Check your starting point

You created a Cloud TPU VM and gcloud says READY. What still has to happen before you trust a JAX run on it?

<details><summary>Compare your reasoning</summary>

Bootstrap ~/jax-tpu-lab/.venv with jax[tpu], verify inside Python that jax.default_backend() == 'tpu' and jax.device_count() matches your slice, and run a synchronized pmap reduction receipt.

</details>

Review: [Select a TPU slice and provision your Cloud TPU VM](01-get-a-tpu-running-and-verify-the-runtime/docs/en.md).

### Build in stages

1. **Select a single-host TPU topology and provision the VM.** Compare v5litepod-1, v5litepod-4, v5litepod-8, and v6e-4 HBM totals, distinguish single-host VMs from multi-host Pod slices, and run gcloud create/describe/delete.

   Lessons: [Select a TPU slice and provision your Cloud TPU VM](01-get-a-tpu-running-and-verify-the-runtime/docs/en.md).

2. **Bootstrap .venv and verify the runtime contract receipt.** Install jax[tpu] in ~/jax-tpu-lab/.venv, run verify_runtime_contract with a synchronized pmap sum of squares, and copy back the SHA-256 receipt.

   Lessons: [Bootstrap .venv and verify a TPU runtime receipt](02-bootstrap-venv-and-verify-tpu-runtime-receipt/docs/en.md).

### Try a changed condition

A script runs on a Cloud TPU VM without raising an error, but throughput is identical to a laptop CPU. What should your receipt check first?

<details><summary>Compare an approach</summary>

Inspect observed_backend and device_count in the JSON receipt; if libtpu was missing or JAX_PLATFORMS=tpu was not set, JAX silently fell back to CPU.

</details>

**Symptom:** You closed your SSH terminal after finishing a run, assuming the Cloud TPU VM stopped billing.

**Check next:** Run gcloud compute tpus tpu-vm delete and confirm gcloud compute tpus tpu-vm list returns 0 items.

### Decide what is ready

Retain the slice HBM table from tpu-01, the verified JSON runtime receipt from tpu-04, and the empty gcloud compute tpus tpu-vm list confirmation.

### Further work

Continue immediately to Phase 20 (TPU Workflows: Launching Jobs & Checkpointed Experiments) to stage code and run resumable training loops.

## Lesson sequence

### 19.01 Select a TPU slice and provision your Cloud TPU VM

[Read the lesson](01-get-a-tpu-running-and-verify-the-runtime/docs/en.md) · [Run the code](01-get-a-tpu-running-and-verify-the-runtime/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Choose a single-host TPU topology, create a Cloud TPU VM with gcloud (or connect to a Colab TPU), verify SSH access, and rehearse clean VM deletion.

**Evidence:** Keep the slice HBM capacity table, single-host vs Pod classification, and gcloud create/describe/delete command log.

**Checkpoint:** Which command sequence proves that your Cloud TPU VM is READY when you start and no longer billing when you finish?

### 19.02 Bootstrap .venv and verify a TPU runtime receipt

[Read the lesson](02-bootstrap-venv-and-verify-tpu-runtime-receipt/docs/en.md) · [Run the code](02-bootstrap-venv-and-verify-tpu-runtime-receipt/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Create ~/jax-tpu-lab/.venv on the TPU VM, install jax[tpu], run a synchronized pmap runtime contract check, and download the signed JSON receipt before deleting the VM.

**Evidence:** Keep the JSON runtime receipt (observed_backend, device_count, pmap_sum_of_squares, receipt_sha256) and the 1-chip vs 4-chip vs 8-chip pmap reference values.

**Checkpoint:** Why must verify_runtime_contract check both jax.default_backend() and a synchronized pmap reduction rather than only checking that import jax succeeds?

## Phase project

A provisioned Cloud TPU VM and a cryptographically hashed JAX runtime receipt.

**Demonstrate:** Select a single-host TPU topology, provision a Cloud TPU VM (or Colab TPU), bootstrap jax[tpu] in an isolated .venv, verify pmap execution, and confirm clean VM deletion.

Project status: implemented staged practice · [Open source](../../projects/tpu-runtime-audit/README.md). Use stages 1, 2, 3, 4 for this phase. Verify the runtime receipt locally on CPU first, then run the exact same check on your Cloud TPU VM and copy back tpu-receipt.txt before deleting the VM. Copy `projects/workload-operations/starter/model.py` to `projects/workload-operations/my_model.py` and write your code in `projects/workload-operations/my_model.py`. Run `python3 projects/workload-operations/tests/check.py --implementation projects/workload-operations/my_model.py --stage 1` from the top-level folder to verify stage 1.

Additional project: [Operate and recover a local JAX workload](../../projects/workload-operations/README.md).

[Primary documentation](https://cloud.google.com/tpu/docs/run-calculation-jax).
