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

1. **Select a single-host TPU topology and provision the VM.** Use the runtime verifier for the active backend. Calculate and verify the chip count and total HBM for v5litepod-4 and v6e-4, confirm both are single-host slices, and list all supplied single-host slices with at least 64 GiB. Retain the target VM create/describe/delete log.

   Lessons: [Select a TPU slice and provision your Cloud TPU VM](01-get-a-tpu-running-and-verify-the-runtime/docs/en.md).

2. **Bootstrap .venv and verify the runtime contract receipt.** Install jax[tpu] in ~/jax-tpu-lab/.venv, run build_runtime_receipt(), and verify its observed reduction against the closed-form result for its actual local_device_count. Then compute the separate theoretical totals for 4 and 8 devices. Copy back the JSON receipt; distinguish its SHA-256 field from a digital signature and calculated expectations from observed device execution.

   Lessons: [Bootstrap .venv and verify a TPU runtime receipt](02-bootstrap-venv-and-verify-tpu-runtime-receipt/docs/en.md).

### Try a changed condition

Your runtime receipt reports four local devices. Is that always four physical chips, and does it establish a production-model run?

<details><summary>Compare an approach</summary>

No. The mapping depends on generation and topology; TPU7x exposes chiplets as devices. Record machine type and device kinds with a synchronized numerical check. A small runtime receipt qualifies only its tested work, and the named control-plane resources still need explicit cleanup.

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

**Intended outcome:** Create ~/jax-tpu-lab/.venv on the TPU VM, install jax[tpu], record the observed backend and local device count alongside a synchronized pmap reduction check, retrieve the JSON receipt, and then delete the VM.

**Evidence:** Keep the JSON runtime receipt fields (backend, local_device_count, per_device_sums, observed_total, expected_total, verified, receipt_sha256). Label the 1-, 4-, and 8-device closed-form values as calculated expectations; only the receipt's local-device result is observed execution.

**Checkpoint:** Why does build_runtime_receipt() synchronize the pmap reduction and compare observed_total with closed_form_sum_of_squares(local_device_count, 8)?

## Phase project

Inspect CPU device inventory, BF16 storage, and collective participant arithmetic.

**Demonstrate:** Select a single-host TPU topology, provision a Cloud TPU VM (or Colab TPU), bootstrap jax[tpu] in an isolated .venv, verify pmap execution, and confirm clean VM deletion.

Project status: implemented staged practice · [Open source](../../projects/tpu-runtime-audit/README.md). Use stages 1, 2, 3, 4 for this phase. Work in projects/tpu-runtime-audit/starter/model.py. Complete stages 1 and 2 for independent calculations and changed-condition diagnosis, then retain the stage 3 evidence with --evidence-dir runs/tpu-runtime-audit. Stage 4 integrates the checks. Learner commands explicitly select the starter; --implementation solution is instructor reference execution. This bounded practice does not certify the entire phase or target hardware.

Additional project: [Operate and recover a local JAX workload](../../projects/workload-operations/README.md).

[Primary documentation](https://cloud.google.com/tpu/docs/run-calculation-jax).
