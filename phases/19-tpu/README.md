# Phase 19: TPU Setup: Provisioning & Runtime Verification

Start here.

Step 1 of the 3-part Cloud TPU mini-course. Choose the right TPU slice, provision a Cloud TPU VM (or Colab TPU), bootstrap ~/jax-tpu-lab/.venv with jax[tpu], verify a synchronized pmap runtime receipt, and practice clean teardown.

Complete this short 2-lesson setup course right after 00: Setup & first steps, then continue directly to 20: TPU Workflows and 21: TPU Systems.

**Prerequisites:** 00: Setup & first steps.

**Hardware:** CPU topology/contract verifier; Cloud TPU VM (v5litepod-4 / v6e-4) or Colab TPU for target runs.

## Study guide: Can you provision a Cloud TPU VM, verify that JAX actually executes on its chips, and prove clean teardown?

Choosing an accelerator begins with a workload and a per-device capacity calculation. Aggregate HBM can look sufficient while a replicated parameter or optimizer array still exceeds the memory available to one device. Work through the placement and headroom assumptions before creating a resource.

Then compare the requested runtime with the observed backend and topology. A numerical check answers whether the tested arithmetic agrees; the receipt answers which environment ran it. Keep the named creation route and its matching cleanup steps together so the experiment has a clear beginning and end.

### Check your starting point

You created a Cloud TPU VM and gcloud says READY. What still has to happen before you trust a JAX run on it?

<details><summary>Compare your reasoning</summary>

Bootstrap ~/jax-tpu-lab/.venv with jax[tpu], verify inside Python that jax.default_backend() == 'tpu' and jax.device_count() matches your slice, and run a synchronized pmap reduction receipt.

</details>

Review: [Select a TPU slice and provision your Cloud TPU VM](01-get-a-tpu-running-and-verify-the-runtime/docs/en.md).

### Build in stages

1. **Select a single-host TPU topology and provision the VM.** Identify the requested API and machine type independently from observed JAX devices/processes. Calculate the 48 GiB replicated and ideal-sharded state ledger on four 16 GiB chips, include the 80% budget and assumed activation cost, and explain why aggregate capacity is insufficient. Keep the named resource create/describe/cleanup log.

   Lessons: [Select a TPU slice and provision your Cloud TPU VM](01-get-a-tpu-running-and-verify-the-runtime/docs/en.md).

2. **Bootstrap .venv and verify the runtime contract receipt.** Install the pinned runtime, run build_runtime_receipt(), compare arithmetic_ok with the local closed-form oracle, and inspect expected backend/count flags and requested_target_ready. Calculate separate 4-/8-device expectations. The target-only TPU_REQUIRE_TARGET path must reject CPU, wrong count or missing saved-source identity. Retrieve the receipt and identify its digest as an integrity field, not a signature.

   Lessons: [Bootstrap .venv and verify a TPU runtime receipt](02-bootstrap-venv-and-verify-tpu-runtime-receipt/docs/en.md).

### Try a changed condition

Your runtime receipt reports four local devices. Is that always four physical chips, and does it establish a production-model run?

<details><summary>Compare an approach</summary>

No. The mapping depends on generation and topology; TPU7x exposes chiplets as devices. Record machine type and device kinds with a synchronized numerical check. A small runtime receipt qualifies only its tested work, and the named control-plane resources still need explicit cleanup.

</details>

**Symptom:** You closed your SSH terminal after finishing a run, assuming the Cloud TPU VM stopped billing.

**Check next:** Use the cleanup command for the route that created the named resource. A queued resource must be deleted through its owning route. Confirm those exact resource names are absent; leave unrelated project resources alone.

### Decide what is ready

Keep the per-device ledger, requested-versus-observed runtime/source/environment identity, synchronized arithmetic and fail-closed target checks. Record absence of the exact named owned VM and queued resource after route-specific cleanup; a project-wide empty list is not required.

### Further work

Continue immediately to Phase 20 (TPU Workflows: Launching Jobs & Checkpointed Experiments) to stage code and run resumable training loops.

## Lesson sequence

### 19.01 Select a TPU slice and provision your Cloud TPU VM

[Read the lesson](01-get-a-tpu-running-and-verify-the-runtime/docs/en.md) · [Run the code](01-get-a-tpu-running-and-verify-the-runtime/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Choose and identify a named single-host TPU topology, distinguish physical chips from JAX devices, compare per-device memory-fit assumptions and verify a requested-target receipt before a route-owned teardown.

**Evidence:** Keep the 48 GiB replicated versus ideal-sharded ledger with headroom/activation assumptions, requested API/machine/chip/process identity, observed process/device/backend receipt and source/environment identity where available. Retain absence checks only for the named owned VM or queued resource.

**Checkpoint:** Which command sequence proves that your Cloud TPU VM is READY when you start and no longer billing when you finish?

### 19.02 Bootstrap .venv and verify a TPU runtime receipt

[Read the lesson](02-bootstrap-venv-and-verify-tpu-runtime-receipt/docs/en.md) · [Run the code](02-bootstrap-venv-and-verify-tpu-runtime-receipt/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Use the pinned TPU runtime, compare synchronized arithmetic with its oracle and distinguish requested backend/device readiness from a CPU rehearsal; retrieve the receipt and clean up only named owned resources.

**Evidence:** Keep arithmetic_ok, expected_backend_ok, expected_topology_ok, requested_target_ready, expected/observed backend and counts, source identity or its explicit absence, observed/expected reduction and receipt digest. Separate calculated 4-/8-device expectations from observed execution and require the gated target preflight to fail closed.

**Checkpoint:** Why does build_runtime_receipt() synchronize the pmap reduction and compare observed_total with closed_form_sum_of_squares(local_device_count, 8)?

## Phase project

Inspect CPU device inventory, BF16 storage, and collective participant arithmetic.

**Demonstrate:** Select a single-host TPU topology, provision a Cloud TPU VM (or Colab TPU), bootstrap jax[tpu] in an isolated .venv, verify pmap execution, and confirm clean VM deletion.

Project status: implemented staged practice · [Open source](../../projects/tpu-runtime-audit/README.md). Use stages 1, 2, 3, 4 for this phase. Work in projects/tpu-runtime-audit/starter/model.py. Complete stages 1 and 2 for independent calculations and changed-condition diagnosis, then retain the stage 3 evidence with --evidence-dir runs/tpu-runtime-audit. Stage 4 integrates the checks. Learner commands explicitly select the starter; --implementation solution is instructor reference execution. This bounded practice does not certify the entire phase or target hardware.

Additional project: [Operate and recover a local JAX workload](../../projects/workload-operations/README.md).

[Primary documentation](https://cloud.google.com/tpu/docs/run-calculation-jax).
