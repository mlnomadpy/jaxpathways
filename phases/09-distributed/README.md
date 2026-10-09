# Phase 09: Distributed training

Specializations.

Partition arrays and training updates, compare all-reduce with reduce-scatter, and restore a complete sharded state across an epoch boundary. Verify CPU placement and arithmetic before separate cluster validation.

Compare against a single-device reference; measure real accelerator performance separately.

**Prerequisites:** 06: Data & checkpoint recovery; 08: Performance diagnosis; 19: TPU Setup: Provisioning & Runtime Verification; 20: TPU Workflows: Launching Jobs & Checkpointed Experiments; 21: TPU Systems: Generations, Memory, Precision & XProf.

**Hardware:** Logical CPU devices for sharding practice; multi-host work needs a real cluster.

## Study guide: Is the distributed update still optimizing the same objective?

Start with the global mathematical operation, then specify ownership and communication. Logical CPU devices help expose placement and reduction mistakes before you spend accelerator time.

### Check your starting point

Two devices see one and three valid examples. Should their mean gradients receive equal weight?

<details><summary>Compare your reasoning</summary>

No. Weight by valid example counts. Sum local gradient numerators and valid counts, then divide once. Padding and empty partitions must not silently change the denominator.

</details>

Review: [A sharded training step](02-a-sharded-training-step/docs/en.md).

### Build in stages

1. **Verify placement and the global update.** Inspect actual shard indices and compare each relevant output with an independent unpartitioned calculation, including padded and unequal partitions.

   Lessons: [Arrays, meshes, and sharding](01-arrays-meshes-and-sharding/docs/en.md) · [A sharded training step](02-a-sharded-training-step/docs/en.md).

2. **Connect communication and recovery.** Inspect collectives without inferring network speed from payload size. Restore data position and optimizer state before comparing subsequent global steps.

   Lessons: [Communication-efficient algorithms](03-communication-efficient-algorithms/docs/en.md) · [Resilient distributed training](04-resilient-distributed-training/docs/en.md).

### Try a changed condition

A final batch leaves one partition empty. Should it contribute a zero mean, be omitted, or fail?

<details><summary>Compare an approach</summary>

Do not compute an undefined local mean. Accumulate a zero numerator and zero valid count for the empty partition, reduce globally and reject a globally empty batch. Compare the result with the valid unpartitioned examples.

</details>

**Symptom:** The gradient grows with device count.

**Check next:** Check whether local means were summed without count normalization. Verify one complete update before interpreting a training curve.

### Decide what is ready

Use sharded-training. Retain actual placements and collectives, independent uneven/empty-partition checks and full-state replay. Label logical CPU devices explicitly.

### Further work

Real network collectives, multi-controller ingestion and host-loss recovery are unqualified. The optional launch instructions are preparation, not execution receipts.

## Lesson sequence

### 09.01 Arrays, meshes, and sharding

[Read the lesson](01-arrays-meshes-and-sharding/docs/en.md) · [Run the code](01-arrays-meshes-and-sharding/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Partition an array over a device mesh and inspect its local shards.

**Evidence:** Padding repairs the shape but adds artificial rows. The mask keeps them out of both numerator and denominator. Checking only divisibility would miss this statistical error.

**Checkpoint:** With row sharding, why does a global column sum need results from other devices?

### 09.02 A sharded training step

[Read the lesson](02-a-sharded-training-step/docs/en.md) · [Run the code](02-a-sharded-training-step/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Run a training step with explicit data and parameter placement.

**Evidence:** Each local gradient already divides by its local count. Summing four equal local means omits division by four. For unequal counts, an unweighted mean also misrepresents examples; sum count-weighted local means and divide by total count.

**Checkpoint:** Four equal-size shards compute local mean gradients. Which aggregation gives the global mean gradient?

### 09.03 Communication-efficient algorithms

[Read the lesson](03-communication-efficient-algorithms/docs/en.md) · [Run the code](03-communication-efficient-algorithms/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Compare all-reduce and reduce-scatter against the same independent global gradient. Explain global result shape versus per-device storage.

**Evidence:** Keep global NumPy gradient checks, local shard shapes, lowered collectives, changed-value replicas and complete scatter/gather payload accounting. Separate the idealized ring model from observed CPU timings.

**Checkpoint:** A program replaces all-reduce with reduce-scatter and immediately gathers the full vector again. What does the idealized ring model predict?

### 09.04 Resilient distributed training

[Read the lesson](04-resilient-distributed-training/docs/en.md) · [Run the code](04-resilient-distributed-training/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Restore a complete training state with explicit target sharding. Compare next sample IDs, optimizer state and losses across an epoch boundary.

**Evidence:** Keep the checkpoint path, saved/restored full state, subsequent sample IDs and losses through a new epoch. Diagnose missing momentum and a changed shuffle key; state the untested multi-host boundary.

**Checkpoint:** The first resumed batch and update match, but sample IDs differ at the next epoch. Which saved value should you inspect first?

## Phase project

A resilient sharded training experiment.

**Demonstrate:** Explain partitioning, communication costs, and recovery behavior with measured evidence.

Project status: implemented staged practice · [Open source](../../projects/sharded-training/README.md). Use stages 1, 2, 3, 4 for this phase. This connected project also uses performance and distributed lessons. Work through its prerequisites before the full integration check; return here with the completed evidence. Copy `projects/sharded-training/starter/model.py` to `projects/sharded-training/my_model.py` and write your code in `projects/sharded-training/my_model.py`. Run `python3 projects/sharded-training/tests/check.py --implementation projects/sharded-training/my_model.py --stage 1` from the top-level folder to verify stages 1, 2, 3, 4.



[Primary documentation](https://docs.jax.dev/en/latest/).
