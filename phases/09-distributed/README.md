# Phase 09: Distributed training

Specializations.

Partition arrays and training updates, compare all-reduce with reduce-scatter, and restore a complete sharded state across an epoch boundary. Verify CPU placement and arithmetic before separate cluster validation.

Compare against a single-device reference; measure real accelerator performance separately.

**Prerequisites:** 06: Data & checkpoint recovery; 08: Performance diagnosis.

**Hardware:** Logical CPU devices for sharding practice; multi-host work needs a real cluster.

## Study guide: Is the distributed update still optimizing the same objective?

Distribution changes where arrays and computations live; it should preserve the intended global objective. Begin by drawing which rows and parameter columns each device owns. Then derive how distinct contributions and valid counts combine into one loss and update.

Use logical CPU devices to make ownership and unequal partitions inspectable. The unsharded calculation gives you a numerical reference; shard indices and local byte counts give you a placement reference. Communication and recovery add separate questions about coordinated work, which require their own receipts.

### Check your starting point

Two devices see one and three valid examples. Should their mean gradients receive equal weight?

<details><summary>Compare your reasoning</summary>

No. Weight by valid example counts. Sum local gradient numerators and valid counts, then divide once. Padding and empty partitions must not silently change the denominator.

</details>

Review: [A sharded training step](02-a-sharded-training-step/docs/en.md).

### Build in stages

1. **Verify placement and the global update.** Inspect actual shard indices and compare each relevant output with an independent unpartitioned calculation, including padded and unequal partitions.

   Lessons: [Arrays, meshes, and sharding](01-arrays-meshes-and-sharding/docs/en.md) · [A sharded training step](02-a-sharded-training-step/docs/en.md).

2. **Connect communication and recovery.** Inspect collectives without inferring network speed from payload size. Restore data position and optimizer state before comparing subsequent global steps. Continue to the 2x2 data/model mesh: predict parameter and momentum slices and per-device bytes, then compare one forward/backward/update with an unsharded NumPy oracle under uneven and empty valid counts.

   Lessons: [Communication-efficient algorithms](03-communication-efficient-algorithms/docs/en.md) · [Resilient distributed training](04-resilient-distributed-training/docs/en.md).

### Try a changed condition

Three shards have valid counts 2, 6 and 0 and local gradient sums 6, 6 and 0. Predict the global mean and the empty-shard contribution.

<details><summary>Compare an approach</summary>

Reduce sums and counts separately: the global mean is \((6+6+0)/(2+6+0)=1.5\). The empty shard contributes a zero numerator and zero count; it must not compute \(0/0\). Reject a globally empty batch. Compare with an independent unsharded calculation.

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

**Intended outcome:** Compute global count-weighted gradients and verify a 2x2 data/model mesh forward, backward and momentum update against an unsharded oracle.

**Evidence:** Keep 1D and 2D shard ownership, per-device parameter/momentum bytes, asymmetric-input forward/backward/update comparisons, unequal and empty valid-count cases and the wrong local-mean control.

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
