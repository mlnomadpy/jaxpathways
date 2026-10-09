# Phase 06: Data & checkpoint recovery

Training systems.

Follow examples from the dataset into a training update, then save enough information to reproduce the next update after interruption. Track experiment configuration, data identity and metric histories with MLflow alongside complete checkpoints.

Compare sample IDs, optimizer memory, random state and losses after restore.

**Prerequisites:** 05: Neural network training.

**Hardware:** CPU first; TPU for system measurements.

## Study guide: Can an interrupted run make exactly the same next update?

Recovery is a contract between data order, model state and publication. Tracking explains which run produced an artifact; a checkpoint supplies the values needed to continue it. Neither replaces the other.

### Check your starting point

The weights restore exactly, but the next loss differs. Which state should you inspect before blaming numerical nondeterminism?

<details><summary>Compare your reasoning</summary>

Compare optimizer moments and count, random keys, data permutation/cursor, preprocessing and configuration. The same weights with a different next batch are a different transition.

</details>

Review: [Save model, optimizer, and random state](03-save-model-optimizer-and-random-state/docs/en.md).

### Build in stages

1. **Preserve record identity and position.** Inspect record IDs through shuffling and batching. State the final-batch policy and save iterator state only at the intended consumption boundary.

   Lessons: [Design an input pipeline](01-design-an-input-pipeline/docs/en.md) · [Load, batch, and checkpoint order with Grain](02-load-batch-and-prefetch-with-grain/docs/en.md).

2. **Prove a committed continuation.** Save complete state, restore in a fresh process and compare future IDs and updates across an epoch boundary. Reject an incomplete or incompatible checkpoint before publishing it.

   Lessons: [Save model, optimizer, and random state](03-save-model-optimizer-and-random-state/docs/en.md) · [Recover data position and resume](04-recover-data-position-and-resume/docs/en.md) · [Asynchronous checkpoints and failure boundaries](05-asynchronous-checkpoints-and-failure-boundaries/docs/en.md).

3. **Connect continuation to lineage.** Retrieve an MLflow run and its artifact, compare data/source identities and metric-step conventions, and distinguish a registry selection from active deployment.

   Lessons: [Track experiments and lineage with MLflow](06-track-experiments-and-lineage-with-mlflow/docs/en.md).

### Try a changed condition

The checkpoint is valid, but the dataset was reordered without changing its filename. Is the saved cursor sufficient?

<details><summary>Compare an approach</summary>

No. The cursor is meaningful relative to a dataset identity and ordering configuration. Verify content/version identity first; reject mismatches rather than silently resuming on different records.

</details>

**Symptom:** The advertised checkpoint step is newer than the restored state.

**Check next:** Inspect completion and publication order. An asynchronous save request is not proof that the artifact is durable or restorable.

### Decide what is ready

Use sharded-training recovery stages and the MLflow extension. Keep uninterrupted and restarted trajectories, a corrupt/incompatible rejection, and a retrieved artifact tied to its run.

### Further work

Network storage failures, concurrent checkpoint writers and true multi-host recovery still need separate failure-injection qualification.

## Lesson sequence

### 06.01 Design an input pipeline

[Read the lesson](01-design-an-input-pipeline/docs/en.md) · [Run the code](01-design-an-input-pipeline/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Build and audit a deterministic batched host pipeline.

**Evidence:** Keep the exact epoch/example ID order, uneven final-batch sizes, hand-derived masked versus unmasked label mean, changed seed order and duplicate/drop audit.

**Checkpoint:** What information is missing from a model-only checkpoint for mid-epoch recovery?

### 06.02 Load, batch, and checkpoint order with Grain

[Read the lesson](02-load-batch-and-prefetch-with-grain/docs/en.md) · [Run the code](02-load-batch-and-prefetch-with-grain/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Build a Grain source/sampler/batch pipeline and verify its record order.

**Evidence:** Keep Grain sampler configuration and exact example IDs for full and partial batches; save iterator bytes, recreate the loader, and compare the next batch before/after restore. Record the incompatible configuration failure.

**Checkpoint:** What makes a fresh loader resume at the saved next batch?

### 06.03 Save model, optimizer, and random state

[Read the lesson](03-save-model-optimizer-and-random-state/docs/en.md) · [Run the code](03-save-model-optimizer-and-random-state/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Checkpoint complete training state with Orbax and replay the next update.

**Evidence:** Keep Orbax save/wait/restore commands, snapshot step, Adam count/moments, exact PRNG words, numerical next-update replay and the parameter-only restart counterexample.

**Checkpoint:** Which checkpoint supports replaying the next Adam update with stochastic targets?

### 06.04 Recover data position and resume

[Read the lesson](04-recover-data-position-and-resume/docs/en.md) · [Run the code](04-recover-data-position-and-resume/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Persist model/optimizer/key and Grain iterator position at one boundary.

**Evidence:** Keep the committed model/optimizer/key/iterator snapshot, uninterrupted and resumed batch-ID ledger across the epoch boundary, exact keys/counts, declared floating tolerances and inconsistent model-only/reader-only restart failures.

**Checkpoint:** Which evidence supports a consistent restart boundary?

### 06.05 Asynchronous checkpoints and failure boundaries

[Read the lesson](05-asynchronous-checkpoints-and-failure-boundaries/docs/en.md) · [Run the code](05-asynchronous-checkpoints-and-failure-boundaries/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Distinguish requested, completed, verified and published checkpoint state. Continue a functional update while an asynchronous writer is alive.

**Evidence:** Keep the requested/completed/accepted step log, the restored snapshot comparison, interrupted-publication result and surfaced finalization error. Explain why directory presence alone does not establish acceptance.

**Checkpoint:** A new checkpoint directory exists but the completion wait raises an exception. Which recovery decision follows this lesson’s protocol?

### 06.06 Track experiments and lineage with MLflow

[Read the lesson](06-track-experiments-and-lineage-with-mlflow/docs/en.md) · [Run the code](06-track-experiments-and-lineage-with-mlflow/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Track, retrieve and compare two JAX experiments using explicit data identity, metric steps and downloaded model artifacts.

**Evidence:** Keep both MLflow run IDs, source/data hashes, complete loss histories, the selection metric and predictions independently recomputed from the downloaded artifact.

**Checkpoint:** A run is tagged champion. What does that establish?

## Phase project

An interrupted run restored with its full state.

**Demonstrate:** Compare resumed and uninterrupted runs under a stated determinism tolerance.

Project status: implemented staged practice · [Open source](../../projects/sharded-training/README.md). Use stages 1, 2, 3 for this phase. This connected project also uses performance and distributed lessons. Work through its prerequisites before the full integration check; return here with the completed evidence. Copy `projects/sharded-training/starter/model.py` to `projects/sharded-training/my_model.py` and write your code in `projects/sharded-training/my_model.py`. Run `python3 projects/sharded-training/tests/check.py --implementation projects/sharded-training/my_model.py --stage 1` from the top-level folder to verify stages 1, 2, 3.

Additional project: [Ship a tracked and containerized model release](../../projects/engineering-release/README.md).

[Primary documentation](https://orbax.readthedocs.io/en/latest/).
