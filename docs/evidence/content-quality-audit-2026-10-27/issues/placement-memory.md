<!-- jaxpathways-content-audit:2026-10-27:placement-memory -->

Audited October 27, 2026, edition 0.17.1, PR #71 head `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`. Priority P2. Finding type: mixed-correction-and-extension.

## Teaching problem

The distributed chapter executes meaningful 1D data sharding, weighted gradients and collectives, but its opening 2D data/model notation never becomes a parameter-sharded learner step. The TPU capacity exercises still ask whether aggregate HBM means a model fits, despite newer prose and starter invariants correctly saying it does not prove per-device fit.

A 48 GiB replicated state does not fit on any 16 GiB device in a 4-device slice, although aggregate HBM is 64 GiB. Ideal full sharding gives 12 GiB/device before activations, temporaries, uneven partitions and headroom. The existing exercise explanation silently assumes sharding without proving placement.

`tpu-01` also equates device_count('tpu') with chip count without a topology qualifier. Its own newer chapter transfer discusses generations/chiplets where those terms must be distinguished. [JAX's TPU hardware reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html) distinguishes chip and TensorCore-level hardware; use topology-specific API observations rather than a universal device-to-chip identity.

## Acceptance criteria

- [ ] Add a bounded 2x2 logical-CPU mesh exercise with explicit parameter, batch and optimizer-state placement, followed by one real forward/backward/update comparison to an unsharded oracle.
- [ ] Record per-device shapes/bytes and collective ownership; include unequal or empty valid-count cases and explain the reduction axes.
- [ ] Replace aggregate-fit conclusions with replicated-versus-sharded per-device ledgers and explicit peak/headroom assumptions.
- [ ] Teach topology-specific process/device/chip terminology and verify it against the selected generation/API receipt.
- [ ] Add training-versus-inference sizing decisions, keeping training state and inference KV cache separate.
- [ ] Keep logical CPU execution distinct from physical TPU qualification and do not require a new paid hardware run to complete the teaching task.

## Audited source

- [distributed-01](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/09-distributed/01-arrays-meshes-and-sharding/lesson.json)
- [distributed-02](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/09-distributed/02-a-sharded-training-step/lesson.json)
- [distributed-03](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/09-distributed/03-communication-efficient-algorithms/lesson.json)
- [distributed-04](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/09-distributed/04-resilient-distributed-training/lesson.json)
- [tpu-01](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/19-tpu/01-get-a-tpu-running-and-verify-the-runtime/lesson.json)
- [tpu-03](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/21-tpu-systems/01-tpu-generations-and-hbm-memory-budgeting/lesson.json)
- [curriculum/course.json](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/curriculum/course.json)

This issue is a source-level content review. Existing CPU receipts and honest toy/replay/target limitations remain valid within their stated scope. This audit did not rerun all training programs, commission human reviews or launch paid hardware. Keep content maturity, physical-device qualification and model-quality evidence separate.
