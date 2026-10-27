<!-- jaxpathways-content-audit:2026-10-27:tpu-readiness -->

Audited October 27, 2026, edition 0.17.1, PR #71 head `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`. Priority P1. Finding type: confirmed-correction.

## Teaching problem

Several learner-facing statements undermine the newer guarded target matrix:

- The Setup chapter milestone still tells learners to "run subsequent course phases on TPU in parallel". `tpu-05` correctly says some scripts are CPU-only and unreviewed scripts cannot be inferred compatible.
- `tpu-01` defaults to the observed backend; its main call passes `expected_backend=jax.default_backend()`. The exact function returns `contract_ok=True` on CPU. This is a valid CPU arithmetic rehearsal, not evidence that the requested TPU is ready.
- `tpu-04` returns `verified=True` for arithmetic success on CPU. The SSH command forces TPU platform selection, but the copied receipt has no expected device-count gate, and the word "verified" does not say which contract passed.
- TPU setup installs unpinned dependencies while the staging chapter requires pinned, source-reviewed environments.
- Cleanup objectives say the whole project list must contain zero TPU nodes. `tpu-04` teaches direct VM deletion even when the preceding setup chose an owning queued resource. The earlier route-specific cleanup guidance is stronger and should be used consistently.
- `tpu-02` promises that temp/fsync/replace means preemption "never corrupts" the latest checkpoint, beyond its controlled single-writer local test and despite more careful caveats later.

CPU probes extracted only the exact function bodies from the audited source: tpu-01 `contract_ok=True`, tpu-04 `verified=True`, observed backend `cpu`. No TPU was provisioned or qualified in this audit.

## Acceptance criteria

- [ ] Replace the blanket Setup milestone with the existing target classification and explicit CPU continuation route.
- [ ] Distinguish arithmetic_ok, expected_backend_ok, expected_topology_ok and overall requested-target readiness; keep CPU rehearsal usable and clearly labeled.
- [ ] Exercise backend mismatch and insufficient device-count failures before any training launch; retain source/environment identity in the receipt.
- [ ] Use the reviewed pinned environment and bind staging decisions to the actual source/target identity rather than a caller-supplied claim alone.
- [ ] Give route-specific named-resource cleanup, including the owning queued resource when applicable; verify absence of those resources without requiring unrelated resources to be removed.
- [ ] Scope checkpoint guarantees to the tested writer/filesystem/publication model and teach an interrupted-save recovery decision.

## Audited source

- [welcome-cpu](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/00-welcome/03-virtual-cpu-devices/lesson.json)
- [welcome-03](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/00-welcome/03-move-your-experiment-to-a-tpu/lesson.json)
- [tpu-01](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/19-tpu/01-get-a-tpu-running-and-verify-the-runtime/lesson.json)
- [tpu-04](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/19-tpu/02-bootstrap-venv-and-verify-tpu-runtime-receipt/lesson.json)
- [tpu-02](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/20-tpu-workflows/01-stage-and-launch-checkpointed-tpu-jobs/lesson.json)
- [tpu-05](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/20-tpu-workflows/02-run-course-scripts-and-multihost-tpu-slices/lesson.json)
- [curriculum/course.json](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/curriculum/course.json)
- [curriculum/hardware-validation.json](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/curriculum/hardware-validation.json)

This issue is a source-level content review. Existing CPU receipts and honest toy/replay/target limitations remain valid within their stated scope. This audit did not rerun all training programs, commission human reviews or launch paid hardware. Keep content maturity, physical-device qualification and model-quality evidence separate.
