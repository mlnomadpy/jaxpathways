<!-- jaxpathways-content-audit:2026-10-27:exercise-contracts -->

Audited October 27, 2026, edition 0.17.1, PR #71 head `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`. Priority P2. Finding type: confirmed-correction.

## Teaching problem

Four exercise contracts are not fulfilled by their supplied solutions:

- `recovery-06`: asks for a third wrong-model MLflow artifact and proof that a low training loss is insufficient. The solution only computes zero-model validation MSE and asserts it is worse. It creates no third MLflow run/artifact and no misleading low-training-loss candidate.
- `deployment-07`: asks to reject a changed artifact. The solution compares original/altered digests, but never sends the altered bytes through the actual loading/request rejection boundary. The separate main bad-hash example does not complete this exercise.
- `transforms-04`: asks for an independent manual matrix-gradient comparison and synchronization explanation. The solution compares eager and JIT outputs only. A manual formula elsewhere in the experiment should be incorporated or explicitly included in the exercise contract.
- `tpu-06`: asks learners to report the observed backend; the solution does not report it, and its wording says two mean errors while the function returns three policy rows.

## Acceptance criteria

- [ ] Provide complete, independently runnable solutions from the declared learner inputs for all four exercises.
- [ ] Persist and reload the third MLflow artifact; show why a misleading training-only choice fails a held-out selection rule.
- [ ] Submit altered artifact bytes to the fresh-process loader and assert the expected rejection and absence of unintended side effects.
- [ ] Derive the matrix gradient independently and explain where device completion is required for the claimed timing.
- [ ] Check output evidence and negative controls, not only that a phrase or assert appears in the solution source.

## Audited source

- [recovery-06](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/06-recovery/06-track-experiments-and-lineage-with-mlflow/lesson.json)
- [deployment-07](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/15-deployment/07-containerize-a-model-service/lesson.json)
- [transforms-04](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/02-transforms/04-compile-a-function-with-jit/lesson.json)
- [tpu-06](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/21-tpu-systems/02-bfloat16-int8-precision-and-xprof-profiling/lesson.json)

This issue is a source-level content review. Existing CPU receipts and honest toy/replay/target limitations remain valid within their stated scope. This audit did not rerun all training programs, commission human reviews or launch paid hardware. Keep content maturity, physical-device qualification and model-quality evidence separate.
