<!-- jaxpathways-content-audit:2026-10-27:scaffold-load -->

Audited October 27, 2026, edition 0.17.1, PR #71 head `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`. Priority P2. Finding type: curriculum-extension.

## Teaching problem

Operations-01 through -05 each embed roughly 192-255 lines, including about 180 lines of shared supervisor/worker scaffolding, while often offering only three build steps. The new concept is difficult to locate in repeated code. Recovery-03/-04 similarly place pack/version helpers before the state/iterator replay story.

This is a teaching-structure issue, not a request to remove runnable programs or weaken process-isolation tests. Learners should be able to explain the mechanism without reverse-engineering the entire support harness first.

## Acceptance criteria

- [ ] Identify reviewed support code versus learner-authored transitions and show a short concept-focused excerpt first.
- [ ] Add a worked execution/state trace and explain each new boundary before presenting the full program.
- [ ] Reuse a versioned common support module in downloadable exercises, or clearly show deltas while keeping each download runnable.
- [ ] Require a small change that tests the central concept and an interpretation task using actual logs/state transitions.
- [ ] Verify script/notebook/book parity and fresh-download execution after reorganizing the teaching material.

## Audited source

- [operations-01](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/16-operations/01-accelerator-jobs-and-runtime-lifecycle/lesson.json)
- [operations-02](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/16-operations/02-health-logs-and-workload-observability/lesson.json)
- [operations-03](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/16-operations/03-failure-recovery-and-operational-runbooks/lesson.json)
- [operations-04](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/16-operations/04-changes-rollback-and-artifact-provenance/lesson.json)
- [operations-05](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/16-operations/05-capacity-utilization-and-operating-cost/lesson.json)
- [recovery-03](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/06-recovery/03-save-model-optimizer-and-random-state/lesson.json)
- [recovery-04](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/06-recovery/04-recover-data-position-and-resume/lesson.json)

This issue is a source-level content review. Existing CPU receipts and honest toy/replay/target limitations remain valid within their stated scope. This audit did not rerun all training programs, commission human reviews or launch paid hardware. Keep content maturity, physical-device qualification and model-quality evidence separate.
