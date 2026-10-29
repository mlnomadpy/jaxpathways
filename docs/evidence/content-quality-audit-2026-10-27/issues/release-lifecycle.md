<!-- jaxpathways-content-audit:2026-10-27:release-lifecycle -->

Audited October 27, 2026, edition 0.17.1, PR #71 head `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`. Priority P2. Finding type: curriculum-extension.

## Teaching problem

Release A and B in operations-04 differ in metadata rather than model behavior. The test proves pointer/content selection, but does not show a consumer restoring the previous model's predictions. Operations-08 includes retirement in its title while its implementation/exercises stop at activation, approval expiry and target checks.

Current content-bound approval and side-effect-free rejection are good foundations. The missing learner outcome is a complete, clearly scoped lifecycle with observed consumer behavior and retirement semantics.

## Acceptance criteria

- [ ] Use two identified, valid artifacts with intentionally different predictions and a consumer that loads the selected artifact.
- [ ] Record predictions before activation, after activation and after rollback; demonstrate that a rejected activation leaves behavior and selection unchanged.
- [ ] Define and exercise retirement/tombstone behavior for active and fallback versions, including cache/reload and rollback eligibility.
- [ ] Add stale approval, retired target and expired approval negative cases plus an audit trail tied to exact bundle identity.
- [ ] Keep the single-writer/local/no-auth scope explicit and avoid presenting pointer files as a production access-control system.

## Audited source

- [operations-04](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/16-operations/04-changes-rollback-and-artifact-provenance/lesson.json)
- [operations-08](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/16-operations/08-modelops-ownership-approval-and-retirement/lesson.json)

This issue is a source-level content review. Existing CPU receipts and honest toy/replay/target limitations remain valid within their stated scope. This audit did not rerun all training programs, commission human reviews or launch paid hardware. Keep content maturity, physical-device qualification and model-quality evidence separate.
