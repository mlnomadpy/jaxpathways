<!-- jaxpathways-content-audit:2026-10-27:ops-evaluation -->

Audited October 27, 2026, edition 0.17.1, PR #71 head `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`. Priority P2. Finding type: curriculum-extension.

## Teaching problem

The MLOps lesson has useful group-leakage and weighted-slice counterexamples, but receives supplied squared errors rather than computing predictions from a reloaded artifact. The LLMOps lesson records real MLflow spans and missing/critical-case bookkeeping, but nine answerable fixtures are duplicates and answers are canned; no prompt/model/retrieval stage produces them. Existing replay labeling is honest and should remain.

Learners need one connected offline workflow to show what is evaluated, where a response failed and why a release gate changes. Paid APIs are unnecessary for this task.

## Acceptance criteria

- [ ] Evaluate a reloaded local artifact and derive per-example and slice metrics from predictions/targets, preserving group-safe splits and weighting.
- [ ] Provide heterogeneous stable case IDs with supported, unsupported, missing, duplicate and out-of-order responses; demonstrate a case that changes the release decision.
- [ ] Trace actual local prompt/retrieval/model boundaries using a bounded local adapter, or clearly identify an immutable captured replay and its limits.
- [ ] Have the learner localize one retrieval failure, one answer-generation failure and one infrastructure failure from trace/metric evidence.
- [ ] Require a changed candidate comparison with the same case IDs, denominators, policy and artifact identity. Do not equate drift alone with model quality.

## Audited source

- [operations-06](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/16-operations/06-mlops-data-contracts-and-release-pipelines/lesson.json)
- [operations-07](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/16-operations/07-llmops-prompts-evaluation-and-traces/lesson.json)

This issue is a source-level content review. Existing CPU receipts and honest toy/replay/target limitations remain valid within their stated scope. This audit did not rerun all training programs, commission human reviews or launch paid hardware. Keep content maturity, physical-device qualification and model-quality evidence separate.
