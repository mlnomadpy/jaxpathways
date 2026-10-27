<!-- jaxpathways-content-audit:2026-10-27:attention-permutation -->

Audited October 27, 2026, edition 0.17.1, PR #71 head `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`. Priority P1. Finding type: confirmed-correction.

## Teaching problem

The main example uses `q = zeros((2, 2))`. Every attention row is uniform, so both queries return the mean of V, `[2, 2]`. The primary exercise asks for paired key/value permutation invariance AND a changed output after permuting only V. With these zero queries, every permutation of V leaves its mean unchanged, so the second required result is impossible.

The solution instead references `q_new` and `actual` from a separate nonuniform experiment. Those inputs are not stated as the exercise starting point. Independent NumPy checks reproduce the invariant zero-query case and a changed result with nonuniform scores.

## Acceptance criteria

- [ ] Keep the zero-query worked example and explain why value-only permutation is also invariant in this special case.
- [ ] Supply an explicit nonuniform Q/K/V fixture in the primary exercise, independent of hidden variables from a previous experiment.
- [ ] Derive paired-permutation invariance and exhibit one value-only permutation that changes this fixture; avoid claiming every mismatch must change every possible output.
- [ ] Run the exercise solution from its declared starter state and compare with an independent NumPy reference.
- [ ] Align course registry exercise/evidence, learner starter, explanation and solution.

## Audited source

- [transformers-01](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/07-transformers/01-attention-from-small-pieces/lesson.json)

This issue is a source-level content review. Existing CPU receipts and honest toy/replay/target limitations remain valid within their stated scope. This audit did not rerun all training programs, commission human reviews or launch paid hardware. Keep content maturity, physical-device qualification and model-quality evidence separate.
