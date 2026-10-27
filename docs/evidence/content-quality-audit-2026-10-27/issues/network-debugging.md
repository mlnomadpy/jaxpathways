<!-- jaxpathways-content-audit:2026-10-27:network-debugging -->

Audited October 27, 2026, edition 0.17.1, PR #71 head `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`. Priority P2. Finding type: curriculum-extension.

## Teaching problem

The chapter already trains a real tiny MLP and teaches binary evaluation, but the dedicated debugging lesson returns to scalar quadratic regression, repeating the earlier transformation/optimization fixture. Its math is correct; it cannot show how to locate a failure inside a multilayer model.

The evaluation lesson is stateless, so it also cannot demonstrate train/eval differences caused by Dropout or BatchNorm state. This is a missing transfer opportunity rather than evidence that its current immutable-evaluation check is wrong.

## Acceptance criteria

- [ ] Reuse the small MLP and introduce controlled failures separately in data, activation/forward computation, gradients and update scale.
- [ ] For each failure ask for a prediction, record one diagnostic that identifies the layer/stage, repair one cause and compare on held-out inputs.
- [ ] Explain activation/gradient distributions and loss/learning-rate plots using the actual generated values; keep finite checks and independent loss references.
- [ ] Add a bounded stateful train/eval extension that distinguishes parameters, mutable running statistics and random keys, including a negative control.

## Audited source

- [networks-05](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/05-networks/05-debug-unstable-learning/lesson.json)
- [networks-03](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/05-networks/03-a-compiled-train-and-evaluation-step/lesson.json)

This issue is a source-level content review. Existing CPU receipts and honest toy/replay/target limitations remain valid within their stated scope. This audit did not rerun all training programs, commission human reviews or launch paid hardware. Keep content maturity, physical-device qualification and model-quality evidence separate.
