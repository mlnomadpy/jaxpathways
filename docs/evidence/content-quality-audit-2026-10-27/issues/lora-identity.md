<!-- jaxpathways-content-audit:2026-10-27:lora-identity -->

Audited October 27, 2026, edition 0.17.1, PR #71 head `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`. Priority P2. Finding type: confirmed-correction.

## Teaching problem

The prose correctly requires base checkpoint hash, target modules, rank, alpha and dtype. The adapter-save exercise writes only A, B, alpha and rank, then reloads against the live base variable. There is no bound base identity or wrong-base rejection. A shape-compatible different base can silently produce a different model.

The current fitted-base, gradient asymmetry, nonunit scaling and merge checks should be retained. This issue closes the gap between the stated artifact contract and its implementation.

## Acceptance criteria

- [ ] Save a manifest binding adapter tensors to an identified base checkpoint, target modules, shapes, dtype, rank and scaling convention.
- [ ] Reload the base and adapter independently in a fresh process and compare unmerged versus merged predictions on changed inputs.
- [ ] Reject a shape-compatible wrong base and altered metadata before activation; explain why matching tensor shapes is insufficient.
- [ ] Include a nonunit alpha/r case and preserve evidence that base parameters remain frozen during adaptation.

## Audited source

- [posttraining-02](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/18-posttraining/02-lora/lesson.json)

This issue is a source-level content review. Existing CPU receipts and honest toy/replay/target limitations remain valid within their stated scope. This audit did not rerun all training programs, commission human reviews or launch paid hardware. Keep content maturity, physical-device qualification and model-quality evidence separate.
