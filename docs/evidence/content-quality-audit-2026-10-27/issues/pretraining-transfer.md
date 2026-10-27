<!-- jaxpathways-content-audit:2026-10-27:pretraining-transfer -->

Audited October 27, 2026, edition 0.17.1, PR #71 head `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`. Priority P2. Finding type: curriculum-extension.

## Teaching problem

The three lessons explain their limited fixtures honestly and include useful independent checks. MLM uses three repeated-token sequences; MIM uses a fixed visible-patch linear decoder; contrastive retrieval is evaluated on four training pairs. These teach objectives, but do not establish learned representation transfer or guide the learner through a saved mask-aware encoder.

The PR stack already contains a saved embedding pilot and image project material. Reuse those foundations and their declared evidence limits instead of duplicating another toy objective or claiming a production model has passed.

## Acceptance criteria

- [ ] Add an explicit small-model route with stage inputs, data recipe, split/source-group policy, tokenizer or patch transform, mask schedule, objective and artifact outputs.
- [ ] For MLM require a learner-controlled corruption/objective change and held-out masked loss against a declared baseline, then a frozen-encoder downstream probe.
- [ ] For MIM vary the mask and distinguish visible encoding from target reconstruction; explain what the frozen representation probe can and cannot show.
- [ ] For contrastive learning include held-out groups, multiple positives and false-negative handling; compare retrieval metrics against an independent ranking reference.
- [ ] Keep data/model/processor/checkpoint identity across stages and link each transfer task to the existing pilot files and prerequisites.
- [ ] Do not require a new 300M run or relabel failed/unperformed project-quality gates. Full production lifecycles remain in #29.

## Audited source

- [pretraining-01](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/17-pretraining/01-mlm/lesson.json)
- [pretraining-02](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/17-pretraining/02-mim/lesson.json)
- [pretraining-03](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/17-pretraining/03-contrastive/lesson.json)
- [projects/embedding-model-300m/README.md](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/projects/embedding-model-300m/README.md)
- [projects/image-model-300m/README.md](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/projects/image-model-300m/README.md)

This issue is a source-level content review. Existing CPU receipts and honest toy/replay/target limitations remain valid within their stated scope. This audit did not rerun all training programs, commission human reviews or launch paid hardware. Keep content maturity, physical-device qualification and model-quality evidence separate.
