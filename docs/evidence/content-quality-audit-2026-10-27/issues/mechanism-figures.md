<!-- jaxpathways-content-audit:2026-10-27:mechanism-figures -->

Audited October 27, 2026, edition 0.17.1, PR #71 head `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`. Priority P2. Finding type: curriculum-extension.

## Teaching problem

Many figures now have useful reading guidance. A few remaining visuals show an outcome without making the chapter's main mechanism visible:

- tpu-01 puts chip counts and GiB on one "count / capacity" axis. Separate units and show process/device/chip ownership.
- networks-02 uses anonymous leaf indices. Show module paths, GraphDef/State separation and shared versus copied state.
- transforms-05 sum/mean bars do not show specialization. Add a signature-to-trace ledger with value-only and shape/static changes.
- rl-05 critic-loss plot does not explain the reverse GAE scan. Add time steps, terminal/cutoff observations, bootstrap and trace masks with a hand-computed value chain.
- kernels-03 memory bars do not show pipeline scheduling. Add buffer-lifetime/load/compute/store overlap with prologue, steady state and drain.
- transforms-02 overlays loss and derivative with different units on one axis. Split panels while keeping matching x coordinates and the zero-gradient interpretation.

Use deterministic SVG/Mermaid/standard plots for exact tensors, timelines and values. Decorative generated images cannot establish these relationships.

## Acceptance criteria

- [ ] Generate each figure from the exact fixture or an explicitly labeled schematic and define axes, units, arrows, masks and legends.
- [ ] Explain one specific observation and why it follows from the code/math, plus a prediction question for a changed input.
- [ ] Include a numerical/structural check for plotted labels and a text equivalent conveying the same mechanism.
- [ ] Check narrow reader, downloadable HTML, print and EPUB legibility. Keep existing useful plots where they answer a different question.

## Audited source

- [tpu-01](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/19-tpu/01-get-a-tpu-running-and-verify-the-runtime/lesson.json)
- [networks-02](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/05-networks/02-model-and-state-with-flax-nnx/lesson.json)
- [transforms-05](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/02-transforms/05-tracing-static-arguments-and-recompilation/lesson.json)
- [rl-05](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/12-rl/05-learn-values-and-gae/lesson.json)
- [kernels-03](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/13-kernels/03-tiling-memory-and-pipelining/lesson.json)
- [transforms-02](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/02-transforms/02-losses-and-value-and-grad/lesson.json)

This issue is a source-level content review. Existing CPU receipts and honest toy/replay/target limitations remain valid within their stated scope. This audit did not rerun all training programs, commission human reviews or launch paid hardware. Keep content maturity, physical-device qualification and model-quality evidence separate.
