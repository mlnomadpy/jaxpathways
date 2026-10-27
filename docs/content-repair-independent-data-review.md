# Independent data review: distributed and recovery exercises

Reviewed the current lesson sources against placement-memory issue #82 and exercise-contracts issue #79. This review made no lesson-source edits.

## Findings

### Confirmed renderer failure in distributed-02

- Source: `phases/09-distributed/02-a-sharded-training-step/lesson.json`, `content.visual.code` (line 245).
- The code sets `visual_data['kind']` to `table`. The lesson figure worker passes executed visual data to `draw_panel`; `scripts/render-lesson-figures.py` supports `heatmap`, `field`, `line`, `bar`, `vectors`, and `flow`, then raises `ValueError` for other kinds at line 115. The separate `content.diagram.panel.kind = table` schema is supported by `draw_mechanism`, but that does not make a `visual.code` table renderable.
- Counterexample: `/private/tmp/jaxpathways-0170-clean-learner-env/bin/python scripts/render-lesson-figures.py --worker --lesson distributed-02 --output-dir /tmp/jaxpathways-independent-figures` fails with `ValueError: Unknown figure kind: table`. The 2x2 lesson script itself passes; the failure is the rendered visual artifact path.

No other concrete data or execution errors were found in the requested lessons.

## Checks and evidence

Exported the full lessons with `lessonScript()` from `scripts/lesson-artifacts.mjs` and ran them using `/private/tmp/jaxpathways-0170-clean-learner-env/bin/python`:

- `distributed-01`: PASS. The 4-device row/column ownership checks and masked ten-row mean complete.
- `distributed-02`: PASS. The 2x2 update matches an independent NumPy loss, gradient, momentum, and parameter-update oracle. Parameters and momentum both use `P(None, "model")`; gradients retain that model-column sharding. The valid-count cases print data-shard counts `[4,4,1,1]` for five valid examples and `[3,3,0,0]` for three, including the empty shard. The four replicated 1D local means sum to four times the global mean; the separate uneven-count example is weighted correctly. The 12-row transfer reports counts `[6,6,1,1]`.
- `recovery-06`: PASS. Its three-run exercise computes actual training errors `{linear-rate-0.02: 0.1401000006, linear-rate-0.15: 5.0763e-10, training-lookup: 0.0}` and held-out errors `{linear-rate-0.02: 0.1199446163, linear-rate-0.15: 5.0763e-10, training-lookup: 6.0}`. It creates three runs in a fresh temporary local experiment, logs each `model.json`, reloads each artifact, verifies predictions, then confirms training-only selection chooses the lookup model while validation selection chooses a different run.

The 2x2 ownership diagram states that feature rows are split on `data` and replicated across `model`; weight columns and momentum are split on `model` and replicated across `data`; gradient columns follow the parameter ownership. The section explains that the global masked numerator combines row and output-column contributions, while the denominator is valid rows times output columns. It explicitly avoids claiming a specific collective lowering or physical network schedule. These labels and caveats align with the tested placements and exercise behavior.
