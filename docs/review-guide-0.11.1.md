# Review the 0.11.1 reader repair

This patch is based on the frozen 0.11.0 candidate, not the ongoing 0.12 model work. The original EPUB already contained learner contracts; the standalone HTML reader omitted them. The actual rendered-reader regression now covers all 104 chapters and all 303 exercise/practice contracts.

1. Open the standalone reader and inspect an exercise and its practice tasks. Inputs, outputs, invariants, hints and TODO code should precede their separate reference reveals. Comparison signs must render as signs rather than literal HTML entity text.
2. Print or save a chapter. Reasoning disclosures expand for print, interactive reveal controls disappear, and chapter-labeled reference solutions and checkpoint reasoning appear in the appendix. On returning to the screen, previously open and closed disclosures should keep their original state.
3. Inspect the four representative chapters in A4 and Letter: welcome-01, optimization-09, transformers-04 and tpu-06. The retained print receipt identifies the browser, source HTML hash, settings and sample artifacts. The sample scope does not establish full-book or native EPUB acceptance.

Use the [0.11 review notes](review-notes-0.11.md) for findings, recording the exact revision you review. Check the [book qualification matrix](book-compatibility.md) and [patch evidence](evidence/reader-parity-0.11.1/verification.json). Original 0.11 hosted checks passed all eleven jobs at d7bc44af; patch CI is tracked separately.

Thirteen developing phases, ten planned capstones, independent human teaching/learner review and broader target qualification remain open. Native reader and screen-reader checks still require their actual sessions. This patch is not a production deployment.
