# Author review guide: 0.16.1

Open the [frozen book preview](http://127.0.0.1:4201/book.html) and keep your observations in [blank feedback notes](review-notes-0.16.1.md). This patch builds on [draft PR #69](https://github.com/mlnomadpy/jaxpathways/pull/69). It has not been merged or deployed. The [qualification packet](evidence/course-review-0.16.1/verification.json) records exact source hashes and artifact identity.

## Read the code guidance

Start with [PPO build step 3](http://127.0.0.1:4201/lesson-rl-03.html#section-13), then [the control-flow lesson](http://127.0.0.1:4201/lesson-state-04.html) and [TPU setup](http://127.0.0.1:4201/lesson-tpu-01.html). Each code-building-block explanation now follows an actual recognized call in the shown source. NumPy and JAX comparisons, JAX and Flax transformations, namespace and method calls, and optimizer losses have distinct explanations. Comments and quoted examples no longer trigger guidance. Unknown aliases are deliberately omitted.

Check whether the explanation helps you predict the function's inputs and output. For least squares, a vector right-hand side produces a residual array with shape `(1,)` under the pinned JAX 0.9.2 runtime; a matrix right-hand side produces one residual per target column. The API review and regression tests retain those distinctions.

This patch updates generated guidance in five lesson exports. All 104 lesson JSON bodies, numeric programs and prior 23 release snapshots are unchanged. A new edition entry records the shared renderer and print improvements without inventing lesson revisions.

## Read a figure before running code

Open [the PPO figure](http://127.0.0.1:4201/lesson-rl-03.html#section-10). Predict whether return must increase every update, read the axes and legend, then compare your explanation with the text below it. The comparison uses different numbers of inner epochs, so it does not establish a compute-matched algorithm ranking. Follow the full-size SVG and data/provenance links if the inline labels are too small.

## Review printed pages

The [offline HTML reader](http://127.0.0.1:4201/downloads/jax-foundations.html) includes print styling. In Chromium, figures now keep their title, prediction and provenance together; mechanism figures have valid caption relationships. Long explanations can continue on another page. A4 and Letter output include `Page N` footers.

The exact qualified local PDFs are `/private/tmp/jaxpathways-print-0.16.1-r1/output/pdf/book-all-a4.pdf` and `/private/tmp/jaxpathways-print-0.16.1-r1/output/pdf/book-all-letter.pdf`. A4 has 1,268 pages and Letter has 1,391 pages. Both pass all 104 visual and 29 mechanism pairings, every page number and geometry checks across 2,659 pages. Raster review covers 31 selected pages, including every previously failing figure pairing, plus a small number of full-resolution legibility samples. This is not a human review of every printed page.

Compare attention (Letter 481), RLHF (Letter 1266 / A4 1158) and the tall confusion-matrix figure (Letter 381 / A4 345). Check axis labels, panel sizes, equation wrapping and code. Three-panel figures have smaller labels; record whether you need the full-size online figure. Historical failed print output remains preserved separately.

## Review status and remaining work

The book stays In development. Thirteen phases remain in development, all ten full capstones remain Planned, and independent expert, learner and career feedback is still needed. Native Apple Books control continued to report the Mac locked after the reported unlock; no new native-reader or assistive-speech pass is claimed. EPUBCheck validates the export structure, not native reading usability.

Local checks passed 191 Node tests, 11 Python EPUB identity tests, 52 desktop/mobile browser cases, 53 formatter inputs, Astro/type/lint/format checks, root and GitHub Pages paths, static links/schema and EPUBCheck 5.4.0 with zero messages. Base PR #69 has 15 hosted checks green; hosted checks for this patch are a separate requirement. Not every numerical lesson or project was freshly executed locally for this renderer patch.

The saved-Transformer PPO bridge is separate work. Its first campaign failed in evaluation after eight in-memory updates, with only update 4 durably saved. That incident, its source and checkpoint are retained. The evaluator repair has no-update checks, but a new campaign remains gated. It is excluded from this review copy. No production model, hardware qualification, merge or deployment is implied.
