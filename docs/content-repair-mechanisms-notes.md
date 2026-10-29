# Content repair mechanism notes

## P2#90 and P2#89: transforms, kernels, and recovery

### Changed lesson sources

- `phases/02-transforms/02-losses-and-value-and-grad/lesson.json`: split value and derivative into separate panels with identical weight coordinates, distinct units, numerical checks at `w=1` and `w=2`, and a constant-offset prediction.
- `phases/02-transforms/05-tracing-static-arguments-and-recompilation/lesson.json`: replace the sum/mean outcome bars with a four-call signature ledger. It distinguishes value-only changes from shape and static-mode changes and labels the trace interpretation as per-process structural evidence, not a universal cache guarantee.
- `phases/13-kernels/03-tiling-memory-and-pipelining/lesson.json`: add a prologue/steady-state/drain schedule naming `x_hbm`, `y_hbm`, `out_hbm`, the actual `axpy_body` consumer and slot reuse conditions. It states that the overlap is schematic and not a CPU timing claim.
- Recovery-03 and recovery-04 now lead with short learner-authored transitions and worked state/log traces. Their full code, support helpers, negative controls and existing downloads remain in place. The recovery-03 interpretation contrasts Adam count 3 to 4 with a reset count of 1. Recovery-04 uses the recorded four-batch continuation after step 2 and asks learners to interpret the reader-only and model-only negative controls.

### Checks performed in this repair

- Parsed all five edited lesson files with `python -m json.tool` successfully.
- Exported each full script through `lessonScript()` in `scripts/lesson-artifacts.mjs` and executed it with `/private/tmp/jaxpathways-0170-clean-learner-env/bin/python`. All five ended with `PASS: transforms-02`, `PASS: transforms-05`, `PASS: kernels-03`, `PASS: recovery-03`, and `PASS: recovery-04`.
- Rendered all five figures with `scripts/render-lesson-figures.py --worker --lesson <id> --output-dir /tmp/jaxpathways-mechanism-figures`; outputs stayed in `/tmp`. Inspected the transforms-05 and kernels-03 mechanism tables, then shortened labels where the first render clipped text. The final versions fit their rendered cells. No repository figures, notebooks, execution receipts, or release metadata were regenerated.

### Tests and artifact work still needed

- Still run repository lesson schema/content checks and lesson visual tests. The direct worker renders verified the configured renderer and mechanism-table schema, but not narrow reader, downloadable HTML, print, EPUB, or notebook rendering.
- After the owner selects the artifact workflow, regenerate the relevant committed `outputs/visual.json`, figure artifacts, and lesson content-hash receipts. The current figure output is temporary under `/tmp/jaxpathways-mechanism-figures`.
- Fresh-download script and notebook parity checks remain necessary for recovery-03 and recovery-04. The exported full scripts did run successfully and retained their negative controls.
- The transforms-05 ledger presents structural specialization inputs, not observed trace counts. Any future visual of trace counts needs a single-process fixture and must label counts as local observations.

### Packaging and central metadata

- The recovery downloads remain standalone and still contain duplicated support helpers. If reducing that duplication, create a versioned shared support module and package a copy alongside each downloadable script/notebook, or retain a generated standalone bundle. A download must not rely on a repository-root import path.
- Central curriculum, release maturity metadata, and generated source parity were outside this repair scope. After lesson source edits, the owner should determine whether those registries or generated script/notebook/book artifacts need updates, then regenerate them through the normal release workflow.
- The existing recovery receipts establish a fixed CPU fixture only. They do not establish separate-process, machine-loss, asynchronous, multi-host, remote-storage, or arbitrary-version recovery.
