# Course review evidence: 0.17.1

This directory preserves a bounded set of existing review receipts, validation outputs, browser figure capture, and historical build failures for the 0.17.1 review. `manifest.json` records each copied file's SHA-256, byte size, and original path. The copies retain the source bytes exactly.

## What the evidence says

The saved-decoder extension remains optional. Its recorded v5 result is `QUALITY_GATE_FAILED`: the strict heldout reward-improvement floor failed. The mechanics and replay checks are separate from heldout quality and do not reverse that result or establish language-model quality.

The five-candidate learner-mechanics review was performed against the earlier ZIP, SHA-256 `3de18328abfc8499f6c2c785aae39d075a48fe01325e9aa7feff31ed408fcb30`. Its total measured exposure was 3 outer updates, 6 actor optimizer steps, 6 critic optimizer steps, and 20 reward-head optimizer steps; it recorded zero heldout quality scores. Two positive mechanics candidates passed and three negative controls were rejected as expected.

The documentation-reconciled 0.17.1 ZIP is a distinct artifact, SHA-256 `fe1318990ce10144237612ab86e879cb14b7e5b871d536a99936cc98ebfbeb13`. It has 40 members: 37 remaining members match the previously checked ZIP, and three Markdown files changed. The final ZIP received clean-extract, source-fidelity, preflight, semantic, and numeric-packet non-run checks with zero optimizer actions. The learner-mechanics result for the earlier ZIP must not be described as a new optimization run on the final ZIP.

The receipts also record that the existing 104 lesson JSON bodies and prior release snapshots remain unchanged. They do not establish broader human preference evaluation, retained behavior, target-scale model quality, hardware qualification, native EPUB-reader usability, or weight-release permission.

## Contents

- `receipts/` contains the independent campaign, starter, actor-leaf, five-candidate, ZIP, learner-plan, environment, and extracted-artifact reviews.
- `validation/` contains the final clean-extract receipt, rights/link report, and exact preflight, semantic, and numeric-packet stdout/stderr logs.
- `images/appendix-figure-browser.png` preserves the appendix figure browser capture.
- `logs/` and `history/` preserve earlier build failures, sandbox and disk failures, a concurrent EPUB-validation failure, the rejected first print layout, a subsequent print timeout, and the catalog-URL test race. The browser race log is retained; its original trace and screenshot were overwritten before archival and are not claimed as retained. The unrelated default-port test attempt is explicitly unqualified.
- `qa/` contains the final complete test, browser, root/Pages deployment, EPUBCheck, PDF pagination and all-page geometry evidence, plus the immutable site file inventory. The final browser run targets the frozen port 4203 build.
- `verification.json` summarizes exact source revisions, artifact hashes, scoped results and remaining work.

The packet omits run-authorized approval files, checkpoints, model weights, NPZ payloads, tokenizer/corpus material, and full private raw v5 campaign evidence. The public failed-quality numeric packet and figure are represented by their review receipts and browser capture, not by private campaign state. Local qualification passed 204 Node tests, 15 Python identity/appendix tests, 58 desktop/mobile browser cases, root and Pages paths, static links/schema, 57 pinned Black inputs and EPUBCheck with zero messages. Both unpruned PDFs pass every folio, 104 visual groups, 29 mechanism groups, one appendix figure and two command blocks per format; all 2,668 pages pass text geometry. Selected raster inspection does not establish human teaching or complete native-reader acceptance. Hosted checks remain separately pending until the draft PR runs. No merge or deployment is authorized in this review packet.
