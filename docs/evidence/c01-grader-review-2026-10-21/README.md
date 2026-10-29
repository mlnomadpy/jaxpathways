# C01 learner-grader review after the 0.13 freeze

The frozen reference model and historical training records are unchanged.
Following the C02 grader findings, root inspected the C01 acceptance boundary
and reproduced two incorrect learner candidates passing its model sub-check:

- An MLM head that always returns zero logits passes the loss/gradient checks.
  Its MLM gradient norm is zero, and its logits differ by about 0.294 from the
  separately reconstructed tied head. The loss oracle accepts candidate logits,
  and a finite difference through the same disconnected candidate compares zero
  with zero.
- Stopping gradients at the encoder output preserves forward logits exactly and
  also passes. The encoder-block MLM gradient norm is zero, compared with about
  0.812 for the unchanged reference on the same fixture.

The separate complete zero-head workflow subsequently finished with exit 0 and
`status: pass`, including fresh-process recovery and full retrieval checks.
Its exact checker output and run-receipt binding are retained here. This proves
a complete-checker false positive for the zero head; the disconnected-gradient
case remains scoped to the model sub-check. No retained original run was changed.

Root also checked the unchanged reference against independent NumPy tied-head
equations and six loss finite differences spanning both encoder blocks, tied
embeddings and the MLM head. Forward error is below 1e-7, and the selected
derivatives agree. This distinguishes a learner-grading defect from a claim that
the actual retained reference used a zero head or a disconnected encoder.

Sol's next contract gives one Luna exclusive checker-only ownership: independent
full encoder/head logits and objective derivatives, sensitivity controls, both
owned incorrect candidates rejected, equivalent implementations accepted, and
all existing checks retained. Full capstone, target, quality and human gates stay
open. Frozen PR #65 and historical source/receipt identities remain unchanged.

The retained checkers reference the external source/runtime paths used for these
observations. `sha256.json` identifies the source and result files. The frozen
model/checker are available at the pinned PR #65 revision; the mutation scripts
construct their owned candidates without editing those files.
