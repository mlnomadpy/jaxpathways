# Author review: edition 0.14.1

This patch preserves all 104 lesson bodies and the 0.14 teaching scope. Start with
the [0.14 review guide](review-guide-0.14.md) for the causal, embedding and local
recovery material. The [0.14.1 preview](http://127.0.0.1:4197/book.html) is a
separate frozen copy; the older 4196 preview remains edition 0.14.0.

The patch repairs the two-file Black failure from PR66. The recovery model is
AST-identical to 0.14.0. Its checker now locates an owned wrong-order mutation by
AST source span rather than formatting, and requires a specific sample/order
mismatch against the independent oracle. The [amendment verification](evidence/recovery-format-amendment-2026-10-21/verification.json)
and [fresh protocol receipt](evidence/recovery-format-amendment-2026-10-21/protocol-evidence.json)
bind the exact current project files. Earlier receipts remain historical.

Native observations are recorded separately in the [book compatibility matrix](book-compatibility.md).
The sampled 0.14.0 HTML and Apple Books pages render equations, a labeled plot and
worked code correctly. This does not qualify every chapter, disclosure behavior,
a second EPUB engine, printing or actual screen-reader speech. The patch changes
edition metadata; it does not turn that sampled earlier rendering into a new
full-reader qualification.

Record feedback in [the patch notes form](review-notes-0.14.1.md). All ten full
model/retrieval capstones remain Planned, thirteen phases remain In Development,
and both small pilot recipes retain their failed task-quality outcomes. Image,
audio and multi-controller source repairs continue separately from this patch.
Hosted results for each pinned commit must be inspected separately. Nothing in
this review series has been merged or published to production.
