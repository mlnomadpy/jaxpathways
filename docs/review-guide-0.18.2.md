# Reader presentation review: 0.18.2

This is a local UI candidate. The published edition remains 0.18.1.

Start with [the arrays chapter](http://127.0.0.1:4210/downloads/jax-foundations.html#chapter-4), then compare the [online lesson](http://127.0.0.1:4210/lesson-arrays-01.html).

The default reading flow keeps the explanation, worked calculations, essential code, figure and figure interpretation together. The expected example result stays visible. Detailed runtime qualifications and the repeated full execution log open only when requested. Hints, learner scaffolds, evidence checklists and references have named disclosures. The chapter's maturity and the figure's measured/calculated status stay visible.

The book cover provides direct reading and contents links. Route choices and phase plans expand on request; the contents reveals lessons one phase at a time. Appendix tables scroll within their own region on phones.

For review, read a chapter without opening optional details, then attempt an exercise. Check whether the lesson still makes sense, whether the control names predict their contents, and whether any information needed to solve the task should move back into the main flow.

Keyboard opening and closing of the execution log was checked in the browser. The existing desktop/mobile print lifecycle checks pass: printing expands supporting material, keeps reference solutions in their existing separate section, and restores the prior disclosure state. Existing renderer/print checks pass after updating the print inspector to recognize the shorter captions; its older caption support remains intact. Astro, type and lint checks pass. Lesson and phase source revisions are unchanged.

No paid training runs, cloud qualification or deployment are part of this presentation pass. The design rules are in [reader-disclosure-design.md](reader-disclosure-design.md).
