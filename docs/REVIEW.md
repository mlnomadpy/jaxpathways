# Current author review: 0.18.1

Start with [the teaching review guide](review-guide-0.18.1.md), [the complete local book](http://127.0.0.1:4210/downloads/jax-foundations.html) and [blank feedback notes](review-notes-0.18.1.md). The remaining 92 lessons now have worked-example and figure explanations, building on the 12 lessons revised in 0.18.0. All 22 chapter introductions connect their learning progression. The [editorial coverage](evidence/course-review-0.18.1/editorial-coverage.json) and [qualification packet](evidence/course-review-0.18.1/verification.json) bind the review to exact sources and exports. The candidate is not merged or deployed; human acceptance and existing scope limits remain open.

---

# Author review: 0.18.0

Start with [the 0.18.0 review guide](review-guide-0.18.0.md) and [blank feedback notes](review-notes-0.18.0.md). The [complete local HTML book](http://127.0.0.1:4210/downloads/jax-foundations.html) and [book overview](http://127.0.0.1:4210/book.html) use the frozen review preview. The guide lists the new lesson routes, workspace requirements and evidence limits. Exact source and artifact identities are in [the qualification packet](evidence/course-review-0.18.0/verification.json). This candidate addresses content issues #74 through #91 on top of PR #71 and is not merged or deployed.

Local reproduction: use a clean checkout of the repair branch, install the pinned Python environment and `npm ci`, run `npm run build`, then `npm run preview -- --host 127.0.0.1 --port 4210`. The PR site workflow retains the generated HTML/EPUB review book as a downloadable artifact. Prior review guides and failed model-quality evidence remain historical records below.

---

# Author review: 0.16.1

Start with the [frozen book preview](http://127.0.0.1:4201/book.html), [review guide](review-guide-0.16.1.md) and [blank feedback notes](review-notes-0.16.1.md). Exact source and artifact hashes are in the [qualification packet](evidence/course-review-0.16.1/verification.json). Implementation commit `9f123dd92f18f910486b4da13b360aeea10e6054` is pinned for review. This patch builds on draft PR #69; no merge or deployment.

API explanations now describe recognized calls in the displayed source, with distinct namespaces and no comment/string false positives. Print grouping keeps all 104 visual prompts and 29 mechanism prompts with their figures in full A4 and Letter output, with correct page footers. All 2,659 PDF pages pass text geometry checks; raster legibility review is sampled. All 104 lesson bodies and 23 historical release snapshots are preserved.

Local qualification passed 191 Node tests, 11 Python EPUB identity tests, 52 browser cases, 53 formatter inputs, root/Pages checks and EPUBCheck with zero messages. Hosted checks for this patch remain a separate gate. Apple Books remains unchecked because native control still reports the Mac locked. Human teaching, native/assistive reading, target validation, 13 developing phases and ten Planned full capstones remain open. The Transformer bridge is excluded while its failed campaign is repaired.

---

# Author review: 0.16.0

Start with the [frozen book preview](http://127.0.0.1:4200/book.html), [sequence PPO guide](http://127.0.0.1:4200/project-guides/posttraining-audit.html), and [review guide](review-guide-0.16.0.md). Keep feedback in the new [blank notes](review-notes-0.16.0.md).

The source and evidence are pinned at `a0715b45b0caa03729e9d17680f258bbc36e3ef1` on the [review branch](https://github.com/mlnomadpy/jaxpathways/tree/codex/sequence-ppo-review), based on draft PR #68. Local qualification passed 177 Node tests, 11 Python identity tests, 52 desktop/mobile browser cases, 53 formatter inputs, root/Pages-path checks and EPUBCheck with zero messages. The extracted 14-member project ZIP passed in a fresh CPU environment. [Exact sources, artifacts and raw checks](evidence/course-review-0.16.0/verification.json) are retained.

This is a synthetic sequence PPO extension with an explained GAE figure, a complete learner scaffold, semantic grading and per-stage edit files. It does not complete full RLHF or the ten full capstones. The 104 lesson bodies remain unchanged. Native reader, human teaching and target qualification remain separate open requirements. This candidate has not been merged or deployed; the frozen 0.15.1 source now has all 15 hosted checks green, while hosted 0.16 checks remain separate.

---

# Author review: 0.15.1

Start with the [frozen book preview](http://127.0.0.1:4199/book.html).
The [review branch](https://github.com/mlnomadpy/jaxpathways/tree/codex/model-recovery-review)
builds on draft PR #67. Implementation commit
`1660731fb3877d193d8a017ec6ee91394cc868a6` is identified in the
[pinned qualification packet](https://github.com/mlnomadpy/jaxpathways/blob/07a1fab3081302905322d480ea706ed209b24629/docs/evidence/course-review-0.15.1/verification.json).
This candidate has not been merged or deployed to production.

Use the [review guide](https://github.com/mlnomadpy/jaxpathways/blob/07a1fab3081302905322d480ea706ed209b24629/docs/review-guide-0.15.1.md)
and [blank feedback notes](review-notes-0.15.1.md). Inspect the
[image pilot and seven explained figures](http://127.0.0.1:4199/project-guides/image-model-300m-pilot.html)
and [two-controller recovery exercise](http://127.0.0.1:4199/project-guides/sharded-training-multiprocess.html).
The catalog, parent plan, audit and agent-facing discovery now expose the same
three bounded CPU pilots. Pilot, preparation-project and full-capstone status
labels describe their own scopes.

Local qualification passed 173 Node tests, 11 focused Python identity tests,
50 desktop/mobile Chromium cases, root/GitHub Pages path checks, and EPUBCheck
5.4.0 with zero errors or warnings. The retained CI formatter pass applies to
49 byte-identical current files. The packet records the exact source and the
whitespace-only helper follow-up. Hosted checks for this new branch remain
separate from the green frozen PR #67.

The image run completed eleven stages and 384 updates, but its selected
classifier failed quality: 0.3185 balanced accuracy versus a 0.40 floor and
0.3638 fixed color control. Its public archive contains source and numeric
evidence, excluding photos, feature vectors, fitted processors and weights.
The recovery extension demonstrates two local POSIX CPU controllers and
whole-cluster restart; remote, TPU, Linux-workflow and storage-durability
qualification remain open.

A separate audio validation incident retains two committed updates and no
completed pilot or final evaluation. Its execution gate is closed, exact
as-run sources and partial artifacts are preserved privately, and the
validation wrapper is now preflight-only. No replacement run is authorized.
Its runtime is excluded from this candidate.

The browser sample shows readable math, a complete labeled plot, the plot
interpretation, working reasoning reveal and code. The Mac relocked before
the final native EPUB check; that check remains pending. Earlier bounded
0.14 Apple Books evidence is retained separately. Thirteen phases remain
In Development, all ten full capstones remain Planned, and independent
teaching/career, broader reader, assistive-technology and target qualification
remain open. The 104 unchanged lesson bodies retain prior matching CPU
receipts; they were not all freshly executed locally for this candidate.

---

# Earlier review: 0.14.1

Start with the [frozen patch preview](http://127.0.0.1:4197/book.html).
The patch is [draft PR #67](https://github.com/mlnomadpy/jaxpathways/pull/67),
stacked on PR #66. The pinned verification packet is at commit
`d34e1e4b82eb474a54fa2703dc92bca164bd7a94`; its exact tested source commit is
`e6cd34698b1118727e168812a6a37a7dc302193f`. No merge or production deployment.

Use the [patch review guide](https://github.com/mlnomadpy/jaxpathways/blob/d34e1e4b82eb474a54fa2703dc92bca164bd7a94/docs/review-guide-0.14.1.md),
[feedback notes](review-notes-0.14.1.md), and
[verification record](https://github.com/mlnomadpy/jaxpathways/blob/d34e1e4b82eb474a54fa2703dc92bca164bd7a94/docs/evidence/course-review-0.14.1/verification.json).
This patch repairs the hosted Black failure and a formatting-sensitive checker
mutation. The reference model AST and all 104 lesson bodies are unchanged.
The 0.14 review guide below still describes the substantive course upgrade.

Local checks pass: 163 automated tests, 49-file Black check, fresh complete
recovery protocol, 46 desktop/mobile browser checks, root/Pages deployment and
EPUBCheck with zero errors or warnings. All 15 hosted checks passed for final PR67 head
`615be5fda0bda99e39292447a94508117b3accd2`. The later [follow-up receipt](evidence/review-qualification-followup-2026-10-21/verification.json)
retains these results separately from the frozen packet and earlier PR66 results.

Native 0.14.0 browser and Apple Books samples show readable math, plot labels,
code and working TOC navigation. The [compatibility matrix](book-compatibility.md)
records the exact edition and limited sample. This is not a new all-reader
qualification for the patch. Disclosure behavior, another EPUB engine, print and
actual assistive-technology acceptance remain open.

Thirteen phases remain In Development and all ten full capstones remain Planned.
Image/audio and multi-controller work continue outside this frozen patch. Record
confusing instructions and changed-input results in the notes before revealing
worked references. Independent teaching and career calibration remain needed.

---

# Earlier review: 0.14.0

Start with the [frozen book preview](http://127.0.0.1:4196/book.html).
The candidate is [draft PR #66](https://github.com/mlnomadpy/jaxpathways/pull/66),
commit `a3df5706b43355a8f218d86fdd100c74ff17ec94`, stacked on PR #65.
It has not been merged or published to production.

Use the [pinned review guide](https://github.com/mlnomadpy/jaxpathways/blob/a3df5706b43355a8f218d86fdd100c74ff17ec94/docs/review-guide-0.14.md),
[local feedback notes](review-notes-0.14.md), and
[local verification record](https://github.com/mlnomadpy/jaxpathways/blob/a3df5706b43355a8f218d86fdd100c74ff17ec94/docs/evidence/course-review-0.14/verification.json).

Review the [causal CPU lifecycle](http://127.0.0.1:4196/project-plans/causal-model-300m.html),
[checkpoint recovery project](http://127.0.0.1:4196/project-guides/recovery-audit.html),
and [embedding pilot](http://127.0.0.1:4196/project-plans/embedding-model-300m.html).
The causal plots explain why decreasing objective values did not meet the declared
QA quality floors. The embedding checker now rejects constant heads and disconnected
gradients. The recovery project tests fresh processes, interrupted publication and
competing writers on a local POSIX filesystem.

Local qualification passed 163 automated tests, 31 project reference suites,
46 desktop/mobile browser checks, root/subpath deployment checks and EPUBCheck
with zero errors or warnings. All 104 unchanged lesson sources match their retained
CPU execution receipts; they were not all freshly executed for this edition.

Hosted checks are separate: the site job failed Black formatting for two recovery
Python files. A follow-up branch is repairing that issue; this frozen commit must
not be described as fully green. All fourteen other hosted checks passed, including the lesson and causal-pilot
jobs. Inspect the exact results in [PR checks](https://github.com/mlnomadpy/jaxpathways/pull/66/checks).

The native browser inspection confirmed readable equations, code, plots and figure
interpretations. After earlier import/capture failures, opening an exact-copy EPUB
with an edition-specific filename through Finder opened 0.14 in Apple Books. The
introduction, TOC navigation, least-squares inline/display math, complete chart and
wrapped Python code rendered correctly in that sample. Disclosure interaction, a
second EPUB engine, all-chapter review, print and assistive-technology acceptance
remain pending.

All ten full model/retrieval capstones remain Planned. Thirteen phases remain In
Development. Image/audio lifecycle and multi-controller recovery work continue
separately. Use this pinned candidate for review, because the main checkout has
additional unfinished worker changes. The broader gap-repair goal remains active.

---

# Earlier review: 0.13.0

Start with the [frozen book preview](http://127.0.0.1:4195/book.html).
The current candidate is [draft PR #65](https://github.com/mlnomadpy/jaxpathways/pull/65),
commit `ae4f1499e63885da6a9bd83dd651bdc45d16c4d0`, stacked on PR #64.
It is prepared for your review and has not been merged or published to production.

Use the [review guide](https://github.com/mlnomadpy/jaxpathways/blob/ae4f1499e63885da6a9bd83dd651bdc45d16c4d0/docs/review-guide-0.13.md),
[blank feedback notes](https://github.com/mlnomadpy/jaxpathways/blob/ae4f1499e63885da6a9bd83dd651bdc45d16c4d0/docs/review-notes-0.13.md),
and [verification record](https://github.com/mlnomadpy/jaxpathways/blob/ae4f1499e63885da6a9bd83dd651bdc45d16c4d0/docs/evidence/course-review-0.13/verification.json).

This edition adds a [staged embedding CPU pilot](http://127.0.0.1:4195/project-plans/embedding-model-300m.html#cpu-pilot),
explained phase-specific objective plots, a source and numeric-evidence download,
one lesson stage dock, reserved figure geometry and mobile status-path wrapping.
It carries forward the 104-lesson review series described below.

All 159 automated tests, 42 browser checks, root/GitHub Pages path checks and
EPUBCheck conformance passed locally. The current full embedding checker passed;
its source and artifact hashes and numerical parity were independently checked.
All 14 hosted checks passed for the pinned draft commit. Native EPUB, actual assistive technology
and independent novice/specialist acceptance remain open.

A subsequent adversarial audit found that the complete embedding checker accepts
a constant MLM head; its model sub-check also accepts a disconnected encoder
gradient. Learner grading for those cases needs repair. Independent NumPy checks
agree with the unchanged reference's tied-head equations and selected gradients.
The next review edition will include stronger acceptance checks.

The small embedding pilot failed its retrieval quality floors and remains In
Development. All ten full model/retrieval capstones remain Planned. Causal,
image and audio workflows are being developed separately and are excluded from
this frozen candidate. Their newest validation findings are retained on the
[next review branch](https://github.com/mlnomadpy/jaxpathways/tree/codex/causal-image-review/docs/evidence).
The broader gap-repair goal remains active.

---

# Earlier review: 0.12.0

Start with the [frozen book preview](http://127.0.0.1:4194/book.html).
The review candidate is [draft PR #64](https://github.com/mlnomadpy/jaxpathways/pull/64),
at commit `93c0b2d1f547d52bb9118e12cb8857e76f9f65c3`. It builds on the
earlier draft upgrades in PRs #61, #62 and #63. It has not been merged or
published to production.

Use the [edition-specific review guide](https://github.com/mlnomadpy/jaxpathways/blob/93c0b2d1f547d52bb9118e12cb8857e76f9f65c3/docs/review-guide-0.12.md)
and [blank review notes](https://github.com/mlnomadpy/jaxpathways/blob/93c0b2d1f547d52bb9118e12cb8857e76f9f65c3/docs/review-notes-0.12.md).
The [complete material audit](https://github.com/mlnomadpy/jaxpathways/blob/93c0b2d1f547d52bb9118e12cb8857e76f9f65c3/docs/learning-material-audit-2026-10-21.md)
lists every phase, project, assessment, pathway and career route, with its
current scope and next contribution. The main checkout contains additional
uncommitted work; use the frozen preview or pinned revision for this review.

## What changed

The review series now provides learner-only Python and notebook scaffolds for
all 104 lessons, with 305 authored exercise/practice contracts. Worked references
and checkpoint answers remain separate. The book, downloadable materials and
website share canonical lesson bodies and visible maturity/evidence labels.

The latest edition deepens adaptive simulation and inverse problems, adds a
configurable natural-text decoder workflow, explains the measured plots, and
corrects TPU precision and setup instructions. Earlier drafts add substantive
probabilistic and reinforcement-learning work, experiment workflows and book
reading improvements. These are bounded implementations, with their limits
described beside the material.

## Suggested review order

| Area                      | Open                                                                                                                             | Look for                                                                                                                                                              |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scientific computing      | [Scientific workflow](http://127.0.0.1:4194/project-guides/scientific-inverse.html)                                              | Solver assumptions, stiff versus nonstiff work counts, sensitivity errors, same-data fitting and identifiability explanations.                                        |
| Transformer foundations   | [Natural-text workflow](http://127.0.0.1:4194/project-guides/text-harness-foundations.html)                                      | Data rights/splits, train-only tokenizer, masks, actual training exposure, cache/recovery and the limits of the six-update pilot.                                     |
| TPU learning              | [TPU course](http://127.0.0.1:4194/phase-guides/tpu.html) and [TPU systems](http://127.0.0.1:4194/phase-guides/tpu-systems.html) | Slice capacities, actual device-count receipts, result dtype versus physical accumulation, and measured versus estimated performance.                                 |
| Learner exercises         | [Book](http://127.0.0.1:4194/book.html)                                                                                          | Attempt a scaffold and changed-input case before revealing a reference. Record where instructions or feedback are insufficient.                                       |
| Discovery and progression | [Projects](http://127.0.0.1:4194/projects.html), pathways and careers                                                            | Whether scope, prerequisites, expected artifacts and contribution needs are easy to find. Career readiness and durations still need practitioner/learner calibration. |

## Verification and remaining work

The frozen candidate passes 154 automated tests, 32 desktop/mobile Chromium
checks and all 31 registered project reference suites. All 104 current lesson
hashes match CPU execution receipts; eight changed lesson script/notebook pairs
were freshly executed for this edition. Root and GitHub Pages subpath checks
cover links, status labels, SEO and schema. Official EPUBCheck 5.4.0 reports
zero errors and zero warnings. See the [source-bound verification record](https://github.com/mlnomadpy/jaxpathways/blob/93c0b2d1f547d52bb9118e12cb8857e76f9f65c3/docs/evidence/course-review-0.12/verification.json).
All 13 [hosted checks](https://github.com/mlnomadpy/jaxpathways/pull/64/checks)
also passed for the pinned candidate commit.

The inventory contains nine Available Draft phases and thirteen In Development
phases. All ten full model/retrieval capstones remain Planned. A separate SciFact
embedding lifecycle pilot is under internal review and is excluded from 0.12.0;
its measured retrieval quality is below its declared release floors.

One known interface follow-up is overlapping duplicate lesson stage navigation
in short viewports. Actual native EPUB/screen-reader sessions, independent novice
and specialist reviews, target-scale model training and broader hardware/service
qualification remain open. Automated conformance and numerical checks do not
establish learner understanding or production readiness.

Record the lesson/section, prediction, observed result, confusing step and
requested correction. Keep teaching, mathematical, interface and hardware
findings distinct so each can receive an appropriate retest.
