# Review the 0.12.0 scientific and decoder edition

This is an author review candidate. It adds deeper scientific and Transformer
workflows and repairs the precision explanation. It carries forward the 0.11.1
reader and print-answer fixes. The complete course still contains developing
phases and planned model lifecycles; this edition does not claim that every gap
has been closed.

Open the [frozen book preview](http://127.0.0.1:4194/book.html),
[decoder workflow](http://127.0.0.1:4194/project-guides/text-harness-foundations.html)
or [scientific project](http://127.0.0.1:4194/project-guides/scientific-inverse.html).
These local URLs show the candidate, not the published website.

The candidate contains 104 lessons across 22 phases, with 305 authored exercise
and practice contracts. Learner-only downloads keep TODO scaffolds separate from
worked references and checkpoint answers. Use the verification record with this
edition to distinguish current executions from historical receipts.

## Verified for this edition

The final checks pass: 154 automated tests, 32 desktop/mobile Chromium checks,
31 registered project reference suites, nine TPU lesson checks and three
hardware-source-binding checks. Eight changed lessons have fresh script and
notebook executions; all 104 lesson content hashes match retained CPU receipts.
The other two changed lesson revisions update metadata only. Root and
`/jaxpathways/` page, link, status, SEO and schema checks pass. Official EPUBCheck
5.4.0 reports zero errors and zero warnings for the retained EPUB bytes.

See [the source-bound verification record](evidence/course-review-0.12/verification.json)
and its logs and inspected screenshots. Hosted checks for this new branch are
separate from these local results. Actual native-reader sessions and independent
human teaching acceptance remain pending.

## Review route

| Stop | Source to inspect | Questions for your review |
| --- | --- | --- |
| Adaptive simulation | [Scientific project](../projects/scientific-inverse/README.md), [Diffrax reference](../projects/scientific-inverse/solution/diffrax_workflow.py), [independent checker](../projects/scientific-inverse/tests/check_diffrax.py) | Can you explain why two accurate stiff solves use different step counts? Are the exact solution, initial-time contract, requested times and tolerances clear? |
| Sensitivity and inverse problems | [Science lesson 3](../phases/10-science/03-differentiate-through-a-solver/lesson.json), [lesson 4](../phases/10-science/04-fit-parameters-and-scale-the-simulation/lesson.json), [retained figures and arrays](../projects/scientific-inverse/reference/diffrax/receipt.json) | Does the error panel make the overlapping derivative curves understandable? Can you distinguish discretization error, measurement noise and an unidentifiable parameter? The fit/profile comparison must use the same observations. |
| Natural-text decoder | [Workflow guide](../projects/text-harness/foundations/README.md), [learner scaffold](../projects/text-harness/foundations/starter.py), [independent checker](../projects/text-harness/foundations/tests/check_foundations.py) | Can you follow the isolated setup, chapter splits, train-only BPE, padding/document masks and next-token loss? Attempt a changed mask/cache case before reading the reference. |
| Interpret the pilot | [Measured plot](../projects/text-harness/foundations/reference/pilot-metrics.png), [raw report](../projects/text-harness/foundations/reference/run/pilot-results.json), [portable evidence index](../projects/text-harness/foundations/reference/evidence-index.json) | Can you explain the step-6 rise, the zoomed training-loss axis and the common byte denominator? Does the text make the six-update scope and repeatedly inspected test chapters clear? |
| Recovery and consumer handoff | [Training state and package code](../projects/text-harness/foundations/train.py) | Can you trace parameters, Adam moments, PRNG, sample order and cursor across a fresh process? Which checks fail for a changed batch configuration or corrupted tokenizer? The larger decoder configuration is formula-only. |
| TPU setup and receipt tasks | [Slice exercise](../phases/19-tpu/01-get-a-tpu-running-and-verify-the-runtime/lesson.json), [runtime receipt lesson](../phases/19-tpu/02-bootstrap-venv-and-verify-tpu-runtime-receipt/lesson.json) | Check both named slice capacities and the inclusive 64/128 GiB filters. Can you distinguish the observed reduction at the actual local device count from theoretical totals for four or eight devices? A truncated hash is not a digital signature. |
| Precision evidence | [TPU precision lesson](../phases/21-tpu-systems/02-bfloat16-int8-precision-and-xprof-profiling/lesson.json), [dtype/IR probe](../phases/21-tpu-systems/02-bfloat16-int8-precision-and-xprof-profiling/labs/precision_contract_probe.py) | Do operands, requested/returned result dtype, compiler IR and physical accumulation stay distinct? Why does one seeded cast comparison differ while another agrees? The CPU result does not qualify TPU lowering. |
| Access and book | Projects, the dedicated decoder guide, the standalone HTML reader and the EPUB | Are the new workflows easy to find? Do the two new optional practices state their separate dependencies? Can you attempt tasks before revealing answers and find printed chapter solutions? |

## Evidence boundaries

One known P2 interface follow-up remains: lesson pages have two independent stage
navigation bars that can overlap in a short viewport. The lesson controls remain
usable. The inspected [precision-page screenshot](evidence/course-review-0.12/precision-intro.png)
records it; consolidation and a single-dock browser regression are next-wave work.

The scientific workflow uses a synthetic equation with a known solution and
known-noise observations on CPU. The three explained figures retain their raw
arrays and source hashes. They do not establish real-instrument validity,
confidence intervals or scientific-domain acceptance.

The decoder pilot trains a 10,880-parameter, two-layer FP32 model for six CPU
updates on retained Project Gutenberg text. It has meaningful numerical,
data-coverage, cache and recovery checks. Its measured losses do not establish
fluent language generation, a blind benchmark, accelerator performance or
approximately-300M training.

The separate embedding lifecycle remains under development and is excluded from
this frozen edition. The ten full model/retrieval capstones keep their planned
status. Existing physical TPU qualifications retain their exact original source
bindings and workload limits.

Independent specialist reviews, novice sessions and career calibration remain
open. The [book matrix](book-compatibility.md) distinguishes EPUB conformance,
the inspected 0.11.1 Chromium print samples and pending native/assistive-technology
sessions. Later text revisions do not inherit an unperformed native reading
session.

Record actual findings in [the blank review notes](review-notes-0.12.md), including
the edition, exact source or artifact hash, expected behavior, observed behavior
and any assistance needed. A passing reference implementation is not evidence
that a learner understood the material.
