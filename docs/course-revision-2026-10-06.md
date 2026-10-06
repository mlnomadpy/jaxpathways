# Whole-course teaching revision — 6 October 2026

This revision implements a teaching pass across all 97 canonical lessons in all 19 phases. It follows the [diagram audit](diagram-audit-2026-10-06.md) and preserves existing runnable examples, staged builds, practice and measured evidence.

## What changed

- Rewrote all 97 introductions around the learner's question, the computation's purpose and a likely misconception.
- Added one tailored worked-reasoning section per lesson: concrete arithmetic, state transitions, shape tracing or diagnosis.
- Added 97 “Pause and reason” questions with explanations revealed after an attempt. These require no form entry and do not award competency.
- Added 22 conceptual/analytic mechanism diagrams covering every phase, placed beside the worked explanation.
- Extended the reader, printable book, Markdown, notebook and EPUB generators. Notebook PNG attachments do not require neighboring image files.
- Corrected categorical program-ID colors and the masked-token probability scale in the existing renderer.

Worked cases include a scalar loss hiding incorrect broadcasting, stable and overshooting gradient steps, unequal minibatch weighting, discrete versus continuous solver derivatives, covariance effects on a sum, inconsistent checkpoint/data boundaries, frozen/current/old policies, and a DPO margin increasing while chosen probability decreases.

This is an explanatory improvement to every lesson, not a claim that every existing paragraph was replaced or every diagram-backlog recommendation implemented.

## Diagrams delivered

| Phase | Mechanism |
| --- | --- |
| Welcome | Working folder → interpreter → saved program → checked result |
| Arrays | Axis alignment and accidental all-pairs residuals |
| Transforms | Mapped observations, shared weights and stacked outputs |
| State | Scan carry versus collected outputs |
| Optimization | Global-norm versus coordinate clipping |
| Networks | Convolution receptive field |
| Recovery | Model, optimizer, key, iterator and step boundary |
| Transformers | Actual single-head pre-LN block, residual and value paths |
| Performance | Dispatch, device completion and measurement interval |
| Distributed | Reduced chunk ownership and all-gather |
| Science | Independent trajectories versus sequential time |
| Probability | Equal marginal variance with different covariance |
| RL | Active, terminal-reward and padding transitions |
| Kernels | Program IDs and partial tile coverage |
| Internals | JVP/VJP input and output spaces |
| Deployment | Quantization, accumulation, bias units and rescaling |
| Operations | Validation, activation and unchanged release on rejection |
| Pretraining | MLM target/input separation; MIM spatial mask |
| Post-training | LoRA branches; RLHF lifetimes; DPO corrected margin |

All 22 rendered mechanisms were visually inspected. Follow-up changes fixed a residual-arrow route, clarified the attention value path, improved a rollout-table label and explained the kernel's actual padding/cropping strategy. These deterministic diagrams contain no generated benchmark imagery.

## Verification and boundaries

- All 97 complete figure experiments passed; all 22 new mechanism assets were visually inspected.
- All 97 scripts and notebooks passed in fresh CPU processes: 194 executions.
- All 97 current canonical content hashes match figure and execution records.
- All 26 independent hand-worked arithmetic checks passed.
- Production build and full checks passed: 62 tests, 68 static pages, 2,998 local references, 97 EPUB chapters, notebook attachments, beginner bundle and full workspace fidelity.
- Astro reported no errors or warnings; lint, formatting and source-code whitespace checks passed.
- The existing local preview returned HTTP 200 for the Transformer lesson. This is a connectivity check, not browser interaction validation.

The earlier diagram audit remains a snapshot of the state before implementation.

This pass does not supply natural-corpus training results, a 300M run, TPU qualification, independent expert review or observed learner outcomes. Existing planned capstones retain their status. Live browser/mobile interaction was not verified; rendered assets and generated markup were inspected and checked.

Further work in the backlog includes detailed async checkpoint timelines, richer attention/loss-mask interactions, spatial reconstruction/error panels from actual arrays, collective communication rounds and full real-data modality lifecycles. Current additions explain mechanisms without relabeling synthetic fixtures as production evidence.
