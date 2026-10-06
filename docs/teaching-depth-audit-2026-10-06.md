# Teaching depth: findings and first repair

Date: 2026-10-06. Scope: structural inventory of all 97 authored entries; close editorial and executable review of the eight pretraining/post-training lessons. This is an authoring review, not independent technical approval or a learner study. The [before inventory](teaching-depth-inventory-2026-10-06.json) preserves the measurements used for triage. Section and exercise counts are signals to inspect, not quality scores.

## The central problem

The newer objective lessons had the same thin shape: five short sections, a complete small training program, one prediction experiment, and one transfer exercise. All eight lacked incremental build instructions. They stated important caveats, but too often left the learner to derive axes, denominators, gradients and failure explanations alone. Their plots mostly showed falling training loss; that did not expose the mechanism or how to diagnose a different result.

A working synthetic example establishes a mechanism under narrow conditions. It does not establish that a learner can prepare real data, make an architecture decision, reproduce a training phase, evaluate generalization or operate a released model. The course needs both the mechanism lesson and a connected, progressively implemented project. More topic names and project specifications do not close that gap.

## Specific failures and repairs

| Lesson | Missing or weak teaching | Repair in this revision | Remaining boundary |
| --- | --- | --- | --- |
| MLM | Shapes and selected-token reduction asserted more than worked through; adding a common scalar to all class logits was a weak masking test because softmax ignores that shift | Token/input/selection walkthrough, axis trace, unequal-mask host oracle, analytic full-logit gradient, class-dependent perturbation, final masked-token probability matrix | Repeated-token data, no full positional/padding-aware text encoder or natural-language evaluation |
| Masked image modeling | One square-patch check and falling MSE did not expose layout, leakage or irreducible ambiguity | Rectangular RGB slicing oracle, hidden-pixel intervention, ambiguous-target optimum, target/predicted hidden-patch comparison | Linear decoder on a fixed spatial pattern; variable masks and real-image encoder transfer still missing |
| Contrastive learning | Loss decrease and same-pair retrieval could be mistaken for semantic quality | Hand-derived two-pair loss, temperature/ranking distinction, two-step finite differences, cosine matrix, source and evaluation-pool reasoning | Four synthetic pairs with a shared linear encoder; CLIP/SigLIP remain planned capstones |
| SFT | Shift rule not supported by enough concrete alignment and reduction examples | Explicit input/next-target table, ragged host likelihood oracle, gradient-position map, context-collision counterexample | Transition-table training, not a pretrained causal Transformer with a real instruction corpus |
| LoRA | Rank-one success on a rank-one target hid capacity and scaling failures | Matrix-gradient derivation/oracle, nonunit-alpha rank-two merge check, independent rank-one residual bound, initial-factor gradient bars | Dense layer; no full-model adapter targeting, real-data adaptation or QLoRA kernels |
| Reward modeling | Score ranking alone hid ambiguity, calibration and identifiability | Worked likelihood values, independent linear gradient in both label directions, conflicting-label optimum, preference-margin plot | Synthetic comparisons, no human-label collection or held-out reward generalization |
| RLHF mechanics | A compact PPO loop left estimator and policy identities too implicit | Four-object state explanation, signed clipping derivation, exact baseline cancellation, regularized finite-action optimum, reference/final policy bars | One-action bandit; no language rollout/value model, token masking or sequence RLHF pipeline |
| DPO | Positive margins could be mistaken for increasing chosen-response probability | Worked margin derivative, falling-chosen-probability counterexample, independent categorical gradient, reference-cache contract, margin bars | Categorical responses; sequence log-probability integration and held-out language behavior remain missing |

Each of these lessons now has a staged build, an independent calculation, a changed-condition task, and a figure interpretation tied to the actual saved data. Equations use the existing KaTeX/MathML pipeline. Time estimates rise from 75 to 105 minutes to allow for derivation and practice; these are editorial estimates, not measured learner completion times.

## What the broader inventory tells us

Before this repair, deployment had six of eight lessons without staged builds and five with at most one experiment and one practice task. Performance had three of five with at most one of each; operations had two of eight. These are priorities for close review, not proof that every lesson in those phases is shallow. Other phases having more sections or tasks does not establish their quality either.

The separate [whole-course audit](full-course-audit-2026-10-06.md) and [roadmap](course-roadmap-2026-10-06.md) identify the larger gaps: real-data training, coherent specialization routes, complete integration projects, benchmark governance and target-runtime evidence. Their earlier source-hash inventories are historical snapshots taken before this rewrite.

## Triage beyond the rewritten lessons

A second pass inspected the problem statements, teaching-section topics and task prompts of all eight deployment, five performance and eight operations lessons. This was not a fresh execution audit of those implementations. It sharpens the next review questions:

- **deployment-01/08:** move from one dense layout or a small dense/normalization architecture to attention, pooling, tokenizer/processor parity and an actual supported checkpoint family.
- **deployment-03/06/07:** connect local export checks and container practice to a complete measured request path and a named target runtime; desktop execution must stay distinct from edge-device evidence.
- **performance-03/04/05:** require a diagnosed optimization decision from the trace, traffic model or rematerialization comparison, then show a changed workload where the recommendation fails or stops helping.
- **operations-06/07/08:** require a linked incident spanning data identity, evaluation, traces and model selection, rather than treating isolated gates as evidence for operating a whole system.

The deployment precision lesson already contains two experiments, worked quantization arithmetic and affine zero-point/bias accounting. It should be reviewed for actual learner transfer and runtime qualification rather than being grouped with the single-experiment lessons solely because it is recent.

## Next repairs, in order

Follow-up: the [deployment teaching revision](deployment-teaching-revision-2026-10-06.md) adds staged construction, semantic mapping checks, fresh-interpreter export, precision accounting, request validation and deadline reasoning across all eight deployment lessons. The larger model-family conversion and named target-runtime work described below remains outstanding.

1. **Deployment and cross-framework conversion:** teach a full input-to-output architecture mapping, layerwise numerical errors, calibration choices and actual export/runtime tests. Distinguish a dense-layer parity demonstration from a supported model-family converter. Require changed layouts, normalization, padding and precision cases.
2. **A real-data text lifecycle:** connect corpus/source splits, tokenizer training, causal and masked objectives, durable resume, domain adaptation and held-out evaluation. A small permitted corpus with fixed source revisions must precede a 300M model recipe.
3. **Natural-image representation learning:** connect image–caption inspection to dual encoders, multi-positive labels, CLIP/SigLIP objective checks and fixed-pool retrieval. A roadmap page stays planned until starter, reference, checks and executed evidence exist.
4. **Performance and operations:** relate each trace or dashboard to a reproduced bottleneck/failure and a verified intervention. A timing plot without workload boundaries, or a run tracker without artifact lineage, is inadequate teaching.
5. **Independent learner walkthroughs:** have a learner explain a changed condition without opening the solution. Record where the explanation, code order or figure failed them; revise the lesson before claiming comprehension.

## A completion gate based on learner work

For each repaired lesson, ask the learner to predict a result, implement a missing part, validate it independently, change one assumption and diagnose a failure. Require them to explain one actual figure with axes, values and limits. Use hidden/changed-condition assessment separately from public reference checks. Do not mark a whole pathway complete because every lesson file has an authored status.

The existing CI proves structure, exports and reference execution. Editorial approval additionally requires a concrete starting point, named symbols and shapes, a worked mechanism, progressively runnable steps, nontrivial transfer and a reasoned connection to the next artifact. Passing CI does not replace that judgment.

## Verification of this revision

- All eight revised figure experiments executed successfully; all eight rendered figures were inspected, with follow-up corrections to axis units and the masked-token probability label.
- All 97 authored scripts and notebooks passed the CPU runner: 194 executions. The current audit finds no stale lesson receipts.
- Strict KaTeX rendering passed for the 571 prose/math fields inspected across the eight lessons.
- Build and checks passed: 57 tests, 67 static pages, 2,910 local references, 97 offline chapters, EPUB references, workspace bundles and all 19 phase guides.
- The refreshed visual inventory retains 92 executed figures and five conceptual diagrams; eight existing figures now include an additional diagnostic panel. These are saved CPU results, not hosted execution or independent learner evaluation.

The website changes remain local until publication. No natural-corpus, target-device or large-model training result was added by this revision.
