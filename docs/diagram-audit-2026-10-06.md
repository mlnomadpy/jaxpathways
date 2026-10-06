# Course diagram audit — 6 October 2026

The course needs more explanations of **how a computation works**, especially tensor axes, state changes, information flow and system boundaries. It already has substantial evidence plots. Adding another loss curve to every lesson would not solve the main problem.

This audit proposes diagrams and acceptance checks; it does not add them to the website. The implementation backlog covers every canonical lesson.

- [97-lesson backlog](diagram-backlog-2026-10-06.md): current visual, proposed change, priority and factual check.
- [Detailed diagram specifications](diagram-specifications-2026-10-06.md): composition, learner questions, explanations and verification for the first implementation batches.
- [Machine-readable inventory](diagram-audit-2026-10-06.json): source paths, content hashes, existing figures, review scope and recommendations.

## What was reviewed

The inventory covers **97 authored lessons across 19 phases**. For each lesson, the review considered its learning problem, core explanation, section topics, visual specification/computation, recorded figure data and exercise. Selected mechanism-heavy lessons also received closer implementation review. This is an explanatory-design audit, not a new line-by-line correctness review of all course code.

All 97 current lesson-content hashes match their figure records; SVG, PNG and execution artifacts exist. This verifies source correspondence, not mathematical correctness or teaching effectiveness. No numerical experiments were rerun for this audit.

There are **92 executed figure specifications and 5 conceptual flows**, containing **120 panels**: 50 line plots, 40 bar plots, 22 heatmaps, 5 flows, 2 decision fields and 1 vector plot. There are also 7 sets of conceptual cards and 2 interactive figures. The backlog accounts for these existing explanations rather than treating them as absent.

Visual inspection covered 9 lesson images: `transforms-03`, `optimization-12`, `transformers-03`, `pretraining-02`, `recovery-03`, `deployment-05`, `posttraining-04`, `distributed-03` and `kernels-01`. Four project images were inspected: audio waveform/spectrogram, cross-modal retrieval, text cache precision and image lifecycle. Project scope also included registry metadata for 19 projects, introductory/stage material for the four modality harnesses, and the figure requirements in the CLIP, SigLIP and retrieval plans. This is not a complete audit of every project implementation.

No live browser or responsive interaction review was completed. Layout findings below are based on source inspection. Actual mobile readability and screen-reader behavior remain to be tested.

## Most consequential findings

| Finding | Evidence in the current course | Change with the greatest teaching value |
| --- | --- | --- |
| Results are often visible while the mechanism remains abstract. | `vmap` has three output bars; the Transformer block has an output-minus-input heatmap. | Add axis mapping and the actual architecture before the evidence plot. Keep the numbers as a check. |
| Spatial concepts sometimes lose their spatial meaning. | Masked-image modeling flattens hidden patches into heatmap rows. | Show the source image, patch coordinates, visible/masked patches, reconstruction and error in spatial layout. |
| State and timing need more than a before/after comparison. | Async checkpoint acceptance, JIT reuse, PPO policy versions and pipelining have distinct intermediate states. | Use timelines, state machines and labeled feedback edges. |
| Some categorical data are encoded as numeric magnitude. | Sample IDs are connected/plotted by height; artifact identities appear as bars; program IDs use a continuous color scale. | Use ordered ID strips, release state machines and discrete tile ownership maps. |
| Broad lesson titles cover mechanisms that the main plot does not demonstrate. | `optimization-12` compares optimizer loss paths but does not visualize clipping or schedules. The walkthrough already acknowledges this. | Add separate, explicitly scoped clipping and schedule examples; preserve the existing qualification. |
| Recorded execution and empirical measurement need more precise labels. | Executed figure code can calculate an analytic standard-error curve, hypothetical roofline or modeled communication payload. | Label provenance per panel: measured, analytically calculated, simulated or conceptual. |

Examples worth preserving include the tangent interaction, attention-mask interaction, gradient-check error curve, projection geometry, Euler error comparisons, held-out error examples and explicit clean-versus-corrupted evaluation. The existing checkpoint diagram correctly merges parallel state inputs; it needs a consistent-step boundary, not a replacement with a sequential chain.

The backlog recommends **7 keep, 18 refine, 67 augment and 5 replace-lead** decisions. These are opportunities, not a requirement to add 67 standalone graphics. Reuse a small vocabulary of diagram components and add a figure only where it answers a specific learner question.

## Implementation order

| Batch | Lessons / scope | Deliverable and reason |
| --- | --- | --- |
| 1. Foundations and math | `arrays-02`, `transforms-03`, `state-01/03`, `optimization-01/08/12` | Axis alignment, mapping, key tree, scan lattice, tensor loss flow, curvature trajectory and clipping. These explain dependencies used throughout the course. |
| 2. Attention and training objectives | `transformers-01/02/03`, `pretraining-01/02/03`, `posttraining-01/02/04/05` | Actual block architecture; attention versus loss masks; spatial reconstruction; contrastive pairing; LoRA, PPO/RLHF and DPO information flow. |
| 3. Recovery and numerical deployment | `recovery-02/04/05`, `deployment-05/08`, `kernels-01` | Consistent checkpoint timeline, integer arithmetic, layerwise conversion checks and categorical tile coverage. |
| 4. Performance and distributed systems | `performance-01/03/05`, `distributed-01/02/03`, `kernels-03` | Host/device timing, liveness, global/local ownership, weighted reductions, collective rounds and buffer lifetimes. |
| 5. Probability, simulation and RL | `science-02/03`, `probability-02/03/04`, `rl-01/02/03/04` | Time/batch lattice, sensitivity chain, uncertainty geometry, sampler diagnostics and rollout/PPO diagrams. |
| 6. Engineering and modality integration | recovery/operations workflows and modality harnesses | Artifact lineage, release state transitions and a shared lifecycle map with modality-specific contracts. |

Start with one complete lesson in each diagram family, including prose, alternative text and an independent check. Use learner explanations to evaluate whether it works before repeating the pattern widely. “Learner can predict the next state or identify the broken contract” is a better success criterion than figure count.

## Modality and project diagrams

| Project | What already works | Proposed addition |
| --- | --- | --- |
| Text harness | Training/attention and measured cache-memory/error figures; explicit token and checkpoint contracts. | A byte-to-token/shifted-target strip, the actual two-head model, and prefill versus one-token decoding with cache position and validity. Do not reuse the single-head lesson diagram unchanged. |
| Image harness | Real synthetic pixel examples, held-out confusion counts, distribution-shift failures and precision comparison. | Image layout/range transformations and the actual CNN receptive field; branch clean/corrupted evaluation from the same immutable split. |
| Audio harness | A waveform and spectrogram of an actual error with time/frequency units and an explanation. | Overlay one 128-sample Hann window and 64-sample hop; connect 15 frames × 65 bins to time-averaged features. Make the loss of temporal order visible. |
| Cross-modal harness | Recorded loss, a real fixture image, similarity matrix and shifted retrieval failures. | Two encoders, normalization and a multi-positive mask; show a query with its ranked candidates and declared positive set. Avoid depicting only diagonal positives when semantic classes supply multiple positives. |
| CLIP plan | Already specifies training/similarity/retrieval evaluation figures. | Add the image-by-text score matrix with row/column normalization and an explicit false-negative case. It remains a planned real-data project. |
| SigLIP plan | Already specifies analytic loss curves, dense/chunked parity, score distributions and fair cost comparisons. | Add independent pair labels and pairwise sigmoid losses beside the CLIP normalization diagram. A shared bias affects thresholds but cannot change within-query ranking by itself. |
| Retrieval plan | Already specifies ranked-result, quality/latency, ANN-memory and migration figures. | Add two separate truths: exact-neighbor agreement and judged relevance. Then show candidate generation → fusion → reranking → eligibility/response, with eligibility applied consistently at the relevant stages. |
| Project workflow guide | Folder layout and a runnable experiment-preparation CLI. | Add source/config/data → prepared workspace → separately executed run → evaluated artifact lineage. Clearly label that preparing the workspace does not train a model. |

For the planned model-building capstones, reuse a lifecycle overview: dataset recipe → tokenizer/processor → objective phases → checkpoint selection → evaluation → precision qualification → export/Hub publication → serving/monitoring. Each stage should name its input artifact, output artifact and gate. The branches differ by modality; MLM followed by contrastive learning is an embedding-model recipe, not a universal causal/image/audio recipe. Do not draw an unexecuted 300M run as completed evidence.

## Choose the medium by the question

| Medium | Use it for | Requirement |
| --- | --- | --- |
| Deterministic SVG | Tensor shapes, token alignment, architecture, patches, spatial ownership, integer arithmetic. | Generate labels and dimensions from explicit structured inputs; keep text selectable. |
| Mermaid rendered during the build | Lineage, release states, high-level workflows and sequence diagrams. | Export SVG and PNG for the website, notebooks, EPUB and offline downloads. Do not require a live Mermaid runtime to read the lesson. |
| Matplotlib from actual arrays | Loss/gradient trajectories, contours, calibration, uncertainty, latency and error. | Save raw values, units, source identity and the exact measurement or analytic assumptions. |
| Small interaction | Axis selection, attention eligibility, clipping threshold, one scan step. | Buttons, direct selection and a short step control fit the course better than forms/dropdowns. Provide a static equivalent. |
| AI-generated illustration | Optional nontechnical context or clearly synthetic illustrative media. | Keep it out of equations, tensor labels, architecture facts, benchmark evidence and plots. Any generated data used in an experiment must be labeled and tracked as such. |

AI illustration is not needed for the priority diagrams. Deterministic rendering gives us higher factual control and lets us regenerate a diagram when the lesson changes. JAX describes `vmap` as automatic vectorization, which supports an axis-mapping diagram rather than a speculative hardware-thread illustration. [JAX automatic vectorization](https://docs.jax.dev/en/latest/automatic-vectorization.html).

## Delivery-system improvements

1. **Place figures beside the relevant concept.** The lesson template renders the main visual after the deep-concept sections. Add section-addressable figures so a learner sees a mask while reading about masks. Keep the final evidence plot near the experiment.
2. **Support several named figures without replacing the current pipeline.** Extend canonical lesson content with stable figure IDs, section placement, semantic type and provenance. Keep the current single visual as a backwards-compatible default during migration.
3. **Separate units from color semantics.** The renderer specially handles `device ID` but not `program ID`. Add an explicit categorical scale. Similarly declare probability bounds and shared limits instead of detecting them from a few exact unit strings.
4. **Use meaningful alternative text.** The website currently uses the title plus “a detailed explanation follows”; rich `visual.alt` content is not consumed there. Keep the nearby walkthrough, add concise semantic alt text and connect a longer description where needed.
5. **Test narrow layouts.** Images have a 480px minimum width inside a keyboard-focusable horizontal scroller. Preserve this fallback, but split dense panels and offer vertical diagram layouts. Confirm readability at 320/375/768px content widths in a real browser before claiming mobile success.
6. **Preserve provenance across formats.** Include source/renderer identities, diagram assumptions, raw arrays for plots, and labeled conceptual status. A source-hash match protects freshness; an independent invariant protects correctness.

Relevant implementation files: [renderer](../scripts/render-lesson-figures.py), [lesson template](../src/lib/lesson-template.js), [visual component](../src/lib/lesson-visuals.js), [visual styles](../src/styles/features/lesson-visuals.css), [distribution builder](../scripts/build-distribution.py).

## Quality gate for every new figure

Each figure must answer one question, name its entities/axes/units, define every arrow, and state its scope. Data flow, control flow and gradient flow need different labels or styles. Frozen, trainable and restored state must be explicit.

Use the following teaching sequence: **predict → read the marks → trace one example → explain the outcome → change one condition**. The explanation must point to a concrete cell, branch, interval or region. Include what the figure cannot establish: a fixture loss is not generalization, an analytic roofline is not a device benchmark, and CPU interpretation is not observed TPU overlap.

Before release, independently check shapes, arithmetic and invariants; inspect rendered SVG and PNG; verify colors are not the only distinction; test keyboard and narrow-screen access; and confirm notebook/EPUB/static exports retain the explanation. The lesson backlog lists a specific acceptance check for each recommendation.

## Validation of this audit delivery

The structured backlog contains exactly the current 97 canonical lesson IDs, in curriculum order, with no omissions or duplicates. Its relative Markdown file links resolve. Current figure content hashes were checked against canonical lesson content.

`npm run build` and `npm run check` passed: 59 tests, 68 static pages, 2,954 local references, curriculum validation and offline distribution checks. These checks validate the existing course and audit delivery; they do not validate the proposed diagrams, which still require implementation and review. No new course figures or numerical lesson changes were made in this audit.
