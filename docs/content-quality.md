# Content quality standards and review history

> Historical counts in the dated findings below are snapshots. Current coverage and priorities are recorded in the [2026-10-06 full audit](full-course-audit-2026-10-06.md), [course expansion plan](course-roadmap-2026-10-06.md) and regenerated [content audit](content-audit.md). The authoring requirements remain applicable.

## Current teaching-depth repair

The [deployment teaching revision](deployment-teaching-revision-2026-10-06.md) deepens all eight deployment and interoperability lessons with staged builds, numerical reasoning, changed-condition exercises and four additional conceptual figures. The phase culminates in a reproducible release dossier; actual container, accelerator and device qualification remain separate work.

The [2026-10-06 depth review](teaching-depth-audit-2026-10-06.md) records the shallow patterns in the newest objective lessons and the concrete repairs to all eight pretraining/post-training lessons. Their larger model integrations still remain work to implement. File availability and structural checks must not be presented as course depth.

## Phase-level quality

A phase must connect individual lessons into learner work. Its canonical `studyGuide` in `curriculum/course.json` must explain the central question, provide a starting diagnostic with reasoning and a repair lesson, group every lesson into ordered build milestones, and ask for transfer to a changed condition. Include an observed failure symptom with a specific diagnostic check, project evidence the learner should retain, and material that remains outside the phase's supported scope.

Generate the browser phase guide and phase README from that same source. Keep lesson IDs stable when repairing sequence, update explicit prerequisites, and validate that milestones cover each lesson exactly once in order. A phase guide should make the next action clear without requiring form entry. It supplements executable lesson practice and project checks; it is not a substitute for either.

Retain an independent calculation for new numerical mechanisms. Evaluation examples must preserve their intended population and report missing work explicitly. Explain new plot panels against their own data, particularly when panels use different fixtures or units. Count availability separately from runtime qualification, expert review and learner comprehension.

## Historical foundation review

The earlier authored count described available files and CPU reference execution. It did not establish course quality. The initial lessons often contained two explanatory paragraphs, one tiny example, and a one-step exercise; that was insufficient for the intended course.

This revision expands the four array lessons and five transformation lessons. Each connects a concrete problem to a mechanism, works through numerical or shape reasoning, includes two prediction-led experiments, and supplies foundation/practice/challenge tasks with reference reasoning. The topics progress from preprocessing and explicit state to derivatives, losses, batching, compilation, and tracing. Their evidence requirements are now specific to the capability being taught.

## Editorial checks

These are questions for a human or authoring agent, not a word-count test:

- Can a learner explain why the calculation works before copying the API call?
- Are the input/output contract, axes, precision assumptions, and known results explicit?
- Does at least one check use an independent derivation or reference implementation?
- Does practice require applying the mechanism to changed conditions rather than repeating the example?
- Does failure diagnosis connect an observed symptom to evidence and a repair?
- Do claims distinguish a bounded example from a general result or performance guarantee?
- Is the required portfolio evidence clear enough for another person to inspect?
- Does the learner understand why this lesson is a prerequisite for the next one?

## Historical whole-course audit and limits at that time

The [before audit](content-audit-before.md) found 53 missing lesson bodies and nine shallow starters among 72 canonical entries. The [current audit](content-audit.md) records every entry, missing teaching components, companion availability and current source-bound CPU receipts. The starter label describes actual instructional gaps rather than a missing word count.

Nine starters (arrays/devices, four state lessons, four optimization lessons) have now been expanded with worked mechanisms, staged code, two prediction experiments and two transfer/diagnosis exercises each. The previously missing jaxpr lesson now includes six teaching sections, typed data-flow diagrams, transformed-program inspection, closed constants, structured loop comparison and a reproduced/repaired tracing failure. The nine earlier array/transformation rewrites and guided setup remain available. There are 20 lesson drafts and 52 unwritten roadmap entries.

This is still an incomplete course. “Teaching structure present” in the audit means that teaching components exist; it does not mean independent technical review or learner validation. Training and deployment have authored preparation; operations, scientific and kernel specializations retain substantial gaps. Two staged projects and three synthesis assessment review drafts exist. Four modality guides connect the lifecycle, but their complete harness implementations remain planned. No TPU execution receipts or independent expert reviews exist.

The build rejects available numerical lessons missing problem, objectives, teaching sections, experiments, transfer practice, diagnosis or references. Setup follows a separate installation walkthrough contract. It also checks that incremental build code, experiments and reference solutions survive generation into Markdown, notebooks, scripts and EPUB. These prevent omissions, not poor pedagogy; the editorial questions above remain necessary.

## Editorial findings across available material

- Setup now explains terminals, folders, environments and OS-specific commands. Python installation support and novice walkthroughs still need testing on each OS.
- Array lessons use unequal axis sizes, independent NumPy references and explicit boundary checks. More figures are needed where layouts are difficult to picture.
- Transformation lessons derive derivatives, separate sample/aggregate gradients and bound compilation/timing claims. Earlier lessons still present some programs as whole examples rather than incrementally built files.
- State lessons now explain key ownership, carry and tree invariants. Key replay is bounded to a tested environment; cond and scan examples do not establish accelerator performance.
- Optimization now works through residuals, analytic gradients, cancellation, stability and optimizer continuation. The datasets are deliberately synthetic; extension to real data belongs in actual model/data lessons.
- Jaxpr now distinguishes abstract programs from hardware execution. Printed variable names and primitive parameters are version-dependent, so tests check numerical behavior instead of entire strings.
- The route catalog and organizer plans name future outcomes; they cannot substitute for executable route projects and exams. Unwritten lesson pages must show an explicit availability notice and never generic exercise/evidence/checkpoint filler.

## Historical authoring order

Author a coherent Flax NNX training phase with a real dataset and parameter/state contracts, then data loading and durable recovery, a Transformer block, and profiling with actual artifacts. Build their integration projects before moving into advanced career routes. Add supported TPU execution and independent review after CPU correctness. This sequence follows the existing prerequisite graph; it should not create another layer of empty “full lessons.”

## CPU practice and course expansion, October 3

This iteration adds 18 drafts: a logical CPU setup lab, two sharding lessons, five neural training lessons, four Grain/Orbax input-and-recovery lessons, four Transformer lessons and two CPU performance lessons. There are now 38 available drafts among 73 entries; 35 remain unwritten. The earlier 20/52 figures above describe the previous audit, not current coverage.

Three parallel agents authored CPU sharding/recovery, neural training and learner UI/performance. Internal cross-reading caught and repaired a missing precision-policy comparison, oversized language-model build increments and unchecked checkpoint step metadata. These are internal peer checks, not independent expert approval. All current generated scripts and notebook code were executed in fresh CPU processes, including observed logical-device counts. A second staged classifier project checks changed seeds, independent numerical references, evaluation isolation and replay.

The attention-mask explorer is an analytic browser illustration backed by matching JAX experiments. Available content and unwritten roadmap entries are now separate views. Copyable snippets, incremental builds and collapsed assembled programs support deliberate practice without inflating the lesson with duplicate visible code.

Remaining work includes asynchronous/durable failure boundaries, accelerator profiling artifacts, communication and multi-host recovery, scientific/probabilistic/RL/inference/operations/kernel content, route synthesis projects/exams, novice walkthroughs and TPU validation. Do not turn logical CPU exercise results into accelerator throughput claims.

## Deployment and interoperability expansion, October 5

Four deployment drafts now cover model/framework contracts, a JAX serialized export round trip, weight/activation/accumulator precision, and edge deployment. The course has 42 authored drafts among 75 entries, with 33 still planned. The new Projects catalog organizes both implemented projects and separately labels future phase project briefs.

The numerical companions check known dense outputs, changed batch/input conditions, rejected export signatures, representation error bounds, zero-channel quantization, held-out activation clipping and end-to-end request preprocessing. Optional real-framework companions compare Keras on JAX, PyTorch and TensorFlow, and convert a TensorFlow model to FP32/INT8 LiteRT with held-out checks. See `deployment-lab-validation.json` for versions, errors and source hashes. None of this establishes target-device performance or independent editorial/learner validation.

## Learner writing and math revision, October 5

The 42 authored lesson openings now introduce a concrete learner question in a more conversational voice. Key mathematical explanations introduce symbols and small worked values before code. The inline follow-up covers variables, shapes, arrays, expressions and numerical examples throughout narrative, hints, exercises and checkpoints, including the regression paragraph identified during review. Display equations and inline notation use a shared KaTeX renderer; executable code and plain-text data-flow diagrams keep their own formatting.

Browser and printable output include accessible MathML; EPUB carries native MathML; notebook and Markdown exports retain compatible TeX delimiters. Rendering, export-fidelity and source checks cover equation presence, escaped prose, unchanged code, malformed TeX and regression against unformatted mathematical shapes/operators. These checks establish source and rendering contracts, not visual browser review or learner comprehension. See [the writing guide](learner-writing-and-math.md) for the continuing authoring standard.

## Math & optimization expansion — 2026-10-05

Phase 04 now has twelve connected lesson drafts. Eight new lessons cover vector geometry, direct least squares and conditioning, Jacobians and directional derivatives, curvature, stable likelihood-based losses, regularization and validation, minibatch gradient noise, and Adam/clipping/schedules. The four existing lessons receive additional mechanisms and counterexamples. The course now contains 50 authored drafts among 83 entries; 33 remain planned.

The new material uses defined KaTeX notation, hand calculations, staged code, independent comparisons, prediction experiments and changed-condition practice. Numerical checks include residual orthogonality, an analytic Jacobian and adjoint identity, exact enumeration of batch variance, a ridge solution, a saddle counterexample and manual Adam recurrences. These checks are bounded evidence about instructor examples; they are not independent editorial review or learner assessment. See [the sequence and scope](math-optimization-expansion.md).

## Visual teaching and execution visibility — 2026-10-05

The visual audit covers all 83 entries. All 50 authored lessons now pair a figure with a prediction and an explanation of its relationship to the code: 45 executed CPU plots and five conceptual diagrams. The reader exposes recorded script output; notebooks contain real per-cell streams and Matplotlib image outputs, with a rerunnable plotting cell. Printable and EPUB editions include the figure assets. The 33 planned entries receive specific visual/evidence requirements without being relabeled authored.

Artifact inspection includes every figure, with refinements for categorical device ownership, shared probability scales, contrast, and checkpoint diagrams that show state components converging at a boundary. Source and cell hashes guard against stale execution output. This remains an authoring audit and reference execution, not independent learner validation. See [the complete audit](visual-learning-audit.md).

## Explain the figure that is actually present

Review each figure against its saved data and the code that produced it. Start by identifying the axes, units, legend, scale, and any initial or pre-update point. Describe a concrete visible pattern and walk through representative values, then connect that pattern to the computation. Distinguish an observation from its inferred cause and name what the plot cannot establish.

Do not describe generic expected behavior when this run shows something more specific: identify tied attention weights, repeated rows, optimizer rebounds, zero-height bars, and overlapping curves where they actually occur. Equal adjacent bars are not overlapping lines. A probability is not accuracy, a gradient is not loss, a categorical device ID is not a performance score, and a timing percentile is not a maximum. Explain logarithmic ratios and mixed linear/log scales explicitly.

Keep walkthroughs in canonical `content.visual.reading` and `content.visual.connection` with paragraph breaks and explicit math delimiters. Their figures and explanations must travel together through the lesson, notebook, Markdown, and books. Timing prose must remain meaningful after reruns: interpret units, sample variation, and measurement boundaries rather than freezing machine-dependent durations into the text. See `docs/plot-interpretation-audit.md` for the reviewed examples and known review limits.
