# Reference review and JAX course blueprint

Reviewed 2026-10-03. Source: [AI Engineering from Scratch at 3be078b](https://github.com/rohitg00/ai-engineering-from-scratch/tree/3be078b37ffd8f0c04953c0678e48f5c6d0c7775). This is a design and pedagogy review, not an endorsement of every technical claim in that repository.

## Scope and evidence

The [inventory](reference-course-inventory.json) covers **all 523 English phase lesson files across 20 phases**: complete file reads by the inventory script, headings, build steps, diagrams, exercise introductions, and companion file counts. I inspected the compact index for every lesson. The corpus contains 923,120 whitespace-separated words, 390 Mermaid blocks, and 531 interactive figure slots. Slots count references, not distinct widgets or successfully rendered diagrams. Counts are descriptive, not a quality score.

Close reading was selective: setup, chain rule/autodiff, CNNs, and excerpts of representative lessons from each phase; lesson template, contributor rules, learner tutor instructions, pathway manifest, project catalog and a staged project manifest; static build/content-source and figure runtime. The CNN reader was also inspected live after its dynamic content loaded. This does **not** claim a line-by-line editorial or technical review of all 923,120 words, execution of reference code, exhaustive mobile/accessibility testing, or validation of every figure. The source revision is fixed; the live site may differ.

To regenerate the structural inventory from a read-only checkout:

```sh
python3 scripts/audit-reference-course.py /path/to/ai-engineering-from-scratch --output docs/reference-course-inventory.json
```

## What makes the reference teach effectively

Its [lesson template](https://github.com/rohitg00/ai-engineering-from-scratch/blob/3be078b37ffd8f0c04953c0678e48f5c6d0c7775/LESSON_TEMPLATE.md) is a connected sequence:

1. **Problem:** a concrete failure, constraint, or task that makes the concept necessary.
2. **Concept:** intuition, a diagram, intermediate values, and terms before API calls.
3. **Build it:** small increments, with explanation of what changed and why.
4. **Use it:** connect the mechanism to a practical library or engineering decision.
5. **Ship it:** a named artifact someone can inspect or reuse.
6. **Exercises:** reproduce, modify, then transfer or investigate a failure.
7. **Terms and references:** correct misconceptions and explain why a source matters.

The CNN lesson does this particularly visibly: architecture lineage → shape traces → branching and residual diagrams → LeNet → reusable VGG block → BasicBlock → complete small ResNet → parameter comparison → pretrained adaptation → harder exercises. The diagrams support specific reasoning; they are not banners inserted between paragraphs. The chain-rule lesson shows actual values and reverse derivatives before constructing an autograd engine incrementally.

The project system goes further: [learner starters, reference solutions, cumulative stages, fixtures, recorded outputs and completion reports](https://github.com/rohitg00/ai-engineering-from-scratch/blob/3be078b37ffd8f0c04953c0678e48f5c6d0c7775/projects/README.md). It separates reference-solution passes from learner completion and describes the scope of public tests. A downloadable notebook alone does not supply this teaching workflow.

## Findings across all phases

The counts below come from the full inventory. Teaching observations are based on the phase index and sampled explanation/build/exercise sections, not every paragraph.

| Phase | Lessons | Pattern to adapt for JAX |
|---|---:|---|
| 00 Setup | 12 | Required-now versus optional-later dependencies; verify the selected route. Improve ours with explicit terminal, folder and interpreter instructions. |
| 01 Math | 22 | Work concrete values, shape contracts and derivatives by hand, then compare with the library. |
| 02 ML | 18 | Change data/model conditions and diagnose error; use repeated experiments rather than a single successful fit. |
| 03 Deep learning | 13 | Construct operations first, then compose a training system; distinguish training state from inference behavior. |
| 04 Vision | 28 | Architecture changes have reasons, shapes and parameter consequences; build a small Flax model before a large pretrained model. |
| 05 NLP | 29 | Inspect representations and evaluate alternatives; include tokenization and packing contracts, not only model calls. |
| 06 Audio | 17 | Distinguish a CPU mechanism demonstration from actual model generation and accelerator requirements. |
| 07 Transformers | 16 | Trace patch/token/head dimensions before implementation; use the same discipline for attention masks in JAX. |
| 08 Generative models | 15 | Isolate the mechanism in a small example before framework integration, e.g. low-rank updates. |
| 09 RL | 12 | Move from estimator intuition to controlled comparisons; vectorized environments require explicit state and keys. |
| 10 LLM construction | 24 | Join isolated components through artifacts, manifests, dependencies and evaluation gates. |
| 11 LLM engineering | 17 | Teach system loops and failure paths, then map the hand-built mechanism to real APIs. |
| 12 Multimodal | 25 | Explain data/mask/loss contracts and label simplified stand-ins honestly. |
| 13 Tools/protocols | 31 | Trace contracts, validation and errors; adapt this discipline to checkpoints, loaders and distributed coordination. |
| 14 Agent engineering | 54 | Include decisions, verification and ownership; a workflow may be more useful than a larger model. |
| 15 Autonomous systems | 22 | Demonstrate crash/replay behavior; apply to training recovery without confusing replay with exactly-once side effects. |
| 16 Multi-agent systems | 25 | Explicit coordination, provenance and independent verification; useful analogies for distributed run evidence. |
| 17 Infrastructure | 28 | Connect performance and cost to workload requirements; separate measured results from calculators. |
| 18 Safety | 30 | Include repeatable evaluation and clearly bounded toy demonstrations. |
| 19 Capstones | 85 | Decompose an end-to-end build into teachable components and cumulative integration checkpoints. |

Late phases often organize builds by implementation contracts rather than headings literally called “Step 1.” A low step-heading count does not prove missing instruction. JAX should adapt relevant teaching patterns; it does not need copies of every AI topic.

## Course structure: one shared foundation, several kinds of work

Keep the four-part progression already represented in our manifest, but make each transition a demonstrated capability:

- **Start here:** open a workspace, choose an interpreter, run and modify a saved file, inspect device/version output. No Git, TPU, Docker, or Node dependency for the first success.
- **Understand JAX:** arrays and shapes → pure functions → derivatives → batching → compilation → explicit random/state → compiled loops. Finish with an independently checked regression experiment.
- **Build a training system:** manual updates → Optax → Flax NNX → data pipeline → checkpoints/recovery → evaluation → profiling. The same small model grows across these lessons.
- **Choose work:** application and systems routes reuse these lessons, then add focused projects.

Career routes should describe work and evidence, not imply guaranteed employment or invent unrelated sets of notebooks:

| Work | Role direction | Evidence project | Additional capabilities |
|---|---|---|---|
| Train and adapt models | ML / training engineer | Recoverable Flax training run | Optax, data loading, evaluation, Orbax, mixed precision, sharding |
| Serve predictions | Inference engineer | Measured prediction service | Serialization, batch shapes, compilation warmup, latency/throughput, load tests |
| Operate accelerator workloads | ML platform / TPU operations engineer | Recovery and run-observability report | Runtime setup, data/checkpoint storage, failure recovery, distributed launch, monitoring |
| Improve compute efficiency | Accelerator performance engineer | Profile-driven optimization dossier | Benchmark synchronization, roofline, HLO, memory, collectives, Pallas |
| Explore new models | Research engineer | Reproducible architecture comparison | Autodiff, masks, Flax modules, controlled experiments, evaluation |
| Differentiate simulations | Scientific ML engineer | Inverse-problem experiment | scan, vmap, numerical precision, differentiable simulation, parameter recovery |
| Move an existing stack to JAX | Framework migration route | Behavioral parity audit | NumPy/PyTorch/Keras bridges, state/PRNG translation, numerical and performance checks |

These are proposed route outcomes. Most advanced lessons remain planned locally. The actual local source has 19 authored lessons and 53 planned briefs; earlier conversation claims of 56 full lessons are not evidence of current files.

**Route contract to implement next:** ordered shared lesson IDs with `required`, bridge/optional classification, prerequisite links, estimated time, hardware and evidence project IDs. Current local manifests select whole phases, which is too coarse for precise career routes. Do not silently migrate the schema: update generator, validators, CLI and route-aware navigation together. A learner can skip already demonstrated prerequisites but should see why a bridge is recommended.

## Lesson contract and a concrete example

A lesson needs one main question, a visible prerequisite, and an achievable artifact. Organize the reader as **Understand → Build → Practice → Keep the result**, while preserving direct headings and deep links. These can be anchors on one page; tabs that hide the concept while coding are not required.

For “Batch a function with vmap”:

1. Problem: the per-example prediction works, but a Python loop obscures batching.
2. Draw one function with input `(D,)`, weights `(D,)`, scalar output; then draw mapped input `(B,D)` and shared weights.
3. Ask which axes change and what output shape to expect.
4. Build one prediction; establish a loop baseline; add `vmap`; compare outputs independently.
5. Explain `in_axes=(0,None)` using the same diagram and variable names.
6. Change the batch size and add gradients; diagnose accidentally mapping weights.
7. Keep code, predictions, assertions and a shape explanation. Timing is optional here and requires warmup/synchronization if included.

For every command: identify **where to run it, which shell, which folder, which environment, what to expect, and what a common failure means**. For every code increment: state whether to create, replace or append, which file to edit, and how to check the intermediate result. Avoid a long finished script masquerading as a guided build.

Introduce unfamiliar words at first use. Require only the packages needed by this lesson. Solutions stay collapsed in the reader; starter and instructor code should be distinguishable. Estimated time must include the actual build/exercise, not just reading.

## Diagram plan

Choose a figure because it answers a question. Every interactive figure needs a prediction prompt, named inputs, observable output, a textual equivalent and a connection to a runnable check.

| JAX concept | Visual / control | Learner question | Independent check |
|---|---|---|---|
| Broadcasting | Align axes and expand size-one dimensions | Which axis is reused? | Small NumPy reference and explicit indexing |
| Autodiff | Computation graph with values and local derivatives | How does the seed reach each input? | Analytic derivative and finite differences |
| vmap | Map axis highlighted, shared arguments stationary | What changes when `in_axes` changes? | Loop versus mapped outputs |
| jit/tracing | Python call → trace → compiled executable → reuse | Which changed argument triggers tracing? | Observed trace events with version/hardware context |
| PRNG | Key-splitting tree, repeated-key branch highlighted | Why do two draws repeat? | Reuse versus split-key experiment |
| scan | Carry and time-indexed input/output shapes | What must stay constant across steps? | Python-loop reference |
| Sharding | Global array partitioned over a named mesh | Which value lives on which device? | Device/sharding inspection on actual target hardware |
| Recovery | Snapshot of params, optimizer, RNG and data cursor | What differs after resume if one component is missing? | Interrupted versus uninterrupted training |
| Performance | Work, memory and communication on a roofline | Which bound is plausible? | Real measured profile; calculator explicitly labeled |
| Pallas | Grid/block/tile access with boundary mask | What happens at a partial tile? | Reference outputs on edge shapes and target device |

Use static SVG for architecture and tensor shapes; Mermaid for dependency/state flow; interactive SVG/canvas for parameter-sensitive mechanisms. Motion must have a purpose and pause/reduced-motion support. Keyboard controls, legible narrow layouts, text descriptions and print fallbacks belong in the figure contract. In-browser analytic illustrations must not claim to run Python JAX or validate TPU behavior.

The reference has both static assets and dynamic figure providers. Unknown figure IDs leave an empty host in its runtime. Our build should reject unknown IDs and retain a textual fallback on runtime failure. Source inspection also confirms the JAX introduction references a batch-normalization distribution widget and a ViT lesson references a batch-statistics inference widget. These may teach related ideas, but they are weak matches to the central mechanisms. Match figures to the exact learning question rather than adding one per lesson to satisfy a count.

## UI and UX blueprint

### Landing page

Each section should answer one learner decision:

1. What can I learn/build here? One concrete promise and **Start here**.
2. Which work interests me? Training, inference, operations/performance, scientific/research; each card names skills, artifact and role directions.
3. Where does my chosen role begin? Career routes with prerequisite bridges and a clear starting lesson.
4. What does learning look like? One actual lesson/project preview and its output.
5. Where is the shared course? Compact phase index with authored/planned labels.

Put organizer tools, evidence editing, API docs, downloads and coverage reports on appropriate pages. Avoid repeating the same routes in multiple large grids, numerical “complete” claims, and generic feature promises. Preserve careers; connect them to real lesson order and projects.

### Pathway / career page

Show outcome, audience, prerequisites, time and hardware first. Then an ordered map of shared foundation → focused lessons → project milestones. Make optional bridges visible without presenting them as mandatory installation. A planned route is discoverable but must not look ready to complete. Remember the selected route in lesson navigation and show its next lesson rather than the next arbitrary phase item.

### Lesson reader

Use a readable central column, route/phase navigation to the left and a short local outline to the right on wide screens. On narrow screens use accessible disclosures, preserve the title and current route, and keep code/diagrams scrollable inside their bounds. The initial viewport should show what the learner will make, prerequisite and first action; auxiliary tool panels must not push teaching far down the page.

Keep one formative checkpoint block and one evidence action. Explain failures with actionable feedback. Distinguish reading, quiz correctness, attempted exercise, self-reported saved evidence and reviewed evidence. Offer notebook/script/source together, but prioritize a beginner workspace on setup. Clear command context beats an ambiguous “Copy command.”

### Project / learner home

Projects need a demo/result, prerequisites, starter download, stages, expected failures, learner-run tests, and separate instructor solution. Keep evidence attached to the project/version/hardware. Learner home should resume one useful next action, show unresolved prerequisites and link saved work. Certificate/readiness states must have explicit evidence criteria; quiz passes alone cannot establish a systems capability.

### Architecture

The original is **static HTML and vanilla JavaScript plus a Node build**, with Markdown lesson sources, generated indexes and dynamic lesson rendering. HTML is not the cause of weak teaching. Our generated JSON/Markdown/notebook approach can support the same coherent experience; a framework migration only helps if it solves a measured routing/rendering/maintenance problem.

Local build outputs must include lesson and project content so a preview does not require unpublished main-branch files. Static JSON APIs and OpenAPI descriptions do not establish parity with server-side content negotiation. Preserve one source of truth and verify browser, notebook, book and CLI outputs against it.

## Implementation order and acceptance criteria

1. **Beginner setup:** downloadable tiny workspace; explicit OS commands, expected results, repair guidance. Confirm a learner can reach a changed output without cloning the repository. macOS/Linux/Windows instructions are distinct; label platforms not actually tested.
2. **One complete foundation module:** arrays → transformations → state → regression project. Replace brief authored lessons with staged builds and matching diagrams before adding advanced titles. Gate quality on prediction, independent check, transfer and diagnosis, not word count.
3. **Lesson reader:** implement required content components and responsive/accessibility behavior around that module; check keyboard use, focus, diagram fallback and narrow code blocks.
4. **Precise career routes:** add lesson-level route manifests and route-preserving navigation, updating the whole contract together. Preserve current roles and add evidence requirements.
5. **Training-system spine:** develop one model across data, optimization, checkpoint, recovery and profiling lessons; then validate on real TPU runtimes with pinned environments.
6. **Focused capstones and learner tools:** add inference, operations, scientific and performance projects with real stages. Expand portfolio/cohort features only when they support those learning flows.

This document is a blueprint, not a claim that these changes are implemented. The next work should demonstrate one complete learner journey before widening the platform.
