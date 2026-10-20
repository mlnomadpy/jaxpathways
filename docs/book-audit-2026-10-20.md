# Book audit, 20 October 2026

Revision audited: `225f724ea4a2e3fbd431c46b7c6c64a15ed65541`, version **0.6.0**.

The book has substantial teaching material and useful numerical checks. Its main weakness is the distance between some advertised outcomes and the evidence learners actually produce. Repair the misleading projects, TPU instructions and book exports before expanding the lesson count again.

This report covers the course as a book and its connected learning system. It inventories **103 lessons in 22 phases, 10 pathways, 7 careers, 29 registered projects, 11 synthesis assessments, 103 primary figures and 29 additional mechanism diagrams**. Ten additional project directories contain planned capstone briefs; only three are in the structured planned-project registry. The [complete inventory](book-audit-2026-10-20.json) records every lesson, phase, route and registered project.

## Method and limits

Every canonical lesson was indexed for teaching sections, experiments, practice, references, code, figures, companion availability and source-bound execution records. Every project manifest and route was inspected for its declared stages and dependencies; reference solutions were compared for duplication. Close reading and targeted code review concentrated on the six TPU lessons, project mismatches, mathematical explanations and exercises in the science/probability/RL/internals branches, representative foundations and optimization material, pre/post-training boundaries, and the new reproducible-experiments workshop. This is comprehensive coverage of the corpus and its structure, with selective technical close review. It is not a sentence-by-sentence proof of all 103 lessons.

The HTML and EPUB were rebuilt and checked structurally. Five saved figures were visually inspected: optimization-12, probability-03, tpu-03, tpu-04 and tpu-06. The other figures were inventoried with their explanations and provenance, not all visually reviewed. No native browser, screen-reader, print pagination or EPUB reading app was reviewed in this audit. No cloud resources were created and no TPU execution was performed. Existing CPU receipts and the completed GitHub CPU smoke step are distinguished from checks freshly executed here.

Review standards: [course-authoring skill](../skills/create-jax-course/SKILL.md), [pedagogy and UX](../skills/create-jax-course/references/pedagogy-and-ux.md), and [content quality](content-quality.md). Historical reports remain snapshots; their old counts are not the baseline for this review.

## What is working

- All 103 lessons have source and expected companion artifacts. All saved primary visual and execution content hashes match current canonical content. There are no missing lesson bodies in the current manifest.
- The 12-lesson optimization sequence contains independent arithmetic and useful counterexamples. Its geometry, conditioning, gradient-noise and optimizer-state references pass freshly executed checks.
- The sampled probability, science, RL and internals explanations distinguish a small worked example from a general claim. Examples include discrete versus continuous solver sensitivity, predictive versus parameter uncertainty, PPO clipping with negative advantages, and derivative-rule contracts.
- The optimization-12 explanation describes optimizer rebounds visible in its actual log-scale plot. Probability-03 explains thinning and the limits of a trace. These are useful standards for repairing weaker figures.
- The image/text/audio/cross-modal harnesses are separate from planned large-model capstones. The README explicitly labels the large-model training and release work as planned. Preserve that distinction.
- The reproducible-experiments workshop connects a question to frozen configuration/data, actual CSV histories, paired runs, failures/retries, plots and a Markdown research notebook. Its four-stage reference suite passes freshly executed checks. Its optional tracker implementations have source and prior evidence, but their integrations were not rerun in this audit.
- The build checks export fidelity, mathematics rendering, IDs, local references, metadata and feedback links. These catch omissions. They do not establish the truth of a plot interpretation or the relevance of a project.

## Priority findings

P1 means repair before presenting the affected work as a dependable learning or qualification path. P2 means improve the teaching or distribution experience after those repairs. These are course-quality priorities, not security severities.

### F01. P1: Nine projects do not assess their advertised subject

**Confirmed.** After removing their introductory docstring and substituting the project ID, these nine `solution/model.py` files are identical:

`welcome-audit`, `arrays-audit`, `transforms-audit`, `recovery-audit`, `performance-audit`, `posttraining-audit`, `tpu-runtime-audit`, `tpu-workflow-audit`, `tpu-systems-audit`.

They all ask for a small normalized encoder, a finite-difference check and a scan training loop. For example, [performance-audit](../projects/performance-audit/project.json) promises compilation boundaries, latency and memory, while its stages require no timing, recompilation measurement, trace or memory evidence. [tpu-runtime-audit](../projects/tpu-runtime-audit/project.json) promises topology, BF16 and collectives, while its [checker](../projects/tpu-runtime-audit/tests/check.py) checks encoder shapes, masked normalization, one derivative and decreasing loss. [recovery-audit](../projects/recovery-audit/project.json) does not require checkpoint continuation. Passing these checks cannot support those titles.

The same checker defaults to `--implementation solution`, although the advertised stage commands omit the implementation flag. A learner following those commands can see a pass without implementing the starter. The stage-1 “deterministic PRNG” claim is tested only through keys and shapes, not repeated seeds or changed-seed behavior. Stage 4 compares endpoint losses and finite evaluation; it does not establish its manifest's claimed monotonic decrease.

**Repair:** remove these from completed phase evidence until each is replaced or explicitly renamed as shared encoder practice. Prefer reusing genuine existing projects where they already cover the capability. Default learner commands to starter or an explicit learner file; instructor checks must explicitly select solution.

**Acceptance:** changing the relevant behavior must fail its own stage. Examples: reset optimizer state on recovery; introduce an extra compilation; omit a collective participant; misstate BF16 storage; provide no profiler artifact. An untouched starter must fail the advertised learner command.

### F02. P1: “Run any course script on TPU” is contradicted by the scripts

**Confirmed by source inspection.** [tpu-05](../phases/20-tpu-workflows/02-run-course-scripts-and-multihost-tpu-slices/lesson.json), section “Run any course lesson script,” selects `distributed-02` and sets `JAX_PLATFORMS=tpu`. The [generated script](../phases/09-distributed/02-a-sharded-training-step/code/main.py) explicitly configures CPU, creates four CPU devices and requests `jax.devices("cpu")`. Other CPU-specific lessons have similar boundaries. Setting an environment variable does not adapt their implementation.

The multi-host command copies `launch.py` but executes only an initialization/device-count one-liner. That verifies discovery, not distributed training, data ownership, a collective, checkpoint recovery or a synchronized model update. The separate copy-back example does not retrieve distinct per-worker run directories. The SSH pipeline also lacks explicit pipeline-failure handling, so `tee` can obscure a failed Python command.

**Repair:** publish a per-script execution matrix: CPU-only, accelerator-compatible single-host, or qualified multi-host. Provide an actual multi-host program with process-local inputs, global-array construction, correct reductions, unique worker outputs and coordinated checkpoint publication. Use reliable remote exit-status handling.

**Acceptance:** a CPU-only lesson is rejected or clearly routed to its CPU mode. Device-only discovery and actual collective/training results get separate receipts. Qualify the latter on a named TPU slice before claiming it works there. JAX's [multi-process guide](https://docs.jax.dev/en/latest/multi_process.html) and [distributed data-loading guide](https://docs.jax.dev/en/latest/distributed_data_loading.html) explain why starting a process on every host is only part of the program.

### F03. P1: A calculated four-chip figure is described as an observed runtime receipt

**Confirmed.** In [tpu-04's canonical visual](../phases/19-tpu/02-bootstrap-venv-and-verify-tpu-runtime-receipt/lesson.json), `visual.code` computes four row sums with NumPy over `range(4)`. The [saved artifact](../phases/19-tpu/02-bootstrap-venv-and-verify-tpu-runtime-receipt/outputs/visual.json) records **one CPU device** and an observed total of **140**. Yet its figure and explanation describe four TPU chips and say the total **10416** proves all four participated.

The arithmetic for the hypothetical four-row input is correct. It is not an observation of four executing chips. A current source hash cannot detect this semantic error.

**Repair:** label this figure “calculated expectation for four devices,” or derive its bars from the actual receipt's `per_device_sums` and actual device count. Keep expected and observed results visibly separate.

**Acceptance:** a one-device reference produces one observed-device bar, or an explicitly hypothetical diagram. A four-device claim must link to a four-device execution record; TPU claims must record backend `tpu`.

### F04. P1: The TPU checkpoint example fails an ordinary resume case and accepts inconsistent saved state

**Reproduced locally.** In [tpu-02](../phases/20-tpu-workflows/01-stage-and-launch-checkpointed-tpu-jobs/lesson.json), `run_tpu_ready_job` defines `state_payload` inside the update loop and accesses it afterward. Run two updates, then resume the completed checkpoint with `steps=2`: it raises `UnboundLocalError`.

A second probe changed saved `w` to `[[100.0]]` while retaining the old `state_hash`. Resuming to step 3 accepted and continued it. Restore prints the saved hash without recomputing it, and the checkpoint does not bind the optimizer/data configuration. A changed learning rate can therefore continue under the same saved identity. `resume=True` with no checkpoint also silently starts a new run.

**Repair:** reuse the stronger recovery contract already taught elsewhere. Define completed-run behavior, verify saved contents and compatibility, reject a missing requested checkpoint, and distinguish continuation from a deliberately changed experiment. Keep the promise about local atomic replacement separate from power-loss durability and distributed checkpoint publication.

**Acceptance:** completed-run resume, corrupted contents, missing checkpoint, incompatible data/optimizer configuration and fresh-process continuation all have explicit, tested outcomes.

### F05. P1: The low-precision lesson mixes storage, arithmetic and hardware claims

**Confirmed code/prose mismatch, with hardware behavior unqualified.** [tpu-06](../phases/21-tpu-systems/02-bfloat16-int8-precision-and-xprof-profiling/lesson.json) labels a policy “int8 per-col (fp32 accum),” but immediately casts INT8 weights back to FP32, applies scales and calls a floating-point dot product. It measures weight-quantization error. It does not demonstrate a native INT8 kernel, low-bit activations, packed runtime storage or reduced device memory traffic.

The prose calls the FP32 result “exact” and says BF16's exponent range prevents underflow/overflow. Both need bounded wording. The figure's mean INT8 error is **0.02322**, while the connection describes approximately **0.01**. The blanket “quarter the weight bytes” omits scale bytes and the dequantized FP32 operand used in this computation.

The BF16 comparison requests different result/accumulation types, but a CPU example does not establish how every partial sum is implemented on a TPU. Its hardware interpretation needs a target-specific check. See [JAX dot semantics](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.dot.html) and Google's [BF16 guide](https://docs.cloud.google.com/tpu/docs/bfloat16).

**Repair:** explicitly separate stored weights, operand types, accumulator/result types, scale metadata and actual kernel execution. Call this an FP32 dequantization reference. Explain measured error against this finite-precision oracle and show hardware evidence separately.

**Acceptance:** the caption and plotted numbers agree; byte counts include metadata; native integer execution is claimed only with an inspected target implementation and observed output/performance evidence. Include a zero-valued weight channel, which currently gives a zero scale in this local example.

### F06. P1: The profiling example discards the artifact learners are supposed to inspect

**Confirmed.** [tpu-06](../phases/21-tpu-systems/02-bfloat16-int8-precision-and-xprof-profiling/lesson.json) captures its worked trace inside `TemporaryDirectory` and returns only its file count and timing. The trace is deleted before a learner can copy or inspect it. A separate shell command attempts to create a persistent trace, but that is a different workload from the executed example and its figure. Counting an XPlane file does not teach trace interpretation.

**Repair:** accept a user-owned trace directory, retain the actual worked workload, and explain an annotated trace: host dispatch, compilation boundary, device work, synchronization and input stalls. Keep CPU and TPU examples explicitly distinct. Report step measurements outside profiler overhead when comparing performance.

**Acceptance:** the printed path exists after execution; it opens in the supported tool; learners answer a concrete question using named events and units from that actual trace. Do not require a TPU merely to understand the CPU trace.

### F07. P1: The required paths and the linear book disagree with their learning goals

**Confirmed.** [models](../learning-paths/models.json) requires **75 lessons**, roughly **98.8 hours of declared lesson time**, for an image-classifier outcome. It includes TPU setup/workflows/systems, Transformers, pretraining, RL and post-training. [tpu](../learning-paths/tpu.json) requires **87 lessons**, roughly **116.4 declared hours**. These are manifest sums, not validated completion estimates.

The phase graph makes all three TPU phases prerequisites for eight advanced phases, including performance and deployment. In the linear HTML/EPUB order, however, those TPU phases occur after the phases that depend on them. The image learner also inherits an RL prerequisite for ordinary supervised/LoRA adaptation through the post-training phase.

**Repair:** define a CPU-completable required core, then explicit image, language, operations and accelerator branches. Separate supervised adaptation from preference/RL prerequisites. Generate the book order from a coherent dependency plan, or make its alternative reading orders explicit.

**Acceptance:** a learner can reach a recoverable exported classifier without provisioning cloud hardware or taking RL. The reader shows the next required lesson and the cost of an optional branch. Every prerequisite is earlier in the selected reading order or clearly introduced as a return visit.

### F08. P1: Saving the HTML book alone loses every figure offline

**Confirmed DOM/source result.** The rebuilt `dist/downloads/jax-foundations.html` is **9,371,752 bytes** and contains **132 images**. All 132 use external site asset paths rather than embedded data. For example, `/figures/welcome-01-mechanism.svg`. Its fonts/CSS are embedded, but its figures are not. A standalone local HTML file cannot resolve those site-root paths.

Source: [HTML book](../src/pages/downloads/jax-foundations.astro), [shared visual renderer](../src/lib/lesson-visuals.js). The EPUB and workspace ZIP package figures separately and pass their structural fidelity checks; this finding concerns the saved HTML experience.

**Repair:** embed the figures in the downloadable HTML, or distribute a clearly labeled HTML-plus-assets archive with relative paths. Keep the web reader as a separate online artifact if useful.

**Acceptance:** move the download into an unrelated folder, disable networking and open it. Every equation, figure and code block must still render. Add asset-resolution checks for the standalone HTML, alongside the existing EPUB checks.

### F09. P1: The full course CI does not finish, and deployment is not gated on it

**Confirmed GitHub results.** [Course checks for this revision](https://github.com/mlnomadpy/jaxpathways/actions/runs/38384112229) ended **cancelled** after approximately 20 minutes, matching the job's timeout budget. Installation, build, all lesson CPU smoke checks, TPU precision reference and math checks passed. Project checks started but were cancelled; optional logger verification and subsequent final checks were skipped. The [Pages deployment](https://github.com/mlnomadpy/jaxpathways/actions/runs/38384112382) succeeded through a separate workflow.

Source: [.github/workflows](../.github/workflows). The Pages workflow does run `npm run check`; that is valuable, but it does not run the complete numerical/project pipeline. A successful deployment is therefore not a completed full-course verification result.

**Repair:** split independent validation jobs, cache dependencies where appropriate and set observed runtime budgets. Gate publication of numerical-content changes on the required verification results. Keep optional hardware qualification separate from ordinary CPU CI.

**Acceptance:** all required jobs for the same commit finish green, with project/logger results visible. A failed or timed-out required gate cannot silently become a published “fully checked” edition.

### F10. P2: The book omits the connective material that makes it a course

**Confirmed omission.** Both book generators concatenate lessons. The HTML contains no phase-guide or project-guide links. It does not include the phase `studyGuide` sequence, pathway decisions, phase project handoffs, careers, practical project-structure guide or reproducible-experiments workshop as book chapters/appendices. Those materials exist elsewhere in the site/workspace, but a reader following this book alone will not encounter their sequence.

Source: [HTML book](../src/pages/downloads/jax-foundations.astro), [EPUB generator](../scripts/build-distribution.py), [project workflow](../content/guides/project-workflow.md), [workshop](../projects/reproducible-experiments/README.md).

**Repair:** add a short “how to use this book,” phase openings and endings, a project evidence checklist, and appendices for experiment tooling. Provide direct links to careers and alternate routes. Keep the main reading sequence focused rather than copying the whole application into every chapter.

**Acceptance:** each phase ends with a specific artifact and a working project handoff. A book-only learner can discover the logging/comparison workshop and the distinction between required and optional work.

### F11. P2: EPUB metadata and reading aids lag behind the HTML

**Confirmed metadata/accessibility differences; reading-app behavior untested.** [build-distribution.py](../scripts/build-distribution.py) hardcodes `dcterms:modified` to **2026-10-03**, uses “JAX Pathways” as every chapter's document title, and emits image titles as alt text rather than the canonical richer descriptions. The HTML visual helper also ignores `visual.alt`; nearby long explanations reduce, but do not eliminate, the lost context.

EPUB code has plain preformatted styling while HTML uses shared code panels. XML, MathML presence, packaged image bytes and internal file references pass. Actual EPUB reader math layout, navigation, long code lines and answer-disclosure behavior remain unverified.

**Repair:** derive edition metadata from the release source, use chapter titles, retain meaningful alternatives and long descriptions, and test one beginner, one equation-heavy and one long-code chapter in a real EPUB reader. Add release/version information to the book cover with a link to edition changes.

**Acceptance:** current edition metadata, correct reading order, usable TOC, readable math/code and figures with a useful text alternative on the named tested readers.

### F12. P2: The solution and code presentation can undercut deliberate practice

**Confirmed template behavior; learner impact needs testing.** [ReaderChapter](../src/components/shared/ReaderChapter.astro) renders build-step code, the complete example, recorded output, exercise-writing guidance based on the solution, and full reference solutions in the main reading flow. The EPUB similarly exposes solutions immediately. Many canonical code comments narrate syntax, for example “Run … to perform the next check or state transition,” rather than explaining a decision or misconception.

**Repair:** show one current implementation increment, its expected result and its independent check. Put assembled code and reference answers behind explicit reveal actions in the web edition; place solutions in a clearly separate book appendix. Keep comments for shapes, invariants, numerical choices and failure boundaries. Do not replace these with new forms or dropdown selectors.

**Acceptance:** learners can attempt an exercise without seeing its answer. A changed-condition task requires a prediction and an explanation, not just copying an asserted value. Test whether the shorter presentation improves completion and explanation quality before rewriting every lesson.

### F13. P1: Small objective labs still do not provide the promised complete model lineage

**Confirmed implementation boundary, not a hidden missing-page claim.** [training-methods](../projects/training-methods/README.md) contains separate objective labs; [weight-conversion](../projects/weight-conversion/README.md) supplies bounded framework parity. They do not constitute one saved Transformer moving through corpus preparation, pretraining, continued training, SFT/LoRA or contrastive adaptation, final evaluation and published model artifacts.

The [ten lifecycle capstones](model-lifecycle-capstones.md), including 300M embedding/causal/image/audio models, generation, CLIP, SigLIP and retrieval, remain specifications. Their directories have no registered `project.json` executable stage contract. The site [planned-project registry](../curriculum/project-plans.json) exposes only CLIP, SigLIP and retrieval as structured project-plan pages; the remaining seven briefs are less integrated into discovery.

**Repair:** finish one complete small real-data lineage first, retaining the same data/tokenizer/model/checkpoint/evaluator identities across stages. A suitable first deliverable is a tiny embedding model progressing from MLM to contrastive learning and held-out retrieval, before a budgeted 300M variant. Follow with distinct causal, image and audio pipelines. Keep plans visibly labeled throughout.

**Acceptance:** a fresh workspace can execute stage handoffs and evaluate reloaded final bytes against a declared baseline. Report actual model size, data exposure, failures and compute. A 300M label requires an actual architecture count and qualified run, not a renamed tiny fixture.

### F14. P2: Career and assessment claims need a capability-to-evidence map

**Confirmed mismatch in current structure.** [Career definitions](../curriculum/course.json) give the training role the TPU route as its default. The research role lists NumPyro among its skills, while its default route and milestone emphasize JAX internals; the authored probability companions implement instructional inference directly rather than an executed NumPyro workflow. Career milestones do not systematically map every advertised skill to assessed evidence.

The [models synthesis assessment](../assessments/models.md) is stronger: it names changed width, independent arithmetic, replay, held-out measurements and the limits of public fixtures. Reuse that specificity. Public solutions and self-reported evidence are appropriate practice aids, but do not establish career readiness or independent review.

**Repair:** for every skill, show the lesson, required learner artifact, review criterion and current qualification boundary. Make TPU practice an explicit specialization where the role does not inherently require it. Label NumPyro and other library extensions honestly or implement their promised workflow.

**Acceptance:** no advertised assessed skill lacks a required artifact and review criterion. Careers explain what a portfolio demonstrates and what still needs external experience or review.

### F15. P2: Capacity and hardware explanations need clearer assumptions

**Confirmed ambiguous/incomplete accounting.** [tpu-03](../phases/21-tpu-systems/01-tpu-generations-and-hbm-memory-budgeting/lesson.json) puts GiB and FLOP/byte bars on one common axis labeled “GiB / (FLOP per byte).” These are different quantities; their heights should not invite comparison.

Its `mixed_bf16_train_state_gib` uses 12 bytes/parameter while the prose describes FP32 master weights, two FP32 Adam moments and gradients. Master weights plus moments already total 12 bytes; gradients and any separate BF16 parameter copy need additional accounting. Which tensors are persistent, transient, replicated or sharded must be explicit. The roofline explanation also treats its idealized bound as a definitive observed bottleneck.

**Repair:** separate memory capacity and ridge point into panels or a table. Show an explicit byte ledger for model weights, masters, moments, gradients, activations, communication buffers, KV cache and temporary copies, with stated exclusions. Use roofline as a hypothesis to investigate with a profile.

**Acceptance:** hand-summed tensor bytes match the estimator, including named exclusions. A target runtime measurement is presented separately from the theoretical budget and published peak specifications.

### F16. P2: Audit/status documentation is itself stale

**Confirmed.** [content-audit.md](content-audit.md) says 28 projects, [README](../README.md) says nineteen registered reference suites, and [model-lifecycle-capstones.md](model-lifecycle-capstones.md) embeds a 19-project snapshot. The registry now has 29. Some documents are clearly dated historical records; others read as current guidance.

**Repair:** generate current counts and status summaries from canonical registries, stamp audit scope/revision, and mark historical snapshots consistently. Separate “authored,” “reference executed,” “hardware qualified” and “independently reviewed.” Do not replace them with a single completion percentage.

**Acceptance:** current navigation/status documentation agrees with the registry; historical reports preserve their date without being presented as current verification.

## Phase-by-phase improvement plan

These are next teaching/review actions, not assertions that every listed extension is essential for every learner. Preserve useful existing material and add depth around an inspectable artifact.

| Phase | Lessons | Retain | Next improvement and useful figure |
| --- | ---: | --- | --- |
| Setup | 4 | Saved-file workflow, explicit interpreter/device distinction | Validate fresh-machine setup with beginners; optional TPU bridge. Diagram terminal → environment → file → output; replace the unrelated encoder audit. |
| Arrays | 4 | Unequal shape cases, independent NumPy checks, fitted preprocessing | Add a variable-length masked batch carried into the next model. Diagram observations/features/mask and reductions with counts. |
| Transforms | 5 | Grad/vmap/jit reasoning and tracing boundaries | Reuse one model across transform composition; show a call ledger with actual tracing/compilation events. Keep timing claims bounded. |
| State | 4 | Explicit key ownership, tree and carry invariants | Carry sample identity, augmentation randomness and optimizer state into recovery. Diagram consumed versus continuation keys across an update. |
| Math & optimization | 12 | Geometry, conditioning, likelihoods, exact variance and optimizer references | Add a phase synthesis that joins split selection, scaling, clipping and optimizer continuation. Use paired runs/error analysis instead of more scalar descent plots. |
| Neural networks | 5 | Explicit and NNX models, train/eval separation, transparent CNN | Add one bounded natural-data pilot and frozen error slices. Show real mistakes, calibration and learning curves for the same saved model. |
| Data & recovery | 6 | Grain/Orbax state and continuation discipline, MLflow | Replace the generic recovery project; connect cursor/config identity to an actual failure drill. Timeline requested → written → published → restored. |
| Transformers | 4 | Attention/masks and tiny causal training | Tokenizer study, complete encoder/decoder configuration and realistic context/data. Trace token IDs, attention visibility, loss mask and cache state through one request. |
| Performance | 5 | Synchronization, actual CPU traces, estimates versus measurements | Replace the encoder audit with compile/steady-state/trace/memory evidence. Annotate an actual trace beside the exact workload and hardware. |
| Distributed training | 4 | Logical CPU meshes, count-weighted reductions and recovery | Actual multi-process data ownership, collective participation and recovery qualification. Show per-process inputs and global-array ownership; CPU logical devices remain a preparation step. |
| Science | 4 | Discretization, discrete sensitivity references, inverse-problem checks | Optional adaptive/stiff solver and noisy-observation study. Couple trajectory, residual and step-size/sensitivity panels; label model bias separately. |
| Probability | 4 | Exact posterior, hand-built HMC/VI and predictive distinctions | Add an executed library workflow if NumPyro remains a career promise. Use multi-panel trace/rank/ESS diagnostics and a deliberately failing predictive check. |
| RL | 4 | Functional environment, masking, sampled updates and exact evaluator | Extend to learned values, GAE and timeout bootstrapping before larger tasks. Pair transition/termination diagrams with seed distributions; preserve the tiny exact oracle. |
| Pallas | 4 | Ownership, interpreted semantics and separate target paths | Real target compilation and measured benefit. Show tile ownership and buffer lifetime beside a captured target trace, with output error and end-to-end timing. |
| Internals | 4 | Typed programs, directional products, custom rules and supported-subset interpreter | Expand transformation-composition failures and extension boundaries. Trace primal/tangent/cotangent values on one shared graph. |
| Deployment | 8 | Bounded framework mapping, serialization and precision contracts | A complete architecture conversion, real concurrent service and one target edge qualification. Connect first-diverging-layer errors to request preprocessing and actual exported bytes. |
| Operations | 8 | Actual local lifecycle, release gates, lineage and rollback drills | Load-driven latency, cancellation/backpressure, delayed labels and incident/SLO practice. Plot request timelines with queue wait, service, timeout and recovery boundaries. |
| Pretraining | 3 | MLM/MIM/contrastive objective and leakage reasoning | Connect one full saved encoder to data recipe, phase handoff and downstream evaluation. Diagram visible inputs versus loss-selected targets and candidate ownership. |
| Post-training | 5 | Response mask, trained dense LoRA, reward/PPO/DPO mechanics | Separate supervised and preference prerequisites; adapt a real saved small Transformer. Compare target task, retained behavior and reference drift after each phase. |
| TPU runtime | 2 | Runtime discovery and explicit observed arithmetic | Correct hypothetical/observed figure distinction and replace generic audit with topology/collective evidence. Display expected versus observed device participation. |
| TPU workflows | 2 | Staging and checkpoint intent | Repair F02/F04 and qualify a real multi-host job. Use a worker/cursor/checkpoint timeline with retained per-worker logs. |
| TPU systems | 2 | Memory budgeting and precision/profile questions | Repair F05/F06/F15; make assumptions explicit and retain trace artifacts. Separate capacity, theoretical roofline and measured workload panels. |

## Recommended sequence of work

| Order | Deliverable | Completion evidence |
| --- | --- | --- |
| 1 | Correct existing claims and broken examples: F01–F06, F15 | Relevant learner commands fail on untouched starters; negative recovery cases pass; plot descriptions match actual runs; traces remain inspectable. |
| 2 | Finish the verification pipeline: F09 | Same-commit full CPU/project/logger CI completes; deployment depends on required gates. |
| 3 | Make the book coherent and portable: F07–F12 | CPU learning order, phase/project handoffs, offline figures, edition metadata and a tested math/code reading sample. |
| 4 | Deliver one realistic complete model lineage: F13 | Frozen real-data recipe, stage checkpoints, objective transitions, independent evaluation, exported reload and a documented failed experiment. |
| 5 | Align roles, projects and review: F14/F16 | Every promised assessed skill maps to evidence; generated status is current; small learner pilots inform the next writing revision. |
| 6 | Extend modalities and qualify accelerators | Distinct causal/image/audio/CLIP/SigLIP/retrieval implementations and named hardware receipts. Scale follows correctness and a measured pilot. |

For the first learner pilot, follow one CPU foundations → classifier → recovery → experiments → exported-inference journey. Observe where the learner needs help, which prediction they make, whether they can explain a figure, and whether they can reproduce and diagnose their own result. Then pilot a separate advanced TPU journey after its repairs. Record reading, coding and waiting time separately before revising duration estimates.

## Verification record

| Check | Result and scope |
| --- | --- |
| Fresh `npm run check` | Passed: Astro/types/lint/format/curriculum/build, 97 automated tests, distribution fidelity, 187 generated pages, 10,215 local references, 22 phase guides and SEO checks. Structural accessibility checks passed; this is not an assistive-technology review. |
| Fresh content inventory | 1 setup walkthrough and 102 lessons with required teaching structure; 103 current CPU receipts, of which 5 use logical multi-device CPU runtimes. No stale content hashes or missing expected artifacts found. Structure is not an editorial score. |
| Fresh `npm run test:math` | Passed geometry, curvature, ridge, exact sampling covariance and two-step Adam references. |
| Fresh `npm run test:tpu-precision` | Passed independent integer/rounded-float references, zero-scale/shape cases, synchronized CPU measurements, real trace retention and failure checks in the separate TPU utility. This does not repair tpu-06's local example. |
| Fresh `npm run test:experiments` | All four stages passed, including real Flax training, paired analysis, retry/replay, reload, timeout and corrupt-evidence checks. |
| Targeted tpu-02 negative probes | Completed-checkpoint resume reproduced `UnboundLocalError`; saved weight alteration with an unchanged hash was accepted. These are failing learning contracts, despite the reference happy path passing. |
| Latest GitHub course run | CPU lesson smoke succeeded; complete course job timed out during project checks. Full 29-project suite and optional logger integration were not freshly rerun in this audit. |
| Book inspection | HTML image dependencies counted; EPUB XML/assets/reference checks passed. Five saved plots inspected visually. Native reading app, browser, print and screen-reader behavior remain to be tested. |

The audit creates this report and its inventory only. It does not change, publish or certify the course material.
