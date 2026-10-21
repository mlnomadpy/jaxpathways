# Learning-material audit: 21 October 2026

Complete source/companion/receipt inventory and per-item scope review; exercises and transfer prompts inspected across all phases. Mathematical/code close review and visual inspection are selective, not a proof of every sentence or a novice/expert acceptance.

Base revision: `7554817ed19a26dabfb1e183220cf84308382906`. This is an internal editorial audit, not independent teaching approval.

## Current assessment

The book is a substantial collection of bounded, runnable practice drafts. It is not yet a fully reviewed curriculum or a set of production-model releases. The foundational mechanisms and numerical contracts are stronger than the broader model-lifecycle, target deployment and career-readiness evidence.

104 current CPU lesson references; 31 source-matching public project receipts; 4 bounded physical target qualifications. 0 independent expert reviews and 0 novice pilots are recorded.

The inventory covers 203 material entries, 409 lesson exercises/checkpoints and 127 project stages. Section counts describe coverage, not quality scores. Duration estimates have not been calibrated with learners.

## Learner labels

- **Available Draft**: Material is available for its stated practice scope. Teaching review and learner validation are still pending.
- **In Development**: Useful material exists, but important parts of this course or project are still being developed. Read its stated scope before starting.
- **Planned**: A specification or brief exists; no runnable capstone implementation is supplied.
- **Needs Repair**: Required material is missing or its current checks fail. Do not rely on it until repaired.

CPU checks, hardware scope, human review and learner completion are separate. Nothing is promoted to reviewed merely because files or a passing reference exist.

## Priority contribution work

### P1: No independent teaching reviews or novice pilots are recorded

Status: **open**.

Recruit consented novices and independent specialists; bind accepted reports to the exact content hash. Automated execution and this internal audit do not count as human acceptance.

Evidence: [curriculum/teaching-reviews.json](../curriculum/teaching-reviews.json), [docs/teaching-review-protocol.md](../docs/teaching-review-protocol.md).

### P1: All ten larger model and retrieval capstones are specifications

Status: **open**.

Implement one licensed-data, baseline-compared model lineage at a time. Existing tiny pilots and harnesses are preparation, not 300M releases.

Evidence: [curriculum/project-plans.json](../curriculum/project-plans.json), [projects/embedding-lineage/README.md](../projects/embedding-lineage/README.md), [projects/causal-lineage/README.md](../projects/causal-lineage/README.md).

### P1: Hardware evidence is real but narrow

Status: **open**.

Retain the four source-matching v5e receipts and their limits. Add fresh-process/worker-loss recovery, target inference, native quantization and scaling evidence separately.

Evidence: [curriculum/hardware-validation.json](../curriculum/hardware-validation.json).

### P2: Choice-specific checkpoint reasoning is implemented; learner validation remains open

Status: **partly-addressed**.

Review the three rationales in every lesson and pilot changed-condition tasks. Author revision is not independent teaching acceptance.

Evidence: [src/scripts/reader.js](../src/scripts/reader.js), [phases](../phases).

### P2: Learner-only contracts are authored; independent assessment calibration remains open

Status: **partly-addressed**.

Review all 305 authored task contracts and pilot unseen transfer cases. Learner-only downloads omit reference solutions and checkpoint answers; this author revision does not establish independent learner success or calibrated grading.

Evidence: [src/lib/code-teaching.js](../src/lib/code-teaching.js), [scripts/lesson-artifacts.mjs](../scripts/lesson-artifacts.mjs), [tests/learner-starters.test.mjs](../tests/learner-starters.test.mjs), [docs/review-guide-0.11.md](../docs/review-guide-0.11.md).

### P2: Figure freshness is checked but interpretation still needs human review

Status: **partly addressed**.

Corrected the two recovery captions: equal leaf norms do not imply elementwise equality, and continuation depends on complete state, inputs, configuration and environment. Human review of arrows, axes, units and causal explanations is still needed.

Evidence: [phases/06-recovery/03-save-model-optimizer-and-random-state/lesson.json](../phases/06-recovery/03-save-model-optimizer-and-random-state/lesson.json), [phases/06-recovery/04-recover-data-position-and-resume/lesson.json](../phases/06-recovery/04-recover-data-position-and-resume/lesson.json), [docs/teaching-review-protocol.md](../docs/teaching-review-protocol.md).

### P2: Career routes and duration estimates are not externally calibrated

Status: **open**.

Ask working practitioners to review role coverage and pilot actual learner journeys. Portfolio fixture passes do not establish job readiness.

Evidence: [curriculum/course.json](../curriculum/course.json), [learning-paths](../learning-paths), [curriculum/teaching-reviews.json](../curriculum/teaching-reviews.json).

### P2: Availability, completion and hardware review were mixed in learner-facing wording

Status: **addressed**.

Added learner-facing maturity, current CPU checks, scoped hardware evidence and teaching-review status. Replaced stale blanket TPU staging instructions. Keep these labels source-bound as the material changes.

Evidence: [src/pages/projects.astro](../src/pages/projects.astro), [src/pages/book.astro](../src/pages/book.astro), [src/pages/downloads/jax-foundations.astro](../src/pages/downloads/jax-foundations.astro), [curriculum/course.json](../curriculum/course.json).

## Every course phase

| Material | Maturity | Current evidence and useful scope | Next contribution |
| --- | --- | --- | --- |
| [welcome: Setup & first steps](https://www.tahabouhsine.com/jaxpathways/phase-guides/welcome.html) | available-draft | 4/4 current CPU references. A concrete saved-file and environment walkthrough, plus CPU/device checks. | Run consented fresh-machine novice pilots on Windows, macOS and Linux; record where learners need help. |
| [arrays: Arrays & pure functions](https://www.tahabouhsine.com/jaxpathways/phase-guides/arrays.html) | available-draft | 4/4 current CPU references. Dense-array axes, preprocessing and update contracts have changed-input checks. | Add a masked variable-length batch that continues into the model project; review constant-feature and ragged-input explanations. |
| [transforms: JAX transformations](https://www.tahabouhsine.com/jaxpathways/phase-guides/transforms.html) | available-draft | 5/5 current CPU references. Gradients, batching, compilation and tracing have independent arithmetic and changed conditions. | Review transformation composition at nonsmooth boundaries and add a single continuing model across the five lessons. |
| [state: State, randomness & control flow](https://www.tahabouhsine.com/jaxpathways/phase-guides/state.html) | available-draft | 4/4 current CPU references. Key ownership, parameter trees, scan carries and branches have explicit invariants. | Connect sample IDs, augmentation keys and optimizer state to a recovery handoff; test that novices can explain key reuse. |
| [optimization: Math & optimization](https://www.tahabouhsine.com/jaxpathways/phase-guides/optimization.html) | available-draft | 12/12 current CPU references. Twelve connected math labs and an optimizer audit cover geometry, conditioning, noise and update state. | Pilot the synthesis with learners; add constrained or nonsmooth optimization only around a new inspectable task, and improve feedback on wrong answers. |
| [networks: Neural network training](https://www.tahabouhsine.com/jaxpathways/phase-guides/networks.html) | available-draft | 5/5 current CPU references. MLP, NNX and CNN examples test gradients, evaluation isolation and synthetic image errors. | Add a licensed natural-image pilot with frozen source-group splits, a simple baseline and per-class error slices. |
| [recovery: Data & checkpoint recovery](https://www.tahabouhsine.com/jaxpathways/phase-guides/recovery.html) | available-draft | 6/6 current CPU references. Grain/Orbax and the recovery audit exercise complete state and input-position contracts. | Review exact-continuation wording against its fixed environment; inject storage and concurrent-writer failures separately from ordinary local restore. |
| [transformers: Transformers & language models](https://www.tahabouhsine.com/jaxpathways/phase-guides/transformers.html) | in-development | 4/4 current CPU references. Attention and packing lessons connect to a separate configurable multi-layer decoder with train-only BPE, a measured natural-text CPU pilot, cache parity and fresh-process full-state recovery. | Review tokenizer and length policies on broader licensed corpora, extend the measured short pilot into staged training and evaluate generation quality. Target-scale training and independent teaching review remain open. |
| [performance: Performance diagnosis](https://www.tahabouhsine.com/jaxpathways/phase-guides/performance.html) | available-draft | 5/5 current CPU references. Warm timing, call ledgers, retained CPU traces and independent memory estimates are available. | Annotate a real target trace with the exact workload and bottleneck; retain distributions and distinguish estimates from measured allocated memory. |
| [distributed: Distributed training](https://www.tahabouhsine.com/jaxpathways/phase-guides/distributed.html) | in-development | 4/4 current CPU references. Logical CPU sharding is checked; separate v5e collectives and same-process multi-host restore receipts exist. | Qualify fresh-process multi-host restart and worker-loss recovery with explicit data ownership; do not infer scaling results from logical CPU devices. |
| [science: Scientific computing](https://www.tahabouhsine.com/jaxpathways/phase-guides/science.html) | in-development | 4/4 current CPU references. Fixed-grid ODE references connect to a separately pinned adaptive Diffrax extension with stiff solves, tolerance-sensitive gradients, noisy inversion and identifiable versus insensitive profiles. | Review the solver and inference assumptions with a domain specialist, extend the synthetic fixture to real measurements and validate uncertainty or model misspecification before broader scientific claims. |
| [probability: Probabilistic modeling](https://www.tahabouhsine.com/jaxpathways/phase-guides/probability.html) | in-development | 4/4 current CPU references. Exact Bayesian regression, instructional HMC/VI and an optional real NumPyro workflow with raw chains, modern diagnostics and paired model-mismatch checks. | Review the probabilistic assumptions with a specialist and extend the synthetic model checks to a licensed real-data case with calibrated predictive evaluation. |
| [rl: Reinforcement learning](https://www.tahabouhsine.com/jaxpathways/phase-guides/rl.html) | in-development | 5/5 current CPU references. Tabular policy objectives, paired-seed evaluation, a separately learned value table and explicit GAE terminal/cutoff/reset checks have bounded CPU references. | Connect the learned critic and GAE to actor updates, then evaluate larger tasks or continuous actions with independent learner and expert review. |
| [kernels: Pallas kernels](https://www.tahabouhsine.com/jaxpathways/phase-guides/kernels.html) | in-development | 4/4 current CPU references. CPU interpretation and a separately qualified two-kernel v5e target are available. | Turn the bounded target receipt into a learner tuning lab with retained trace interpretation and end-to-end comparison; GPU and general pipeline claims remain unqualified. |
| [internals: Autodiff & JAX internals](https://www.tahabouhsine.com/jaxpathways/phase-guides/internals.html) | available-draft | 4/4 current CPU references. Jaxpr, directional products, custom rules and a supported-subset interpreter have independent derivative checks. | Add transformation-composition failures and effect/extension boundaries; have an independent reviewer check the supported subset and numerical assumptions. |
| [deployment: Deployment, interoperability & edge AI](https://www.tahabouhsine.com/jaxpathways/phase-guides/deployment.html) | in-development | 8/8 current CPU references. Dense framework conversion, exports, precision contracts and process boundaries are checked. | Execute and document a real container service, concurrent request workload and named edge-device/delegate report; dense parity is not full architecture portability. |
| [operations: Workload operations](https://www.tahabouhsine.com/jaxpathways/phase-guides/operations.html) | in-development | 8/8 current CPU references. Local job incidents, checksummed recovery and release/rollback drills are runnable. | Add authenticated approvals, multi-writer release control, load-driven SLO incidents and application-specific LLM evaluation beyond deterministic fixtures. |
| [pretraining: Self-supervised pretraining: masked and contrastive learning](https://www.tahabouhsine.com/jaxpathways/phase-guides/pretraining.html) | in-development | 3/3 current CPU references. Small masked-token, masked-image and contrastive objective labs run on CPU. | Integrate the objectives with saved encoder architectures, licensed data recipes and downstream probes; the larger modality capstones are still plans. |
| [posttraining: Post-training: SFT, LoRA, reward models and RLHF](https://www.tahabouhsine.com/jaxpathways/phase-guides/posttraining.html) | in-development | 5/5 current CPU references. Response masks, dense LoRA, learned rewards, PPO-bandit and DPO mechanics are implemented. | Adapt a real saved Transformer and compare target quality, retained behavior and reference drift; sequence PPO/value models and human preferences are not supplied. |
| [tpu: TPU Setup: Provisioning & Runtime Verification](https://www.tahabouhsine.com/jaxpathways/phase-guides/tpu.html) | in-development | 2/2 current CPU references. Provisioning guidance and runtime verification connect to separate bounded physical v5e evidence. | Review generation-specific topology and memory assumptions, test a learner provisioning journey and keep pricing illustrative rather than a live quote. |
| [tpu-workflows: TPU Workflows: Launching Jobs & Checkpointed Experiments](https://www.tahabouhsine.com/jaxpathways/phase-guides/tpu-workflows.html) | in-development | 2/2 current CPU references. Reviewed launch targets, checkpoint contracts and a bounded multi-host receipt exist. | Qualify fresh-process and worker-loss recovery, retained per-worker evidence and cleanup in an observed learner run; device discovery alone is not training qualification. |
| [tpu-systems: TPU Systems: Generations, Memory, Precision & XProf](https://www.tahabouhsine.com/jaxpathways/phase-guides/tpu-systems.html) | in-development | 2/2 current CPU references. Memory/precision models and retained profiling paths connect to narrow v5e targets. | Teach target trace interpretation and native low-bit kernel/storage evidence; distinguish per-chip capacity, sharded state and dequantized reference arithmetic. |

## Every runnable project

| Material | Maturity | Current evidence and useful scope | Next contribution |
| --- | --- | --- | --- |
| [regression-audit: Build a regression training audit](https://www.tahabouhsine.com/jaxpathways/project-guides/regression-audit.html) | available-draft | CPU reference checker passed for current sources. Independent regression gradients and changed-condition training. | Pilot the rubric with unseen data and retain a reviewer-scored evidence package. |
| [mlp-classifier: Build a reproducible classifier audit](https://www.tahabouhsine.com/jaxpathways/project-guides/mlp-classifier.html) | available-draft | CPU reference checker passed for current sources. Small synthetic classifier with objective, gradient and replay checks. | Bridge to a licensed real dataset with frozen splits, error analysis and reviewer feedback. |
| [scientific-inverse: Audit a differentiable scientific inverse problem](https://www.tahabouhsine.com/jaxpathways/project-guides/scientific-inverse.html) | available-draft | CPU reference checker passed for current sources. Fixed-grid simulation plus an optional executed Diffrax workflow with adaptive/stiff solves, sensitivity refinement, same-data noisy fitting and explained identifiability figures. | Add licensed real measurements, an explicit uncertainty or discrepancy model and domain-expert review. Existing evidence is synthetic CPU computation. |
| [policy-evaluation: Train and audit a policy from interaction](https://www.tahabouhsine.com/jaxpathways/project-guides/policy-evaluation.html) | available-draft | CPU reference checker passed for current sources. Tabular environment, policy objectives, held-out evaluation and a separately checked learned value table with GAE. | Integrate the critic/GAE with actor updates and evaluate a larger task with a fixed held-out protocol and learner review. |
| [bayesian-regression: Audit a Bayesian regression model](https://www.tahabouhsine.com/jaxpathways/project-guides/bayesian-regression.html) | available-draft | CPU reference checker passed for current sources. Exact regression oracle and an optional, separately pinned NumPyro NUTS workflow with raw-chain diagnostics and paired predictive checks. | Validate assumptions, diagnostics and model mismatch interpretation with specialist and learner review, then extend to a real-data example. |
| [derivative-audit: Audit derivatives and compile a tiny transformation](https://www.tahabouhsine.com/jaxpathways/project-guides/derivative-audit.html) | available-draft | CPU reference checker passed for current sources. Directional products, custom rules and a small primitive transformation. | Review higher-order composition and unsupported-primitive failure cases. |
| [deployment-audit: Release a trained computation with a deployment audit](https://www.tahabouhsine.com/jaxpathways/project-guides/deployment-audit.html) | available-draft | CPU reference checker passed for current sources. Small artifact adaptation, precision and request capacity contracts. | Attach a real container run, load test and target runtime report. |
| [workload-operations: Operate and recover a local JAX workload](https://www.tahabouhsine.com/jaxpathways/project-guides/workload-operations.html) | available-draft | CPU reference checker passed for current sources. Real local process incidents, complete recovery and rollback. | Extend from local drills to authenticated, concurrent cloud release and SLO failure cases. |
| [audio-harness: Build an audio waveform-to-inference harness](https://www.tahabouhsine.com/jaxpathways/project-guides/audio-harness.html) | available-draft | CPU reference checker passed for current sources. Synthetic waveform classification with recovery, precision and export. | Add licensed recordings, speaker/source-held-out splits and task-appropriate baselines; this is not an ASR model. |
| [cross-modal-harness: Build and audit image–text retrieval](https://www.tahabouhsine.com/jaxpathways/project-guides/cross-modal-harness.html) | available-draft | CPU reference checker passed for current sources. Small paired-branch contrastive system with recovery and export. | Add natural paired data, duplicate-positive handling and held-out multimodal retrieval. |
| [kernel-audit: Audit a Pallas kernel and its target evidence](https://www.tahabouhsine.com/jaxpathways/project-guides/kernel-audit.html) | available-draft | CPU reference checker passed for current sources. Interpreted semantics plus two separately observed v5e target kernels. | Reproduce the bounded target on new shapes and interpret the trace before claiming a useful speedup. |
| [image-harness: Build a recoverable image classifier through exported inference](https://www.tahabouhsine.com/jaxpathways/project-guides/image-harness.html) | available-draft | CPU reference checker passed for current sources. Synthetic CNN lifecycle with augmentation, recovery and frozen errors. | Add real photos, source-group leakage checks and an observed edge/runtime report. |
| [foundation-toolkit: Connect arrays, transformations and explicit state](https://www.tahabouhsine.com/jaxpathways/project-guides/foundation-toolkit.html) | available-draft | CPU reference checker passed for current sources. Environment, masked arrays, gradient batches and replayed simulation. | Test the end-to-end beginner handoff without revealing reference answers. |
| [sharded-training: Resume and profile a sharded training run](https://www.tahabouhsine.com/jaxpathways/project-guides/sharded-training.html) | available-draft | CPU reference checker passed for current sources. Logical CPU ownership, weighted updates, recovery and trace evidence. | Qualify fresh-process multi-host data and recovery behavior on an explicitly named target. |
| [text-harness: Build a causal Transformer training and inference harness](https://www.tahabouhsine.com/jaxpathways/project-guides/text-harness.html) | available-draft | CPU reference checker passed for current sources. Tiny causal training, recovery, KV cache and export, plus a separate train-only BPE decoder workflow with natural chapter splits, measured short-run results and an explained baseline comparison. | Broaden source and tokenizer studies, complete larger staged training and assess generation on genuinely unseen material. Repeatedly inspected Alice chapters are not a blind benchmark. |
| [optimizer-audit: Audit conditioning, gradient noise and optimizer decisions](https://www.tahabouhsine.com/jaxpathways/project-guides/optimizer-audit.html) | available-draft | CPU reference checker passed for current sources. Geometry, conditioning, enumerated gradient noise and optimizer-state evidence. | Pilot the synthesis with learner implementations and review failure explanations. |
| [engineering-release: Ship a tracked and containerized model release](https://www.tahabouhsine.com/jaxpathways/project-guides/engineering-release.html) | available-draft | CPU reference checker passed for current sources. Tracked training, slice gates and deterministic LLM application fixtures. | Evaluate a real application with versioned prompts/retrieval, adversarial cases and human review. |
| [training-methods: Audit pretraining and post-training objectives](https://www.tahabouhsine.com/jaxpathways/project-guides/training-methods.html) | available-draft | CPU reference checker passed for current sources. Eight small masked, contrastive, adaptation and preference objectives. | Connect each objective to real saved architectures and phase-to-phase checkpoint lineage. |
| [weight-conversion: Convert PyTorch weights and audit Flax architecture parity](https://www.tahabouhsine.com/jaxpathways/project-guides/weight-conversion.html) | available-draft | CPU reference checker passed for current sources. Real framework dense mappings and first-divergence numerical checks. | Convert a complete saved attention architecture, including normalization, masks and input gradients. |
| [welcome-audit: Check the CPU environment and a saved matrix arithmetic identity](https://www.tahabouhsine.com/jaxpathways/project-guides/welcome-audit.html) | available-draft | CPU reference checker passed for current sources. Environment evidence and independent arithmetic with learner-owned implementation. | Run an unaided novice walkthrough and review the retained environment record. |
| [arrays-audit: Audit masked batch moments and padding-sensitive denominators](https://www.tahabouhsine.com/jaxpathways/project-guides/arrays-audit.html) | available-draft | CPU reference checker passed for current sources. Array shape, statistics and changed-input diagnostics. | Add ragged/masked data and assess the learner explanation of each axis. |
| [transforms-audit: Compare scalar, vmap, autodiff, and changed-shape JIT behavior](https://www.tahabouhsine.com/jaxpathways/project-guides/transforms-audit.html) | available-draft | CPU reference checker passed for current sources. Differentiation, batching and tracing invariants. | Add composed transformations with a deliberately incorrect derivative boundary. |
| [recovery-audit: Resume optimizer and PRNG state across an interrupted training update](https://www.tahabouhsine.com/jaxpathways/project-guides/recovery-audit.html) | available-draft | CPU reference checker passed for current sources. Local checkpoint contract and independent continuation checks. | Inject incomplete writes, stale manifests and incompatible input state in separate processes. |
| [performance-audit: Measure CPU JIT latency and changed-shape traces with retained JAXPR evidence](https://www.tahabouhsine.com/jaxpathways/project-guides/performance-audit.html) | available-draft | CPU reference checker passed for current sources. Compilation/timing distinctions and retained evidence. | Add a target-backed workload and a reviewed trace, not a generic timing improvement claim. |
| [posttraining-audit: Audit response-token SFT masking and preference-pair loss](https://www.tahabouhsine.com/jaxpathways/project-guides/posttraining-audit.html) | available-draft | CPU reference checker passed for current sources. Bounded adaptation and preference-contract checks. | Transfer the contracts to a saved Transformer and document retained behavior. |
| [tpu-runtime-audit: Inspect CPU device inventory, BF16 storage, and collective participant arithmetic](https://www.tahabouhsine.com/jaxpathways/project-guides/tpu-runtime-audit.html) | available-draft | CPU reference checker passed for current sources. CPU-verifiable runtime and participation evidence contracts. | Observe a learner-owned TPU receipt on a declared topology and keep the exact numerical reference. |
| [tpu-workflow-audit: Gate simulated multi-worker checkpoint publication on complete receipts](https://www.tahabouhsine.com/jaxpathways/project-guides/tpu-workflow-audit.html) | available-draft | CPU reference checker passed for current sources. Launch, checkpoint and recovery evidence contracts. | Add a fresh-process multi-host restart and failure-injection qualification. |
| [tpu-systems-audit: Estimate optimizer-state bytes and calculate supplied-rate scaling efficiency](https://www.tahabouhsine.com/jaxpathways/project-guides/tpu-systems-audit.html) | available-draft | CPU reference checker passed for current sources. Precision/memory/profiling contracts with bounded assumptions. | Attach actual native low-bit execution and an interpreted target trace. |
| [reproducible-experiments: Reproducible JAX/Flax experiments workshop](https://www.tahabouhsine.com/jaxpathways/project-guides/reproducible-experiments.html) | available-draft | CPU reference checker passed for current sources. Six recorded CPU runs, paired comparisons, logging and a research notebook. | Pilot learner-owned hypotheses, failed/retried runs and the optional logger adapters. |
| [embedding-lineage: Trace a tiny bidirectional encoder from pretraining to retrieval](https://www.tahabouhsine.com/jaxpathways/project-guides/embedding-lineage.html) | in-development | CPU reference checker passed for current sources. A real-text CPU pilot with MLM, contrastive continuation, adaptation and held-out retrieval; quality is weak. | Improve the frozen data/evaluation recipe against BM25 before scaling or claiming a useful released embedding model. |
| [causal-lineage: Trace a tiny causal Transformer from pretraining to instruction-style adaptation](https://www.tahabouhsine.com/jaxpathways/project-guides/causal-lineage.html) | in-development | CPU reference checker passed for current sources. A real-text CPU Transformer pilot with continuation, held-out perplexity and export; quality is weak. | Improve baseline-relative held-out quality and tokenizer/data decisions before scaling or publishing a production model. |

## Every planned capstone

| Material | Maturity | Current evidence and useful scope | Next contribution |
| --- | --- | --- | --- |
| [clip-model: Train a CLIP image–text model](https://www.tahabouhsine.com/jaxpathways/project-plans/clip-model.html) | planned | Specification only; no runnable capstone. Learn paired representations, test zero-shot classification, and qualify both retrieval directions before releasing the encoders. | Implement the scoped data recipe, architecture, training stages, independent evaluation and exported reload before attempting the target scale. |
| [siglip-model: Train and compare a SigLIP model](https://www.tahabouhsine.com/jaxpathways/project-plans/siglip-model.html) | planned | Specification only; no runnable capstone. Replace the softmax objective with pairwise sigmoid training. Audit labels, reduction, negatives and distributed gradients against a controlled CLIP baseline. | Implement the scoped data recipe, architecture, training stages, independent evaluation and exported reload before attempting the target scale. |
| [retrieval-system: Build a measured retrieval service](https://www.tahabouhsine.com/jaxpathways/project-plans/retrieval-system.html) | planned | Specification only; no runnable capstone. Take a qualified text or cross-modal encoder through exact search, approximate indexes, hybrid retrieval, reranking, evaluation and safe index migration. | Implement the scoped data recipe, architecture, training stages, independent evaluation and exported reload before attempting the target scale. |
| [embedding-model-300m: Build a staged embedding model](https://www.tahabouhsine.com/jaxpathways/project-plans/embedding-model-300m.html) | planned | Specification only; no runnable capstone. MLM to contrastive pretraining and retrieval fine-tuning, with frozen data, MTEB evaluation and release gates. | Implement the scoped data recipe, architecture, training stages, independent evaluation and exported reload before attempting the target scale. |
| [causal-model-300m: Build a causal language model](https://www.tahabouhsine.com/jaxpathways/project-plans/causal-model-300m.html) | planned | Specification only; no runnable capstone. Connect tokenizer/data preparation, next-token pretraining, adaptation, generation evaluation and model publication. | Implement the scoped data recipe, architecture, training stages, independent evaluation and exported reload before attempting the target scale. |
| [image-model-300m: Build an image understanding model](https://www.tahabouhsine.com/jaxpathways/project-plans/image-model-300m.html) | planned | Specification only; no runnable capstone. Carry a frozen image recipe through representation learning, task adaptation and measured evaluation. | Implement the scoped data recipe, architecture, training stages, independent evaluation and exported reload before attempting the target scale. |
| [image-generation-model: Build an image generation pipeline](https://www.tahabouhsine.com/jaxpathways/project-plans/image-generation-model.html) | planned | Specification only; no runnable capstone. Qualify image representations, denoising training, conditioning and sampling with separate quality and runtime evidence. | Implement the scoped data recipe, architecture, training stages, independent evaluation and exported reload before attempting the target scale. |
| [audio-model-300m: Build an audio understanding model](https://www.tahabouhsine.com/jaxpathways/project-plans/audio-model-300m.html) | planned | Specification only; no runnable capstone. Version waveform and acoustic representations through training, recognition or event adaptation and robustness evaluation. | Implement the scoped data recipe, architecture, training stages, independent evaluation and exported reload before attempting the target scale. |
| [audio-generation-model: Build an audio generation pipeline](https://www.tahabouhsine.com/jaxpathways/project-plans/audio-generation-model.html) | planned | Specification only; no runnable capstone. Connect codec qualification, sequence or denoising training, conditioning and waveform evaluation. | Implement the scoped data recipe, architecture, training stages, independent evaluation and exported reload before attempting the target scale. |
| [multimodal-model: Build a grounded multimodal model](https://www.tahabouhsine.com/jaxpathways/project-plans/multimodal-model.html) | planned | Specification only; no runnable capstone. Qualify encoders, connector alignment, instruction adaptation and grounded evaluation, keeping retrieval a distinct outcome. | Implement the scoped data recipe, architecture, training stages, independent evaluation and exported reload before attempting the target scale. |

## Every synthesis assessment

| Material | Maturity | Current evidence and useful scope | Next contribution |
| --- | --- | --- | --- |
| [foundations: Regression synthesis](https://www.tahabouhsine.com/jaxpathways/assessments/foundations.html) | available-draft | Written tasks, public scaffolds and rubric; no calibrated grading study. Defend a JAX regression experiment with independent gradient checks, reproducible training, learning-rate analysis and a repair for a broadcasting bug. | Trial the tasks with unseen learner implementations and record rubric disagreements, changed cases and reviewer reasoning. |
| [models: Classifier and image lifecycle synthesis](https://www.tahabouhsine.com/jaxpathways/assessments/models.html) | available-draft | Written tasks, public scaffolds and rubric; no calibrated grading study. Assess a JAX image classifier through stable losses, preprocessing, checkpoint recovery, precision tradeoffs and verified exported inference. | Trial the tasks with unseen learner implementations and record rubric disagreements, changed cases and reviewer reasoning. |
| [math: Math & optimization synthesis](https://www.tahabouhsine.com/jaxpathways/assessments/math.html) | available-draft | Written tasks, public scaffolds and rubric; no calibrated grading study. Explain JAX optimizer behavior using least-squares geometry, Hessian eigenvalues, minibatch gradient noise, clipping and independent Adam update checks. | Trial the tasks with unseen learner implementations and record rubric disagreements, changed cases and reviewer reasoning. |
| [science: Scientific inverse-problem synthesis](https://www.tahabouhsine.com/jaxpathways/assessments/science.html) | available-draft | Written tasks, public scaffolds and rubric; no calibrated grading study. Defend a differentiable JAX inverse problem with conservation checks, sensitivity analysis, parameter recovery and explained numerical error. | Trial the tasks with unseen learner implementations and record rubric disagreements, changed cases and reviewer reasoning. |
| [rl: Reinforcement-learning synthesis](https://www.tahabouhsine.com/jaxpathways/assessments/rl.html) | available-draft | Written tasks, public scaffolds and rubric; no calibrated grading study. Assess JAX reinforcement learning with episode-boundary tests, independent policy expectations, clipped updates and held-out evaluation. | Trial the tasks with unseen learner implementations and record rubric disagreements, changed cases and reviewer reasoning. |
| [probability: Bayesian regression synthesis](https://www.tahabouhsine.com/jaxpathways/assessments/probability.html) | available-draft | Written tasks, public scaffolds and rubric; no calibrated grading study. Compare exact Bayesian regression with HMC and variational inference in JAX, then diagnose approximation error and posterior predictive behavior. | Trial the tasks with unseen learner implementations and record rubric disagreements, changed cases and reviewer reasoning. |
| [internals: Derivative synthesis](https://www.tahabouhsine.com/jaxpathways/assessments/internals.html) | available-draft | Written tasks, public scaffolds and rubric; no calibrated grading study. Defend JAX derivative transformations with rectangular Jacobians, custom VJP rules, primitive extensions and independent numerical comparisons. | Trial the tasks with unseen learner implementations and record rubric disagreements, changed cases and reviewer reasoning. |
| [ship: Shipping synthesis](https://www.tahabouhsine.com/jaxpathways/assessments/ship.html) | available-draft | Written tasks, public scaffolds and rubric; no calibrated grading study. Justify a JAX model release with adaptation comparisons, artifact checks, quantized inference measurements and transparent service capacity estimates. | Trial the tasks with unseen learner implementations and record rubric disagreements, changed cases and reviewer reasoning. |
| [ops: Workload operations synthesis](https://www.tahabouhsine.com/jaxpathways/assessments/ops.html) | available-draft | Written tasks, public scaffolds and rubric; no calibrated grading study. Rehearse JAX workload incidents with process outcome checks, complete-state recovery, model validation, rollback drills and capacity reasoning. | Trial the tasks with unseen learner implementations and record rubric disagreements, changed cases and reviewer reasoning. |
| [scale: Profiling, partitioning and kernel synthesis](https://www.tahabouhsine.com/jaxpathways/assessments/scale.html) | available-draft | Written tasks, public scaffolds and rubric; no calibrated grading study. Evaluate a sharded JAX workload with independent numerical checks, profiler evidence, partitioning analysis and a bounded Pallas kernel decision. | Trial the tasks with unseen learner implementations and record rubric disagreements, changed cases and reviewer reasoning. |
| [tpu: Training-system synthesis](https://www.tahabouhsine.com/jaxpathways/assessments/tpu.html) | available-draft | Written tasks, public scaffolds and rubric; no calibrated grading study. Assess a JAX causal training system through checkpoint recovery, cache precision, exported inference and a separate TPU runtime qualification plan. | Trial the tasks with unseen learner implementations and record rubric disagreements, changed cases and reviewer reasoning. |

## Every learning path

| Material | Maturity | Current evidence and useful scope | Next contribution |
| --- | --- | --- | --- |
| [foundations: Start with JAX](https://www.tahabouhsine.com/jaxpathways/learning-paths/foundations.html) | available-draft | Linked practice and capstone references; route comprehension unvalidated. Learn how arrays, derivatives and explicit state fit together by building and checking a regression experiment. Use the deeper math lessons when you need to explain why an update works or fails. | Pilot this complete route, verify its prerequisite effort and review the final learner evidence. Teaching review is still pending. |
| [models: Train neural networks](https://www.tahabouhsine.com/jaxpathways/learning-paths/models.html) | in-development | Linked practice and capstone references; route comprehension unvalidated. Train a CPU image classifier, preserve its complete state through interruption, inspect held-out errors and verify its exported inference. Language modeling and TPU work are optional branches. | Pilot this complete route, verify its prerequisite effort and review the final learner evidence. Parts still in development: deployment. |
| [science: Simulate and discover](https://www.tahabouhsine.com/jaxpathways/learning-paths/science.html) | in-development | Linked practice and capstone references; route comprehension unvalidated. Simulate physical dynamics, check numerical and derivative accuracy, and recover unknown parameters with held-out evidence. Build a reusable CPU inverse-problem audit before extending to adaptive solvers or accelerators. | Pilot this complete route, verify its prerequisite effort and review the final learner evidence. Parts still in development: science. |
| [probability: Model uncertainty](https://www.tahabouhsine.com/jaxpathways/learning-paths/probability.html) | in-development | Linked practice and capstone references; route comprehension unvalidated. Express assumptions as probability models, verify a known posterior, then implement HMC and variational inference with explicit diagnostics. Use predictive checks to distinguish a sampler problem from a model that does not explain the data. | Pilot this complete route, verify its prerequisite effort and review the final learner evidence. Parts still in development: probability. |
| [rl: Learn by interacting](https://www.tahabouhsine.com/jaxpathways/learning-paths/rl.html) | in-development | Linked practice and capstone references; route comprehension unvalidated. Train a policy from actual environment interaction, verify its objective against an independent small-state reference, and compare multiple seeded agents on fixed held-out evaluation streams. | Pilot this complete route, verify its prerequisite effort and review the final learner evidence. Parts still in development: rl. |
| [scale: Make it fast. Make it scale.](https://www.tahabouhsine.com/jaxpathways/learning-paths/scale.html) | in-development | Linked practice and capstone references; route comprehension unvalidated. Profile a correct workload, preserve its global objective while partitioning arrays, and audit actual Pallas kernels before requesting target measurements. CPU semantics are available; accelerator speed and multi-host claims require their own receipts. | Pilot this complete route, verify its prerequisite effort and review the final learner evidence. Parts still in development: distributed, kernels. |
| [internals: Understand JAX itself](https://www.tahabouhsine.com/jaxpathways/learning-paths/internals.html) | available-draft | Linked practice and capstone references; route comprehension unvalidated. Trace a function into jaxpr, verify forward and reverse sensitivities, write exact custom derivative rules, and compile a small transformation whose limits you can explain. | Pilot this complete route, verify its prerequisite effort and review the final learner evidence. Teaching review is still pending. |
| [tpu: Learn TPUs and Google Cloud](https://www.tahabouhsine.com/jaxpathways/learning-paths/tpu.html) | in-development | Linked practice and capstone references; route comprehension unvalidated. Build and recover an experiment on CPU, then learn GCP provisioning, TPU generations, training and inference memory, precision policies and XProf profiling. Connect measured error and real target traces to model engineering. | Pilot this complete route, verify its prerequisite effort and review the final learner evidence. Parts still in development: tpu, tpu-workflows, tpu-systems, transformers, pretraining, rl, posttraining, distributed, deployment, operations. |
| [ship: Ship and adapt JAX workloads](https://www.tahabouhsine.com/jaxpathways/learning-paths/ship.html) | in-development | Linked practice and capstone references; route comprehension unvalidated. Move trained computations into a measured inference service. Check framework/precision parity, package a non-root Docker image, verify readiness and connect evaluation, model identity and rollback evidence. | Pilot this complete route, verify its prerequisite effort and review the final learner evidence. Parts still in development: deployment. |
| [ops: Operate JAX workloads](https://www.tahabouhsine.com/jaxpathways/learning-paths/ops.html) | in-development | Linked practice and capstone references; route comprehension unvalidated. Operate reproducible model workloads from tracked experiments to containerized inference. Practice failure recovery, MLOps release gates, LLMOps evaluation and tracing, and ModelOps ownership, approval and rollback. | Pilot this complete route, verify its prerequisite effort and review the final learner evidence. Parts still in development: distributed, deployment, operations. |

## Every career guide

| Material | Maturity | Current evidence and useful scope | Next contribution |
| --- | --- | --- | --- |
| [training-engineer: ML & training engineer](https://www.tahabouhsine.com/jaxpathways/career-guides/training-engineer.html) | available-draft | Internal capability map; no external role or hiring validation. Turn a dataset and model idea into a reproducible experiment. You define the objective, diagnose learning, keep held-out evaluation isolated and prove a saved run can continue correctly. | Practitioners: review the responsibilities, missing skills, evidence criteria and prerequisite burden for this role. |
| [research-engineer: Research engineer](https://www.tahabouhsine.com/jaxpathways/career-guides/research-engineer.html) | available-draft | Internal capability map; no external role or hiring validation. Turn a proposed method into an experiment that can support or refute a claim. Your work connects mathematical assumptions, a simple baseline, independent checks and carefully bounded conclusions. | Practitioners: review the responsibilities, missing skills, evidence criteria and prerequisite burden for this role. |
| [scientific-ml: Scientific ML engineer](https://www.tahabouhsine.com/jaxpathways/career-guides/scientific-ml.html) | available-draft | Internal capability map; no external role or hiring validation. Connect observations to a numerical or physical model. You check derivatives and solver behavior, fit parameters and explain how noise, conditioning and model assumptions affect the answer. | Practitioners: review the responsibilities, missing skills, evidence criteria and prerequisite burden for this role. |
| [rl-engineer: RL engineer](https://www.tahabouhsine.com/jaxpathways/career-guides/rl-engineer.html) | available-draft | Internal capability map; no external role or hiring validation. Build experiments in which actions change later observations. You keep environment state, random sampling and policy updates reproducible, and evaluate improvement across seeds and conditions. | Practitioners: review the responsibilities, missing skills, evidence criteria and prerequisite burden for this role. |
| [performance-engineer: Accelerator & performance engineer](https://www.tahabouhsine.com/jaxpathways/career-guides/performance-engineer.html) | available-draft | Internal capability map; no external role or hiring validation. Measure where a workload spends time and memory, then improve the limiting operation while preserving its numerical contract. Explain when an optimization helps and when it does not. | Practitioners: review the responsibilities, missing skills, evidence criteria and prerequisite burden for this role. |
| [inference-engineer: Inference & deployment engineer](https://www.tahabouhsine.com/jaxpathways/career-guides/inference-engineer.html) | available-draft | Internal capability map; no external role or hiring validation. Package a trained model so another process or device can use it reliably. Own preprocessing, exported predictions, precision choices and measurements under a defined request workload. | Practitioners: review the responsibilities, missing skills, evidence criteria and prerequisite burden for this role. |
| [operations-engineer: JAX / accelerator operations engineer](https://www.tahabouhsine.com/jaxpathways/career-guides/operations-engineer.html) | available-draft | Internal capability map; no external role or hiring validation. Make model workloads observable and recoverable. Track progress and resource use, diagnose failed or stalled runs and rehearse recovery before relying on it. | Practitioners: review the responsibilities, missing skills, evidence criteria and prerequisite burden for this role. |

## Every modality track

| Material | Maturity | Current evidence and useful scope | Next contribution |
| --- | --- | --- | --- |
| [image: Image: from pixels to a deployed classifier](https://www.tahabouhsine.com/jaxpathways/tracks/image.html) | in-development | Bounded harness practice; full modality capstones planned. A connected CPU image harness is runnable: a trained CNN, full fresh-process recovery, interpreted clean/corrupted evaluation, calibrated integer arithmetic, serialized inference and measured rollout/rollback. Local image ingestion is implemented; real-photo quality and target-device performance remain unverified. | Connect the runnable harness to licensed natural data, meaningful task benchmarks and a separately qualified deployment target. |
| [text: Text: from tokens to reproducible generation](https://www.tahabouhsine.com/jaxpathways/tracks/text.html) | in-development | Bounded harness practice; full modality capstones planned. Runnable four-stage CPU byte-Transformer harness: real training, complete fresh-process recovery, fixed held-out evaluation, FP32/BF16/INT8 KV caches, six inference exports and profiler traces. Includes a local document loader. External-corpus quality, service load and physical TPU qualification remain extensions. An additional tiny bidirectional embedding pilot connects three objective phases and reports held-out retrieval against BM25, including a poor measured outcome. A separate 70,704-parameter causal pilot connects real-text next-token pretraining, response-masked continuation adaptation, chapter-held-out perplexity against a unigram baseline, and verified export sampling. Its weak evaluation and repetitive sample are retained. | Connect the runnable harness to licensed natural data, meaningful task benchmarks and a separately qualified deployment target. |
| [audio: Audio: from waveforms to an edge sound classifier](https://www.tahabouhsine.com/jaxpathways/tracks/audio.html) | in-development | Bounded harness practice; full modality capstones planned. A complete CPU waveform-to-inference harness is runnable: trained spectral classifier, full fresh-process recovery, fixed evaluation, calibrated precision and complete exported artifacts. PCM WAV ingestion is available; natural-recording quality and target-edge performance remain unmeasured. | Connect the runnable harness to licensed natural data, meaningful task benchmarks and a separately qualified deployment target. |
| [cross-modal: Cross-modal: connect images and text](https://www.tahabouhsine.com/jaxpathways/tracks/cross-modal.html) | in-development | Bounded harness practice; full modality capstones planned. A connected CPU image–text retrieval harness is runnable: both encoders train, complete state restores in a fresh process, calibrated precision policies export and reload, and a measured release gate supports rollback. The synthetic four-concept fixture is verified separately from natural-language or real-image retrieval quality. | Connect the runnable harness to licensed natural data, meaningful task benchmarks and a separately qualified deployment target. |

## Every practical guide

| Material | Maturity | Current evidence and useful scope | Next contribution |
| --- | --- | --- | --- |
| [project-workflow: Tools, folders and experiment organization](https://www.tahabouhsine.com/jaxpathways/project-workflow.html) | available-draft | Workspace checks available; novice review pending. A practical guide with a downloadable practice workspace. | Pilot the setup and evidence handoff with a new learner; review platform-specific and hardware claims. |
| [tpu-gcp: TPU and Google Cloud workflow](https://www.tahabouhsine.com/jaxpathways/tpu-gcp.html) | available-draft | Workspace checks available; novice review pending. A practical guide with a downloadable practice workspace. | Pilot the setup and evidence handoff with a new learner; review platform-specific and hardware claims. |
| [tpu-performance: TPU precision and profiling](https://www.tahabouhsine.com/jaxpathways/tpu-performance.html) | available-draft | Workspace checks available; novice review pending. A practical guide with a downloadable practice workspace. | Pilot the setup and evidence handoff with a new learner; review platform-specific and hardware claims. |

## Book formats

| Material | Maturity | Current evidence and useful scope | Next contribution |
| --- | --- | --- | --- |
| [reader: JAX Pathways book](https://www.tahabouhsine.com/jaxpathways/book.html) | in-development | Automated export checks; native reader and teaching qualification incomplete. HTML and EPUB include all authored chapters, figures, practical appendices and solutions. | Review math/code on native EPUB readers, print/PDF and assistive technology; contribute chapter-level teaching reviews. |

## Every lesson and its practice

| Material | Maturity | Current evidence and useful scope | Next contribution |
| --- | --- | --- | --- |
| [welcome-01: Set up your learning workspace](https://www.tahabouhsine.com/jaxpathways/lesson-welcome-01.html) | available-draft | CPU reference checked. Download a starter workspace, select its Python environment, run a saved program, and verify one deliberate change. | Run consented fresh-machine novice pilots on Windows, macOS and Linux; record where learners need help. |
| [welcome-02: Meet your arrays and devices](https://www.tahabouhsine.com/jaxpathways/lesson-welcome-02.html) | available-draft | CPU reference checked. Create an array, inspect its dtype and device, and explain the result. | Run consented fresh-machine novice pilots on Windows, macOS and Linux; record where learners need help. |
| [welcome-cpu: Practice with four virtual CPU devices](https://www.tahabouhsine.com/jaxpathways/lesson-welcome-cpu.html) | available-draft | Logical CPU reference checked. Start four logical CPU devices, inspect row-sharded and replicated arrays, and diagnose an indivisible batch. | Run consented fresh-machine novice pilots on Windows, macOS and Linux; record where learners need help. |
| [welcome-03: Move your experiment to a TPU](https://www.tahabouhsine.com/jaxpathways/lesson-welcome-03.html) | available-draft | CPU reference checked. Use the recorded CPU reference to check arithmetic and placement, then run only a named, source-reviewed TPU target with pinned dependencies and verify its observed backend receipt. | Run consented fresh-machine novice pilots on Windows, macOS and Linux; record where learners need help. |
| [arrays-01: From NumPy to jax.numpy](https://www.tahabouhsine.com/jaxpathways/lesson-arrays-01.html) | available-draft | CPU reference checked. Build feature-wise preprocessing with fitted statistics and verify behavior on new and constant data. | Add a masked variable-length batch that continues into the model project; review constant-feature and ragged-input explanations. |
| [arrays-02: Shapes, broadcasting, and dtypes](https://www.tahabouhsine.com/jaxpathways/lesson-arrays-02.html) | available-draft | CPU reference checked. Specify broadcasting contracts and detect silent pairwise-loss and dtype mistakes. | Add a masked variable-length batch that continues into the model project; review constant-feature and ragged-input explanations. |
| [arrays-03: Pure functions and explicit inputs](https://www.tahabouhsine.com/jaxpathways/lesson-arrays-03.html) | available-draft | CPU reference checked. Design an explicit parameter/data interface and return new state without mutating old containers. | Add a masked variable-length batch that continues into the model project; review constant-feature and ragged-input explanations. |
| [arrays-04: Immutable updates and indexing](https://www.tahabouhsine.com/jaxpathways/lesson-arrays-04.html) | available-draft | CPU reference checked. Implement functional indexed updates and distinguish accumulated contributions from shape-preserving masks. | Add a masked variable-length batch that continues into the model project; review constant-feature and ragged-input explanations. |
| [first-gradient: Your first gradient](https://www.tahabouhsine.com/jaxpathways/lesson-first-gradient.html) | available-draft | CPU reference checked. Explain local sensitivity and the chain rule, then compare analytic, numerical, and automatic derivatives. | Review transformation composition at nonsmooth boundaries and add a single continuing model across the five lessons. |
| [transforms-02: Losses and value_and_grad](https://www.tahabouhsine.com/jaxpathways/lesson-transforms-02.html) | available-draft | CPU reference checked. Derive and audit a scalar loss, its parameter gradient, and stable versus unstable updates. | Review transformation composition at nonsmooth boundaries and add a single continuing model across the five lessons. |
| [transforms-03: Batch a function with vmap](https://www.tahabouhsine.com/jaxpathways/lesson-transforms-03.html) | available-draft | CPU reference checked. Specify batching axes and connect per-example gradients to a reduced batch objective. | Review transformation composition at nonsmooth boundaries and add a single continuing model across the five lessons. |
| [transforms-04: Compile a function with jit](https://www.tahabouhsine.com/jaxpathways/lesson-transforms-04.html) | available-draft | CPU reference checked. Validate compiled loss/gradient values and produce a correctly synchronized latency report. | Review transformation composition at nonsmooth boundaries and add a single continuing model across the five lessons. |
| [transforms-05: Tracing, static arguments, and recompilation](https://www.tahabouhsine.com/jaxpathways/lesson-transforms-05.html) | available-draft | CPU reference checked. Explain the staging boundary and repair configuration versus runtime-control-flow mistakes. | Review transformation composition at nonsmooth boundaries and add a single continuing model across the five lessons. |
| [state-01: Random keys without surprises](https://www.tahabouhsine.com/jaxpathways/lesson-state-01.html) | available-draft | CPU reference checked. Split keys and show reproducibility without accidental key reuse. | Connect sample IDs, augmentation keys and optimizer state to a recovery handoff; test that novices can explain key reuse. |
| [state-02: Pytrees and structured parameters](https://www.tahabouhsine.com/jaxpathways/lesson-state-02.html) | available-draft | CPU reference checked. Represent parameters and state in a nested structure and transform its leaves. | Connect sample IDs, augmentation keys and optimizer state to a recovery handoff; test that novices can explain key reuse. |
| [state-03: Compiled loops with lax.scan](https://www.tahabouhsine.com/jaxpathways/lesson-state-03.html) | available-draft | CPU reference checked. Replace a fixed-length Python loop with scan and compare final states. | Connect sample IDs, augmentation keys and optimizer state to a recovery handoff; test that novices can explain key reuse. |
| [state-04: Branches with lax.cond](https://www.tahabouhsine.com/jaxpathways/lesson-state-04.html) | available-draft | CPU reference checked. Implement a data-dependent branch that works inside a compiled function. | Connect sample IDs, augmentation keys and optimizer state to a recovery handoff; test that novices can explain key reuse. |
| [optimization-05: Vectors, norms, and projections](https://www.tahabouhsine.com/jaxpathways/lesson-optimization-05.html) | available-draft | CPU reference checked. Read a dot product geometrically and separate a vector into a projection and an orthogonal residual. | Pilot the synthesis with learners; add constrained or nonsmooth optimization only around a new inspectable task, and improve feedback on wrong answers. |
| [optimization-01: Linear algebra and loss intuition](https://www.tahabouhsine.com/jaxpathways/lesson-optimization-01.html) | available-draft | CPU reference checked. Connect a matrix-vector product to predictions and a scalar regression loss. | Pilot the synthesis with learners; add constrained or nonsmooth optimization only around a new inspectable task, and improve feedback on wrong answers. |
| [optimization-06: Least squares, rank, and conditioning](https://www.tahabouhsine.com/jaxpathways/lesson-optimization-06.html) | available-draft | CPU reference checked. Fit a linear model with a direct solver and diagnose nonunique or sensitive coefficients. | Pilot the synthesis with learners; add constrained or nonsmooth optimization only around a new inspectable task, and improve feedback on wrong answers. |
| [optimization-07: The chain rule, Jacobians, and directions](https://www.tahabouhsine.com/jaxpathways/lesson-optimization-07.html) | available-draft | CPU reference checked. Compute a small Jacobian by hand and connect forward sensitivities to reverse-mode gradients. | Pilot the synthesis with learners; add constrained or nonsmooth optimization only around a new inspectable task, and improve feedback on wrong answers. |
| [optimization-02: Gradient checking and numerical accuracy](https://www.tahabouhsine.com/jaxpathways/lesson-optimization-02.html) | available-draft | CPU reference checked. Compare autodiff with finite differences over several step sizes. | Pilot the synthesis with learners; add constrained or nonsmooth optimization only around a new inspectable task, and improve feedback on wrong answers. |
| [optimization-08: Curvature, Hessians, and learning rates](https://www.tahabouhsine.com/jaxpathways/lesson-optimization-08.html) | available-draft | CPU reference checked. Use curvature to explain learning-rate stability, slow directions and why a zero gradient need not be a minimum. | Pilot the synthesis with learners; add constrained or nonsmooth optimization only around a new inspectable task, and improve feedback on wrong answers. |
| [optimization-03: Write gradient descent yourself](https://www.tahabouhsine.com/jaxpathways/lesson-optimization-03.html) | available-draft | CPU reference checked. Fit a regression model using explicit parameter updates. | Pilot the synthesis with learners; add constrained or nonsmooth optimization only around a new inspectable task, and improve feedback on wrong answers. |
| [optimization-09: Probability, likelihood, and stable losses](https://www.tahabouhsine.com/jaxpathways/lesson-optimization-09.html) | available-draft | CPU reference checked. Derive binary cross-entropy from a probability model and compute it safely from logits. | Pilot the synthesis with learners; add constrained or nonsmooth optimization only around a new inspectable task, and improve feedback on wrong answers. |
| [optimization-10: Regularization and honest validation](https://www.tahabouhsine.com/jaxpathways/lesson-optimization-10.html) | available-draft | CPU reference checked. Derive a ridge solution, distinguish data loss from penalized loss, and select a penalty using held-out data. | Pilot the synthesis with learners; add constrained or nonsmooth optimization only around a new inspectable task, and improve feedback on wrong answers. |
| [optimization-11: Minibatches, expectation, and gradient noise](https://www.tahabouhsine.com/jaxpathways/lesson-optimization-11.html) | available-draft | CPU reference checked. Explain an unbiased minibatch gradient and verify how sampling and reduction affect its variance. | Pilot the synthesis with learners; add constrained or nonsmooth optimization only around a new inspectable task, and improve feedback on wrong answers. |
| [optimization-04: Optimize with Optax](https://www.tahabouhsine.com/jaxpathways/lesson-optimization-04.html) | available-draft | CPU reference checked. Replace the handwritten update with an optimizer and track its state. | Pilot the synthesis with learners; add constrained or nonsmooth optimization only around a new inspectable task, and improve feedback on wrong answers. |
| [optimization-12: Adam, clipping, and learning-rate schedules](https://www.tahabouhsine.com/jaxpathways/lesson-optimization-12.html) | available-draft | CPU reference checked. Explain Adam’s moments, inspect clipping and schedule order, and compare optimizers with a controlled update budget. | Pilot the synthesis with learners; add constrained or nonsmooth optimization only around a new inspectable task, and improve feedback on wrong answers. |
| [networks-01: Build a tiny multilayer perceptron](https://www.tahabouhsine.com/jaxpathways/lesson-networks-01.html) | available-draft | CPU reference checked. Implement a small model with explicit parameters before adding a model library. | Add a licensed natural-image pilot with frozen source-group splits, a simple baseline and per-class error slices. |
| [networks-02: Model and state with Flax NNX](https://www.tahabouhsine.com/jaxpathways/lesson-networks-02.html) | available-draft | CPU reference checked. Express a model and inspect the distinction between parameters and other state. | Add a licensed natural-image pilot with frozen source-group splits, a simple baseline and per-class error slices. |
| [networks-03: A compiled train and evaluation step](https://www.tahabouhsine.com/jaxpathways/lesson-networks-03.html) | available-draft | CPU reference checked. Build train and eval steps and check that evaluation does not update weights. | Add a licensed natural-image pilot with frozen source-group splits, a simple baseline and per-class error slices. |
| [networks-04: Train a CNN on a small dataset](https://www.tahabouhsine.com/jaxpathways/lesson-networks-04.html) | available-draft | CPU reference checked. Train a classifier with a fixed data split and report held-out metrics. | Add a licensed natural-image pilot with frozen source-group splits, a simple baseline and per-class error slices. |
| [networks-05: Debug unstable learning](https://www.tahabouhsine.com/jaxpathways/lesson-networks-05.html) | available-draft | CPU reference checked. Introduce a bad learning rate and diagnose it using loss and gradient evidence. | Add a licensed natural-image pilot with frozen source-group splits, a simple baseline and per-class error slices. |
| [recovery-01: Design an input pipeline](https://www.tahabouhsine.com/jaxpathways/lesson-recovery-01.html) | available-draft | CPU reference checked. Build and audit a deterministic batched host pipeline. | Review exact-continuation wording against its fixed environment; inject storage and concurrent-writer failures separately from ordinary local restore. |
| [recovery-02: Load, batch, and checkpoint order with Grain](https://www.tahabouhsine.com/jaxpathways/lesson-recovery-02.html) | available-draft | CPU reference checked. Build a Grain source/sampler/batch pipeline and verify its record order. | Review exact-continuation wording against its fixed environment; inject storage and concurrent-writer failures separately from ordinary local restore. |
| [recovery-03: Save model, optimizer, and random state](https://www.tahabouhsine.com/jaxpathways/lesson-recovery-03.html) | available-draft | CPU reference checked. Checkpoint complete training state with Orbax and replay the next update. | Review exact-continuation wording against its fixed environment; inject storage and concurrent-writer failures separately from ordinary local restore. |
| [recovery-04: Recover data position and resume](https://www.tahabouhsine.com/jaxpathways/lesson-recovery-04.html) | available-draft | CPU reference checked. Persist model/optimizer/key and Grain iterator position at one boundary. | Review exact-continuation wording against its fixed environment; inject storage and concurrent-writer failures separately from ordinary local restore. |
| [recovery-05: Asynchronous checkpoints and failure boundaries](https://www.tahabouhsine.com/jaxpathways/lesson-recovery-05.html) | available-draft | CPU reference checked. Distinguish requested, completed, verified and published checkpoint state. Continue a functional update while an asynchronous writer is alive. | Review exact-continuation wording against its fixed environment; inject storage and concurrent-writer failures separately from ordinary local restore. |
| [recovery-06: Track experiments and lineage with MLflow](https://www.tahabouhsine.com/jaxpathways/lesson-recovery-06.html) | available-draft | CPU reference checked. Track, retrieve and compare two JAX experiments using explicit data identity, metric steps and downloaded model artifacts. | Review exact-continuation wording against its fixed environment; inject storage and concurrent-writer failures separately from ordinary local restore. |
| [transformers-01: Attention from small pieces](https://www.tahabouhsine.com/jaxpathways/lesson-transformers-01.html) | available-draft | CPU reference checked. Implement scaled dot-product attention and test its output shape. | Review tokenizer and length policies on broader licensed corpora, extend the measured short pilot into staged training and evaluate generation quality. Target-scale training and independent teaching review remain open. |
| [transformers-02: Masking, tokenization, and sequence packing](https://www.tahabouhsine.com/jaxpathways/lesson-transformers-02.html) | available-draft | CPU reference checked. Construct a causal mask and prevent packed sequences from attending across boundaries. | Review tokenizer and length policies on broader licensed corpora, extend the measured short pilot into staged training and evaluate generation quality. Target-scale training and independent teaching review remain open. |
| [transformers-03: A Transformer block and mixed precision](https://www.tahabouhsine.com/jaxpathways/lesson-transformers-03.html) | available-draft | CPU reference checked. Build a residual attention block and compare numerical behavior across precision choices. | Review tokenizer and length policies on broader licensed corpora, extend the measured short pilot into staged training and evaluate generation quality. Target-scale training and independent teaching review remain open. |
| [transformers-04: Train, checkpoint, and generate](https://www.tahabouhsine.com/jaxpathways/lesson-transformers-04.html) | available-draft | CPU reference checked. Train a tiny causal model, restore it, and generate from a fixed seed. | Review tokenizer and length policies on broader licensed corpora, extend the measured short pilot into staged training and evaluate generation quality. Target-scale training and independent teaching review remain open. |
| [performance-01: Benchmark asynchronous work correctly](https://www.tahabouhsine.com/jaxpathways/lesson-performance-01.html) | available-draft | CPU reference checked. Warm up a computation and synchronize results before reporting timing. | Annotate a real target trace with the exact workload and bottleneck; retain distributions and distinguish estimates from measured allocated memory. |
| [performance-02: Diagnose recompilation and host synchronization](https://www.tahabouhsine.com/jaxpathways/lesson-performance-02.html) | available-draft | CPU reference checked. Find a shape-changing call or host read and connect it to observed overhead. | Annotate a real target trace with the exact workload and bottleneck; retain distributions and distinguish estimates from measured allocated memory. |
| [performance-03: Read an XProf trace](https://www.tahabouhsine.com/jaxpathways/lesson-performance-03.html) | available-draft | CPU reference checked. Capture an actual warmed CPU trace with named steps. Identify nested intervals and avoid double-counting their durations. | Annotate a real target trace with the exact workload and bottleneck; retain distributions and distinguish estimates from measured allocated memory. |
| [performance-04: Reason with HLO and the roofline model](https://www.tahabouhsine.com/jaxpathways/lesson-performance-04.html) | available-draft | CPU reference checked. Locate matrix multiplication and activation in StableHLO. Derive the operation and minimum byte counts for a declared matrix contract. | Annotate a real target trace with the exact workload and bottleneck; retain distributions and distinguish estimates from measured allocated memory. |
| [performance-05: Memory, rematerialization, and optimization tradeoffs](https://www.tahabouhsine.com/jaxpathways/lesson-performance-05.html) | available-draft | CPU reference checked. Explain why reverse-mode differentiation needs forward intermediates. Compare ordinary and rematerialized gradients with an independent reverse recurrence. | Annotate a real target trace with the exact workload and bottleneck; retain distributions and distinguish estimates from measured allocated memory. |
| [distributed-01: Arrays, meshes, and sharding](https://www.tahabouhsine.com/jaxpathways/lesson-distributed-01.html) | available-draft | Logical CPU reference checked. Partition an array over a device mesh and inspect its local shards. | Qualify fresh-process multi-host restart and worker-loss recovery with explicit data ownership; do not infer scaling results from logical CPU devices. |
| [distributed-02: A sharded training step](https://www.tahabouhsine.com/jaxpathways/lesson-distributed-02.html) | available-draft | Logical CPU reference checked. Run a training step with explicit data and parameter placement. | Qualify fresh-process multi-host restart and worker-loss recovery with explicit data ownership; do not infer scaling results from logical CPU devices. |
| [distributed-03: Communication-efficient algorithms](https://www.tahabouhsine.com/jaxpathways/lesson-distributed-03.html) | available-draft | Logical CPU reference checked. Compare all-reduce and reduce-scatter against the same independent global gradient. Explain global result shape versus per-device storage. | Qualify fresh-process multi-host restart and worker-loss recovery with explicit data ownership; do not infer scaling results from logical CPU devices. |
| [distributed-04: Resilient distributed training](https://www.tahabouhsine.com/jaxpathways/lesson-distributed-04.html) | available-draft | Logical CPU reference checked. Restore a complete training state with explicit target sharding. Compare next sample IDs, optimizer state and losses across an epoch boundary. | Qualify fresh-process multi-host restart and worker-loss recovery with explicit data ownership; do not infer scaling results from logical CPU devices. |
| [science-01: A functional physical simulation](https://www.tahabouhsine.com/jaxpathways/lesson-science-01.html) | available-draft | CPU reference checked. Implement a pure scalar state transition with units and a fixed time grid. Separate a discrete implementation oracle from continuous approximation error. | Review the solver and inference assumptions with a domain specialist, extend the synthetic fixture to real measurements and validate uncertainty or model misspecification before broader scientific claims. |
| [science-02: Vectorize trajectories and scan time](https://www.tahabouhsine.com/jaxpathways/lesson-science-02.html) | available-draft | CPU reference checked. Construct and verify a fourth-order fixed-step transition. Distinguish batch-major from time-major trajectory layouts. | Review the solver and inference assumptions with a domain specialist, extend the synthetic fixture to real measurements and validate uncertainty or model misspecification before broader scientific claims. |
| [science-03: Differentiate through a solver](https://www.tahabouhsine.com/jaxpathways/lesson-science-03.html) | available-draft | CPU reference checked. Derive an independent discrete solver sensitivity. Check an observation-loss gradient at a nonoptimal parameter. | Review the solver and inference assumptions with a domain specialist, extend the synthetic fixture to real measurements and validate uncertainty or model misspecification before broader scientific claims. |
| [science-04: Fit parameters and scale the simulation](https://www.tahabouhsine.com/jaxpathways/lesson-science-04.html) | available-draft | CPU reference checked. Fit a constrained physical parameter through an executed simulator. Separate training fit, held-out prediction and solver refinement. | Review the solver and inference assumptions with a domain specialist, extend the synthetic fixture to real measurements and validate uncertainty or model misspecification before broader scientific claims. |
| [probability-01: Distributions and sampling](https://www.tahabouhsine.com/jaxpathways/lesson-probability-01.html) | available-draft | CPU reference checked. Distinguish samples, density, variance and standard error Check normal moments against independent analytic expectations | Review the probabilistic assumptions with a specialist and extend the synthetic model checks to a licensed real-data case with calibrated predictive evaluation. |
| [probability-02: Build a Bayesian regression](https://www.tahabouhsine.com/jaxpathways/lesson-probability-02.html) | available-draft | CPU reference checked. State the prior, likelihood and observation-noise assumptions Derive posterior precision and check a closed-form special case | Review the probabilistic assumptions with a specialist and extend the synthetic model checks to a licensed real-data case with calibrated predictive evaluation. |
| [probability-03: Hamiltonian Monte Carlo and sampler diagnostics](https://www.tahabouhsine.com/jaxpathways/lesson-probability-03.html) | available-draft | CPU reference checked. Implement leapfrog with the correct half-step boundaries Check reversibility and an independent analytic gradient | Review the probabilistic assumptions with a specialist and extend the synthetic model checks to a licensed real-data case with calibrated predictive evaluation. |
| [probability-04: Variational inference and model checking](https://www.tahabouhsine.com/jaxpathways/lesson-probability-04.html) | available-draft | CPU reference checked. Derive the diagonal Gaussian variational optimum for a correlated target Separate optimization error from variational-family error | Review the probabilistic assumptions with a specialist and extend the synthetic model checks to a licensed real-data case with calibrated predictive evaluation. |
| [rl-01: Write a functional environment](https://www.tahabouhsine.com/jaxpathways/lesson-rl-01.html) | available-draft | CPU reference checked. Implement reset and step as pure functions with explicit state. Distinguish termination, time-limit truncation and an absorbing finished state. | Connect the learned critic and GAE to actor updates, then evaluate larger tasks or continuous actions with independent learner and expert review. |
| [rl-02: Batch environments and scan rollouts](https://www.tahabouhsine.com/jaxpathways/lesson-rl-02.html) | available-draft | CPU reference checked. Build a scan over vmapped transitions with explicit key ownership. Reconstruct episode returns and reward-to-go from padded trajectories. | Connect the learned critic and GAE to actor updates, then evaluate larger tasks or continuous actions with independent learner and expert review. |
| [rl-03: Policy gradients and PPO](https://www.tahabouhsine.com/jaxpathways/lesson-rl-03.html) | available-draft | CPU reference checked. Derive and implement the score-function policy gradient for complete episodes. Train with fresh stochastic rollouts and a fixed behavior policy per PPO batch. | Connect the learned critic and GAE to actor updates, then evaluate larger tasks or continuous actions with independent learner and expert review. |
| [rl-04: Evaluate agents across seeds](https://www.tahabouhsine.com/jaxpathways/lesson-rl-04.html) | available-draft | CPU reference checked. Separate training seeds, evaluation streams and individual episodes. Report per-agent means and variability with an explicit aggregation unit. | Connect the learned critic and GAE to actor updates, then evaluate larger tasks or continuous actions with independent learner and expert review. |
| [rl-05: Learn value functions and generalized advantage estimation](https://www.tahabouhsine.com/jaxpathways/lesson-rl-05.html) | available-draft | CPU reference checked. Fit a tabular value function from retained trajectories and estimate GAE with terminal and collector-cutoff semantics explicit. | Connect the learned critic and GAE to actor updates, then evaluate larger tasks or continuous actions with independent learner and expert review. |
| [kernels-01: Pallas grids and BlockSpecs](https://www.tahabouhsine.com/jaxpathways/lesson-kernels-01.html) | available-draft | CPU reference checked. Implement an actual Pallas Ref-writing kernel with grid and BlockSpecs. Derive block coordinates and distinguish them from element offsets. | Turn the bounded target receipt into a learner tuning lab with retained trace interpretation and end-to-end comparison; GPU and general pipeline claims remain unqualified. |
| [kernels-02: A first TPU kernel](https://www.tahabouhsine.com/jaxpathways/lesson-kernels-02.html) | available-draft | CPU reference checked. Implement a real TPU-targeted Pallas launch with a broadcast BlockSpec. Specify target layout and float32 compute/output-cast policies. | Turn the bounded target receipt into a learner tuning lab with retained trace interpretation and end-to-end comparison; GPU and general pipeline claims remain unqualified. |
| [kernels-03: Tiling, memory, and pipelining](https://www.tahabouhsine.com/jaxpathways/lesson-kernels-03.html) | available-draft | CPU reference checked. Execute actual emit_pipeline semantics under explicit CPU TPU-layout simulation. Compare synchronous and buffered schedules against an independent oracle. | Turn the bounded target receipt into a learner tuning lab with retained trace interpretation and end-to-end comparison; GPU and general pipeline claims remain unqualified. |
| [kernels-04: Iterate with correctness and performance evidence](https://www.tahabouhsine.com/jaxpathways/lesson-kernels-04.html) | available-draft | CPU reference checked. Build a representative shape/dtype correctness matrix for an actual Pallas pipeline. Separate kernel implementation error from input/output precision differences. | Turn the bounded target receipt into a learner tuning lab with retained trace interpretation and end-to-end comparison; GPU and general pipeline claims remain unqualified. |
| [internals-01: Read your first jaxpr](https://www.tahabouhsine.com/jaxpathways/lesson-internals-01.html) | available-draft | CPU reference checked. Inspect a small transformed function and identify its operations. | Add transformation-composition failures and effect/extension boundaries; have an independent reviewer check the supported subset and numerical assumptions. |
| [internals-02: JVPs, VJPs, and higher-order derivatives](https://www.tahabouhsine.com/jaxpathways/lesson-internals-02.html) | available-draft | CPU reference checked. Compute and interpret JVP and VJP products with correct domains and shapes. Verify the adjoint dot-product identity against a hand-derived Jacobian. | Add transformation-composition failures and effect/extension boundaries; have an independent reviewer check the supported subset and numerical assumptions. |
| [internals-03: Custom derivative rules](https://www.tahabouhsine.com/jaxpathways/lesson-internals-03.html) | available-draft | CPU reference checked. Implement actual custom_jvp and custom_vjp rules with correct return and residual contracts. Verify seed linearity, stable extreme values and second derivatives. | Add transformation-composition failures and effect/extension boundaries; have an independent reviewer check the supported subset and numerical assumptions. |
| [internals-04: Lowering, compilation, and a tiny transformation](https://www.tahabouhsine.com/jaxpathways/lesson-internals-04.html) | available-draft | CPU reference checked. Interpret literals, captured constants and variables in a supported flat jaxpr. Implement actual forward derivative rules without calling jax.jvp inside the interpreter. | Add transformation-composition failures and effect/extension boundaries; have an independent reviewer check the supported subset and numerical assumptions. |
| [deployment-02: Adapt a pretrained model and choose a post-training objective](https://www.tahabouhsine.com/jaxpathways/lesson-deployment-02.html) | available-draft | CPU reference checked. Pretrain and reload an artifact with recorded data and checkpoint hashes Compare supervised adaptation and teacher distillation from the same starting weights | Execute and document a real container service, concurrent request workload and named edge-device/delegate report; dense parity is not full architecture portability. |
| [deployment-01: Move models between JAX, Keras, TensorFlow and PyTorch](https://www.tahabouhsine.com/jaxpathways/lesson-deployment-01.html) | available-draft | CPU reference checked. Verify parameter layouts and inference semantics before switching frameworks. | Execute and document a real container service, concurrent request workload and named edge-device/delegate report; dense parity is not full architecture portability. |
| [deployment-08: Convert PyTorch weights to Flax and locate numerical errors](https://www.tahabouhsine.com/jaxpathways/lesson-deployment-08.html) | available-draft | CPU reference checked. Convert a real PyTorch state dictionary to Flax NNX and validate intermediate outputs, gradients and reloaded artifacts with explicit numerical tolerances. | Execute and document a real container service, concurrent request workload and named edge-device/delegate report; dense parity is not full architecture portability. |
| [deployment-03: Export a computation and verify its serving contract](https://www.tahabouhsine.com/jaxpathways/lesson-deployment-03.html) | available-draft | CPU reference checked. Round-trip an exported inference function and reject incompatible requests. | Execute and document a real container service, concurrent request workload and named edge-device/delegate report; dense parity is not full architecture portability. |
| [deployment-05: Choose weight, activation and accumulation precision](https://www.tahabouhsine.com/jaxpathways/lesson-deployment-05.html) | available-draft | CPU reference checked. Compare floating-point policies, weight-only quantization and calibrated integer activations. | Execute and document a real container service, concurrent request workload and named edge-device/delegate report; dense parity is not full architecture portability. |
| [deployment-07: Containerize a model service and verify its boundary](https://www.tahabouhsine.com/jaxpathways/lesson-deployment-07.html) | available-draft | CPU reference checked. Package an inference service and verify model identity, request validation and readiness across process and container boundaries. | Execute and document a real container service, concurrent request workload and named edge-device/delegate report; dense parity is not full architecture portability. |
| [deployment-04: Inference capacity, batching, and autoscaling](https://www.tahabouhsine.com/jaxpathways/lesson-deployment-04.html) | available-draft | CPU reference checked. Measure completed inference after warming each batch shape Report batch latency and example throughput with correct units | Execute and document a real container service, concurrent request workload and named edge-device/delegate report; dense parity is not full architecture portability. |
| [deployment-06: Deploy at the edge: conversion, budgets and device checks](https://www.tahabouhsine.com/jaxpathways/lesson-deployment-06.html) | available-draft | CPU reference checked. Define an edge runtime contract and measure an end-to-end batch-one request. | Execute and document a real container service, concurrent request workload and named edge-device/delegate report; dense parity is not full architecture portability. |
| [operations-01: Accelerator jobs and runtime lifecycle](https://www.tahabouhsine.com/jaxpathways/lesson-operations-01.html) | available-draft | CPU reference checked. Launch and reap a real bounded subprocess with explicit runtime requirements. Distinguish successful completion, worker failure and supervisor timeout. | Add authenticated approvals, multi-writer release control, load-driven SLO incidents and application-specific LLM evaluation beyond deterministic fixtures. |
| [operations-02: Health, logs, and workload observability](https://www.tahabouhsine.com/jaxpathways/lesson-operations-02.html) | available-draft | CPU reference checked. Build useful summaries from structured events emitted by a real worker. Define throughput with explicit numerator, denominator and synchronization boundary. | Add authenticated approvals, multi-writer release control, load-driven SLO incidents and application-specific LLM evaluation beyond deterministic fixtures. |
| [operations-03: Failure recovery and operational runbooks](https://www.tahabouhsine.com/jaxpathways/lesson-operations-03.html) | available-draft | CPU reference checked. Rehearse a real worker failure and resume from an atomic complete-state checkpoint. Verify the next update and every later state against an uninterrupted reference. | Add authenticated approvals, multi-writer release control, load-driven SLO incidents and application-specific LLM evaluation beyond deterministic fixtures. |
| [operations-04: Changes, rollback, and artifact provenance](https://www.tahabouhsine.com/jaxpathways/lesson-operations-04.html) | available-draft | CPU reference checked. Separate content identity, validation evidence and active selection. Reject corrupt or failed artifacts before a pointer change. | Add authenticated approvals, multi-writer release control, load-driven SLO incidents and application-specific LLM evaluation beyond deterministic fixtures. |
| [operations-05: Capacity, utilization, and operating cost](https://www.tahabouhsine.com/jaxpathways/lesson-operations-05.html) | available-draft | CPU reference checked. Measure end-to-end local service time at more than one workload length. Compute offered load, worker reserve and illustrative cost with consistent units. | Add authenticated approvals, multi-writer release control, load-driven SLO incidents and application-specific LLM evaluation beyond deterministic fixtures. |
| [operations-06: MLOps: data contracts, CI gates and monitoring](https://www.tahabouhsine.com/jaxpathways/lesson-operations-06.html) | available-draft | CPU reference checked. Reject data leakage and incomplete evaluation, compute count-weighted slice metrics and distinguish drift investigation from model promotion. | Add authenticated approvals, multi-writer release control, load-driven SLO incidents and application-specific LLM evaluation beyond deterministic fixtures. |
| [operations-07: LLMOps: version prompts, evaluate behavior and trace requests](https://www.tahabouhsine.com/jaxpathways/lesson-operations-07.html) | available-draft | CPU reference checked. Evaluate a versioned text application with a complete case inventory, slice-level answer contracts and inspectable MLflow traces. | Add authenticated approvals, multi-writer release control, load-driven SLO incidents and application-specific LLM evaluation beyond deterministic fixtures. |
| [operations-08: ModelOps: own, approve, roll back and retire a model](https://www.tahabouhsine.com/jaxpathways/lesson-operations-08.html) | available-draft | CPU reference checked. Bind a release decision to immutable evidence, reject changed or expired approvals and preserve active selection after failure. | Add authenticated approvals, multi-writer release control, load-driven SLO incidents and application-specific LLM evaluation beyond deterministic fixtures. |
| [pretraining-01: Masked language modeling: predict hidden tokens](https://www.tahabouhsine.com/jaxpathways/lesson-pretraining-01.html) | available-draft | CPU reference checked. Separate the clean targets from the corrupted input. Normalize over selected target positions. | Integrate the objectives with saved encoder architectures, licensed data recipes and downstream probes; the larger modality capstones are still plans. |
| [pretraining-02: Masked image modeling: reconstruct missing patches](https://www.tahabouhsine.com/jaxpathways/lesson-pretraining-02.html) | available-draft | CPU reference checked. Preserve patch order and pixel order. Keep hidden pixels out of the encoder. | Integrate the objectives with saved encoder architectures, licensed data recipes and downstream probes; the larger modality capstones are still plans. |
| [pretraining-03: Contrastive learning: views, positives and negatives](https://www.tahabouhsine.com/jaxpathways/lesson-pretraining-03.html) | available-draft | CPU reference checked. Define the positive pair before augmenting. Normalize embeddings and introduce temperature. | Integrate the objectives with saved encoder architectures, licensed data recipes and downstream probes; the larger modality capstones are still plans. |
| [posttraining-01: Supervised fine-tuning with response-only token loss](https://www.tahabouhsine.com/jaxpathways/lesson-posttraining-01.html) | available-draft | CPU reference checked. Align predictions with their target tokens. Separate the attention mask from the loss mask. | Adapt a real saved Transformer and compare target quality, retained behavior and reference drift; sequence PPO/value models and human preferences are not supplied. |
| [posttraining-02: LoRA: adapt, save and merge low-rank updates](https://www.tahabouhsine.com/jaxpathways/lesson-posttraining-02.html) | available-draft | CPU reference checked. Separate the frozen base from trainable adapters. Understand initialization and the first gradient. | Adapt a real saved Transformer and compare target quality, retained behavior and reference drift; sequence PPO/value models and human preferences are not supplied. |
| [posttraining-03: Learn a reward model from pairwise preferences](https://www.tahabouhsine.com/jaxpathways/lesson-posttraining-03.html) | available-draft | CPU reference checked. Represent a preference as a comparison. Fit score differences with a stable likelihood. | Adapt a real saved Transformer and compare target quality, retained behavior and reference drift; sequence PPO/value models and human preferences are not supplied. |
| [posttraining-04: RLHF mechanics: a frozen reward and PPO policy updates](https://www.tahabouhsine.com/jaxpathways/lesson-posttraining-04.html) | available-draft | CPU reference checked. Keep three identities separate. Collect actions before optimizing them. | Adapt a real saved Transformer and compare target quality, retained behavior and reference drift; sequence PPO/value models and human preferences are not supplied. |
| [posttraining-05: Direct preference optimization and reference-corrected margins](https://www.tahabouhsine.com/jaxpathways/lesson-posttraining-05.html) | available-draft | CPU reference checked. Compare chosen and rejected response log probabilities. Subtract the reference preference. | Adapt a real saved Transformer and compare target quality, retained behavior and reference drift; sequence PPO/value models and human preferences are not supplied. |
| [tpu-01: Select a TPU slice and provision your Cloud TPU VM](https://www.tahabouhsine.com/jaxpathways/lesson-tpu-01.html) | available-draft | CPU reference checked. Choose a single-host TPU topology, create a Cloud TPU VM with gcloud (or connect to a Colab TPU), verify SSH access, and rehearse clean VM deletion. | Review generation-specific topology and memory assumptions, test a learner provisioning journey and keep pricing illustrative rather than a live quote. |
| [tpu-04: Bootstrap .venv and verify a TPU runtime receipt](https://www.tahabouhsine.com/jaxpathways/lesson-tpu-04.html) | available-draft | CPU reference checked. Create ~/jax-tpu-lab/.venv on the TPU VM, install jax[tpu], record the observed backend and local device count alongside a synchronized pmap reduction check, retrieve the JSON receipt, and then delete the VM. | Review generation-specific topology and memory assumptions, test a learner provisioning journey and keep pricing illustrative rather than a live quote. |
| [tpu-02: Stage and launch checkpointed TPU jobs](https://www.tahabouhsine.com/jaxpathways/lesson-tpu-02.html) | available-draft | CPU reference checked. Stage launch.py to a TPU VM, separate first-step XLA compilation warmup from steady-state step timing, and verify that resuming from an atomic checkpoint matches an uninterrupted run. | Qualify fresh-process and worker-loss recovery, retained per-worker evidence and cleanup in an observed learner run; device discovery alone is not training qualification. |
| [tpu-05: Review scripts before TPU staging](https://www.tahabouhsine.com/jaxpathways/lesson-tpu-05.html) | available-draft | CPU reference checked. Classify the exact program before staging: keep `distributed-02` on CPU and generate a TPU launch only for a reviewed target whose backend, dependencies, topology, and data ownership are explicit. Distinguish multi-host discovery from a qualified distributed update. | Qualify fresh-process and worker-loss recovery, retained per-worker evidence and cleanup in an observed learner run; device discovery alone is not training qualification. |
| [tpu-03: TPU generations and HBM memory budgeting](https://www.tahabouhsine.com/jaxpathways/lesson-tpu-03.html) | available-draft | CPU reference checked. Compare TPU v5e, v5p, and v6e specifications and roofline ridge points, budget FP32 vs mixed-BF16 training state and KV-cache HBM, and size the minimum chip count for a model. | Teach target trace interpretation and native low-bit kernel/storage evidence; distinguish per-chip capacity, sharded state and dequantized reference arithmetic. |
| [tpu-06: bfloat16, int8 precision, and XProf profiling on TPU](https://www.tahabouhsine.com/jaxpathways/lesson-tpu-06.html) | available-draft | CPU reference checked. Measure BF16 operand and requested result-dtype error against a finite-precision FP32 JAX reference, inspect CPU dtype and compiler evidence, and retain a warmed XProf trace with its observed backend. | Teach target trace interpretation and native low-bit kernel/storage evidence; distinguish per-chip capacity, sharded state and dequantized reference arithmetic. |

## Every exercise, checkpoint and project stage

These are public practice tasks, not independently administered exams. A reference run does not establish learner mastery.

| Task | Material | Status | Scope |
| --- | --- | --- | --- |
| `welcome-01:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `welcome-01:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `welcome-02:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `welcome-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `welcome-02:practice-1` | Add a new observation | available-draft | CPU reference checked; reference solution, not learner grading |
| `welcome-02:practice-2` | Diagnose an impossible reshape | available-draft | CPU reference checked; reference solution, not learner grading |
| `welcome-cpu:exercise` | Main exercise | available-draft | Logical CPU reference checked; reference solution, not learner grading |
| `welcome-cpu:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `welcome-cpu:practice-1` | Move from four rows to twelve | available-draft | Logical CPU reference checked; reference solution, not learner grading |
| `welcome-cpu:practice-2` | Repair a batch that does not divide evenly | available-draft | Logical CPU reference checked; reference solution, not learner grading |
| `welcome-03:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `welcome-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `welcome-03:practice-1` | Test row order without changing the model | available-draft | CPU reference checked; reference solution, not learner grading |
| `arrays-01:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `arrays-01:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `arrays-01:practice-1` | Test the constant-feature policy | available-draft | CPU reference checked; reference solution, not learner grading |
| `arrays-01:practice-2` | Detect accidental batch-dependent preprocessing | available-draft | CPU reference checked; reference solution, not learner grading |
| `arrays-02:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `arrays-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `arrays-02:practice-1` | Use both kinds of offsets together | available-draft | CPU reference checked; reference solution, not learner grading |
| `arrays-02:practice-2` | Reject an aligned-loss contract violation | available-draft | CPU reference checked; reference solution, not learner grading |
| `arrays-03:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `arrays-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `arrays-03:practice-1` | Update the bias without changing either parameter input | available-draft | CPU reference checked; reference solution, not learner grading |
| `arrays-03:practice-2` | Return predictions and metrics together | available-draft | CPU reference checked; reference solution, not learner grading |
| `arrays-04:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `arrays-04:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `arrays-04:practice-1` | Replace a region of a matrix | available-draft | CPU reference checked; reference solution, not learner grading |
| `arrays-04:practice-2` | Keep shape while masking values | available-draft | CPU reference checked; reference solution, not learner grading |
| `first-gradient:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `first-gradient:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `first-gradient:practice-1` | Differentiate the parameter you intended | available-draft | CPU reference checked; reference solution, not learner grading |
| `first-gradient:practice-2` | Repair an objective without hiding its meaning | available-draft | CPU reference checked; reference solution, not learner grading |
| `transforms-02:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `transforms-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `transforms-02:practice-1` | Make the data boundary explicit | available-draft | CPU reference checked; reference solution, not learner grading |
| `transforms-02:practice-2` | Audit reduction and batch replication | available-draft | CPU reference checked; reference solution, not learner grading |
| `transforms-03:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `transforms-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `transforms-03:practice-1` | Turn sensitivities into training gradients | available-draft | CPU reference checked; reference solution, not learner grading |
| `transforms-03:practice-2` | Prove the mean-gradient identity with an experiment | available-draft | CPU reference checked; reference solution, not learner grading |
| `transforms-04:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `transforms-04:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `transforms-04:practice-1` | Check that a new value remains runtime data | available-draft | CPU reference checked; reference solution, not learner grading |
| `transforms-04:practice-2` | Build a report whose timing claim can be checked | available-draft | CPU reference checked; reference solution, not learner grading |
| `transforms-05:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `transforms-05:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `transforms-05:practice-1` | Keep runtime values dynamic | available-draft | CPU reference checked; reference solution, not learner grading |
| `transforms-05:practice-2` | Audit a shape-dependent specialization | available-draft | CPU reference checked; reference solution, not learner grading |
| `state-01:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `state-01:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `state-01:practice-1` | Replay a noisy prediction | available-draft | CPU reference checked; reference solution, not learner grading |
| `state-01:practice-2` | Diagnose key reuse in a loop | available-draft | CPU reference checked; reference solution, not learner grading |
| `state-02:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `state-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `state-02:practice-1` | Extend to a nested model | available-draft | CPU reference checked; reference solution, not learner grading |
| `state-02:practice-2` | Repair a mismatched update tree | available-draft | CPU reference checked; reference solution, not learner grading |
| `state-03:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `state-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `state-03:practice-1` | Record a different output | available-draft | CPU reference checked; reference solution, not learner grading |
| `state-03:practice-2` | Diagnose a growing carry | available-draft | CPU reference checked; reference solution, not learner grading |
| `state-04:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `state-04:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `state-04:practice-1` | Match a changed piecewise rule | available-draft | CPU reference checked; reference solution, not learner grading |
| `state-04:practice-2` | Repair incompatible branch shapes | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-05:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-05:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `optimization-05:practice-1` | Handle an undefined direction | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-05:practice-2` | Use two perpendicular directions | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-01:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-01:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `optimization-01:practice-1` | Add an unseen observation | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-01:practice-2` | Reject and repair a column target | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-06:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-06:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `optimization-06:practice-1` | Change feature units | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-06:practice-2` | Explain the squared conditioning | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-07:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-07:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `optimization-07:practice-1` | Find the broken derivative path | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-07:practice-2` | Transfer to unequal dimensions | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-02:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `optimization-02:practice-1` | Move to a different smooth objective | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-02:practice-2` | Catch the wrong objective despite correct gradients | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-08:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-08:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `optimization-08:practice-1` | Reject the zero-gradient shortcut | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-08:practice-2` | Transfer to a nonquadratic function | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-03:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `optimization-03:practice-1` | Transfer the stability analysis | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-03:practice-2` | Reproduce an uphill update and repair it | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-09:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-09:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `optimization-09:practice-1` | Fit a constant probability | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-09:practice-2` | Catch a reduction mistake | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-10:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-10:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `optimization-10:practice-1` | Do not accidentally shrink the intercept | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-10:practice-2` | Catch preprocessing leakage | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-11:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-11:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `optimization-11:practice-1` | Accumulate before updating | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-11:practice-2` | Detect a biased sampler | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-04:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-04:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `optimization-04:practice-1` | Replay a momentum continuation | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-04:practice-2` | Catch double subtraction | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-12:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-12:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `optimization-12:practice-1` | Catch schedule restart | available-draft | CPU reference checked; reference solution, not learner grading |
| `optimization-12:practice-2` | Check transformation order | available-draft | CPU reference checked; reference solution, not learner grading |
| `networks-01:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `networks-01:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `networks-01:practice-1` | Reorder hidden units without changing predictions | available-draft | CPU reference checked; reference solution, not learner grading |
| `networks-01:practice-2` | Diagnose accidental pairwise broadcasting | available-draft | CPU reference checked; reference solution, not learner grading |
| `networks-02:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `networks-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `networks-02:practice-1` | Change a clone without changing its source | available-draft | CPU reference checked; reference solution, not learner grading |
| `networks-02:practice-2` | Diagnose a shared module reference | available-draft | CPU reference checked; reference solution, not learner grading |
| `networks-03:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `networks-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `networks-03:practice-1` | Aggregate unequal batches | available-draft | CPU reference checked; reference solution, not learner grading |
| `networks-03:practice-2` | Catch an evaluation function that trains | available-draft | CPU reference checked; reference solution, not learner grading |
| `networks-04:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `networks-04:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `networks-04:practice-1` | Read a confusion matrix as conditional rates | available-draft | CPU reference checked; reference solution, not learner grading |
| `networks-04:practice-2` | Catch a channel-order mistake | available-draft | CPU reference checked; reference solution, not learner grading |
| `networks-05:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `networks-05:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `networks-05:practice-1` | Rescale the objective and compensate the update | available-draft | CPU reference checked; reference solution, not learner grading |
| `networks-05:practice-2` | Diagnose a nonfinite input before an update | available-draft | CPU reference checked; reference solution, not learner grading |
| `recovery-01:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `recovery-01:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `recovery-01:practice-1` | Fingerprint meaning, not just a row count | available-draft | CPU reference checked; reference solution, not learner grading |
| `recovery-01:practice-2` | Catch a feature/label shuffle mismatch | available-draft | CPU reference checked; reference solution, not learner grading |
| `recovery-02:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `recovery-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `recovery-02:practice-1` | Replay through an epoch boundary | available-draft | CPU reference checked; reference solution, not learner grading |
| `recovery-02:practice-2` | Reject a changed sampler | available-draft | CPU reference checked; reference solution, not learner grading |
| `recovery-03:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `recovery-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `recovery-03:practice-1` | Show what weights-only recovery loses | available-draft | CPU reference checked; reference solution, not learner grading |
| `recovery-03:practice-2` | Reject an incompatible update configuration | available-draft | CPU reference checked; reference solution, not learner grading |
| `recovery-04:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `recovery-04:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `recovery-04:practice-1` | Reject an incompatible resume before consuming data | available-draft | CPU reference checked; reference solution, not learner grading |
| `recovery-04:practice-2` | Reject changed batch size before consuming | available-draft | CPU reference checked; reference solution, not learner grading |
| `recovery-05:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `recovery-05:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `recovery-05:practice-1` | Keep only a genuinely accepted target | available-draft | CPU reference checked; reference solution, not learner grading |
| `recovery-06:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `recovery-06:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `recovery-06:practice-1` | Change a data value | available-draft | CPU reference checked; reference solution, not learner grading |
| `transformers-01:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `transformers-01:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `transformers-01:practice-1` | Use a different value dimension | available-draft | CPU reference checked; reference solution, not learner grading |
| `transformers-01:practice-2` | Expose a wrong softmax axis | available-draft | CPU reference checked; reference solution, not learner grading |
| `transformers-02:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `transformers-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `transformers-02:practice-1` | Repair padding-query NaNs | available-draft | CPU reference checked; reference solution, not learner grading |
| `transformers-02:practice-2` | Detect packed-example leakage | available-draft | CPU reference checked; reference solution, not learner grading |
| `transformers-03:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `transformers-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `transformers-03:practice-1` | Check normalization without autodiff | available-draft | CPU reference checked; reference solution, not learner grading |
| `transformers-03:practice-2` | Repair a residual width mismatch | available-draft | CPU reference checked; reference solution, not learner grading |
| `transformers-04:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `transformers-04:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `transformers-04:practice-1` | Reject an incompatible checkpoint | available-draft | CPU reference checked; reference solution, not learner grading |
| `transformers-04:practice-2` | Make prompt failures explicit | available-draft | CPU reference checked; reference solution, not learner grading |
| `performance-01:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `performance-01:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `performance-01:practice-1` | Benchmark a different equivalent workload | available-draft | CPU reference checked; reference solution, not learner grading |
| `performance-01:practice-2` | Repair a timer that includes setup but omits completion | available-draft | CPU reference checked; reference solution, not learner grading |
| `performance-02:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `performance-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `performance-02:practice-1` | Pad a final batch while preserving the objective | available-draft | CPU reference checked; reference solution, not learner grading |
| `performance-02:practice-2` | Reproduce and repair runtime Python branching | available-draft | CPU reference checked; reference solution, not learner grading |
| `performance-03:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `performance-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `performance-03:practice-1` | Compute a union instead of summing overlap | available-draft | CPU reference checked; reference solution, not learner grading |
| `performance-04:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `performance-04:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `performance-04:practice-1` | Keep the same shapes but change storage precision | available-draft | CPU reference checked; reference solution, not learner grading |
| `performance-05:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `performance-05:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `performance-05:practice-1` | Check higher-order differentiation | available-draft | CPU reference checked; reference solution, not learner grading |
| `distributed-01:exercise` | Main exercise | available-draft | Logical CPU reference checked; reference solution, not learner grading |
| `distributed-01:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `distributed-01:practice-1` | Change both dimensions | available-draft | Logical CPU reference checked; reference solution, not learner grading |
| `distributed-01:practice-2` | Pad an uneven batch without corrupting its mean | available-draft | Logical CPU reference checked; reference solution, not learner grading |
| `distributed-02:exercise` | Main exercise | available-draft | Logical CPU reference checked; reference solution, not learner grading |
| `distributed-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `distributed-02:practice-1` | Transfer to a different target model | available-draft | Logical CPU reference checked; reference solution, not learner grading |
| `distributed-02:practice-2` | Diagnose summed local means | available-draft | Logical CPU reference checked; reference solution, not learner grading |
| `distributed-03:exercise` | Main exercise | available-draft | Logical CPU reference checked; reference solution, not learner grading |
| `distributed-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `distributed-03:practice-1` | Account for the complete communication cycle | available-draft | Logical CPU reference checked; reference solution, not learner grading |
| `distributed-04:exercise` | Main exercise | available-draft | Logical CPU reference checked; reference solution, not learner grading |
| `distributed-04:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `distributed-04:practice-1` | Find a delayed random-state failure | available-draft | Logical CPU reference checked; reference solution, not learner grading |
| `science-01:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `science-01:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `science-01:practice-1` | Energy does not check every invariant | available-draft | CPU reference checked; reference solution, not learner grading |
| `science-01:practice-2` | Keep the physical horizon fixed | available-draft | CPU reference checked; reference solution, not learner grading |
| `science-02:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `science-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `science-02:practice-1` | Save only what the objective needs | available-draft | CPU reference checked; reference solution, not learner grading |
| `science-02:practice-2` | Pairing versus Cartesian products | available-draft | CPU reference checked; reference solution, not learner grading |
| `science-03:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `science-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `science-03:practice-1` | Weight observations without losing normalization | available-draft | CPU reference checked; reference solution, not learner grading |
| `science-03:practice-2` | Diagnose a detached backward path | available-draft | CPU reference checked; reference solution, not learner grading |
| `science-03:practice-3` | Check adaptive log-rate sensitivity at multiple tolerances | available-draft | CPU reference checked; reference solution, not learner grading |
| `science-04:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `science-04:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `science-04:practice-1` | Infer a different physical quantity | available-draft | CPU reference checked; reference solution, not learner grading |
| `science-04:practice-2` | Compare noisy inference with an independent search | available-draft | CPU reference checked; reference solution, not learner grading |
| `science-04:practice-3` | Fit and independently audit an adaptive noisy rate estimate | available-draft | CPU reference checked; reference solution, not learner grading |
| `probability-01:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `probability-01:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `probability-01:practice-1` | Detect a missing normalization | available-draft | CPU reference checked; reference solution, not learner grading |
| `probability-01:practice-2` | Demonstrate key ownership | available-draft | CPU reference checked; reference solution, not learner grading |
| `probability-02:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `probability-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `probability-02:practice-1` | Move away from centered inputs | available-draft | CPU reference checked; reference solution, not learner grading |
| `probability-02:practice-2` | Verify total predictive variance | available-draft | CPU reference checked; reference solution, not learner grading |
| `probability-03:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `probability-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `probability-03:practice-1` | Catch chains trapped at different levels | available-draft | CPU reference checked; reference solution, not learner grading |
| `probability-03:practice-2` | Verify a directional derivative | available-draft | CPU reference checked; reference solution, not learner grading |
| `probability-04:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `probability-04:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `probability-04:practice-1` | A correct diagonal is not a correct covariance | available-draft | CPU reference checked; reference solution, not learner grading |
| `probability-04:practice-2` | Repair the changed mean without hiding noise | available-draft | CPU reference checked; reference solution, not learner grading |
| `rl-01:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `rl-01:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `rl-01:practice-1` | Separate the two target meanings | available-draft | CPU reference checked; reference solution, not learner grading |
| `rl-01:practice-2` | Enumerate all one-step moves | available-draft | CPU reference checked; reference solution, not learner grading |
| `rl-02:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `rl-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `rl-02:practice-1` | Catch a shifted mask | available-draft | CPU reference checked; reference solution, not learner grading |
| `rl-02:practice-2` | Check a deterministic policy limit | available-draft | CPU reference checked; reference solution, not learner grading |
| `rl-03:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `rl-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `rl-03:practice-1` | Inspect a frozen behavior reference | available-draft | CPU reference checked; reference solution, not learner grading |
| `rl-03:practice-2` | Change the time budget | available-draft | CPU reference checked; reference solution, not learner grading |
| `rl-04:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `rl-04:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `rl-04:practice-1` | Expose an unequal-batch averaging bug | available-draft | CPU reference checked; reference solution, not learner grading |
| `rl-04:practice-2` | Distinguish spread from standard error | available-draft | CPU reference checked; reference solution, not learner grading |
| `rl-05:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `rl-05:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `rl-05:practice-1` | Find a reset leak | available-draft | CPU reference checked; reference solution, not learner grading |
| `rl-05:practice-2` | Choose the finite-horizon contract | available-draft | CPU reference checked; reference solution, not learner grading |
| `kernels-01:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `kernels-01:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `kernels-01:practice-1` | Reject a broadcast that changes the contract | available-draft | CPU reference checked; reference solution, not learner grading |
| `kernels-01:practice-2` | Measure padding overhead without claiming runtime | available-draft | CPU reference checked; reference solution, not learner grading |
| `kernels-02:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `kernels-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `kernels-02:practice-1` | Reject hidden bias-axis broadcasting | available-draft | CPU reference checked; reference solution, not learner grading |
| `kernels-02:practice-2` | Check a permutation invariant | available-draft | CPU reference checked; reference solution, not learner grading |
| `kernels-03:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `kernels-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `kernels-03:practice-1` | Reject an unsupported pipeline tile | available-draft | CPU reference checked; reference solution, not learner grading |
| `kernels-03:practice-2` | Estimate the price of a third input buffer | available-draft | CPU reference checked; reference solution, not learner grading |
| `kernels-04:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `kernels-04:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `kernels-04:practice-1` | Reject unsupported dtype mixing | available-draft | CPU reference checked; reference solution, not learner grading |
| `kernels-04:practice-2` | Bound the end-to-end opportunity | available-draft | CPU reference checked; reference solution, not learner grading |
| `internals-01:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `internals-01:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `internals-01:practice-1` | Compare unrolled and scanned recurrences | available-draft | CPU reference checked; reference solution, not learner grading |
| `internals-01:practice-2` | Reproduce and repair a traced boolean failure | available-draft | CPU reference checked; reference solution, not learner grading |
| `internals-02:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `internals-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `internals-02:practice-1` | Test linearity at a fixed point | available-draft | CPU reference checked; reference solution, not learner grading |
| `internals-02:practice-2` | Check curvature symmetry without a full Hessian | available-draft | CPU reference checked; reference solution, not learner grading |
| `internals-03:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `internals-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `internals-03:practice-1` | Check common-shift invariance of curvature | available-draft | CPU reference checked; reference solution, not learner grading |
| `internals-03:practice-2` | Audit a nonlinear seed rule before registering it | available-draft | CPU reference checked; reference solution, not learner grading |
| `internals-04:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `internals-04:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `internals-04:practice-1` | Make a constant into a differentiable input | available-draft | CPU reference checked; reference solution, not learner grading |
| `internals-04:practice-2` | Verify multiple outputs and negation | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-02:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `deployment-02:practice-1` | Keep source-task evidence | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-02:practice-2` | Check the loss without autodiff | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-02:practice-3` | Choose under a declared retention constraint | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-01:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-01:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `deployment-01:practice-1` | Detect a preprocessing mismatch | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-01:practice-2` | Map a second layer and locate an omitted activation | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-08:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-08:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `deployment-08:practice-1` | Reject a plausible but wrong tensor layout | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-08:practice-2` | Check a nonuniform output sensitivity | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-03:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `deployment-03:practice-1` | Validate a request before inference | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-03:practice-2` | Prove that request validation precedes exported inference | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-05:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-05:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `deployment-05:practice-1` | Compute W8A8 with a wide accumulator | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-05:practice-2` | Change the code origin without changing the values | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-05:practice-3` | Account for actual low-bit storage | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-07:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-07:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `deployment-07:practice-1` | Reject a nonfinite request | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-07:practice-2` | Check changed behavior after a legitimate artifact replacement | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-04:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-04:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `deployment-04:practice-1` | Expose burst sensitivity | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-04:practice-2` | Reject a broken timeline | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-04:practice-3` | Measure the margin below a response deadline | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-06:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-06:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `deployment-06:practice-1` | Make incomplete deployment evidence explicit | available-draft | CPU reference checked; reference solution, not learner grading |
| `deployment-06:practice-2` | Decide whether a model speedup meets the request budget | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-01:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-01:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `operations-01:practice-1` | Reject invalid work before spawning | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-01:practice-2` | Preserve a successful checkpoint | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-02:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `operations-02:practice-1` | Catch an average-of-rates error | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-02:practice-2` | Change the checkpoint cadence | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-03:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `operations-03:practice-1` | Reject accidental checkpoint corruption | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-03:practice-2` | Explain a weights-only mismatch by hand | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-04:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-04:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `operations-04:practice-1` | Reject an incomplete provenance record | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-04:practice-2` | Check rollback without history | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-05:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-05:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `operations-05:practice-1` | Account for failed attempts | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-05:practice-2` | Reject impossible capacity inputs | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-06:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-06:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `operations-06:practice-1` | Change the population mix | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-07:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-07:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `operations-07:practice-1` | Check abstention structure | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-07:practice-2` | Keep failed work in the evaluation inventory | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-08:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `operations-08:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `operations-08:practice-1` | Review a changed owner | available-draft | CPU reference checked; reference solution, not learner grading |
| `pretraining-01:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `pretraining-01:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `pretraining-01:practice-1` | Reject a missing learning signal | available-draft | CPU reference checked; reference solution, not learner grading |
| `pretraining-01:practice-2` | Derive the output gradient without autodiff | available-draft | CPU reference checked; reference solution, not learner grading |
| `pretraining-02:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `pretraining-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `pretraining-02:practice-1` | Expose the leakage shortcut | available-draft | CPU reference checked; reference solution, not learner grading |
| `pretraining-02:practice-2` | Audit every patch of a rectangular color image | available-draft | CPU reference checked; reference solution, not learner grading |
| `pretraining-03:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `pretraining-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `pretraining-03:practice-1` | Duplicate a source and inspect the objective | available-draft | CPU reference checked; reference solution, not learner grading |
| `pretraining-03:practice-2` | Check the encoder gradient through normalization | available-draft | CPU reference checked; reference solution, not learner grading |
| `posttraining-01:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `posttraining-01:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `posttraining-01:practice-1` | Reject a batch with no answer targets | available-draft | CPU reference checked; reference solution, not learner grading |
| `posttraining-01:practice-2` | Create a failure this model cannot fix | available-draft | CPU reference checked; reference solution, not learner grading |
| `posttraining-02:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `posttraining-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `posttraining-02:practice-1` | Check the merge on new inputs | available-draft | CPU reference checked; reference solution, not learner grading |
| `posttraining-02:practice-2` | Test nonunit scaling and an impossible rank-one target | available-draft | CPU reference checked; reference solution, not learner grading |
| `posttraining-03:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `posttraining-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `posttraining-03:practice-1` | Show that feature scale can change scores without changing ranking | available-draft | CPU reference checked; reference solution, not learner grading |
| `posttraining-03:practice-2` | Derive the linear reward gradient | available-draft | CPU reference checked; reference solution, not learner grading |
| `posttraining-04:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `posttraining-04:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `posttraining-04:practice-1` | Verify detached behavior information | available-draft | CPU reference checked; reference solution, not learner grading |
| `posttraining-04:practice-2` | Compare with the finite-action regularized optimum | available-draft | CPU reference checked; reference solution, not learner grading |
| `posttraining-05:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `posttraining-05:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `posttraining-05:practice-1` | Freeze the reference numerically and in autodiff | available-draft | CPU reference checked; reference solution, not learner grading |
| `posttraining-05:practice-2` | Derive a one-pair categorical gradient | available-draft | CPU reference checked; reference solution, not learner grading |
| `tpu-01:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `tpu-01:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `tpu-01:practice-1` | Check whether a model state fits on a single-host TPU slice | available-draft | CPU reference checked; reference solution, not learner grading |
| `tpu-01:practice-2` | Verify the pmap sum-of-squares formula for 4 TPU devices | available-draft | CPU reference checked; reference solution, not learner grading |
| `tpu-04:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `tpu-04:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `tpu-04:practice-1` | Detect a silent fallback from 4 TPU chips to 1 CPU device | available-draft | CPU reference checked; reference solution, not learner grading |
| `tpu-04:practice-2` | Estimate hourly billing savings from immediate teardown | available-draft | CPU reference checked; reference solution, not learner grading |
| `tpu-02:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `tpu-02:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `tpu-02:practice-1` | Inspect the JSONL event sequence from a resumed run | available-draft | CPU reference checked; reference solution, not learner grading |
| `tpu-02:practice-2` | Verify atomic checkpoint file replacement | available-draft | CPU reference checked; reference solution, not learner grading |
| `tpu-05:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `tpu-05:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `tpu-05:practice-1` | Filter multi-host slices from the catalog | available-draft | CPU reference checked; reference solution, not learner grading |
| `tpu-05:practice-2` | Compute per-host vs global batch size on a 4-host Pod slice | available-draft | CPU reference checked; reference solution, not learner grading |
| `tpu-03:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `tpu-03:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `tpu-03:practice-1` | Compare arithmetic intensity against the TPU v5e and v6e ridge points | available-draft | CPU reference checked; reference solution, not learner grading |
| `tpu-03:practice-2` | Size a Trillium v6e-4 slice for a 7B training state | available-draft | CPU reference checked; reference solution, not learner grading |
| `tpu-06:exercise` | Main exercise | available-draft | CPU reference checked; reference solution, not learner grading |
| `tpu-06:checkpoint` | Understanding checkpoint | available-draft | Authored answer and rationale; uncalibrated checkpoint |
| `tpu-06:practice-1` | Compare per-channel vs per-tensor INT8 quantization error | available-draft | CPU reference checked; reference solution, not learner grading |
| `tpu-06:practice-2` | Verify non-empty XProf *.xplane.pb byte size | available-draft | CPU reference checked; reference solution, not learner grading |
| `regression-audit:stage-1` | Prediction and objective | available-draft | Public reference stage checked; no independent learner assessment |
| `regression-audit:stage-2` | Gradient evidence | available-draft | Public reference stage checked; no independent learner assessment |
| `regression-audit:stage-3` | Compiled training | available-draft | Public reference stage checked; no independent learner assessment |
| `regression-audit:stage-4` | Training audit | available-draft | Public reference stage checked; no independent learner assessment |
| `mlp-classifier:stage-1` | Model and shape contract | available-draft | Public reference stage checked; no independent learner assessment |
| `mlp-classifier:stage-2` | Objective and derivative evidence | available-draft | Public reference stage checked; no independent learner assessment |
| `mlp-classifier:stage-3` | Training audit | available-draft | Public reference stage checked; no independent learner assessment |
| `scientific-inverse:stage-1` | Simulation and numerical references | available-draft | Public reference stage checked; no independent learner assessment |
| `scientific-inverse:stage-2` | Objective and solver sensitivity | available-draft | Public reference stage checked; no independent learner assessment |
| `scientific-inverse:stage-3` | Parameter recovery and an inference report | available-draft | Public reference stage checked; no independent learner assessment |
| `policy-evaluation:stage-1` | Functional environment | available-draft | Public reference stage checked; no independent learner assessment |
| `policy-evaluation:stage-2` | Batched trajectories | available-draft | Public reference stage checked; no independent learner assessment |
| `policy-evaluation:stage-3` | Policy objectives | available-draft | Public reference stage checked; no independent learner assessment |
| `policy-evaluation:stage-4` | Training and held-out evaluation | available-draft | Public reference stage checked; no independent learner assessment |
| `policy-evaluation:stage-5` | Value learning and generalized advantage estimation | available-draft | Public reference stage checked; no independent learner assessment |
| `bayesian-regression:stage-1` | Exact posterior and density | available-draft | Public reference stage checked; no independent learner assessment |
| `bayesian-regression:stage-2` | Sampler and diagnostic audit | available-draft | Public reference stage checked; no independent learner assessment |
| `bayesian-regression:stage-3` | Predictive uncertainty and mismatch | available-draft | Public reference stage checked; no independent learner assessment |
| `derivative-audit:stage-1` | Directional products and exact curvature | available-draft | Public reference stage checked; no independent learner assessment |
| `derivative-audit:stage-2` | Exact custom derivative contracts | available-draft | Public reference stage checked; no independent learner assessment |
| `derivative-audit:stage-3` | A compiled primitive-level transformation | available-draft | Public reference stage checked; no independent learner assessment |
| `deployment-audit:stage-1` | Trained artifact and adaptation provenance | available-draft | Public reference stage checked; no independent learner assessment |
| `deployment-audit:stage-2` | Export and precision contract | available-draft | Public reference stage checked; no independent learner assessment |
| `deployment-audit:stage-3` | Request timing and capacity evidence | available-draft | Public reference stage checked; no independent learner assessment |
| `workload-operations:stage-1` | Real process lifecycle | available-draft | Public reference stage checked; no independent learner assessment |
| `workload-operations:stage-2` | Workload observability | available-draft | Public reference stage checked; no independent learner assessment |
| `workload-operations:stage-3` | Complete-state recovery | available-draft | Public reference stage checked; no independent learner assessment |
| `workload-operations:stage-4` | Artifact selection and rollback | available-draft | Public reference stage checked; no independent learner assessment |
| `workload-operations:stage-5` | Capacity and cost | available-draft | Public reference stage checked; no independent learner assessment |
| `audio-harness:stage-1` | Waveform and spectral contract | available-draft | Public reference stage checked; no independent learner assessment |
| `audio-harness:stage-2` | Training, recovery and held-out errors | available-draft | Public reference stage checked; no independent learner assessment |
| `audio-harness:stage-3` | Precision and complete exported artifacts | available-draft | Public reference stage checked; no independent learner assessment |
| `audio-harness:stage-4` | Measured inference and boundary drills | available-draft | Public reference stage checked; no independent learner assessment |
| `cross-modal-harness:stage-1` | Pair and preprocessing contracts | available-draft | Public reference stage checked; no independent learner assessment |
| `cross-modal-harness:stage-2` | Symmetric contrastive objective | available-draft | Public reference stage checked; no independent learner assessment |
| `cross-modal-harness:stage-3` | Training and complete recovery | available-draft | Public reference stage checked; no independent learner assessment |
| `cross-modal-harness:stage-4` | Precision and exported branches | available-draft | Public reference stage checked; no independent learner assessment |
| `cross-modal-harness:stage-5` | Inference, measured release and rollback | available-draft | Public reference stage checked; no independent learner assessment |
| `kernel-audit:stage-1` | Grid and precision contract | available-draft | Public reference stage checked; no independent learner assessment |
| `kernel-audit:stage-2` | Fused target-oriented kernel | available-draft | Public reference stage checked; no independent learner assessment |
| `kernel-audit:stage-3` | Pipeline semantics and measurement protocol | available-draft | Public reference stage checked; no independent learner assessment |
| `image-harness:stage-1` | Image data and preprocessing | available-draft | Public reference stage checked; no independent learner assessment |
| `image-harness:stage-2` | Transparent CNN baseline | available-draft | Public reference stage checked; no independent learner assessment |
| `image-harness:stage-3` | Actual augmented training | available-draft | Public reference stage checked; no independent learner assessment |
| `image-harness:stage-4` | Full-state recovery | available-draft | Public reference stage checked; no independent learner assessment |
| `image-harness:stage-5` | Frozen evaluation and errors | available-draft | Public reference stage checked; no independent learner assessment |
| `image-harness:stage-6` | Calibrated precision | available-draft | Public reference stage checked; no independent learner assessment |
| `image-harness:stage-7` | Serialized inference artifact | available-draft | Public reference stage checked; no independent learner assessment |
| `image-harness:stage-8` | Measured serving adapter | available-draft | Public reference stage checked; no independent learner assessment |
| `foundation-toolkit:stage-1` | Report the completed environment | available-draft | Public reference stage checked; no independent learner assessment |
| `foundation-toolkit:stage-2` | Process observations with a mask | available-draft | Public reference stage checked; no independent learner assessment |
| `foundation-toolkit:stage-3` | Compile a batch of gradients | available-draft | Public reference stage checked; no independent learner assessment |
| `foundation-toolkit:stage-4` | Replay a batched simulation | available-draft | Public reference stage checked; no independent learner assessment |
| `sharded-training:stage-1` | State and input ownership | available-draft | Public reference stage checked; no independent learner assessment |
| `sharded-training:stage-2` | Count-weighted sharded updates | available-draft | Public reference stage checked; no independent learner assessment |
| `sharded-training:stage-3` | Fresh-process complete recovery | available-draft | Public reference stage checked; no independent learner assessment |
| `sharded-training:stage-4` | Trace-backed performance evidence | available-draft | Public reference stage checked; no independent learner assessment |
| `text-harness:stage-1` | Tokens, causal computation and loss | available-draft | Public reference stage checked; no independent learner assessment |
| `text-harness:stage-2` | Actual training and full recovery | available-draft | Public reference stage checked; no independent learner assessment |
| `text-harness:stage-3` | Real KV-cache precision | available-draft | Public reference stage checked; no independent learner assessment |
| `text-harness:stage-4` | Exported inference and profiler evidence | available-draft | Public reference stage checked; no independent learner assessment |
| `optimizer-audit:stage-1` | Derive and check the local geometry | available-draft | Public reference stage checked; no independent learner assessment |
| `optimizer-audit:stage-2` | Change coordinates without changing predictions | available-draft | Public reference stage checked; no independent learner assessment |
| `optimizer-audit:stage-3` | Measure noise by enumerating the sampler | available-draft | Public reference stage checked; no independent learner assessment |
| `optimizer-audit:stage-4` | Audit optimizer state and update order | available-draft | Public reference stage checked; no independent learner assessment |
| `optimizer-audit:stage-5` | Separate prediction and coefficient stability | available-draft | Public reference stage checked; no independent learner assessment |
| `engineering-release:stage-1` | Data identity and slice gates | available-draft | Public reference stage checked; no independent learner assessment |
| `engineering-release:stage-2` | LLM application evaluation | available-draft | Public reference stage checked; no independent learner assessment |
| `engineering-release:stage-3` | Model ownership and release transitions | available-draft | Public reference stage checked; no independent learner assessment |
| `engineering-release:stage-4` | Tracked training to inference | available-draft | Public reference stage checked; no independent learner assessment |
| `training-methods:stage-1` | Masked language modeling | available-draft | Public reference stage checked; no independent learner assessment |
| `training-methods:stage-2` | Masked image reconstruction | available-draft | Public reference stage checked; no independent learner assessment |
| `training-methods:stage-3` | Paired contrastive learning | available-draft | Public reference stage checked; no independent learner assessment |
| `training-methods:stage-4` | Response-only SFT | available-draft | Public reference stage checked; no independent learner assessment |
| `training-methods:stage-5` | Frozen-base LoRA | available-draft | Public reference stage checked; no independent learner assessment |
| `training-methods:stage-6` | Preference reward model | available-draft | Public reference stage checked; no independent learner assessment |
| `training-methods:stage-7` | Learned-reward PPO | available-draft | Public reference stage checked; no independent learner assessment |
| `training-methods:stage-8` | Direct preference optimization | available-draft | Public reference stage checked; no independent learner assessment |
| `weight-conversion:stage-1` | Map and compare real frameworks | available-draft | Public reference stage checked; no independent learner assessment |
| `weight-conversion:stage-2` | Locate numerical disagreement | available-draft | Public reference stage checked; no independent learner assessment |
| `weight-conversion:stage-3` | Reload converted weights independently | available-draft | Public reference stage checked; no independent learner assessment |
| `welcome-audit:stage-1` | Contract and independent calculation | available-draft | Public reference stage checked; no independent learner assessment |
| `welcome-audit:stage-2` | Changed condition and diagnosis | available-draft | Public reference stage checked; no independent learner assessment |
| `welcome-audit:stage-3` | Retained evidence | available-draft | Public reference stage checked; no independent learner assessment |
| `welcome-audit:stage-4` | Integrated project check | available-draft | Public reference stage checked; no independent learner assessment |
| `arrays-audit:stage-1` | Contract and independent calculation | available-draft | Public reference stage checked; no independent learner assessment |
| `arrays-audit:stage-2` | Changed condition and diagnosis | available-draft | Public reference stage checked; no independent learner assessment |
| `arrays-audit:stage-3` | Retained evidence | available-draft | Public reference stage checked; no independent learner assessment |
| `arrays-audit:stage-4` | Integrated project check | available-draft | Public reference stage checked; no independent learner assessment |
| `transforms-audit:stage-1` | Contract and independent calculation | available-draft | Public reference stage checked; no independent learner assessment |
| `transforms-audit:stage-2` | Changed condition and diagnosis | available-draft | Public reference stage checked; no independent learner assessment |
| `transforms-audit:stage-3` | Retained evidence | available-draft | Public reference stage checked; no independent learner assessment |
| `transforms-audit:stage-4` | Integrated project check | available-draft | Public reference stage checked; no independent learner assessment |
| `recovery-audit:stage-1` | Contract and independent calculation | available-draft | Public reference stage checked; no independent learner assessment |
| `recovery-audit:stage-2` | Changed condition and diagnosis | available-draft | Public reference stage checked; no independent learner assessment |
| `recovery-audit:stage-3` | Retained evidence | available-draft | Public reference stage checked; no independent learner assessment |
| `recovery-audit:stage-4` | Integrated project check | available-draft | Public reference stage checked; no independent learner assessment |
| `performance-audit:stage-1` | Contract and independent calculation | available-draft | Public reference stage checked; no independent learner assessment |
| `performance-audit:stage-2` | Changed condition and diagnosis | available-draft | Public reference stage checked; no independent learner assessment |
| `performance-audit:stage-3` | Retained evidence | available-draft | Public reference stage checked; no independent learner assessment |
| `performance-audit:stage-4` | Integrated project check | available-draft | Public reference stage checked; no independent learner assessment |
| `posttraining-audit:stage-1` | Contract and independent calculation | available-draft | Public reference stage checked; no independent learner assessment |
| `posttraining-audit:stage-2` | Changed condition and diagnosis | available-draft | Public reference stage checked; no independent learner assessment |
| `posttraining-audit:stage-3` | Retained evidence | available-draft | Public reference stage checked; no independent learner assessment |
| `posttraining-audit:stage-4` | Integrated project check | available-draft | Public reference stage checked; no independent learner assessment |
| `tpu-runtime-audit:stage-1` | Contract and independent calculation | available-draft | Public reference stage checked; no independent learner assessment |
| `tpu-runtime-audit:stage-2` | Changed condition and diagnosis | available-draft | Public reference stage checked; no independent learner assessment |
| `tpu-runtime-audit:stage-3` | Retained evidence | available-draft | Public reference stage checked; no independent learner assessment |
| `tpu-runtime-audit:stage-4` | Integrated project check | available-draft | Public reference stage checked; no independent learner assessment |
| `tpu-workflow-audit:stage-1` | Contract and independent calculation | available-draft | Public reference stage checked; no independent learner assessment |
| `tpu-workflow-audit:stage-2` | Changed condition and diagnosis | available-draft | Public reference stage checked; no independent learner assessment |
| `tpu-workflow-audit:stage-3` | Retained evidence | available-draft | Public reference stage checked; no independent learner assessment |
| `tpu-workflow-audit:stage-4` | Integrated project check | available-draft | Public reference stage checked; no independent learner assessment |
| `tpu-systems-audit:stage-1` | Contract and independent calculation | available-draft | Public reference stage checked; no independent learner assessment |
| `tpu-systems-audit:stage-2` | Changed condition and diagnosis | available-draft | Public reference stage checked; no independent learner assessment |
| `tpu-systems-audit:stage-3` | Retained evidence | available-draft | Public reference stage checked; no independent learner assessment |
| `tpu-systems-audit:stage-4` | Integrated project check | available-draft | Public reference stage checked; no independent learner assessment |
| `reproducible-experiments:stage-1` | Freeze the experimental question and configuration | available-draft | Public reference stage checked; no independent learner assessment |
| `reproducible-experiments:stage-2` | Log complete post-update observations | available-draft | Public reference stage checked; no independent learner assessment |
| `reproducible-experiments:stage-3` | Compare complete paired seed results | available-draft | Public reference stage checked; no independent learner assessment |
| `reproducible-experiments:stage-4` | Execute, diagnose and write the research notebook | available-draft | Public reference stage checked; no independent learner assessment |
| `embedding-lineage:stage-1` | Masked-token pretraining | in-development | Public reference stage checked; no independent learner assessment |
| `embedding-lineage:stage-2` | Contrastive continuation | in-development | Public reference stage checked; no independent learner assessment |
| `embedding-lineage:stage-3` | Supervised query-positive adaptation | in-development | Public reference stage checked; no independent learner assessment |
| `embedding-lineage:stage-4` | Held-out retrieval and export | in-development | Public reference stage checked; no independent learner assessment |
| `causal-lineage:stage-1` | Causal next-token pretraining | in-development | Public reference stage checked; no independent learner assessment |
| `causal-lineage:stage-2` | Instruction-style continuation, held-out perplexity and export | in-development | Public reference stage checked; no independent learner assessment |

## Source-bound hardware evidence

These records qualify their declared experiments only; they do not certify an entire lesson, generation, production model or career path.

- [v5e1-training-recovery-precision-20261020](../docs/evidence/tpu-qualification-2026-10-20/v5e1/smoke-v1/execution-tpu.json): One v5e chip: bounded training, resume and single-device dot/precision checks; no multi-host or model-quality qualification. Source hashes match.
- [v5e1-pallas-20261020](../['docs/evidence/tpu-qualification-2026-10-20/v5e1/pallas-v2/fused-float32.json', 'docs/evidence/tpu-qualification-2026-10-20/v5e1/pallas-v2/fused-bfloat16.json', 'docs/evidence/tpu-qualification-2026-10-20/v5e1/pallas-v2/pipeline-float32.json', 'docs/evidence/tpu-qualification-2026-10-20/v5e1/pallas-v2/pipeline-bfloat16.json']): One v5e chip, two kernels, FP32/BF16, logical shape257x513 and five timing samples; not a general speed or scaling claim. Source hashes match.
- [v5e4-collectives-20261020](../docs/evidence/tpu-qualification-2026-10-20/v5e4/collectives-v1/execution-collectives.json): One v5e host with four devices: deterministic psum, reduce-scatter and all-gather numerical, placement, StableHLO and trace checks; not multi-host or scaling qualification. Source hashes match.
- [v5e16-multihost-checkpoint-20261020](../['docs/evidence/tpu-qualification-2026-10-20/v5e16/worker-0/multihost-v4/process-000/execution-multihost.json', 'docs/evidence/tpu-qualification-2026-10-20/v5e16/worker-1/multihost-v4/process-002/execution-multihost.json', 'docs/evidence/tpu-qualification-2026-10-20/v5e16/worker-2/multihost-v4/process-001/execution-multihost.json', 'docs/evidence/tpu-qualification-2026-10-20/v5e16/worker-3/multihost-v4/process-003/execution-multihost.json']): Four VM controllers and sixteen v5e devices: process-local synthetic data, global updates, Orbax GCS save at step4, same-process abstract sharded restore and replay through step8. No worker failure, fresh-process restart, elastic recovery, production quality or scaling measurement. Source hashes match.

## Limits and next review

All canonical sources, task prompts, stage contracts, companion existence and stored source hashes were inventoried. Per-phase scope judgments are recorded in curriculum/content-review.json. Technical close reading and visual inspection are selective; no new human study, paid accelerator run or full native-reader qualification was performed. The two corrected recovery captions were followed by fresh CPU script/notebook runs and regenerated figures. Other numerical checks come from source-matching receipts and the completed 0.8.2 deployment, rather than new executions during this audit. Wrong-choice feedback, realistic datasets, target-specific outcomes and human review need further work.

Do not solve these gaps by inflating word counts, randomizing quiz choices, or labelling public fixture passes as professional readiness. Start with one contribution and retain inspectable evidence.
