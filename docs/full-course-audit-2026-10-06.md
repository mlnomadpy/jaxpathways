# Full curriculum audit — 2026-10-06

The course has a working foundation: **19 phases, 97 authored lessons, 19 registered projects, 10 pathways and 7 career guides**. The next priority is to connect that material to realistic datasets and complete model lifecycles. Many requested subjects already have an executable introduction; adding another introductory lesson would miss the remaining gap.

The proposed work is specified in the [course roadmap](course-roadmap-2026-10-06.md). It is **planned work**, not additional authored lessons or a claim that the existing material is production-qualified.

## Scope and method

Reviewed the canonical phase and lesson inventory, lesson teaching structures and exercises, all project manifests and README scope statements, all route and career mappings, modality guides, assessment coverage, tutor instructions, generation/checking contracts, and previous audit reports. Closer implementation inspection focused on the four modality harnesses, objective labs, conversion, precision and engineering boundaries. The [inventory](course-audit-inventory-2026-10-06.json) records all 97 lesson sources, content hashes, project stages, route membership and receipt matching.

This is a complete **curriculum coverage and progression audit**, with selective technical inspection. It is not a line-by-line independent proof of every program or mathematical statement. Existing numerical execution receipts were checked against current sources; numerical suites were not rerun because this work changes planning documents only. No live browser, learner trial, external reviewer, new dataset experiment, accelerator, edge-device or cloud qualification was performed.

## Additional capstone requirements

The subsequent [model lifecycle capstone catalog](model-lifecycle-capstones.md) addresses the missing connection between method labs and entire model-development programs. Seven specifications cover embeddings, causal language, image/audio understanding, image/audio generation and grounded multimodal generation, with data recipes, objective transitions, benchmark/monitoring protocols, consumer packaging and Hugging Face publication. The [tooling plan](model-lifecycle-tooling.md) adds W&B alongside MLflow and scoped ecosystem tools. These plans do not change the verified counts below or establish trained production models.

## What is already present

- Every canonical lesson has an authored source and current matching CPU script/notebook receipt: **97/97**. All **19/19** registered project source hashes match their passed reference receipts.
- **92 executed figures and 5 conceptual diagrams** are represented in the sources. Plot presence does not establish that a learner understands it; prior interpretation reviews and their limitations remain relevant.
- All four modality harnesses actually implement training, recovery, evaluation, precision, export and measured local inference. Their core demonstrations use controlled synthetic data.
- MLM, masked image reconstruction, paired contrastive learning, response-only SFT, LoRA, reward modeling, PPO-based preference optimization and DPO exist. Their small models deliberately isolate the objective.
- Actual PyTorch-to-Flax conversion exists for a dense–LayerNorm–GELU–dense network. Keras JAX/TensorFlow/PyTorch backend labs and desktop LiteRT FP32/INT8 conversion have separate recorded evidence.
- MLflow, Docker, MLOps, LLMOps and ModelOps exist. The engineering project includes real tracked training, registry retrieval, a local service and separately recorded Docker qualification.
- The text harness already has genuine prefill and one-token KV-cache decoding, FP32/BF16/INT8 cache storage, and serialized inference. **Do not schedule “add a KV cache” as if absent.**
- Eleven synthesis assessments are review drafts. Their public reference tests and review rubrics are useful, but do not constitute independent learner assessment or a credential.

Sources: [course manifest](../curriculum/course.json), [project registry](../curriculum/projects.json), [lesson receipts](../curriculum/validation.json), [project receipts](../curriculum/project-validation.json), [framework lab receipt](deployment-lab-validation.json), [engineering evidence](engineering-operations-expansion.md), [latest objective expansion](pretraining-posttraining-expansion.md).

## Highest-priority findings

| Priority | Finding and evidence | Consequence | Planned resolution |
| --- | --- | --- | --- |
| P0 | Objective lessons and complete harnesses are separate. `training-methods` explicitly leaves connecting its objectives to the modality harnesses as follow-up. | A learner can pass an SFT mask or LoRA algebra check without adapting a saved Transformer. | TEXT + ADAPT: reuse one model, tokenizer, dataset identity, checkpoint and evaluator through pretraining and adaptation. |
| P0 | Natural-data workflows are mostly loader contracts. Text is short synthetic reversal data with a 12-token inference context; vision uses bars; audio uses generated tones; retrieval uses four concepts. | Correct plumbing does not establish data curation, realistic model quality or generalization. | DATA + EVAL first; then one reproducible modest real-data run per modality. Preserve the synthetic fixtures as correctness oracles. |
| P0 | `models` contains **69 lessons** and mandates Transformers, pretraining, RL and post-training while its capstone is an image classifier. `tpu` contains **73 lessons**. | The route has expanded beyond its advertised outcome; image learners take unrelated prerequisites. | Route repair before expansion: required core plus explicit modality/method branches; retain existing lesson IDs and progress. |
| P0 | `posttraining` has phase prerequisites `pretraining` and `rl`, applying RL preparation even to SFT and LoRA. | Method prerequisites are conflated with an entire phase. | Separate supervised adaptation from preference/RL learning in route composition. RL becomes required for sequence PPO, not ordinary SFT. |
| P0 | Career skill lists outpace milestone evidence. Training mentions pre/post-training and parity but milestones stop at performance; research names NumPyro while its default route/milestone is the tiny internals audit. | Career pages imply broader practice than the compulsory artifacts demonstrate. | Map each skill to a lesson, artifact and review criterion; label library extensions and target qualification explicitly. |
| P1 | Engineering extensions on **9/10 routes** contain lessons outside the main route. The download tells learners to follow their prerequisites, but the extra dependency work is not part of the route list. | A beginner can arrive at operations material requiring distributed training without seeing the full detour. | Expose a prerequisite-resolved optional branch and its first useful artifact. Do not silently append all operations to every route. |
| P1 | Precision covers meaningful arithmetic and some real runtime paths, but packed INT4/QLoRA/QAT/FP8 training and hardware execution are not complete. | “Supports 8-bit” can be misread as native full-model low-bit training or acceleration. | NUM + SERVE: publish per-tensor precision policies, actual storage/kernel evidence and device-specific qualification. |
| P1 | Serving lessons contain genuine export/local calls plus queue simulation; the engineering service is a bounded demonstration. | Concurrent streaming, cancellation, backpressure, load-driven tail latency and real rollout behavior remain untested. | SERVE + OPS with a local concurrent service, load generator and failure drills before any cloud extension. |
| P1 | Lesson/project structure is checked automatically; independent learner transfer, onboarding and review reliability are unmeasured. | More authored pages can increase effort without increasing understanding. | Changed-data capstones, explicit assessment coverage, small learner pilots and revised explanations based on observed errors. |

## Phase-by-phase audit

“Extend” means the existing teaching remains useful. “Qualify” means the capability needs new observed evidence rather than simply another explanation. Roadmap codes identify the work package.

| Existing phase | Lessons | Existing strength | Remaining gap and action |
| --- | ---: | --- | --- |
| Setup & first steps | 4 | Saved programs, interpreter/device reports, logical CPU placement and explicit TPU target check | Qualify fresh-machine setup on supported OSes; introduce accelerator placement when needed rather than blocking an ordinary CPU start. ROUTE. |
| Arrays & pure functions | 4 | Shapes, broadcasting, fitted preprocessing, immutable updates and independent references | Add padding/length masks for variable-length data in DATA/TEXT/AUDIO; sparse and ragged representations remain an elective. |
| JAX transformations | 5 | Derivative checks, batching, compilation and tracing | Extend composition examples to realistic state and nonsmooth boundaries; revisit these concepts within the model work rather than duplicating introductions. RESEARCH. |
| State, randomness & control flow | 4 | Explicit keys, trees, loops and branches | Carry randomness through augmentation and distributed data ownership; asynchronous/concurrent behavior belongs in recovery/SCALE. |
| Math & optimization | 12 | Geometry, conditioning, derivatives, stable objectives, sampling, Adam and independent numerical checks | Statistical comparison/calibration is still fragmented. Add EVAL; offer constrained/proximal optimization later. Do not add more scalar descent examples first. |
| Neural network training | 5 | Actual NNX training, transparent CNN and train/eval separation | Natural images, controlled augmentations, transfer and dense prediction need VISION. Add a model-state comparison with dropout/normalization where relevant. |
| Data & checkpoint recovery | 6 | Grain, Orbax, full-state replay, async publication and MLflow | Corpus preparation, streaming shards, data mixtures and multi-writer/network failure are incomplete. DATA then SCALE/OPS. |
| Transformers & language models | 4 | Causal attention, masks, packing, actual tiny training and generation | Tokenizer study, realistic context/data, complete encoder/decoder architecture and downstream evaluation need TEXT. RoPE/GQA belong here as extensions, not prerequisites to the first simple decoder. |
| Performance diagnosis | 5 | Synchronized measurements, actual traces, compiler/roofline reasoning and rematerialization | Qualification on real accelerators and sustained workloads is missing. SCALE; distinguish compiler estimates, allocated bytes and measured peak memory. |
| Distributed training | 4 | Count-weighted updates, meshes, collectives and CPU recovery | Real multi-host ownership/recovery, tensor/FSDP/context parallelism and checkpoint resharding need SCALE. Logical devices do not qualify network behavior. |
| Scientific computing | 4 | ODE convergence, sensitivities and inverse problems | Adaptive/stiff solvers, events, uncertain measurements and PDE/domain validation need a specialist RESEARCH branch. |
| Probabilistic modeling | 4 | Known posterior, instructional HMC/VI and predictive checks | Executed NumPyro hierarchical inference, modern diagnostics and posterior predictive failure studies need RESEARCH. |
| Reinforcement learning | 4 | Functional environment, real updates, rollout masks and seeded evaluation | Learned value functions, GAE, timeout bootstrapping, continuous actions and larger environments need PREF plus an RL elective. |
| Pallas kernels | 4 | Actual interpreted kernels, ownership/precision checks and executable target path | Real device compilation, races, synchronization and end-to-end benefit need SCALE qualification. More interpreted examples alone will not supply it. |
| Autodiff & JAX internals | 4 | Jaxpr, JVP/VJP, custom rules and a bounded interpreter | Transformation composition, effects and supported extension interfaces need a focused RESEARCH elective. |
| Deployment, interoperability & edge AI | 8 | Actual export, dense framework parity, affine quantization, containers and desktop edge conversion | Full architecture mappings, native low-bit runtime support and device measurements need BRIDGE/NUM/SERVE. Split navigation by adaptation, portability, precision and serving. |
| Workload operations | 8 | Local failure/recovery, lineage, release gates, MLflow traces and ModelOps | Real traffic, canary/shadow rollout, delayed labels, authenticated review, concurrency and incident/SLO drills need OPS. |
| Self-supervised pretraining | 3 | Executed MLM, hidden-patch reconstruction and paired embedding objectives | Full encoders, augmentations, dataset-scale negatives and downstream probes need TEXT/VISION/MULTI. Reconstruction or contrastive loss is not downstream quality. |
| Post-training | 5 | Correct objective bookkeeping, trained dense LoRA, reward/PPO/DPO mechanisms | Saved Transformer SFT/LoRA, preference data, sequence PPO/value models, forgetting and quantized adaptation need ADAPT/PREF/NUM. |

## Modality coverage and next evidence

| Harness | What actually exists | Next learner deliverable | Evidence required before calling the extension complete |
| --- | --- | --- | --- |
| [Text](../projects/text-harness/README.md) | One-block, two-head byte Transformer; reversal strings; recovery; cache policies; prefill/decode exports | A modest real-corpus model reused for SFT and LoRA | Document-disjoint splits, tokenizer identity, base/downstream evaluation, restored training, per-layer conversion parity and saved-model service calls |
| [Image](../projects/image-harness/README.md) | CNN on synthetic images; raw-image adapter; augmentation; recovery; calibrated integer reference; export | A natural-image classifier, then a ViT masked-pretraining/probe branch | Group-disjoint real data, augmentation ablation, class errors, random-init/frozen-probe/full-FT comparisons, preprocessing parity |
| [Audio](../projects/audio-harness/README.md) | PCM/STFT features; tone classifier; real WAV loader; recovery; precision; waveform inference | A real recording classifier, then temporal recognition | Speaker/recording split, sample-rate/channel conventions, resampling and mel checks, duration/noise slices, valid-time masks and eventually CTC/WER |
| [Cross-modal](../projects/cross-modal-harness/README.md) | Linear image/text embeddings over four semantic concepts; grouped positives; retrieval; both exported branches | Natural image–caption retrieval; later audio–text and conditioned generation | Pair provenance, multi-positive identities, frozen retrieval pool, Recall@K in both directions, hard negatives, missing-modality behavior |

The loaders are useful existing work. Do not call real-data ingestion wholly absent. What is missing is an executed, documented learning experiment on an appropriate real dataset and the associated model/data changes. The cross-modal parser intentionally accepts only two-word orientation/thickness descriptions; unrestricted captions require a new encoder, not a relaxed validator.

## Pathway and career alignment

| Route | Current issue or boundary | Proposed outcome and branch |
| --- | --- | --- |
| foundations | 29 lessons; study advice gives a shorter math sequence but the route lists the full phase | First regression artifact, with deeper math checkpoints exposed as a clearly explained continuation |
| models | Image capstone after mandatory text/RL/post-training | Shared training core → image **or** text → relevant adaptation → deployment; assess each chosen branch |
| tpu | 73 lessons; CPU text evidence plus unqualified target work | Shared text lifecycle → systems scaling → explicit target qualification; RLHF optional |
| ship | Broad deployment phase; engineering extensions outside the route | Saved model → parity → precision → exported service → release evidence; modality-specific adapter branch |
| ops | Core artifact is local process operation | Shared engineering foundation → local service incidents → optional cluster operation; separate these milestones |
| scale | Logical devices and interpreted kernels | Reference correctness → device profiling → distributed ownership/recovery → measured target optimization |
| rl | Tabular policy capstone | Tabular fundamentals → value/GAE → environment benchmark; optional preference optimization |
| probability | Good exact-reference introduction; NumPyro is optional | Exact model → executable library inference → hierarchical predictive audit |
| science | Scalar cooling inverse problem | Known-system audit → adaptive solver → domain-specific inverse problem and uncertainty |
| internals | Small derivative/interpreter artifact | Verified transform → composition/failure cases → supported extension task |

| Career | Additional demonstrable milestone required |
| --- | --- |
| ML & training engineer | Curated dataset, complete pretraining/adaptation experiment, honest comparison and checkpoint identity. Keep ordinary ML and LLM specialization choices visible. |
| Research engineer | Reproduction with a baseline, ablation, repeated-run uncertainty and an independent evaluation; explicitly optional NumPyro or model-method branch. |
| Scientific ML engineer | Solver convergence, measurement/noise model, identifiability and held-out physical checks for a less trivial system. |
| RL engineer | Value/advantage correctness, multi-seed training, held-out task variations and timeout/termination diagnosis. |
| Performance engineer | Profile-guided improvement under a fixed numerical contract on a named actual target, plus an unchanged or worse result honestly retained. |
| Inference engineer | Architecture parity, precision-quality tradeoff, concurrency/cancellation, measured service tail latency and a separate edge qualification if chosen. |
| Operations engineer | Immutable release, observed rollout/rollback, monitoring with delayed labels, incident evidence, access boundaries and recovery under failure. |

Keep careers discoverable in navigation. Career descriptions should describe work and point to evidence, without turning lesson availability or public checker passes into job-readiness claims.

## Teaching and learner-experience gaps

1. **Teach the figure's question.** For each new figure name axes/units/population, show a representative value, explain the mechanism, and state a counterexample or limitation. Include errors, data examples, calibration plots, attention masks, retrieval grids and service timelines—not only decreasing loss curves.
2. **Keep a continuous artifact.** Each next lesson should load the preceding lesson's output and say which file/function changes. A sequence of unrelated complete scripts leaves integration work to the learner.
3. **Use friendly mathematical explanations.** Concrete example → worked numbers → defined KaTeX notation → code → changed condition. Inline mathematical expressions belong in KaTeX too; executable code remains code. Define vocabulary on first use.
4. **Reduce route effort.** Offer visible build-goal links, a recommended next action and prerequisite explanations. Preserve the user's preference for avoiding mandatory dropdowns/text forms. Advanced branches can be optional disclosures or linked guides.
5. **Keep tutor entry useful.** Preserve the existing `npx skills add . --skill start-learning learn-jax jax-course-guide check-jax` workflow and downloadable workspace. Verify exported plans resolve new prerequisites and preserve `LEARNING.md`; do not imply every chatbot can execute local tools.
6. **Assessment must match claims.** Add separate text/adaptation and modality extension rubrics rather than considering an image capstone sufficient. Use changed data/architecture, an injected fault and a reasoned interpretation. Solutions and public checks remain formative.
7. **Test learning, not just formatting.** Observe a small pilot of beginners and experienced learners completing a coherent slice. Record stalls, misconceptions, hint use and successful transfer. Existing minutes are estimates; do not claim validated completion time.
8. **Keep audit history legible.** Older reports still contain superseded counts and “missing” projects. Their history should remain available, with this dated audit and roadmap as the current planning entry points.

## Verification in this audit

- Fresh `npm run build`: passed, **64 pages**.
- Fresh `npm run check`: passed, **56 tests**, all 19 phase-guide checks, **2,785 local references**, curriculum, types, formatting, lint, workspace/project downloads and 97-chapter offline/EPUB checks.
- Recomputed current canonical content hashes: **97/97** match the recorded passed script/notebook results dated `2026-10-06T01:27:51.059573+00:00`.
- Recomputed complete registered project file hashes using the reference runner's hashing convention: **19/19** match the passed project receipt dated `2026-10-06T01:27:02.045848+00:00`.
- Fresh `npm run audit:content` and `npm run audit:visuals`: passed; one setup walkthrough plus 96 lessons with teaching structure, 92 executed plots and five conceptual diagrams. These inspect structure and provenance, not learner comprehension.
- No numerical lesson/project edits; no fresh CPU numerical runs, new Docker run, deployment, browser inspection or paid infrastructure operation.

The source-bound evidence supports preserving the current material while expanding it. The roadmap's first release should demonstrate the missing integration on one manageable text lifecycle before multiplying the same promises across modalities.
