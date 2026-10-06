# Course expansion plan — 2026-10-06

Status: **proposed curriculum plan; lessons below are not yet authored**. This plan follows the [full audit](full-course-audit-2026-10-06.md) and preserves the existing 97 lessons, 19 projects and learner progress.

Build one connected learning journey first:

**Real dataset → complete small Transformer → pretraining → held-out evaluation → full fine-tuning versus LoRA → framework conversion → precision-aware inference → tracked local release.**

Then apply the same engineering discipline to images, audio and cross-modal systems. Extend shared projects and teaching rather than cloning a complete training stack for each method. MaxText is an advanced reference to study after the small system is understandable; it is not the beginner runtime.

## Priorities and delivery boundaries

There are **61 candidate lesson units across 16 work packages** below. These are teaching scopes, not 61 promised new pages: some should become substantial extensions of existing lessons. **Fourteen units form the proposed first release**; the rest are sequenced follow-ups and electives. Do not add them to authored counts or public lesson navigation until their canonical lessons, exercises and evidence exist.

- **P0 — Repair progression and complete one lifecycle.** Route/career alignment plus the fourteen units below. Keep this bounded to a small model and a manageable dataset.
- **P1 — Complete the requested modality and engineering depth.** Natural-image/audio/retrieval work, full encoders, preference learning, broader conversion, precision, serving and release operations.
- **P2 — Specialize after evidence.** Accelerator qualification, extended RL, generative modeling, retrieval/tool applications, and deeper scientific/probabilistic/compiler work. These are goal-specific branches, not required detours for every learner.

No calendar or learner-time promises are assigned before the first slice is implemented and piloted. Cloud/device work has separate runtime and cost decisions. A CPU correctness lab can ship while its target-qualification status remains visibly pending.

## First release: fourteen connected units

| Order | Unit | What the learner retains |
| ---: | --- | --- |
| 1 | DATA-1 | A versioned source manifest, rights/attribution notes, rejected records and a reproducible sample |
| 2 | DATA-2 | Document/group-disjoint train/validation/test splits and a leakage report |
| 3 | DATA-3 | A frozen tokenizer and preprocessing contract with round-trip and special-token checks |
| 4 | EVAL-1 | A fixed evaluation inventory, baseline predictions and independently verified metrics |
| 5 | TEXT-1 | A model configuration and complete causal decoder with tested masks and positions |
| 6 | TEXT-2 | A trained checkpoint whose full state and next data batches recover correctly |
| 7 | TEXT-3 | Held-out generation and token reports, including failures and cache/full-forward parity |
| 8 | EVAL-2 | A repeated-run comparison with uncertainty and an explicit “inconclusive” option |
| 9 | ADAPT-1 | Full fine-tuning of that identified base, with response masks and base-task retention results |
| 10 | ADAPT-2 | LoRA on the same model/data, verified frozen leaves, saved adapters and merge parity |
| 11 | BRIDGE-1 | PyTorch/Flax Transformer mapping with per-layer and gradient error localization |
| 12 | SERVE-1 | A restored inference artifact behind a validated request API, using an existing supported precision policy |
| 13 | SERVE-2 | A small concurrent streaming service with bounded queueing, cancellation and measured load behavior |
| 14 | OPS-1 | MLflow lineage and a containerized release that connects all preceding artifacts |

Existing lessons remain prerequisites: foundations; networks/recovery; Transformers; `posttraining-01/02` before ADAPT; `deployment-08` before BRIDGE; export/precision/container lessons before SERVE/OPS. First-release precision uses the existing FP32/BF16/INT8 teaching where supported and validated. Packed INT4, QLoRA, FP8 and target speed claims are separate work.

**Release acceptance:** another learner can follow the guide from source data to a fresh-process loaded model, change one declared condition, diagnose an injected error and explain the observed quality/precision/latency tradeoff. All reported runs use saved artifacts, fixed evaluation definitions and recorded environments. A modest run need not produce a useful general chatbot; the report must say what the model learned and where it fails.

## Flagship model-development capstones

The [model lifecycle capstone catalog](model-lifecycle-capstones.md) now specifies **seven planned projects**: text embeddings, causal language generation, image understanding, image generation, audio understanding, audio generation and grounded multimodal generation. Each reuses its existing harness and has its own data recipe, objectives, evaluation, monitoring and release contract. The six additions extend the original embedding capstone; they are not yet registered executable projects or added to authored counts.

The [approximately 300M embedding project](../projects/embedding-model-300m/README.md) keeps its MLM → broad contrastive → supervised-pair sequence. The [causal project](../projects/causal-model-300m/README.md) uses next-token pretraining → domain continuation → SFT/full-FT versus LoRA → optional preference adaptation. Image and audio understanding have task-specific representation/adaptation choices; their separate generative projects teach denoising or acoustic-token generation. Multimodal generation connects qualified encoders to the causal decoder and evaluates grounding. Count all frozen and trainable components when describing model size.

The [shared tooling plan](model-lifecycle-tooling.md) adds **Weights & Biases** runs, artifacts, media tables, recovery/offline behavior and bounded sweeps alongside the existing MLflow work. It also scopes optional data-versioning, model-specific evaluators, generative pipeline references, serving and telemetry tools. Learners choose a minimal stack; core learning does not require every service.

Add **G7: model-development capstone** after the relevant G5/G6 preparation: tiny correctness run → measured real-data recipe pilot → separately budgeted target-scale training → phase-by-phase evidence → qualified service and immutable downloaded release. TEXT-4 remains required for the bidirectional embedding model, while TEXT-1..3/5 prepares the causal model. GEN/AUDIO and connector tutorials must precede generative image/audio/multimodal projects. Production readiness requires declared quality, reliability and runtime gates; model size does not establish it.

These capstones reuse the 61 planned units and add required project tutorials for phase transitions, tool adapters, benchmark governance, pipeline assembly, Hub packaging and operation. Cloud runs, benchmark scores, tracker integrations and publication remain unperformed.

## Route repair before adding content

Planning code: **ROUTE**. This is integration work, not an additional course package.

1. Keep shared foundations, then expose image, text, audio and cross-modal build goals as visible links. Offer a recommended next action with prerequisite reasoning; avoid mandatory questionnaires or dropdowns.
2. Make SFT/LoRA a supervised-adaptation branch. Require environment/RL/value preparation only for the preference/PPO branch. Keep stable lesson URLs; change grouping and route metadata carefully.
3. Keep the `models` image outcome achievable without mandatory language RLHF. Offer the text journey separately; make the TPU route's target work explicit and leave post-training optional.
4. Resolve optional engineering prerequisites into a visible branch. All required lessons must be ordered and available before their exercise is offered. Report extra work instead of silently expanding the selected goal.
5. Rewrite career milestones to point to the chosen route's actual artifact and review question. Distinguish a library mentioned in reading from an executed library project.
6. Preserve local progress and tutor plans. Preview changes in canonical manifests, generated career plans, offline books and project bundles before migrating route composition. Add meaningful checks for dependency order and unchanged progress IDs.
7. Keep the landing page concise: build goal, start/continue, tutor command, projects and careers. Put this full roadmap in a guide; do not turn it into homepage marketing copy.

## Course work packages

Each row names a lesson-sized question, its inspectable output and a figure/check that teaches the mechanism. “First release” refers to the fourteen units above. Other units are planned follow-ups. Unit codes here are planning IDs, not new live lesson IDs.

### DATA — Datasets, tokenization and input systems

Placement: expand recovery/data teaching and add a reusable data guide. Depends on arrays/state and basic train/eval separation. Reuse existing modality loaders. All modalities and careers benefit. First three units are P0; the fourth is P1.

| Unit | Lesson and artifact | Check and visualization |
| --- | --- | --- |
| DATA-1 | **Which data may this experiment use?** Build a manifest with source revision, byte hashes, attribution, schema, IDs and preprocessing version; retain rejected records and the sample-selection rule. | Replay the sample from source files; reject corruption/unknown schema. Show representative accepted/rejected examples and class/source counts. |
| DATA-2 | **Can duplicates leak the answer?** Split by document/person/recording/source before chunking or augmentation; compare exact and near-duplicate handling and time-based splits where appropriate. | Inject cross-split duplicates and group leakage. Show a source-to-split diagram and counts before/after deduplication; do not claim near-duplicate heuristics prove zero contamination. |
| DATA-3 | **What becomes a token or model input?** Compare a byte baseline with a trained subword tokenizer; freeze normalization, vocabulary, special IDs and maximum-length policy using training data only. | Round-trip multilingual/empty/boundary inputs; test EOS/PAD/UNK behavior. Plot length distributions under each tokenizer with a common document population. |
| DATA-4 | **Can a streaming mixture resume the same experiment?** Build shard sampling, mixture weights, length bucketing/packing, padding masks and a resumable cursor. | Count IDs/tokens across interruption and epoch/shard boundaries; diagnose sampler bias. Plot padding waste and observed source proportions against declared targets. |

Dataset selection is an implementation gate: choose a modest dataset with inspectable provenance and redistribution terms, record its revision/checksum, size, labels and intended limits, and supply acquisition instructions. An offline synthetic fixture remains available for numerical checks. Download availability, license suitability and runtime budget must be checked before declaring a real-data lab runnable.

### EVAL — Evaluation, uncertainty and experiment design

Placement: shared evaluation material after the relevant model basics, linked from optimization and all capstones. Existing fixed splits, metrics and seed comparisons are the starting point. First two units are P0; remaining units P1.

| Unit | Lesson and artifact | Check and visualization |
| --- | --- | --- |
| EVAL-1 | **What exactly is the metric measuring?** Freeze case IDs, reductions, split responsibilities, baseline and missing-response policy; produce per-example outputs and an error taxonomy. | Independent metric oracle, unequal batches and missing-case rejection. Show confusion/error examples or token NLL slices with counts. |
| EVAL-2 | **Is the apparent improvement repeatable?** Compare controlled runs across seeds and paired evaluation cases; distinguish uncertainty over training runs from uncertainty over examples. | Change the resampling unit and explain the difference; retain inconclusive comparisons. Plot paired differences/intervals, with no universal significance threshold. |
| EVAL-3 | **Can confidence support a decision?** Fit calibration on a separate split; compare thresholds, abstention, imbalance and out-of-distribution failures. | No final-test fitting; empty/small-slice handling. Reliability and risk–coverage plots with denominators and binning caveats. |
| EVAL-4 | **Did pretraining improve representations?** Compare random initialization, frozen linear probe, nearest-neighbor retrieval and full fine-tuning under declared budgets. | Freeze the correct parameters and keep selection data separate. Show downstream score versus compute and concrete failure cases; pretraining loss alone is insufficient. |

### TEXT — Complete language-model pretraining

Placement: extend Transformers/pretraining and `text-harness`. Depends on DATA-1..3, EVAL-1 and existing causal Transformer lessons. First three units are P0; final two P1.

| Unit | Lesson and artifact | Check and visualization |
| --- | --- | --- |
| TEXT-1 | **How do the pieces form a complete decoder?** Assemble configurable embeddings, multiple blocks, normalization, feed-forward layers, positions and a language head using the frozen tokenizer. | Causal independence, singleton/variable valid lengths and independent small attention values. Annotated tensor/parameter diagram. Start with ordinary attention and a simple supported position scheme. |
| TEXT-2 | **What does a real pretraining run require?** Train on the selected corpus; record valid tokens, optimizer/schedule clocks, accumulation, validation cadence, checkpoint identity and data position. | Interrupted versus uninterrupted next updates and token-weighted microbatch equivalence. Learning curves use comparable token budgets and separately marked validation observations. |
| TEXT-3 | **What has this language model learned?** Compare held-out token likelihood, a simple baseline and fixed generation prompts; run cached decoding on the saved model. | EOS/length/temperature policies, fixed-seed sampling and full-forward/cache parity. Display complete failed and successful continuations beside quantitative errors. |
| TEXT-4 | **How does an encoder learn from masked context?** Extend the MLM objective to a complete bidirectional encoder, corruption policy and a downstream classification/probe task. | Selected-position loss, no target copying, padding and changed-mask tests. Mask diagram plus downstream probe results; do not label the small model a reproduction of full BERT. |
| TEXT-5 | **What changes with RoPE and grouped-query attention?** Extend the same architecture and cache layout; investigate position limits and context changes. | Head-group mapping and position-offset parity against an independent reference; compare cache bytes and errors across lengths. Long-context generalization needs its own evaluation. |

### ADAPT — Fine-tuning and adapters on a saved model

Placement: supervised branch of post-training. Depends on TEXT-1..3, EVAL-1, existing response-mask and LoRA labs; **does not require RL**. First two units P0, remaining P1.

| Unit | Lesson and artifact | Check and visualization |
| --- | --- | --- |
| ADAPT-1 | **Can we change behavior without losing the base task?** Full fine-tuning from an identified checkpoint; conversation/source splits, chat formatting, response-only targets and base-task evaluation. | Shifted role masks, truncated answer handling and immutable tokenizer/base identity. Plot adaptation and retention metrics separately; report the nature of demonstration labels. |
| ADAPT-2 | **Where should LoRA live in a Transformer?** Apply adapters to named attention/feed-forward projections, track trainable leaves, save/reload and compare with full fine-tuning. | Frozen base leaves, initialization gradients, rank/alpha policies, new-input merge parity and double-merge rejection. Compare quality, parameter counts and measured memory without conflating them. |
| ADAPT-3 | **Which adaptation budget is worth using?** Compare ranks, target modules, schedules and frozen-head/partial/full tuning using a fixed selection protocol. | Repeated seeds, common data/evaluation and declared unequal resource budgets. Plot a measured quality/resource frontier and retain failures. |
| ADAPT-4 | **Can someone else load the adapter correctly?** Package base/tokenizer/template versions, adapter targets, precision, license notes and serving configuration; rehearse compatibility rejection and rollback. | Fresh-process reload, wrong-base rejection and canary predictions. Show artifact dependency graph and a failure-localization table. |

### VISION — Image learning and masked pretraining

Placement: extend networks/pretraining and `image-harness`. Depends on DATA/EVAL basics and existing CNN/MIM lessons. P1; dense prediction is an elective within this branch.

| Unit | Lesson and artifact | Check and visualization |
| --- | --- | --- |
| VISION-1 | **Does the image pipeline preserve the task?** Train a natural-image CNN baseline with class mapping, source-group splits and controlled augmentation. | EXIF/channel/range/resize parity and augmentation label validity. Show raw/transformed images, confusion matrix and error gallery. |
| VISION-2 | **How does a ViT reconstruct hidden patches?** Build patch embeddings, positions, visible-only encoder and reconstruction decoder; connect to the existing masked-loss lab. | Independent patch round-trip, hidden-pixel leakage test, changed mask ratios. Show original/masked/reconstructed images and hidden-region error, not only total loss. |
| VISION-3 | **Are the image features transferable?** Run EVAL-4 on supervised, masked and contrastive representations with declared augmentations. | Frozen-probe and full-FT distinction; independent splits. Downstream scores and nearest neighbors include plausible-looking failures. |
| VISION-4 | **What changes when predictions have locations?** Add a small segmentation task first, then a detection extension with box/mask transforms and task metrics. | Empty target, ignored pixel, resize and coordinate checks. Overlay predictions/errors and derive IoU on a hand-checkable mask; avoid a generic accuracy claim. |

### AUDIO — Signals, temporal models and recognition

Placement: a dedicated audio branch instead of hiding all signal theory in a project README. Extend `audio-harness`. Depends on arrays/state, DATA/EVAL basics and neural training. P1.

| Unit | Lesson and artifact | Check and visualization |
| --- | --- | --- |
| AUDIO-1 | **What does the model hear?** Decode real recordings with sample rate/channel/units, explicit resampling and windowing; retain speaker/recording identities. | Sine/impulse oracle, aliasing demonstration, sample-count/duration checks. Waveform and frequency plots connect samples, seconds and Hz. |
| AUDIO-2 | **What information does a spectrogram retain?** Compare STFT, log-mel features, normalization and silence behavior. | Window/hop/bin conventions, stable log and train-only statistics. Plot the same recording in waveform/STFT/mel views with explained axes. |
| AUDIO-3 | **Can a temporal model generalize to new recordings?** Train a CNN/temporal encoder with length masks, augmentation and speaker-disjoint evaluation. | Padding invariance, changed-duration/noise conditions and fresh-process recovery. Show errors by duration/noise/speaker, with sample counts. |
| AUDIO-4 | **How can audio align with a shorter transcript?** Implement CTC mechanics and a small recognition experiment before a streaming extension. | Enumerate a tiny alignment oracle, repeated labels/blanks and invalid lengths; derive CER/WER edit counts. Show frame logits, collapsed tokens and alignment ambiguity. |

### MULTI — Cross-modal representations and fusion

Placement: extend `cross-modal-harness`; depends on DATA, EVAL-4 and the relevant text/image/audio encoders. P1. Keep retrieval and generative claims separate.

| Unit | Lesson and artifact | Check and visualization |
| --- | --- | --- |
| MULTI-1 | **Which pairs are genuinely positive?** Natural image–caption retrieval with actual encoders, shared embeddings and multi-positive/group-aware labels. | Pair shuffling, duplicate semantics, normalization and both retrieval directions. Similarity matrix and ranked results include hard negatives. |
| MULTI-2 | **Does the representation cross domains or modalities?** Add an audio–text branch and fixed-pool Recall@K evaluation, with missing/corrupted modality cases. | No retrieval-pool leakage; tie/positive-set handling and independent rank calculation. Retrieval grids and per-domain errors with pool sizes. |
| MULTI-3 | **How does one modality condition another?** Freeze/unfreeze an encoder, train a projection or cross-attention bridge, and evaluate a small conditioned-generation task. | Mask/shape contracts, modality ablation and pair counterfactuals. Compare outputs with correct, absent and swapped context. |

### PREF — Preference learning and sequence RL

Placement: advanced post-training branch. Depends on the existing RL/objective lessons and ADAPT-1; value/GAE preparation precedes sequence PPO. P1; continuous-action environment work remains a later RL specialization.

| Unit | Lesson and artifact | Check and visualization |
| --- | --- | --- |
| PREF-1 | **What does a preference label mean?** Create a rubric and data schema, retain disagreement/ties and source/prompt splits, and train a sequence reward head. | Chosen/rejected direction, prompt leakage and held-out ranking; compare length effects. Margin/disagreement plots; synthetic or heuristic labels must be identified as such. |
| PREF-2 | **How should a trajectory assign credit?** Train a value function and implement GAE with terminal versus truncated bootstrap conventions. | Tiny independently enumerated trajectories and changed discount/lambda. Plot reward, value, advantage and valid-time masks on the same episode. |
| PREF-3 | **What makes sequence PPO different from the bandit lab?** Use variable-length sampled responses, frozen behavior/reference policies, learned values and explicit token KL/reward conventions. | Padding/EOS, signed clipping, old-log-prob reuse, advantage normalization and reward exploitation drills. Plot external quality, reward, KL and length separately. |
| PREF-4 | **How does sequence DPO compare?** Optimize chosen/rejected response log-probability sums under a frozen reference and compare to SFT using held-out judgments. | Response masks, sum-versus-average convention, length effects, reference caching and reversed preferences. Explain margin/length/quality plots without treating preference fit as general alignment. |

GAE and QLoRA are separate technical mechanisms, not names for the existing PPO and LoRA examples. Use the [GAE paper](https://arxiv.org/abs/1506.02438) for the advantage derivation. A GRPO/verifiable-reward comparison may follow PREF-3; it is not a prerequisite or a substitute for teaching the value-based method accurately.

### NUM — Training and inference precision

Placement: expand deployment precision with links back to optimization and adaptation. Depends on existing `deployment-05`, EVAL and relevant model arithmetic; NUM-4 also needs ADAPT-2. P1, with hardware-dependent extensions P2.

| Unit | Lesson and artifact | Check and visualization |
| --- | --- | --- |
| NUM-1 | **Which tensors need which numerical range?** Write separate policies for parameter storage, optimizer/master state, activations, reductions, gradients and cache; compare FP32/BF16/FP16 and loss-scaling behavior. | Overflow/underflow, nonfinite-update handling and gradient comparison. Plot ranges and update errors with units; FP8 is an explicitly target-dependent extension. |
| NUM-2 | **Where does post-training quantization lose information?** Compare per-tensor/channel/group scales, zero points, calibration choices, outliers and held-out task effects. | Independent quantize/dequantize and affine operator oracle; zero-channel and clipping cases. Range histograms, clipping fractions and layer/task errors. |
| NUM-3 | **Can training compensate for quantization?** Add fake quantization/QAT with an explicitly defined surrogate gradient, then export to one supported integer runtime. | Distinguish the surrogate derivative from the true rounding derivative; check exported versus fake-quant behavior and rounding conventions. Show code grids and held-out quality. |
| NUM-4 | **What changes when the frozen base is low-bit?** Implement a bounded quantized-adapter experiment, then a supported packed-QLoRA path with recorded base format and compute policy. | Packed bytes including scales/metadata, frozen-base checks, adapter reload and quality. Actual memory/latency require runtime measurements; dequantized FP32 matmuls are not native low-bit acceleration. |

Track a compatibility matrix by framework/runtime/device and **weights / activations / accumulator / optimizer / cache**, with representation-only, executed-reference and native-kernel states. Include INT8/INT4, FP16/BF16/FP32 and target-specific FP8; unsupported combinations must be explicit. The [QLoRA paper](https://arxiv.org/abs/2305.14314) informs NUM-4; [LiteRT's operator specification](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec) informs compatible integer deployment contracts. Neither source proves our implementation or hardware performance.

### BRIDGE — Framework portability and architecture parity

Placement: extend `weight-conversion` and existing Keras/TensorFlow labs. Depends on `deployment-01/08`; use the relevant complete architecture. BRIDGE-1 is P0, the others P1.

| Unit | Lesson and artifact | Check and visualization |
| --- | --- | --- |
| BRIDGE-1 | **Can two frameworks compute the same Transformer?** Map embeddings, fused/split QKV, heads, dense layouts, normalization, positions and tied output weights for one declared architecture. | Multiple batches/lengths/dtypes, per-layer forward and input-gradient checks, cache outputs, exact keys and fresh-process artifact reload. Plot first-diverging-layer absolute/relative error; tolerances are operator-specific. |
| BRIDGE-2 | **Which state travels with a CNN?** Convert one Keras/TensorFlow/PyTorch/JAX image model through individually declared paths, including convolution layout, padding and normalization state. | Raw-image preprocessing parity and independent receptive-field calculations. Show where channel/layout/epsilon errors first appear; no universal converter claim. |
| BRIDGE-3 | **What survives serialization or a backend change?** Compare Keras backend switching with explicit weight mapping and supported inference export; retain metadata and reject unsupported operations. | Fresh environments/processes, shape/dtype contracts, serialization round trip and unsupported-op failures. A compatibility table records tested versions; optimizer-state portability is a separate promise. |

The [Keras migration guide](https://keras.io/guides/migrating_to_keras_3/) informs the distinction between backend-compatible operations and framework-specific code. Preserve separately pinned optional environments where dependency requirements differ from the core course.

### SERVE — Inference systems and edge qualification

Placement: deployment branch extending text/image/audio exports and `engineering-release`. Depends on export, precision, containers and fixed evaluation. First two units P0; remaining P1 with target work requiring access to the actual device.

| Unit | Lesson and artifact | Check and visualization |
| --- | --- | --- |
| SERVE-1 | **Can a saved model answer a validated request?** Serve the actual restored artifact with preprocessing/tokenizer identity, startup/readiness, explicit shape limits and precision selection. | Golden requests, malformed/nonfinite/oversized inputs, digest failure, process restart and fresh export parity. Trace raw input through the response boundary. |
| SERVE-2 | **What happens when several users arrive together?** Add bounded queues, streaming, cancellation, deadlines, backpressure and graceful shutdown to a small local service. | Real concurrent requests and canceled/slow clients; no abandoned cache/state leaks. Timeline and load curves separate queue delay, time to first token, inter-token latency and complete-request latency. |
| SERVE-3 | **When do batching and cache scheduling help?** Compare a simple baseline with dynamic/continuous batching and cache allocation under a declared workload. | Mixed lengths, fairness, capacity exhaustion and output parity. Throughput/tail-latency/memory curves include sample counts and startup exclusions; paged cache is an advanced extension. |
| SERVE-4 | **Does the complete request work on the edge device?** Choose one supported runtime/device, carry preprocessing through conversion and inspect delegate/fallback behavior. | Device canaries, warm/cold/end-to-end latency, peak memory, sustained thermal behavior and energy only where measurable. Clearly separate desktop conversion results from actual device evidence. |

### OPS — ML, LLM and model operations in one release

Placement: extend existing operations and `engineering-release`; reuse its working MLflow/Docker foundations. OPS-1 P0, remaining P1. Shared core requires local serving, not prior multi-host accelerator training.

| Unit | Lesson and artifact | Check and visualization |
| --- | --- | --- |
| OPS-1 | **Can we reconstruct the released experiment?** Track the connected data/tokenizer/base/adapter/evaluation/export/image bundle with MLflow or the planned equivalent W&B adapter and load the selected immutable model in its container. | Retrieve rather than merely log; fail stale hashes and wrong versions; compare the container response to the saved artifact. Lineage graph links actual IDs. |
| OPS-2 | **When should live traffic move to a candidate?** Add shadow/canary rollout, bounded quality/latency gates, authenticated review boundaries and actual traffic rollback. | Concurrent promotion conflict, failed health/quality gate, missing reviewer authority and old-version restoration. Plot traffic allocation and incident timeline. |
| OPS-3 | **Which changes deserve an operational response?** Monitor drift, delayed labels, training-serving skew and SLOs with known measurement windows. | Inject data and service failures separately; rehearse alerts/runbooks. Plot observed versus expected signals; drift alone must not trigger automatic retraining. |
| OPS-4 | **How do we operate a model application?** Version prompt/retrieval/tool contracts, trace actual local model calls, enforce access/retention boundaries and rehearse model retirement. | Unsupported answers, prompt injection/tool misuse fixtures, secrets-redaction checks and revoked-artifact behavior. Case-level traces accompany aggregate quality and cost measurements. |

Use [MLflow's official documentation](https://mlflow.org/docs/latest/ml/) for the selected tracking/registry APIs. Existing synthetic LLM replay remains useful for gate testing, but the new application milestone must identify which cases actually executed a model and which were fixtures.

### SCALE — Distributed training and target performance

Placement: advanced systems branch extending `sharded-training`, `kernel-audit` and the text harness. Depends on recovery, performance and distributed foundations; TEXT for the LLM branch. P2 qualification, with local correctness preparation available earlier.

| Unit | Lesson and artifact | Check and visualization |
| --- | --- | --- |
| SCALE-1 | **Which dimension should each device own?** Compare data, fully sharded and tensor parallelism on a complete small model; introduce context/pipeline parallelism only after those contracts. | Global token denominator, uneven masks, single-device gradient reference. Placement diagrams and communication-volume models distinguish estimates from measured traffic. |
| SCALE-2 | **Can a checkpoint survive a new mesh or failed host?** Reshard model/optimizer state and coordinate input ownership/restore across processes. | Duplicate/missing records, rank loss, incomplete storage and incompatible topology. An interruption timeline identifies the accepted recovery boundary. |
| SCALE-3 | **Does the target trace support the proposed optimization?** Profile a full training/inference workload, memory and collectives; compare a kernel or rematerialization change with the strong baseline. | Numerical parity, synchronized repeated timings and end-to-end measurements on a named target. Show both local kernel and whole-step results. |
| SCALE-4 | **How does this map to MaxText?** Read a pinned MaxText version and trace configuration, model, input, checkpoint, evaluation and precision contracts; reproduce one supported small target run when resources exist. | Map course artifacts to actual reference files and record version/config/device. No tutorial completion or upstream benchmark is a substitute for our run receipt. |

The inspected [MaxText overview](https://github.com/AI-Hypercomputer/maxtext) describes a JAX LLM reference targeting TPU/GPU training and recommends released distributions over assuming `main` is production-ready. At implementation, pin a release/commit and inspect its actual files; this audit only read the overview. Do not copy commands from a different revision or assume compatibility with the course's core Python environment.

### RESEARCH — Deeper mathematical and scientific branches

Placement: four **separate electives**, linked to their existing phases and specialist career milestones. P2. These scopes are follow-up starts, not complete replacements for full specialist courses.

| Unit | Lesson and artifact | Check and visualization |
| --- | --- | --- |
| RESEARCH-1 | **What changes with constraints or nonsmooth objectives?** Extend optimization with a small projected/proximal problem and explicit stationarity conditions. | Independent closed-form case and boundary counterexample. Plot feasible set, iterates and objective; ordinary smooth gradient checks have limits. |
| RESEARCH-2 | **Does a hierarchical Bayesian model explain the data?** Use an actual inference library, multiple chains, modern diagnostics and posterior predictive checks. | Known special case, simulation-based checks, poor geometry and model mismatch. Trace/rank/predictive plots explain distinct failure mechanisms. |
| RESEARCH-3 | **Can an adaptive solver support an inverse problem?** Implement a less trivial system with tolerance control, events or stiffness and uncertain measurements. | Convergence, sensitivity, conservation where applicable and held-out prediction. Distinguish discretization bias from noise and non-identifiability. |
| RESEARCH-4 | **Does a custom transformation compose correctly?** Extend the internals project with a supported API boundary and explicit batching/autodiff/JIT combinations. | Compare transformed results and reject unsupported effects/shapes. Annotated program/data-flow graph plus an injected rule error. |

### GEN — Generative models beyond autoregressive text

Placement: optional image/audio generative branch after probability basics, neural training and EVAL. P2; extend modality harness contracts rather than creating an unrelated infrastructure stack.

| Unit | Lesson and artifact | Check and visualization |
| --- | --- | --- |
| GEN-1 | **What does a latent-variable model optimize?** Build a small VAE with reconstruction and KL accounting. | Analytic Gaussian KL, reparameterization check and posterior-collapse diagnosis. Reconstruction/latent plots show limitations. |
| GEN-2 | **How does denoising become generation?** Train a small diffusion/denoising model with a declared noise schedule and prediction parameterization. | Forward moments and reverse-step oracle; consistent training/sampling conventions. Show fixed-example noise levels and actual sampling trajectories. |
| GEN-3 | **How do conditioning and guidance change outputs?** Add class or cross-modal conditioning and compare sampling policies under a fixed budget. | Fixed noise seeds, absent/swapped conditions and held-out evaluation. Sample grids are labeled selections and accompanied by aggregate evidence. |

### RAG — Retrieval and tool-using model applications

Placement: optional LLM application branch after TEXT, EVAL, SERVE and OPS basics. P2. Useful for inference/operations roles, but separate from the requirement to learn model training.

| Unit | Lesson and artifact | Check and visualization |
| --- | --- | --- |
| RAG-1 | **Did retrieval find the evidence?** Build a versioned local document index, chunker and retrieval baseline with corpus/query split rules. | Independent ranking cases, stale index and duplicate/chunk leakage tests. Show retrieved passages and recall by query class. |
| RAG-2 | **Does the answer follow the retrieved evidence?** Evaluate a local grounded-answer pipeline with citations, unsupported questions and abstention. | Separate retrieval failure from generation failure; source/version attribution and adversarial-document cases. Case traces connect question, evidence and answer. |
| RAG-3 | **Can a model use a tool within a reliable boundary?** Add typed tool arguments, idempotency, bounded retries, authorization and error recovery. | Invalid arguments, prompt injection, timeout, duplicate calls and unauthorized actions using disposable fixtures. A sequence diagram shows decisions and observed effects. |

## Project strategy and artifact handoffs

Use existing projects as the spine. Add isolated stages or explicit extension modules without invalidating their introductory checker contracts.

| Existing project | Planned extension | Artifact handed to the next stage |
| --- | --- | --- |
| `text-harness` | DATA/TEXT/ADAPT natural-data lifecycle | Data/tokenizer/config identities, full training checkpoint, base/adapted evaluation, adapter bundle |
| `image-harness` | VISION plus EVAL-4 | Image/preprocessing contract, trained CNN/ViT, probe results and raw-input canaries |
| `audio-harness` | AUDIO | Waveform/feature contract, temporal model, length-aware metrics and golden recordings |
| `cross-modal-harness` | MULTI | Pair manifest, encoder versions, frozen candidate pool, two branch exports and retrieval report |
| `training-methods` | Connect verified objective functions to those harnesses | Mask/reduction/objective configuration and changed-condition checks |
| `weight-conversion` | BRIDGE architecture-specific mappings | Converted weights, architecture schema, per-layer errors and reload evidence |
| `deployment-audit` | NUM + SERVE | Runtime-specific artifact, calibration identity, precision policy and measured request results |
| `engineering-release` | OPS integration | MLflow IDs, immutable release manifest, image identity, gate evidence, rollout/rollback record |
| `sharded-training` / `kernel-audit` | SCALE | Mesh/data-ownership contract, target traces, recovery and numerical/performance receipts |
| Research and RL projects | RESEARCH/PREF specialist branches | Baseline, independent oracle, changed task, error report and bounded conclusion |

The conceptual shared contract is configuration, data identity, training state, evaluation, export and request handling. Reuse a small common implementation only after two actual harnesses demonstrate the same need. Keep tokenization, image transforms, audio preprocessing, pairing and decoding explicit; do not force every modality through a premature generic trainer.

## Prerequisite and release gates

Each package is implemented internally in dependency order. First-release subsets do not wait for every later unit in their package.

| Gate | Required work | Definition of done |
| --- | --- | --- |
| G0: progression | ROUTE and current audit | No required step has missing preparation; career/route claims resolve to actual artifacts; existing progress IDs survive |
| G1: reproducible data | DATA-1..3 + EVAL-1 | Pinned manageable dataset, separate split roles, checked tokenizer and baseline, offline fixture and acquisition instructions |
| G2: complete text training | TEXT-1..3 + EVAL-2 | Actual run, recovery, baseline/seed comparison, failed generations retained and cache/export contracts carried forward |
| G3: adaptation and portability | ADAPT-1..2 + BRIDGE-1 | Same saved base, real full-FT/LoRA comparison, frozen-leaf checks, per-layer/gradient parity and fresh-process conversion reload |
| G4: local release | SERVE-1..2 + OPS-1 | Actual restored model service, real concurrent/canceled requests, immutable tracked/containerized release and measured boundaries |
| G5: modality expansion | VISION/AUDIO then MULTI, with EVAL-3..4 | Each modality has real-data evidence plus its own preprocessing, failure modes and changed-condition assessment |
| G6: advanced qualification | PREF/NUM/SERVE/OPS/SCALE extensions | Algorithm-specific correctness plus observed target/runtime evidence where the claim requires it |

PREF also depends on learned value/GAE before sequence PPO. NUM-4 depends on actual adapter training. MULTI requires the corresponding encoders before fusion. RAG retrieval can begin independently, but answer/tool operation requires model and service preparation. RESEARCH and GEN are optional after their own prerequisites; they do not block the core release.

## Authoring, figures and assessment acceptance

For every unit:

1. State one learner question, prior knowledge, one likely misconception, hardware and the artifact to keep.
2. Explain one worked mechanism with defined KaTeX notation; build code in meaningful increments with expected intermediate outputs.
3. Supply an independent oracle/invariant and a failure to diagnose. A passing copied implementation or a falling loss is insufficient.
4. Include at least one figure that answers the question. Record real plotting data, label analytic illustrations, explain axes/units/legend, a visible observation, its mechanism and its limits.
5. Require transfer to changed data, dimensions, masks, precision or runtime conditions. Explain low-quality results rather than selecting a lucky run.
6. Preserve artifact identity through training, checkpointing, evaluation, conversion and serving. Test loaded artifacts in fresh processes.
7. Add a matching project stage and reviewer rubric where a new capability is claimed. Use independent evaluation cases and failure explanations; leave assessment status as a draft until actually reviewed.
8. Generate browser/Markdown/notebook/CLI/offline outputs from canonical sources. Run build/check, CPU numerical suites after numerical changes, relevant project checks and figure/source fidelity checks. Optional device/framework checks have separate receipts.
9. Pilot the connected slice with learners before broad expansion. Ask them to explain an unseen plot and diagnose a new failure; use observed mistakes to revise teaching and hints.

References checked for planning include [MAE](https://arxiv.org/abs/2111.06377) for masked image encoders and [CLIP](https://arxiv.org/abs/2103.00020) for image–text representation learning. These guide scope and mechanisms; the planned small experiments need their own results and should not inherit the papers' model-quality claims. Version-sensitive APIs and additional specialist sources must be checked again during authoring.
