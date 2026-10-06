# Model lifecycle capstones

Status: **ten planned project specifications**, 2026-10-06. The existing course still has 19 registered executable projects. These plans are additional work; no new large model, score, cloud run or public release is claimed.

Every capstone starts from an existing harness and follows a complete model-development cycle: task/data contract → architecture and input representation → pretraining → task-specific adaptation → evaluation → precision/conversion → service qualification → Hugging Face publication → operation. The model's task determines its objectives. Masked language modeling is one route, not a universal first phase.

## Choose the model to build

| Planned project | Learning sequence | Final capability |
| --- | --- | --- |
| [Text embedding model, approximately 300M](../projects/embedding-model-300m/README.md) | MLM → broad contrastive continuation with a joint-objective ablation → supervised contrastive pairs | Qualified text embeddings and a version-compatible retrieval index |
| [Causal language model, approximately 300M](../projects/causal-model-300m/README.md) | Next-token pretraining → domain continuation → SFT/full-FT versus LoRA → optional DPO or sequence RL | A measured small generative language model with reproducible decoding and a bounded service |
| [Image understanding model, approximately 300M](../projects/image-model-300m/README.md) | Supervised baseline → selected masked/contrastive/distillation pretraining experiment → classification/retrieval or dense-prediction adaptation | An image model with verified preprocessing and task-specific evaluation |
| [Image generation model](../projects/image-generation-model/README.md) | Image/latent representation → denoising diffusion baseline → conditioning/domain adaptation → sampler and quality qualification | A complete reproducible image-generation pipeline |
| [Audio understanding model, approximately 300M](../projects/audio-model-300m/README.md) | Acoustic representation learning → speech recognition or sound-event adaptation → task/domain robustness | A recognizer or event model with a waveform-to-output contract |
| [Audio generation model](../projects/audio-generation-model/README.md) | Codec/representation qualification → generative sequence or denoising training → controlled conditioning → waveform evaluation | A bounded text/condition-to-audio system with a complete decoder |
| [Multimodal model](../projects/multimodal-model/README.md) | Qualified encoders/decoder → connector alignment → grounded instruction tuning → optional preference adaptation | A grounded image/audio-to-text model; cross-modal retrieval remains a distinct sibling outcome |
| [CLIP image–text model](../projects/clip-model/README.md) | Paired softmax pretraining → domain continuation → curated pairs/adaptation → zero-shot and bidirectional retrieval | Qualified dual encoders, processors and a reproducible retrieval handoff |
| [SigLIP image–text model](../projects/siglip-model/README.md) | Pairwise sigmoid oracle → negative/reduction audit → controlled pretraining and adaptation → CLIP comparison | A separately qualified sigmoid-trained dual encoder, including scale and bias |
| [Retrieval system](../projects/retrieval-system/README.md) | Corpus/qrels → exact and lexical baselines → ANN → hybrid/reranking → service and index migration | A measured text or cross-modal search service with compatible model/index releases |

For the generative and multimodal projects, approximately 300M is an initial **component-budget hypothesis**, not a claim about the whole pipeline. Report total, trainable, frozen and active parameter counts, plus auxiliary encoders/codecs/decoders. A 300M denoiser plus a large text encoder is not a 300M deployment. Smaller, measured pilots precede any final size choice.

## Common project contract

Each linked specification inherits these requirements and supplies its own objectives, recipes, benchmarks and release details:

1. **Define the use case.** Name languages/domains, input/output limits, baseline, quality floors, permitted slice regressions and runtime/service targets before final evaluation.
2. **Version the recipe.** Record source revisions, hashes, grouping/splits, rights/attribution, filtering, deduplication, mixture weights and sampling units. Isolate final test material and disclose known overlap. Data acquisition remains an implementation task.
3. **Freeze representations.** Tokenizer, processor, codec, positions, class IDs and normalization are part of the model. A change requires a new artifact identity and renewed evaluation.
4. **Use a scale ladder.** Tiny CPU correctness fixture → measured small real-data pilot → separately budgeted target-scale run. From-scratch and warm-start variants have distinct provenance and compute reports.
5. **Hand off actual checkpoints.** Every phase records its parent, full state, objective/reduction, data cursor and configuration. Declare optimizer/schedule resets between phases; test exact resume within phases.
6. **Observe actual learning.** Log real outputs, gradients, source exposure, numerical health and quality. Explain figures against their data and use KaTeX for mathematical notation. Include a changed condition and a failure diagnosis at every stage.
7. **Separate development from release testing.** Pin evaluator/task/dataset versions, baselines, decoding or preprocessing and the full expected task inventory. Keep failed/missing tasks visible and avoid selecting recipes on final test scores.
8. **Qualify exported bytes.** Reload in a fresh process; compare native and converted/reduced-precision outputs and task quality. Record actual hardware, memory and synchronized timing boundaries.
9. **Publish a usable system.** Package model plus processors/configuration/required auxiliary components, model card and canaries; verify an immutable Hub download. A hosted weight file alone is insufficient.
10. **Rehearse operation.** Exercise concurrency, timeouts, failure recovery, monitoring and rollback appropriate to the product. Production readiness is conditional on the declared task/runtime gates.

Use the [tooling guide](model-lifecycle-tooling.md) for MLflow, Weights & Biases and focused ecosystem choices. Tracking backends receive a common local metric/artifact record; changing dashboards must not change the training algorithm or require a hosted account for core learning.

## Shared publishing and monitoring lessons

Reuse the [Hub publication workflow](../projects/embedding-model-300m/RELEASE.md), adapting the consumer format and canaries to each project. The embedding-specific model/index migration applies only where embeddings feed a persistent index. Other projects version their complete inference pipeline and restore a compatible pipeline on rollback.

Reuse the [monitoring contract](../projects/embedding-model-300m/MONITORING.md) for identifiers, counts, numerical health, resource measurements and recovery. Replace embedding-specific metrics with the corresponding project table. A causal model needs token/generation measurements; diffusion needs timestep/denoising/sampler measurements; audio needs seconds, alignment and waveform measurements.

## Curriculum placement and new integration teaching

These projects apply the existing 61 planned lesson units, with required project tutorials for the additional integration work. They are not automatically another 61 lessons and are not shallow template pages.

| Tutorial to author | Connected curriculum | Learner evidence |
| --- | --- | --- |
| Reproducible lifecycle configuration and phase transitions | DATA, TEXT, ADAPT, VISION, AUDIO, GEN | Parent/child checkpoint lineage and objective/schedule transition test |
| Equivalent MLflow and W&B tracking | OPS-1 and all capstones | Same recorded metrics/artifacts through both adapters, offline recovery, explicit step semantics |
| W&B bounded sweeps and artifact comparisons | EVAL-2, ADAPT-3, OPS | Fair pilot comparisons, resource caps and selected configuration bound to artifact IDs |
| Model-specific evaluation adapters | EVAL plus each specialist branch | Independent metric oracle, full task inventory and a fixed serialized-model evaluation |
| Image/audio generative pipeline assembly | GEN, AUDIO, MULTI | Representation/codec identity, denoiser or token model, and correct full-pipeline decoding |
| Multimodal connector and grounded SFT | MULTI-3, ADAPT-1, AUDIO/VISION | Swapped/absent modality ablations, modality masks and grounded-answer evidence |
| Portable packaging and cold Hub loading | BRIDGE, SERVE, OPS | Downloaded immutable revision reproduces canaries and advertised consumer behavior |
| Service telemetry and rollback | SERVE/OPS | Correlated incident trace/metrics and restoration of a compatible pipeline |

## Alignment and retrieval integration tutorials

The CLIP and SigLIP projects extend MULTI-1 with distinct objective derivations, duplicate-positive handling, dense/chunked/distributed gradient checks and controlled real-image experiments. MULTI-2 supplies cross-domain and later audio–text transfer. The retrieval project deepens RAG-1 with exact versus approximate ranking, relevance judgments, hybrid search, reranking, deletion and index migration. These are required project tutorials within the existing roadmap, not additional authored lesson entries.

Teach CLIP's symmetric softmax baseline before the SigLIP comparison. Reuse identical data, processors, towers and development budgets where the experiment calls for control. Then carry qualified encoder artifacts into retrieval. Define a common score/embedding contract, but preserve each model's normalization, scale, bias and input semantics.

The [public plan registry](../curriculum/project-plans.json) makes these three new specifications discoverable from Projects and the relevant modality guides. The other seven lifecycle specifications remain repository planning documents. All ten remain outside the executable project registry.

## Implementation sequence

Keep the first-release fourteen-unit small text lifecycle intact. Then implement the causal and embedding capstones from their shared text/data/state foundations. Build image and audio understanding after their dedicated preparation, followed by image/audio generation once representation and decoder lessons exist. Multimodal generation consumes qualified component checkpoints from those projects rather than unexplained imports. Public-tool integrations are verified in bounded local/offline exercises before any optional account sync or large run.

Each project must gain a learner starter, substantive reference implementation, independent stage checks, executed figures and assessment rubric before registration as authored. Until then its status remains planned and its specification is discoverable here and from the course roadmap.
