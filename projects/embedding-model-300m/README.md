# Flagship capstone: build and release a 300M embedding model

**Status: planned project specification, 2026-10-06.** No 300M model has been trained, benchmarked, production-qualified or published by this project. It is not yet part of the registered executable project count. This is the advanced integration capstone for the [course expansion plan](../../docs/course-roadmap-2026-10-06.md), within the [seven-project lifecycle catalog](../../docs/model-lifecycle-capstones.md). Its embedding branches remain below; causal, image/audio generation and grounded multimodal models now have separate project specifications.

The learner will make and defend a sequence of decisions: what the model should retrieve, which data it may learn from, how inputs become representations, which objective belongs at each stage, whether the representation improves, and whether someone else can load and operate the released model.

“Production grade” is the intended outcome **for a declared use case and runtime**, earned through quality, reliability and release gates. A parameter count or an MTEB average alone is verified separately from it.

## Project guides

- [Data recipes, tokenizer, architecture and three training phases](TRAINING_RECIPE.md)
- [Evaluation, MTEB, retrieval and release benchmarks](EVALUATION.md)
- [MLflow, training metrics, monitoring and incident decisions](MONITORING.md)
- [Image, audio and cross-modal versions of the capstone](MODALITIES.md)
- [Packaging, numerical parity and Hugging Face publication](RELEASE.md)

Use the [shared tooling plan](../../docs/model-lifecycle-tooling.md) for equivalent Weights & Biases and MLflow tracking, bounded sweeps and artifact lineage. The tracking adapter is independent of the training objective.

## The connected journey

```mermaid
flowchart LR
  A[Use case and evaluation contract] --> B[Curated data and frozen tokenizer]
  B --> C[Phase 1: masked pretraining]
  C --> D[Phase 2: broad contrastive pretraining]
  D --> E[Phase 3: supervised pair fine-tuning]
  E --> F[Quality and robustness gates]
  F --> G[Conversion and precision checks]
  G --> H[Service and release qualification]
  H --> I[Hugging Face release]
```

Evaluation and monitoring run throughout this sequence. Every phase produces a named checkpoint, a data/objective/configuration manifest, metrics and a decision report. A rejected stage remains an instructive result; it is not promoted by renaming its status.

## Reuse the existing harnesses

| Existing work | Reuse | Required new work |
| --- | --- | --- |
| [Text harness](../text-harness/README.md) | Token/data identity, explicit training state, recovery, export and measurement patterns | Bidirectional encoder, document pooling, real corpus and pair batches. The current causal mask, next-token objective and KV cache do not become an embedding encoder unchanged. |
| [Training methods](../training-methods/README.md) | Masked-loss and contrastive correctness oracles | Complete encoder, distributed candidate gathering, real pair provenance and stage handoffs |
| [Sharded training](../sharded-training/README.md) | Global reductions, state ownership, replay and profiling discipline | Encoder/pair workload on real accelerators, cross-device negatives and target qualification |
| [Weight conversion](../weight-conversion/README.md) | Mapping validation and first-diverging-layer analysis | Encoder attention, positions, pooling and tokenizer parity; existing dense mapping is preparation |
| [Deployment audit](../deployment-audit/README.md) | Export, precision and request-boundary contracts | Embedding-specific quality, index compatibility and target service measurements |
| [Engineering release](../engineering-release/README.md) | MLflow lineage, container and release policy | Multi-phase run lineage, benchmark gates, Hub package and embedding-index migration |
| [Image](../image-harness/README.md), [audio](../audio-harness/README.md), [cross-modal](../cross-modal-harness/README.md) | Their modality-specific preprocessing, state and release contracts | The modality branches in MODALITIES.md; no claim that text MLM fits every modality |

## Prerequisites and scale ladder

The project follows DATA, EVAL, TEXT-4, NUM, BRIDGE, SERVE, OPS and SCALE preparation, plus the relevant VISION/AUDIO/MULTI units. The earlier decoder lifecycle is useful preparation but does not replace **TEXT-4's bidirectional encoder** or this project's embedding evaluation. SFT/LoRA/RLHF are optional comparisons; instruction tuning is not the final objective of this retrieval model.

Use three configurations sharing the same contracts:

1. **Correctness run:** tiny encoder and controlled fixtures, suitable for CPU; independent loss, masking, pooling and resume checks. This is where learners debug mechanics.
2. **Recipe pilot:** a smaller encoder and a bounded real-data subset; compare recipes and observe throughput/memory on the intended training runtime. Report its own size and budget.
3. **Approximately 300M run:** the selected encoder, full frozen recipe, accelerator receipts, staged checkpoints and measured release evidence. Do not relabel the pilot as this run.

The teaching must include a full from-scratch lineage. A warm-started model can be a separately labeled lower-cost comparison, retaining its original training provenance and tokenizer. No accelerator count, training duration, data budget or price is promised before the pilot measures the actual workload. Cloud allocation/publication are later execution steps, not performed by this specification.

## Stages and learner evidence

| Stage | Learner work | Artifact and pass condition |
| --- | --- | --- |
| 0 — Product contract | Choose the retrieval task, language/domain coverage, output dimension, maximum input length, hardware and service constraints | `target-contract` with baseline, quality floors, acceptable regression margins, latency/memory/load targets and evaluation roles fixed before final testing |
| 1 — Data recipe | Source, deduplicate, split, filter and sample data; isolate benchmark test material | `data-manifest`, source/group leakage report, per-source acceptance statistics and reproducible shards |
| 2 — Input/architecture | Compare tokenizers, freeze one, implement encoder/pooling and count actual parameters | `tokenizer-report`, architecture config, parameter-tree count and independent numerical checks |
| 3 — Phase 1 MLM | Pretrain the complete encoder using selected-token loss | Recoverable `phase1` checkpoint; held-out masked loss, downstream probes, data/token counters and recovery drill |
| 4 — Phase 2 contrastive pretraining | Train broad weakly supervised pairs; compare contrastive-only with a retained-MLM objective | `phase2` checkpoint; positive/negative audit, component losses, candidate counts, retrieval improvement and representation health |
| 5 — Phase 3 pair fine-tuning | Use curated task pairs, reviewed hard negatives and fixed query/document conventions | `phase3` checkpoint; domain quality and general-retention comparison, no final-test tuning |
| 6 — Evaluation | Evaluate each stage and baselines under a pinned benchmark protocol | Raw task results, exact task coverage, contamination/overlap disclosure, failure cases, uncertainty and release decision |
| 7 — Portability/precision | Reload, convert where needed, calibrate and compare supported inference policies | Layer/pooling/embedding parity, retrieval quality, storage and measured target results for every advertised format |
| 8 — Service qualification | Load the candidate in its container, exercise concurrent encoding and an index migration | Load/failure report, bounded queueing, timeouts, canaries, model/index identity and rollback rehearsal |
| 9 — Hub release | Assemble, upload a candidate, download its immutable revision and rerun consumer checks | Model card, tokenizer/processor, weights, loader, benchmark/config records, commit ID and successful cold-load receipt |
| 10 — Operate and defend | Monitor the deployed embedding/index pair; investigate an injected quality or data failure | Runbook, incident timeline, rollback to a compatible model/index, and a learner explanation of what the evidence supports |

These artifact names are proposed outputs, not existing files or runnable commands. Starter functions, reference implementations and stage checks must be authored before project registration. Each stage needs a prediction, an actual executed figure, a changed condition and a failure diagnosis. The model card is written from the final receipts rather than from planned targets.

## Portfolio review

The final submission includes the learner's code and commands; data/recipe/tokenizer decisions; three checkpoint identities; one recipe ablation; actual training and representation-health plots; reproducible benchmark outputs; negative results; numerical/precision comparisons; service measurements; and a release/rollback dossier. An independent reviewer should be able to answer:

- Why did this source mixture and tokenizer match the intended task?
- What changed at each objective boundary, and what observation justified promotion?
- What prevents target leakage, false-negative misuse and benchmark contamination?
- Which improvement survives changed seeds/data and which remains uncertain?
- Can a clean consumer reproduce the embeddings and the reported retrieval behavior?
- Which runtime, language, domain and operating conditions were actually qualified?

## Planned implementation order

Build the tiny bidirectional encoder and independent checks first; connect the three objectives with full-state checkpoints; implement data/evaluation/monitoring manifests; qualify the smaller real-data pilot; then select the 300M recipe and compute budget. Implement release packaging and a local benchmark adapter before spending on the large run. Roll the proven pattern into each modality with its own objective and evaluation contract.

This capstone reuses the roadmap's 61 lesson units and adds explicit project-stage teaching for data-mixture decisions, objective transitions, embedding diagnostics, benchmark governance, Hub publication and index lifecycle. Those stage tutorials are required project work; they are not silently counted as already authored lessons.
