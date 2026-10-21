# Flagship capstone: build and release a 300M embedding model

**Full capstone status: planned. Bounded C01 SciFact CPU workflow: in development.** No 300M model has been trained, benchmarked, production-qualified or published by this project. The executed C01 pilot below demonstrates a small, real-data lifecycle workflow; it does not qualify the target architecture, general embedding quality or a production runtime. Full-capstone registration and target-scale acceptance remain pending. This is the advanced integration capstone for the [course expansion plan](../../docs/course-roadmap-2026-10-06.md), within the [seven-project lifecycle catalog](../../docs/model-lifecycle-capstones.md). Its embedding branches remain below; causal, image/audio generation and grounded multimodal models now have separate project specifications.

The learner will make and defend a sequence of decisions: what the model should retrieve, which data it may learn from, how inputs become representations, which objective belongs at each stage, whether the representation improves, and whether someone else can load and operate the released model.

“Production grade” is the intended outcome **for a declared use case and runtime**, earned through quality, reliability and release gates. A parameter count or an MTEB average alone is verified separately from it.

## Project guides

- [Data recipes, tokenizer, architecture and three training phases](TRAINING_RECIPE.md)
- [Evaluation, MTEB, retrieval and release benchmarks](EVALUATION.md)
- [MLflow, training metrics, monitoring and incident decisions](MONITORING.md)
- [Image, audio and cross-modal versions of the capstone](MODALITIES.md)
- [Packaging, numerical parity and Hugging Face publication](RELEASE.md)

## C01 SciFact CPU pilot

The first executable workflow is under `data/`, `model.py`, `objectives.py`,
`trainer.py`, `evaluation.py`, `package_runtime.py`, `run.py`, and `tests/`.
It implements a real bidirectional RoPE Transformer, a tied masked-token head,
two matched weak-pair phase-2 branches, and a human-evidence multi-positive
phase 3. It saves a full-coverage exact BM25 baseline, a random-initialized
encoder diagnostic, and a pinned external MiniLM-L6-v2 comparison. All quality
numbers are measured on the full SciFact corpus with the frozen official-dev
query inventory; this is a custom final protocol because the official test
split has no public gold judgments.

The official 3.1 MB data archive and every split inventory are byte- and
membership-hashed by [`data/frozen_recipe.json`](data/frozen_recipe.json).
The preparer writes normalized data outside the source tree and rejects any
archive whose SHA-256 differs. Claims/evidence terms and corpus abstract terms
are separate; see [`data/LICENSES.md`](data/LICENSES.md). The full candidate
corpus remains available at retrieval time, while its complete final-connected
document group and all final query text are excluded from tokenizer and
training exposure. Candidate-corpus exposure is disclosed as transductive.
There is a separate disclosed recipe-development document exposure: 29
development-component documents are among the 4,888 permitted corpus documents
used to fit the tokenizer, and the deterministic 128-document MLM sample
contains 3 of them. Recipe-development claim text and human query/evidence pairs
are excluded from training. Treat development retrieval metrics as document-text
transductive; they are not a clean unseen-document estimate.

### Numerical conventions for your implementation

Use the same conventions when you compare your model with the reference.
Both the encoder MLP and the masked-token head use the tanh approximation
to GELU, with cubic coefficient 0.044715. Layer normalization uses the mean
squared deviation across the final feature axis, then adds an epsilon of
0.00001 to that variance before taking its square root. A different GELU
approximation or epsilon can change a parity check even when shapes agree.

The masked-token head applies a dense transform with bias, GELU and layer
normalization to each encoder state. It then multiplies by the transpose of
the input token embedding matrix and adds a vocabulary bias. The shared
embedding must receive gradients from both its input and output uses.
Compute cross entropy only at selected masked-token positions, retaining
the loss sum and selected-token count before dividing. Padding positions
must not contribute to that count.

Before running a training phase, predict what should happen when you change
one selected target, a valid input token or a head parameter. Check raw logits
and selected loss values, then compare autodiff with finite differences of an
independent forward calculation. Matching outputs alone cannot establish that
gradients reach the encoder. Likewise, a finite-difference comparison with a
zero reference sensitivity cannot establish that a parameter is connected.

Create an isolated environment from the fully pinned lock before running the
CPU workflow. The baseline weights are about 90.9 MB and download from a pinned
public revision on first run. The offline option requires that exact snapshot
to exist in the output directory; missing files are errors.

On Linux, install the CPU-only PyTorch wheel before the lock to avoid pulling CUDA packages; this is installation guidance, not Linux execution qualification.

```sh
python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu 'torch==2.11.0'
```

POSIX shell:

```sh
C01_TMP="${TMPDIR:-/tmp}"
mkdir -p "$C01_TMP"
python3 -m venv "$C01_TMP/jaxpathways-embedding-env"
. "$C01_TMP/jaxpathways-embedding-env/bin/activate"
python -m pip install -r projects/embedding-model-300m/requirements-embedding-lock.txt
python projects/embedding-model-300m/data/prepare.py --download --output "$C01_TMP/jaxpathways-scifact-c01-data"
python projects/embedding-model-300m/tests/check.py --implementation solution --stage all --data "$C01_TMP/jaxpathways-scifact-c01-data" --output "$C01_TMP/jaxpathways-embedding-c01-check"
```

PowerShell:

```powershell
py -3 -m venv $env:TEMP\jaxpathways-embedding-env
& $env:TEMP\jaxpathways-embedding-env\Scripts\Activate.ps1
python -m pip install -r projects/embedding-model-300m/requirements-embedding-lock.txt
python projects/embedding-model-300m/data/prepare.py --download --output "$env:TEMP\jaxpathways-scifact-c01-data"
python projects/embedding-model-300m/tests/check.py --implementation solution --stage all --data "$env:TEMP\jaxpathways-scifact-c01-data" --output "$env:TEMP\jaxpathways-embedding-c01-check"
```

`run.py --stage phase1`, `--stage phase2`, `--stage phase3`, and `--stage all`
are explicit executable milestones. Each command builds its prerequisites
from the frozen recipe; an unavailable dependency, missing dataset, or missing
pinned baseline is an error. The default run directory is a fresh external
temporary folder. To work on the learner implementation, pass
`--implementation starter`; it intentionally raises until the required model
APIs are implemented.

The pilot precommits its small data and update budget in the frozen recipe.
The phase-2 joint branch is the declared phase-3 parent; development data does
not select a branch after the fact. The final quality floors are gates, not
claims of success. A short CPU run can establish mechanics, exact resume,
source coverage and local consumer parity only. MTEB, quality-generalization,
human review, full target training, accelerator qualification and remote Hub
download remain open.

Use the [shared tooling plan](../../docs/model-lifecycle-tooling.md) for equivalent Weights & Biases and MLflow tracking, bounded sweeps and artifact lineage. The tracking adapter is independent of the training objective.

## Measured pilot results

The retained `final8` run is a bounded CPU pilot, not a completed 300M capstone. Its retrieval figure compares seven systems on the same full corpus of 5,183 abstracts and the frozen 300-query SciFact development inventory. Only 188 queries have human evidence-document judgments; the other 112 remain in the inventory but are excluded from macro metric denominators. Recall@k divides retrieved judged positives by that query's number of relevant evidence documents, nDCG@k uses binary evidence relevance and its ideal ranking, and MRR is the reciprocal rank of the first relevant document. The horizontal axis is metric value from 0 to 1; blue bars are Recall@10 and orange bars are nDCG@10.

![Measured final8 retrieval metrics](reference/evidence/final8/figures/final-retrieval.svg)

| System | Recall@10 | nDCG@10 | MRR |
| --- | ---: | ---: | ---: |
| BM25 | 0.9242021 | 0.8218117 | 0.7959756 |
| Pinned MiniLM-L6-v2 baseline | 0.9117021 | 0.7742753 | 0.7353779 |
| Random initialization | 0.0159574 | 0.0120312 | 0.0149667 |
| Phase 1 MLM | 0.0265957 | 0.0091670 | 0.0089226 |
| Phase 2 contrastive-only | 0.0212766 | 0.0083259 | 0.0100976 |
| Phase 2 with retained MLM | 0.0265957 | 0.0099271 | 0.0101971 |
| Phase 3 human-evidence pairs | 0.0212766 | 0.0092948 | 0.0109189 |

The precommitted phase-3 floors were Recall@10 ≥ 0.30 and nDCG@10 ≥ 0.10; the observed 0.0212766 and 0.0092948 fail both. The result is a negative pilot outcome: it does not support useful retrieval quality or promotion to a larger run. BM25 and the pinned MiniLM baseline are strong references on this small task, and every trained checkpoint remains far below them.

Each panel has its own phase-local x axis—8, 8, 8, and 71 updates—and its own
zero-preserving y axis, so values from different objectives are not compared
on a shared scale. Each panel names its loss units: phase 1 uses nats per
selected masked token; phase 2 contrastive-only uses the contrastive objective;
its joint branch combines contrastive and weighted MLM loss; phase 3 uses
multi-positive contrastive loss. Learner runs and the retained teaching
figure use the same generic `plots.py::loss_figure` renderer; the evidence-
bound wrapper checks the frozen final8 inputs before calling it. Its panel
renderer and figure are bound to
the unchanged trace and figure data by [`teaching/manifest.json`](reference/teaching/manifest.json)
and the explicit [`teaching_extension`](reference/evidence-index.json) entry.

Update 71 in phase 3 contains one query, one positive pair, and one candidate
document. Its loss of 0 reflects a single-example final batch and is not
evidence of retrieval quality. The historical combined plot remains
byte-for-byte unchanged at
[`evidence/final8/figures/training-objectives.svg`](reference/evidence/final8/figures/training-objectives.svg); its original hash is recorded alongside the new plot hash.

![Four phase-specific views of the final8 training objectives](reference/teaching/objective-panels.svg)

Actual training exposure from the saved traces was:

| Stage | Consumed examples | Input exposure | Additional note |
| --- | --- | --- | --- |
| Phase 1 MLM | 32 distinct documents | 2,048 nonpadding token positions; 281 selected masked positions | 2 documents are from the disclosed recipe-development document-text subset |
| Phase 2 contrastive-only | 32 weak pairs | 3,659 nonpadding pair-input tokens | Same pair inventory as the joint branch |
| Phase 2 with retained MLM | 32 weak pairs plus 32 MLM documents | 3,659 pair-input tokens; 2,048 MLM token positions; 292 selected masked positions | 1 consumed MLM document is from the recipe-development document-text subset |
| Phase 3 human-evidence pairs | 281 distinct fit queries and 212 positive documents | 12,916 query-token positions; 20,096 candidate-document token positions | Candidate documents are encoded during the objective; counts are not unique training pairs |

The tokenizer and frozen source exposure, per-query metrics, raw figure data, seven complete ranking arrays, shared qrels and source hashes are recorded in the [evidence index](reference/evidence-index.json). The [full pilot record](reference/README.md) explains the axes, exposure, provenance, omissions and qualification limits. The two phase-2 branches had matched pair exposure; the joint branch also consumed its retained MLM batches.

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

The artifact names in this full-capstone stage table are proposed stage-level deliverables, not claims that every full-capstone acceptance artifact already exists. The bounded C01 implementation files and measured evidence are identified above; they do not complete the full capstone or its target-scale gates. Each full stage still needs a prediction, an executed figure, a changed condition and a failure diagnosis. The model card is written from final receipts rather than planned targets.

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

## Implementation gate and real-data lineage

Do not promote a pilot until its source-group split, tokenizer, MLM/contrastive/pair-stage checkpoint ancestry, complete training state and held-out retrieval protocol have receipts. The real-data pilot must disclose overlap, source exposure, permissions and a frozen test set; a synthetic fit only verifies code paths.

Before this plan can change from **planned** to **authored**, provide independently checkable dataset/recipe manifests, staged parent-to-child checkpoint records with full resume state, real held-out evaluation outputs, and runnable plan-specific starter/reference/checks. Keep small CPU fixtures labeled as correctness evidence; they do not establish target-scale training, quality or production readiness.

## Recipe decision rehearsal before target-scale training

MLM validation improves while query-to-document retrieval degrades after contrastive continuation.

**Decide:** Treat MLM, unsupervised contrastive continuation and supervised pair adaptation as distinct checkpoints with frozen parent identities. Compare pooling, length policy, source mixture and hard-negative policy one at a time on development tasks. Use a pinned MTEB task/version protocol only after checking data overlap, and preserve an untouched final evaluation.

**Review packet:** A checkpoint lineage graph, tokens/pairs seen by source, objective-specific denominators, retrieval baselines, per-task/slice results and a cold-load Hugging Face artifact test. The existing embedding-lineage pilot remains preparation, not a 300M reference.

This is a design exercise for a planned implementation. Record a hypothesis, a controlled pilot, a stopping criterion and the evidence needed to reject the recipe before requesting target-scale compute. Do not mark the capstone implemented or publish a model card with invented metrics.
