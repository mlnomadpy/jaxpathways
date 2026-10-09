# Evaluation: demonstrate useful representations

Status: planned evaluation protocol for the [flagship capstone](README.md). No benchmark result is asserted. Dataset versions, exact task IDs and acceptance thresholds must be frozen before the candidate's final evaluation.

## Separate three evaluation jobs

| Tier | When | Purpose |
| --- | --- | --- |
| Correctness smoke | Every relevant code change and before a run | Independently check masking, pooling, order, shapes, normalization, loss and numerical parity; tiny controlled inputs |
| Development panel | Periodically during pilots/training and at phase boundaries | Choose recipes/checkpoints using designated development data; source/length/domain slices and an economical retrieval/probe panel |
| Frozen release evaluation | On the selected release candidate and every advertised precision/runtime variant | Report the predeclared benchmark suite, domain holdout, robustness and system gates; do not select hyperparameters on these results |

A final result that changes the recipe turns the inspected data into development evidence. Record that fact and reserve a new independent holdout for the next final claim. Benchmark-required probe training may use its permitted training split under the official protocol; that is different from tuning our encoder on the benchmark test set.

## Text benchmark matrix

| Capability | Planned benchmark/protocol | Metrics and interpretation |
| --- | --- | --- |
| General text embeddings | Pin **MTEB(eng, v2)** for the initial English model; expand the identifier into exact task versions/splits | Preserve each task's primary metric, category summaries, missing/failed tasks and official aggregation. Do not mix v1/v2 scores or average incompatible raw metrics ad hoc. |
| Out-of-domain retrieval | Declared [BEIR](https://github.com/beir-cellar/beir) subset or full feasible suite, with trained-on/overlapping tasks labeled | nDCG@10, Recall@K and MRR where appropriate; retrieval corpus, qrels, query counts, tie/self-match conventions and exact versus ANN retrieval |
| Intended product domain | A frozen, source/time-separated query–document set reflecting actual relevance definitions | Primary retrieval metric plus critical slices, empty/no-relevant-result cases and error taxonomy; this determines the product gate |
| Semantic similarity and discrimination | Relevant MTEB similarity, pair/classification and clustering tasks | Use task-defined Spearman/accuracy/AP/clustering metrics and the prescribed probe; a retrieval score alone is verified separately from all representation properties |
| Multilingual extension | Pin **MTEB(Multilingual, v2)** only for a deliberately trained/claimed language set, plus cross-lingual retrieval/mining tasks | Per-language/task results, macro summaries, task coverage and low-resource gaps; an English run cannot imply multilingual qualification |
| Long-input and perturbation behavior | Frozen domain tests with changed length, important content position, punctuation, spelling, boilerplate and duplicates | Quality by length/condition, truncation rate and paired regressions; maximum accepted tokens alone do not prove long-context quality |
| Production retrieval stack | Same embeddings under exact search and the selected ANN/index configuration | Model quality separated from ANN recall, index build time/bytes, query latency and model/index-version compatibility |

The current [MTEB benchmark definitions](https://docs.mteb.org/overview/available_benchmarks/) include English, multilingual and multimodal suites. Do not describe MTEB as permanently text-only or hard-code an old dataset count. MTEB and BEIR may overlap; report coverage without counting shared tasks as independent evidence. “Everything” means complete coverage of declared capabilities, not running every public benchmark regardless of relevance or access.

## Freeze the benchmark manifest

Record evaluator package version/commit; benchmark identifier; expanded task IDs and dataset revisions; splits/subsets/languages; model checkpoint and tokenizer hashes; pooling/normalization; query/document prefixes; maximum lengths/truncation; output dimension/dtype; device; batch sizes; metric definitions; seeds; and raw-result paths. Retain the exact settings of external baselines too.

Include expected task inventory and per-task state: pending, completed, failed, unavailable or excluded-with-reason. Missing tasks never disappear from an average silently. Pinning a benchmark name without its expanded task list is insufficient because benchmark definitions can change. Explicitly mark parameter selection on a task or source overlap; do not present those tasks as untouched generalization evidence.

Implement and version an MTEB adapter against the [current model protocol](https://docs.mteb.org/get_started/usage/defining_the_model/). Test ordered output across batches, query/document modes, empty inputs, Unicode, singleton batches and length limits. A bare `encode(list_of_strings)` assumption may not match the selected framework release. Retain a small independent retrieval/metric oracle so passing an adapter does not conceal wrong ranking semantics.

## Compare the full lineage

Evaluate random initialization as a diagnostic where meaningful, the pooled Phase 1 encoder, Phase 2, Phase 3, a lexical baseline for retrieval and at least one pinned competitive encoder baseline. Compare equivalent preprocessing, test populations and index settings; clearly label instruction/prefix differences. Baseline selection should match intended language, use case and resource envelope rather than inventing a size-only leaderboard contest.

Retain full-precision and each advertised reduced-precision result. Measure pairwise cosine/logit error, embedding norm, nearest-neighbor/rank changes and task quality independently: small elementwise error can still change a close ranking. For dimension truncation, evaluate each advertised dimension; slicing a vector is not automatically a valid smaller embedding.

Use paired query-level comparisons and uncertainty at the appropriate unit (query/source group, speaker or image identity). Compare training seeds in pilots. If only one full-scale run fits the budget, state that its training variability is unmeasured rather than manufacturing confidence from many test examples.

## Quality, robustness and system release gates

Before final testing, fill in numeric target-domain minimums, maximum general/slice regressions, allowed missing-task policy, precision parity budgets, p95 latency under a stated request distribution/concurrency, memory/startup limits and recovery expectations. These values come from the target contract and pilot, not from a universal benchmark score.

Candidate promotion requires all relevant gates and an explicit decision record. Examples of automatic rejection are an incomplete mandatory evaluation inventory, nonfinite/zero embeddings outside the declared policy, tokenizer mismatch, corrupted weights, unacceptable critical-slice regression or an unbounded service queue. Statistical non-improvement may justify an inconclusive decision rather than a false win.

The learner should explain these figures:

- Stage comparison on identical tasks: what improved after MLM, contrastive pretraining and supervised pairs, and what regressed?
- Per-domain/language/length results with counts and missing coverage, not only a single average.
- Positive/negative cosine distributions and hard retrieval failures; high cosine does not necessarily mean relevance.
- Exact versus ANN search quality and runtime; identify which loss comes from the model and which from the index.
- Quality versus embedding dimension/precision/latency, with measured hardware and request boundaries.

Publish raw task outputs and aggregate-generation code where sharing is permitted, an overlap statement, failure cases, supported scope and unresolved limitations. Publishing a model and submitting to a leaderboard are separate actions with separate protocols.
