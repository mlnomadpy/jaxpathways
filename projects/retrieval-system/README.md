# Build a measured retrieval service

Status: **planned capstone**. This is a system specification; the real-corpus index, service, benchmark results and downloadable workspace remain to be implemented. The [cross-modal harness](../cross-modal-harness/README.md) provides runnable synthetic preparation.

## The question and the delivered system

Can the system return useful items quickly, keep deleted items out, and survive a model upgrade without silently mixing incompatible vectors? Build a search service that returns ranked IDs, scores, source/version metadata and inspected evidence. It accepts a qualified text embedding encoder or paired encoders from the [CLIP](../clip-model/README.md) or [SigLIP](../siglip-model/README.md) projects. A declared audio–text encoder can become another branch after audio pairing is qualified; an audio classifier does not automatically supply a shared text space.

Use document retrieval as the first complete path, then add image–text retrieval with the same index/evaluation contracts and modality-specific preprocessing. Keep generation optional: finding evidence and generating an answer require different evaluations. A retrieval service can be complete without a chatbot.

## Preparation and boundaries

Complete paired contrastive learning, data splits, inference capacity, precision and the [engineering release project](../engineering-release/README.md). Use its container and tracking practices, plus the [weight-conversion project](../weight-conversion/README.md) when importing an encoder. The capstone adds real-corpus ingestion, relevance judgments, index implementations, service integration and migrations; existing toy checks do not cover those additions.

Start with a tiny hand-judged corpus, move to a measured real-data pilot, then qualify the actual corpus size, query workload and runtime. There is no new 300M parameter target here: consume an independently qualified encoder and count all model/index storage in the deployment budget.

## Stage 1: version the corpus and evaluation contract

Record stable document/media IDs, source revisions, rights, content hashes, language/domain, timestamps and access metadata. Define chunk boundaries and stable chunk-to-parent mapping for documents; define processor versions and media metadata for images/audio. Keep original evidence locatable. Split related sources before training an encoder or reranker and isolate final test queries from tuning. A retrieval evaluation corpus may contain the relevant held-out documents; that is different from leaking their query labels into training.

Create qrels: query-ID/item-ID relevance judgments, with a declared binary or graded scale. Include multiple positives, near duplicates, no-answer queries and intentionally stale/deleted items. Unjudged documents are not automatically known irrelevant; state how the chosen benchmark treats them. Fix whether evaluation ranks chunks or parents and how duplicate chunks count.

**Check:** change chunking or preprocessing without rebuilding the index. The manifest must reject the mismatch rather than serve incompatible vectors.

## Stage 2: establish exact and lexical baselines

Implement a tiny independent exact dot-product ranker over unit-normalized vectors. Test norm handling, stable tie policy, multiple positives, empty results, filters and malformed dimensions. Record the embedding space identity: encoder checkpoint, query/document prompts, tokenizer/processor, pooling, normalization, dimension and dtype. Matching vector shapes do not imply compatible spaces.

Build a lexical BM25 baseline for document text, captions or transcripts where available; images themselves are not BM25 tokens. Compare dense and lexical failures on the same queries. Begin with exact search before choosing an approximate nearest-neighbor (ANN) index such as an appropriate Faiss configuration. Pin the library and index parameters when implementation begins.

## Stage 3: distinguish search approximation from relevance

For each query, let \(G_k\) be its exact top-\(k\) IDs and \(A_k\) its ANN top-\(k\) IDs. With enough eligible candidates and a fixed tie rule:

\[
\mathrm{ANNRecall}@k=\frac{|A_k\cap G_k|}{k}.
\]

This asks whether the index reproduced the exact encoder ranking. It does not ask whether those items are useful. Relevance recall instead divides the number of retrieved relevant items by the total number of judged relevant items for that query. Report both measures and their aggregation rules.

For example, with relevant IDs A and C and results B, A at cutoff two, relevance recall is \(1/2\) and reciprocal rank is \(1/2\). The index could still have ANN recall of one if B, A is also the exact top-two ranking. That means the approximation is faithful while the model's first result is wrong. These are hand-worked expectations, not benchmark results.

For graded relevance, define gain \(g(r)=2^r-1\), where \(r\) is the relevance grade, and compute:

\[
\mathrm{DCG}@k=\sum_{i=1}^{k}\frac{2^{r_i}-1}{\log_2(i+1)},\qquad
\mathrm{nDCG}@k=\frac{\mathrm{DCG}@k}{\mathrm{IDCG}@k}.
\]

IDCG is the ideal ordering under the same judgments and cutoff. State the convention for zero ideal gain; separately count queries with no judged positives. Validate Recall@K, MRR and nDCG on independent hand cases, then compare with a pinned evaluator. Report query counts, uncertainty and domain/language slices.

## Stage 4: qualify ANN, hybrid retrieval and reranking

Sweep the chosen ANN search/build parameters on development queries, keeping final test queries untouched. Measure index build time, bytes, memory, exact-neighbor recall, relevance, latency and concurrency on the named hardware. Apply eligibility filters consistently to the exact oracle and candidate search; filtering after top-K can underfill results or expose forbidden items. Test deletions and rebuild/update semantics explicitly.

Combine lexical and dense candidates with a documented method such as reciprocal-rank fusion. If using scores, calibrate them before blending; cosine and BM25 scales are not interchangeable. Train or adapt a cross-encoder reranker using only permitted training judgments and mined negatives from training material, with false-negative audits. Keep the negative miner/checkpoint version. For image–text reranking, use a scorer that actually accepts those modalities rather than assuming a text reranker works.

Compare lexical, dense exact, dense ANN, hybrid and reranked systems with a common candidate budget. A reranker cannot recover a relevant item that never reached its candidate set. Report candidate recall alongside final ranking quality and added latency.

## Stage 5: inspect figures and benchmark the frozen system

- **Query/ranked-result panel:** label query, source IDs, relevance grades, ranks and component scores. Explain whether one failure came from missing candidates or incorrect reranking.
- **Quality–latency plot:** horizontal axis is measured p95 end-to-end latency in milliseconds; vertical axis is nDCG or relevance recall. Label corpus size, hardware, concurrency and index settings. Better quality at higher latency is a tradeoff; different workloads are not comparable points.
- **ANN recall–memory plot:** show exact-neighbor recall and index bytes across settings. Explain why matching exact neighbors cannot prove semantic usefulness.
- **Migration comparison:** plot old/new query results and rank changes on frozen canaries. Trace one changed result to encoder, index, filtering or reranker versions before blaming quantization.

Every figure needs raw measurements, units, corpus/query IDs and an explanation of what it requires separate verification to establish. Use BEIR tasks for a declared text-retrieval scope and relevant MTEB retrieval tasks when qualifying a text encoder; list actual tasks and failures instead of claiming the whole benchmark. Use fixed COCO/Flickr30k-style protocols for image–text retrieval where permitted. Audio retrieval needs a separately selected, rights-reviewed audio–text benchmark and explicit caption/clip handling; task selection remains an implementation decision.

## Stage 6: precision and the service boundary

Separate encoder weight/activation precision from stored-vector precision and ANN compression. Product quantization of the index is different from quantizing model weights. Compare each change alone and combined against the float/exact baseline, with calibration/training data isolated from final tests. Measure rank/quality regressions, storage, memory and actual latency; smaller bytes do not guarantee faster requests.

Containerize an API that validates query type/size, top-K bounds, timeouts, filtering and declared authorization. Measure tokenization/decoding, query encoding, index lookup, filtering and reranking separately and end to end. Test cold start, overload, cancellation, missing artifacts and malformed requests. Return empty/no-answer behavior under a declared validation policy rather than trusting an arbitrary cosine threshold.

## Stage 7: migrate, publish and monitor

Create an immutable manifest binding encoder, processors, prompts, corpus snapshot, index, reranker, precision and evaluation report. New encoder weights require re-embedding and rebuilding unless compatibility is actually established. Use a shadow index and canary queries, then atomically switch the complete compatible bundle. Rehearse rollback and deletion propagation; no request may combine a new query encoder with an old incompatible index. Never infer compatibility from equal dimension.

Publish releasable encoder weights/processors to Hugging Face with a model card and immutable revision; publish a rebuild recipe and allowed demo corpus separately from private/restricted source content. Test cold reconstruction and consumer canaries. Track experiment artifacts through local records and MLflow/W&B, including corpus/query revisions, metrics, mining provenance and precision policy. Serve telemetry through a focused stack such as OpenTelemetry/Prometheus when implemented.

Monitor p50/p95/p99 latency, error/timeout rates, saturation, index freshness, filter/deletion correctness, empty-result rate, embedding drift and judged relevance on a delayed evaluation stream. Log stage timings and artifact IDs. Clicks are biased feedback and do not directly measure relevance. Keep raw queries/documents out of telemetry by default and define appropriate retention for the chosen use case.

## Completion evidence and teaching still to implement

Supply versioned corpus/qrels, independent metric oracles, exact and lexical baselines, ANN tradeoffs, hybrid/reranker ablations, interpreted error cases, service load results, cold rebuild, compatibility rejection, deletion and rollback tests. Author progressive starter/reference/check files for those stages before registration as runnable. Production claims require measured quality and service gates on the actual target corpus/runtime.

## Primary references

- [BEIR evaluation toolkit](https://github.com/beir-cellar/beir): retrieval evaluation protocols and baseline integrations.
- [Faiss index selection guide](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index): index tradeoffs; qualify the selected implementation on the target workload.
- [MTEB](https://github.com/embeddings-benchmark/mteb): select and pin relevant text-embedding retrieval tasks; do not substitute its aggregate for the service audit.
