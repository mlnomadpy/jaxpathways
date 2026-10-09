# Monitor the experiment and the released system

Status: planned monitoring implementation for the [flagship capstone](README.md). Extend the existing real MLflow experiment and release labs. The [shared tooling plan](../../docs/model-lifecycle-tooling.md) adds an equivalent Weights & Biases adapter with runs, artifacts, media tables and bounded sweeps; choose one primary tracker and preserve the same local metric/artifact contract. The W&B integration is planned, not yet executed.

## MLflow lineage

Create a project experiment with distinct runs for data/tokenizer decisions, pilots, Phase 1, Phase 2, Phase 3, evaluation, conversion, quantization and release. Use parent/child relationships where useful and retain explicit immutable input/output artifact IDs. A phase transition starts a new run with its parent checkpoint identity; a process retry within a phase retains the recovery lineage and attempt number.

Log source commit plus dirty-source snapshot/hash, dependency/container versions, device topology, precision, complete config, data recipe, tokenizer/processor, preprocessing/pooling, objective reduction, global batch/candidate definitions, seeds and checkpoint acceptance state. Track **attempted updates, applied optimizer updates, consumed valid tokens/pairs and elapsed time** separately. Do not overwrite metrics at a reused local step after recovery without an attempt/continuation convention.

Store configs/manifests, compressed summaries, explained figures, raw evaluation results, checkpoint references and incident records as artifacts. Verify that a chosen run's exact model can be retrieved and loaded. Dashboard screenshots and mutable aliases alone are insufficient. Keep raw documents, restricted recordings, credentials and unnecessary per-example text out of routine logs.

## Metrics, cadence and response

Cadences below are starting measurement policies. Tune them after profiling their overhead. Compute local summaries on device, aggregate correctly, and avoid a host synchronization or full-layer histogram at every update.

| Signal | What to record | Starting cadence | What an unexpected result should trigger |
| --- | --- | --- | --- |
| Data and progress | Valid tokens, selected MLM targets, positive pairs, padding/truncation, source/language shares, dedup/rejection rates, sample cursor | Counts each update; log aggregated windows every 10–50 updates | Compare actual mixture/exposure and denominator; stop if split/identity contracts fail |
| Phase 1 objective | Loss sum and selected count, masked accuracy, per-source/length held-out results | Training windows; fixed probes at configured token milestones | Inspect masking/target leakage, source imbalance and corrupted examples before tuning rates |
| Phase 2/3 objectives | Separate contrastive/MLM components, weights, temperature, anchor/candidate/positive counts, active hard negatives and pair confidence | Training windows; pair audit per data/mining revision | Investigate denominator changes, false negatives or a dominant objective; do not compare raw losses from different candidate pools as identical |
| Update stability | Loss finiteness, gradient norm before/after clipping, clipping fraction, parameter/update norms, learning rate, optimizer step, skipped-update count | Cheap summaries each window; fail fast for nonfinite state | Preserve the last accepted checkpoint and failing configuration; inspect data, loss scaling and update order before resuming |
| Layer numerics | Sampled activation/gradient RMS and extrema, normalization variance, overflow/underflow indicators, optimizer-state norms | Sampled every 100–500 updates or on an alert | Find the first layer/operation that diverges; compare a bounded FP32 reference |
| Representation health | Pre/post-normalization norms, zero-vector rate, cosine positive/negative gap, per-dimension variance and covariance spectrum/effective-rank summary | Fixed held-out sample at evaluation boundaries | Examine collapse, anisotropy, duplicates or label leakage; low effective rank alone is not an automatic diagnosis |
| Quality/retention | Frozen development retrieval/probes, source/length slices, general-task retention and qualitative errors | Scheduled valid-token/pair milestones; every phase transition | Halt promotion or investigate a regression; final benchmarks remain outside frequent tuning |
| Systems | Completed tokens/pairs per second, step distribution, data wait, recompilation, communication, peak device memory, host memory and storage | Aggregated windows; periodic target traces | Separate input stalls, compilation and compute; do not equate device utilization with useful throughput |
| Recovery | Save request/completion/accepted step, checkpoint age, upload duration/errors, restoration result | Every save/restore and injected failure | Never publish an incomplete checkpoint as accepted; stop duplicate writers and verify replay |
| Budget | Consumed tokens/pairs, elapsed/reserved runtime, storage and estimated versus actual cost where observable | Each run summary and budget boundary | Stop at configured bounds and reassess using measured throughput; retain actual versus assumed prices separately |

Effective rank and similarity statistics depend on sample selection and centering. Keep that procedure fixed, save the sample IDs and compare to earlier checkpoints. Changing the sample at every step can create apparent representation changes. Log positive/negative statistics by source/length as well as globally; metadata shortcuts may look like successful learning.

## Alert decisions, not decorative charts

Hard contract failures—nonfinite state, zero valid denominator, source/split mismatch, broken checkpoint checksum or impossible pair labels—stop the affected run. Soft trends such as rising clipping, changing norms, stalled validation or high data wait trigger a bounded diagnostic run. Define rolling windows and thresholds from pilots, record the reason for any change, and avoid treating a single noisy observation as a universal failure.

Learners must rehearse at least: wrong masking denominator; duplicated positive treated as negative; missing optimizer/sampler state; nonfinite precision failure; stalled input; and a rejected phase checkpoint. Each incident includes symptom, relevant plot/log, independent check, repair and observed outcome.

## After deployment

Track request/error counts, rejected lengths/payloads, queue depth, timeouts, cancellation, batch sizes, cold-start time, end-to-end p50/p95/p99 with sample counts, throughput, resource use and accepted model/tokenizer/processor/index IDs. A few local samples requires separate verification to establish a reliable tail percentile.

Embedding-specific service monitoring includes output dimension/dtype/norm, zero/nonfinite outputs, truncation rates, input/domain/language shifts and query/index incompatibility. Retrieval quality requires delayed labels, sampled judgments or a controlled evaluation panel; input drift or click changes alone do not prove a model regression.

Replacing an embedding model generally requires re-embedding documents and building a compatible index. Use a versioned candidate index, replayed canary queries, staged traffic and rollback to the **matching model plus index**, not just old weights. Any new model, tokenizer, pooling, normalization, dimension or precision policy can change the embedding space.

## Required dashboards and learner explanations

Keep four linked views: data exposure, objective/numerical health, retrieval/representation quality, and system/recovery behavior. Every chart names units, counts, phase boundaries and whether values are measured or estimated. Add short run-specific explanations: what changed, which evidence supports the cause, which competing explanation remains, and what action followed.

The [MLflow tracking documentation](https://mlflow.org/docs/latest/ml/tracking/) is the API reference to verify during implementation. The existing [engineering release guide](../engineering-release/README.md) provides the local executed starting point. No metrics from the proposed 300M run exist yet.
