# SciFact C01 encoder — measured local model card

## Status and intended use

This is an in-development, CPU-trained two-layer pilot for educational SciFact evidence-retrieval experiments. It is not the proposed 300M model and has not passed its precommitted pilot retrieval floors. Regardless of that pilot result, it is not a qualified biomedical search system, general-purpose embedding model, or production candidate. The custom local JAX package API is described in `package.json`; this is not a `transformers.AutoModel` or Sentence Transformers class.

## Measured evaluation

The evaluation used all 5,183 SciFact abstracts as candidates and all 300 official development claims as a fixed custom final inventory. Metrics include the 188 claims with documented human evidence document IDs; the remaining 112 are listed as unjudged and excluded from metric denominators. Relevance comes only from the original evidence annotations. The official test split has no public judgments.

| System | Recall@1 | Recall@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Phase 3 human-evidence pilot | 0.0000 | 0.0213 | 0.0109 | 0.0093 |
| Phase 1 MLM | 0.0000 | 0.0266 | 0.0089 | 0.0092 |
| Frozen MiniLM-L6-v2 baseline | 0.6126 | 0.9117 | 0.7354 | 0.7743 |
| BM25 lexical baseline | 0.7017 | 0.9242 | 0.7960 | 0.8218 |

Precommitted pilot floors were Recall@10 >= 0.30 and nDCG@10 >= 0.10. The phase-3 pilot **failed** these floors. This is one deterministic seed and a small architecture; it gives no estimate of between-seed uncertainty. The fixed held-out corpus was rerun for engineering verification and is not a blinded benchmark. MTEB, expert review, broad-domain quality, target-scale training, target-runtime evaluation, and service qualification remain unrun.

The frozen external comparator is `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. It reports Apache-2.0, but its model metadata lists S2ORC and MS MARCO and possible SciFact overlap is unresolved; it is contextual, not contamination-free evidence.

## Architecture and input behavior

The executed pilot has 2 bidirectional pre-LN Transformer layers, hidden width 32, 4 attention heads, GELU two-matrix MLPs, RoPE positions, tied MLM embeddings, and float32 weights. The bundled byte-level BPE tokenizer adds BOS/EOS, truncates at 64 tokens, mean-pools content tokens only, and L2-normalizes. Empty-content inputs map to zero vectors. Query/document prefixes are not used. These describe the measured pilot, not the target architecture.

## Data provenance and rights

The source is the pinned AllenAI SciFact release, archive SHA-256 `11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be`. Its claims and evidence annotations are CC BY 4.0; the scientific abstract corpus originates from S2ORC under ODC-By 1.0; upstream code is Apache-2.0. See the bundled `DATA_PROVENANCE.md` for attribution and source links. The package assigns no separate model-weight license; do not redistribute it until source-rights implications have been reviewed.

The prepared MLM inventory contained 128 documents and 8192 nonpadding token positions; these are prepared input totals, not consumed totals. Phase 1 actually consumed 32 distinct documents (2048 nonpadding input tokens; 281 selected masked-token positions). The joint phase-2 branch consumed 32 distinct retained-MLM documents (2048 nonpadding tokens; 292 selected masked-token positions) alongside 32 weak pairs. Phase 3 consumed 281 distinct human-evidence fit queries, 212 positive documents, 12916 nonpadding query tokens, and 20096 nonpadding candidate-document tokens. Exact consumed document/query ID inventories and per-query metrics are in bundled `MEASURED_EVIDENCE.json`.

Recipe-development query text and final query text did not enter training pairs. Development-component document text entered the eligible tokenizer corpus (29 documents; 3 in the prepared MLM inventory), but the actual phase-1 and retained-MLM consumed subsets contained 2 and 1 such documents respectively. The final-connected document group and final query text were excluded from training. This is a disclosed transductive setup, not an unseen-document estimate.

## Reproducibility

The external source-bound receipt binds the exact candidate, recipe, data outputs, tokenizer, environment, checkpoints, report, and run artifacts. This README is generated from measured report values and its SHA-256 is recorded in `package.json`. Pinned MiniLM weights are not bundled; their revision and file hashes are recorded in the separate baseline receipt.
