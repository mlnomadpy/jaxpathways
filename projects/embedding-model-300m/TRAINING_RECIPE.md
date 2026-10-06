# Training recipe: from masked tokens to useful embeddings

Status: proposed experiments for the [flagship project](README.md). Settings here are starting hypotheses, not a validated recipe. The learner must retain the comparison that selects each final value.

## Start with the task and the population

The first implementation targets English text retrieval and general text similarity. Multilingual capability is a separate branch with explicit language sampling, tokenizer analysis and per-language evaluation. The choice keeps the first release assessable; it does not prevent a later multilingual recipe.

Write what a relevant result means, what requests look like, typical/maximum document lengths, latency and memory constraints, and the expected domains. Freeze independent train, recipe-development and final evaluation populations before selecting data. Keep a lexical retrieval baseline and an identified pretrained encoder baseline alongside our from-scratch stages.

## Choose and audit data mixtures

Every source records dataset/revision, acquisition method, content hashes, source/group IDs, language/domain, rights and attribution notes, filtering/dedup versions, split role and retained counts. Preserve an exclusion/contamination report. A dataset card is evidence to inspect, not blanket permission to redistribute its underlying records.

Possible source families are curated web/educational text, reference/documentation text, technical/domain material, question–passage data and licensed curated pairs. [FineWeb-Edu's dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) is one candidate to inspect for the English corpus; no source revision, download or suitability decision has been completed here. Choose actual sources and pin them during implementation.

| Phase | Pilot mixture hypothesis | Sampling unit and selection question |
| --- | --- | --- |
| 1: MLM | 50% curated general text, 20% reference/documentation, 20% technical/domain text, 10% question/answer prose | Fractions of **consumed non-padding tokens**. Does extra domain text improve the target domain without damaging general probes? |
| 2: broad contrastive | 50% related spans from the same source, 25% weak query–passage pairs, 15% title–body pairs, 10% other audited semantic pairs | Fractions of **positive pairs**. Are pairs actually semantically relevant rather than merely sharing boilerplate or a source? |
| 3: supervised contrastive | 60% target query–relevant-document pairs, 25% semantic/paraphrase pairs, 15% general retrieval pairs retained for coverage | Fractions of **positive pairs**. Does specialization improve target retrieval while respecting the declared general-quality regression margin? |

These percentages each sum to 100%; they are not claims of optimality. Run a small baseline mixture and one controlled change under equal measured token/compute budgets. Report both token and pair exposure; pair percentages alone can hide large length differences. Cap repeated exposure of small sources, version the sampling schedule and compare intended versus observed shares. For multilingual work, choose language shares from task needs and pilot data; report low-resource repetition rather than letting a smoothing exponent hide it.

Deduplicate before splitting/chunking; retain near-duplicate clusters and source-group identity. Audit benchmark queries and labeled examples against every stage, including mined pairs, teacher-generated labels and tokenizer-training samples. If benchmark corpus documents overlap unlabeled pretraining, disclose the exact overlap and follow that benchmark's policy; do not claim a universal zero-shot result. Final query labels and judgments never enter recipe selection. Test-only data cannot become hard negatives.

## Tokenizer and input decisions

Train candidate BPE/unigram tokenizers on the training corpus only, with a byte-level baseline for comparison. Record normalization, whitespace/case handling, byte fallback, vocabulary, special tokens and software version. Compare token counts per source/language, truncation frequency, unknown/fallback behavior and throughput. More vocabulary consumes model parameters and changes sequence length; it is not automatically better.

Freeze the chosen tokenizer before Phase 1. Define PAD, MASK and boundary IDs, selected-token eligibility, document boundaries and maximum length. Introduce `query:`/`passage:` or other task prefixes deliberately before contrastive training, and preserve their exact use in evaluation and release. Compare prefixes on development data rather than assuming they help all tasks. Changing vocabulary after MLM requires a separately documented embedding migration and renewed training/parity checks.

For embedding output, begin with padding-aware mean pooling of the final token states, a declared special-token inclusion policy, then L2 normalization with a defined zero-vector rule. Compare CLS pooling on the same development protocol. Return stable dimensions and preserve input order across batching. Do not pool padding, masked pretraining artifacts or another packed document into a representation.

## A concrete approximately 300M architecture candidate

Use a bidirectional pre-normalized Transformer with 24 blocks, width 1,024, 16 attention heads, head width 64, a two-matrix GELU feed-forward layer of width 3,072, vocabulary 49,152, bias-free attention/feed-forward linear layers, two affine LayerNorms per block and one final affine LayerNorm. RoPE supplies positions without learned position parameters. Initial context is 512 tokens; a 2,048-token extension requires separate long-input training and evaluation. No wider context is implied.

For this exact candidate, with \(L=24\), \(d=1024\), \(f=3072\), \(V=49152\):

\[
P_{\mathrm{encoder}}=Vd+L(4d^2+2df+4d)+2d=302{,}090{,}240.
\]

The terms are token embeddings, Q/K/V/output matrices, two feed-forward matrices, the block LayerNorm parameters and final LayerNorm. Mean pooling and normalization add no trainable parameters. A tied MLM decoder with a dense transform/bias, affine norm and vocabulary bias adds 1,100,800 parameters, yielding 303,191,040 during MLM. Remove only the MLM-specific head from the inference package. Count unique tied parameters once.

The arithmetic was checked while writing this plan; the actual architecture has not been implemented or allocated. Its implementation must count the parameter tree and agree with the derivation. This is an original configuration hypothesis, not a named pretrained model's architecture. For comparison, the authors' [ModernBERT overview](https://huggingface.co/blog/modernbert) describes encoder models of different sizes and design choices; reusing its name or checkpoint would require matching its actual configuration.

At 302,090,240 parameters, BF16 weights alone occupy about 0.604 decimal GB. FP32 parameters, gradients and two Adam moments total about 4.833 GB before activations, extra copies, buffers, communication or allocator overhead. These are storage calculations, **not training-memory or device-count recommendations**. Measure peak memory at actual lengths, batch sizes, objective mixtures and sharding.

## Phase 1: masked language modeling

Question: can context recover hidden content without leaking the answer? Use a full bidirectional encoder, original token targets and a separately generated corruption mask. A 15% eligible-token selection rate with an explicitly documented replacement policy is a pilot baseline; compare span/whole-word masking only as separately tested variants. Boundaries and PAD are ineligible, and an empty selected-target batch must be rejected or handled under a documented policy.

\[
L_{\mathrm{MLM}}=-\frac{\sum_{b,t}m_{bt}\log p_\theta(x_{bt}\mid\widetilde{x}_b)}{\sum_{b,t}m_{bt}}.
\]

Here \(x\) is the clean target, \(\widetilde{x}\) the corrupted input and \(m\) the selected-position mask. Packing must preserve document attention boundaries and loss counts. For distributed/microbatched training, combine loss sums and selected counts; averaging local means is wrong when counts differ.

Retain FP32 sensitive reductions/optimizer state under a tested mixed-precision policy. Select optimizer, rate, warmup, decay, global valid-token budget, length schedule and clipping from measured pilots; record decay exclusions and whether a scheduled step advances after a rejected update. Include an uninterrupted-versus-restored comparison with dropout, mask RNG, sampler state and optimizer clocks.

Inspect held-out selected-token loss/accuracy by source and length, gradient stability and a fixed frozen-encoder retrieval/probe panel. Exponentiated masked-token loss is not autoregressive perplexity and should not be compared as such. MLM success alone does not qualify sentence embeddings.

## Phase 2: broad contrastive pretraining

Question: can related inputs become useful neighbors? Initialize from the identified Phase 1 checkpoint. Train clean paired views through the encoder and pooling path. Keep pair provenance, confidence and positive-group IDs; same-document spans can be weak positives, but are not guaranteed to mean the same thing.

For normalized embeddings \(q_i,d_j\), temperature \(\tau>0\), eligible candidate set \(C_i\) and positive set \(P_i\subseteq C_i\), one explicit multi-positive objective is:

\[
L_{q\to d}=-\frac1N\sum_i\log
\frac{\sum_{j\in P_i}\exp(q_i^\top d_j/\tau)}
{\sum_{j\in C_i}\exp(q_i^\top d_j/\tau)}.
\]

This log-sum-positive objective is one choice, not identical to averaging separate per-positive cross-entropies. Explain which behavior is desired. Add a reverse-direction term for genuinely symmetric pair tasks; asymmetric query–document tasks need a deliberate direction choice. Use stable logsumexp, valid-anchor counts, duplicate masking and independently checked gradients.

Compare **contrastive-only** continuation against **joint MLM plus contrastive** continuation:

\[
L=\lambda_mL_{\mathrm{MLM}}+\lambda_cL_{\mathrm{contrastive}}.
\]

The joint arm uses separately specified corrupted MLM inputs and clean contrastive views. Log both component losses and gradients. Pilot normalized weights and schedules; a raw sum of differently scaled losses is not a justified recipe. The promotion decision uses retrieval, representation health and retention, not the combined loss alone. [E5](https://arxiv.org/abs/2212.03533) and [multi-stage GTE](https://arxiv.org/abs/2308.03281) motivate staged contrastive study; this proposed joint schedule is our experiment, not a reproduction claim.

Cross-device negatives require a defined all-gather/gradient contract and a one-device oracle. Ordinary gradient accumulation increases examples per update but **does not increase the contrastive denominator** across microbatches. If gradient caching, a memory bank or cross-batch negatives is added, teach and verify that algorithm separately. Record effective candidate counts and false-negative policies so loss comparisons remain interpretable.

## Phase 3: supervised contrastive fine-tuning

Question: does the representation rank relevant results for the intended task? Continue from Phase 2 on curated query/positive pairs and reviewed hard negatives, with source-disjoint validation. Mine negatives from a pinned training corpus using an identified model/index; retain known positives and near-duplicate groups. Teacher labeling is optional and records teacher/version/prompt/confidence; synthetic labels are not human judgments.

Begin with the chosen contrastive objective. Treat distillation, margin losses and dimension-truncation objectives as explicit later ablations. Semantic pairs and retrieval pairs may need distinct direction/prefix conventions. A random unrelated example and a plausible-but-incorrect hard negative teach different distinctions; inspect false negatives before increasing hardness.

Use smaller update budgets selected on development results, a fixed general-retention panel and task-specific quality gates. Re-mining creates a new recipe revision. Retain Phase 1/2/3 results under the same evaluator so the learner can see both gains and forgetting.

## Stage transition contract

Each child run records its parent checkpoint hash, tokenizer/pooling identity, data revision, objective/reduction, precision and distributed configuration. Decide explicitly whether optimizer state and schedule clocks continue or reset; a changed objective is not an exact resume. Test exact resume **within** each phase. Promotion snapshots contain complete training state; inference releases contain the selected encoder and its consumer contract.

Before a large run, fill in the actual per-phase token/pair budget, maximum updates, checkpoint/evaluation cadence, storage retention, measured throughput/peak memory, wall-time estimate, stop conditions and available hardware. Estimate time from steady-state measured throughput plus evaluation/checkpoint/startup overhead; retain uncertainty. No generic “tokens per parameter” rule or downstream score is guaranteed by this plan.

Required figures: source mixture/length histograms, corruption/attention masks, per-source MLM curves, positive/negative similarity distributions, covariance spectrum, phase-by-phase retrieval and retention, and actual quality-versus-runtime tradeoffs. Each must include explained units and population, an observed value, a mechanism and a limitation.
