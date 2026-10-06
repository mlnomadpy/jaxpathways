# Train a CLIP image–text model

Status: **planned capstone**. This is a staged implementation specification. The real-data trainer, reference checks, executed figures and downloadable workspace are not implemented yet. Start now with the [CPU cross-modal harness](../cross-modal-harness/README.md); its synthetic results do not establish natural-image quality.

## The question and the finished artifact

Can a caption and its image meet in the same embedding space, even when we never trained a classifier for that caption's concept? Build two encoders with a shared embedding dimension, then release the processors, weights and evidence needed to use them independently. The image encoder consumes pixels; the text encoder consumes token IDs. Their inputs differ, but their normalized outputs can be compared by a dot product.

This project teaches CLIP-style alignment and zero-shot transfer. The [SigLIP project](../siglip-model/README.md) changes the training objective under a controlled comparison. The [retrieval project](../retrieval-system/README.md) consumes either qualified checkpoint and builds a search service.

## Preparation and scale

Complete the cross-modal harness, paired contrastive pretraining, image inputs, attention/masking, full-state recovery and precision lessons. The [training-methods project](../training-methods/README.md) is the objective preparation; the [engineering release project](../engineering-release/README.md) supplies tracking and container practice.

Use three budgets: a tiny correctness fixture, a measured real-image pilot, and a separately budgeted target-scale run. Approximately 300M is an optional total dual-encoder budget, including both towers and projections. Report the image/text allocation, frozen/trainable counts, input resolution, text length and embedding dimension. A 300M image encoder plus a text encoder exceeds that total. Compare from-scratch and pretrained initialization as separate experiments with separate provenance.

## Stage 1: choose and inspect the pair recipe

Create a versioned manifest with image ID, caption ID, source/group ID, rights, image checksum, text and split. Inspect contact sheets with the actual caption below each image. Split by source/near-duplicate group before training; multiple captions of the same image must stay together. Pin resize/crop/channel normalization and tokenizer assets, padding, truncation and pooling. Fit learned assets only on training material.

Begin with a small documented image–caption collection whose rights permit the experiment. Treat COCO/Flickr30k as candidate evaluation protocols, not automatically clean training sources. Pin the specific split and exclude evaluation overlap from all training sources. Choose a source mixture from measured pilot quality and exposure, not arbitrary percentages. Keep source weights, examples seen, caption lengths and filtering attrition in the recipe.

**Check and failure:** shuffle captions without changing tensor shapes. The pair audit must detect the corruption; shape checks alone cannot. Mark known multiple positives and uncertain pair relationships explicitly.

## Stage 2: make the contrastive calculation visible

Let \(u_i\) and \(v_j\) be unit-normalized image and text embeddings; \(n\) is the number of paired examples. A learned log-scale \(a\) gives logits \(s_{ij}=\exp(a)u_i^\top v_j\). First implement the original one-positive-per-row/column case:

\[
L_{\mathrm{CLIP}}=-\frac{1}{2n}\sum_{i=1}^{n}\left[\log\frac{\exp(s_{ii})}{\sum_j\exp(s_{ij})}+\log\frac{\exp(s_{ii})}{\sum_j\exp(s_{ji})}\right].
\]

Each image must distinguish its paired caption from the other captions, and each caption must distinguish its image. Use stable log-softmax. With two pairs and all logits zero, each direction gives probability \(1/2\) to the correct match, so the mean loss is \(\log 2\). A confidently wrong match increases the loss. This is an analytic check, not a training result.

Add a separately named multi-positive variant after the diagonal baseline: define the positive mask and whether the objective averages positive log-probabilities or sums positive probability mass. Those objectives differ. Never silently use diagonal labels when two captions legitimately describe the same image. Audit both forward values and gradients against an independent small calculation, including changed batch size and permuted pair order.

## Stage 3: pretrain, resume and adapt

Phase A trains the paired objective from a declared initialization. Phase B continues on a justified domain mixture while checking broad-domain retention. Phase C uses curated domain pairs and reviewed hard negatives; compare frozen-tower, full fine-tuning and a supported adapter configuration. There is no mandatory MLM phase for this dual-encoder project. Any auxiliary objective is a separate ablation with its own weight and denominator.

Save parameters, optimizer, scale, random state, sampler cursor, preprocessing and recipe identity. Within a phase, compare several uninterrupted/resumed updates in a fresh process. Between phases, record the parent checkpoint and explicit schedule/optimizer resets. Keep the same held-out pool throughout recipe selection. Stop on nonfinite values and diagnose gradients, clipping and scale before increasing model size.

For distributed training, specify local anchors, global candidates, label offsets, gather-gradient behavior and the reduction over devices. Compare values and encoder gradients with a single-device global-batch oracle. Accumulating gradients from isolated microbatches does not reproduce a larger pool of negatives.

## Stage 4: evaluate what alignment bought you

Report image-to-text and text-to-image retrieval separately, with fixed candidate pools, positive sets and Recall@1/5/10. State whether recall means any-positive hit rate or the fraction of all relevant items recovered. Add zero-shot classification with pinned label vocabulary, prompt templates and prompt-ensemble rule; select prompts on development data only. Include a frozen pretrained baseline, random initialization, shuffled pairs, and domain/language slices with counts. Pin COCO/Flickr30k and classification protocols actually used; a text-only MTEB score cannot substitute for image–text evidence.

## Figures learners must produce and explain

- **Pair contact sheet:** show original image, transformed input and caption. Explain a crop that removes the named object and why this weakens supervision.
- **Similarity matrix:** rows are held-out images, columns are captions, color is cosine similarity. Mark every valid positive. A bright diagonal is helpful only under this ordering; bright off-diagonal cells may be valid duplicate semantics or failures. Inspect the actual examples.
- **Learning and retrieval curves:** horizontal axis is pairs seen, vertical axes are training loss and held-out retrieval on separate panels. Falling loss with flat retrieval suggests overfitting or misaligned evaluation; it does not prove which cause without further checks.
- **Ranked gallery:** place the query beside top results, stable IDs, ranks and relevance judgments. Explain one hard negative and one annotation ambiguity without cherry-picking only successes.

All plots must retain raw measurements, seed, checkpoint and pool identity. These figures are required outputs, not existing measured results.

## Stage 5: qualify precision, publish and operate

Compare float32 with supported bfloat16/float16 and calibrated integer variants, distinguishing weight, activation and accumulation precision. Measure embedding drift, rank changes and task quality; an allclose check alone does not qualify retrieval. Calibrate on separate development data. Check each tower independently and together after serialization; use the [weight-conversion project](../weight-conversion/README.md) as preparation, extending its dense-layer checks to actual attention, pooling and processor semantics.

Publish both towers, projections, tokenizer, image processor, scale, architecture, license/data provenance, evaluation inventory and model card. Test a cold load at an immutable Hugging Face revision with fixed image/text canaries. Publish only a format supported by the advertised consumer, verified in a clean environment. Release no score until its run exists.

Record loss denominators, source exposure, positive/negative similarity, embedding norms, logit scale, encoder gradient norms, updates, throughput and memory in a local event record; send the same record to MLflow or W&B. Include inspectable image-caption tables, artifact lineage and explicit optimizer-step versus examples-seen axes. Containerize the consumer, measure real request latency and rehearse paired model/index rollback through the retrieval project.

## Completion evidence and teaching still to implement

Deliver the source manifest, objective oracle, resumed-state comparison, pilot ablations, both retrieval directions, zero-shot protocol, interpreted plots, precision/runtime table, cold-load canaries and release/rollback rehearsal. Authored status requires staged starter/solution/check files and executed evidence for each stage. Natural-image quality, accelerator performance and production readiness remain unverified until the corresponding gates pass.

## Primary references

- [CLIP paper](https://arxiv.org/abs/2103.00020): source for the dual-encoder softmax objective and zero-shot transfer formulation.
- [OpenAI CLIP implementation](https://github.com/openai/CLIP): reference preprocessing and inference contracts; pin a revision when implementing parity.
