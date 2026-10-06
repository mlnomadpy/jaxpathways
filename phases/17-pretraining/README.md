# Phase 17: Self-supervised pretraining: masked and contrastive learning

Training systems.

Train small masked-token, hidden-patch and paired-view models. Audit information leakage, reduction counts and pair identity before interpreting a falling objective.

Predict the mask, denominator or preference direction before training. Keep independent objective checks and changed-condition evidence.

**Prerequisites:** 07: Transformers & language models.

**Hardware:** Executed CPU synthetic objectives; large-scale and human-feedback evaluation remain separate.

## Study guide: What learning signal can unlabelled data provide?

Train small masked-token, hidden-patch and paired-view models. Audit information leakage, reduction counts and pair identity before interpreting a falling objective.

### Check your starting point

If an image reconstruction is perfect before training, what boundary should you inspect?

<details><summary>Compare your reasoning</summary>

Check whether hidden pixels entered the encoder or were copied into the prediction. A valid objective can still be defeated by leaked inputs.

</details>

Review: [Masked language modeling: predict hidden tokens](01-mlm/docs/en.md).

### Build in stages

1. **Masked language modeling: predict hidden tokens.** Masked language modeling predicts selected original tokens from a corrupted input. It differs from causal next-token prediction: allowed context can come from both sides. Our repeated-token fixture makes the correct answer inspectable; this is a small objective and attention experiment, not a BERT reproduction or a language benchmark.

   Lessons: [Masked language modeling: predict hidden tokens](01-mlm/docs/en.md).

2. **Masked image modeling: reconstruct missing patches.** We split small synthetic images into patches, expose only two patches to a linear encoder/decoder map and train on the other two. This teaches masking, spatial order and reconstruction accounting. It is not a full ViT masked autoencoder; the simple image generator makes independent checks possible.

   Lessons: [Masked image modeling: reconstruct missing patches](02-mim/docs/en.md).

3. **Contrastive learning: views, positives and negatives.** The lab trains a shared linear encoder with a symmetric cross-view contrastive loss. Each row’s paired view is its positive and other rows are negatives. This resembles the paired objective used in cross-modal retrieval; it is not the full SimCLR denominator, which includes additional same-view negatives.

   Lessons: [Contrastive learning: views, positives and negatives](03-contrastive/docs/en.md).

### Try a changed condition

A batch contains duplicate semantic pairs. Should each duplicate automatically be a negative?

<details><summary>Compare an approach</summary>

No. Decide positives by source or semantic identity and use a compatible objective; the duplicate-pair exercise shows how a diagonal-only loss penalizes duplicates.

</details>

**Symptom:** A low masked loss does not transfer to the downstream task.

**Check next:** Check target leakage, mask distribution, encoder-versus-projection features and a held-out downstream protocol.

### Decide what is ready

Complete stages 1–3 of training-methods with independent loss references, real training curves, changed masks or pairs and explicit synthetic-data limits.

### Further work

Full BERT/ViT masked autoencoders, natural corpora, augmentation studies and downstream probe benchmarks remain extensions. The current labs execute small attention, linear reconstruction and embedding models.

## Lesson sequence

### 17.01 Masked language modeling: predict hidden tokens

[Read the lesson](01-mlm/docs/en.md) · [Run the code](01-mlm/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Separate the clean targets from the corrupted input. Normalize over selected target positions.

**Evidence:** Keep corrupted inputs, selected-target counts, the log-vocabulary baseline, independent masked-loss checks, trained parameters and changed-mask results.

**Checkpoint:** Why are visible input tokens useful even when their output positions are excluded from the loss?

### 17.02 Masked image modeling: reconstruct missing patches

[Read the lesson](02-mim/docs/en.md) · [Run the code](02-mim/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Preserve patch order and pixel order. Keep hidden pixels out of the encoder.

**Evidence:** Keep patch-order slices, visible-only feature shapes, hidden-patch counts, independent MSE and held-amplitude reconstruction results.

**Checkpoint:** Why must the encoder input be inspected separately from reconstruction loss?

### 17.03 Contrastive learning: views, positives and negatives

[Read the lesson](03-contrastive/docs/en.md) · [Run the code](03-contrastive/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Define the positive pair before augmenting. Normalize embeddings and introduce temperature.

**Evidence:** Keep source pairing, temperature, stable two-direction loss checks, collapse and duplicate baselines, training outputs and broken-pair diagnosis.

**Checkpoint:** What does successful retrieval on the training pairs establish?

## Phase project

An audited pretraining and adaptation objective toolkit.

**Demonstrate:** Complete stages 1–3 of training-methods with independent loss references, real training curves, changed masks or pairs and explicit synthetic-data limits.

Project status: implemented staged practice · [Open source](../../projects/training-methods/README.md). Use stages 1, 2, 3 for this phase.



[Primary documentation](https://docs.jax.dev/en/latest/).
