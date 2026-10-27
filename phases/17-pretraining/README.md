# Phase 17: Self-supervised pretraining: masked and contrastive learning

Training systems.

Train small masked-token, hidden-patch and paired-view models. Audit information leakage, reduction counts and pair identity before interpreting a falling objective.

Predict the mask, denominator or preference direction before training. Keep independent objective checks and changed-condition evidence.

**Prerequisites:** 07: Transformers & language models.

**Hardware:** Executed CPU synthetic objectives; large-scale and human-feedback evaluation remain separate.

## Study guide: What learning signal can unlabelled data provide?

Self-supervised learning creates targets from the data itself. That makes the information boundary especially important: a hidden token or patch must remain a target rather than leaking into the input, and a contrastive positive must keep its source identity when views move around a batch.

Work through one selected token, one patch and one row of a similarity matrix. Then read the recorded objectives beside their baselines and held-out checks. Saving and reloading an encoder gives the next exercise a concrete starting point; the evaluation determines whether the learned representation helps under a changed condition.

### Check your starting point

If an image reconstruction is perfect before training, what boundary should you inspect?

<details><summary>Compare your reasoning</summary>

Check whether hidden pixels entered the encoder or were copied into the prediction. A valid objective can still be defeated by leaked inputs.

</details>

Review: [Masked image modeling: reconstruct missing patches](02-mim/docs/en.md).

### Build in stages

1. **Masked language modeling: predict hidden tokens.** Trace clean targets, corrupted IDs, attention axes and selected counts. Reproduce unequal-mask loss with a host calculation, derive the full logit gradient, and explain the final probability matrix.

   Lessons: [Masked language modeling: predict hidden tokens](01-mlm/docs/en.md).

2. **Masked image modeling: reconstruct missing patches.** Verify every patch on a rectangular RGB input, intervene on hidden pixels, and explain an ambiguous target that no deterministic decoder can reconstruct perfectly. Compare actual hidden targets with predictions.

   Lessons: [Masked image modeling: reconstruct missing patches](02-mim/docs/en.md).

3. **Contrastive learning: views, positives and negatives.** Derive a two-pair loss, change temperature without changing ranking, compare encoder gradients with finite differences, and diagnose broken identity or duplicate false negatives from the similarity matrix.

   Lessons: [Contrastive learning: views, positives and negatives](03-contrastive/docs/en.md).

### Try a changed condition

Two rows in a contrastive batch refer to the same semantic positive. What assumption does a diagonal-only target violate?

<details><summary>Compare an approach</summary>

It assumes one unique positive per row and treats the duplicate positive as a negative. Write the positive matrix before choosing an objective. A multi-positive loss or deduplication policy changes the task definition; report which one is used and keep evaluation groups separated from training.

</details>

**Symptom:** A low masked loss does not transfer to the downstream task.

**Check next:** Check target leakage, mask distribution, encoder-versus-projection features and a held-out downstream protocol.

### Decide what is ready

Complete stages 1–3 of training-methods with independent loss references, real training curves, changed masks or pairs and explicit synthetic-data limits. Add the new lesson evidence: unequal-count gradient oracle, rectangular patch/intervention checks, irreducible ambiguity example and temperature/ranking comparison. Explain each saved figure without copying its caption. Complete the saved-encoder handoffs: record data/mask/processor identities, reload the artifacts, evaluate held-out groups and report frozen probes against independent baselines. Keep the demonstrated MLM and MIM limitations visible rather than calling objective completion a quality pass.

### Further work

Full BERT/ViT masked autoencoders, natural corpora, augmentation studies and downstream probe benchmarks remain extensions. The current labs execute small attention, linear reconstruction and embedding models.

## Lesson sequence

### 17.01 Masked language modeling: predict hidden tokens

[Read the lesson](01-mlm/docs/en.md) · [Run the code](01-mlm/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Audit selected-token MLM loss and transfer a saved small encoder to held-out masks and a frozen representation probe.

**Evidence:** Keep clean/corrupted masks, selected-position denominators and independent gradients, plus saved encoder/processor identity, held-out loss versus uniform baseline, a frozen probe and honest changed-condition failures.

**Checkpoint:** Why are visible input tokens useful even when their output positions are excluded from the loss?

### 17.02 Masked image modeling: reconstruct missing patches

[Read the lesson](02-mim/docs/en.md) · [Run the code](02-mim/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Audit visible-only masked reconstruction and compare a saved small encoder across changed masks and a frozen representation probe.

**Evidence:** Keep patch-order and ambiguity controls, saved encoder/mask/transform identity, changed-mask held-out reconstruction against a visible-mean baseline and the frozen probe. Separate objective mechanics from representation quality.

**Checkpoint:** Why must the encoder input be inspected separately from reconstruction loss?

### 17.03 Contrastive learning: views, positives and negatives

[Read the lesson](03-contrastive/docs/en.md) · [Run the code](03-contrastive/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Train a bounded contrastive objective and check held-out grouped retrieval with multiple positives and a saved encoder identity.

**Evidence:** Keep pair/temperature/false-negative checks, held-out source-group membership, saved encoder identity, multiple-positive Recall@k and mAP calculations against an independent ranking oracle.

**Checkpoint:** What does successful retrieval on the training pairs establish?

## Phase project

An audited pretraining and adaptation objective toolkit.

**Demonstrate:** Complete stages 1–3 of training-methods with independent loss references, real training curves, changed masks or pairs and explicit synthetic-data limits.

Project status: implemented staged practice · [Open source](../../projects/training-methods/README.md). Use stages 1, 2, 3 for this phase. Copy `projects/training-methods/starter/methods.py` to `projects/training-methods/my_methods.py` and write your code in `projects/training-methods/my_methods.py`. Run `python3 projects/training-methods/tests/check.py --implementation projects/training-methods/my_methods.py --stage 1` from the top-level folder to verify stages 1, 2, 3.

Additional project: [Trace a tiny bidirectional encoder from pretraining to retrieval](../../projects/embedding-lineage/README.md).

[Primary documentation](https://docs.jax.dev/en/latest/).
