# Phase 07: Transformers & language models

Training systems.

Build attention and causal masking from small arrays, then connect tokenization, mixed precision, training and generation.

Inspect what each token can attend to and count only valid targets in the loss.

**Prerequisites:** 06: Data & checkpoint recovery.

**Hardware:** CPU for small blocks; TPU or GPU for training experiments.

## Study guide: Can you prove that a predicted token cannot see its future?

Attention is a weighted mixing operation whose axes and masks are testable. Build a causal model from these contracts before interpreting a generated sentence as evidence of learning.

### Check your starting point

A query has equal scores for three allowed keys. What are the attention weights, and what happens if one key is masked?

<details><summary>Compare your reasoning</summary>

The weights are equal thirds. Masking one key leaves equal halves on the two allowed keys and zero on the masked key. Normalize over keys, not queries.

</details>

Review: [Attention from small pieces](01-attention-from-small-pieces/docs/en.md).

### Build in stages

1. **Audit attention and targets.** Check one attention row by hand, token/target shifts, padding denominators and segment boundaries. Perturb future tokens and inspect earlier outputs.

   Lessons: [Attention from small pieces](01-attention-from-small-pieces/docs/en.md) · [Masking, tokenization, and sequence packing](02-masking-tokenization-and-sequence-packing/docs/en.md).

2. **Train, resume and decode one small model.** Preserve residual shapes and sensitive arithmetic, compare the next resumed update and state the decoding policy. Continue into the text harness for cache/full-prefix parity.

   Lessons: [A Transformer block and mixed precision](03-a-transformer-block-and-mixed-precision/docs/en.md) · [Train, checkpoint, and generate](04-train-checkpoint-and-generate/docs/en.md).

### Try a changed condition

Two documents are packed into one sequence. Causal masking passes, but one document still changes the other’s predictions. What is missing?

<details><summary>Compare an approach</summary>

Causality only blocks future positions. Require matching segment identities too, align targets within each document, and exclude padding or invalid boundaries from the loss. Test a change to the other document while holding the target document fixed.

</details>

**Symptom:** Padding produces NaNs in attention.

**Check next:** Count permitted keys in every query row. A fully masked softmax needs an explicit invalid-query policy; its output and loss must not become valid training evidence.

### Decide what is ready

Use text-harness stages 1–4. Keep independent attention arithmetic, future-token invariance, resumed Adam state and full-prefix/cache parity with declared precision.

### Further work

Real-corpus curation, tokenizer tradeoff studies, long-context evaluation and live generative quality need further labs beyond the tiny byte-model harness.

## Lesson sequence

### 07.01 Attention from small pieces

[Read the lesson](01-attention-from-small-pieces/docs/en.md) · [Run the code](01-attention-from-small-pieces/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Implement scaled dot-product attention and test its output shape.

**Evidence:** Save the Q/K/V axis diagram, uniform hand calculation, NumPy nonuniform reference, paired permutation check, and wrong-axis repair.

**Checkpoint:** Which dimension should softmax normalize for one query to combine all key/value pairs?

### 07.02 Masking, tokenization, and sequence packing

[Read the lesson](02-masking-tokenization-and-sequence-packing/docs/en.md) · [Run the code](02-masking-tokenization-and-sequence-packing/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Construct a causal mask and prevent packed sequences from attending across boundaries.

**Evidence:** Keep token/target alignment, the two masks, future perturbation, segment-leakage comparison and all-masked-row repair. Verify that masks and ignored loss positions enforce the same boundaries.

**Checkpoint:** What must a packed causal mask enforce?

### 07.03 A Transformer block and mixed precision

[Read the lesson](03-a-transformer-block-and-mixed-precision/docs/en.md) · [Run the code](03-a-transformer-block-and-mixed-precision/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Build a residual attention block and compare numerical behavior across precision choices.

**Evidence:** Keep the architecture diagram, zero-projection identity, causal perturbation, independent normalization, finite-difference coordinate check, precision discrepancy and residual-width repair.

**Checkpoint:** If all projection matrices are zero in this block, what should the output be?

### 07.04 Train, checkpoint, and generate

[Read the lesson](04-train-checkpoint-and-generate/docs/en.md) · [Run the code](04-train-checkpoint-and-generate/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Train a tiny causal model, restore it, and generate from a fixed seed.

**Evidence:** Save the target alignment, uniform-loss derivation, training/held-out metrics with corpus limitations, snapshot continuation check, generated cycle and metadata/prompt rejection repairs.

**Checkpoint:** Why is saving only parameters insufficient to replay the next Adam update?

## Phase project

A checkpointed causal language model.

**Demonstrate:** Verify masking, sequence handling, restoration, and generation before comparing speed.

Project status: implemented staged practice · [Open source](../../projects/text-harness/README.md). Use stages 1, 2, 3, 4 for this phase. Begin with token contracts and training. Complete the linked performance and deployment prerequisites before cache, export and profiling checks.



[Primary documentation](https://flax.readthedocs.io/).
