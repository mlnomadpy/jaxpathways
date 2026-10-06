# Phase 01: Arrays & pure functions

Shared foundations.

Treat every axis as part of a contract. Work through normalization, broadcasting and immutable updates, then diagnose mistakes that still return plausible numbers.

Draw the input and output shapes before running each change.

**Prerequisites:** 00: Setup & first steps.

**Hardware:** CPU.

## Study guide: Can you spot a wrong axis even when the program runs?

Array bugs often return plausible numbers. Use unequal dimensions and hand-computed rows so that a successful operation cannot hide a mistaken meaning. Carry that habit into preprocessing and model inputs.

### Check your starting point

A matrix has three observations and two features. You need one mean per feature. Which axis should be reduced, and how many values remain?

<details><summary>Compare your reasoning</summary>

Reduce the observation axis, axis 0. Two feature means remain. Reducing axis 1 instead returns three observation means, which answers a different question.

</details>

Review: [From NumPy to jax.numpy](01-from-numpy-to-jax-numpy/docs/en.md).

### Build in stages

1. **Write the data contract.** Annotate observation and feature axes; calculate one normalized row by hand. Compare a feature bias with an observation bias using unequal dimensions.

   Lessons: [From NumPy to jax.numpy](01-from-numpy-to-jax-numpy/docs/en.md) · [Shapes, broadcasting, and dtypes](02-shapes-broadcasting-and-dtypes/docs/en.md).

2. **Make preprocessing safe to reuse.** Pass parameters explicitly, preserve the caller’s arrays and check indexed accumulation separately from replacement. Carry shape, mask and valid-count checks into foundation-toolkit stage 2.

   Lessons: [Pure functions and explicit inputs](03-pure-functions-and-explicit-inputs/docs/en.md) · [Immutable updates and indexing](04-immutable-updates-and-indexing/docs/en.md).

### Try a changed condition

Normalize a matrix with a constant feature and one masked observation. What should the constant feature and the mean denominator do?

<details><summary>Compare an approach</summary>

Define the constant-feature policy before division, for example a unit fallback scale. Compute statistics only from valid observations; masking the final output alone leaves the fitted statistics contaminated. Reject a feature with no valid observations or declare an explicit policy.

</details>

**Symptom:** Predictions look reasonable but adding a bias changes the wrong dimension.

**Check next:** Write the two shapes before broadcasting and test one row and one column independently.

### Decide what is ready

Retain unequal-shape, constant-column and masked-row cases, a NumPy or hand reference, and proof that inputs were unchanged. Explain every axis in the output.

### Further work

Sparse and ragged inputs need additional contract-focused labs; this phase deliberately teaches fixed dense arrays first.

## Lesson sequence

### 01.01 From NumPy to jax.numpy

[Read the lesson](01-from-numpy-to-jax-numpy/docs/en.md) · [Run the code](01-from-numpy-to-jax-numpy/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Build feature-wise preprocessing with fitted statistics and verify behavior on new and constant data.

**Evidence:** Keep the fitted mean/scale, hand-derived training values, NumPy comparison, constant-feature test, and together-versus-separate new-batch results. Explain why re-fitting on each inference batch changes the problem.

**Checkpoint:** Why does x.mean(axis=0) return two values?

### 01.02 Shapes, broadcasting, and dtypes

[Read the lesson](02-shapes-broadcasting-and-dtypes/docs/en.md) · [Run the code](02-shapes-broadcasting-and-dtypes/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Specify broadcasting contracts and detect silent pairwise-loss and dtype mistakes.

**Evidence:** Keep an axis sketch, asymmetric offset checks, aligned-versus-pairwise residual arrays, a rejected shape mismatch, and the float32 precision observation. Explain the intended residual contract.

**Checkpoint:** Which offset shape adds one scalar to each row of a (2, 3) batch?

### 01.03 Pure functions and explicit inputs

[Read the lesson](03-pure-functions-and-explicit-inputs/docs/en.md) · [Run the code](03-pure-functions-and-explicit-inputs/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Design an explicit parameter/data interface and return new state without mutating old containers.

**Evidence:** Keep both parameter configurations, their known predictions, a container-alias experiment, a non-mutating update, and an evaluation loss checked by hand. Explain which state the caller owns.

**Checkpoint:** What makes predict easier to transform and test?

### 01.04 Immutable updates and indexing

[Read the lesson](04-immutable-updates-and-indexing/docs/en.md) · [Run the code](04-immutable-updates-and-indexing/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Implement functional indexed updates and distinguish accumulated contributions from shape-preserving masks.

**Evidence:** Keep original and updated arrays, a repeated-index integer reference, the whole-matrix replacement check, and compact-versus-fixed-shape outputs. Explain replacement, accumulation, and name rebinding.

**Checkpoint:** After y = x.at[1].set(20.), what happens to x?

## Phase project

A tested array-processing function.

**Demonstrate:** Predict output shapes and explain why a pure function is easier to transform.

Project status: implemented staged practice · [Open source](../../projects/foundation-toolkit/README.md). Use stages 2 for this phase.



[Primary documentation](https://docs.jax.dev/en/latest/beginner_guide.html).
