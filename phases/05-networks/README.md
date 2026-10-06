# Phase 05: Neural network training

Training systems.

Build the same ideas into small neural networks. Understand hidden layers, parameter state, stable objectives and evaluation before moving to a small image classifier.

Explain a loss curve together with held-out errors; a lower training loss alone is not sufficient.

**Prerequisites:** 04: Math & optimization.

**Hardware:** CPU first; accelerator for larger experiments.

## Study guide: Did the network learn the task, or exploit an evaluation mistake?

Move from explicit matrix operations to stateful modules while preserving independent checks. Learn to distinguish capacity, optimization, data quality and evaluation errors before adding layers.

### Check your starting point

A classifier predicts the majority class on every example and reports high accuracy. What information is missing?

<details><summary>Compare your reasoning</summary>

Class counts and per-class errors are missing. A majority baseline can score highly on imbalanced data while missing every rare example. Inspect the confusion matrix and the denominator for each class.

</details>

Review: [Train a CNN on a small dataset](04-train-a-cnn-on-a-small-dataset/docs/en.md).

### Build in stages

1. **Connect equations to module state.** Check a forward pass and unreduced loss independently, then identify parameter and nonparameter state in the NNX model.

   Lessons: [Build a tiny multilayer perceptron](01-build-a-tiny-multilayer-perceptron/docs/en.md) · [Model and state with Flax NNX](02-model-and-state-with-flax-nnx/docs/en.md).

2. **Train and evaluate under separate contracts.** Keep held-out examples out of updates, verify evaluation leaves state unchanged and read confusion rows as true-class denominators.

   Lessons: [A compiled train and evaluation step](03-a-compiled-train-and-evaluation-step/docs/en.md) · [Train a CNN on a small dataset](04-train-a-cnn-on-a-small-dataset/docs/en.md).

3. **Diagnose one failure at a time.** Use recorded input ranges, losses, gradient norms and update sizes to distinguish a scale problem from a data or shape problem.

   Lessons: [Debug unstable learning](05-debug-unstable-learning/docs/en.md).

### Try a changed condition

Training loss falls, but recall for a rare class is zero. Write the next three checks before changing the model.

<details><summary>Compare an approach</summary>

Inspect class counts and labels; compare a constant predictor and per-class confusion; then inspect score distributions and choose any threshold on validation data. Report final metrics on held-out data without using it to tune the repair.

</details>

**Symptom:** Evaluation results change even though you did not call an optimizer.

**Check next:** Compare state snapshots and train/eval flags; stochastic or running-statistic behavior may still be active.

### Decide what is ready

Use mlp-classifier before the larger image harness. Retain independent forward/loss checks, two runs with declared seeds, state-pure evaluation and class-specific failure analysis.

### Further work

Natural-image dataset ingestion, richer augmentation studies and pretrained vision transfer remain larger follow-up labs; synthetic images do not establish real-photo generalization.

## Lesson sequence

### 05.01 Build a tiny multilayer perceptron

[Read the lesson](01-build-a-tiny-multilayer-perceptron/docs/en.md) · [Run the code](01-build-a-tiny-multilayer-perceptron/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Implement a small model with explicit parameters before adding a model library.

**Evidence:** Keep the 33-parameter shape calculation, independent NumPy cross-entropy, train and held-out seeds, initial/final loss, nearby held-out accuracy, and one repaired score/label shape failure.

**Checkpoint:** Explain why removing tanh makes two affine layers collapse to a linear XOR boundary, then verify a changed label convention.

### 05.02 Model and state with Flax NNX

[Read the lesson](02-model-and-state-with-flax-nnx/docs/en.md) · [Run the code](02-model-and-state-with-flax-nnx/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Express a model and inspect the distinction between parameters and other state.

**Evidence:** Keep the NumPy forward comparison, count of 13 trainable scalars and one diagnostic counter, counter independence after copying state, and logits for batch size one and five.

**Checkpoint:** Explain why exclusion from Param gradients does not prevent forward state mutation, and demonstrate safe snapshot independence.

### 05.03 A compiled train and evaluation step

[Read the lesson](03-a-compiled-train-and-evaluation-step/docs/en.md) · [Run the code](03-a-compiled-train-and-evaluation-step/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Build train and eval steps and check that evaluation does not update weights.

**Evidence:** Keep pre/post parameter snapshots around evaluation, optimizer step count 80, independent held-out loss and accuracy, complete replay comparison, and a deliberately bad held-out update on an isolated clone.

**Checkpoint:** Identify which weights a pre-update loss describes and prove evaluation changes neither parameters nor optimizer step count.

### 05.04 Train a CNN on a small dataset

[Read the lesson](04-train-a-cnn-on-a-small-dataset/docs/en.md) · [Run the code](04-train-a-cnn-on-a-small-dataset/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Train a classifier with a fixed data split and report held-out metrics.

**Evidence:** Keep the 8-by-8-to-6-by-6 shape diagram, NumPy patch dot product, seeds 20/21, independent held-out cross-entropy, 24-observation confusion matrix, constant-class baseline, and layout failure repair.

**Checkpoint:** Explain what global spatial mean pooling discards and verify that confusion counts and trace reproduce the reported held-out accuracy.

### 05.05 Debug unstable learning

[Read the lesson](05-debug-unstable-learning/docs/en.md) · [Run the code](05-debug-unstable-learning/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Introduce a bad learning rate and diagnose it using loss and gradient evidence.

**Evidence:** Keep the analytic error recurrence, per-step weight/loss/gradient norm for rates 0.1 and 0.3, observed 3.24 loss multiplier, changed-scale comparison, clipping bound, and one repaired nonfinite-input case.

**Checkpoint:** Derive why scaling features and targets by ten changes this squared-loss curvature by 100 without treating that bound as universal.

## Phase project

Build and audit a neural classifier.

**Demonstrate:** Keep train and evaluation behavior separate and reproduce metrics from a saved configuration.

Project status: implemented staged practice · [Open source](../../projects/mlp-classifier/README.md).



[Primary documentation](https://flax.readthedocs.io/).
