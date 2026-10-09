# Models synthesis: defend training, recovery and image deployment

**Scope:** classifier fundamentals plus a connected image lifecycle: raw-pixel preprocessing, CNN training, full-state recovery, frozen error analysis, calibrated precision, actual exported inference and CPU measurements. This assessment builds on the [classifier](../project.html?id=mlp-classifier) and [image-harness](../project.html?id=image-harness) projects.

[Open classifier fundamentals](../project.html?id=mlp-classifier) and [the image lifecycle harness](../project.html?id=image-harness). Keep your evidence in [My learning](../notebook.html#portfolio).
The image tasks assess training through recovered and exported inference. Language-model-specific masking, generation and text-serving contracts belong to the separate text harness; this image evidence does not silently certify them.

Complete `networks-01` through `networks-05`, the classifier project, the recovery lessons and the deployment/precision lessons linked by `projects/image-harness/README.md`. Retain your own implementation; the public tests and provided solution remain separate evidence.

## Prepare an assessment workspace

Use the tested CPU environment and record Python, JAX, NumPy, Flax and Optax versions, actual backend/device count and dtype. From the checkout, run:

```sh
# Run run command in terminal using the course Python environment
python projects/mlp-classifier/tests/check.py --stage 3 --implementation projects/mlp-classifier/my_model.py
```

This project checker accepts an implementation path. If your file has another name, replace the path above. Keep a separate `assessment.py` harness and `report.md`; do not modify public tests to relax their contracts. Record predictions before execution. The work is suitable for several study sessions; there is no claimed time-to-mastery.

## Task 1: transfer the model to a new width and singleton input

Build the same two-layer tanh model at hidden width **five**, rather than the project’s default eight:

```text
input (B,2) → Linear(2,5) → tanh → Linear(5,1) → logits (B,)
```

Derive the parameter count and identify every kernel/bias shape. Verify it against NNX parameter state, excluding optimizer memory and non-parameter variables.

On a fresh disposable model, assign these explicit parameters so the forward calculation is reproducible:

```text
hidden kernel = [[1,0,-1,2,0.5], [0,1,1,-1,2]]
hidden bias   = [0,0.5,-0.5,0,1]
output kernel = [[1],[-2],[0.5],[0],[1]]
output bias   = [-0.25]
input         = [[1,-2]]
```

Write the five pre-activations, their tanh values and the final logit by hand or an independent NumPy expression. Compare the NNX forward result within `rtol=1e-5, atol=1e-6`. Then check batches of one, three and four inputs: the output must remain `(B,)`, including `(1,)` for the singleton case.

**Keep:** shape diagram, parameter derivation, known-parameter forward calculation, NNX comparison and the singleton-axis check.

## Task 2: defend the objective at extreme logits

For logits `[-100,0,100]` and labels `[0,1,1]`, derive the stable mean binary cross-entropy:

```text
mean(logaddexp(0,logit) - label*logit)
```

Explain what the zero-logit observation contributes. Predict the derivative when one scalar bias shift is added to all three logits. Compare that derivative with autodiff and central differences at `0.01` and `0.002`. Use an independent host `logaddexp` objective for the difference calculation.

In a disposable diagnostic function, compute sigmoid probabilities and then `-label*log(p)-(1-label)*log(1-p)` in float32. Show whether its extreme-logit result is finite and identify the unstable intermediate operation. Restore the logit-based objective and demonstrate finite values. Do not repair the test by deleting extreme observations.

Finally, on a newly initialized width-five NNX model, compare the derivative of the output bias at two seeds against your independent finite-difference calculation. Preserve/restore the perturbed parameter after every comparison. Report tolerances and explain why excessively small perturbations increase floating-point cancellation error.

**Keep:** scalar derivation, bias-shift derivative, stable/unstable results and two seeded parameter-gradient comparisons.

## Task 3: separate fitting, evaluation and reproducibility

Train your width-five model on the four XOR corners. Use seeds **2** and **5**, learning rate `0.03`, and **160** updates. For each seed, record the full pre-update loss history. Replay that seed/configuration independently and compare the full history and final model state within `rtol=1e-6, atol=1e-6`.

Define two held-out sets **before** fitting:

- Use `numpy.random.default_rng(202)` and `numpy.random.default_rng(303)` independently.
- Repeat each of the four corners ten times and add Gaussian input noise with standard deviation `0.18`.
- Derive each target from the sign rule `x0*x1 < 0`, using that set’s perturbed inputs.
- Keep these arrays separate from every training update.

Evaluate all four seed × held-out combinations. Verify loss and accuracy against independent NumPy logits and stable loss calculations. State the threshold/tie convention; the project classifies a logit strictly greater than zero as positive. Keep sample count and error IDs alongside each metric.

A newly changed width/seed/noise fixture has **no guaranteed accuracy threshold** in this assessment. Low accuracy is an observation to diagnose, not a reason to resample the held-out set until it looks good. Acceptance here depends on correct, isolated measurement and reasoned interpretation; public project thresholds still apply to its original fixture.

Snapshot every NNX model-state leaf before and after evaluation and verify exact equality. Snapshot optimizer state too if your assessment retains it. Show one comparison where the same model is evaluated on both fixed sets without fitting between them. Explain why four aggregate accuracies measures fixed-split performance before subgroup fairness testing.

**Keep:** the frozen held-out generation rule, seed/configuration table, histories, replay comparisons, error IDs, independent metric checks and evaluation-state comparison.

## Task 4: diagnose unequal-batch metrics

An evaluator processed one batch with 7 rows and 6 correct predictions, and another with 33 rows and 21 correct predictions. The two batch mean losses are `0.2` and `1.1`.

1. Derive global accuracy and global mean loss using counts.
2. Calculate the wrong unweighted mean of batch metrics and quantify each discrepancy.
3. Partition one of your 40-row held-out arrays into 7 and 33 rows. Compare count-weighted batch metrics with evaluation of the full array. State whether this particular model/set happens to make the wrong unweighted accuracy agree; the constructed count example above must still expose the bug.
4. Add an implementation guard or aggregation contract that requires positive counts and a nonempty total. Explain how an empty evaluation set should be handled rather than dividing by zero.

**Keep:** independent arithmetic, the constructed wrong result, full-vs-partitioned model evaluation and the empty-input policy.

## Task 5: reject malformed labels without hiding the axis bug

Give the classifier predictions of shape `(B,)` and labels of shape `(B,1)` for `B=3`. Predict the broadcast result if subtraction/product is allowed. Your objective should raise `ValueError` before loss arithmetic. Check the guard through both training and evaluation entry points.

For an intentionally column-shaped input format, validate that there is exactly one label column and convert explicitly with `labels[:,0]` at the caller boundary. Verify that the converted loss equals the original vector-label loss. Repeat with a singleton batch to ensure batch dimensions survive. Reject a multi-column target rather than squeezing it until a computation runs.

**Keep:** shapes, the expected error, caller conversion, equality check and multi-column/singleton checks.

## Task 6: defend raw-image and source-group contracts

Complete the image harness on its public fixture, then use training seed \(5\), a dataset of \(105\) generated images and a minibatch size of \(7\). Create held-out data with an independently declared seed and disjoint IDs/source groups before fitting.

Derive the CNN parameter count and check one convolution receptive field against a separate host calculation. Give the adapter a solid blue RGB image of size \(12\times16\); derive its normalized grayscale value under the declared RGB coefficients. Compare uint8 NHWC, transposed NCHW and explicitly declared unit-float paths. Reject malformed channel counts, an empty batch and out-of-range floating inputs.

Create an actual local PNG from a sample, load it through the external manifest API, and retain its file hash, source, permission/license note, label and group. Create a second manifest that leaks a source group across the split and prove it is rejected. Explain why merely changing IDs is verified separately from independence, and why near-duplicate or related-subject detection still requires data review.

Keep raw and preprocessed thumbnails with the axis, range, resize and EXIF contracts. This synthetic or self-generated PNG evidence verifies ingestion; it is verified separately from accuracy on real photographs.

## Task 7: resume the next image and augmentation, not just the weights

Using the changed dataset/batch configuration, interrupt at update \(17\), save a complete checkpoint, and continue \(9\) updates both uninterrupted and after restoration in a **fresh interpreter**. Record the actual next image IDs, augmented-input hashes and pre-update minibatch losses. Compare parameters, momentum, PRNG key, permutation, position, epoch and step at the same continuation boundary.

The sampler drops incomplete epoch tails under its declared policy. Explain where the interruption lies relative to the current permutation and why retaining only an epoch number would be insufficient. Show one rejected restore after changing a pixel and one after changing the optimizer configuration. Preserve the failure messages rather than weakening the compatibility checks.

Record source/runtime versions and the exact checkpoint data contract. Explain what this local atomic replacement and integrity check is verified separately from about hostile writers, remote storage or cross-device bitwise replay.

## Task 8: interpret error structure and precision tradeoffs

Freeze clean and declared-corruption evaluation sets before tuning. Evaluate in one batch and unequal batches; compare count-weighted metrics. Keep full confusion counts and error IDs. Draw aligned clean/corrupted thumbnails for at least one off-diagonal error and explain what information was removed or changed. Do not resample errors or labels until the accuracy looks satisfactory.

Calibrate integer scales on training/calibration images only, using predeclared percentiles \(95\) and \(30\). Preserve their data identity and compare float32, the source FP16/BF16 policies and the integer reference on the same held-out images. Record accuracy, cross-entropy, maximum logit difference and input/hidden clipping counts. Equal class accuracy does not imply equal logits or confidence.

Independently compute one integer convolution dot product. Show the maximum nine-product accumulation bound for signed symmetric int8 values, explain why int16 is unsafe, and state which operations in this reference remain float32. Compare parameter payload bytes separately from complete serialized-artifact bytes, and measure actual execution latency separately from storage size.

## Task 9: reload the artifact and measure the actual inference path

Export the trained image model and preprocessing/class contract. In another interpreter, load and infer batches of \(1\), \(2\), \(4\) and \(7\) raw images. Explain how the adapter decomposes unsupported graph batch shapes into supported calls. Compare against the source model with declared tolerances and against the integer reference for the integer path.

Reject a mismatched preprocessing version and corrupted graph bytes. Keep a last verified runtime while checking a candidate; prove failed load/canary validation leaves the active handle unchanged and rehearse rollback. Distinguish this local in-process selection from fleet rollout, concurrent deployment or rollback of external side effects.

Warm up, then time at least \(15\) completed inference repetitions for a declared batch and precision. Record the individual samples, actual device, units, percentile convention and throughput denominator. State that host preprocessing is included and file decoding is excluded, or explicitly add and separately measure decoding. The small sample set does not justify a strong production tail-latency claim.

Conclude with evidence boundaries: synthetic image fixture; actual local file ingestion; completed CPU graph and integer-reference execution; no independently validated edge/accelerator deployment. Name the target-specific parity, kernel-support, memory and end-to-end device measurements still needed before making that deployment claim.

## Evidence package and reviewer decision

Submit your model implementation, assessment harness, environment/commands and report. Distinguish observed results from predictions and copied reference output. Explain one actual failure and repair. If a test remains unresolved, keep its output and describe the next diagnostic step.

### Model transfer

- **Accept:** Width-five parameter count, explicit-parameter numerical forward pass and three batch-size contracts agree independently
- **Revise:** Only default-width output is shown; all variables are counted as parameters; singleton batch becomes a scalar

### Objective and derivatives

- **Accept:** Extreme logits remain finite with the stable objective; the analytic bias-shift result and two NNX seed checks agree within justified tolerances
- **Revise:** Probabilities/logs produce nonfinite loss without diagnosis; an oracle reuses the same autodiff; perturbed state is not restored

### Training and measurement

- **Accept:** Both configured seeds replay, fixed held-out sets stay out of updates, error IDs/independent metrics are recorded and evaluation leaves state unchanged
- **Revise:** Held-out data is repeatedly selected to meet a score; evaluation trains the model; only final loss or a single lucky seed is reported

### Aggregation

- **Accept:** Constructed count arithmetic and real 7/33 split agree with the global metric; empty totals have an explicit contract
- **Revise:** Batch means are averaged without count weights; an accidental equal accuracy is treated as a proof

### Shape diagnosis

- **Accept:** Malformed shapes are rejected at the loss boundary, deliberate conversion preserves vector labels and singleton/multi-column cases are checked
- **Revise:** Broadcasting is silently accepted; unqualified squeeze destroys batch meaning

### Connected image lifecycle

- **Accept:** Raw-image and group contracts, fresh-process full-state next-update recovery, fixed confusion/error evidence, calibrated arithmetic and actual exported/reloaded inference are all demonstrated under changed conditions
- **Revise:** Only weights resume; calibration uses final test examples; image ingestion is implied from tabular data; exported files are copied parameters without an executed runtime; failed candidate replaces the verified runtime

### Evidence

- **Accept:** Runnable learner-owned files, actual CPU output, quantitative reasoning and limitations are inspectable
- **Revise:** Public solution execution is presented as learner work; outputs are asserted without a reproducing command; TPU or job-readiness claims exceed evidence

Use **accept**, **revise**, or **not demonstrated** per task and name the missing evidence. An overall accepted local assessment requires every task and its evidence to be accepted. A reviewer can request a new width, fixed data seed or unequal-batch split to check transfer. This is an original synthesis assessment; no reviewer or credential is automatically supplied by the course.

Reviewer numerical references are separate. After your attempt, [download the reviewer notes](models-reviewer.md). These notes are public, not a secret examination key; reading them is separate from demonstrating the tasks.

## Training-method extensions: changed objectives and architecture parity

After the pretraining and post-training phases, use the training-methods project with an independently changed fixture. Compare two mask patterns with equal supervised counts; show why revealing clean hidden targets would invalidate the result. For contrastive learning, duplicate a source pair and explain the diagonal-positive penalty before choosing a multi-positive policy. Retain input identity, the independent objective calculation and training observations.

For adaptation, change adapter rank or scaling, retain the base weights exactly and verify merged predictions on new inputs. Compare original-task and adapted-task measurements separately. For text objectives, draw the shifted response mask and distinguish SFT, learned-reward PPO and DPO by their data, reference and rollout requirements. Synthetic preference success is not human alignment evidence.

Use the weight-conversion project to introduce one deliberate operation mismatch. Report the earliest differing intermediate tensor, the numerical error budget and the repaired result, then reload the converted artifact in a fresh process. Passing the provided MLP is verified separately from parity for a convolutional or attention architecture.
