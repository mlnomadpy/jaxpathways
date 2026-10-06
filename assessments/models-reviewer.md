# Models synthesis: reviewer notes

Use these after inspecting the learner’s attempt. Numerical answers alone are insufficient evidence. [Return to the assignment](models.md).

Width five has `2*5 + 5 + 5*1 + 1 = 21` parameters. The specified singleton input gives hidden pre-activations `[1,-1.5,-3.5,4,-2.5]`. The scalar logit is `tanh(1)-2*tanh(-1.5)+0.5*tanh(-3.5)+tanh(-2.5)-0.25`, approximately `0.8361874163` in an independent NumPy float64 calculation. Its five tanh activations are approximately `[0.761594156,-0.905148254,-0.998177898,0.999329300,-0.986614298]`. The output shape is `(1,)`, not scalar `()`.

The extreme-logit loss is approximately `log(2)/3 = 0.23104906`. The common-bias derivative is `mean(sigmoid(logits)-labels)`, approximately `-1/6`; saturated endpoints contribute negligibly. A probability-based expression can round to zero or one, encounter `log(0)` and create indeterminate zero-times-infinity terms. The stable logit expression avoids that representation failure. Exact float32 printing and finite-difference errors vary; inspect the learner’s arithmetic rather than demanding one bit pattern.

The constructed global accuracy is `(6+21)/(7+33)=27/40=0.675`; the unweighted mean is `(6/7+21/33)/2≈0.746753247`, overstating by about `0.071753247`. Global mean loss is `(7*0.2+33*1.1)/40=0.9425`, while the unweighted mean is `0.65`. Weighting follows example counts. If rows carry unequal sample weights, the aggregation denominator must follow that objective’s weight sum instead.

Training acceptance has no hardcoded threshold for the changed fixture. Seek a valid replay and faithful held-out measurement, then ask what observed error locations imply. Poor scores may arise from insufficient optimization, width, input distribution or a code error; the learner must distinguish these with evidence. A source checker passing at its original width/seed does not resolve this changed condition.

For malformed labels, logits `(3,)` combine with labels `(3,1)` into a `(3,3)` pairwise matrix. An explicit caller conversion is defensible after checking the single-label-column contract. Never accept arbitrary squeezing as an explanation.


## Connected image lifecycle reviewer guidance

The image model has \(3\cdot3\cdot1\cdot6+6+6\cdot3+3=81\) parameters. A solid blue RGB input under the declared channel conversion becomes grayscale \(0.0722\), then normalized value \(2(0.0722)-1=-0.8556\). The floor-index resize leaves a solid image unchanged. Other color-management conventions are not interchangeable without changing the artifact contract.

With \(105\) images and batch size \(7\), one complete permutation contains \(15\) batches. After update \(17\), the changed fixture has completed two batches of its next permutation, so position is \(14\) and zero-based epoch is \(1\), assuming the specified initial state and sampler policy. The first resumed update is \(18\), using the next seven indices from that saved permutation. Do not accept an epoch-only checkpoint or a newly generated random permutation as an exact continuation. The actual saved IDs, augmented-input hashes, losses and optimizer/key/cursor states must agree in the new process.

For symmetric int8 values clipped to \([-127,127]\), the nine-term convolution magnitude bound is \(9\cdot127^2=145161\); the six-term dense bound is \(96774\). Both exceed signed int16 capacity and fit int32. Bias addition, ReLU, pooling and dequantization remain float32 in this reference. The implementation must execute integer products and integer accumulation rather than rounding values but multiplying floats while claiming int8 arithmetic.

The reference public fixture produced a diagonal clean confusion matrix with \(20\) examples per class, but its fixed destructive occlusion/noise set produced counts \(((6,14,0),(20,0,0),(20,0,0))\), or \(6/60=0.10\) accuracy. These values explain the supplied figure, not a target to force in the assessment’s changed seed/data setting. A learner should retain mismatches, confirm the data-generation and pairing rules, and explain visible evidence rather than resampling to obtain these exact counts.

The public reference’s int8 path retained clean accuracy while its largest logit discrepancy was approximately \(0.368\); three of \(360\) held-out pooled activation values clipped under its training-only 99th-percentile calibration. The assessment changes percentiles to \(95\) and \(30\), so require actual changed-condition measurements and do not impose the reference’s quality/clipping counts. The aggressive policy should be interpreted from its measured clipping and error structure, not described as automatically better because of smaller ranges.

A seven-image request uses one batch-four graph and three batch-one calls. A two-image request uses two singleton calls. Acceptance requires actual deserialized computation in a fresh process, raw preprocessing parity and integrity/version rejection; a parameter NPZ alone is not the serialized JAX export being assessed.

The runtime canary in this project is an output-parity gate against declared expected logits, useful for conversion/preprocessing checks. It is not a general quality gate for intentionally changed weights and does not authenticate the publisher. A production replacement would need a separate quality/authorization policy. Require unchanged active state after a rejected candidate and a clear account of what rollback can and cannot undo.

Report complete local timing samples without promising a fixed latency across machines. Counts, fixed signatures, array dtypes and parity are correctness contracts; latency is measured evidence tied to a specific workload, process, device and version. No CPU result should be relabeled as a mobile or accelerator deployment receipt.

## Review the training-method extension

Require a changed example and an independent denominator or probability calculation. Check that hidden targets do not enter the encoder, pair identity survives augmentation, and response masks shift with targets. For LoRA, inspect unchanged base arrays and rank/scaling metadata rather than only a decreasing loss. Ask the learner to explain why the two-zero-factor initialization stalls.

For conversion, require the first mismatching layer and explicit near-zero handling. Reject claims based only on matching parameter shapes or final argmax. Confirm that the saved target artifact, rather than the live source object, supplies the final inference result. Preference claims must identify synthetic versus human label provenance and distinguish frozen reference from rollout behavior policy.
