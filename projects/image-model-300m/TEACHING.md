# C03 flower-model teaching guide

> **Quality: FAIL.** The selected pilot has 0.3185224830 macro balanced accuracy on the 547-image held-out set, below the predeclared 0.40 floor and below the fixed color-centroid control at 0.3638358559. This is a faithful, short, one-seed CPU pilot—not an accepted classifier or a 300M model result.

This project is a small five-class exercise over daisy, dandelion, rose, sunflower, and tulip photos. Its local CPU candidate has 25,605 parameters. “300M” describes the larger planned capstone goal; the full-size model and all full-scale capstones remain **Planned**. There is no human acceptance or target-scale qualification, and no rights clearance for releasing weights or photos.

## How to read the experiment

The recipe separates data before training. `fit` is for supervised fitting and unsupervised pretraining. `recipe_dev` is used to compare frozen probes, select the deploy arm, and choose its temperature. The held-out `final` split is loaded only after the pretraining and deploy choices are frozen. It reports the result once; it is not a tuning set. Images from the same source component stay together across the split.

The supervised CNN and small ViT are controls. MAE pretraining learns to reconstruct selected masked normalized-pixel targets. SimCLR pretraining learns from two augmented views of each source image. Both pretraining stages see 128 source images in this pilot; MAE uses selected reconstruction targets, while SimCLR processes 256 views. Frozen probes test the learned representations with a small classifier. The experiment also compares partial and full adaptation, then transfers the chosen parent into the deployment evaluation. Every stage starts from its recipe-declared parent weights and resets the optimizer and sampler for that phase; it does not quietly continue an unrelated stage optimizer. Checkpoint metadata records the parameter, optimizer, sampler, trace, and source-ID ancestry needed for an exact same-stage recovery. The accepted v14 attempt completed without interruption or resume, so no recovery was exercised here.

Loss plots use a one-based update counter local to each phase. The axes show supervised cross-entropy in nats/example, masked reconstruction mean squared normalized-pixel target, and SimCLR NT-Xent in nats/anchor in separate figures. Their values have different meanings and units, so compare curves only within the same objective and phase.

Development results select `baseline-vit` under the frozen candidate order. Temperature is selected on development logits only: T=1.0201723142. The published probability values are raw, uncalibrated softmax probabilities at T=1. The reported calibrated NLL applies that development-frozen temperature to final logits.

The quality gates tell a mixed story. Final macro balanced accuracy is 0.3185224830 and fails the 0.40 floor; the fixed color baseline reaches 0.3638358559. Brightness +0.05 lowers balanced accuracy by 0.0249026268, within the 0.15 ceiling. Calibrated NLL is 1.5087931145 versus raw NLL 1.5089652910, within the 0.05 degradation ceiling. Selected retrieval mAP is 0.2346509574 over 547 queries. Passing robustness and calibration checks do not offset the failed classification gate.

For a local prediction, a private consumer package from a reader-owned run can be loaded offline with `LocalImageClassifier.from_pretrained(path, local_files_only=True)`, then passed raw JPEG/PNG bytes with `predict_bytes`. It returns five logits, their T=1 softmax probabilities, the argmax class index, and a preprocessing record. The class index is the highest-scoring class for that input; the probability is not a calibrated confidence because this runtime uses T=1. The public packet deliberately omits the fitted weights and processor, so it cannot itself make predictions.

The figures are deterministic SVGs and PNGs derived only from the public numeric arrays. Each loss figure has its own objective and y-axis units. The numeric packet publishes all 12 comparison arms' ordered development/final IDs, class labels, and raw five-class logits. Retrieval results use rank ordinals into the final ID/class/component table, so every query ranking and AP can be rebuilt without publishing image pixels or embeddings.

## Verify the public evidence

These commands need Python and NumPy only. They do not read the source archive, fetch data, or import JAX.

```sh
python3 -m pip install -r tools/requirements-numeric.txt
python3 tools/check_v14_packet.py
python3 tools/build_v14_figures.py
python3 tools/make_packet_manifest.py
python3 tools/check_v14_packet.py
python3 tools/test_v14_packet_controls.py
```

The negative-control command proves rejection of a changed source byte, changed public logit and label, changed approval byte, tampered manifest and figure input, and accidental photo or weight payloads. The checker compares the checked-in source files with the copied, exact v14 recipe and approval; checks the source inventory and split identities; and independently reconstructs classification metrics, the development temperature, brightness drop, calibration NLL, retrieval AP/mAP, and training exposure totals. It checks that the selected arm was frozen before final data load and that no image, feature-vector, checkpoint, or weight payload is present. The figure builder writes `reference/evidence/v14/figure-data.json` plus SVG and PNG plots from these public inputs.

## Optional local CPU reproduction

A reader may choose to run a new local reproduction. That command is separate from the completed Sol authorization and does not imply another optimization approval. It needs the pinned official archive and a prepared copy of the frozen v7 data outside this checkout. The full preparation can contain image pixels; do not add it to this project or public evidence. With the exact reviewed recipe and approval copied in this packet, the run command is:

```sh
JAX_PLATFORMS=cpu python3 run.py \
  --data /absolute/path/to/prepared-v7 \
  --archive /absolute/path/to/flower_photos-clean.tgz \
  --recipe reference/evidence/v14/approved-v14-recipe.json \
  --approval reference/evidence/v14/approved-v14-approval.json \
  --output /absolute/path/to/a-new-empty-output-directory \
  --implementation solution
```

Use a fresh external output directory. This reproduces the approved short CPU recipe only if the source/archive/data identities match; it is not a route to tune from the inspected final split. Do not publish the resulting checkpoints, fitted processor, consumer bundle, or source photos.
