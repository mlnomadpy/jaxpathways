# Capstone: train and release a 300M image understanding model

**Status: planned specification.** Inherits [common lifecycle gates](../../docs/model-lifecycle-capstones.md) and [tooling](../../docs/model-lifecycle-tooling.md). Build from the [image harness](../image-harness/README.md) and VISION/DATA/EVAL/NUM/BRIDGE/SERVE preparation. The companion [image-generation project](../image-generation-model/README.md) has different objectives and outputs.

Outcome: a natural-image model qualified for a chosen classification or retrieval task, with an optional separately assessed segmentation/detection branch. A candidate ViT uses 24 blocks, width 1,024, 16 heads and MLP width 4,096, near the 300M backbone scale before task-specific counts. Choose patch size/resolution, positions, normalization and heads in the pilot; count backbone, training-only decoder and deployed head independently.

## Data and representation recipe

Select a documented corpus relevant to the task, retaining source/entity IDs, duplicates, labels/caption provenance, resolution/aspect ratios and rights/attribution. Split groups before augmentation; isolate final benchmark identities. A pilot can compare 70% broad-domain and 30% target-domain images, measured by sampled images with pixel/token exposure also reported. The fractions are hypotheses, and actual datasets/revisions remain an implementation decision.

Freeze raw-file decoding, EXIF orientation, RGB/channel/range rules, resize/interpolation/crop and class mapping. Patches are the initial input representation; a text tokenizer is unnecessary for the unimodal classifier. Audit crops/color changes against the label semantics. Keep a small CNN and a supervised ViT baseline to determine whether expensive self-supervision helps.

## Model-development phases

| Phase | Work and objective | Evidence |
| --- | --- | --- |
| 0: baseline and correctness | Train the transparent supervised baseline with cross-entropy; check class counts, patch layout and model gradients | Baseline checkpoint, error gallery, hand-checkable receptive field/patch round trip |
| 1: representation pretraining | Choose a masked patch-reconstruction experiment **or** contrastive views **or** teacher–student self-distillation, against the supervised baseline | Objective-specific checkpoint; masks/pairs/teacher-update checks and a fixed probe panel |
| 2: representation/domain continuation | Test a new domain mixture or objective transition only if pilot evidence supports it | Continued checkpoint, resource-matched ablation and unchanged-task retention |
| 3: downstream adaptation | Compare frozen probe, partial/LoRA tuning and full fine-tuning for classification or image retrieval | Independent downstream splits, trained-leaf inventory and class/instance relevance contract |
| 4: optional dense prediction | Add a real segmentation/detection head with spatial labels and matched augmentation | Task-head checkpoint, ignored/empty-target and coordinate checks; classification success does not qualify this branch |
| 5: release | Convert/quantize/export the selected model and processor, qualify service/device and publish | Cold-load raw-image canaries, task/precision/runtime report and rollback |

For classification, independently derive the target log probability on a tiny batch. For reconstruction, score only the chosen hidden patch population; for contrastive training, inspect positive definitions and false negatives; for self-distillation, preserve the target network/centering/temperature/update rules. These methods are alternatives or tested transitions, not synonyms for MLM. Record optimizer/schedule changes and full RNG/augmentation/data state at each handoff.

## Evaluation

Predeclare a task-specific primary benchmark and domain holdout. Classification uses top-1/top-k where appropriate, per-class recall/confusion, calibration and imbalance-aware summaries. Representation quality adds frozen linear/k-NN probes. Image retrieval uses declared relevant identities, candidate pools and Recall@K/mAP. Segmentation uses a pinned IoU protocol; detection uses the chosen AP protocol with actual boxes and matching rules. Use suitable licensed/access-controlled datasets and pin their splits rather than promising a benchmark without its data.

Compare random-init/supervised/pretrained variants with the same probe/training budget. Audit near duplicates and source overlap. Add domain, corruption, resolution and rare-class slices with counts. The [DINOv2 reference](https://github.com/facebookresearch/dinov2) is useful for representation/probe workflows; our architecture, data and results require their own evidence.

## Monitoring and explained figures

Log images and patch tokens consumed, source/class proportions, failed decoding, crop/augmentation statistics, objective components, gradient/update norms, precision failures, memory, step times and fixed validation metrics. For a teacher–student method track teacher/student identities and updates; for contrastive training track pair counts and representation collapse diagnostics.

Use W&B image tables or equivalent MLflow artifacts for a fixed, permitted sample. Explain raw versus augmented images, masked reconstructions, learning/probe curves, confusion matrices, nearest-neighbor failures and per-class changes after adaptation. A visually plausible reconstruction alone does not demonstrate classification or retrieval quality.

## Release and acceptance

Package the actual processor, backbone/head weights, labels or retrieval contract, positions/resolution policy and loader. Compare preprocessing and intermediate tensors across Keras/TensorFlow/PyTorch/JAX only for the declared tested path. Qualify calibrated precision on raw images, batch-one and variable-batch serving, malformed input and a real target if edge performance is advertised.

Apply the [Hub release protocol](../embedding-model-300m/RELEASE.md) with image-specific canaries. A clean download must reproduce predictions, not merely load the file. Retain backbone/head/processor/container identities and a compatible previous release for rollback. The final learner dossier includes recipe choice, objective ablation, exact recovery, task failures, measured tradeoffs and release evidence.
