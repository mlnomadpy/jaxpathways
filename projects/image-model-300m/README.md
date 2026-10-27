# Capstone: train and release a 300M image understanding model

**Status: full 300M capstone planned; bounded natural-image CPU pilot in development.** The implementation below is a source-bound teaching pilot, not a completed or quality-qualified model release. It has not performed flower-image optimizer updates. Target-scale training, independent human acceptance, 300M qualification, serving, and public weight redistribution remain open. It inherits [common lifecycle gates](../../docs/model-lifecycle-capstones.md) and [tooling](../../docs/model-lifecycle-tooling.md). The companion [image-generation project](../image-generation-model/README.md) has different objectives and outputs.

## Bounded real-image pilot currently authored

The pilot uses the upstream TensorFlow `flower_photos` archive, not generated pixels. Its frozen inventory starts from 3,670 JPEG files. Nine images without image-level attribution are excluded. Two exact-content images with conflicting class labels and two redundant same-class duplicates are also excluded, leaving 3,657 images. Whole source components are assigned to fit (2,562 images, 794 components), recipe-development (548, 360), or final held-out (547, 374) splits. Components join exact creator attribution, raw-byte identity, and EXIF-transposed RGB-pixel identity. The final split remains source-group-disjoint. The upstream credit file is an attribution notice, not the complete CC BY 2.0 legal code; see [`data/LICENSE.txt`](data/LICENSE.txt) and the current source record in [`data/recipe-review-v10.json`](data/recipe-review-v10.json). No source photographs are copied into consumer packages.

The locked processor accepts one-frame JPEG or PNG, applies EXIF orientation, composites alpha on RGB gray, converts to RGB, contains the full image in a 64×64 Lanczos letterbox, then maps `uint8` to `[-1, 1]`. The CPU pilot model is a two-block, width-32 bidirectional ViT; a small two-convolution CNN is an architecture control. The precommitted stage graph compares supervised ViT/CNN baselines, visible-only masked patch reconstruction (MAE), multi-positive SimCLR, fresh frozen-feature linear probes, full fine-tuning, and a partial fine-tune of the dev-selected pretrained branch. Branch choices use recipe-development data only. Every stage has deterministic per-ID/view/dropout/mask randomness, no-drop epoch-tail carryover, AdamW state, and per-update checkpoint recovery. The recipe binds source files, the exact data rows/arrays, optimizer budgets, selection ties, evaluation conditions, and pilot quality floors.

The frozen split JSON stores image IDs, labels, class indices, and source identities, while perceptual hashes live only in the trusted `source-manifest.json` record inventory. Before any model initialization, the pipeline performs a unique exact-ID join and checks label, frozen class index, creator, raw-byte hash, decoded-pixel hash, source URL, and row-to-array label agreement. It rejects duplicate or missing IDs and hashes that are not 16 hexadecimal digits. The enriched rows feed both the batchwise SimCLR relation and the source-pair audit; the data-stage checker exercises this same loader against the actual frozen v7 arrays without an optimizer update.

The lifecycle code remains behind a pre-optimization review gate. The checks use real archive bytes only for independent lineage verification. All optimizer, interruption/resume, metric-reconstruction and consumer cold-load graph tests use checker-owned synthetic RGB data; they do not change flower model parameters or estimate flower quality. The full graph checks all 11 declared stages, candidate API binding, objective-specific exposure counters, development-only selection before final-split loading, retained classifier/retrieval artifacts and fresh-process checkpoint continuation. Run the learner stages after creating the isolated environment and materializing the pinned archive:

```sh
python3 -m venv /private/tmp/jaxpathways-image-env
/private/tmp/jaxpathways-image-env/bin/python -m pip install -r projects/image-model-300m/requirements-image-lock.txt
/private/tmp/jaxpathways-image-env/bin/python projects/image-model-300m/data/prepare.py \
  --download --cache /private/tmp/jaxpathways-flower-photos-clean.tgz \
  --output /private/tmp/jaxpathways-c03-data-frozen-split-20261021-v7
/private/tmp/jaxpathways-image-env/bin/python projects/image-model-300m/tests/check.py \
  --implementation solution --stage all \
  --data /private/tmp/jaxpathways-c03-data-frozen-split-20261021-v7 \
  --archive /private/tmp/jaxpathways-flower-photos-clean.tgz \
  --output /private/tmp/jaxpathways-c03-check
```

The first command installs project-local pins without changing the shared JAX environment. `--stage all` runs the model, objective, augmentation, source-data, resume, runner, view-order composition, full synthetic graph, evaluation, offline consumer, and dynamic model-card checks. A starter run is expected to fail at the first unimplemented TODO. The strict runner independently verifies the source-bound recipe and archive before loading the optimization pipeline. With the current recipe it exits as `blocked_preoptimization_review`, reports zero updates, and does not load arrays into the model/training pipeline. Do not edit the recipe gate locally to force a run. An approved later recipe revision must bind the exact source/data snapshot and its separate reviewer approval before flower-image training.

After such approval, the declared invocation is:

```sh
/private/tmp/jaxpathways-image-env/bin/python projects/image-model-300m/run.py \
  --recipe projects/image-model-300m/data/recipe-review-v10.json \
  --data /private/tmp/jaxpathways-c03-data-frozen-split-20261021-v7 \
  --archive /private/tmp/jaxpathways-flower-photos-clean.tgz \
  --approval /path/to/reviewer-approval.json \
  --output /private/tmp/jaxpathways-c03-pilot-run
```

The held-out inventory is a small five-class flower task, not a broad vision benchmark. A passing CPU check would establish neither target-scale feasibility nor general representation quality, and the current source has no observed flower training metrics to report. The later run must retain its actual exposure, per-class and corruption results, retrieval rankings, failed as well as passed precommitted floors, and exact source-bound checkpoint lineage. Model packaging is local/offline correctness evidence only; there is no standalone model-weight license claim or public release approval.

The approved runner exposes resumable execution through `--resume`. It still checks the active recipe, data lineage, approval receipt, and complete candidate API/dependency source inventory before continuing the existing journal:

```sh
/private/tmp/jaxpathways-image-env/bin/python projects/image-model-300m/run.py \
  --recipe projects/image-model-300m/data/recipe-review-v10.json \
  --data /private/tmp/jaxpathways-c03-data-frozen-split-20261021-v7 \
  --archive /private/tmp/jaxpathways-flower-photos-clean.tgz \
  --approval /path/to/reviewer-approval.json \
  --output /private/tmp/jaxpathways-c03-pilot-run \
  --resume
```

## Full capstone target, not yet achieved

The eventual outcome is a natural-image model qualified for a chosen classification or retrieval task, with an optional separately assessed segmentation/detection branch. A candidate ViT uses 24 blocks, width 1,024, 16 heads and MLP width 4,096, near the 300M backbone scale before task-specific counts. Choose patch size/resolution, positions, normalization and heads in the pilot; count backbone, training-only decoder and deployed head independently.

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

## Implementation gate and real-data lineage

Do not promote until source images, class/instance labels, rights, duplicate/entity groups, processor settings and split identities are frozen. Link representation and task-head checkpoints to objective, full state and source exposure; real held-out task metrics and per-class/domain counts are required, while synthetic pixels only validate preprocessing and architecture.

Before this plan can change from **planned** to **authored**, provide independently checkable dataset/recipe manifests, staged parent-to-child checkpoint records with full resume state, real held-out evaluation outputs, and runnable plan-specific starter/reference/checks. Keep small CPU fixtures labeled as correctness evidence; they do not establish target-scale training, quality or production readiness.

## Recipe decision rehearsal before target-scale training

A masked-image objective improves reconstruction but frozen-feature classification does not improve.

**Decide:** Reconstruction, representation quality and downstream adaptation are different objectives. Keep the evaluation processor fixed and compare an independently trained supervised baseline, frozen linear probe and declared fine-tuning budget. Inspect whether pixel-space error rewards texture while the downstream task needs semantic grouping.

**Review packet:** Masked/visible patches with reconstruction, per-class errors, probe/fine-tune tables, data-group overlap checks and parent/processor identities. Reject a general visual-quality claim supported only by reconstruction loss.

This is a design exercise for a planned implementation. Record a hypothesis, a controlled pilot, a stopping criterion and the evidence needed to reject the recipe before requesting target-scale compute. Do not mark the capstone implemented or publish a model card with invented metrics.
