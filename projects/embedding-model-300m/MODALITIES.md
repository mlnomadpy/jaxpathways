# One lifecycle, modality-specific learning

Status: planned branches of the [flagship capstone](README.md). Each reuses the data/state/evaluation/release contracts and the corresponding existing harness. The approximately 300M text model is the first concrete target. For other branches, declare **total, per-encoder, frozen and trainable parameter counts** before implementation. A dual encoder with two 300M towers is not a 300M system.

The shared pattern is masked/self-supervised representation learning → broad contrastive learning → task-pair specialization → evaluation → qualified release. The masking unit, valid positive, negative semantics and benchmark protocol differ. No single objective is required to outperform all alternatives; include a controlled baseline and a phase ablation.

For complete non-embedding projects, use the [lifecycle catalog](../../docs/model-lifecycle-capstones.md): image classification/dense prediction, speech/event recognition, image/audio generation and grounded multimodal generation have separate specifications and acceptance criteria. This guide retains the representation/retrieval branches.

## Image embeddings

Start with the image harness, VISION and EVAL-4. Curate natural images with source/entity/near-duplicate groups, original resolution/aspect-ratio metadata and documented transformations. Split before augmentation. A different crop may cease to depict the relevant object; audit augmentations against the task.

1. **Masked pretraining:** patch-based encoder/decoder with hidden patches excluded from encoder input and loss restricted to declared targets. Compare mask ratio/patch size on a pilot. The reconstruction decoder is training-only unless explicitly retained. A [MAE-style approach](https://arxiv.org/abs/2111.06377) is a reference, not a guarantee of retrieval quality.
2. **Broad representation learning:** contrastive views with task-valid crops and group-aware positives. Compare a frozen/self-distillation baseline if appropriate; do not call self-distillation a contrastive loss or silently introduce an untracked teacher.
3. **Pair fine-tuning:** instance/product/semantic retrieval pairs with audited hard negatives and identity-disjoint splits. Classification-label equality and instance equality are different definitions of a positive.
4. **Evaluation:** frozen linear/k-NN probes on a declared image classification task; held-out image retrieval with Recall@K/mAP; cross-domain/corruption/resolution slices; downstream segmentation only if that capability is claimed. [DINOv2's model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md) illustrates distinct image-representation uses and evaluation choices.
5. **Release:** encoder weights, image processor (EXIF/color/resize/crop/interpolation/range), pooling, embedding dimension/normalization, raw-image canaries and tested framework/device contracts.

Monitor per-source/augmentation exposure, reconstruction on hidden regions, view similarity and false positives, representation health, probe/retrieval quality, preprocessing time and memory. Required figures show actual crops/masks/reconstructions, nearest-neighbor successes/failures and stage comparisons.

## Audio embeddings

Start with the audio harness and AUDIO. Choose the **speech** or **general sound** task before mixing data. Retain sample rate, channel policy, recording/speaker/source identity, duration, transcript/caption provenance and permissions. Split by recording/speaker/entity as required; overlapping windows from the same recording cannot straddle a purported independent split.

1. **Masked pretraining:** choose spectrogram reconstruction or masked latent prediction. Define time/frequency masks, padding and normalization. A latent-target/teacher method needs a separately specified target generator and stop-gradient/update rules; it is not automatically text MLM on audio tokens.
2. **Broad contrastive learning:** views of the same recording, semantically related recordings or aligned audio/text. Label which notion of similarity is intended. Pitch/noise/time-stretch changes can destroy speaker, event or linguistic labels.
3. **Pair fine-tuning:** task-specific positive/negative pairs for audio retrieval or speaker identity, with group separation and hard-negative inspection. Speaker identity and semantic similarity can demand different invariances; publish separate variants if necessary.
4. **Evaluation:** use appropriate [SUPERB](https://arxiv.org/abs/2105.01051) tasks/probes for speech; speaker verification may require EER under its prescribed protocol, classification uses task metrics, and ASR uses WER only with an actual recognition decoder. For sound embeddings, use suitable event classification/retrieval with task-defined mAP/Recall@K and duration/noise slices. Do not imply a pooled embedding alone performs transcription.
5. **Release:** waveform decoder/processor, resampling/window/mel conventions, valid-length pooling, input duration limits, golden audio examples and supported export operators.

Monitor valid audio seconds, silence/clipping, padding, sample-rate mismatches, speaker/source balance, masked objective and embedding health, plus quality by duration/noise. Explain waveform, spectrogram and retrieved-audio examples alongside metrics.

## Image–text and audio–text embeddings

Start with independently qualified encoders and the cross-modal harness. Curate pairs with provenance, caption/transcript quality, source/entity groups, multi-positive relationships and wrong-pair audits. Frozen encoders versus jointly trained towers is an explicit experiment, as is each tower's learning rate and parameter budget.

1. **Unimodal initialization:** reuse the selected text/image/audio pretraining checkpoints. Record whether their data overlap, and preserve each processor. Training all towers from scratch is a separately budgeted experiment.
2. **Cross-modal contrastive pretraining:** project representations into a declared shared space; train both retrieval directions where meaningful, with masked known positives/duplicates and stable temperature. Missing/corrupted modalities need a policy. Use [CLIP](https://arxiv.org/abs/2103.00020) and [CLAP](https://github.com/LAION-AI/CLAP) as mechanism references.
3. **Pair fine-tuning:** specialize with curated image-caption or audio-caption pairs and challenging semantic negatives. Test swapped attributes, negation, counting and near-duplicate entities rather than relying only on broad category differences.
4. **Evaluation:** image–text retrieval in both directions on a pinned suitable image-caption suite, zero-shot classification with fixed prompts if claimed, and cross-domain/compositional cases. Audio–text retrieval can use prescribed AudioCaps/Clotho protocols when compatible and accessible. Record candidate pools and multiple captions; training on a benchmark split changes the interpretation of its score. Current MTEB/MIEB suites can supplement these tests when their adapters match the model.
5. **Release:** both encoders, projection heads, text/image/audio processors, similarity/normalization/temperature conventions, parameter inventory and paired canaries. Test each tower independently and the complete pair score after cold download.

Monitor both directional losses, valid-pair counts, candidate pools, tower gradient/update norms, projection norms, temperature and retrieval by direction/domain. A strong tower can conceal a failing partner in a combined loss. Show similarity matrices, ranked cross-modal examples and deliberate wrong-pair failures.

## Shared promotion rule

Every branch must pass the same stage-artifact, recovery, numerical parity, precision-quality, service and publication gates, with its own task metrics and baselines. Reuse MLflow schema and release tooling; retain modality-specific data and preprocessing code. A text benchmark cannot certify an image/audio encoder, and a generic “multimodal average” complements per-direction/per-domain evidence.

Implement text end to end first, then image and audio independently, then cross-modal alignment using their saved encoders. This order makes the final multimodal project visibly combine earlier learner work rather than importing unexplained pretrained components.
