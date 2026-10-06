# Connected image harness implementation and evidence

## Delivered scope

`projects/image-harness` is a connected eight-stage CPU image system, not the existing tabular classifier relabeled as image work. It contains a real 81-parameter CNN, explicit raw-image preprocessing, synthetic-image training, complete-state recovery, frozen clean/corrupted evaluation, lower floating/integer precision, actual serialized JAX graphs, fresh-process inference and completed CPU measurements. Starter/reference implementations and cumulative independent checks are separate.

The project README teaches the mechanisms and interprets actual learning curves, confusion counts and aligned failure thumbnails. Two SVG/PNG figures are generated from the executed reference. `assessments/models.md` retains the prior five classifier fundamentals tasks and adds four changed-condition image lifecycle tasks; reviewer notes retain the original arithmetic and add image/recovery/precision/export guidance.

## Executed command

```sh
python3 projects/image-harness/tests/check.py --implementation solution --stage 8 --figures --report projects/image-harness/validation.json
```

All eight stages passed. The starter parses and fails at its first unimplemented learner operation, as intended. The validation report contains implementation/checker hashes, actual environment, complete metrics, learning-curve values, confusion/error IDs, calibration clipping, serialization manifest and individual timing samples. No network dataset or cloud runtime is required.

Executed environment: Python 3.14.3, JAX 0.9.2, NumPy 2.4.4, CPU `TFRT_CPU_0`; Pillow 12.2.0 for actual local PNG decoding. The root integrator should pin Pillow 12.2.0 in the shared CPU requirements if exact decoder reproducibility is part of distribution policy; this subtask did not edit that shared file.

## What the checks prove

1. **Pixels and provenance.** Independent uint8/range/layout/channel/resize calculations; a red RGB constant maps to -0.5748; real PNG save/read/manifest ingestion; corrupt-file rejection; source-group leakage and input-contract guards. The implementation also rejects exact image-byte duplication across splits. No external photograph accuracy is claimed.
2. **Actual CNN.** Valid 3×3 convolution, ReLU, spatial mean and dense head; 81 parameters. An independent host loop checks receptive fields and pooled logits for singleton/changed batches at two seeds. A host float64 finite difference checks a model derivative, and malformed label axes are rejected.
3. **Learning.** Two hundred genuine augmented minibatch updates, tiny-batch fit, full seeded replay and another training seed. Post-update train and frozen held-out loss curves use the same unaugmented loss definition. The sampler’s incomplete-tail-drop policy is explicit.
4. **Recovery.** A checkpoint at update 13 is restored in a new interpreter, which continues eleven updates. Next sample IDs, augmented-input hashes, losses and all parameter/momentum/key/permutation/position/epoch fields match the uninterrupted run. Changed data and optimizer settings reject recovery.
5. **Evaluation.** Independent logits, one full batch versus unequal batches, unchanged model state, full confusion counts and a fixed declared corruption set. Corruptions are applied using a separate random stream after clean generation so error thumbnails are genuinely paired.
6. **Precision.** Executed float16/bfloat16 input/weight representations with float32 accumulation; actual int8 matrix products with int32 accumulation; independent Python integer receptive-field sum; bounded accumulator analysis; zero-channel scales; training-only calibration identity; reported clipping under ordinary and aggressive percentiles; unsupported int16 accumulation rejected.
7. **Export and inference.** Actual JAX serialized computations for fixed batch signatures 1 and 4, deserialization in a fresh interpreter, source parity on raw batches 1/2/4/7, exact integer-artifact parity, preprocessing mismatch and corrupt-payload rejection. The serving adapter partitions requests into supported graph calls instead of pretending unsupported signatures work.
8. **Operation.** A candidate passes load and declared canary parity before replacing the caller’s active handle. The check rehearses verified replacement, rollback and corruption that leaves the prior handle unchanged. Timings warm up and complete twelve repetitions, including host preprocessing and synchronized inference, excluding file decoding.

## Representative observed results

The reference seed’s final unaugmented training loss is approximately 0.0031243. Both tested training seeds reached clean held-out accuracy 1.0 on the 60-image synthetic set. The integer reference also retained accuracy 1.0 while its maximum logit error was about 0.367922: unchanged class predictions do not mean identical numeric outputs.

Clean confusion counts are 20 on each diagonal. Under the declared center-column occlusion plus noise, the matrix is `[[6,14,0],[20,0,0],[20,0,0]]`, with rows as true classes and columns as predictions. Accuracy is 0.10. This intentionally destructive distribution shift demonstrates substantial failure despite excellent clean results; labels retain the original generating class, and some decisive information has been removed.

The ordinary training-only calibration clips 0 of 3840 input values and 3 of 360 pooled hidden values on clean held-out data. The predeclared 30th-percentile diagnostic clips 2674 input and 253 hidden values. These counts are kept as evidence rather than used to tune calibration on final test data.

Float32 parameter payload occupies 324 bytes. The reported integer payload, including numeric scales/biases/calibration metadata, is 160 bytes; a complete bundle includes serialized graphs, manifests and container overhead and is larger. No storage reduction is used as evidence of native int8 speed.

The CPU timing report retains all actual samples for float32 deserialized JAX inference and the NumPy integer reference. Host/JAX dispatch overhead matters for this tiny model. There is no fixed speed promise, no native mobile-int8 kernel claim and no claim that twelve samples establish production tail latency.

## Figure review

Both generated figures were opened and visually inspected. The lifecycle panel distinguishes post-update full-set learning curves, clean and corrupted confusion counts, and accuracy across two arithmetic policies. The failure panel uses aligned clean/corrupted images, fixed grayscale intensity limits and explicit true/predicted class labels. The README describes the actual off-diagonal counts and equal-accuracy/nonzero-logit-error observation.

## Integration mapping for the root

Shared registries/frontend and route manifests remain root-owned:

- Add `projects/image-harness/project.json` to `curriculum/projects.json`.
- Modality track `image`: set status according to the existing implemented-project convention; include `image-harness` in projectIds, retain `mlp-classifier` as preparation, and describe the eight executed lifecycle stages. Replace “connected harness still to be implemented” with available synthetic CPU harness plus actual local-image ingestion; keep real-photo generalization and target-device qualification as unvalidated extensions.
- Models pathway capstone can reference `image-harness`, status authored, assessment `models`, with a title/outcome describing a recoverable image classifier through verified exported CPU inference. The assessment has been expanded in place, so update the existing `models` assessment registry’s projectId to `image-harness` and title/scope accordingly. The classifier project may continue linking to that same assessment’s foundational tasks.
- Consider the pathway prerequisite closure: the project requires networks, recovery, deployment/precision and performance lessons. Existing models routes without deployment should either include those phase prerequisites or visibly link the explicit bridge rather than implying all capstone prerequisites already lie on that route. The manifest lists exact requiredLessonIds.
- Keep language-model-specific masking/decoding separate: the image capstone does not assess Transformer behavior merely because the models route also includes Transformer lessons. The text harness should provide that complementary evidence.
- Register the image project checker in shared project validation with cumulative stage 8. Starter/reference/README/outputs/validation travel in its project ZIP.
- Pin the tested Pillow decoder if the shared environment is intended to reproduce external-file ingestion exactly.

No shared manifest, frontend, generator or other project file was edited. Shared distribution generation, project-registry tests, site checks and assessment rendering remain integration gates.

## Explicit limits and transfer

The trained dataset is an original synthetic bar/cross fixture. The external manifest path ingests actual user-provided PNG/JPEG-compatible local files and records source/license/groups/file hashes, but it supplies no external collection and proves no real-photo generalization. Subject grouping, near-duplicate review, lawful data use and label quality remain responsibilities of a real-data study.

Quantized affine arithmetic is genuine int8/int32, while dequantization, biases, ReLU and pooling are float32. FP16/BF16 correctness is checked in the source model; the serving adapter explicitly supports the float32 serialized graph and the integer reference. Calibration and final evaluation remain separate.

The completed target is CPU JAX export/deserialization plus a CPU integer reference. The README connects to the existing deployment/optional LiteRT labs and states the additional device preprocessing parity, runtime/kernel support, memory and decode/transfer/model measurements required for an actual edge target. No GPU/TPU/browser/mobile/LiteRT device receipt, production serving system, independent expert review or professional credential is claimed.

Primary references inspected: current official JAX convolution accumulation/dimension contracts, JAX serialized export documentation and Pillow EXIF transposition documentation. Installed APIs were verified by the executed checks rather than inferred from examples alone.
