# Cross-modal harness execution evidence

The five-stage project trains both image and text encoders on a controlled four-concept paired retrieval task. It includes scaffold, reference, independent checks, real exports, measured requests, explained plots and a local paired-file ingestion boundary. It uses a symmetric multi-positive contrastive objective so duplicate class descriptions are not falsely treated as negatives.

```sh
python3 projects/cross-modal-harness/tests/check.py --implementation solution --stage all
python3 projects/cross-modal-harness/examples/figures.py
```

All five stages pass on CPU. Checks cover provenance and group separation, independent NumPy objectives and finite differences, changed seeds, fixed held-out retrieval, full momentum/random/order/cursor recovery in another process, training-only calibration, an independent INT64 arithmetic oracle, twelve genuine serialized image/text endpoints, singleton parity and fresh-process reload. Actual request measurements, measured release rejection, rollback and corrupted-artifact rejection complete the lifecycle.

Review caught two boundary defects. Flooring the squared norm before the square root now keeps gradients finite for blank images and zero parameters, including an otherwise normal batch with one blank image. Checkpoints now bind the optimizer/objective configuration and implementation source hash; changing the learning rate causes rejection. Both fixes were independently checked, and the full five-stage checker was rerun after adding regression fixtures.

The final implementation and figure-runner hashes match outputs/evidence.json. Both seeds 7 and 9 achieve 20/20 on the clean fixture and only 5/20 under the declared image shift. The report retains these paired controls. The four-panel figure was visually inspected: observed training loss, an actual held-out image, clean class-average cosine similarities and a shifted confusion matrix. Its guide explains row/column meaning and the large off-diagonal errors, rather than describing only the successful run.

The supported policies are FP32, W8A32 and W8A8 with explicit INT32 accumulation. Exports and completed inference execute locally; they do not demonstrate native integer acceleration. Quantized export files can be larger for this tiny model because metadata dominates. Vocabulary is four literal concepts, so these results do not establish arbitrary caption understanding, real-world multimodal quality or device deployment performance.
