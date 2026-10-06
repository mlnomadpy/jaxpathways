# Integration review: modality lifecycle and target evidence

Reviewed 2026-10-05. This is an internal agent review, not independent human approval. Canonical sources and project implementations were read-only during the running course smoke. Scope: cross-modal harness implementation, public checker, README, figure generator and recorded outputs; distributed-03 and distributed-04; welcome-03 TPU bridge; current pathway and career descriptions. Image already had its separate self-audit; text and sharded-training were still under active integration by their owners.

## Confirmed findings at review time

### P1 — A valid blank image poisons the trained image weights

`projects/cross-modal-harness/solution/model.py:84–85` divides by `maximum(norm(x), 1e-6)`. Forward values at zero are finite, but the norm's derivative at zero is undefined before the outer maximum. A completely black float32 image satisfies the published input contract. With even one such image in a four-class batch, the loss remains finite while the image gradient, momentum and updated image weights become NaN.

Read-only CPU reproduction with the installed JAX 0.9.2:

```python
data = m.fixture(11, per_class=1)
data['images'][0] = 0
state = m.initial_state(data, batch_size=4)
updated, event = m.transition(state, data)
print(event['loss'])  # 2.205064058303833, finite
print(np.isfinite(updated['params']['image']).all())  # False
print(np.isfinite(updated['momentum']['image']).all())  # False
```

The stage-two checks use nonzero generated bars and miss this accepted boundary. Fix the denominator by flooring the squared norm before taking its square root, for example `sqrt(maximum(sum(x*x, axis=-1, keepdims=True), 1e-12))`. Add a finite-gradient test at zero embeddings and a full transition with one valid blank image. Keep the independent objective oracle and regenerate project export/figure evidence after the change. Forward finiteness alone is insufficient for a training primitive.

### P2 — Checkpoint acceptance does not bind the training configuration

`projects/cross-modal-harness/solution/model.py:149–164` saves preprocessing `CONTRACT` and dynamic state, but not the optimizer/objective configuration or implementation identity. Momentum `0.85`, learning rate `0.015`, and temperature `0.2` live only in code at lines 92 and 102–103. Changing one while reusing an existing checkpoint is accepted as the same run. A read-only in-memory replacement of `_update` confirmed that `load_checkpoint` still succeeds without any mismatch. Existing tests check changed data, batch size and damaged bytes, not changed optimizer/objective behavior.

The supplied fresh-process equality check correctly establishes replay under the same source and environment. It does not establish configuration-safe resume after a learner changes the training code. Save an explicit training contract including these constants and compare the requested contract at load; retain implementation/runtime identity or state its required compatibility policy. Add a changed-learning-rate or temperature rejection case and explain the difference between resume and an intentional new experiment initialized from saved weights.

### P2 — The distributed restart instruction has no corresponding resume mode

`phases/09-distributed/04-resilient-distributed-training/lesson.json:33` instructs restarting the worker group and restoring the accepted checkpoint. The supplied executable code at line 48, repeated in build steps at line 148, always initializes fresh state, takes one update and raises `FileExistsError` if `step-one` already exists. It cannot be rerun on the accepted shared directory to perform the described restore. The existing-directory protection itself is reasonable; the mismatch is the promise attached to the command.

Either add explicit fresh/resume modes and test a second CPU process against an existing checkpoint, or describe this command precisely as a fresh coordinated save/load drill and point actual worker restart to the operations/sharded project. Retain the accurate caveat that no live multi-host failure has been exercised. This issue does not invalidate the existing in-process Orbax state and next-update comparisons.

## Figure and mathematical checks

- Cross-modal multi-positive row/column objective agrees with its stated definition; same-class duplicate positives and the duplication-invariance argument are appropriate for this particular objective.
- The recorded retrieval plot was visually inspected. It uses readable cell text and distinct units for cosine similarity and confusion counts. Its loss, diagonal similarities, confusion row totals and 20/20 versus 5/20 statements match `outputs/evidence.json`.
- Clean and shifted recorded fixtures change both noise seed and shift. A read-only paired control confirmed 20/20 clean and 5/20 shifted for **each** of seeds 7 and 9, supporting the stated translation failure. Prefer recording one of these paired controls to make the causal comparison explicit in the artifact itself.
- W8A8 really casts quantized operands to INT32 before the dot product and rescales afterward; the independent INT64 oracle and overflow bound match that arithmetic. Documentation correctly makes no native INT8 speed claim. Artifact-size and local-request timing scope are stated honestly.
- Distributed-03's global-gradient formula, local element counts and idealized ring byte arithmetic are consistent. Every-local-replica value checking occurs in the exercise solution, while the initial demonstration checks global values and local shapes. There is no false claim that modeled payload bytes are observed network traffic.
- Distributed-04's epoch crossing and plotted pre-update losses are described consistently. CPU replay evidence is explicitly separated from multi-host failure evidence.
- The TPU bridge's first prediction, constant increment, changed-bias answer and reversal answer are correct. It checks an explicitly requested backend and completed output placement. Its CPU receipt is not presented as a TPU execution result.

## Route and career integration observations

At review time the models route already points to the authored image harness. Probability/science correctly label NumPyro/Diffrax as optional extensions. Shipping and operations distinguish local CPU evidence, simulated arrival/capacity assumptions, and unmeasured target devices. Career descriptions are aspirational responsibilities and include explicit learning-route scope notes; no concrete unsupported hardware qualification claim was found in the reviewed descriptions.

Pending integration reminders, rather than findings against the unfinished work: `learning-paths/tpu.json:22,40` still calls the capstone planned; `curriculum/modality-tracks.json:168` still calls the text harness planned. Their owners are actively authoring/integrating these. Cross-modal prerequisites at line 428 say dedicated alignment/pairing/retrieval lessons remain planned even though the connected project now teaches those mechanisms; distinguish a future dedicated lesson phase from available project instruction so learners do not mistake it for an unavailable harness.

## Review limits

No browser access, real accelerator run, external dataset download or complete second corpus run was performed. Small CPU probes used temporary directories or in-memory changes only. The parent process owns final integration and the full corpus checks. Source locations above refer to the reviewed version and may move as the owner applies fixes.

## Resolution verification, later on 2026-10-05

The author applied all three fixes above; this reviewer independently rechecked them without editing their sources.

- **Blank-image gradient: resolved.** Normalization now floors the squared norm before square root. The original mixed four-class batch with one completely blank image still has finite loss `2.205064058303833`, and both parameter matrices and momentum matrices now remain finite after the actual compiled update. Added public tests also cover zero inputs and zero weights.
- **Training configuration: resolved within the stated same-source contract.** Checkpoint metadata now includes `TRAINING` and an implementation source hash. Saving under the normal learning rate and changing the requested rate to `0.3` before loading now raises `ValueError`. The source includes explicit checks for optimizer/objective configuration and implementation identity, while unchanged fresh-process replay remains covered by the author's full five-stage run.
- **Executable restart: resolved on CPU.** Independently extracted the current distributed-04 assembled code into a temporary file, ran it with a declared `COURSE_CHECKPOINT_DIR`, then launched a second Python process with the same directory and `COURSE_RESUME=1`. Both processes succeeded and produced identical stdout. SHA-256 hashes of all 12 files in the accepted checkpoint were unchanged. This confirms the documented fresh-process resume path on logical CPU devices, without asserting multi-host qualification.

## Foundations toolkit review

Reviewed the new `projects/foundation-toolkit/solution/toolkit.py` and `tests/check.py`; all four public stages passed independently on CPU. Masked statistics, per-example regularized gradients, explicit random-state transitions and the fresh-process trajectory check are coherent within their declared teaching scope.

### P2 — Validate saved array dtypes before JAX conversion

At review time, `projects/foundation-toolkit/solution/toolkit.py:81–84` loaded arrays with `jnp.asarray` before calling `validate_state`. Under the default JAX configuration, this silently narrowed unsupported 64-bit arrays before the validator could see their stored dtype. A checkpoint whose `step` was an INT64 scalar `2**32` was accepted as the INT32 value zero. This bypassed the explicit INT32 step contract and could silently reset reported progress. Float64 positions and uint64 keys had the same boundary problem.

Reproduction: take the normal initial NumPy state, replace `step` with `np.array(2**32, dtype=np.int64)`, save with `np.savez` into a temporary file, and call `load_state`; the returned step is zero without an exception.

Fix by loading raw NumPy arrays, validating their exact stored shapes/dtypes/values first, and only then converting accepted arrays to JAX. Add rejection fixtures for wrong stored float/key/step dtypes, including this overflowing counter. No source edits were made by the reviewer.

**Resolved and independently verified:** the author now validates the raw NumPy archive before conversion. Fresh temporary archives with INT64 overflowing step, float64 position and uint64 key each raise `ValueError`. The author also added these malformed-archive cases to the public checker.
