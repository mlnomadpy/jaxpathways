# Independent review: operations and performance repairs

Review scope: `operations-01` through `operations-08`, `performance-03`, and their diagrams, against scaffold-load (#87), release-lifecycle (#88), operations-evaluation (#89), and profiler-teaching (#81). This was read-only for lesson/source files. Bounded CPU checks used the pinned local environment; no cloud or paid service was used.

## Findings needing correction

### 1. High: operations-08 can activate an unapproved artifact key and reload changed bytes

In `phases/16-operations/08-modelops-ownership-approval-and-retirement/lesson.json`, `content.code` defines `activate(store, version, bundle, ...)` separately from the approved `bundle['version']`. `verify()` checks the bundle hash, target, expiry and tombstone, but does not require `version == bundle['version']`. `activate()` writes the caller-supplied `version` to `active.json`. `consumer_predict()` then loads `artifacts/<selected-version>.json` without comparing its contents with the approved bundle's `artifact_hash`.

I executed the current lesson code and then used its own fixture values in a temporary directory:

```python
activate(root, 'C', bundles['B'], approvals['B'], 150, 'cpu-demo', 'reviewer-B', [])
consumer_predict(root, 1.0, {})
```

This was accepted with `current == 'C'`, although the approval was for bundle B; the consumer returned C's prediction `4.0`, while B's approved artifact predicts `3.4`. Replacing `artifacts/C.json` under that unchanged selection with slope 99, then calling `consumer_predict(..., force_reload=True)`, returned `99.0`. The original C bundle's stored artifact hash still matched its own bundled bytes. The cached path also trusts a version-keyed object, so it does not establish content identity either.

This breaks the exact-bundle approval and artifact identity claims. Validate the requested version against the bundle version, persist a bundle/content digest in the selection, and verify the loaded artifact against that digest on cache miss/reload. The existing stale/expired/wrong-target and tombstone tests pass, but do not exercise these two mismatches.

### 2. Medium: operations-06's changed-target comparison is not a changed candidate comparison

`phases/16-operations/06-mlops-data-contracts-and-release-pipelines/lesson.json`, `content.code`, writes the first model to `candidate.json` and reloads it for per-row predictions and count-weighted MSE. This repairs the original supplied-error-list problem. However, the changed comparison changes `changed[-1]['y']` from 19 to 17.1, then writes `train_linear(rows)` to `changed.json`, where `rows` is still the original dataset. Training uses only training rows, so the serialized model and its identity are unchanged. Evaluation then uses the changed test target.

The gate flips because the held-out label was changed: the original model's high-slice MSE changes from 4 to 0.01 and the run passes. This is a useful label-correction sensitivity check, and the lesson names it a target change. It does not show a changed model candidate against the same frozen labels, so it does not meet the candidate-comparison transfer criterion. Add a separate one-parameter candidate change while preserving the ten test IDs, targets, slice weights, threshold and artifact reload path; keep the current label-correction case clearly named as such.

### 3. High: checked-in learner/download artifacts still teach the earlier lessons

The canonical lesson JSON changes are not reflected in several checked-in `code/main.py`, `code/learner.py`, and `notebooks/exercise.ipynb` companions. The exact current `lessonScript()`/`learnerScript()` generator outputs differ from the checked-in files. Concrete examples:

- `phases/16-operations/07-llmops-prompts-evaluation-and-traces/lesson.json`, `content.code`, now runs a five-ID local retrieval plus deterministic adapter with retrieval, generation and infrastructure diagnostics. Its checked-in `code/main.py` still imports MLflow and evaluates nine repeated canned cases; it does not contain the current retrieval/model boundary implementation. Its checked-in `code/learner.py:3` asks for an invented-citation check, while the lesson exercise asks about `case-sensitive-refund-77` and changing a prompt policy across five IDs. The notebook likewise contains the old citation exercise.
- `phases/16-operations/06-mlops-data-contracts-and-release-pipelines/code/main.py:25` still defines `train_linear` only after the older schema and `gate_metrics(errors, slices, limit)` consumes supplied squared errors. The current lesson's `content.code` computes predictions from a serialized/reloaded linear artifact in `evaluate_reloaded`. Its checked-in learner file and exercise notebook still ask about averaging two supplied slice means, not changing the high-slice held-out target.
- `phases/16-operations/08-modelops-ownership-approval-and-retirement/code/main.py` still has the prior metadata-only `verify_release()` flow and no consumer prediction, cache, tombstone or retirement implementation. Its `code/learner.py:3` asks only for an evaluation-hash approval invalidation, while the current lesson exercise covers active/fallback retirement and tombstoning B.
- `phases/16-operations/01-accelerator-jobs-and-runtime-lifecycle/code/main.py:3-5` still starts with the repeated full “local artifact helpers” scaffold. The current lesson's first build step is the short timeout-classification function. Its learner script and notebook omit the current `min_devices=999` rejected-runtime task.
- The same source/companion mismatch appears across operations-02 through -05. For example operations-04's current lesson exercise records consumer predictions before activation, after activation, after rejection and after rollback; its checked-in learner script still asks only for release-note ID rollback, and its notebook contains that older task.

For parity checking, I generated the current reference scripts from `lessonScript()` and the learner scripts from `learnerScript()` in `scripts/lesson-artifacts.mjs`. All nine generated reference scripts (operations-01 through -08 and performance-03) passed in `/private/tmp/jaxpathways-0170-clean-learner-env/bin/python`. That establishes that canonical source examples run. It does not make the checked-in notebooks/downloads current. Regenerate and execute the learner, exercise, documentation and reference-script companions from the repaired JSON before considering #87/#89 staging/parity complete.

## Reviewed mechanisms that passed

- **Operations-04 release behavior:** two compatible artifacts have distinct local consumer predictions at x=1. Publication leaves A selected, activation of B changes the loaded prediction, rejected activation preserves pointer bytes and behavior, and rollback selects A. The implementation uses content-addressed artifacts and states its local single-writer/no-auth scope. `content.diagram` is readable and its branch labels match the pass/fail behavior.
- **Operations-07 case identity and diagnostics:** the local adapter records case, candidate, prompt, index and model IDs; retrieval and generation inputs/outputs are captured. Stable heterogeneous IDs include supported, unsupported, retrieval failure, generation failure and timeout. Responses are reversed then paired by ID, and missing, duplicate and unknown IDs are rejected. The baseline is 1/5 and the grounded candidate is 5/5 under the same five cases. This is accurately labeled deterministic local-adapter evidence, not hosted LLM or general quality evidence.
- **Operations-08 tested lifecycle paths:** normal before/after/rollback predictions are 3.0, 3.4 and 3.0. Stale, expired and wrong-target approvals leave selection and prediction unchanged; active/fallback retirement is rejected; fallback clearing permits tombstoning; tombstoned activation and rollback are rejected; cached and forced-reload reads reject a retired selection; audit entries include actor and bundle hash. The version-key/content-hash holes in Finding 1 are not covered by those checks.
- **Performance-03 retained trace:** `data/retained_cpu_timeline_p2_81.json`'s source SHA-256 matches `code/capture_retained_cpu_timeline_p2_81.py`; every retained `start + duration == end`. I reran the capture to a temporary output with the pinned CPU environment. It emitted the same six named CPU events and event nesting, with new timing values as expected from runtime variability. The stored 2 ms condition correctly gives a 2665 us parent containing 2520 us `input_wait` and 119 us `model_and_wait`, leaving 26 us; parent plus children is the wrong 5304 us sum. The 0.5 ms comparison changes the wait in the predicted direction. The lesson explicitly limits the claim to annotated CPU host spans and infers no accelerator kernels or collectives. Both `performance-03` and `operations-04` mechanism figures rendered successfully; their table/branch labels fit at full size.

## Commands and bounded checks

Generated scripts with `lessonScript()` for operations-01 through -08 and performance-03, then ran each under the pinned CPU interpreter with a 40-second per-script timeout. All ended with exit code 0 and `PASS`. Representative output included operations-01 observed `started`, `ready`, `stall_injected` before timeout and reaped the worker; operations-03 restored step 2 and matched every resumed state hash; operations-04 preserved rejection behavior and rolled back to A; operations-07 passed the five-case comparison; operations-08 completed its declared lifecycle checks; performance-03 recomputed `(outer, child union, remainder, wrong sum) = (2665, 2639, 26, 5304)` us.

The retained trace reproduction was loaded with `importlib`, its module output path was redirected to a temporary directory, and `main()` was run with `/private/tmp/jaxpathways-0170-clean-learner-env/bin/python`. This avoided changing the checked-in evidence file.

The two generated figure checks were:

```sh
python scripts/render-lesson-figures.py --lesson operations-04 --output-dir /tmp/ops-figure-checks
python scripts/render-lesson-figures.py --lesson performance-03 --output-dir /tmp/ops-figure-checks
```

Both returned `PASS`.

## Follow-up resolution check (2026-10-27)

I re-read the latest canonical lesson JSON, reran the generated references, and independently replayed the previously failing paths. The two source findings above are resolved in these exact lesson versions:

- `phases/16-operations/06-mlops-data-contracts-and-release-pipelines/lesson.json` SHA-256: `54a3d50adc46c883454a51e2536bab1e681393cbd3f7f3f4e3689b535f952bac`.
- `phases/16-operations/08-modelops-ownership-approval-and-retirement/lesson.json` SHA-256: `454c3a862c59e4f218cbf85790be7c948175d6361ac44ee99dc38d7c8f01bff4`.

### Operations-06 resolution

The lesson now serializes and reloads distinct artifact A and candidate B. Both are evaluated on the same ten test IDs, targets, splits, slice assignments/counts, and 0.5 MSE gate. Candidate A's high-slice MSE is 4 and fails. Candidate B changes a model prediction through `slice_offsets={'high': 1.9}`, has a different artifact fingerprint, and reduces high-slice MSE to 0.01 and overall MSE to 0.01, so it passes. A separate label-correction sensitivity check keeps artifact A fixed and is clearly distinguished from the model-candidate comparison. The exercise's B offset 0.5 counterexample correctly reports high-slice MSE 2.25 and a failing decision.

I regenerated and ran the current reference with:

```sh
node --input-type=module - <<'JS'
import fs from 'node:fs';import {lessonScript} from './scripts/lesson-artifacts.mjs';
const p='phases/16-operations/06-mlops-data-contracts-and-release-pipelines/lesson.json';
const d=JSON.parse(fs.readFileSync(p));
fs.writeFileSync('/tmp/operations-06-latest-review.py',lessonScript(d));
JS
/private/tmp/jaxpathways-0170-clean-learner-env/bin/python /tmp/operations-06-latest-review.py
```

It ended `PASS: operations-06`. Output verified A's high-slice MSE 4 and failed gate, B's 0.01 and passing gate, plus the separate changed-target case. The prior operations-06 finding is resolved.

### Operations-08 resolution

The current `verify()` takes the requested version and rejects a version/bundle mismatch. The active pointer records the version, bundle digest and artifact digest. Activation checks the stored bytes against the approved bundle before pointer mutation. The consumer uses all three identity values in its cache key and verifies both reloaded disk bytes and cache-hit objects against the selected artifact digest.

I regenerated and ran the current operations-08 reference. It ended `PASS: operations-08`. I then independently executed its `content.code` in a separate namespace and temporary registry, and replayed three negative controls:

1. Activate version C using B's valid bundle and approval. It raised `requested version does not match approved bundle`; pointer and cache bytes stayed unchanged.
2. Replace C's artifact file with slope 99 and force a reload. It raised `reloaded artifact bytes do not match approved bundle`; pointer and cache stayed unchanged.
3. Corrupt C's already cached object. It raised `cached artifact does not match approved bundle`; pointer and cache stayed unchanged.

The prior operations-08 finding is resolved for the declared local single-writer teaching fixture.

### Companion regeneration remains pending

This follow-up checked canonical lesson source only and generated temporary reference scripts for execution. Checked-in `code/main.py`, `code/learner.py`, documentation and notebooks were not refreshed or verified as a complete distribution. As requested, companion regeneration and the parent's full smoke/parity checks remain pending; the earlier companion-staleness finding is not yet resolved.
