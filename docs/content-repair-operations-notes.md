# Operations content repair notes

Scope: P2 issue #87 (`ops-evaluation`) and the ModelOps retirement portion of #88 (`release-lifecycle`). Edited only operations 06, 07 and 08 lesson content. The demos are self-contained in each lesson so generated downloads do not need external helper files.

## Implemented

- **Operations 06, MLOps:** validates a group-safe manifest, trains a tiny linear model, serializes and reloads its local artifact, and computes per-example predictions and squared errors. Slice and overall MSE use observed predictions and targets. The baseline has overall MSE 0.409 and fails the high-slice gate; changing a held-out target while keeping the ten IDs and policy fixed produces overall MSE 0.01 and passes. An absent required slice fails.
- **Operations 07, LLMOps:** executes local lexical retrieval and a deterministic extractive adapter with explicit prompt/model boundaries. Five heterogeneous stable IDs cover supported and unsupported questions, a retrieval omission, a malformed generated response and a timeout. Immutable trace dictionaries retain the prompt, retrieved context, response, stage diagnosis and version IDs. Responses are paired by ID despite reversed arrival order; missing, duplicate and unknown IDs are rejected. Baseline is 1/5, candidate is 5/5 over the same five IDs. The demo is explicitly not a hosted LLM or model-quality benchmark.
- **Operations 08, ModelOps:** writes three distinct local artifacts, selects them through an active pointer and has a consumer load the selected artifact to predict. It observes A at 3.0 before B, B at 3.4 after activation, and A at 3.0 after rollback. Stale, expired, wrong-target, version-to-bundle mismatch and retired target cases fail without changing pointer or consumer output. The active selection stores bundle and artifact digests; the consumer checks both cached objects and reloaded disk bytes against the selected artifact digest before prediction. The checks reject a C selection with B's approval, a C artifact file replaced by slope 99, and a corrupted C cache entry without pointer/cache mutation. Active/fallback retirement is rejected; explicit fallback clearing enables tombstones. Cached and forced-reload reads of a stale pointer cannot serve a tombstoned artifact. Audit records bind the action to actor and exact bundle hash. Scope states local, single-writer and unauthenticated.

## Verification

Command run from the checkout:

```sh
/private/tmp/jaxpathways-0170-clean-learner-env/bin/python - <<'PY'
import json, subprocess, sys, tempfile
from pathlib import Path
for n in ('06-mlops-data-contracts-and-release-pipelines',
          '07-llmops-prompts-evaluation-and-traces',
          '08-modelops-ownership-approval-and-retirement'):
    p = Path('phases/16-operations') / n / 'lesson.json'
    content = json.loads(p.read_text())['content']
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / 'lesson.py'
        script.write_text(content['code'])
        run = subprocess.run([sys.executable, str(script)], text=True, capture_output=True)
        if run.returncode:
            raise SystemExit(run.stderr)
    for i, experiment in enumerate(content.get('experiments', [])):
        compile(experiment['code'], f'{n}:experiment-{i}', 'exec')
    for i, practice in enumerate(content.get('practice', [])):
        compile(practice['learnerStarter']['code'], f'{n}:practice-{i}', 'exec')
print('JSON parse, lesson execution, experiment compilation, and learner starter compilation passed.')
PY
```

Result: all three embedded lesson programs exited 0; experiment and learner starter code compiled. The MLOps output reported baseline pass `False`, changed held-out target pass `True`, missing-slice pass `False`. The LLMOps output reported baseline `1/5`, candidate `5/5`, with retrieval, generation and infrastructure diagnostics. The ModelOps output reported predictions `3.0`, `3.4`, `3.0`, rejected invalid transitions and confirmed tombstone/cache/reload checks.

## Central metadata needs

No generator registry, release maturity registry, curriculum source, global code helper or maturity change is needed. The lesson payload already carries the self-contained source and expected output. Keep exported lesson scripts synchronized with each lesson's `content.code`, experiment examples and practice starters. The local ModelOps reviewer labels are teaching identities, not authenticated reviewers. The LLMOps adapter traces establish only the executed local fixture behavior.

The normal artifact exporter was also exercised. `lessonScript` from `scripts/lesson-artifacts.mjs` generated the downloadable Python scripts in `/private/tmp`, and the course CPU interpreter ran each to `PASS: operations-06`, `PASS: operations-07` and `PASS: operations-08`. This caught stale build-step, plot and exercise references; those lesson payloads now use the standalone demo and synchronized visual/solution code.

`node scripts/validate-curriculum.mjs` was attempted but cannot run from this checkout before a curriculum build: it fails opening the absent `public/curriculum.json`. No generated public artifact was created as part of this repair.

## Independent review fixes and rerun

After the independent review, operations-06 now compares two distinct reloaded artifact hashes against the same fixed evaluation rows and gate. Candidate A is rejected at high MSE 4.0; candidate B changes one slice-offset parameter, gets high/overall MSE 0.01 and passes. The target-correction sensitivity case remains separate. Operations-08 now requires the requested activation version to match the approved bundle version, stores bundle/artifact hashes in the selection, validates disk bytes before activation/rollback, and validates cached and reloaded consumer artifacts before returning predictions. Both mismatched-version approval and changed-C-byte controls verify unchanged pointer and cache state.

Reran `lessonScript()` exports for operations-06, -07 and -08 and executed each generated script with `/private/tmp/jaxpathways-0170-clean-learner-env/bin/python`; all exited 0 and ended in `PASS: operations-06`, `PASS: operations-07`, and `PASS: operations-08`. The rerun also exercises the new MLOps candidate comparison and ModelOps version, disk-tamper and cache-tamper negatives.
