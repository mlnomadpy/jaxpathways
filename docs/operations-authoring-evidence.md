# Workload operations authoring evidence

## Implemented course slice

All five canonical operations lessons now contain executable CPU work, staged construction, explicit mathematical/measurement contracts, prediction experiments, original and transfer exercises, diagnosis, checkpoints and source-bound figures:

1. Real child-process lifecycle: successful execution, rejected runtime requirement, injected stall, bounded termination and reaping.
2. Actual structured progress/checkpoint events, synchronized update timings, work counts, separate update/job throughput and checkpoint lag.
3. Controlled failure after uncheckpointed work, complete-state recovery, next-update/full-continuation equality, incompatible/corrupt restore rejection and a concrete runbook.
4. Source/config/data identity, content-addressed publication, measured held-out model gate, atomic local selection, failed-candidate pointer preservation and tested rollback.
5. Measured local jobs of different lengths, explicit timing boundaries, capacity/reserve/cost arithmetic with labeled hypothetical demand and prices.

The practical artifact is `projects/workload-operations`: a supplied inspectable JAX worker, learner supervisor/storage/planning API, reference implementation, five cumulative stages, public independent checks, source-hashed validation receipt and a bounded operational runbook. The source is original teaching material, not imported cloud infrastructure code.

`assessments/ops.md` and `assessments/ops-reviewer.md` require changed seeds, cadence, failure point, workload length, actual model degradation and independently checked planning arithmetic. They remain public review drafts with no automatically supplied expert reviewer.

## Executed verification

Every lesson's complete example, experiments, original solution and distinct transfer solutions ran in a fresh CPU process. All passed. After adding the measured release gate, the entire changed operations-04 example/experiments/solutions were rerun. Figure workers ran the actual examples and figure code, emitting SVG, PNG and JSON provenance; stale hashes were regenerated after text/figure refinements.

Project reference command:

```sh
python3 projects/workload-operations/tests/check.py --implementation solution --stage 5 --report projects/workload-operations/validation.json
```

All five stages passed. The checker executes real child processes and does not manufacture event rows to claim process success:

- A successful JAX worker; a nonzero preflight runtime failure; a real sleeping child terminated and reaped after a bounded deadline.
- Raw-event counts and independent throughput arithmetic, monotonic ordering, finite observed loss and positive synchronized durations.
- An interrupted job with committed step 2, restoration of steps 3–8, every subsequent full-state hash and loss identical to uninterrupted execution, equal final checkpoints, and an idempotent already-completed restore.
- Configuration incompatibility and accidental checkpoint corruption rejected before continuation.
- Independent SHA-256 calculation, publish-without-activation, two checked selections, rollback, failed declared gate and stored-payload corruption leaving the selection unchanged.
- A candidate with actual degraded model parameters rejected by recomputed held-out prediction error even when its validation flag claims success.
- Known dimensional capacity cases, measured local service-time input, reserve rounding and invalid/nonfinite-input rejection.

`validation.json` records source/checker hashes, environment and actual process/metric/recovery/registry/planning results. Timing values vary by host and rerun and are never embedded in prose as stable performance promises.

## Numerical and visual evidence

The worker is a real jitted JAX momentum update with randomized minibatches of a fixed small regression fixture. Complete checkpoints include parameters, momentum, PRNG key, step, schema, source/config/data identities and a checksum. Warmup is synchronized without consuming the actual training state; observed update timers also synchronize completion.

The release gate independently evaluates parameters on host inputs (-1.5, -0.37, 0.22, 1.5), whose known targets follow y = 2x + 1. The recorded eight-update reference parameters are approximately (1.1121758, 1.0252824), with held-out MSE 0.9255983. The deliberately degraded parameters (100, -100) produce MSE 22192.7553, independently recomputed using ordinary Python arithmetic. The permissive MSE limit of 1 is an illustrative fixture gate, not a production quality standard.

The recovery figure zooms to resumed updates 3–8 and explains actual loss rebounds shared by both runs, including the jump near update 4. Visual inspection confirmed that the two continuation curves overlap as expected. The capacity figure was revised into separately scaled wall-time and update-time panels to avoid a legend covering measured labels and to make timing boundaries readable. The text explicitly tells learners not to compare bar heights between separately scaled panels.

The capacity hand check uses 90 seconds/job × 80 jobs/hour = 2 worker-hours/hour. Four workers imply load 0.5; a 25% reserve requires at least three workers. A hypothetical hourly rate of 2 gives active-time cost 0.05 per job and a reserved four-worker budget of 8 per hour. These are deliberately invented scenarios, not actual cloud prices.

## Integration metadata

The root integrator owns shared manifests, generator runs and course-wide checks:

- Mark existing `operations-01` through `operations-05` authored at their existing canonical paths. Derive their manifest exercise/evidence/check fields from the authored material.
- Phase `operations`: set `projectId` to `workload-operations`; describe the available local CPU job lifecycle, observability, recovery, provenance and capacity sequence. Hardware should say CPU for the executed labs; accelerator/cluster transfer requires separate target validation.
- Add `projects/workload-operations/project.json` to the project registry.
- Assessment registration: `{id: ops, projectId: workload-operations, title: Workload operations synthesis, status: review-draft, scope: project-synthesis, source: assessments/ops.md, url: assessments/ops.html}`.
- Point the ops pathway's authored capstone at `workload-operations` and the ops assessment. Title/outcome should describe the actual observable, restartable local workload rather than assert an accelerator deployment receipt.
- Replace operations planned-only route/career copy with the available bounded capabilities; preserve explicit target-validation limits.

No shared manifests, generators or frontend files were edited by this subtask. Full generated script/notebook smoke, export and site checks are still required after registration. Individual authoring receipts do not substitute for those integration gates.

## Deliberate limits

One direct child per supervised job; bounded logs captured in memory; one writer; small JSON checkpoints; local filesystem; CPU execution. Local replace/file-fsync does not establish remote-store atomicity or machine-power-loss durability. A checksum is not authenticated provenance. The validation flag is not a signed attestation; measured host prediction checks establish only the fixture's declared gate.

A prepared accelerator host can request `backend="gpu"` or `"tpu"` through the executable launch API; observed discovery is checked and a missing runtime fails. Those paths were not executed and do not allocate resources. Multi-host initialization, scheduler integration, process-tree cancellation, trusted publishers, concurrent deployment control, durable streaming telemetry and real target capacity/price measurements remain explicit deployment requirements. This slice does not label simulated fleet data as actual operations or claim an independently reviewed professional qualification.

Primary sources inspected: Python subprocess lifecycle/timeout documentation, Python atomic replace/fsync documentation and JAX asynchronous dispatch guidance. The APIs were exercised in the pinned local environment. Learner-facing examples stay within temporary directories and incur no external runtime spending.
