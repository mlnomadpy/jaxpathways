# TPU content repair notes

P2 repairs for audit issues #81 (profiler teaching), #82 (placement-aware memory and topology), and #90 (TPU topology figures) are complete. The P1 TPU changes from gate 2d36e4b5 are preserved.

## Lesson changes

- phases/21-tpu-systems/02-bfloat16-int8-precision-and-xprof-profiling/lesson.json now profiles a fixed-input 128×128 BF16 dot, bounds calls to 1–16, checks warmup/profile/unprofiled outputs for finite values, and measures a separate unprofiled wall-time loop. The receipt reports JAX version and observed backend. The quiz no longer claims the trace measures only device work or collective communication; the shown function contains no collective. The exercise reports all three dtype/error rows and checks five steps as a changed input. Build-step code matches the main profiler implementation.
- phases/19-tpu/01-get-a-tpu-running-and-verify-the-runtime/lesson.json distinguishes JAX processes/devices from physical chips, qualifies mappings by generation, machine type and API, restores the direct/Flex-start provisioning and pinned environment commands, and keeps the target preflight separate from cleanup. The topology visual uses separate chip-count and HBM-GiB panels. Its text equivalent and graph identify host/process/device/chip ownership. The capacity exercise compares replicated and ideal-sharded per-device state and checks a 52 GiB changed case.
- phases/21-tpu-systems/01-tpu-generations-and-hbm-memory-budgeting/lesson.json adds a placement-explicit per-device ledger, activation allowance, and 48/52 GiB worked cases. The 7B sizing case compares ideal-sharded v6e-4/v6e-8 with 8 GiB/device activation allowance and reports the inference KV cache separately. The v5e minimum-chip estimate is labeled as an ideal-sharding arithmetic lower bound. The figure now includes a per-device placement panel and the diagram describes replication, sharding, activation headroom and separate inference memory.
- tpu-04, tpu-02 and tpu-05 retain their P1 repairs and were not changed for these P2 findings.

No release maturity or physical TPU qualification evidence was changed. No TPU, cloud, or paid workloads were run.

## Verification

All dynamic checks used /private/tmp/jaxpathways-0170-clean-learner-env/bin/python.

Exact commands run:

- /private/tmp/jaxpathways-0170-clean-learner-env/bin/python /tmp/verify_p2_tpu.py
- /private/tmp/jaxpathways-0170-clean-learner-env/bin/python /tmp/verify_tpu_buildsteps.py
- python -m json.tool phases/19-tpu/01-get-a-tpu-running-and-verify-the-runtime/lesson.json >/dev/null; repeated for tpu-04, tpu-02, tpu-05, tpu-03 and tpu-06.

Observed results:

- tpu-01: CPU sum-of-squares arithmetic passed; requested TPU/4 readiness remained false. The independent backend and insufficient-device checks passed. The two visual panels rendered from the supplied topology values.
- tpu-03: 48 GiB replicated state was 48 GiB/device and failed; ideal four-way sharding was 12 GiB/device but failed with a 1 GiB/device activation allowance; the changed 52 GiB ideal shard was 13 GiB/device and failed. The 7B ideal-shard case failed on four v6e chips and passed the stated arithmetic budget on eight; the separate 8192-token BF16 KV estimate was 1 GiB. The three-panel visual data checks passed.
- tpu-06: CPU backend was recorded as cpu; seed-42 returned dtypes were float32, bfloat16, float32 and all three mean errors were reported. Three- and five-step fixed-input traces were finite and emitted XPlane files; 17 steps were rejected. The tpu-06 build-step profiler implementation matched the main lesson implementation.
- tpu-02 resume/hash and tpu-05 CPU-route checks also passed when rerunning the six-lesson P1/P2 validation harness.
- All six assigned TPU lesson JSON files parse successfully.

The temporary verification harnesses are /tmp/verify_p2_tpu.py and /tmp/verify_tpu_buildsteps.py in this workspace session; they are not course artifacts.

## Parent sync

The lesson objective change to per-device replicated/sharded sizing and separate inference KV sizing is in tpu-03. The tpu-01 objective continues to require named route-owned cleanup. Regenerate the parent-owned lesson/book visual outputs for tpu-01 and tpu-03 from their updated visual data. Preserve content maturity and evidence status; these source-level CPU checks do not qualify physical TPU hardware.

## Independent review follow-up

The independent review gaps are repaired in the owned lesson JSON sources:

- tpu-01's primary exercise and solution now both calculate the 48 GiB replicated/ideal-sharded per-chip ledger, include the activation allowance, and report requested API/machine/chip/process values separately from observed process and JAX-device counts. The receipt records `jax.process_count()`, and says physical chip count comes from the named resource contract rather than device count.
- tpu-01 and tpu-04 keep CPU arithmetic as a rehearsal and assert that a requested TPU target is not ready on CPU. Their documented TPU preflight invocations set `TPU_REQUIRE_TARGET=1`; the scripts exit nonzero unless the backend is TPU and the global JAX device count exactly matches four. The tpu-04 command uses `set -euo pipefail` so `tee` cannot hide that failure.
- tpu-01 and tpu-04 create the remote lab directory over SSH before the first SCP. tpu-06 also ensures its trace script destination exists before SCP.
- tpu-04 practice and starter use `observed_backend`, `device_count`, and other keys actually emitted by its receipt; absent keys return false.
- tpu-03's memory section now displays the per-device HBM placement equation instead of an unrelated tokens/s equation.
- tpu-06 defaults and explicitly runs the main trace for five profiled steps, matching the command label. The full main code and build step import `tempfile`, so the retained-XPlane practice runs after the lesson script.

## Independent review verification

Using `/private/tmp/jaxpathways-0170-clean-learner-env/bin/python` with `JAX_PLATFORMS=cpu`:

- `/private/tmp/jaxpathways-0170-clean-learner-env/bin/python /tmp/verify_p2_tpu.py` passed all six CPU lesson checks and TPU-06 function/build-step synchronization.
- `/private/tmp/jaxpathways-0170-clean-learner-env/bin/python /tmp/verify_tpu_buildsteps.py` passed the tpu-01, tpu-03 and tpu-06 complete build-step sequences.
- `JAX_PLATFORMS=cpu /private/tmp/jaxpathways-0170-clean-learner-env/bin/python /tmp/verify_full_tpu_lessons.py` ran all six complete lesson scripts, all six solutions and all 12 practice solutions, including the tpu-04 receipt predicate and tpu-06 XPlane file-size check.
- A separate CPU subprocess check ran tpu-01 and tpu-04 with `TPU_REQUIRE_TARGET=1`; both exited status 1 with `Requested TPU target is not ready`.
- A direct CPU build-step loop executed all 19 build steps across the six lessons, and `python -m json.tool` parsed all six JSON files.

No TPU/cloud resources or paid workloads were used. These checks establish CPU behavior only. The retained XProf timeline artifact for issue #81 is owned by the parent performance-03 repair; the tpu-06 lesson still needs to be connected to that retained example during integration. Parent should regenerate lesson-derived files and visuals after integrating source JSON changes. No central metadata changes are required from this subtask; preserve existing maturity and evidence qualification.

## TPU integration checker follow-up

Updated only `scripts/check-tpu-lessons.mjs` for the current lesson contracts. The Phase 19 check now runs the actual CPU receipts, confirms arithmetic passes while TPU readiness is false, checks process/device/chip/API fields separately, and tests backend and device-count mismatches. A new negative-control test verifies that `TPU_REQUIRE_TARGET=1` rejects inline execution without source identity and rejects a saved verifier run on CPU. The gate remains strict. The tpu-06 trace check now expects the five-step CPU trace output and checks the no-collective statement from source; the precision caption assertion matches the current lesson text. Existing checkpoint tamper, precision raw-array tamper, memory ledger, trace-file and analytic provenance checks remain.

Exact command and result:

- `PATH=/private/tmp/jaxpathways-0170-clean-learner-env/bin:$PATH node scripts/check-tpu-lessons.mjs` — 10 tests passed, 0 failed. The provided PATH ensures its Python subprocesses use the reviewed CPU JAX environment.

## Central metadata contracts to sync

I did not edit `curriculum/course.json` or lesson files in this integration-check task. The current registry still has stale contracts for parent integration:

- tpu-01's registry objective and Phase 19 study-guide milestone still center topology filtering / aggregate HBM, not the lesson's replicated-versus-ideal-sharded 48 GiB per-chip exercise and observed-process/device versus requested-chip receipt comparison. Its learner starter also retains the old `min_devices` and aggregate HBM filter contract.
- tpu-01 registry evidence only names the HBM table, Pod classification and provisioning command log; it does not yet list the target readiness receipt/source identity or separate observed process/device and requested physical-chip fields.
- tpu-04 registry evidence still requires `verified`; the current receipt emits `requested_target_ready`, `expected_backend_ok` and `expected_topology_ok`. Its guide describes the synchronized receipt and calculated 4-/8-device values but could name the exact current readiness fields.
- tpu-03's study-guide milestone still asks for a minimum chip count from an 80% aggregate budget without the explicit placement assumption and separate training/inference decision now used in the lesson.

No cloud/TPU calls were made. The checker modifies no lesson, curriculum, generated asset or evidence source.
