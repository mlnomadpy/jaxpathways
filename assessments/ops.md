# Operations synthesis: rehearse an incident and defend the recovery

**Scope:** a real local CPU JAX subprocess, explicit lifecycle and measurement boundaries, complete-state recovery, content-addressed artifact selection, actual held-out release validation and transparent capacity arithmetic. **TODO (Cloud & Accelerator Extension):** Run the same supervisor and rollback drills against a multi-host Cloud TPU VM slice.

[Open the project](../project.html?id=workload-operations). Complete `operations-01` through `operations-05`, then keep your implementation and an independent `assessment.py` harness. Run every drill only in a caller-owned temporary directory, with bounded child-process deadlines. Do not terminate unrelated processes or allocate external infrastructure.

## Establish the environment

From the course or extracted project workspace:

```sh
# Run run command in terminal using the course Python environment
python3 projects/workload-operations/tests/check.py --implementation projects/workload-operations/my_model.py --stage 5
```

Record source hashes, Python/JAX/NumPy versions, actual backend/device count, dtype and storage location. Keep public test execution separate from the changed-condition evidence below.

### Worked verification scaffold

Run the baseline verification suite with `python3 assessments/check_assessments.py`, and use the starter scaffold below to verify your numerical contracts:

```python
# Worked starter scaffold: harmonic vs arithmetic throughput & active-time capacity plan
import jax.numpy as jnp

# Two equal batches of 100 examples taking 2s and 6s: true rate is 200 / 8 = 25 ex/s.
batch_examples = jnp.array([100.0, 100.0])
batch_seconds = jnp.array([2.0, 6.0])
true_throughput = jnp.sum(batch_examples) / jnp.sum(batch_seconds)
naive_mean_rate = jnp.mean(batch_examples / batch_seconds)
print("True throughput:", float(true_throughput), "Naive average of rates:", float(naive_mean_rate))
```

## Task 1: classify real process outcomes

Run three jobs: a normal job with \(7\) updates and batch size \(5\); a job requiring more devices than are actually available; and a job that stalls after completing \(2\) updates. Choose and record a bounded deadline that allows this environment to initialize before the injected stall.

Retain the process exit, stdout/stderr, parsed events and supervisor duration. For each result, distinguish the requested runtime from observed discovery and identify the last completed lifecycle boundary. Prove the timed-out process has been reaped before attempting another job. Correlate the supervisor timeout with the injected event and trace to pinpoint the exact stalled stage.

State the extra cancellation contract that would be required if the worker launched descendants. Do not implement process-tree termination against unrelated processes.

## Task 2: recompute operational signals

For the normal job, independently calculate examples processed, total synchronized update seconds, update throughput, complete-job throughput and update duty fraction from raw events. Include units and timing boundaries.

Construct a diagnostic example with two equal-size batches processed in \(2\) and \(6\) seconds. Show why averaging their rates gives a different result from total examples divided by total elapsed time. Explain why neither update duty fraction nor a low loss proves accelerator hardware utilization or checkpoint freshness.

For the stalled run, report the latest observed progress and the latest committed checkpoint. Separate completed-but-uncheckpointed updates from an update that may never have completed.

## Task 3: recover changed conditions, including the next update

Use seed \(17\), total budget \(11\) updates, checkpoint cadence \(3\), and injected failure after update \(7\). Run a matching uninterrupted reference in a different directory. Before executing, predict the committed step and first update after restoration.

Follow an explicit runbook: confirm failed process exit; retain incident logs; inspect checkpoint integrity and provenance; select a single writer; restore; verify the first next update; complete; compare final state. Compare complete state hashes and losses for every resumed update, not only a final metric.

Then perform two separate negative drills: alter one checkpoint momentum value without its checksum, and change the training learning rate before restore. Require failures that name the violated contract. Do not make a failing run pass by deleting fields or recalculating a corrupted artifact’s checksum.

Explain why restarting from the original random seed or restoring parameters without momentum changes the experiment. Distinguish file-level atomic selection from power-loss durability and arbitrary remote storage guarantees.

## Task 4: gate an actual changed model and rehearse rollback

Publish a validated artifact from an actual completed job, retaining source/config/data identities. Record the active content ID. Publish a candidate with deliberately degraded parameters and a claimed passing validation flag. Independently calculate its mean squared prediction error on the project’s fixed held-out inputs using ordinary host arithmetic.

Prove activation rejects that model and leaves the active pointer byte-for-byte unchanged. This exercise must fail on actual prediction error, not merely a hardcoded false status.

Next publish a compatible, passing artifact with an explicit metadata change, activate it, and roll back. Verify the exact selected ID and recomputed payload digest. Corrupt a stored candidate without changing its filename and confirm it cannot be activated. Explain why a content digest establishes integrity but does not authenticate a publisher, and why selecting old weights cannot undo external side effects.

The local acceptance threshold is an illustrative fixture criterion. Explain what new evidence and authorization would be required for a real deployment gate, including concurrent writers and trusted validators.

## Task 5: justify capacity under declared assumptions

Measure at least \(3\) repetitions of two local job lengths, keeping individual complete-job and synchronized-update durations. State whether child runtime/compiler caches persist and what the process boundary includes. A few samples do not establish a high percentile; report them without relabeling their maximum as a reliable tail estimate.

Use a representative measured duration in an explicitly hypothetical plan: \(900\) arrivals per hour, a reserve of \(30\)% and a planning rate of \(2\) currency units per worker-hour. Calculate the minimum integer worker count, nominal load, hourly reserved budget and active-time cost per job. Independently verify the rounding and seconds-to-hours conversion.

Add a scenario where \(120\) completed jobs require \(150\) equal-duration attempts. Report cost per completed job under the same active-time model. Explain how bursts, idle reservation, retries, storage and actual target hardware could change the plan. Do not present the assumed price as a live cloud quotation.

## Evidence and review

Use **accept**, **revise**, or **not demonstrated** for each capability:

- Actual lifecycle outcomes and reaped children, with useful failure traces.
- Correct work counts and timing boundaries recomputed independently.
- Full-state next-update recovery under changed seed/cadence, plus incompatible/corrupt restore rejection.
- A measured prediction-quality gate, pointer preservation, exact content rollback and integrity checks.
- Measured local duration separated from demand/price assumptions, with checked dimensions and honest uncertainty.
- A runnable incident runbook naming what was observed, what action followed and what confirmed the repair.

Acceptance requires inspectable evidence for every task. A reviewer may request another failure point, changed cadence or candidate artifact. Public [reviewer notes](ops-reviewer.md) support review after the attempt; no external expert review or credential is supplied automatically.

## Engineering extension: tracked release and application regression

After `recovery-06`, `deployment-07` and `operations-06` through `operations-08`, complete the [engineering release project](../project.html?id=engineering-release). Keep this extension separate from the local workload evidence above.

Track two comparable training runs with real MLflow metadata and artifacts. Retrieve the chosen run through the API, verify its data/code identity and reload the immutable registered model version. Change one validation row or metric denominator and explain why the old comparison is no longer sufficient.

Construct a slice regression where the aggregate still passes. Retain the independently calculated counts, means and rejection decision. For the text application, add a changed expected answer, an unsupported case and an invented citation; record prompt/retrieval/evaluator versions and actual traces. Clearly identify replayed fixtures versus responses produced by a live model.

Bind approval to model, evaluation, target and image identity. Demonstrate that changed evaluation evidence, expiration and target mismatch cannot change the active release. Revalidate the previous bundle before rollback. Identify which component authenticates the reviewer in a deployed system; a reviewer string in this fixture does not.

When Docker is available, retain the actual image identity, model digest, non-root user, internal readiness/prediction results and bad-digest rejection. If it is unavailable, mark that part unexecuted rather than substituting a process check. Write a short ownership/retirement plan and identify which monitoring signals lead to investigation, rollback or retraining.
