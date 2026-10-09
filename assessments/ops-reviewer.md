# Operations synthesis: reviewer notes

[Return to the assignment](ops.md). Inspect the actual process and artifact evidence; a table of hardcoded status names is not sufficient.

## Lifecycle and metrics

A successful changed job records progress steps \(1\) through \(7\), with \(5\) examples per step, for \(35\) processed examples. Its completion requires a successful exit and the completed event. The impossible runtime requirement should fail before any training progress. The injected stall after two completed updates must be located from its trace and the child must exit after supervisor cancellation.

For two batches of \(8\) examples taking \(2\) and \(6\) seconds, average rate is \((4+4/3)/2=8/3\) examples per second, while total throughput is \(16/8=2\) examples per second. Other declared equal batch sizes are acceptable when the arithmetic is consistent. Do not accept hardware-utilization language for a ratio of application timing boundaries.

A stalled in-flight update is not an additional completed update merely because it was requested. Check the last actual progress event and committed checkpoint event.

## Changed recovery drill

Failure after update \(7\) with cadence \(3\) leaves committed step \(6\). The first resumed update is \(7\); its state should match uninterrupted update \(7\), followed by matching steps through \(11\). The progress event at the original failed run’s step \(7\) does not advance the checkpoint boundary.

Accept full parameter, momentum, key, step and provenance equality in the pinned runtime, with wall time excluded from deterministic state. If the learner changes environment or numeric precision, require a reasoned reproducibility contract instead of silently weakening exact checks. Corrupt payload and incompatible training config must be rejected separately.

File fsync followed by local atomic replace is not a blanket claim of power-loss durability or remote-storage atomicity. One writer is part of the demonstrated contract. A changed worker that launches descendants needs process-group or scheduler-level cancellation beyond this direct-child example.

## Actual release validation

The release fixture evaluates \(x=(-1.5,-0.37,0.22,1.5)\) against \(y=2x+1\), accepting mean squared error at most \(1\). For deliberately degraded parameters \((100,-100)\), residual is \(98x-101\); its mean squared error is \(22192.7553\), exceeding the threshold regardless of a claimed passing flag. The learner must compute and report the actual value rather than toggle validation to false.

The baseline project’s eight-update model has parameters near \((1.11218,1.02528)\) in the recorded environment, with held-out MSE around \(0.92560\). This is a permissive teaching gate, not a high-quality deployed model or proof of generalization. If the learner uses a changed actual run, evaluate that candidate on its merits and preserve any rejection rather than raising the threshold after seeing it.

Content identity uses sorted compact JSON, with nonfinite values rejected. The checker should recompute the digest from payload bytes under the declared encoding. Failed gates and corrupt artifacts must leave the active pointer unchanged. Rollback verifies the previous artifact through the same checks; it cannot undo data migrations or prior external requests.

## Capacity arithmetic

For representative measured duration \(t\) seconds, arrival rate \(900\) jobs/hour implies offered demand \(900t/3600=t/4\) worker-hours per hour. With \(30\)% reserve, minimum count is the ceiling of \((t/4)/0.7\), bounded below by one under the project contract. With \(m\) workers, nominal load is \(t/(4m)\), hourly reserved budget is \(2m\), and active-time cost per job is \(2t/3600=t/1800\).

If \(150\) attempts produce \(120\) completions, active-time cost per completion is \(1.25\) times per-attempt cost. Reserved-capacity billing and idle time can give a different result. Require explicit units, measured samples, hypothetical prices and workload boundaries. Three samples do not justify a strong tail-latency claim.

Review the runbook as a sequence of evidence-driven actions. Require the observed symptom, diagnosis, actual repair and verification result, plus material limits. Passing a public reference test or copying the instructor result is verified separately from independent operational competence.

## Engineering extension review

Require real retrieved MLflow records and model reload, not a mock dashboard or a printed dictionary. Confirm comparable metric boundaries and frozen data identity. Recalculate slice metrics and inspect the case that an aggregate hides. Prompt replay verifies the evaluator, not live-model quality. Inspect the actual container receipt when available; check its source/image/model identity and do not infer cloud rollout or load capacity. Changed/expired approval must preserve the active pointer, and rollback must revalidate compatibility and approval. Ask how identity, access control, concurrent promotion and retention would be enforced outside the single-writer fixture.
