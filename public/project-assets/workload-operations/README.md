# Operate, recover and roll back a local JAX workload

Build a supervisor and runbook around a real JAX subprocess. The supplied worker fits a small linear model with momentum and randomized minibatches. You implement the operational contracts: process lifetime, measured signals, complete-state recovery checks, artifact selection and capacity arithmetic.

This project executes locally on CPU inside temporary directories. It does not provision cloud resources or establish accelerator/fleet performance. It supplies a working operational foundation whose remaining deployment contracts are explicit.

## Prepare

Use the pinned `requirements-cpu.txt` in the course workspace or project ZIP. From its extracted top folder:

```sh
# Run run command in terminal using the course Python environment
python3 -m pip install -r requirements-cpu.txt
cp projects/workload-operations/starter/model.py projects/workload-operations/my_model.py
python3 projects/workload-operations/tests/check.py --stage 1 --implementation projects/workload-operations/my_model.py
```

PowerShell preparation:

```powershell
# Copy the starter template into your editable workspace file
Copy-Item projects/workload-operations/starter/model.py projects/workload-operations/my_model.py
```

The starter includes an inspectable worker string so the exercise can focus on operation of an actual numerical job. Read its state/update/checkpoint contract. Keep the instructor reference in `solution/model.py` separate from your implementation.

## Worker and configuration contract

Defaults: seed 3, learning rate 0.04, momentum 0.8, minibatch 8, total updates 8, checkpoint cadence 2, backend `cpu`, minimum devices 1, resume false, no failure or stall injection, injected-stall duration 10 seconds. The fixed 64-example dataset follows `y = 2*x + 1` on evenly spaced inputs from -1 to 1.

Positive integer sizes are required. Batch size cannot exceed 64 when sampling without replacement. Momentum is in `[0,1)` and the example learning rate in `(0,1)`. The teacher workload is deliberately small and bounded.

`WORKER` is written as `worker.py` inside a caller-owned run directory. It receives that directory as an argument, reads `config.json`, records observed backend/device count, compiles without consuming the true initial state and emits structured JSON lines. JAX completion is synchronized before an update duration is recorded. Checkpoints contain schema, parameters, momentum, random key, completed step, source/config/data hashes and an integrity checksum. The checkpoint is replaced only after its complete file is flushed.

The worker never launches descendants. The supervisor owns its direct child and must terminate/kill as necessary and wait after a timeout. Do not change external processes, inspect unrelated logs or write outside the selected temporary run/store directories.

## Stage 1: process lifecycle

Implement `default_config`, `digest`, `atomic_json` and `launch`. Use an argument list without a shell. Capture stdout/stderr and retain malformed output separately. Classify completion only after a successful exit plus a completed event. A timeout must stop and reap the actual child. Keep `run.json` with the same observed result returned by the function.

```sh
# Run run command in terminal using the course Python environment
python3 projects/workload-operations/tests/check.py --stage 1 --implementation projects/workload-operations/my_model.py
```

The checker executes a successful worker, an impossible runtime requirement and an injected stall terminated by a deadline. Keep real event sequences and the final process outcomes as evidence.

## Stage 2: observability with units

Implement `summarize`. Sum processed examples and synchronized update times. Report update-path throughput, complete-job throughput and measured update duty fraction separately. Use committed checkpoint events to report uncheckpointed completed updates. Do not label the duty fraction as hardware utilization.

```sh
# Run run command in terminal using the course Python environment
python3 projects/workload-operations/tests/check.py --stage 2 --implementation projects/workload-operations/my_model.py
```

Check the arithmetic independently from raw events, not by comparing one summary helper with another copy of itself. Preserve monotonic timestamps only as local duration evidence.

## Stage 3: a tested recovery runbook

Use your supervisor to launch an uninterrupted reference and another job that fails after update 3. Confirm the failed process has exited, preserve its logs, validate the committed checkpoint at step 2, then resume the compatible run. Compare the next update and every later complete-state hash with the uninterrupted reference.

```sh
# Run run command in terminal using the course Python environment
python3 projects/workload-operations/tests/check.py --stage 3 --implementation projects/workload-operations/my_model.py
```

The checker also tests an already-completed resume, incompatible configuration and corrupted checkpoint. Record the next-update comparison and a failure diagnosis. A plausible final loss is insufficient recovery evidence. Checksums detect accidental corruption; they do not authenticate an adversarial writer.

## Stage 4: provenance and reversible selection

Implement `publish`, `read_artifact`, `activate` and `rollback`. Publication writes `artifacts/<content-id>.json` without activating it. Activation independently checks model MSE on fixed held-out inputs `(-1.5,-0.37,0.22,1.5)` against the known relationship `y=2*x+1`, requiring MSE at most 1. This is an illustrative acceptance criterion, not a production recommendation. It also verifies content, declared validation and required provenance (`worker_hash`, `config_hash`, `data_hash`, `state`) before atomically replacing `active.json`. That pointer records current and previous IDs. Rollback uses the same validation path.

```sh
# Run run command in terminal using the course Python environment
python3 projects/workload-operations/tests/check.py --stage 4 --implementation projects/workload-operations/my_model.py
```

The checker independently calculates content identity, exercises both valid selections, rehearses rollback and proves a failed gate, actually degraded model or corrupt payload leaves the pointer unchanged. This is a trusted single-writer local store. A boolean validation record is not a signed attestation, and the pointer is not a distributed deployment system.

## Stage 5: capacity and cost assumptions

Implement `capacity`. Convert measured seconds per job and arrivals per hour into offered worker-hours per hour. Divide by worker count for nominal load, and use a ceiling for the minimum integer count with reserve. Retain the hourly reserved budget and active-time hypothetical cost per job as separate values.

```sh
# Run run command in terminal using the course Python environment
python3 projects/workload-operations/tests/check.py --stage 5 --implementation projects/workload-operations/my_model.py
```

Independent known cases check dimensions and rounding. Another case uses the actual local job duration with clearly hypothetical demand and prices. Reject invalid/nonfinite inputs. Do not turn local CPU time into a cloud price or accelerator throughput claim.

## Evidence bundle

Keep your implementation, environment, configuration, actual events, checkpoint hashes, full and resumed state comparisons, failed-gate trace, rollback IDs and capacity assumptions. Explain one changed checkpoint cadence and one changed workload duration. The generated `validation.json` records the instructor reference checks, source hashes and measured local results; it is not learner evidence.

The synthesis in `assessments/ops.md` requires changed conditions and a written runbook. Reviewer notes are public. Passing the reference fixture alone is verified separately from job readiness or an independently reviewed qualification.

## Optional transfer to a prepared accelerator host

After installing and validating the appropriate JAX runtime on a host you control, the same supervisor can request an actual target:

```python
# Reference snippet
# Run `launch` to compute `result`.
result = launch(run_directory, default_config(backend="gpu", min_devices=1))
```

Use `backend="tpu"` for an appropriately configured TPU environment. The worker records discovery and rejects a mismatch. These paths are not part of the CPU receipt and were not executed here. They do not allocate machines, initialize a multi-host mesh or submit scheduler jobs. Add actual target installation, distributed coordination, storage guarantees, cancellation signals and scheduler integration before making claims about that deployment.

## Limitations

The example captures bounded logs in memory, has one writer, supervises one child, uses one local filesystem and saves small JSON states. Local atomic replacement plus file fsync does not by itself prove power-loss durability or remote-store semantics. Real systems also need bounded streaming logs, trusted release gates, process-tree cancellation, concurrency control and target-specific performance measurements. All of these are explicit transfer requirements, not silently simulated capabilities.
