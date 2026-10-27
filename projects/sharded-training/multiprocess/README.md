# L10B: restart a real two-controller CPU job

This optional continuation of the NumPy ownership example asks whether two separately launched JAX processes compute one weighted update and whether a whole job can recover after a worker dies. It is a local CPU protocol exercise. Two local processes are not two machines, and this result does not establish network, accelerator, speed, TPU, or production durability claims.

## Start here

Complete the two APIs in `starter/model.py`, then run the stages below. `local_numerator_and_gradient` returns the masked **sum** of squared-error loss and gradient plus the valid count. `global_transition` divides the gathered gradient sums by the one global valid count, then applies momentum and updates the weights. The reference checker computes expected math independently from the ordered sample IDs.

Run commands from the repository root. Python 3.14.3, JAX/JAXlib 0.9.2, and NumPy 2.4.4 are the versions recorded in the current receipt. Use an activated course environment's `python3`, or set `JAXPATHWAYS_PYTHON` to the exact Python executable that should launch worker processes. The report records the interpreter and library versions used. Source and runtime identities are frozen with an approved experiment; changing them under an old checkpoint approval must fail before JAX initialization or restore.

```sh
python3 projects/sharded-training/multiprocess/check.py \
  --stage math --candidate projects/sharded-training/multiprocess/solution/model.py

JAXPATHWAYS_PYTHON=python3 \
python3 projects/sharded-training/multiprocess/check.py \
  --stage all --candidate projects/sharded-training/multiprocess/solution/model.py
```

The checker writes a report and scratch work to an OS temporary directory when `--output` and `--work` are omitted. To keep evidence at a chosen location, pass both options, for example `--work "$PWD/.l10b-run" --output "$PWD/.l10b-run/report.json"`; use a new directory for each run.

`math` checks the two APIs and semantic controls without starting worker processes. `checkpoint` compares uninterrupted training with a fresh-process restore and checks the first post-restore IDs and complete optimizer/sampler state. `worker-loss` checks worker termination at a committed boundary and during an incomplete write, then restarts the whole cluster from `CURRENT`. `all` runs all three stages and the process-level negative controls. A failed stage means its named contract did not pass; a generic process error is not evidence that a particular negative control worked. If the host denies binding to `127.0.0.1`, a process stage is blocked and must not be reported as passing; rerun on a local terminal that permits loopback sockets. `math` can still run on a host that blocks loopback.

Individual process stages use separate scratch directories and reports:

```sh
python3 projects/sharded-training/multiprocess/check.py \
  --stage checkpoint --candidate projects/sharded-training/multiprocess/solution/model.py
python3 projects/sharded-training/multiprocess/check.py \
  --stage worker-loss --candidate projects/sharded-training/multiprocess/solution/model.py
```

## Worked update: uneven shards and an empty rank

The source-bound local CPU run in `/private/tmp/l10b-final-preflight-all-20261021/report.json` has SHA-256 `ba21af2c1e37d1862a53b119fc4ed56ac0a5f1172d6c13d78b73df16f30856d5`. Its step-1 transition records ordered IDs `[2, 3, 0]`: rank 0 owns `[2, 3]` and rank 1 owns `[0]`, so their valid counts are `[2, 1]`. The pre-update loss is about `1.0208334`. After the update, the local float32 fixture has `w ≈ [0.13686667, -0.0222]`, `v ≈ [-2.7373333, 0.444]`, key `17`, epoch `0`, cursor `3`, and step `1`.

At the epoch tail, IDs `[1, 4]` both belong to rank 0; counts are `[2, 0]`. Rank 1 contributes zero numerator and zero count. From the recorded step-1 state, the global gradient is approximately `[-0.8187333, 1.8652667]`. Divide the summed gradient numerator once by the global count `2`, then apply momentum:

```text
v2 = 0.8 * [-2.7373333, 0.444] + [-0.8187333, 1.8652667]
   ≈ [-3.0086, 2.2204667]
w2 = [0.13686667, -0.0222] - 0.05 * v2
   ≈ [0.28729668, -0.13322334]
```

The resulting state has key `17`, epoch `0`, cursor `5`, and step `2`. Do not average rank means: that would give the one-example shard the same weight as the two-example shard. Do not divide the empty shard by zero. These rounded values describe this local float32 fixture, not a general performance result.

The receipt binds the computation to these source hashes: `check.py` `f9ffbef4d9f38de45c0bff63f25119aec6e1cbcc5beaca635118f79f8c7e72ad`; `solution/model.py` `a7a7c1aca9009d284c41014ea1c6d1f5b7a9b4b1b60991491f5073f1a46ea8a5`; `starter/model.py` `166311a5dd9f47b77e0fdbed1cc4e116c4d0a96932fede3c7463f6af4770272d`; `supervisor.py` `30c0e9724565089e9796b8be39e6c763a3f874ad3fd8c9b8c8de0a877d1db330`; and `worker.py` `f9838429d447308e54588c82523de0370e0525d92c748367e2a6fdccb94bbb03`.

## What the recovery run observed

The figure below distinguishes two events from the same named run. After a complete generation-1 commit, rank 1 was killed with return code `-9`; rank 0 was observed alive and later returned `-6` during JAX shutdown. In a separate incomplete-write event, rank 0 had only `rank-0.json.tmp`, rank 1 had no final shard, and no generation-2 manifest existed when rank 1 was killed with `-9`. Rank 0 was alive when observed and later exited with return code `0` after the checker stopped it. `CURRENT` still named step 1. Both recovery checks started a new two-process attempt and restored the whole cluster; neither demonstrates survivor continuity.

![Two observed kill-and-restart sequences from the source-bound local CPU run. A committed step 1 is restored by a fresh two-controller attempt after rank 1 receives SIGKILL (-9); rank 0 was observed alive and later returned -6 during JAX shutdown. In a separate incomplete generation-2 write, only rank-0.json.tmp exists, rank 1 has no final shard, and no manifest exists; rank 1 receives SIGKILL (-9), rank 0 exits 0 after checker cleanup, CURRENT remains at step 1, and a fresh two-controller attempt replays IDs 1 and 4. Both cases restart the whole cluster.](figures/recovery-sequence.svg)

The standalone worker-loss report is `/private/tmp/l10b-final-preflight-worker-loss-20261021/report.json`, SHA-256 `fbf1a2838ac55497377c406aa64de443655d584c0569b7eba4b4ac855cccec0f`. It covers the same source hashes listed above and records the actual PIDs, per-rank return codes, checkpoint files, and restart comparisons. The recovery protocol synchronizes rank writes before publishing `CURRENT.json`; its local filesystem observation does not qualify remote storage or power-loss durability.

The checker also rejects changed source/configuration under an old approval, missing approval beside existing checkpoint artifacts, changed world or rank ownership, duplicate sample ownership, local-mean averaging, stale sampler state, early commit without every rank, stale checkpoint lineage, and a rechecksummed malformed state schema. A copy with only a trailing comment still passes the candidate math check; formatting is not used as a behavioral proxy.

The single-controller sharded project and the physical multi-host target retain their separate scopes and evidence. This extension establishes only the bounded two-controller CPU exercise described here.
