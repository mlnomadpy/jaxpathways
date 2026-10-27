# Distributed content repair notes

Scope: P2#82 distributed half, for the 2026-10-27 placement and memory audit. Only the two distributed lesson JSON files listed below were edited. TPU memory-ledger lessons remain assigned to the TPU agent. No course-wide generators, rendered outputs, central release metadata, or commits were changed.

## Implemented

- `distributed-01` now explains the opening `P_d`/`P_m` notation and directs readers to the parameter-sharded 2D task in `distributed-02`.
- `distributed-02` retains its existing 1D data-parallel, weighted-gradient exercise and adds a complete 2x2 logical-CPU mesh forward/backward/momentum-update task. The batch, target, parameter, momentum, mask, loss and gradient placements are explicit. It checks the loss, full gradient, updated parameter matrix and updated momentum against an independent NumPy oracle.
- The task prints actual local shapes and float32 bytes for each addressable device shard, including parameter, momentum and gradient ownership. It exercises valid-count pairs 4/1 and 3/0. The 4/1 case also computes and rejects the incorrect equal average of local means; that negative control is 0.068350 versus the correct 0.048550 global loss.
- Lesson text explains the scalar loss's reductions over data contributions and model output columns, and why a gradient keeps model-sharded parameter columns. The diagram values use the exercise's array shapes and byte counts. This is logical ownership evidence only, not a device capacity or communication-performance claim.

## Execution receipt

Ran both the full `distributed-02` `content.code` and a generated whole-lesson script from `scripts/lesson-artifacts.mjs` in a clean process using `/private/tmp/jaxpathways-0170-clean-learner-env/bin/python` (JAX 0.9.2). The lesson's preamble configures four CPU devices before initializing the CPU backend. The generated script exercised build steps, main code, visual code, experiments, reference solutions, and practice in renderer order. All assertions passed. The representative 2D output from the full content-code run was:

```text
Initial loss: 4.875
Gradient: [-5.25 -0.75] updated weights: [0.52500004 0.075     ]
Updated loss: 2.6784372329711914
valid=5; naive equal mean of nonempty shard means=0.068350 (wrong); weighted global loss=0.048550
valid=5; device valid counts=[4, 4, 1, 1]; loss=0.048550
valid=3; empty data shard contributes zero sum and zero count
valid=3; device valid counts=[3, 3, 0, 0]; loss=0.037083
batch features per device (device id, local shape, bytes): [(0, (4, 4), 64), (1, (4, 4), 64), (2, (4, 4), 64), (3, (4, 4), 64)]
targets per device (device id, local shape, bytes): [(0, (4, 2), 32), (1, (4, 2), 32), (2, (4, 2), 32), (3, (4, 2), 32)]
parameters per device (device id, local shape, bytes): [(0, (4, 2), 32), (1, (4, 2), 32), (2, (4, 2), 32), (3, (4, 2), 32)]
momentum state per device (device id, local shape, bytes): [(0, (4, 2), 32), (1, (4, 2), 32), (2, (4, 2), 32), (3, (4, 2), 32)]
parameter gradient per device (device id, local shape, bytes): [(0, (4, 2), 32), (1, (4, 2), 32), (2, (4, 2), 32), (3, (4, 2), 32)]
updated parameters per device (device id, local shape, bytes): [(0, (4, 2), 32), (1, (4, 2), 32), (2, (4, 2), 32), (3, (4, 2), 32)]
updated momentum per device (device id, local shape, bytes): [(0, (4, 2), 32), (1, (4, 2), 32), (2, (4, 2), 32), (3, (4, 2), 32)]
2D parameter-sharded forward/backward/update matches the unsharded NumPy oracle
```

The generated whole-lesson script also completed the 12-row, seven-valid-example transfer with data-shard counts `[6, 1]` (duplicated over model coordinates), loss `0.032743`, and `(4,2)` parameter shards of 32 bytes per device. It ended with `PASS: distributed-02`.

## Central metadata follow-up for the parent

- Consider adding an explicit lesson link/prerequisite from `distributed-01` to the 2D parameter-sharding section in `distributed-02`, if the central curriculum metadata models intra-phase routes.
- The lesson keeps the established virtual CPU runtime contract at `deviceCount: 4`; the 2x2 mesh shape is stated and executed in the lesson code and diagram. Consider adding a centrally modeled mesh-shape field if the curriculum schema supports it.
- The other audit acceptance items are assigned to the TPU half: replace aggregate HBM conclusions with per-device ledgers, qualify process/device/chip terminology against API receipts, and separate training state from inference KV-cache sizing. No physical TPU claim or paid hardware run was added here.
