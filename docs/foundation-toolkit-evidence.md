# Foundation toolkit execution evidence

The four-stage project connects environment inspection, masked array statistics, transformed derivatives and explicit random state. It supplies a learner scaffold, reference, cumulative checker, runner and two-panel figure. Welcome, arrays, transforms and state each point to the corresponding stage; learners do not need to finish all four before completing the first phase.

Executed on the installed CPU environment (Python 3.14.3, JAX 0.9.2, NumPy 2.4.4):

```sh
python3 projects/foundation-toolkit/tests/check.py --implementation solution --stage all
python3 projects/foundation-toolkit/run.py
```

All four stages pass. Independent checks include valid-row mean/scale calculations with a constant column and singleton, analytic per-example derivatives at changed parameters, deterministic scan transitions and fresh-process replay of seven updates after restoring five completed steps. Changed random keys produce different trajectories. The environment report observes actual completed row sums 3 and 12.

Review found that converting checkpoint arrays to JAX before validating their stored dtypes could silently narrow an overflowing 64-bit step. The loader now validates raw NumPy arrays first. Wrong float, key and counter dtypes, including an INT64 step of 2**32, are rejected. A separate reviewer reran all stages and independently reproduced these rejections; see integration-review.md.

The final source and runner hashes match outputs/report.json. The actual PNG was inspected: three seeded trajectories use arbitrary coordinate units, with crosses marking final positions; paired gradient bars agree with the independent residual formula. The prose explains the axes, matching bars and what neither panel establishes. This is a synthetic state/array contract exercise, not a physical model or hardware benchmark.
