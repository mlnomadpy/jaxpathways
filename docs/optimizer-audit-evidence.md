# Optimizer audit authoring evidence

Authored and executed 2026-10-05. This is original synthetic teaching material and internal author review, not independent expert or learner approval.

## Deliverables and integration

- `projects/optimizer-audit/project.json`: authored project, pathway `foundations`, CPU.
- `solution/optimization.py`: transparent quadratic geometry, training-only coordinate scaling, exhaustive minibatch enumeration, compiled GD/momentum/Adam training, clipping/schedules and augmented ridge least squares.
- `starter/optimization.py`: five corresponding implementation tasks; explicit validation, generated fixtures and loop scaffolding remain supplied.
- `tests/check.py`: five cumulative stages with independent numerical references and changed conditions.
- `examples/figures.py`: reruns the full checker, executes the reference experiments and writes three PNG/SVG figures plus source-bound numeric evidence.
- `README.md`: mechanism-first stage instructions, worked calculation, interpreted actual plots, distinct transfer tasks and numerical diagnosis.

Register project ID `optimizer-audit` for phase `optimization` with stages `1`–`5`. Required lessons are `optimization-01` through `optimization-12`. Preserve `regression-audit` as the initial four-lesson project and existing foundations assessment. This project links to that assessment and the phase README without rewriting shared sources. Phase integration evidence: explain loss/geometry; verify derivatives; diagnose conditioning, curvature and sampling noise; justify optimizer and regularization choices using held-out measurements.

## Executed checks

```sh
python3 projects/optimizer-audit/tests/check.py --implementation solution --stage all
python3 projects/optimizer-audit/examples/figures.py
```

All five stages passed. Each cumulative stage `1` through `5` was also executed in its own fresh process. The unfinished starter fails at its intended Stage 1 `NotImplementedError`. All project Python sources parsed, all 91 inline/display TeX expressions in the README compiled through installed KaTeX, and source/checker/figure hashes match `outputs/evidence.json`.

Environment: Python 3.14.3, JAX 0.9.2, NumPy 2.4.4, CPU. Geometry and optimizer trajectories use JAX float32; independent oracles and direct ridge audit use NumPy float64. Installed JAX autodiff, Hessian and scan signatures were inspected; official JAX autodiff and Optax optimizer/transformation documentation were read. No new runtime packages were installed.

Independent coverage includes multiple feature/row counts, singleton observations, two penalty values, exact gradient/Hessian algebra, directional differences, input purity, zero feature scales, prediction-preserving coordinate mapping, every ordered iid batch of sizes one through three, exact covariance scaling, incorrect unequal-batch averaging, full NumPy optimizer trajectories, pre/post metric conventions, schedule boundary, clipping order, finite spectral divergence, rank-deficient ridge closed form and changed-design stationarity.

## Recorded observations and interpretation

- Raw Hessian condition number is about 999.76; RMS scaling reduces it to 4.99. Equal 80-update runs with rate `0.9/L` give full-training MSE 1.294 raw versus 0.01930 scaled. This compares coordinate methods, not equal wall-clock budgets.
- Exact covariance traces for iid batch sizes one, two and three are 6.8472, 3.4236 and 2.2824. The README states the with-replacement condition and distinguishes this from Monte Carlo estimates.
- The shared 150-batch comparison executes SGD, momentum, Adam, clipped Adam and clipped Adam with decay. All curves use the same fixed validation MSE; method-specific rates and state conventions are declared. Clipping before Adam does not bound the final update by rate times the threshold.
- The recorded ridge sweep selects **zero** penalty on validation, with final one-time test MSE about 0.02345. Positive ridge greatly stabilizes coefficients but does not improve this validation score. The README preserves this actual result and explains why prediction and identification differ.

All three PNGs were visually inspected: coordinate paths/curvature/stability; noise/optimizer/clipping behavior; ridge quality/coefficients/sensitivity. Axis units, logarithmic scales, categorical penalty spacing, update indexing and representative values match the executed data. The regularization sensitivity curves have different units, explicitly explained in the text.

## Boundaries

The public examples are small original synthetic CPU experiments. No production convergence, algorithm ranking, runtime speedup, external-data quality or accelerator execution is claimed. Equal updates/examples do not establish equally tuned methods or equal runtime. The optimizer comparison's held-out set is validation data; ridge uses a separate test set once after validation selection. The actual reference source and figure receipt are inspectable and remain separate from learner-written evidence. Root owns manifest registration, whole-site generation and final full-project/corpus execution.
