# Audit a differentiable scientific inverse problem

Build a reusable cooling simulator, infer an unknown positive rate, and defend the result with numerical and physical evidence. Complete `science-01` through `science-04` first. This project uses synthetic exponential cooling so independent answers are available; it is not a validation of an actual instrument or a general ODE solver.

## Start and run

Use the course CPU environment in `requirements-cpu.txt`. Run commands from the checkout or extracted bundle's top folder. The scaffold enables float64 before creating arrays.

```sh
cp projects/scientific-inverse/starter/model.py projects/scientific-inverse/my_model.py
python3 projects/scientific-inverse/tests/check.py --implementation projects/scientific-inverse/my_model.py --stage 1
```

In Windows PowerShell, use `Copy-Item` instead of `cp`. Keep `my_model.py` as your implementation; the reference stays in `solution/model.py`.

## Stage 1: model and state contract

Implement `simulate(rate, initials, steps=40, dt=0.05)`. Use classical RK4 and explicit scan state. Return a batch-major array `(B, steps+1)` that includes the starting value in column zero. Reject empty or non-vector initials, a nonscalar rate, nonpositive step count and nonpositive step size. Keep the rate differentiable, so do not convert it to a Python float inside the simulation.

Before running, derive the discrete multiplier and predict whether a larger rate should raise or lower the endpoint. Tests compare the entire trajectory with an independent RK4 polynomial and the continuous exponential at two rate/grid settings, including a singleton batch and permuted trajectory order.

Save a shape diagram and a refinement table. Reproduce an overly large Euler step separately; explain why a finite, differentiable result can still violate physical behavior.

## Stage 2: loss and sensitivity

Implement `loss(log_rate, initials, observations, dt=0.05)`. Convert the log parameter to a positive rate with `exp`, simulate the same observation times, and return mean squared error over all batch/time entries. Require exactly `(B, steps+1)` observations with at least two times; do not rely on broadcasting.

```sh
python3 projects/scientific-inverse/tests/check.py --implementation projects/scientific-inverse/my_model.py --stage 2
```

Derive the log-rate chain rule before checking it. The checker uses an independent host polynomial and finite differences at two nonoptimal parameters. Save those comparisons, a justified tolerance, and a deliberate detached-gradient failure with its repair. Explain the difference between a derivative of the discretization and a sensitivity of the continuous model.

## Stage 3: inference and measurement

Implement `fit` using the explicit log-rate gradient update documented in the starter. Return the fitted rate and the complete loss history including the initial pre-update value. Implement `evaluate` without fitting or mutating inputs. Report MSE, RMSE, maximum absolute error, trajectory count and observation count.

```sh
python3 projects/scientific-inverse/tests/check.py --implementation projects/scientific-inverse/my_model.py --stage all
```

The checks change the generating rate, initial guess and initial conditions; compare full replay; evaluate frozen new initial conditions; refine the simulation grid; and compare one noisy-data fit with an independent dense analytic grid search. A parameter fit that passes only the lesson's original constants is insufficient.

Save a report with data-generation rules and seeds, units, train/held-out split, environment/device/dtype, full learning histories, residual plots, numerical tolerances, an identifiability diagnosis and observed failure/repair. Explain what would change with real measurements: noise model, uncertain initial conditions, model misspecification, measurement times and parameter uncertainty. A single noisy fit is not a confidence interval.

## Inspect the reference after your attempt

```sh
python3 projects/scientific-inverse/tests/check.py --implementation solution --stage all
```

Public reference checks establish behavior on bounded CPU fixtures. They do not establish independent scientific review, a certified competency, or accelerator performance. Continue with the [science synthesis assessment](../../assessments/science.md) for a changed problem and reviewer rubric.
