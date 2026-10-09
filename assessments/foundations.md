# Foundations synthesis: defend a regression experiment

**Scope:** arrays, pure functions, autodiff, vectorization, compiled state transitions and optimization on CPU. Complete the changed-condition numerical tasks below and record your verification outputs in your portfolio.

[Open the project workspace](../project.html?id=regression-audit). Keep your evidence in [My learning](../notebook.html#portfolio).

Complete the regression-audit project first. Transfer your `regression-audit` implementation to the modified fixture below and verify each calculation against an independent NumPy reference.

## Before starting

Use the CPU environment from the setup lesson. Record Python, JAX and NumPy versions, backend, device count and precision. Read `projects/regression-audit/README.md` and its checker. Run the public stages from the course checkout and retain the output:

```sh
# Run run command in terminal using the course Python environment
python projects/regression-audit/tests/check.py --stage all --implementation starter
```

The project checker accepts `starter` or `solution`; it does not accept an arbitrary file path. If you keep your assessment implementation separately, import and run that file in your own assessment harness. State which file each command tested. Do not edit the public checker to manufacture a pass.

Create an assessment folder containing your implementation, `verify.py`, observed output and `report.md`. Start by writing predictions for Tasks 1–3 before executing their JAX checks. Estimated effort is two or more study sessions; debugging and reviewer discussion can take longer.

## Task 1: change the objective, then derive its first update

Use three observations with unequal, non-centered inputs:

```text
x = [-2, 0, 3]
y = -1.5*x + 0.25
initial weight = 0.5
initial bias = -0.5
learning rate for this first update = 0.025
```

Your model is `prediction = weight*x + bias`. The objective is the **mean** squared residual, with both prediction and target shape `(3,)`.

1. Write the three targets, predictions and residuals by hand.
2. Derive the scalar loss and both parameter derivatives from the residuals. Explain why the nonzero input mean couples the weight and bias behavior.
3. Derive one update without using `jax.grad`. Then implement the same update with JAX and compare each component to your independent derivation.
4. Compare central differences at perturbations `1e-1`, `1e-2`, `1e-3` and `1e-5` using the same initial parameters. Report dtype, absolute error and your reason for choosing a useful perturbation. You are not required to make the smallest perturbation the most accurate.

The independent oracle must not call the same JAX objective or derivative it checks. A host NumPy calculation of residuals and the derived formulas is acceptable. For analytic comparisons use float32 `rtol=1e-5, atol=1e-5`; if you choose different tolerances, justify them with the observed error and scale. Do not loosen tolerances solely until a failing implementation passes.

**Keep:** signed residual table, derivation, independent host check, automatic/finite-difference table and the new weight/bias after one update.

## Task 2: train on changed data and verify the transition contract

Train the changed fixture with a constant rate of `0.1` for `200` updates, starting from Task 1’s initial parameters. Express the repeated update with `lax.scan` and keep a loss history with exactly 200 entries. Specify whether each entry describes the pre-update or post-update parameters, and verify the first entry accordingly.

Check all of the following:

- The starting parameter pytree and the caller’s input arrays are unchanged after training.
- A host loop implementing your derived update tracks the JAX parameters at updates 1, 2, 10 and 200 within `rtol=1e-5, atol=1e-5`.
- The recovered weight and bias are within `1e-4` of the known target values.
- Predictions agree with the target line at held-out inputs `[-1.25, 0.75, 2.5]`; those rows never participate in the update.
- Repeating the run with the same explicit inputs reproduces the full history and final parameters within your stated tolerances.

Use `jax.make_jaxpr` to inspect a four-step version and explain where the repeated transition appears. A printed jaxpr is a program representation; it is not a measurement of speed or TPU efficiency. Time measurements are optional and must separate compilation from synchronized execution if included.

**Keep:** parameter/history checks at named boundaries, caller-state comparison, held-out results, replay results and an explained loop representation.

## Task 3: explain a rate that fails

Return to the public project’s centered 21-point fixture `x = linspace(-1,1,21)`, target `2*x+1` and zero initial parameters. Derive the gradient-descent recurrence for the **bias error** using the zero mean of x. State the interval of constant learning rates that contracts this bias error.

Run rates `0.1`, `0.5` and `1.1` for 30 updates. Predict the sign and magnitude pattern of the bias error at each rate before running. Report the first five bias errors, initial/final loss and whether all values remain finite. Explain why “finite” and “improving” are different checks.

Do not infer that every optimizer or dataset has this same rate boundary. The recurrence belongs to this quadratic objective and this centered fixture. Explain what changes when the input mean is nonzero, as in Task 1.

**Keep:** the recurrence, rate predictions, actual bias/error trajectories and a diagnosis connected to the update equation.

## Task 4: repair a plausible shape bug

In a disposable copy of your loss, remove the shape guard and pass predictions of shape `(3,)` with targets of shape `(3,1)`. Predict the residual shape before execution. Show its scalar mean is a different objective, not merely another storage layout.

Repair the public interface so incompatible shapes raise `ValueError` before subtraction. Then provide a separate, explicit conversion from column targets to vector targets when the caller intentionally supplies that representation. Verify the converted objective against Task 1 and demonstrate that valid singleton batches retain shape `(1,)`.

**Keep:** wrong residual matrix, wrong objective, caught interface error, intentional conversion and singleton-batch check. An error string without an explanation of the axes is insufficient.

## Submit an inspectable evidence package

Your report should let a reviewer reproduce the changed experiment without your notebook’s hidden state. Include:

- The tested implementation path, CPU environment and exact commands.
- Predictions recorded before execution and the independent calculations.
- Assertions with actual observed output for all four tasks.
- Loss-history convention, chosen tolerances and shape/state contracts.
- Failure evidence plus the repaired behavior.
- What was not tested: external-data generalization, TPU execution, distributed recovery and independently reviewed production readiness.

Keep learner-written work distinct from public-check output and reference solutions. If a task is unresolved, describe the evidence that failed and the next diagnostic question rather than labeling it complete.

## Reviewer decision rubric

### Changed objective

- **Accept:** All signed values, gradients and the update match an independent derivation; finite-difference error is explained with dtype/scale
- **Revise:** Expected values come from the same autodiff call; axes or reduction are unexplained; tolerance hides a mismatch

### Compiled training

- **Accept:** Four named update boundaries, 200-entry convention, unchanged caller state, changed-target recovery, held-out checks and full replay are demonstrated
- **Revise:** Only final training loss is shown; history indexing is ambiguous; inputs/state are mutated; predictions are hard-coded

### Rate diagnosis

- **Accept:** Bias-error recurrence correctly predicts the three rate behaviors, and observed finite/improving checks are distinguished
- **Revise:** A plot is described as “unstable” without a mechanism; one dataset’s threshold is claimed universally

### Shape repair

- **Accept:** Pairwise broadcasting is explained, the interface rejects it, explicit conversion restores the intended objective and singleton shapes work
- **Revise:** Silent squeezing hides incompatible axes; a convenient numeric result is accepted without an interface contract

### Evidence

- **Accept:** An identifiable implementation, reproducible CPU command/environment and bounded claims accompany every task
- **Revise:** Reference execution is presented as learner work; screenshots replace runnable evidence; unsupported TPU/readiness claims appear

Review tasks individually as **accept**, **revise**, or **not demonstrated** and explain the reason. An overall local assessment is accepted only when every task and its evidence are accepted. A reviewer may ask the learner to explain one changed condition live. Public checks alone do not decide this assessment, and acceptance does not issue a certificate.

Reviewer numerical references are separate. After your attempt, [download the reviewer notes](foundations-reviewer.md). These notes are public, not a secret examination key; reading them is separate from demonstrating the tasks.
