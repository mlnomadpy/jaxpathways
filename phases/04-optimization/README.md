# Phase 04: Math & optimization

Shared foundations.

Connect geometry and probability to the updates in a training loop. Begin with linear algebra, gradient checks and gradient descent, then investigate conditioning, curvature, likelihoods, regularization and optimizer behavior.

For the initial regression project, focus on Linear algebra and loss intuition, Gradient checking and numerical accuracy, Write gradient descent yourself, and Optimize with Optax. The other eight lessons deepen the mathematical explanation; return to them as needed.

**Prerequisites:** 03: State, randomness & control flow.

**Hardware:** CPU.

## Study guide: Can you explain an optimizer’s behavior from the objective and data?

Use a model small enough to solve independently. Connect residuals, gradients, curvature and update state before comparing optimizer names. A lower training loss is useful evidence, but it does not by itself establish better held-out predictions.

### Check your starting point

A model predicts one value for each observation. Targets accidentally have an extra trailing singleton axis. Why might mean squared error still run?

<details><summary>Compare your reasoning</summary>

Broadcasting can construct a pairwise residual matrix instead of one residual per observation. Assert identical prediction and target shapes, then compute one residual by hand before reducing.

</details>

Review: [Linear algebra and loss intuition](01-linear-algebra-and-loss-intuition/docs/en.md).

### Build in stages

1. **Audit the objective and derivative.** Work through vectors, residuals, rank and directional derivatives. Use a linear solve and a finite-difference sweep as independent checks.

   Lessons: [Vectors, norms, and projections](05-vectors-norms-and-projections/docs/en.md) · [Linear algebra and loss intuition](01-linear-algebra-and-loss-intuition/docs/en.md) · [Least squares, rank, and conditioning](06-least-squares-rank-and-conditioning/docs/en.md) · [The chain rule, Jacobians, and directions](07-chain-rule-jacobians-and-directional-derivatives/docs/en.md) · [Gradient checking and numerical accuracy](02-gradient-checking-and-numerical-accuracy/docs/en.md).

2. **Connect dynamics to evaluation.** Predict a stable versus divergent quadratic update, inspect stable likelihood arithmetic and separate fitting, validation selection and final test reporting.

   Lessons: [Curvature, Hessians, and learning rates](08-curvature-hessians-and-learning-rates/docs/en.md) · [Write gradient descent yourself](03-write-gradient-descent-yourself/docs/en.md) · [Probability, likelihood, and stable losses](09-probability-likelihood-and-stable-losses/docs/en.md) · [Regularization and honest validation](10-regularization-and-validation/docs/en.md).

3. **Make optimizer comparisons fair.** Count-weight unequal batches, preserve optimizer moments and schedule position, then change feature scale while retaining the prediction contract.

   Lessons: [Minibatches, expectation, and gradient noise](11-minibatches-expectation-and-gradient-noise/docs/en.md) · [Optimize with Optax](04-optimize-with-optax/docs/en.md) · [Adam, clipping, and learning-rate schedules](12-adam-clipping-and-learning-rate-schedules/docs/en.md).

### Try a changed condition

Rescale one feature by a factor of ten. Is the same learning rate automatically a fair comparison?

<details><summary>Compare an approach</summary>

No. The parameterization changes directional curvature. Map coefficients to preserve predictions, verify the objective and compare trajectories under stated rates. Fit preprocessing on training data only; never use test performance to choose the rate.

</details>

**Symptom:** Gradient descent diverges while all values initially remain finite.

**Check next:** Check the update sign, loss reduction, feature scales and largest quadratic curvature before trying a different seed.

### Decide what is ready

Complete regression-audit, then optimizer-audit. Keep an analytic reference, stable/unstable trajectories, a resumed optimizer comparison and an untouched final evaluation split.

### Further work

Constrained, nonsmooth and large-scale stochastic optimization are not covered comprehensively. Current ranking evidence is limited to declared synthetic objectives.

## Lesson sequence

### 04.01 Vectors, norms, and projections

[Read the lesson](05-vectors-norms-and-projections/docs/en.md) · [Run the code](05-vectors-norms-and-projections/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Read a dot product geometrically and separate a vector into a projection and an orthogonal residual.

**Evidence:** Keep the hand-computed projection, perpendicular-residual check, rescaling experiment, zero-direction repair and changed-basis calculation.

**Checkpoint:** Does doubling a nonzero direction double the projection onto its line?

### 04.02 Linear algebra and loss intuition

[Read the lesson](01-linear-algebra-and-loss-intuition/docs/en.md) · [Run the code](01-linear-algebra-and-loss-intuition/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Connect a matrix-vector product to predictions and a scalar regression loss.

**Evidence:** Keep the row-by-row predictions, analytic weight/bias derivatives, pairwise-broadcast counterexample, and checked-loss rejection/repair. Include the analytic vector-gradient derivation and mean-versus-sum comparison.

**Checkpoint:** What shape should one scalar target per observation have for predictions of shape (n,)?

### 04.03 Least squares, rank, and conditioning

[Read the lesson](06-least-squares-rank-and-conditioning/docs/en.md) · [Run the code](06-least-squares-rank-and-conditioning/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Fit a linear model with a direct solver and diagnose nonunique or sensitive coefficients.

**Evidence:** Save the hand solution, residual orthogonality, duplicate-column ambiguity, perturbation table and NumPy comparison.

**Checkpoint:** Can zero training error prove that fitted weights are uniquely determined?

### 04.04 The chain rule, Jacobians, and directions

[Read the lesson](07-chain-rule-jacobians-and-directional-derivatives/docs/en.md) · [Run the code](07-chain-rule-jacobians-and-directional-derivatives/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Compute a small Jacobian by hand and connect forward sensitivities to reverse-mode gradients.

**Evidence:** Keep the handwritten Jacobian, JVP and VJP checks, adjoint identity, finite-direction check and stop-gradient diagnosis.

**Checkpoint:** Which object does a reverse-mode pullback map to input sensitivity?

### 04.05 Gradient checking and numerical accuracy

[Read the lesson](02-gradient-checking-and-numerical-accuracy/docs/en.md) · [Run the code](02-gradient-checking-and-numerical-accuracy/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Compare autodiff with finite differences over several step sizes.

**Evidence:** Save analytic and numerical derivatives, the step-size/error table with dtype, the nonquadratic transfer check, and the omitted-term failure/repair. Add the one-sided slopes showing why a symmetric probe cannot certify differentiability at a kink.

**Checkpoint:** Does passing one gradient check prove a loss implementation is correct for all inputs?

### 04.06 Curvature, Hessians, and learning rates

[Read the lesson](08-curvature-hessians-and-learning-rates/docs/en.md) · [Run the code](08-curvature-hessians-and-learning-rates/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Use curvature to explain learning-rate stability, slow directions and why a zero gradient need not be a minimum.

**Evidence:** Keep the Hessian and Hessian-vector checks, stable/divergent trajectories, rescaling comparison and saddle counterexample.

**Checkpoint:** A smooth function has zero gradient at a point. What can we conclude?

### 04.07 Write gradient descent yourself

[Read the lesson](03-write-gradient-descent-yourself/docs/en.md) · [Run the code](03-write-gradient-descent-yourself/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Fit a regression model using explicit parameter updates.

**Evidence:** Keep the two hand-computed steps, rate stability derivation, edge/divergent runs, scaled-curvature transfer and uphill-update repair. Include the tiny-update counterexample and explain your stopping criteria.

**Checkpoint:** What makes rate 1.1 unstable for this particular quadratic?

### 04.08 Probability, likelihood, and stable losses

[Read the lesson](09-probability-likelihood-and-stable-losses/docs/en.md) · [Run the code](09-probability-likelihood-and-stable-losses/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Derive binary cross-entropy from a probability model and compute it safely from logits.

**Evidence:** Keep the hand log-loss, analytic gradient, extreme-logit failure/repair, multiclass shift check and empirical-probability calculation.

**Checkpoint:** Why compute binary cross-entropy from logits?

### 04.09 Regularization and honest validation

[Read the lesson](10-regularization-and-validation/docs/en.md) · [Run the code](10-regularization-and-validation/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Derive a ridge solution, distinguish data loss from penalized loss, and select a penalty using held-out data.

**Evidence:** Keep the ridge gradient/solve comparison, penalty sweep with separate train/validation losses, duplication check and bias-mask repair.

**Checkpoint:** Which loss should select the penalty in this exercise?

### 04.10 Minibatches, expectation, and gradient noise

[Read the lesson](11-minibatches-expectation-and-gradient-noise/docs/en.md) · [Run the code](11-minibatches-expectation-and-gradient-noise/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Explain an unbiased minibatch gradient and verify how sampling and reduction affect its variance.

**Evidence:** Keep the exact per-example mean, enumerated batch-variance calculation, uneven-batch repair, PRNG replay and gradient-accumulation check.

**Checkpoint:** When does averaging batch means reproduce the dataset mean?

### 04.11 Optimize with Optax

[Read the lesson](04-optimize-with-optax/docs/en.md) · [Run the code](04-optimize-with-optax/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Replace the handwritten update with an optimizer and track its state.

**Evidence:** Save the analytic first update, manual/Optax equivalence, retained/reset momentum comparison, continuation replay and double-subtraction repair. Include the zero-gradient momentum step and explain why retained state still moves the parameter.

**Checkpoint:** What should a training step retain besides updated parameters for a stateful optimizer?

### 04.12 Adam, clipping, and learning-rate schedules

[Read the lesson](12-adam-clipping-and-learning-rate-schedules/docs/en.md) · [Run the code](12-adam-clipping-and-learning-rate-schedules/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Explain Adam’s moments, inspect clipping and schedule order, and compare optimizers with a controlled update budget.

**Evidence:** Keep two hand-checked Adam steps, clipping-direction checks, schedule values and boundary, equal-budget loss traces, and the optimization decision memo.

**Checkpoint:** Does clipping gradients before Adam guarantee that final updates obey the same norm threshold?

## Phase project

A fitted regression model.

**Demonstrate:** Explain prediction geometry and loss assumptions; verify derivatives; diagnose conditioning, curvature and gradient noise; justify an optimizer using held-out evidence.

Project status: implemented staged practice · [Open source](../../projects/optimizer-audit/README.md). Use stages 1, 2, 3, 4, 5 for this phase. Complete the regression audit after the four core lessons. This deeper project follows all twelve optimization lessons and connects curvature, conditioning, gradient noise, optimizer state and regularization.



[Primary documentation](https://docs.jax.dev/en/latest/automatic-differentiation.html).
