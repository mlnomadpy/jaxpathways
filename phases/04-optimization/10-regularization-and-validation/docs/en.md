# Regularization and honest validation

Phase 04: Math & optimization · about 75 minutes · CPU

## What you will be able to do

- Derive a ridge solution, distinguish data loss from penalized loss, and select a penalty using held-out data.
- State the exact objective
- Work through shrinkage by hand
- Keep fitting, choosing and reporting separate
- Choose which parameters to penalize

## The problem

A model can fit the training data closely and still make poor predictions on new data. Can we trade a little training fit for a simpler model? We will make that tradeoff visible with a dataset small enough to solve by hand, while keeping the validation data separate from fitting.

## The idea

Regularization changes which solutions training prefers. Validation helps choose settings using data that did not fit the model, and a final test estimates performance after those choices are fixed. Keeping these jobs separate makes the reported result interpretable.

## Give each split one job

Imagine trying several penalty strengths. For each setting, fit weights on the training split, then score the frozen result on validation. Pick a setting using those validation results. Repeatedly using test performance to make that choice quietly turns the test into another validation set.

Preprocessing belongs inside the same boundary: fit means, scales and feature selection using training data only. Applying a transformation to validation is allowed; allowing validation to choose the transformation's fitted statistics changes the experiment.

The regularization plot may show training error improving while validation error worsens. That gap motivates model selection, but one small fixture does not prove a universal best penalty. Preserve split identities and the selection rule so another person can reconstruct the decision.

### Pause and reason

After selecting a penalty, you inspect the test score and change the penalty again. What has changed?

<details><summary>Compare your reasoning</summary>

The test data has influenced model selection. Its score no longer represents a single untouched final evaluation under the original protocol. Use a fresh evaluation protocol or disclose the repeated selection.

</details>

## State the exact objective

Let $n$ be the training sample count and $\lambda\ge0$ the penalty strength. We use mean squared error plus $\lambda\|w\|_2^2$, without a factor of one half. Differentiating gives $2X^\mathsf{T}(Xw-y)/n+2\lambda w$. Setting it to zero gives the linear system below. Other conventions change the numerical meaning of the same penalty value, so copy the objective along with a reported hyperparameter.

$$
L_\lambda(w)=\frac1n\|Xw-y\|_2^2+\lambda\|w\|_2^2,\qquad (X^\mathsf{T}X+n\lambda I)w=X^\mathsf{T}y
$$

## Work through shrinkage by hand

Train a slope without a bias on inputs $(-1,1)$ and targets $(-3,3)$. The data loss is $(w-3)^2$, so the ridge solution is $w=3/(1+\lambda)$. At $\lambda=0$, it fits the training slope $3$. At $\lambda=1/2$, it gives slope $2$. Our deliberately constructed validation targets follow slope $2$, making the tradeoff easy to inspect. Real validation curves are noisier and do not guarantee that positive regularization helps.

## Keep fitting, choosing and reporting separate

Fit weights on training examples for each candidate $\lambda$. Choose using validation prediction loss, not the training objective with its penalty. Validation targets must never enter the fit or preprocessing statistics. After many choices, validation itself can be overfit; a final untouched test set estimates the selected procedure. This tiny exercise has a training and validation split only, so its result is a demonstration, not a final generalization estimate.

## Choose which parameters to penalize

A bias can represent the overall target level, so many models exclude it from weight regularization. For an augmented parameter vector $(w,b)$, use a diagonal penalty mask with entries $(1,0)$. Scaling a feature also changes the size of its coefficient, and therefore the penalty. Standardize using training statistics when that matches your modeling choice, then reuse those statistics unchanged for validation and inference. Ridge adds curvature in penalized directions; it cannot repair leaked data or a wrong target.

## Create separate training and validation examples

Create main.py. Keep validation arrays separate from the solver inputs.

```python
# Step 1 — Create separate training and validation examples: The helper solves the explicitly stated mean-loss convention.
# Import jax for this computation.
import jax
import jax.numpy as jnp
# Construct `X` via `jnp.array([[-1.0], [1.0]])`
X = jnp.array([[-1.0], [1.0]])
# Construct `y` via `jnp.array([-3.0, 3.0])`
y = jnp.array([-3.0, 3.0])
# Construct `Xv` via `jnp.array([[-2.0], [2.0]])`
Xv = jnp.array([[-2.0], [2.0]])
# Construct `yv` via `jnp.array([-4.0, 4.0])`
yv = jnp.array([-4.0, 4.0])

# Function `ridge(X, y, lam)` implementing this stage's computation:
def ridge(X, y, lam):
    # Return `jnp.linalg.solve(X.T @ X + X.shape[0] * lam * jnp.eye(X.shape[1]), X.T @ y)` to the caller.
    return jnp.linalg.solve(X.T @ X + X.shape[0] * lam * jnp.eye(X.shape[1]), X.T @ y)

# Function `mse(w, X, y)` implementing this stage's computation:
def mse(w, X, y):
    # Return `jnp.mean((X @ w - y) ** 2)` to the caller.
    return jnp.mean((X @ w - y) ** 2)
```

The helper solves the explicitly stated mean-loss convention. We use a small well-conditioned system here; this is not a general replacement for robust least-squares algorithms.

## Verify the penalized optimum

Append the hand solution and its gradient check.

```python
# Step 2 — Verify the penalized optimum: The data gradient and penalty gradient cancel at w=2.
lam = 0.5
# Run `ridge` to compute `w`.
w = ridge(X, y, lam)
# Compute `objective` from `lambda w: mse(w, X, y) + lam * jnp.dot(w, w)`
objective = lambda w: mse(w, X, y) + lam * jnp.dot(w, w)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(w, jnp.array([2.0]), atol=1e-06)
# Check numerical equivalence within tolerance: `jnp.allclose(jax.grad(objective)(w), jnp.zeros(1), atol=1e-06)`
assert jnp.allclose(jax.grad(objective)(w), jnp.zeros(1), atol=1e-06)
```

The data gradient and penalty gradient cancel at $w=2$.

## Separate the quantities you report

Append the three losses and run python main.py.

```python
# Step 3 — Separate the quantities you report: Training prediction loss is 1, validation loss is 0, and the...
train = mse(w, X, y)
# Run `mse` to compute `validation`.
validation = mse(w, Xv, yv)
# Run `objective` to compute `penalized`.
penalized = objective(w)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jnp.array([train, validation, penalized]), jnp.array([1.0, 0.0, 3.0]))
# Print the observed values to compare against the expected result.
print('train / validation / objective:', train, validation, penalized)
```

Training prediction loss is $1$, validation loss is $0$, and the penalized objective is $3$. They answer different questions.

## Run the example

```python
# Step 1 — Create separate training and validation examples: The helper solves the explicitly stated mean-loss convention.
# Import jax for this computation.
import jax
import jax.numpy as jnp
# Construct `X` via `jnp.array([[-1.0], [1.0]])`
X = jnp.array([[-1.0], [1.0]])
# Construct `y` via `jnp.array([-3.0, 3.0])`
y = jnp.array([-3.0, 3.0])
# Construct `Xv` via `jnp.array([[-2.0], [2.0]])`
Xv = jnp.array([[-2.0], [2.0]])
# Construct `yv` via `jnp.array([-4.0, 4.0])`
yv = jnp.array([-4.0, 4.0])

# Function `ridge(X, y, lam)` implementing this stage's computation:
def ridge(X, y, lam):
    # Return `jnp.linalg.solve(X.T @ X + X.shape[0] * lam * jnp.eye(X.shape[1]), X.T @ y)` to the caller.
    return jnp.linalg.solve(X.T @ X + X.shape[0] * lam * jnp.eye(X.shape[1]), X.T @ y)

# Function `mse(w, X, y)` implementing this stage's computation:
def mse(w, X, y):
    # Return `jnp.mean((X @ w - y) ** 2)` to the caller.
    return jnp.mean((X @ w - y) ** 2)

# Step 2 — Verify the penalized optimum: The data gradient and penalty gradient cancel at w=2.
lam = 0.5
# Run `ridge` to compute `w`.
w = ridge(X, y, lam)
# Compute `objective` from `lambda w: mse(w, X, y) + lam * jnp.dot(w, w)`
objective = lambda w: mse(w, X, y) + lam * jnp.dot(w, w)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(w, jnp.array([2.0]), atol=1e-06)
# Check numerical equivalence within tolerance: `jnp.allclose(jax.grad(objective)(w), jnp.zeros(1), atol=1e-06)`
assert jnp.allclose(jax.grad(objective)(w), jnp.zeros(1), atol=1e-06)

# Step 3 — Separate the quantities you report: Training prediction loss is 1, validation loss is 0, and the...
train = mse(w, X, y)
# Run `mse` to compute `validation`.
validation = mse(w, Xv, yv)
# Run `objective` to compute `penalized`.
penalized = objective(w)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jnp.array([train, validation, penalized]), jnp.array([1.0, 0.0, 3.0]))
# Print the observed values to compare against the expected result.
print('train / validation / objective:', train, validation, penalized)
```

Expected: The regularized slope is $2$; data loss and penalized objective are reported separately.

## Training error and validation error select different penalties

**Predict:** Does the smallest training error imply the best validation result?

![Training error and validation error select different penalties](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis varies the ridge penalty $\lambda$. For each penalty, the model is fitted again. The vertical axis shows mean squared prediction error on training and validation data; it does not include the penalty term.

At $\lambda=0$, training error is zero and validation error is $4$. At $\lambda=0.5$, training error rises to $1$, but validation error falls to zero. Increasing the penalty to $2$ makes both errors $4$. The validation curve therefore has a minimum between the extremes.

### Connect it to the computation

The fitted weight is $w=3/(1+\lambda)$. The training targets favor slope $3$, while the validation targets in this constructed example favor slope $2$. Moderate shrinkage moves the model toward the validation relationship; too much shrinkage moves it past that relationship and toward underfitting.

Select the penalty using the validation minimum, not the intersection of the curves or the smallest training error. Here that choice is $0.5$. A zero validation error is a property of this tiny synthetic example, not a promise that regularization removes all error on real data.

```python
# Compute figure data for: Training error and validation error select different penalties
# Generate a uniform grid of points in `grid`.
grid = jnp.linspace(0.0, 2.0, 41)
# Compute `visual_data` from `{'kind': 'line', 'x': grid.tolist(), 'xlabel': 'ridg...`
visual_data = {'kind': 'line', 'x': grid.tolist(), 'xlabel': 'ridge penalty', 'ylabel': 'mean squared prediction error', 'series': [{'label': 'training', 'y': [float(mse(ridge(X, y, float(a)), X, y)) for a in grid]}, {'label': 'validation', 'y': [float(mse(ridge(X, y, float(a)), Xv, yv)) for a in grid]}]}
```

## Recorded reference execution

CPU run: 2026-10-08T14:02:13.081899+00:00. JAX 0.9.2.

```text
train / validation / objective: 1.0 0.0 3.0
train / validation / objective: 1.0 0.0 3.0
penalty / train / validation: [0.  0.5 2. ] [0. 1. 4.] [4. 0. 4.]
PASS: optimization-10

```

## Sweep the penalty

**Predict before running:** Will the penalty with the lowest training error also have the lowest validation error?

```python
# Experiment — Sweep the penalty: Regularization deliberately trades training fit against a...
# Construct `candidates` via `jnp.array([0.0, 0.5, 2.0])`
candidates = jnp.array([0.0, 0.5, 2.0])
# Combine or mask array elements to form `fits`.
fits = jnp.stack([ridge(X, y, float(a)) for a in candidates])
# Construct `training` via `jnp.array([mse(a, X, y) for a in fits])`
training = jnp.array([mse(a, X, y) for a in fits])
# Construct `validation` via `jnp.array([mse(a, Xv, yv) for a in fits])`
validation = jnp.array([mse(a, Xv, yv) for a in fits])
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(fits[:, 0], jnp.array([3.0, 2.0, 1.0]))
# Check numerical equivalence within tolerance: `jnp.allclose(training, jnp.array([0.0, 1.0, 4.0]))`
assert jnp.allclose(training, jnp.array([0.0, 1.0, 4.0]))
# Check numerical equivalence within tolerance: `jnp.allclose(validation, jnp.array([4.0, 0.0, 4.0]))`
assert jnp.allclose(validation, jnp.array([4.0, 0.0, 4.0]))
# Assert invariant `int(jnp.argmin(training)) != int(jnp.argmin(validation))` holds
assert int(jnp.argmin(training)) != int(jnp.argmin(validation))
# Print the observed values to compare against the expected result.
print('penalty / train / validation:', candidates, training, validation)
```

**Expected:** Validation selects $\lambda=0.5$; training error prefers $\lambda=0$.

Regularization deliberately trades training fit against a preference for smaller weights.

## Duplicate the training set

**Predict before running:** If every training example is repeated, should the same mean-loss objective choose different weights?

```python
# Experiment — Duplicate the training set: Using a mean data loss keeps the penalty tradeoff unchanged...
repeat = ridge(jnp.tile(X, (2, 1)), jnp.tile(y, 2), lam)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(repeat, w, atol=1e-06)
```

**Expected:** The weights stay fixed.

Using a mean data loss keeps the penalty tradeoff unchanged under exact dataset duplication.

## Make it yours

For a target slope of $4$ and penalty $\lambda=1$, derive the fitted slope and verify its gradient.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.zeros / jnp.ones(shape, dtype=...)` — Allocates a tensor of the given `shape` initialized with constants.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `jax.grad(loss_fn)(params, ...)` — Transforms a scalar-output function into a function returning the gradient PyTree with the same structure as `params`.

**Step-by-step implementation plan:**
1. Construct `y4` via `jnp.array([-4.0, 4.0])`
2. Run `ridge` to compute `w4`.
3. Verify that the numerical values match the expected reference within tolerance.
4. Check numerical equivalence within tolerance: `jnp.allclose(jax.grad(lambda z: mse(z, X, y4) + jnp.dot(z, z))(w4...`

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: For a target slope of 4 and penalty \lambda=1, derive the fitted slope...
# Construct `y4` via `jnp.array([-4.0, 4.0])`
y4 = jnp.array(...)  # TODO: compute y4
# Run `ridge` to compute `w4`.
w4 = ridge(...)  # TODO: compute w4
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(w4, jnp.array([2.0]))  # TODO: complete assertion check
# Check numerical equivalence within tolerance: `jnp.allclose(jax.grad(lambda z: mse(z, X, y4) + jnp.dot(z, z))(w4...`
assert jnp.allclose(jax.grad(lambda z: mse(z, X, y4) + jnp.dot(z, z))(w4), jnp.zeros(1), atol=1e-06)  # TODO: complete assertion check
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: For a target slope of 4 and penalty \lambda=1, derive the fitted slope...
# Construct `y4` via `jnp.array([-4.0, 4.0])`
y4 = jnp.array([-4.0, 4.0])
# Run `ridge` to compute `w4`.
w4 = ridge(X, y4, 1.0)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(w4, jnp.array([2.0]))
# Check numerical equivalence within tolerance: `jnp.allclose(jax.grad(lambda z: mse(z, X, y4) + jnp.dot(z, z))(w4...`
assert jnp.allclose(jax.grad(lambda z: mse(z, X, y4) + jnp.dot(z, z))(w4), jnp.zeros(1), atol=1e-06)
```

</details>

## Do not accidentally shrink the intercept

**Transfer / diagnosis**

Fit constant targets $(5,5,5)$ using inputs $(-1,0,1)$ and a bias. Compare penalizing only the slope with penalizing both parameters at $\lambda=1$.

<details><summary>Hint</summary>

Use a diagonal mask $\operatorname{diag}(1,0)$.

</details>

### How to write: Do not accidentally shrink the intercept — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.zeros / jnp.ones(shape, dtype=...)` — Allocates a tensor of the given `shape` initialized with constants.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Construct `x` via `jnp.array([-1.0, 0.0, 1.0])`
2. Construct `A` via `jnp.stack([x, jnp.ones_like(x)], axis=1)`
3. Compute `target` from `jnp.full((3,), 5.0)`
4. Construct `mask` via `jnp.diag(jnp.array([1.0, 0.0]))`
5. Perform matrix contraction / projection to compute `correct`.

**Starter code scaffold (fill in the TODOs):**

```python
# Do not accidentally shrink the intercept (Transfer / diagnosis): The mask expresses a modeling choice: shrink feature effects...
# Construct `x` via `jnp.array([-1.0, 0.0, 1.0])`
x = jnp.array(...)  # TODO: compute x
# Construct `A` via `jnp.stack([x, jnp.ones_like(x)], axis=1)`
A = jnp.stack(...)  # TODO: compute A
# Compute `target` from `jnp.full((3,), 5.0)`
target = jnp.full(...)  # TODO: compute target
# Construct `mask` via `jnp.diag(jnp.array([1.0, 0.0]))`
mask = jnp.diag(...)  # TODO: compute mask
# Perform matrix contraction / projection to compute `correct`.
correct = jnp.linalg.solve(...)  # TODO: compute correct
# Run `ridge` to compute `all_penalized`.
all_penalized = ridge(...)  # TODO: compute all_penalized
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(correct, jnp.array([0.0, 5.0]))  # TODO: complete assertion check
# Check numerical equivalence within tolerance: `jnp.allclose(all_penalized, jnp.array([0.0, 2.5]))`
assert jnp.allclose(all_penalized, jnp.array([0.0, 2.5]))  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Do not accidentally shrink the intercept (Transfer / diagnosis): The mask expresses a modeling choice: shrink feature effects...
# Construct `x` via `jnp.array([-1.0, 0.0, 1.0])`
x = jnp.array([-1.0, 0.0, 1.0])
# Construct `A` via `jnp.stack([x, jnp.ones_like(x)], axis=1)`
A = jnp.stack([x, jnp.ones_like(x)], axis=1)
# Compute `target` from `jnp.full((3,), 5.0)`
target = jnp.full((3,), 5.0)
# Construct `mask` via `jnp.diag(jnp.array([1.0, 0.0]))`
mask = jnp.diag(jnp.array([1.0, 0.0]))
# Perform matrix contraction / projection to compute `correct`.
correct = jnp.linalg.solve(A.T @ A + 3 * mask, A.T @ target)
# Run `ridge` to compute `all_penalized`.
all_penalized = ridge(A, target, 1.0)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(correct, jnp.array([0.0, 5.0]))
# Check numerical equivalence within tolerance: `jnp.allclose(all_penalized, jnp.array([0.0, 2.5]))`
assert jnp.allclose(all_penalized, jnp.array([0.0, 2.5]))
```

The mask expresses a modeling choice: shrink feature effects while preserving the constant target level.

</details>

## Catch preprocessing leakage

**Transfer / diagnosis**

Training inputs are $(0,2)$ and validation input is $100$. Compare training-only centering with centering all inputs together. Which can be reused for a truly unseen example?

<details><summary>Hint</summary>

Fit the mean on training data; do not refit it on validation data.

</details>

### How to write: Catch preprocessing leakage — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `x.sum(axis=...) / x.mean(axis=..., keepdims=...)` — Reduces values along the named `axis` (the axis you name is collapsed unless `keepdims=True`).
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Construct `train_x` via `jnp.array([0.0, 2.0])`
2. Construct `val_x` via `jnp.array([100.0])`
3. Aggregate array values to compute `train_mean`.
4. Aggregate array values to compute `leaked_mean`.
5. Verify that the numerical values match the expected reference within tolerance.

**Starter code scaffold (fill in the TODOs):**

```python
# Catch preprocessing leakage (Transfer / diagnosis): Validation data must not influence fitted preprocessing.
# Construct `train_x` via `jnp.array([0.0, 2.0])`
train_x = jnp.array(...)  # TODO: compute train_x
# Construct `val_x` via `jnp.array([100.0])`
val_x = jnp.array(...)  # TODO: compute val_x
# Aggregate array values to compute `train_mean`.
train_mean = jnp.mean(...)  # TODO: compute train_mean
# Aggregate array values to compute `leaked_mean`.
leaked_mean = jnp.mean(...)  # TODO: compute leaked_mean
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(val_x - train_mean, jnp.array([99.0]))  # TODO: complete assertion check
# Check numerical equivalence within tolerance: `not jnp.allclose(train_mean, leaked_mean)`
assert not jnp.allclose(train_mean, leaked_mean)  # TODO: complete assertion check
# Check numerical equivalence within tolerance: `jnp.allclose(jnp.mean(train_x - train_mean), 0.0)`
assert jnp.allclose(jnp.mean(train_x - train_mean), 0.0)  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Catch preprocessing leakage (Transfer / diagnosis): Validation data must not influence fitted preprocessing.
# Construct `train_x` via `jnp.array([0.0, 2.0])`
train_x = jnp.array([0.0, 2.0])
# Construct `val_x` via `jnp.array([100.0])`
val_x = jnp.array([100.0])
# Aggregate array values to compute `train_mean`.
train_mean = jnp.mean(train_x)
# Aggregate array values to compute `leaked_mean`.
leaked_mean = jnp.mean(jnp.concatenate([train_x, val_x]))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(val_x - train_mean, jnp.array([99.0]))
# Check numerical equivalence within tolerance: `not jnp.allclose(train_mean, leaked_mean)`
assert not jnp.allclose(train_mean, leaked_mean)
# Check numerical equivalence within tolerance: `jnp.allclose(jnp.mean(train_x - train_mean), 0.0)`
assert jnp.allclose(jnp.mean(train_x - train_mean), 0.0)
```

Validation data must not influence fitted preprocessing. Reusing training statistics preserves the meaning of held-out evaluation.

</details>

## Check your understanding

Which loss should select the penalty in this exercise?

1. The penalized training objective alone
2. Unpenalized validation prediction loss
3. A repeatedly inspected final test loss

<details><summary>Answer and explanation</summary>

Unpenalized validation prediction loss

Validation compares predictive performance of candidates fitted on training data. A separate final test set is for reporting after selection.

</details>

## Diagnose the result

If increasing $\lambda$ improves the penalized objective but worsens predictions, check which quantity is being compared. Inspect the mean-versus-sum convention, bias mask, feature units and where preprocessing statistics were fitted.

## Carry forward

- Regularization deliberately trades training fit against a preference for smaller weights.
- Using a mean data loss keeps the penalty tradeoff unchanged under exact dataset duplication.

## Keep your evidence

Keep the ridge gradient/solve comparison, penalty sweep with separate train/validation losses, duplication check and bias-mask repair.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [Stanford CS229 notes: regularization and model selection](https://cs229.stanford.edu/notes2022fall/main_notes.pdf)
- [JAX linear solve](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.solve.html)

