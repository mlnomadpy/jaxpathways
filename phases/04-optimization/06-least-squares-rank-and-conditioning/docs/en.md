# Least squares, rank, and conditioning

Phase 04: Math & optimization · about 75 minutes · CPU

## What you will be able to do

- Fit a linear model with a direct solver and diagnose nonunique or sensitive coefficients.
- Solve a fit you can verify by hand
- Use a solver, not an explicit inverse
- Separate rank from fit quality
- Measure sensitivity rather than guessing

## The problem

Before training a linear model step by step, can we solve it directly? A tiny least-squares fit gives us a reference answer. Then we will make two features nearly identical and see why a good prediction can hide unreliable coefficients.

## The idea

Least squares finds parameters whose predictions are close to the targets under a squared-error objective. Whether those parameters are uniquely determined and stable is a separate question. Nearly redundant features can make weights sensitive even when predictions look good.

## A good fit can hide uncertain parameters

Suppose two feature columns are identical. A prediction using weights $w_1$ and $w_2$ depends only on their sum. The pairs $(1,2)$ and $(2,1)$ therefore produce identical predictions. A low residual cannot tell us which pair is the intended explanation.

If the columns are almost, rather than exactly, identical, tiny data changes can lead to large changes in the separate weights. This is conditioning: how strongly the solution responds to perturbations. Inspect parameter changes as well as prediction error.

The geometric view projects the target vector onto the space spanned by the columns. A target component outside that space remains as residual even after a correct solve. A residual and an unstable parameter estimate are different problems and call for different diagnostics.

### Pause and reason

What evidence would distinguish an unavoidable residual from a faulty least-squares implementation?

<details><summary>Compare your reasoning</summary>

Compare with an independent solver or known projection, inspect rank, and check the residual's orthogonality to the column space within tolerance. A nonzero residual alone does not imply a bug.

</details>

## Solve a fit you can verify by hand

Take rows $(1,0)$, $(0,1)$, $(1,1)$ and targets $y=(1,2,2)$. These targets cannot all be matched. The best weights are $w=(2/3,5/3)$, with predictions $(2/3,5/3,7/3)$. The residual $r=Xw-y=(-1/3,-1/3,1/3)$ has zero dot product with each column. The mean squared error is $1/9$.

$$
\min_w\|Xw-y\|_2^2,\qquad X^\mathsf{T}(Xw-y)=0
$$

## Use a solver, not an explicit inverse

The orthogonality equation implies $X^\mathsf{T}Xw=X^\mathsf{T}y$. This is useful for reasoning, but forming an inverse is not a good default algorithm. Squaring a full-rank matrix into its Gram matrix squares its condition number in the Euclidean norm. Use `jnp.linalg.lstsq` for the direct reference; it also reports numerical rank and singular values. We recompute the residual so its meaning is explicit.

## Separate rank from fit quality

Rank counts independent column directions. If both columns are the same vector $a$, then predictions depend only on $w_0+w_1$. Infinitely many weight pairs give the same prediction. The least-squares solver chooses a minimum-length solution; that convention is separate from which feature caused the target. A zero residual is therefore not evidence that the individual coefficients are identifiable.

## Measure sensitivity rather than guessing

Singular values measure how strongly a matrix stretches different unit directions. For a full-column-rank matrix, the ratio $\kappa=\sigma_{\max}/\sigma_{\min}$ is its condition number. A tiny smallest singular value means some weight changes barely affect predictions. The inverse question—recovering weights from targets—can then amplify small disturbances. Our perturbation experiment is deliberately aligned with such a weak direction. It illustrates sensitivity, not a universal noise model. Rank decisions also depend on the solver cutoff and dtype.

$$
\kappa_2(X)=\frac{\sigma_{\max}(X)}{\sigma_{\min}(X)}
$$

## Construct an imperfect fit

Create main.py and add this small matrix and target. Work out the proposed weights on paper.

```python
# Step 1 — Construct an imperfect fit: No weights can match all three targets, so a nonzero residual is...
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np
# Construct `X` via `jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])`
X = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
# Construct `y` via `jnp.array([1.0, 2.0, 2.0])`
y = jnp.array([1.0, 2.0, 2.0])
# Construct `expected` via `jnp.array([2 / 3, 5 / 3])`
expected = jnp.array([2 / 3, 5 / 3])
```

No weights can match all three targets, so a nonzero residual is expected.

## Solve and inspect the residual

Append the direct solve and compute the error from the actual predictions.

```python
# Step 2 — Solve and inspect the residual: Perpendicular residuals verify optimality for this unconstrained...
w, _, rank, singular = jnp.linalg.lstsq(X, y, rcond=None)
# Perform matrix contraction / projection to compute `r`.
r = X @ w - y
# Assert invariant `int(rank) == 2` holds
assert int(rank) == 2
# Assert that `jnp.allclose(w, expected, atol=2e-06)`.
assert jnp.allclose(w, expected, atol=2e-06)
# Assert that `jnp.allclose(X.T @ r, jnp.zeros(2), atol=2e-06)`.
assert jnp.allclose(X.T @ r, jnp.zeros(2), atol=2e-06)
```

Perpendicular residuals verify optimality for this unconstrained least-squares problem.

## Compare independent references

Append the mean loss and a NumPy float64 solve; run python main.py.

```python
# Step 3 — Compare independent references: The hand result and a separate numerical implementation agree...
# Assert that `jnp.allclose(jnp.mean(r * r), 1 / 9, atol=2e-06)`.
assert jnp.allclose(jnp.mean(r * r), 1 / 9, atol=2e-06)
# Convert `np_w` to a host NumPy array for inspection or verification.
np_w = np.linalg.lstsq(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), rcond=None)[0]
# Assert that `np.allclose(np.asarray(w), np_w, atol=2e-06)`.
assert np.allclose(np.asarray(w), np_w, atol=2e-06)
# Print the observed values to compare against the expected result.
print('weights / rank / MSE:', w, rank, jnp.mean(r * r))
```

The hand result and a separate numerical implementation agree within float32 tolerance.

## Run the example

```python
# Step 1 — Construct an imperfect fit: No weights can match all three targets, so a nonzero residual is...
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np
# Construct `X` via `jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])`
X = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
# Construct `y` via `jnp.array([1.0, 2.0, 2.0])`
y = jnp.array([1.0, 2.0, 2.0])
# Construct `expected` via `jnp.array([2 / 3, 5 / 3])`
expected = jnp.array([2 / 3, 5 / 3])

# Step 2 — Solve and inspect the residual: Perpendicular residuals verify optimality for this unconstrained...
w, _, rank, singular = jnp.linalg.lstsq(X, y, rcond=None)
# Perform matrix contraction / projection to compute `r`.
r = X @ w - y
# Assert invariant `int(rank) == 2` holds
assert int(rank) == 2
# Assert that `jnp.allclose(w, expected, atol=2e-06)`.
assert jnp.allclose(w, expected, atol=2e-06)
# Assert that `jnp.allclose(X.T @ r, jnp.zeros(2), atol=2e-06)`.
assert jnp.allclose(X.T @ r, jnp.zeros(2), atol=2e-06)

# Step 3 — Compare independent references: The hand result and a separate numerical implementation agree...
# Assert that `jnp.allclose(jnp.mean(r * r), 1 / 9, atol=2e-06)`.
assert jnp.allclose(jnp.mean(r * r), 1 / 9, atol=2e-06)
# Convert `np_w` to a host NumPy array for inspection or verification.
np_w = np.linalg.lstsq(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), rcond=None)[0]
# Assert that `np.allclose(np.asarray(w), np_w, atol=2e-06)`.
assert np.allclose(np.asarray(w), np_w, atol=2e-06)
# Print the observed values to compare against the expected result.
print('weights / rank / MSE:', w, rank, jnp.mean(r * r))
```

Expected: Weights near $(2/3,5/3)$, rank $2$, mean squared error near $1/9$.

## Least squares balances unavoidable residuals

**Predict:** Can every target be matched at the same time?

![Least squares balances unavoidable residuals](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

For each observation, compare the target bar with the least-squares prediction beside it. Targets are $(1,2,2)$; predictions are approximately $(0.667,1.667,2.333)$. The fitted model falls short on the first two rows and exceeds the target on the third.

Each vertical gap has magnitude $1/3$, but the signs differ. Using prediction minus target, the residual vector is $(-1/3,-1/3,1/3)$. The mean squared error is therefore $1/9$, despite no bar pair matching exactly.

### Connect it to the computation

The feature matrix makes the third prediction equal the sum of the first two. Matching the first two targets exactly would force the third prediction to be $3$, not its target $2$. An exact fit is impossible with these features.

Least squares balances this conflict. For each feature column, the residual contributions cancel, giving $X^{\mathsf T}r=0$. That orthogonality, rather than zero residual on every row, is the optimality condition. The unequal bar heights are an unavoidable limitation of this model, not evidence that the solver failed.

```python
# Compute figure data for: Least squares balances unavoidable residuals
# Perform matrix contraction / projection to compute `visual_data`.
visual_data = {'kind': 'bar', 'labels': ['row 0', 'row 1', 'row 2'], 'ylabel': 'target / prediction', 'series': [{'label': 'target', 'y': y.tolist()}, {'label': 'least-squares fit', 'y': (X @ w).tolist()}]}
```

## Recorded reference execution

CPU run: 2026-10-08T14:01:51.477308+00:00. JAX 0.9.2.

```text
weights / rank / MSE: [0.6666669 1.6666665] 2 0.11111113
weights / rank / MSE: [0.6666669 1.6666665] 2 0.11111113
condition / target change / weight change: 200.00504 0.0010000002 0.14142144
PASS: optimization-06

```

## Duplicate a feature

**Predict before running:** For identical columns, can we tell whether a coefficient belongs to the first or second column?

```python
# Experiment — Duplicate a feature: Fit quality and coefficient uniqueness are different questions.
# Construct `a` via `jnp.array([1.0, 2.0, 3.0])`
a = jnp.array([1.0, 2.0, 3.0])
# Combine or mask array elements to form `duplicate`.
duplicate = jnp.stack([a, a], axis=1)
# Run `jnp.linalg.lstsq` to compute `(w_dup, _, rank_dup, _)`.
w_dup, _, rank_dup, _ = jnp.linalg.lstsq(duplicate, 3 * a, rcond=None)
# Assert invariant `int(rank_dup) == 1` holds
assert int(rank_dup) == 1
# Assert that `jnp.allclose(duplicate @ w_dup, 3 * a, atol=3e-06)`.
assert jnp.allclose(duplicate @ w_dup, 3 * a, atol=3e-06)
# Assert that `jnp.allclose(duplicate @ jnp.array([1.0, 2.0]), duplicate @ jnp.array([0.0, 3.0]))`.
assert jnp.allclose(duplicate @ jnp.array([1.0, 2.0]), duplicate @ jnp.array([0.0, 3.0]))
```

**Expected:** Rank $1$; different weight pairs make identical predictions.

Fit quality and coefficient uniqueness are different questions.

## Perturb a weak direction

**Predict before running:** If two columns differ by only $0.01$ in one row, can a target change of $0.001$ move a weight by much more?

```python
# Experiment — Perturb a weak direction: Nearly dependent columns can amplify target perturbations while...
# Construct `near` via `jnp.array([[1.0, 1.0], [0.0, 0.01], [0.0, 0.0]])`
near = jnp.array([[1.0, 1.0], [0.0, 0.01], [0.0, 0.0]])
# Construct `base` via `near @ jnp.array([1.0, 1.0])`
base = near @ jnp.array([1.0, 1.0])
# Construct `changed` via `base + jnp.array([0.0, 0.001, 0.0])`
changed = base + jnp.array([0.0, 0.001, 0.0])
# Run `jnp.linalg.lstsq` to compute `w0`.
w0 = jnp.linalg.lstsq(near, base, rcond=None)[0]
# Run `jnp.linalg.lstsq` to compute `w1`.
w1 = jnp.linalg.lstsq(near, changed, rcond=None)[0]
# Assert that `jnp.allclose(w1 - w0, jnp.array([-0.1, 0.1]), atol=5e-05)`.
assert jnp.allclose(w1 - w0, jnp.array([-0.1, 0.1]), atol=5e-05)
# Print the observed values to compare against the expected result.
print('condition / target change / weight change:', jnp.linalg.cond(near), jnp.linalg.norm(changed - base), jnp.linalg.norm(w1 - w0))
```

**Expected:** The target change has length $0.001$, but the weight change has length about $0.1414$.

Nearly dependent columns can amplify target perturbations while preserving a close prediction fit.

## Make it yours

Add a constant bias column to inputs $(-1,0,1)$ and fit targets $(-1,1,3)$. Recover slope $2$ and bias $1$.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Construct `x` via `jnp.array([-1.0, 0.0, 1.0])`
2. Construct `design` via `jnp.stack([x, jnp.ones_like(x)], axis=1)`
3. Run `jnp.linalg.lstsq` to compute `fit`.
4. Assert that `jnp.allclose(fit, jnp.array([2.0, 1.0]), atol=2e-06)`.

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Add a constant bias column to inputs (-1,0,1) and fit targets (-1,1,3).
# Construct `x` via `jnp.array([-1.0, 0.0, 1.0])`
x = jnp.array(...)  # TODO: compute x
# Construct `design` via `jnp.stack([x, jnp.ones_like(x)], axis=1)`
design = jnp.stack(...)  # TODO: compute design
# Run `jnp.linalg.lstsq` to compute `fit`.
fit = jnp.linalg.lstsq(...)  # TODO: compute fit
# Assert that `jnp.allclose(fit, jnp.array([2.0, 1.0]), atol=2e-06)`.
assert jnp.allclose(fit, jnp.array([2.0, 1.0]), atol=2e-06)  # TODO: complete assertion check
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Add a constant bias column to inputs (-1,0,1) and fit targets (-1,1,3).
# Construct `x` via `jnp.array([-1.0, 0.0, 1.0])`
x = jnp.array([-1.0, 0.0, 1.0])
# Construct `design` via `jnp.stack([x, jnp.ones_like(x)], axis=1)`
design = jnp.stack([x, jnp.ones_like(x)], axis=1)
# Run `jnp.linalg.lstsq` to compute `fit`.
fit = jnp.linalg.lstsq(design, 2 * x + 1, rcond=None)[0]
# Assert that `jnp.allclose(fit, jnp.array([2.0, 1.0]), atol=2e-06)`.
assert jnp.allclose(fit, jnp.array([2.0, 1.0]), atol=2e-06)
```

</details>

## Change feature units

**Transfer / diagnosis**

Multiply the first feature of $X$ by $100$. Predict the new weights and compare predictions.

<details><summary>Hint</summary>

The first weight must divide by $100$ to represent the same model.

</details>

### How to write: Change feature units — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Construct `scaled` via `X * jnp.array([100.0, 1.0])`
2. Run `jnp.linalg.lstsq` to compute `ws`.
3. Assert that `jnp.allclose(ws, expected / jnp.array([100.0, 1.0]), atol=3e-06)`.
4. Assert that `jnp.allclose(scaled @ ws, X @ w, atol=3e-06)`.

**Starter code scaffold (fill in the TODOs):**

```python
# Change feature units (Transfer / diagnosis): Feature units affect coefficients and conditioning.
# Construct `scaled` via `X * jnp.array([100.0, 1.0])`
scaled = ...  # TODO: compute scaled
# Run `jnp.linalg.lstsq` to compute `ws`.
ws = jnp.linalg.lstsq(...)  # TODO: compute ws
# Assert that `jnp.allclose(ws, expected / jnp.array([100.0, 1.0]), atol=3e-06)`.
assert jnp.allclose(ws, expected / jnp.array([100.0, 1.0]), atol=3e-06)  # TODO: complete assertion check
# Assert that `jnp.allclose(scaled @ ws, X @ w, atol=3e-06)`.
assert jnp.allclose(scaled @ ws, X @ w, atol=3e-06)  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Change feature units (Transfer / diagnosis): Feature units affect coefficients and conditioning.
# Construct `scaled` via `X * jnp.array([100.0, 1.0])`
scaled = X * jnp.array([100.0, 1.0])
# Run `jnp.linalg.lstsq` to compute `ws`.
ws = jnp.linalg.lstsq(scaled, y, rcond=None)[0]
# Assert that `jnp.allclose(ws, expected / jnp.array([100.0, 1.0]), atol=3e-06)`.
assert jnp.allclose(ws, expected / jnp.array([100.0, 1.0]), atol=3e-06)
# Assert that `jnp.allclose(scaled @ ws, X @ w, atol=3e-06)`.
assert jnp.allclose(scaled @ ws, X @ w, atol=3e-06)
```

Feature units affect coefficients and conditioning. They need not change the fitted prediction space.

</details>

## Explain the squared conditioning

**Transfer / diagnosis**

Compare the condition numbers of the diagonal matrix with entries $(1,0.1)$ and its Gram matrix. Why is a solver preferable to forming the inverse?

<details><summary>Hint</summary>

The diagonal entries are singular values here; squaring them changes the ratio from $10$ to $100$.

</details>

### How to write: Explain the squared conditioning — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Construct `D` via `jnp.diag(jnp.array([1.0, 0.1]))`
2. Assert that `jnp.allclose(jnp.linalg.cond(D), 10.0)`.
3. Assert that `jnp.allclose(jnp.linalg.cond(D.T @ D), 100.0, rtol=2e-06)`.

**Starter code scaffold (fill in the TODOs):**

```python
# Explain the squared conditioning (Transfer / diagnosis): The normal equations can worsen numerical sensitivity.
# Construct `D` via `jnp.diag(jnp.array([1.0, 0.1]))`
D = jnp.diag(...)  # TODO: compute D
# Assert that `jnp.allclose(jnp.linalg.cond(D), 10.0)`.
assert jnp.allclose(jnp.linalg.cond(D), 10.0)  # TODO: complete assertion check
# Assert that `jnp.allclose(jnp.linalg.cond(D.T @ D), 100.0, rtol=2e-06)`.
assert jnp.allclose(jnp.linalg.cond(D.T @ D), 100.0, rtol=2e-06)  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Explain the squared conditioning (Transfer / diagnosis): The normal equations can worsen numerical sensitivity.
# Construct `D` via `jnp.diag(jnp.array([1.0, 0.1]))`
D = jnp.diag(jnp.array([1.0, 0.1]))
# Assert that `jnp.allclose(jnp.linalg.cond(D), 10.0)`.
assert jnp.allclose(jnp.linalg.cond(D), 10.0)
# Assert that `jnp.allclose(jnp.linalg.cond(D.T @ D), 100.0, rtol=2e-06)`.
assert jnp.allclose(jnp.linalg.cond(D.T @ D), 100.0, rtol=2e-06)
```

The normal equations can worsen numerical sensitivity. A small residual alone is verified separately from coefficient accuracy.

</details>

## Check your understanding

Can zero training error prove that fitted weights are uniquely determined?

1. Yes, zero error is a unique solution
2. No; dependent columns can produce many equally good weights
3. Only if we use float64

<details><summary>Answer and explanation</summary>

No; dependent columns can produce many equally good weights

Independent feature directions determine uniqueness. Extra precision cannot create information missing from the design matrix.

</details>

## Diagnose the result

If a fit changes sharply after a small target perturbation, inspect singular values and feature units. If rank drops, inspect duplicate columns and the cutoff. Compare residuals and predictions as well as coefficients.

## Carry forward

- Fit quality and coefficient uniqueness are different questions.
- Nearly dependent columns can amplify target perturbations while preserving a close prediction fit.

## Keep your evidence

Save the hand solution, residual orthogonality, duplicate-column ambiguity, perturbation table and NumPy comparison.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [JAX least-squares solver](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.lstsq.html)
- [MIT: positive definite matrices and least squares](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-16-projection-matrices-and-least-squares/)

