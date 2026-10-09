# Shapes, broadcasting, and dtypes

Phase 01: Arrays & pure functions · about 50 minutes · CPU

## What you will be able to do

- Apply the trailing-dimension broadcasting rule before execution.
- Distinguish feature-wise and observation-wise offsets with explicit singleton axes.
- Find a silent pairwise-residual bug using values and shape checks.
- State dtype and precision assumptions for numerical experiments.

## The problem

Your code runs, but has it added the bias to the right axis? This is a common array puzzle: a legal operation can still answer the wrong question. We’ll use small tables to see exactly which values combine, then catch a broadcasting mistake that turns one error per example into a whole matrix of errors.

## The idea

Before adding or subtracting arrays, name what each axis represents. Broadcasting aligns dimensions from the right and reuses singleton dimensions. A legal broadcast can still answer the wrong question, especially when a final mean hides the resulting shape.

## Catch the scalar that hides a shape mistake

Take predictions $[1,3]$ and matching targets $[1,3]$. The intended residuals are $[0,0]$, so mean squared error is zero. If the targets are stored as a column, subtraction instead compares each target against both predictions. The residual matrix has entries $0,2,-2,0$, and its mean squared error is $2$.

Both calculations return a scalar after averaging. Looking only at the final loss, or checking that autodiff can differentiate it, will not expose the problem. Inspect the residual shape before reducing it and compare against a known aligned example.

This is also why a reshape should express a data contract. A singleton feature axis can mean one offset per observation; a singleton observation axis can mean one offset per feature. They are different operations even when an unlucky square fixture makes both expressions run.

$$
(A + b)_{i,j} = A_{i,j} + b_j, \qquad A \in \mathbb{R}^{M \times N},\; b \in \mathbb{R}^{1 \times N}
$$

### Align axes before reducing

**Predict:** Why is checking that the loss is scalar insufficient?

![Align axes before reducing](../outputs/mechanism.svg)

*Architecture and dataflow mechanism diagram.*

Feature bias reuses a vector across observations; row offsets reuse each scalar across features. The final row creates all-pairs residuals. A mean would hide the unwanted $(n,n)$ intermediate. Each row describes a different shape contract.

### Pause and reason

Why is checking that the loss is scalar insufficient?

<details><summary>Compare your reasoning</summary>

A mean can reduce an incorrectly broadcast matrix to a scalar. Check prediction/target alignment and an independently known loss before reduction, not only the final result's rank.

</details>

## Work the shape rule one dimension at a time

Align $(2, 3)$ and $(3,)$ as $(2, 3)$ and $(1, 3)$. The trailing $3$ matches, and the leading $1$ can broadcast to $2$. By contrast, $(2,)$ aligns as $(1, 2)$: trailing $2$ conflicts with $3$. That expression cannot add one scalar per row without another axis.

Represent row offsets as $(2, 1)$. The leading $2$ matches observations and the trailing $1$ broadcasts over the three features. A singleton axis is a statement about how data should be reused, not a random reshape to silence an exception.

```text
feature bias: (2,3) + (1,3) → (2,3)
row offset:  (2,3) + (2,1) → (2,3)
wrong row offset: (2,3) + (2,) → incompatible
```

## Do not test axes with only square arrays

If a batch happens to have $B=D$, a vector of length $B$ can accidentally fit as a feature vector. The expression succeeds while doing the wrong calculation. A test with two observations and three features forces those meanings apart.

Use distinguishable values, such as feature bias $[10, 20, 30]$ and row offsets $[100, 200]$. Then inspect at least two rows and columns. Shape checks alone do not detect every semantically wrong operation; the values need a known reference too.

## A legal broadcast can change your loss

Predictions with shape $(B,)$ and targets with shape $(B, 1)$ broadcast to $(B, B)$. Each row then compares one target to every prediction. A mean over that matrix is a scalar, so a later gradient transform can accept it even though the objective is wrong.

For predictions $[1, 3, 5]$ and matching targets, the intended residual vector is all zero. A column target produces rows $[0, 2, 4]$, $[-2, 0, 2]$, $[-4, -2, 0]$, and a positive mean-squared error. Independent expected values expose the bug immediately. Validate prediction/target shape equality at the boundary where aligned examples are required.

```text
(B,) − (B,1) → (B,B)
Intended aligned-example contract: prediction.shape == target.shape
```

## Shape and dtype are separate contracts

Shape says how values are arranged; dtype says how each value is represented. Use explicit float32 inputs for this course experiment so the precision assumption is visible. Default JAX precision and X64 configuration can differ from expectations brought from a NumPy workflow.

A correctly shaped float32 calculation can still lose a small change at large magnitude. That is numerical representation, not an axis error. Record both shape and dtype in debugging output, and choose comparisons appropriate to their scale. Enabling a different dtype is not a substitute for correcting a broadcast.

## Make boundary checks before reducing

A reduction often hides shape information. Before averaging residuals, check whether the array is the expected $(B,)$ vector or an unintended matrix. Likewise, check that feature statistics have one entry per feature and that row offsets have the observation dimension.

These checks belong at meaningful boundaries, rather than after every elementary addition. A descriptive failure before a loss is calculated is more useful than a plausible scalar whose semantics have already been lost.

## Step 1: Set up imports and input tensors

Import the required JAX modules and define the initial inputs for shapes, broadcasting, and dtypes.

```python
import jax.numpy as jnp
# Construct `batch` via `jnp.array([[1., 2., 3.], [4., 5., 6.]], dtype=jnp.fl...`
batch = jnp.array([[1., 2., 3.], [4., 5., 6.]], dtype=jnp.float32)
```

Establishing explicit input shapes and dtypes first makes the downstream transformation contract deterministic.

## Step 2: Apply the core JAX transformation

Write the core computation and transformation step over the initialized inputs.

```python
bias = jnp.array([10., 20., 30.], dtype=jnp.float32)
# Compute `y` from `batch + bias`
y = batch + bias
```

This stage executes the primary numerical transformation and binds the intermediate outputs.

## Step 3: Verify shapes and numerical invariants

Check that the resulting arrays satisfy the expected shape, dtype, and numerical tolerances.

```python
assert y.shape == (2, 3)
# Assert that `jnp.allclose(y[1], jnp.array([14., 25., 36.]))`.
assert jnp.allclose(y[1], jnp.array([14., 25., 36.]))
```

These assertions lock in the exact numerical contract before you run the full experiment and variations.

## Run the example

```python
# Shapes, broadcasting, and dtypes: Before adding or subtracting arrays, name what each axis represents.
# Import jax.numpy for this computation.
import jax.numpy as jnp
# Construct `batch` via `jnp.array([[1., 2., 3.], [4., 5., 6.]], dtype=jnp.fl...`
batch = jnp.array([[1., 2., 3.], [4., 5., 6.]], dtype=jnp.float32)
# Construct `bias` via `jnp.array([10., 20., 30.], dtype=jnp.float32)`
bias = jnp.array([10., 20., 30.], dtype=jnp.float32)
# Compute `y` from `batch + bias`
y = batch + bias
# Print the observed values to compare against the expected result.
print(y)
# Print diagnostic summary of the computed outputs.
print("Shape:", y.shape, "dtype:", y.dtype)
# Check tensor shape invariant: `y.shape == (2, 3)`
assert y.shape == (2, 3)
# Assert that `jnp.allclose(y[1], jnp.array([14., 25., 36.]))`.
assert jnp.allclose(y[1], jnp.array([14., 25., 36.]))
```

Expected: Rows: $[11., 22., 33.]$ and $[14., 25., 36.]$; shape $(2, 3)$, dtype float32.

## Broadcasting repeats the bias across rows

**Predict:** Why are both rows of the difference equal to $(10,20,30)$?

![Broadcasting repeats the bias across rows](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

This heatmap shows the added amount, computed as the output minus the input batch. It does not show the final output array. Rows are observations and columns are features; the cell values and color bar measure added bias.

Both rows read $(10,20,30)$. The repeated vertical bands show that feature $0$ receives $10$ in every observation, feature $1$ receives $20$, and feature $2$ receives $30$.

### Connect it to the computation

The bias has shape $(3,)$, which aligns with the last axis of the $(2,3)$ batch. Broadcasting applies the same feature-wise addition to both rows. For example, the input $(1,2,3)$ becomes $(11,22,33)$, while $(4,5,6)$ becomes $(14,25,36)$.

Identical rows in this picture mean identical additions, not identical predictions. If you intended a different offset for each observation, you would need a row-wise bias with a compatible shape, such as $(2,1)$; this figure would then have horizontal bands.

```python
# Compute figure data for: Broadcasting repeats the bias across rows
# Compute `visual_data` from `{'kind': 'heatmap', 'values': (y - batch).tolist(), ...`
visual_data = {'kind': 'heatmap', 'values': (y - batch).tolist(), 'rows': ['observation 0', 'observation 1'], 'columns': ['feature 0', 'feature 1', 'feature 2'], 'unit': 'added bias'}
```

## Recorded reference execution

CPU run: 2026-10-09T14:07:35.210470+00:00. JAX 0.9.2.

```text
[[11. 22. 33.]
 [14. 25. 36.]]
Shape: (2, 3) dtype: float32
aligned residuals: [0. 0. 0.]
pairwise residuals: [[ 0.  2.  4.]
 [-2.  0.  2.]
 [-4. -2.  0.]]
float32 large + 1 − large: 0.0
PASS: arrays-02

```

## Make the pairwise bug visible

**Predict before running:** Matching predictions and targets should have zero loss. Predict the matrix created by a column target and its mean-square value.

```python
# Experiment — Make the pairwise bug visible: The final result is scalar in both cases.
# Construct `predictions` via `jnp.array([1., 3., 5.])`
predictions = jnp.array([1., 3., 5.])
# Construct `targets` via `jnp.array([1., 3., 5.])`
targets = jnp.array([1., 3., 5.])
# Compute `correct_residuals` from `predictions - targets`
correct_residuals = predictions - targets
# Compute `pairwise_residuals` from `predictions - targets[:, None]`
pairwise_residuals = predictions - targets[:, None]
# Print the observed values to compare against the expected result.
print("aligned residuals:", correct_residuals)
# Print diagnostic summary of the computed outputs.
print("pairwise residuals:", pairwise_residuals)
# Check tensor shape invariant: `correct_residuals.shape == (3,)`
assert correct_residuals.shape == (3,)
# Check tensor shape invariant: `pairwise_residuals.shape == (3, 3)`
assert pairwise_residuals.shape == (3, 3)
# Assert that `jnp.allclose(jnp.mean(correct_residuals ** 2), 0.)`.
assert jnp.allclose(jnp.mean(correct_residuals ** 2), 0.)
# Assert that `jnp.allclose(jnp.mean(pairwise_residuals ** 2), 16. / 3.)`.
assert jnp.allclose(jnp.mean(pairwise_residuals ** 2), 16. / 3.)
```

**Expected:** The incorrect broadcast produces shape $(3, 3)$ and loss $16/3$, despite perfect aligned predictions.

The final result is scalar in both cases. Checking only the final loss shape would miss this objective bug.

## Observe a precision limit separately

**Predict before running:** Does float32 preserve adding $1$ to $100$,$000$,$000$? Is the shape relevant to that result?

```python
# Experiment — Observe a precision limit separately: This scalar example isolates precision from broadcasting.
# Construct `large` via `jnp.array(100_000_000., dtype=jnp.float32)`
large = jnp.array(100_000_000., dtype=jnp.float32)
# Print the observed values to compare against the expected result.
print("float32 large + 1 − large:", float((large + 1.) - large))
# Assert invariant `float((large + 1.) - large) == 0.` holds
assert float((large + 1.) - large) == 0.
# Check tensor shape invariant: `large.shape == ()`
assert large.shape == ()
```

**Expected:** The recorded difference is zero because the representable spacing at this magnitude exceeds $1$.

This scalar example isolates precision from broadcasting. Autodiff and compilation do not make every arithmetic operation exact.

## Make it yours

Add a different scalar offset to each row using offsets $[100., 200.]$. Make its shape explicit.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Construct `offsets` via `jnp.array([100., 200.])[:, None]`
2. Compute `z` from `batch + offsets`
3. Check tensor shape invariant: `offsets.shape == (2, 1)`
4. Assert that `jnp.allclose(z, jnp.array([[101.,102.,103.],[204.,205.,206.]]))`.

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Add a different scalar offset to each row using offsets [100., 200.].
# Construct `offsets` via `jnp.array([100., 200.])[:, None]`
offsets = jnp.array(...)  # TODO: compute offsets
# Compute `z` from `batch + offsets`
z = ...  # TODO: compute z
# Check tensor shape invariant: `offsets.shape == (2, 1)`
assert offsets.shape  # TODO: complete assertion check
# Assert that `jnp.allclose(z, jnp.array([[101.,102.,103.],[204.,205.,206.]]))`.
assert jnp.allclose(z, jnp.array([[101.,102.,103.],[204.,205.,206.]]))  # TODO: complete assertion check
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Add a different scalar offset to each row using offsets [100., 200.].
# Construct `offsets` via `jnp.array([100., 200.])[:, None]`
offsets = jnp.array([100., 200.])[:, None]
# Compute `z` from `batch + offsets`
z = batch + offsets
# Check tensor shape invariant: `offsets.shape == (2, 1)`
assert offsets.shape == (2, 1)
# Assert that `jnp.allclose(z, jnp.array([[101.,102.,103.],[204.,205.,206.]]))`.
assert jnp.allclose(z, jnp.array([[101.,102.,103.],[204.,205.,206.]]))
```

</details>

## Use both kinds of offsets together

**Practice**

Add the feature bias and row offsets to the original batch. Derive every row before running, then check with an explicitly constructed array.

<details><summary>Hint</summary>

Use bias shape $(3,)$ and offsets shape $(2, 1)$. Do not swap their semantic roles.

</details>

### How to write: Use both kinds of offsets together — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Construct `combined` via `batch + bias + jnp.array([100., 200.])[:, None]`
2. Check tensor shape invariant: `combined.shape == (2, 3)`
3. Assert that `jnp.allclose(combined, jnp.array([[111., 122., 133.], [214., 225., 236.]]))`.

**Starter code scaffold (fill in the TODOs):**

```python
# Use both kinds of offsets together (Practice): The first row receives 100 in every feature and the second...
# Construct `combined` via `batch + bias + jnp.array([100., 200.])[:, None]`
combined = ...  # TODO: compute combined
# Check tensor shape invariant: `combined.shape == (2, 3)`
assert combined.shape  # TODO: complete assertion check
# Assert that `jnp.allclose(combined, jnp.array([[111., 122., 133.], [214., 225., 236.]]))`.
assert jnp.allclose(combined, jnp.array([[111., 122., 133.], [214., 225., 236.]]))  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Use both kinds of offsets together (Practice): The first row receives 100 in every feature and the second...
# Construct `combined` via `batch + bias + jnp.array([100., 200.])[:, None]`
combined = batch + bias + jnp.array([100., 200.])[:, None]
# Check tensor shape invariant: `combined.shape == (2, 3)`
assert combined.shape == (2, 3)
# Assert that `jnp.allclose(combined, jnp.array([[111., 122., 133.], [214., 225., 236.]]))`.
assert jnp.allclose(combined, jnp.array([[111., 122., 133.], [214., 225., 236.]]))
```

The first row receives $100$ in every feature and the second receives $200$, while the feature bias differs by column.

</details>

## Reject an aligned-loss contract violation

**Challenge**

Write an MSE function that requires prediction and target shapes to match. Test zero loss for matching vectors and rejection of a column target. Explain why an automatic reshape might hide a data-ordering problem.

<details><summary>Hint</summary>

Raise a descriptive ValueError before subtracting. The check is based on shape metadata, not array truth values.

</details>

### How to write: Reject an aligned-loss contract violation — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `x.sum(axis=...) / x.mean(axis=..., keepdims=...)` — Reduces values along the named `axis` (the axis you name is collapsed unless `keepdims=True`).
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Guard input contract (`prediction.shape != target.shape`) and fail fast if violated.
2. Return `jnp.mean((prediction - target) ** 2)` to the caller.
3. Assert that `jnp.allclose(aligned_mse(predictions, targets), 0.)`.
4. Run the boundary check and catch the expected exception:

**Starter code scaffold (fill in the TODOs):**

```python
# Reject an aligned-loss contract violation (Challenge): Fix shapes at the data/model interface according to the...
def aligned_mse(prediction, target):
    # Guard input contract (`prediction.shape != target.shape`) and fail fast if violated.
    if prediction.shape != target.shape:
        raise ValueError("aligned prediction and target shapes must match")
    # Return `jnp.mean((prediction - target) ** 2)` to the caller.
    return ...  # TODO: return computed result
# Assert that `jnp.allclose(aligned_mse(predictions, targets), 0.)`.
assert jnp.allclose(aligned_mse(predictions, targets), 0.)  # TODO: complete assertion check
# Run the boundary check and catch the expected exception:
try:
    aligned_mse(predictions, targets[:, None])
except ValueError:
    pass
else:
    raise AssertionError("Column targets must be rejected")
```

<details><summary>Reference solution and reasoning</summary>

```python
# Reject an aligned-loss contract violation (Challenge): Fix shapes at the data/model interface according to the...
def aligned_mse(prediction, target):
    # Guard input contract (`prediction.shape != target.shape`) and fail fast if violated.
    if prediction.shape != target.shape:
        raise ValueError("aligned prediction and target shapes must match")
    # Return `jnp.mean((prediction - target) ** 2)` to the caller.
    return jnp.mean((prediction - target) ** 2)
# Assert that `jnp.allclose(aligned_mse(predictions, targets), 0.)`.
assert jnp.allclose(aligned_mse(predictions, targets), 0.)
# Run the boundary check and catch the expected exception:
try:
    aligned_mse(predictions, targets[:, None])
except ValueError:
    pass
else:
    raise AssertionError("Column targets must be rejected")
```

Fix shapes at the data/model interface according to the intended example order. Blindly reshaping inside a loss can conceal mismatched semantics.

</details>

## Check your understanding

Which offset shape adds one scalar to each row of a $(2, 3)$ batch?

1. $(2,)$
2. $(2, 1)$
3. $(3, 2)$

<details><summary>Answer and explanation</summary>

$(2, 1)$

$(2, 1)$ broadcasts across the three columns. Shape $(2,)$ conflicts with the trailing dimension $3$.

</details>

## Diagnose the result

When broadcasting fails, align trailing dimensions on paper. When it succeeds with a surprising loss, inspect the residual shape before reduction and compare with hand-derived values. When a small change disappears, inspect dtype and magnitude. Use asymmetric shape tests so accidentally equal dimension sizes do not conceal a wrong axis.

## Carry forward

- Broadcasting compatibility does not imply semantic correctness.
- A singleton axis declares reuse along a particular dimension.
- Prediction/target mismatch can silently create all-pairs residuals.
- Check dtype separately from shape and record numerical scale.

## Keep your evidence

Keep an axis sketch, asymmetric offset checks, aligned-versus-pairwise residual arrays, a rejected shape mismatch, and the float32 precision observation. Explain the intended residual contract.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [NumPy: broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [JAX: default dtypes and X64](https://docs.jax.dev/en/latest/default_dtypes.html)

