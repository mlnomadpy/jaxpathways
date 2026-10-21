# Meet your arrays and devices

Phase 00: Setup & first steps · about 45 minutes · CPU

This worked reference is validated on CPU. Accelerator compatibility requires a separate source review and target execution receipt. Use the execution matrix to review device selection and data ownership before adapting a separate target program.

## What you will be able to do

- Read the table before using an API
- Derive a reduction and its output shape
- Distinguish dtype from mathematical value
- Inspect placement without assuming an accelerator

## The problem

Imagine a small table: two observations, with three measurements in each row. If we ask JAX to add the values, should it add across the rows or down the columns? Let’s work out both answers before running any code. You’ll learn to check an array’s shape, number format and device, so a plausible-looking result does not hide a mistake.

## The idea

An array is a collection of values with a shape, a dtype and a device. Shape tells us how to index the values; dtype tells us how they are represented; device tells us where the computation lives. Start by predicting a small reduction before asking JAX to perform it.

## Read one array two ways

Take the two rows $[1,2,3]$ and $[4,5,6]$. Summing across each row gives $[6,15]$: one answer per observation. Summing down the rows gives $[5,7,9]$: one answer per feature. The values are the same; the question determines the axis.

In Python, the row sums use `axis=1`, because the feature axis is reduced. The feature sums use `axis=0`, because the observation axis is reduced. The axis you name is the one you consume, not the one you keep.

Read the heatmap by locating one row and one column before looking at its color. Color helps compare values, but labels establish which observation and feature you are comparing. A device label is a separate property; moving the same values to another device does not change what a row means.

$$
Y_{i,j} = \alpha X_{i,j} + \beta, \qquad X \in \mathbb{R}^{M \times N}
$$

### Pause and reason

If the batch gains a third observation with the same three features, which reduction still returns three values?

<details><summary>Compare your reasoning</summary>

The feature sums still return three values because `axis=0` removes the observation axis. Row sums now return three values too, but for a different reason. This is why unequal axis sizes make debugging easier.

</details>

## Read the table before using an API

Start with the six values $0$ through $5$. Reshape them into two rows: $[0, 1, 2]$ and $[3, 4, 5]$. The first axis selects an observation; the second selects a measurement. Shape $(2, 3)$ describes that organization, not the numerical values. Reshape preserves element count and ordering here; it does not create new measurements.

A length-six array and a two-by-three array contain the same values but invite different operations. In your notebook, write an axis label beside each dimension. This becomes important when a later model expects one prediction per row.

```text
flat [0,1,2,3,4,5] → rows [[0,1,2],[3,4,5]]
axis 0: observations (2)
axis 1: measurements (3)
```

## Derive a reduction and its output shape

Summing along axis $1$ removes the measurement axis and leaves one number per observation. The first row sums to $3$, the second to $12$, so the result has shape $(2,)$. Summing along axis $0$ instead combines observations and leaves three measurement totals: $[3, 5, 7]$.

A reduction can also keep the reduced axis with `keepdims=True.` Column means then have shape $(1, 3)$, which states explicitly that the same three means are subtracted from every row. Derive the numbers and shape before using sum or mean.

```text
(2,3) --sum(axis=1)--> (2,) : [3,12]
(2,3) --sum(axis=0)--> (3,) : [3,5,7]
```

## Distinguish dtype from mathematical value

Dtype describes storage and arithmetic, not whether a value happens to be an integer. The value $3$ can be represented as int32 or float32. This lesson asks for float32 explicitly so later differentiation uses an inexact input. Do not infer precision from the way Python prints a scalar.

Float32 cannot represent every real number, or even every large integer. An exact-looking printed value is not a guarantee of exact arithmetic. Keep dtype in an experiment record and compare with tolerances when arithmetic is approximate.

## Inspect placement without assuming an accelerator

`jax.devices()` lists available devices; `x.devices()` describes the placement of this array. A CPU result is useful evidence of CPU execution. It is separate from that a TPU runtime is installed or that a model scales across devices.

Creating a NumPy copy brings values into host memory for inspection. Converting a scalar with float also requires a concrete value. Those are useful boundaries for tests and reports; repeated transfers inside a training loop can distort performance. Use device metadata to inspect placement and an explicit synchronization point for timing later.

## Prepare the inputs

Create main.py in your lesson workspace. Add this first block; use the environment from setup.

```python
# Step 1 — Prepare the inputs
import jax
import jax.numpy as jnp
```

These explicit inputs define the case that the later checks will verify.

## Build the computation

Append this block below the inputs in the same file.

```python
# Step 2 — Build the computation: x has two observations and three features.
x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
```

$x$ has two observations and three features. `axis=0` removes the observation dimension, so the means describe features; the centered result retains both axes.

## Run and check the result

Append the checks, save main.py, and run python main.py from this folder using your course environment.

```python
# Step 3 — Run and check the result
print("Shape:", x.shape)
print("Dtype:", x.dtype)
print("Devices:", x.devices())
print("Row sums:", x.sum(axis=1))
# Check tensor shape invariant: `x.shape == (2, 3)`
assert x.shape == (2, 3)
assert jnp.allclose(x.sum(axis=1), jnp.array([3., 12.]))
```

Compare the output to the expected result below before making the exercise change.

## Run the example

```python
# Step 1 — Prepare the inputs
import jax
import jax.numpy as jnp
# Step 2 — Build the computation: x has two observations and three features.
x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
# Step 3 — Run and check the result
print("Shape:", x.shape)
print("Dtype:", x.dtype)
print("Devices:", x.devices())
print("Row sums:", x.sum(axis=1))
# Check tensor shape invariant: `x.shape == (2, 3)`
assert x.shape == (2, 3)
assert jnp.allclose(x.sum(axis=1), jnp.array([3., 12.]))
```

Expected: Shape: $(2, 3)$; Dtype: float32; Row sums: [ $3$. $12$.]. Device names vary.

## Rows are groups of observations

**Predict:** Which row should sum to $12$?

![Rows are groups of observations](../../phases/00-welcome/02-meet-your-arrays-and-devices/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis selects a feature column; the vertical axis selects an observation row. Each cell contains one array value. The color bar maps larger numbers to darker purple: the bottom-right cell is darkest because it contains $5$.

Read across the top row to find $(0,1,2)$, then across the bottom row to find $(3,4,5)$. The picture therefore represents an array with shape $(2,3)$: two observations, each with three features.

### Connect it to the computation

Now connect the picture to a reduction. Summing the feature columns within each row gives $0+1+2=3$ and $3+4+5=12$. That is why summing along axis $1$ returns two values. Summing down rows instead would return $(3,5,7)$, one value per feature.

The row labels are positions, not times or class labels. The darker lower row tells us its stored values are larger; it does not imply that the observation is more important.

```python
visual_data = {'kind': 'heatmap', 'values': x.tolist(), 'rows': ['row 0', 'row 1'], 'columns': ['feature 0', 'feature 1', 'feature 2'], 'unit': 'array value'}
```

## Recorded reference execution

CPU run: 2026-10-21T16:03:16.771051+00:00. JAX 0.9.2.

```text
Shape: (2, 3)
Dtype: float32
Devices: {CpuDevice(id=0)}
Row sums: [ 3. 12.]
Shape: (2, 3)
Dtype: float32
Devices: {CpuDevice(id=0)}
Row sums: [ 3. 12.]
Means: [[1.5 2.5 3.5]]
Centered: [[-1.5 -1.5 -1.5]
 [ 1.5  1.5  1.5]]
Host dtype: float32
Expected element-count mismatch
PASS: welcome-02

```

## Center each measurement

**Predict before running:** Predict the three column means and every centered value. Which mean is shared by the two observations?

```python
# Experiment — Center each measurement: The calculation reuses one mean per column.
# Reduce along axis=0 to compute `means`.
means=x.mean(axis=0,keepdims=True)
centered=x-means
print("Means:",means)
print("Centered:",centered)
# Check tensor shape invariant: `means.shape==(1,3)`
assert means.shape==(1,3)
assert jnp.allclose(centered,jnp.array([[-1.5,-1.5,-1.5],[1.5,1.5,1.5]]))
```

**Expected:** Means $[[1.5,2.5,3.5]]$; centered rows are $-1.5$ and $+1.5$.

The calculation reuses one mean per column. An explicit hand calculation verifies both the values and the meaning of the reduction.

## Keep a host reference

**Predict before running:** Will copying to NumPy preserve values and shape? Does that copy describe device-resident computation?

```python
# Experiment — Keep a host reference: The host reference is an inspection copy.
import numpy as np
# Transfer the device result to the host for an independent NumPy comparison.
host=np.asarray(x)
# Check tensor shape invariant: `host.shape==(2,3)`
assert host.shape==(2,3)
np.testing.assert_array_equal(host,np.arange(6).reshape(2,3))
print("Host dtype:",host.dtype)
```

**Expected:** A NumPy array with the same two-by-three values.

The host reference is an inspection copy. Its existence says nothing about accelerator throughput.

## Make it yours

Compute column sums instead. State the resulting shape before running the code.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Inputs:**
- A two-row, three-feature JAX array named `x`.
- The lesson has checked its dtype and device.

**Outputs:**
- Column totals and an explicit output-shape check.

**Invariants to check:**
- Reduce the observation axis and preserve the feature axis.
- Check shape separately from numeric values.

**Hints:**
- Read the dimensions before choosing a reduction axis.
- The result retains the feature dimension.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp

def column_summary(x):
    # TODO: reduce observations and return feature-wise values and shape
    values = ...
    shape = ...
    return values, shape
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Compute column sums instead.
# Reduce along axis=0 to compute `column_sums`.
column_sums = x.sum(axis=0)
# Check tensor shape invariant: `column_sums.shape == (3,)`
assert column_sums.shape == (3,)
assert jnp.allclose(column_sums, jnp.array([3., 5., 7.]))
```

</details>

## Add a new observation

**Practice**

Append $[6,7,8]$. Predict row sums and column means, then check both.

<details><summary>Hint</summary>

Keep observation rows and measurement columns.

</details>

### How to write: Add a new observation — Step-by-step recipe & starter scaffold

**Inputs:**
- The original two-row table and the new observation `[6, 7, 8]`.

**Outputs:**
- The extended table, row totals, and feature means.

**Invariants to check:**
- Appending adds an observation without changing feature count.
- Row and feature reductions preserve opposite axes.

**Hints:**
- Concatenate along the observation axis.
- Write down the table shape before reducing.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp

def extend_and_summarize(table, new_row):
    # TODO: append one observation and compute both summaries
    extended = ...
    row_totals = ...
    feature_means = ...
    return extended, row_totals, feature_means
```

<details><summary>Reference solution and reasoning</summary>

```python
# Add a new observation (Practice)
table=jnp.arange(9,dtype=jnp.float32).reshape(3,3)
assert jnp.allclose(table.sum(axis=1),jnp.array([3.,12.,21.]))
assert jnp.allclose(table.mean(axis=0),jnp.array([3.,4.,5.]))
```

Changing the observation count changes axis $0$, while the measurement count stays three.

</details>

## Diagnose an impossible reshape

**Challenge**

Try reshaping six values to $(2,4)$. Explain the failure and repair the intended three-measurement table.

<details><summary>Hint</summary>

Count required elements before changing the code.

</details>

### How to write: Diagnose an impossible reshape — Step-by-step recipe & starter scaffold

**Inputs:**
- A six-value array and target shape `(2, 4)`.
- The intended table has three measurements per row.

**Outputs:**
- A captured incompatible reshape and a repaired three-feature layout.

**Invariants to check:**
- Reshape preserves the number of elements.
- The repaired table retains every value.

**Hints:**
- Count required elements before reshaping.
- Catch the shape error without discarding data.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp

def diagnose_reshape(values):
    # TODO: attempt the requested shape and capture its incompatibility
    attempted = ...
    repaired = ...
    return attempted, repaired
```

<details><summary>Reference solution and reasoning</summary>

```python
# Diagnose an impossible reshape (Challenge): An incompatible reshape is an element-count error.
try:
    x.reshape(2,4)
except TypeError:
    print("Expected element-count mismatch")
else:
    raise AssertionError("Expected reshape to fail")
assert x.reshape(2,3).size==6
```

An incompatible reshape is an element-count error. Changing a dtype or moving devices cannot fix it. Choose a shape whose product matches six.

</details>

## Check your understanding

For an array of shape $(2, 3)$, what shape results from `sum(axis=0)`?

1. $(2,)$
2. $(3,)$
3. $(2, 3)$

<details><summary>Answer and explanation</summary>

**1. $(2,)$**

This would describe reducing the columns within each row. Axis zero runs down the observations; write the two row vectors and add them element by element.

**2. $(3,)$ (correct)**

Axis $0$ is reduced. The three columns remain, so the result has shape $(3,)$.

**3. $(2, 3)$**

A reduction removes the selected axis unless keepdims is requested. Count the three column sums left after combining the two rows.

</details>

## Diagnose the result

An incompatible reshape is an element-count error. Changing a dtype or moving devices cannot fix it. Choose a shape whose product matches six.

## Carry forward

- The calculation reuses one mean per column. An explicit hand calculation verifies both the values and the meaning of the reduction.
- The host reference is an inspection copy. Its existence says nothing about accelerator throughput.

## Keep your evidence

Keep the labeled table, row/column calculations, centered-data check, array dtype/device report and the failed reshape with its element-count diagnosis.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [JAX arrays](https://docs.jax.dev/en/latest/101/arrays.html)
- [JAX data placement](https://docs.jax.dev/en/latest/201/placement.html)

