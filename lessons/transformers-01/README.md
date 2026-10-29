# Attention from small pieces

Phase 07: Transformers & language models · about 90 minutes · CPU

This worked reference is validated on CPU. Accelerator compatibility requires a separate source review and target execution receipt. Use the execution matrix to review device selection and data ownership before adapting a separate target program.

## What you will be able to do

- Track query, key and value axes
- Derive the uniform-query result
- Scale scores and keep softmax stable
- Test relationships that should survive a change

## The problem

When a model reads a sequence, how can one position use information from another? We’ll build a single attention head as a weighted average. Start with a query whose answer we know, then change the query and watch which values it emphasizes. Small arrays let us see every part of the calculation.

## The idea

Attention uses queries and keys to determine weights, then combines values using those weights. Scores describe compatibility; values provide the content being mixed. The attention matrix is an intermediate computation rather than the model's complete output.

## Calculate one weighted value mixture

Attention answers two linked questions: which keys should this query use, and what content do their values contribute? Start with weights we can inspect directly: $0.25(2,0)+0.75(0,4)=(0.5,3)$. The weights sum to one, but the output coordinates need not.

On a heatmap, follow one query row across the eligible keys, then apply those weights to the corresponding value rows. If you change a value while holding weights fixed, the mixture changes but the heatmap does not. Predict the mixture before reading the computed output.

### Pause and reason

Can an attention-weight heatmap alone reconstruct the output?

<details><summary>Compare your reasoning</summary>

No. You also need the value vectors and any output projection. Weights explain the mixing coefficients, not the content being mixed.

</details>

## Track query, key and value axes

The shapes explain that recipe. With $Q\in\mathbb{R}^{2\times2}$ and $K\in\mathbb{R}^{3\times2}$, $QK^\mathsf{T}$ has shape $(2,3)$: one row per query and one column per key. Softmax over the last axis gives each query its own distribution across keys. Multiplying by three value rows produces two output rows; the value width $D_v$ can differ from key width $D_k$.

A rectangular example exposes a transposed-axis mistake that square shapes can hide. In the code, check each intermediate shape and ask: is this reduction choosing among keys for each query?

```text
Q (Tq, Dk) @ Kᵀ (Dk, Tk) → scores (Tq, Tk)
softmax over Tk → weights (Tq, Tk)
weights @ V (Tk, Dv) → output (Tq, Dv)
```

## Derive the uniform-query result

A zero query has dot product zero with every key, so softmax receives $(0,0,0)$ and returns $(1/3,1/3,1/3)$. The three values sum to $(6,6)$, so their mean is $(2,2)$; the recorded run prints that output for both queries. This gives a known answer for checking both normalization and value mixing.

Keep this special case separate from the nonuniform permutation exercise: uniform weights make every value-only permutation preserve the mean. The exercise declares a nonzero query and a fixture where changing value order changes its output. A heatmap row is a weighted average, not an index selecting one value.

## Scale scores and keep softmax stable

As feature width grows, dot products can grow in scale under common initialization assumptions. Dividing by $\sqrt{D_k}$ moderates that scale; it does not prevent every possible input from producing saturated weights. The independent NumPy reference subtracts each row maximum before exponentiating, while JAX softmax uses a stable normalization.

Adding the same finite constant to every score in one row leaves its softmax unchanged. Use that invariance as a check, then inspect unusually concentrated weights in context rather than treating scaling as a guarantee.

## Test relationships that should survive a change

Use invariants to check that the computation still means the same thing after a controlled edit. Adding a shared vector $c$ to every value should add $c$ to each output because each weight row sums to one. Permuting keys with their paired values preserves the same weighted terms; permuting values alone usually changes which content each score selects.

The recorded nonuniform fixture prints original and paired outputs as $(2.67485,1.32515)$, while the value-only result is $(2,2.67485)$. The separate zero-query example still stays at its mean under any value permutation. Use the exercise's explicit fixture to distinguish the general behavior from that uniform exception. Since this head is unmasked, it can read future positions and is not yet a next-token model.

## Read attention as matching, then mixing

The compact equation assigns a job to each stage: $Q$ and $K$ form compatibility scores, row-wise softmax turns them into weights $A$, and $AV$ mixes the content. The score divisor uses key width $d_k$; each row of $A$ is nonnegative and sums to one.

For the zero-query example, all three scores tie, so the value mixture is their mean, $(2,2)$. Check this by hand before moving from the single head to learned projections.

$$
\begin{aligned}S&=\frac{QK^\mathsf{T}}{\sqrt{d_k}}\\A_{ij}&=\frac{\exp(S_{ij}-m_i)}{\sum_t\exp(S_{it}-m_i)},\quad m_i=\max_t S_{it}\\O&=AV\end{aligned}
$$

## From the attention head to a decoder stack

The small Q/K/V exercise isolates one weighted mixture. A decoder adds learned projections, multiple heads, normalization, residual paths, token embeddings and causal blocks. In the optional text foundations, compare the independent NumPy attention row with the model output before interpreting a generated token. Its small CPU pilot is documented in `projects/text-harness/foundations/README.md`; it is a separate workflow from the original four-stage one-block harness.

## 1. Prepare the experiment

In the CPU environment from setup, create main.py and add this block. Write down the dimensions and the known result before continuing.

```python
# Step 1 — 1. Prepare the experiment
import jax
import jax.numpy as jnp
import numpy as np
```

These imports provide JAX array transformations and NumPy reference calculations. Define the numerical functions next; the final build block supplies the concrete fixture and checks.

## 2. Implement the mechanism

Append this block in the same file. Follow the shape diagram above and identify each reduction or state transition.

```python
# Step 2 — 2. Implement the mechanism
def attend(q, k, v):
    scores = q @ k.T / jnp.sqrt(q.shape[-1])
    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ v, weights
```

Multiplying $Q\in\mathbb{R}^{T_q\times D_k}$ by $K^\mathsf{T}\in\mathbb{R}^{D_k\times T_k}$ gives scores of shape $(T_q,T_k)$. Softmax normalizes each query’s weights across the $T_k$ keys. Multiplying by $V\in\mathbb{R}^{T_k\times D_v}$ then gives an output of shape $(T_q,D_v)$.

## 3. Run and check

Append this block and run python main.py. Compare its output with your prediction. An assertion failure means a stated contract needs investigation.

```python
# Step 3 — 3. Run and check: Uniform weights 1/3; both output rows [2, 2].
q = jnp.zeros((2, 2))
k = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
v = jnp.array([[2., 0.], [0., 4.], [4., 2.]])
output, weights = attend(q, k, v)
print("Weights / output:", weights, output)
assert jnp.allclose(weights, jnp.full((2, 3), 1/3))
assert jnp.allclose(output, jnp.array([[2., 2.], [2., 2.]]))
assert jnp.allclose(jnp.sum(weights, axis=-1), 1.)
```

Uniform weights $1/3$; both output rows $[2, 2]$. The separate permutation exercise declares its own nonuniform Q/K/V fixture and does not depend on the later `q_new` experiment.

## Run the example

```python
# Step 1 — 1. Prepare the experiment
import jax
import jax.numpy as jnp
import numpy as np

# Step 2 — 2. Implement the mechanism
def attend(q, k, v):
    scores = q @ k.T / jnp.sqrt(q.shape[-1])
    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ v, weights

# Step 3 — 3. Run and check: Uniform weights 1/3; both output rows [2, 2].
q = jnp.zeros((2, 2))
k = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
v = jnp.array([[2., 0.], [0., 4.], [4., 2.]])
output, weights = attend(q, k, v)
print("Weights / output:", weights, output)
assert jnp.allclose(weights, jnp.full((2, 3), 1/3))
assert jnp.allclose(output, jnp.array([[2., 2.], [2., 2.]]))
assert jnp.allclose(jnp.sum(weights, axis=-1), 1.)
```

Expected: Uniform weights $1/3$; both output rows $[2, 2]$.

## Each query distributes one unit of attention

**Predict:** Why do the initial rows have equal weights?

![Each query distributes one unit of attention](../../phases/07-transformers/01-attention-from-small-pieces/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

Rows are queries, columns are keys, and each cell is the attention weight assigned to that key; the shared color scale runs from $0$ to $1$. Read horizontally because each query's weights normalize across keys. In the zero-query panel every cell is about $0.3333$. In the changed-query panel, query 0 assigns about $(0.472,0.0565,0.472)$, and query 1 assigns $(0.0565,0.472,0.472)$.

### Connect it to the computation

The changed queries $(3,0)$ and $(0,3)$ score keys $(1,0),(0,1),(1,1)$; key 2 ties with the matching coordinate, so each row has two largest weights. The code applies softmax to these scores, then multiplies the resulting row by the value matrix. In the uniform panel that weighted sum is $(2,2)$; the changed panel's weights alone do not tell you its output until you combine them with values. Next, predict how changing one value row affects output without changing these weights.

```python
changed_q = jnp.array([[3.0, 0.0], [0.0, 3.0]])
_, focused = attend(changed_q, k, v)
visual_data = {'kind': 'panels', 'panels': [{'kind': 'heatmap', 'title': 'Zero queries', 'values': weights.tolist(), 'rows': ['query 0', 'query 1'], 'columns': ['key 0', 'key 1', 'key 2'], 'unit': 'attention weight'}, {'kind': 'heatmap', 'title': 'Changed queries', 'values': focused.tolist(), 'rows': ['query 0', 'query 1'], 'columns': ['key 0', 'key 1', 'key 2'], 'unit': 'attention weight'}]}
```

## Recorded reference execution

CPU run: 2026-10-29T13:57:22.821893+00:00. JAX 0.9.2.

```text
Weights / output: [[0.33333334 0.33333334 0.33333334]
 [0.33333334 0.33333334 0.33333334]] [[2. 2.]
 [2. 2.]]
Weights / output: [[0.33333334 0.33333334 0.33333334]
 [0.33333334 0.33333334 0.33333334]] [[2. 2.]
 [2. 2.]]
original / paired / value-only: [[2.6748495 1.3251503]] [[2.6748495 1.3251503]] [[2.        2.6748495]]
zero-query mean: [[2. 2.]]
PASS: transformers-01

```

## Compare a nonzero query with NumPy

**Predict before running:** Which keys become more relevant for query $[2, 0]$?

```python
# Experiment — Compare a nonzero query with NumPy
q_new = jnp.array([[2., 0.]])
actual, w_new = attend(q_new, k, v)
# Transfer the device result to the host for an independent NumPy comparison.
scores_np = np.asarray(q_new) @ np.asarray(k).T / np.sqrt(2)
# Reduce along axis=-1 to compute `raw`.
raw = np.exp(scores_np - scores_np.max(axis=-1, keepdims=True))
# Reduce along axis=-1 to compute `reference_w`.
reference_w = raw / raw.sum(axis=-1, keepdims=True)
np.testing.assert_allclose(w_new, reference_w, rtol=1e-6, atol=1e-6)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(actual, reference_w @ np.asarray(v), rtol=1e-6, atol=1e-6)
```

**Expected:** Keys $0$ and $2$ receive equal larger weights than key $1$.

A separate NumPy calculation checks nonuniform weighting and the output axis.

## Shift every value vector

**Predict before running:** If every $V$ row gains $[5, -2]$, what happens to each output?

```python
# Experiment — Shift every value vector: Row-normalized attention preserves a shared value translation.
shift = jnp.array([5., -2.])
shifted, _ = attend(q_new, k, v + shift)
assert jnp.allclose(shifted, actual + shift)
```

**Expected:** Each output shifts by $[5, -2]$.

Row-normalized attention preserves a shared value translation.

## Make it yours

Start from the explicit independent fixture below: q_ex = [[2, 0]], k_ex = [[1, 0], [0, 1], [1, 1]], and v_ex = [[2, 0], [0, 4], [4, 2]]. Implement `attend` using the declared starter. Show algebraically why permuting keys and their paired values together preserves the weighted sum. Then test order [2, 0, 1] for both the paired permutation and a value-only permutation. Compare each result to the original output with NumPy. State why the separate zero-query worked example cannot show a value-only change.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Inputs:**
- Independent nonuniform fixture: q_ex=[[2,0]], k_ex=[[1,0],[0,1],[1,1]], v_ex=[[2,0],[0,4],[4,2]], and permutation order [2,0,1].

**Outputs:**
- Original attention output, paired-permutation output, and value-only-permutation output compared with an independent NumPy reference.

**Invariants to check:**
- Permuting keys and their paired values together preserves the weighted sum.
- A value-only permutation can change the result for this nonuniform fixture; this is not guaranteed for every query/value set.
- Zero queries produce a uniform-weight mean that is invariant under value permutations.

**Hints:**
- Implement scaled scores, softmax over keys, then weights @ values.
- Write the paired permutation as a reindexing of the same sum.
- Run the full solution using only the fixture declared in this starter.

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs a JAX array from Python/NumPy values.

**Starter code scaffold (fill in the TODOs):**

```python
import jax
import jax.numpy as jnp

def attend(q, k, v):
    """Return attention outputs and normalized key weights."""
    # TODO: compute scaled query-key scores, normalize over keys, and mix values.
    raise NotImplementedError

q_ex = jnp.array([[2., 0.]])
k_ex = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
v_ex = jnp.array([[2., 0.], [0., 4.], [4., 2.]])
order = jnp.array([2, 0, 1])
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: all inputs are declared here, independent of prior experiments.
import jax
import jax.numpy as jnp
import numpy as np

def attend(q, k, v):
    scores = q @ k.T / jnp.sqrt(q.shape[-1])
    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ v, weights

q_ex = jnp.array([[2., 0.]])
k_ex = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
v_ex = jnp.array([[2., 0.], [0., 4.], [4., 2.]])
order = jnp.array([2, 0, 1])
original, weights = attend(q_ex, k_ex, v_ex)
paired, paired_weights = attend(q_ex, k_ex[order], v_ex[order])
value_only, _ = attend(q_ex, k_ex, v_ex[order])
assert jnp.allclose(paired, original)
assert not jnp.allclose(value_only, original)

# Independent NumPy reference for the original and both permutations.
def numpy_attend(q, k, v):
    scores = q @ k.T / np.sqrt(q.shape[-1])
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    w = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    return w @ v
np.testing.assert_allclose(original, numpy_attend(np.asarray(q_ex), np.asarray(k_ex), np.asarray(v_ex)), rtol=1e-6, atol=1e-6)
np.testing.assert_allclose(paired, numpy_attend(np.asarray(q_ex), np.asarray(k_ex[order]), np.asarray(v_ex[order])), rtol=1e-6, atol=1e-6)
np.testing.assert_allclose(value_only, numpy_attend(np.asarray(q_ex), np.asarray(k_ex), np.asarray(v_ex[order])), rtol=1e-6, atol=1e-6)

# For any permutation p, sum_i w_i V_{p(i)} generally changes when w is nonuniform.
# With q=0, all w_i=1/3 and the output is mean(V), invariant under every permutation.
zero_query, _ = attend(jnp.zeros((1, 2)), k_ex, v_ex)
assert jnp.allclose(zero_query, v_ex.mean(axis=0, keepdims=True))
print("original / paired / value-only:", original, paired, value_only)
print("zero-query mean:", zero_query)
```

</details>

## Use a different value dimension

**Transfer / diagnosis**

Make each value a single scalar $[3, 6, 9]$. Check the output shape and zero-query mean.

<details><summary>Hint</summary>

The score matrix does not depend on Dv.

</details>

### How to write: Use a different value dimension — Step-by-step recipe & starter scaffold

**Inputs:**
- Make each value a single scalar $[3, 6, 9]$. Check the output shape and zero-query mean.

**Outputs:**
- One scalar attention output per query and a zero-query mean check.

**Invariants to check:**
- The key/value row pairing remains aligned and the query axis is preserved.

**Hints:**
- The score matrix does not depend on Dv.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp

def practice_use_a_different_value_dimension(values):
    """Implement the use a different value dimension check."""
    # TODO: Construct scalar values with shape (3,1), attend with a zero query, and assert the output shape and arithmetic mean.
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
# Use a different value dimension (Transfer / diagnosis)
scalar_v = jnp.array([[3.], [6.], [9.]])
scalar_out, _ = attend(q, k, scalar_v)
# Check tensor shape invariant: `scalar_out.shape == (2, 1)`
assert scalar_out.shape == (2, 1)
assert jnp.allclose(scalar_out, 6.)
```

Queries and keys share $Dk$; values may have an independent feature dimension.

</details>

## Expose a wrong softmax axis

**Transfer / diagnosis**

Normalize zero scores along axis $0$ and compare row sums with the intended contract. Repair the axis.

<details><summary>Hint</summary>

Two queries and three keys make the incorrect result visible.

</details>

### How to write: Expose a wrong softmax axis — Step-by-step recipe & starter scaffold

**Inputs:**
- Normalize zero scores along axis $0$ and compare row sums with the intended contract. Repair the axis.

**Outputs:**
- Both row-sum diagnostics and a corrected attention-weight matrix.

**Invariants to check:**
- For each query, probabilities sum across key positions.

**Hints:**
- Two queries and three keys make the incorrect result visible.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp

def practice_expose_a_wrong_softmax_axis(scores):
    """Implement the expose a wrong softmax axis check."""
    # TODO: Normalize a zero score matrix across axis zero, inspect column sums, and implement the key-axis normalization.
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
# Expose a wrong softmax axis (Transfer / diagnosis): A legal operation can normalize the wrong axis.
wrong_w = jax.nn.softmax(jnp.zeros((2, 3)), axis=0)
assert jnp.allclose(wrong_w.sum(axis=-1), 1.5)
right_w = jax.nn.softmax(jnp.zeros((2, 3)), axis=-1)
assert jnp.allclose(right_w.sum(axis=-1), 1.)
```

A legal operation can normalize the wrong axis. Rectangular shapes expose the mismatch.

</details>

## Check your understanding

Which dimension should softmax normalize for one query to combine all key/value pairs?

1. The query dimension
2. The key dimension
3. The value feature dimension

<details><summary>Answer and explanation</summary>

**1. The query dimension**

Normalizing over queries changes which probabilities sum to one. For each query, sum its weights over the available keys.

**2. The key dimension (correct)**

One query needs a probability distribution across keys; each row must sum to one.

**3. The value feature dimension**

The value-feature axis belongs to the weighted output representation. Attention probabilities normalize over candidate key positions.

</details>

## Diagnose the result

If weights sum to $1.5$ rather than $1$, inspect the softmax axis on the $(2,3)$ score matrix. If paired key/value permutation changes the output, check that the same order was applied to both arrays. Repair the axis or pairing, then rerun the independent NumPy and translation checks.

## Carry forward

- A separate NumPy calculation checks nonuniform weighting and the output axis.
- Row-normalized attention preserves a shared value translation.

## Keep your evidence

Keep the Q/K/V axis diagram, zero-query hand mean, independent nonuniform fixture and NumPy output, paired permutation invariance and one value-only permutation result. Explain the zero-query exception and repair a wrong softmax axis.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [JAX softmax API](https://docs.jax.dev/en/latest/_autosummary/jax.nn.softmax.html)
- [JAX attention API](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html)
- [Optax cross entropy](https://optax.readthedocs.io/en/latest/api/losses.html)

