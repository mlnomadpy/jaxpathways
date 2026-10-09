# Masking, tokenization, and sequence packing

Phase 07: Transformers & language models · about 90 minutes · CPU

## What you will be able to do

- Build a causal mask from positions
- Keep token IDs and targets aligned
- Combine causal and segment boundaries
- Handle padding without an all-masked softmax

## The problem

An attention head should not always see every token. For example, a next-token model must not peek at the future. We’ll build a causal mask one row at a time, then add padding and boundaries between packed examples. Before training, you’ll be able to explain exactly which positions may read each other.

## The idea

An attention mask controls which positions supply context. A loss mask controls which targets contribute to the objective. Tokenization, target shifting and sequence packing decide the positions to which those different rules apply.

## Keep context eligibility separate from scored targets

At input position $t$, causal next-token training predicts target $t+1$. A response-only role flag must follow that shifted target. Using unshifted role flags can score the wrong token at the prompt/response boundary.

An unscored prompt token can still influence a scored response through attention. Removing its direct loss does not remove it from the computation. A blocked attention edge instead prevents a context contribution.

Packed documents need segment boundaries in addition to a causal triangle. A later segment must not read an unrelated earlier segment merely because its positions occur earlier in the tensor. Trace one query and target across these rules before trusting a batch average.

### Pause and reason

Why is a triangular mask alone insufficient for packed documents?

<details><summary>Compare your reasoning</summary>

It blocks future positions but can allow one document to attend to another earlier document. Segment eligibility must enforce the intended document separation.

</details>

## Build a causal mask from positions

For query position i, allow keys j ≤ i. Including the diagonal lets a token use its own representation. The model predicts the next token from that prefix; the target array is shifted one position ahead. Excluding the diagonal is a different convention that requires changing how inputs and labels are aligned.

With values $2$, $4$, $8$, $16$ and uniform permitted scores, the outputs are prefix means: $2$, $3$, $14/3$ and $7.5$. Changing the last value must not change the first three outputs. This counterfactual tests causal isolation directly.

```text
allowed key columns
query 0: 1 0 0 0
query 1: 1 1 0 0
query 2: 1 1 1 0
query 3: 1 1 1 1
```

## Keep token IDs and targets aligned

For vocabulary {a:$0$, b:$1$, c:$2$}, abcab becomes $[0,1,2,0,1]$. Inputs $[0,1,2,0]$ pair with next-token targets $[1,2,0,1]$. Do not train the last token to predict the first token of an unrelated packed example. A separator or an ignored loss position must express that boundary.

Production tokenizers add Unicode normalization, special control IDs, and subword merges on top of this exact index-alignment contract. Validate unknown characters explicitly rather than silently assigning an existing ID.

## Combine causal and segment boundaries

Packing segments $[0,0,1,1]$ means positions $2$ and $3$ belong to a new example. Require equal segment IDs as well as causal order. With our values, packed means become $2$, $3$, $8$ and $12$. Position $2$ must not read earlier positions from segment $0$ even though they are causally prior.

Segment IDs must identify separate examples within a packed row. Reusing an ID for two unrelated spans could reconnect them. The mask is only one part of packing: target eligibility must also stop at each segment boundary.

## Handle padding without an all-masked softmax

An all-masked row represented by all −infinity scores has no normalized distribution and can produce NaN. Padding queries create precisely that row if you forbid every key. Provide a harmless fallback key for those queries, then replace their outputs with zero and ignore their loss. Every real query still uses its original permitted keys.

The fallback is a numerical convention, not permission for padding to carry semantic information. Test that perturbing padded values does not change real outputs. Also reject any real query with no valid key when that violates your model contract.

## Turn the allowed positions into weights

For causal attention, query position $i$ may use key position $j$ only when $j\le i$. We leave an allowed score unchanged and replace a blocked score with negative infinity before softmax. This gives blocked positions zero weight when the row has at least one allowed key. If a row is entirely blocked, an ordinary softmax has no valid distribution; handle that case explicitly, as the code does.

$$
\widetilde S_{ij}=\begin{cases}S_{ij}&\text{if key }j\text{ is allowed for query }i\\-\infty&\text{otherwise}\end{cases}
$$

## 1. Prepare the experiment

In the CPU environment from setup, create main.py and add this block. Write down the dimensions and the known result before continuing.

```python
# Step 1 — 1. Prepare the experiment: These imports provide JAX array transformations and NumPy...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np
```

These imports provide JAX array transformations and NumPy reference calculations. Define the numerical functions next; the final build block supplies the concrete fixture and checks.

## 2. Implement the mechanism

Append this block in the same file. Follow the shape diagram above and identify each reduction or state transition.

```python
# Step 2 — 2. Implement the mechanism: allowed key columns query 0: 1 0 0 0 query 1: 1 1 0 0 query 2: 1 1...
def masked_mean(values, allowed):
    # Combine or mask array elements to form `scores`.
    scores = jnp.where(allowed, 0., -jnp.inf)
    # Apply nonlinear activation or probability normalization to compute `weights`.
    weights = jax.nn.softmax(scores, axis=-1)
    # Return `(weights @ values, weights)` to the caller.
    return weights @ values, weights

# Function `make_mask(segment_ids, valid)` implementing this stage's computation:
def make_mask(segment_ids, valid):
    # Construct `positions` via `jnp.arange(segment_ids.shape[0])`
    positions = jnp.arange(segment_ids.shape[0])
    # Compute `causal` from `positions[:, None] >= positions[None, :]`
    causal = positions[:, None] >= positions[None, :]
    # Compute `same_segment` from `segment_ids[:, None] == segment_ids[None, :]`
    same_segment = segment_ids[:, None] == segment_ids[None, :]
    # Return `causal & same_segment & valid[:, None] & valid[None, :]` to the caller.
    return causal & same_segment & valid[:, None] & valid[None, :]
```

allowed key columns
query $0$: $1$ $0$ $0$ $0$
query $1$: $1$ $1$ $0$ $0$
query $2$: $1$ $1$ $1$ $0$
query $3$: $1$ $1$ $1$ $1$

## 3. Run and check

Append this block and run python main.py. Compare its output with your prediction. An assertion failure means a stated contract needs investigation.

```python
# Step 3 — 3. Run and check: Causal means [2, 3, 4.666667, 7.5]; no weight above the diagonal.
# Construct `values` via `jnp.array([[2.], [4.], [8.], [16.]])`
values = jnp.array([[2.], [4.], [8.], [16.]])
# Construct `causal` via `jnp.arange(4)[:, None] >= jnp.arange(4)[None, :]`
causal = jnp.arange(4)[:, None] >= jnp.arange(4)[None, :]
# Run `masked_mean` to compute `(out, weights)`.
out, weights = masked_mean(values, causal)
# Assert that `jnp.allclose(out[:, 0], jnp.array([2., 3., 14/3, 7.5]))`.
assert jnp.allclose(out[:, 0], jnp.array([2., 3., 14/3, 7.5]))
# Assert that `jnp.allclose(jnp.triu(weights, 1), 0.)`.
assert jnp.allclose(jnp.triu(weights, 1), 0.)
# Print the observed values to compare against the expected result.
print("Causal means:", out[:, 0])
```

Causal means $[2, 3, 4.666667, 7.5]$; no weight above the diagonal.

## Run the example

```python
# Step 1 — 1. Prepare the experiment: These imports provide JAX array transformations and NumPy...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Step 2 — 2. Implement the mechanism: allowed key columns query 0: 1 0 0 0 query 1: 1 1 0 0 query 2: 1 1...
def masked_mean(values, allowed):
    # Combine or mask array elements to form `scores`.
    scores = jnp.where(allowed, 0., -jnp.inf)
    # Apply nonlinear activation or probability normalization to compute `weights`.
    weights = jax.nn.softmax(scores, axis=-1)
    # Return `(weights @ values, weights)` to the caller.
    return weights @ values, weights

# Function `make_mask(segment_ids, valid)` implementing this stage's computation:
def make_mask(segment_ids, valid):
    # Construct `positions` via `jnp.arange(segment_ids.shape[0])`
    positions = jnp.arange(segment_ids.shape[0])
    # Compute `causal` from `positions[:, None] >= positions[None, :]`
    causal = positions[:, None] >= positions[None, :]
    # Compute `same_segment` from `segment_ids[:, None] == segment_ids[None, :]`
    same_segment = segment_ids[:, None] == segment_ids[None, :]
    # Return `causal & same_segment & valid[:, None] & valid[None, :]` to the caller.
    return causal & same_segment & valid[:, None] & valid[None, :]

# Step 3 — 3. Run and check: Causal means [2, 3, 4.666667, 7.5]; no weight above the diagonal.
# Construct `values` via `jnp.array([[2.], [4.], [8.], [16.]])`
values = jnp.array([[2.], [4.], [8.], [16.]])
# Construct `causal` via `jnp.arange(4)[:, None] >= jnp.arange(4)[None, :]`
causal = jnp.arange(4)[:, None] >= jnp.arange(4)[None, :]
# Run `masked_mean` to compute `(out, weights)`.
out, weights = masked_mean(values, causal)
# Assert that `jnp.allclose(out[:, 0], jnp.array([2., 3., 14/3, 7.5]))`.
assert jnp.allclose(out[:, 0], jnp.array([2., 3., 14/3, 7.5]))
# Assert that `jnp.allclose(jnp.triu(weights, 1), 0.)`.
assert jnp.allclose(jnp.triu(weights, 1), 0.)
# Print the observed values to compare against the expected result.
print("Causal means:", out[:, 0])
```

Expected: Causal means $[2, 3, 4.666667, 7.5]$; no weight above the diagonal.

## A causal mask blocks future information

**Predict:** Which cells must remain zero?

![A causal mask blocks future information](../../phases/07-transformers/02-masking-tokenization-and-sequence-packing/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

Rows select query positions and columns select key positions. The diagonal and cells to its left are allowed; the white cells to the right of the diagonal have exactly zero attention weight because they refer to future tokens.

Query $0$ puts all its weight on key $0$. Query $1$ splits weight equally across keys $0$ and $1$. The last row assigns $1/4$ to every key because, at the last position, all four positions are available.

### Connect it to the computation

The allowed scores are equal in this constructed example, so each row averages its visible prefix. The causal mask determines which entries must be zero; it does not generally make the remaining entries uniform. Learned scores can distribute weight unevenly within the allowed region.

For values $(2,4,8,16)$, query $2$ returns $(2+4+8)/3=14/3$, not the full-sequence average $7.5$. The zero in its last column is what prevents the future value $16$ from influencing that result. Verify normalization across rows, not down columns.

```python
# Compute figure data for: A causal mask blocks future information
# Compute `visual_data` from `{'kind': 'heatmap', 'values': weights.tolist(), 'row...`
visual_data = {'kind': 'heatmap', 'values': weights.tolist(), 'rows': ['query ' + str(i) for i in range(4)], 'columns': ['key ' + str(i) for i in range(4)], 'unit': 'attention weight'}
```

## Recorded reference execution

CPU run: 2026-10-08T14:03:30.420100+00:00. JAX 0.9.2.

```text
Causal means: [2.       3.       4.666667 7.5     ]
Causal means: [2.       3.       4.666667 7.5     ]
PASS: transformers-02

```

## Perturb the future

**Predict before running:** If the last value becomes $1000$, which outputs may change?

```python
# Experiment — Perturb the future: Perturbation tests whether earlier representations depend on...
changed = values.at[-1, 0].set(1000.)
# Run `masked_mean` to compute `(changed_out, _)`.
changed_out, _ = masked_mean(changed, causal)
# Assert that `jnp.allclose(changed_out[:3], out[:3])`.
assert jnp.allclose(changed_out[:3], out[:3])
# Assert that `not jnp.allclose(changed_out[-1], out[-1])`.
assert not jnp.allclose(changed_out[-1], out[-1])
```

**Expected:** Only the last output changes.

Perturbation tests whether earlier representations depend on future information.

## Pack two independent examples

**Predict before running:** What are the means for segments $[0,0,1,1]$?

```python
# Experiment — Pack two independent examples: Causal order alone does not isolate independent packed examples.
# Construct `segments` via `jnp.array([0, 0, 1, 1])`
segments = jnp.array([0, 0, 1, 1])
# Construct `valid` via `jnp.ones(4, dtype=bool)`
valid = jnp.ones(4, dtype=bool)
# Run `make_mask` to compute `packed`.
packed = make_mask(segments, valid)
# Run `masked_mean` to compute `(packed_out, _)`.
packed_out, _ = masked_mean(values, packed)
# Assert that `jnp.allclose(packed_out[:, 0], jnp.array([2., 3., 8., 12.]))`.
assert jnp.allclose(packed_out[:, 0], jnp.array([2., 3., 8., 12.]))
```

**Expected:** Packed means $[2, 3, 8, 12]$.

Causal order alone does not isolate independent packed examples.

## Make it yours

Encode abcab with a three-character vocabulary. Create shifted inputs/targets, and a loss-eligibility vector for packed segments $[0,0,0,1,1]$ that excludes a cross-segment next-token target.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.

**Step-by-step implementation plan:**
1. Construct `ids` via `jnp.array([vocab[c] for c in "abcab"])`
2. Assert invariant `jnp.array_equal(ids[:-1], jnp.array([0,1,2,0]))` holds
3. Assert invariant `jnp.array_equal(ids[1:], jnp.array([1,2,0,1]))` holds
4. Construct `seg` via `jnp.array([0,0,0,1,1])`
5. Compute `loss_valid` from `seg[:-1] == seg[1:]`

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Encode abcab with a three-character vocabulary.
vocab = ...  # TODO: compute vocab
# Construct `ids` via `jnp.array([vocab[c] for c in "abcab"])`
ids = jnp.array(...)  # TODO: compute ids
# Assert invariant `jnp.array_equal(ids[:-1], jnp.array([0,1,2,0]))` holds
assert jnp.array_equal(ids[:-1], jnp.array([0,1,2,0]))  # TODO: complete assertion check
# Assert invariant `jnp.array_equal(ids[1:], jnp.array([1,2,0,1]))` holds
assert jnp.array_equal(ids[1:], jnp.array([1,2,0,1]))  # TODO: complete assertion check
# Construct `seg` via `jnp.array([0,0,0,1,1])`
seg = jnp.array(...)  # TODO: compute seg
# Compute `loss_valid` from `seg[:-1] == seg[1:]`
loss_valid = ...  # TODO: compute loss_valid
# Assert invariant `jnp.array_equal(loss_valid, jnp.array([True,True,False,True]))` holds
assert jnp.array_equal(loss_valid, jnp.array([True,True,False,True]))  # TODO: complete assertion check
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Encode abcab with a three-character vocabulary.
vocab = {"a":0, "b":1, "c":2}
# Construct `ids` via `jnp.array([vocab[c] for c in "abcab"])`
ids = jnp.array([vocab[c] for c in "abcab"])
# Assert invariant `jnp.array_equal(ids[:-1], jnp.array([0,1,2,0]))` holds
assert jnp.array_equal(ids[:-1], jnp.array([0,1,2,0]))
# Assert invariant `jnp.array_equal(ids[1:], jnp.array([1,2,0,1]))` holds
assert jnp.array_equal(ids[1:], jnp.array([1,2,0,1]))
# Construct `seg` via `jnp.array([0,0,0,1,1])`
seg = jnp.array([0,0,0,1,1])
# Compute `loss_valid` from `seg[:-1] == seg[1:]`
loss_valid = seg[:-1] == seg[1:]
# Assert invariant `jnp.array_equal(loss_valid, jnp.array([True,True,False,True]))` holds
assert jnp.array_equal(loss_valid, jnp.array([True,True,False,True]))
```

</details>

## Repair padding-query NaNs

**Transfer / diagnosis**

Use `valid=[True,True,False,False]`. Reproduce the undefined all-masked rows, then give padded queries a fallback and zero their outputs.

<details><summary>Hint</summary>

Insert a diagonal fallback only for invalid queries.

</details>

### How to write: Repair padding-query NaNs — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.zeros / jnp.ones(shape, dtype=...)` — Allocates a tensor of the given `shape` initialized with constants.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `jnp.isfinite(x)` — Returns a boolean mask verifying that no element is `NaN` or `Inf`.

**Step-by-step implementation plan:**
1. Construct `pad_valid` via `jnp.array([True,True,False,False])`
2. Construct `pad_mask` via `make_mask(jnp.zeros(4, dtype=int), pad_valid)`
3. Combine or mask array elements to form `(bad_pad, _)`.
4. Confirm that all computed values remain finite (no NaN or Inf).
5. Compute `fallback` from `(~pad_valid[:, None]) & jnp.eye(4, dtype=bool)`

**Starter code scaffold (fill in the TODOs):**

```python
# Repair padding-query NaNs (Transfer / diagnosis): A fallback makes normalization defined; zeroing and loss...
# Construct `pad_valid` via `jnp.array([True,True,False,False])`
pad_valid = jnp.array(...)  # TODO: compute pad_valid
# Construct `pad_mask` via `make_mask(jnp.zeros(4, dtype=int), pad_valid)`
pad_mask = make_mask(...)  # TODO: compute pad_mask
# Combine or mask array elements to form `(bad_pad, _)`.
bad_pad, _ = masked_mean(...)  # TODO: compute bad_pad, _
# Confirm that all computed values remain finite (no NaN or Inf).
assert not jnp.all(jnp.isfinite(bad_pad))  # TODO: complete assertion check
# Compute `fallback` from `(~pad_valid[:, None]) & jnp.eye(4, dtype=bool)`
fallback = ...  # TODO: compute fallback
# Combine or mask array elements to form `(pad_out, _)`.
pad_out, _ = masked_mean(...)  # TODO: compute pad_out, _
# Combine or mask array elements to form `pad_out`.
pad_out = jnp.where(...)  # TODO: compute pad_out
# Confirm that all computed values remain finite (no NaN or Inf).
assert jnp.all(jnp.isfinite(pad_out))  # TODO: complete assertion check
# Assert that `jnp.allclose(pad_out[:,0], jnp.array([2.,3.,0.,0.]))`.
assert jnp.allclose(pad_out[:,0], jnp.array([2.,3.,0.,0.]))  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Repair padding-query NaNs (Transfer / diagnosis): A fallback makes normalization defined; zeroing and loss...
# Construct `pad_valid` via `jnp.array([True,True,False,False])`
pad_valid = jnp.array([True,True,False,False])
# Construct `pad_mask` via `make_mask(jnp.zeros(4, dtype=int), pad_valid)`
pad_mask = make_mask(jnp.zeros(4, dtype=int), pad_valid)
# Combine or mask array elements to form `(bad_pad, _)`.
bad_pad, _ = masked_mean(values, pad_mask)
# Confirm that all computed values remain finite (no NaN or Inf).
assert not jnp.all(jnp.isfinite(bad_pad))
# Compute `fallback` from `(~pad_valid[:, None]) & jnp.eye(4, dtype=bool)`
fallback = (~pad_valid[:, None]) & jnp.eye(4, dtype=bool)
# Combine or mask array elements to form `(pad_out, _)`.
pad_out, _ = masked_mean(values, pad_mask | fallback)
# Combine or mask array elements to form `pad_out`.
pad_out = jnp.where(pad_valid[:, None], pad_out, 0.)
# Confirm that all computed values remain finite (no NaN or Inf).
assert jnp.all(jnp.isfinite(pad_out))
# Assert that `jnp.allclose(pad_out[:,0], jnp.array([2.,3.,0.,0.]))`.
assert jnp.allclose(pad_out[:,0], jnp.array([2.,3.,0.,0.]))
```

A fallback makes normalization defined; zeroing and loss masking retain the semantic padding contract.

</details>

## Detect packed-example leakage

**Transfer / diagnosis**

Compare causal-only and segment-aware outputs at query $2$. Explain which prior values caused the difference.

<details><summary>Hint</summary>

Query $2$ should see only value $8$ from its own segment.

</details>

### How to write: Detect packed-example leakage — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Assert that `jnp.allclose(out[2,0], 14/3)`.
2. Assert that `jnp.allclose(packed_out[2,0], 8.)`.
3. Assert that `not jnp.allclose(out[2,0], packed_out[2,0])`.

**Starter code scaffold (fill in the TODOs):**

```python
# Detect packed-example leakage (Transfer / diagnosis): The earlier values 2 and 4 belong to another example; they...
# Assert that `jnp.allclose(out[2,0], 14/3)`.
assert jnp.allclose(out[2,0], 14/3)  # TODO: complete assertion check
# Assert that `jnp.allclose(packed_out[2,0], 8.)`.
assert jnp.allclose(packed_out[2,0], 8.)  # TODO: complete assertion check
# Assert that `not jnp.allclose(out[2,0], packed_out[2,0])`.
assert not jnp.allclose(out[2,0], packed_out[2,0])  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Detect packed-example leakage (Transfer / diagnosis): The earlier values 2 and 4 belong to another example; they...
# Assert that `jnp.allclose(out[2,0], 14/3)`.
assert jnp.allclose(out[2,0], 14/3)
# Assert that `jnp.allclose(packed_out[2,0], 8.)`.
assert jnp.allclose(packed_out[2,0], 8.)
# Assert that `not jnp.allclose(out[2,0], packed_out[2,0])`.
assert not jnp.allclose(out[2,0], packed_out[2,0])
```

The earlier values $2$ and $4$ belong to another example; they must be excluded despite being past positions.

</details>

## Check your understanding

What must a packed causal mask enforce?

1. Causal order only
2. Same segment and permitted causal keys, plus explicit padding behavior
3. Equal token IDs

<details><summary>Answer and explanation</summary>

Same segment and permitted causal keys, plus explicit padding behavior

Packing combines positions in storage, not their semantic context. Segment and validity constraints retain example boundaries.

</details>

## Diagnose the result

NaN in a padded output can mean that its mask row permits no keys; count allowed keys before blaming the optimizer. Add a fallback only for invalid queries, zero their outputs, and ignore their loss. If a packed output includes another example’s values, inspect equal-segment and causal conditions separately. Recheck the hand-derived $[2,3,8,12]$ result.

## Carry forward

- Perturbation tests whether earlier representations depend on future information.
- Causal order alone does not isolate independent packed examples.

## Keep your evidence

Keep token/target alignment, the two masks, future perturbation, segment-leakage comparison and all-masked-row repair. Verify that masks and ignored loss positions enforce the same boundaries.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [JAX softmax API](https://docs.jax.dev/en/latest/_autosummary/jax.nn.softmax.html)
- [JAX attention API](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html)
- [Optax cross entropy](https://optax.readthedocs.io/en/latest/api/losses.html)

