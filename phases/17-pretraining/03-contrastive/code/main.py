"""Contrastive learning: views, positives and negatives: worked experiments and reference solutions. CPU checks."""

# 1. Define both retrieval directions
# Step 1 — 1. Define both retrieval directions: Row and column log-softmax share scores but normalize different...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Function `paired_contrastive(left, right, temperature)` implementing this stage's computation:
def paired_contrastive(left, right, temperature=0.2):
    # Reduce across the target axis to summarize `left`.
    left = left / jnp.maximum(jnp.linalg.norm(left, axis=-1, keepdims=True), 1e-6)
    # Reduce across the target axis to summarize `right`.
    right = right / jnp.maximum(jnp.linalg.norm(right, axis=-1, keepdims=True), 1e-6)
    # Perform matrix contraction / projection to compute `scores`.
    scores = left @ right.T / temperature
    # Return `-0.5 * (jnp.mean(jnp.diag(jax.nn.log_softmax(scores, axis=1))) + jnp.mean(jnp.diag(jax.nn.log_softmax(scores, axis=0))))` to the caller.
    return -0.5 * (
        jnp.mean(jnp.diag(jax.nn.log_softmax(scores, axis=1)))
        + jnp.mean(jnp.diag(jax.nn.log_softmax(scores, axis=0)))
    )

# 2. Construct paired views and an encoder
# Step 2 — 2. Construct paired views and an encoder: The fixed perturbations preserve source identity.
# Construct `left` via `jnp.array(`
left = jnp.array(
    [[1.0, 0.0, 0.2], [0.0, 1.0, -0.2], [-1.0, 0.0, 0.1], [0.0, -1.0, -0.1]]
)
# Construct `right` via `left + jnp.array(`
right = left + jnp.array(
    [[0.02, -0.01, 0.0], [-0.01, 0.02, 0.0], [0.01, 0.01, 0.0], [-0.02, -0.01, 0.0]]
)
# Construct `w` via `jnp.array([[0.2, 0.1], [0.1, 0.1], [0.02, -0.01]])`
w = jnp.array([[0.2, 0.1], [0.1, 0.1], [0.02, -0.01]])
# Compute `history` from `[]`
history = []
# Differentiate the objective to obtain `step` via automatic differentiation.
step = jax.jit(
    jax.value_and_grad(lambda w: paired_contrastive(left @ w, right @ w, 0.2))
)

# 3. Train, then inspect identity and symmetry
# Step 3 — 3. Train, then inspect identity and symmetry: The host calculation checks both denominators.
for _ in range(60):
    # Run `step` to compute `(value, g)`.
    value, g = step(w)
    # Append the current step result to `history`.
    history.append(float(value))
    # Compute `w` from `w - 0.03 * g`
    w = w - 0.03 * g

# Assert invariant `history[-1] < history[0]` holds
assert history[-1] < history[0]
# Perform matrix / vector contraction (`@`) to compute `zi`.
zi = left @ w
# Perform matrix / vector contraction (`@`) to compute `zt`.
zt = right @ w
# Compute `zi` from `zi / jnp.linalg.norm(zi, axis=1, keepdims=True)`
zi = zi / jnp.linalg.norm(zi, axis=1, keepdims=True)
# Compute `zt` from `zt / jnp.linalg.norm(zt, axis=1, keepdims=True)`
zt = zt / jnp.linalg.norm(zt, axis=1, keepdims=True)
# Perform matrix contraction / projection to compute `scores`.
scores = zi @ zt.T / 0.2
# Convert `host` to a host NumPy array for inspection or verification.
host = np.asarray(scores, dtype=np.float64)

# Function `host_ce(a)` implementing this stage's computation:
def host_ce(a):
    # Return `np.mean(np.log(np.exp(a - a.max(1, keepdims=True)).sum(1)) + a.max(1) - np.diag(a))` to the caller.
    return np.mean(
        np.log(np.exp(a - a.max(1, keepdims=True)).sum(1)) + a.max(1) - np.diag(a)
    )

# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(
    paired_contrastive(left @ w, right @ w, 0.2),
    0.5 * (host_ce(host) + host_ce(host.T)),
    atol=1e-6,
)
# Assert invariant `np.array_equal(np.argmax(host, axis=1), np.arange(4))` holds
assert np.array_equal(np.argmax(host, axis=1), np.arange(4))
# Create device-backed JAX array `permutation`.
permutation = jnp.array([2, 0, 3, 1])
# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(
    paired_contrastive(left @ w, right @ w),
    paired_contrastive((left @ w)[permutation], (right @ w)[permutation]),
    atol=1e-6,
)
# Print diagnostic summary of the computed outputs.
print(
    'Paired contrastive initial/final:',
    history[0],
    history[-1],
    '; all four nearest pairs correct',
)

# Step 1 — 1. Define both retrieval directions: Row and column log-softmax share scores but normalize different...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Function `paired_contrastive(left, right, temperature)` implementing this stage's computation:
def paired_contrastive(left, right, temperature=0.2):
    # Reduce across the target axis to summarize `left`.
    left = left / jnp.maximum(jnp.linalg.norm(left, axis=-1, keepdims=True), 1e-6)
    # Reduce across the target axis to summarize `right`.
    right = right / jnp.maximum(jnp.linalg.norm(right, axis=-1, keepdims=True), 1e-6)
    # Perform matrix contraction / projection to compute `scores`.
    scores = left @ right.T / temperature
    # Return `-0.5 * (jnp.mean(jnp.diag(jax.nn.log_softmax(scores, axis=1))) + jnp.mean(jnp.diag(jax.nn.log_softmax(scores, axis=0))))` to the caller.
    return -0.5 * (
        jnp.mean(jnp.diag(jax.nn.log_softmax(scores, axis=1)))
        + jnp.mean(jnp.diag(jax.nn.log_softmax(scores, axis=0)))
    )

# Step 2 — 2. Construct paired views and an encoder: The fixed perturbations preserve source identity.
# Construct `left` via `jnp.array(`
left = jnp.array(
    [[1.0, 0.0, 0.2], [0.0, 1.0, -0.2], [-1.0, 0.0, 0.1], [0.0, -1.0, -0.1]]
)
# Construct `right` via `left + jnp.array(`
right = left + jnp.array(
    [[0.02, -0.01, 0.0], [-0.01, 0.02, 0.0], [0.01, 0.01, 0.0], [-0.02, -0.01, 0.0]]
)
# Construct `w` via `jnp.array([[0.2, 0.1], [0.1, 0.1], [0.02, -0.01]])`
w = jnp.array([[0.2, 0.1], [0.1, 0.1], [0.02, -0.01]])
# Compute `history` from `[]`
history = []
# Differentiate the objective to obtain `step` via automatic differentiation.
step = jax.jit(
    jax.value_and_grad(lambda w: paired_contrastive(left @ w, right @ w, 0.2))
)

# Step 3 — 3. Train, then inspect identity and symmetry: The host calculation checks both denominators.
for _ in range(60):
    # Run `step` to compute `(value, g)`.
    value, g = step(w)
    # Append the current step result to `history`.
    history.append(float(value))
    # Compute `w` from `w - 0.03 * g`
    w = w - 0.03 * g

# Assert invariant `history[-1] < history[0]` holds
assert history[-1] < history[0]
# Perform matrix / vector contraction (`@`) to compute `zi`.
zi = left @ w
# Perform matrix / vector contraction (`@`) to compute `zt`.
zt = right @ w
# Compute `zi` from `zi / jnp.linalg.norm(zi, axis=1, keepdims=True)`
zi = zi / jnp.linalg.norm(zi, axis=1, keepdims=True)
# Compute `zt` from `zt / jnp.linalg.norm(zt, axis=1, keepdims=True)`
zt = zt / jnp.linalg.norm(zt, axis=1, keepdims=True)
# Perform matrix contraction / projection to compute `scores`.
scores = zi @ zt.T / 0.2
# Convert `host` to a host NumPy array for inspection or verification.
host = np.asarray(scores, dtype=np.float64)

# Function `host_ce(a)` implementing this stage's computation:
def host_ce(a):
    # Return `np.mean(np.log(np.exp(a - a.max(1, keepdims=True)).sum(1)) + a.max(1) - np.diag(a))` to the caller.
    return np.mean(
        np.log(np.exp(a - a.max(1, keepdims=True)).sum(1)) + a.max(1) - np.diag(a)
    )

# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(
    paired_contrastive(left @ w, right @ w, 0.2),
    0.5 * (host_ce(host) + host_ce(host.T)),
    atol=1e-6,
)
# Assert invariant `np.array_equal(np.argmax(host, axis=1), np.arange(4))` holds
assert np.array_equal(np.argmax(host, axis=1), np.arange(4))
# Create device-backed JAX array `permutation`.
permutation = jnp.array([2, 0, 3, 1])
# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(
    paired_contrastive(left @ w, right @ w),
    paired_contrastive((left @ w)[permutation], (right @ w)[permutation]),
    atol=1e-6,
)
# Print diagnostic summary of the computed outputs.
print(
    'Paired contrastive initial/final:',
    history[0],
    history[-1],
    '; all four nearest pairs correct',
)

# Figure data experiment
# Compute figure data for: Contrastive learning: views, positives and negatives — recorded experiment
# Compute `visual_data` from `{'kind':'line','xlabel':'completed parameter updates...`
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'symmetric contrastive loss (nats)','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
# Loop over `panel` in `visual_data.get('panels', [visual_data])`:
for panel in visual_data.get('panels',[visual_data]):
    # Compute `panel['x']` from `panel['series'][0]['x']`
    panel['x']=panel['series'][0]['x']

# Convert `extra_panel` to a host NumPy array for inspection or verification.
extra_panel={'kind':'heatmap','values':np.asarray(zi@zt.T).tolist(),'rows':['left 0','left 1','left 2','left 3'],'columns':['right 0','right 1','right 2','right 3'],'unit':'cosine similarity','diverging':True,'xlabel':'right-view source ID','ylabel':'left-view source ID','title':'Final cross-view similarities before temperature scaling'}
# Compute `visual_data` from `{"panels":[*visual_data.get("panels",[visual_data]),...`
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}

# Experiment: Measure the collapsed baseline
# Experiment — Measure the collapsed baseline: Equal similarity gives each candidate probability one quarter.
# Construct `collapsed` via `jnp.ones((4, 2))`
collapsed = jnp.ones((4, 2))
# Evaluate `paired_contrastive(collapsed, collapsed)` and convert the result into Python scalar/collection `collapse_loss`.
collapse_loss = float(paired_contrastive(collapsed, collapsed))
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(collapse_loss, np.log(4), atol=1e-6)`
np.testing.assert_allclose(collapse_loss, np.log(4), atol=1e-6)
# Print the observed values to compare against the expected result.
print('Collapsed baseline:', collapse_loss)

# Experiment: Separate temperature from ranking
# Experiment — Separate temperature from ranking: The independent binary-softmax expression checks scale handling...
# Compute `orthogonal` from `jnp.eye(2)`
orthogonal = jnp.eye(2)
# Iterate over `tau` to step through the computation:
for tau in [0.2, 1.0, 2.0]:
    # Evaluate `paired_contrastive(orthogonal, orthogonal, tau)` and convert the result into Python scalar/collection `observed`.
    observed = float(paired_contrastive(orthogonal, orthogonal, tau))
    # Check numerical equivalence within tolerance: `np.testing.assert_allclose(observed, np.logaddexp(0.0, -1.0 / tau...`
    np.testing.assert_allclose(observed, np.logaddexp(0.0, -1.0 / tau), atol=1e-6)
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_array_equal(
        np.argmax(np.asarray(orthogonal @ orthogonal.T) / tau, axis=1), [0, 1]
    )
    # Print diagnostic summary of the computed outputs.
    print('Temperature/loss:', tau, observed)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Permute only the right-hand embeddings after training and demonstrate...
wrong_pair_loss = float(paired_contrastive(left @ w, (right @ w)[permutation]))
# Assert invariant `wrong_pair_loss > float(paired_contrastive(left @ w, right @ w))` holds
assert wrong_pair_loss > float(paired_contrastive(left @ w, right @ w))
# Print the observed values to compare against the expected result.
print('Incorrect pair mapping loss:', wrong_pair_loss)

# Reference practice: Duplicate a source and inspect the objective
# Duplicate a source and inspect the objective (Transfer): The extra log(2) comes from duplicating each denominator...
duplicated = float(
    paired_contrastive(jnp.tile(left @ w, (2, 1)), jnp.tile(right @ w, (2, 1)))
)
# Perform matrix contraction / projection to compute `original`.
original = float(paired_contrastive(left @ w, right @ w))
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(duplicated - original, np.log(2), atol...`
np.testing.assert_allclose(duplicated - original, np.log(2), atol=1e-5)
# Print the observed values to compare against the expected result.
print('Duplicated diagonal-only penalty:', duplicated - original)

# Reference practice: Check the encoder gradient through normalization
# Check the encoder gradient through normalization (Challenge): This checks the complete input-to-normalized-similarity chain.
# Construct `audit_w` via `jnp.array([[0.2, 0.1], [0.1, 0.1], [0.02, -0.01]])`
audit_w = jnp.array([[0.2, 0.1], [0.1, 0.1], [0.02, -0.01]])
# Perform matrix contraction / projection to compute `objective`.
objective = lambda weights: paired_contrastive(left @ weights, right @ weights, 0.2)
# Differentiate the objective to obtain `auto` via automatic differentiation.
auto = float(jax.grad(objective)(audit_w)[0, 0])
# Iterate over `epsilon` to step through the computation:
for epsilon in [1e-3, 5e-4]:
    # Construct `delta` via `jnp.zeros_like(audit_w).at[0, 0].set(epsilon)`
    delta = jnp.zeros_like(audit_w).at[0, 0].set(epsilon)
    # Evaluate `(objective(audit_w + delta) - objective(audit_w - delta)) / (2 * epsilon)` and convert the result into Python scalar/collection `estimate`.
    estimate = float(
        (objective(audit_w + delta) - objective(audit_w - delta)) / (2 * epsilon)
    )
    # Check numerical equivalence within tolerance: `np.testing.assert_allclose(estimate, auto, rtol=3e-3, atol=1e-3)`
    np.testing.assert_allclose(estimate, auto, rtol=3e-3, atol=1e-3)
    # Print diagnostic summary of the computed outputs.
    print('Step / finite difference / autodiff:', epsilon, estimate, auto)
print("PASS: pretraining-03")
