"""Masked language modeling: predict hidden tokens: worked experiments and reference solutions. CPU checks."""

# 1. Define corruption and the selected-token objective
# Step 1 — 1. Define corruption and the selected-token objective: The functions separate the input boundary from the differentiable...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Function `masked_ce(logits, targets, selected)` implementing this stage's computation:
def masked_ce(logits, targets, selected):
    # Caller validates a nonempty mask before a transformed training step.
    logp = jax.nn.log_softmax(logits, axis=-1)
    # Compute `nll` from `-jnp.take_along_axis(logp, targets[..., None], axis=...`
    nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    # Return `jnp.sum(jnp.where(selected, nll, 0.0)) / jnp.sum(selected)` to the caller.
    return jnp.sum(jnp.where(selected, nll, 0.0)) / jnp.sum(selected)

# Function `validate_mask(selected, shape)` implementing this stage's computation:
def validate_mask(selected, shape):
    # Convert `a` to a host NumPy array for inspection or verification.
    a = np.asarray(selected)
    # Guard input contract (`a.shape != shape or a.dtype != np.bool_ or (not a.any())`) and fail fast if violated.
    if a.shape != shape or a.dtype != np.bool_ or not a.any():
        raise ValueError('a nonempty Boolean mask of the target shape is required')

# Function `corrupt_tokens(tokens, selected, mask_id)` implementing this stage's computation:
def corrupt_tokens(tokens, selected, mask_id):
    # Run `validate_mask` to perform the next check or state transition.
    validate_mask(selected, tokens.shape)
    # Return `jnp.where(selected, mask_id, tokens)` to the caller.
    return jnp.where(selected, mask_id, tokens)

# Function `mlm_logits(p, corrupted)` implementing this stage's computation:
def mlm_logits(p, corrupted):
    # Compute `h` from `p['embedding'][corrupted]`
    h = p['embedding'][corrupted]
    # Bidirectional single-head attention, deliberately no causal mask.
    scores = h @ jnp.swapaxes(h, -1, -2) / jnp.sqrt(h.shape[-1])
    # Perform matrix / vector contraction (`@`) to compute `context`.
    context = jax.nn.softmax(scores, axis=-1) @ h
    # Return `context @ p['head']` to the caller.
    return context @ p['head']

# 2. Construct a batch whose answers you can inspect
# Step 2 — 2. Construct a batch whose answers you can inspect: Before continuing, inspect corrupted: each row must contain mask...
# Construct `tokens` via `jnp.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]...`
tokens = jnp.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]], jnp.int32)
# Construct `selected` via `jnp.array([[False, True, False, False]] * 3)`
selected = jnp.array([[False, True, False, False]] * 3)
# Run `corrupt_tokens` to compute `corrupted`.
corrupted = corrupt_tokens(tokens, selected, 3)
# Create or split explicit PRNG key(s) (`key`) for reproducible randomness.
key = jax.random.key(7)
# Sample deterministic random values into `p` using an explicit PRNG key.
p = {
    'embedding': jax.random.normal(key, (4, 6)) * 0.2,
    'head': jnp.zeros((6, 3)),
}
# Compute `loss` from `lambda p: masked_ce(mlm_logits(p, corrupted), tokens...`
loss = lambda p: masked_ce(mlm_logits(p, corrupted), tokens, selected)
# Differentiate the objective to obtain `step` via automatic differentiation.
step = jax.jit(jax.value_and_grad(loss))
# Compute `history` from `[]`
history = []

# 3. Train, then change the corruption
# Step 3 — 3. Train, then change the corruption: The loop measures loss before each update.
for _ in range(100):
    # Run `step` to compute `(value, g)`.
    value, g = step(p)
    # Append the current step result to `history`.
    history.append(float(value))
    # Transform every leaf of the parameter PyTree (`p`).
    p = jax.tree.map(lambda a, b: a - 0.4 * b, p, g)

# Assert invariant `history[-1] < history[0] * 0.15` holds
assert history[-1] < history[0] * 0.15
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(history[0], np.log(3), atol=1e-6)`
np.testing.assert_allclose(history[0], np.log(3), atol=1e-6)
# Changing the clean target after constructing corrupted input cannot change the forward pass.
changed_targets = tokens.at[:, 1].set((tokens[:, 1] + 1) % 3)
# Verify that the numerical values match the expected reference within tolerance.
assert not np.isclose(
    float(masked_ce(mlm_logits(p, corrupted), changed_targets, selected)),
    history[-1],
)
# Construct `held_mask` via `jnp.array([[False, False, True, False]] * 3)`
held_mask = jnp.array([[False, False, True, False]] * 3)
# Evaluate `masked_ce(mlm_logits(p, corrupt_tokens(tokens, held_mask, 3)), tokens, held_mask)` and convert the result into Python scalar/collection `held_loss`.
held_loss = float(
    masked_ce(mlm_logits(p, corrupt_tokens(tokens, held_mask, 3)), tokens, held_mask)
)
# Assert invariant `held_loss < 0.2` holds
assert held_loss < 0.2
# Print the observed values to compare against the expected result.
print('MLM initial/final/changed-mask:', history[0], history[-1], held_loss)

# Step 1 — 1. Define corruption and the selected-token objective: The functions separate the input boundary from the differentiable...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Function `masked_ce(logits, targets, selected)` implementing this stage's computation:
def masked_ce(logits, targets, selected):
    # Caller validates a nonempty mask before a transformed training step.
    logp = jax.nn.log_softmax(logits, axis=-1)
    # Compute `nll` from `-jnp.take_along_axis(logp, targets[..., None], axis=...`
    nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    # Return `jnp.sum(jnp.where(selected, nll, 0.0)) / jnp.sum(selected)` to the caller.
    return jnp.sum(jnp.where(selected, nll, 0.0)) / jnp.sum(selected)

# Function `validate_mask(selected, shape)` implementing this stage's computation:
def validate_mask(selected, shape):
    # Convert `a` to a host NumPy array for inspection or verification.
    a = np.asarray(selected)
    # Guard input contract (`a.shape != shape or a.dtype != np.bool_ or (not a.any())`) and fail fast if violated.
    if a.shape != shape or a.dtype != np.bool_ or not a.any():
        raise ValueError('a nonempty Boolean mask of the target shape is required')

# Function `corrupt_tokens(tokens, selected, mask_id)` implementing this stage's computation:
def corrupt_tokens(tokens, selected, mask_id):
    # Run `validate_mask` to perform the next check or state transition.
    validate_mask(selected, tokens.shape)
    # Return `jnp.where(selected, mask_id, tokens)` to the caller.
    return jnp.where(selected, mask_id, tokens)

# Function `mlm_logits(p, corrupted)` implementing this stage's computation:
def mlm_logits(p, corrupted):
    # Compute `h` from `p['embedding'][corrupted]`
    h = p['embedding'][corrupted]
    # Bidirectional single-head attention, deliberately no causal mask.
    scores = h @ jnp.swapaxes(h, -1, -2) / jnp.sqrt(h.shape[-1])
    # Perform matrix / vector contraction (`@`) to compute `context`.
    context = jax.nn.softmax(scores, axis=-1) @ h
    # Return `context @ p['head']` to the caller.
    return context @ p['head']

# Step 2 — 2. Construct a batch whose answers you can inspect: Before continuing, inspect corrupted: each row must contain mask...
# Construct `tokens` via `jnp.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]...`
tokens = jnp.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]], jnp.int32)
# Construct `selected` via `jnp.array([[False, True, False, False]] * 3)`
selected = jnp.array([[False, True, False, False]] * 3)
# Run `corrupt_tokens` to compute `corrupted`.
corrupted = corrupt_tokens(tokens, selected, 3)
# Create or split explicit PRNG key(s) (`key`) for reproducible randomness.
key = jax.random.key(7)
# Sample deterministic random values into `p` using an explicit PRNG key.
p = {
    'embedding': jax.random.normal(key, (4, 6)) * 0.2,
    'head': jnp.zeros((6, 3)),
}
# Compute `loss` from `lambda p: masked_ce(mlm_logits(p, corrupted), tokens...`
loss = lambda p: masked_ce(mlm_logits(p, corrupted), tokens, selected)
# Differentiate the objective to obtain `step` via automatic differentiation.
step = jax.jit(jax.value_and_grad(loss))
# Compute `history` from `[]`
history = []

# Step 3 — 3. Train, then change the corruption: The loop measures loss before each update.
for _ in range(100):
    # Run `step` to compute `(value, g)`.
    value, g = step(p)
    # Append the current step result to `history`.
    history.append(float(value))
    # Transform every leaf of the parameter PyTree (`p`).
    p = jax.tree.map(lambda a, b: a - 0.4 * b, p, g)

# Assert invariant `history[-1] < history[0] * 0.15` holds
assert history[-1] < history[0] * 0.15
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(history[0], np.log(3), atol=1e-6)`
np.testing.assert_allclose(history[0], np.log(3), atol=1e-6)
# Changing the clean target after constructing corrupted input cannot change the forward pass.
changed_targets = tokens.at[:, 1].set((tokens[:, 1] + 1) % 3)
# Verify that the numerical values match the expected reference within tolerance.
assert not np.isclose(
    float(masked_ce(mlm_logits(p, corrupted), changed_targets, selected)),
    history[-1],
)
# Construct `held_mask` via `jnp.array([[False, False, True, False]] * 3)`
held_mask = jnp.array([[False, False, True, False]] * 3)
# Evaluate `masked_ce(mlm_logits(p, corrupt_tokens(tokens, held_mask, 3)), tokens, held_mask)` and convert the result into Python scalar/collection `held_loss`.
held_loss = float(
    masked_ce(mlm_logits(p, corrupt_tokens(tokens, held_mask, 3)), tokens, held_mask)
)
# Assert invariant `held_loss < 0.2` holds
assert held_loss < 0.2
# Print the observed values to compare against the expected result.
print('MLM initial/final/changed-mask:', history[0], history[-1], held_loss)

# Figure data experiment
# Compute figure data for: Masked language modeling: predict hidden tokens — recorded experiment
# Compute `visual_data` from `{'kind':'line','xlabel':'completed parameter updates...`
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'selected-token cross-entropy (nats)','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
# Loop over `panel` in `visual_data.get('panels', [visual_data])`:
for panel in visual_data.get('panels',[visual_data]):
    # Compute `panel['x']` from `panel['series'][0]['x']`
    panel['x']=panel['series'][0]['x']

# Convert `final_probabilities` to a host NumPy array for inspection or verification.
final_probabilities=np.asarray(jax.nn.softmax(mlm_logits(p,corrupted)[:,1,:],axis=-1))
# Compute `extra_panel` from `{'kind':'heatmap','values':final_probabilities.tolis...`
extra_panel={'kind':'heatmap','values':final_probabilities.tolist(),'rows':['target 0','target 1','target 2'],'columns':['class 0','class 1','class 2'],'unit':'masked-token probability','xlabel':'predicted vocabulary class','ylabel':'masked example','title':'Final predictions at the selected position'}
# Compute `visual_data` from `{"panels":[*visual_data.get("panels",[visual_data]),...`
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}

# Experiment: Show that unsupervised positions have no direct loss gradient
# Experiment — Show that unsupervised positions have no direct loss gradient: Visible input tokens can still influence selected predictions...
logits = mlm_logits(p, corrupted)
# Compute `changed_logits` from `jnp.where(`
changed_logits = jnp.where(
    selected[..., None], logits, logits + jnp.array([100.0, -50.0, 7.0])
)
# Execute `np.testing.assert_allclose(`
np.testing.assert_allclose(
    masked_ce(changed_logits, tokens, selected),
    masked_ce(logits, tokens, selected),
    atol=1e-6,
)
# Differentiate the objective to obtain `grad_logits` via automatic differentiation.
grad_logits = jax.grad(masked_ce)(logits, tokens, selected)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(np.asarray(grad_logits)[~np.asarray(selected)], 0.0)
# Print the observed values to compare against the expected result.
print('Unselected output logits have zero direct loss gradient.')

# Experiment: Use unequal selection counts and a host probability calculation
# Experiment — Use unequal selection counts and a host probability calculation: The second sequence supplies three of four selected tokens.
# Construct `audit_probs` via `jnp.array([`
audit_probs = jnp.array([
    [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.3, 0.5]],
    [[0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.3, 0.4, 0.3]],
])
# Construct `audit_targets` via `jnp.array([[0, 1, 2], [1, 0, 2]])`
audit_targets = jnp.array([[0, 1, 2], [1, 0, 2]])
# Construct `audit_mask` via `jnp.array([[True, False, False], [True, True, True]])`
audit_mask = jnp.array([[True, False, False], [True, True, True]])
# Run `jnp.log` to compute `audit_logits`.
audit_logits = jnp.log(audit_probs)
# Compute `host_losses` from `-np.log(`
host_losses = -np.log(
    np.asarray(audit_probs)[
        np.arange(2)[:, None], np.arange(3)[None, :], np.asarray(audit_targets)
    ]
)
# Aggregate array values to compute `expected`.
expected = host_losses[np.asarray(audit_mask)].mean()
# Evaluate `masked_ce(audit_logits, audit_targets, audit_mask)` and convert the result into Python scalar/collection `observed`.
observed = float(masked_ce(audit_logits, audit_targets, audit_mask))
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(observed, expected, atol=1e-6)`
np.testing.assert_allclose(observed, expected, atol=1e-6)
# Aggregate array values to compute `sequence_mean`.
sequence_mean = np.mean(
    [host_losses[i][np.asarray(audit_mask[i])].mean() for i in range(2)]
)
# Verify that the numerical values match the expected reference within tolerance.
assert not np.isclose(observed, sequence_mean)
# Print diagnostic summary of the computed outputs.
print('Token-weighted / sequence-weighted:', observed, sequence_mean)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Select the first position instead of the second and evaluate the...
# Construct `first_mask` via `jnp.array([[True, False, False, False]] * 3)`
first_mask = jnp.array([[True, False, False, False]] * 3)
# Evaluate `masked_ce(mlm_logits(p, corrupt_tokens(tokens, first_mask, 3)), tokens, first_mask)` and convert the result into Python scalar/collection `first_loss`.
first_loss = float(
    masked_ce(mlm_logits(p, corrupt_tokens(tokens, first_mask, 3)), tokens, first_mask)
)
# Assert invariant `first_loss < 0.2` holds
assert first_loss < 0.2
# Print the observed values to compare against the expected result.
print('Changed position loss:', first_loss)

# Reference practice: Reject a missing learning signal
# Reject a missing learning signal (Transfer): A skipped batch requires an explicit training policy.
# Run the boundary check and catch the expected exception:
try:
    corrupt_tokens(tokens, jnp.zeros_like(tokens, dtype=bool), 3)
except ValueError:
    print('Empty mask rejected')
else:
    raise AssertionError('empty mask accepted')

# Reference practice: Derive the output gradient without autodiff
# Derive the output gradient without autodiff (Challenge): This verifies the entire derivative tensor, including zeros...
# Reduce along axis=-1 to compute `analytic`.
analytic = (
    (jax.nn.softmax(audit_logits, axis=-1) - jax.nn.one_hot(audit_targets, 3))
    * audit_mask[..., None]
    / audit_mask.sum()
)
# Differentiate the objective to obtain gradients ``.
np.testing.assert_allclose(
    jax.grad(masked_ce)(audit_logits, audit_targets, audit_mask),
    analytic,
    atol=1e-6,
)
# Print the observed values to compare against the expected result.
print('Independent masked softmax gradient agrees.')
print("PASS: pretraining-01")
