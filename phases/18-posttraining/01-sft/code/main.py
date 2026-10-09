"""Supervised fine-tuning with response-only token loss: worked experiments and reference solutions. CPU checks."""

# 1. Define sequence log-probability sums
# Step 1 — 1. Define sequence log-probability sums: The slice removes the final logit and first token.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Function `response_logps(logits, tokens, response_mask)` implementing this stage's computation:
def response_logps(logits, tokens, response_mask):
    # logits at t predict token t+1; the role mask belongs to the target token.
    logp = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
    # Run `jnp.take_along_axis` to compute `selected`.
    selected = jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1)[..., 0]
    # Return `jnp.sum(jnp.where(response_mask[:, 1:], selected, 0.0), axis=-1)` to the caller.
    return jnp.sum(jnp.where(response_mask[:, 1:], selected, 0.0), axis=-1)

# Function `validate_mask(selected, shape)` implementing this stage's computation:
def validate_mask(selected, shape):
    # Convert `a` to a host NumPy array for inspection or verification.
    a = np.asarray(selected)
    # Guard input contract (`a.shape != shape or a.dtype != np.bool_ or (not a.any())`) and fail fast if violated.
    if a.shape != shape or a.dtype != np.bool_ or not a.any():
        raise ValueError('a nonempty Boolean mask of the target shape is required')

# 2. Align prompt, response and target masks
# Token 0/1 is a prompt; 2/3 is its response; 4 marks the end.
tokens = jnp.array([[0, 2, 4], [1, 3, 4]], jnp.int32)
# Construct `roles` via `jnp.array([[False, True, True]] * 2)`
roles = jnp.array([[False, True, True]] * 2)
# Run `validate_mask` to perform the next check or state transition.
validate_mask(roles[:, 1:], tokens[:, 1:].shape)
# Construct `p` via `jnp.zeros((5, 5))`
p = jnp.zeros((5, 5))
# Compute `history` from `[]`
history = []

# Function `sft_objective(p)` implementing this stage's computation:
def sft_objective(p):
    # Return `-jnp.sum(response_logps(p[tokens], tokens, roles)) / jnp.sum(roles[:, 1:])` to the caller.
    return -jnp.sum(response_logps(p[tokens], tokens, roles)) / jnp.sum(roles[:, 1:])

# Differentiate the objective to obtain `step` via automatic differentiation.
step = jax.jit(jax.value_and_grad(sft_objective))

# 3. Fit the transition table and check the final position
# Step 3 — 3. Fit the transition table and check the final position: The table learns only local transitions.
for _ in range(100):
    # Run `step` to compute `(value, g)`.
    value, g = step(p)
    # Append the current step result to `history`.
    history.append(float(value))
    # Compute `p` from `p - 0.5 * g`
    p = p - 0.5 * g

# Compute `np.testing.assert_allclose(history[0], np.log(5), atol` as `1e-6)`.
np.testing.assert_allclose(history[0], np.log(5), atol=1e-6)
# Assert invariant `history[-1] < 0.15` holds
assert history[-1] < 0.15
# Assert invariant `np.array_equal(np.argmax(np.asarray(p)[[0, 1]], axis=1), [2, 3])` holds
assert np.array_equal(np.argmax(np.asarray(p)[[0, 1]], axis=1), [2, 3])
# The final position predicts nothing and has no contribution.
base_logits = p[tokens]
# Compute `changed` from `base_logits.at[:, -1, :].set(100.0)`
changed = base_logits.at[:, -1, :].set(100.0)
# Execute `np.testing.assert_allclose(`
np.testing.assert_allclose(
    response_logps(changed, tokens, roles),
    response_logps(base_logits, tokens, roles),
)
# Print the observed values to compare against the expected result.
print('Response-only token NLL initial/final:', history[0], history[-1])

# Step 1 — 1. Define sequence log-probability sums: The slice removes the final logit and first token.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Function `response_logps(logits, tokens, response_mask)` implementing this stage's computation:
def response_logps(logits, tokens, response_mask):
    # logits at t predict token t+1; the role mask belongs to the target token.
    logp = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
    # Run `jnp.take_along_axis` to compute `selected`.
    selected = jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1)[..., 0]
    # Return `jnp.sum(jnp.where(response_mask[:, 1:], selected, 0.0), axis=-1)` to the caller.
    return jnp.sum(jnp.where(response_mask[:, 1:], selected, 0.0), axis=-1)

# Function `validate_mask(selected, shape)` implementing this stage's computation:
def validate_mask(selected, shape):
    # Convert `a` to a host NumPy array for inspection or verification.
    a = np.asarray(selected)
    # Guard input contract (`a.shape != shape or a.dtype != np.bool_ or (not a.any())`) and fail fast if violated.
    if a.shape != shape or a.dtype != np.bool_ or not a.any():
        raise ValueError('a nonempty Boolean mask of the target shape is required')

# Token 0/1 is a prompt; 2/3 is its response; 4 marks the end.
tokens = jnp.array([[0, 2, 4], [1, 3, 4]], jnp.int32)
# Construct `roles` via `jnp.array([[False, True, True]] * 2)`
roles = jnp.array([[False, True, True]] * 2)
# Run `validate_mask` to perform the next check or state transition.
validate_mask(roles[:, 1:], tokens[:, 1:].shape)
# Construct `p` via `jnp.zeros((5, 5))`
p = jnp.zeros((5, 5))
# Compute `history` from `[]`
history = []

# Function `sft_objective(p)` implementing this stage's computation:
def sft_objective(p):
    # Return `-jnp.sum(response_logps(p[tokens], tokens, roles)) / jnp.sum(roles[:, 1:])` to the caller.
    return -jnp.sum(response_logps(p[tokens], tokens, roles)) / jnp.sum(roles[:, 1:])

# Differentiate the objective to obtain `step` via automatic differentiation.
step = jax.jit(jax.value_and_grad(sft_objective))

# Step 3 — 3. Fit the transition table and check the final position: The table learns only local transitions.
for _ in range(100):
    # Run `step` to compute `(value, g)`.
    value, g = step(p)
    # Append the current step result to `history`.
    history.append(float(value))
    # Compute `p` from `p - 0.5 * g`
    p = p - 0.5 * g

# Compute `np.testing.assert_allclose(history[0], np.log(5), atol` as `1e-6)`.
np.testing.assert_allclose(history[0], np.log(5), atol=1e-6)
# Assert invariant `history[-1] < 0.15` holds
assert history[-1] < 0.15
# Assert invariant `np.array_equal(np.argmax(np.asarray(p)[[0, 1]], axis=1), [2, 3])` holds
assert np.array_equal(np.argmax(np.asarray(p)[[0, 1]], axis=1), [2, 3])
# The final position predicts nothing and has no contribution.
base_logits = p[tokens]
# Compute `changed` from `base_logits.at[:, -1, :].set(100.0)`
changed = base_logits.at[:, -1, :].set(100.0)
# Execute `np.testing.assert_allclose(`
np.testing.assert_allclose(
    response_logps(changed, tokens, roles),
    response_logps(base_logits, tokens, roles),
)
# Print the observed values to compare against the expected result.
print('Response-only token NLL initial/final:', history[0], history[-1])

# Figure data experiment
# Compute figure data for: Supervised fine-tuning with response-only token loss — recorded experiment
# Compute `visual_data` from `{'kind':'line','xlabel':'completed parameter updates...`
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'response-token NLL (nats)','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
# Loop over `panel` in `visual_data.get('panels', [visual_data])`:
for panel in visual_data.get('panels',[visual_data]):
    # Compute `panel['x']` from `panel['series'][0]['x']`
    panel['x']=panel['series'][0]['x']

# Allocate initialized array `initial_output_grads` with the specified shape and dtype.
initial_output_grads=jax.grad(lambda values:-jnp.sum(response_logps(values,tokens,roles))/jnp.sum(roles[:,1:]))(jnp.zeros((2,3,5)))
# Convert `extra_panel` to a host NumPy array for inspection or verification.
extra_panel={'kind':'heatmap','values':np.asarray(jnp.linalg.norm(initial_output_grads,axis=-1)).tolist(),'rows':['prompt 0 / answer 2','prompt 1 / answer 3'],'columns':['input position 0','input position 1','input position 2'],'unit':'output-logit gradient L2 norm','xlabel':'position producing the prediction','ylabel':'sequence','title':'Initial supervised gradient by output position'}
# Compute `visual_data` from `{"panels":[*visual_data.get("panels",[visual_data]),...`
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}

# Experiment: Prove the target-mask shift
# Experiment — Prove the target-mask shift: The prediction made at the prompt position is supervised because...
# Differentiate the objective to obtain `g` via automatic differentiation.
g = jax.grad(sft_objective)(jnp.zeros((5, 5)))
# Assert that `np.linalg.norm(np.asarray(g)[0]) > 0 and np.linalg.norm(np.asarray(g)[1]) > 0`.
assert np.linalg.norm(np.asarray(g)[0]) > 0 and np.linalg.norm(np.asarray(g)[1]) > 0
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(np.asarray(g)[4], 0.0)
# Print the observed values to compare against the expected result.
print('Prompt rows learn to predict first answers; the final end-token row has no target.')

# Experiment: Check a ragged batch against a scalar loop
# Experiment — Check a ragged batch against a scalar loop: Padding and role masks affect the target after shifting.
# Construct `ragged_tokens` via `jnp.array([[0, 2, 4, 4], [1, 3, 2, 4]])`
ragged_tokens = jnp.array([[0, 2, 4, 4], [1, 3, 2, 4]])
# Construct `ragged_roles` via `jnp.array([[False, True, True, False], [False, True,...`
ragged_roles = jnp.array([[False, True, True, False], [False, True, True, True]])
# Construct and reshape `ragged_logits` into the target tensor dimensions.
ragged_logits = jnp.arange(2 * 4 * 5, dtype=jnp.float32).reshape(2, 4, 5) / 19
# Run `response_logps` to compute `observed`.
observed = response_logps(ragged_logits, ragged_tokens, ragged_roles)
# Convert `arr` to a host NumPy array for inspection or verification.
arr = np.asarray(ragged_logits, dtype=np.float64)
# Compute `expected` from `[]`
expected = []
# Iterate over `row` to step through the computation:
for row in range(2):
    # Compute `total` from `0.0`
    total = 0.0
    # Iterate over `position` to step through the computation:
    for position in range(3):
        # Branch on condition `ragged_roles[row, position + 1]`:
        if ragged_roles[row, position + 1]:
            values = arr[row, position]
            log_normalizer = values.max() + np.log(np.exp(values - values.max()).sum())
            total += values[int(ragged_tokens[row, position + 1])] - log_normalizer
    # Append the current step result to `expected`.
    expected.append(total)
# Compute `np.testing.assert_allclose(observed, expected, atol` as `1e-6)`.
np.testing.assert_allclose(observed, expected, atol=1e-6)
# Print the observed values to compare against the expected result.
print('Supervised targets per sequence:', np.asarray(ragged_roles[:, 1:].sum(1)))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Score only the first response token in each sequence and compare the...
first_only = roles.at[:, 2].set(False)
# Aggregate array values to compute `value`.
value = -jnp.sum(response_logps(p[tokens], tokens, first_only)) / jnp.sum(
    first_only[:, 1:]
)
# Assert invariant `int(jnp.sum(first_only[:, 1:])) == 2` holds
assert int(jnp.sum(first_only[:, 1:])) == 2
# Ensure all array elements remain finite: `np.isfinite(float(value))`
assert np.isfinite(float(value))
# Print the observed values to compare against the expected result.
print('Two first-response targets:', float(value))

# Reference practice: Reject a batch with no answer targets
# Reject a batch with no answer targets (Transfer): An empty answer mask should trigger a clear skip/rejection...
# Run the boundary check and catch the expected exception:
try:
    validate_mask(jnp.zeros_like(roles[:, 1:]), tokens[:, 1:].shape)
except ValueError:
    print('No-target SFT batch rejected')
else:
    raise AssertionError('empty supervision accepted')

# Reference practice: Create a failure this model cannot fix
# Create a failure this model cannot fix (Challenge): A model that conditions on the earlier context can represent...
# Construct `contexts` via `jnp.array([[1, 0], [3, 0]])`
contexts = jnp.array([[1, 0], [3, 0]])
# Compute `last_logits` from `p[contexts[:, -1]]`
last_logits = p[contexts[:, -1]]
# Execute `np.testing.assert_array_equal(last_logits[0], last_logits[1]`
np.testing.assert_array_equal(last_logits[0], last_logits[1])
# Construct `conflicting_targets` via `jnp.array([2, 3])`
conflicting_targets = jnp.array([2, 3])
# Assert invariant `conflicting_targets[0] != conflicting_targets[1]` holds
assert conflicting_targets[0] != conflicting_targets[1]
# Print the observed values to compare against the expected result.
print('Identical final tokens force identical table predictions despite different contexts.')
print("PASS: posttraining-01")
