"""Batch a function with vmap: worked experiments and reference solutions. CPU checks."""

# Step 1: Set up imports and input tensors
import jax
import jax.numpy as jnp

# Step 2: Apply the core JAX transformation
def predict(weight, x):
    # Return `jnp.dot(weight, x)` to the caller.
    return jnp.dot(weight, x)
# Construct `weight` via `jnp.array([2., -1.])`
weight = jnp.array([2., -1.])

# Step 3: Verify shapes and numerical invariants
batch = jnp.array([[1., 1.], [2., 3.], [4., 0.]])
# Vectorize across the batch dimension with `jax.vmap` (`batched`).
batched = jax.vmap(predict, in_axes=(None, 0))
# Print the observed values to compare against the expected result.
# Assert that `jnp.allclose(batched(weight, batch), jnp.array([1.,1.,8.]))`.
assert jnp.allclose(batched(weight, batch), jnp.array([1.,1.,8.]))

# Batch a function with vmap: Vectorization lets us describe one example clearly and then...
# Import jax for this computation.
import jax
import jax.numpy as jnp
# Function `predict(weight, x)` implementing this stage's computation:
def predict(weight, x):
    # Return `jnp.dot(weight, x)` to the caller.
    return jnp.dot(weight, x)
# Construct `weight` via `jnp.array([2., -1.])`
weight = jnp.array([2., -1.])
# Construct `batch` via `jnp.array([[1., 1.], [2., 3.], [4., 0.]])`
batch = jnp.array([[1., 1.], [2., 3.], [4., 0.]])
# Vectorize across the batch dimension with `jax.vmap` (`batched`).
batched = jax.vmap(predict, in_axes=(None, 0))
# Print the observed values to compare against the expected result.
print(batched(weight, batch))
# Assert that `jnp.allclose(batched(weight, batch), jnp.array([1.,1.,8.]))`.
assert jnp.allclose(batched(weight, batch), jnp.array([1.,1.,8.]))

# Figure data experiment
# Compute figure data for: Vectorization keeps one result per row
# Compute `visual_data` from `{'kind': 'bar', 'labels': ['row 0', 'row 1', 'row 2'...`
visual_data = {'kind': 'bar', 'labels': ['row 0', 'row 1', 'row 2'], 'ylabel': 'prediction', 'series': [{'label': 'vmap output', 'y': batched(weight, batch).tolist()}]}

# Experiment: Verify against a loop and a matrix product
# Experiment — Verify against a loop and a matrix product: The loop makes example selection explicit; the matrix product...
loop_predictions = jnp.stack([predict(weight, row) for row in batch])
# Run `batched` to compute `vector_predictions`.
vector_predictions = batched(weight, batch)
# Perform matrix / vector contraction (`@`) to compute `matrix_predictions`.
matrix_predictions = batch @ weight
# Check tensor shape invariant: `vector_predictions.shape == (3,)`
assert vector_predictions.shape == (3,)
# Assert that `jnp.allclose(vector_predictions, loop_predictions)`.
assert jnp.allclose(vector_predictions, loop_predictions)
# Assert that `jnp.allclose(vector_predictions, matrix_predictions)`.
assert jnp.allclose(vector_predictions, matrix_predictions)
# Print the observed values to compare against the expected result.
print("loop / vmap / matmul:", loop_predictions, vector_predictions, matrix_predictions)

# Experiment: Move the example axis
# Experiment — Move the example axis: The examples axis and output placement are separate choices.
column_batch = batch.T
# Vectorize across the batch dimension with `jax.vmap` (`column_predict`).
column_predict = jax.vmap(predict, in_axes=(None, 1))
# Assert that `jnp.allclose(column_predict(weight, column_batch), vector_predictions)`.
assert jnp.allclose(column_predict(weight, column_batch), vector_predictions)
# Function `feature_pair(row)` implementing this stage's computation:
def feature_pair(row):
    # Return `jnp.array([jnp.sum(row), jnp.sum(row ** 2)])` to the caller.
    return jnp.array([jnp.sum(row), jnp.sum(row ** 2)])
# Vectorize across the batch dimension with `jax.vmap` (`features_rows`).
features_rows = jax.vmap(feature_pair)(batch)
# Vectorize across the batch dimension with `jax.vmap` (`features_columns`).
features_columns = jax.vmap(feature_pair, out_axes=1)(batch)
# Check tensor shape invariant: `features_rows.shape == (3, 2)`
assert features_rows.shape == (3, 2)
# Check tensor shape invariant: `features_columns.shape == (2, 3)`
assert features_columns.shape == (2, 3)
# Assert that `jnp.allclose(features_rows.T, features_columns)`.
assert jnp.allclose(features_rows.T, features_columns)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Foundation · Compute each prediction gradient with respect to the...
# Differentiate the objective to obtain `per_example_gradient` via automatic differentiation.
per_example_gradient = jax.vmap(jax.grad(predict), in_axes=(None, 0))(weight, batch)
# Check tensor shape invariant: `per_example_gradient.shape == (3, 2)`
assert per_example_gradient.shape == (3, 2)
# Assert that `jnp.allclose(per_example_gradient, batch)`.
assert jnp.allclose(per_example_gradient, batch)

# Reference practice: Turn sensitivities into training gradients
# Turn sensitivities into training gradients (Practice): The first residual is 1, so the first loss gradient is [2, 2].
# Construct `targets` via `jnp.array([0., 2., 7.])`
targets = jnp.array([0., 2., 7.])
# Function `single_loss(parameters, row, target)` implementing this stage's computation:
def single_loss(parameters, row, target):
    # Return `(predict(parameters, row) - target) ** 2` to the caller.
    return (predict(parameters, row) - target) ** 2
# Differentiate the objective to obtain `individual_grads` via automatic differentiation.
individual_grads = jax.vmap(jax.grad(single_loss), in_axes=(None, 0, 0))(weight, batch, targets)
# Perform matrix contraction / projection to compute `manual_grads`.
manual_grads = 2. * (batch @ weight - targets)[:, None] * batch
# Check tensor shape invariant: `individual_grads.shape == (3, 2)`
assert individual_grads.shape == (3, 2)
# Assert that `jnp.allclose(individual_grads, manual_grads)`.
assert jnp.allclose(individual_grads, manual_grads)
# Assert that `jnp.allclose(individual_grads[0], jnp.array([2., 2.]))`.
assert jnp.allclose(individual_grads[0], jnp.array([2., 2.]))

# Reference practice: Prove the mean-gradient identity with an experiment
# Prove the mean-gradient identity with an experiment (Challenge): The aggregate has parameter shape, while per-example...
def batch_mean_loss(parameters, rows, labels):
    # Return `jnp.mean(jax.vmap(single_loss, in_axes=(None, 0, 0))(parameters, rows, labels))` to the caller.
    return jnp.mean(jax.vmap(single_loss, in_axes=(None, 0, 0))(parameters, rows, labels))
# Iterate over `labels` to step through the computation:
for labels in (targets, targets + 0.5):
    # Differentiate the objective to obtain `each` via automatic differentiation.
    each = jax.vmap(jax.grad(single_loss), in_axes=(None, 0, 0))(weight, batch, labels)
    # Differentiate the objective to obtain `aggregate` via automatic differentiation.
    aggregate = jax.grad(batch_mean_loss)(weight, batch, labels)
    # Check tensor shape invariant: `aggregate.shape == weight.shape`
    assert aggregate.shape == weight.shape
    # Assert that `jnp.allclose(aggregate, jnp.mean(each, axis=0))`.
    assert jnp.allclose(aggregate, jnp.mean(each, axis=0))
print("PASS: transforms-03")
