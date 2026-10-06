"""Batch a function with vmap: worked experiments and reference solutions. CPU checks."""



import jax
import jax.numpy as jnp
def predict(weight, x):
    return jnp.dot(weight, x)
weight = jnp.array([2., -1.])
batch = jnp.array([[1., 1.], [2., 3.], [4., 0.]])
batched = jax.vmap(predict, in_axes=(None, 0))
print(batched(weight, batch))
assert jnp.allclose(batched(weight, batch), jnp.array([1.,1.,8.]))

# Figure data experiment
visual_data = {'kind': 'bar', 'labels': ['row 0', 'row 1', 'row 2'], 'ylabel': 'prediction', 'series': [{'label': 'vmap output', 'y': batched(weight, batch).tolist()}]}

# Experiment: Verify against a loop and a matrix product
loop_predictions = jnp.stack([predict(weight, row) for row in batch])
vector_predictions = batched(weight, batch)
matrix_predictions = batch @ weight
assert vector_predictions.shape == (3,)
assert jnp.allclose(vector_predictions, loop_predictions)
assert jnp.allclose(vector_predictions, matrix_predictions)
print("loop / vmap / matmul:", loop_predictions, vector_predictions, matrix_predictions)

# Experiment: Move the example axis
column_batch = batch.T
column_predict = jax.vmap(predict, in_axes=(None, 1))
assert jnp.allclose(column_predict(weight, column_batch), vector_predictions)
def feature_pair(row):
    return jnp.array([jnp.sum(row), jnp.sum(row ** 2)])
features_rows = jax.vmap(feature_pair)(batch)
features_columns = jax.vmap(feature_pair, out_axes=1)(batch)
assert features_rows.shape == (3, 2)
assert features_columns.shape == (2, 3)
assert jnp.allclose(features_rows.T, features_columns)

# Reference solution. Try the exercise before reading this.
per_example_gradient = jax.vmap(jax.grad(predict), in_axes=(None, 0))(weight, batch)
assert per_example_gradient.shape == (3, 2)
assert jnp.allclose(per_example_gradient, batch)

# Reference practice: Turn sensitivities into training gradients
targets = jnp.array([0., 2., 7.])
def single_loss(parameters, row, target):
    return (predict(parameters, row) - target) ** 2
individual_grads = jax.vmap(jax.grad(single_loss), in_axes=(None, 0, 0))(weight, batch, targets)
manual_grads = 2. * (batch @ weight - targets)[:, None] * batch
assert individual_grads.shape == (3, 2)
assert jnp.allclose(individual_grads, manual_grads)
assert jnp.allclose(individual_grads[0], jnp.array([2., 2.]))

# Reference practice: Prove the mean-gradient identity with an experiment
def batch_mean_loss(parameters, rows, labels):
    return jnp.mean(jax.vmap(single_loss, in_axes=(None, 0, 0))(parameters, rows, labels))
for labels in (targets, targets + 0.5):
    each = jax.vmap(jax.grad(single_loss), in_axes=(None, 0, 0))(weight, batch, labels)
    aggregate = jax.grad(batch_mean_loss)(weight, batch, labels)
    assert aggregate.shape == weight.shape
    assert jnp.allclose(aggregate, jnp.mean(each, axis=0))
print("PASS: transforms-03")
