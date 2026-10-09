"""Pure functions and explicit inputs: worked experiments and reference solutions. CPU checks."""



# Pure functions and explicit inputs: A pure function makes its changing inputs visible and returns...
# Import jax.numpy for this computation.
import jax.numpy as jnp
# Function `predict(params, x)` implementing this stage's computation:
def predict(params, x):
    # Return `params['weight'] * x + params['bias']` to the caller.
    return params["weight"] * x + params["bias"]
# Initialize array `params` with explicit values and shape.
params = {"weight": jnp.array(2.), "bias": jnp.array(1.)}
# Initialize array `x` with explicit values and shape.
x = jnp.array([0., 1., 2.])
# Print the observed values to compare against the expected result.
print(predict(params, x))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))

# Figure data experiment
# Compute figure data for: Explicit parameters make predictions inspectable
# Create device-backed JAX array `changed`.
changed = {**params, 'bias': jnp.array(3.0)}
# Evaluate `visual_data` from the current inputs and state.
visual_data = {'kind': 'line', 'x': x.tolist(), 'xlabel': 'input', 'ylabel': 'prediction', 'series': [{'label': 'bias 1', 'y': predict(params, x).tolist()}, {'label': 'bias 3', 'y': predict(changed, x).tolist()}]}

# Experiment: Expose a mutable container alias
# Experiment — Expose a mutable container alias: The mutation concerns Python container bindings.
# Initialize array `container` with explicit values and shape.
container = {"weight": jnp.array(2.), "bias": jnp.array(1.)}
# Evaluate `alias` from the current inputs and state.
alias = container
# Initialize array `alias['weight']` with explicit values and shape.
alias["weight"] = jnp.array(9.)
# Verify contract: `float(container['weight']) == 9.0`.
assert float(container["weight"]) == 9.
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert float(params["weight"]) == 2.
# Print the observed values to compare against the expected result.
print("aliased container weight:", float(container["weight"]))

# Experiment: Replay two independent model configurations
# Experiment — Replay two independent model configurations: Independent configurations reveal the benefit of explicit inputs...
# Initialize array `first_params` with explicit values and shape.
first_params = {"weight": jnp.array(2.), "bias": jnp.array(1.)}
# Initialize array `second_params` with explicit values and shape.
second_params = {"weight": jnp.array(-1.), "bias": jnp.array(4.)}
# Run `predict` to compute `first_output`.
first_output = predict(first_params, x)
# Run `predict` to compute `second_output`.
second_output = predict(second_params, x)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(first_output, jnp.array([1., 3., 5.]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(second_output, jnp.array([4., 3., 2.]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(predict(first_params, x), first_output)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(predict(second_params, x), second_output)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Create new parameters with weight 3 while retaining the original bias.
# Initialize array `updated` with explicit values and shape.
updated = {**params, "weight": jnp.array(3.)}
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(predict(updated, x), jnp.array([1.,4.,7.]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert float(params["weight"]) == 2.0

# Reference practice: Update the bias without changing either parameter input
# Update the bias without changing either parameter input (Practice): The caller can evaluate both the old and new model without...
def with_bias(parameters, new_bias):
    # Return `{**parameters, 'bias': jnp.asarray(new_bias)}` to the caller.
    return {**parameters, "bias": jnp.asarray(new_bias)}
# Run `with_bias` to compute `changed_bias`.
changed_bias = with_bias(params, -2.)
# Verify contract: `changed_bias is not params`.
assert changed_bias is not params
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(predict(changed_bias, x), jnp.array([-2., 0., 2.]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))

# Reference practice: Return predictions and metrics together
# Return predictions and metrics together (Challenge): The second model has residuals [3, 0, -3], giving mean...
def evaluate(parameters, inputs, targets):
    # Run `predict` to compute `predictions`.
    predictions = predict(parameters, inputs)
    # Return `(predictions, jnp.mean((predictions - targets) ** 2))` to the caller.
    return predictions, jnp.mean((predictions - targets) ** 2)
# Initialize array `labels` with explicit values and shape.
labels = jnp.array([1., 3., 5.])
# Run `evaluate` to compute `(outputs, error)`.
outputs, error = evaluate(first_params, x, labels)
# Run `evaluate` to compute `(other_outputs, other_error)`.
other_outputs, other_error = evaluate(second_params, x, labels)
# Verify that the output tensor shape matches our prediction.
assert error.shape == ()
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(error, 0.)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(other_error, 6.)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(outputs, first_output)
# Print the observed values to compare against the expected result.
print("evaluation losses:", float(error), float(other_error))
print("PASS: arrays-03")
