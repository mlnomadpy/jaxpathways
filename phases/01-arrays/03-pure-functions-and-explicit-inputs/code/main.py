"""Pure functions and explicit inputs: worked experiments and reference solutions. CPU checks."""

# Step 1: Set up imports and input tensors
import jax.numpy as jnp

# Step 2: Apply the core JAX transformation
def predict(params, x):
    # Return `params['weight'] * x + params['bias']` to the caller.
    return params["weight"] * x + params["bias"]
# Construct `params` via `{"weight": jnp.array(2.), "bias": jnp.array(1.)}`
params = {"weight": jnp.array(2.), "bias": jnp.array(1.)}

# Step 3: Verify shapes and numerical invariants
x = jnp.array([0., 1., 2.])
# Print the observed values to compare against the expected result.
# Assert that `jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))`.
assert jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))

# Pure functions and explicit inputs: A pure function makes its changing inputs visible and returns...
# Import jax.numpy for this computation.
import jax.numpy as jnp
# Function `predict(params, x)` implementing this stage's computation:
def predict(params, x):
    # Return `params['weight'] * x + params['bias']` to the caller.
    return params["weight"] * x + params["bias"]
# Construct `params` via `{"weight": jnp.array(2.), "bias": jnp.array(1.)}`
params = {"weight": jnp.array(2.), "bias": jnp.array(1.)}
# Construct `x` via `jnp.array([0., 1., 2.])`
x = jnp.array([0., 1., 2.])
# Print the observed values to compare against the expected result.
print(predict(params, x))
# Assert that `jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))`.
assert jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))

# Figure data experiment
# Compute figure data for: Explicit parameters make predictions inspectable
# Create device-backed JAX array `changed`.
changed = {**params, 'bias': jnp.array(3.0)}
# Compute `visual_data` from `{'kind': 'line', 'x': x.tolist(), 'xlabel': 'input',...`
visual_data = {'kind': 'line', 'x': x.tolist(), 'xlabel': 'input', 'ylabel': 'prediction', 'series': [{'label': 'bias 1', 'y': predict(params, x).tolist()}, {'label': 'bias 3', 'y': predict(changed, x).tolist()}]}

# Experiment: Expose a mutable container alias
# Experiment — Expose a mutable container alias: The mutation concerns Python container bindings.
# Construct `container` via `{"weight": jnp.array(2.), "bias": jnp.array(1.)}`
container = {"weight": jnp.array(2.), "bias": jnp.array(1.)}
# Compute `alias` from `container`
alias = container
# Construct `alias["weight"]` via `jnp.array(9.)`
alias["weight"] = jnp.array(9.)
# Assert invariant `float(container["weight"]) == 9.` holds
assert float(container["weight"]) == 9.
# Assert invariant `float(params["weight"]) == 2.` holds
assert float(params["weight"]) == 2.
# Print the observed values to compare against the expected result.
print("aliased container weight:", float(container["weight"]))

# Experiment: Replay two independent model configurations
# Experiment — Replay two independent model configurations: Independent configurations reveal the benefit of explicit inputs...
# Construct `first_params` via `{"weight": jnp.array(2.), "bias": jnp.array(1.)}`
first_params = {"weight": jnp.array(2.), "bias": jnp.array(1.)}
# Construct `second_params` via `{"weight": jnp.array(-1.), "bias": jnp.array(4.)}`
second_params = {"weight": jnp.array(-1.), "bias": jnp.array(4.)}
# Run `predict` to compute `first_output`.
first_output = predict(first_params, x)
# Run `predict` to compute `second_output`.
second_output = predict(second_params, x)
# Assert that `jnp.allclose(first_output, jnp.array([1., 3., 5.]))`.
assert jnp.allclose(first_output, jnp.array([1., 3., 5.]))
# Assert that `jnp.allclose(second_output, jnp.array([4., 3., 2.]))`.
assert jnp.allclose(second_output, jnp.array([4., 3., 2.]))
# Assert that `jnp.allclose(predict(first_params, x), first_output)`.
assert jnp.allclose(predict(first_params, x), first_output)
# Assert that `jnp.allclose(predict(second_params, x), second_output)`.
assert jnp.allclose(predict(second_params, x), second_output)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Create new parameters with weight 3 while retaining the original bias.
# Construct `updated` via `{**params, "weight": jnp.array(3.)}`
updated = {**params, "weight": jnp.array(3.)}
# Assert that `jnp.allclose(predict(updated, x), jnp.array([1.,4.,7.]))`.
assert jnp.allclose(predict(updated, x), jnp.array([1.,4.,7.]))
# Assert invariant `float(params["weight"]) == 2.0` holds
assert float(params["weight"]) == 2.0

# Reference practice: Update the bias without changing either parameter input
# Update the bias without changing either parameter input (Practice): The caller can evaluate both the old and new model without...
def with_bias(parameters, new_bias):
    # Return `{**parameters, 'bias': jnp.asarray(new_bias)}` to the caller.
    return {**parameters, "bias": jnp.asarray(new_bias)}
# Run `with_bias` to compute `changed_bias`.
changed_bias = with_bias(params, -2.)
# Assert invariant `changed_bias is not params` holds
assert changed_bias is not params
# Assert that `jnp.allclose(predict(changed_bias, x), jnp.array([-2., 0., 2.]))`.
assert jnp.allclose(predict(changed_bias, x), jnp.array([-2., 0., 2.]))
# Assert that `jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))`.
assert jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))

# Reference practice: Return predictions and metrics together
# Return predictions and metrics together (Challenge): The second model has residuals [3, 0, -3], giving mean...
def evaluate(parameters, inputs, targets):
    # Run `predict` to compute `predictions`.
    predictions = predict(parameters, inputs)
    # Return `(predictions, jnp.mean((predictions - targets) ** 2))` to the caller.
    return predictions, jnp.mean((predictions - targets) ** 2)
# Construct `labels` via `jnp.array([1., 3., 5.])`
labels = jnp.array([1., 3., 5.])
# Run `evaluate` to compute `(outputs, error)`.
outputs, error = evaluate(first_params, x, labels)
# Run `evaluate` to compute `(other_outputs, other_error)`.
other_outputs, other_error = evaluate(second_params, x, labels)
# Check tensor shape invariant: `error.shape == ()`
assert error.shape == ()
# Assert that `jnp.allclose(error, 0.)`.
assert jnp.allclose(error, 0.)
# Assert that `jnp.allclose(other_error, 6.)`.
assert jnp.allclose(other_error, 6.)
# Assert that `jnp.allclose(outputs, first_output)`.
assert jnp.allclose(outputs, first_output)
# Print the observed values to compare against the expected result.
print("evaluation losses:", float(error), float(other_error))
print("PASS: arrays-03")
