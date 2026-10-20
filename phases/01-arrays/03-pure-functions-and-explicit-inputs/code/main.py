"""Pure functions and explicit inputs: worked experiments and reference solutions. CPU checks."""

# Import JAX array operations
import jax.numpy as jnp

# Define a prediction from explicit parameters
def predict(params, x):
    return params["weight"] * x + params["bias"]
params = {"weight": jnp.array(2.), "bias": jnp.array(1.)}

# Evaluate the predictor on three inputs
x = jnp.array([0., 1., 2.])
assert jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))

# Pure functions and explicit inputs: A pure function makes its changing inputs visible and returns...
import jax.numpy as jnp
def predict(params, x):
    return params["weight"] * x + params["bias"]
params = {"weight": jnp.array(2.), "bias": jnp.array(1.)}
x = jnp.array([0., 1., 2.])
print(predict(params, x))
assert jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))

# Figure data experiment
# Create device-backed JAX array `changed`.
changed = {**params, 'bias': jnp.array(3.0)}
visual_data = {'kind': 'line', 'x': x.tolist(), 'xlabel': 'input', 'ylabel': 'prediction', 'series': [{'label': 'bias 1', 'y': predict(params, x).tolist()}, {'label': 'bias 3', 'y': predict(changed, x).tolist()}]}

# Experiment: Expose a mutable container alias
# Experiment — Expose a mutable container alias: The mutation concerns Python container bindings.
container = {"weight": jnp.array(2.), "bias": jnp.array(1.)}
alias = container
alias["weight"] = jnp.array(9.)
assert float(container["weight"]) == 9.
assert float(params["weight"]) == 2.
print("aliased container weight:", float(container["weight"]))

# Experiment: Replay two independent model configurations
# Experiment — Replay two independent model configurations: Independent configurations reveal the benefit of explicit inputs...
first_params = {"weight": jnp.array(2.), "bias": jnp.array(1.)}
second_params = {"weight": jnp.array(-1.), "bias": jnp.array(4.)}
first_output = predict(first_params, x)
second_output = predict(second_params, x)
assert jnp.allclose(first_output, jnp.array([1., 3., 5.]))
assert jnp.allclose(second_output, jnp.array([4., 3., 2.]))
assert jnp.allclose(predict(first_params, x), first_output)
assert jnp.allclose(predict(second_params, x), second_output)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Create new parameters with weight 3 while retaining the original bias.
updated = {**params, "weight": jnp.array(3.)}
assert jnp.allclose(predict(updated, x), jnp.array([1.,4.,7.]))
assert float(params["weight"]) == 2.0

# Reference practice: Update the bias without changing either parameter input
# Update the bias without changing either parameter input (Practice): The caller can evaluate both the old and new model without...
def with_bias(parameters, new_bias):
    return {**parameters, "bias": jnp.asarray(new_bias)}
changed_bias = with_bias(params, -2.)
assert changed_bias is not params
assert jnp.allclose(predict(changed_bias, x), jnp.array([-2., 0., 2.]))
assert jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))

# Reference practice: Return predictions and metrics together
# Return predictions and metrics together (Challenge): The second model has residuals [3, 0, -3], giving mean...
def evaluate(parameters, inputs, targets):
    predictions = predict(parameters, inputs)
    return predictions, jnp.mean((predictions - targets) ** 2)
labels = jnp.array([1., 3., 5.])
outputs, error = evaluate(first_params, x, labels)
other_outputs, other_error = evaluate(second_params, x, labels)
# Check tensor shape invariant: `error.shape == ()`
assert error.shape == ()
assert jnp.allclose(error, 0.)
assert jnp.allclose(other_error, 6.)
assert jnp.allclose(outputs, first_output)
print("evaluation losses:", float(error), float(other_error))
print("PASS: arrays-03")
