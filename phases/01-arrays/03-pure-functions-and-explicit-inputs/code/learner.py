"""Pure functions and explicit inputs · learner-only exercises (3/3 contracts ready)"""

# Exercise: Create new parameters with weight \(3\) while retaining the original bias. Show that the original parameters still produce the original predictions.

# Inputs:
# - A parameter mapping with weight and bias.
# - A requested new weight and the prediction inputs.

# Outputs:
# - A new parameter mapping and predictions under both versions.

# Invariants to check:
# - The original mapping remains unchanged.
# - The retained field keeps its original value.
# - Original predictions can be replayed.

# Hints:
# - Return a new mapping instead of assigning into the input.
# - Pass the changed mapping explicitly to prediction.

import jax.numpy as jnp

def update_weight(params, new_weight):
    # TODO: return a new mapping with only the requested field changed
    raise NotImplementedError




# Update the bias without changing either parameter input (Practice):
# Implement `with_bias(params, new_bias)`, returning a new dictionary. Verify outputs for new bias \(-2\) and replay the original prediction.

# Inputs:
# - The original parameters, a new bias value, and the lesson prediction function.

# Outputs:
# - A new mapping and predictions for both parameter versions.

# Invariants to check:
# - The returned mapping is a distinct container.
# - Changing bias does not alter the original prediction.

# Hints:
# - Create a copy with one updated field.
# - Check identity and numerical replay separately.

import jax.numpy as jnp

def with_bias(params, new_bias, predict, inputs):
    # TODO: return a new mapping and compare both prediction paths
    raise NotImplementedError




# Return predictions and metrics together (Challenge):
# Write `evaluate(params, inputs, targets)` that returns predictions and a scalar MSE. Test two model configurations on targets \([1, 3, 5]\). Keep printing outside the function.

# Inputs:
# - Two parameter configurations, inputs, and targets.

# Outputs:
# - Predictions and scalar loss for both configurations.

# Invariants to check:
# - Evaluation returns numerical data without printing.
# - The loss reduces example errors to one scalar.
# - Parameter inputs are not mutated.

# Hints:
# - Keep reporting outside the function.
# - Use the supplied prediction function.

import jax.numpy as jnp

def evaluate(params, inputs, targets, predict):
    # TODO: return predictions and scalar mean squared error
    raise NotImplementedError

