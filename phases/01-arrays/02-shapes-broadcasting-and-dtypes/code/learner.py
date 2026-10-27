"""Shapes, broadcasting, and dtypes · learner-only exercises (3/3 contracts ready)"""

# Exercise: Add a different scalar offset to each row using offsets \([100., 200.]\). Make its shape explicit.

# Inputs:
# - A `(2, 3)` batch and one scalar row offset per observation.

# Outputs:
# - An explicit offset shape and the row-offset result.

# Invariants to check:
# - The feature axis remains length three.
# - Each row receives its own scalar offset.
# - The batch shape is unchanged.

# Hints:
# - Make the singleton feature dimension explicit.
# - Keep row offsets distinct from feature biases.

import jax.numpy as jnp

def add_row_offsets(batch, row_offsets):
    # TODO: arrange offsets to broadcast over features
    offsets_by_row = ...
    result = ...
    return result, offsets_by_row




# Use both kinds of offsets together (Practice):
# Add the feature bias and row offsets to the original batch. Derive every row before running, then check with an explicitly constructed array.

# Inputs:
# - The lesson batch, feature bias, and two row offsets.

# Outputs:
# - The combined table with both offsets applied.

# Invariants to check:
# - Feature bias and row offsets have different axes.
# - The output retains the original batch shape.

# Hints:
# - Give each offset an explicit shape before adding.
# - Verify every row and feature.

import jax.numpy as jnp

def combine_offsets(batch, feature_bias, row_offsets):
    # TODO: apply feature and observation offsets with distinct shapes
    raise NotImplementedError




# Reject an aligned-loss contract violation (Challenge):
# Write an MSE function that requires prediction and target shapes to match. Test zero loss for matching vectors and rejection of a column target. Explain why an automatic reshape might hide a data-ordering problem.

# Inputs:
# - Prediction and target arrays with matching vector shapes, then a column-shaped target.

# Outputs:
# - A scalar MSE for aligned vectors and a descriptive mismatch error.

# Invariants to check:
# - Shape metadata is checked before subtraction.
# - Do not silently reshape or broadcast mismatched data.

# Hints:
# - Compare shape tuples before arithmetic.
# - Use a clear exception for invalid alignment.

import jax.numpy as jnp

def aligned_mse(prediction, target):
    # TODO: reject shape mismatch before computing scalar mean squared error
    raise NotImplementedError

