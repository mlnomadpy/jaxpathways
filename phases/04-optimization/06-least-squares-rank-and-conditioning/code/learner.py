"""Least squares, rank, and conditioning · learner-only exercises (3/3 contracts ready)"""

# Exercise: Add a constant bias column to inputs \((-1,0,1)\) and fit targets \((-1,1,3)\). Recover slope \(2\) and bias \(1\).

# Inputs:
# - A design matrix, target vector, and an explicit constant feature for the intercept.

# Outputs:
# - Least-squares coefficients and predictions.

# Invariants to check:
# - The coefficient order is documented; predictions are recomputed from the design matrix.

# Hints:
# - Use a linear solve or least-squares routine, not an explicitly formed inverse.
# - Compare prediction parity when only feature units change.

import jax.numpy as jnp

def fit_with_bias(features, targets):
    """Least-squares coefficients and predictions."""
    # TODO: Build the bias-augmented design matrix and solve the least-squares system without forming its inverse.
    raise NotImplementedError




# Change feature units (Transfer / diagnosis):
# Multiply the first feature of \(X\) by \(100\). Predict the new weights and compare predictions.

# Inputs:
# - Multiply the first feature of \(X\) by \(100\). Predict the new weights and compare predictions.

# Outputs:
# - Original/scaled coefficients and prediction parity.

# Invariants to check:
# - The feature transform is inverted when interpreting its coefficient; predictions use matched rows.

# Hints:
# - The first weight must divide by \(100\) to represent the same model.

import jax.numpy as jnp

def practice_change_feature_units(features, targets):
    """Implement the change feature units check."""
    # TODO: Scale only the first feature column by 100, refit, and compare predictions before comparing coefficients.
    raise NotImplementedError




# Explain the squared conditioning (Transfer / diagnosis):
# Compare the condition numbers of the diagonal matrix with entries \((1,0.1)\) and its Gram matrix. Why is a solver preferable to forming the inverse?

# Inputs:
# - Compare the condition numbers of the diagonal matrix with entries \((1,0.1)\) and its Gram matrix. Why is a solver preferable to forming the inverse?

# Outputs:
# - Both condition numbers and a solve result for a declared right-hand side.

# Invariants to check:
# - The Gram conditioning comparison uses the same nonzero singular directions.

# Hints:
# - The diagonal entries are singular values here; squaring them changes the ratio from \(10\) to \(100\).

import jax.numpy as jnp

def practice_explain_the_squared_conditioning(matrix):
    """Implement the explain the squared conditioning check."""
    # TODO: Compute condition numbers for the diagonal matrix and its Gram matrix, then compare solve-based and inverse-based procedures.
    raise NotImplementedError

