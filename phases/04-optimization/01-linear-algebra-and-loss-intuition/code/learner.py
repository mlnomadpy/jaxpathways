"""Linear algebra and loss intuition · learner-only exercises (3/3 contracts ready)"""

# Exercise: Set the bias to zero. Predict each residual and calculate the loss by hand, then compare it with JAX.

# Inputs:
# - A feature matrix X, one weight per column, aligned target vector, and a scalar bias.

# Outputs:
# - Predictions with one scalar per row and a mean squared error.

# Invariants to check:
# - Prediction and target arrays stay one-dimensional and have identical shape; each row contributes one residual.

# Hints:
# - Check the row/feature axes before reducing.
# - For the zero-bias exercise, compute the row predictions first and keep the supplied targets aligned.

import jax.numpy as jnp

def linear_model_loss(weights, features, targets):
    """Predictions with one scalar per row and a mean squared error."""
    # TODO: Compute one prediction per feature row, form aligned residuals, and reduce their squares with a mean.
    raise NotImplementedError




# Add an unseen observation (Transfer / diagnosis):
# Extend \(X\) with \([2, -1]\), supply its target, and check predictions using ordinary arithmetic.

# Inputs:
# - Extend \(X\) with \([2, -1]\), supply its target, and check predictions using ordinary arithmetic.

# Outputs:
# - Prediction for the added row and a direct-arithmetic parity check.

# Invariants to check:
# - The added target stays paired to its row and the original training rows are unchanged.

# Hints:
# - The new target is the dot product plus bias.

import jax.numpy as jnp

def practice_add_an_unseen_observation(features, targets):
    """Implement the add an unseen observation check."""
    # TODO: Append exactly one feature row and target, evaluate its prediction, and compare with the direct row calculation.
    raise NotImplementedError




# Reject and repair a column target (Transfer / diagnosis):
# Write a loss that rejects mismatched prediction/target shapes. Demonstrate rejection and then restore the intended one-target-per-row format.

# Inputs:
# - Write a loss that rejects mismatched prediction/target shapes. Demonstrate rejection and then restore the intended one-target-per-row format.

# Outputs:
# - A shape-mismatch error followed by a correctly aligned scalar loss.

# Invariants to check:
# - The loss reduces B paired residuals rather than a B by B broadcast matrix.

# Hints:
# - Compare shapes before subtraction; reshape only this known scalar-target dataset.

import jax.numpy as jnp

def practice_reject_and_repair_a_column_target(predictions, targets):
    """Implement the reject and repair a column target check."""
    # TODO: Add an exact shape guard, demonstrate rejection of (B,) versus (B, 1), then reshape only the target to one value per row.
    raise NotImplementedError

