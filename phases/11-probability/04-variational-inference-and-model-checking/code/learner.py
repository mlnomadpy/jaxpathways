"""Variational inference and model checking · learner-only exercises (3/3 contracts ready)"""

# Exercise: Replace the target correlation by zero and rederive the optimal diagonal variances and minimum KL. Verify that the diagonal family can now represent the target exactly.

# Inputs:
# - The target mean and covariance after setting the target correlation to zero.

# Outputs:
# - The optimized diagonal-family variances, the minimum KL value, and whether the family represents this target exactly.

# Invariants to check:
# - The result uses the exercise’s KL direction; when the target covariance is diagonal, the optimized diagonal family can match its covariance exactly.

# Hints:
# - Recompute the target precision after setting the correlation to zero.
# - Evaluate the KL at the optimized diagonal covariance and compare the two covariance matrices.

import jax.numpy as jnp

def check_diagonal_family_at_zero_correlation(mean, target_covariance):
    # TODO: Set the target correlation to zero and evaluate the optimal diagonal approximation and minimum KL.
    raise NotImplementedError




# A correct diagonal is not a correct covariance (Transfer / diagnosis):
# Compare the variance of the sum of two target coordinates with that under the optimized diagonal approximation.

# Inputs:
# - A full target covariance and a projection direction.

# Outputs:
# - Projected variance under the full target and a diagonal approximation.

# Invariants to check:
# - The full covariance projection includes off-diagonal contributions.

# Hints:
# - Evaluate v.T @ covariance @ v for each covariance choice.

def compare_projected_uncertainty(covariance, direction, diagonal_variance):
    """Return the projected variances under full and diagonal covariance."""
    # TODO: compute both quadratic forms from the supplied values.
    raise NotImplementedError



# Repair the changed mean without hiding noise (Transfer / diagnosis):
# Replace the zero mean by the known curved mean in the stress fixture. Draw fresh noisy targets and compare residual spread to the stated noise scale.

# Inputs:
# - Inputs, observed targets, a known curved mean, a noise scale, and a random key.

# Outputs:
# - Observed and replicated discrepancy summaries before/after the mean repair.

# Invariants to check:
# - Use the same discrepancy on observed and replicated data.

# Hints:
# - Keep the noise scale fixed when adding the missing mean term.

def compare_curved_mean_repair(X, y, corrected_mean, noise_scale, key):
    """Return paired posterior predictive discrepancy summaries."""
    # TODO: evaluate the same residual statistic under both mean models.
    raise NotImplementedError
