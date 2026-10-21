"""Build a Bayesian regression · learner-only exercises (3/3 contracts ready)"""

# Exercise: Strengthen the prior by changing its scale from two to one half. Predict the direction of change in the weight norm and check it. Explain why a tighter posterior need not mean a better model.

# Inputs:
# - A design matrix with an explicit intercept column, outcomes, known noise scale, and two prior scales.

# Outputs:
# - Posterior mean and covariance for both prior choices.

# Invariants to check:
# - The covariance is symmetric positive definite.
# - The stronger zero-centered prior cannot increase the posterior mean norm in this fixed fixture.

# Hints:
# - Build posterior precision from prior and likelihood precision.
# - Solve linear systems instead of explicitly inverting a matrix.
# - Interpret the changed posterior separately from predictive model quality.

def compare_prior_strength(X, y, noise_scale, prior_scales):
    """Return posterior moments under two prior scales."""
    # TODO: solve both Gaussian precision systems and compare their means.
    raise NotImplementedError



# Move away from centered inputs (Transfer / diagnosis):
# Use design rows \((1,0),(1,1),(1,2)\). Compare the JAX result with a host NumPy linear solve and show the posterior covariance is not diagonal.

# Inputs:
# - The requested design rows, outcomes, known noise scale, and prior scale.

# Outputs:
# - The JAX posterior mean/covariance, host NumPy solve results, and an off-diagonal covariance check.

# Invariants to check:
# - The two solves use the same precision system; the shifted design yields a full posterior covariance with a nonzero off-diagonal entry.

# Hints:
# - Construct the precision matrix in float64 on the host for an independent solve.
# - Check the covariance off-diagonal directly instead of reducing it to a diagonal summary.

import numpy as np

def compare_posterior_with_host_solve(posterior, outcomes, noise_scale, prior_scale):
    # TODO: Solve the requested design case with JAX and independent host NumPy arithmetic.
    raise NotImplementedError




# Verify total predictive variance (Transfer / diagnosis):
# Draw posterior weights and independent observation noise. Compare empirical prediction variance at input zero to the analytic observation variance.

# Inputs:
# - Posterior weight mean/covariance, observation noise scale, an explicit random key, and the requested sample count.

# Outputs:
# - Empirical prediction variance from sampled weights plus observation noise and analytic observation variance at input zero.

# Invariants to check:
# - The sampled observation variance is compared with the analytic predictive variance, including the noise contribution.

# Hints:
# - Draw parameter samples from the full posterior covariance.
# - Use a separate subkey for observation noise and evaluate the prediction at input zero.

import jax
import jax.numpy as jnp

def compare_sampled_and_analytic_observation_variance(mean, covariance, noise_scale, key, sample_count):
    # TODO: Sample posterior weights and independent observation noise, then compare variances at x=0.
    raise NotImplementedError

