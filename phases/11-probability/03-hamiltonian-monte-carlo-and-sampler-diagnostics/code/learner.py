"""Hamiltonian Monte Carlo and sampler diagnostics · learner-only exercises (3/3 contracts ready)"""

# Exercise: Increase the step size to 1.2 with the same leapfrog count. Record acceptance and positive energy errors. Explain why keeping more rejected samples is not a repair for unstable integration.

# Inputs:
# - A target mean and covariance, chain starts, and a deliberately large leapfrog step size.

# Outputs:
# - Per-chain acceptance summaries, signed energy errors, and a saved position trace.

# Invariants to check:
# - A rejected transition retains the previous state.
# - A large positive energy error lowers acceptance probability.
# - The diagnostic report names its R-hat variant and does not infer convergence from acceptance alone.

# Hints:
# - Keep momentum and acceptance random keys independent.
# - Inspect energy error and trace movement together.
# - Compare with the exact Gaussian target before interpreting a sampler statistic.

def summarize_unstable_trajectory(key, mean, covariance, starts, step_size):
    """Return transition summaries and the chain states for inspection."""
    # TODO: run the changed-step experiment and preserve per-chain evidence.
    raise NotImplementedError



# Catch chains trapped at different levels (Transfer / diagnosis):
# Construct four arrays with similar within-chain variation but means separated by four units. Verify that classical split R-hat flags disagreement.

# Inputs:
# - Four chains with similar within-chain variation and means separated by four units.

# Outputs:
# - The classical split R-hat and a check that it flags the separated chain means.

# Invariants to check:
# - Chain identity is retained through splitting; a classical split diagnostic is reported, not a rank-normalized modern diagnostic.

# Hints:
# - Split each chain into two halves before computing within- and between-chain variation.
# - Keep the chain offsets and within-chain noise on separate axes.

import jax.numpy as jnp

def check_classical_split_rhat(chains, split_rhat):
    # TODO: Construct the specified separated chains and evaluate the classical split diagnostic.
    raise NotImplementedError




# Verify a directional derivative (Transfer / diagnosis):
# At a nonzero position compare the autodiff directional derivative with the analytic precision-matrix expression. Then compare to a central difference.

# Inputs:
# - The target mean and precision matrix, the log-density gradient, a nonzero position, a direction, and a finite-difference step.

# Outputs:
# - The analytic precision-matrix directional derivative, autodiff directional derivative, and central-difference estimate.

# Invariants to check:
# - All three derivatives use the same nonzero position and direction; the central difference perturbs only along that direction.

# Hints:
# - Form the analytic gradient action from the precision matrix and centered position.
# - Compare a symmetric log-density difference with the autodiff directional derivative.

import jax
import jax.numpy as jnp

def compare_gaussian_directional_derivatives(log_density, mean, precision, point, direction, epsilon):
    # TODO: Compare the analytic precision action, autodiff derivative, and central difference.
    raise NotImplementedError

