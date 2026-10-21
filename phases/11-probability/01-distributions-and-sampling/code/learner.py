"""Distributions and sampling · learner-only exercises (3/3 contracts ready)"""

# Exercise: Change the sensor to mean \(-1\), standard deviation \(0.5\), and draw \(30000\) readings from a new key. Predict the variance and the standard error of the sample mean, then verify both moments.

# Inputs:
# - A seed for a fresh observation key; mean and standard deviation of the sensor.

# Outputs:
# - One reading vector, its sample variance, and the standard error of its sample mean.

# Invariants to check:
# - The input key is not reused for another random draw.
# - The reported standard error uses the number of readings, not the variance alone.

# Hints:
# - Split a key before sampling.
# - For independent readings, standard error scales as standard deviation divided by the square root of the sample count.
# - Compare empirical moments with finite-sample tolerances rather than exact equality.

def compare_sample_moments(key, mean, scale, count):
    """Return readings, sample variance, and standard error of the mean."""
    # TODO: split the key, draw readings, and compute both summaries.
    raise NotImplementedError



# Detect a missing normalization (Transfer / diagnosis):
# Compare scales \(1\) and \(2\) at their common mean. Show why a residual-only objective incorrectly ties them.

# Inputs:
# - The shared mean and two normal scales, 1 and 2.

# Outputs:
# - The two log densities at the common mean and the change in the normalized density across scales.

# Invariants to check:
# - The residual-only term is identical at the shared mean, while the full log densities differ because scale changes the normalizer.

# Hints:
# - Evaluate both normalized log densities at the same zero-residual point.
# - Compare the full expression with the residual-only term.

import jax.numpy as jnp

def compare_normal_scales_at_common_mean(normal_logpdf):
    # TODO: Evaluate scales one and two at their common mean and expose the residual-only tie.
    raise NotImplementedError




# Demonstrate key ownership (Transfer / diagnosis):
# Produce two separately owned standard-normal draws and show replay of the first. Explain why replay alone does not test independence.

# Inputs:
# - An explicit JAX key, two child-work labels, and a changed seed.

# Outputs:
# - Two reproducible but independent random arrays.

# Invariants to check:
# - Reusing a key replays its random values; splitting creates independent child streams.

# Hints:
# - Split once for each random operation. Verify replay and changed-seed behavior separately.

def check_key_ownership(key, seed):
    """Return two draws and a replay/changed-seed comparison."""
    # TODO: allocate child keys and record the random-stream checks.
    raise NotImplementedError
