"""Adam, clipping, and learning-rate schedules · learner-only exercises (3/3 contracts ready)"""

# Exercise: Compare global norm clipping with coordinate-wise clipping on \((3,4)\). Check which preserves the ratio of the coordinates.

# Inputs:
# - The gradient vector (3, 4) and the global and coordinate clipping bound used in the lesson.

# Outputs:
# - The globally clipped vector and coordinate-wise clipped vector, their coordinate ratios, and which operation preserves the original ratio.

# Invariants to check:
# - Global norm clipping scales both coordinates by one factor, while coordinate clipping can change their ratio.

# Hints:
# - Compute the vector norm before the global rescaling.
# - Apply coordinate clipping independently to each component and compare ratios.

import jax.numpy as jnp

def compare_global_and_coordinate_clipping(gradient, max_norm):
    # TODO: Compute both clipped vectors for the supplied gradient and report their coordinate ratios.
    raise NotImplementedError




# Catch schedule restart (Transfer / diagnosis):
# After five scheduled SGD steps with constant gradient \(1\), compare a retained optimizer state with a freshly initialized state for the next update.

# Inputs:
# - After five scheduled SGD steps with constant gradient \(1\), compare a retained optimizer state with a freshly initialized state for the next update.

# Outputs:
# - Both parameter updates and the scheduler/optimizer states.

# Invariants to check:
# - Only the optimizer-state reset differs between the two continuations.

# Hints:
# - The retained count has reached the final rate; a reset count starts at \(0\).

import jax.numpy as jnp
import optax

def practice_catch_schedule_restart(gradient, state, next_step):
    """Implement the catch schedule restart check."""
    # TODO: Apply the next scheduled update with retained state and with reset state after the same first five steps.
    raise NotImplementedError




# Check transformation order (Transfer / diagnosis):
# For gradient \((3,4)\), compare clipping before SGD at rate \(0.1\) with clipping after it. Write a decision memo explaining which quantity you intended to bound.

# Inputs:
# - For gradient \((3,4)\), compare clipping before SGD at rate \(0.1\) with clipping after it. Write a decision memo explaining which quantity you intended to bound.

# Outputs:
# - Both transformed gradients and their global norms.

# Invariants to check:
# - The declared bound names whether it applies to raw gradients or scaled updates.

# Hints:
# - The un-clipped SGD update already has norm \(0.5\), below the threshold \(1\).

import jax
import jax.numpy as jnp
import optax

def practice_check_transformation_order(gradient, learning_rate, max_norm):
    """Implement the check transformation order check."""
    # TODO: Apply clipping before the learning-rate transform and after it, recording both bounded quantities.
    raise NotImplementedError

