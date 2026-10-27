"""Vectorize trajectories and scan time · learner-only exercises (3/3 contracts ready)"""

# Exercise: Add a fourth initial condition, verify the changed output shape and exponential values, then verify that reordering initial conditions reorders rows only.

# Inputs:
# - Rate 0.7, a four-element initial-condition vector, and the lesson time grid with 41 points.
# - A permutation of the four initial conditions.

# Outputs:
# - Batched trajectory with shape (4, 41), checked against initial * exp(-rate*time).
# - Reordered trajectories whose rows follow the changed initial-condition order.

# Invariants to check:
# - Each initial condition scales the same scalar decay trajectory.
# - Reordering inputs changes rows only; time columns remain unchanged.
# - The batch/time axes have shape (4, 41).

# Hints:
# - Vectorize over initial values while preserving the time axis.
# - Compare all values with the analytic exponential and apply the same row permutation.

import numpy as np
import jax
import jax.numpy as jnp

def verify_batched_trajectories(solve, initial_values, rate, times, permutation):
    # TODO: evaluate four starts, check shape and exponential values, then row reordering.
    raise NotImplementedError("Check analytic trajectories and preserve batch/time axes")



# Save only what the objective needs (Practice):
# Implement a final-state-only solve and compare it with the final column of the saved batch. Explain which allocation was removed.

# Inputs:
# - A vector of four initial conditions, fixed scalar rate, and saved time grid.

# Outputs:
# - Final-state-only values and the final column from the trajectory solve.
# - A short explanation naming the intermediate-trajectory allocation no longer retained.

# Invariants to check:
# - Both methods agree at the final time.
# - The final-only solve avoids retaining intermediate states.
# - Final-only evaluation carries the state without storing every time point.

# Hints:
# - Use the same integrator, step count, and initial values in both paths.
# - Compare arrays by shape and values, not a single selected element.

import numpy as np
import jax
import jax.numpy as jnp

def compare_final_only(initial_values, rate, dt, steps):
    # TODO: implement a scan that returns only its carry, then compare with saved states.
    raise NotImplementedError("Compare final-only integration with the saved trajectory")



# Pairing versus Cartesian products (Challenge):
# Build the full grid of three rates and three initial values. Show that its diagonal agrees with the paired-rate experiment.

# Inputs:
# - Three rates and three initial values for the scalar decay model.

# Outputs:
# - Paired outputs and a full rate-by-initial-value grid.

# Invariants to check:
# - Paired evaluation returns one output per matching input pair.
# - Cartesian evaluation returns a (3, 3) grid; its diagonal matches paired results.

# Hints:
# - Use vmap over both arguments for pairing.
# - Nest vectorization over rates and initial values for the Cartesian grid.

import numpy as np
import jax
import jax.numpy as jnp

def compare_pairing_and_grid(solve, rates, initials):
    # TODO: construct paired and Cartesian evaluations.
    # Return both arrays and an explicit diagonal consistency check.
    raise NotImplementedError("Distinguish paired mapping from a Cartesian product")
