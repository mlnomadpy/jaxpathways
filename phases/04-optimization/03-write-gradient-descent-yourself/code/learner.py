"""Write gradient descent yourself · learner-only exercises (3/3 contracts ready)"""

# Exercise: Run ten steps with learning rate \(1.1\). Predict whether the error contracts before executing; show evidence of divergence.

# Inputs:
# - An initial scalar parameter, step size, and fixed number of updates.

# Outputs:
# - The parameter/loss history for the requested update schedule.

# Invariants to check:
# - Each update uses the gradient evaluated at the current parameter; report divergence rather than clipping it away.

# Hints:
# - Use value_and_grad to measure the loss and gradient at one point.
# - Check whether the update changes the loss in the direction predicted by curvature.

import jax
import jax.numpy as jnp
import optax

def gradient_descent(initial, rate, steps):
    """The parameter/loss history for the requested update schedule."""
    # TODO: Record parameter and loss after each explicit gradient-descent transition, including the high-rate case.
    raise NotImplementedError




# Transfer the stability analysis (Transfer / diagnosis):
# For \(4(w-3)^2\), derive a stable rate and verify convergence from zero.

# Inputs:
# - For \(4(w-3)^2\), derive a stable rate and verify convergence from zero.

# Outputs:
# - A finite loss history that contracts under the derived rate.

# Invariants to check:
# - The rate is checked against the new curvature rather than copied from the earlier example.

# Hints:
# - The error multiplier is \(1-8\eta\); use \(\eta=0.1\).

import jax.numpy as jnp

def practice_transfer_the_stability_analysis(rate):
    """Implement the transfer the stability analysis check."""
    # TODO: Derive the update multiplier for the changed quadratic, select a stable rate, and retain loss history from zero.
    raise NotImplementedError




# Reproduce an uphill update and repair it (Transfer / diagnosis):
# Show how adding the gradient increases loss, then verify subtraction at the same starting point.

# Inputs:
# - Show how adding the gradient increases loss, then verify subtraction at the same starting point.

# Outputs:
# - Both candidate losses and an explicit direction check.

# Invariants to check:
# - The candidates share the same starting point and learning rate.

# Hints:
# - The derivative at zero is \(-4\).

import jax
import jax.numpy as jnp

def practice_reproduce_an_uphill_update_and_repair_it(initial, rate):
    """Implement the reproduce an uphill update and repair it check."""
    # TODO: Apply a plus-gradient step and a minus-gradient step from the same initial value, then compare their losses.
    raise NotImplementedError

