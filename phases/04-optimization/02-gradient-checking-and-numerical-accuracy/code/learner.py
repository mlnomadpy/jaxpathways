"""Gradient checking and numerical accuracy · learner-only exercises (3/3 contracts ready)"""

# Exercise: Check the gradient again at \(w=[2., -1.]\). Explain why this checks a minimum but does not prove correctness everywhere.

# Inputs:
# - A differentiable scalar objective and a vector of evaluation points.

# Outputs:
# - A report comparing autodiff with central finite differences at multiple step sizes.

# Invariants to check:
# - Compare the same scalar objective and coordinate at every step size; finite differences are estimates, not a proof for all inputs.

# Hints:
# - Check a smooth point, and compare several perturbation scales.
# - Keep the analytic derivative independent from the finite-difference implementation.

import jax
import jax.numpy as jnp

def gradient_check_report(point, epsilons):
    """A report comparing autodiff with central finite differences at multiple step sizes."""
    # TODO: Evaluate autodiff and central-difference gradients at the supplied point across the requested perturbations.
    raise NotImplementedError




# Move to a different smooth objective (Transfer / diagnosis):
# Check the derivative of `sum(sin(z))` at \([0.2, -0.7]\) using cosine and central differences.

# Inputs:
# - Check the derivative of `sum(sin(z))` at \([0.2, -0.7]\) using cosine and central differences.

# Outputs:
# - Autodiff, analytic derivative, and finite-difference values for both coordinates.

# Invariants to check:
# - All methods evaluate the same scalar objective at the requested point.

# Hints:
# - Use cosine as the analytic reference; here truncation error also matters.

import jax
import jax.numpy as jnp

def practice_move_to_a_different_smooth_objective(point):
    """Implement the move to a different smooth objective check."""
    # TODO: Differentiate the sine-sum objective at both supplied coordinates and compare each result with cosine and central differences.
    raise NotImplementedError




# Catch the wrong objective despite correct gradients (Transfer / diagnosis):
# A model is meant to use both coordinates. Show that an objective omitting the second coordinate has a correct derivative for itself but fails the intended analytic reference. Repair it.

# Inputs:
# - A model is meant to use both coordinates. Show that an objective omitting the second coordinate has a correct derivative for itself but fails the intended analytic reference. Repair it.

# Outputs:
# - A counterexample for the incomplete objective and a passing repaired comparison.

# Invariants to check:
# - The derivative is checked against the intended function, not only the function that produced it.

# Hints:
# - Differentiate \((z_0-2)^2\) and compare both coordinates.

import jax
import jax.numpy as jnp

def practice_catch_the_wrong_objective_despite_correct_gradients(point):
    """Implement the catch the wrong objective despite correct gradients check."""
    # TODO: Construct the objective that drops the second coordinate, compare its gradient to the intended two-coordinate reference, then restore the missing dependence.
    raise NotImplementedError

