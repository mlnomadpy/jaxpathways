"""Curvature, Hessians, and learning rates · learner-only exercises (3/3 contracts ready)"""

# Exercise: For curvatures \((2,8)\), derive the stable interval and check one step at rate \(0.1\) from \((1,1)\).

# Inputs:
# - A point, positive quadratic curvatures, and a proposed gradient-descent rate.

# Outputs:
# - One updated point and an independently evaluated loss change.

# Invariants to check:
# - Stability is determined by the largest curvature; check the actual update at the supplied rate.

# Hints:
# - For a quadratic, inspect each coordinate multiplier after one step.
# - Use a nonzero direction when checking the Hessian-vector product.

import jax.numpy as jnp

def quadratic_step(point, curvatures, rate):
    """One updated point and an independently evaluated loss change."""
    # TODO: Use each curvature to check the proposed update multiplier and evaluate one changed-rate step.
    raise NotImplementedError




# Reject the zero-gradient shortcut (Transfer / diagnosis):
# At the origin of \(S(w)=w_0^2-w_1^2\), show zero gradient and both a lower and higher nearby value.

# Inputs:
# - At the origin of \(S(w)=w_0^2-w_1^2\), show zero gradient and both a lower and higher nearby value.

# Outputs:
# - One stationary point with a lower and a higher nearby objective value.

# Invariants to check:
# - A zero gradient is not accepted as evidence of a local minimum.

# Hints:
# - Move by \(0.1\) along each axis separately.

import jax
import jax.numpy as jnp

def practice_reject_the_zero_gradient_shortcut(point, curvatures):
    """Implement the reject the zero-gradient shortcut check."""
    # TODO: Evaluate the saddle objective at the origin and on both coordinate-axis perturbations.
    raise NotImplementedError




# Transfer to a nonquadratic function (Transfer / diagnosis):
# For \(F(w)=\sum_i e^{w_i}\), verify a Hessian-vector product at \((0,\log 2)\) along \((1,3)\).

# Inputs:
# - For \(F(w)=\sum_i e^{w_i}\), verify a Hessian-vector product at \((0,\log 2)\) along \((1,3)\).

# Outputs:
# - A finite HVP matching the elementwise analytic curvature action.

# Invariants to check:
# - The product uses the full nonuniform direction and the specified point.

# Hints:
# - The analytic Hessian is diagonal with entries \(e^{w_i}\).

import jax
import jax.numpy as jnp

def practice_transfer_to_a_nonquadratic_function(point, direction):
    """Implement the transfer to a nonquadratic function check."""
    # TODO: Compute a Hessian-vector product for the exponential sum at the requested point and direction.
    raise NotImplementedError

