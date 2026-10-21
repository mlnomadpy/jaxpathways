"""Memory, rematerialization, and optimization tradeoffs · learner-only exercises (2/2 contracts ready)"""

# Exercise: Use the direction \(v=(1,\ldots,1)\) and central differences to independently check a directional derivative of the original objective at the existing input. Compare it with the gradient dot direction and report the perturbation.

# Inputs:
# - The original objective, its existing evaluation point, an all-ones direction, and the finite-difference perturbation.

# Outputs:
# - The central-difference directional derivative, gradient dot direction, and their discrepancy at the stated perturbation.

# Invariants to check:
# - The derivative uses the original objective and existing input; the finite-difference direction is the requested all-ones vector.

# Hints:
# - Evaluate the scalar objective at the point plus and minus epsilon times the direction.
# - Compute the autodiff gradient once and take its dot product with the same direction.

import jax
import jax.numpy as jnp

def check_directional_derivative(objective, point, direction, epsilon):
    # TODO: Compare the directional central difference with the gradient dot direction.
    raise NotImplementedError




# Check higher-order differentiation (Transfer):
# Compare Hessian-vector products of the ordinary and rematerialized objectives along a nonuniform direction. Explain why equal first derivatives alone would be a weaker test.

# Inputs:
# - Compare Hessian-vector products of the ordinary and rematerialized objectives along a nonuniform direction. Explain why equal first derivatives alone would be a weaker test.

# Outputs:
# - Both HVP vectors and an independent parity check.

# Invariants to check:
# - The comparison uses the same point, direction, and objective for both differentiation paths.

# Hints:
# - Apply jvp to each gradient function with the same primal point and tangent.

import jax
import jax.numpy as jnp
import numpy as np
from jax.ad_checkpoint import print_saved_residuals

def practice_check_higher_order_differentiation(objective, point, direction):
    """Implement the check higher-order differentiation check."""
    # TODO: Compare ordinary and rematerialized Hessian-vector products along the nonuniform direction.
    raise NotImplementedError

