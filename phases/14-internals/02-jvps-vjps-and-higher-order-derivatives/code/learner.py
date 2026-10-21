"""JVPs, VJPs, and higher-order derivatives · learner-only exercises (3/3 contracts ready)"""

# Exercise: Repeat the Jacobian, JVP, VJP and adjoint checks at \(x=(-0.2,0.8)\) with \(v=(-0.5,1.3)\) and \(u=(1.1,-0.4)\). Explain which quantities belong to input and output space.

# Inputs:
# - Vector function mapping R² to R².
# - Changed base point x=(-0.2, 0.8), tangent v=(-0.5, 1.3), and output cotangent u=(1.1, -0.4).

# Outputs:
# - Jacobian, JVP, VJP, and adjoint inner products at the changed point.

# Invariants to check:
# - JVP belongs to output space and VJP belongs to input space.
# - The two adjoint inner products agree within numerical tolerance.

# Hints:
# - Use the changed primal and seeds for every transform.
# - Check each result shape before comparing scalar inner products.

import jax
import jax.numpy as jnp

def inspect_changed_linearization(function, x, v, u):
    # TODO: compute Jacobian, JVP, VJP, and both adjoint inner products.
    raise NotImplementedError("Check primal, tangent, and cotangent spaces separately")



# Test linearity at a fixed point (Practice):
# Check the JVP of \(2v-w\) against the same combination of separate JVPs. Then show why changing the base point is a different operation.

# Inputs:
# - A fixed base point, tangent directions v and w, and f(v,w)=2v-w.

# Outputs:
# - JVP for a combined direction and sum of separate JVPs.

# Invariants to check:
# - The base point remains fixed for the linearity comparison.
# - Changing the base point is not tangent linearity.

# Hints:
# - Use jax.jvp at one shared primal input.
# - Compare one combined seed with the linear combination of separate seeds.

import jax
import jax.numpy as jnp

def check_jvp_linearity(function, base, v, w):
    # TODO: compare combined and separate JVPs at the same base.
    raise NotImplementedError("Test tangent linearity without changing the primal point")



# Check curvature symmetry without a full Hessian (Challenge):
# For two directions \(v,w\), verify \(w^\mathsf{T}Hv=v^\mathsf{T}Hw\), then verify one product with finite differences.

# Inputs:
# - A scalar-valued twice-differentiable function, base point, and directions v,w.

# Outputs:
# - Both mixed Hessian directional products and a finite-difference check.

# Invariants to check:
# - Mixed products agree within a stated tolerance.
# - Finite differences independently check one Hessian-vector product.

# Hints:
# - Compute Hessian-vector products with forward-over-reverse or jvp(grad).
# - Perturb the gradient along one direction for the finite-difference comparison.

import jax
import jax.numpy as jnp

def check_hessian_symmetry(function, point, v, w, epsilon):
    # TODO: compare the two mixed products and finite-difference one product.
    raise NotImplementedError("Verify curvature symmetry without materializing the Hessian")
