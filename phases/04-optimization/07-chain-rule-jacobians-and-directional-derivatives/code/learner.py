"""The chain rule, Jacobians, and directions · learner-only exercises (3/3 contracts ready)"""

# Exercise: Replace the scalar loss by \(L(x)=f_0(x)+2f_1(x)\). Predict its gradient at \((2,3)\) before using `grad`.

# Inputs:
# - A vector input, a vector-valued function, and output weights.

# Outputs:
# - The gradient of the weighted scalar output.

# Invariants to check:
# - The VJP maps output sensitivities back to the input dimension.

# Hints:
# - Form the scalar objective explicitly before differentiating.
# - Use an independent matrix product to check the changed output dimension.

import jax
import jax.numpy as jnp

def weighted_output_gradient(point, weights):
    """The gradient of the weighted scalar output."""
    # TODO: Form the weighted scalar output and check its gradient/VJP against the requested input and output shapes.
    raise NotImplementedError




# Find the broken derivative path (Transfer / diagnosis):
# A colleague wraps the first input in `jax.lax.stop_gradient`. The forward outputs are unchanged. Show why the gradient no longer matches the intended mathematical function.

# Inputs:
# - A colleague wraps the first input in `jax.lax.stop_gradient`. The forward outputs are unchanged. Show why the gradient no longer matches the intended mathematical function.

# Outputs:
# - Forward-output parity and the two derivative results.

# Invariants to check:
# - Stopping a gradient does not change the forward value but changes the derivative path.

# Hints:
# - Compare the derivative of the sum of outputs with \(J^\mathsf{T}(1,1)\).

import jax
import jax.numpy as jnp

def practice_find_the_broken_derivative_path(point):
    """Implement the find the broken derivative path check."""
    # TODO: Differentiate the intended composition and a version with stop_gradient on the first input; compare both with an independent derivative.
    raise NotImplementedError




# Transfer to unequal dimensions (Transfer / diagnosis):
# For a matrix \(A\in\mathbb{R}^{3\times2}\), check that the VJP of \(Ax\) maps an output sensitivity of length \(3\) into a vector of length \(2\).

# Inputs:
# - For a matrix \(A\in\mathbb{R}^{3\times2}\), check that the VJP of \(Ax\) maps an output sensitivity of length \(3\) into a vector of length \(2\).

# Outputs:
# - A length-two input cotangent checked against the matrix transpose product.

# Invariants to check:
# - The cotangent shape follows the output, and the VJP result follows the input.

# Hints:
# - The reference is \(A^\mathsf{T}c\), not \(Ac\).

import jax
import jax.numpy as jnp

def practice_transfer_to_unequal_dimensions(matrix, input, output_sensitivity):
    """Implement the transfer to unequal dimensions check."""
    # TODO: Build a 3-by-2 matrix product and apply a VJP to a length-three cotangent.
    raise NotImplementedError

