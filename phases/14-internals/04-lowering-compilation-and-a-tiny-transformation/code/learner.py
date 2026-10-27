"""Lowering, compilation, and a tiny transformation · learner-only exercises (3/3 contracts ready)"""

# Exercise: Trace a new polynomial using only supported primitives, verify its interpreted derivative at two points, and compile that transformed computation independently.

# Inputs:
# - A three-element vector x, a second point x+0.6, and a three-element tangent direction.
# - A polynomial using only the interpreter-supported add, multiply, negation, and reduction operations.

# Outputs:
# - Scalar primal and tangent at both points from the interpreted transformation.
# - Matching scalar primal/tangent pairs from a separately compiled transformed computation.

# Invariants to check:
# - The interpreted derivative agrees with an independent directional check at both points.
# - The compiled computation preserves the traced vector input signature.

# Hints:
# - Inspect the primitives emitted by jax.make_jaxpr before interpreting the program.
# - Lower and compile the transformed function separately from the interpreted check.

import jax
import jax.numpy as jnp

def inspect_supported_polynomial(points, direction):
    # TODO: trace a supported vector polynomial, interpret its tangent at both points, then compile it.
    raise NotImplementedError("Keep the supported primitive set and interpreted/compiled checks explicit")



# Make a constant into a differentiable input (Practice):
# Replace the captured offset by an explicit argument, seed that argument with a nonzero direction, and verify its contribution to the total tangent.

# Inputs:
# - Three-element primal vector x and captured offset vector with shape (3,).
# - A nonzero direction for x and a nonzero three-element direction for the explicit offset argument.

# Outputs:
# - JVP tangent including the contribution from the newly explicit offset argument.

# Invariants to check:
# - A closed-over offset receives no tangent; an explicit argument receives its supplied tangent.
# - The explicit-argument JVP matches an independent directional derivative.

# Hints:
# - Compare the closed-over function signature with the explicit signature.
# - Seed the added argument with a nonzero tangent.

import jax
import jax.numpy as jnp

def compare_captured_and_explicit(value, offset, value_tangent, offset_tangent):
    # TODO: trace the explicit offset as an input and include both seed directions.
    raise NotImplementedError("Expose the vector offset as a differentiable argument")



# Verify multiple outputs and negation (Challenge):
# Trace a supported function returning both a scalar sum and a negated vector. Check the tuple of primal/tangent outputs and reject a mismatched tangent shape.

# Inputs:
# - A three-element vector and a same-shaped tangent.
# - A supported function returning the scalar sum of elementwise squares and the negated vector.
# - A deliberately mismatched tangent shape.

# Outputs:
# - Tuple-structured scalar/vector primals and matching scalar/vector tangents.
# - A clear shape rejection for the mismatched tangent.

# Invariants to check:
# - The scalar tangent is the directional derivative of the square-sum output.
# - The vector tangent preserves shape and sign for the negated output.

# Hints:
# - Use multiplication and reduction rather than an unsupported power primitive.
# - Validate tangent shape against the traced primal signature.

import jax
import jax.numpy as jnp

def check_multi_output_jvp(values, tangent):
    # TODO: define and trace the two-output function, then check matching and malformed seeds.
    raise NotImplementedError("Preserve the tuple output structure and reject a mismatched tangent")
