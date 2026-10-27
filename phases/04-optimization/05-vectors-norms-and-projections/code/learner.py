"""Vectors, norms, and projections · learner-only exercises (3/3 contracts ready)"""

# Exercise: Project \(u=(2,-1,2)\) onto \(a=(1,0,1)\). Check the residual by hand first.

# Inputs:
# - Two same-shaped vectors and a nonzero projection direction.

# Outputs:
# - The projection and orthogonal residual.

# Invariants to check:
# - The residual is orthogonal to the direction; a zero direction is rejected.

# Hints:
# - Compute the scalar coefficient from a dot product and squared norm.
# - Use an inner-product check on the residual instead of checking only its length.

import jax.numpy as jnp

def project_onto(vector, direction):
    """The projection and orthogonal residual."""
    # TODO: Guard against a zero direction, compute the projection coefficient, and verify the residual dot product.
    raise NotImplementedError




# Handle an undefined direction (Transfer / diagnosis):
# Show that a zero direction produces a clear input error instead of a misleading projection.

# Inputs:
# - Show that a zero direction produces a clear input error instead of a misleading projection.

# Outputs:
# - A clear rejection for zero direction and a finite projection for a nonzero direction.

# Invariants to check:
# - A projection is returned only when its direction has nonzero squared norm.

# Hints:
# - Use a try/except block; a silent successful result should fail the check.

import jax.numpy as jnp

def practice_handle_an_undefined_direction(vector, direction):
    """Implement the handle an undefined direction check."""
    # TODO: Check the direction norm before dividing and raise a clear ValueError for the zero vector.
    raise NotImplementedError




# Use two perpendicular directions (Transfer / diagnosis):
# Reconstruct \(u\) from projections onto \((1,1)\) and \((1,-1)\). Would adding independent projections still work for arbitrary nonorthogonal directions?

# Inputs:
# - Reconstruct \(u\) from projections onto \((1,1)\) and \((1,-1)\). Would adding independent projections still work for arbitrary nonorthogonal directions?

# Outputs:
# - Reconstructed vector and residual norm for the perpendicular basis.

# Invariants to check:
# - The sum is interpreted as a reconstruction only for the stated orthogonal directions.

# Hints:
# - Check the directions have zero dot product. Nonorthogonal directions can count the same component twice.

import jax.numpy as jnp

def practice_use_two_perpendicular_directions(vector, first_direction, second_direction):
    """Implement the use two perpendicular directions check."""
    # TODO: Compute each orthogonal projection, add the two components, and measure the reconstruction residual.
    raise NotImplementedError

