"""Vectors, norms, and projections: worked experiments and reference solutions. CPU checks."""

# Draw the two vectors
# Step 1 — Draw the two vectors: The length is 5; the component along the horizontal unit direction...
# Import jax.numpy for this computation.
import jax.numpy as jnp
# Construct `u` via `jnp.array([3.0, 4.0])`
u = jnp.array([3.0, 4.0])
# Construct `a` via `jnp.array([1.0, 0.0])`
a = jnp.array([1.0, 0.0])
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jnp.linalg.norm(u), 5.0)
# Check numerical equivalence within tolerance: `jnp.allclose(jnp.dot(u, a), 3.0)`
assert jnp.allclose(jnp.dot(u, a), 3.0)

# Build the projection
# Step 2 — Build the projection: The leftover r contains what the chosen direction cannot explain.
def project(u, a):
    # Run `jnp.dot` to compute `denominator`.
    denominator = jnp.dot(a, a)
    # Guard input contract (`float(denominator) == 0.0`) and fail fast if violated.
    if float(denominator) == 0.0:
        raise ValueError('Projection needs a nonzero direction')
    # Return `jnp.dot(a, u) / denominator * a` to the caller.
    return jnp.dot(a, u) / denominator * a
# Run `project` to compute `p`.
p = project(u, a)
# Compute `r` from `u - p`
r = u - p

# Check with geometry
# Step 3 — Check with geometry: You should see [3,0] and [0,4]; the squared lengths sum to 25.
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(p, jnp.array([3.0, 0.0]))
# Check numerical equivalence within tolerance: `jnp.allclose(jnp.dot(a, r), 0.0, atol=1e-06)`
assert jnp.allclose(jnp.dot(a, r), 0.0, atol=1e-06)
# Check numerical equivalence within tolerance: `jnp.allclose(jnp.dot(p, p) + jnp.dot(r, r), jnp.dot(u, u))`
assert jnp.allclose(jnp.dot(p, p) + jnp.dot(r, r), jnp.dot(u, u))
# Print the observed values to compare against the expected result.
print('projection / residual:', p, r)

# Step 1 — Draw the two vectors: The length is 5; the component along the horizontal unit direction...
# Import jax.numpy for this computation.
import jax.numpy as jnp
# Construct `u` via `jnp.array([3.0, 4.0])`
u = jnp.array([3.0, 4.0])
# Construct `a` via `jnp.array([1.0, 0.0])`
a = jnp.array([1.0, 0.0])
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jnp.linalg.norm(u), 5.0)
# Check numerical equivalence within tolerance: `jnp.allclose(jnp.dot(u, a), 3.0)`
assert jnp.allclose(jnp.dot(u, a), 3.0)

# Step 2 — Build the projection: The leftover r contains what the chosen direction cannot explain.
def project(u, a):
    # Run `jnp.dot` to compute `denominator`.
    denominator = jnp.dot(a, a)
    # Guard input contract (`float(denominator) == 0.0`) and fail fast if violated.
    if float(denominator) == 0.0:
        raise ValueError('Projection needs a nonzero direction')
    # Return `jnp.dot(a, u) / denominator * a` to the caller.
    return jnp.dot(a, u) / denominator * a
# Run `project` to compute `p`.
p = project(u, a)
# Compute `r` from `u - p`
r = u - p

# Step 3 — Check with geometry: You should see [3,0] and [0,4]; the squared lengths sum to 25.
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(p, jnp.array([3.0, 0.0]))
# Check numerical equivalence within tolerance: `jnp.allclose(jnp.dot(a, r), 0.0, atol=1e-06)`
assert jnp.allclose(jnp.dot(a, r), 0.0, atol=1e-06)
# Check numerical equivalence within tolerance: `jnp.allclose(jnp.dot(p, p) + jnp.dot(r, r), jnp.dot(u, u))`
assert jnp.allclose(jnp.dot(p, p) + jnp.dot(r, r), jnp.dot(u, u))
# Print the observed values to compare against the expected result.
print('projection / residual:', p, r)

# Figure data experiment
# Compute figure data for: Projection and leftover form a right triangle
# Compute `visual_data` from `{'kind': 'vectors', 'arrows': [{'label': 'input u', ...`
visual_data = {'kind': 'vectors', 'arrows': [{'label': 'input u', 'start': [0, 0], 'end': u.tolist()}, {'label': 'projection p', 'start': [0, 0], 'end': p.tolist()}, {'label': 'residual r', 'start': p.tolist(), 'end': u.tolist()}]}

# Experiment: Rescale the direction
# Experiment — Rescale the direction: A line does not change when its nonzero direction is rescaled.
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(project(u, 2 * a), p)
# Check numerical equivalence within tolerance: `jnp.allclose(project(u, -a), p)`
assert jnp.allclose(project(u, -a), p)
# Check numerical equivalence within tolerance: `jnp.allclose(jnp.dot(u, 2 * a), 2 * jnp.dot(u, a))`
assert jnp.allclose(jnp.dot(u, 2 * a), 2 * jnp.dot(u, a))

# Experiment: Rotate the line
# Experiment — Rotate the line: Changing direction changes what can be explained; merely...
# Construct `diagonal` via `jnp.array([1.0, 1.0])`
diagonal = jnp.array([1.0, 1.0])
# Run `project` to compute `p2`.
p2 = project(u, diagonal)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(p2, jnp.array([3.5, 3.5]))
# Check numerical equivalence within tolerance: `jnp.allclose(jnp.dot(diagonal, u - p2), 0.0, atol=1e-06)`
assert jnp.allclose(jnp.dot(diagonal, u - p2), 0.0, atol=1e-06)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Project u=(2,-1,2) onto a=(1,0,1).
# Construct `u3` via `jnp.array([2.0, -1.0, 2.0])`
u3 = jnp.array([2.0, -1.0, 2.0])
# Construct `a3` via `jnp.array([1.0, 0.0, 1.0])`
a3 = jnp.array([1.0, 0.0, 1.0])
# Run `project` to compute `p3`.
p3 = project(u3, a3)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(p3, jnp.array([2.0, 0.0, 2.0]))
# Check numerical equivalence within tolerance: `jnp.allclose(jnp.dot(a3, u3 - p3), 0.0)`
assert jnp.allclose(jnp.dot(a3, u3 - p3), 0.0)

# Reference practice: Handle an undefined direction
# Handle an undefined direction (Transfer / diagnosis): Rejecting an undefined input is part of the function contract.
# Run the boundary check and catch the expected exception:
try:
    project(u, jnp.zeros_like(a))
except ValueError as exc:
    assert 'nonzero' in str(exc)
else:
    raise AssertionError('Zero direction was accepted')

# Reference practice: Use two perpendicular directions
# Use two perpendicular directions (Transfer / diagnosis): Adding separate projections reconstructs a vector in an...
# Construct `a1` via `jnp.array([1.0, 1.0])`
a1 = jnp.array([1.0, 1.0])
# Construct `a2` via `jnp.array([1.0, -1.0])`
a2 = jnp.array([1.0, -1.0])
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(project(u, a1) + project(u, a2), u)
# Check numerical equivalence within tolerance: `not jnp.allclose(project(u, a) + project(u, a1), u)`
assert not jnp.allclose(project(u, a) + project(u, a1), u)
print("PASS: optimization-05")
