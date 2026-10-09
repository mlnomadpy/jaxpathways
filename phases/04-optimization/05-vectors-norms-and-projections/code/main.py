"""Vectors, norms, and projections: worked experiments and reference solutions. CPU checks."""

# Draw the two vectors
# Step 1 — Draw the two vectors: The length is 5; the component along the horizontal unit direction...
# Import jax.numpy for this computation.
import jax.numpy as jnp
# Initialize array `u` with explicit values and shape.
u = jnp.array([3.0, 4.0])
# Initialize array `a` with explicit values and shape.
a = jnp.array([1.0, 0.0])
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jnp.linalg.norm(u), 5.0)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
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
# Evaluate `r` from the current inputs and state.
r = u - p

# Check with geometry
# Step 3 — Check with geometry: You should see [3,0] and [0,4]; the squared lengths sum to 25.
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(p, jnp.array([3.0, 0.0]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.dot(a, r), 0.0, atol=1e-06)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.dot(p, p) + jnp.dot(r, r), jnp.dot(u, u))
# Print the observed values to compare against the expected result.
print('projection / residual:', p, r)

# Step 1 — Draw the two vectors: The length is 5; the component along the horizontal unit direction...
# Import jax.numpy for this computation.
import jax.numpy as jnp
# Initialize array `u` with explicit values and shape.
u = jnp.array([3.0, 4.0])
# Initialize array `a` with explicit values and shape.
a = jnp.array([1.0, 0.0])
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jnp.linalg.norm(u), 5.0)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
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
# Evaluate `r` from the current inputs and state.
r = u - p

# Step 3 — Check with geometry: You should see [3,0] and [0,4]; the squared lengths sum to 25.
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(p, jnp.array([3.0, 0.0]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.dot(a, r), 0.0, atol=1e-06)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.dot(p, p) + jnp.dot(r, r), jnp.dot(u, u))
# Print the observed values to compare against the expected result.
print('projection / residual:', p, r)

# Figure data experiment
# Compute figure data for: Projection and leftover form a right triangle
# Evaluate `visual_data` from the current inputs and state.
visual_data = {'kind': 'vectors', 'arrows': [{'label': 'input u', 'start': [0, 0], 'end': u.tolist()}, {'label': 'projection p', 'start': [0, 0], 'end': p.tolist()}, {'label': 'residual r', 'start': p.tolist(), 'end': u.tolist()}]}

# Experiment: Rescale the direction
# Experiment — Rescale the direction: A line does not change when its nonzero direction is rescaled.
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(project(u, 2 * a), p)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(project(u, -a), p)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.dot(u, 2 * a), 2 * jnp.dot(u, a))

# Experiment: Rotate the line
# Experiment — Rotate the line: Changing direction changes what can be explained; merely...
# Initialize array `diagonal` with explicit values and shape.
diagonal = jnp.array([1.0, 1.0])
# Run `project` to compute `p2`.
p2 = project(u, diagonal)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(p2, jnp.array([3.5, 3.5]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.dot(diagonal, u - p2), 0.0, atol=1e-06)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Project u=(2,-1,2) onto a=(1,0,1).
# Initialize array `u3` with explicit values and shape.
u3 = jnp.array([2.0, -1.0, 2.0])
# Initialize array `a3` with explicit values and shape.
a3 = jnp.array([1.0, 0.0, 1.0])
# Run `project` to compute `p3`.
p3 = project(u3, a3)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(p3, jnp.array([2.0, 0.0, 2.0]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
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
# Initialize array `a1` with explicit values and shape.
a1 = jnp.array([1.0, 1.0])
# Initialize array `a2` with explicit values and shape.
a2 = jnp.array([1.0, -1.0])
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(project(u, a1) + project(u, a2), u)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert not jnp.allclose(project(u, a) + project(u, a1), u)
print("PASS: optimization-05")
