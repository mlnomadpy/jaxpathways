"""Vectors, norms, and projections: worked experiments and reference solutions. CPU checks."""

# Draw the two vectors
import jax.numpy as jnp
u = jnp.array([3.0, 4.0])
a = jnp.array([1.0, 0.0])
assert jnp.allclose(jnp.linalg.norm(u), 5.0)
assert jnp.allclose(jnp.dot(u, a), 3.0)

# Build the projection
def project(u, a):
    denominator = jnp.dot(a, a)
    if float(denominator) == 0.0:
        raise ValueError('Projection needs a nonzero direction')
    return jnp.dot(a, u) / denominator * a
p = project(u, a)
r = u - p

# Check with geometry
assert jnp.allclose(p, jnp.array([3.0, 0.0]))
assert jnp.allclose(jnp.dot(a, r), 0.0, atol=1e-06)
assert jnp.allclose(jnp.dot(p, p) + jnp.dot(r, r), jnp.dot(u, u))
print('projection / residual:', p, r)

import jax.numpy as jnp
u = jnp.array([3.0, 4.0])
a = jnp.array([1.0, 0.0])
assert jnp.allclose(jnp.linalg.norm(u), 5.0)
assert jnp.allclose(jnp.dot(u, a), 3.0)

def project(u, a):
    denominator = jnp.dot(a, a)
    if float(denominator) == 0.0:
        raise ValueError('Projection needs a nonzero direction')
    return jnp.dot(a, u) / denominator * a
p = project(u, a)
r = u - p

assert jnp.allclose(p, jnp.array([3.0, 0.0]))
assert jnp.allclose(jnp.dot(a, r), 0.0, atol=1e-06)
assert jnp.allclose(jnp.dot(p, p) + jnp.dot(r, r), jnp.dot(u, u))
print('projection / residual:', p, r)

# Figure data experiment
visual_data = {'kind': 'vectors', 'arrows': [{'label': 'input u', 'start': [0, 0], 'end': u.tolist()}, {'label': 'projection p', 'start': [0, 0], 'end': p.tolist()}, {'label': 'residual r', 'start': p.tolist(), 'end': u.tolist()}]}

# Experiment: Rescale the direction
assert jnp.allclose(project(u, 2 * a), p)
assert jnp.allclose(project(u, -a), p)
assert jnp.allclose(jnp.dot(u, 2 * a), 2 * jnp.dot(u, a))

# Experiment: Rotate the line
diagonal = jnp.array([1.0, 1.0])
p2 = project(u, diagonal)
assert jnp.allclose(p2, jnp.array([3.5, 3.5]))
assert jnp.allclose(jnp.dot(diagonal, u - p2), 0.0, atol=1e-06)

# Reference solution. Try the exercise before reading this.
u3 = jnp.array([2.0, -1.0, 2.0])
a3 = jnp.array([1.0, 0.0, 1.0])
p3 = project(u3, a3)
assert jnp.allclose(p3, jnp.array([2.0, 0.0, 2.0]))
assert jnp.allclose(jnp.dot(a3, u3 - p3), 0.0)

# Reference practice: Handle an undefined direction
try:
    project(u, jnp.zeros_like(a))
except ValueError as exc:
    assert 'nonzero' in str(exc)
else:
    raise AssertionError('Zero direction was accepted')

# Reference practice: Use two perpendicular directions
a1 = jnp.array([1.0, 1.0])
a2 = jnp.array([1.0, -1.0])
assert jnp.allclose(project(u, a1) + project(u, a2), u)
assert not jnp.allclose(project(u, a) + project(u, a1), u)
print("PASS: optimization-05")
