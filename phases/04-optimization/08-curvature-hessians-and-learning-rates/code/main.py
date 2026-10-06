"""Curvature, Hessians, and learning rates: worked experiments and reference solutions. CPU checks."""

# Write the bowl
import jax
import jax.numpy as jnp

def loss(w):
    return 0.5 * (w[0] ** 2 + 10 * w[1] ** 2)
w = jnp.array([1.0, 1.0])
H = jnp.diag(jnp.array([1.0, 10.0]))
assert jnp.allclose(jax.grad(loss)(w), H @ w)

# Check second derivatives
assert jnp.allclose(jax.hessian(loss)(w), H)
v = jnp.array([2.0, -1.0])
hv = jax.jvp(jax.grad(loss), (w,), (v,))[1]
assert jnp.allclose(hv, jnp.array([2.0, -10.0]))

# Run an explicit trajectory
def run(rate, steps=20):

    def step(w, _):
        w = w - rate * jax.grad(loss)(w)
        return (w, loss(w))
    return jax.lax.scan(step, jnp.array([1.0, 1.0]), None, length=steps)
final, history = run(0.1)
assert jnp.allclose(final, jnp.array([0.9 ** 20, 0.0]), atol=2e-06)
assert jnp.all(jnp.diff(history) <= 1e-06)
print('final weights / loss:', final, history[-1])

import jax
import jax.numpy as jnp

def loss(w):
    return 0.5 * (w[0] ** 2 + 10 * w[1] ** 2)
w = jnp.array([1.0, 1.0])
H = jnp.diag(jnp.array([1.0, 10.0]))
assert jnp.allclose(jax.grad(loss)(w), H @ w)

assert jnp.allclose(jax.hessian(loss)(w), H)
v = jnp.array([2.0, -1.0])
hv = jax.jvp(jax.grad(loss), (w,), (v,))[1]
assert jnp.allclose(hv, jnp.array([2.0, -10.0]))

def run(rate, steps=20):

    def step(w, _):
        w = w - rate * jax.grad(loss)(w)
        return (w, loss(w))
    return jax.lax.scan(step, jnp.array([1.0, 1.0]), None, length=steps)
final, history = run(0.1)
assert jnp.allclose(final, jnp.array([0.9 ** 20, 0.0]), atol=2e-06)
assert jnp.all(jnp.diff(history) <= 1e-06)
print('final weights / loss:', final, history[-1])

# Figure data experiment
points = [jnp.array([1.0, 1.0])]
for _ in range(20):
    points.append(points[-1] - 0.1 * jax.grad(loss)(points[-1]))
trace = jnp.stack(points)
visual_data = {'kind': 'line', 'x': list(range(21)), 'xlabel': 'completed update', 'ylabel': 'coordinate value', 'series': [{'label': 'curvature 1', 'y': trace[:, 0].tolist()}, {'label': 'curvature 10', 'y': trace[:, 1].tolist()}]}

# Experiment: Cross the stability boundary
edge, _ = run(0.2)
bad, bad_history = run(0.21)
assert jnp.allclose(jnp.abs(edge[1]), 1.0, atol=2e-06)
assert jnp.abs(bad[1]) > 6.0
assert bad_history[-1] > loss(w)

# Experiment: Change coordinates
z = jnp.array([w[0], jnp.sqrt(10.0) * w[1]])
round_loss = lambda z: 0.5 * jnp.dot(z, z)
z_next = z - jax.grad(round_loss)(z)
assert jnp.allclose(z_next, jnp.zeros(2))
assert jnp.allclose(round_loss(z), loss(w))

# Reference solution. Try the exercise before reading this.
H2 = jnp.diag(jnp.array([2.0, 8.0]))
w_next = w - 0.1 * (H2 @ w)
assert jnp.allclose(w_next, jnp.array([0.8, 0.2]), atol=1e-06)
assert jnp.allclose(2 / jnp.max(jnp.linalg.eigvalsh(H2)), 0.25)

# Reference practice: Reject the zero-gradient shortcut
saddle = lambda z: z[0] ** 2 - z[1] ** 2
zero = jnp.zeros(2)
assert jnp.allclose(jax.grad(saddle)(zero), zero)
assert saddle(jnp.array([0.0, 0.1])) < saddle(zero)
assert saddle(jnp.array([0.1, 0.0])) > saddle(zero)
assert jnp.allclose(jnp.linalg.eigvalsh(jax.hessian(saddle)(zero)), jnp.array([-2.0, 2.0]))

# Reference practice: Transfer to a nonquadratic function
point = jnp.array([0.0, jnp.log(2.0)])
direction = jnp.array([1.0, 3.0])
product = jax.jvp(jax.grad(lambda z: jnp.sum(jnp.exp(z))), (point,), (direction,))[1]
assert jnp.allclose(product, jnp.array([1.0, 6.0]), atol=1e-06)
print("PASS: optimization-08")
