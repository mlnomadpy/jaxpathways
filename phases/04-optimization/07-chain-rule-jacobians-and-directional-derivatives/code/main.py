"""The chain rule, Jacobians, and directions: worked experiments and reference solutions. CPU checks."""

# Write the vector function
import jax
import jax.numpy as jnp

def f(x):
    return jnp.array([x[0] ** 2 + x[1], x[0] * x[1]])
x = jnp.array([2.0, 3.0])
v = jnp.array([1.0, -1.0])
J = jnp.array([[4.0, 1.0], [3.0, 2.0]])
assert jnp.allclose(f(x), jnp.array([7.0, 6.0]))

# Push a direction and pull a sensitivity
out, jv = jax.jvp(f, (x,), (v,))
_, pullback = jax.vjp(f, x)
c = out
jt_c = pullback(c)[0]
assert jnp.allclose(jv, jnp.array([3.0, 1.0]))
assert jnp.allclose(jt_c, jnp.array([46.0, 19.0]))

# Connect to a scalar loss
def loss(x):
    return 0.5 * jnp.sum(f(x) ** 2)
assert jnp.allclose(jax.jacfwd(f)(x), J)
assert jnp.allclose(jax.grad(loss)(x), J.T @ c)
assert jnp.allclose(jnp.dot(c, jv), jnp.dot(v, jt_c))
print('Jv / loss gradient:', jv, jt_c)

import jax
import jax.numpy as jnp

def f(x):
    return jnp.array([x[0] ** 2 + x[1], x[0] * x[1]])
x = jnp.array([2.0, 3.0])
v = jnp.array([1.0, -1.0])
J = jnp.array([[4.0, 1.0], [3.0, 2.0]])
assert jnp.allclose(f(x), jnp.array([7.0, 6.0]))

out, jv = jax.jvp(f, (x,), (v,))
_, pullback = jax.vjp(f, x)
c = out
jt_c = pullback(c)[0]
assert jnp.allclose(jv, jnp.array([3.0, 1.0]))
assert jnp.allclose(jt_c, jnp.array([46.0, 19.0]))

def loss(x):
    return 0.5 * jnp.sum(f(x) ** 2)
assert jnp.allclose(jax.jacfwd(f)(x), J)
assert jnp.allclose(jax.grad(loss)(x), J.T @ c)
assert jnp.allclose(jnp.dot(c, jv), jnp.dot(v, jt_c))
print('Jv / loss gradient:', jv, jt_c)

# Figure data experiment
visual_data = {'kind': 'heatmap', 'values': jax.jacfwd(f)(x).tolist(), 'rows': ['output 0', 'output 1'], 'columns': ['input 0', 'input 1'], 'unit': 'partial derivative'}

# Experiment: Check a finite input move
h = 0.01
fd = (f(x + h * v) - f(x - h * v)) / (2 * h)
assert jnp.allclose(fd, J @ v, atol=0.0001, rtol=0.0001)

# Experiment: Change the point
x2 = jnp.array([0.0, 1.0])
J2 = jnp.array([[0.0, 1.0], [1.0, 0.0]])
assert jnp.allclose(jax.jacrev(f)(x2), J2)
assert jnp.allclose(jax.jvp(f, (x2,), (v,))[1], J2 @ v)

# Reference solution. Try the exercise before reading this.
def weighted_loss(x):
    return f(x)[0] + 2 * f(x)[1]
assert jnp.allclose(jax.grad(weighted_loss)(x), jnp.array([10.0, 5.0]))

# Reference practice: Find the broken derivative path
def detached(x):
    a = jax.lax.stop_gradient(x[0])
    return jnp.array([a * a + x[1], a * x[1]])
assert jnp.allclose(detached(x), f(x))
assert jnp.allclose(jax.grad(lambda z: jnp.sum(detached(z)))(x), jnp.array([0.0, 3.0]))
assert jnp.allclose(jax.grad(lambda z: jnp.sum(f(z)))(x), jnp.array([7.0, 3.0]))

# Reference practice: Transfer to unequal dimensions
A = jnp.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
c3 = jnp.array([1.0, -1.0, 2.0])
_, back = jax.vjp(lambda z: A @ z, x)
assert back(c3)[0].shape == (2,)
assert jnp.allclose(back(c3)[0], jnp.array([3.0, 0.0]))
print("PASS: optimization-07")
