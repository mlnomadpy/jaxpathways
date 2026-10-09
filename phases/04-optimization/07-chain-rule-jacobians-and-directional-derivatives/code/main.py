"""The chain rule, Jacobians, and directions: worked experiments and reference solutions. CPU checks."""

# Write the vector function
# Step 1 — Write the vector function: The matrix is written from the derivatives, not obtained from an...
# Import jax for this computation.
import jax
import jax.numpy as jnp

# Function `f(x)` implementing this stage's computation:
def f(x):
    # Return `jnp.array([x[0] ** 2 + x[1], x[0] * x[1]])` to the caller.
    return jnp.array([x[0] ** 2 + x[1], x[0] * x[1]])
# Construct `x` via `jnp.array([2.0, 3.0])`
x = jnp.array([2.0, 3.0])
# Construct `v` via `jnp.array([1.0, -1.0])`
v = jnp.array([1.0, -1.0])
# Construct `J` via `jnp.array([[4.0, 1.0], [3.0, 2.0]])`
J = jnp.array([[4.0, 1.0], [3.0, 2.0]])
# Assert that `jnp.allclose(f(x), jnp.array([7.0, 6.0]))`.
assert jnp.allclose(f(x), jnp.array([7.0, 6.0]))

# Push a direction and pull a sensitivity
# Step 2 — Push a direction and pull a sensitivity: Forward mode follows an input direction; reverse mode starts with...
# Compute exact directional derivative / Jacobian / Hessian (`(out, jv)`).
out, jv = jax.jvp(f, (x,), (v,))
# Compute exact directional derivative / Jacobian / Hessian (`(_, pullback)`).
_, pullback = jax.vjp(f, x)
# Compute `c` from `out`
c = out
# Run `pullback` to compute `jt_c`.
jt_c = pullback(c)[0]
# Assert that `jnp.allclose(jv, jnp.array([3.0, 1.0]))`.
assert jnp.allclose(jv, jnp.array([3.0, 1.0]))
# Assert that `jnp.allclose(jt_c, jnp.array([46.0, 19.0]))`.
assert jnp.allclose(jt_c, jnp.array([46.0, 19.0]))

# Connect to a scalar loss
# Step 3 — Connect to a scalar loss: Both routes give directional loss derivative 27.
def loss(x):
    # Return `0.5 * jnp.sum(f(x) ** 2)` to the caller.
    return 0.5 * jnp.sum(f(x) ** 2)
# Assert that `jnp.allclose(jax.jacfwd(f)(x), J)`.
assert jnp.allclose(jax.jacfwd(f)(x), J)
# Assert that `jnp.allclose(jax.grad(loss)(x), J.T @ c)`.
assert jnp.allclose(jax.grad(loss)(x), J.T @ c)
# Assert that `jnp.allclose(jnp.dot(c, jv), jnp.dot(v, jt_c))`.
assert jnp.allclose(jnp.dot(c, jv), jnp.dot(v, jt_c))
# Print the observed values to compare against the expected result.
print('Jv / loss gradient:', jv, jt_c)

# Step 1 — Write the vector function: The matrix is written from the derivatives, not obtained from an...
# Import jax for this computation.
import jax
import jax.numpy as jnp

# Function `f(x)` implementing this stage's computation:
def f(x):
    # Return `jnp.array([x[0] ** 2 + x[1], x[0] * x[1]])` to the caller.
    return jnp.array([x[0] ** 2 + x[1], x[0] * x[1]])
# Construct `x` via `jnp.array([2.0, 3.0])`
x = jnp.array([2.0, 3.0])
# Construct `v` via `jnp.array([1.0, -1.0])`
v = jnp.array([1.0, -1.0])
# Construct `J` via `jnp.array([[4.0, 1.0], [3.0, 2.0]])`
J = jnp.array([[4.0, 1.0], [3.0, 2.0]])
# Assert that `jnp.allclose(f(x), jnp.array([7.0, 6.0]))`.
assert jnp.allclose(f(x), jnp.array([7.0, 6.0]))

# Step 2 — Push a direction and pull a sensitivity: Forward mode follows an input direction; reverse mode starts with...
# Compute exact directional derivative / Jacobian / Hessian (`(out, jv)`).
out, jv = jax.jvp(f, (x,), (v,))
# Compute exact directional derivative / Jacobian / Hessian (`(_, pullback)`).
_, pullback = jax.vjp(f, x)
# Compute `c` from `out`
c = out
# Run `pullback` to compute `jt_c`.
jt_c = pullback(c)[0]
# Assert that `jnp.allclose(jv, jnp.array([3.0, 1.0]))`.
assert jnp.allclose(jv, jnp.array([3.0, 1.0]))
# Assert that `jnp.allclose(jt_c, jnp.array([46.0, 19.0]))`.
assert jnp.allclose(jt_c, jnp.array([46.0, 19.0]))

# Step 3 — Connect to a scalar loss: Both routes give directional loss derivative 27.
def loss(x):
    # Return `0.5 * jnp.sum(f(x) ** 2)` to the caller.
    return 0.5 * jnp.sum(f(x) ** 2)
# Assert that `jnp.allclose(jax.jacfwd(f)(x), J)`.
assert jnp.allclose(jax.jacfwd(f)(x), J)
# Assert that `jnp.allclose(jax.grad(loss)(x), J.T @ c)`.
assert jnp.allclose(jax.grad(loss)(x), J.T @ c)
# Assert that `jnp.allclose(jnp.dot(c, jv), jnp.dot(v, jt_c))`.
assert jnp.allclose(jnp.dot(c, jv), jnp.dot(v, jt_c))
# Print the observed values to compare against the expected result.
print('Jv / loss gradient:', jv, jt_c)

# Figure data experiment
# Compute figure data for: A Jacobian maps inputs to output sensitivities
# Compute higher-order Jacobian or Hessian curvature matrix `visual_data`.
visual_data = {'kind': 'heatmap', 'values': jax.jacfwd(f)(x).tolist(), 'rows': ['output 0', 'output 1'], 'columns': ['input 0', 'input 1'], 'unit': 'partial derivative'}

# Experiment: Check a finite input move
# Experiment — Check a finite input move: A numerical perturbation provides a check independent of autodiff.
h = 0.01
# Compute `fd` from `(f(x + h * v) - f(x - h * v)) / (2 * h)`
fd = (f(x + h * v) - f(x - h * v)) / (2 * h)
# Assert that `jnp.allclose(fd, J @ v, atol=0.0001, rtol=0.0001)`.
assert jnp.allclose(fd, J @ v, atol=0.0001, rtol=0.0001)

# Experiment: Change the point
# Experiment — Change the point: The Jacobian describes sensitivity at the current point; it is...
# Construct `x2` via `jnp.array([0.0, 1.0])`
x2 = jnp.array([0.0, 1.0])
# Construct `J2` via `jnp.array([[0.0, 1.0], [1.0, 0.0]])`
J2 = jnp.array([[0.0, 1.0], [1.0, 0.0]])
# Assert that `jnp.allclose(jax.jacrev(f)(x2), J2)`.
assert jnp.allclose(jax.jacrev(f)(x2), J2)
# Assert that `jnp.allclose(jax.jvp(f, (x2,), (v,))[1], J2 @ v)`.
assert jnp.allclose(jax.jvp(f, (x2,), (v,))[1], J2 @ v)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Replace the scalar loss by L(x)=f_0(x)+2f_1(x).
def weighted_loss(x):
    # Return `f(x)[0] + 2 * f(x)[1]` to the caller.
    return f(x)[0] + 2 * f(x)[1]
# Assert that `jnp.allclose(jax.grad(weighted_loss)(x), jnp.array([10.0, 5.0]))`.
assert jnp.allclose(jax.grad(weighted_loss)(x), jnp.array([10.0, 5.0]))

# Reference practice: Find the broken derivative path
# Find the broken derivative path (Transfer / diagnosis): Equal forward values do not imply equal differentiation...
# Define `detached(x)` to evaluate the objective and its automatic derivatives:
def detached(x):
    # Run `jax.lax.stop_gradient` to compute `a`.
    a = jax.lax.stop_gradient(x[0])
    # Return `jnp.array([a * a + x[1], a * x[1]])` to the caller.
    return jnp.array([a * a + x[1], a * x[1]])
# Assert that `jnp.allclose(detached(x), f(x))`.
assert jnp.allclose(detached(x), f(x))
# Assert that `jnp.allclose(jax.grad(lambda z: jnp.sum(detached(z)))(x), jnp.array([0.0, 3.0]))`.
assert jnp.allclose(jax.grad(lambda z: jnp.sum(detached(z)))(x), jnp.array([0.0, 3.0]))
# Assert that `jnp.allclose(jax.grad(lambda z: jnp.sum(f(z)))(x), jnp.array([7.0, 3.0]))`.
assert jnp.allclose(jax.grad(lambda z: jnp.sum(f(z)))(x), jnp.array([7.0, 3.0]))

# Reference practice: Transfer to unequal dimensions
# Transfer to unequal dimensions (Transfer / diagnosis): Unequal input and output dimensions make a mistaken...
# Construct `A` via `jnp.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])`
A = jnp.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
# Construct `c3` via `jnp.array([1.0, -1.0, 2.0])`
c3 = jnp.array([1.0, -1.0, 2.0])
# Compute exact directional derivative / Jacobian / Hessian (`(_, back)`).
_, back = jax.vjp(lambda z: A @ z, x)
# Check tensor shape invariant: `back(c3)[0].shape == (2,)`
assert back(c3)[0].shape == (2,)
# Assert that `jnp.allclose(back(c3)[0], jnp.array([3.0, 0.0]))`.
assert jnp.allclose(back(c3)[0], jnp.array([3.0, 0.0]))
print("PASS: optimization-07")
