"""Read your first jaxpr: worked experiments and reference solutions. CPU checks."""

# 1. Establish the input signature
# Step 1 — 1. Establish the input signature: The vector has shape (3,) and dtype float32; these determine the...
import jax
import jax.numpy as jnp

x = jnp.array([1., 2., 3.], dtype=jnp.float32)

# 2. Write the pure function
# Step 2 — 2. Write the pure function: Each assignment names a new value.
def shifted_square_sum(x):
    shifted = x + 1.
    squared = shifted * shifted
    return jnp.sum(squared)

# 3. Inspect and independently verify
# Step 3 — 3. Inspect and independently verify: Forward operations include add, mul and reduce_sum, with output...
# Trace or lower the function to inspect its compiler representation (`closed`).
closed = jax.make_jaxpr(shifted_square_sum)(x)
print("Forward program:")
print(closed)
for equation in closed.jaxpr.eqns:
    print("Operation:", equation.primitive.name,
          "output shapes:", [v.aval.shape for v in equation.outvars])
value = shifted_square_sum(x)
# Differentiate the objective to obtain `gradient` via automatic differentiation.
gradient = jax.grad(shifted_square_sum)(x)
print("Value / gradient:", value, gradient)
assert jnp.allclose(value, 29.)
assert jnp.allclose(gradient, jnp.array([4., 6., 8.]))

# Step 1 — 1. Establish the input signature: The vector has shape (3,) and dtype float32; these determine the...
import jax
import jax.numpy as jnp

x = jnp.array([1., 2., 3.], dtype=jnp.float32)
# Step 2 — 2. Write the pure function: Each assignment names a new value.
def shifted_square_sum(x):
    shifted = x + 1.
    squared = shifted * shifted
    return jnp.sum(squared)

# Step 3 — 3. Inspect and independently verify: Forward operations include add, mul and reduce_sum, with output...
# Trace or lower the function to inspect its compiler representation (`closed`).
closed = jax.make_jaxpr(shifted_square_sum)(x)
print("Forward program:")
print(closed)
for equation in closed.jaxpr.eqns:
    print("Operation:", equation.primitive.name,
          "output shapes:", [v.aval.shape for v in equation.outvars])
value = shifted_square_sum(x)
# Differentiate the objective to obtain `gradient` via automatic differentiation.
gradient = jax.grad(shifted_square_sum)(x)
print("Value / gradient:", value, gradient)
assert jnp.allclose(value, 29.)
assert jnp.allclose(gradient, jnp.array([4., 6., 8.]))

# Figure data experiment
visual_data = {'kind': 'heatmap', 'values': jnp.stack([x, x + 1, (x + 1) ** 2]).tolist(), 'rows': ['input', 'add 1', 'multiply by self'], 'columns': ['coordinate 0', 'coordinate 1', 'coordinate 2'], 'unit': 'intermediate value'}

# Experiment: Trace and verify the derivative
# Experiment — Trace and verify the derivative: The new input checks the derivative independently of the...
# Differentiate the objective to obtain `derivative_program` via automatic differentiation.
derivative_program = jax.make_jaxpr(jax.grad(shifted_square_sum))(x)
print("Derivative program:", derivative_program)
other = jnp.array([-1., 0., 2.])
assert jnp.allclose(shifted_square_sum(other), 10.)
assert jnp.allclose(jax.grad(shifted_square_sum)(other), jnp.array([0., 2., 6.]))
assert jnp.allclose(jax.grad(shifted_square_sum)(other), 2*(other+1))

# Experiment: Inspect a captured array
# Experiment — Inspect a captured array: The function signature determines ordinary inputs; closed...
offset = jnp.array([1., 2., 3.])
def captured(z):
    return jnp.sum(z + offset)
def explicit(z, offset):
    return jnp.sum(z + offset)
# Trace or lower the function to inspect its compiler representation (`captured_program`).
captured_program = jax.make_jaxpr(captured)(x)
# Trace or lower the function to inspect its compiler representation (`explicit_program`).
explicit_program = jax.make_jaxpr(explicit)(x, offset)
print("Captured:", captured_program)
print("Explicit:", explicit_program)
assert len(captured_program.jaxpr.invars) == 1
assert len(explicit_program.jaxpr.invars) == 2
assert jnp.allclose(captured(x), explicit(x, offset))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Change the scalar offset from 1 to 2.
def shifted_two(z):
    shifted = z + 2.
    return jnp.sum(shifted * shifted)
print(jax.make_jaxpr(shifted_two)(x))
assert jnp.allclose(shifted_two(x), 50.)
assert jnp.allclose(jax.grad(shifted_two)(x), jnp.array([6., 8., 10.]))

# Reference practice: Compare unrolled and scanned recurrences
# Compare unrolled and scanned recurrences (Transfer / diagnosis): The outer scan equation has a nested body.
def unrolled(z):
    for _ in range(4):
        z = 0.5*z + 1.
    return z
# Define `scanned(z)` to carry state across steps with `jax.lax.scan`:
def scanned(z):
    def step(carry, _):
        new = 0.5*carry + 1.
        return new, new
    return jax.lax.scan(step, z, None, length=4)[0]
print("Unrolled:", jax.make_jaxpr(unrolled)(jnp.array(0.)))
print("Scanned:", jax.make_jaxpr(scanned)(jnp.array(0.)))
assert jnp.allclose(unrolled(jnp.array(0.)), 1.875)
assert jnp.allclose(scanned(jnp.array(0.)), 1.875)

# Reference practice: Reproduce and repair a traced boolean failure
# Reproduce and repair a traced boolean failure (Transfer / diagnosis): The repair makes runtime control flow explicit and verifies...
def bad_branch(z):
    if z[0] > 0:
        return jnp.sum(z)
    return -jnp.sum(z)
try:
    jax.make_jaxpr(bad_branch)(x)
except jax.errors.TracerBoolConversionError:
    print("Expected failure: traced boolean used by Python if")
else:
    raise AssertionError("expected tracing failure")
def repaired(z):
    return jax.lax.cond(z[0] > 0, lambda v: jnp.sum(v), lambda v: -jnp.sum(v), z)
print("Repaired:", jax.make_jaxpr(repaired)(x))
assert jnp.allclose(jax.jit(repaired)(x), 6.)
assert jnp.allclose(jax.jit(repaired)(-x), 6.)
print("PASS: internals-01")
