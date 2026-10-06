"""Read your first jaxpr: worked experiments and reference solutions. CPU checks."""

# 1. Establish the input signature
import jax
import jax.numpy as jnp

x = jnp.array([1., 2., 3.], dtype=jnp.float32)

# 2. Write the pure function
def shifted_square_sum(x):
    shifted = x + 1.
    squared = shifted * shifted
    return jnp.sum(squared)

# 3. Inspect and independently verify
closed = jax.make_jaxpr(shifted_square_sum)(x)
print("Forward program:")
print(closed)
for equation in closed.jaxpr.eqns:
    print("Operation:", equation.primitive.name,
          "output shapes:", [v.aval.shape for v in equation.outvars])
value = shifted_square_sum(x)
gradient = jax.grad(shifted_square_sum)(x)
print("Value / gradient:", value, gradient)
assert jnp.allclose(value, 29.)
assert jnp.allclose(gradient, jnp.array([4., 6., 8.]))

import jax
import jax.numpy as jnp

x = jnp.array([1., 2., 3.], dtype=jnp.float32)
def shifted_square_sum(x):
    shifted = x + 1.
    squared = shifted * shifted
    return jnp.sum(squared)

closed = jax.make_jaxpr(shifted_square_sum)(x)
print("Forward program:")
print(closed)
for equation in closed.jaxpr.eqns:
    print("Operation:", equation.primitive.name,
          "output shapes:", [v.aval.shape for v in equation.outvars])
value = shifted_square_sum(x)
gradient = jax.grad(shifted_square_sum)(x)
print("Value / gradient:", value, gradient)
assert jnp.allclose(value, 29.)
assert jnp.allclose(gradient, jnp.array([4., 6., 8.]))

# Figure data experiment
visual_data = {'kind': 'heatmap', 'values': jnp.stack([x, x + 1, (x + 1) ** 2]).tolist(), 'rows': ['input', 'add 1', 'multiply by self'], 'columns': ['coordinate 0', 'coordinate 1', 'coordinate 2'], 'unit': 'intermediate value'}

# Experiment: Trace and verify the derivative
derivative_program = jax.make_jaxpr(jax.grad(shifted_square_sum))(x)
print("Derivative program:", derivative_program)
other = jnp.array([-1., 0., 2.])
assert jnp.allclose(shifted_square_sum(other), 10.)
assert jnp.allclose(jax.grad(shifted_square_sum)(other), jnp.array([0., 2., 6.]))
assert jnp.allclose(jax.grad(shifted_square_sum)(other), 2*(other+1))

# Experiment: Inspect a captured array
offset = jnp.array([1., 2., 3.])
def captured(z):
    return jnp.sum(z + offset)
def explicit(z, offset):
    return jnp.sum(z + offset)
captured_program = jax.make_jaxpr(captured)(x)
explicit_program = jax.make_jaxpr(explicit)(x, offset)
print("Captured:", captured_program)
print("Explicit:", explicit_program)
assert len(captured_program.jaxpr.invars) == 1
assert len(explicit_program.jaxpr.invars) == 2
assert jnp.allclose(captured(x), explicit(x, offset))

# Reference solution. Try the exercise before reading this.
def shifted_two(z):
    shifted = z + 2.
    return jnp.sum(shifted * shifted)
print(jax.make_jaxpr(shifted_two)(x))
assert jnp.allclose(shifted_two(x), 50.)
assert jnp.allclose(jax.grad(shifted_two)(x), jnp.array([6., 8., 10.]))

# Reference practice: Compare unrolled and scanned recurrences
def unrolled(z):
    for _ in range(4):
        z = 0.5*z + 1.
    return z
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
