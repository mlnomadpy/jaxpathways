"""Branches with lax.cond: worked experiments and reference solutions. CPU checks."""

# Prepare the inputs
import jax
import jax.numpy as jnp

# Build the computation
@jax.jit
def magnitude(x):
    return jax.lax.cond(x >= 0., lambda v: v, lambda v: -v, x)

# Run and check the result
print(float(magnitude(-3.)))
assert jnp.allclose(magnitude(-3.), 3.)
assert jnp.allclose(magnitude(2.), 2.)

import jax
import jax.numpy as jnp
@jax.jit
def magnitude(x):
    return jax.lax.cond(x >= 0., lambda v: v, lambda v: -v, x)
print(float(magnitude(-3.)))
assert jnp.allclose(magnitude(-3.), 3.)
assert jnp.allclose(magnitude(2.), 2.)

# Figure data experiment
grid = jnp.linspace(-3.0, 3.0, 25)
visual_data = {'kind': 'line', 'x': grid.tolist(), 'xlabel': 'input', 'ylabel': 'magnitude', 'series': [{'label': 'lax.cond output', 'y': jax.vmap(magnitude)(grid).tolist()}]}

# Experiment: Reproduce the Python truth-test failure
def eager_branch(v):
    if v>=0.:
        return v
    return -v
assert eager_branch(-3.)==3.
try:
    jax.jit(eager_branch)(jnp.array(-3.))
except jax.errors.TracerBoolConversionError:
    print("Expected traced Python Boolean failure")
else:
    raise AssertionError("Expected branch error")
for v in [-3.,0.,2.]:
    assert jnp.allclose(magnitude(v),abs(v))

# Experiment: Check a batched numerical selection
values=jnp.array([-3.,-1.,2.,4.])
assert jnp.allclose(jax.vmap(magnitude)(values),jnp.abs(values))
assert jnp.allclose(jax.vmap(jax.grad(magnitude))(values),jnp.array([-1.,-1.,1.,1.]))

# Reference solution. Try the exercise before reading this.
@jax.jit
def choose_update(x):
    return jax.lax.cond(x > 0., lambda v: v * 2., lambda v: v - 2., x)
assert jnp.allclose(choose_update(3.), 6.)
assert jnp.allclose(choose_update(-3.), -5.)
assert jnp.allclose(choose_update(0.), -2.)

# Reference practice: Match a changed piecewise rule
def piecewise(v):
    return jax.lax.cond(v>1.,lambda z:z*z,lambda z:2*z,v)
assert jnp.allclose(jax.vmap(piecewise)(jnp.array([-1.,1.,2.])),jnp.array([-2.,2.,4.]))
assert jnp.allclose(jax.grad(piecewise)(-1.),2.)
assert jnp.allclose(jax.grad(piecewise)(2.),4.)

# Reference practice: Repair incompatible branch shapes
try:
    jax.lax.cond(True,lambda x:x,lambda x:jnp.stack([x,x]),jnp.array(2.))
except TypeError:
    print("Expected branch shape mismatch")
else:
    raise AssertionError("Expected result type error")
def pair(v):
    return jax.lax.cond(v>=0.,lambda x:jnp.stack([x,x*x]),lambda x:jnp.stack([-x,x*x]),v)
assert jnp.allclose(pair(jnp.array(-2.)),jnp.array([2.,4.]))
print("PASS: state-04")
