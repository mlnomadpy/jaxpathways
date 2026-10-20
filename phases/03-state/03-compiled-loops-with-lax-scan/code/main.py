"""Compiled loops with lax.scan: worked experiments and reference solutions. CPU checks."""

# Prepare the inputs
# Step 1 — Prepare the inputs
import jax
import jax.numpy as jnp

# Build the computation
# Step 2 — Build the computation: step returns the next scalar carry and one recorded scalar.
def step(carry, increment):
    next_value = 0.5 * carry + increment
    return next_value, next_value
increments = jnp.array([1., 1., 1., 1.])

# Run and check the result
# Step 3 — Run and check the result
final, history = jax.lax.scan(step, jnp.array(0.), increments)
print("History:", history)
print("Final:", float(final))
assert jnp.allclose(history, jnp.array([1.,1.5,1.75,1.875]))
assert jnp.allclose(final, history[-1])

# Step 1 — Prepare the inputs
import jax
import jax.numpy as jnp
# Step 2 — Build the computation: step returns the next scalar carry and one recorded scalar.
def step(carry, increment):
    next_value = 0.5 * carry + increment
    return next_value, next_value
increments = jnp.array([1., 1., 1., 1.])
# Step 3 — Run and check the result
final, history = jax.lax.scan(step, jnp.array(0.), increments)
print("History:", history)
print("Final:", float(final))
assert jnp.allclose(history, jnp.array([1.,1.5,1.75,1.875]))
assert jnp.allclose(final, history[-1])

# Figure data experiment
visual_data = {'kind': 'line', 'x': [1, 2, 3, 4], 'xlabel': 'completed step', 'ylabel': 'carry value', 'series': [{'label': 'scan history', 'y': history.tolist()}, {'label': 'fixed point 2', 'y': [2.0] * 4}]}

# Experiment: Compare scan to the Python recurrence
# Experiment — Compare scan to the Python recurrence
inputs=jnp.array([1.,-1.,2.,0.])
state=0.
reference=[]
for inc in [1.,-1.,2.,0.]:
    state=0.5*state+inc
    reference.append(state)
last,observed=jax.lax.scan(step,jnp.array(0.),inputs)
assert jnp.allclose(observed,jnp.array([1.,-0.5,1.75,0.875]))
assert jnp.allclose(observed,jnp.array(reference))
assert jnp.allclose(last,state)

# Experiment: Verify state and sensitivity together
# Experiment — Verify state and sensitivity together: The scalar objective is the final state.
# Define `terminal(d)` to carry state across steps with `jax.lax.scan`:
def terminal(d):
    return jax.lax.scan(lambda c,u:(d*c+u,d*c+u),jnp.array(0.),increments)[0]
value,sensitivity=jax.value_and_grad(terminal)(0.5)
assert jnp.allclose(value,1.875)
assert jnp.allclose(sensitivity,2.75)
assert jnp.allclose(jax.jit(terminal)(0.5),value)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Make the decay an explicit scalar argument.
# Define `simulate(decay)` to carry state across steps with `jax.lax.scan`:
def simulate(decay):
    def transition(carry, increment):
        value = decay * carry + increment
        return value, value
    return jax.lax.scan(transition, jnp.array(0.), increments)[0]
assert jnp.allclose(jax.grad(simulate)(0.5), 2.75)

# Reference practice: Record a different output
# Record a different output (Practice): Carry and history need not have the same interpretation.
last,energy=jax.lax.scan(lambda c,u:(0.5*c+u,(0.5*c+u)**2),jnp.array(0.),increments)
assert jnp.allclose(last,1.875)
assert jnp.allclose(energy,jnp.array([1.,2.25,3.0625,3.515625]))

# Reference practice: Diagnose a growing carry
# Diagnose a growing carry (Challenge): The error identifies incompatible carry input/output types.
def growing(c,u):
    return jnp.concatenate([c,u[None]]),u
try:
    jax.lax.scan(growing,jnp.zeros((0,)),increments)
except TypeError:
    print("Expected carry-shape mismatch")
else:
    raise AssertionError("Expected scan type failure")
_,saved=jax.lax.scan(lambda c,u:(c,u),jnp.array(0.),increments)
assert jnp.array_equal(saved,increments)
print("PASS: state-03")
