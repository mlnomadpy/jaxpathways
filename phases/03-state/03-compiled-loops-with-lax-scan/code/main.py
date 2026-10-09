"""Compiled loops with lax.scan: worked experiments and reference solutions. CPU checks."""

# Prepare the inputs
# Step 1 — Prepare the inputs: These explicit inputs define the case that the later checks will...
# Import jax for this computation.
import jax
import jax.numpy as jnp

# Build the computation
# Step 2 — Build the computation: step returns the next scalar carry and one recorded scalar.
def step(carry, increment):
    # Compute `next_value` from `0.5 * carry + increment`
    next_value = 0.5 * carry + increment
    # Return `(next_value, next_value)` to the caller.
    return next_value, next_value
# Construct `increments` via `jnp.array([1., 1., 1., 1.])`
increments = jnp.array([1., 1., 1., 1.])

# Run and check the result
# Step 3 — Run and check the result: Compare the output to the expected result below before making the...
# Run compiled structured control flow via `jax.lax` (`(final, history)`).
final, history = jax.lax.scan(step, jnp.array(0.), increments)
# Print the observed values to compare against the expected result.
print("History:", history)
# Print diagnostic summary of the computed outputs.
print("Final:", float(final))
# Assert that `jnp.allclose(history, jnp.array([1.,1.5,1.75,1.875]))`.
assert jnp.allclose(history, jnp.array([1.,1.5,1.75,1.875]))
# Assert that `jnp.allclose(final, history[-1])`.
assert jnp.allclose(final, history[-1])

# Step 1 — Prepare the inputs: These explicit inputs define the case that the later checks will...
# Import jax for this computation.
import jax
import jax.numpy as jnp
# Step 2 — Build the computation: step returns the next scalar carry and one recorded scalar.
def step(carry, increment):
    # Compute `next_value` from `0.5 * carry + increment`
    next_value = 0.5 * carry + increment
    # Return `(next_value, next_value)` to the caller.
    return next_value, next_value
# Construct `increments` via `jnp.array([1., 1., 1., 1.])`
increments = jnp.array([1., 1., 1., 1.])
# Step 3 — Run and check the result: Compare the output to the expected result below before making the...
# Run compiled structured control flow via `jax.lax` (`(final, history)`).
final, history = jax.lax.scan(step, jnp.array(0.), increments)
# Print the observed values to compare against the expected result.
print("History:", history)
# Print diagnostic summary of the computed outputs.
print("Final:", float(final))
# Assert that `jnp.allclose(history, jnp.array([1.,1.5,1.75,1.875]))`.
assert jnp.allclose(history, jnp.array([1.,1.5,1.75,1.875]))
# Assert that `jnp.allclose(final, history[-1])`.
assert jnp.allclose(final, history[-1])

# Figure data experiment
# Compute figure data for: The carry approaches a fixed point
# Compute `visual_data` from `{'kind': 'line', 'x': [1, 2, 3, 4], 'xlabel': 'compl...`
visual_data = {'kind': 'line', 'x': [1, 2, 3, 4], 'xlabel': 'completed step', 'ylabel': 'carry value', 'series': [{'label': 'scan history', 'y': history.tolist()}, {'label': 'fixed point 2', 'y': [2.0] * 4}]}

# Experiment: Compare scan to the Python recurrence
# Experiment — Compare scan to the Python recurrence: The loop is an independent control-flow reference and the fixed...
# Construct `inputs` via `jnp.array([1.,-1.,2.,0.])`
inputs=jnp.array([1.,-1.,2.,0.])
# Compute `state` from `0.`
state=0.
# Compute `reference` from `[]`
reference=[]
# Iterate over `inc` to step through the computation:
for inc in [1.,-1.,2.,0.]:
    # Compute `state` from `0.5*state+inc`
    state=0.5*state+inc
    # Append the current step result to `reference`.
    reference.append(state)
# Run compiled structured control flow via `jax.lax` (`(last, observed)`).
last,observed=jax.lax.scan(step,jnp.array(0.),inputs)
# Assert that `jnp.allclose(observed,jnp.array([1.,-0.5,1.75,0.875]))`.
assert jnp.allclose(observed,jnp.array([1.,-0.5,1.75,0.875]))
# Assert that `jnp.allclose(observed,jnp.array(reference))`.
assert jnp.allclose(observed,jnp.array(reference))
# Assert that `jnp.allclose(last,state)`.
assert jnp.allclose(last,state)

# Experiment: Verify state and sensitivity together
# Experiment — Verify state and sensitivity together: The scalar objective is the final state.
# Define `terminal(d)` to carry state across steps with `jax.lax.scan`:
def terminal(d):
    # Return `jax.lax.scan(lambda c, u: (d * c + u, d * c + u), jnp.array(0.0), increments)[0]` to the caller.
    return jax.lax.scan(lambda c,u:(d*c+u,d*c+u),jnp.array(0.),increments)[0]
# Differentiate the objective to obtain `(value, sensitivity)` via automatic differentiation.
value,sensitivity=jax.value_and_grad(terminal)(0.5)
# Assert that `jnp.allclose(value,1.875)`.
assert jnp.allclose(value,1.875)
# Assert that `jnp.allclose(sensitivity,2.75)`.
assert jnp.allclose(sensitivity,2.75)
# Assert that `jnp.allclose(jax.jit(terminal)(0.5),value)`.
assert jnp.allclose(jax.jit(terminal)(0.5),value)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Make the decay an explicit scalar argument.
# Define `simulate(decay)` to carry state across steps with `jax.lax.scan`:
def simulate(decay):
    # Function `transition(carry, increment)` implementing this stage's computation:
    def transition(carry, increment):
        # Compute `value` from `decay * carry + increment`
        value = decay * carry + increment
        # Return `(value, value)` to the caller.
        return value, value
    # Return `jax.lax.scan(transition, jnp.array(0.0), increments)[0]` to the caller.
    return jax.lax.scan(transition, jnp.array(0.), increments)[0]
# Assert that `jnp.allclose(jax.grad(simulate)(0.5), 2.75)`.
assert jnp.allclose(jax.grad(simulate)(0.5), 2.75)

# Reference practice: Record a different output
# Record a different output (Practice): Carry and history need not have the same interpretation.
# Run compiled structured control flow via `jax.lax` (`(last, energy)`).
last,energy=jax.lax.scan(lambda c,u:(0.5*c+u,(0.5*c+u)**2),jnp.array(0.),increments)
# Assert that `jnp.allclose(last,1.875)`.
assert jnp.allclose(last,1.875)
# Assert that `jnp.allclose(energy,jnp.array([1.,2.25,3.0625,3.515625]))`.
assert jnp.allclose(energy,jnp.array([1.,2.25,3.0625,3.515625]))

# Reference practice: Diagnose a growing carry
# Diagnose a growing carry (Challenge): The error identifies incompatible carry input/output types.
def growing(c,u):
    # Return `(jnp.concatenate([c, u[None]]), u)` to the caller.
    return jnp.concatenate([c,u[None]]),u
# Run the boundary check and catch the expected exception:
try:
    jax.lax.scan(growing,jnp.zeros((0,)),increments)
except TypeError:
    print("Expected carry-shape mismatch")
else:
    raise AssertionError("Expected scan type failure")
# Run compiled structured control flow via `jax.lax` (`(_, saved)`).
_,saved=jax.lax.scan(lambda c,u:(c,u),jnp.array(0.),increments)
# Assert invariant `jnp.array_equal(saved,increments)` holds
assert jnp.array_equal(saved,increments)
print("PASS: state-03")
