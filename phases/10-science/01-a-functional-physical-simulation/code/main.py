"""A functional physical simulation: worked experiments and reference solutions. CPU checks."""

# Define a state transition
# Step 1 — Define a state transition: A scalar state enters and a new scalar leaves.
# Import jax for this computation.
import jax
# Update state in place with the new values.
jax.config.update("jax_enable_x64", True)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np

# Function `euler_step(rate, value, dt)` implementing this stage's computation:
def euler_step(rate, value, dt):
    # Return `value - dt * rate * value` to the caller.
    return value - dt*rate*value

# Define `euler_solve(rate, initial, steps, dt)` to carry state across steps with `jax.lax.scan`:
def euler_solve(rate, initial, steps, dt):
    # Function `advance(value, unused)` implementing this stage's computation:
    def advance(value, unused):
        # Run `euler_step` to compute `updated`.
        updated = euler_step(rate, value, dt)
        # Return `(updated, updated)` to the caller.
        return updated, updated
    # Run compiled structured control flow via `jax.lax` (`(_, tail)`).
    _, tail = jax.lax.scan(advance, initial, None, length=steps)
    # Return `jnp.concatenate((jnp.atleast_1d(initial), tail))` to the caller.
    return jnp.concatenate((jnp.atleast_1d(initial), tail))

# Compare the simulation with a physical reference
# Step 2 — Compare the simulation with a physical reference: The geometric sequence verifies implementation correctness.
# Construct `rate, initial, dt, steps` via `0.7, jnp.array(2.0), 0.5, 4`
rate, initial, dt, steps = 0.7, jnp.array(2.0), 0.5, 4
# Run `euler_solve` to compute `trajectory`.
trajectory = euler_solve(rate, initial, steps, dt)
# Compute `times` from `np.arange(steps+1)*dt`
times = np.arange(steps+1)*dt
# Compute `continuous` from `2.0*np.exp(-rate*times)`
continuous = 2.0*np.exp(-rate*times)
# Compute `discrete` from `2.0*(1-rate*dt)**np.arange(steps+1)`
discrete = 2.0*(1-rate*dt)**np.arange(steps+1)
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(trajectory, discrete, rtol=1e-12, atol...`
np.testing.assert_allclose(trajectory, discrete, rtol=1e-12, atol=1e-12)
# Assert invariant `np.all(np.diff(np.asarray(trajectory)) < 0)` holds
assert np.all(np.diff(np.asarray(trajectory)) < 0)
# Check numerical equivalence within tolerance: `abs(float(trajectory[-1])-continuous[-1]) > 0.1`
assert abs(float(trajectory[-1])-continuous[-1]) > 0.1
# Print the observed values to compare against the expected result.
print("Euler final:", float(trajectory[-1]), "analytic final:", continuous[-1])

# Refine the time grid without changing the experiment
# Step 3 — Refine the time grid without changing the experiment: Every run stops at the same physical time.
# Compute `grid_sizes` from `np.array([4, 8, 16, 32])`
grid_sizes = np.array([4, 8, 16, 32])
# Compute `endpoint_errors` from `np.array([abs(float(euler_solve(rate, initial, int(n...`
endpoint_errors = np.array([abs(float(euler_solve(rate, initial, int(n), 2.0/n)[-1])-continuous[-1]) for n in grid_sizes])
# Assert invariant `np.all(np.diff(endpoint_errors) < 0)` holds
assert np.all(np.diff(endpoint_errors) < 0)
# Assert invariant `1.7 < endpoint_errors[-2]/endpoint_errors[-1] < 2.3` holds
assert 1.7 < endpoint_errors[-2]/endpoint_errors[-1] < 2.3
# Print the observed values to compare against the expected result.
print("Endpoint errors:", endpoint_errors)
# Print diagnostic summary of the computed outputs.
print("Last refinement ratio:", endpoint_errors[-2]/endpoint_errors[-1])

# Step 1 — Define a state transition: A scalar state enters and a new scalar leaves.
# Import jax for this computation.
import jax
# Update state in place with the new values.
jax.config.update("jax_enable_x64", True)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np

# Function `euler_step(rate, value, dt)` implementing this stage's computation:
def euler_step(rate, value, dt):
    # Return `value - dt * rate * value` to the caller.
    return value - dt*rate*value

# Define `euler_solve(rate, initial, steps, dt)` to carry state across steps with `jax.lax.scan`:
def euler_solve(rate, initial, steps, dt):
    # Function `advance(value, unused)` implementing this stage's computation:
    def advance(value, unused):
        # Run `euler_step` to compute `updated`.
        updated = euler_step(rate, value, dt)
        # Return `(updated, updated)` to the caller.
        return updated, updated
    # Run compiled structured control flow via `jax.lax` (`(_, tail)`).
    _, tail = jax.lax.scan(advance, initial, None, length=steps)
    # Return `jnp.concatenate((jnp.atleast_1d(initial), tail))` to the caller.
    return jnp.concatenate((jnp.atleast_1d(initial), tail))

# Step 2 — Compare the simulation with a physical reference: The geometric sequence verifies implementation correctness.
# Construct `rate, initial, dt, steps` via `0.7, jnp.array(2.0), 0.5, 4`
rate, initial, dt, steps = 0.7, jnp.array(2.0), 0.5, 4
# Run `euler_solve` to compute `trajectory`.
trajectory = euler_solve(rate, initial, steps, dt)
# Compute `times` from `np.arange(steps+1)*dt`
times = np.arange(steps+1)*dt
# Compute `continuous` from `2.0*np.exp(-rate*times)`
continuous = 2.0*np.exp(-rate*times)
# Compute `discrete` from `2.0*(1-rate*dt)**np.arange(steps+1)`
discrete = 2.0*(1-rate*dt)**np.arange(steps+1)
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(trajectory, discrete, rtol=1e-12, atol...`
np.testing.assert_allclose(trajectory, discrete, rtol=1e-12, atol=1e-12)
# Assert invariant `np.all(np.diff(np.asarray(trajectory)) < 0)` holds
assert np.all(np.diff(np.asarray(trajectory)) < 0)
# Check numerical equivalence within tolerance: `abs(float(trajectory[-1])-continuous[-1]) > 0.1`
assert abs(float(trajectory[-1])-continuous[-1]) > 0.1
# Print the observed values to compare against the expected result.
print("Euler final:", float(trajectory[-1]), "analytic final:", continuous[-1])

# Step 3 — Refine the time grid without changing the experiment: Every run stops at the same physical time.
# Compute `grid_sizes` from `np.array([4, 8, 16, 32])`
grid_sizes = np.array([4, 8, 16, 32])
# Compute `endpoint_errors` from `np.array([abs(float(euler_solve(rate, initial, int(n...`
endpoint_errors = np.array([abs(float(euler_solve(rate, initial, int(n), 2.0/n)[-1])-continuous[-1]) for n in grid_sizes])
# Assert invariant `np.all(np.diff(endpoint_errors) < 0)` holds
assert np.all(np.diff(endpoint_errors) < 0)
# Assert invariant `1.7 < endpoint_errors[-2]/endpoint_errors[-1] < 2.3` holds
assert 1.7 < endpoint_errors[-2]/endpoint_errors[-1] < 2.3
# Print the observed values to compare against the expected result.
print("Endpoint errors:", endpoint_errors)
# Print diagnostic summary of the computed outputs.
print("Last refinement ratio:", endpoint_errors[-2]/endpoint_errors[-1])

# Figure data experiment
# Compute figure data for: A correct Euler program still has time-step error
# Create device-backed JAX array `fine`.
fine = np.asarray(euler_solve(0.7, jnp.array(2.0), 8, 0.25))[::2]
# Convert `visual_data` to a host NumPy array for inspection or verification.
visual_data = {'kind':'line','x':times.tolist(),'xlabel':'time (seconds)','ylabel':'excess temperature','series':[{'label':'Euler h=0.5','y':np.asarray(trajectory).tolist()},{'label':'Euler h=0.25 (sampled)','y':fine.tolist()},{'label':'analytic exponential','y':continuous.tolist()}]}

# Experiment: Stable does not mean physically positive
# Experiment — Stable does not mean physically positive: The magnitude decays because |1-kh|=0.4, but crossing below...
# Construct `oscillatory` via `euler_solve(0.7, jnp.array(2.0), 4, 2.0)`
oscillatory = euler_solve(0.7, jnp.array(2.0), 4, 2.0)
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(oscillatory, [2.0, -0.8, 0.32, -0.128,...`
np.testing.assert_allclose(oscillatory, [2.0, -0.8, 0.32, -0.128, 0.0512], atol=1e-12)
# Print the observed values to compare against the expected result.
print("Stable but sign-alternating:", np.asarray(oscillatory))

# Experiment: Reproduce numerical instability
# Experiment — Reproduce numerical instability: The multiplier is -1.1.
# Construct `unstable` via `euler_solve(0.7, jnp.array(2.0), 6, 3.0)`
unstable = euler_solve(0.7, jnp.array(2.0), 6, 3.0)
# Check numerical equivalence within tolerance: `abs(float(unstable[-1])) > 2.0`
assert abs(float(unstable[-1])) > 2.0
# Assert invariant `2*np.exp(-0.7*18) < 1e-4` holds
assert 2*np.exp(-0.7*18) < 1e-4
# Print the observed values to compare against the expected result.
print("Unstable numerical endpoint:", float(unstable[-1]))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Change the rate to 0.3, the initial excess to 5, and use twenty steps...
# Construct `changed` via `euler_solve(0.3, jnp.array(5.0), 20, 0.1)`
changed = euler_solve(0.3, jnp.array(5.0), 20, 0.1)
# Compute `reference` from `5.0*(1-0.3*0.1)**np.arange(21)`
reference = 5.0*(1-0.3*0.1)**np.arange(21)
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(changed, reference, rtol=1e-12)`
np.testing.assert_allclose(changed, reference, rtol=1e-12)
# Assert invariant `float(changed[0]) == 5.0` holds
assert float(changed[0]) == 5.0
# Check numerical equivalence within tolerance: `abs(float(changed[-1])-5*np.exp(-0.6)) < 0.03`
assert abs(float(changed[-1])-5*np.exp(-0.6)) < 0.03
# Print the observed values to compare against the expected result.
print("Changed rate, initial state and grid verified")

# Reference practice: Energy does not check every invariant
# Energy does not check every invariant (Practice): A scalar diagnostic can establish decay without capturing...
# Iterate over `h` to step through the computation:
for h in (0.5, 2.0):
    # Construct `u` via `np.asarray(euler_solve(0.7, jnp.array(2.0), 6, h))`
    u = np.asarray(euler_solve(0.7, jnp.array(2.0), 6, h))
    # Assert invariant `np.all(np.diff(u*u/2) < 0)` holds
    assert np.all(np.diff(u*u/2) < 0)
# Construct `u` via `np.asarray(euler_solve(0.7, jnp.array(2.0), 6, 3.0))`
u = np.asarray(euler_solve(0.7, jnp.array(2.0), 6, 3.0))
# Assert invariant `np.all(np.diff(u*u/2) > 0)` holds
assert np.all(np.diff(u*u/2) > 0)
# Print the observed values to compare against the expected result.
print("Energy checks distinguish growth, but not sign preservation")

# Reference practice: Keep the physical horizon fixed
# Keep the physical horizon fixed (Challenge): A refinement study must hold the modeled experiment fixed;...
horizon = 2.0
# Compute `errors` from `[]`
errors = []
# Iterate over `n` to step through the computation:
for n in (10, 20, 40):
    # Compute `h` from `horizon/n`
    h = horizon/n
    # Construct `final` via `euler_solve(0.7, jnp.array(2.0), n, h)[-1]`
    final = euler_solve(0.7, jnp.array(2.0), n, h)[-1]
    # Append the current step result to `errors`.
    errors.append(abs(float(final)-2*np.exp(-0.7*horizon)))
# Assert invariant `errors[2] < errors[1] < errors[0]` holds
assert errors[2] < errors[1] < errors[0]
# Print the observed values to compare against the expected result.
print("Same-horizon errors:", errors)
print("PASS: science-01")
