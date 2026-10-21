"""A functional physical simulation: worked experiments and reference solutions. CPU checks."""

# Define a state transition
# Step 1 — Define a state transition: A scalar state enters and a new scalar leaves.
import jax
# Enable 64-bit arithmetic before importing jax.numpy or creating arrays.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

def euler_step(rate, value, dt):
    return value - dt*rate*value

# Define `euler_solve(rate, initial, steps, dt)` to carry state across steps with `jax.lax.scan`:
def euler_solve(rate, initial, steps, dt):
    def advance(value, unused):
        updated = euler_step(rate, value, dt)
        return updated, updated
    _, tail = jax.lax.scan(advance, initial, None, length=steps)
    return jnp.concatenate((jnp.atleast_1d(initial), tail))

# Compare the simulation with a physical reference
# Step 2 — Compare the simulation with a physical reference: The geometric sequence verifies implementation correctness.
rate, initial, dt, steps = 0.7, jnp.array(2.0), 0.5, 4
trajectory = euler_solve(rate, initial, steps, dt)
times = np.arange(steps+1)*dt
continuous = 2.0*np.exp(-rate*times)
discrete = 2.0*(1-rate*dt)**np.arange(steps+1)
np.testing.assert_allclose(trajectory, discrete, rtol=1e-12, atol=1e-12)
assert np.all(np.diff(np.asarray(trajectory)) < 0)
assert abs(float(trajectory[-1])-continuous[-1]) > 0.1
print("Euler final:", float(trajectory[-1]), "analytic final:", continuous[-1])

# Refine the time grid without changing the experiment
# Step 3 — Refine the time grid without changing the experiment: Every run stops at the same physical time.
grid_sizes = np.array([4, 8, 16, 32])
endpoint_errors = np.array([abs(float(euler_solve(rate, initial, int(n), 2.0/n)[-1])-continuous[-1]) for n in grid_sizes])
assert np.all(np.diff(endpoint_errors) < 0)
assert 1.7 < endpoint_errors[-2]/endpoint_errors[-1] < 2.3
print("Endpoint errors:", endpoint_errors)
print("Last refinement ratio:", endpoint_errors[-2]/endpoint_errors[-1])

# Step 1 — Define a state transition: A scalar state enters and a new scalar leaves.
import jax
# Enable 64-bit arithmetic before importing jax.numpy or creating arrays.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

def euler_step(rate, value, dt):
    return value - dt*rate*value

# Define `euler_solve(rate, initial, steps, dt)` to carry state across steps with `jax.lax.scan`:
def euler_solve(rate, initial, steps, dt):
    def advance(value, unused):
        updated = euler_step(rate, value, dt)
        return updated, updated
    _, tail = jax.lax.scan(advance, initial, None, length=steps)
    return jnp.concatenate((jnp.atleast_1d(initial), tail))

# Step 2 — Compare the simulation with a physical reference: The geometric sequence verifies implementation correctness.
rate, initial, dt, steps = 0.7, jnp.array(2.0), 0.5, 4
trajectory = euler_solve(rate, initial, steps, dt)
times = np.arange(steps+1)*dt
continuous = 2.0*np.exp(-rate*times)
discrete = 2.0*(1-rate*dt)**np.arange(steps+1)
np.testing.assert_allclose(trajectory, discrete, rtol=1e-12, atol=1e-12)
assert np.all(np.diff(np.asarray(trajectory)) < 0)
assert abs(float(trajectory[-1])-continuous[-1]) > 0.1
print("Euler final:", float(trajectory[-1]), "analytic final:", continuous[-1])

# Step 3 — Refine the time grid without changing the experiment: Every run stops at the same physical time.
grid_sizes = np.array([4, 8, 16, 32])
endpoint_errors = np.array([abs(float(euler_solve(rate, initial, int(n), 2.0/n)[-1])-continuous[-1]) for n in grid_sizes])
assert np.all(np.diff(endpoint_errors) < 0)
assert 1.7 < endpoint_errors[-2]/endpoint_errors[-1] < 2.3
print("Endpoint errors:", endpoint_errors)
print("Last refinement ratio:", endpoint_errors[-2]/endpoint_errors[-1])

# Figure data experiment
fine = np.asarray(euler_solve(0.7, jnp.array(2.0), 8, 0.25))[::2]
# Transfer the device result to the host for an independent NumPy comparison.
visual_data = {'kind':'line','x':times.tolist(),'xlabel':'time (seconds)','ylabel':'excess temperature','series':[{'label':'Euler h=0.5','y':np.asarray(trajectory).tolist()},{'label':'Euler h=0.25 (sampled)','y':fine.tolist()},{'label':'analytic exponential','y':continuous.tolist()}]}

# Experiment: Stable does not mean physically positive
# Experiment — Stable does not mean physically positive
oscillatory = euler_solve(0.7, jnp.array(2.0), 4, 2.0)
np.testing.assert_allclose(oscillatory, [2.0, -0.8, 0.32, -0.128, 0.0512], atol=1e-12)
print("Stable but sign-alternating:", np.asarray(oscillatory))

# Experiment: Reproduce numerical instability
# Experiment — Reproduce numerical instability: The multiplier is -1.1.
unstable = euler_solve(0.7, jnp.array(2.0), 6, 3.0)
assert abs(float(unstable[-1])) > 2.0
assert 2*np.exp(-0.7*18) < 1e-4
print("Unstable numerical endpoint:", float(unstable[-1]))

# Reference solution. Try the exercise before reading this.
# Exercise solution
changed = euler_solve(0.3, jnp.array(5.0), 20, 0.1)
reference = 5.0*(1-0.3*0.1)**np.arange(21)
np.testing.assert_allclose(changed, reference, rtol=1e-12)
assert float(changed[0]) == 5.0
assert abs(float(changed[-1])-5*np.exp(-0.6)) < 0.03
print("Changed rate, initial state and grid verified")

# Reference practice: Energy does not check every invariant
# Energy does not check every invariant (Practice)
for h in (0.5, 2.0):
    u = np.asarray(euler_solve(0.7, jnp.array(2.0), 6, h))
    assert np.all(np.diff(u*u/2) < 0)
u = np.asarray(euler_solve(0.7, jnp.array(2.0), 6, 3.0))
assert np.all(np.diff(u*u/2) > 0)
print("Energy checks distinguish growth, but not sign preservation")

# Reference practice: Keep the physical horizon fixed
# Keep the physical horizon fixed (Challenge)
horizon = 2.0
errors = []
for n in (10, 20, 40):
    h = horizon/n
    final = euler_solve(0.7, jnp.array(2.0), n, h)[-1]
    errors.append(abs(float(final)-2*np.exp(-0.7*horizon)))
assert errors[2] < errors[1] < errors[0]
print("Same-horizon errors:", errors)
print("PASS: science-01")
