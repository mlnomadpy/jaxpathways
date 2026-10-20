"""Vectorize trajectories and scan time: worked experiments and reference solutions. CPU checks."""

# Use a higher-order scalar transition
# Step 1 — Use a higher-order scalar transition: Runge–Kutta uses four slopes inside one step.
import jax
# Enable 64-bit arithmetic before importing jax.numpy or creating arrays.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

def rk4_step(rate, value, dt):
    a = -rate*value
    b = -rate*(value + dt*a/2)
    c = -rate*(value + dt*b/2)
    d = -rate*(value + dt*c)
    return value + dt*(a + 2*b + 2*c + d)/6

# Define `solve(rate, initial, steps, dt)` to carry state across steps with `jax.lax.scan`:
def solve(rate, initial, steps=40, dt=0.05):
    def advance(value, unused):
        next_value = rk4_step(rate, value, dt)
        return next_value, next_value
    _, tail = jax.lax.scan(advance, initial, None, length=steps)
    return jnp.concatenate((jnp.atleast_1d(initial), tail))

# Batch initial conditions while sharing the model
# Step 2 — Batch initial conditions while sharing the model: The batch axis is independent initial conditions, not time.
initials = jnp.array([0.5, 1.0, 2.0])
rate, dt, steps = 0.7, 0.05, 40
# Vectorize across the batch dimension with `jax.vmap` (`simulate_batch`).
simulate_batch = jax.jit(jax.vmap(lambda initial: solve(rate, initial, steps, dt)))
trajectories = simulate_batch(initials)
times = np.arange(steps+1)*dt
# Transfer the device result to the host for an independent NumPy comparison.
oracle = np.asarray(initials)[:,None]*np.exp(-rate*times[None,:])
# Check tensor shape invariant: `trajectories.shape == (3, 41)`
assert trajectories.shape == (3, 41)
np.testing.assert_allclose(trajectories, oracle, rtol=3e-8, atol=1e-10)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(trajectories)[2], 4*np.asarray(trajectories)[0], rtol=1e-12)
print("Shape:", trajectories.shape, "endpoints:", np.asarray(trajectories)[:,-1])

# Change composition and verify axis meaning
# Step 3 — Change composition and verify axis meaning: Array arithmetic applies the independent scalar dynamics to all...
# Define `scan_batch(rate, values, steps, dt)` to carry state across steps with `jax.lax.scan`:
def scan_batch(rate, values, steps=40, dt=0.05):
    def advance(state, unused):
        updated = rk4_step(rate, state, dt)
        return updated, updated
    _, tail = jax.lax.scan(advance, values, None, length=steps)
    return jnp.concatenate((values[None,:], tail), axis=0)
time_major = scan_batch(rate, initials)
# Check tensor shape invariant: `time_major.shape == (41, 3)`
assert time_major.shape == (41, 3)
np.testing.assert_allclose(time_major.T, trajectories, rtol=1e-12, atol=1e-12)
print("scan(vectors) transposed equals vmap(scan)")

# Step 1 — Use a higher-order scalar transition: Runge–Kutta uses four slopes inside one step.
import jax
# Enable 64-bit arithmetic before importing jax.numpy or creating arrays.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

def rk4_step(rate, value, dt):
    a = -rate*value
    b = -rate*(value + dt*a/2)
    c = -rate*(value + dt*b/2)
    d = -rate*(value + dt*c)
    return value + dt*(a + 2*b + 2*c + d)/6

# Define `solve(rate, initial, steps, dt)` to carry state across steps with `jax.lax.scan`:
def solve(rate, initial, steps=40, dt=0.05):
    def advance(value, unused):
        next_value = rk4_step(rate, value, dt)
        return next_value, next_value
    _, tail = jax.lax.scan(advance, initial, None, length=steps)
    return jnp.concatenate((jnp.atleast_1d(initial), tail))

# Step 2 — Batch initial conditions while sharing the model: The batch axis is independent initial conditions, not time.
initials = jnp.array([0.5, 1.0, 2.0])
rate, dt, steps = 0.7, 0.05, 40
# Vectorize across the batch dimension with `jax.vmap` (`simulate_batch`).
simulate_batch = jax.jit(jax.vmap(lambda initial: solve(rate, initial, steps, dt)))
trajectories = simulate_batch(initials)
times = np.arange(steps+1)*dt
# Transfer the device result to the host for an independent NumPy comparison.
oracle = np.asarray(initials)[:,None]*np.exp(-rate*times[None,:])
# Check tensor shape invariant: `trajectories.shape == (3, 41)`
assert trajectories.shape == (3, 41)
np.testing.assert_allclose(trajectories, oracle, rtol=3e-8, atol=1e-10)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(np.asarray(trajectories)[2], 4*np.asarray(trajectories)[0], rtol=1e-12)
print("Shape:", trajectories.shape, "endpoints:", np.asarray(trajectories)[:,-1])

# Step 3 — Change composition and verify axis meaning: Array arithmetic applies the independent scalar dynamics to all...
# Define `scan_batch(rate, values, steps, dt)` to carry state across steps with `jax.lax.scan`:
def scan_batch(rate, values, steps=40, dt=0.05):
    def advance(state, unused):
        updated = rk4_step(rate, state, dt)
        return updated, updated
    _, tail = jax.lax.scan(advance, values, None, length=steps)
    return jnp.concatenate((values[None,:], tail), axis=0)
time_major = scan_batch(rate, initials)
# Check tensor shape invariant: `time_major.shape == (41, 3)`
assert time_major.shape == (41, 3)
np.testing.assert_allclose(time_major.T, trajectories, rtol=1e-12, atol=1e-12)
print("scan(vectors) transposed equals vmap(scan)")

# Figure data experiment
# Transfer the device result to the host for an independent NumPy comparison.
visual_data={'kind':'line','x':times.tolist(),'xlabel':'time (seconds)','ylabel':'excess temperature','series':[{'label':f'initial {float(u):g}','y':np.asarray(trajectories)[i].tolist()} for i,u in enumerate(initials)]}

# Experiment: Compare RK4 with an independent polynomial
# Experiment — Compare RK4 with an independent polynomial: This checks the discrete algorithm independently of the...
z = -rate*dt
multiplier = 1 + z + z*z/2 + z**3/6 + z**4/24
polynomial = np.asarray(initials)[:,None]*multiplier**np.arange(steps+1)[None,:]
np.testing.assert_allclose(trajectories, polynomial, rtol=1e-12, atol=1e-12)
print("RK4 polynomial maximum difference:", np.max(np.abs(np.asarray(trajectories)-polynomial)))

# Experiment: Let each trajectory have its own rate
# Experiment — Let each trajectory have its own rate: Mapping both axes pairs corresponding entries.
rates = jnp.array([0.2, 0.7, 1.1])
# Vectorize across the batch dimension with `jax.vmap` (`paired`).
paired = jax.vmap(lambda k, u: solve(k,u), in_axes=(0,0))(rates,initials)
# Transfer the device result to the host for an independent NumPy comparison.
paired_reference = np.asarray(initials)[:,None]*np.exp(-np.asarray(rates)[:,None]*times)
np.testing.assert_allclose(paired,paired_reference,rtol=2e-7,atol=1e-10)
print("Paired-rate endpoints:", np.asarray(paired)[:,-1])

# Reference solution. Try the exercise before reading this.
# Exercise solution: Add a fourth initial condition, verify the changed output shape and...
changed_initials = jnp.array([0.5, 1.0, 2.0, 3.0])
# Vectorize across the batch dimension with `jax.vmap` (`changed`).
changed = jax.vmap(lambda u: solve(0.7,u))(changed_initials)
# Check tensor shape invariant: `changed.shape == (4,41)`
assert changed.shape == (4,41)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(changed,np.asarray(changed_initials)[:,None]*np.exp(-0.7*times),rtol=3e-8)
order = jnp.array([3,0,2,1])
# Vectorize across the batch dimension with `jax.vmap` (`reordered`).
reordered = jax.vmap(lambda u: solve(0.7,u))(changed_initials[order])
np.testing.assert_allclose(reordered,changed[order],rtol=1e-12)
print("Permutation preserves trajectory identity")

# Reference practice: Save only what the objective needs
# Save only what the objective needs (Practice): The explicit trajectory output is removed.
# Define `final_only(rate, initial, steps, dt)` to carry state across steps with `jax.lax.scan`:
def final_only(rate, initial, steps=40, dt=0.05):
    def advance(value, unused):
        return rk4_step(rate,value,dt), None
    return jax.lax.scan(advance,initial,None,length=steps)[0]
# Vectorize across the batch dimension with `jax.vmap` (`finals`).
finals = jax.vmap(lambda u: final_only(0.7,u))(initials)
np.testing.assert_allclose(finals,trajectories[:,-1],rtol=1e-12)
print("Only final states returned:", np.asarray(finals))

# Reference practice: Pairing versus Cartesian products
# Pairing versus Cartesian products (Challenge): Paired batches and Cartesian products are different experiments.
# Vectorize across the batch dimension with `jax.vmap` (`grid`).
grid = jax.vmap(lambda k: jax.vmap(lambda u: solve(k,u))(initials))(rates)
# Check tensor shape invariant: `grid.shape == (3,3,41)`
assert grid.shape == (3,3,41)
# Create evenly spaced index values in ``.
np.testing.assert_allclose(grid[jnp.arange(3),jnp.arange(3)],paired,rtol=1e-12)
print("Grid axes: rate, initial condition, time", grid.shape)
print("PASS: science-02")
