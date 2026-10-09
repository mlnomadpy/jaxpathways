"""Vectorize trajectories and scan time: worked experiments and reference solutions. CPU checks."""

# Use a higher-order scalar transition
# Step 1 — Use a higher-order scalar transition: Runge–Kutta uses four slopes inside one step.
# Import jax for this computation.
import jax
# Update state in place with the new values.
jax.config.update("jax_enable_x64", True)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np

# Function `rk4_step(rate, value, dt)` implementing this stage's computation:
def rk4_step(rate, value, dt):
    # Compute `a` from `-rate*value`
    a = -rate*value
    # Compute `b` from `-rate*(value + dt*a/2)`
    b = -rate*(value + dt*a/2)
    # Compute `c` from `-rate*(value + dt*b/2)`
    c = -rate*(value + dt*b/2)
    # Compute `d` from `-rate*(value + dt*c)`
    d = -rate*(value + dt*c)
    # Return `value + dt * (a + 2 * b + 2 * c + d) / 6` to the caller.
    return value + dt*(a + 2*b + 2*c + d)/6

# Define `solve(rate, initial, steps, dt)` to carry state across steps with `jax.lax.scan`:
def solve(rate, initial, steps=40, dt=0.05):
    # Function `advance(value, unused)` implementing this stage's computation:
    def advance(value, unused):
        # Run `rk4_step` to compute `next_value`.
        next_value = rk4_step(rate, value, dt)
        # Return `(next_value, next_value)` to the caller.
        return next_value, next_value
    # Run compiled structured control flow via `jax.lax` (`(_, tail)`).
    _, tail = jax.lax.scan(advance, initial, None, length=steps)
    # Return `jnp.concatenate((jnp.atleast_1d(initial), tail))` to the caller.
    return jnp.concatenate((jnp.atleast_1d(initial), tail))

# Batch initial conditions while sharing the model
# Step 2 — Batch initial conditions while sharing the model: The batch axis is independent initial conditions, not time.
# Construct `initials` via `jnp.array([0.5, 1.0, 2.0])`
initials = jnp.array([0.5, 1.0, 2.0])
# Compute `rate, dt, steps` from `0.7, 0.05, 40`
rate, dt, steps = 0.7, 0.05, 40
# Vectorize across the batch dimension with `jax.vmap` (`simulate_batch`).
simulate_batch = jax.jit(jax.vmap(lambda initial: solve(rate, initial, steps, dt)))
# Run `simulate_batch` to compute `trajectories`.
trajectories = simulate_batch(initials)
# Compute `times` from `np.arange(steps+1)*dt`
times = np.arange(steps+1)*dt
# Convert `oracle` to a host NumPy array for inspection or verification.
oracle = np.asarray(initials)[:,None]*np.exp(-rate*times[None,:])
# Check tensor shape invariant: `trajectories.shape == (3, 41)`
assert trajectories.shape == (3, 41)
# Compute `np.testing.assert_allclose(trajectories, oracle, rtol` as `3e-8, atol=1e-10)`.
np.testing.assert_allclose(trajectories, oracle, rtol=3e-8, atol=1e-10)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(trajectories)[2], 4*np.asarray(trajectories)[0], rtol=1e-12)
# Print the observed values to compare against the expected result.
print("Shape:", trajectories.shape, "endpoints:", np.asarray(trajectories)[:,-1])

# Change composition and verify axis meaning
# Step 3 — Change composition and verify axis meaning: Array arithmetic applies the independent scalar dynamics to all...
# Define `scan_batch(rate, values, steps, dt)` to carry state across steps with `jax.lax.scan`:
def scan_batch(rate, values, steps=40, dt=0.05):
    # Function `advance(state, unused)` implementing this stage's computation:
    def advance(state, unused):
        # Run `rk4_step` to compute `updated`.
        updated = rk4_step(rate, state, dt)
        # Return `(updated, updated)` to the caller.
        return updated, updated
    # Run compiled structured control flow via `jax.lax` (`(_, tail)`).
    _, tail = jax.lax.scan(advance, values, None, length=steps)
    # Return `jnp.concatenate((values[None, :], tail), axis=0)` to the caller.
    return jnp.concatenate((values[None,:], tail), axis=0)
# Run compiled structured control flow via `jax.lax` (`time_major`).
time_major = scan_batch(rate, initials)
# Check tensor shape invariant: `time_major.shape == (41, 3)`
assert time_major.shape == (41, 3)
# Compute `np.testing.assert_allclose(time_major.T, trajectories, rtol` as `1e-12, atol=1e-12)`.
np.testing.assert_allclose(time_major.T, trajectories, rtol=1e-12, atol=1e-12)
# Print the observed values to compare against the expected result.
print("scan(vectors) transposed equals vmap(scan)")

# Step 1 — Use a higher-order scalar transition: Runge–Kutta uses four slopes inside one step.
# Import jax for this computation.
import jax
# Update state in place with the new values.
jax.config.update("jax_enable_x64", True)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np

# Function `rk4_step(rate, value, dt)` implementing this stage's computation:
def rk4_step(rate, value, dt):
    # Compute `a` from `-rate*value`
    a = -rate*value
    # Compute `b` from `-rate*(value + dt*a/2)`
    b = -rate*(value + dt*a/2)
    # Compute `c` from `-rate*(value + dt*b/2)`
    c = -rate*(value + dt*b/2)
    # Compute `d` from `-rate*(value + dt*c)`
    d = -rate*(value + dt*c)
    # Return `value + dt * (a + 2 * b + 2 * c + d) / 6` to the caller.
    return value + dt*(a + 2*b + 2*c + d)/6

# Define `solve(rate, initial, steps, dt)` to carry state across steps with `jax.lax.scan`:
def solve(rate, initial, steps=40, dt=0.05):
    # Function `advance(value, unused)` implementing this stage's computation:
    def advance(value, unused):
        # Run `rk4_step` to compute `next_value`.
        next_value = rk4_step(rate, value, dt)
        # Return `(next_value, next_value)` to the caller.
        return next_value, next_value
    # Run compiled structured control flow via `jax.lax` (`(_, tail)`).
    _, tail = jax.lax.scan(advance, initial, None, length=steps)
    # Return `jnp.concatenate((jnp.atleast_1d(initial), tail))` to the caller.
    return jnp.concatenate((jnp.atleast_1d(initial), tail))

# Step 2 — Batch initial conditions while sharing the model: The batch axis is independent initial conditions, not time.
# Construct `initials` via `jnp.array([0.5, 1.0, 2.0])`
initials = jnp.array([0.5, 1.0, 2.0])
# Compute `rate, dt, steps` from `0.7, 0.05, 40`
rate, dt, steps = 0.7, 0.05, 40
# Vectorize across the batch dimension with `jax.vmap` (`simulate_batch`).
simulate_batch = jax.jit(jax.vmap(lambda initial: solve(rate, initial, steps, dt)))
# Run `simulate_batch` to compute `trajectories`.
trajectories = simulate_batch(initials)
# Compute `times` from `np.arange(steps+1)*dt`
times = np.arange(steps+1)*dt
# Convert `oracle` to a host NumPy array for inspection or verification.
oracle = np.asarray(initials)[:,None]*np.exp(-rate*times[None,:])
# Check tensor shape invariant: `trajectories.shape == (3, 41)`
assert trajectories.shape == (3, 41)
# Compute `np.testing.assert_allclose(trajectories, oracle, rtol` as `3e-8, atol=1e-10)`.
np.testing.assert_allclose(trajectories, oracle, rtol=3e-8, atol=1e-10)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(trajectories)[2], 4*np.asarray(trajectories)[0], rtol=1e-12)
# Print the observed values to compare against the expected result.
print("Shape:", trajectories.shape, "endpoints:", np.asarray(trajectories)[:,-1])

# Step 3 — Change composition and verify axis meaning: Array arithmetic applies the independent scalar dynamics to all...
# Define `scan_batch(rate, values, steps, dt)` to carry state across steps with `jax.lax.scan`:
def scan_batch(rate, values, steps=40, dt=0.05):
    # Function `advance(state, unused)` implementing this stage's computation:
    def advance(state, unused):
        # Run `rk4_step` to compute `updated`.
        updated = rk4_step(rate, state, dt)
        # Return `(updated, updated)` to the caller.
        return updated, updated
    # Run compiled structured control flow via `jax.lax` (`(_, tail)`).
    _, tail = jax.lax.scan(advance, values, None, length=steps)
    # Return `jnp.concatenate((values[None, :], tail), axis=0)` to the caller.
    return jnp.concatenate((values[None,:], tail), axis=0)
# Run compiled structured control flow via `jax.lax` (`time_major`).
time_major = scan_batch(rate, initials)
# Check tensor shape invariant: `time_major.shape == (41, 3)`
assert time_major.shape == (41, 3)
# Compute `np.testing.assert_allclose(time_major.T, trajectories, rtol` as `1e-12, atol=1e-12)`.
np.testing.assert_allclose(time_major.T, trajectories, rtol=1e-12, atol=1e-12)
# Print the observed values to compare against the expected result.
print("scan(vectors) transposed equals vmap(scan)")

# Figure data experiment
# Compute figure data for: Batching preserves independent physical trajectories
# Convert `visual_data` to a host NumPy array for inspection or verification.
visual_data={'kind':'line','x':times.tolist(),'xlabel':'time (seconds)','ylabel':'excess temperature','series':[{'label':f'initial {float(u):g}','y':np.asarray(trajectories)[i].tolist()} for i,u in enumerate(initials)]}

# Experiment: Compare RK4 with an independent polynomial
# Experiment — Compare RK4 with an independent polynomial: This checks the discrete algorithm independently of the...
z = -rate*dt
# Compute `multiplier` from `1 + z + z*z/2 + z**3/6 + z**4/24`
multiplier = 1 + z + z*z/2 + z**3/6 + z**4/24
# Compute `polynomial` from `np.asarray(initials)[:,None]*multiplier**np.arange(s...`
polynomial = np.asarray(initials)[:,None]*multiplier**np.arange(steps+1)[None,:]
# Compute `np.testing.assert_allclose(trajectories, polynomial, rtol` as `1e-12, atol=1e-12)`.
np.testing.assert_allclose(trajectories, polynomial, rtol=1e-12, atol=1e-12)
# Print the observed values to compare against the expected result.
print("RK4 polynomial maximum difference:", np.max(np.abs(np.asarray(trajectories)-polynomial)))

# Experiment: Let each trajectory have its own rate
# Experiment — Let each trajectory have its own rate: Mapping both axes pairs corresponding entries.
# Construct `rates` via `jnp.array([0.2, 0.7, 1.1])`
rates = jnp.array([0.2, 0.7, 1.1])
# Vectorize across the batch dimension with `jax.vmap` (`paired`).
paired = jax.vmap(lambda k, u: solve(k,u), in_axes=(0,0))(rates,initials)
# Convert `paired_reference` to a host NumPy array for inspection or verification.
paired_reference = np.asarray(initials)[:,None]*np.exp(-np.asarray(rates)[:,None]*times)
# Compute `np.testing.assert_allclose(paired,paired_reference,rtol` as `2e-7,atol=1e-10)`.
np.testing.assert_allclose(paired,paired_reference,rtol=2e-7,atol=1e-10)
# Print the observed values to compare against the expected result.
print("Paired-rate endpoints:", np.asarray(paired)[:,-1])

# Reference solution. Try the exercise before reading this.
# Exercise solution: Add a fourth initial condition, verify the changed output shape and...
# Construct `changed_initials` via `jnp.array([0.5, 1.0, 2.0, 3.0])`
changed_initials = jnp.array([0.5, 1.0, 2.0, 3.0])
# Vectorize across the batch dimension with `jax.vmap` (`changed`).
changed = jax.vmap(lambda u: solve(0.7,u))(changed_initials)
# Check tensor shape invariant: `changed.shape == (4,41)`
assert changed.shape == (4,41)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(changed,np.asarray(changed_initials)[:,None]*np.exp(-0.7*times),rtol=3e-8)
# Construct `order` via `jnp.array([3,0,2,1])`
order = jnp.array([3,0,2,1])
# Vectorize across the batch dimension with `jax.vmap` (`reordered`).
reordered = jax.vmap(lambda u: solve(0.7,u))(changed_initials[order])
# Compute `np.testing.assert_allclose(reordered,changed[order],rtol` as `1e-12)`.
np.testing.assert_allclose(reordered,changed[order],rtol=1e-12)
# Print the observed values to compare against the expected result.
print("Permutation preserves trajectory identity")

# Reference practice: Save only what the objective needs
# Save only what the objective needs (Practice): The explicit trajectory output is removed.
# Define `final_only(rate, initial, steps, dt)` to carry state across steps with `jax.lax.scan`:
def final_only(rate, initial, steps=40, dt=0.05):
    # Function `advance(value, unused)` implementing this stage's computation:
    def advance(value, unused):
        # Return `(rk4_step(rate, value, dt), None)` to the caller.
        return rk4_step(rate,value,dt), None
    # Return `jax.lax.scan(advance, initial, None, length=steps)[0]` to the caller.
    return jax.lax.scan(advance,initial,None,length=steps)[0]
# Vectorize across the batch dimension with `jax.vmap` (`finals`).
finals = jax.vmap(lambda u: final_only(0.7,u))(initials)
# Compute `np.testing.assert_allclose(finals,trajectories[:,-1],rtol` as `1e-12)`.
np.testing.assert_allclose(finals,trajectories[:,-1],rtol=1e-12)
# Print the observed values to compare against the expected result.
print("Only final states returned:", np.asarray(finals))

# Reference practice: Pairing versus Cartesian products
# Pairing versus Cartesian products (Challenge): Paired batches and Cartesian products are different experiments.
# Vectorize across the batch dimension with `jax.vmap` (`grid`).
grid = jax.vmap(lambda k: jax.vmap(lambda u: solve(k,u))(initials))(rates)
# Check tensor shape invariant: `grid.shape == (3,3,41)`
assert grid.shape == (3,3,41)
# Create evenly spaced index values in ``.
np.testing.assert_allclose(grid[jnp.arange(3),jnp.arange(3)],paired,rtol=1e-12)
# Print the observed values to compare against the expected result.
print("Grid axes: rate, initial condition, time", grid.shape)
print("PASS: science-02")
