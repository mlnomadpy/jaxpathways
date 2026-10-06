"""Vectorize trajectories and scan time: worked experiments and reference solutions. CPU checks."""

# Use a higher-order scalar transition
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

def rk4_step(rate, value, dt):
    a = -rate*value
    b = -rate*(value + dt*a/2)
    c = -rate*(value + dt*b/2)
    d = -rate*(value + dt*c)
    return value + dt*(a + 2*b + 2*c + d)/6

def solve(rate, initial, steps=40, dt=0.05):
    def advance(value, unused):
        next_value = rk4_step(rate, value, dt)
        return next_value, next_value
    _, tail = jax.lax.scan(advance, initial, None, length=steps)
    return jnp.concatenate((jnp.atleast_1d(initial), tail))

# Batch initial conditions while sharing the model
initials = jnp.array([0.5, 1.0, 2.0])
rate, dt, steps = 0.7, 0.05, 40
simulate_batch = jax.jit(jax.vmap(lambda initial: solve(rate, initial, steps, dt)))
trajectories = simulate_batch(initials)
times = np.arange(steps+1)*dt
oracle = np.asarray(initials)[:,None]*np.exp(-rate*times[None,:])
assert trajectories.shape == (3, 41)
np.testing.assert_allclose(trajectories, oracle, rtol=3e-8, atol=1e-10)
np.testing.assert_allclose(np.asarray(trajectories)[2], 4*np.asarray(trajectories)[0], rtol=1e-12)
print("Shape:", trajectories.shape, "endpoints:", np.asarray(trajectories)[:,-1])

# Change composition and verify axis meaning
def scan_batch(rate, values, steps=40, dt=0.05):
    def advance(state, unused):
        updated = rk4_step(rate, state, dt)
        return updated, updated
    _, tail = jax.lax.scan(advance, values, None, length=steps)
    return jnp.concatenate((values[None,:], tail), axis=0)
time_major = scan_batch(rate, initials)
assert time_major.shape == (41, 3)
np.testing.assert_allclose(time_major.T, trajectories, rtol=1e-12, atol=1e-12)
print("scan(vectors) transposed equals vmap(scan)")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

def rk4_step(rate, value, dt):
    a = -rate*value
    b = -rate*(value + dt*a/2)
    c = -rate*(value + dt*b/2)
    d = -rate*(value + dt*c)
    return value + dt*(a + 2*b + 2*c + d)/6

def solve(rate, initial, steps=40, dt=0.05):
    def advance(value, unused):
        next_value = rk4_step(rate, value, dt)
        return next_value, next_value
    _, tail = jax.lax.scan(advance, initial, None, length=steps)
    return jnp.concatenate((jnp.atleast_1d(initial), tail))

initials = jnp.array([0.5, 1.0, 2.0])
rate, dt, steps = 0.7, 0.05, 40
simulate_batch = jax.jit(jax.vmap(lambda initial: solve(rate, initial, steps, dt)))
trajectories = simulate_batch(initials)
times = np.arange(steps+1)*dt
oracle = np.asarray(initials)[:,None]*np.exp(-rate*times[None,:])
assert trajectories.shape == (3, 41)
np.testing.assert_allclose(trajectories, oracle, rtol=3e-8, atol=1e-10)
np.testing.assert_allclose(np.asarray(trajectories)[2], 4*np.asarray(trajectories)[0], rtol=1e-12)
print("Shape:", trajectories.shape, "endpoints:", np.asarray(trajectories)[:,-1])

def scan_batch(rate, values, steps=40, dt=0.05):
    def advance(state, unused):
        updated = rk4_step(rate, state, dt)
        return updated, updated
    _, tail = jax.lax.scan(advance, values, None, length=steps)
    return jnp.concatenate((values[None,:], tail), axis=0)
time_major = scan_batch(rate, initials)
assert time_major.shape == (41, 3)
np.testing.assert_allclose(time_major.T, trajectories, rtol=1e-12, atol=1e-12)
print("scan(vectors) transposed equals vmap(scan)")

# Figure data experiment
visual_data={'kind':'line','x':times.tolist(),'xlabel':'time (seconds)','ylabel':'excess temperature','series':[{'label':f'initial {float(u):g}','y':np.asarray(trajectories)[i].tolist()} for i,u in enumerate(initials)]}

# Experiment: Compare RK4 with an independent polynomial
z = -rate*dt
multiplier = 1 + z + z*z/2 + z**3/6 + z**4/24
polynomial = np.asarray(initials)[:,None]*multiplier**np.arange(steps+1)[None,:]
np.testing.assert_allclose(trajectories, polynomial, rtol=1e-12, atol=1e-12)
print("RK4 polynomial maximum difference:", np.max(np.abs(np.asarray(trajectories)-polynomial)))

# Experiment: Let each trajectory have its own rate
rates = jnp.array([0.2, 0.7, 1.1])
paired = jax.vmap(lambda k, u: solve(k,u), in_axes=(0,0))(rates,initials)
paired_reference = np.asarray(initials)[:,None]*np.exp(-np.asarray(rates)[:,None]*times)
np.testing.assert_allclose(paired,paired_reference,rtol=2e-7,atol=1e-10)
print("Paired-rate endpoints:", np.asarray(paired)[:,-1])

# Reference solution. Try the exercise before reading this.
changed_initials = jnp.array([0.5, 1.0, 2.0, 3.0])
changed = jax.vmap(lambda u: solve(0.7,u))(changed_initials)
assert changed.shape == (4,41)
np.testing.assert_allclose(changed,np.asarray(changed_initials)[:,None]*np.exp(-0.7*times),rtol=3e-8)
order = jnp.array([3,0,2,1])
reordered = jax.vmap(lambda u: solve(0.7,u))(changed_initials[order])
np.testing.assert_allclose(reordered,changed[order],rtol=1e-12)
print("Permutation preserves trajectory identity")

# Reference practice: Save only what the objective needs
def final_only(rate, initial, steps=40, dt=0.05):
    def advance(value, unused):
        return rk4_step(rate,value,dt), None
    return jax.lax.scan(advance,initial,None,length=steps)[0]
finals = jax.vmap(lambda u: final_only(0.7,u))(initials)
np.testing.assert_allclose(finals,trajectories[:,-1],rtol=1e-12)
print("Only final states returned:", np.asarray(finals))

# Reference practice: Pairing versus Cartesian products
grid = jax.vmap(lambda k: jax.vmap(lambda u: solve(k,u))(initials))(rates)
assert grid.shape == (3,3,41)
np.testing.assert_allclose(grid[jnp.arange(3),jnp.arange(3)],paired,rtol=1e-12)
print("Grid axes: rate, initial condition, time", grid.shape)
print("PASS: science-02")
