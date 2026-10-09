"""Hamiltonian Monte Carlo and sampler diagnostics: worked experiments and reference solutions. CPU checks."""

# 1. Write a known target and its gradient
# Step 1 — 1. Write a known target and its gradient: Potential energy is the negative log-density up to a constant,...
# Import numpy for this computation.
import numpy as np
import jax
import jax.numpy as jnp
# Initialize array `target_mean` with explicit values and shape.
target_mean = jnp.array([1.,-1.])
# Initialize array `target_cov` with explicit values and shape.
target_cov = jnp.array([[1.,.8],[.8,1.]])
# Initialize array `precision` with explicit values and shape.
precision = jnp.linalg.solve(target_cov,jnp.eye(2))
# Function `potential(q)` implementing this stage's computation:
def potential(q):
    # Evaluate `delta` from the current inputs and state.
    delta=q-target_mean
    # Return `0.5 * delta @ precision @ delta` to the caller.
    return .5*delta@precision@delta
# Differentiate the objective to obtain `force` via automatic differentiation.
force=jax.grad(potential)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(force(jnp.array([0.,0.])),precision@(-target_mean))

# 2. Integrate then accept or reject
# Step 2 — 2. Integrate then accept or reject: Leapfrog approximates energy conservation.
def leapfrog(q,p,step_size,steps):
    # Evaluate `p` from the current inputs and state.
    p=p-.5*step_size*force(q)
    # Function `step(i, state)` implementing this stage's computation:
    def step(i,state):
        # Evaluate `(q, p)` from the current inputs and state.
        q,p=state
        # Evaluate `q` from the current inputs and state.
        q=q+step_size*p
        # Combine or mask array elements to form `p`.
        p=p-jnp.where(i<steps-1,step_size,.5*step_size)*force(q)
        # Return `(q, p)` to the caller.
        return q,p
    # Return `jax.lax.fori_loop(0, steps, step, (q, p))` to the caller.
    return jax.lax.fori_loop(0,steps,step,(q,p))

# Define `chain(key, initial, step_size, leapfrog_steps...)` to carry state across steps with `jax.lax.scan`:
def chain(key,initial,step_size=.25,leapfrog_steps=7,draws=1600):
    # Function `transition(state, _)` implementing this stage's computation:
    def transition(state,_):
        # Evaluate `(q, key)` from the current inputs and state.
        q,key=state
        # Create or split explicit PRNG key(s) (`(key, kp, ku)`) for reproducible randomness.
        key,kp,ku=jax.random.split(key,3)
        # Sample deterministic random values into `p` using an explicit PRNG key.
        p=jax.random.normal(kp,q.shape)
        # Run `leapfrog` to compute `(proposal, new_p)`.
        proposal,new_p=leapfrog(q,p,step_size,leapfrog_steps)
        # Reduce across the target axis to summarize `error`.
        error=potential(proposal)+.5*jnp.sum(new_p**2)-potential(q)-.5*jnp.sum(p**2)
        # Draw pseudorandom samples for `accept` using the explicit RNG state.
        accept=jnp.isfinite(error)&(jnp.log(jax.random.uniform(ku)) < -error)
        # Combine or mask array elements to form `next_q`.
        next_q=jnp.where(accept,proposal,q)
        # Return `((next_q, key), (next_q, accept, error))` to the caller.
        return (next_q,key),(next_q,accept,error)
    # Run compiled structured control flow via `jax.lax` (`(_, record)`).
    _,record=jax.lax.scan(transition,(initial,key),None,length=draws)
    # Return `record` to the caller.
    return record

# 3. Compare chains and an analytic oracle
# Step 3 — 3. Compare chains and an analytic oracle: The tolerances test this seeded Gaussian fixture only.
# Initialize array `starts` with explicit values and shape.
starts=jnp.array([[-4.,-4.],[-4.,4.],[4.,-4.],[4.,4.]])
# Create or split explicit PRNG key(s) (`records`) for reproducible randomness.
records=jax.vmap(lambda key,start:chain(key,start))(jax.random.split(jax.random.key(71),4),starts)
# Evaluate `samples` from the current inputs and state.
samples=records[0][:,400:,:]
# Function `split_rhat(values)` implementing this stage's computation:
def split_rhat(values):
    # Classical split R-hat for one scalar coordinate; not rank-normalized.
    half=values.shape[1]//2
    # Combine or mask array elements to form `split`.
    split=jnp.concatenate([values[:,:half],values[:,-half:]],axis=0)
    # Reduce along axis=1 to compute `within`.
    within=jnp.var(split,axis=1,ddof=1).mean()
    # Reduce along axis=1 to compute `between`.
    between=half*jnp.var(split.mean(axis=1),ddof=1)
    # Return `jnp.sqrt(((half - 1) * within / half + between / half) / within)` to the caller.
    return jnp.sqrt(((half-1)*within/half+between/half)/within)
# Vectorize across the batch dimension with `jax.vmap` (`rhats`).
rhats=jax.vmap(split_rhat,in_axes=2)(samples)
# Construct and reshape `pooled` into the target tensor dimensions.
pooled=samples.reshape(-1,2)
# Verify contract: `jnp.max(jnp.abs(pooled.mean(0) - target_mean)) < 0.15`.
assert jnp.max(jnp.abs(pooled.mean(0)-target_mean))<.15
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.max(jnp.abs(jnp.cov(pooled.T)-target_cov))<.2
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.all(rhats<1.05)
# Print the observed values to compare against the expected result.
print('mean:',pooled.mean(0),'covariance:',jnp.cov(pooled.T))
# Print diagnostic summary of the computed outputs.
print('classical split R-hat:',rhats,'acceptance:',records[1].mean(axis=1))

# Step 1 — 1. Write a known target and its gradient: Potential energy is the negative log-density up to a constant,...
# Import numpy for this computation.
import numpy as np
import jax
import jax.numpy as jnp
# Initialize array `target_mean` with explicit values and shape.
target_mean = jnp.array([1.,-1.])
# Initialize array `target_cov` with explicit values and shape.
target_cov = jnp.array([[1.,.8],[.8,1.]])
# Initialize array `precision` with explicit values and shape.
precision = jnp.linalg.solve(target_cov,jnp.eye(2))
# Function `potential(q)` implementing this stage's computation:
def potential(q):
    # Evaluate `delta` from the current inputs and state.
    delta=q-target_mean
    # Return `0.5 * delta @ precision @ delta` to the caller.
    return .5*delta@precision@delta
# Differentiate the objective to obtain `force` via automatic differentiation.
force=jax.grad(potential)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(force(jnp.array([0.,0.])),precision@(-target_mean))

# Step 2 — 2. Integrate then accept or reject: Leapfrog approximates energy conservation.
def leapfrog(q,p,step_size,steps):
    # Evaluate `p` from the current inputs and state.
    p=p-.5*step_size*force(q)
    # Function `step(i, state)` implementing this stage's computation:
    def step(i,state):
        # Evaluate `(q, p)` from the current inputs and state.
        q,p=state
        # Evaluate `q` from the current inputs and state.
        q=q+step_size*p
        # Combine or mask array elements to form `p`.
        p=p-jnp.where(i<steps-1,step_size,.5*step_size)*force(q)
        # Return `(q, p)` to the caller.
        return q,p
    # Return `jax.lax.fori_loop(0, steps, step, (q, p))` to the caller.
    return jax.lax.fori_loop(0,steps,step,(q,p))

# Define `chain(key, initial, step_size, leapfrog_steps...)` to carry state across steps with `jax.lax.scan`:
def chain(key,initial,step_size=.25,leapfrog_steps=7,draws=1600):
    # Function `transition(state, _)` implementing this stage's computation:
    def transition(state,_):
        # Evaluate `(q, key)` from the current inputs and state.
        q,key=state
        # Create or split explicit PRNG key(s) (`(key, kp, ku)`) for reproducible randomness.
        key,kp,ku=jax.random.split(key,3)
        # Sample deterministic random values into `p` using an explicit PRNG key.
        p=jax.random.normal(kp,q.shape)
        # Run `leapfrog` to compute `(proposal, new_p)`.
        proposal,new_p=leapfrog(q,p,step_size,leapfrog_steps)
        # Reduce across the target axis to summarize `error`.
        error=potential(proposal)+.5*jnp.sum(new_p**2)-potential(q)-.5*jnp.sum(p**2)
        # Draw pseudorandom samples for `accept` using the explicit RNG state.
        accept=jnp.isfinite(error)&(jnp.log(jax.random.uniform(ku)) < -error)
        # Combine or mask array elements to form `next_q`.
        next_q=jnp.where(accept,proposal,q)
        # Return `((next_q, key), (next_q, accept, error))` to the caller.
        return (next_q,key),(next_q,accept,error)
    # Run compiled structured control flow via `jax.lax` (`(_, record)`).
    _,record=jax.lax.scan(transition,(initial,key),None,length=draws)
    # Return `record` to the caller.
    return record

# Step 3 — 3. Compare chains and an analytic oracle: The tolerances test this seeded Gaussian fixture only.
# Initialize array `starts` with explicit values and shape.
starts=jnp.array([[-4.,-4.],[-4.,4.],[4.,-4.],[4.,4.]])
# Create or split explicit PRNG key(s) (`records`) for reproducible randomness.
records=jax.vmap(lambda key,start:chain(key,start))(jax.random.split(jax.random.key(71),4),starts)
# Evaluate `samples` from the current inputs and state.
samples=records[0][:,400:,:]
# Function `split_rhat(values)` implementing this stage's computation:
def split_rhat(values):
    # Classical split R-hat for one scalar coordinate; not rank-normalized.
    half=values.shape[1]//2
    # Combine or mask array elements to form `split`.
    split=jnp.concatenate([values[:,:half],values[:,-half:]],axis=0)
    # Reduce along axis=1 to compute `within`.
    within=jnp.var(split,axis=1,ddof=1).mean()
    # Reduce along axis=1 to compute `between`.
    between=half*jnp.var(split.mean(axis=1),ddof=1)
    # Return `jnp.sqrt(((half - 1) * within / half + between / half) / within)` to the caller.
    return jnp.sqrt(((half-1)*within/half+between/half)/within)
# Vectorize across the batch dimension with `jax.vmap` (`rhats`).
rhats=jax.vmap(split_rhat,in_axes=2)(samples)
# Construct and reshape `pooled` into the target tensor dimensions.
pooled=samples.reshape(-1,2)
# Verify contract: `jnp.max(jnp.abs(pooled.mean(0) - target_mean)) < 0.15`.
assert jnp.max(jnp.abs(pooled.mean(0)-target_mean))<.15
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.max(jnp.abs(jnp.cov(pooled.T)-target_cov))<.2
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.all(rhats<1.05)
# Print the observed values to compare against the expected result.
print('mean:',pooled.mean(0),'covariance:',jnp.cov(pooled.T))
# Print diagnostic summary of the computed outputs.
print('classical split R-hat:',rhats,'acceptance:',records[1].mean(axis=1))

# Figure data experiment
# Compute figure data for: Four HMC traces around a known posterior mean
# Create evenly spaced index values in `indices`.
indices=np.arange(0,400,10)
# Convert `visual_data` to a host NumPy array for inspection or verification.
visual_data={"kind":"line","x":indices.tolist(),"xlabel":"retained transition (after fixed discard)","ylabel":"first parameter coordinate","series":[{"label":"chain "+str(i+1),"y":np.asarray(samples[i,indices,0]).tolist()} for i in range(4)]}

# Experiment: Reverse one trajectory
# Experiment — Reverse one trajectory: Reversibility is an integrator invariant, independent of whether...
# Initialize array `q0` with explicit values and shape.
q0=jnp.array([.2,-.4]);p0=jnp.array([.3,.7])
# Run `leapfrog` to compute `(q1, p1)`.
q1,p1=leapfrog(q0,p0,.15,9)
# Run `leapfrog` to compute `(q2, p2)`.
q2,p2=leapfrog(q1,-p1,.15,9)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(q2,q0,atol=2e-6)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(p2,-p0,atol=2e-6)

# Experiment: Separate replay from a changed chain
# Experiment — Separate replay from a changed chain: Replay tests state ownership.
# Create or split explicit PRNG key(s) (`replay`) for reproducible randomness.
replay=chain(jax.random.key(9),jnp.zeros(2),draws=50)[0]
# Create or split explicit PRNG key(s) (`again`) for reproducible randomness.
again=chain(jax.random.key(9),jnp.zeros(2),draws=50)[0]
# Create or split explicit PRNG key(s) (`changed`) for reproducible randomness.
changed=chain(jax.random.key(10),jnp.zeros(2),draws=50)[0]
# Verify contract: `jnp.array_equal(replay, again)`.
assert jnp.array_equal(replay,again)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert not jnp.array_equal(replay,changed)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Increase the step size to 1.2 with the same leapfrog count.
# Create or split explicit PRNG key(s) (`bad`) for reproducible randomness.
bad=chain(jax.random.key(71),starts[0],step_size=1.2,draws=100)
# Verify contract: `bad[1].mean() < 0.2`.
assert bad[1].mean()<.2
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.max(bad[2])>100.
# Print the observed values to compare against the expected result.
print("bad acceptance:",float(bad[1].mean()),"maximum energy error:",float(jnp.max(bad[2])))

# Reference practice: Catch chains trapped at different levels
# Catch chains trapped at different levels (Transfer / diagnosis): Between-chain variation dominates within-chain variation, so...
# Create or split explicit PRNG key(s) (`fake`) for reproducible randomness.
fake=jax.random.normal(jax.random.key(4),(4,600))*.1+jnp.arange(4)[:,None]*4
# Verify contract: `split_rhat(fake) > 5`.
assert split_rhat(fake)>5

# Reference practice: Verify a directional derivative
# Verify a directional derivative (Transfer / diagnosis): A sign error in potential or a transposed geometry can be...
# Initialize array `q` with explicit values and shape.
q=jnp.array([.4,.8]);v=jnp.array([.7,-.2]);eps=.001
# Perform matrix / vector contraction (`@`) to compute `analytic`.
analytic=(precision@(q-target_mean))@v
# Perform matrix / vector contraction (`@`) to compute `auto`.
auto=force(q)@v
# Evaluate `finite` from the current inputs and state.
finite=(potential(q+eps*v)-potential(q-eps*v))/(2*eps)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(auto,analytic,rtol=1e-6)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(finite,analytic,rtol=.005,atol=.002)
print("PASS: probability-03")
