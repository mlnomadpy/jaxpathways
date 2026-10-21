"""Hamiltonian Monte Carlo and sampler diagnostics: worked experiments and reference solutions. CPU checks."""

# 1. Write a known target and its gradient
# Step 1 — 1. Write a known target and its gradient
import numpy as np
import jax
import jax.numpy as jnp
target_mean = jnp.array([1.,-1.])
target_cov = jnp.array([[1.,.8],[.8,1.]])
precision = jnp.linalg.solve(target_cov,jnp.eye(2))
def potential(q):
    delta=q-target_mean
    return .5*delta@precision@delta
force=jax.grad(potential)
assert jnp.allclose(force(jnp.array([0.,0.])),precision@(-target_mean))

# 2. Integrate then accept or reject
# Step 2 — 2. Integrate then accept or reject: Leapfrog approximates energy conservation.
def leapfrog(q,p,step_size,steps):
    p=p-.5*step_size*force(q)
    def step(i,state):
        q,p=state
        q=q+step_size*p
        p=p-jnp.where(i<steps-1,step_size,.5*step_size)*force(q)
        return q,p
    return jax.lax.fori_loop(0,steps,step,(q,p))

# Define `chain(key, initial, step_size, leapfrog_steps...)` to carry state across steps with `jax.lax.scan`:
def chain(key,initial,step_size=.25,leapfrog_steps=7,draws=1600):
    def transition(state,_):
        q,key=state
        # Create or split explicit PRNG key(s) (`(key, kp, ku)`) for reproducible randomness.
        key,kp,ku=jax.random.split(key,3)
        # Sample deterministic random values into `p` using an explicit PRNG key.
        p=jax.random.normal(kp,q.shape)
        proposal,new_p=leapfrog(q,p,step_size,leapfrog_steps)
        error=potential(proposal)+.5*jnp.sum(new_p**2)-potential(q)-.5*jnp.sum(p**2)
        accept=jnp.isfinite(error)&(jnp.log(jax.random.uniform(ku)) < -error)
        next_q=jnp.where(accept,proposal,q)
        return (next_q,key),(next_q,accept,error)
    _,record=jax.lax.scan(transition,(initial,key),None,length=draws)
    return record

# 3. Compare chains and an analytic oracle
# Step 3 — 3. Compare chains and an analytic oracle: The tolerances test this seeded Gaussian fixture only.
starts=jnp.array([[-4.,-4.],[-4.,4.],[4.,-4.],[4.,4.]])
# Create or split explicit PRNG key(s) (`records`) for reproducible randomness.
records=jax.vmap(lambda key,start:chain(key,start))(jax.random.split(jax.random.key(71),4),starts)
samples=records[0][:,400:,:]
def split_rhat(values):
    # Classical split R-hat for one scalar coordinate; not rank-normalized.
    half=values.shape[1]//2
    split=jnp.concatenate([values[:,:half],values[:,-half:]],axis=0)
    # Reduce along axis=1 to compute `within`.
    within=jnp.var(split,axis=1,ddof=1).mean()
    # Reduce along axis=1 to compute `between`.
    between=half*jnp.var(split.mean(axis=1),ddof=1)
    return jnp.sqrt(((half-1)*within/half+between/half)/within)
# Vectorize across the batch dimension with `jax.vmap` (`rhats`).
rhats=jax.vmap(split_rhat,in_axes=2)(samples)
pooled=samples.reshape(-1,2)
assert jnp.max(jnp.abs(pooled.mean(0)-target_mean))<.15
assert jnp.max(jnp.abs(jnp.cov(pooled.T)-target_cov))<.2
assert jnp.all(rhats<1.05)
print('mean:',pooled.mean(0),'covariance:',jnp.cov(pooled.T))
print('classical split R-hat:',rhats,'acceptance:',records[1].mean(axis=1))

# Step 1 — 1. Write a known target and its gradient
import numpy as np
import jax
import jax.numpy as jnp
target_mean = jnp.array([1.,-1.])
target_cov = jnp.array([[1.,.8],[.8,1.]])
precision = jnp.linalg.solve(target_cov,jnp.eye(2))
def potential(q):
    delta=q-target_mean
    return .5*delta@precision@delta
force=jax.grad(potential)
assert jnp.allclose(force(jnp.array([0.,0.])),precision@(-target_mean))

# Step 2 — 2. Integrate then accept or reject: Leapfrog approximates energy conservation.
def leapfrog(q,p,step_size,steps):
    p=p-.5*step_size*force(q)
    def step(i,state):
        q,p=state
        q=q+step_size*p
        p=p-jnp.where(i<steps-1,step_size,.5*step_size)*force(q)
        return q,p
    return jax.lax.fori_loop(0,steps,step,(q,p))

# Define `chain(key, initial, step_size, leapfrog_steps...)` to carry state across steps with `jax.lax.scan`:
def chain(key,initial,step_size=.25,leapfrog_steps=7,draws=1600):
    def transition(state,_):
        q,key=state
        # Create or split explicit PRNG key(s) (`(key, kp, ku)`) for reproducible randomness.
        key,kp,ku=jax.random.split(key,3)
        # Sample deterministic random values into `p` using an explicit PRNG key.
        p=jax.random.normal(kp,q.shape)
        proposal,new_p=leapfrog(q,p,step_size,leapfrog_steps)
        error=potential(proposal)+.5*jnp.sum(new_p**2)-potential(q)-.5*jnp.sum(p**2)
        accept=jnp.isfinite(error)&(jnp.log(jax.random.uniform(ku)) < -error)
        next_q=jnp.where(accept,proposal,q)
        return (next_q,key),(next_q,accept,error)
    _,record=jax.lax.scan(transition,(initial,key),None,length=draws)
    return record

# Step 3 — 3. Compare chains and an analytic oracle: The tolerances test this seeded Gaussian fixture only.
starts=jnp.array([[-4.,-4.],[-4.,4.],[4.,-4.],[4.,4.]])
# Create or split explicit PRNG key(s) (`records`) for reproducible randomness.
records=jax.vmap(lambda key,start:chain(key,start))(jax.random.split(jax.random.key(71),4),starts)
samples=records[0][:,400:,:]
def split_rhat(values):
    # Classical split R-hat for one scalar coordinate; not rank-normalized.
    half=values.shape[1]//2
    split=jnp.concatenate([values[:,:half],values[:,-half:]],axis=0)
    # Average the sample variance within each split chain.
    within=jnp.var(split,axis=1,ddof=1).mean()
    # Measure how far split-chain means vary from one another, scaled by half the draws.
    between=half*jnp.var(split.mean(axis=1),ddof=1)
    return jnp.sqrt(((half-1)*within/half+between/half)/within)
# Vectorize across the batch dimension with `jax.vmap` (`rhats`).
rhats=jax.vmap(split_rhat,in_axes=2)(samples)
pooled=samples.reshape(-1,2)
assert jnp.max(jnp.abs(pooled.mean(0)-target_mean))<.15
assert jnp.max(jnp.abs(jnp.cov(pooled.T)-target_cov))<.2
assert jnp.all(rhats<1.05)
print('mean:',pooled.mean(0),'covariance:',jnp.cov(pooled.T))
print('classical split R-hat:',rhats,'acceptance:',records[1].mean(axis=1))

# Figure data experiment
indices=np.arange(0,400,10)
# Transfer the device result to the host for an independent NumPy comparison.
visual_data={"kind":"line","x":indices.tolist(),"xlabel":"retained transition (after fixed discard)","ylabel":"first parameter coordinate","series":[{"label":"chain "+str(i+1),"y":np.asarray(samples[i,indices,0]).tolist()} for i in range(4)]}

# Experiment: Reverse one trajectory
# Experiment — Reverse one trajectory
q0=jnp.array([.2,-.4])
p0=jnp.array([.3,.7])
q1,p1=leapfrog(q0,p0,.15,9)
q2,p2=leapfrog(q1,-p1,.15,9)
np.testing.assert_allclose(q2,q0,atol=2e-6)
np.testing.assert_allclose(p2,-p0,atol=2e-6)

# Experiment: Separate replay from a changed chain
# Experiment — Separate replay from a changed chain: Replay tests state ownership.
# Create or split explicit PRNG key(s) (`replay`) for reproducible randomness.
replay=chain(jax.random.key(9),jnp.zeros(2),draws=50)[0]
# Create or split explicit PRNG key(s) (`again`) for reproducible randomness.
again=chain(jax.random.key(9),jnp.zeros(2),draws=50)[0]
# Create or split explicit PRNG key(s) (`changed`) for reproducible randomness.
changed=chain(jax.random.key(10),jnp.zeros(2),draws=50)[0]
assert jnp.array_equal(replay,again)
assert not jnp.array_equal(replay,changed)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Increase the step size to 1.2 with the same leapfrog count.
# Create or split explicit PRNG key(s) (`bad`) for reproducible randomness.
bad=chain(jax.random.key(71),starts[0],step_size=1.2,draws=100)
assert bad[1].mean()<.2
assert jnp.max(bad[2])>100.
print("bad acceptance:",float(bad[1].mean()),"maximum energy error:",float(jnp.max(bad[2])))

# Reference practice: Catch chains trapped at different levels
# Catch chains trapped at different levels (Transfer / diagnosis)
# Create or split explicit PRNG key(s) (`fake`) for reproducible randomness.
fake=jax.random.normal(jax.random.key(4),(4,600))*.1+jnp.arange(4)[:,None]*4
assert split_rhat(fake)>5

# Reference practice: Verify a directional derivative
# Verify a directional derivative (Transfer / diagnosis)
q=jnp.array([.4,.8])
v=jnp.array([.7,-.2])
eps=.001
analytic=(precision@(q-target_mean))@v
auto=force(q)@v
finite=(potential(q+eps*v)-potential(q-eps*v))/(2*eps)
np.testing.assert_allclose(auto,analytic,rtol=1e-6)
np.testing.assert_allclose(finite,analytic,rtol=.005,atol=.002)
print("PASS: probability-03")
