"""Random keys without surprises: worked experiments and reference solutions. CPU checks."""

# Prepare the inputs
# Step 1 — Prepare the inputs: These explicit inputs define the case that the later checks will...
# Import jax for this computation.
import jax
import jax.numpy as jnp

# Build the computation
# Step 2 — Build the computation: random.split returns two new keys.
# Create or split explicit PRNG key(s) (`key`) for reproducible randomness.
key = jax.random.key(42)
# Create or split explicit PRNG key(s) (`(key, sample_key)`) for reproducible randomness.
key, sample_key = jax.random.split(key)
# Sample deterministic random values into `a` using an explicit PRNG key.
a = jax.random.normal(sample_key, (3,))
# Sample deterministic random values into `repeated` using an explicit PRNG key.
repeated = jax.random.normal(sample_key, (3,))
# Create or split explicit PRNG key(s) (`(key, next_key)`) for reproducible randomness.
key, next_key = jax.random.split(key)
# Sample deterministic random values into `b` using an explicit PRNG key.
b = jax.random.normal(next_key, (3,))

# Run and check the result
# Step 3 — Run and check the result: Compare the output to the expected result below before making the...
# Print the observed values to compare against the expected result.
print("Replay equal:", bool(jnp.array_equal(a, repeated)))
# Print diagnostic summary of the computed outputs.
print("Fresh draw equal:", bool(jnp.array_equal(a, b)))
# Assert invariant `jnp.array_equal(a, repeated)` holds
assert jnp.array_equal(a, repeated)
# Assert invariant `not jnp.array_equal(a, b)` holds
assert not jnp.array_equal(a, b)

# Step 1 — Prepare the inputs: These explicit inputs define the case that the later checks will...
# Import jax for this computation.
import jax
import jax.numpy as jnp
# Step 2 — Build the computation: random.split returns two new keys.
# Create or split explicit PRNG key(s) (`key`) for reproducible randomness.
key = jax.random.key(42)
# Create or split explicit PRNG key(s) (`(key, sample_key)`) for reproducible randomness.
key, sample_key = jax.random.split(key)
# Sample deterministic random values into `a` using an explicit PRNG key.
a = jax.random.normal(sample_key, (3,))
# Sample deterministic random values into `repeated` using an explicit PRNG key.
repeated = jax.random.normal(sample_key, (3,))
# Create or split explicit PRNG key(s) (`(key, next_key)`) for reproducible randomness.
key, next_key = jax.random.split(key)
# Sample deterministic random values into `b` using an explicit PRNG key.
b = jax.random.normal(next_key, (3,))
# Step 3 — Run and check the result: Compare the output to the expected result below before making the...
# Print the observed values to compare against the expected result.
print("Replay equal:", bool(jnp.array_equal(a, repeated)))
# Print diagnostic summary of the computed outputs.
print("Fresh draw equal:", bool(jnp.array_equal(a, b)))
# Assert invariant `jnp.array_equal(a, repeated)` holds
assert jnp.array_equal(a, repeated)
# Assert invariant `not jnp.array_equal(a, b)` holds
assert not jnp.array_equal(a, b)

# Figure data experiment
# Compute figure data for: Reused keys replay the same sample
# Compute `visual_data` from `{'kind': 'line', 'x': [0, 1, 2], 'xlabel': 'sample c...`
visual_data = {'kind': 'line', 'x': [0, 1, 2], 'xlabel': 'sample coordinate', 'ylabel': 'normal draw', 'series': [{'label': 'first draw', 'y': a.tolist()}, {'label': 'same key replay', 'y': repeated.tolist()}, {'label': 'split next draw', 'y': b.tolist()}]}

# Experiment: Implement one random state transition
# Experiment — Implement one random state transition: Replay compares complete state transitions.
def noise_step(key):
    # Create or split explicit PRNG key(s) (`(next_key, draw_key)`) for reproducible randomness.
    next_key,draw_key=jax.random.split(key)
    # Return `(next_key, jax.random.normal(draw_key, (2,)))` to the caller.
    return next_key,jax.random.normal(draw_key,(2,))
# Create or split explicit PRNG key(s) (`root`) for reproducible randomness.
root=jax.random.key(9)
# Run `noise_step` to compute `(k1, n1)`.
k1,n1=noise_step(root)
# Run `noise_step` to compute `(k2, n2)`.
k2,n2=noise_step(k1)
# Create or split explicit PRNG key(s) (`(r1, replay1)`) for reproducible randomness.
r1,replay1=noise_step(jax.random.key(9))
# Run `noise_step` to compute `(r2, replay2)`.
r2,replay2=noise_step(r1)
# Assert invariant `jnp.array_equal(n1,replay1) and jnp.array_equal(n2,replay2)` holds
assert jnp.array_equal(n1,replay1) and jnp.array_equal(n2,replay2)
# Assert invariant `jnp.array_equal(jax.random.key_data(k2),jax.random.key_data(r2))` holds
assert jnp.array_equal(jax.random.key_data(k2),jax.random.key_data(r2))

# Experiment: Keep identity under reordering
# Experiment — Keep identity under reordering: fold_in makes the assignment explicit.
# Create or split explicit PRNG key(s) (`base`) for reproducible randomness.
base=jax.random.key(11)
# Function `by_id(ids)` implementing this stage's computation:
def by_id(ids):
    # Return `jax.vmap(lambda i: jax.random.normal(jax.random.fold_in(base, i), ()))(ids)` to the caller.
    return jax.vmap(lambda i:jax.random.normal(jax.random.fold_in(base,i),()))(ids)
# Construct `forward` via `by_id(jnp.array([3,8]))`
forward=by_id(jnp.array([3,8]))
# Construct `reverse` via `by_id(jnp.array([8,3]))`
reverse=by_id(jnp.array([8,3]))
# Assert invariant `jnp.array_equal(forward,reverse[::-1])` holds
assert jnp.array_equal(forward,reverse[::-1])

# Reference solution. Try the exercise before reading this.
# Exercise solution: Split a fresh seed into four keys, then use vmap to draw two normal...
def draw(seed):
    # Create or split explicit PRNG key(s) (`keys`) for reproducible randomness.
    keys = jax.random.split(jax.random.key(seed), 4)
    # Return `jax.vmap(lambda k: jax.random.normal(k, (2,)))(keys)` to the caller.
    return jax.vmap(lambda k: jax.random.normal(k, (2,)))(keys)
# Check tensor shape invariant: `draw(7).shape == (4, 2)`
assert draw(7).shape == (4, 2)
# Assert invariant `jnp.array_equal(draw(7), draw(7))` holds
assert jnp.array_equal(draw(7), draw(7))

# Reference practice: Replay a noisy prediction
# Replay a noisy prediction (Practice): Holding noise fixed isolates the changed numerical input.
def noisy_prediction(seed,x):
    # Create or split explicit PRNG key(s) (`eps`) for reproducible randomness.
    eps=jax.random.normal(jax.random.key(seed),x.shape)
    # Return `2 * x + 1 + eps` to the caller.
    return 2*x+1+eps
# Construct `inputs` via `jnp.array([0.,1.,2.])`
inputs=jnp.array([0.,1.,2.])
# Assert that `jnp.array_equal(noisy_prediction(4,inputs),noisy_prediction(4,inputs))`.
assert jnp.array_equal(noisy_prediction(4,inputs),noisy_prediction(4,inputs))
# Assert that `jnp.allclose(noisy_prediction(4,inputs+1)-noisy_prediction(4,inputs),2.,atol=1e-6)`.
assert jnp.allclose(noisy_prediction(4,inputs+1)-noisy_prediction(4,inputs),2.,atol=1e-6)

# Reference practice: Diagnose key reuse in a loop
# Diagnose key reuse in a loop (Challenge): The repeated-row symptom comes from key allocation.
# Create or split explicit PRNG key(s) (`root`) for reproducible randomness.
root=jax.random.key(5)
# Sample deterministic random values into `broken` using an explicit PRNG key.
broken=jnp.stack([jax.random.normal(root,(2,)) for _ in range(3)])
# Assert invariant `jnp.array_equal(broken[0],broken[1])` holds
assert jnp.array_equal(broken[0],broken[1])
# Compute `current` from `root`
current=root
# Compute `rows` from `[]`
rows=[]
# Repeat the update loop over `range(3)` steps:
for _ in range(3):
    # Run `noise_step` to compute `(current, noise)`.
    current,noise=noise_step(current)
    # Append the current step result to `rows`.
    rows.append(noise)
# Combine or mask array elements to form `fixed`.
fixed=jnp.stack(rows)
# Check tensor shape invariant: `fixed.shape==(3,2)`
assert fixed.shape==(3,2)
# Assert invariant `not jnp.array_equal(fixed[0],fixed[1])` holds
assert not jnp.array_equal(fixed[0],fixed[1])
print("PASS: state-01")
