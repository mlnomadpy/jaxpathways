"""Random keys without surprises: worked experiments and reference solutions. CPU checks."""

# Prepare the inputs
import jax
import jax.numpy as jnp

# Build the computation
key = jax.random.key(42)
key, sample_key = jax.random.split(key)
a = jax.random.normal(sample_key, (3,))
repeated = jax.random.normal(sample_key, (3,))
key, next_key = jax.random.split(key)
b = jax.random.normal(next_key, (3,))

# Run and check the result
print("Replay equal:", bool(jnp.array_equal(a, repeated)))
print("Fresh draw equal:", bool(jnp.array_equal(a, b)))
assert jnp.array_equal(a, repeated)
assert not jnp.array_equal(a, b)

import jax
import jax.numpy as jnp
key = jax.random.key(42)
key, sample_key = jax.random.split(key)
a = jax.random.normal(sample_key, (3,))
repeated = jax.random.normal(sample_key, (3,))
key, next_key = jax.random.split(key)
b = jax.random.normal(next_key, (3,))
print("Replay equal:", bool(jnp.array_equal(a, repeated)))
print("Fresh draw equal:", bool(jnp.array_equal(a, b)))
assert jnp.array_equal(a, repeated)
assert not jnp.array_equal(a, b)

# Figure data experiment
visual_data = {'kind': 'line', 'x': [0, 1, 2], 'xlabel': 'sample coordinate', 'ylabel': 'normal draw', 'series': [{'label': 'first draw', 'y': a.tolist()}, {'label': 'same key replay', 'y': repeated.tolist()}, {'label': 'split next draw', 'y': b.tolist()}]}

# Experiment: Implement one random state transition
def noise_step(key):
    next_key,draw_key=jax.random.split(key)
    return next_key,jax.random.normal(draw_key,(2,))
root=jax.random.key(9)
k1,n1=noise_step(root)
k2,n2=noise_step(k1)
r1,replay1=noise_step(jax.random.key(9))
r2,replay2=noise_step(r1)
assert jnp.array_equal(n1,replay1) and jnp.array_equal(n2,replay2)
assert jnp.array_equal(jax.random.key_data(k2),jax.random.key_data(r2))

# Experiment: Keep identity under reordering
base=jax.random.key(11)
def by_id(ids):
    return jax.vmap(lambda i:jax.random.normal(jax.random.fold_in(base,i),()))(ids)
forward=by_id(jnp.array([3,8]))
reverse=by_id(jnp.array([8,3]))
assert jnp.array_equal(forward,reverse[::-1])

# Reference solution. Try the exercise before reading this.
def draw(seed):
    keys = jax.random.split(jax.random.key(seed), 4)
    return jax.vmap(lambda k: jax.random.normal(k, (2,)))(keys)
assert draw(7).shape == (4, 2)
assert jnp.array_equal(draw(7), draw(7))

# Reference practice: Replay a noisy prediction
def noisy_prediction(seed,x):
    eps=jax.random.normal(jax.random.key(seed),x.shape)
    return 2*x+1+eps
inputs=jnp.array([0.,1.,2.])
assert jnp.array_equal(noisy_prediction(4,inputs),noisy_prediction(4,inputs))
assert jnp.allclose(noisy_prediction(4,inputs+1)-noisy_prediction(4,inputs),2.,atol=1e-6)

# Reference practice: Diagnose key reuse in a loop
root=jax.random.key(5)
broken=jnp.stack([jax.random.normal(root,(2,)) for _ in range(3)])
assert jnp.array_equal(broken[0],broken[1])
current=root
rows=[]
for _ in range(3):
    current,noise=noise_step(current)
    rows.append(noise)
fixed=jnp.stack(rows)
assert fixed.shape==(3,2)
assert not jnp.array_equal(fixed[0],fixed[1])
print("PASS: state-01")
