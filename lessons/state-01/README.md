# Random keys without surprises

Phase 03: State, randomness & control flow · about 60 minutes · CPU

## What you will be able to do

- Treat a key as an explicit input
- Carry the continuation key explicitly
- Separate replay from a statistical claim
- Allocate randomness by sample identity

## The problem

You saved the seed, yet your simulation repeats exactly the same noise at every step. Let’s follow the random key. In JAX, using a key again repeats the same draw; splitting it gives you keys for separate draws. We’ll practice both deliberate replay and fresh sampling before adding randomness to a training loop.

## The idea

A random sampler in JAX receives an explicit key. Reusing the same key with the same sampler repeats the same computation. Splitting gives us separately owned keys, so the program can show which operation receives which randomness.

## A key is an input, not a moving cursor

Imagine a training step that needs dropout randomness and a key for the next step. Split the incoming key, use one child for dropout, and return the other as the continuation key. The drawing should branch once; it should not show a sampler secretly advancing the parent key.

If two augmentation calls receive the same key and otherwise identical inputs, they can produce the same augmentation. Different variable names do not make the underlying key values different. This is a data-flow bug that ordinary shape checks will miss.

For recovery, save the continuation key at the same boundary as model and data position. Reconstructing an initial seed and restarting the random stream is not equivalent to restoring the next key after many training steps.

### Pause and reason

Does calling a sampler consume or mutate the key variable?

<details><summary>Compare your reasoning</summary>

No. The key is an explicit value. Your program must choose new child keys and carry the next key forward; otherwise repeated calls can replay the same sample.

</details>

## Treat a key as an explicit input

A JAX random operation is deterministic given its key and other arguments. The key is not a mutable generator that automatically advances after a call. Calling normal twice with the same key, shape and dtype reproduces the draw. That property is useful for replay and dangerous when accidental reuse makes samples correlated.

A seed is only the initial description of an experiment. Splitting describes where subsequent random choices come from. Write down which function owns the continuation key and which subkey is consumed by a draw. Do not pass one subkey to two independent components.

```text
seed → root key
root key --split--> continuation key + sample key
sample key --normal(shape=(3,))--> one sample
```

## Carry the continuation key explicitly

In the worked program, `key, sample_key = split(key)` replaces the variable key with a continuation value. The old key was not mutated; Python rebound a name. sample_key belongs to the immediate draw. The next split uses the retained continuation, not sample_key.

This convention gives each training or simulation step a small state transition: receive key → split → sample → return next key. Later, checkpoints need to store that next key alongside model and optimizer state. Storing parameters alone cannot reproduce the next random batch.

## Separate replay from a statistical claim

For this fixed seed, fresh subkeys produce visibly different draws. That observation is not proof of statistical independence, and different random draws can theoretically coincide. Equality of a single pair is therefore not a general test of randomness quality.

The reliable replay test is to reconstruct the same program from the same seed and compare its results. A change to key allocation can alter all later samples. Do not promise identical streams across arbitrary JAX versions, PRNG implementations, shapes or devices.

## Allocate randomness by sample identity

Splitting a root into four keys is useful for one fixed batch. If the batch membership changes, assignment by position may associate a different draw with a particular sample. fold_in lets you combine a stable integer identity with a base key, which can make per-example randomness stable under reordering.

Keep identifiers unique where distinct randomness is intended. Reusing an identifier repeats its derived key. This is deterministic bookkeeping, not a security primitive or a substitute for recording the PRNG implementation.

## Prepare the inputs

Create main.py in your lesson workspace. Add this first block; use the environment from setup.

```python
import jax
import jax.numpy as jnp
```

These explicit inputs define the case that the later checks will verify.

## Build the computation

Append this block below the inputs in the same file.

```python
key = jax.random.key(42)
key, sample_key = jax.random.split(key)
a = jax.random.normal(sample_key, (3,))
repeated = jax.random.normal(sample_key, (3,))
key, next_key = jax.random.split(key)
b = jax.random.normal(next_key, (3,))
```

random.split returns two new keys. noise_step returns the continuation key as part of its result so the caller can use fresh noise on the next transition.

## Run and check the result

Append the checks, save main.py, and run python main.py from this folder using your course environment.

```python
print("Replay equal:", bool(jnp.array_equal(a, repeated)))
print("Fresh draw equal:", bool(jnp.array_equal(a, b)))
assert jnp.array_equal(a, repeated)
assert not jnp.array_equal(a, b)
```

Compare the output to the expected result below before making the exercise change.

## Run the example

```python
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
```

Expected: Replay equal: True; Fresh draw equal: False.

## Reused keys replay the same sample

**Predict:** Will a repeated draw lie on top of the original?

![Reused keys replay the same sample](../../phases/03-state/01-random-keys-without-surprises/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis selects a coordinate within a three-element random draw; the vertical axis gives its sampled value. The solid first-draw line and dashed same-key replay line lie on top of each other. The dash-dot line comes from a new split key.

The replay repeats approximately $(0.606,0.799,-0.909)$, while the split-key draw is approximately $(-0.211,-1.363,-0.045)$. Two curves are hard to distinguish precisely because their coordinates are equal.

### Connect it to the computation

A JAX random function is deterministic given the same key and arguments. Calling it again with that key replays the result; it does not silently advance a hidden random state. Splitting produces a new key for a new draw.

The lines only connect coordinates to make equality visible. This is not a time series or a probability density, and three different values cannot establish statistical independence. The practical check is reproducibility: reuse for deliberate replay, split for the next stochastic operation.

```python
visual_data = {'kind': 'line', 'x': [0, 1, 2], 'xlabel': 'sample coordinate', 'ylabel': 'normal draw', 'series': [{'label': 'first draw', 'y': a.tolist()}, {'label': 'same key replay', 'y': repeated.tolist()}, {'label': 'split next draw', 'y': b.tolist()}]}
```

## Recorded reference execution

CPU run: 2026-10-06T22:58:43.761358+00:00. JAX 0.9.2.

```text
Replay equal: True
Fresh draw equal: False
Replay equal: True
Fresh draw equal: False
PASS: state-01

```

## Implement one random state transition

**Predict before running:** Predict which part of the return value must be reused for the next draw.

```python
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
```

**Expected:** Reconstructing the two steps reproduces both samples and the continuation key.

Replay compares complete state transitions. It catches lost continuation state, not only a repeated final number.

## Keep identity under reordering

**Predict before running:** If examples $3$ and $8$ swap order, should their identity-derived noise swap too?

```python
base=jax.random.key(11)
def by_id(ids):
    return jax.vmap(lambda i:jax.random.normal(jax.random.fold_in(base,i),()))(ids)
forward=by_id(jnp.array([3,8]))
reverse=by_id(jnp.array([8,3]))
assert jnp.array_equal(forward,reverse[::-1])
```

**Expected:** Noise follows the identifiers, not their batch positions.

fold_in makes the assignment explicit. Repeated IDs would deliberately reproduce a draw.

## Make it yours

Split a fresh seed into four keys, then use vmap to draw two normal values from each. Replay from the seed and compare.

<details><summary>Reference solution</summary>

```python
def draw(seed):
    keys = jax.random.split(jax.random.key(seed), 4)
    return jax.vmap(lambda k: jax.random.normal(k, (2,)))(keys)
assert draw(7).shape == (4, 2)
assert jnp.array_equal(draw(7), draw(7))
```

</details>

## Replay a noisy prediction

**Practice**

Draw one noise value per example for prediction $2x+1$. Reconstruct the same batch from the seed, then change only $x$.

<details><summary>Hint</summary>

Reuse the random construction for a controlled comparison, not accidentally across independent training steps.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
def noisy_prediction(seed,x):
    eps=jax.random.normal(jax.random.key(seed),x.shape)
    return 2*x+1+eps
inputs=jnp.array([0.,1.,2.])
assert jnp.array_equal(noisy_prediction(4,inputs),noisy_prediction(4,inputs))
assert jnp.allclose(noisy_prediction(4,inputs+1)-noisy_prediction(4,inputs),2.,atol=1e-6)
```

Holding noise fixed isolates the changed numerical input. That is a deliberately coupled experiment.

</details>

## Diagnose key reuse in a loop

**Challenge**

Write the broken loop that draws with one key three times, demonstrate repeated rows, then repair it by threading the continuation.

<details><summary>Hint</summary>

The draw function does not mutate the key.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
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
```

The repeated-row symptom comes from key allocation. The last inequality checks this fixed demonstration, not every possible random stream.

</details>

## Check your understanding

What happens if the same key is used twice for the same random operation?

1. The generator automatically advances
2. The same sample is reproduced
3. JAX guarantees an exception

<details><summary>Answer and explanation</summary>

The same sample is reproduced

Keys are explicit immutable inputs. Identical inputs reproduce the draw; split keys to describe independent randomness.

</details>

## Diagnose the result

The repeated-row symptom comes from key allocation. The last inequality checks this fixed demonstration, not every possible random stream.

## Carry forward

- Replay compares complete state transitions. It catches lost continuation state, not only a repeated final number.
- fold_in makes the assignment explicit. Repeated IDs would deliberately reproduce a draw.

## Keep your evidence

Keep a key-ownership tree, two-step replay including the final key, the per-ID reorder check, and the broken/repaired repeated-noise loop.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [JAX pseudorandom numbers](https://docs.jax.dev/en/latest/101/random.html)

