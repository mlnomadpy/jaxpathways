"""LoRA: adapt, save and merge low-rank updates: worked experiments and reference solutions. CPU checks."""

# 1. Define the frozen-plus-adapter forward pass
# Step 1 — 1. Define the frozen-plus-adapter forward pass: The base is an input to the forward pass but is not in the adapter...
import jax
import jax.numpy as jnp
import numpy as np

def lora_forward(x, base, a, b, alpha=1.0):
    rank = a.shape[1]
    return x @ base + (alpha / rank) * (x @ a @ b)

# 2. Train a base, freeze it and initialize the adapter
# Step 2 — 2. Train a base, freeze it and initialize the adapter: This step includes the source-task training.
x = jnp.asarray(np.random.default_rng(4).normal(size=(24, 6)), jnp.float32)
# Construct and reshape `source_w` into the target tensor dimensions.
source_w = jnp.arange(24, dtype=jnp.float32).reshape(6, 4) / 40
base = jnp.zeros_like(source_w)
# Differentiate the objective to obtain `base_step` via automatic differentiation.
base_step = jax.jit(jax.grad(lambda w: jnp.mean((x @ w - x @ source_w) ** 2)))
for _ in range(150):
    base = base - 0.15 * base_step(base)
assert float(jnp.mean((x @ base - x @ source_w) ** 2)) < 1e-4
# Convert `frozen` to a host NumPy array for inspection or verification.
frozen = np.asarray(base).copy()
u = jnp.array([[0.2], [-0.3], [0.1], [0.2], [-0.1], [0.4]])
v = jnp.array([[0.5, -0.2, 0.3, 0.1]])
# Perform matrix / vector contraction (`@`) to compute `target`.
target = x @ (base + u @ v)
# Initialize explicit deterministic PRNG key `a`.
a = jax.random.normal(jax.random.key(4), (6, 1)) * 0.1
# Allocate initialized array `b` with the specified shape and dtype.
b = jnp.zeros((1, 4))
params = (a, b)
loss = lambda ab: jnp.mean((lora_forward(x, base, *ab, alpha=1.0) - target) ** 2)
# Evaluate both scalar loss and parameter gradients in one pass (`step`).
step = jax.jit(jax.value_and_grad(loss))
history = []
initial_grads = step(params)[1]
# Allocate initialized array `` with the specified shape and dtype.
np.testing.assert_array_equal(initial_grads[0], np.zeros((6, 1)))
assert np.linalg.norm(initial_grads[1]) > 0

# 3. Adapt and verify the serving merge
# Step 3 — 3. Adapt and verify the serving merge: The reference target has rank one by construction.
for _ in range(250):
    value, g = step(params)
    history.append(float(value))
    # Transform every leaf of the parameter PyTree (`params`).
    params = jax.tree.map(lambda a, b: a - 0.4 * b, params, g)

a, b = params
assert history[-1] < history[0] * 0.02
# Execute `np.testing.assert_array_equal(base, frozen)`
np.testing.assert_array_equal(base, frozen)
merged = base + a @ b
np.testing.assert_allclose(
    lora_forward(x, base, a, b), x @ merged, rtol=1e-5, atol=1e-6
)
print(
    'LoRA initial/final:',
    history[0],
    history[-1],
    '; trainable:',
    a.size + b.size,
    'base:',
    base.size,
)

# Step 1 — 1. Define the frozen-plus-adapter forward pass: The base is an input to the forward pass but is not in the adapter...
import jax
import jax.numpy as jnp
import numpy as np

def lora_forward(x, base, a, b, alpha=1.0):
    rank = a.shape[1]
    return x @ base + (alpha / rank) * (x @ a @ b)

# Step 2 — 2. Train a base, freeze it and initialize the adapter: This step includes the source-task training.
x = jnp.asarray(np.random.default_rng(4).normal(size=(24, 6)), jnp.float32)
# Construct and reshape `source_w` into the target tensor dimensions.
source_w = jnp.arange(24, dtype=jnp.float32).reshape(6, 4) / 40
base = jnp.zeros_like(source_w)
# Differentiate the objective to obtain `base_step` via automatic differentiation.
base_step = jax.jit(jax.grad(lambda w: jnp.mean((x @ w - x @ source_w) ** 2)))
for _ in range(150):
    base = base - 0.15 * base_step(base)
assert float(jnp.mean((x @ base - x @ source_w) ** 2)) < 1e-4
# Convert `frozen` to a host NumPy array for inspection or verification.
frozen = np.asarray(base).copy()
u = jnp.array([[0.2], [-0.3], [0.1], [0.2], [-0.1], [0.4]])
v = jnp.array([[0.5, -0.2, 0.3, 0.1]])
# Perform matrix / vector contraction (`@`) to compute `target`.
target = x @ (base + u @ v)
# Initialize explicit deterministic PRNG key `a`.
a = jax.random.normal(jax.random.key(4), (6, 1)) * 0.1
# Allocate initialized array `b` with the specified shape and dtype.
b = jnp.zeros((1, 4))
params = (a, b)
loss = lambda ab: jnp.mean((lora_forward(x, base, *ab, alpha=1.0) - target) ** 2)
# Evaluate both scalar loss and parameter gradients in one pass (`step`).
step = jax.jit(jax.value_and_grad(loss))
history = []
initial_grads = step(params)[1]
# Allocate initialized array `` with the specified shape and dtype.
np.testing.assert_array_equal(initial_grads[0], np.zeros((6, 1)))
assert np.linalg.norm(initial_grads[1]) > 0

# Step 3 — 3. Adapt and verify the serving merge: The reference target has rank one by construction.
for _ in range(250):
    value, g = step(params)
    history.append(float(value))
    # Transform every leaf of the parameter PyTree (`params`).
    params = jax.tree.map(lambda a, b: a - 0.4 * b, params, g)

a, b = params
assert history[-1] < history[0] * 0.02
# Execute `np.testing.assert_array_equal(base, frozen)`
np.testing.assert_array_equal(base, frozen)
merged = base + a @ b
np.testing.assert_allclose(
    lora_forward(x, base, a, b), x @ merged, rtol=1e-5, atol=1e-6
)
print(
    'LoRA initial/final:',
    history[0],
    history[-1],
    '; trainable:',
    a.size + b.size,
    'base:',
    base.size,
)

# Figure data experiment
# Compute figure data for: LoRA: adapt, save and merge low-rank updates — recorded experiment
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'adaptation MSE (squared output units)','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']

extra_panel={'kind':'bar','x':[0,1],'labels':['factor A','factor B'],'series':[{'label':'initial gradient norm','y':[float(jnp.linalg.norm(initial_grads[0])),float(jnp.linalg.norm(initial_grads[1]))]}],'xlabel':'trainable factor','ylabel':'gradient L2 norm','title':'Why the first update changes only B'}
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}

# Experiment: Show why two zero factors cannot start learning
# Experiment — Show why two zero factors cannot start learning: Each factor’s derivative contains the other factor.
zero = (jnp.zeros_like(a), jnp.zeros_like(b))
# Differentiate the objective to obtain `zero_grad` via automatic differentiation.
zero_grad = jax.grad(loss)(zero)
assert all(
    np.array_equal(np.asarray(g), np.zeros(g.shape)) for g in zero_grad
)
print('Both zero factors produce zero first gradients.')

# Experiment: Check both factor gradients with matrix calculus
# Experiment — Check both factor gradients with matrix calculus: Nonzero factors exercise both gradient paths, while nonunit...
rng = np.random.default_rng(14)
# Create device-backed JAX array `a_probe`.
a_probe = jnp.asarray(rng.normal(size=(6, 2)) * 0.1, jnp.float32)
# Create device-backed JAX array `b_probe`.
b_probe = jnp.asarray(rng.normal(size=(2, 4)) * 0.1, jnp.float32)
scale = 3.0 / 2
# Evaluate the compound expression for `output_gradient`.
output_gradient = (
    2 * (lora_forward(x, base, a_probe, b_probe, 3.0) - target) / target.size
)
# Perform matrix / vector contraction (`@`) to compute `expected_a`.
expected_a = scale * x.T @ output_gradient @ b_probe.T
# Perform matrix / vector contraction (`@`) to compute `expected_b`.
expected_b = scale * a_probe.T @ x.T @ output_gradient
# Differentiate the objective to obtain `(actual_a, actual_b)` via automatic differentiation.
actual_a, actual_b = jax.grad(
    lambda aa, bb: jnp.mean(
        (lora_forward(x, base, aa, bb, 3.0) - target) ** 2
    ),
    argnums=(0, 1),
)(a_probe, b_probe)
np.testing.assert_allclose(actual_a, expected_a, atol=1e-6)
np.testing.assert_allclose(actual_b, expected_b, atol=1e-6)
print('Both LoRA factor gradients match the matrix-calculus oracle at alpha/r = 1.5.')

# Reference solution. Try the exercise before reading this.
# Exercise solution: Save both adapter factors and their rank/scaling metadata, reload them...
import tempfile
from pathlib import Path

# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / 'adapter.npz'
    # Convert `` to a host NumPy array for inspection or verification.
    np.savez(
        path,
        a=np.asarray(a),
        b=np.asarray(b),
        alpha=np.array(1.0),
        rank=np.array(1),
    )
    # Enter `np.load(path, allow_pickle=False)` context block:
    with np.load(path, allow_pickle=False) as z:
        # Check tensor shape invariant: `int(z['rank']) == z['a'].shape[1]`
        assert int(z['rank']) == z['a'].shape[1]
        np.testing.assert_allclose(
            lora_forward(x, base, z['a'], z['b'], float(z['alpha'])),
            x @ merged,
            atol=1e-6,
            rtol=1e-5,
        )
print('Adapter round trip and merge checked.')

# Reference practice: Check the merge on new inputs
# Check the merge on new inputs (Transfer): The second check detects an operational merge error that...
new_x = jnp.asarray(np.random.default_rng(91).normal(size=(7, 6)), jnp.float32)
np.testing.assert_allclose(
    lora_forward(new_x, base, a, b), new_x @ merged, atol=1e-6, rtol=1e-5
)
assert np.linalg.norm(np.asarray(new_x @ (base + 2 * a @ b) - new_x @ merged)) > 1e-3
print('New-input merge passes; accidental double merge changes outputs.')

# Reference practice: Test nonunit scaling and an impossible rank-one target
# Test nonunit scaling and an impossible rank-one target (Challenge): The residual predicts a capacity limit independently of an...
rng = np.random.default_rng(8)
# Create device-backed JAX array `a2`.
a2 = jnp.asarray(rng.normal(size=(6, 2)), jnp.float32)
# Create device-backed JAX array `b2`.
b2 = jnp.asarray(rng.normal(size=(2, 4)), jnp.float32)
np.testing.assert_allclose(
    lora_forward(x, base, a2, b2, alpha=3.0),
    x @ (base + 1.5 * a2 @ b2),
    rtol=1e-5,
    atol=2e-6,
)
target_update = np.diag([3.0, 1.0])
u_s, s_s, v_s = np.linalg.svd(target_update)
# Perform matrix / vector contraction (`@`) to compute `rank_one`.
rank_one = (u_s[:, :1] * s_s[:1]) @ v_s[:1]
# Execute `np.testing.assert_allclose(np.sum((target_update - rank_one) ** 2), 1.0)`.
np.testing.assert_allclose(np.sum((target_update - rank_one) ** 2), 1.0)
# Construct an identity matrix ``.
np.testing.assert_allclose(
    np.mean((np.eye(2) @ target_update - np.eye(2) @ rank_one) ** 2), 0.25
)
print('Nonunit scaling passes; best rank-one identity-input MSE is 0.25.')
print("PASS: posttraining-02")
