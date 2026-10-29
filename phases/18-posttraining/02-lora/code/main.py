"""LoRA: adapt, save and merge low-rank updates: worked experiments and reference solutions. CPU checks."""

# 1. Define the frozen-plus-adapter forward pass
# Step 1 — 1. Define the frozen-plus-adapter forward pass
import jax
import jax.numpy as jnp
import numpy as np

def lora_forward(x, base, a, b, alpha=1.0):
    rank = a.shape[1]
    return x @ base + (alpha / rank) * (x @ a @ b)

# 2. Train a base, freeze it and initialize the adapter
# Step 2 — 2. Train a base, freeze it and initialize the adapter: This step includes the source-task training.
x = jnp.asarray(np.random.default_rng(4).normal(size=(24, 6)), jnp.float32)
source_w = jnp.arange(24, dtype=jnp.float32).reshape(6, 4) / 40
base = jnp.zeros_like(source_w)
base_step = jax.jit(jax.grad(lambda w: jnp.mean((x @ w - x @ source_w) ** 2)))
for _ in range(150):
    base = base - 0.15 * base_step(base)
assert float(jnp.mean((x @ base - x @ source_w) ** 2)) < 1e-4
# Transfer the device result to the host for an independent NumPy comparison.
frozen = np.asarray(base).copy()
u = jnp.array([[0.2], [-0.3], [0.1], [0.2], [-0.1], [0.4]])
v = jnp.array([[0.5, -0.2, 0.3, 0.1]])
target = x @ (base + u @ v)
a = jax.random.normal(jax.random.key(4), (6, 1)) * 0.1
b = jnp.zeros((1, 4))
params = (a, b)
loss = lambda ab: jnp.mean((lora_forward(x, base, *ab, alpha=1.0) - target) ** 2)
step = jax.jit(jax.value_and_grad(loss))
history = []
initial_grads = step(params)[1]
np.testing.assert_array_equal(initial_grads[0], np.zeros((6, 1)))
assert np.linalg.norm(initial_grads[1]) > 0

# 3. Adapt and verify the serving merge
# Step 3 — 3. Adapt and verify the serving merge: The reference target has rank one by construction.
for _ in range(250):
    value, g = step(params)
    history.append(float(value))
    params = jax.tree.map(lambda a, b: a - 0.4 * b, params, g)

a, b = params
assert history[-1] < history[0] * 0.02
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

# Step 1 — 1. Define the frozen-plus-adapter forward pass
import jax
import jax.numpy as jnp
import numpy as np

def lora_forward(x, base, a, b, alpha=1.0):
    rank = a.shape[1]
    return x @ base + (alpha / rank) * (x @ a @ b)

# Step 2 — 2. Train a base, freeze it and initialize the adapter: This step includes the source-task training.
x = jnp.asarray(np.random.default_rng(4).normal(size=(24, 6)), jnp.float32)
source_w = jnp.arange(24, dtype=jnp.float32).reshape(6, 4) / 40
base = jnp.zeros_like(source_w)
base_step = jax.jit(jax.grad(lambda w: jnp.mean((x @ w - x @ source_w) ** 2)))
for _ in range(150):
    base = base - 0.15 * base_step(base)
assert float(jnp.mean((x @ base - x @ source_w) ** 2)) < 1e-4
# Transfer the device result to the host for an independent NumPy comparison.
frozen = np.asarray(base).copy()
u = jnp.array([[0.2], [-0.3], [0.1], [0.2], [-0.1], [0.4]])
v = jnp.array([[0.5, -0.2, 0.3, 0.1]])
target = x @ (base + u @ v)
a = jax.random.normal(jax.random.key(4), (6, 1)) * 0.1
b = jnp.zeros((1, 4))
params = (a, b)
loss = lambda ab: jnp.mean((lora_forward(x, base, *ab, alpha=1.0) - target) ** 2)
step = jax.jit(jax.value_and_grad(loss))
history = []
initial_grads = step(params)[1]
np.testing.assert_array_equal(initial_grads[0], np.zeros((6, 1)))
assert np.linalg.norm(initial_grads[1]) > 0

# Step 3 — 3. Adapt and verify the serving merge: The reference target has rank one by construction.
for _ in range(250):
    value, g = step(params)
    history.append(float(value))
    params = jax.tree.map(lambda a, b: a - 0.4 * b, params, g)

a, b = params
assert history[-1] < history[0] * 0.02
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
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'adaptation MSE (squared output units)','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']

extra_panel={'kind':'bar','x':[0,1],'labels':['factor A','factor B'],'series':[{'label':'initial gradient norm','y':[float(jnp.linalg.norm(initial_grads[0])),float(jnp.linalg.norm(initial_grads[1]))]}],'xlabel':'trainable factor','ylabel':'gradient L2 norm','title':'Why the first update changes only B'}
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}

# Experiment: Show why two zero factors cannot start learning
# Experiment — Show why two zero factors cannot start learning: Each factor’s derivative contains the other factor.
zero = (jnp.zeros_like(a), jnp.zeros_like(b))
zero_grad = jax.grad(loss)(zero)
assert all(
    np.array_equal(np.asarray(g), np.zeros(g.shape)) for g in zero_grad
)
print('Both zero factors produce zero first gradients.')

# Experiment: Check both factor gradients with matrix calculus
# Experiment — Check both factor gradients with matrix calculus
rng = np.random.default_rng(14)
a_probe = jnp.asarray(rng.normal(size=(6, 2)) * 0.1, jnp.float32)
b_probe = jnp.asarray(rng.normal(size=(2, 4)) * 0.1, jnp.float32)
scale = 3.0 / 2
output_gradient = (
    2 * (lora_forward(x, base, a_probe, b_probe, 3.0) - target) / target.size
)
expected_a = scale * x.T @ output_gradient @ b_probe.T
expected_b = scale * a_probe.T @ x.T @ output_gradient
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
# Primary adapter-save exercise.
# Use the complete following practice: “Enforce base identity in a fresh-process adapter reload.”
# It writes a base-hash-bound manifest, checks target module/rank/alpha/scaling/dtypes/shapes,
# validates in a fresh interpreter, compares merged/unmerged predictions on new inputs,
# and rejects a shape-compatible wrong base before activation.
# The earlier unbound adapter.npz pattern is not a reusable adapter artifact.

# Reference practice: Check the merge on new inputs
# Check the merge on new inputs (Transfer)
new_x = jnp.asarray(np.random.default_rng(91).normal(size=(7, 6)), jnp.float32)
np.testing.assert_allclose(
    lora_forward(new_x, base, a, b), new_x @ merged, atol=1e-6, rtol=1e-5
)
assert np.linalg.norm(np.asarray(new_x @ (base + 2 * a @ b) - new_x @ merged)) > 1e-3
print('New-input merge passes; accidental double merge changes outputs.')

# Reference practice: Test nonunit scaling and an impossible rank-one target
# Test nonunit scaling and an impossible rank-one target (Challenge)
rng = np.random.default_rng(8)
a2 = jnp.asarray(rng.normal(size=(6, 2)), jnp.float32)
b2 = jnp.asarray(rng.normal(size=(2, 4)), jnp.float32)
np.testing.assert_allclose(
    lora_forward(x, base, a2, b2, alpha=3.0),
    x @ (base + 1.5 * a2 @ b2),
    rtol=1e-5,
    atol=2e-6,
)
target_update = np.diag([3.0, 1.0])
u_s, s_s, v_s = np.linalg.svd(target_update)
rank_one = (u_s[:, :1] * s_s[:1]) @ v_s[:1]
np.testing.assert_allclose(np.sum((target_update - rank_one) ** 2), 1.0)
np.testing.assert_allclose(
    np.mean((np.eye(2) @ target_update - np.eye(2) @ rank_one) ** 2), 0.25
)
print('Nonunit scaling passes; best rank-one identity-input MSE is 0.25.')

# Reference practice: Enforce base identity in a fresh-process adapter reload
# Fresh-process adapter bundle round trip with base identity and negative controls.
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

rng = np.random.default_rng(815)
base2 = np.asarray(base, dtype=np.float32).copy()
base_before_adapter_fit = base2.copy()
alpha, rank = 3.0, 2
scale = alpha / rank
# A bounded rank-two teaching target; fit the factors before serializing them.
u2 = jnp.asarray([[0.2, -0.1], [-0.3, 0.1], [0.1, 0.2], [0.2, -0.2], [-0.1, 0.1], [0.4, 0.2]], jnp.float32)
v2 = jnp.asarray([[0.5, -0.2, 0.3, 0.1], [-0.1, 0.4, 0.2, -0.3]], jnp.float32)
target2 = x @ (jnp.asarray(base2) + scale * u2 @ v2)
a2 = jax.random.normal(jax.random.key(815), (6, rank)) * 0.1
b2 = jnp.zeros((rank, 4), jnp.float32)
fit2 = jax.jit(jax.value_and_grad(lambda aa, bb: jnp.mean(
    (lora_forward(x, jnp.asarray(base2), aa, bb, alpha) - target2) ** 2
), argnums=(0, 1)))
fit_history = []
for _ in range(300):
    fit_loss, (grad_a, grad_b) = fit2(a2, b2)
    fit_history.append(float(fit_loss))
    a2 = a2 - 0.2 * grad_a
    b2 = b2 - 0.2 * grad_b
assert fit_history[-1] < fit_history[0] * 1e-3
np.testing.assert_array_equal(base2, base_before_adapter_fit)
a2 = np.asarray(a2, dtype=np.float32)
b2 = np.asarray(b2, dtype=np.float32)
changed_x = rng.normal(size=(5, 6)).astype(np.float32)
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    np.save(root / "base.npy", base2, allow_pickle=False)
    np.savez(root / "adapter.npz", A=a2, B=b2)
    with np.load(root / "adapter.npz", allow_pickle=False) as trained_bundle:
        np.testing.assert_array_equal(trained_bundle["A"], a2)
        np.testing.assert_array_equal(trained_bundle["B"], b2)
    base_hash = hashlib.sha256((root / "base.npy").read_bytes()).hexdigest()
    manifest = {
        "schema": 1, "base_sha256": base_hash,
        "target_modules": ["decoder.block.0.proj"],
        "base_shape": list(base2.shape), "base_dtype": str(base2.dtype),
        "rank": rank, "alpha": alpha, "scaling": "alpha_over_rank",
        "A_shape": list(a2.shape), "B_shape": list(b2.shape),
        "A_dtype": str(a2.dtype), "B_dtype": str(b2.dtype),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    # New interpreter reloads all three artifacts from disk and validates before activation.
    child = r"""import hashlib,json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); m=json.loads((root/'manifest.json').read_text())
base_path=Path(sys.argv[2]) if len(sys.argv)>2 else root/'base.npy'
base_bytes=base_path.read_bytes()
if hashlib.sha256(base_bytes).hexdigest()!=m['base_sha256']: raise ValueError('base checkpoint identity mismatch')
base=np.load(base_path,allow_pickle=False)
with np.load(root/'adapter.npz',allow_pickle=False) as z: A=z['A']; B=z['B']
if list(base.shape)!=m['base_shape'] or str(base.dtype)!=m['base_dtype']: raise ValueError('base shape/dtype mismatch')
if list(A.shape)!=m['A_shape'] or list(B.shape)!=m['B_shape'] or str(A.dtype)!=m['A_dtype'] or str(B.dtype)!=m['B_dtype']: raise ValueError('adapter tensor metadata mismatch')
if m['rank']!=A.shape[1] or B.shape[0]!=m['rank'] or m['scaling']!='alpha_over_rank' or m['alpha']!=3.0: raise ValueError('rank/scaling metadata mismatch')
if m['target_modules']!=['decoder.block.0.proj']: raise ValueError('target module mismatch')
x=np.load(root/'inputs.npy',allow_pickle=False); scale=m['alpha']/m['rank']
separate=x@base+scale*(x@A@B); merged=x@(base+scale*A@B)
np.testing.assert_allclose(separate,merged,rtol=1e-6,atol=1e-6)
np.save(root/'child-predictions.npy',separate,allow_pickle=False)
print('fresh-process validated',m['base_sha256'],float(np.max(np.abs(separate-merged))))"""
    np.save(root / "inputs.npy", changed_x, allow_pickle=False)
    subprocess.run([sys.executable, "-c", child, str(root)], check=True, capture_output=True, text=True)
    child_predictions = np.load(root / "child-predictions.npy", allow_pickle=False)
    np.testing.assert_allclose(child_predictions, changed_x @ base2 + 1.5 * (changed_x @ a2 @ b2), rtol=1e-6, atol=1e-6)
    # Shape-compatible wrong base must fail on the base digest before activation.
    wrong = base2.copy(); wrong[0, 0] += 0.25
    np.save(root / "wrong-base.npy", wrong, allow_pickle=False)
    result = subprocess.run([sys.executable, "-c", child, str(root), str(root / "wrong-base.npy")], capture_output=True, text=True)
    assert result.returncode != 0 and "base checkpoint identity mismatch" in result.stderr
    # Each incompatible metadata edit must fail before predictions are activated.
    negatives = [
        ("rank", 1, "rank/scaling metadata mismatch"),
        ("alpha", 4.0, "rank/scaling metadata mismatch"),
        ("target_modules", ["decoder.block.1.proj"], "target module mismatch"),
        ("A_shape", [6, 1], "adapter tensor metadata mismatch"),
    ]
    for field, value, message in negatives:
        bad = dict(manifest, **{field: value})
        (root / "manifest.json").write_text(json.dumps(bad, sort_keys=True))
        result = subprocess.run([sys.executable, "-c", child, str(root)], capture_output=True, text=True)
        assert result.returncode != 0 and message in result.stderr, (field, result.stderr)
print("rank-two alpha/r=1.5 teaching-fit loss:", fit_history[0], fit_history[-1])
print("Fresh-process trained-factor round trip passed at alpha/r=1.5; wrong same-shape base and altered rank, alpha, target, and shape metadata were rejected.")
print("PASS: posttraining-02")
