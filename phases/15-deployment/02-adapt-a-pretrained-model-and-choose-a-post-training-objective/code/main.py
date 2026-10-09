"""Adapt a pretrained model and choose a post-training objective: worked experiments and reference solutions. CPU checks."""

# 1. Pretrain a small model and retain its source data identity
# Step 1 — 1. Pretrain a small model and retain its source data identity: The same stable cross-entropy accepts binary labels or teacher...
# Import hashlib for this computation.
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

# Function `design(key, n)` implementing this stage's computation:
def design(key,n):
    # Sample deterministic random values into `features` using an explicit PRNG key.
    features=jax.random.normal(key,(n,2))
    # Return `jnp.concatenate([features, jnp.ones((n, 1))], axis=1)` to the caller.
    return jnp.concatenate([features,jnp.ones((n,1))],axis=1)
# Create or split explicit PRNG key(s) (`source_X`) for reproducible randomness.
source_X=design(jax.random.key(10),256)
# Construct `source_targets` via `jax.nn.sigmoid(source_X@jnp.array([1.2,-.8,.2]))`
source_targets=jax.nn.sigmoid(source_X@jnp.array([1.2,-.8,.2]))
# Create or split explicit PRNG key(s) (`adapt_X`) for reproducible randomness.
adapt_X=design(jax.random.key(11),48)
# Create or split explicit PRNG key(s) (`held_X`) for reproducible randomness.
held_X=design(jax.random.key(12),512)
# Construct `teacher` via `jnp.array([.6,1.4,-.3])`
teacher=jnp.array([.6,1.4,-.3])
# Perform matrix contraction / projection to compute `soft_targets`.
soft_targets=jax.nn.sigmoid(adapt_X@teacher)
# Create or split explicit PRNG key(s) (`hard_targets`) for reproducible randomness.
hard_targets=jax.random.bernoulli(jax.random.key(13),soft_targets).astype(jnp.float32)
# Perform matrix contraction / projection to compute `held_prob`.
held_prob=jax.nn.sigmoid(held_X@teacher)
# Function `objective(w, X, targets)` implementing this stage's computation:
def objective(w,X,targets):
    # Perform matrix contraction / projection to compute `logits`.
    logits=X@w
    # Return `jnp.mean(jnp.logaddexp(0.0, logits) - targets * logits)` to the caller.
    return jnp.mean(jnp.logaddexp(0.,logits)-targets*logits)
# Function `train(initial, X, targets, steps, ...)` implementing this stage's computation:
def train(initial,X,targets,steps=300,rate=.15):
    # Function `step(w, _)` implementing this stage's computation:
    def step(w,_):
        # Evaluate both scalar loss and parameter gradients in one pass (`(loss, grad)`).
        loss,grad=jax.value_and_grad(objective)(w,X,targets)
        # Return `(w - rate * grad, loss)` to the caller.
        return w-rate*grad,loss
    # Return `jax.lax.scan(step, initial, None, length=steps)` to the caller.
    return jax.lax.scan(step,initial,None,length=steps)
# Allocate initialized array `(base, pretrain_history)` with the specified shape and dtype.
base,pretrain_history=train(jnp.zeros(3),source_X,source_targets)
# Assert invariant `pretrain_history[-1]<pretrain_history[0]-.1` holds
assert pretrain_history[-1]<pretrain_history[0]-.1
# Convert `source_hash` to a host NumPy array for inspection or verification.
source_hash=hashlib.sha256(np.asarray(source_X).tobytes()+np.asarray(source_targets).tobytes()).hexdigest()

# 2. Save and reload before adapting
# Step 2 — 2. Save and reload before adapting: Both branches begin with identical pretrained parameters and a...
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory() as directory:
    # Read or serialize artifact data on disk (`checkpoint`).
    checkpoint=Path(directory)/'pretrained.npz'
    # Convert `` to a host NumPy array for inspection or verification.
    np.savez(checkpoint,weights=np.asarray(base))
    # Compute deterministic cryptographic digest `digest` for provenance verification.
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    # Compute deterministic cryptographic digest `manifest` for provenance verification.
    manifest={'model':'three-parameter logistic synthetic fixture','source_data_sha256':source_hash,
              'weights_sha256':digest,'source_seed':10,'pretrain_steps':300,'learning_rate':.15,
              'objective':'binary cross entropy with synthetic source probabilities',
              'jax':jax.__version__,'dtype':'float32','optimizer_state_included':False}
    # Read or serialize artifact data on disk (`manifest_path`).
    manifest_path=Path(directory)/'manifest.json'
    # Read or serialize artifact data on disk (``).
    manifest_path.write_text(json.dumps(manifest))
    # Read or serialize artifact data on disk (`recorded`).
    recorded=json.loads(manifest_path.read_text())
    # Assert invariant `hashlib.sha256(checkpoint.read_bytes()).hexdigest()==recorded['we...` holds
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==recorded['weights_sha256']
    # Enter managed runtime/context scope for this block:
    with np.load(checkpoint,allow_pickle=False) as saved:
        # Create device-backed JAX array `restored`.
        restored=jnp.asarray(saved['weights'])
# Execute `np.testing.assert_array_equal(restored,base)`
np.testing.assert_array_equal(restored,base)
# Compute `frozen` from `np.array(restored,copy=True)`
frozen=np.array(restored,copy=True)
# Run `train` to compute `(sft, sft_history)`.
sft,sft_history=train(restored,adapt_X,hard_targets,steps=200)
# Run `train` to compute `(distilled, distill_history)`.
distilled,distill_history=train(restored,adapt_X,soft_targets,steps=200)
# Execute `np.testing.assert_array_equal(restored,frozen)`
np.testing.assert_array_equal(restored,frozen)

# 3. Compare on a held-out criterion shared by both objectives
# Step 3 — 3. Compare on a held-out criterion shared by both objectives: The teacher is the known generating rule in this fixture, so...
# Construct `w_check` via `jnp.array([.2,-.3,.1])`
w_check=jnp.array([.2,-.3,.1])
# Convert `host_X` to a host NumPy array for inspection or verification.
host_X=np.asarray(adapt_X,dtype=np.float64)
# Convert `host_w` to a host NumPy array for inspection or verification.
host_w=np.asarray(w_check,dtype=np.float64)
# Perform matrix contraction / projection to compute `host_p`.
host_p=1/(1+np.exp(-(host_X@host_w)))
# Iterate over `targets` to step through the computation:
for targets in (hard_targets,soft_targets):
    # Convert `expected` to a host NumPy array for inspection or verification.
    expected=host_X.T@(host_p-np.asarray(targets))/len(host_X)
    # Differentiate the objective to obtain gradients ``.
    np.testing.assert_allclose(jax.grad(objective)(w_check,adapt_X,targets),expected,rtol=1e-5,atol=1e-6)
# Compute `metrics` from `{}`
metrics={}
# Iterate over `(name, weights)` to step through the computation:
for name,weights in [('pretrained',restored),('supervised',sft),('teacher',distilled)]:
    # Perform matrix contraction / projection to compute `metrics[name]`.
    metrics[name]={'held_cross_entropy':float(objective(weights,held_X,held_prob)),
                   'held_brier':float(jnp.mean((jax.nn.sigmoid(held_X@weights)-held_prob)**2)),
                   'source_cross_entropy':float(objective(weights,source_X,source_targets))}
# Assert invariant `metrics['supervised']['held_cross_entropy']<metrics['pretrained']...` holds
assert metrics['supervised']['held_cross_entropy']<metrics['pretrained']['held_cross_entropy']
# Assert invariant `metrics['teacher']['held_cross_entropy']<metrics['pretrained']['h...` holds
assert metrics['teacher']['held_cross_entropy']<metrics['pretrained']['held_cross_entropy']
# Print the observed values to compare against the expected result.
print(json.dumps(metrics,indent=2))
# Print diagnostic summary of the computed outputs.
print('checkpoint source hash:',source_hash)

# Step 1 — 1. Pretrain a small model and retain its source data identity: The same stable cross-entropy accepts binary labels or teacher...
# Import hashlib for this computation.
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

# Function `design(key, n)` implementing this stage's computation:
def design(key,n):
    # Sample deterministic random values into `features` using an explicit PRNG key.
    features=jax.random.normal(key,(n,2))
    # Return `jnp.concatenate([features, jnp.ones((n, 1))], axis=1)` to the caller.
    return jnp.concatenate([features,jnp.ones((n,1))],axis=1)
# Create or split explicit PRNG key(s) (`source_X`) for reproducible randomness.
source_X=design(jax.random.key(10),256)
# Construct `source_targets` via `jax.nn.sigmoid(source_X@jnp.array([1.2,-.8,.2]))`
source_targets=jax.nn.sigmoid(source_X@jnp.array([1.2,-.8,.2]))
# Create or split explicit PRNG key(s) (`adapt_X`) for reproducible randomness.
adapt_X=design(jax.random.key(11),48)
# Create or split explicit PRNG key(s) (`held_X`) for reproducible randomness.
held_X=design(jax.random.key(12),512)
# Construct `teacher` via `jnp.array([.6,1.4,-.3])`
teacher=jnp.array([.6,1.4,-.3])
# Perform matrix contraction / projection to compute `soft_targets`.
soft_targets=jax.nn.sigmoid(adapt_X@teacher)
# Create or split explicit PRNG key(s) (`hard_targets`) for reproducible randomness.
hard_targets=jax.random.bernoulli(jax.random.key(13),soft_targets).astype(jnp.float32)
# Perform matrix contraction / projection to compute `held_prob`.
held_prob=jax.nn.sigmoid(held_X@teacher)
# Function `objective(w, X, targets)` implementing this stage's computation:
def objective(w,X,targets):
    # Perform matrix contraction / projection to compute `logits`.
    logits=X@w
    # Return `jnp.mean(jnp.logaddexp(0.0, logits) - targets * logits)` to the caller.
    return jnp.mean(jnp.logaddexp(0.,logits)-targets*logits)
# Function `train(initial, X, targets, steps, ...)` implementing this stage's computation:
def train(initial,X,targets,steps=300,rate=.15):
    # Function `step(w, _)` implementing this stage's computation:
    def step(w,_):
        # Evaluate both scalar loss and parameter gradients in one pass (`(loss, grad)`).
        loss,grad=jax.value_and_grad(objective)(w,X,targets)
        # Return `(w - rate * grad, loss)` to the caller.
        return w-rate*grad,loss
    # Return `jax.lax.scan(step, initial, None, length=steps)` to the caller.
    return jax.lax.scan(step,initial,None,length=steps)
# Allocate initialized array `(base, pretrain_history)` with the specified shape and dtype.
base,pretrain_history=train(jnp.zeros(3),source_X,source_targets)
# Assert invariant `pretrain_history[-1]<pretrain_history[0]-.1` holds
assert pretrain_history[-1]<pretrain_history[0]-.1
# Convert `source_hash` to a host NumPy array for inspection or verification.
source_hash=hashlib.sha256(np.asarray(source_X).tobytes()+np.asarray(source_targets).tobytes()).hexdigest()

# Step 2 — 2. Save and reload before adapting: Both branches begin with identical pretrained parameters and a...
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory() as directory:
    # Read or serialize artifact data on disk (`checkpoint`).
    checkpoint=Path(directory)/'pretrained.npz'
    # Convert `` to a host NumPy array for inspection or verification.
    np.savez(checkpoint,weights=np.asarray(base))
    # Compute deterministic cryptographic digest `digest` for provenance verification.
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    # Compute deterministic cryptographic digest `manifest` for provenance verification.
    manifest={'model':'three-parameter logistic synthetic fixture','source_data_sha256':source_hash,
              'weights_sha256':digest,'source_seed':10,'pretrain_steps':300,'learning_rate':.15,
              'objective':'binary cross entropy with synthetic source probabilities',
              'jax':jax.__version__,'dtype':'float32','optimizer_state_included':False}
    # Read or serialize artifact data on disk (`manifest_path`).
    manifest_path=Path(directory)/'manifest.json'
    # Read or serialize artifact data on disk (``).
    manifest_path.write_text(json.dumps(manifest))
    # Read or serialize artifact data on disk (`recorded`).
    recorded=json.loads(manifest_path.read_text())
    # Assert invariant `hashlib.sha256(checkpoint.read_bytes()).hexdigest()==recorded['we...` holds
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==recorded['weights_sha256']
    # Enter managed runtime/context scope for this block:
    with np.load(checkpoint,allow_pickle=False) as saved:
        # Create device-backed JAX array `restored`.
        restored=jnp.asarray(saved['weights'])
# Execute `np.testing.assert_array_equal(restored,base)`
np.testing.assert_array_equal(restored,base)
# Compute `frozen` from `np.array(restored,copy=True)`
frozen=np.array(restored,copy=True)
# Run `train` to compute `(sft, sft_history)`.
sft,sft_history=train(restored,adapt_X,hard_targets,steps=200)
# Run `train` to compute `(distilled, distill_history)`.
distilled,distill_history=train(restored,adapt_X,soft_targets,steps=200)
# Execute `np.testing.assert_array_equal(restored,frozen)`
np.testing.assert_array_equal(restored,frozen)

# Step 3 — 3. Compare on a held-out criterion shared by both objectives: The teacher is the known generating rule in this fixture, so...
# Construct `w_check` via `jnp.array([.2,-.3,.1])`
w_check=jnp.array([.2,-.3,.1])
# Convert `host_X` to a host NumPy array for inspection or verification.
host_X=np.asarray(adapt_X,dtype=np.float64)
# Convert `host_w` to a host NumPy array for inspection or verification.
host_w=np.asarray(w_check,dtype=np.float64)
# Perform matrix contraction / projection to compute `host_p`.
host_p=1/(1+np.exp(-(host_X@host_w)))
# Iterate over `targets` to step through the computation:
for targets in (hard_targets,soft_targets):
    # Convert `expected` to a host NumPy array for inspection or verification.
    expected=host_X.T@(host_p-np.asarray(targets))/len(host_X)
    # Differentiate the objective to obtain gradients ``.
    np.testing.assert_allclose(jax.grad(objective)(w_check,adapt_X,targets),expected,rtol=1e-5,atol=1e-6)
# Compute `metrics` from `{}`
metrics={}
# Iterate over `(name, weights)` to step through the computation:
for name,weights in [('pretrained',restored),('supervised',sft),('teacher',distilled)]:
    # Perform matrix contraction / projection to compute `metrics[name]`.
    metrics[name]={'held_cross_entropy':float(objective(weights,held_X,held_prob)),
                   'held_brier':float(jnp.mean((jax.nn.sigmoid(held_X@weights)-held_prob)**2)),
                   'source_cross_entropy':float(objective(weights,source_X,source_targets))}
# Assert invariant `metrics['supervised']['held_cross_entropy']<metrics['pretrained']...` holds
assert metrics['supervised']['held_cross_entropy']<metrics['pretrained']['held_cross_entropy']
# Assert invariant `metrics['teacher']['held_cross_entropy']<metrics['pretrained']['h...` holds
assert metrics['teacher']['held_cross_entropy']<metrics['pretrained']['held_cross_entropy']
# Print the observed values to compare against the expected result.
print(json.dumps(metrics,indent=2))
# Print diagnostic summary of the computed outputs.
print('checkpoint source hash:',source_hash)

# Figure data experiment
# Compute figure data for: Two adaptation objectives evaluated on the same held-out target
# Compute `names` from `["pretrained","supervised","teacher"]`
names=["pretrained","supervised","teacher"]
# Compute `visual_data` from `{"kind":"bar","x":[0,1,2],"labels":names,"xlabel":"m...`
visual_data={"kind":"bar","x":[0,1,2],"labels":names,"xlabel":"model after training stage","ylabel":"mean cross-entropy (nats)","series":[{"label":"held-out target","y":[metrics[n]["held_cross_entropy"] for n in names]},{"label":"source retention","y":[metrics[n]["source_cross_entropy"] for n in names]}]}

# Experiment: Check a stable extreme logit
# Experiment — Check a stable extreme logit: logaddexp evaluates softplus stably; a naive log(1+exp(z)) can...
# Construct `extreme` via `objective(jnp.array([1000.]),jnp.ones((1,1)),jnp.zer...`
extreme=objective(jnp.array([1000.]),jnp.ones((1,1)),jnp.zeros(1))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.isfinite(extreme) and jnp.allclose(extreme,1000.)

# Experiment: Reveal teacher error
# Experiment — Reveal teacher error: Distillation optimizes fidelity to its teacher, which is not...
wrong,_=train(restored,adapt_X,1-soft_targets,steps=200)
# Assert invariant `objective(wrong,held_X,held_prob)>objective(distilled,held_X,held...` holds
assert objective(wrong,held_X,held_prob)>objective(distilled,held_X,held_prob)
# Print the observed values to compare against the expected result.
print("wrong-teacher held loss:",float(objective(wrong,held_X,held_prob)))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Adapt for zero steps and verify that predictions exactly match the...
unchanged,_=train(restored,adapt_X,hard_targets,steps=0)
# Execute `np.testing.assert_array_equal(unchanged,restored)`
np.testing.assert_array_equal(unchanged,restored)
# Run `train` to compute `(short, _)`.
short,_=train(restored,adapt_X,hard_targets,steps=10)
# Assert invariant `not jnp.array_equal(short,restored)` holds
assert not jnp.array_equal(short,restored)
# Execute `np.testing.assert_array_equal(restored,frozen)`
np.testing.assert_array_equal(restored,frozen)

# Reference practice: Keep source-task evidence
# Keep source-task evidence (Transfer / diagnosis): The two tasks prefer different parameters; the measurement...
retention=[float(objective(w,source_X,source_targets)) for w in (restored,sft,distilled)]
# Assert invariant `retention[1]>retention[0] and retention[2]>retention[0]` holds
assert retention[1]>retention[0] and retention[2]>retention[0]
# Print the observed values to compare against the expected result.
print("source retention losses:",retention)

# Reference practice: Check the loss without autodiff
# Check the loss without autodiff (Transfer / diagnosis): Independent algebra checks normalization and reduction as...
# Compute `probe` from `np.array([-.4,.2,.3])`
probe=np.array([-.4,.2,.3])
# Perform matrix / vector contraction (`@`) to compute `z`.
z=host_X@probe
# Aggregate array values to compute `expected`.
expected=np.mean(np.logaddexp(0,z)-np.asarray(soft_targets)*z)
# Create device-backed JAX array ``.
np.testing.assert_allclose(objective(jnp.asarray(probe),adapt_X,soft_targets),expected,rtol=1e-6)

# Reference practice: Choose under a declared retention constraint
# Choose under a declared retention constraint (Transfer / diagnosis): The gate separates optimization from acceptance.
retention_budget = .3
# Compute `eligible` from `[]`
eligible = []
# Iterate over `name` to step through the computation:
for name in ('supervised', 'teacher'):
    # Compute `delta` from `metrics[name]['source_cross_entropy'] - metrics['pre...`
    delta = metrics[name]['source_cross_entropy'] - metrics['pretrained']['source_cross_entropy']
    # Compute `improved` from `metrics[name]['held_cross_entropy'] < metrics['pretr...`
    improved = metrics[name]['held_cross_entropy'] < metrics['pretrained']['held_cross_entropy']
    # Compute `accepted_candidate` from `improved and delta <= retention_budget`
    accepted_candidate = improved and delta <= retention_budget
    # Print diagnostic summary of the computed outputs.
    print(name, 'source loss increase', delta, 'passes declared gate', accepted_candidate)
    # Branch on condition `accepted_candidate`:
    if accepted_candidate: eligible.append(name)
# Run `min` to compute `selected`.
selected = min(eligible, key=lambda n: metrics[n]['held_cross_entropy']) if eligible else 'pretrained'
# Assert invariant `selected == 'pretrained' or selected in eligible` holds
assert selected == 'pretrained' or selected in eligible
# Assert invariant `all(metrics[n]['source_cross_entropy']-metrics['pretrained']['sou...` holds
assert all(metrics[n]['source_cross_entropy']-metrics['pretrained']['source_cross_entropy'] <= retention_budget for n in eligible)
# Print the observed values to compare against the expected result.
print('Candidate selected under the declared fixture budget:', selected)
print("PASS: deployment-02")
