# Save model, optimizer, and random state

Phase 06: Data & checkpoint recovery · about 100 minutes · CPU

## What you will be able to do

- Checkpoint complete training state with Orbax and replay the next update.
- Explain Adam memory and continuation-key requirements.
- Restore known shapes/structure and validate version/data/configuration identity.
- Distinguish tested local synchronous saving from production durability.

## The problem

A saved file is useful only if it contains what you need to continue. We’ll train a tiny regression model, save its weights, optimizer state, random key and step with Orbax, and check the next update after restoring. Then we’ll leave out pieces on purpose to see why they matter.

## The idea

A training checkpoint needs enough state to reproduce the next update. Weights alone do not capture optimizer memory or random-state ownership. Think of the checkpoint as the complete input to a future state transition.

## Save the inputs to the next update

Two runs with identical weights but different Adam moments can receive the same gradient and take different next steps. Different random keys can change augmentation or dropout before the gradient is computed.

The existing diagram merges state components into one snapshot. Its arrows mean membership in the snapshot, not a sequence in which parameters must be computed before moments or randomness. Every component must agree on the completed-step boundary.

Loading bytes successfully checks serialization. Test continuation by restoring into a fresh process and comparing the next transition with an uninterrupted run. Add iterator state through the data-recovery procedure that follows this lesson.

$$
\text{sha256}(\text{serialize}(\mathcal{C}_t)) = \text{sha256}(\text{deserialize}(\text{path}))
$$

### Pause and reason

What is stronger evidence than matching loaded weights?

<details><summary>Compare your reasoning</summary>

Match the next example identities, random state, optimizer update and resulting parameters. This tests the checkpoint's purpose rather than only its stored parameter values.

</details>

## Inventory everything the next step reads

step reads parameters, Adam memory and key_data, then consumes $x$ and $y$. It returns new parameters, new optimizer state, the continuation key and an incremented step count. Adam includes first and second gradient moments plus a count for bias correction. Reinitializing it from saved parameters discards those moments.

The noisy targets deliberately make randomness matter, so a key reset cannot pass invisibly. We serialize key_data as uint32 words and explicitly reconstruct a threefry2x32 key. Raw words alone do not identify every possible PRNG implementation; the contract records that choice. The learning rate and Adam identity are recorded because optimizer state does not encode every update rule.

```text
params + Adam moments/count + continuation key + step
                 + next batch → next update
```

## Save to a real local checkpoint directory

StandardCheckpointer may perform asynchronous work internally. We call wait_until_finished immediately after save and before restore. save writes an actual Orbax checkpoint under an absolute path in a TemporaryDirectory, then restore reads it back. Calling save alone is not our completion claim. No restore is accepted just because the directory exists: we compare the complete state and its next transition. All files are removed when the temporary workspace is cleaned up; copy a learner-run checkpoint to your chosen evidence folder if you want to keep it.

This example does not implement a cloud retention policy, an asynchronous manager, multi-process synchronization or protection against every storage failure. A successful synchronous local save demonstrates the tested library path and numerical replay. It is not evidence of production durability under a machine or disk failure.

## Restore into an explicit state template

The target supplies a known model dictionary and the Adam namedtuple/pytree structure initialized by the same optimizer definition. That prevents relying on an untyped collection of arrays as if it were a complete model. We then compare pytree structure and every leaf, including the continuation key and count.

The compatibility contract is encoded as JSON bytes in a fixed-size uint8 buffer with a separate length. This keeps the restore target shapes known without pickling Python objects. The 4096-byte capacity is a tutorial simplification: oversized metadata fails explicitly. For a different model shape or optimizer, define a migration deliberately rather than claiming a generic restore.

## Version and data checks belong to the application

The payload includes schema version, package versions, learning rate, optimizer identity, key implementation and a digest of this exact input/target pair. validate_contract rejects any mismatch before continuing training. A checksum can identify changed data; it does not repair it or authenticate a hostile checkpoint.

The strongest local recovery check is the next loss and full updated state against an uninterrupted transition at the same boundary. We use float32 and allclose with `rtol=1e-6`, `atol=1e-7` for numerical leaves. Keys and counters are integer state whose expected values should remain identical. Changing hardware, reductions or library versions needs new validation; this lesson establishes only the tested CPU environment.

## Prepare the data and state

Create main.py in your course workspace. Use the tested course environment and add this block first.

```python
# Step 1 — Prepare the data and state: This setup makes the dataset and software assumptions explicit.
# Import jax for this computation.
import jax
# Update state in place with the new values.
jax.config.update("jax_platforms","cpu")
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tempfile
import json
import hashlib
from pathlib import Path
# Configure or step the Optax optimizer state (`tx`).
tx = optax.adam(0.05)
# Function `initial_state()` implementing this stage's computation:
def initial_state():
    # Construct `params` via `{"weight":jnp.array(0.,jnp.float32),"bias":jnp.array...`
    params = {"weight":jnp.array(0.,jnp.float32),"bias":jnp.array(0.,jnp.float32)}
    # Return `{'params': params, 'optimizer': tx.init(params), 'key_data': jax.random.key_data(jax.random.key(7, impl='threefry2x32')), 'step': jnp.array(0, jnp.int32)}` to the caller.
    return {"params":params,"optimizer":tx.init(params),
            "key_data":jax.random.key_data(jax.random.key(7,impl="threefry2x32")),
            "step":jnp.array(0,jnp.int32)}
# Function `pack_bytes(raw, capacity)` implementing this stage's computation:
def pack_bytes(raw,capacity=4096):
    # Guard input contract (`len(raw) > capacity`) and fail fast if violated.
    if len(raw)>capacity:
        raise ValueError("Snapshot exceeds tutorial fixed-byte capacity")
    # Allocate initialized array `buffer` with the specified shape and dtype.
    buffer = np.zeros(capacity,dtype=np.uint8)
    # Run `np.frombuffer` to compute `buffer[:len(raw)]`.
    buffer[:len(raw)] = np.frombuffer(raw,dtype=np.uint8)
    # Return `{'buffer': buffer, 'length': np.array(len(raw), dtype=np.int32)}` to the caller.
    return {"buffer":buffer,"length":np.array(len(raw),dtype=np.int32)}
# Function `unpack_bytes(packed)` implementing this stage's computation:
def unpack_bytes(packed):
    # Evaluate `packed['length']` and convert the result into Python scalar/collection `length`.
    length = int(packed["length"])
    # Guard input contract (`not 0 <= length <= len(packed['buffer'])`) and fail fast if violated.
    if not 0 <= length <= len(packed["buffer"]):
        raise ValueError("Invalid snapshot length")
    # Return `np.asarray(packed['buffer'], dtype=np.uint8)[:length].tobytes()` to the caller.
    return np.asarray(packed["buffer"],dtype=np.uint8)[:length].tobytes()
# Function `same_tree(left, right)` implementing this stage's computation:
def same_tree(left,right):
    # Assert invariant `jax.tree.structure(left)==jax.tree.structure(right)` holds
    assert jax.tree.structure(left)==jax.tree.structure(right)
    # Iterate over `(a, b)` to step through the computation:
    for a,b in zip(jax.tree.leaves(left),jax.tree.leaves(right)):
        # Convert `(host_a, host_b)` to a host NumPy array for inspection or verification.
        host_a,host_b = np.asarray(a),np.asarray(b)
        # Check tensor shape invariant: `host_a.shape==host_b.shape and host_a.dtype==host_b.dtype`
        assert host_a.shape==host_b.shape and host_a.dtype==host_b.dtype
        # Branch on condition `np.issubdtype(host_a.dtype, np.integer)`:
        if np.issubdtype(host_a.dtype,np.integer):
            np.testing.assert_array_equal(host_a,host_b)
        else:
            np.testing.assert_allclose(host_a,host_b,rtol=1e-6,atol=1e-7)
# Define `step(state, x, y)` to evaluate the objective and its automatic derivatives:
def step(state,x,y):
    # Sample deterministic random values into `key` using an explicit PRNG key.
    key = jax.random.wrap_key_data(state["key_data"],impl="threefry2x32")
    # Create or split explicit PRNG key(s) (`(next_key, sample_key)`) for reproducible randomness.
    next_key,sample_key = jax.random.split(key)
    # Sample deterministic random values into `target_noise` using an explicit PRNG key.
    target_noise = 0.01*jax.random.normal(sample_key,y.shape)
    # Function `objective(params)` implementing this stage's computation:
    def objective(params):
        # Compute `residual` from `params["weight"]*x+params["bias"]-(y+target_noise)`
        residual = params["weight"]*x+params["bias"]-(y+target_noise)
        # Return `jnp.mean(residual ** 2)` to the caller.
        return jnp.mean(residual**2)
    # Evaluate both scalar loss and parameter gradients in one pass (`(value, gradient)`).
    value,gradient = jax.value_and_grad(objective)(state["params"])
    # Run `tx.update` to compute `(updates, opt_state)`.
    updates,opt_state = tx.update(gradient,state["optimizer"],state["params"])
    # Initialize explicit deterministic PRNG key `next_state`.
    next_state = {"params":optax.apply_updates(state["params"],updates),
                  "optimizer":opt_state,"key_data":jax.random.key_data(next_key),
                  "step":state["step"]+1}
    # Return `(next_state, value)` to the caller.
    return next_state,value
# Function `package_versions()` implementing this stage's computation:
def package_versions():
    # Import importlib.metadata for this computation.
    import importlib.metadata
    # Return `{name: importlib.metadata.version(name) for name in ('jax', 'numpy', 'optax', 'orbax-checkpoint')}` to the caller.
    return {name:importlib.metadata.version(name) for name in ("jax","numpy","optax","orbax-checkpoint")}
# Function `encode_contract(contract)` implementing this stage's computation:
def encode_contract(contract):
    # Return `pack_bytes(json.dumps(contract, sort_keys=True).encode())` to the caller.
    return pack_bytes(json.dumps(contract,sort_keys=True).encode())
# Function `validate_contract(saved, expected)` implementing this stage's computation:
def validate_contract(saved,expected):
    # Read or serialize artifact data on disk (`actual`).
    actual = json.loads(unpack_bytes(saved))
    # Guard input contract (`actual != expected`) and fail fast if violated.
    if actual != expected:
        raise ValueError("Checkpoint contract differs: verify dataset, optimizer configuration, key implementation and package versions")
# Run `tempfile.TemporaryDirectory` to compute `workspace`.
workspace = tempfile.TemporaryDirectory()
# Read or serialize artifact data on disk (`root`).
root = Path(workspace.name)
```

This setup makes the dataset and software assumptions explicit. No external dataset or accelerator is required.

## Build the pipeline or state transition

Append this block to the same file; follow the named state objects through each function.

```python
# Step 2 — Build the pipeline or state transition: The function boundaries expose which inputs determine the next...
# Construct `x` via `jnp.array([0.,0.5,1.],dtype=jnp.float32)`
x = jnp.array([0.,0.5,1.],dtype=jnp.float32)
# Compute `y` from `2*x-1`
y = 2*x-1
# Convert `contract` to a host NumPy array for inspection or verification.
contract = {"schema":1,"learning_rate":0.05,"optimizer":"adam",
            "key_impl":"threefry2x32","versions":package_versions(),
            "data_sha256":hashlib.sha256(np.asarray(x).tobytes()+np.asarray(y).tobytes()).hexdigest()}
# Run `initial_state` to compute `state`.
state = initial_state()
# Repeat the update loop over `range(3)` steps:
for _ in range(3):
    # Run `step` to compute `(state, _)`.
    state,_ = step(state,x,y)
# Compute `checkpoint_path` from `root/"step_3"`
checkpoint_path = root/"step_3"
# Compute `payload` from `{"training":state,"contract":encode_contract(contract)}`
payload = {"training":state,"contract":encode_contract(contract)}
# Enter `ocp.StandardCheckpointer()` context block:
with ocp.StandardCheckpointer() as checkpointer:
    # Run `checkpointer.save` to perform the next check or state transition.
    checkpointer.save(checkpoint_path,payload)
    # Run `checkpointer.wait_until_finished` to perform the next check or state transition.
    checkpointer.wait_until_finished()
    # Run `checkpointer.restore` to compute `restored`.
    restored = checkpointer.restore(checkpoint_path,target={"training":initial_state(),"contract":encode_contract(contract)})
# Run `validate_contract` to perform the next check or state transition.
validate_contract(restored["contract"],contract)
```

The function boundaries expose which inputs determine the next output and which state must be preserved.

## Run the comparison

Append the checks, save main.py and run python main.py. Predict what should agree before running.

```python
# Step 3 — Run the comparison: The comparison checks the next behavior, not merely whether a save...
same_tree(state,restored["training"])
# Run `step` to compute `(expected_next, expected_loss)`.
expected_next,expected_loss = step(state,x,y)
# Run `step` to compute `(restored_next, restored_loss)`.
restored_next,restored_loss = step(restored["training"],x,y)
# Run `same_tree` to perform the next check or state transition.
same_tree(expected_next,restored_next)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(expected_loss),np.asarray(restored_loss),rtol=1e-6)
# Assert invariant `int(restored_next["step"])==4` holds
assert int(restored_next["step"])==4
# Print the observed values to compare against the expected result.
print("Restored step:",int(restored["training"]["step"]))
# Print diagnostic summary of the computed outputs.
print("Optimizer count:",int(restored["training"]["optimizer"][0].count))
# Print diagnostic summary of the computed outputs.
print("Next update and loss agree; checkpoint exists:",checkpoint_path.exists())
```

The comparison checks the next behavior, not merely whether a save call succeeded.

## Run the example

```python
# Step 1 — Prepare the data and state: This setup makes the dataset and software assumptions explicit.
# Import jax for this computation.
import jax
# Update state in place with the new values.
jax.config.update("jax_platforms","cpu")
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tempfile
import json
import hashlib
from pathlib import Path
# Configure or step the Optax optimizer state (`tx`).
tx = optax.adam(0.05)
# Function `initial_state()` implementing this stage's computation:
def initial_state():
    # Construct `params` via `{"weight":jnp.array(0.,jnp.float32),"bias":jnp.array...`
    params = {"weight":jnp.array(0.,jnp.float32),"bias":jnp.array(0.,jnp.float32)}
    # Return `{'params': params, 'optimizer': tx.init(params), 'key_data': jax.random.key_data(jax.random.key(7, impl='threefry2x32')), 'step': jnp.array(0, jnp.int32)}` to the caller.
    return {"params":params,"optimizer":tx.init(params),
            "key_data":jax.random.key_data(jax.random.key(7,impl="threefry2x32")),
            "step":jnp.array(0,jnp.int32)}
# Function `pack_bytes(raw, capacity)` implementing this stage's computation:
def pack_bytes(raw,capacity=4096):
    # Guard input contract (`len(raw) > capacity`) and fail fast if violated.
    if len(raw)>capacity:
        raise ValueError("Snapshot exceeds tutorial fixed-byte capacity")
    # Allocate initialized array `buffer` with the specified shape and dtype.
    buffer = np.zeros(capacity,dtype=np.uint8)
    # Run `np.frombuffer` to compute `buffer[:len(raw)]`.
    buffer[:len(raw)] = np.frombuffer(raw,dtype=np.uint8)
    # Return `{'buffer': buffer, 'length': np.array(len(raw), dtype=np.int32)}` to the caller.
    return {"buffer":buffer,"length":np.array(len(raw),dtype=np.int32)}
# Function `unpack_bytes(packed)` implementing this stage's computation:
def unpack_bytes(packed):
    # Evaluate `packed['length']` and convert the result into Python scalar/collection `length`.
    length = int(packed["length"])
    # Guard input contract (`not 0 <= length <= len(packed['buffer'])`) and fail fast if violated.
    if not 0 <= length <= len(packed["buffer"]):
        raise ValueError("Invalid snapshot length")
    # Return `np.asarray(packed['buffer'], dtype=np.uint8)[:length].tobytes()` to the caller.
    return np.asarray(packed["buffer"],dtype=np.uint8)[:length].tobytes()
# Function `same_tree(left, right)` implementing this stage's computation:
def same_tree(left,right):
    # Assert invariant `jax.tree.structure(left)==jax.tree.structure(right)` holds
    assert jax.tree.structure(left)==jax.tree.structure(right)
    # Iterate over `(a, b)` to step through the computation:
    for a,b in zip(jax.tree.leaves(left),jax.tree.leaves(right)):
        # Convert `(host_a, host_b)` to a host NumPy array for inspection or verification.
        host_a,host_b = np.asarray(a),np.asarray(b)
        # Check tensor shape invariant: `host_a.shape==host_b.shape and host_a.dtype==host_b.dtype`
        assert host_a.shape==host_b.shape and host_a.dtype==host_b.dtype
        # Branch on condition `np.issubdtype(host_a.dtype, np.integer)`:
        if np.issubdtype(host_a.dtype,np.integer):
            np.testing.assert_array_equal(host_a,host_b)
        else:
            np.testing.assert_allclose(host_a,host_b,rtol=1e-6,atol=1e-7)
# Define `step(state, x, y)` to evaluate the objective and its automatic derivatives:
def step(state,x,y):
    # Sample deterministic random values into `key` using an explicit PRNG key.
    key = jax.random.wrap_key_data(state["key_data"],impl="threefry2x32")
    # Create or split explicit PRNG key(s) (`(next_key, sample_key)`) for reproducible randomness.
    next_key,sample_key = jax.random.split(key)
    # Sample deterministic random values into `target_noise` using an explicit PRNG key.
    target_noise = 0.01*jax.random.normal(sample_key,y.shape)
    # Function `objective(params)` implementing this stage's computation:
    def objective(params):
        # Compute `residual` from `params["weight"]*x+params["bias"]-(y+target_noise)`
        residual = params["weight"]*x+params["bias"]-(y+target_noise)
        # Return `jnp.mean(residual ** 2)` to the caller.
        return jnp.mean(residual**2)
    # Evaluate both scalar loss and parameter gradients in one pass (`(value, gradient)`).
    value,gradient = jax.value_and_grad(objective)(state["params"])
    # Run `tx.update` to compute `(updates, opt_state)`.
    updates,opt_state = tx.update(gradient,state["optimizer"],state["params"])
    # Initialize explicit deterministic PRNG key `next_state`.
    next_state = {"params":optax.apply_updates(state["params"],updates),
                  "optimizer":opt_state,"key_data":jax.random.key_data(next_key),
                  "step":state["step"]+1}
    # Return `(next_state, value)` to the caller.
    return next_state,value
# Function `package_versions()` implementing this stage's computation:
def package_versions():
    # Import importlib.metadata for this computation.
    import importlib.metadata
    # Return `{name: importlib.metadata.version(name) for name in ('jax', 'numpy', 'optax', 'orbax-checkpoint')}` to the caller.
    return {name:importlib.metadata.version(name) for name in ("jax","numpy","optax","orbax-checkpoint")}
# Function `encode_contract(contract)` implementing this stage's computation:
def encode_contract(contract):
    # Return `pack_bytes(json.dumps(contract, sort_keys=True).encode())` to the caller.
    return pack_bytes(json.dumps(contract,sort_keys=True).encode())
# Function `validate_contract(saved, expected)` implementing this stage's computation:
def validate_contract(saved,expected):
    # Read or serialize artifact data on disk (`actual`).
    actual = json.loads(unpack_bytes(saved))
    # Guard input contract (`actual != expected`) and fail fast if violated.
    if actual != expected:
        raise ValueError("Checkpoint contract differs: verify dataset, optimizer configuration, key implementation and package versions")
# Run `tempfile.TemporaryDirectory` to compute `workspace`.
workspace = tempfile.TemporaryDirectory()
# Read or serialize artifact data on disk (`root`).
root = Path(workspace.name)

# Step 2 — Build the pipeline or state transition: The function boundaries expose which inputs determine the next...
# Construct `x` via `jnp.array([0.,0.5,1.],dtype=jnp.float32)`
x = jnp.array([0.,0.5,1.],dtype=jnp.float32)
# Compute `y` from `2*x-1`
y = 2*x-1
# Convert `contract` to a host NumPy array for inspection or verification.
contract = {"schema":1,"learning_rate":0.05,"optimizer":"adam",
            "key_impl":"threefry2x32","versions":package_versions(),
            "data_sha256":hashlib.sha256(np.asarray(x).tobytes()+np.asarray(y).tobytes()).hexdigest()}
# Run `initial_state` to compute `state`.
state = initial_state()
# Repeat the update loop over `range(3)` steps:
for _ in range(3):
    # Run `step` to compute `(state, _)`.
    state,_ = step(state,x,y)
# Compute `checkpoint_path` from `root/"step_3"`
checkpoint_path = root/"step_3"
# Compute `payload` from `{"training":state,"contract":encode_contract(contract)}`
payload = {"training":state,"contract":encode_contract(contract)}
# Enter `ocp.StandardCheckpointer()` context block:
with ocp.StandardCheckpointer() as checkpointer:
    # Run `checkpointer.save` to perform the next check or state transition.
    checkpointer.save(checkpoint_path,payload)
    # Run `checkpointer.wait_until_finished` to perform the next check or state transition.
    checkpointer.wait_until_finished()
    # Run `checkpointer.restore` to compute `restored`.
    restored = checkpointer.restore(checkpoint_path,target={"training":initial_state(),"contract":encode_contract(contract)})
# Run `validate_contract` to perform the next check or state transition.
validate_contract(restored["contract"],contract)

# Step 3 — Run the comparison: The comparison checks the next behavior, not merely whether a save...
same_tree(state,restored["training"])
# Run `step` to compute `(expected_next, expected_loss)`.
expected_next,expected_loss = step(state,x,y)
# Run `step` to compute `(restored_next, restored_loss)`.
restored_next,restored_loss = step(restored["training"],x,y)
# Run `same_tree` to perform the next check or state transition.
same_tree(expected_next,restored_next)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(np.asarray(expected_loss),np.asarray(restored_loss),rtol=1e-6)
# Assert invariant `int(restored_next["step"])==4` holds
assert int(restored_next["step"])==4
# Print the observed values to compare against the expected result.
print("Restored step:",int(restored["training"]["step"]))
# Print diagnostic summary of the computed outputs.
print("Optimizer count:",int(restored["training"]["optimizer"][0].count))
# Print diagnostic summary of the computed outputs.
print("Next update and loss agree; checkpoint exists:",checkpoint_path.exists())
```

Expected: Saved/restored training step $3$ and Adam count $3$; next step $4$, loss and full state agree within stated tolerances. Actual checkpoint path is temporary.

## Parameter parity: uninterrupted step vs. Orbax-restored step

**Predict:** Predict the maximum absolute difference between the uninterrupted next state and the Orbax-restored next state.

![Parameter parity: uninterrupted step vs. Orbax-restored step](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

Adjacent bars for each parameter leaf have identical L2 norms between the uninterrupted run and the Orbax-restored run.

### Connect it to the computation

Verifying exact leaf equality after deserialization guarantees that the restored checkpoint reproduces the exact next training step.

```python
# Compare parameter leaf norms between `expected_next` and `restored_next`:
saved_norms = [float(jnp.linalg.norm(a)) for a in jax.tree.leaves(expected_next['params'])]
restored_norms = [float(jnp.linalg.norm(a)) for a in jax.tree.leaves(restored_next['params'])]
visual_data = {'kind': 'bar', 'x': list(range(len(saved_norms))), 'labels': [f'leaf {i}' for i in range(len(saved_norms))], 'xlabel': 'checkpoint parameter leaf index', 'ylabel': 'L2 norm', 'series': [{'label': 'uninterrupted step', 'y': saved_norms}, {'label': 'restored from Orbax', 'y': restored_norms}]}
```

## Recorded reference execution

CPU run: 2026-10-09T14:07:57.216589+00:00. JAX 0.9.2.

```text
Restored step: 3
Optimizer count: 3
Next update and loss agree; checkpoint exists: True
Restored step: 3
Optimizer count: 3
Next update and loss agree; checkpoint exists: True
Model-only checkpoint produces a different Adam update
Reset key changes the noisy objective and continuation stream
Full restore Adam count: 4; reset memory count: 1; next parameters differ.
Expected changed configuration rejection
Expected exact key-data mismatch
PASS: recovery-03

```

## Model-only restoration loses Adam history

**Predict before running:** With the same parameters, key and input, will a freshly initialized optimizer reproduce the next update?

```python
# Experiment — Model-only restoration loses Adam history: Saved weights give the same pre-update predictions but not...
model_only = dict(restored["training"])
# Run `tx.init` to compute `model_only['optimizer']`.
model_only["optimizer"] = tx.init(model_only["params"])
# Run `step` to compute `(wrong_next, _)`.
wrong_next,_ = step(model_only,x,y)
# Assert that `any(not np.allclose(np.asarray(a),np.asarray(b),rtol=1e-6,atol=1e-7) for a,b in zip(jax.tree`.
assert any(not np.allclose(np.asarray(a),np.asarray(b),rtol=1e-6,atol=1e-7) for a,b in zip(jax.tree.leaves(wrong_next["params"]),jax.tree.leaves(expected_next["params"])))
# Assert invariant `int(wrong_next["optimizer"][0].count)==1` holds
assert int(wrong_next["optimizer"][0].count)==1
# Print the observed values to compare against the expected result.
print("Model-only checkpoint produces a different Adam update")
```

**Expected:** Fresh optimizer count becomes $1$ and parameters differ from the full-state continuation.

Saved weights give the same pre-update predictions but not necessarily the same optimizer transition.

## Reset the random stream

**Predict before running:** What changes if parameters and Adam memory restore correctly but the key resets to seed seven?

```python
# Experiment — Reset the random stream: A seed identifies the initial stream; a saved continuation key...
reset_random = dict(restored["training"])
# Run `initial_state` to compute `reset_random['key_data']`.
reset_random["key_data"] = initial_state()["key_data"]
# Run `step` to compute `(reset_next, reset_loss)`.
reset_next,reset_loss = step(reset_random,x,y)
# Assert that `not np.isclose(float(reset_loss),float(expected_loss),rtol=1e-6,atol=1e-7)`.
assert not np.isclose(float(reset_loss),float(expected_loss),rtol=1e-6,atol=1e-7)
# Assert invariant `not np.array_equal(np.asarray(reset_next["key_data"])` holds
assert not np.array_equal(np.asarray(reset_next["key_data"]),np.asarray(expected_next["key_data"]))
# Print the observed values to compare against the expected result.
print("Reset key changes the noisy objective and continuation stream")
```

**Expected:** The noisy objective and continuation key differ.

A seed identifies the initial stream; a saved continuation key identifies the current boundary.

## Make it yours

Advance to step five, save to a new directory, restore and verify step six against uninterrupted execution.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `step(...)` — Call `step` with your updated parameters or inputs from this lesson's workspace.
- `ocp.StandardCheckpointer(...)` — Call `ocp.StandardCheckpointer` with your updated parameters or inputs from this lesson's workspace.
- `assert condition` — Verify that the observed output shape, status, or numerical value satisfies the contract.

**Step-by-step implementation plan:**
1. Repeat the update loop over `range(2)` steps:
2. Run `step` to compute `(later, _)`.
3. Enter `ocp.StandardCheckpointer()` context block:
4. Read or serialize artifact data on disk (`later_path`).
5. Run `cp.save` to perform the next check or state transition.

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Advance to step five, save to a new directory, restore and verify step...
later = ...  # TODO: compute later
# Repeat the update loop over `range(2)` steps:
# Run `step` to compute `(later, _)`.
for _ in range(2):later,_=step(later,x,y)
# Enter `ocp.StandardCheckpointer()` context block:
with ocp.StandardCheckpointer() as cp:
    # Read or serialize artifact data on disk (`later_path`).
    later_path = Path(...)  # TODO: compute later_path
    # Run `cp.save` to perform the next check or state transition.
    cp.save(later_path,{"training":later,"contract":encode_contract(contract)})
    # Run `cp.wait_until_finished` to perform the next check or state transition.
    cp.wait_until_finished()
    # Run `cp.restore` to compute `recovered`.
    recovered = cp.restore(...)  # TODO: compute recovered
# Run `validate_contract` to perform the next check or state transition.
validate_contract(recovered["contract"],contract)
# Run `step` to compute `(left, _)`.
# Run `step` to compute `(right, _)`.
left,_ = step(...)  # TODO: compute left,_
right,_ = step(...)  # TODO: compute right,_
# Run `same_tree` to perform the next check or state transition.
same_tree(left,right)
# Assert invariant `int(right["step"])==6` holds
assert int(right["step"])  # TODO: complete assertion check
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Advance to step five, save to a new directory, restore and verify step...
later = state
# Repeat the update loop over `range(2)` steps:
# Run `step` to compute `(later, _)`.
for _ in range(2):later,_=step(later,x,y)
# Enter `ocp.StandardCheckpointer()` context block:
with ocp.StandardCheckpointer() as cp:
    # Read or serialize artifact data on disk (`later_path`).
    later_path = Path(tempfile.mkdtemp(dir=root))/"step_5"
    # Run `cp.save` to perform the next check or state transition.
    cp.save(later_path,{"training":later,"contract":encode_contract(contract)})
    # Run `cp.wait_until_finished` to perform the next check or state transition.
    cp.wait_until_finished()
    # Run `cp.restore` to compute `recovered`.
    recovered = cp.restore(later_path,target={"training":initial_state(),"contract":encode_contract(contract)})
# Run `validate_contract` to perform the next check or state transition.
validate_contract(recovered["contract"],contract)
# Run `step` to compute `(left, _)`.
# Run `step` to compute `(right, _)`.
left,_=step(later,x,y)
right,_=step(recovered["training"],x,y)
# Run `same_tree` to perform the next check or state transition.
same_tree(left,right)
# Assert invariant `int(right["step"])==6` holds
assert int(right["step"])==6
```

</details>

## Show what weights-only recovery loses

**Transfer**

Keep the restored parameters but reset Adam’s memory. Compare the next update with the full-state restore. Keep data and random state identical so the optimizer memory is the only changed factor.

<details><summary>Hint</summary>

Create a new mapping and replace only its optimizer entry.

</details>

### How to write: Show what weights-only recovery loses — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Evaluate `full, optimizer=tx.init(full['params'])` and convert the result into Python scalar/collection `weights_only`.
2. Run `step` to compute `(full_next, _)`.
3. Run `step` to compute `(reset_next, _)`.
4. Assert invariant `int(full_next['optimizer'][0].count)==4` holds
5. Assert invariant `int(reset_next['optimizer'][0].count)==1` holds

**Starter code scaffold (fill in the TODOs):**

```python
# Show what weights-only recovery loses (Transfer): A checkpoint can return plausible predictions yet fail to...
full = ...  # TODO: compute full
# Evaluate `full, optimizer=tx.init(full['params'])` and convert the result into Python scalar/collection `weights_only`.
weights_only = dict(...)  # TODO: compute weights_only
# Run `step` to compute `(full_next, _)`.
full_next,_ = step(...)  # TODO: compute full_next,_
# Run `step` to compute `(reset_next, _)`.
reset_next,_ = step(...)  # TODO: compute reset_next,_
# Assert invariant `int(full_next['optimizer'][0].count)==4` holds
assert int(full_next['optimizer'][0].count)  # TODO: complete assertion check
# Assert invariant `int(reset_next['optimizer'][0].count)==1` holds
assert int(reset_next['optimizer'][0].count)  # TODO: complete assertion check
# Assert that `any(not np.allclose(a,b,rtol=1e-6,atol=1e-7) for a,b in zip(jax.tree.leaves(full_next['param`.
assert any(not np.allclose(a,b,rtol=1e-6,atol=1e-7) for a,b  # TODO: complete assertion check
# Print the observed values to compare against the expected result.
print('Full restore Adam count: 4; reset memory count: 1; next parameters differ.')
```

<details><summary>Reference solution and reasoning</summary>

```python
# Show what weights-only recovery loses (Transfer): A checkpoint can return plausible predictions yet fail to...
full=restored['training']
# Evaluate `full, optimizer=tx.init(full['params'])` and convert the result into Python scalar/collection `weights_only`.
weights_only=dict(full,optimizer=tx.init(full['params']))
# Run `step` to compute `(full_next, _)`.
full_next,_=step(full,x,y)
# Run `step` to compute `(reset_next, _)`.
reset_next,_=step(weights_only,x,y)
# Assert invariant `int(full_next['optimizer'][0].count)==4` holds
assert int(full_next['optimizer'][0].count)==4
# Assert invariant `int(reset_next['optimizer'][0].count)==1` holds
assert int(reset_next['optimizer'][0].count)==1
# Assert that `any(not np.allclose(a,b,rtol=1e-6,atol=1e-7) for a,b in zip(jax.tree.leaves(full_next['param`.
assert any(not np.allclose(a,b,rtol=1e-6,atol=1e-7) for a,b in zip(jax.tree.leaves(full_next['params']),jax.tree.leaves(reset_next['params'])))
# Print the observed values to compare against the expected result.
print('Full restore Adam count: 4; reset memory count: 1; next parameters differ.')
```

A checkpoint can return plausible predictions yet fail to resume the original training trajectory. Adam’s moments and count affect the next update even when the weights and batch match.

</details>

## Reject an incompatible update configuration

**Challenge**

Attempt to restore the contract under learning rate $0.1$. Show the rejection, then validate the original configuration. Also inspect shape and dtype equality of restored parameter leaves.

<details><summary>Hint</summary>

A compatible pytree can still represent the wrong update rule.

</details>

### How to write: Reject an incompatible update configuration — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `configuration(...)` — Call `configuration` with your updated parameters or inputs from this lesson's workspace.
- `validate_contract(...)` — Call `validate_contract` with your updated parameters or inputs from this lesson's workspace.
- `assert condition` — Verify that the observed output shape, status, or numerical value satisfies the contract.

**Step-by-step implementation plan:**
1. Run the boundary check and catch the expected exception:
2. Run `validate_contract` to perform the next check or state transition.
3. Iterate over `(a, b)` to step through the computation:
4. Check tensor shape invariant: `a.shape==b.shape and a.dtype==b.dtype`
5. Integer PRNG words must be checked exactly, not with relative float tolerance.

**Starter code scaffold (fill in the TODOs):**

```python
# Reject an incompatible update configuration (Challenge): Array restoration does not verify that a learning rate, data...
changed_contract = dict(...)  # TODO: compute changed_contract
# Run the boundary check and catch the expected exception:
try:
    validate_contract(restored["contract"],changed_contract)
except ValueError:
    print("Expected changed configuration rejection")
else:
    raise AssertionError("Expected contract mismatch")
# Run `validate_contract` to perform the next check or state transition.
validate_contract(restored["contract"],contract)
# Iterate over `(a, b)` to step through the computation:
for a,b in zip(jax.tree.leaves(state["params"]),jax.tree.leaves(restored["training"]["params"])):
    # Check tensor shape invariant: `a.shape==b.shape and a.dtype==b.dtype`
    assert a.shape  # TODO: complete assertion check
# Integer PRNG words must be checked exactly, not with relative float tolerance.
changed_key = dict(...)  # TODO: compute changed_key
# Compute `changed_key["key_data"]` from `state["key_data"].at[0].add(np.uint32(1))`
changed_key["key_data"] = ...  # TODO: compute changed_key["key_data"]
# Run the boundary check and catch the expected exception:
try:
    same_tree(state,changed_key)
except AssertionError:
    print("Expected exact key-data mismatch")
else:
    raise AssertionError("Expected changed PRNG word rejection")
```

<details><summary>Reference solution and reasoning</summary>

```python
# Reject an incompatible update configuration (Challenge): Array restoration does not verify that a learning rate, data...
changed_contract = dict(contract,learning_rate=0.1)
# Run the boundary check and catch the expected exception:
try:
    validate_contract(restored["contract"],changed_contract)
except ValueError:
    print("Expected changed configuration rejection")
else:
    raise AssertionError("Expected contract mismatch")
# Run `validate_contract` to perform the next check or state transition.
validate_contract(restored["contract"],contract)
# Iterate over `(a, b)` to step through the computation:
for a,b in zip(jax.tree.leaves(state["params"]),jax.tree.leaves(restored["training"]["params"])):
    # Check tensor shape invariant: `a.shape==b.shape and a.dtype==b.dtype`
    assert a.shape==b.shape and a.dtype==b.dtype
# Integer PRNG words must be checked exactly, not with relative float tolerance.
changed_key = dict(state)
# Compute `changed_key["key_data"]` from `state["key_data"].at[0].add(np.uint32(1))`
changed_key["key_data"] = state["key_data"].at[0].add(np.uint32(1))
# Run the boundary check and catch the expected exception:
try:
    same_tree(state,changed_key)
except AssertionError:
    print("Expected exact key-data mismatch")
else:
    raise AssertionError("Expected changed PRNG word rejection")
```

Array restoration does not verify that a learning rate, data revision or PRNG implementation agrees with the saved experiment. Refuse an incompatible continuation before applying an update.

</details>

## Check your understanding

Which checkpoint supports replaying the next Adam update with stochastic targets?

1. Parameters alone.
2. Parameters, Adam state/count, continuation key, step and compatible data/configuration.
3. A seed plus the last printed loss.

<details><summary>Answer and explanation</summary>

Parameters, Adam state/count, continuation key, step and compatible data/configuration.

The update reads all of those inputs; matching predictions before an update cannot replace preserving transition state.

</details>

## Diagnose the result

Array restoration does not verify that a learning rate, data revision or PRNG implementation agrees with the saved experiment. Refuse an incompatible continuation before applying an update.

## Carry forward

- Inventory everything the next step reads
- Save to a real local checkpoint directory
- Restore into an explicit state template
- Version and data checks belong to the application

## Keep your evidence

Keep Orbax save/wait/restore commands, snapshot step, Adam count/moments, exact PRNG words, numerical next-update replay and the parameter-only restart counterexample.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [Orbax StandardCheckpointer](https://orbax.readthedocs.io/en/latest/api_reference/checkpoint.checkpointers.html)
- [Optax Adam](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adam)
- [JAX random key data](https://docs.jax.dev/en/latest/_autosummary/jax.random.key_data.html)

