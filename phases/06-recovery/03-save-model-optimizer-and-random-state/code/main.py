"""Save model, optimizer, and random state: worked experiments and reference solutions. CPU checks."""

# Prepare the data and state
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

# Build the pipeline or state transition
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

# Run the comparison
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

# Figure data experiment
# Compare parameter leaf norms between `expected_next` and `restored_next`:
saved_norms = [float(jnp.linalg.norm(a)) for a in jax.tree.leaves(expected_next['params'])]
restored_norms = [float(jnp.linalg.norm(a)) for a in jax.tree.leaves(restored_next['params'])]
visual_data = {'kind': 'bar', 'x': list(range(len(saved_norms))), 'labels': [f'leaf {i}' for i in range(len(saved_norms))], 'xlabel': 'checkpoint parameter leaf index', 'ylabel': 'L2 norm', 'series': [{'label': 'uninterrupted step', 'y': saved_norms}, {'label': 'restored from Orbax', 'y': restored_norms}]}

# Experiment: Model-only restoration loses Adam history
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

# Experiment: Reset the random stream
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

# Reference solution. Try the exercise before reading this.
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

# Reference practice: Show what weights-only recovery loses
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

# Reference practice: Reject an incompatible update configuration
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
print("PASS: recovery-03")
