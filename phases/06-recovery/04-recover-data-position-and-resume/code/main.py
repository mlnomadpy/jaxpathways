"""Recover data position and resume: worked experiments and reference solutions. CPU checks."""

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

# Import required JAX, NumPy, and standard-library modules.
import numpy as np
import grain.python as grain
# Module/container class `TutorialSource` encapsulating model parameters and forward pass:
class TutorialSource:
    # Function `__init__(self, count)` implementing this stage's computation:
    def __init__(self,count=12):
        # Compute `self.count` from `count`
        self.count = count
    # Function `__len__(self)` implementing this stage's computation:
    def __len__(self):
        # Return `self.count` to the caller.
        return self.count
    # Function `__getitem__(self, index)` implementing this stage's computation:
    def __getitem__(self,index):
        # Cast or evaluate `value` in explicit floating-point precision.
        value = np.float32(index/10)
        # Return `{'id': np.int64(index), 'x': value, 'y': np.float32(2 * value - 1)}` to the caller.
        return {"id":np.int64(index),"x":value,"y":np.float32(2*value-1)}
    # Function `__repr__(self)` implementing this stage's computation:
    def __repr__(self):
        # Return `f'TutorialSource(v1,n={self.count})'` to the caller.
        return f"TutorialSource(v1,n={self.count})"
# Function `make_loader(seed, batch_size, count, drop_remainder)` implementing this stage's computation:
def make_loader(seed=42,batch_size=4,count=12,drop_remainder=False):
    # Return `grain.DataLoader(data_source=TutorialSource(count), sampler=grain.IndexSampler(num_records=count, num_epochs=2, shuffle=True, seed=seed), operations=[grain.Batch(batch_size, drop_remainder=drop_remainder)], worker_count=0)` to the caller.
    return grain.DataLoader(
        data_source=TutorialSource(count),
        sampler=grain.IndexSampler(num_records=count,num_epochs=2,shuffle=True,seed=seed),
        operations=[grain.Batch(batch_size,drop_remainder=drop_remainder)],
        worker_count=0)

# Build the pipeline or state transition
# Step 2 — Build the pipeline or state transition: The function boundaries expose which inputs determine the next...
def data_contract(seed=42):
    # Run `TutorialSource` to compute `source`.
    source = TutorialSource()
    # Compute `values` from `np.array([[source[i]["x"],source[i]["y"]] for i in r...`
    values = np.array([[source[i]["x"],source[i]["y"]] for i in range(len(source))],dtype=np.float32)
    # Return `{'schema': 1, 'seed': seed, 'batch_size': 4, 'count': 12, 'source': 'TutorialSource(v1,n=12)', 'sha256': hashlib.sha256(values.tobytes()).hexdigest(), 'learning_rate': 0.05, 'optimizer': 'adam', 'key_impl': 'threefry2x32', 'versions': dict(package_versions(), grain=__import__('importlib.metadata', fromlist=['version']).version('grain'))}` to the caller.
    return {"schema":1,"seed":seed,"batch_size":4,"count":12,"source":"TutorialSource(v1,n=12)",
            "sha256":hashlib.sha256(values.tobytes()).hexdigest(),
            "learning_rate":0.05,"optimizer":"adam","key_impl":"threefry2x32",
            "versions":dict(package_versions(),grain=__import__("importlib.metadata",fromlist=["version"]).version("grain"))}
# Function `consume(state, iterator, count)` implementing this stage's computation:
def consume(state,iterator,count):
    # Evaluate `losses` from the current inputs and state.
    # Compute `losses` from `[]`
    losses=[]
    orders=[]
    # Repeat the update loop over `range(count)` steps:
    for _ in range(count):
        # Run `next` to compute `batch`.
        batch = next(iterator)
        # Append the current step result to `orders`.
        orders.append(batch["id"].copy())
        # Create device-backed JAX array `(state, value)`.
        state,value = step(state,jnp.asarray(batch["x"]),jnp.asarray(batch["y"]))
        # Append the current step result to `losses`.
        losses.append(float(value))
    # Return `(state, np.array(losses), orders)` to the caller.
    return state,np.array(losses),orders
# Run `data_contract` to compute `contract`.
contract = data_contract()
# Run `iter` to compute `reference_iterator`.
reference_iterator = iter(make_loader())
# Run `consume` to compute `(reference_state, _, _)`.
reference_state,_,_ = consume(initial_state(),reference_iterator,2)
# Compute `payload` from `{"training":reference_state,"pipeline":pack_bytes(re...`
payload = {"training":reference_state,"pipeline":pack_bytes(reference_iterator.get_state()),"contract":encode_contract(contract)}
# Compute `resume_path` from `root/"completed_step_2"`
resume_path = root/"completed_step_2"
# Enter `ocp.StandardCheckpointer()` context block:
with ocp.StandardCheckpointer() as cp:
    # Run `cp.save` to perform the next check or state transition.
    cp.save(resume_path,payload)
    # Run `cp.wait_until_finished` to perform the next check or state transition.
    cp.wait_until_finished()
    # Run `cp.restore` to compute `restored_payload`.
    restored_payload = cp.restore(resume_path,target={"training":initial_state(),"pipeline":pack_bytes(b""),"contract":encode_contract(contract)})
# Run `validate_contract` to perform the next check or state transition.
validate_contract(restored_payload["contract"],contract)
# Run `iter` to compute `resumed_iterator`.
resumed_iterator = iter(make_loader())
# Run `resumed_iterator.set_state` to perform the next check or state transition.
resumed_iterator.set_state(unpack_bytes(restored_payload["pipeline"]))

# Run the comparison
# Step 3 — Run the comparison: The comparison checks the next behavior, not merely whether a save...
reference_final,reference_losses,reference_ids = consume(reference_state,reference_iterator,4)
# Run `consume` to compute `(resumed_final, resumed_losses, resumed_ids)`.
resumed_final,resumed_losses,resumed_ids = consume(restored_payload["training"],resumed_iterator,4)
# Iterate over `(a, b)` to step through the computation:
for a,b in zip(reference_ids,resumed_ids):np.testing.assert_array_equal(a,b)
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(reference_losses,resumed_losses,rtol=1...`
np.testing.assert_allclose(reference_losses,resumed_losses,rtol=1e-6,atol=1e-7)
# Run `same_tree` to perform the next check or state transition.
same_tree(reference_final,resumed_final)
# Assert invariant `int(resumed_final["step"])==6` holds
assert int(resumed_final["step"])==6
# Print the observed values to compare against the expected result.
print("Restored next four ID batches:",[a.tolist() for a in resumed_ids])
# Print diagnostic summary of the computed outputs.
print("Restored next four losses:",resumed_losses.tolist())
# Print diagnostic summary of the computed outputs.
print("Full state agrees at step six")

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

# Import required JAX, NumPy, and standard-library modules.
import numpy as np
import grain.python as grain
# Module/container class `TutorialSource` encapsulating model parameters and forward pass:
class TutorialSource:
    # Function `__init__(self, count)` implementing this stage's computation:
    def __init__(self,count=12):
        # Compute `self.count` from `count`
        self.count = count
    # Function `__len__(self)` implementing this stage's computation:
    def __len__(self):
        # Return `self.count` to the caller.
        return self.count
    # Function `__getitem__(self, index)` implementing this stage's computation:
    def __getitem__(self,index):
        # Cast or evaluate `value` in explicit floating-point precision.
        value = np.float32(index/10)
        # Return `{'id': np.int64(index), 'x': value, 'y': np.float32(2 * value - 1)}` to the caller.
        return {"id":np.int64(index),"x":value,"y":np.float32(2*value-1)}
    # Function `__repr__(self)` implementing this stage's computation:
    def __repr__(self):
        # Return `f'TutorialSource(v1,n={self.count})'` to the caller.
        return f"TutorialSource(v1,n={self.count})"
# Function `make_loader(seed, batch_size, count, drop_remainder)` implementing this stage's computation:
def make_loader(seed=42,batch_size=4,count=12,drop_remainder=False):
    # Return `grain.DataLoader(data_source=TutorialSource(count), sampler=grain.IndexSampler(num_records=count, num_epochs=2, shuffle=True, seed=seed), operations=[grain.Batch(batch_size, drop_remainder=drop_remainder)], worker_count=0)` to the caller.
    return grain.DataLoader(
        data_source=TutorialSource(count),
        sampler=grain.IndexSampler(num_records=count,num_epochs=2,shuffle=True,seed=seed),
        operations=[grain.Batch(batch_size,drop_remainder=drop_remainder)],
        worker_count=0)

# Step 2 — Build the pipeline or state transition: The function boundaries expose which inputs determine the next...
def data_contract(seed=42):
    # Run `TutorialSource` to compute `source`.
    source = TutorialSource()
    # Compute `values` from `np.array([[source[i]["x"],source[i]["y"]] for i in r...`
    values = np.array([[source[i]["x"],source[i]["y"]] for i in range(len(source))],dtype=np.float32)
    # Return `{'schema': 1, 'seed': seed, 'batch_size': 4, 'count': 12, 'source': 'TutorialSource(v1,n=12)', 'sha256': hashlib.sha256(values.tobytes()).hexdigest(), 'learning_rate': 0.05, 'optimizer': 'adam', 'key_impl': 'threefry2x32', 'versions': dict(package_versions(), grain=__import__('importlib.metadata', fromlist=['version']).version('grain'))}` to the caller.
    return {"schema":1,"seed":seed,"batch_size":4,"count":12,"source":"TutorialSource(v1,n=12)",
            "sha256":hashlib.sha256(values.tobytes()).hexdigest(),
            "learning_rate":0.05,"optimizer":"adam","key_impl":"threefry2x32",
            "versions":dict(package_versions(),grain=__import__("importlib.metadata",fromlist=["version"]).version("grain"))}
# Function `consume(state, iterator, count)` implementing this stage's computation:
def consume(state,iterator,count):
    # Evaluate `losses` from the current inputs and state.
    # Compute `losses` from `[]`
    losses=[]
    orders=[]
    # Repeat the update loop over `range(count)` steps:
    for _ in range(count):
        # Run `next` to compute `batch`.
        batch = next(iterator)
        # Append the current step result to `orders`.
        orders.append(batch["id"].copy())
        # Create device-backed JAX array `(state, value)`.
        state,value = step(state,jnp.asarray(batch["x"]),jnp.asarray(batch["y"]))
        # Append the current step result to `losses`.
        losses.append(float(value))
    # Return `(state, np.array(losses), orders)` to the caller.
    return state,np.array(losses),orders
# Run `data_contract` to compute `contract`.
contract = data_contract()
# Run `iter` to compute `reference_iterator`.
reference_iterator = iter(make_loader())
# Run `consume` to compute `(reference_state, _, _)`.
reference_state,_,_ = consume(initial_state(),reference_iterator,2)
# Compute `payload` from `{"training":reference_state,"pipeline":pack_bytes(re...`
payload = {"training":reference_state,"pipeline":pack_bytes(reference_iterator.get_state()),"contract":encode_contract(contract)}
# Compute `resume_path` from `root/"completed_step_2"`
resume_path = root/"completed_step_2"
# Enter `ocp.StandardCheckpointer()` context block:
with ocp.StandardCheckpointer() as cp:
    # Run `cp.save` to perform the next check or state transition.
    cp.save(resume_path,payload)
    # Run `cp.wait_until_finished` to perform the next check or state transition.
    cp.wait_until_finished()
    # Run `cp.restore` to compute `restored_payload`.
    restored_payload = cp.restore(resume_path,target={"training":initial_state(),"pipeline":pack_bytes(b""),"contract":encode_contract(contract)})
# Run `validate_contract` to perform the next check or state transition.
validate_contract(restored_payload["contract"],contract)
# Run `iter` to compute `resumed_iterator`.
resumed_iterator = iter(make_loader())
# Run `resumed_iterator.set_state` to perform the next check or state transition.
resumed_iterator.set_state(unpack_bytes(restored_payload["pipeline"]))

# Step 3 — Run the comparison: The comparison checks the next behavior, not merely whether a save...
reference_final,reference_losses,reference_ids = consume(reference_state,reference_iterator,4)
# Run `consume` to compute `(resumed_final, resumed_losses, resumed_ids)`.
resumed_final,resumed_losses,resumed_ids = consume(restored_payload["training"],resumed_iterator,4)
# Iterate over `(a, b)` to step through the computation:
for a,b in zip(reference_ids,resumed_ids):np.testing.assert_array_equal(a,b)
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(reference_losses,resumed_losses,rtol=1...`
np.testing.assert_allclose(reference_losses,resumed_losses,rtol=1e-6,atol=1e-7)
# Run `same_tree` to perform the next check or state transition.
same_tree(reference_final,resumed_final)
# Assert invariant `int(resumed_final["step"])==6` holds
assert int(resumed_final["step"])==6
# Print the observed values to compare against the expected result.
print("Restored next four ID batches:",[a.tolist() for a in resumed_ids])
# Print diagnostic summary of the computed outputs.
print("Restored next four losses:",resumed_losses.tolist())
# Print diagnostic summary of the computed outputs.
print("Full state agrees at step six")

# Figure data experiment
# Compare step losses for the 4 post-checkpoint updates (`reference_losses` vs `resumed_losses`):
visual_data = {'kind': 'line', 'x': [3, 4, 5, 6], 'xlabel': 'training step after checkpoint', 'ylabel': 'batch loss', 'series': [{'label': 'uninterrupted reference_losses', 'y': [float(v) for v in reference_losses]}, {'label': 'checkpoint resumed_losses', 'y': [float(v) for v in resumed_losses]}]}

# Experiment: Restart only the reader
# Experiment — Restart only the reader: A model step counter does not automatically seek a data iterator.
fresh = iter(make_loader())
# Run `consume` to compute `(wrong_state, wrong_losses, wrong_ids)`.
wrong_state,wrong_losses,wrong_ids = consume(restored_payload["training"],fresh,1)
# Assert invariant `not np.array_equal(wrong_ids[0],reference_ids[0])` holds
assert not np.array_equal(wrong_ids[0],reference_ids[0])
# Check numerical equivalence within tolerance: `not np.isclose(wrong_losses[0],reference_losses[0],rtol=1e-6,atol...`
assert not np.isclose(wrong_losses[0],reference_losses[0],rtol=1e-6,atol=1e-7)
# Print the observed values to compare against the expected result.
print("Missing iterator restore repeats earlier IDs and changes the next loss")

# Experiment: Restore only the reader
# Experiment — Restore only the reader: Input replay and model-state replay are complementary...
reader_only = iter(make_loader())
reader_only.set_state(unpack_bytes(restored_payload["pipeline"]))
# Run `consume` to compute `(wrong_model, wrong_model_losses, correct_ids)`.
wrong_model,wrong_model_losses,correct_ids = consume(initial_state(),reader_only,1)
# Execute `np.testing.assert_array_equal(correct_ids[0],reference_ids[0`
np.testing.assert_array_equal(correct_ids[0],reference_ids[0])
# Verify that the numerical values match the expected reference within tolerance.
assert not np.isclose(wrong_model_losses[0],reference_losses[0],rtol=1e-6,atol=1e-7)
# Assert invariant `int(wrong_model["step"])==1` holds
assert int(wrong_model["step"])==1
# Print the observed values to compare against the expected result.
print("Correct IDs alone do not restore the training transition")

# Reference solution. Try the exercise before reading this.
# Exercise solution: Repeat the protocol after one batch and compare the next five batches...
other_iterator = iter(make_loader())
# Run `consume` to compute `(other_state, _, _)`.
other_state,_,_ = consume(initial_state(),other_iterator,1)
# Compute `other_payload` from `{"training":other_state,"pipeline":pack_bytes(other_...`
other_payload = {"training":other_state,"pipeline":pack_bytes(other_iterator.get_state()),"contract":encode_contract(contract)}
# Enter `ocp.StandardCheckpointer()` context block:
with ocp.StandardCheckpointer() as cp:
    # Read or serialize artifact data on disk (`other_path`).
    other_path = Path(tempfile.mkdtemp(dir=root))/"completed_step_1"
    # Run `cp.save` to perform the next check or state transition.
    cp.save(other_path,other_payload)
    # Run `cp.wait_until_finished` to perform the next check or state transition.
    cp.wait_until_finished()
    # Run `cp.restore` to compute `other_saved`.
    other_saved = cp.restore(other_path,target={"training":initial_state(),"pipeline":pack_bytes(b""),"contract":encode_contract(contract)})
# Run `iter` to compute `other_resumed`.
# Execute the next step of the computation.
other_resumed = iter(make_loader())
other_resumed.set_state(unpack_bytes(other_saved["pipeline"]))
# Run `consume` to compute `(a, loss_a, ids_a)`.
a,loss_a,ids_a = consume(other_state,other_iterator,5)
# Run `consume` to compute `(b, loss_b, ids_b)`.
b,loss_b,ids_b = consume(other_saved["training"],other_resumed,5)
# Iterate over `(l, r)` to step through the computation:
for l,r in zip(ids_a,ids_b):np.testing.assert_array_equal(l,r)
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(loss_a,loss_b,rtol=1e-6,atol=1e-7)`
np.testing.assert_allclose(loss_a,loss_b,rtol=1e-6,atol=1e-7)
# Run `same_tree` to perform the next check or state transition.
same_tree(a,b)
# Assert invariant `int(b["step"])==6` holds
assert int(b["step"])==6

# Reference practice: Reject an incompatible resume before consuming data
# Reject an incompatible resume before consuming data (Transfer): Exact state replay only makes sense under a compatible data...
# Iterate over `(field, value)` to step through the computation:
for field,value in [('seed',43),('sha256','different-dataset')]:
    # Evaluate `contract, **{field: value}` and convert the result into Python scalar/collection `incompatible`.
    incompatible=dict(contract,**{field:value})
    # Run the boundary check and catch the expected exception:
    try:validate_contract(restored_payload['contract'],incompatible)
    except ValueError:print('Rejected incompatible',field)
    else:raise AssertionError('Incompatible resume was accepted')
# Run `validate_contract` to perform the next check or state transition.
validate_contract(restored_payload['contract'],contract)
# Print the observed values to compare against the expected result.
print('Original run contract remains valid.')

# Reference practice: Reject changed batch size before consuming
# Reject changed batch size before consuming (Challenge): Changing batching changes consumption semantics.
changed = dict(contract,batch_size=3)
# Run the boundary check and catch the expected exception:
try:
    validate_contract(restored_payload["contract"],changed)
except ValueError:
    print("Expected batch-configuration mismatch")
else:
    raise AssertionError("Expected incompatible restore rejection")
# Run `validate_contract` to perform the next check or state transition.
validate_contract(restored_payload["contract"],contract)
# Run `iter` to compute `fixed_reader`.
# Execute the next step of the computation.
fixed_reader = iter(make_loader())
fixed_reader.set_state(unpack_bytes(restored_payload["pipeline"]))
# Execute `np.testing.assert_array_equal(next(fixed_reader)["id"],refer`
np.testing.assert_array_equal(next(fixed_reader)["id"],reference_ids[0])
print("PASS: recovery-04")
