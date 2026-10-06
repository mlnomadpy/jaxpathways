"""Save model, optimizer, and random state: worked experiments and reference solutions. CPU checks."""

# Prepare the data and state
import jax
jax.config.update("jax_platforms","cpu")
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tempfile
import json
import hashlib
from pathlib import Path
tx = optax.adam(0.05)
def initial_state():
    params = {"weight":jnp.array(0.,jnp.float32),"bias":jnp.array(0.,jnp.float32)}
    return {"params":params,"optimizer":tx.init(params),
            "key_data":jax.random.key_data(jax.random.key(7,impl="threefry2x32")),
            "step":jnp.array(0,jnp.int32)}
def pack_bytes(raw,capacity=4096):
    if len(raw)>capacity:
        raise ValueError("Snapshot exceeds tutorial fixed-byte capacity")
    buffer = np.zeros(capacity,dtype=np.uint8)
    buffer[:len(raw)] = np.frombuffer(raw,dtype=np.uint8)
    return {"buffer":buffer,"length":np.array(len(raw),dtype=np.int32)}
def unpack_bytes(packed):
    length = int(packed["length"])
    if not 0 <= length <= len(packed["buffer"]):
        raise ValueError("Invalid snapshot length")
    return np.asarray(packed["buffer"],dtype=np.uint8)[:length].tobytes()
def same_tree(left,right):
    assert jax.tree.structure(left)==jax.tree.structure(right)
    for a,b in zip(jax.tree.leaves(left),jax.tree.leaves(right)):
        host_a,host_b = np.asarray(a),np.asarray(b)
        assert host_a.shape==host_b.shape and host_a.dtype==host_b.dtype
        if np.issubdtype(host_a.dtype,np.integer):
            np.testing.assert_array_equal(host_a,host_b)
        else:
            np.testing.assert_allclose(host_a,host_b,rtol=1e-6,atol=1e-7)
def step(state,x,y):
    key = jax.random.wrap_key_data(state["key_data"],impl="threefry2x32")
    next_key,sample_key = jax.random.split(key)
    target_noise = 0.01*jax.random.normal(sample_key,y.shape)
    def objective(params):
        residual = params["weight"]*x+params["bias"]-(y+target_noise)
        return jnp.mean(residual**2)
    value,gradient = jax.value_and_grad(objective)(state["params"])
    updates,opt_state = tx.update(gradient,state["optimizer"],state["params"])
    next_state = {"params":optax.apply_updates(state["params"],updates),
                  "optimizer":opt_state,"key_data":jax.random.key_data(next_key),
                  "step":state["step"]+1}
    return next_state,value
def package_versions():
    import importlib.metadata
    return {name:importlib.metadata.version(name) for name in ("jax","numpy","optax","orbax-checkpoint")}
def encode_contract(contract):
    return pack_bytes(json.dumps(contract,sort_keys=True).encode())
def validate_contract(saved,expected):
    actual = json.loads(unpack_bytes(saved))
    if actual != expected:
        raise ValueError("Checkpoint contract differs: verify dataset, optimizer configuration, key implementation and package versions")
workspace = tempfile.TemporaryDirectory()
root = Path(workspace.name)

# Build the pipeline or state transition
x = jnp.array([0.,0.5,1.],dtype=jnp.float32)
y = 2*x-1
contract = {"schema":1,"learning_rate":0.05,"optimizer":"adam",
            "key_impl":"threefry2x32","versions":package_versions(),
            "data_sha256":hashlib.sha256(np.asarray(x).tobytes()+np.asarray(y).tobytes()).hexdigest()}
state = initial_state()
for _ in range(3):
    state,_ = step(state,x,y)
checkpoint_path = root/"step_3"
payload = {"training":state,"contract":encode_contract(contract)}
with ocp.StandardCheckpointer() as checkpointer:
    checkpointer.save(checkpoint_path,payload)
    checkpointer.wait_until_finished()
    restored = checkpointer.restore(checkpoint_path,target={"training":initial_state(),"contract":encode_contract(contract)})
validate_contract(restored["contract"],contract)

# Run the comparison
same_tree(state,restored["training"])
expected_next,expected_loss = step(state,x,y)
restored_next,restored_loss = step(restored["training"],x,y)
same_tree(expected_next,restored_next)
np.testing.assert_allclose(np.asarray(expected_loss),np.asarray(restored_loss),rtol=1e-6)
assert int(restored_next["step"])==4
print("Restored step:",int(restored["training"]["step"]))
print("Optimizer count:",int(restored["training"]["optimizer"][0].count))
print("Next update and loss agree; checkpoint exists:",checkpoint_path.exists())

import jax
jax.config.update("jax_platforms","cpu")
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tempfile
import json
import hashlib
from pathlib import Path
tx = optax.adam(0.05)
def initial_state():
    params = {"weight":jnp.array(0.,jnp.float32),"bias":jnp.array(0.,jnp.float32)}
    return {"params":params,"optimizer":tx.init(params),
            "key_data":jax.random.key_data(jax.random.key(7,impl="threefry2x32")),
            "step":jnp.array(0,jnp.int32)}
def pack_bytes(raw,capacity=4096):
    if len(raw)>capacity:
        raise ValueError("Snapshot exceeds tutorial fixed-byte capacity")
    buffer = np.zeros(capacity,dtype=np.uint8)
    buffer[:len(raw)] = np.frombuffer(raw,dtype=np.uint8)
    return {"buffer":buffer,"length":np.array(len(raw),dtype=np.int32)}
def unpack_bytes(packed):
    length = int(packed["length"])
    if not 0 <= length <= len(packed["buffer"]):
        raise ValueError("Invalid snapshot length")
    return np.asarray(packed["buffer"],dtype=np.uint8)[:length].tobytes()
def same_tree(left,right):
    assert jax.tree.structure(left)==jax.tree.structure(right)
    for a,b in zip(jax.tree.leaves(left),jax.tree.leaves(right)):
        host_a,host_b = np.asarray(a),np.asarray(b)
        assert host_a.shape==host_b.shape and host_a.dtype==host_b.dtype
        if np.issubdtype(host_a.dtype,np.integer):
            np.testing.assert_array_equal(host_a,host_b)
        else:
            np.testing.assert_allclose(host_a,host_b,rtol=1e-6,atol=1e-7)
def step(state,x,y):
    key = jax.random.wrap_key_data(state["key_data"],impl="threefry2x32")
    next_key,sample_key = jax.random.split(key)
    target_noise = 0.01*jax.random.normal(sample_key,y.shape)
    def objective(params):
        residual = params["weight"]*x+params["bias"]-(y+target_noise)
        return jnp.mean(residual**2)
    value,gradient = jax.value_and_grad(objective)(state["params"])
    updates,opt_state = tx.update(gradient,state["optimizer"],state["params"])
    next_state = {"params":optax.apply_updates(state["params"],updates),
                  "optimizer":opt_state,"key_data":jax.random.key_data(next_key),
                  "step":state["step"]+1}
    return next_state,value
def package_versions():
    import importlib.metadata
    return {name:importlib.metadata.version(name) for name in ("jax","numpy","optax","orbax-checkpoint")}
def encode_contract(contract):
    return pack_bytes(json.dumps(contract,sort_keys=True).encode())
def validate_contract(saved,expected):
    actual = json.loads(unpack_bytes(saved))
    if actual != expected:
        raise ValueError("Checkpoint contract differs: verify dataset, optimizer configuration, key implementation and package versions")
workspace = tempfile.TemporaryDirectory()
root = Path(workspace.name)

x = jnp.array([0.,0.5,1.],dtype=jnp.float32)
y = 2*x-1
contract = {"schema":1,"learning_rate":0.05,"optimizer":"adam",
            "key_impl":"threefry2x32","versions":package_versions(),
            "data_sha256":hashlib.sha256(np.asarray(x).tobytes()+np.asarray(y).tobytes()).hexdigest()}
state = initial_state()
for _ in range(3):
    state,_ = step(state,x,y)
checkpoint_path = root/"step_3"
payload = {"training":state,"contract":encode_contract(contract)}
with ocp.StandardCheckpointer() as checkpointer:
    checkpointer.save(checkpoint_path,payload)
    checkpointer.wait_until_finished()
    restored = checkpointer.restore(checkpoint_path,target={"training":initial_state(),"contract":encode_contract(contract)})
validate_contract(restored["contract"],contract)

same_tree(state,restored["training"])
expected_next,expected_loss = step(state,x,y)
restored_next,restored_loss = step(restored["training"],x,y)
same_tree(expected_next,restored_next)
np.testing.assert_allclose(np.asarray(expected_loss),np.asarray(restored_loss),rtol=1e-6)
assert int(restored_next["step"])==4
print("Restored step:",int(restored["training"]["step"]))
print("Optimizer count:",int(restored["training"]["optimizer"][0].count))
print("Next update and loss agree; checkpoint exists:",checkpoint_path.exists())

# Experiment: Model-only restoration loses Adam history
model_only = dict(restored["training"])
model_only["optimizer"] = tx.init(model_only["params"])
wrong_next,_ = step(model_only,x,y)
assert any(not np.allclose(np.asarray(a),np.asarray(b),rtol=1e-6,atol=1e-7) for a,b in zip(jax.tree.leaves(wrong_next["params"]),jax.tree.leaves(expected_next["params"])))
assert int(wrong_next["optimizer"][0].count)==1
print("Model-only checkpoint produces a different Adam update")

# Experiment: Reset the random stream
reset_random = dict(restored["training"])
reset_random["key_data"] = initial_state()["key_data"]
reset_next,reset_loss = step(reset_random,x,y)
assert not np.isclose(float(reset_loss),float(expected_loss),rtol=1e-6,atol=1e-7)
assert not np.array_equal(np.asarray(reset_next["key_data"]),np.asarray(expected_next["key_data"]))
print("Reset key changes the noisy objective and continuation stream")

# Reference solution. Try the exercise before reading this.
later = state
for _ in range(2):later,_=step(later,x,y)
with ocp.StandardCheckpointer() as cp:
    later_path = Path(tempfile.mkdtemp(dir=root))/"step_5"
    cp.save(later_path,{"training":later,"contract":encode_contract(contract)})
    cp.wait_until_finished()
    recovered = cp.restore(later_path,target={"training":initial_state(),"contract":encode_contract(contract)})
validate_contract(recovered["contract"],contract)
left,_=step(later,x,y);right,_=step(recovered["training"],x,y)
same_tree(left,right)
assert int(right["step"])==6

# Reference practice: Show what weights-only recovery loses
full=restored['training']
weights_only=dict(full,optimizer=tx.init(full['params']))
full_next,_=step(full,x,y)
reset_next,_=step(weights_only,x,y)
assert int(full_next['optimizer'][0].count)==4
assert int(reset_next['optimizer'][0].count)==1
assert any(not np.allclose(a,b,rtol=1e-6,atol=1e-7) for a,b in zip(jax.tree.leaves(full_next['params']),jax.tree.leaves(reset_next['params'])))
print('Full restore Adam count: 4; reset memory count: 1; next parameters differ.')

# Reference practice: Reject an incompatible update configuration
changed_contract = dict(contract,learning_rate=0.1)
try:
    validate_contract(restored["contract"],changed_contract)
except ValueError:
    print("Expected changed configuration rejection")
else:
    raise AssertionError("Expected contract mismatch")
validate_contract(restored["contract"],contract)
for a,b in zip(jax.tree.leaves(state["params"]),jax.tree.leaves(restored["training"]["params"])):
    assert a.shape==b.shape and a.dtype==b.dtype
# Integer PRNG words must be checked exactly, not with relative float tolerance.
changed_key = dict(state)
changed_key["key_data"] = state["key_data"].at[0].add(np.uint32(1))
try:
    same_tree(state,changed_key)
except AssertionError:
    print("Expected exact key-data mismatch")
else:
    raise AssertionError("Expected changed PRNG word rejection")
print("PASS: recovery-03")
