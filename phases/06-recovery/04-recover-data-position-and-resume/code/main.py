"""Recover data position and resume: worked experiments and reference solutions. CPU checks."""

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

import numpy as np
import grain.python as grain
class TutorialSource:
    def __init__(self,count=12):
        self.count = count
    def __len__(self):
        return self.count
    def __getitem__(self,index):
        value = np.float32(index/10)
        return {"id":np.int64(index),"x":value,"y":np.float32(2*value-1)}
    def __repr__(self):
        return f"TutorialSource(v1,n={self.count})"
def make_loader(seed=42,batch_size=4,count=12,drop_remainder=False):
    return grain.DataLoader(
        data_source=TutorialSource(count),
        sampler=grain.IndexSampler(num_records=count,num_epochs=2,shuffle=True,seed=seed),
        operations=[grain.Batch(batch_size,drop_remainder=drop_remainder)],
        worker_count=0)

# Build the pipeline or state transition
def data_contract(seed=42):
    source = TutorialSource()
    values = np.array([[source[i]["x"],source[i]["y"]] for i in range(len(source))],dtype=np.float32)
    return {"schema":1,"seed":seed,"batch_size":4,"count":12,"source":"TutorialSource(v1,n=12)",
            "sha256":hashlib.sha256(values.tobytes()).hexdigest(),
            "learning_rate":0.05,"optimizer":"adam","key_impl":"threefry2x32",
            "versions":dict(package_versions(),grain=__import__("importlib.metadata",fromlist=["version"]).version("grain"))}
def consume(state,iterator,count):
    losses=[];orders=[]
    for _ in range(count):
        batch = next(iterator)
        orders.append(batch["id"].copy())
        state,value = step(state,jnp.asarray(batch["x"]),jnp.asarray(batch["y"]))
        losses.append(float(value))
    return state,np.array(losses),orders
contract = data_contract()
reference_iterator = iter(make_loader())
reference_state,_,_ = consume(initial_state(),reference_iterator,2)
payload = {"training":reference_state,"pipeline":pack_bytes(reference_iterator.get_state()),"contract":encode_contract(contract)}
resume_path = root/"completed_step_2"
with ocp.StandardCheckpointer() as cp:
    cp.save(resume_path,payload)
    cp.wait_until_finished()
    restored_payload = cp.restore(resume_path,target={"training":initial_state(),"pipeline":pack_bytes(b""),"contract":encode_contract(contract)})
validate_contract(restored_payload["contract"],contract)
resumed_iterator = iter(make_loader())
resumed_iterator.set_state(unpack_bytes(restored_payload["pipeline"]))

# Run the comparison
reference_final,reference_losses,reference_ids = consume(reference_state,reference_iterator,4)
resumed_final,resumed_losses,resumed_ids = consume(restored_payload["training"],resumed_iterator,4)
for a,b in zip(reference_ids,resumed_ids):np.testing.assert_array_equal(a,b)
np.testing.assert_allclose(reference_losses,resumed_losses,rtol=1e-6,atol=1e-7)
same_tree(reference_final,resumed_final)
assert int(resumed_final["step"])==6
print("Restored next four ID batches:",[a.tolist() for a in resumed_ids])
print("Restored next four losses:",resumed_losses.tolist())
print("Full state agrees at step six")

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

import numpy as np
import grain.python as grain
class TutorialSource:
    def __init__(self,count=12):
        self.count = count
    def __len__(self):
        return self.count
    def __getitem__(self,index):
        value = np.float32(index/10)
        return {"id":np.int64(index),"x":value,"y":np.float32(2*value-1)}
    def __repr__(self):
        return f"TutorialSource(v1,n={self.count})"
def make_loader(seed=42,batch_size=4,count=12,drop_remainder=False):
    return grain.DataLoader(
        data_source=TutorialSource(count),
        sampler=grain.IndexSampler(num_records=count,num_epochs=2,shuffle=True,seed=seed),
        operations=[grain.Batch(batch_size,drop_remainder=drop_remainder)],
        worker_count=0)

def data_contract(seed=42):
    source = TutorialSource()
    values = np.array([[source[i]["x"],source[i]["y"]] for i in range(len(source))],dtype=np.float32)
    return {"schema":1,"seed":seed,"batch_size":4,"count":12,"source":"TutorialSource(v1,n=12)",
            "sha256":hashlib.sha256(values.tobytes()).hexdigest(),
            "learning_rate":0.05,"optimizer":"adam","key_impl":"threefry2x32",
            "versions":dict(package_versions(),grain=__import__("importlib.metadata",fromlist=["version"]).version("grain"))}
def consume(state,iterator,count):
    losses=[];orders=[]
    for _ in range(count):
        batch = next(iterator)
        orders.append(batch["id"].copy())
        state,value = step(state,jnp.asarray(batch["x"]),jnp.asarray(batch["y"]))
        losses.append(float(value))
    return state,np.array(losses),orders
contract = data_contract()
reference_iterator = iter(make_loader())
reference_state,_,_ = consume(initial_state(),reference_iterator,2)
payload = {"training":reference_state,"pipeline":pack_bytes(reference_iterator.get_state()),"contract":encode_contract(contract)}
resume_path = root/"completed_step_2"
with ocp.StandardCheckpointer() as cp:
    cp.save(resume_path,payload)
    cp.wait_until_finished()
    restored_payload = cp.restore(resume_path,target={"training":initial_state(),"pipeline":pack_bytes(b""),"contract":encode_contract(contract)})
validate_contract(restored_payload["contract"],contract)
resumed_iterator = iter(make_loader())
resumed_iterator.set_state(unpack_bytes(restored_payload["pipeline"]))

reference_final,reference_losses,reference_ids = consume(reference_state,reference_iterator,4)
resumed_final,resumed_losses,resumed_ids = consume(restored_payload["training"],resumed_iterator,4)
for a,b in zip(reference_ids,resumed_ids):np.testing.assert_array_equal(a,b)
np.testing.assert_allclose(reference_losses,resumed_losses,rtol=1e-6,atol=1e-7)
same_tree(reference_final,resumed_final)
assert int(resumed_final["step"])==6
print("Restored next four ID batches:",[a.tolist() for a in resumed_ids])
print("Restored next four losses:",resumed_losses.tolist())
print("Full state agrees at step six")

# Experiment: Restart only the reader
fresh = iter(make_loader())
wrong_state,wrong_losses,wrong_ids = consume(restored_payload["training"],fresh,1)
assert not np.array_equal(wrong_ids[0],reference_ids[0])
assert not np.isclose(wrong_losses[0],reference_losses[0],rtol=1e-6,atol=1e-7)
print("Missing iterator restore repeats earlier IDs and changes the next loss")

# Experiment: Restore only the reader
reader_only = iter(make_loader());reader_only.set_state(unpack_bytes(restored_payload["pipeline"]))
wrong_model,wrong_model_losses,correct_ids = consume(initial_state(),reader_only,1)
np.testing.assert_array_equal(correct_ids[0],reference_ids[0])
assert not np.isclose(wrong_model_losses[0],reference_losses[0],rtol=1e-6,atol=1e-7)
assert int(wrong_model["step"])==1
print("Correct IDs alone do not restore the training transition")

# Reference solution. Try the exercise before reading this.
other_iterator = iter(make_loader())
other_state,_,_ = consume(initial_state(),other_iterator,1)
other_payload = {"training":other_state,"pipeline":pack_bytes(other_iterator.get_state()),"contract":encode_contract(contract)}
with ocp.StandardCheckpointer() as cp:
    other_path = Path(tempfile.mkdtemp(dir=root))/"completed_step_1"
    cp.save(other_path,other_payload)
    cp.wait_until_finished()
    other_saved = cp.restore(other_path,target={"training":initial_state(),"pipeline":pack_bytes(b""),"contract":encode_contract(contract)})
other_resumed = iter(make_loader());other_resumed.set_state(unpack_bytes(other_saved["pipeline"]))
a,loss_a,ids_a = consume(other_state,other_iterator,5)
b,loss_b,ids_b = consume(other_saved["training"],other_resumed,5)
for l,r in zip(ids_a,ids_b):np.testing.assert_array_equal(l,r)
np.testing.assert_allclose(loss_a,loss_b,rtol=1e-6,atol=1e-7)
same_tree(a,b)
assert int(b["step"])==6

# Reference practice: Reject an incompatible resume before consuming data
for field,value in [('seed',43),('sha256','different-dataset')]:
    incompatible=dict(contract,**{field:value})
    try:validate_contract(restored_payload['contract'],incompatible)
    except ValueError:print('Rejected incompatible',field)
    else:raise AssertionError('Incompatible resume was accepted')
validate_contract(restored_payload['contract'],contract)
print('Original run contract remains valid.')

# Reference practice: Reject changed batch size before consuming
changed = dict(contract,batch_size=3)
try:
    validate_contract(restored_payload["contract"],changed)
except ValueError:
    print("Expected batch-configuration mismatch")
else:
    raise AssertionError("Expected incompatible restore rejection")
validate_contract(restored_payload["contract"],contract)
fixed_reader = iter(make_loader());fixed_reader.set_state(unpack_bytes(restored_payload["pipeline"]))
np.testing.assert_array_equal(next(fixed_reader)["id"],reference_ids[0])
print("PASS: recovery-04")
