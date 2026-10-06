"""Train, checkpoint, and generate: worked experiments and reference solutions. CPU checks."""

# 1. Prepare packages and the vocabulary
import jax
import jax.numpy as jnp
import numpy as np
import optax
import json
from pathlib import Path
from tempfile import TemporaryDirectory

VOCAB = {"a":0, "b":1, "c":2}
CONTEXT = 4

# 2. Reuse the verified causal block
def layer_norm(x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean((x-mean)**2, axis=-1, keepdims=True)
    return (x-mean) / jnp.sqrt(variance + 1e-5)

def init_block(key, width=8):
    keys = jax.random.split(key, 6)
    shapes = [(width,width)]*4 + [(width,2*width),(2*width,width)]
    return {name: jax.random.normal(k,s)*0.1 for name,k,s in zip(
        ["q","k","v","o","up","down"], keys, shapes)}

def block_forward(p, x):
    h = layer_norm(x)
    q, k, v = h@p["q"], h@p["k"], h@p["v"]
    scores = q@k.T / jnp.sqrt(x.shape[-1])
    allowed = jnp.arange(x.shape[0])[:,None] >= jnp.arange(x.shape[0])[None,:]
    weights = jax.nn.softmax(jnp.where(allowed, scores, -jnp.inf), axis=-1)
    residual = x + (weights@v)@p["o"]
    return residual + jax.nn.gelu(layer_norm(residual)@p["up"])@p["down"]

# 3. Add embeddings and the vocabulary head
def init_lm(key):
    embed_key, block_key, head_key = jax.random.split(key, 3)
    return {"embed":jax.random.normal(embed_key,(3,8))*0.1,
            "position":jnp.arange(CONTEXT*8,dtype=jnp.float32).reshape(CONTEXT,8)*0.001,
            "block":init_block(block_key),
            "head":jax.random.normal(head_key,(8,3))*0.1}

def logits(p, tokens):
    h = p["embed"][tokens] + p["position"][:tokens.shape[0]]
    return layer_norm(block_forward(p["block"],h))@p["head"]

def make_examples(text):
    ids = np.array([VOCAB[c] for c in text],dtype=np.int32)
    rows = np.stack([ids[i:i+CONTEXT+1] for i in range(len(ids)-CONTEXT)])
    return jnp.array(rows[:,:-1]), jnp.array(rows[:,1:])

# 4. Define the loss and compiled state transition
train_x, train_y = make_examples("abc"*12)
held_x, held_y = make_examples("bca"*6)
def lm_loss(p, x, y):
    predictions = jax.vmap(logits,in_axes=(None,0))(p,x)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(predictions,y))
optimizer = optax.adam(0.03)
@jax.jit
def train_step(p, state, x, y):
    value, gradients = jax.value_and_grad(lm_loss)(p,x,y)
    updates, next_state = optimizer.update(gradients,state,p)
    return optax.apply_updates(p,updates), next_state, value

# 5. Run the first twenty updates
params = init_lm(jax.random.key(3))
state = optimizer.init(params)
initial = float(lm_loss(params,train_x,train_y))
for _ in range(20):
    params, state, _ = train_step(params,state,train_x,train_y)
# Save numeric leaves only, with a schema checked against this program's template.

# 6. Write and validate the snapshot format
def save_snapshot(folder, p, state, step):
    leaves, _ = jax.tree.flatten((p,state))
    np.savez(folder/"state.npz", **{f"leaf_{i}":np.asarray(x) for i,x in enumerate(leaves)})
    manifest = {"schema":1,"step":step,"vocabulary":VOCAB,"context":CONTEXT,
                "shapes":[list(x.shape) for x in leaves],
                "dtypes":[str(x.dtype) for x in leaves]}
    (folder/"manifest.json").write_text(json.dumps(manifest))

def load_snapshot(folder):
    meta = json.loads((folder/"manifest.json").read_text())
    if meta["schema"] != 1 or meta["vocabulary"] != VOCAB or meta["context"] != CONTEXT:
        raise ValueError("checkpoint architecture/vocabulary mismatch")
    if type(meta["step"]) is not int or meta["step"] < 0:
        raise ValueError("invalid checkpoint step")
    template = init_lm(jax.random.key(0))
    template_leaves, structure = jax.tree.flatten((template,optimizer.init(template)))
    arrays = []
    with np.load(folder/"state.npz",allow_pickle=False) as data:
        if set(data.files) != {f"leaf_{i}" for i in range(len(template_leaves))}:
            raise ValueError("checkpoint leaf count mismatch")
        for i, leaf in enumerate(template_leaves):
            saved = data[f"leaf_{i}"]
            if saved.shape != leaf.shape or saved.dtype != np.asarray(leaf).dtype:
                raise ValueError("checkpoint leaf contract mismatch")
            arrays.append(jnp.array(saved))
    restored_p, restored_state = jax.tree.unflatten(structure,arrays)
    if int(restored_state[0].count) != meta["step"]:
        raise ValueError("checkpoint step and Adam count disagree")
    return restored_p, restored_state, meta["step"]

# 7. Verify resume and evaluate the final fit
with TemporaryDirectory() as directory:
    folder = Path(directory)
    save_snapshot(folder,params,state,20)
    restored, restored_state, step = load_snapshot(folder)
    assert step == 20
    next_a, state_a, _ = train_step(params,state,train_x,train_y)
    next_b, state_b, _ = train_step(restored,restored_state,train_x,train_y)
    assert all(jnp.allclose(a,b,atol=1e-7) for a,b in zip(
        jax.tree.leaves((next_a,state_a)),jax.tree.leaves((next_b,state_b))))
for _ in range(80):
    params, state, _ = train_step(params,state,train_x,train_y)
held_logits = jax.vmap(logits,in_axes=(None,0))(params,held_x)
accuracy = jnp.mean(jnp.argmax(held_logits,axis=-1)==held_y)
final_loss = lm_loss(params,held_x,held_y)
print("Initial / held loss / held accuracy:",initial,float(final_loss),float(accuracy))
assert final_loss < 0.05
assert accuracy > 0.99

# 8. Define greedy decoding
def generate(p, prompt, count):
    tokens = [VOCAB[c] for c in prompt]
    if not tokens:
        raise ValueError("generation needs a nonempty prompt")
    inverse = {i:c for c,i in VOCAB.items()}
    for _ in range(count):
        window = jnp.array(tokens[-CONTEXT:],dtype=jnp.int32)
        new = int(jnp.argmax(logits(p,window)[-1]))
        tokens.append(new)
    return "".join(inverse[i] for i in tokens)

# 9. Check the expected continuation
continuation = generate(params,"abca",8)
print("Greedy continuation:",continuation)
assert continuation == "abcabcabcabc"

import jax
import jax.numpy as jnp
import numpy as np
import optax
import json
from pathlib import Path
from tempfile import TemporaryDirectory

VOCAB = {"a":0, "b":1, "c":2}
CONTEXT = 4
def layer_norm(x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean((x-mean)**2, axis=-1, keepdims=True)
    return (x-mean) / jnp.sqrt(variance + 1e-5)

def init_block(key, width=8):
    keys = jax.random.split(key, 6)
    shapes = [(width,width)]*4 + [(width,2*width),(2*width,width)]
    return {name: jax.random.normal(k,s)*0.1 for name,k,s in zip(
        ["q","k","v","o","up","down"], keys, shapes)}

def block_forward(p, x):
    h = layer_norm(x)
    q, k, v = h@p["q"], h@p["k"], h@p["v"]
    scores = q@k.T / jnp.sqrt(x.shape[-1])
    allowed = jnp.arange(x.shape[0])[:,None] >= jnp.arange(x.shape[0])[None,:]
    weights = jax.nn.softmax(jnp.where(allowed, scores, -jnp.inf), axis=-1)
    residual = x + (weights@v)@p["o"]
    return residual + jax.nn.gelu(layer_norm(residual)@p["up"])@p["down"]

def init_lm(key):
    embed_key, block_key, head_key = jax.random.split(key, 3)
    return {"embed":jax.random.normal(embed_key,(3,8))*0.1,
            "position":jnp.arange(CONTEXT*8,dtype=jnp.float32).reshape(CONTEXT,8)*0.001,
            "block":init_block(block_key),
            "head":jax.random.normal(head_key,(8,3))*0.1}

def logits(p, tokens):
    h = p["embed"][tokens] + p["position"][:tokens.shape[0]]
    return layer_norm(block_forward(p["block"],h))@p["head"]

def make_examples(text):
    ids = np.array([VOCAB[c] for c in text],dtype=np.int32)
    rows = np.stack([ids[i:i+CONTEXT+1] for i in range(len(ids)-CONTEXT)])
    return jnp.array(rows[:,:-1]), jnp.array(rows[:,1:])

train_x, train_y = make_examples("abc"*12)
held_x, held_y = make_examples("bca"*6)
def lm_loss(p, x, y):
    predictions = jax.vmap(logits,in_axes=(None,0))(p,x)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(predictions,y))
optimizer = optax.adam(0.03)
@jax.jit
def train_step(p, state, x, y):
    value, gradients = jax.value_and_grad(lm_loss)(p,x,y)
    updates, next_state = optimizer.update(gradients,state,p)
    return optax.apply_updates(p,updates), next_state, value

params = init_lm(jax.random.key(3))
state = optimizer.init(params)
initial = float(lm_loss(params,train_x,train_y))
for _ in range(20):
    params, state, _ = train_step(params,state,train_x,train_y)
# Save numeric leaves only, with a schema checked against this program's template.
def save_snapshot(folder, p, state, step):
    leaves, _ = jax.tree.flatten((p,state))
    np.savez(folder/"state.npz", **{f"leaf_{i}":np.asarray(x) for i,x in enumerate(leaves)})
    manifest = {"schema":1,"step":step,"vocabulary":VOCAB,"context":CONTEXT,
                "shapes":[list(x.shape) for x in leaves],
                "dtypes":[str(x.dtype) for x in leaves]}
    (folder/"manifest.json").write_text(json.dumps(manifest))

def load_snapshot(folder):
    meta = json.loads((folder/"manifest.json").read_text())
    if meta["schema"] != 1 or meta["vocabulary"] != VOCAB or meta["context"] != CONTEXT:
        raise ValueError("checkpoint architecture/vocabulary mismatch")
    if type(meta["step"]) is not int or meta["step"] < 0:
        raise ValueError("invalid checkpoint step")
    template = init_lm(jax.random.key(0))
    template_leaves, structure = jax.tree.flatten((template,optimizer.init(template)))
    arrays = []
    with np.load(folder/"state.npz",allow_pickle=False) as data:
        if set(data.files) != {f"leaf_{i}" for i in range(len(template_leaves))}:
            raise ValueError("checkpoint leaf count mismatch")
        for i, leaf in enumerate(template_leaves):
            saved = data[f"leaf_{i}"]
            if saved.shape != leaf.shape or saved.dtype != np.asarray(leaf).dtype:
                raise ValueError("checkpoint leaf contract mismatch")
            arrays.append(jnp.array(saved))
    restored_p, restored_state = jax.tree.unflatten(structure,arrays)
    if int(restored_state[0].count) != meta["step"]:
        raise ValueError("checkpoint step and Adam count disagree")
    return restored_p, restored_state, meta["step"]

with TemporaryDirectory() as directory:
    folder = Path(directory)
    save_snapshot(folder,params,state,20)
    restored, restored_state, step = load_snapshot(folder)
    assert step == 20
    next_a, state_a, _ = train_step(params,state,train_x,train_y)
    next_b, state_b, _ = train_step(restored,restored_state,train_x,train_y)
    assert all(jnp.allclose(a,b,atol=1e-7) for a,b in zip(
        jax.tree.leaves((next_a,state_a)),jax.tree.leaves((next_b,state_b))))
for _ in range(80):
    params, state, _ = train_step(params,state,train_x,train_y)
held_logits = jax.vmap(logits,in_axes=(None,0))(params,held_x)
accuracy = jnp.mean(jnp.argmax(held_logits,axis=-1)==held_y)
final_loss = lm_loss(params,held_x,held_y)
print("Initial / held loss / held accuracy:",initial,float(final_loss),float(accuracy))
assert final_loss < 0.05
assert accuracy > 0.99

def generate(p, prompt, count):
    tokens = [VOCAB[c] for c in prompt]
    if not tokens:
        raise ValueError("generation needs a nonempty prompt")
    inverse = {i:c for c,i in VOCAB.items()}
    for _ in range(count):
        window = jnp.array(tokens[-CONTEXT:],dtype=jnp.int32)
        new = int(jnp.argmax(logits(p,window)[-1]))
        tokens.append(new)
    return "".join(inverse[i] for i in tokens)

continuation = generate(params,"abca",8)
print("Greedy continuation:",continuation)
assert continuation == "abcabcabcabc"

# Figure data experiment
probs = jax.nn.softmax(held_logits[0], axis=-1)
visual_data = {'kind': 'heatmap', 'values': probs.tolist(), 'rows': ['position ' + str(i) for i in range(CONTEXT)], 'columns': list(VOCAB.keys()), 'unit': 'next-token probability'}

# Experiment: Verify uniform-logit cross entropy
zero_head = {**params,"head":jnp.zeros_like(params["head"])}
assert jnp.allclose(lm_loss(zero_head,train_x,train_y),jnp.log(3.),atol=1e-6)

# Experiment: Test causality at the final model output
example = jnp.array([0,1,2,0])
original = logits(params,example)
modified = logits(params,example.at[-1].set(1))
assert jnp.allclose(original[:3],modified[:3],atol=1e-5)

# Reference solution. Try the exercise before reading this.
assert generate(params,"bcab",12) == "bcabcabcabcabcab"

# Reference practice: Reject an incompatible checkpoint
with TemporaryDirectory() as directory:
    folder = Path(directory)
    save_snapshot(folder,params,state,100)
    metadata_path = folder/"manifest.json"
    correct = metadata_path.read_text()
    metadata = json.loads(correct)
    metadata["vocabulary"] = {"a":1,"b":0,"c":2}
    metadata_path.write_text(json.dumps(metadata))
    try:
        load_snapshot(folder)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong vocabulary accepted")
    forged_step = json.loads(correct)
    forged_step["step"] = 101
    metadata_path.write_text(json.dumps(forged_step))
    try:
        load_snapshot(folder)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong step accepted")
    metadata_path.write_text(correct)
    recovered, _, recovered_step = load_snapshot(folder)
    assert recovered_step == 100
    assert jnp.allclose(logits(recovered,jnp.array([0,1,2,0])),logits(params,jnp.array([0,1,2,0])))

# Reference practice: Make prompt failures explicit
try:
    generate(params,"",1)
except ValueError:
    pass
else:
    raise AssertionError("empty prompt accepted")
try:
    generate(params,"x",1)
except KeyError:
    pass
else:
    raise AssertionError("unknown token accepted")
assert generate(params,"abca",0) == "abca"
print("PASS: transformers-04")
