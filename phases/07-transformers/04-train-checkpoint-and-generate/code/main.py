"""Train, checkpoint, and generate: worked experiments and reference solutions. CPU checks."""

# 1. Prepare packages and the vocabulary
# Step 1 — 1. Prepare packages and the vocabulary: Vocabulary size is three, model width is eight, and the context...
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
# Step 2 — 2. Reuse the verified causal block: The block returns one model-width vector per token; it has no...
def layer_norm(x):
    # Reduce along axis=-1 to compute `mean`.
    mean = jnp.mean(x, axis=-1, keepdims=True)
    # Reduce along axis=-1 to compute `variance`.
    variance = jnp.mean((x-mean)**2, axis=-1, keepdims=True)
    return (x-mean) / jnp.sqrt(variance + 1e-5)

def init_block(key, width=8):
    # Create or split explicit PRNG key(s) (`keys`) for reproducible randomness.
    keys = jax.random.split(key, 6)
    shapes = [(width,width)]*4 + [(width,2*width),(2*width,width)]
    return {name: jax.random.normal(k,s)*0.1 for name,k,s in zip(
        ["q","k","v","o","up","down"], keys, shapes)}

def block_forward(p, x):
    h = layer_norm(x)
    q, k, v = h@p["q"], h@p["k"], h@p["v"]
    scores = q@k.T / jnp.sqrt(x.shape[-1])
    # Create evenly spaced index values in `allowed`.
    allowed = jnp.arange(x.shape[0])[:,None] >= jnp.arange(x.shape[0])[None,:]
    # Apply nonlinear activation or probability normalization to compute `weights`.
    weights = jax.nn.softmax(jnp.where(allowed, scores, -jnp.inf), axis=-1)
    residual = x + (weights@v)@p["o"]
    return residual + jax.nn.gelu(layer_norm(residual)@p["up"])@p["down"]

# 3. Add embeddings and the vocabulary head
# Step 3 — 3. Add embeddings and the vocabulary head: The head has three columns; make_examples shifts each window by...
def init_lm(key):
    # Create or split explicit PRNG key(s) (`(embed_key, block_key, head_key)`) for reproducible randomness.
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
    # Combine or mask array elements to form `rows`.
    rows = np.stack([ids[i:i+CONTEXT+1] for i in range(len(ids)-CONTEXT)])
    return jnp.array(rows[:,:-1]), jnp.array(rows[:,1:])

# 4. Define the loss and compiled state transition
# Step 4 — 4. Define the loss and compiled state transition: train_step returns new parameters, optimizer state and the...
train_x, train_y = make_examples("abc"*12)
held_x, held_y = make_examples("bca"*6)
def lm_loss(p, x, y):
    # Vectorize across the batch dimension with `jax.vmap` (`predictions`).
    predictions = jax.vmap(logits,in_axes=(None,0))(p,x)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(predictions,y))
# Configure or step the Optax optimizer state (`optimizer`).
optimizer = optax.adam(0.03)
# Define and JIT-compile `train_step(p, state, x, y)` so XLA traces and fuses the operations:
@jax.jit
def train_step(p, state, x, y):
    # Differentiate the objective to obtain `(value, gradients)` via automatic differentiation.
    value, gradients = jax.value_and_grad(lm_loss)(p,x,y)
    # Apply the computed gradient updates to update the model parameters.
    updates, next_state = optimizer.update(gradients,state,p)
    return optax.apply_updates(p,updates), next_state, value

# 5. Run the first twenty updates
# Step 5 — 5. Run the first twenty updates: Adam count should now equal twenty; initialization is the only...
# Create or split explicit PRNG key(s) (`params`) for reproducible randomness.
params = init_lm(jax.random.key(3))
state = optimizer.init(params)
initial = float(lm_loss(params,train_x,train_y))
for _ in range(20):
    params, state, _ = train_step(params,state,train_x,train_y)
# Save numeric leaves only, with a schema checked against this program's template.

# 6. Write and validate the snapshot format
# Step 6 — 6. Write and validate the snapshot format: The loader checks architecture, vocabulary, numeric leaf...
def save_snapshot(folder, p, state, step):
    leaves, _ = jax.tree.flatten((p,state))
    # Convert `` to a host NumPy array for inspection or verification.
    np.savez(folder/"state.npz", **{f"leaf_{i}":np.asarray(x) for i,x in enumerate(leaves)})
    manifest = {"schema":1,"step":step,"vocabulary":VOCAB,"context":CONTEXT,
                "shapes":[list(x.shape) for x in leaves],
                "dtypes":[str(x.dtype) for x in leaves]}
    # Write the serialized artifact payload to disk.
    (folder/"manifest.json").write_text(json.dumps(manifest))

def load_snapshot(folder):
    meta = json.loads((folder/"manifest.json").read_text())
    # Guard input contract (`meta['schema'] != 1 or meta['vocabulary'] != VOCAB or meta['context'] != CONTEXT`) and fail fast if violated.
    if meta["schema"] != 1 or meta["vocabulary"] != VOCAB or meta["context"] != CONTEXT:
        raise ValueError("checkpoint architecture/vocabulary mismatch")
    # Guard input contract (`type(meta['step']) is not int or meta['step'] < 0`) and fail fast if violated.
    if type(meta["step"]) is not int or meta["step"] < 0:
        raise ValueError("invalid checkpoint step")
    # Create or split explicit PRNG key(s) (`template`) for reproducible randomness.
    template = init_lm(jax.random.key(0))
    template_leaves, structure = jax.tree.flatten((template,optimizer.init(template)))
    arrays = []
    with np.load(folder/"state.npz",allow_pickle=False) as data:
        # Guard input contract (`set(data.files) != {f'leaf_{i}' for i in range(len(template_leaves))}`) and fail fast if violated.
        if set(data.files) != {f"leaf_{i}" for i in range(len(template_leaves))}:
            raise ValueError("checkpoint leaf count mismatch")
        for i, leaf in enumerate(template_leaves):
            saved = data[f"leaf_{i}"]
            # Guard input contract (`saved.shape != leaf.shape or saved.dtype != np.asarray(leaf).dtype`) and fail fast if violated.
            if saved.shape != leaf.shape or saved.dtype != np.asarray(leaf).dtype:
                raise ValueError("checkpoint leaf contract mismatch")
            # Create device-backed JAX array ``.
            arrays.append(jnp.array(saved))
    restored_p, restored_state = jax.tree.unflatten(structure,arrays)
    # Guard input contract (`int(restored_state[0].count) != meta['step']`) and fail fast if violated.
    if int(restored_state[0].count) != meta["step"]:
        raise ValueError("checkpoint step and Adam count disagree")
    return restored_p, restored_state, meta["step"]

# 7. Verify resume and evaluate the final fit
# Step 7 — 7. Verify resume and evaluate the final fit: Uninterrupted and restored next transitions agree; the held-out...
# Create an isolated temporary directory to run and inspect artifacts safely:
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
# Vectorize across the batch dimension with `jax.vmap` (`held_logits`).
held_logits = jax.vmap(logits,in_axes=(None,0))(params,held_x)
# Reduce along axis=-1 to compute `accuracy`.
accuracy = jnp.mean(jnp.argmax(held_logits,axis=-1)==held_y)
final_loss = lm_loss(params,held_x,held_y)
print("Initial / held loss / held accuracy:",initial,float(final_loss),float(accuracy))
assert final_loss < 0.05
assert accuracy > 0.99

# 8. Define greedy decoding
# Step 8 — 8. Define greedy decoding: The generator recomputes a short window each step.
def generate(p, prompt, count):
    tokens = [VOCAB[c] for c in prompt]
    # Guard input contract (`not tokens`) and fail fast if violated.
    if not tokens:
        raise ValueError("generation needs a nonempty prompt")
    inverse = {i:c for c,i in VOCAB.items()}
    for _ in range(count):
        window = jnp.array(tokens[-CONTEXT:],dtype=jnp.int32)
        new = int(jnp.argmax(logits(p,window)[-1]))
        tokens.append(new)
    return "".join(inverse[i] for i in tokens)

# 9. Check the expected continuation
# Step 9 — 9. Check the expected continuation: The expected continuation is abcabcabcabc, with held-out loss...
continuation = generate(params,"abca",8)
print("Greedy continuation:",continuation)
assert continuation == "abcabcabcabc"

# Step 1 — 1. Prepare packages and the vocabulary: Vocabulary size is three, model width is eight, and the context...
import jax
import jax.numpy as jnp
import numpy as np
import optax
import json
from pathlib import Path
from tempfile import TemporaryDirectory

VOCAB = {"a":0, "b":1, "c":2}
CONTEXT = 4
# Step 2 — 2. Reuse the verified causal block: The block returns one model-width vector per token; it has no...
def layer_norm(x):
    # Reduce along axis=-1 to compute `mean`.
    mean = jnp.mean(x, axis=-1, keepdims=True)
    # Reduce along axis=-1 to compute `variance`.
    variance = jnp.mean((x-mean)**2, axis=-1, keepdims=True)
    return (x-mean) / jnp.sqrt(variance + 1e-5)

def init_block(key, width=8):
    # Create or split explicit PRNG key(s) (`keys`) for reproducible randomness.
    keys = jax.random.split(key, 6)
    shapes = [(width,width)]*4 + [(width,2*width),(2*width,width)]
    return {name: jax.random.normal(k,s)*0.1 for name,k,s in zip(
        ["q","k","v","o","up","down"], keys, shapes)}

def block_forward(p, x):
    h = layer_norm(x)
    q, k, v = h@p["q"], h@p["k"], h@p["v"]
    scores = q@k.T / jnp.sqrt(x.shape[-1])
    # Create evenly spaced index values in `allowed`.
    allowed = jnp.arange(x.shape[0])[:,None] >= jnp.arange(x.shape[0])[None,:]
    # Apply nonlinear activation or probability normalization to compute `weights`.
    weights = jax.nn.softmax(jnp.where(allowed, scores, -jnp.inf), axis=-1)
    residual = x + (weights@v)@p["o"]
    return residual + jax.nn.gelu(layer_norm(residual)@p["up"])@p["down"]

# Step 3 — 3. Add embeddings and the vocabulary head: The head has three columns; make_examples shifts each window by...
def init_lm(key):
    # Create or split explicit PRNG key(s) (`(embed_key, block_key, head_key)`) for reproducible randomness.
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
    # Combine or mask array elements to form `rows`.
    rows = np.stack([ids[i:i+CONTEXT+1] for i in range(len(ids)-CONTEXT)])
    return jnp.array(rows[:,:-1]), jnp.array(rows[:,1:])

# Step 4 — 4. Define the loss and compiled state transition: train_step returns new parameters, optimizer state and the...
train_x, train_y = make_examples("abc"*12)
held_x, held_y = make_examples("bca"*6)
def lm_loss(p, x, y):
    # Vectorize across the batch dimension with `jax.vmap` (`predictions`).
    predictions = jax.vmap(logits,in_axes=(None,0))(p,x)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(predictions,y))
# Configure or step the Optax optimizer state (`optimizer`).
optimizer = optax.adam(0.03)
# Define and JIT-compile `train_step(p, state, x, y)` so XLA traces and fuses the operations:
@jax.jit
def train_step(p, state, x, y):
    # Differentiate the objective to obtain `(value, gradients)` via automatic differentiation.
    value, gradients = jax.value_and_grad(lm_loss)(p,x,y)
    # Apply the computed gradient updates to update the model parameters.
    updates, next_state = optimizer.update(gradients,state,p)
    return optax.apply_updates(p,updates), next_state, value

# Step 5 — 5. Run the first twenty updates: Adam count should now equal twenty; initialization is the only...
# Create or split explicit PRNG key(s) (`params`) for reproducible randomness.
params = init_lm(jax.random.key(3))
state = optimizer.init(params)
initial = float(lm_loss(params,train_x,train_y))
for _ in range(20):
    params, state, _ = train_step(params,state,train_x,train_y)
# Save numeric leaves only, with a schema checked against this program's template.
# Step 6 — 6. Write and validate the snapshot format: The loader checks architecture, vocabulary, numeric leaf...
def save_snapshot(folder, p, state, step):
    leaves, _ = jax.tree.flatten((p,state))
    # Convert `` to a host NumPy array for inspection or verification.
    np.savez(folder/"state.npz", **{f"leaf_{i}":np.asarray(x) for i,x in enumerate(leaves)})
    manifest = {"schema":1,"step":step,"vocabulary":VOCAB,"context":CONTEXT,
                "shapes":[list(x.shape) for x in leaves],
                "dtypes":[str(x.dtype) for x in leaves]}
    # Write the serialized artifact payload to disk.
    (folder/"manifest.json").write_text(json.dumps(manifest))

def load_snapshot(folder):
    meta = json.loads((folder/"manifest.json").read_text())
    # Guard input contract (`meta['schema'] != 1 or meta['vocabulary'] != VOCAB or meta['context'] != CONTEXT`) and fail fast if violated.
    if meta["schema"] != 1 or meta["vocabulary"] != VOCAB or meta["context"] != CONTEXT:
        raise ValueError("checkpoint architecture/vocabulary mismatch")
    # Guard input contract (`type(meta['step']) is not int or meta['step'] < 0`) and fail fast if violated.
    if type(meta["step"]) is not int or meta["step"] < 0:
        raise ValueError("invalid checkpoint step")
    # Create or split explicit PRNG key(s) (`template`) for reproducible randomness.
    template = init_lm(jax.random.key(0))
    template_leaves, structure = jax.tree.flatten((template,optimizer.init(template)))
    arrays = []
    with np.load(folder/"state.npz",allow_pickle=False) as data:
        # Guard input contract (`set(data.files) != {f'leaf_{i}' for i in range(len(template_leaves))}`) and fail fast if violated.
        if set(data.files) != {f"leaf_{i}" for i in range(len(template_leaves))}:
            raise ValueError("checkpoint leaf count mismatch")
        for i, leaf in enumerate(template_leaves):
            saved = data[f"leaf_{i}"]
            # Guard input contract (`saved.shape != leaf.shape or saved.dtype != np.asarray(leaf).dtype`) and fail fast if violated.
            if saved.shape != leaf.shape or saved.dtype != np.asarray(leaf).dtype:
                raise ValueError("checkpoint leaf contract mismatch")
            # Create device-backed JAX array ``.
            arrays.append(jnp.array(saved))
    restored_p, restored_state = jax.tree.unflatten(structure,arrays)
    # Guard input contract (`int(restored_state[0].count) != meta['step']`) and fail fast if violated.
    if int(restored_state[0].count) != meta["step"]:
        raise ValueError("checkpoint step and Adam count disagree")
    return restored_p, restored_state, meta["step"]

# Step 7 — 7. Verify resume and evaluate the final fit: Uninterrupted and restored next transitions agree; the held-out...
# Create an isolated temporary directory to run and inspect artifacts safely:
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
# Vectorize across the batch dimension with `jax.vmap` (`held_logits`).
held_logits = jax.vmap(logits,in_axes=(None,0))(params,held_x)
# Reduce along axis=-1 to compute `accuracy`.
accuracy = jnp.mean(jnp.argmax(held_logits,axis=-1)==held_y)
final_loss = lm_loss(params,held_x,held_y)
print("Initial / held loss / held accuracy:",initial,float(final_loss),float(accuracy))
assert final_loss < 0.05
assert accuracy > 0.99

# Step 8 — 8. Define greedy decoding: The generator recomputes a short window each step.
def generate(p, prompt, count):
    tokens = [VOCAB[c] for c in prompt]
    # Guard input contract (`not tokens`) and fail fast if violated.
    if not tokens:
        raise ValueError("generation needs a nonempty prompt")
    inverse = {i:c for c,i in VOCAB.items()}
    for _ in range(count):
        window = jnp.array(tokens[-CONTEXT:],dtype=jnp.int32)
        new = int(jnp.argmax(logits(p,window)[-1]))
        tokens.append(new)
    return "".join(inverse[i] for i in tokens)

# Step 9 — 9. Check the expected continuation: The expected continuation is abcabcabcabc, with held-out loss...
continuation = generate(params,"abca",8)
print("Greedy continuation:",continuation)
assert continuation == "abcabcabcabc"

# Figure data experiment
# Compute figure data for: Inspect next-token probabilities before greedy decoding
# Apply nonlinear activation or probability normalization to compute `probs`.
probs = jax.nn.softmax(held_logits[0], axis=-1)
visual_data = {'kind': 'heatmap', 'values': probs.tolist(), 'rows': ['position ' + str(i) for i in range(CONTEXT)], 'columns': list(VOCAB.keys()), 'unit': 'next-token probability'}

# Experiment: Verify uniform-logit cross entropy
# Experiment — Verify uniform-logit cross entropy: A known probability distribution independently checks target...
zero_head = {**params,"head":jnp.zeros_like(params["head"])}
assert jnp.allclose(lm_loss(zero_head,train_x,train_y),jnp.log(3.),atol=1e-6)

# Experiment: Test causality at the final model output
# Experiment — Test causality at the final model output: Causal independence should hold for the full model, not only the...
example = jnp.array([0,1,2,0])
original = logits(params,example)
modified = logits(params,example.at[-1].set(1))
assert jnp.allclose(original[:3],modified[:3],atol=1e-5)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Generate twelve new tokens from prompt bcab.
assert generate(params,"bcab",12) == "bcabcabcabcabcab"

# Reference practice: Reject an incompatible checkpoint
# Reject an incompatible checkpoint (Transfer / diagnosis): The metadata is part of the model contract even when...
# Create an isolated temporary directory to run and inspect artifacts safely:
with TemporaryDirectory() as directory:
    folder = Path(directory)
    save_snapshot(folder,params,state,100)
    metadata_path = folder/"manifest.json"
    correct = metadata_path.read_text()
    metadata = json.loads(correct)
    metadata["vocabulary"] = {"a":1,"b":0,"c":2}
    # Write the serialized artifact payload to disk.
    metadata_path.write_text(json.dumps(metadata))
    try:
        load_snapshot(folder)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong vocabulary accepted")
    forged_step = json.loads(correct)
    forged_step["step"] = 101
    # Write the serialized artifact payload to disk.
    metadata_path.write_text(json.dumps(forged_step))
    try:
        load_snapshot(folder)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong step accepted")
    # Write the serialized artifact payload to disk.
    metadata_path.write_text(correct)
    recovered, _, recovered_step = load_snapshot(folder)
    assert recovered_step == 100
    assert jnp.allclose(logits(recovered,jnp.array([0,1,2,0])),logits(params,jnp.array([0,1,2,0])))

# Reference practice: Make prompt failures explicit
# Make prompt failures explicit (Transfer / diagnosis): A decoding interface needs a defined input contract; neither...
# Run the boundary check and catch the expected exception:
try:
    generate(params,"",1)
except ValueError:
    pass
else:
    raise AssertionError("empty prompt accepted")
# Run the boundary check and catch the expected exception:
try:
    generate(params,"x",1)
except KeyError:
    pass
else:
    raise AssertionError("unknown token accepted")
assert generate(params,"abca",0) == "abca"
print("PASS: transformers-04")
