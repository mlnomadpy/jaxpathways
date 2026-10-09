"""Train, checkpoint, and generate: worked experiments and reference solutions. CPU checks."""

# 1. Prepare packages and the vocabulary
# Step 1 — 1. Prepare packages and the vocabulary: Vocabulary size is three, model width is eight, and the context...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np
import optax
import json
from pathlib import Path
from tempfile import TemporaryDirectory

# Evaluate `VOCAB` from the current inputs and state.
VOCAB = {"a":0, "b":1, "c":2}
# Evaluate `CONTEXT` from the current inputs and state.
CONTEXT = 4

# 2. Reuse the verified causal block
# Step 2 — 2. Reuse the verified causal block: The block returns one model-width vector per token; it has no...
def layer_norm(x):
    # Reduce along axis=-1 to compute `mean`.
    mean = jnp.mean(x, axis=-1, keepdims=True)
    # Reduce along axis=-1 to compute `variance`.
    variance = jnp.mean((x-mean)**2, axis=-1, keepdims=True)
    # Return `(x - mean) / jnp.sqrt(variance + 1e-05)` to the caller.
    return (x-mean) / jnp.sqrt(variance + 1e-5)

# Function `init_block(key, width)` implementing this stage's computation:
def init_block(key, width=8):
    # Create or split explicit PRNG key(s) (`keys`) for reproducible randomness.
    keys = jax.random.split(key, 6)
    # Evaluate `shapes` from the current inputs and state.
    shapes = [(width,width)]*4 + [(width,2*width),(2*width,width)]
    # Return `{name: jax.random.normal(k, s) * 0.1 for name, k, s in zip(['q', 'k', 'v', 'o', 'up', 'down'], keys, shapes)}` to the caller.
    return {name: jax.random.normal(k,s)*0.1 for name,k,s in zip(
        ["q","k","v","o","up","down"], keys, shapes)}

# Function `block_forward(p, x)` implementing this stage's computation:
def block_forward(p, x):
    # Run `layer_norm` to compute `h`.
    h = layer_norm(x)
    # Perform matrix contraction / projection to compute `(q, k, v)`.
    q, k, v = h@p["q"], h@p["k"], h@p["v"]
    # Perform matrix contraction / projection to compute `scores`.
    scores = q@k.T / jnp.sqrt(x.shape[-1])
    # Create evenly spaced index values in `allowed`.
    allowed = jnp.arange(x.shape[0])[:,None] >= jnp.arange(x.shape[0])[None,:]
    # Apply nonlinear activation or probability normalization to compute `weights`.
    weights = jax.nn.softmax(jnp.where(allowed, scores, -jnp.inf), axis=-1)
    # Perform matrix contraction / projection to compute `residual`.
    residual = x + (weights@v)@p["o"]
    # Return `residual + jax.nn.gelu(layer_norm(residual) @ p['up']) @ p['down']` to the caller.
    return residual + jax.nn.gelu(layer_norm(residual)@p["up"])@p["down"]

# 3. Add embeddings and the vocabulary head
# Step 3 — 3. Add embeddings and the vocabulary head: The head has three columns; make_examples shifts each window by...
def init_lm(key):
    # Create or split explicit PRNG key(s) (`(embed_key, block_key, head_key)`) for reproducible randomness.
    embed_key, block_key, head_key = jax.random.split(key, 3)
    # Return `{'embed': jax.random.normal(embed_key, (3, 8)) * 0.1, 'position': jnp.arange(CONTEXT * 8, dtype=jnp.float32).reshape(CONTEXT, 8) * 0.001, 'block': init_block(block_key), 'head': jax.random.normal(head_key, (8, 3)) * 0.1}` to the caller.
    return {"embed":jax.random.normal(embed_key,(3,8))*0.1,
            "position":jnp.arange(CONTEXT*8,dtype=jnp.float32).reshape(CONTEXT,8)*0.001,
            "block":init_block(block_key),
            "head":jax.random.normal(head_key,(8,3))*0.1}

# Function `logits(p, tokens)` implementing this stage's computation:
def logits(p, tokens):
    # Evaluate `h` from the current inputs and state.
    h = p["embed"][tokens] + p["position"][:tokens.shape[0]]
    # Return `layer_norm(block_forward(p['block'], h)) @ p['head']` to the caller.
    return layer_norm(block_forward(p["block"],h))@p["head"]

# Function `make_examples(text)` implementing this stage's computation:
def make_examples(text):
    # Initialize array `ids` with explicit values and shape.
    ids = np.array([VOCAB[c] for c in text],dtype=np.int32)
    # Combine or mask array elements to form `rows`.
    rows = np.stack([ids[i:i+CONTEXT+1] for i in range(len(ids)-CONTEXT)])
    # Return `(jnp.array(rows[:, :-1]), jnp.array(rows[:, 1:]))` to the caller.
    return jnp.array(rows[:,:-1]), jnp.array(rows[:,1:])

# 4. Define the loss and compiled state transition
# Step 4 — 4. Define the loss and compiled state transition: train_step returns new parameters, optimizer state and the...
train_x, train_y = make_examples("abc"*12)
# Run `make_examples` to compute `(held_x, held_y)`.
held_x, held_y = make_examples("bca"*6)
# Function `lm_loss(p, x, y)` implementing this stage's computation:
def lm_loss(p, x, y):
    # Vectorize across the batch dimension with `jax.vmap` (`predictions`).
    predictions = jax.vmap(logits,in_axes=(None,0))(p,x)
    # Return `jnp.mean(optax.softmax_cross_entropy_with_integer_labels(predictions, y))` to the caller.
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(predictions,y))
# Configure or step the Optax optimizer state (`optimizer`).
optimizer = optax.adam(0.03)
# Define and JIT-compile `train_step(p, state, x, y)` so XLA traces and fuses the operations:
@jax.jit
# Function `train_step(p, state, x, y)` implementing this stage's computation:
def train_step(p, state, x, y):
    # Differentiate the objective to obtain `(value, gradients)` via automatic differentiation.
    value, gradients = jax.value_and_grad(lm_loss)(p,x,y)
    # Apply the computed gradient updates to update the model parameters.
    updates, next_state = optimizer.update(gradients,state,p)
    # Return `(optax.apply_updates(p, updates), next_state, value)` to the caller.
    return optax.apply_updates(p,updates), next_state, value

# 5. Run the first twenty updates
# Step 5 — 5. Run the first twenty updates: Adam count should now equal twenty; initialization is the only...
# Create or split explicit PRNG key(s) (`params`) for reproducible randomness.
params = init_lm(jax.random.key(3))
# Run `optimizer.init` to compute `state`.
state = optimizer.init(params)
# Evaluate `lm_loss(params, train_x, train_y)` and convert the result into Python scalar/collection `initial`.
initial = float(lm_loss(params,train_x,train_y))
# Repeat the update loop over `range(20)` steps:
for _ in range(20):
    # Run `train_step` to compute `(params, state, _)`.
    params, state, _ = train_step(params,state,train_x,train_y)
# Save numeric leaves only, with a schema checked against this program's template.

# 6. Write and validate the snapshot format
# Step 6 — 6. Write and validate the snapshot format: The loader checks architecture, vocabulary, numeric leaf...
def save_snapshot(folder, p, state, step):
    # Run `jax.tree.flatten` to compute `(leaves, _)`.
    leaves, _ = jax.tree.flatten((p,state))
    # Convert `` to a host NumPy array for inspection or verification.
    np.savez(folder/"state.npz", **{f"leaf_{i}":np.asarray(x) for i,x in enumerate(leaves)})
    # Evaluate `manifest` from the current inputs and state.
    manifest = {"schema":1,"step":step,"vocabulary":VOCAB,"context":CONTEXT,
                "shapes":[list(x.shape) for x in leaves],
                "dtypes":[str(x.dtype) for x in leaves]}
    # Read or serialize artifact data on disk (``).
    (folder/"manifest.json").write_text(json.dumps(manifest))

# Function `load_snapshot(folder)` implementing this stage's computation:
def load_snapshot(folder):
    # Read or serialize artifact data on disk (`meta`).
    meta = json.loads((folder/"manifest.json").read_text())
    # Guard input contract (`meta['schema'] != 1 or meta['vocabulary'] != VOCAB or meta['context'] != CONTEXT`) and fail fast if violated.
    if meta["schema"] != 1 or meta["vocabulary"] != VOCAB or meta["context"] != CONTEXT:
        raise ValueError("checkpoint architecture/vocabulary mismatch")
    # Guard input contract (`type(meta['step']) is not int or meta['step'] < 0`) and fail fast if violated.
    if type(meta["step"]) is not int or meta["step"] < 0:
        raise ValueError("invalid checkpoint step")
    # Create or split explicit PRNG key(s) (`template`) for reproducible randomness.
    template = init_lm(jax.random.key(0))
    # Run `jax.tree.flatten` to compute `(template_leaves, structure)`.
    template_leaves, structure = jax.tree.flatten((template,optimizer.init(template)))
    # Evaluate `arrays` from the current inputs and state.
    arrays = []
    # Enter managed runtime/context scope for this block:
    with np.load(folder/"state.npz",allow_pickle=False) as data:
        # Guard input contract (`set(data.files) != {f'leaf_{i}' for i in range(len(template_leaves))}`) and fail fast if violated.
        if set(data.files) != {f"leaf_{i}" for i in range(len(template_leaves))}:
            raise ValueError("checkpoint leaf count mismatch")
        # Loop over `(i, leaf)` in `enumerate(template_leaves)`:
        for i, leaf in enumerate(template_leaves):
            # Evaluate `saved` from the current inputs and state.
            saved = data[f"leaf_{i}"]
            # Guard input contract (`saved.shape != leaf.shape or saved.dtype != np.asarray(leaf).dtype`) and fail fast if violated.
            if saved.shape != leaf.shape or saved.dtype != np.asarray(leaf).dtype:
                raise ValueError("checkpoint leaf contract mismatch")
            # Create device-backed JAX array ``.
            arrays.append(jnp.array(saved))
    # Run `jax.tree.unflatten` to compute `(restored_p, restored_state)`.
    restored_p, restored_state = jax.tree.unflatten(structure,arrays)
    # Guard input contract (`int(restored_state[0].count) != meta['step']`) and fail fast if violated.
    if int(restored_state[0].count) != meta["step"]:
        raise ValueError("checkpoint step and Adam count disagree")
    # Return `(restored_p, restored_state, meta['step'])` to the caller.
    return restored_p, restored_state, meta["step"]

# 7. Verify resume and evaluate the final fit
# Step 7 — 7. Verify resume and evaluate the final fit: Uninterrupted and restored next transitions agree; the held-out...
# Create an isolated temporary directory to run and inspect artifacts safely:
with TemporaryDirectory() as directory:
    # Read or serialize artifact data on disk (`folder`).
    folder = Path(directory)
    # Run `save_snapshot` to perform the next check or state transition.
    save_snapshot(folder,params,state,20)
    # Run `load_snapshot` to compute `(restored, restored_state, step)`.
    restored, restored_state, step = load_snapshot(folder)
    # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
    assert step == 20
    # Run `train_step` to compute `(next_a, state_a, _)`.
    next_a, state_a, _ = train_step(params,state,train_x,train_y)
    # Run `train_step` to compute `(next_b, state_b, _)`.
    next_b, state_b, _ = train_step(restored,restored_state,train_x,train_y)
    # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
    assert all(jnp.allclose(a,b,atol=1e-7) for a,b in zip(
        jax.tree.leaves((next_a,state_a)),jax.tree.leaves((next_b,state_b))))
# Repeat the update loop over `range(80)` steps:
for _ in range(80):
    # Run `train_step` to compute `(params, state, _)`.
    params, state, _ = train_step(params,state,train_x,train_y)
# Vectorize across the batch dimension with `jax.vmap` (`held_logits`).
held_logits = jax.vmap(logits,in_axes=(None,0))(params,held_x)
# Reduce along axis=-1 to compute `accuracy`.
accuracy = jnp.mean(jnp.argmax(held_logits,axis=-1)==held_y)
# Run `lm_loss` to compute `final_loss`.
final_loss = lm_loss(params,held_x,held_y)
# Print the observed values to compare against the expected result.
print("Initial / held loss / held accuracy:",initial,float(final_loss),float(accuracy))
# Verify contract: `final_loss < 0.05`.
assert final_loss < 0.05
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert accuracy > 0.99

# 8. Define greedy decoding
# Step 8 — 8. Define greedy decoding: The generator recomputes a short window each step.
def generate(p, prompt, count):
    # Evaluate `tokens` from the current inputs and state.
    tokens = [VOCAB[c] for c in prompt]
    # Guard input contract (`not tokens`) and fail fast if violated.
    if not tokens:
        raise ValueError("generation needs a nonempty prompt")
    # Evaluate `inverse` from the current inputs and state.
    inverse = {i:c for c,i in VOCAB.items()}
    # Repeat the update loop over `range(count)` steps:
    for _ in range(count):
        # Initialize array `window` with explicit values and shape.
        window = jnp.array(tokens[-CONTEXT:],dtype=jnp.int32)
        # Evaluate `jnp.argmax(logits(p, window)[-1])` and convert the result into Python scalar/collection `new`.
        new = int(jnp.argmax(logits(p,window)[-1]))
        # Append the current step result to `tokens`.
        tokens.append(new)
    # Return `''.join((inverse[i] for i in tokens))` to the caller.
    return "".join(inverse[i] for i in tokens)

# 9. Check the expected continuation
# Step 9 — 9. Check the expected continuation: The expected continuation is abcabcabcabc, with held-out loss...
continuation = generate(params,"abca",8)
# Print the observed values to compare against the expected result.
print("Greedy continuation:",continuation)
# Verify contract: `continuation == 'abcabcabcabc'`.
assert continuation == "abcabcabcabc"

# Step 1 — 1. Prepare packages and the vocabulary: Vocabulary size is three, model width is eight, and the context...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np
import optax
import json
from pathlib import Path
from tempfile import TemporaryDirectory

# Evaluate `VOCAB` from the current inputs and state.
VOCAB = {"a":0, "b":1, "c":2}
# Evaluate `CONTEXT` from the current inputs and state.
CONTEXT = 4
# Step 2 — 2. Reuse the verified causal block: The block returns one model-width vector per token; it has no...
def layer_norm(x):
    # Reduce along axis=-1 to compute `mean`.
    mean = jnp.mean(x, axis=-1, keepdims=True)
    # Reduce along axis=-1 to compute `variance`.
    variance = jnp.mean((x-mean)**2, axis=-1, keepdims=True)
    # Return `(x - mean) / jnp.sqrt(variance + 1e-05)` to the caller.
    return (x-mean) / jnp.sqrt(variance + 1e-5)

# Function `init_block(key, width)` implementing this stage's computation:
def init_block(key, width=8):
    # Create or split explicit PRNG key(s) (`keys`) for reproducible randomness.
    keys = jax.random.split(key, 6)
    # Evaluate `shapes` from the current inputs and state.
    shapes = [(width,width)]*4 + [(width,2*width),(2*width,width)]
    # Return `{name: jax.random.normal(k, s) * 0.1 for name, k, s in zip(['q', 'k', 'v', 'o', 'up', 'down'], keys, shapes)}` to the caller.
    return {name: jax.random.normal(k,s)*0.1 for name,k,s in zip(
        ["q","k","v","o","up","down"], keys, shapes)}

# Function `block_forward(p, x)` implementing this stage's computation:
def block_forward(p, x):
    # Run `layer_norm` to compute `h`.
    h = layer_norm(x)
    # Perform matrix contraction / projection to compute `(q, k, v)`.
    q, k, v = h@p["q"], h@p["k"], h@p["v"]
    # Perform matrix contraction / projection to compute `scores`.
    scores = q@k.T / jnp.sqrt(x.shape[-1])
    # Create evenly spaced index values in `allowed`.
    allowed = jnp.arange(x.shape[0])[:,None] >= jnp.arange(x.shape[0])[None,:]
    # Apply nonlinear activation or probability normalization to compute `weights`.
    weights = jax.nn.softmax(jnp.where(allowed, scores, -jnp.inf), axis=-1)
    # Perform matrix contraction / projection to compute `residual`.
    residual = x + (weights@v)@p["o"]
    # Return `residual + jax.nn.gelu(layer_norm(residual) @ p['up']) @ p['down']` to the caller.
    return residual + jax.nn.gelu(layer_norm(residual)@p["up"])@p["down"]

# Step 3 — 3. Add embeddings and the vocabulary head: The head has three columns; make_examples shifts each window by...
def init_lm(key):
    # Create or split explicit PRNG key(s) (`(embed_key, block_key, head_key)`) for reproducible randomness.
    embed_key, block_key, head_key = jax.random.split(key, 3)
    # Return `{'embed': jax.random.normal(embed_key, (3, 8)) * 0.1, 'position': jnp.arange(CONTEXT * 8, dtype=jnp.float32).reshape(CONTEXT, 8) * 0.001, 'block': init_block(block_key), 'head': jax.random.normal(head_key, (8, 3)) * 0.1}` to the caller.
    return {"embed":jax.random.normal(embed_key,(3,8))*0.1,
            "position":jnp.arange(CONTEXT*8,dtype=jnp.float32).reshape(CONTEXT,8)*0.001,
            "block":init_block(block_key),
            "head":jax.random.normal(head_key,(8,3))*0.1}

# Function `logits(p, tokens)` implementing this stage's computation:
def logits(p, tokens):
    # Evaluate `h` from the current inputs and state.
    h = p["embed"][tokens] + p["position"][:tokens.shape[0]]
    # Return `layer_norm(block_forward(p['block'], h)) @ p['head']` to the caller.
    return layer_norm(block_forward(p["block"],h))@p["head"]

# Function `make_examples(text)` implementing this stage's computation:
def make_examples(text):
    # Initialize array `ids` with explicit values and shape.
    ids = np.array([VOCAB[c] for c in text],dtype=np.int32)
    # Combine or mask array elements to form `rows`.
    rows = np.stack([ids[i:i+CONTEXT+1] for i in range(len(ids)-CONTEXT)])
    # Return `(jnp.array(rows[:, :-1]), jnp.array(rows[:, 1:]))` to the caller.
    return jnp.array(rows[:,:-1]), jnp.array(rows[:,1:])

# Step 4 — 4. Define the loss and compiled state transition: train_step returns new parameters, optimizer state and the...
train_x, train_y = make_examples("abc"*12)
# Run `make_examples` to compute `(held_x, held_y)`.
held_x, held_y = make_examples("bca"*6)
# Function `lm_loss(p, x, y)` implementing this stage's computation:
def lm_loss(p, x, y):
    # Vectorize across the batch dimension with `jax.vmap` (`predictions`).
    predictions = jax.vmap(logits,in_axes=(None,0))(p,x)
    # Return `jnp.mean(optax.softmax_cross_entropy_with_integer_labels(predictions, y))` to the caller.
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(predictions,y))
# Configure or step the Optax optimizer state (`optimizer`).
optimizer = optax.adam(0.03)
# Define and JIT-compile `train_step(p, state, x, y)` so XLA traces and fuses the operations:
@jax.jit
# Function `train_step(p, state, x, y)` implementing this stage's computation:
def train_step(p, state, x, y):
    # Differentiate the objective to obtain `(value, gradients)` via automatic differentiation.
    value, gradients = jax.value_and_grad(lm_loss)(p,x,y)
    # Apply the computed gradient updates to update the model parameters.
    updates, next_state = optimizer.update(gradients,state,p)
    # Return `(optax.apply_updates(p, updates), next_state, value)` to the caller.
    return optax.apply_updates(p,updates), next_state, value

# Step 5 — 5. Run the first twenty updates: Adam count should now equal twenty; initialization is the only...
# Create or split explicit PRNG key(s) (`params`) for reproducible randomness.
params = init_lm(jax.random.key(3))
# Run `optimizer.init` to compute `state`.
state = optimizer.init(params)
# Evaluate `lm_loss(params, train_x, train_y)` and convert the result into Python scalar/collection `initial`.
initial = float(lm_loss(params,train_x,train_y))
# Repeat the update loop over `range(20)` steps:
for _ in range(20):
    # Run `train_step` to compute `(params, state, _)`.
    params, state, _ = train_step(params,state,train_x,train_y)
# Save numeric leaves only, with a schema checked against this program's template.
# Step 6 — 6. Write and validate the snapshot format: The loader checks architecture, vocabulary, numeric leaf...
def save_snapshot(folder, p, state, step):
    # Run `jax.tree.flatten` to compute `(leaves, _)`.
    leaves, _ = jax.tree.flatten((p,state))
    # Convert `` to a host NumPy array for inspection or verification.
    np.savez(folder/"state.npz", **{f"leaf_{i}":np.asarray(x) for i,x in enumerate(leaves)})
    # Evaluate `manifest` from the current inputs and state.
    manifest = {"schema":1,"step":step,"vocabulary":VOCAB,"context":CONTEXT,
                "shapes":[list(x.shape) for x in leaves],
                "dtypes":[str(x.dtype) for x in leaves]}
    # Read or serialize artifact data on disk (``).
    (folder/"manifest.json").write_text(json.dumps(manifest))

# Function `load_snapshot(folder)` implementing this stage's computation:
def load_snapshot(folder):
    # Read or serialize artifact data on disk (`meta`).
    meta = json.loads((folder/"manifest.json").read_text())
    # Guard input contract (`meta['schema'] != 1 or meta['vocabulary'] != VOCAB or meta['context'] != CONTEXT`) and fail fast if violated.
    if meta["schema"] != 1 or meta["vocabulary"] != VOCAB or meta["context"] != CONTEXT:
        raise ValueError("checkpoint architecture/vocabulary mismatch")
    # Guard input contract (`type(meta['step']) is not int or meta['step'] < 0`) and fail fast if violated.
    if type(meta["step"]) is not int or meta["step"] < 0:
        raise ValueError("invalid checkpoint step")
    # Create or split explicit PRNG key(s) (`template`) for reproducible randomness.
    template = init_lm(jax.random.key(0))
    # Run `jax.tree.flatten` to compute `(template_leaves, structure)`.
    template_leaves, structure = jax.tree.flatten((template,optimizer.init(template)))
    # Evaluate `arrays` from the current inputs and state.
    arrays = []
    # Enter managed runtime/context scope for this block:
    with np.load(folder/"state.npz",allow_pickle=False) as data:
        # Guard input contract (`set(data.files) != {f'leaf_{i}' for i in range(len(template_leaves))}`) and fail fast if violated.
        if set(data.files) != {f"leaf_{i}" for i in range(len(template_leaves))}:
            raise ValueError("checkpoint leaf count mismatch")
        # Loop over `(i, leaf)` in `enumerate(template_leaves)`:
        for i, leaf in enumerate(template_leaves):
            # Evaluate `saved` from the current inputs and state.
            saved = data[f"leaf_{i}"]
            # Guard input contract (`saved.shape != leaf.shape or saved.dtype != np.asarray(leaf).dtype`) and fail fast if violated.
            if saved.shape != leaf.shape or saved.dtype != np.asarray(leaf).dtype:
                raise ValueError("checkpoint leaf contract mismatch")
            # Create device-backed JAX array ``.
            arrays.append(jnp.array(saved))
    # Run `jax.tree.unflatten` to compute `(restored_p, restored_state)`.
    restored_p, restored_state = jax.tree.unflatten(structure,arrays)
    # Guard input contract (`int(restored_state[0].count) != meta['step']`) and fail fast if violated.
    if int(restored_state[0].count) != meta["step"]:
        raise ValueError("checkpoint step and Adam count disagree")
    # Return `(restored_p, restored_state, meta['step'])` to the caller.
    return restored_p, restored_state, meta["step"]

# Step 7 — 7. Verify resume and evaluate the final fit: Uninterrupted and restored next transitions agree; the held-out...
# Create an isolated temporary directory to run and inspect artifacts safely:
with TemporaryDirectory() as directory:
    # Read or serialize artifact data on disk (`folder`).
    folder = Path(directory)
    # Run `save_snapshot` to perform the next check or state transition.
    save_snapshot(folder,params,state,20)
    # Run `load_snapshot` to compute `(restored, restored_state, step)`.
    restored, restored_state, step = load_snapshot(folder)
    # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
    assert step == 20
    # Run `train_step` to compute `(next_a, state_a, _)`.
    next_a, state_a, _ = train_step(params,state,train_x,train_y)
    # Run `train_step` to compute `(next_b, state_b, _)`.
    next_b, state_b, _ = train_step(restored,restored_state,train_x,train_y)
    # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
    assert all(jnp.allclose(a,b,atol=1e-7) for a,b in zip(
        jax.tree.leaves((next_a,state_a)),jax.tree.leaves((next_b,state_b))))
# Repeat the update loop over `range(80)` steps:
for _ in range(80):
    # Run `train_step` to compute `(params, state, _)`.
    params, state, _ = train_step(params,state,train_x,train_y)
# Vectorize across the batch dimension with `jax.vmap` (`held_logits`).
held_logits = jax.vmap(logits,in_axes=(None,0))(params,held_x)
# Reduce along axis=-1 to compute `accuracy`.
accuracy = jnp.mean(jnp.argmax(held_logits,axis=-1)==held_y)
# Run `lm_loss` to compute `final_loss`.
final_loss = lm_loss(params,held_x,held_y)
# Print the observed values to compare against the expected result.
print("Initial / held loss / held accuracy:",initial,float(final_loss),float(accuracy))
# Verify contract: `final_loss < 0.05`.
assert final_loss < 0.05
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert accuracy > 0.99

# Step 8 — 8. Define greedy decoding: The generator recomputes a short window each step.
def generate(p, prompt, count):
    # Evaluate `tokens` from the current inputs and state.
    tokens = [VOCAB[c] for c in prompt]
    # Guard input contract (`not tokens`) and fail fast if violated.
    if not tokens:
        raise ValueError("generation needs a nonempty prompt")
    # Evaluate `inverse` from the current inputs and state.
    inverse = {i:c for c,i in VOCAB.items()}
    # Repeat the update loop over `range(count)` steps:
    for _ in range(count):
        # Initialize array `window` with explicit values and shape.
        window = jnp.array(tokens[-CONTEXT:],dtype=jnp.int32)
        # Evaluate `jnp.argmax(logits(p, window)[-1])` and convert the result into Python scalar/collection `new`.
        new = int(jnp.argmax(logits(p,window)[-1]))
        # Append the current step result to `tokens`.
        tokens.append(new)
    # Return `''.join((inverse[i] for i in tokens))` to the caller.
    return "".join(inverse[i] for i in tokens)

# Step 9 — 9. Check the expected continuation: The expected continuation is abcabcabcabc, with held-out loss...
continuation = generate(params,"abca",8)
# Print the observed values to compare against the expected result.
print("Greedy continuation:",continuation)
# Verify contract: `continuation == 'abcabcabcabc'`.
assert continuation == "abcabcabcabc"

# Figure data experiment
# Compute figure data for: Inspect next-token probabilities before greedy decoding
# Apply nonlinear activation or probability normalization to compute `probs`.
probs = jax.nn.softmax(held_logits[0], axis=-1)
# Evaluate `visual_data` from the current inputs and state.
visual_data = {'kind': 'heatmap', 'values': probs.tolist(), 'rows': ['position ' + str(i) for i in range(CONTEXT)], 'columns': list(VOCAB.keys()), 'unit': 'next-token probability'}

# Experiment: Verify uniform-logit cross entropy
# Experiment — Verify uniform-logit cross entropy: A known probability distribution independently checks target...
# Initialize array `zero_head` with explicit values and shape.
zero_head = {**params,"head":jnp.zeros_like(params["head"])}
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(lm_loss(zero_head,train_x,train_y),jnp.log(3.),atol=1e-6)

# Experiment: Test causality at the final model output
# Experiment — Test causality at the final model output: Causal independence should hold for the full model, not only the...
# Initialize array `example` with explicit values and shape.
example = jnp.array([0,1,2,0])
# Run `logits` to compute `original`.
original = logits(params,example)
# Run `logits` to compute `modified`.
modified = logits(params,example.at[-1].set(1))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(original[:3],modified[:3],atol=1e-5)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Generate twelve new tokens from prompt bcab.
# Verify contract: `generate(params, 'bcab', 12) == 'bcabcabcabcabcab'`.
assert generate(params,"bcab",12) == "bcabcabcabcabcab"

# Reference practice: Reject an incompatible checkpoint
# Reject an incompatible checkpoint (Transfer / diagnosis): The metadata is part of the model contract even when...
# Create an isolated temporary directory to run and inspect artifacts safely:
with TemporaryDirectory() as directory:
    # Read or serialize artifact data on disk (`folder`).
    folder = Path(directory)
    # Run `save_snapshot` to perform the next check or state transition.
    save_snapshot(folder,params,state,100)
    # Evaluate `metadata_path` from the current inputs and state.
    metadata_path = folder/"manifest.json"
    # Read or serialize artifact data on disk (`correct`).
    correct = metadata_path.read_text()
    # Read or serialize artifact data on disk (`metadata`).
    metadata = json.loads(correct)
    # Evaluate `metadata['vocabulary']` from the current inputs and state.
    metadata["vocabulary"] = {"a":1,"b":0,"c":2}
    # Read or serialize artifact data on disk (``).
    metadata_path.write_text(json.dumps(metadata))
    try:
        load_snapshot(folder)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong vocabulary accepted")
    # Read or serialize artifact data on disk (`forged_step`).
    forged_step = json.loads(correct)
    # Evaluate `forged_step['step']` from the current inputs and state.
    forged_step["step"] = 101
    # Read or serialize artifact data on disk (``).
    metadata_path.write_text(json.dumps(forged_step))
    try:
        load_snapshot(folder)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong step accepted")
    # Read or serialize artifact data on disk (``).
    metadata_path.write_text(correct)
    # Run `load_snapshot` to compute `(recovered, _, recovered_step)`.
    recovered, _, recovered_step = load_snapshot(folder)
    # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
    assert recovered_step == 100
    # Verify that the output satisfies the expected shape, finite-value, or numerical contract.
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
# Verify contract: `generate(params, 'abca', 0) == 'abca'`.
assert generate(params,"abca",0) == "abca"
print("PASS: transformers-04")
