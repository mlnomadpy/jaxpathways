"""Supervised fine-tuning with response-only token loss: worked experiments and reference solutions. CPU checks."""

# 1. Define sequence log-probability sums
# Step 1 — 1. Define sequence log-probability sums: The slice removes the final logit and first token.
import jax
import jax.numpy as jnp
import numpy as np

def response_logps(logits, tokens, response_mask):
    # logits at t predict token t+1; the role mask belongs to the target token.
    logp = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
    selected = jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1)[..., 0]
    return jnp.sum(jnp.where(response_mask[:, 1:], selected, 0.0), axis=-1)

def validate_mask(selected, shape):
    # Transfer the device result to the host for an independent NumPy comparison.
    a = np.asarray(selected)
    if a.shape != shape or a.dtype != np.bool_ or not a.any():
        raise ValueError('a nonempty Boolean mask of the target shape is required')

# 2. Align prompt, response and target masks
# Token 0/1 is a prompt; 2/3 is its response; 4 marks the end.
tokens = jnp.array([[0, 2, 4], [1, 3, 4]], jnp.int32)
roles = jnp.array([[False, True, True]] * 2)
validate_mask(roles[:, 1:], tokens[:, 1:].shape)
p = jnp.zeros((5, 5))
history = []

def sft_objective(p):
    return -jnp.sum(response_logps(p[tokens], tokens, roles)) / jnp.sum(roles[:, 1:])

step = jax.jit(jax.value_and_grad(sft_objective))

# 3. Fit the transition table and check the final position
# Step 3 — 3. Fit the transition table and check the final position: The table learns only local transitions.
for _ in range(100):
    value, g = step(p)
    history.append(float(value))
    p = p - 0.5 * g

np.testing.assert_allclose(history[0], np.log(5), atol=1e-6)
assert history[-1] < 0.15
assert np.array_equal(np.argmax(np.asarray(p)[[0, 1]], axis=1), [2, 3])
# The final position predicts nothing and has no contribution.
base_logits = p[tokens]
changed = base_logits.at[:, -1, :].set(100.0)
np.testing.assert_allclose(
    response_logps(changed, tokens, roles),
    response_logps(base_logits, tokens, roles),
)
print('Response-only token NLL initial/final:', history[0], history[-1])

# Step 1 — 1. Define sequence log-probability sums: The slice removes the final logit and first token.
import jax
import jax.numpy as jnp
import numpy as np

def response_logps(logits, tokens, response_mask):
    # logits at t predict token t+1; the role mask belongs to the target token.
    logp = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
    selected = jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1)[..., 0]
    return jnp.sum(jnp.where(response_mask[:, 1:], selected, 0.0), axis=-1)

def validate_mask(selected, shape):
    # Transfer the device result to the host for an independent NumPy comparison.
    a = np.asarray(selected)
    if a.shape != shape or a.dtype != np.bool_ or not a.any():
        raise ValueError('a nonempty Boolean mask of the target shape is required')

# Token 0/1 is a prompt; 2/3 is its response; 4 marks the end.
tokens = jnp.array([[0, 2, 4], [1, 3, 4]], jnp.int32)
roles = jnp.array([[False, True, True]] * 2)
validate_mask(roles[:, 1:], tokens[:, 1:].shape)
p = jnp.zeros((5, 5))
history = []

def sft_objective(p):
    return -jnp.sum(response_logps(p[tokens], tokens, roles)) / jnp.sum(roles[:, 1:])

step = jax.jit(jax.value_and_grad(sft_objective))

# Step 3 — 3. Fit the transition table and check the final position: The table learns only local transitions.
for _ in range(100):
    value, g = step(p)
    history.append(float(value))
    p = p - 0.5 * g

np.testing.assert_allclose(history[0], np.log(5), atol=1e-6)
assert history[-1] < 0.15
assert np.array_equal(np.argmax(np.asarray(p)[[0, 1]], axis=1), [2, 3])
# The final position predicts nothing and has no contribution.
base_logits = p[tokens]
changed = base_logits.at[:, -1, :].set(100.0)
np.testing.assert_allclose(
    response_logps(changed, tokens, roles),
    response_logps(base_logits, tokens, roles),
)
print('Response-only token NLL initial/final:', history[0], history[-1])

# Figure data experiment
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'response-token NLL (nats)','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']

initial_output_grads=jax.grad(lambda values:-jnp.sum(response_logps(values,tokens,roles))/jnp.sum(roles[:,1:]))(jnp.zeros((2,3,5)))
# Transfer the device result to the host for an independent NumPy comparison.
extra_panel={'kind':'heatmap','values':np.asarray(jnp.linalg.norm(initial_output_grads,axis=-1)).tolist(),'rows':['prompt 0 / answer 2','prompt 1 / answer 3'],'columns':['input position 0','input position 1','input position 2'],'unit':'output-logit gradient L2 norm','xlabel':'position producing the prediction','ylabel':'sequence','title':'Initial supervised gradient by output position'}
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}

# Experiment: Prove the target-mask shift
# Experiment — Prove the target-mask shift
g = jax.grad(sft_objective)(jnp.zeros((5, 5)))
assert np.linalg.norm(np.asarray(g)[0]) > 0 and np.linalg.norm(np.asarray(g)[1]) > 0
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_array_equal(np.asarray(g)[4], 0.0)
print('Prompt rows learn to predict first answers; the final end-token row has no target.')

# Experiment: Check a ragged batch against a scalar loop
# Experiment — Check a ragged batch against a scalar loop: Padding and role masks affect the target after shifting.
ragged_tokens = jnp.array([[0, 2, 4, 4], [1, 3, 2, 4]])
ragged_roles = jnp.array([[False, True, True, False], [False, True, True, True]])
ragged_logits = jnp.arange(2 * 4 * 5, dtype=jnp.float32).reshape(2, 4, 5) / 19
observed = response_logps(ragged_logits, ragged_tokens, ragged_roles)
# Transfer the device result to the host for an independent NumPy comparison.
arr = np.asarray(ragged_logits, dtype=np.float64)
expected = []
for row in range(2):
    total = 0.0
    for position in range(3):
        if ragged_roles[row, position + 1]:
            values = arr[row, position]
            log_normalizer = values.max() + np.log(np.exp(values - values.max()).sum())
            total += values[int(ragged_tokens[row, position + 1])] - log_normalizer
    expected.append(total)
np.testing.assert_allclose(observed, expected, atol=1e-6)
print('Supervised targets per sequence:', np.asarray(ragged_roles[:, 1:].sum(1)))

# Reference solution. Try the exercise before reading this.
# Exercise solution
first_only = roles.at[:, 2].set(False)
value = -jnp.sum(response_logps(p[tokens], tokens, first_only)) / jnp.sum(
    first_only[:, 1:]
)
assert int(jnp.sum(first_only[:, 1:])) == 2
# Ensure all array elements remain finite: `np.isfinite(float(value))`
assert np.isfinite(float(value))
print('Two first-response targets:', float(value))

# Reference practice: Reject a batch with no answer targets
# Reject a batch with no answer targets (Transfer)
try:
    validate_mask(jnp.zeros_like(roles[:, 1:]), tokens[:, 1:].shape)
except ValueError:
    print('No-target SFT batch rejected')
else:
    raise AssertionError('empty supervision accepted')

# Reference practice: Create a failure this model cannot fix
# Create a failure this model cannot fix (Challenge)
contexts = jnp.array([[1, 0], [3, 0]])
last_logits = p[contexts[:, -1]]
np.testing.assert_array_equal(last_logits[0], last_logits[1])
conflicting_targets = jnp.array([2, 3])
assert conflicting_targets[0] != conflicting_targets[1]
print('Identical final tokens force identical table predictions despite different contexts.')

# Reference practice: Run one response-masked SFT update from the saved foundations decoder
# Full-workspace saved-decoder SFT transfer. Requires the pinned foundation package and requirements.
import hashlib, json, sys, tempfile
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from tokenizers import Tokenizer

ROOT = Path.cwd()
FOUNDATIONS = ROOT / "projects/text-harness/foundations"
PACKAGE = FOUNDATIONS / "reference/run/inference-package"
sys.path.insert(0, str(FOUNDATIONS))
import model as decoder
import train
manifest = json.loads((PACKAGE / "package.json").read_text())
expected_source_hashes = {'model.py': '94a88ae7fd6a251b5b190b4631496e04bb71a31c1bfe418511fdd37ffef034c9', 'config.py': '5c05801a7ec1a15b3978c87c77ac4e64d1b056c60487cbcff3cdf72475c8bb64', 'train.py': 'ee9f3bc73505ce8e00fe9680589d97805593e5c8c035cba4709b5f334eb27a2b'}
for name, expected in expected_source_hashes.items():
    actual = hashlib.sha256((FOUNDATIONS / name).read_bytes()).hexdigest()
    assert actual == expected == manifest["identity"]["code_sha256"][name], (name, actual)

assert manifest["params_sha256"] == "271cb6457de5da3f727c343c64b7bd350c00e503ea7ac847eecf51fbd5558802"
assert manifest["processor_sha256"] == "c92b534443cc6a1eacd5e78406b2c42e59553df704a07102a03c639d3694f586"
params, config, loaded = train.load_model_package(PACKAGE, manifest["identity"])
assert hashlib.sha256((PACKAGE / "tokenizer.json").read_bytes()).hexdigest() == loaded["processor_sha256"]
tokenizer = Tokenizer.from_file(str(PACKAGE / "tokenizer.json"))
parent = jax.tree.map(lambda leaf: jnp.array(leaf), params)
parent_snapshot = [np.asarray(leaf).copy() for leaf in jax.tree.leaves(parent)]
PAD, BOS, EOS = 0, 1, 2

def encode(text):
    return tokenizer.encode(text, add_special_tokens=False).ids

def packed(prompt, response):
    prefix = [BOS] + encode(prompt)
    answer = encode(response) + [EOS]
    ids = prefix + answer
    if len(ids) > config.context:
        raise ValueError("fixture sequence exceeds saved model context")
    tokens = np.full((1, config.context), PAD, np.int32)
    tokens[0, :len(ids)] = ids
    target_positions = np.arange(1, config.context)
    # Mask belongs to targets at positions 1..T-1, not input positions 0..T-2.
    target_mask = (target_positions >= len(prefix)) & (target_positions < len(ids))
    target_mask &= tokens[0, 1:] != PAD
    return jnp.asarray(tokens), jnp.asarray(target_mask), len(answer)

def sequence_nll(model_params, prompt, response):
    tokens, mask, count = packed(prompt, response)
    logits = decoder.logits(model_params, tokens, config, valid_mask=tokens != PAD)
    log_probs = jax.nn.log_softmax(logits[:, :-1].astype(jnp.float32), axis=-1)
    targets = tokens[:, 1:]
    selected = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    return -jnp.sum(jnp.where(mask[None, :], selected, 0.0)) / count

train_pairs = [("Q: ", "yes"), ("A: ", "help")]
heldout = [("R: ", "clear"), ("B: ", "unknown")]
train_prompt_groups = {tuple([BOS] + encode(prompt)) for prompt, _ in train_pairs}
heldout_prompt_groups = {tuple([BOS] + encode(prompt)) for prompt, _ in heldout}
assert train_prompt_groups.isdisjoint(heldout_prompt_groups)
heldout_before = np.asarray([float(sequence_nll(parent, p, r)) for p,r in heldout])
# One learner-controlled change: the declared SFT step size. Optimizer state is new.
learning_rate = 1e-4

def objective(candidate):
    return jnp.mean(jnp.stack([sequence_nll(candidate, p, r) for p,r in train_pairs]))

train_before = float(objective(params))
loss, grads = jax.value_and_grad(objective)(params)
params = jax.tree.map(lambda w,g: w - learning_rate*g, params, grads)
train_after = float(objective(params))
heldout_after = np.asarray([float(sequence_nll(params, p, r)) for p,r in heldout])
for before, after in zip(heldout_before, heldout_after):
    assert np.isfinite(before) and np.isfinite(after)
np.testing.assert_array_equal(params["token_embed"].shape, parent["token_embed"].shape)
assert all(np.array_equal(saved, np.asarray(leaf)) for saved,leaf in zip(parent_snapshot, jax.tree.leaves(parent)))

# Save an inference child with numeric state and an explicit parent/tokenizer/config binding.
with tempfile.TemporaryDirectory() as folder:
    output = Path(folder)
    leaves, _ = jax.tree.flatten(params)
    np.savez(output / "child-params.npz", **{f"leaf_{i:04d}": np.asarray(x) for i,x in enumerate(leaves)})
    child_bytes = (output / "child-params.npz").read_bytes()
    child_manifest = {
        "schema": 1, "parent_params_sha256": manifest["params_sha256"],
        "tokenizer_sha256": manifest["processor_sha256"],
        "model_config_sha256": manifest["model_config_sha256"],
        "child_params_sha256": hashlib.sha256(child_bytes).hexdigest(),
        "posttraining_stage": "single_response_masked_sft_step",
        "learning_rate": learning_rate, "optimizer_state": "new; not included",
        "train_examples": len(train_pairs), "heldout_examples": len(heldout),
    }
    (output / "manifest.json").write_text(json.dumps(child_manifest, sort_keys=True, indent=2))
    saved = json.loads((output / "manifest.json").read_text())
    assert saved["parent_params_sha256"] == manifest["params_sha256"]
    assert saved["child_params_sha256"] == hashlib.sha256((output / "child-params.npz").read_bytes()).hexdigest()
print("saved decoder identity:", manifest["params_sha256"], manifest["processor_sha256"])
print("response target count includes EOS and excludes prompt/PAD; example counts:", [packed(p,r)[2] for p,r in train_pairs])
print("one SFT step train NLL before/after:", train_before, train_after)
print("heldout response NLL baseline/after:", heldout_before.tolist(), heldout_after.tolist())
print("child inference bundle parent-bound; no optimizer resume or quality claim")
print("PASS: posttraining-01")
