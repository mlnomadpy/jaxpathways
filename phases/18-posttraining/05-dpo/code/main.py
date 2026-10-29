"""Direct preference optimization and reference-corrected margins: worked experiments and reference solutions. CPU checks."""

# 1. Define the reference-corrected pair objective
# Step 1 — 1. Define the reference-corrected pair objective
import jax
import jax.numpy as jnp
import numpy as np

def dpo_loss(policy_logps, reference_logps, chosen, rejected, beta=0.2):
    margin = (policy_logps[chosen] - policy_logps[rejected]) - jax.lax.stop_gradient(
        reference_logps[chosen] - reference_logps[rejected]
    )
    return jnp.mean(jax.nn.softplus(-beta * margin))

# 2. Freeze reference probabilities and comparison IDs
# Step 2 — 2. Freeze reference probabilities and comparison IDs: The initial policy equals a nonuniform reference.
reference = jax.nn.log_softmax(jnp.array([0.2, 0.0, -0.2]))
chosen = jnp.array([0, 0, 1])
rejected = jnp.array([1, 2, 2])
theta = jnp.array([0.2, 0.0, -0.2])
history = []
loss = lambda theta: dpo_loss(
    jax.nn.log_softmax(theta), reference, chosen, rejected, 0.3
)
step = jax.jit(jax.value_and_grad(loss))
np.testing.assert_allclose(loss(theta), np.log(2), atol=1e-6)

# 3. Fit the policy and test cancellation
# Step 3 — 3. Fit the policy and test cancellation: The host stable-softplus calculation checks final margins.
for _ in range(120):
    value, g = step(theta)
    history.append(float(value))
    theta = theta - 0.4 * g

assert history[-1] < history[0] * 0.5
policy = jax.nn.log_softmax(theta)
# Transfer the device result to the host for an independent NumPy comparison.
margin = np.asarray(
    (policy[chosen] - policy[rejected]) - (reference[chosen] - reference[rejected]),
    np.float64,
)
np.testing.assert_allclose(
    loss(theta), np.mean(np.logaddexp(0, -0.3 * margin)), atol=1e-6
)
assert np.all(margin > 0)
np.testing.assert_allclose(loss(theta + 50), loss(theta), atol=1e-6)
print(
    'DPO initial/final:',
    history[0],
    history[-1],
    '; reference-corrected margins:',
    margin,
)

# Step 1 — 1. Define the reference-corrected pair objective
import jax
import jax.numpy as jnp
import numpy as np

def dpo_loss(policy_logps, reference_logps, chosen, rejected, beta=0.2):
    margin = (policy_logps[chosen] - policy_logps[rejected]) - jax.lax.stop_gradient(
        reference_logps[chosen] - reference_logps[rejected]
    )
    return jnp.mean(jax.nn.softplus(-beta * margin))

# Step 2 — 2. Freeze reference probabilities and comparison IDs: The initial policy equals a nonuniform reference.
reference = jax.nn.log_softmax(jnp.array([0.2, 0.0, -0.2]))
chosen = jnp.array([0, 0, 1])
rejected = jnp.array([1, 2, 2])
theta = jnp.array([0.2, 0.0, -0.2])
history = []
loss = lambda theta: dpo_loss(
    jax.nn.log_softmax(theta), reference, chosen, rejected, 0.3
)
step = jax.jit(jax.value_and_grad(loss))
np.testing.assert_allclose(loss(theta), np.log(2), atol=1e-6)

# Step 3 — 3. Fit the policy and test cancellation: The host stable-softplus calculation checks final margins.
for _ in range(120):
    value, g = step(theta)
    history.append(float(value))
    theta = theta - 0.4 * g

assert history[-1] < history[0] * 0.5
policy = jax.nn.log_softmax(theta)
# Transfer the device result to the host for an independent NumPy comparison.
margin = np.asarray(
    (policy[chosen] - policy[rejected]) - (reference[chosen] - reference[rejected]),
    np.float64,
)
np.testing.assert_allclose(
    loss(theta), np.mean(np.logaddexp(0, -0.3 * margin)), atol=1e-6
)
assert np.all(margin > 0)
np.testing.assert_allclose(loss(theta + 50), loss(theta), atol=1e-6)
print(
    'DPO initial/final:',
    history[0],
    history[-1],
    '; reference-corrected margins:',
    margin,
)

# Figure data experiment
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'DPO preference loss (nats)','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']

extra_panel={'kind':'bar','x':[0,1,2],'labels':['0 preferred to 1','0 preferred to 2','1 preferred to 2'],'series':[{'label':'final corrected margin','y':margin.tolist()}],'xlabel':'preference pair','ylabel':'reference-corrected log ratio','title':'Relative preference changes behind the loss'}
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}

# Experiment: Verify the reference cancellation
# Experiment — Verify the reference cancellation: The initial policy is not uniform.
initial = dpo_loss(reference, reference, chosen, rejected, 0.3)
np.testing.assert_allclose(initial, np.log(2), atol=1e-6)
uncorrected = jnp.mean(
    jax.nn.softplus(-0.3 * (reference[chosen] - reference[rejected]))
)
assert not np.isclose(float(initial), float(uncorrected))
print('Corrected/unadjusted initial loss:', float(initial), float(uncorrected))

# Experiment: Improve the ratio while lowering the chosen probability
# Experiment — Improve the ratio while lowering the chosen probability: The rejected probability falls further.
example_ref = jnp.log(jnp.array([0.4, 0.4, 0.2]))
example_policy = jnp.log(jnp.array([0.3, 0.1, 0.6]))
choice = jnp.array([0])
reject = jnp.array([1])
before = float(dpo_loss(example_ref, example_ref, choice, reject, 0.3))
after = float(dpo_loss(example_policy, example_ref, choice, reject, 0.3))
assert after < before and float(jnp.exp(example_policy[0])) < float(
    jnp.exp(example_ref[0])
)
np.testing.assert_allclose(after, np.logaddexp(0.0, -0.3 * np.log(3)), atol=1e-6)
print('Loss before/after:', before, after, '; chosen probability: 0.4 -> 0.3')

# Reference solution. Try the exercise before reading this.
# Exercise solution
reverse = float(dpo_loss(policy, reference, rejected, chosen, 0.3))
assert reverse > float(dpo_loss(policy, reference, chosen, rejected, 0.3))
print('Reversed-label loss:', reverse)

# Reference practice: Freeze the reference numerically and in autodiff
# Freeze the reference numerically and in autodiff (Transfer)
reference_grad = jax.grad(dpo_loss, argnums=1)(
    policy, reference, chosen, rejected, 0.3
)
np.testing.assert_array_equal(
    reference_grad, np.zeros(reference_grad.shape)
)
print('Reference gradient is zero.')

# Reference practice: Derive a one-pair categorical gradient
# Derive a one-pair categorical gradient (Challenge)
probe = jnp.array([0.6, -0.3, 0.1])
choice = jnp.array([0])
reject = jnp.array([1])
beta = 0.3
one_pair = lambda logits: dpo_loss(
    jax.nn.log_softmax(logits), reference, choice, reject, beta
)
margin_value = float((probe[0] - probe[1]) - (reference[0] - reference[1]))
factor = -beta / (1 + np.exp(beta * margin_value))
np.testing.assert_allclose(
    jax.grad(one_pair)(probe), [factor, -factor, 0.0], atol=1e-6
)
print('One-pair logit gradient agrees with independent margin derivative.')

# Reference practice: Full-workspace transfer: score saved sequences and run one preference update
# Saved-sequence preference handoff. Run from the repository root in the foundations CPU environment.
import hashlib, json, sys
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
tokenizer = Tokenizer.from_file(str(PACKAGE / "tokenizer.json"))
assert hashlib.sha256((PACKAGE / "tokenizer.json").read_bytes()).hexdigest() == loaded["processor_sha256"]
reference = jax.tree.map(lambda x: jnp.array(x), params)
PAD, EOS = 0, 2

def encode(text):
    return tokenizer.encode(text, add_special_tokens=False).ids

def score_jax(model_params, prompt, response):
    prefix = [1] + encode(prompt)  # Explicit BOS; exact prefix IDs are part of the example.
    answer = encode(response) + [EOS]
    full = prefix + answer
    if len(full) > config.context:
        raise ValueError("fixture example exceeds the saved model context")
    padded = np.full((1, config.context), PAD, dtype=np.int32)
    padded[0, :len(full)] = full
    valid = padded != PAD
    logits = decoder.logits(model_params, jnp.asarray(padded), config,
                            valid_mask=jnp.asarray(valid))
    targets = jnp.asarray(padded[:, 1:])
    log_probs = jax.nn.log_softmax(logits[:, :-1].astype(jnp.float32), axis=-1)
    selected = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0][0]
    target_positions = jnp.arange(1, padded.shape[1])
    mask = (target_positions >= len(prefix)) & (target_positions < len(full))
    mask = mask & (targets[0] != PAD)
    return jnp.sum(jnp.where(mask, selected, 0.0))

def score_one(model_params, prompt, response):
    prefix = [1] + encode(prompt)  # Explicit BOS; exact prefix IDs are part of the example.
    answer = encode(response) + [EOS]
    full = prefix + answer
    if len(full) > config.context:
        raise ValueError("fixture example exceeds the saved model context")
    padded = np.full((1, config.context), PAD, dtype=np.int32)
    padded[0, :len(full)] = full
    valid = padded != PAD
    logits = decoder.logits(model_params, jnp.asarray(padded), config,
                            valid_mask=jnp.asarray(valid))
    target_ids = padded[:, 1:]
    logp = np.asarray(jax.nn.log_softmax(logits[:, :-1].astype(jnp.float32), axis=-1))
    selected = np.take_along_axis(logp, target_ids[..., None], axis=-1)[..., 0][0]
    target_positions = np.arange(1, padded.shape[1])
    response_mask = (target_positions >= len(prefix)) & (target_positions < len(full))
    response_mask &= target_ids[0] != PAD
    return float(np.sum(selected[response_mask], dtype=np.float64)), selected, response_mask, full

# Hand check and independent NumPy aggregation on an unequal-length pair.
prompt = "Q: "
short = score_one(reference, prompt, "yes")
long = score_one(reference, prompt, "I do not know")
assert int(short[2].sum()) == len(encode("yes")) + 1
assert int(long[2].sum()) == len(encode("I do not know")) + 1
np.testing.assert_allclose(short[0], np.sum(short[1][short[2]], dtype=np.float64), atol=1e-6)
# Independently recompute log-softmax and the shifted sum in host NumPy.
def numpy_reference_score(prompt_text, response_text):
    prefix_ids = [1] + encode(prompt_text)
    answer_ids = encode(response_text) + [EOS]
    tokens = prefix_ids + answer_ids
    batch = np.full((1, config.context), PAD, dtype=np.int32)
    batch[0, :len(tokens)] = tokens
    logits = np.asarray(decoder.logits(reference, jnp.asarray(batch), config,
                                       valid_mask=jnp.asarray(batch != PAD)), dtype=np.float64)[0, :-1]
    targets = batch[0, 1:]
    positions = np.arange(1, batch.shape[1])
    mask = (positions >= len(prefix_ids)) & (positions < len(tokens)) & (targets != PAD)
    terms=[]
    for t in np.flatnonzero(mask):
        row=logits[t]
        peak=np.max(row)
        terms.append(row[targets[t]] - peak - np.log(np.exp(row - peak).sum()))
    return float(np.sum(terms, dtype=np.float64)), terms, mask
numpy_short, hand_terms, independent_mask = numpy_reference_score(prompt, "yes")
np.testing.assert_array_equal(independent_mask, short[2])
np.testing.assert_allclose(short[0], numpy_short, rtol=1e-6, atol=1e-6)
print("hand sum terms:", np.round(hand_terms, 6).tolist(), "sum:", float(np.sum(hand_terms)))
# A deliberately shifted-mask canary: position len(prefix)-1 predicts the first answer token.
assert short[2][len([1] + encode(prompt)) - 1]
assert not short[2][0]  # The first shifted target is still inside the prompt.

train_pairs = [
    ("Q: ", "yes", "no"),
    ("A: ", "I can help", "I cannot tell"),
]
# Freeze before-update baselines on separate held-out preference cases and a source-retention case.
heldout = [
    ("R: ", "clear", "unclear", 1.0, "unanimous"),
    ("R: ", "unknown", "maybe", 0.5, "disagreement"),
]
retention = [("Alice", "yes")]

def evaluate(model_params, pairs):
    rows=[]
    for p, good, bad, vote, slice_name in pairs:
        good_lp=score_one(model_params,p,good)[0]
        bad_lp=score_one(model_params,p,bad)[0]
        margin = good_lp - bad_lp
        predicted_probability = 1.0 / (1.0 + np.exp(-margin))
        predicted_chosen = float(predicted_probability > 0.5)
        hard_accuracy = None if vote == 0.5 else float(predicted_chosen == float(vote > 0.5))
        expected_annotator_agreement = vote * predicted_chosen + (1.0 - vote) * (1.0 - predicted_chosen)
        rows.append({"margin":margin,"predicted_probability":predicted_probability,
                     "vote_fraction":vote,"slice":slice_name,"ambiguous":vote == 0.5,
                     "hard_accuracy":hard_accuracy,
                     "expected_annotator_agreement":expected_annotator_agreement})
    return rows

def retention_score(model_params):
    return float(np.mean([score_one(model_params,p,r)[0] for p,r in retention]))

heldout_before=evaluate(reference, heldout)
retention_before=retention_score(reference)
train_scores=[(score_one(params,p,a)[0],score_one(params,p,b)[0]) for p,a,b in train_pairs]

def dpo_loss(candidate):
    margins=[]
    for p,good,bad in train_pairs:
        pi_good=score_jax(candidate,p,good)
        pi_bad=score_jax(candidate,p,bad)
        ref_good=score_jax(reference,p,good)
        ref_bad=score_jax(reference,p,bad)
        margins.append((pi_good-pi_bad)-(ref_good-ref_bad))
    return jnp.mean(jax.nn.softplus(-0.1*jnp.stack(margins)))

loss, grads = jax.value_and_grad(dpo_loss)(params)
# One small, declared learner-controlled update; reference and held-out records are unchanged.
params = jax.tree.map(lambda w,g: w - 1e-4*g, params, grads)
heldout_after=evaluate(params, heldout)
retention_after=retention_score(params)
# NumPy is the independent host aggregation oracle, not a separate model implementation.
np.testing.assert_allclose(float(loss), float(np.logaddexp(0, -0.1*np.zeros(2)).mean()), atol=1e-5)
assert all(np.isfinite(row["margin"]) for row in heldout_after)
assert all(row["hard_accuracy"] is None and row["ambiguous"] for row in heldout_after if row["vote_fraction"] == 0.5)
print("scoring: summed response logps", short[0], long[0], "; response token counts", int(short[2].sum()), int(long[2].sum()))
print("DPO loss before one update:", float(loss))
print("heldout preference baseline/after:", heldout_before, heldout_after)
print("disagreement slice baseline/after:", [x for x in heldout_before if x['slice']=='disagreement'], [x for x in heldout_after if x['slice']=='disagreement'])
print("source retention baseline/after:", retention_before, retention_after)

# Changed-condition replay of the existing sequence-PPO appendix evidence.
# No model update or retuning: keep recorded held-out prompts and responses fixed.
ppo_path = ROOT / "projects/posttraining-audit/transformer_lineage/teaching/v5/evaluation.json"
ppo = json.loads(ppo_path.read_text())
heldout_ppo = ppo["splits"]["heldout"]["episodes"]
parent_rewards = np.asarray([r["parentFittedRewardScore"] for r in heldout_ppo], np.float64)
child_rewards = np.asarray([r["childFittedRewardScore"] for r in heldout_ppo], np.float64)
matched_responses = np.asarray([r["parentActionIds"] == r["childActionIds"] for r in heldout_ppo])
assert len(heldout_ppo) == 8 and int(matched_responses.sum()) == 8
for scale in (0.5, 1.0, 2.0):
    delta = float(np.mean(scale * child_rewards) - np.mean(scale * parent_rewards))
    print("recorded PPO heldout reward scale / child-minus-parent:", scale, delta)
assert ppo["qualityGate"]["heldoutRewardFloorPassed"] is False
print("recorded heldout response matches:", int(matched_responses.sum()), "/", len(heldout_ppo),
      "; strict heldout reward floor remains FAILED at every scale")
print("Synthetic objective check only. No human preference or response-quality claim.")
print("PASS: posttraining-05")
