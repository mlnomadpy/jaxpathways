"""RLHF mechanics: a frozen reward and PPO policy updates: worked experiments and reference solutions. CPU checks."""

# 1. Define the reward likelihood and signed PPO loss
# Step 1 — 1. Define the reward likelihood and signed PPO loss: Keep reward fitting and policy loss separate.
import jax
import jax.numpy as jnp
import numpy as np

def preference_loss(w, chosen, rejected):
    margin = (chosen - rejected) @ w
    return jnp.mean(jax.nn.softplus(-margin))

# Define `ppo_loss(logits, old_logps, actions, advantages...)` to evaluate the objective and its automatic derivatives:
def ppo_loss(
    logits, old_logps, actions, advantages, reference_logps, beta=0.1, clip=0.2
):
    logp = jax.nn.log_softmax(logits)
    ratios = jnp.exp(logp[actions] - jax.lax.stop_gradient(old_logps))
    advantages = jax.lax.stop_gradient(advantages)
    clipped = jnp.clip(ratios, 1.0 - clip, 1.0 + clip)
    surrogate = jnp.mean(jnp.minimum(ratios * advantages, clipped * advantages))
    # Exact categorical KL in this one-prompt, one-action teaching environment.
    kl = jnp.sum(
        jnp.exp(logp) * (logp - jax.lax.stop_gradient(reference_logps))
    )
    return -surrogate + beta * kl

# 2. Fit and freeze reward, then identify reference state
# Step 2 — 2. Fit and freeze reward, then identify reference state: The reference is stored before policy updates.
features = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
chosen = features[jnp.array([0, 0, 1])]
rejected = features[jnp.array([1, 2, 2])]
reward_w = jnp.zeros(2)
# Differentiate the objective to obtain `reward_step` via automatic differentiation.
reward_step = jax.jit(jax.grad(lambda w: preference_loss(w, chosen, rejected)))
for _ in range(100):
    reward_w = reward_w - 0.1 * reward_step(reward_w)
rewards = jax.lax.stop_gradient(features @ reward_w)
# A fixed reference policy and a learned reward, with one terminal response action.
reference = jax.nn.log_softmax(jnp.array([0.2, 0.0, -0.2]))
theta = jnp.array([0.2, 0.0, -0.2])
# Transfer the device result to the host for an independent NumPy comparison.
reference_copy = np.asarray(reference).copy()
# Create or split explicit PRNG key(s) (`key`) for reproducible randomness.
key = jax.random.key(11)
history = []
kl_history = []
step = jax.jit(jax.value_and_grad(ppo_loss))

# 3. Collect rollouts and reuse each batch correctly
# Step 3 — 3. Collect rollouts and reuse each batch correctly: Each outer iteration collects a fresh batch; each inner step...
for iteration in range(40):
    old = jax.nn.log_softmax(theta)
    probs = jnp.exp(old)
    # Create or split explicit PRNG key(s) (`(key, draw)`) for reproducible randomness.
    key, draw = jax.random.split(key)
    # Draw pseudorandom samples for `actions` using the explicit RNG state.
    actions = jax.random.categorical(draw, old, shape=(128,))
    old_selected = jax.lax.stop_gradient(old[actions])
    advantage = jax.lax.stop_gradient(
        rewards[actions] - jnp.sum(probs * rewards)
    )
    for _ in range(3):
        _, g = step(
            theta, old_selected, actions, advantage, reference, 0.2, 0.2
        )
        theta = theta - 0.15 * g
    logp = jax.nn.log_softmax(theta)
    history.append(float(jnp.sum(jnp.exp(logp) * rewards)))
    kl_history.append(float(jnp.sum(jnp.exp(logp) * (logp - reference))))

np.testing.assert_array_equal(reference, reference_copy)
initial_reward = float(jnp.sum(jnp.exp(reference) * rewards))
assert history[-1] > initial_reward and kl_history[-1] > 0
# Independent finite-action expectation: no evaluation sampling error here.
np.testing.assert_allclose(
    history[-1],
    sum(float(a) * float(b) for a, b in zip(jnp.exp(logp), rewards)),
    rtol=1e-6,
)
print(
    'Reward initial/final and final KL:',
    initial_reward,
    history[-1],
    kl_history[-1],
)
print(
    'Synthetic preference bandit with actual sampled PPO updates; synthetic preference target or language-generation claim.'
)

# Step 1 — 1. Define the reward likelihood and signed PPO loss: Keep reward fitting and policy loss separate.
import jax
import jax.numpy as jnp
import numpy as np

def preference_loss(w, chosen, rejected):
    margin = (chosen - rejected) @ w
    return jnp.mean(jax.nn.softplus(-margin))

def ppo_loss(
    logits, old_logps, actions, advantages, reference_logps, beta=0.1, clip=0.2
):
    logp = jax.nn.log_softmax(logits)
    ratios = jnp.exp(logp[actions] - jax.lax.stop_gradient(old_logps))
    advantages = jax.lax.stop_gradient(advantages)
    clipped = jnp.clip(ratios, 1.0 - clip, 1.0 + clip)
    surrogate = jnp.mean(jnp.minimum(ratios * advantages, clipped * advantages))
    # Exact categorical KL in this one-prompt, one-action teaching environment.
    kl = jnp.sum(
        jnp.exp(logp) * (logp - jax.lax.stop_gradient(reference_logps))
    )
    return -surrogate + beta * kl

# Step 2 — 2. Fit and freeze reward, then identify reference state: The reference is stored before policy updates.
features = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
chosen = features[jnp.array([0, 0, 1])]
rejected = features[jnp.array([1, 2, 2])]
reward_w = jnp.zeros(2)
# Differentiate the objective to obtain `reward_step` via automatic differentiation.
reward_step = jax.jit(jax.grad(lambda w: preference_loss(w, chosen, rejected)))
for _ in range(100):
    reward_w = reward_w - 0.1 * reward_step(reward_w)
rewards = jax.lax.stop_gradient(features @ reward_w)
# A fixed reference policy and a learned reward, with one terminal response action.
reference = jax.nn.log_softmax(jnp.array([0.2, 0.0, -0.2]))
theta = jnp.array([0.2, 0.0, -0.2])
# Transfer the device result to the host for an independent NumPy comparison.
reference_copy = np.asarray(reference).copy()
# Create or split explicit PRNG key(s) (`key`) for reproducible randomness.
key = jax.random.key(11)
history = []
kl_history = []
step = jax.jit(jax.value_and_grad(ppo_loss))

# Step 3 — 3. Collect rollouts and reuse each batch correctly: Each outer iteration collects a fresh batch; each inner step...
for iteration in range(40):
    old = jax.nn.log_softmax(theta)
    probs = jnp.exp(old)
    # Create or split explicit PRNG key(s) (`(key, draw)`) for reproducible randomness.
    key, draw = jax.random.split(key)
    # Draw pseudorandom samples for `actions` using the explicit RNG state.
    actions = jax.random.categorical(draw, old, shape=(128,))
    old_selected = jax.lax.stop_gradient(old[actions])
    advantage = jax.lax.stop_gradient(
        rewards[actions] - jnp.sum(probs * rewards)
    )
    for _ in range(3):
        _, g = step(
            theta, old_selected, actions, advantage, reference, 0.2, 0.2
        )
        theta = theta - 0.15 * g
    logp = jax.nn.log_softmax(theta)
    history.append(float(jnp.sum(jnp.exp(logp) * rewards)))
    kl_history.append(float(jnp.sum(jnp.exp(logp) * (logp - reference))))

np.testing.assert_array_equal(reference, reference_copy)
initial_reward = float(jnp.sum(jnp.exp(reference) * rewards))
assert history[-1] > initial_reward and kl_history[-1] > 0
# Independent finite-action expectation: no evaluation sampling error here.
np.testing.assert_allclose(
    history[-1],
    sum(float(a) * float(b) for a, b in zip(jnp.exp(logp), rewards)),
    rtol=1e-6,
)
print(
    'Reward initial/final and final KL:',
    initial_reward,
    history[-1],
    kl_history[-1],
)
print(
    'Synthetic preference bandit with actual sampled PPO updates; synthetic preference target or language-generation claim.'
)

# Figure data experiment
visual_data={'panels':[{'kind':'line','xlabel':'completed PPO rollout batches','ylabel':'expected learned reward','series':[{'label':'exact policy expectation','x':list(range(1,len(history)+1)),'y':history}]},{'kind':'line','xlabel':'completed PPO rollout batches','ylabel':'KL to fixed reference (nats)','series':[{'label':'exact categorical KL','x':list(range(1,len(kl_history)+1)),'y':kl_history}]}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']

# Transfer the device result to the host for an independent NumPy comparison.
extra_panel={'kind':'bar','x':[0,1,2],'labels':['action 0','action 1','action 2'],'series':[{'label':'fixed reference','y':np.asarray(jnp.exp(reference)).tolist()},{'label':'trained policy','y':np.asarray(jnp.exp(logp)).tolist()}],'xlabel':'terminal response action','ylabel':'probability','title':'How increased reward redistributes action probability'}
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}

# Experiment: Inspect both clipping directions
# Experiment — Inspect both clipping directions: The negative-advantage case selects the more negative product.
ratios = np.array([1.5, 0.5])
advantages = np.array([2.0, -2.0])
terms = np.minimum(
    ratios * advantages, np.clip(ratios, 0.8, 1.2) * advantages
)
np.testing.assert_allclose(terms, [2.4, -1.6])
print('Clipped surrogate terms:', terms)

# Experiment: Enumerate the baseline cancellation
# Experiment — Enumerate the baseline cancellation: Enumeration removes sampling error.
probe = jnp.array([0.3, -0.2, 0.1])
probe_probs = jax.nn.softmax(probe)
constant = float(jnp.sum(probe_probs * rewards))
score_jacobian = jax.jacrev(jax.nn.log_softmax)(probe)
# Reduce along axis=0 to compute `enumerated`.
enumerated = jnp.sum(
    probe_probs[:, None] * (rewards - constant)[:, None] * score_jacobian,
    axis=0,
)
# Differentiate the objective to obtain `exact` via automatic differentiation.
exact = jax.grad(lambda logits: jnp.sum(jax.nn.softmax(logits) * rewards))(
    probe
)
np.testing.assert_allclose(enumerated, exact, atol=1e-6)
print('Enumerated baseline score gradient matches exact expected-reward gradient.')

# Reference solution. Try the exercise before reading this.
# Exercise solution: Check the KL at the fixed reference and compare it with the final...
zero_kl = float(jnp.sum(jnp.exp(reference) * (reference - reference)))
assert zero_kl == 0.0 and kl_history[-1] > zero_kl
print('Reference/final KL:', zero_kl, kl_history[-1])

# Reference practice: Verify detached behavior information
# Verify detached behavior information (Transfer): Detaching is not enough if you recompute old probabilities...
# Differentiate the objective to obtain `(old_grad, adv_grad)` via automatic differentiation.
old_grad, adv_grad = jax.grad(ppo_loss, argnums=(1, 3))(
    theta, old_selected, actions, advantage, reference, 0.2, 0.2
)
np.testing.assert_array_equal(old_grad, np.zeros(old_grad.shape))
np.testing.assert_array_equal(adv_grad, np.zeros(adv_grad.shape))
print('Behavior log probabilities and advantages are detached.')

# Reference practice: Compare with the finite-action regularized optimum
# Compare with the finite-action regularized optimum (Challenge): An independently derived optimum bounds this finite-action...
beta = 0.2
optimal_logp = jax.nn.log_softmax(reference + rewards / beta)

def regularized(logp):
    return jnp.sum(jnp.exp(logp) * rewards) - beta * jnp.sum(
        jnp.exp(logp) * (logp - reference)
    )

optimum = float(regularized(optimal_logp))
trained = float(regularized(logp))
baseline = float(regularized(reference))
assert optimum >= trained - 1e-5 and optimum >= baseline - 1e-5
print(
    'Regularized objective reference/trained/optimum:',
    baseline,
    trained,
    optimum,
)
print("PASS: posttraining-04")
